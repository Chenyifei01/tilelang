"""Vocab-parallel embedding gradient reduce (scatter-add) for Ascend NPUs.

Design: ``embedding_grad_reduce/design.md``. ``embedding_lookup`` gathers rows;
this operator is its reverse: for every token position ``j`` it accumulates
``grad_output[j]`` into row ``indices[j]`` of the embedding-gradient table,
so repeated ids add up. The table lives in global memory across the whole run
(``vocab * hidden`` is far larger than UB for DeepSeek V4), which rules out any
on-chip reduction of the table itself: every token is one random row read plus
one atomic row add.

Everything runs on the NPU. The wrapper does no tensor arithmetic at all: it
checks scalar preconditions and launches kernels. Two device kernels cover the
general case:

``prepare``  each core scans its token slice, drops ids outside
             ``[vocab_start, vocab_end)`` and the padding id, compacts the
             survivors into per-core scratch lists and records its hit count
             (optionally histogramming ids for ``scale_grad_by_freq``);
``reduce``   each core iterates exactly its own hit count -- read the grad row,
             cast to fp32, atomic-add it into the destination row.

The full-table case (``vocab_start == 0``, no padding, no frequency scaling)
skips ``prepare`` entirely: every token is a hit, so one kernel reads the ids
directly and the whole thing is a single launch.

The token count, the hit count and ``grad_output``'s row count are all dynamic,
so one compiled binary serves every training step; only ``num_rows``/``H``/
dtypes/``whole_table``/``scale_by_freq`` participate in specialization.

Two disciplines carried over from ``embedding_lookup`` (see its design doc):
MTE movement stays unconditional, and conditional work stays on the V pipe.
The full-table chunk tail is handled by clamping the token index and scaling by
0, never by guarding a DMA or atomic store. The compacted path needs no tail
trick at all because each core's loop bound is its own hit count.

Duplicate ids are resolved by the hardware store-mode atomic add
(``T.set_atomic("add", ...)``); the order is therefore not fixed, so repeated
runs may differ in the last fp32 ulp.
"""

import argparse

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.profiler import do_bench

# Measured AIV count on Ascend 950DT (embedding_lookup design.md §1.4).
NUM_AIV = 56
# UB envelope used for the row pipeline; the hardware limit is ~216 KiB and
# the scheduler needs headroom for flags/reservations.
UB_PLAN = 192 * 1024
# Tokens one core scans (and may compact) in the fused kernel; the lists live
# in UB, so this also bounds the tokens one call may process.
PREP_BLK = 1536
MAX_TOKENS = NUM_AIV * PREP_BLK

_DTYPE_STR = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
}
_STR_TO_TORCH = {v: k for k, v in _DTYPE_STR.items()}
_ID_DTYPES = {torch.int32: "int32", torch.int64: "int64"}

_SCRATCH: dict = {}


def _counts(num_rows, device):
    """Cached per-row hit histogram (zeroed by a kernel before every use)."""
    key = ("counts", num_rows, device.index)
    buf = _SCRATCH.get(key)
    if buf is None:
        buf = torch.empty(num_rows, dtype=torch.int32, device=device)
        _SCRATCH[key] = buf
    return buf


def _vf_threads(H):
    """SIMT VF width for the bf16->fp32 cast row.

    Measured on 950DT (H=7168, 32768 tokens): the cast is the hard serial
    floor at 128 threads (706.9 us, vec=99.8%, MTE2 64%), while 256/512
    threads make it overlap with the memory pipes (495.8/493.4 us, MTE2 ~98%,
    MTE3 ~95%). The cast:memory ratio is H-independent, so the wide domain is
    used for every production hidden size; tiny rows keep the cheap 128-thread
    domain to avoid paying the wider VF's fixed cost.
    """
    return 512 if H >= 512 else 128


def _pipeline_config(H, esize_src, esize_acc, list_bytes=0):
    """(versions, num_stages) for the per-token row pipeline.

    Each in-flight token owns one source row (``in_dtype``) and one
    accumulator row (``acc_dtype``); the pipeline depth is the largest value
    that keeps ``versions * (src + acc) + list_bytes`` inside ``UB_PLAN``
    (``list_bytes`` is the fused kernel's per-core id copy and compacted
    lists). Rows are copied one at a time, so a row byte count that is not a
    multiple of 32 B is fine (the lookup's multi-row pitch constraint does not
    apply).
    """
    per_stage = H * (esize_src + esize_acc)
    room = UB_PLAN - list_bytes
    versions = max(2, min(4, room // per_stage))
    assert versions * per_stage + list_bytes <= UB_PLAN, (
        f"hidden={H} needs {per_stage} B per stage; two source/accumulator row "
        f"versions plus the compacted lists do not fit the UB plan"
    )
    return versions, versions


def _list_bytes(id_dtype, split_unique):
    """UB held by the per-core id copy plus the compacted ``(row, id)`` lists.

    Split mode stages every hit in ``all_*``, gathers the per-row counts into
    ``uniq`` and then partitions into the plain/atomic lists; the atomic-only
    mode writes straight into its two lists.
    """
    esize_id = 8 if id_dtype == "int64" else 4
    lists = 7 if split_unique else 2  # all(2) + uniq(1) + plain(2) + atom(2)
    return PREP_BLK * (esize_id + 4 * lists)


@tilelang.jit(target="ascend")
def embedding_grad_reduce_kernel(
    num_rows,
    H,
    in_dtype="bfloat16",
    acc_dtype="float32",
    scale_grad_by_freq=False,
    id_dtype="int32",
    split_unique=False,
):
    """Build and JIT-cache the scatter-add kernel.

    Compile-time: ``num_rows``/``H``/the dtypes/``scale_grad_by_freq``/
    ``split_unique``. Runtime (dynamic): the token count ``n`` and the table
    row count, so one binary serves every step.

    The kernel takes the caller's ids plus a per-row occurrence histogram
    (``Counts``), produced by its own two tiny launches. Each core compacts its
    own hits into two UB lists and then runs two pipelined passes:

    * rows whose id occurs exactly once go to the plain-store list -- a single
      writer means a normal store is enough, so only ``read + write``
      (6 B/elem) moves instead of an atomic read-modify-write (10 B/elem);
    * rows that occur more than once go to the atomic-add list.

    The split is per row, not per token: every writer of a row takes the same
    branch, so the two lists never touch the same row and no ordering is
    required between them. ``split_unique=False`` sends everything to the
    atomic list, which is what a caller accumulating into a non-zero table
    (``out`` reused across calls) must use. ``scale_grad_by_freq`` divides
    every contribution by its id count.

    Duplicate ids accumulate through the hardware store-mode atomic add; the
    order across cores and tokens is not fixed (see the module docstring).
    """
    assert num_rows > 0 and H > 0
    esize_src = 2 if in_dtype in ("bfloat16", "float16") else 4
    esize_acc = 2 if acc_dtype in ("bfloat16", "float16") else 4
    vf_threads = _vf_threads(H)
    versions, num_stages = _pipeline_config(H, esize_src, esize_acc, _list_bytes(id_dtype, split_unique))
    n = T.dynamic("n")
    rows = T.dynamic("rows")

    @T.prim_func
    def fused(
        Ids: T.Tensor((n,), id_dtype),
        Counts: T.Tensor((rows,), "int32"),
        GradOut: T.Tensor((n, H), in_dtype),
        GradWeight: T.Tensor((num_rows, H), acc_dtype),
        vocab_start: T.int32,
        vocab_end: T.int32,
        padding_idx: T.int32,
    ):
        with T.Kernel(NUM_AIV) as core:
            id_ub = T.alloc_shared((PREP_BLK,), id_dtype)
            all_src = T.alloc_shared((PREP_BLK,), "int32")
            all_dst = T.alloc_shared((PREP_BLK,), "int32")
            uniq = T.alloc_shared((PREP_BLK,), "int32")
            plain_src = T.alloc_shared((PREP_BLK,), "int32")
            plain_dst = T.alloc_shared((PREP_BLK,), "int32")
            atom_src = T.alloc_shared((PREP_BLK,), "int32")
            atom_dst = T.alloc_shared((PREP_BLK,), "int32")
            src_ub = T.alloc_shared((H,), in_dtype)
            acc_ub = T.alloc_shared((H,), acc_dtype)
            T.annotate_buffer_versions({src_ub: versions, acc_ub: versions})
            n_all = T.alloc_var("int32", init=0)
            n_plain = T.alloc_var("int32", init=0)
            n_atom = T.alloc_var("int32", init=0)
            chunk = (n + NUM_AIV - 1) // NUM_AIV
            base = core * chunk
            T.copy(Ids[base : base + PREP_BLK], id_ub)
            # Scalar filter + compact. Every pipeline below runs exactly its own
            # count, so no MTE is ever conditionally dispatched.
            for j in T.serial(T.min(chunk, PREP_BLK)):
                t = base + j
                idv = T.cast(id_ub[j], "int32")
                if (t < n) and (idv >= vocab_start) and (idv < vocab_end) and (idv != padding_idx):
                    all_src[n_all] = t
                    all_dst[n_all] = idv - vocab_start
                    n_all += 1
            if split_unique:
                # The per-row counts are scattered in GM; gather them in the VF
                # so the latencies overlap instead of serializing the scalar
                # loop above.
                with T.SimtVF(threads=256):
                    for j in T.Parallel(PREP_BLK):
                        uniq[j] = Counts[T.if_then_else(j < n_all, all_dst[j], 0)]
                # Partition in UB only.
                for j in T.serial(n_all):
                    if uniq[j] == 1:
                        plain_src[n_plain] = all_src[j]
                        plain_dst[n_plain] = all_dst[j]
                        n_plain += 1
                    else:
                        atom_src[n_atom] = all_src[j]
                        atom_dst[n_atom] = all_dst[j]
                        n_atom += 1
            else:
                n_atom = n_all
                for j in T.serial(n_all):
                    atom_src[j] = all_src[j]
                    atom_dst[j] = all_dst[j]
            # Single-writer rows: a plain store is correct and moves 4 B/elem
            # instead of the atomic read-modify-write's 8 B/elem.
            for i in T.Pipelined(n_plain, num_stages=num_stages):
                row = plain_dst[i]
                T.copy(GradOut[plain_src[i], :], src_ub)
                with T.SimtVF(threads=vf_threads):
                    for j in T.Parallel(H):
                        if scale_grad_by_freq:
                            acc_ub[j] = T.cast(src_ub[j], acc_dtype) * (
                                T.cast(1, "float32") / T.cast(Counts[row], "float32")
                            )
                        else:
                            acc_ub[j] = T.cast(src_ub[j], acc_dtype)
                T.copy(acc_ub, GradWeight[row, :])
            # Shared rows: hardware store-mode atomic add.
            T.set_atomic("add", acc_dtype)
            for i in T.Pipelined(n_atom, num_stages=num_stages):
                row = atom_dst[i]
                T.copy(GradOut[atom_src[i], :], src_ub)
                with T.SimtVF(threads=vf_threads):
                    for j in T.Parallel(H):
                        if scale_grad_by_freq:
                            acc_ub[j] = T.cast(src_ub[j], acc_dtype) * (
                                T.cast(1, "float32") / T.cast(Counts[row], "float32")
                            )
                        else:
                            acc_ub[j] = T.cast(src_ub[j], acc_dtype)
                T.copy(acc_ub, GradWeight[row, :])
            T.set_atomic_none()

    return fused


@tilelang.jit(target="ascend")
def embedding_grad_reduce_hist_kernel(id_dtype="int32"):
    """Per-row occurrence histogram, used only by ``scale_grad_by_freq``.

    Counts must be complete before any ``1/count`` is applied, so this runs as
    its own launch ahead of the fused kernel (rows touched by several cores
    make it impossible inside one kernel without a grid barrier).
    """
    n = T.dynamic("n")
    rows = T.dynamic("rows")

    @T.prim_func
    def histogram(
        Ids: T.Tensor((n,), id_dtype),
        Counts: T.Tensor((rows,), "int32"),
        vocab_start: T.int32,
        vocab_end: T.int32,
        padding_idx: T.int32,
    ):
        with T.Kernel(NUM_AIV) as core:
            id_ub = T.alloc_shared((PREP_BLK,), id_dtype)
            chunk = (n + NUM_AIV - 1) // NUM_AIV
            base = core * chunk
            T.copy(Ids[base : base + PREP_BLK], id_ub)
            for j in T.serial(T.min(chunk, PREP_BLK)):
                t = base + j
                idv = T.cast(id_ub[j], "int32")
                if (t < n) and (idv >= vocab_start) and (idv < vocab_end) and (idv != padding_idx):
                    T.atomic_add(Counts[idv - vocab_start], 1)

    return histogram


@tilelang.jit(target="ascend")
def embedding_grad_reduce_zero_kernel():
    """Zero the histogram (one launch before ``hist``)."""
    rows = T.dynamic("rows")

    @T.prim_func
    def zero(Counts: T.Tensor((rows,), "int32")):
        with T.Kernel(NUM_AIV) as core:
            chunk = (rows + NUM_AIV - 1) // NUM_AIV
            with T.SimtVF(threads=512):
                for j in T.Parallel(chunk):
                    Counts[T.min(core * chunk + j, rows - 1)] = 0

    return zero


def ref_program(grad_output, indices, num_weights, padding_idx=-1, scale_grad_by_freq=False,
                vocab_start_index=0, vocab_end_index=None):
    """Golden: ATen ``embedding_dense_backward`` semantics on CPU, fp32.

    Ids outside ``[vocab_start_index, vocab_end_index)`` and the padding row
    contribute nothing. ``scale_grad_by_freq`` divides by the count of valid
    occurrences (README §3 golden). Duplicate ids accumulate.
    """
    grad_output = grad_output.reshape(-1, grad_output.shape[-1])
    idx = indices.reshape(-1)
    if vocab_end_index is None:
        vocab_end_index = vocab_start_index + num_weights
    go = grad_output.float().cpu()
    idx = idx.cpu().long()
    out = torch.zeros(num_weights, go.shape[1], dtype=torch.float32)
    valid = (idx >= vocab_start_index) & (idx < vocab_end_index)
    if padding_idx >= 0:
        valid &= idx != padding_idx
    idx_v = idx[valid]
    go_v = go[valid]
    if scale_grad_by_freq:
        counts = torch.bincount(idx_v, minlength=vocab_end_index).clamp_min(1)
        go_v = go_v / counts[idx_v].unsqueeze(-1)
    out.index_add_(0, idx_v - vocab_start_index, go_v)
    return out


def embedding_grad_reduce(
    grad_output: torch.Tensor,
    indices: torch.Tensor,
    num_weights: int | None = None,
    padding_idx: int = -1,
    scale_grad_by_freq: bool = False,
    vocab_start_index: int = 0,
    vocab_end_index: int | None = None,
    out: torch.Tensor | None = None,
    out_zeroed: bool = False,
) -> torch.Tensor:
    """Accumulate ``grad_output`` into ``grad_weight[indices]`` (fp32 table).

    ``num_weights`` is the number of rows this rank owns; with
    ``vocab_end_index`` set it defaults to ``vocab_end_index - vocab_start_index``.
    Ids outside ``[vocab_start_index, vocab_end_index)`` and ``padding_idx``
    are dropped inside the kernel (not by the host).

    ``out`` may be an existing fp32 ``[num_weights, H]`` table. When the table
    is known to hold zeros -- a freshly allocated one, which is what autograd
    hands a backward -- pass ``out_zeroed=True`` (or omit ``out`` and let the
    wrapper allocate it): the kernel then writes single-contribution rows with
    plain stores, moving 6 B/elem instead of the 10 B/elem an atomic
    read-modify-write needs. For a table that already holds values (e.g. MTP
    accumulating several branches into one embedding), leave ``out_zeroed``
    false and every row goes through the atomic add.

    Returned dtype is fp32 (internal accumulation precision), not
    ``grad_output``'s dtype.

    Caller contract: ``indices`` holds ids in ``[0, vocab_size)`` for the
    *full* table; ids may repeat. ``grad_output`` must be row-major contiguous.
    """
    H = grad_output.shape[-1]
    assert num_weights is not None or vocab_end_index is not None, (
        "pass num_weights (rows owned by this rank) or vocab_end_index"
    )
    if vocab_end_index is None:
        vocab_end_index = vocab_start_index + num_weights
    if num_weights is None:
        num_weights = vocab_end_index - vocab_start_index
    assert num_weights == vocab_end_index - vocab_start_index, (
        f"num_weights={num_weights} must equal vocab_end_index - vocab_start_index"
    )
    assert grad_output.is_contiguous(), "grad_output must be row-major contiguous"
    in_dtype = _DTYPE_STR.get(grad_output.dtype)
    if in_dtype is None:
        raise TypeError(f"unsupported grad_output dtype: {grad_output.dtype}")
    id_dtype = _ID_DTYPES.get(indices.dtype)
    if id_dtype is None:
        raise TypeError(f"unsupported indices dtype: {indices.dtype}")

    if out is None:
        out = torch.zeros(num_weights, H, dtype=torch.float32, device=grad_output.device)
        out_zeroed = True
    else:
        assert out.dtype == torch.float32, "out must be fp32 (internal accumulation dtype)"
        assert out.shape == (num_weights, H), f"out must be {(num_weights, H)}, got {tuple(out.shape)}"
        assert out.is_contiguous(), "out must be row-major contiguous"

    go = grad_output.reshape(-1, H)  # view: grad_output is contiguous
    idx = indices.reshape(-1)  # view for contiguous ids (no host computation)
    assert idx.numel() == go.shape[0], (
        f"indices has {idx.numel()} entries but grad_output has {go.shape[0]} rows"
    )
    n = idx.numel()
    if n == 0:  # nothing to scatter
        return out

    assert n <= MAX_TOKENS, (
        f"one core scans at most {PREP_BLK} tokens, so this build handles "
        f"{MAX_TOKENS} tokens per call; got {n}"
    )
    device = go.device
    # Counts span cores, so the histogram (and its zeroing) has to finish before
    # the fused kernel can split unique rows or divide by 1/count.
    counts = _counts(num_weights, device)
    embedding_grad_reduce_zero_kernel()(counts)
    embedding_grad_reduce_hist_kernel(id_dtype)(idx, counts, vocab_start_index, vocab_end_index, padding_idx)
    embedding_grad_reduce_kernel(
        num_weights, H, in_dtype, "float32", scale_grad_by_freq, id_dtype, out_zeroed
    )(idx, counts, go, out, vocab_start_index, vocab_end_index, padding_idx)
    return out


def _traffic_bytes(ids, hits, H, esize_src, esize_acc):
    """Bytes actually moved: one grad-row read per hit, one plain write per
    single-contribution row, one atomic read-modify-write per duplicate token."""
    uniq = int(torch.unique(ids, return_counts=False).numel()) if hits else 0
    dups = hits - uniq
    return hits * H * esize_src + uniq * H * esize_acc + dups * H * 2 * esize_acc


def run_regression_perf(num_tokens=32768, H=7168, dtype="bfloat16", vocab_size=129280):
    """Full-table bandwidth regression (read each grad row, write each row once)."""
    ids = torch.randint(0, vocab_size, (num_tokens,), dtype=torch.int32).npu()
    go = torch.randn(num_tokens, H, device="npu").to(_STR_TO_TORCH[dtype])
    gw = torch.zeros(vocab_size, H, dtype=torch.float32, device="npu")
    latency = do_bench(
        lambda: embedding_grad_reduce(go, ids, vocab_size, out=gw, out_zeroed=True),
        backend="msprof",
        _n_warmup=30,
        _n_repeat=50,
    )
    num_bytes = _traffic_bytes(ids.cpu(), num_tokens, H, go.element_size(), gw.element_size())
    bandwidth = num_bytes / (latency * 1e6)
    print(
        f"embedding_grad_reduce (num_tokens={num_tokens}, H={H}, {dtype}): "
        f"{latency * 1000:.2f} us | {bandwidth:.2f} GB/s"
    )
    return latency


def run_sharded_perf(num_tokens=32768, H=7168, dtype="bfloat16", vocab_size=129280, tp=8):
    """TP=``tp`` simulation: end-to-end wrapper (on-device prepare + reduce)."""
    shard = vocab_size // tp
    ids = torch.randint(0, vocab_size, (num_tokens,), dtype=torch.int32).npu()
    go = torch.randn(num_tokens, H, device="npu").to(_STR_TO_TORCH[dtype])
    gw = torch.zeros(shard, H, dtype=torch.float32, device="npu")
    start, end = 3 * shard, 4 * shard  # arbitrary rank 3 of 8
    latency = do_bench(
        lambda: embedding_grad_reduce(
            go, ids, vocab_end_index=end, vocab_start_index=start, out=gw, out_zeroed=True
        ),
        backend="msprof",
        _n_warmup=10,
        _n_repeat=30,
    )
    hit_ids = ids[(ids >= start) & (ids < end)]
    hits = int(hit_ids.numel())
    num_bytes = _traffic_bytes(hit_ids.cpu(), hits, H, go.element_size(), gw.element_size())
    bandwidth = num_bytes / (latency * 1e6)
    print(
        f"embedding_grad_reduce shard (TP={tp}, hits={hits}/{num_tokens}, H={H}, {dtype}): "
        f"{latency * 1000:.2f} us | {bandwidth:.2f} GB/s"
    )
    return latency


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bench", action=argparse.BooleanOptionalAction, default=True, help="benchmark after correctness checks"
    )
    args = parser.parse_args()
    # Sharded scenario with duplicate ids and padding (README §6 checklist).
    num_tokens, vocab_size, vocab_start, vocab_end, H = 4096, 8192, 512, 4608, 4096
    torch.manual_seed(0)
    ids = torch.randint(0, vocab_size, (num_tokens,), dtype=torch.int32).npu()
    ids[: num_tokens // 8] = ids[-num_tokens // 8 :]  # force duplicates
    go = torch.randn(num_tokens, H, device="npu").to(torch.bfloat16)
    got = embedding_grad_reduce(go, ids, padding_idx=0, vocab_start_index=vocab_start, vocab_end_index=vocab_end)
    expect = ref_program(go, ids, vocab_end - vocab_start, 0, False, vocab_start, vocab_end).npu()
    torch.testing.assert_close(got, expect, rtol=1e-5, atol=1e-5)
    print("embedding_grad_reduce: correctness passed")
    if args.bench:
        run_regression_perf()
        run_sharded_perf()
