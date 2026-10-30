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
# Tokens scanned per core in one ``prepare`` pass; the compacted list lives in
# UB, so this also bounds the tokens one call may process.
PREP_BLK = 4096
MAX_TOKENS = NUM_AIV * PREP_BLK

_DTYPE_STR = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
}
_STR_TO_TORCH = {v: k for k, v in _DTYPE_STR.items()}
_ID_DTYPES = {torch.int32: "int32", torch.int64: "int64"}

_SCRATCH: dict = {}


def _scratch(cap, device):
    """Cached int32 scratch: compacted source rows and destination ids."""
    key = ("lists", cap, device.index)
    buf = _SCRATCH.get(key)
    if buf is None:
        buf = (
            torch.empty(cap, dtype=torch.int32, device=device),
            torch.empty(cap, dtype=torch.int32, device=device),
            torch.empty(NUM_AIV, dtype=torch.int32, device=device),
        )
        _SCRATCH[key] = buf
    return buf


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


def _pipeline_config(H, esize_src, esize_acc):
    """(versions, num_stages) for the per-token row pipeline.

    Each in-flight token owns one source row (``in_dtype``) and one
    accumulator row (``acc_dtype``); the pipeline depth is the largest value
    that keeps ``versions * (src + acc)`` inside ``UB_PLAN``. Rows are copied
    one at a time, so a row byte count that is not a multiple of 32 B is fine
    (the lookup's multi-row pitch constraint does not apply).
    """
    per_stage = H * (esize_src + esize_acc)
    versions = max(2, min(4, UB_PLAN // per_stage))
    assert versions * per_stage <= UB_PLAN, (
        f"hidden={H} needs {per_stage} B per stage; two source/accumulator row "
        f"versions do not fit the UB plan"
    )
    return versions, versions


def embedding_grad_reduce_prepare_fwd(scale_by_freq=False, id_dtype="int32"):
    """Build the filtering/compaction kernel.

    Each core scans ``chunk = ceil(n / NUM_AIV)`` tokens, keeps those with
    ``vocab_start <= id < vocab_end`` and ``id != padding_idx``, and writes
    ``(grad row, local id)`` pairs into its own scratch region. The per-core
    hit count goes to ``Hits``. With ``scale_by_freq`` the kernel also bumps a
    per-row histogram, which the reduce kernel turns into ``1/count``.
    """
    n = T.dynamic("n")
    cap = T.dynamic("cap")
    rows = T.dynamic("rows")
    # Each core owns a PREP_BLK-entry slice of the scratch (``obase``), so the
    # fixed-length stores never spill into a neighbour's region.

    if scale_by_freq:

        @T.prim_func
        def main(
            Ids: T.Tensor((n,), id_dtype),
            SrcOut: T.Tensor((cap,), "int32"),
            DstOut: T.Tensor((cap,), "int32"),
            Hits: T.Tensor((NUM_AIV,), "int32"),
            Counts: T.Tensor((rows,), "int32"),
            vocab_start: T.int32,
            vocab_end: T.int32,
            padding_idx: T.int32,
        ):
            with T.Kernel(NUM_AIV) as core:
                id_ub = T.alloc_shared((PREP_BLK,), id_dtype)
                src_ub = T.alloc_shared((PREP_BLK,), "int32")
                dst_ub = T.alloc_shared((PREP_BLK,), "int32")
                pos = T.alloc_var("int32", init=0)
                chunk = (n + NUM_AIV - 1) // NUM_AIV
                base = core * chunk
                obase = core * PREP_BLK
                T.copy(Ids[base : base + PREP_BLK], id_ub)
                # Scalar filter + append: the compaction order is per core, so
                # the scratch writes stay inside this core's region.
                for j in T.serial(T.min(chunk, PREP_BLK)):
                    t = base + j
                    idv = T.cast(id_ub[j], "int32")
                    if (t < n) and (idv >= vocab_start) and (idv < vocab_end) and (idv != padding_idx):
                        src_ub[pos] = t
                        dst_ub[pos] = idv - vocab_start
                        T.atomic_add(Counts[idv - vocab_start], 1)
                        pos += 1
                T.copy(src_ub, SrcOut[obase : obase + PREP_BLK])
                T.copy(dst_ub, DstOut[obase : obase + PREP_BLK])
                Hits[core] = pos

        return main

    @T.prim_func
    def main(
        Ids: T.Tensor((n,), id_dtype),
        SrcOut: T.Tensor((cap,), "int32"),
        DstOut: T.Tensor((cap,), "int32"),
        Hits: T.Tensor((NUM_AIV,), "int32"),
        vocab_start: T.int32,
        vocab_end: T.int32,
        padding_idx: T.int32,
    ):
        with T.Kernel(NUM_AIV) as core:
            id_ub = T.alloc_shared((PREP_BLK,), id_dtype)
            src_ub = T.alloc_shared((PREP_BLK,), "int32")
            dst_ub = T.alloc_shared((PREP_BLK,), "int32")
            pos = T.alloc_var("int32", init=0)
            chunk = (n + NUM_AIV - 1) // NUM_AIV
            base = core * chunk
            obase = core * PREP_BLK
            T.copy(Ids[base : base + PREP_BLK], id_ub)
            # Scalar filter + append: the compaction order is per core, so the
            # scratch writes stay inside this core's region.
            for j in T.serial(T.min(chunk, PREP_BLK)):
                t = base + j
                idv = T.cast(id_ub[j], "int32")
                if (t < n) and (idv >= vocab_start) and (idv < vocab_end) and (idv != padding_idx):
                    src_ub[pos] = t
                    dst_ub[pos] = idv - vocab_start
                    pos += 1
            T.copy(src_ub, SrcOut[obase : obase + PREP_BLK])
            T.copy(dst_ub, DstOut[obase : obase + PREP_BLK])
            Hits[core] = pos

    return main


def embedding_grad_reduce_zero_fwd():
    """Build the histogram-zeroing kernel (one launch before ``prepare``)."""
    rows = T.dynamic("rows")

    @T.prim_func
    def main(Counts: T.Tensor((rows,), "int32")):
        with T.Kernel(NUM_AIV) as core:
            chunk = (rows + NUM_AIV - 1) // NUM_AIV
            with T.SimtVF(threads=512):
                for j in T.Parallel(chunk):
                    Counts[T.min(core * chunk + j, rows - 1)] = 0

    return main


def embedding_grad_reduce_update_fwd(num_rows, H, in_dtype="bfloat16", acc_dtype="float32"):
    """Build the compacted-list accumulator kernel (sharded / filtered path).

    ``Hits[core]`` is this core's entry count, so the loop needs no tail clamp:
    every iteration is a real contribution.
    """
    esize_src = 2 if in_dtype in ("bfloat16", "float16") else 4
    esize_acc = 2 if acc_dtype in ("bfloat16", "float16") else 4
    versions, num_stages = _pipeline_config(H, esize_src, esize_acc)
    vf_threads = _vf_threads(H)
    cap = T.dynamic("cap")
    ntot = T.dynamic("ntot")

    @T.prim_func
    def main(
        SrcRows: T.Tensor((cap,), "int32"),
        DstIds: T.Tensor((cap,), "int32"),
        Hits: T.Tensor((NUM_AIV,), "int32"),
        GradOut: T.Tensor((ntot, H), in_dtype),
        GradWeight: T.Tensor((num_rows, H), acc_dtype),
    ):
        with T.Kernel(NUM_AIV) as core:
            src_ub = T.alloc_shared((H,), in_dtype)
            acc_ub = T.alloc_shared((H,), acc_dtype)
            T.annotate_buffer_versions({src_ub: versions, acc_ub: versions})
            # The scratch is cut into NUM_AIV equal PREP_BLK regions.
            base = core * (cap // NUM_AIV)
            T.set_atomic("add", acc_dtype)
            for i in T.Pipelined(Hits[core], num_stages=num_stages):
                T.copy(GradOut[SrcRows[base + i], :], src_ub)
                with T.SimtVF(threads=vf_threads):
                    for j in T.Parallel(H):
                        acc_ub[j] = T.cast(src_ub[j], acc_dtype)
                T.copy(acc_ub, GradWeight[DstIds[base + i], :])
            T.set_atomic_none()

    return main


def embedding_grad_reduce_scaled_fwd(num_rows, H, in_dtype="bfloat16", acc_dtype="float32"):
    """Same as :func:`embedding_grad_reduce_update_fwd` with ``1/count`` scaling."""
    esize_src = 2 if in_dtype in ("bfloat16", "float16") else 4
    esize_acc = 2 if acc_dtype in ("bfloat16", "float16") else 4
    versions, num_stages = _pipeline_config(H, esize_src, esize_acc)
    vf_threads = _vf_threads(H)
    cap = T.dynamic("cap")
    ntot = T.dynamic("ntot")
    rows = T.dynamic("rows")

    @T.prim_func
    def main(
        SrcRows: T.Tensor((cap,), "int32"),
        DstIds: T.Tensor((cap,), "int32"),
        Hits: T.Tensor((NUM_AIV,), "int32"),
        Counts: T.Tensor((rows,), "int32"),
        GradOut: T.Tensor((ntot, H), in_dtype),
        GradWeight: T.Tensor((num_rows, H), acc_dtype),
    ):
        with T.Kernel(NUM_AIV) as core:
            src_ub = T.alloc_shared((H,), in_dtype)
            acc_ub = T.alloc_shared((H,), acc_dtype)
            T.annotate_buffer_versions({src_ub: versions, acc_ub: versions})
            # The scratch is cut into NUM_AIV equal PREP_BLK regions.
            base = core * (cap // NUM_AIV)
            T.set_atomic("add", acc_dtype)
            for i in T.Pipelined(Hits[core], num_stages=num_stages):
                row = DstIds[base + i]
                scale = T.cast(1, "float32") / T.cast(Counts[row], "float32")
                T.copy(GradOut[SrcRows[base + i], :], src_ub)
                with T.SimtVF(threads=vf_threads):
                    for j in T.Parallel(H):
                        acc_ub[j] = T.cast(src_ub[j], acc_dtype) * scale
                T.copy(acc_ub, GradWeight[row, :])
            T.set_atomic_none()

    return main


def embedding_grad_reduce_whole_fwd(num_rows, H, in_dtype="bfloat16", acc_dtype="float32", id_dtype="int32"):
    """Build the whole-table kernel: every token is a hit, ids read in place."""
    esize_src = 2 if in_dtype in ("bfloat16", "float16") else 4
    esize_acc = 2 if acc_dtype in ("bfloat16", "float16") else 4
    versions, num_stages = _pipeline_config(H, esize_src, esize_acc)
    vf_threads = _vf_threads(H)
    n = T.dynamic("n")

    @T.prim_func
    def main(
        Ids: T.Tensor((n,), id_dtype),
        GradOut: T.Tensor((n, H), in_dtype),
        GradWeight: T.Tensor((num_rows, H), acc_dtype),
    ):
        with T.Kernel(NUM_AIV) as core:
            src_ub = T.alloc_shared((H,), in_dtype)
            acc_ub = T.alloc_shared((H,), acc_dtype)
            T.annotate_buffer_versions({src_ub: versions, acc_ub: versions})
            chunk = (n + NUM_AIV - 1) // NUM_AIV
            T.set_atomic("add", acc_dtype)
            for i in T.Pipelined(chunk, num_stages=num_stages):
                t = core * chunk + i
                tt = T.min(t, n - 1)
                # Clamped tail rows read a real row and add zero to a real row:
                # no conditional DMA or atomic store.
                g_scale = T.if_then_else(t < n, T.cast(1, "float32"), T.cast(0, "float32"))
                T.copy(GradOut[tt, :], src_ub)
                with T.SimtVF(threads=vf_threads):
                    for j in T.Parallel(H):
                        acc_ub[j] = T.cast(src_ub[j], acc_dtype) * g_scale
                T.copy(acc_ub, GradWeight[Ids[tt], :])
            T.set_atomic_none()

    return main


@tilelang.jit(target="ascend")
def _prepare_kernel(scale_by_freq, id_dtype):
    return embedding_grad_reduce_prepare_fwd(scale_by_freq, id_dtype)


@tilelang.jit(target="ascend")
def _zero_kernel():
    return embedding_grad_reduce_zero_fwd()


@tilelang.jit(target="ascend")
def _update_kernel(num_rows, H, in_dtype, acc_dtype, scale_by_freq):
    if scale_by_freq:
        return embedding_grad_reduce_scaled_fwd(num_rows, H, in_dtype, acc_dtype)
    return embedding_grad_reduce_update_fwd(num_rows, H, in_dtype, acc_dtype)


@tilelang.jit(target="ascend")
def _whole_kernel(num_rows, H, in_dtype, acc_dtype, id_dtype):
    return embedding_grad_reduce_whole_fwd(num_rows, H, in_dtype, acc_dtype, id_dtype)


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
) -> torch.Tensor:
    """Accumulate ``grad_output`` into ``grad_weight[indices]`` (fp32 table).

    ``num_weights`` is the number of rows this rank owns; with
    ``vocab_end_index`` set it defaults to ``vocab_end_index - vocab_start_index``.
    Ids outside ``[vocab_start_index, vocab_end_index)`` and ``padding_idx``
    are dropped by the ``prepare`` kernel (not by the host).

    ``out`` may be an existing fp32 ``[num_weights, H]`` table to accumulate
    into (the training-framework path, e.g. MTP reusing the same embedding);
    when omitted a zero table is allocated and returned. Returned dtype is
    fp32 (internal accumulation precision), not ``grad_output``'s dtype.

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

    device = go.device
    whole_table = (
        vocab_start_index == 0
        and vocab_end_index == num_weights
        and padding_idx < 0
        and not scale_grad_by_freq
    )
    if whole_table:
        kernel = _whole_kernel(num_weights, H, in_dtype, "float32", id_dtype)
        kernel(idx, go, out)
        return out

    assert n <= MAX_TOKENS, (
        f"this build handles at most {MAX_TOKENS} tokens per call "
        f"(NUM_AIV * PREP_BLK); got {n}"
    )
    src, dst, hits = _scratch(MAX_TOKENS, device)
    counts = _counts(num_weights, device) if scale_grad_by_freq else None
    if counts is not None:
        _zero_kernel()(counts)
        _prepare_kernel(True, id_dtype)(idx, src, dst, hits, counts,
                                        vocab_start_index, vocab_end_index, padding_idx)
        _update_kernel(num_weights, H, in_dtype, "float32", True)(src, dst, hits, counts, go, out)
    else:
        _prepare_kernel(False, id_dtype)(idx, src, dst, hits,
                                         vocab_start_index, vocab_end_index, padding_idx)
        _update_kernel(num_weights, H, in_dtype, "float32", False)(src, dst, hits, go, out)
    return out


def run_regression_perf(num_tokens=32768, H=7168, dtype="bfloat16", vocab_size=129280):
    """Full-table bandwidth regression (read each grad row + atomic add it)."""
    ids = torch.randint(0, vocab_size, (num_tokens,), dtype=torch.int32).npu()
    go = torch.randn(num_tokens, H, device="npu").to(_STR_TO_TORCH[dtype])
    gw = torch.zeros(vocab_size, H, dtype=torch.float32, device="npu")
    latency = do_bench(
        lambda: embedding_grad_reduce(go, ids, vocab_size, out=gw),
        backend="msprof",
        _n_warmup=30,
        _n_repeat=50,
    )
    # Logical traffic: one grad-row read plus one atomic read-modify-write of
    # the destination row per token; ids/src/inv are negligible.
    num_bytes = num_tokens * H * (go.element_size() + 2 * gw.element_size())
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
        lambda: embedding_grad_reduce(go, ids, vocab_end_index=end, vocab_start_index=start, out=gw),
        backend="msprof",
        _n_warmup=10,
        _n_repeat=30,
    )
    hits = int(((ids >= start) & (ids < end)).sum().item())
    num_bytes = hits * H * (go.element_size() + 2 * gw.element_size())
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
