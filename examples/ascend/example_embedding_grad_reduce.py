"""Vocab-parallel embedding gradient reduce (scatter-add) for Ascend NPUs.

Design: ``embedding_grad_reduce/design.md``. ``embedding_lookup`` gathers rows;
this operator is its reverse: for every token position ``j`` it accumulates
``grad_output[j]`` into row ``indices[j]`` of the embedding-gradient table,
so repeated ids add up. The table lives in global memory across the whole run
(``vocab * hidden`` is far larger than UB for DeepSeek V4), which rules out any
on-chip reduction of the table itself: every token is one random row read plus
one atomic row add.

The kernel is a pure AIV kernel on the persistent-style model: a fixed grid
(``T.Kernel(NUM_AIV)``) splits the (runtime, data-dependent) compacted token
list into equal contiguous chunks, and each chunk runs a token-granularity
multi-buffered row pipeline (``T.Pipelined``). The token count is dynamic so a
vocab-parallel training step -- where the number of ids that hit this rank's
shard changes every step -- reuses one compiled binary instead of recompiling.

Two disciplines carried over from ``embedding_lookup`` (see its design doc):
MTE movement stays unconditional, and conditional work stays on the V pipe.
Here that means the chunk tail is handled by clamping the token index and
scaling the row by 0 in the V pipe, never by guarding a DMA or atomic store:
conditionally dispatched MTE stores are unsound under the auto-scheduler.

Unlike the gather, the destination row is indexed at runtime and duplicate ids
must accumulate. Cross-core and intra-core duplicates are resolved by the
hardware store-mode atomic add (``T.set_atomic("add", ...)``); the call order
is therefore not fixed, so repeated runs may differ in the last fp32 ulp.
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

_DTYPE_STR = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
}
_STR_TO_TORCH = {v: k for k, v in _DTYPE_STR.items()}


_ARANGE_CACHE: dict = {}


def _arange_rows(n, device):
    """Cached int32 ``0..n-1``; the kernel never writes it, so sharing is safe."""
    key = (n, device.index if device.index is not None else torch.npu.current_device())
    rows = _ARANGE_CACHE.get(key)
    if rows is None:
        rows = torch.arange(n, dtype=torch.int32, device=device)
        _ARANGE_CACHE[key] = rows
    return rows


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


def embedding_grad_reduce_fwd(num_rows, H, in_dtype="bfloat16", acc_dtype="float32", scale_by_freq=False):
    """Build the kernel prim_func.

    ``num_rows``/``H``/``in_dtype``/``acc_dtype``/``scale_by_freq`` are
    compile-time specialized; the number of compacted tokens ``n`` and the
    number of rows of ``GradOut`` are dynamic, so one binary serves every
    step's shard hit count and every token count.

    ``Ids`` are local row indices into ``GradWeight`` (already restricted to
    the caller's vocab shard and with ``padding_idx`` removed),
    ``SrcRows`` are the matching rows of ``GradOut``, and ``InvCounts`` is the
    per-token ``1 / count`` weight (all ones when ``scale_grad_by_freq`` is
    off). The wrapper builds all three; the kernel itself has no data-dependent
    branch.
    """
    assert num_rows > 0 and H > 0
    esize_src = 2 if in_dtype in ("bfloat16", "float16") else 4
    esize_acc = 2 if acc_dtype in ("bfloat16", "float16") else 4
    versions, num_stages = _pipeline_config(H, esize_src, esize_acc)
    vf_threads = _vf_threads(H)

    n = T.dynamic("n")
    ntot = T.dynamic("ntot")

    @T.prim_func
    def main(
        Ids: T.Tensor((n,), "int32"),
        SrcRows: T.Tensor((n,), "int32"),
        InvCounts: T.Tensor((n,), "float32"),
        GradOut: T.Tensor((ntot, H), in_dtype),
        GradWeight: T.Tensor((num_rows, H), acc_dtype),
    ):
        with T.Kernel(NUM_AIV) as core:
            src_ub = T.alloc_shared((H,), in_dtype)
            acc_ub = T.alloc_shared((H,), acc_dtype)
            T.annotate_buffer_versions({src_ub: versions, acc_ub: versions})
            # Equal contiguous token chunks; the last chunks may run past the
            # compacted list, which the clamp + zero scale below neutralizes.
            chunk = (n + NUM_AIV - 1) // NUM_AIV
            T.set_atomic("add", acc_dtype)
            for i in T.Pipelined(chunk, num_stages=num_stages):
                t = core * chunk + i
                tt = T.min(t, n - 1)
                # Clamped tail rows read a real row and add zero to a real
                # row: no conditional DMA or atomic store.
                if scale_by_freq:
                    g_scale = T.if_then_else(t < n, InvCounts[tt], T.cast(0, "float32"))
                else:
                    g_scale = T.if_then_else(t < n, T.cast(1, "float32"), T.cast(0, "float32"))
                T.copy(GradOut[SrcRows[tt], :], src_ub)
                with T.SimtVF(threads=vf_threads):
                    for j in T.Parallel(H):
                        acc_ub[j] = T.cast(src_ub[j], acc_dtype) * g_scale
                # Hardware store-mode atomic: duplicate ids accumulate, order
                # across cores and tokens is not fixed (see module docstring).
                T.copy(acc_ub, GradWeight[Ids[tt], :])
            T.set_atomic_none()

    return main


@tilelang.jit(target="ascend")
def embedding_grad_reduce_kernel(num_rows, H, in_dtype="bfloat16", acc_dtype="float32", scale_by_freq=False):
    """JIT-cached kernel per compile-time signature."""
    return embedding_grad_reduce_fwd(num_rows, H, in_dtype, acc_dtype, scale_by_freq)


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
    are skipped (they belong to another rank / receive no gradient).

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

    if out is None:
        out = torch.zeros(num_weights, H, dtype=torch.float32, device=grad_output.device)
    else:
        assert out.dtype == torch.float32, "out must be fp32 (internal accumulation dtype)"
        assert out.shape == (num_weights, H), f"out must be {(num_weights, H)}, got {tuple(out.shape)}"
        assert out.is_contiguous(), "out must be row-major contiguous"

    go = grad_output.reshape(-1, H)
    idx = indices.reshape(-1)
    assert idx.numel() == go.shape[0], (
        f"indices has {idx.numel()} entries but grad_output has {go.shape[0]} rows"
    )
    if idx.dtype != torch.int32:
        idx = idx.to(torch.int32)

    # Full-table, no-padding, no-scaling fast path: the caller guarantees all
    # ids are in range, so pass them straight through (no mask / compaction).
    whole_table = (
        vocab_start_index == 0
        and vocab_end_index == num_weights
        and padding_idx < 0
        and not scale_grad_by_freq
    )
    if whole_table:
        ids = idx.contiguous()
        src_rows = _arange_rows(go.shape[0], go.device)
        inv_counts = None
    else:
        valid = (idx >= vocab_start_index) & (idx < vocab_end_index)
        if padding_idx >= 0:
            valid &= idx != padding_idx
        src_rows_i64 = valid.nonzero().flatten()
        src_rows = src_rows_i64.to(torch.int32)
        local = idx[src_rows_i64].to(torch.int64) - vocab_start_index
        ids = local.to(torch.int32)
        inv_counts = None
        if scale_grad_by_freq:
            counts = torch.zeros(num_weights, dtype=torch.float32, device=idx.device)
            counts.index_add_(0, local, torch.ones_like(ids, dtype=torch.float32))
            inv_counts = (1.0 / counts.index_select(0, local)).to(torch.float32)

    if ids.numel() == 0:  # nothing hits this shard: no launch
        return out
    if inv_counts is None:
        inv_counts = torch.ones(ids.numel(), dtype=torch.float32, device=idx.device)
    assert ids.numel() == src_rows.numel() == inv_counts.numel()

    kernel = embedding_grad_reduce_kernel(num_weights, H, in_dtype, "float32", scale_grad_by_freq)
    # Launch only: the kernel is stream-ordered with the caller's other work, so
    # no device-wide synchronize here (same convention as the other Ascend
    # examples). Consumers that read ``out`` on the host synchronize implicitly.
    kernel(ids, src_rows, inv_counts, go, out)
    return out


def run_regression_perf(num_tokens=32768, H=7168, dtype="bfloat16", vocab_size=129280):
    """Full-table bandwidth regression (read each grad row + atomic add it)."""
    kernel = tilelang.compile(
        embedding_grad_reduce_fwd(vocab_size, H, dtype, "float32"), target="ascend"
    )
    ids = torch.randint(0, vocab_size, (num_tokens,), dtype=torch.int32).npu()
    src_rows = torch.arange(num_tokens, dtype=torch.int32).npu()
    inv = torch.ones(num_tokens, dtype=torch.float32).npu()
    go = torch.randn(num_tokens, H, device="npu").to(_STR_TO_TORCH[dtype])
    gw = torch.zeros(vocab_size, H, dtype=torch.float32, device="npu")
    latency = do_bench(
        lambda: kernel(ids, src_rows, inv, go, gw), backend="msprof", _n_warmup=30, _n_repeat=50
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
    """TP=``tp`` simulation: end-to-end wrapper (host compaction + kernel)."""
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
