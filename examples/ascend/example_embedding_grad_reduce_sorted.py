"""Sorted (non-atomic) embedding gradient reduce for Ascend NPUs.

Experimental alternative to ``example_embedding_grad_reduce.py``. The atomic
example pays ``read + atomic read-modify-write`` per token (2 + 4 + 4 bytes per
element); this variant sorts the token list by id, so equal ids are adjacent,
gives every id group to exactly one core, accumulates the group on chip, and
writes the destination row once with a plain store (2 + 4 bytes per element).
A row therefore has exactly one writer, which also makes the result bit-exact
across repeated runs -- something the atomic path cannot promise.

Wrapper note: unlike the shipped atomic example (whose preprocessing now runs
inside a device kernel), this experimental variant still builds its group plan
with torch NPU ops from Python -- ``torch.sort`` plus a handful of small
metadata kernels. Moving that on device needs a counting sort written in
TileLang (histogram -> per-core prefix -> cross-core offsets -> atomic scatter);
budgeted at ~85-90 us it would cost more than the ~63 us plan it replaces, so
it is deliberately left as a host-plan experiment for now.

The price is moved into the wrapper: sorting ``n`` ids and building the group
plan is O(n log n) host-side work spread over several small NPU kernels
(~63 us at 32768 tokens, ~35 us at 4k tokens because launch overhead dominates).
For the production shape (32768 tokens, H=7168) the kernel drops from
~789-840 us to ~520 us, so end-to-end is ~584 us vs ~805-840 us for the atomic
kernel. A sharded step that hits only ~4k tokens is not worth it: the kernel
falls to ~75 us but the plan still costs ~35 us, so the atomic kernel wins
(~136 us vs ~143 us end-to-end).

Two more consequences of the design:

* The destination row is *written*, not accumulated. ``out`` must be zero-filled
  (a reused, non-zero table needs the atomic kernel, or the caller must add the
  previous contents afterwards).
* Load balance follows the group count. With ``vocab >= 56`` and few duplicates
  every core gets work; a tiny vocabulary (say 8 rows) puts almost all tokens on
  a handful of cores, where the atomic kernel is far better.

The pipeline keeps the source row multi-buffered and the accumulator
single-buffered: the accumulator is a loop-carried value, and the store has to
drain it before the next group overwrites it. The store itself is the only
data-dependent MTE instruction in the kernel, guarded by "last token of this
group"; the id group never spans two cores, so no row is written twice.
"""

import argparse

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.profiler import do_bench

from example_embedding_grad_reduce import _DTYPE_STR, _STR_TO_TORCH, ref_program

# Measured AIV count on Ascend 950DT.
NUM_AIV = 56
# UB envelope; the accumulator takes one row, the rest is read-ahead depth.
UB_PLAN = 192 * 1024

_ONES_CACHE: dict = {}


def _ones(n, device):
    """Cached float32 ones; only read by the kernel, so sharing is safe."""
    key = (n, device.index if device.index is not None else torch.npu.current_device())
    buf = _ONES_CACHE.get(key)
    if buf is None:
        buf = torch.ones(n, dtype=torch.float32, device=device)
        _ONES_CACHE[key] = buf
    return buf


def _vf_threads(H):
    """SIMT VF width for the cast row (measured in the atomic example)."""
    return 512 if H >= 512 else 128


def _pipeline_versions(H, esize_src):
    """Read-ahead depth: all but one accumulator row is spent on source rows.

    The accumulator is single-buffered (loop-carried), so the read depth can go
    deeper than the atomic kernel's. Measured at H=7168/32768 tokens: 2 -> 557us,
    3 -> 548, 4 -> 532, 6 -> 523, 8 -> 541 (deeper starts to cost scheduling).
    """
    room = UB_PLAN - H * 4  # one fp32 accumulator row
    return max(2, min(6, room // (H * esize_src)))


def embedding_grad_reduce_sorted_fwd(num_rows, H, in_dtype="bfloat16", acc_dtype="float32", versions=None):
    """Build the sorted scatter-add kernel.

    ``Ids`` is the id of every token in group order (the destination row),
    ``SrcRows`` the matching ``GradOut`` row, ``Scales`` the per-token
    ``1/count`` weight, and ``TokenStart`` the ``[NUM_AIV + 1]`` offsets that
    cut the token list into whole-group runs, one run per core.
    """
    assert num_rows > 0 and H > 0
    esize_src = 2 if in_dtype in ("bfloat16", "float16") else 4
    versions = versions or _pipeline_versions(H, esize_src)
    vf_threads = _vf_threads(H)
    n = T.dynamic("n")  # compacted tokens
    ntot = T.dynamic("ntot")  # rows of GradOut (pre-compaction token count)

    @T.prim_func
    def main(
        Ids: T.Tensor((n,), "int32"),
        TokenStart: T.Tensor((NUM_AIV + 1,), "int32"),
        SrcRows: T.Tensor((n,), "int32"),
        Scales: T.Tensor((n,), "float32"),
        GradOut: T.Tensor((ntot, H), in_dtype),
        GradWeight: T.Tensor((num_rows, H), acc_dtype),
    ):
        with T.Kernel(NUM_AIV) as core:
            src_ub = T.alloc_shared((H,), in_dtype)
            acc_ub = T.alloc_shared((H,), acc_dtype)
            T.annotate_buffer_versions({src_ub: versions, acc_ub: 1})
            # The first token of every group overwrites the whole accumulator;
            # zeroing it once keeps the buffer initialized for the read side of
            # the select and costs a single row per core.
            with T.SimtVF(threads=vf_threads):
                for j in T.Parallel(H):
                    acc_ub[j] = T.cast(0, acc_dtype)
            t0 = TokenStart[core]
            cnt = TokenStart[core + 1] - t0
            for i in T.Pipelined(cnt, num_stages=versions):
                t = t0 + i
                tt = T.min(t, n - 1)
                T.copy(GradOut[SrcRows[tt], :], src_ub)
                # Group boundaries are runtime scalars; the core range starts and
                # ends on a group boundary, so i == 0 / i == cnt - 1 force them.
                first = T.if_then_else(i == 0, True, Ids[tt] != Ids[T.max(tt - 1, 0)])
                scale = Scales[tt]
                with T.SimtVF(threads=vf_threads):
                    for j in T.Parallel(H):
                        contrib = T.cast(src_ub[j], acc_dtype) * scale
                        acc_ub[j] = T.if_then_else(first, contrib, acc_ub[j] + contrib)
                tail = T.if_then_else(i == cnt - 1, True, Ids[tt] != Ids[T.min(tt + 1, n - 1)])
                if tail:
                    # Plain store, one writer per row: no atomic, no RMW read.
                    T.copy(acc_ub, GradWeight[Ids[tt], :])

    return main


@tilelang.jit(target="ascend")
def embedding_grad_reduce_sorted_kernel(num_rows, H, in_dtype="bfloat16", acc_dtype="float32"):
    """JIT-cached kernel per compile-time signature."""
    return embedding_grad_reduce_sorted_fwd(num_rows, H, in_dtype, acc_dtype)


def sorted_plan(ids, scale_grad_by_freq=False):
    """Group plan for the sorted kernel.

    Returns ``(sorted_ids, token_start, src_rows, scales)`` in the kernel's
    argument order: ids in group order, the per-core token offsets, the matching
    ``GradOut`` rows, and the per-token ``1/count`` (all ones when frequency
    scaling is off).
    """
    n = ids.numel()
    device = ids.device
    sorted_ids, order = torch.sort(ids)
    head = torch.empty(n, dtype=torch.bool, device=device)
    head[0] = True
    head[1:] = sorted_ids[1:] != sorted_ids[:-1]
    starts = torch.nonzero(head).flatten()
    num_groups = starts.numel()
    # `starts_ext[num_groups] == n` closes the last group.
    starts_ext = torch.cat([starts, torch.tensor([n], dtype=starts.dtype, device=device)])
    # Whole groups per core; the remainder stays on the first cores.
    groups_per_core = (num_groups + NUM_AIV - 1) // NUM_AIV
    core_group = torch.clamp(torch.arange(NUM_AIV + 1, device=device) * groups_per_core, max=num_groups)
    token_start = starts_ext.index_select(0, core_group).to(torch.int32)

    if scale_grad_by_freq:
        counts = starts_ext[1:] - starts_ext[:-1]
        group_scale = (1.0 / counts).to(torch.float32)
        scales = group_scale.repeat_interleave(counts).to(torch.float32)
    else:
        scales = _ones(n, device)
    return sorted_ids.to(torch.int32), token_start, order.to(torch.int32), scales


def embedding_grad_reduce_sorted(
    grad_output: torch.Tensor,
    indices: torch.Tensor,
    num_weights: int | None = None,
    padding_idx: int = -1,
    scale_grad_by_freq: bool = False,
    vocab_start_index: int = 0,
    vocab_end_index: int | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Accumulate ``grad_output`` into ``grad_weight[indices]`` via sorted groups.

    Same contract as the atomic example, with two differences: ``out`` must be
    zero-filled (rows are written, not accumulated), and the result is bit-exact
    across repeated runs because every row has a single writer.
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

    whole_table = (
        vocab_start_index == 0
        and vocab_end_index == num_weights
        and padding_idx < 0
        and not scale_grad_by_freq
    )
    if not whole_table:
        valid = (idx >= vocab_start_index) & (idx < vocab_end_index)
        if padding_idx >= 0:
            valid &= idx != padding_idx
        hit_rows = valid.nonzero().flatten()
        ids = (idx[hit_rows].to(torch.int64) - vocab_start_index).to(torch.int32)
    else:
        hit_rows = None
        ids = idx.contiguous()

    if ids.numel() == 0:  # nothing hits this shard: no launch
        return out

    sorted_ids, token_start, permutation, scales = sorted_plan(ids, scale_grad_by_freq)
    # `permutation` orders the compacted token list by id. Without compaction it
    # already is the GradOut row order; otherwise translate it through the hits.
    if hit_rows is None:
        src_rows = permutation.to(torch.int32)
    else:
        src_rows = hit_rows.to(torch.int32).index_select(0, permutation.to(torch.int64))

    kernel = embedding_grad_reduce_sorted_kernel(num_weights, H, in_dtype, "float32")
    kernel(sorted_ids, token_start, src_rows, scales, go, out)
    return out


def run_regression_perf(num_tokens=32768, H=7168, dtype="bfloat16", vocab_size=129280):
    """Full-table end-to-end (sort plan + kernel) and kernel-only latency."""
    kernel = tilelang.compile(embedding_grad_reduce_sorted_fwd(vocab_size, H, dtype), target="ascend")
    ids = torch.randint(0, vocab_size, (num_tokens,), dtype=torch.int32).npu()
    go = torch.randn(num_tokens, H, device="npu").to(_STR_TO_TORCH[dtype])
    gw = torch.zeros(vocab_size, H, dtype=torch.float32, device="npu")
    sorted_ids, token_start, src_rows, scales = sorted_plan(ids)

    kernel_latency = do_bench(
        lambda: kernel(sorted_ids, token_start, src_rows, scales, go, gw),
        backend="msprof",
        _n_warmup=30,
        _n_repeat=50,
    )
    plan_latency = do_bench(lambda: sorted_plan(ids), backend="msprof", _n_warmup=10, _n_repeat=30)
    end_to_end = do_bench(
        lambda: embedding_grad_reduce_sorted(go, ids, vocab_size, out=gw),
        backend="msprof",
        _n_warmup=10,
        _n_repeat=30,
    )
    # Logical traffic: one grad-row read plus one plain write of the row.
    num_bytes = num_tokens * H * (go.element_size() + gw.element_size())
    print(
        f"embedding_grad_reduce_sorted (num_tokens={num_tokens}, H={H}, {dtype}): "
        f"kernel {kernel_latency * 1000:.2f} us | {num_bytes / (kernel_latency * 1e6):.2f} GB/s"
    )
    print(
        f"  plan (sort+group) {plan_latency * 1000:.2f} us | end-to-end {end_to_end * 1000:.2f} us | "
        f"{num_bytes / (end_to_end * 1e6):.2f} GB/s"
    )
    return kernel_latency


def run_sharded_perf(num_tokens=32768, H=7168, dtype="bfloat16", vocab_size=129280, tp=8):
    """TP=``tp`` simulation: end-to-end wrapper (compaction + sort + kernel)."""
    shard = vocab_size // tp
    ids = torch.randint(0, vocab_size, (num_tokens,), dtype=torch.int32).npu()
    go = torch.randn(num_tokens, H, device="npu").to(_STR_TO_TORCH[dtype])
    gw = torch.zeros(shard, H, dtype=torch.float32, device="npu")
    start, end = 3 * shard, 4 * shard  # arbitrary rank 3 of 8
    latency = do_bench(
        lambda: embedding_grad_reduce_sorted(
            go, ids, vocab_end_index=end, vocab_start_index=start, out=gw
        ),
        backend="msprof",
        _n_warmup=10,
        _n_repeat=30,
    )
    hits = int(((ids >= start) & (ids < end)).sum().item())
    num_bytes = hits * H * (go.element_size() + gw.element_size())
    print(
        f"embedding_grad_reduce_sorted shard (TP={tp}, hits={hits}/{num_tokens}, H={H}, {dtype}): "
        f"{latency * 1000:.2f} us | {num_bytes / (latency * 1e6):.2f} GB/s"
    )
    return latency


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    num_tokens, vocab_size, vocab_start, vocab_end, H = 4096, 8192, 512, 4608, 4096
    torch.manual_seed(0)
    ids = torch.randint(0, vocab_size, (num_tokens,), dtype=torch.int32).npu()
    ids[: num_tokens // 8] = ids[-num_tokens // 8 :]  # force duplicates
    go = torch.randn(num_tokens, H, device="npu").to(torch.bfloat16)
    got = embedding_grad_reduce_sorted(go, ids, padding_idx=0, vocab_start_index=vocab_start, vocab_end_index=vocab_end)
    expect = ref_program(go, ids, vocab_end - vocab_start, 0, False, vocab_start, vocab_end).npu()
    torch.testing.assert_close(got, expect, rtol=1e-5, atol=1e-5)
    print("embedding_grad_reduce_sorted: correctness passed")

    if args.bench:
        run_regression_perf()
        run_sharded_perf()
