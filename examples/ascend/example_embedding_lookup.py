"""Vocab-parallel embedding lookup (forward gather) for Ascend NPUs.

Design: ``embedding_lookup/design.md``. Pure AIV kernel on the persistent
execution model: a fixed grid iterates work tiles (``T.Persistent``), each
tile runs a token-granularity multi-buffered row pipeline (``T.Pipelined``).
A tile's rows leave in one block MTE3 (stream-copy transaction shape,
design.md §6.3 T8). When ``num_tokens`` is not a multiple of ``tile_t`` the
output is padded up to one and the wrapper slices the pad rows off, so the
block store still runs for every tile (-16% at H=4096, nt=32766); a tiny batch
whose grid is not filled keeps the per-row path, which starts storing after the
first row. For 4-12 KiB rows (the V4-Flash class) the miss load is skipped -- a
miss row is clamped to an L2-resident row and then zeroed, so the read is pure
waste; measured -39% at TP=8 with TP=1 neutral (design.md §6.3 T9). For >=12 KiB
rows the skip is not enabled: it regresses TP=1 by ~20% and gains only ~3% at
TP=8. Stores remain unconditional with respect to hit/miss; conditional MTE
*stores* are unsound under the auto-scheduler (design.md §9.1 S6), so do not
reintroduce them.
"""

import argparse

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.carver.arch.driver import get_num_vector_cores
from tilelang.profiler import do_bench

# Ascend 950 physical UB is 248 KiB (src/ascend/target_utils.h,
# kAscendUBMemoryBytes). A kernel containing T.SimtVF loses 32 KiB to the VF
# thread context (kAscendSimtContextBytes), so the scheduler's shared-memory
# limit here is 216 KiB; plan at that limit. (Measured boundary: (4,3)@H=7168 =
# 168 KiB compiles, (4,4) = 224 KiB fails with capacity -8192 B.)
UB_PHYSICAL = 248 * 1024
SIMT_VF_CONTEXT = 32 * 1024
UB_PLAN = UB_PHYSICAL - SIMT_VF_CONTEXT  # 216 KiB

# Row-byte class boundaries shared by _tile_config and skip_miss_load.
ROW_CLASS_MID = 4 * 1024
ROW_CLASS_LARGE = 12 * 1024

_DTYPE_STR = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
}
_STR_TO_TORCH = {v: k for k, v in _DTYPE_STR.items()}


def _tile_config(H, esize):
    """(tile_t, versions, num_stages) for the row pipeline (design.md §6.3 T1/T8).

    Swept optima on 950DT with the block-store fast path: row classes
    (>=12 KiB -> tt4/v3: 2843 GB/s; 4-12 KiB -> tt8/v2: 3038 GB/s -- above the
    stream-copy baseline); small rows keep a deeper pipeline (tt8/v4). The
    budget guard shrinks tile_t rather than lose double buffering (fp32 rows);
    the budget is the scheduler's 216 KiB (248 KiB physical minus the 32 KiB
    SimtVF thread context), not the raw physical size. Rows larger than
    UB_PLAN/2 fall back to a single buffer (versions=1); a row that still does
    not fit (including ids_ub) raises ValueError (no row-dimension tiling).
    Non-32B row bytes force single-row tiles (S8).
    """
    row = H * esize
    if row + 4 > UB_PLAN:
        # Even one row plus ids_ub does not fit; fail early with a clear error
        # instead of letting the Z3 scheduler report a negative capacity.
        raise ValueError(
            f"H={H} ({row} B/row) exceeds the usable UB budget "
            f"({UB_PLAN} B, including ids_ub); row-dimension tiling is not supported"
        )
    if row % 32 != 0:
        return 1, 2, 2
    if row >= ROW_CLASS_LARGE:
        tile_t, versions, num_stages = 4, 3, 3
    elif row >= ROW_CLASS_MID:
        tile_t, versions, num_stages = 8, 2, 2
    else:
        tile_t, versions, num_stages = 8, 4, 4
    if tile_t * row * versions > UB_PLAN:  # e.g. fp32 H=7168: keep (3,2,2)
        tile_t = max(1, UB_PLAN // (2 * row))
        versions, num_stages = 2, 2
    if tile_t * row * versions > UB_PLAN:
        # Very large rows (row > UB_PLAN/2): double buffering cannot fit, fall
        # back to a single buffer. Correctness is unchanged; the pipeline loses
        # the load/store overlap for these shapes.
        versions, num_stages = 1, 1
    if tile_t * row * versions + tile_t * 4 > UB_PLAN:
        # Safety net for configurations that fit one row but not the chosen
        # tile_t/versions; row-dimension tiling is out of scope.
        raise ValueError(
            f"H={H} ({row} B/row) exceeds the usable UB budget "
            f"({UB_PLAN} B, including ids_ub); row-dimension tiling is not supported"
        )
    return tile_t, versions, num_stages


@tilelang.jit(out_idx=[2], target="ascend")
def embedding_lookup_fwd(num_tokens, shard_rows, H, dtype="bfloat16", ids_dtype="int32"):
    """JIT-cached kernel per compile-time signature (design.md §3.1, §4.3).

    ``num_tokens``/``shard_rows``/``H``/``dtype``/``ids_dtype`` are compile-time
    specialized; ``vocab_start``/``vocab_end`` stay runtime scalars so one
    binary serves all TP sharding configs. The persistent grid is capped by the
    tile count, so tiny batches shrink the grid. The AIV count is queried from
    the device (950DT/950PR etc. differ) instead of being hard-coded.

    ``ids_dtype`` lets the kernel read int32 and int64 ids directly and narrow
    them in the scalar path, instead of the wrapper launching a separate cast
    kernel (measured 1.1-1.9 us of device time, plus one host dispatch, per
    call). The narrow is lossless for ``vocab < 2^31``; only dtypes outside
    {int32, int64} are still cast in the wrapper.

    A tile's rows leave in one block MTE3 (T8: +6% at H=7168, +16% at H=4096
    over the per-row store). That path needs the whole tile in range, so when
    ``num_tokens`` is not a multiple of ``tile_t`` the output is padded up to
    one and the wrapper returns ``out[:num_tokens]``: the pad rows are written
    from the planner-clamped tail ids, cost at most ``tile_t-1`` extra rows,
    and no store becomes conditional (S6 still holds). Padding only pays once
    the grid is filled (>= one tile per AIV; measured crossover at 512 tokens
    for 8 KiB rows: per-row 7.52 us vs block 4.59 us); below that the grid is
    underfilled and a per-row store that starts after the first row beats a
    block store that waits for the whole tile (nt=128: 2.41 vs 2.81 us), so
    tiny batches keep the per-row path even when ``num_tokens`` is a tile
    multiple (design.md §6.3 T16).
    """
    assert num_tokens > 0 and shard_rows > 0 and H > 0
    esize = 2 if dtype in ("bfloat16", "float16") else 4
    tile_t, versions, num_stages = _tile_config(H, esize)
    # A miss reads a clamped L2-resident row and then zeroes it: the read is
    # waste. Skipping it pays off only for 4-12 KiB rows, where the fill/store
    # pipeline can still overlap (-39% at H=4096 TP=8, neutral at TP=1). For
    # >=12 KiB rows it regresses TP=1 by ~20% and gains ~3% at TP=8.
    skip_miss_load = ROW_CLASS_MID <= H * esize < ROW_CLASS_LARGE
    num_tiles = (num_tokens + tile_t - 1) // tile_t
    aiv_count = get_num_vector_cores()
    # Block MTE3 once the grid is filled (one tile per AIV or more); the
    # measured crossover is exactly grid fill. Below it the per-row store wins
    # even for divisible num_tokens: with a single tile per core, a block store
    # that waits for the whole tile loses to a per-row store that starts after
    # the first row (nt=128 @H=4096: 2.41 vs 2.81 us; nt=512 with 64 tiles:
    # 4.59 vs 7.52 us, design.md §6.3 T16).
    block_store = num_tiles >= aiv_count
    out_rows = num_tiles * tile_t if block_store else num_tokens
    num_cores = min(aiv_count, num_tiles)

    @T.prim_func
    def main(
        Ids: T.Tensor((num_tokens,), ids_dtype),
        W: T.Tensor((shard_rows, H), dtype),
        Out: T.Tensor((out_rows, H), dtype),
        vocab_start: T.int32,
        vocab_end: T.int32,
    ):
        with T.Kernel(num_cores) as core_id:
            ids_ub = T.alloc_shared((tile_t,), ids_dtype)
            row_ub = T.alloc_shared((tile_t, H), dtype)
            T.annotate_buffer_versions({row_ub: versions})
            for tile in T.Persistent([num_tiles], num_cores, core_id):
                t0 = tile * tile_t
                # Static full-width id tile: the copy planner requires provably
                # equal src/dst extents (design.md §9.1 S1). A tail tile's
                # out-of-range lanes are clamped by the planner: with the block
                # store they only ever write pad rows the wrapper slices off,
                # and on the per-row path the guard below drops them.
                T.copy(Ids[t0 : t0 + tile_t], ids_ub)
                if block_store:
                    for i in T.Pipelined(tile_t, num_stages=num_stages):
                        tok_id = T.cast(ids_ub[i], "int32")  # no-op for int32 ids
                        local = tok_id - vocab_start
                        hit = (tok_id >= vocab_start) and (tok_id < vocab_end)
                        if skip_miss_load:  # 4-12 KiB rows: miss read is waste
                            if hit:
                                T.copy(W[T.min(T.max(local, 0), shard_rows - 1), :], row_ub[i, :])
                        else:
                            T.copy(W[T.min(T.max(local, 0), shard_rows - 1), :], row_ub[i, :])
                        if not hit:  # miss: vectorized zero (V pipe)
                            with T.SimtVF(threads=128):
                                for j in T.Parallel(H):
                                    row_ub[i, j] = T.cast(0, dtype)
                    # one block store for the whole tile (tile_t*H contiguous);
                    # out_rows covers the padded tail, so this is unconditional
                    T.copy(row_ub, Out[t0 : t0 + tile_t, :])
                else:  # tiny batch on a non-divisible token count; guard is live
                    for i in T.Pipelined(tile_t, num_stages=num_stages):
                        tk = t0 + i
                        if tk < num_tokens:  # tail-tile guard
                            tok_id = T.cast(ids_ub[i], "int32")  # no-op for int32
                            local = tok_id - vocab_start
                            hit = (tok_id >= vocab_start) and (tok_id < vocab_end)
                            if skip_miss_load:  # 4-12 KiB rows: miss read is waste
                                if hit:
                                    T.copy(W[T.min(T.max(local, 0), shard_rows - 1), :], row_ub[i, :])
                            else:
                                T.copy(W[T.min(T.max(local, 0), shard_rows - 1), :], row_ub[i, :])
                            if not hit:  # miss: vectorized zero (V pipe)
                                with T.SimtVF(threads=128):
                                    for j in T.Parallel(H):
                                        row_ub[i, j] = T.cast(0, dtype)
                            T.copy(row_ub[i, :], Out[tk, :])

    return main


def ref_program(ids, weight, vocab_start, vocab_end):
    """Golden: ids outside [vocab_start, vocab_end) produce zero rows."""
    mask = (ids >= vocab_start) & (ids < vocab_end)
    local = (ids - vocab_start).masked_fill(~mask, 0)
    out = torch.nn.functional.embedding(local, weight)
    return out * mask.unsqueeze(-1).to(weight.dtype)


def embedding_lookup(
    input_ids: torch.Tensor,
    weight: torch.Tensor,
    vocab_start_index: int = 0,
    vocab_end_index: int | None = None,
) -> torch.Tensor:
    """Gather rows of ``weight`` for ids in ``[vocab_start_index, vocab_end_index)``;
    ids outside the range produce zero rows. Returns ``(*input_ids.shape, H)``.

    Caller contract (design.md §3.2): ids in ``[0, vocab_size)``, ``weight``
    row-major contiguous on NPU with
    ``weight.shape[0] == vocab_end - vocab_start``. int32 and int64 ids are both
    read directly by the kernel (the builder specializes on the ids dtype; the
    int64 -> int32 narrow is lossless for vocab < 2^31), so no cast kernel is
    launched for the two common dtypes; other dtypes are cast here.

    The kernel works on a flat token axis: for contiguous ids ``reshape(-1)``
    is a zero-copy view; a non-contiguous input incurs one copy
    (``ids.numel() * ids.element_size()`` bytes). The wrapper does not
    synchronize, matching the other Ascend examples; callers own any required
    stream synchronization.
    """
    orig_shape = tuple(input_ids.shape)
    shard_rows, H = weight.shape
    if vocab_end_index is None:
        vocab_end_index = vocab_start_index + shard_rows
    assert shard_rows == vocab_end_index - vocab_start_index, (
        f"weight.shape[0]={shard_rows} must equal vocab_end_index - "
        f"vocab_start_index={vocab_end_index - vocab_start_index}"
    )
    assert weight.is_contiguous(), "weight must be row-major contiguous"
    dtype = _DTYPE_STR.get(weight.dtype)
    if dtype is None:
        raise TypeError(f"unsupported weight dtype: {weight.dtype}")
    # The kernel indexes a flat token axis. For contiguous [B,S] ids this is a
    # zero-copy view; a non-contiguous input triggers one copy (ids.numel() *
    # ids.element_size() bytes). int32/int64 feed the kernel directly; any
    # other dtype pays one conversion kernel here.
    ids = input_ids.reshape(-1)
    if ids.dtype == torch.int32:
        ids_dtype = "int32"
    elif ids.dtype == torch.int64:
        ids_dtype = "int64"
    else:
        ids = ids.to(torch.int32)
        ids_dtype = "int32"
    if ids.numel() == 0:  # wrapper short-circuit, no kernel launch (design.md §4.4)
        return torch.empty(orig_shape + (H,), device=weight.device, dtype=weight.dtype)
    num_tokens = ids.numel()
    kernel = embedding_lookup_fwd(num_tokens, shard_rows, H, dtype, ids_dtype)
    out = kernel(ids, weight, vocab_start_index, vocab_end_index)
    # The block-store path pads the output up to a tile multiple; slicing dim 0
    # keeps the view contiguous, so the reshape below stays zero-copy.
    return out[:num_tokens].reshape(orig_shape + (H,))


def run_regression_perf(
    num_tokens=32768, H=7168, dtype="bfloat16", vocab_size=129280, id_vocab_size=None
):
    """Bandwidth regression (design.md §6.1/§8.2).

    ``vocab_size`` is the local weight rows (TP shard); ``id_vocab_size`` is the
    global id range. Leaving it None means TP=1 (all ids hit). Passing
    ``vocab_size=16160, id_vocab_size=129280`` simulates TP=8 (12.5% hit).
    """
    id_vocab_size = vocab_size if id_vocab_size is None else id_vocab_size
    kernel = embedding_lookup_fwd(num_tokens, vocab_size, H, dtype)
    ids = torch.randint(0, id_vocab_size, (num_tokens,), dtype=torch.int32).npu()
    w = torch.randn(vocab_size, H, device="npu").to(_STR_TO_TORCH[dtype])
    latency = do_bench(
        lambda: kernel(ids, w, 0, vocab_size), backend="msprof", _n_warmup=30, _n_repeat=50
    )
    # Read = hit rows only (miss reads are clamped/L2 or skipped); write = all
    # rows; ids are int32 (4 B/token), independent of the weight dtype.
    hit_ratio = ((ids >= 0) & (ids < vocab_size)).float().mean().item()
    num_bytes = (hit_ratio * num_tokens * H + num_tokens * H) * w.element_size() + num_tokens * 4
    bandwidth = num_bytes / (latency * 1e6)
    print(
        f"embedding_lookup (num_tokens={num_tokens}, H={H}, rows={vocab_size}, "
        f"id_range={id_vocab_size}, hit={hit_ratio:.3f}, {dtype}): "
        f"{latency * 1000:.2f} us | {bandwidth:.2f} GB/s"
    )
    return latency


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bench", action=argparse.BooleanOptionalAction, default=True, help="benchmark after correctness checks"
    )
    args = parser.parse_args()
    # Mixed hit/miss sharded scenario: shard [512, 4608) of vocab 8192 (§9.1 S6 stress shape).
    num_tokens, vocab_size, vocab_start, vocab_end, H = 4096, 8192, 512, 4608, 4096
    ids = torch.randint(0, vocab_size, (num_tokens,), dtype=torch.int32).npu()
    w = torch.randn(vocab_end - vocab_start, H, device="npu").to(torch.bfloat16)
    out = embedding_lookup(ids, w, vocab_start, vocab_end)
    assert torch.equal(out, ref_program(ids, w, vocab_start, vocab_end)), "output mismatch"
    print("embedding_lookup: correctness passed")
    if args.bench:
        # TP=1 production shapes (H=4096 also exercises the conditional-load code path).
        run_regression_perf(H=4096)
        run_regression_perf()
        # TP=8 simulation: local shard 16160, global ids 129280 -> 12.5% hit,
        # exercising the miss-read skip on the 4-12 KiB row class.
        run_regression_perf(H=4096, vocab_size=16160, id_vocab_size=129280)
        # Non-divisible token count: 32766 % 8 != 0 takes the padded block-store
        # path, which must match run_regression_perf(H=4096) above.
        run_regression_perf(num_tokens=32766, H=4096, vocab_size=16160, id_vocab_size=129280)
