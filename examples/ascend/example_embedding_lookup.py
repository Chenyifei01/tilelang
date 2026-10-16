"""Vocab-parallel embedding lookup (forward gather) for Ascend NPUs.

Design: ``embedding_lookup/design.md``. Pure AIV kernel on the persistent
execution model: a fixed grid iterates work tiles (``T.Persistent``), each
tile runs a token-granularity multi-buffered row pipeline (``T.Pipelined``).
When ``num_tokens`` is divisible by ``tile_t`` (compile-time) the tile's rows
leave in one block MTE3 (stream-copy transaction shape, design.md §6.3 T8);
otherwise rows store per-row under a tail guard. Loads and stores are
unconditional with respect to hit/miss; only the miss zero-fill is a
conditional V-pipe op -- conditionally dispatched MTE stores are unsound
under the auto-scheduler (design.md §9.1 S6), so do not reintroduce them.
"""

import argparse

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.profiler import do_bench

# Measured AIV count on Ascend 950DT (design.md §1.4); the persistent grid is
# additionally capped by the tile count, so tiny batches shrink the grid.
NUM_AIV = 56
# Verified-working UB envelope for row_ub versioning (design.md §6.3 T1 sweep:
# allocations up to 192 KiB compile and run; the ~216 KiB hardware limit leaves
# headroom for ids_ub and scheduler reservations).
UB_PLAN = 192 * 1024

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
    budget guard shrinks tile_t rather than lose double buffering (fp32 rows).
    Non-32B row bytes force single-row tiles (S8).
    """
    row = H * esize
    if row % 32 != 0:
        return 1, 2, 2
    if row >= 12 * 1024:
        tile_t, versions, num_stages = 4, 3, 3
    elif row >= 4 * 1024:
        tile_t, versions, num_stages = 8, 2, 2
    else:
        tile_t, versions, num_stages = 8, 4, 4
    if tile_t * row * versions > UB_PLAN:  # e.g. fp32 H=7168: keep (tile_t,2)
        tile_t = max(1, UB_PLAN // (2 * row))
        versions, num_stages = 2, 2
    return tile_t, versions, num_stages


def embedding_lookup_fwd(num_tokens, shard_rows, H, dtype="bfloat16"):
    """Build the kernel prim_func (design.md §4.3).

    ``num_tokens``/``shard_rows``/``H``/``dtype`` are compile-time specialized;
    ``vocab_start``/``vocab_end`` stay runtime scalars so one binary serves all
    TP sharding configs. When ``num_tokens`` is divisible by ``tile_t`` the
    tail guard is compile-time dead and the tile's rows leave in one block
    MTE3 (T8: +6% at H=7168, +16% at H=4096 over the per-row store); otherwise
    the per-row store path is used.
    """
    assert num_tokens > 0 and shard_rows > 0 and H > 0
    esize = 2 if dtype in ("bfloat16", "float16") else 4
    tile_t, versions, num_stages = _tile_config(H, esize)
    block_store = num_tokens % tile_t == 0  # no tail tile: block MTE3 is safe
    num_tiles = (num_tokens + tile_t - 1) // tile_t
    num_cores = min(NUM_AIV, num_tiles)

    @T.prim_func
    def main(
        Ids: T.Tensor((num_tokens,), "int32"),
        W: T.Tensor((shard_rows, H), dtype),
        Out: T.Tensor((num_tokens, H), dtype),
        vocab_start: T.int32,
        vocab_end: T.int32,
    ):
        with T.Kernel(num_cores) as core_id:
            ids_ub = T.alloc_shared((tile_t,), "int32")
            row_ub = T.alloc_shared((tile_t, H), dtype)
            T.annotate_buffer_versions({row_ub: versions})
            for tile in T.Persistent([num_tiles], num_cores, core_id):
                t0 = tile * tile_t
                # Static full-width id tile: the copy planner requires provably
                # equal src/dst extents (design.md §9.1 S1). The tail tile's
                # out-of-range lanes are clamped by the planner and masked by
                # the guard below.
                T.copy(Ids[t0 : t0 + tile_t], ids_ub)
                if block_store:
                    for i in T.Pipelined(tile_t, num_stages=num_stages):
                        tok_id = ids_ub[i]
                        local = tok_id - vocab_start
                        hit = (tok_id >= vocab_start) and (tok_id < vocab_end)
                        T.copy(W[T.min(T.max(local, 0), shard_rows - 1), :], row_ub[i, :])
                        if not hit:  # miss: vectorized zero (V pipe)
                            with T.SimtVF(threads=128):
                                for j in T.Parallel(H):
                                    row_ub[i, j] = T.cast(0, dtype)
                    # one block store for the whole tile (tile_t*H contiguous)
                    T.copy(row_ub, Out[t0 : t0 + tile_t, :])
                else:
                    for i in T.Pipelined(tile_t, num_stages=num_stages):
                        tk = t0 + i
                        if tk < num_tokens:  # tail-tile guard
                            tok_id = ids_ub[i]
                            local = tok_id - vocab_start
                            hit = (tok_id >= vocab_start) and (tok_id < vocab_end)
                            T.copy(W[T.min(T.max(local, 0), shard_rows - 1), :], row_ub[i, :])
                            if not hit:  # miss: vectorized zero (V pipe)
                                with T.SimtVF(threads=128):
                                    for j in T.Parallel(H):
                                        row_ub[i, j] = T.cast(0, dtype)
                            T.copy(row_ub[i, :], Out[tk, :])

    return main


@tilelang.jit(out_idx=[2], target="ascend")
def embedding_lookup_kernel(num_tokens, shard_rows, H, dtype="bfloat16"):
    """JIT-cached kernel per compile-time signature (design.md §3.1)."""
    return embedding_lookup_fwd(num_tokens, shard_rows, H, dtype)


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

    Caller contract (design.md §3.2): ids in ``[0, vocab_size)`` (int64 inputs
    are converted to int32 losslessly for vocab < 2^31), ``weight`` row-major
    contiguous on NPU with ``weight.shape[0] == vocab_end - vocab_start``.
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
    ids = input_ids.reshape(-1)
    if ids.dtype != torch.int32:
        ids = ids.to(torch.int32)
    if ids.numel() == 0:  # wrapper short-circuit, no kernel launch (design.md §4.4)
        return torch.empty(orig_shape + (H,), device=weight.device, dtype=weight.dtype)
    kernel = embedding_lookup_kernel(ids.numel(), shard_rows, H, dtype)
    out = kernel(ids, weight, vocab_start_index, vocab_end_index)
    torch.npu.synchronize()
    return out.reshape(orig_shape + (H,))


def run_regression_perf(num_tokens=32768, H=7168, dtype="bfloat16", vocab_size=129280):
    """Full-table (TP=1) bandwidth regression (design.md §6.1/§8.2)."""
    kernel = tilelang.compile(
        embedding_lookup_fwd(num_tokens, vocab_size, H, dtype), out_idx=[2], target="ascend"
    )
    ids = torch.randint(0, vocab_size, (num_tokens,), dtype=torch.int32).npu()
    w = torch.randn(vocab_size, H, device="npu").to(_STR_TO_TORCH[dtype])
    latency = do_bench(
        lambda: kernel(ids, w, 0, vocab_size), backend="msprof", _n_warmup=30, _n_repeat=50
    )
    # TP=1 full table: every id hits -- one row read + one row written per
    # token; the 4 B/token id traffic is negligible but counted.
    num_bytes = (2 * num_tokens * H + num_tokens) * w.element_size()
    bandwidth = num_bytes / (latency * 1e6)
    print(
        f"embedding_lookup (num_tokens={num_tokens}, H={H}, {dtype}): "
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
        run_regression_perf()
