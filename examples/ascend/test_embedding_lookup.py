"""Correctness tests for the vocab-parallel embedding lookup example.

Cases map to ``embedding_lookup/design.md`` §8.1 (C1-C20). Comparisons are
bit-exact via ``torch.equal`` unless noted (design.md §5: gather is pure
address movement, no rounding). C14-C20 are the generalization matrix:
row-class boundaries, large H / single-buffer fallback, dtype x H, input
forms, TP shard sums, id patterns, wave boundaries, and config invariants.
"""

import hashlib

import pytest
import torch
import tilelang
import tilelang.testing

from example_embedding_lookup import (
    SIMT_VF_CONTEXT,
    UB_PHYSICAL,
    UB_PLAN,
    _tile_config,
    embedding_lookup,
    embedding_lookup_fwd,
    ref_program,
)


def _rand_ids(num_tokens, vocab, seed=0):
    torch.manual_seed(seed)
    return torch.randint(0, vocab, (num_tokens,), dtype=torch.int32).npu()


def _rand_weight(shard_rows, H, dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed + 1)
    return torch.randn(shard_rows, H, device="npu").to(dtype)


@tilelang.testing.requires_ascend
def test_c1_full_table_gather():
    nt, vocab, H = 64, 32, 256
    ids, w = _rand_ids(nt, vocab), _rand_weight(vocab, H)
    out = embedding_lookup(ids, w, 0, vocab)
    assert out.shape == (nt, H)
    assert torch.equal(out, ref_program(ids, w, 0, vocab))


@tilelang.testing.requires_ascend
def test_c2_boundary_ids():
    vocab, vocab_start, vocab_end, H = 32, 8, 24, 256
    boundary = torch.tensor([0, vocab - 1, vocab_start, vocab_end - 1, vocab_end, vocab_start - 1], dtype=torch.int32)
    ids = torch.cat([boundary.npu(), _rand_ids(58, vocab, seed=2)])
    w = _rand_weight(vocab_end - vocab_start, H, seed=3)
    out = embedding_lookup(ids, w, vocab_start, vocab_end)
    assert torch.equal(out, ref_program(ids, w, vocab_start, vocab_end))


@tilelang.testing.requires_ascend
def test_c3_tp4_shard_sum():
    """TP=4 simulation: per-shard lookups summed reconstruct the full table (README §3)."""
    vocab, shard_rows, H, nt = 64, 16, 128, 64
    torch.manual_seed(4)
    full_w = torch.randn(vocab, H, device="npu").to(torch.bfloat16)
    ids = torch.randint(0, vocab, (nt,), dtype=torch.int32).npu()
    total = torch.zeros(nt, H, device="npu").to(torch.bfloat16)
    for rank in range(4):
        start = rank * shard_rows
        total += embedding_lookup(ids, full_w[start : start + shard_rows], start, start + shard_rows)
    expect = torch.nn.functional.embedding(ids, full_w)
    torch.testing.assert_close(total, expect, rtol=1e-5, atol=1e-5)


@tilelang.testing.requires_ascend
def test_c4_duplicate_ids():
    vocab, vocab_start, vocab_end, H = 32, 4, 28, 128
    base = _rand_ids(16, vocab, seed=5)
    ids = torch.cat([base, base, base])  # every id repeated 3x
    w = _rand_weight(vocab_end - vocab_start, H, seed=6)
    out = embedding_lookup(ids, w, vocab_start, vocab_end)
    assert torch.equal(out, ref_program(ids, w, vocab_start, vocab_end))


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("num_tokens", [1, 7, 8, 9])  # tile_t=8 at H=64 bf16: 7/9 straddle the tail
def test_c5_tail_tiles(num_tokens):
    vocab, vocab_start, vocab_end, H = 32, 8, 24, 64
    ids = _rand_ids(num_tokens, vocab, seed=7)
    w = _rand_weight(vocab_end - vocab_start, H, seed=8)
    out = embedding_lookup(ids, w, vocab_start, vocab_end)
    assert torch.equal(out, ref_program(ids, w, vocab_start, vocab_end))


@tilelang.testing.requires_ascend
def test_c5_empty_tokens():
    """num_tokens=0: wrapper short-circuits without launching (design.md §4.4)."""
    out = embedding_lookup(torch.empty(0, dtype=torch.int32).npu(), _rand_weight(16, 64), 8, 24)
    assert out.shape == (0, 64)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_c6_dtypes(dtype):
    nt, vocab, vocab_start, vocab_end, H = 32, 16, 4, 12, 128
    ids = _rand_ids(nt, vocab, seed=9)
    w = _rand_weight(vocab_end - vocab_start, H, dtype, seed=10)
    out = embedding_lookup(ids, w, vocab_start, vocab_end)
    assert torch.equal(out, ref_program(ids, w, vocab_start, vocab_end))


@tilelang.testing.requires_ascend
def test_c6_int64_ids_via_api():
    """int64 ids are converted to int32 in the wrapper (design.md §3.2)."""
    nt, vocab, vocab_start, vocab_end, H = 32, 16, 4, 12, 128
    ids64 = _rand_ids(nt, vocab, seed=11).to(torch.int64)
    w = _rand_weight(vocab_end - vocab_start, H, seed=12)
    out = embedding_lookup(ids64, w, vocab_start, vocab_end)
    assert torch.equal(out, ref_program(ids64.to(torch.int32), w, vocab_start, vocab_end))


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("H", [4080, 4095])
def test_c7_unaligned_H(H):
    """Odd H: 4080 has 32B-aligned rows (full path); 4095 does not and the
    kernel degrades to single-row tiles to avoid pitched DMA (design.md §4.5)."""
    nt, vocab, vocab_start, vocab_end = 64, 32, 8, 24
    ids = _rand_ids(nt, vocab, seed=13)
    w = _rand_weight(vocab_end - vocab_start, H, seed=14)
    out = embedding_lookup(ids, w, vocab_start, vocab_end)
    assert torch.equal(out, ref_program(ids, w, vocab_start, vocab_end))


@tilelang.testing.requires_ascend
def test_c8_2d_ids_interface():
    vocab, vocab_start, vocab_end, H = 32, 8, 24, 128
    ids2d = _rand_ids(64, vocab, seed=15).reshape(2, 32)
    w = _rand_weight(vocab_end - vocab_start, H, seed=16)
    out = embedding_lookup(ids2d, w, vocab_start, vocab_end)
    assert out.shape == (2, 32, H)
    expect = ref_program(ids2d.reshape(-1), w, vocab_start, vocab_end).reshape(2, 32, H)
    assert torch.equal(out, expect)


@tilelang.testing.requires_ascend
def test_c9_compile_cache_reuse():
    """MTP-style double lookup reuses the cached kernel, no recompile (design.md C9)."""
    nt, shard_rows, H = 32, 16, 128
    k1 = embedding_lookup_fwd(nt, shard_rows, H, "bfloat16")
    k2 = embedding_lookup_fwd(nt, shard_rows, H, "bfloat16")
    assert k1 is k2
    ids = _rand_ids(nt, 32, seed=17)
    w = _rand_weight(shard_rows, H, seed=18)
    out1 = embedding_lookup(ids, w, 8, 24)
    out2 = embedding_lookup(ids, w, 8, 24)
    assert torch.equal(out1, out2)
    assert torch.equal(out2, ref_program(ids, w, 8, 24))


@tilelang.testing.requires_ascend
@pytest.mark.parametrize(
    "H, nt",
    [
        (256, 64),   # unconditional-load class (small rows)
        (4096, 64),  # conditional miss-load class, block store
        (4096, 66),  # conditional miss-load class, per-row tail guard
        (7168, 64),  # unconditional-load class, block store
        (7168, 66),  # unconditional-load class, per-row tail guard
    ],
)
def test_c10_repeat_compile_regression(H, nt, monkeypatch):
    """Gate against schedule variance (design.md §8.1 C10 / §9.1 S6): repeated
    compiles of the same source must all be bit-exact; the per-compile source
    hash is recorded so a failure attributes to a specific binary. Covers both
    store paths (design.md §4.3): nt divisible by tile_t -> block store, else
    per-row with the tail guard active. The kernel disk cache is disabled so
    every rep is a genuine recompile -- with the cache on, reps 2+ would be
    cache hits and gate nothing."""
    monkeypatch.setenv("TILELANG_DISABLE_CACHE", "1")
    shard_rows, vocab, vocab_start, vocab_end = 16, 32, 3, 19
    for rep in range(3):
        # ``get_tir`` bypasses the JIT builder's in-memory kernel cache, so each
        # rep is a genuine frontend + compile (the disk cache is disabled above).
        kern = tilelang.compile(
            embedding_lookup_fwd.get_tir(nt, shard_rows, H), out_idx=[2], target="ascend"
        )
        src_hash = hashlib.sha256(kern.get_kernel_source().encode()).hexdigest()[:12]
        for r in range(3):
            torch.manual_seed(r)
            ids = torch.randint(0, vocab, (nt,), dtype=torch.int32).npu()
            w = torch.randn(shard_rows, H, device="npu").to(torch.bfloat16)
            out = kern(ids, w, vocab_start, vocab_end)
            torch.npu.synchronize()
            assert out.cpu().equal(ref_program(ids, w, vocab_start, vocab_end).cpu()), (
                f"H={H} nt={nt} rep={rep} src_hash={src_hash} run={r}"
            )
        print(f"C10 H={H} nt={nt} rep={rep}: src_hash={src_hash} 3/3 bit-exact")


@tilelang.testing.requires_ascend
def test_c11_persistent_multi_wave():
    """num_tiles > num_cores: the persistent outer loop wraps waves (production
    runs 8192 tiles / 56 cores = 146 waves; C1-C10 never exceed one wave).
    nt=1024 at tile_t=8 gives 128 tiles on 56 cores -- 16 cores take 3 tiles,
    40 take 2 (uneven distribution exercised). Mixed both-side misses."""
    vocab, shard_rows, vocab_start, vocab_end, H, nt = 64, 48, 8, 56, 64, 1024
    ids = _rand_ids(nt, vocab, seed=19)
    w = _rand_weight(shard_rows, H, seed=20)
    out = embedding_lookup(ids, w, vocab_start, vocab_end)
    assert torch.equal(out, ref_program(ids, w, vocab_start, vocab_end))


def test_c12_tile_config_table():
    """_tile_config is correctness-critical (UB capacity) and host-only, so the
    documented table (design.md §4.2) is asserted directly, plus two invariants:
    every config fits UB_PLAN, and non-32B rows force single-row tiles."""
    # Ascend 950: 248 KiB physical UB, 32 KiB reserved for the SimtVF thread
    # context (src/ascend/transform/auto_schedule.cc GetSharedMemoryLimit).
    assert UB_PHYSICAL == 248 * 1024
    assert SIMT_VF_CONTEXT == 32 * 1024
    assert UB_PLAN == 216 * 1024
    assert _tile_config(7168, 2) == (4, 3, 3)  # V4-Pro bf16
    assert _tile_config(4096, 2) == (8, 2, 2)  # V4-Flash bf16
    assert _tile_config(4080, 2) == (8, 2, 2)  # aligned odd H, 4-12 KiB class
    assert _tile_config(256, 2) == (8, 4, 4)
    assert _tile_config(64, 2) == (8, 4, 4)
    assert _tile_config(4095, 2) == (1, 2, 2)  # non-32B rows: single-row tiles
    assert _tile_config(7168, 4) == (3, 2, 2)  # budget guard shrinks fp32 rows
    assert _tile_config(4096, 4) == (4, 3, 3)  # 192 KiB, inside the 216 KiB budget
    for H in [1, 15, 64, 255, 256, 1024, 2048, 3584, 4080, 4095, 4096, 7168, 8192]:
        for esize in [2, 4]:
            tile_t, versions, num_stages = _tile_config(H, esize)
            row = H * esize
            assert versions >= 1 and num_stages == versions
            assert tile_t * row * versions + tile_t * 4 <= UB_PLAN, (H, esize)
            if row % 32 != 0:
                assert tile_t == 1, (H, esize)


def test_c13_wrapper_contract_rejects():
    """Caller contract enforcement (design.md §3.2): non-contiguous weight,
    vocab_end mismatch, and unsupported dtypes must raise before any launch."""
    H = 256
    w = torch.randn(16, H, device="npu").to(torch.bfloat16)
    ids = _rand_ids(8, 32, seed=21)
    with pytest.raises(AssertionError):
        embedding_lookup(ids, w[:, ::2], 0, 16)  # non-contiguous weight
    with pytest.raises(AssertionError):
        embedding_lookup(ids, w, 0, 100)  # vocab_end != vocab_start + shard_rows
    with pytest.raises(TypeError):
        # int table: unsupported dtype rejected. (torch.float64 is silently cast
        # to float32 by the NPU device itself, so it cannot probe this path.)
        embedding_lookup(ids, w.to(torch.int32), 0, 16)


# --------------------------------------------------------------------------
# Generalization matrix (C14-C20)
# --------------------------------------------------------------------------


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("H", [2047, 2048, 6143, 6144, 8192, 16384, 32768, 65536])
def test_c14_row_class_boundaries(H):
    """Row-byte class boundaries and large rows.

    2048/6144 are exactly 4 KiB/12 KiB rows; 2047/6143 are their non-32B
    neighbours (single-row tiles). 8192/16384 exercise the budget guard;
    32768 uses 2 versions at tile_t=1; 65536 (128 KiB row) uses the
    single-buffer fallback (versions=1)."""
    nt, vocab, start, end = 64, 32, 8, 24
    ids = _rand_ids(nt, vocab, seed=31)
    w = _rand_weight(end - start, H, seed=32)
    out = embedding_lookup(ids, w, start, end)
    assert out.shape == (nt, H)
    assert torch.equal(out, ref_program(ids, w, start, end))


@tilelang.testing.requires_ascend
@pytest.mark.parametrize(
    "dtype, H",
    [
        (torch.float16, 7168),
        (torch.float32, 4096),
        (torch.float32, 7168),  # budget guard -> (3,2,2), per-row for nt=64
        (torch.float32, 8192),
    ],
)
def test_c15_dtype_large_H(dtype, H):
    """dtype x large H: fp16/fp32 beyond the C6 H=128 case."""
    nt, vocab, start, end = 64, 32, 8, 24
    ids = _rand_ids(nt, vocab, seed=33)
    w = _rand_weight(end - start, H, dtype, seed=34)
    out = embedding_lookup(ids, w, start, end)
    assert out.dtype == dtype
    assert torch.equal(out, ref_program(ids, w, start, end))


@tilelang.testing.requires_ascend
def test_c16_input_forms():
    """3-D ids, 0-D id, non-contiguous 2-D ids, and empty batches."""
    vocab, start, end, H = 32, 8, 24, 128
    w = _rand_weight(end - start, H, seed=41)

    ids3 = _rand_ids(24, vocab, seed=42).reshape(2, 3, 4)
    out3 = embedding_lookup(ids3, w, start, end)
    assert out3.shape == (2, 3, 4, H)
    assert torch.equal(out3, ref_program(ids3.reshape(-1), w, start, end).reshape(2, 3, 4, H))

    id0 = torch.tensor(10, dtype=torch.int32).npu()
    out0 = embedding_lookup(id0, w, start, end)
    assert out0.shape == (H,)
    assert torch.equal(out0, ref_program(id0.reshape(1), w, start, end).reshape(H))

    ids2 = _rand_ids(64, vocab, seed=43).reshape(2, 32).t()  # [32,2], non-contiguous
    assert not ids2.is_contiguous()
    out2 = embedding_lookup(ids2, w, start, end)
    assert out2.shape == (32, 2, H)
    assert torch.equal(out2, ref_program(ids2.reshape(-1), w, start, end).reshape(32, 2, H))

    empty = torch.empty(2, 0, dtype=torch.int32).npu()
    assert embedding_lookup(empty, w, start, end).shape == (2, 0, H)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("tp", [2, 4, 8])
def test_c17_shard_sum(tp):
    """TP=2/4/8: per-shard lookups summed reconstruct the full table."""
    vocab, H, nt = 1024, 64, 256
    shard = vocab // tp
    torch.manual_seed(51)
    full_w = torch.randn(vocab, H, device="npu").to(torch.bfloat16)
    ids = torch.randint(0, vocab, (nt,), dtype=torch.int32).npu()
    total = torch.zeros(nt, H, device="npu").to(torch.bfloat16)
    for rank in range(tp):
        s = rank * shard
        total += embedding_lookup(ids, full_w[s : s + shard], s, s + shard)
    torch.testing.assert_close(total, torch.nn.functional.embedding(ids, full_w), rtol=1e-5, atol=1e-5)


@tilelang.testing.requires_ascend
def test_c17_production_tp8_shard():
    """Production TP=8 shard size: 129280 / 8 = 16160 rows."""
    vocab, shard_rows, H, nt = 129280, 16160, 64, 512
    torch.manual_seed(52)
    full_w = torch.randn(vocab, H, device="npu").to(torch.bfloat16)
    ids = torch.randint(0, vocab, (nt,), dtype=torch.int32).npu()
    total = torch.zeros(nt, H, device="npu").to(torch.bfloat16)
    for rank in range(8):
        s = rank * shard_rows
        total += embedding_lookup(ids, full_w[s : s + shard_rows], s, s + shard_rows)
    torch.testing.assert_close(total, torch.nn.functional.embedding(ids, full_w), rtol=1e-5, atol=1e-5)


@tilelang.testing.requires_ascend
def test_c17_shard_edge_ranges():
    """All-miss shard (zero rows) and a one-row shard at the table tail."""
    vocab, H = 128, 64
    w = _rand_weight(16, H, seed=53)
    ids = torch.randint(16, vocab, (64,), dtype=torch.int32).npu()  # all outside [0,16)
    out = embedding_lookup(ids, w, 0, 16)
    assert torch.equal(out, torch.zeros_like(out))

    w1 = _rand_weight(1, H, seed=54)
    ids1 = torch.tensor([vocab - 1, 0, vocab - 1, 5], dtype=torch.int32).npu()
    out1 = embedding_lookup(ids1, w1, vocab - 1, vocab)
    assert torch.equal(out1, ref_program(ids1, w1, vocab - 1, vocab))


@tilelang.testing.requires_ascend
def test_c18_id_patterns():
    """All-hit, all-miss, alternating, duplicates, and defensive out-of-range ids."""
    vocab, start, end, H = 64, 8, 56, 128
    w = _rand_weight(end - start, H, seed=55)
    patterns = {
        "all_hit": torch.full((64,), 20, dtype=torch.int32),
        "all_miss": torch.zeros(64, dtype=torch.int32),
        "alternating": torch.tensor([20 if i % 2 else 0 for i in range(64)], dtype=torch.int32),
        "out_of_range": torch.tensor([-1, -100, vocab, vocab + 100, start - 1, end, 0, 1], dtype=torch.int32),
        "duplicates": torch.tensor([20] * 64, dtype=torch.int32),
    }
    for name, ids in patterns.items():
        ids = ids.npu()
        out = embedding_lookup(ids, w, start, end)
        assert torch.equal(out, ref_program(ids, w, start, end)), name


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("nt", [447, 448, 449])
def test_c19_wave_boundaries(nt):
    """56 cores x tile_t=8: 447/448/449 tokens hit the last partial wave,
    an exact single wave, and the first two-wave case."""
    vocab, start, end, H = 64, 8, 56, 64
    ids = _rand_ids(nt, vocab, seed=56)
    w = _rand_weight(end - start, H, seed=57)
    out = embedding_lookup(ids, w, start, end)
    assert torch.equal(out, ref_program(ids, w, start, end))


def test_c20_tile_config_boundaries():
    """Host-only config table at the class boundaries and the very-large-row
    fallback/limit introduced for H=65536 (128 KiB row)."""
    assert _tile_config(16, 2) == (8, 4, 4)
    assert _tile_config(15, 2) == (1, 2, 2)   # 30 B row: non-32B
    assert _tile_config(17, 2) == (1, 2, 2)   # 34 B row: non-32B
    assert _tile_config(2048, 2) == (8, 2, 2)  # exactly 4 KiB
    assert _tile_config(2047, 2) == (1, 2, 2)  # 4094 B: non-32B, below 4 KiB
    assert _tile_config(6144, 2) == (4, 3, 3)  # exactly 12 KiB
    assert _tile_config(6143, 2) == (1, 2, 2)  # 12286 B: non-32B, below 12 KiB
    assert _tile_config(8192, 2) == (4, 3, 3)
    assert _tile_config(16384, 2) == (3, 2, 2)  # budget guard
    assert _tile_config(32768, 2) == (1, 2, 2)  # one row x 2 versions fits
    assert _tile_config(65536, 2) == (1, 1, 1)  # 128 KiB row: single buffer
    assert _tile_config(32768, 4) == (1, 1, 1)  # 128 KiB fp32 row: single buffer
    with pytest.raises(ValueError):
        _tile_config(110592, 2)  # 216 KiB row + ids does not fit
    with pytest.raises(ValueError):
        _tile_config(55296, 4)  # 216 KiB fp32 row + ids does not fit

    for H in [1, 15, 16, 17, 31, 32, 33, 64, 255, 256, 1024, 2047, 2048, 4095, 4096,
              6143, 6144, 7168, 8192, 16384, 32768, 65536]:
        for esize in (2, 4):
            try:
                tile_t, versions, num_stages = _tile_config(H, esize)
            except ValueError:
                assert H * esize + 4 > UB_PLAN, (H, esize)  # only legal when one row cannot fit
                continue
            row = H * esize
            assert versions >= 1 and num_stages == versions
            assert tile_t * row * versions + tile_t * 4 <= UB_PLAN, (H, esize)
            if row % 32 != 0:
                assert tile_t == 1, (H, esize)
