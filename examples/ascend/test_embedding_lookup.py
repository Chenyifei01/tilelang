"""Correctness tests for the vocab-parallel embedding lookup example.

Cases map to ``embedding_lookup/design.md`` §8.1 (C1-C10). Comparisons are
bit-exact via ``torch.equal`` unless noted (design.md §5: gather is pure
address movement, no rounding).
"""

import hashlib

import pytest
import torch
import tilelang
import tilelang.testing

from example_embedding_lookup import (
    embedding_lookup,
    embedding_lookup_fwd,
    embedding_lookup_kernel,
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
    k1 = embedding_lookup_kernel(nt, shard_rows, H, "bfloat16")
    k2 = embedding_lookup_kernel(nt, shard_rows, H, "bfloat16")
    assert k1 is k2
    ids = _rand_ids(nt, 32, seed=17)
    w = _rand_weight(shard_rows, H, seed=18)
    out1 = embedding_lookup(ids, w, 8, 24)
    out2 = embedding_lookup(ids, w, 8, 24)
    assert torch.equal(out1, out2)
    assert torch.equal(out2, ref_program(ids, w, 8, 24))


@tilelang.testing.requires_ascend
@pytest.mark.parametrize(
    "H, nt", [(256, 64), (7168, 64), (7168, 66)]  # 64: block store at both H; 66: per-row tail guard
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
        kern = tilelang.compile(
            embedding_lookup_fwd(nt, shard_rows, H), out_idx=[2], target="ascend"
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
