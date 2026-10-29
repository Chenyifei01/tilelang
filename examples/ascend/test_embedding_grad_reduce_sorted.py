"""Correctness tests for the sorted (non-atomic) embedding gradient reduce.

The sorted kernel writes every destination row exactly once, so on top of the
usual golden comparison these cases also assert bit-exact repeatability -- the
property the atomic kernel cannot provide.
"""

import pytest
import torch

import tilelang
import tilelang.testing

from example_embedding_grad_reduce import ref_program
from example_embedding_grad_reduce_sorted import (
    NUM_AIV,
    embedding_grad_reduce_sorted,
    sorted_plan,
)


def _rand_ids(num_tokens, vocab, seed=0):
    torch.manual_seed(seed)
    return torch.randint(0, vocab, (num_tokens,), dtype=torch.int32).npu()


def _rand_grad(num_tokens, H, dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed + 1)
    return torch.randn(num_tokens, H, device="npu").to(dtype)


def _expect(go, ids, num_weights, padding_idx=-1, scale=False, start=0, end=None):
    return ref_program(go, ids, num_weights, padding_idx, scale, start, end).npu()


def _assert_rows(got, expect, tol=0.0):
    if tol == 0.0:
        assert torch.equal(got, expect), (got - expect).abs().max().item()
    else:
        torch.testing.assert_close(got, expect, rtol=tol, atol=tol)


@tilelang.testing.requires_ascend
def test_c1_full_table_scatter():
    """Distinct ids: every row is exactly one grad row placed by its id."""
    nt, vocab, H = 128, 256, 128
    torch.manual_seed(0)
    ids = torch.randperm(vocab)[:nt].to(torch.int32).npu()
    go = _rand_grad(nt, H)
    got = embedding_grad_reduce_sorted(go, ids, vocab)
    assert got.shape == (vocab, H) and got.dtype == torch.float32
    _assert_rows(got, _expect(go, ids, vocab))


@tilelang.testing.requires_ascend
def test_c2_duplicate_ids_accumulate():
    """Repeated ids are summed on chip before the single store."""
    vocab, H = 32, 64
    base = _rand_ids(64, vocab, seed=2)
    ids = torch.cat([base, base, base])
    go = _rand_grad(ids.numel(), H, seed=3)
    got = embedding_grad_reduce_sorted(go, ids, vocab)
    _assert_rows(got, _expect(go, ids, vocab), tol=1e-5)


@tilelang.testing.requires_ascend
def test_c3_shard_and_padding():
    """Out-of-shard and padding ids are dropped before sorting."""
    vocab, start, end, H = 32, 8, 24, 256
    pad = start  # padding id inside the shard, so the skip is not the range filter
    boundary = torch.tensor([0, vocab - 1, start, end - 1, end, start - 1, -1], dtype=torch.int32)
    ids = torch.cat([boundary.npu(), _rand_ids(57, vocab, seed=4)])
    ids[7:23] = pad  # frequent padding id
    go = _rand_grad(ids.numel(), H, seed=5)
    got = embedding_grad_reduce_sorted(
        go, ids, padding_idx=pad, vocab_end_index=end, vocab_start_index=start
    )
    expect = _expect(go, ids, end - start, padding_idx=pad, start=start, end=end)
    assert torch.count_nonzero(got[pad - start]) == 0
    _assert_rows(got, expect, tol=1e-6)


@tilelang.testing.requires_ascend
def test_c4_scale_grad_by_freq():
    """Each group is divided by its own id count."""
    vocab, H = 24, 64
    ids = _rand_ids(80, vocab, seed=14)
    ids = torch.cat([ids, ids[:16]])
    go = _rand_grad(ids.numel(), H, seed=15)
    got = embedding_grad_reduce_sorted(go, ids, vocab, scale_grad_by_freq=True)
    _assert_rows(got, _expect(go, ids, vocab, scale=True), tol=1e-6)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_c5_dtypes(dtype):
    nt, vocab, H = 200, 32, 96
    ids = _rand_ids(nt, vocab, seed=10)
    go = _rand_grad(nt, H, dtype, seed=11)
    got = embedding_grad_reduce_sorted(go, ids, vocab)
    assert got.dtype == torch.float32
    _assert_rows(got, _expect(go, ids, vocab), tol=1e-5)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("num_tokens", [1, 2, 7, 55, 56, 57, 999])
def test_c6_token_sweep(num_tokens):
    """Token counts below, at and above the core count."""
    vocab, H = 1024, 64
    ids = _rand_ids(num_tokens, vocab, seed=18)
    go = _rand_grad(num_tokens, H, seed=19)
    got = embedding_grad_reduce_sorted(go, ids, vocab)
    _assert_rows(got, _expect(go, ids, vocab), tol=1e-6)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("vocab", [1, 2, 3, 17, 257, 4096])
def test_c7_vocab_sweep(vocab):
    """One group (single core) up to a mid-size table.

    vocab=1 puts all tokens in one group, so the whole reduction runs on one
    core -- correct, but this is the shape where the atomic kernel wins.
    """
    H, nt = 64, 128
    ids = _rand_ids(nt, vocab, seed=58)
    go = _rand_grad(nt, H, seed=59)
    got = embedding_grad_reduce_sorted(go, ids, vocab)
    assert got.shape == (vocab, H)
    _assert_rows(got, _expect(go, ids, vocab), tol=1e-4 if vocab <= 3 else 1e-5)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("H", [1, 7, 64, 129, 512, 513, 4095, 4096, 7168])
def test_c8_hidden_sweep(H):
    """Hidden sizes on both sides of the 512-thread VF boundary."""
    vocab, nt = 64, 100
    ids = _rand_ids(nt, vocab, seed=40)
    go = _rand_grad(nt, H, seed=41)
    got = embedding_grad_reduce_sorted(go, ids, vocab)
    _assert_rows(got, _expect(go, ids, vocab), tol=1e-6)


@tilelang.testing.requires_ascend
def test_c9_repeat_runs_are_bit_exact():
    """One writer per row: repeated launches reproduce the table bit-exactly.

    This is the property the atomic kernel cannot guarantee, and it also guards
    the conditional store: a dropped or doubled store would show up here.
    """
    vocab, H, nt = 4096, 256, 8192
    ids = _rand_ids(nt, vocab, seed=70)
    go = _rand_grad(nt, H, seed=71)
    ref = None
    for rep in range(4):
        out = torch.zeros(vocab, H, dtype=torch.float32, device="npu")
        embedding_grad_reduce_sorted(go, ids, vocab, out=out)
        if ref is None:
            ref = out.clone()
            _assert_rows(out, _expect(go, ids, vocab), tol=1e-5)
        else:
            assert torch.equal(out, ref), f"run {rep} differs from the first run"


@tilelang.testing.requires_ascend
def test_c10_empty_and_all_miss():
    """No hits, empty input, and an all-miss shard produce zeros."""
    vocab, start, end, H = 32, 16, 32, 64
    ids = _rand_ids(64, 16, seed=20)
    go = _rand_grad(64, H, seed=21)
    got = embedding_grad_reduce_sorted(go, ids, vocab_end_index=end, vocab_start_index=start)
    assert torch.count_nonzero(got) == 0

    empty = torch.empty(0, dtype=torch.int32).npu()
    got = embedding_grad_reduce_sorted(torch.empty(0, H, device="npu", dtype=torch.bfloat16), empty, vocab)
    assert got.shape == (vocab, H) and torch.count_nonzero(got) == 0


@tilelang.testing.requires_ascend
def test_c11_plan_invariants():
    """Host-only: the plan is monotone, covers every token and cuts on groups."""
    vocab, nt = 1000, 5000
    ids = _rand_ids(nt, vocab, seed=72)
    sorted_ids, token_start, order, scales = sorted_plan(ids)

    assert int(token_start[0]) == 0 and int(token_start[-1]) == nt
    assert bool(torch.all(token_start[1:] >= token_start[:-1]))
    assert bool(torch.all(sorted_ids[1:] >= sorted_ids[:-1]))
    assert bool(torch.all(scales == 1.0))
    assert order.numel() == nt and sorted_ids.numel() == nt
    # Every core boundary sits on a group boundary (or on the end of the list).
    head = torch.ones(nt, dtype=torch.bool, device="npu")
    head[1:] = sorted_ids[1:] != sorted_ids[:-1]
    starts = torch.nonzero(head).flatten()
    for c in range(NUM_AIV + 1):
        cut = int(token_start[c])
        assert cut == nt or bool(torch.any(starts == cut))
    assert bool(torch.equal(sorted_ids, ids.index_select(0, order.long())))


@tilelang.testing.requires_ascend
def test_c12_repeat_plan_is_stable():
    """The sort plan is deterministic for the same ids."""
    vocab, nt = 512, 2048
    ids = _rand_ids(nt, vocab, seed=73)
    first = sorted_plan(ids)
    for _ in range(3):
        again = sorted_plan(ids)
        assert torch.equal(first[0], again[0])
        assert torch.equal(first[1], again[1])
        assert torch.equal(first[2], again[2])


@tilelang.testing.requires_ascend
def test_c13_contract_rejects():
    H, vocab = 64, 16
    ids = _rand_ids(8, vocab, seed=26)
    go = _rand_grad(8, H, seed=27)
    with pytest.raises(AssertionError):
        embedding_grad_reduce_sorted(go[:, ::2], ids, vocab)
    with pytest.raises(AssertionError):
        embedding_grad_reduce_sorted(go, ids, num_weights=12, vocab_end_index=vocab + 4)
    with pytest.raises(AssertionError):
        embedding_grad_reduce_sorted(
            go, ids, vocab, out=torch.zeros(vocab, H, device="npu", dtype=torch.bfloat16)
        )
    with pytest.raises(AssertionError):
        embedding_grad_reduce_sorted(go, ids, vocab, out=torch.zeros(vocab, H + 1, device="npu", dtype=torch.float32))
    with pytest.raises(TypeError):
        embedding_grad_reduce_sorted(go.to(torch.int32), ids, vocab)


def test_c14_pipeline_versions_table():
    """Host-only: read-ahead depth respects the UB envelope at every H."""
    from example_embedding_grad_reduce_sorted import UB_PLAN, _pipeline_versions

    for H in [1, 8, 64, 1024, 4095, 4096, 7168, 8192]:
        for esize_src in (2, 4):
            versions = _pipeline_versions(H, esize_src)
            assert versions >= 2
            assert versions * H * esize_src + H * 4 <= UB_PLAN, (H, esize_src)
