"""Correctness tests for the vocab-parallel embedding gradient reduce example.

Cases map to ``embedding_grad_reduce/design.md`` §6 (README §6 checklist).
Rows that receive exactly one contribution are bit-exact (the hardware atomic
adds that value to a zero row); rows with repeated ids accumulate in an
unspecified order, so those cases compare with an fp32 tolerance.
"""

import pytest
import torch

import tilelang
import tilelang.testing

from example_embedding_grad_reduce import (
    UB_PLAN,
    _pipeline_config,
    embedding_grad_reduce,
    embedding_grad_reduce_kernel,
    ref_program,
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
    got = embedding_grad_reduce(go, ids, vocab)
    assert got.shape == (vocab, H) and got.dtype == torch.float32
    _assert_rows(got, _expect(go, ids, vocab))


@tilelang.testing.requires_ascend
def test_c2_duplicate_ids_accumulate():
    """The core semantics: repeated ids add, across cores and within a chunk."""
    vocab, H = 32, 64
    base = _rand_ids(64, vocab, seed=2)
    ids = torch.cat([base, base, base])  # every id appears 3x on average
    go = _rand_grad(ids.numel(), H, seed=3)
    got = embedding_grad_reduce(go, ids, vocab)
    _assert_rows(got, _expect(go, ids, vocab), tol=1e-5)


@tilelang.testing.requires_ascend
def test_c3_boundary_and_out_of_shard_ids():
    """Rows at vocab_start / vocab_end-1 hit; ids outside the shard are skipped."""
    vocab, start, end, H = 32, 8, 24, 256
    boundary = torch.tensor(
        [0, vocab - 1, start, end - 1, end, start - 1, -1], dtype=torch.int32
    )
    ids = torch.cat([boundary.npu(), _rand_ids(57, vocab, seed=4)])
    go = _rand_grad(ids.numel(), H, seed=5)
    got = embedding_grad_reduce(go, ids, vocab_end_index=end, vocab_start_index=start)
    _assert_rows(got, _expect(go, ids, end - start, start=start, end=end), tol=1e-6)


@tilelang.testing.requires_ascend
def test_c4_padding_idx():
    """The padding row receives no gradient even when it is a frequent id."""
    vocab, H, pad = 16, 64, 7
    ids = _rand_ids(96, vocab, seed=6)
    ids[:32] = pad
    go = _rand_grad(96, H, seed=7)
    got = embedding_grad_reduce(go, ids, vocab, padding_idx=pad)
    expect = _expect(go, ids, vocab, padding_idx=pad)
    assert torch.count_nonzero(expect[pad]) == 0
    assert torch.count_nonzero(got[pad]) == 0
    _assert_rows(got, expect, tol=1e-6)


@tilelang.testing.requires_ascend
def test_c5_tp4_shards_reconstruct_full_table():
    """TP=4 simulation: each rank reduces its own row range; concatenation
    matches the full-table golden (shards never overlap)."""
    vocab, shard, H, nt = 64, 16, 128, 512
    torch.manual_seed(8)
    ids = torch.randint(0, vocab, (nt,), dtype=torch.int32).npu()
    go = _rand_grad(nt, H, seed=9)
    parts = []
    for rank in range(4):
        start = rank * shard
        parts.append(embedding_grad_reduce(go, ids, vocab_end_index=start + shard, vocab_start_index=start))
    got = torch.cat(parts, dim=0)
    expect = _expect(go, ids, vocab)
    # Each shard is an independent fp32 sum; only accumulation order differs.
    _assert_rows(got, expect, tol=1e-5)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_c6_dtypes(dtype):
    nt, vocab, H = 64, 32, 96
    ids = _rand_ids(nt, vocab, seed=10)
    go = _rand_grad(nt, H, dtype, seed=11)
    got = embedding_grad_reduce(go, ids, vocab)
    assert got.dtype == torch.float32
    _assert_rows(got, _expect(go, ids, vocab), tol=1e-6)


@tilelang.testing.requires_ascend
def test_c7_int64_and_2d_indices():
    """int64 ids are converted in the wrapper; leading dims are flattened."""
    vocab, H = 32, 128
    ids2d = _rand_ids(64, vocab, seed=12).reshape(2, 32)
    go2d = _rand_grad(64, H, seed=13).reshape(2, 32, H)
    got = embedding_grad_reduce(go2d, ids2d.to(torch.int64), vocab)
    _assert_rows(got, _expect(go2d.reshape(-1, H), ids2d.reshape(-1), vocab), tol=1e-6)


@tilelang.testing.requires_ascend
def test_c8_scale_grad_by_freq():
    """Each occurrence is divided by id_count (README §3 golden)."""
    vocab, H = 24, 64
    ids = _rand_ids(80, vocab, seed=14)
    ids = torch.cat([ids, ids[:16]])  # create frequency > 1
    go = _rand_grad(ids.numel(), H, seed=15)
    got = embedding_grad_reduce(go, ids, vocab, scale_grad_by_freq=True)
    expect = _expect(go, ids, vocab, scale=True)
    _assert_rows(got, expect, tol=1e-6)


@tilelang.testing.requires_ascend
def test_c9_in_place_accumulation():
    """``out=`` accumulates in place (MTP / repeated backward reuse)."""
    vocab, H, nt = 32, 64, 128
    ids = _rand_ids(nt, vocab, seed=16)
    go = _rand_grad(nt, H, seed=17)
    out = torch.zeros(vocab, H, dtype=torch.float32, device="npu")
    embedding_grad_reduce(go, ids, vocab, out=out)
    embedding_grad_reduce(go, ids, vocab, out=out)
    _assert_rows(out, _expect(go, ids, vocab) * 2, tol=2e-5)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("num_tokens", [1, 7, 8, 9, 55, 56, 57, 1000])
def test_c10_chunk_tails(num_tokens):
    """Chunks are ceil(n/56) tokens; the clamped tail must add nothing.

    A tail that re-added the last token would show up as an O(1) row error,
    far above the duplicate-accumulation tolerance used here.
    """
    vocab, H = 1024, 64
    ids = _rand_ids(num_tokens, vocab, seed=18)
    go = _rand_grad(num_tokens, H, seed=19)
    got = embedding_grad_reduce(go, ids, vocab)
    _assert_rows(got, _expect(go, ids, vocab), tol=1e-6)


@tilelang.testing.requires_ascend
def test_c11_no_hits_and_empty():
    """No id belongs to this rank, or no tokens at all: zeros, no launch."""
    vocab, start, end, H = 32, 16, 32, 64
    ids = _rand_ids(64, 16, seed=20)  # all in [0, 16): outside the shard
    go = _rand_grad(64, H, seed=21)
    got = embedding_grad_reduce(go, ids, vocab_end_index=end, vocab_start_index=start)
    assert torch.count_nonzero(got) == 0

    empty = torch.empty(0, dtype=torch.int32).npu()
    got = embedding_grad_reduce(torch.empty(0, H, device="npu", dtype=torch.bfloat16), empty, vocab)
    assert got.shape == (vocab, H) and torch.count_nonzero(got) == 0


@tilelang.testing.requires_ascend
def test_c12_dynamic_token_count_reuses_binary():
    """The token count is dynamic: one kernel object serves every hit count."""
    vocab, H = 16, 64
    k1 = embedding_grad_reduce_kernel(vocab, H, "bfloat16", "float32", False, "int32", True)
    k2 = embedding_grad_reduce_kernel(vocab, H, "bfloat16", "float32", False, "int32", True)
    assert k1 is k2
    for nt in (10, 300):
        ids = _rand_ids(nt, vocab, seed=22)
        go = _rand_grad(nt, H, seed=23)
        got = embedding_grad_reduce(go, ids, vocab)
        _assert_rows(got, _expect(go, ids, vocab), tol=1e-6)


@tilelang.testing.requires_ascend
def test_c12_repeat_runs_are_stable(monkeypatch):
    """A binary with distinct ids must reproduce bit-exactly across runs;
    the kernel is recompiled from source each rep (cache disabled)."""
    monkeypatch.setenv("TILELANG_DISABLE_CACHE", "1")
    vocab, H, nt = 200, 64, 200
    torch.manual_seed(24)
    ids = torch.randperm(vocab).to(torch.int32).npu()
    go = _rand_grad(nt, H, seed=25)
    expect = _expect(go, ids, vocab)
    for _ in range(2):
        got = embedding_grad_reduce(go, ids, vocab)
        _assert_rows(got, expect)


@tilelang.testing.requires_ascend
def test_c13_wrapper_contract_rejects():
    H, vocab = 64, 16
    ids = _rand_ids(8, vocab, seed=26)
    go = _rand_grad(8, H, seed=27)
    with pytest.raises(AssertionError):
        embedding_grad_reduce(go[:, ::2], ids, vocab)  # non-contiguous grad
    with pytest.raises(AssertionError):
        embedding_grad_reduce(go, ids, num_weights=12, vocab_end_index=vocab + 4)  # mismatch
    with pytest.raises(AssertionError):
        embedding_grad_reduce(go, ids, vocab, out=torch.zeros(vocab, H, device="npu", dtype=torch.bfloat16))
    with pytest.raises(AssertionError):
        embedding_grad_reduce(go, ids, vocab, out=torch.zeros(vocab, H + 1, device="npu", dtype=torch.float32))
    with pytest.raises(TypeError):
        embedding_grad_reduce(go.to(torch.int32), ids, vocab)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_c16_production_hidden(dtype):
    """Production hidden size: the two-row pipeline must fit UB at H=7168."""
    vocab, H = 8, 7168
    ids = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 1, 1], dtype=torch.int32).npu()
    go = _rand_grad(ids.numel(), H, dtype, seed=32)
    got = embedding_grad_reduce(go, ids, vocab)
    _assert_rows(got, _expect(go, ids, vocab), tol=1e-6)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("H", [4080, 4095])
def test_c17_unaligned_hidden(H):
    """H not divisible by the 512-thread VF domain (4080 = 32B-aligned rows,
    4095 = unaligned): the cast loop's remainder must be handled."""
    vocab, nt = 16, 32
    ids = _rand_ids(nt, vocab, seed=33)
    go = _rand_grad(nt, H, seed=34)
    got = embedding_grad_reduce(go, ids, vocab)
    _assert_rows(got, _expect(go, ids, vocab), tol=1e-6)


@tilelang.testing.requires_ascend
def test_c15_aten_autograd_equivalence():
    """The operator must match the gradient produced by the forward gather."""
    vocab, H, nt = 64, 32, 128
    w = torch.randn(vocab, H, dtype=torch.float32, device="npu", requires_grad=True)
    ids = _rand_ids(nt, vocab, seed=30)
    g = _rand_grad(nt, H, torch.float32, seed=31)
    torch.nn.functional.embedding(ids.long(), w).backward(g)
    got = embedding_grad_reduce(g, ids, vocab)
    torch.testing.assert_close(got, w.grad, rtol=1e-5, atol=1e-5)


def test_c14_pipeline_config_table():
    """Host-only config invariants (UB budget, minimum depth, large-H guard)."""
    assert _pipeline_config(7168, 2, 4) == (4, 4)
    assert _pipeline_config(4096, 2, 4) == (4, 4)
    assert _pipeline_config(7168, 4, 4) == (3, 3)
    for H in [1, 8, 64, 255, 256, 1024, 2048, 4095, 4096, 7168, 8192]:
        for esize_src, esize_acc in [(2, 4), (4, 4), (2, 2)]:
            versions, stages = _pipeline_config(H, esize_src, esize_acc)
            assert versions >= 2 and stages == versions
            assert versions * H * (esize_src + esize_acc) <= UB_PLAN, (H, esize_src, esize_acc)
    with pytest.raises(AssertionError):
        _pipeline_config(32768, 4, 4)  # two fp32 row versions cannot fit UB


# ---------------------------------------------------------------------------
# Generalization sweep (C18-C27): hidden size, token count, dtype x shape,
# vocabulary size, id patterns, shard matrix. Every case compares against the
# ATen-semantics golden; tolerance is 1e-6 unless heavy duplicate ids make the
# fp32 accumulation order visible.
# ---------------------------------------------------------------------------


@tilelang.testing.requires_ascend
@pytest.mark.parametrize(
    "H",
    [
        1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33,
        63, 64, 65, 127, 128, 129, 255, 256, 257,
        511, 512, 513, 1023, 1024, 2047, 2048,
        4095, 4096, 7168, 8192,
    ],
)
def test_c18_hidden_sweep(H):
    """Every hidden size, not just the production ones.

    Covers both sides of the 512-thread SIMT VF boundary (H < 512 switches to
    128 threads), the 512-element stride of the cast loop, and rows whose byte
    count is not a multiple of 32 B (odd H, 4095).
    """
    vocab, nt = 64, 100
    ids = _rand_ids(nt, vocab, seed=40)
    go = _rand_grad(nt, H, seed=41)
    got = embedding_grad_reduce(go, ids, vocab)
    assert got.shape == (vocab, H) and got.dtype == torch.float32
    _assert_rows(got, _expect(go, ids, vocab), tol=1e-6)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("num_tokens", [1, 2, 3, 7, 8, 9, 15, 16, 17, 54, 55, 56, 57, 111, 112, 113, 999])
def test_c19_token_sweep(num_tokens):
    """Token counts around the 56-core chunk boundary and far beyond it.

    chunk = ceil(n / 56), so these sizes exercise one partial chunk, exactly
    one chunk, one full plus a tail, and multi-wave grids with a clamped tail.
    """
    vocab, H = 1024, 64
    ids = _rand_ids(num_tokens, vocab, seed=42)
    go = _rand_grad(num_tokens, H, seed=43)
    got = embedding_grad_reduce(go, ids, vocab)
    _assert_rows(got, _expect(go, ids, vocab), tol=1e-6)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("H", [16, 128, 1024, 4096])
def test_c20_dtype_x_hidden(dtype, H):
    """Every supported input dtype across narrow, wide and production rows.

    fp32 input at H=1024 uses 3 buffer versions instead of 4 (UB budget), so
    both pipeline depths are exercised.
    """
    vocab, nt = 32, 200
    ids = _rand_ids(nt, vocab, seed=44)
    go = _rand_grad(nt, H, dtype, seed=45)
    got = embedding_grad_reduce(go, ids, vocab)
    assert got.dtype == torch.float32
    _assert_rows(got, _expect(go, ids, vocab), tol=1e-5)


def _numel(shape):
    total = 1
    for dim in shape:
        total *= dim
    return total


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("shape", [(1,), (2, 3), (2, 2, 3)])
def test_c21_id_and_grad_shapes(id_dtype, shape):
    """Leading id/grad dims are flattened, int64 ids are converted."""
    vocab, H = 16, 64
    numel = _numel(shape)
    ids = _rand_ids(numel, vocab, seed=46).reshape(shape).to(id_dtype)
    go = _rand_grad(numel, H, seed=47).reshape(*shape, H)
    got = embedding_grad_reduce(go, ids, vocab)
    assert got.shape == (vocab, H)
    _assert_rows(got, _expect(go.reshape(-1, H), ids.reshape(-1), vocab), tol=1e-6)


@tilelang.testing.requires_ascend
def test_c22_scalar_and_noncontiguous_ids():
    """A 0-D id with a 1-D grad row, and a transposed (non-contiguous) id view."""
    vocab, H = 16, 64
    go1d = _rand_grad(1, H, seed=48).reshape(H)
    ids0d = torch.tensor(3, dtype=torch.int32).npu()
    _assert_rows(
        embedding_grad_reduce(go1d, ids0d, vocab),
        _expect(go1d.reshape(1, H), ids0d.reshape(1), vocab),
    )

    nt = 64
    ids_t = _rand_ids(nt * 2, vocab, seed=49).reshape(nt, 2).T  # non-contiguous view
    go = _rand_grad(nt * 2, H, seed=50)
    _assert_rows(
        embedding_grad_reduce(go, ids_t, vocab),
        _expect(go, ids_t.reshape(-1), vocab),
    )


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("tp", [2, 4, 8])
def test_c23_tp_shard_matrix(tp):
    """TP=2/4/8 shards reconstruct the full-table gradient (no overlap)."""
    vocab, H, nt = 1024, 128, 2048
    shard = vocab // tp
    torch.manual_seed(50 + tp)
    ids = torch.randint(0, vocab, (nt,), dtype=torch.int32).npu()
    go = _rand_grad(nt, H, seed=51)
    parts = [
        embedding_grad_reduce(go, ids, vocab_start_index=r * shard, vocab_end_index=(r + 1) * shard)
        for r in range(tp)
    ]
    got = torch.cat(parts, dim=0)
    assert got.shape == (vocab, H)
    _assert_rows(got, _expect(go, ids, vocab), tol=1e-5)


@tilelang.testing.requires_ascend
def test_c24_single_row_and_empty_shards():
    """A one-row shard and a shard that misses every id (zero fill, no write)."""
    vocab, H, nt = 64, 64, 512
    torch.manual_seed(55)
    ids = torch.randint(0, vocab, (nt,), dtype=torch.int32).npu()
    go = _rand_grad(nt, H, seed=56)

    one_row = embedding_grad_reduce(go, ids, vocab_start_index=vocab - 1, vocab_end_index=vocab)
    assert one_row.shape == (1, H)
    full = _expect(go, ids, vocab)
    _assert_rows(one_row, full[vocab - 1 : vocab], tol=1e-5)

    miss_ids = torch.randint(0, 8, (nt,), dtype=torch.int32).npu()  # inside [0, 8)
    empty = embedding_grad_reduce(go, miss_ids, vocab_start_index=32, vocab_end_index=vocab)
    assert torch.count_nonzero(empty) == 0


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("pattern", ["all_hit", "all_miss", "alternating", "heavy_duplicates", "out_of_range"])
def test_c25_id_patterns(pattern):
    """Hit distribution: dense, empty, mixed, highly duplicated, out of range."""
    vocab, H, nt, start, end = 64, 64, 256, 16, 48

    if pattern == "all_hit":
        ids = torch.arange(nt, dtype=torch.int32).npu() % (end - start) + start
    elif pattern == "all_miss":
        ids = torch.randint(0, start, (nt,), dtype=torch.int32).npu()
    elif pattern == "alternating":
        base = torch.arange(nt, dtype=torch.int32).npu()
        ids = torch.where(base % 2 == 0, base % (end - start) + start, base % start)
    elif pattern == "heavy_duplicates":
        ids = torch.randint(start, start + 3, (nt,), dtype=torch.int32).npu()  # 3 hot rows
    else:  # out_of_range: negative, beyond vocab, and at the shard edges
        noise = torch.tensor([-1, -7, vocab, vocab + 5, start - 1, end], dtype=torch.int32).npu()
        ids = torch.cat([noise, torch.randint(0, vocab, (nt - noise.numel(),), dtype=torch.int32).npu()])

    go = _rand_grad(ids.numel(), H, seed=57)
    got = embedding_grad_reduce(go, ids, vocab_end_index=end, vocab_start_index=start)
    expect = _expect(go, ids, end - start, start=start, end=end)
    if pattern == "all_miss":
        assert torch.count_nonzero(got) == 0
    # ~85 contributions per row: the fp32 sum order alone accounts for ~1e-5.
    tol = 1e-4 if pattern == "heavy_duplicates" else 1e-6
    _assert_rows(got, expect, tol=tol)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("vocab", [1, 2, 3, 17, 257, 4096])
def test_c26_vocab_sweep(vocab):
    """Vocabulary sizes from a single row to a mid-size table.

    vocab=1 makes every token collide on row 0 (worst-case atomic contention);
    the golden still defines the fp32 sum, so the tolerance scales with the
    duplicate count (128 contributions on the single row).
    """
    H, nt = 64, 128
    ids = _rand_ids(nt, vocab, seed=58)
    go = _rand_grad(nt, H, seed=59)
    got = embedding_grad_reduce(go, ids, vocab)
    assert got.shape == (vocab, H)
    _assert_rows(got, _expect(go, ids, vocab), tol=1e-4 if vocab <= 3 else 1e-5)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("padding_idx", [-1, 0])
@pytest.mark.parametrize("scale", [False, True])
def test_c27_shard_padding_scale_matrix(padding_idx, scale):
    """Sharding composed with padding_idx and scale_grad_by_freq."""
    vocab, H, nt, tp = 64, 128, 512, 4
    shard = vocab // tp
    torch.manual_seed(60)
    ids = torch.randint(0, vocab, (nt,), dtype=torch.int32).npu()
    if padding_idx >= 0:
        ids[:64] = padding_idx  # frequent padding id inside the first shard
    go = _rand_grad(nt, H, seed=61)
    parts = [
        embedding_grad_reduce(
            go,
            ids,
            padding_idx=padding_idx,
            scale_grad_by_freq=scale,
            vocab_start_index=r * shard,
            vocab_end_index=(r + 1) * shard,
        )
        for r in range(tp)
    ]
    got = torch.cat(parts, dim=0)
    expect = _expect(go, ids, vocab, padding_idx, scale)
    if padding_idx >= 0:
        assert torch.count_nonzero(got[padding_idx]) == 0
    _assert_rows(got, expect, tol=1e-5)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("H", [7168, 8192])
def test_c28_fp32_large_hidden(H):
    """fp32 input at production width: 3 buffer versions (not 4) must fit UB."""
    vocab, nt = 16, 32
    ids = _rand_ids(nt, vocab, seed=62)
    go = _rand_grad(nt, H, torch.float32, seed=63)
    got = embedding_grad_reduce(go, ids, vocab)
    _assert_rows(got, _expect(go, ids, vocab), tol=1e-6)
