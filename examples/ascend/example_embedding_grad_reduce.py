"""Vocab-parallel embedding gradient reduce (scatter-add) for Ascend NPUs.

Design: ``embedding_grad_reduce/design.md``. ``embedding_lookup`` gathers rows;
this operator is its reverse: for every token position ``j`` it accumulates
``grad_output[j]`` into row ``indices[j]`` of the embedding-gradient table,
so repeated ids add up. The table lives in global memory across the whole run
(``vocab * hidden`` is far larger than UB for DeepSeek V4), which rules out any
on-chip reduction of the table itself: every token is one random row read plus
one atomic row add.

Everything runs on the NPU. The wrapper does no tensor arithmetic at all: it
checks scalar preconditions and launches one kernel. That kernel samples the
first id block for the shard hit ratio and then runs one of two strategies
(the same one on every core -- a mix would let a plain store and an atomic add
race on the same row):

* **row partition** (hit ratio >= 1/4: the full-table call). Core ``c`` owns
  rows ``[c*R, (c+1)*R)``, so each row has exactly one writer core. The ids are
  streamed in ``SCAN_BLK`` blocks and compacted in SIMD into a ``(token, row)``
  list; each drain assigns the first token of a row a plain store and the
  repeats an atomic add. Streaming (rather than keeping the index array in UB,
  128 KiB at n=32768) is what lets the row pipeline stay 3-4 buffers deep.
* **token partition** (low hit ratio: one rank of a vocab-parallel job). Each
  core scans only its own token slice, so the scan cost follows the hits
  instead of ``n``; every store is an atomic add.

Either way the pipeline runs exactly as many iterations as the core has
compacted tokens, so no DMA and no atomic store is predicated on a runtime
tail.

The token count, the hit count and ``grad_output``'s row count are all dynamic,
so one compiled binary serves every training step; only ``num_rows``/``H``/
dtypes/``scale_grad_by_freq`` participate in specialization.

One discipline carried over from ``embedding_lookup`` (see its design doc):
MTE movement stays unconditional and conditional work stays on the V pipe.

Duplicate ids are resolved by the hardware store-mode atomic add
(``T.set_atomic("add", ...)``), and in the row-partitioned path by one plain
store followed by atomic adds; the order is therefore not fixed, so repeated
runs may differ in the last fp32 ulp.

``out=`` must be zeroed on entry: the operator produces the gradient of this
call, it does not accumulate into a pre-existing table.
"""

import argparse

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S
from tilelang.profiler import do_bench

# Measured AIV count on Ascend 950DT (embedding_lookup design.md §1.4).
NUM_AIV = 56
# UB envelope used for the row pipeline; the hardware limit is ~216 KiB with a
# VF region, and the scheduler needs headroom for flags/reservations.
UB_PLAN = 208 * 1024
# ids streamed per scan block (4 KiB as int32). The scan never holds the whole
# index array: keeping it resident costs 128 KiB at n=32768 and pins the row
# pipeline to a single buffer version, which measured 14-29% slower.
SCAN_BLK = 1024
# Hits buffered before a drain runs the row pipeline; a block adds at most
# SCAN_BLK more, hence the compaction buffer size.
DRAIN_CAP = 1024
LIST_CAP = SCAN_BLK + DRAIN_CAP
# Vector lanes of the SIMD compaction.
LANES = 64

_DTYPE_STR = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
}
_STR_TO_TORCH = {v: k for k, v in _DTYPE_STR.items()}
_ID_DTYPES = {torch.int32: "int32", torch.int64: "int64"}

def _inv_counts(idx, start, end, pad, num_rows):
    """Per-token ``1/count`` for ``scale_grad_by_freq`` (host-side, O(n)).

    The count of an id aggregates over every core that holds it, which a
    single kernel cannot do without a grid barrier, so this rarely used flag
    keeps a host-side count. The default path does no host tensor work.
    """
    valid = (idx >= start) & (idx < end)
    if pad >= 0:
        valid &= idx != pad
    local = (idx.to(torch.int64) - start).masked_fill(~valid, 0)
    counts = torch.zeros(num_rows, dtype=torch.float32, device=idx.device)
    counts.index_add_(0, local, valid.to(torch.float32))
    counts.clamp_(min=1.0)
    scales = (1.0 / counts.index_select(0, local)).to(torch.float32)
    return torch.where(valid, scales, torch.zeros_like(scales))


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


def _list_bytes(id_dtype, words):
    """UB held by the scan buffers and the compaction/split lists.

    Everything except the row pipeline: the streamed ids block (int64 gets an
    extra staging buffer), the ``(token, row)`` compaction pair, the four
    plain/atom lists, the running-count vector and the first-writer bitmap.
    """
    esize_id = 8 if id_dtype == "int64" else 4
    b = SCAN_BLK * 4                                   # int32 ids block
    if esize_id == 8:
        b += SCAN_BLK * 8                              # int64 staging block
    b += 2 * LIST_CAP * 4                              # tok_list, row_list
    b += 4 * DRAIN_CAP * 4                             # plain/atom tok/row
    b += 2 * LANES * 4                                 # cnt_ub, sample count
    b += words * 4                                     # first-writer bitmap
    return b


@tilelang.jit(target="ascend")
def embedding_grad_reduce_kernel(
    num_rows,
    H,
    in_dtype="bfloat16",
    acc_dtype="float32",
    scale_grad_by_freq=False,
    id_dtype="int32",
):
    """Build and JIT-cache the scatter-add kernel.

    Compile-time: ``num_rows``/``H``/the dtypes/``scale_grad_by_freq``. Runtime
    (dynamic): the token count ``n`` and ``grad_output``'s row count, so one
    compiled binary serves every step and every shard hit count.

    One kernel, two write strategies, chosen once per launch from a sample of
    the first id block (same value on every core, so the strategies can never
    mix):

    * **row partition** (shard hit ratio >= 1/4, i.e. the full-table call). The
      table is split into ``NUM_AIV`` row ranges; core ``c`` owns rows
      ``[c*R, (c+1)*R)``, so a row has exactly one writer core and its first
      token can be a plain store while repeats atomic-add. The ids are
      streamed in ``SCAN_BLK`` blocks (compacted in SIMD into a ``(token, row)``
      list) -- keeping them resident costs 128 KiB and pins the row pipeline to
      a single buffer version, measured 14-29% slower. Each list is drained
      when full and once more on the last block.
    * **token partition** (low hit ratio, e.g. one rank of a vocab-parallel
      job). Each core scans only its own token slice, so the scan cost follows
      the hits instead of ``n``; every store is an atomic add.

    ``Scales`` carries the per-token ``1/count`` weight when
    ``scale_grad_by_freq`` is on. Duplicate ids accumulate; the order across
    cores and tokens is not fixed (see the module docstring).
    """
    assert num_rows > 0 and H > 0
    assert id_dtype in ("int32", "int64")
    esize_src = 2 if in_dtype in ("bfloat16", "float16") else 4
    esize_acc = 2 if acc_dtype in ("bfloat16", "float16") else 4
    vf_threads = _vf_threads(H)
    R = (num_rows + NUM_AIV - 1) // NUM_AIV
    words = (R + 31) // 32
    versions, num_stages = _pipeline_config(H, esize_src, esize_acc, _list_bytes(id_dtype, words))
    n = T.dynamic("n")
    cast_ids = id_dtype != "int32"

    @T.prim_func
    def fused(
        Ids: T.Tensor((n,), id_dtype),
        GradOut: T.Tensor((n, H), in_dtype),
        GradWeight: T.Tensor((num_rows, H), acc_dtype),
        vocab_start: T.int32,
        vocab_end: T.int32,
        padding_idx: T.int32,
    ):
        with T.Kernel(NUM_AIV) as core:
            id_src = T.alloc_shared((SCAN_BLK,), id_dtype)
            id_blk = T.alloc_shared((SCAN_BLK,), "int32")
            tok_list = T.alloc_shared((LIST_CAP,), "uint32")
            row_list = T.alloc_shared((LIST_CAP,), "uint32")
            bits_ub = T.alloc_shared((words,), "int32")
            cnt_ub = T.alloc_shared((LANES,), "int32")
            smp_ub = T.alloc_shared((LANES,), "int32")
            plain_tok = T.alloc_shared((DRAIN_CAP,), "int32")
            plain_row = T.alloc_shared((DRAIN_CAP,), "int32")
            atom_tok = T.alloc_shared((DRAIN_CAP,), "int32")
            atom_row = T.alloc_shared((DRAIN_CAP,), "int32")
            src_ub = T.alloc_shared((H,), in_dtype)
            acc_ub = T.alloc_shared((H,), acc_dtype)
            T.annotate_buffer_versions({src_ub: versions, acc_ub: versions})

            lo = vocab_start + core * R
            hi = T.min(vocab_start + (core + 1) * R, vocab_end)
            nblk = T.ceildiv(n, SCAN_BLK)

            for w in T.serial(words):
                bits_ub[w] = 0
            with T.SimdVF():
                full0 = S.pset(32, 'PAT_ALL')
                S.vsts(cnt_ub[0], S.vdup(0, T.int32, full0), full0)
            T.ascend_pipe_barrier("PIPE_V")

            # Which strategy? Sample the first block for the *shard* hit ratio.
            # Every core sees the same value, so all cores agree -- mixing the
            # two strategies would let a plain store and an atomic add race on
            # the same row.
            if cast_ids:
                T.copy(Ids[0 : SCAN_BLK], id_src)
                with T.SimtVF(threads=vf_threads):
                    for j in T.Parallel(SCAN_BLK):
                        id_blk[j] = T.cast(id_src[j], "int32")
            else:
                T.copy(Ids[0 : SCAN_BLK], id_blk)
            T.ascend_pipe_barrier("PIPE_MTE2")
            with T.SimdVF():
                full = S.pset(32, 'PAT_ALL')
                zero = S.vdup(0, T.int32, full)
                one = S.vdup(1, T.int32, full)
                idx = S.vci(T.int32(0), "int32")
                lane_last = S.vdup(LANES - 1, T.int32, full)
                base = S.alloc_var(T.int32)
                base = zero
                for c in range(SCAN_BLK // LANES):
                    v = S.vld(id_blk[c * LANES])
                    tok = S.vci(T.int32(c * LANES), "int32")
                    # Shard membership only: identical on every core, so the
                    # strategy choice cannot differ between cores (a mix of
                    # row-partitioned plain stores and token-partitioned
                    # atomic adds would race on the same row).
                    m = S.pand(S.vcmps(v, vocab_start, full, 'ge'), S.vcmps(v, vocab_end, full, 'lt'), full)
                    m = S.pand(m, S.vcmp(v, S.vdup(padding_idx, T.int32, full), full, 'ne'), full)
                    m = S.pand(m, S.vcmps(tok, n, full, 'lt'), full)
                    x = S.vsel(one, zero, m)
                    for s in (1, 2, 4, 8, 16, 32):
                        sidx = S.vmaxs(S.vadd(idx, S.vdup(-s, T.int32, full), full), 0, full)
                        sh = S.vsel(S.vselr(x, sidx), zero, S.vcmps(idx, s, full, 'ge'))
                        x = S.vadd(x, sh, full)
                    base = S.vadd(base, S.vselr(x, lane_last), full)
                S.vsts(smp_ub[0], base, full)
            T.ascend_pipe_barrier("PIPE_V")
            high = T.if_then_else(4 * smp_ub[0] >= T.min(SCAN_BLK, n), 1, 0)
            # The token partition buffers every hit of one core, so it needs
            # the slice to fit the list.
            too_big = T.if_then_else(LIST_CAP < T.ceildiv(n, NUM_AIV), 1, 0)
            row_mode = T.if_then_else(T.max(high, too_big) != 0, 1, 0)

            # ---- row partition: one writer per row, plain store then adds ----
            # One extra block past the end so the tail drain always runs; the
            # overflow read is masked by `tok < n` inside the scan.
            for blk in T.serial(nblk + 1):
                if row_mode == 1:
                    if cast_ids:
                        T.copy(Ids[blk * SCAN_BLK : blk * SCAN_BLK + SCAN_BLK], id_src)
                        with T.SimtVF(threads=vf_threads):
                            for j in T.Parallel(SCAN_BLK):
                                id_blk[j] = T.cast(id_src[j], "int32")
                    else:
                        T.copy(Ids[blk * SCAN_BLK : blk * SCAN_BLK + SCAN_BLK], id_blk)
                    T.ascend_pipe_barrier("PIPE_MTE2")
                    with T.SimdVF():
                        full = S.pset(32, 'PAT_ALL')
                        zero = S.vdup(0, T.int32, full)
                        one = S.vdup(1, T.int32, full)
                        neg_one = S.vdup(-1, T.int32, full)
                        idx = S.vci(T.int32(0), "int32")
                        lane_last = S.vdup(LANES - 1, T.int32, full)
                        base = S.alloc_var(T.int32)
                        base = S.vld(cnt_ub[0])
                        for c in range(SCAN_BLK // LANES):
                            v = S.vld(id_blk[c * LANES])
                            tok = S.vci(T.int32(blk * SCAN_BLK + c * LANES), "int32")
                            m = S.pand(S.vcmps(v, lo, full, 'ge'), S.vcmps(v, hi, full, 'lt'), full)
                            m = S.pand(m, S.vcmp(v, S.vdup(padding_idx, T.int32, full), full, 'ne'), full)
                            m = S.pand(m, S.vcmps(tok, n, full, 'lt'), full)
                            x = S.vsel(one, zero, m)
                            for s in (1, 2, 4, 8, 16, 32):
                                sidx = S.vmaxs(S.vadd(idx, S.vdup(-s, T.int32, full), full), 0, full)
                                sh = S.vsel(S.vselr(x, sidx), zero, S.vcmps(idx, s, full, 'ge'))
                                x = S.vadd(x, sh, full)
                            p = S.vadd(S.vadd(x, neg_one, full), base, full)
                            p = S.vsel(p, zero, m)
                            S.vscatter(T.reinterpret(tok, "uint32x64"), tok_list[0],
                                       T.reinterpret(p, "uint32x64"), m)
                            S.vscatter(T.reinterpret(S.vadd(v, S.vdup(-vocab_start, T.int32, full), full),
                                                     "uint32x64"), row_list[0],
                                       T.reinterpret(p, "uint32x64"), m)
                            base = S.vadd(base, S.vselr(x, lane_last), full)
                        S.vsts(cnt_ub[0], base, full)
                    T.ascend_pipe_barrier("PIPE_V")

                    need = T.if_then_else(DRAIN_CAP <= cnt_ub[0], 1,
                                          T.if_then_else(blk == nblk, 1, 0))
                    if need != 0:
                        np_ = T.alloc_var("int32", init=0)
                        na_ = T.alloc_var("int32", init=0)
                        cnt = T.min(cnt_ub[0], LIST_CAP)
                        for i in T.serial(cnt):
                            grow = T.cast(row_list[i], "int32")
                            row = grow - core * R
                            cur = bits_ub[row >> 5]
                            # (cur >> s) & 1 rather than (cur & (1 << s)): with
                            # s == 31 the masked value is INT32_MIN and the
                            # duplicated `0 < x` test the backend emits would
                            # drop every repeat for that row.
                            hit = (cur >> (row & 31)) & 1
                            if hit == 0:
                                bits_ub[row >> 5] = cur | (1 << (row & 31))
                                plain_tok[np_] = T.cast(tok_list[i], "int32")
                                plain_row[np_] = grow
                                np_ += 1
                            else:
                                atom_tok[na_] = T.cast(tok_list[i], "int32")
                                atom_row[na_] = grow
                                na_ += 1
                        for i in T.Pipelined(np_, num_stages=num_stages):
                            T.copy(GradOut[plain_tok[i], :], src_ub)
                            with T.SimtVF(threads=vf_threads):
                                for j in T.Parallel(H):
                                    acc_ub[j] = T.cast(src_ub[j], acc_dtype)
                            T.copy(acc_ub, GradWeight[plain_row[i], :])
                        T.set_atomic("add", acc_dtype)
                        for i in T.Pipelined(na_, num_stages=num_stages):
                            T.copy(GradOut[atom_tok[i], :], src_ub)
                            with T.SimtVF(threads=vf_threads):
                                for j in T.Parallel(H):
                                    acc_ub[j] = T.cast(src_ub[j], acc_dtype)
                            T.copy(acc_ub, GradWeight[atom_row[i], :])
                        T.set_atomic_none()
                        with T.SimdVF():
                            full_r = S.pset(32, 'PAT_ALL')
                            S.vsts(cnt_ub[0], S.vdup(0, T.int32, full_r), full_r)
                        T.ascend_pipe_barrier("PIPE_V")

            # ---- token partition: scan only this core's slice, all-atomic ----
            if row_mode == 0:
                pos = T.alloc_var("int32", init=0)
                chunk = T.ceildiv(n, NUM_AIV)
                tbase = core * chunk
                for b in T.serial(T.ceildiv(chunk, SCAN_BLK)):
                    if cast_ids:
                        T.copy(Ids[tbase + b * SCAN_BLK : tbase + b * SCAN_BLK + SCAN_BLK], id_src)
                        with T.SimtVF(threads=vf_threads):
                            for j in T.Parallel(SCAN_BLK):
                                id_blk[j] = T.cast(id_src[j], "int32")
                    else:
                        T.copy(Ids[tbase + b * SCAN_BLK : tbase + b * SCAN_BLK + SCAN_BLK], id_blk)
                    for j in T.serial(T.min(SCAN_BLK, chunk - b * SCAN_BLK)):
                        t = tbase + b * SCAN_BLK + j
                        idv = id_blk[j]
                        if (t < n) and (vocab_start <= idv) and (idv < vocab_end) and (idv != padding_idx):
                            tok_list[pos] = T.cast(t, "uint32")
                            row_list[pos] = T.cast(idv - vocab_start, "uint32")
                            pos += 1
                T.set_atomic("add", acc_dtype)
                for i in T.Pipelined(pos, num_stages=num_stages):
                    tok = T.cast(tok_list[i], "int32")
                    T.copy(GradOut[tok, :], src_ub)
                    with T.SimtVF(threads=vf_threads):
                        for j in T.Parallel(H):
                            acc_ub[j] = T.cast(src_ub[j], acc_dtype)
                    T.copy(acc_ub, GradWeight[T.cast(row_list[i], "int32"), :])
                T.set_atomic_none()

    @T.prim_func
    def fused_scaled(
        Ids: T.Tensor((n,), id_dtype),
        Scales: T.Tensor((n,), "float32"),
        GradOut: T.Tensor((n, H), in_dtype),
        GradWeight: T.Tensor((num_rows, H), acc_dtype),
        vocab_start: T.int32,
        vocab_end: T.int32,
        padding_idx: T.int32,
    ):
        with T.Kernel(NUM_AIV) as core:
            id_src = T.alloc_shared((SCAN_BLK,), id_dtype)
            id_blk = T.alloc_shared((SCAN_BLK,), "int32")
            tok_list = T.alloc_shared((LIST_CAP,), "uint32")
            row_list = T.alloc_shared((LIST_CAP,), "uint32")
            bits_ub = T.alloc_shared((words,), "int32")
            cnt_ub = T.alloc_shared((LANES,), "int32")
            smp_ub = T.alloc_shared((LANES,), "int32")
            plain_tok = T.alloc_shared((DRAIN_CAP,), "int32")
            plain_row = T.alloc_shared((DRAIN_CAP,), "int32")
            atom_tok = T.alloc_shared((DRAIN_CAP,), "int32")
            atom_row = T.alloc_shared((DRAIN_CAP,), "int32")
            src_ub = T.alloc_shared((H,), in_dtype)
            acc_ub = T.alloc_shared((H,), acc_dtype)
            T.annotate_buffer_versions({src_ub: versions, acc_ub: versions})

            lo = vocab_start + core * R
            hi = T.min(vocab_start + (core + 1) * R, vocab_end)
            nblk = T.ceildiv(n, SCAN_BLK)

            for w in T.serial(words):
                bits_ub[w] = 0
            with T.SimdVF():
                full0 = S.pset(32, 'PAT_ALL')
                S.vsts(cnt_ub[0], S.vdup(0, T.int32, full0), full0)
            T.ascend_pipe_barrier("PIPE_V")

            # Which strategy? Sample the first block for the *shard* hit ratio.
            # Every core sees the same value, so all cores agree -- mixing the
            # two strategies would let a plain store and an atomic add race on
            # the same row.
            if cast_ids:
                T.copy(Ids[0 : SCAN_BLK], id_src)
                with T.SimtVF(threads=vf_threads):
                    for j in T.Parallel(SCAN_BLK):
                        id_blk[j] = T.cast(id_src[j], "int32")
            else:
                T.copy(Ids[0 : SCAN_BLK], id_blk)
            T.ascend_pipe_barrier("PIPE_MTE2")
            with T.SimdVF():
                full = S.pset(32, 'PAT_ALL')
                zero = S.vdup(0, T.int32, full)
                one = S.vdup(1, T.int32, full)
                idx = S.vci(T.int32(0), "int32")
                lane_last = S.vdup(LANES - 1, T.int32, full)
                base = S.alloc_var(T.int32)
                base = zero
                for c in range(SCAN_BLK // LANES):
                    v = S.vld(id_blk[c * LANES])
                    tok = S.vci(T.int32(c * LANES), "int32")
                    # Shard membership only: identical on every core, so the
                    # strategy choice cannot differ between cores (a mix of
                    # row-partitioned plain stores and token-partitioned
                    # atomic adds would race on the same row).
                    m = S.pand(S.vcmps(v, vocab_start, full, 'ge'), S.vcmps(v, vocab_end, full, 'lt'), full)
                    m = S.pand(m, S.vcmp(v, S.vdup(padding_idx, T.int32, full), full, 'ne'), full)
                    m = S.pand(m, S.vcmps(tok, n, full, 'lt'), full)
                    x = S.vsel(one, zero, m)
                    for s in (1, 2, 4, 8, 16, 32):
                        sidx = S.vmaxs(S.vadd(idx, S.vdup(-s, T.int32, full), full), 0, full)
                        sh = S.vsel(S.vselr(x, sidx), zero, S.vcmps(idx, s, full, 'ge'))
                        x = S.vadd(x, sh, full)
                    base = S.vadd(base, S.vselr(x, lane_last), full)
                S.vsts(smp_ub[0], base, full)
            T.ascend_pipe_barrier("PIPE_V")
            high = T.if_then_else(4 * smp_ub[0] >= T.min(SCAN_BLK, n), 1, 0)
            # The token partition buffers every hit of one core, so it needs
            # the slice to fit the list.
            too_big = T.if_then_else(LIST_CAP < T.ceildiv(n, NUM_AIV), 1, 0)
            row_mode = T.if_then_else(T.max(high, too_big) != 0, 1, 0)

            # ---- row partition: one writer per row, plain store then adds ----
            # One extra block past the end so the tail drain always runs; the
            # overflow read is masked by `tok < n` inside the scan.
            for blk in T.serial(nblk + 1):
                if row_mode == 1:
                    if cast_ids:
                        T.copy(Ids[blk * SCAN_BLK : blk * SCAN_BLK + SCAN_BLK], id_src)
                        with T.SimtVF(threads=vf_threads):
                            for j in T.Parallel(SCAN_BLK):
                                id_blk[j] = T.cast(id_src[j], "int32")
                    else:
                        T.copy(Ids[blk * SCAN_BLK : blk * SCAN_BLK + SCAN_BLK], id_blk)
                    T.ascend_pipe_barrier("PIPE_MTE2")
                    with T.SimdVF():
                        full = S.pset(32, 'PAT_ALL')
                        zero = S.vdup(0, T.int32, full)
                        one = S.vdup(1, T.int32, full)
                        neg_one = S.vdup(-1, T.int32, full)
                        idx = S.vci(T.int32(0), "int32")
                        lane_last = S.vdup(LANES - 1, T.int32, full)
                        base = S.alloc_var(T.int32)
                        base = S.vld(cnt_ub[0])
                        for c in range(SCAN_BLK // LANES):
                            v = S.vld(id_blk[c * LANES])
                            tok = S.vci(T.int32(blk * SCAN_BLK + c * LANES), "int32")
                            m = S.pand(S.vcmps(v, lo, full, 'ge'), S.vcmps(v, hi, full, 'lt'), full)
                            m = S.pand(m, S.vcmp(v, S.vdup(padding_idx, T.int32, full), full, 'ne'), full)
                            m = S.pand(m, S.vcmps(tok, n, full, 'lt'), full)
                            x = S.vsel(one, zero, m)
                            for s in (1, 2, 4, 8, 16, 32):
                                sidx = S.vmaxs(S.vadd(idx, S.vdup(-s, T.int32, full), full), 0, full)
                                sh = S.vsel(S.vselr(x, sidx), zero, S.vcmps(idx, s, full, 'ge'))
                                x = S.vadd(x, sh, full)
                            p = S.vadd(S.vadd(x, neg_one, full), base, full)
                            p = S.vsel(p, zero, m)
                            S.vscatter(T.reinterpret(tok, "uint32x64"), tok_list[0],
                                       T.reinterpret(p, "uint32x64"), m)
                            S.vscatter(T.reinterpret(S.vadd(v, S.vdup(-vocab_start, T.int32, full), full),
                                                     "uint32x64"), row_list[0],
                                       T.reinterpret(p, "uint32x64"), m)
                            base = S.vadd(base, S.vselr(x, lane_last), full)
                        S.vsts(cnt_ub[0], base, full)
                    T.ascend_pipe_barrier("PIPE_V")

                    need = T.if_then_else(DRAIN_CAP <= cnt_ub[0], 1,
                                          T.if_then_else(blk == nblk, 1, 0))
                    if need != 0:
                        np_ = T.alloc_var("int32", init=0)
                        na_ = T.alloc_var("int32", init=0)
                        cnt = T.min(cnt_ub[0], LIST_CAP)
                        for i in T.serial(cnt):
                            grow = T.cast(row_list[i], "int32")
                            row = grow - core * R
                            cur = bits_ub[row >> 5]
                            # (cur >> s) & 1 rather than (cur & (1 << s)): with
                            # s == 31 the masked value is INT32_MIN and the
                            # duplicated `0 < x` test the backend emits would
                            # drop every repeat for that row.
                            hit = (cur >> (row & 31)) & 1
                            if hit == 0:
                                bits_ub[row >> 5] = cur | (1 << (row & 31))
                                plain_tok[np_] = T.cast(tok_list[i], "int32")
                                plain_row[np_] = grow
                                np_ += 1
                            else:
                                atom_tok[na_] = T.cast(tok_list[i], "int32")
                                atom_row[na_] = grow
                                na_ += 1
                        for i in T.Pipelined(np_, num_stages=num_stages):
                            scale = Scales[plain_tok[i]]
                            T.copy(GradOut[plain_tok[i], :], src_ub)
                            with T.SimtVF(threads=vf_threads):
                                for j in T.Parallel(H):
                                    acc_ub[j] = T.cast(src_ub[j], acc_dtype) * scale
                            T.copy(acc_ub, GradWeight[plain_row[i], :])
                        T.set_atomic("add", acc_dtype)
                        for i in T.Pipelined(na_, num_stages=num_stages):
                            scale = Scales[atom_tok[i]]
                            T.copy(GradOut[atom_tok[i], :], src_ub)
                            with T.SimtVF(threads=vf_threads):
                                for j in T.Parallel(H):
                                    acc_ub[j] = T.cast(src_ub[j], acc_dtype) * scale
                            T.copy(acc_ub, GradWeight[atom_row[i], :])
                        T.set_atomic_none()
                        with T.SimdVF():
                            full_r = S.pset(32, 'PAT_ALL')
                            S.vsts(cnt_ub[0], S.vdup(0, T.int32, full_r), full_r)
                        T.ascend_pipe_barrier("PIPE_V")

            # ---- token partition: scan only this core's slice, all-atomic ----
            if row_mode == 0:
                pos = T.alloc_var("int32", init=0)
                chunk = T.ceildiv(n, NUM_AIV)
                tbase = core * chunk
                for b in T.serial(T.ceildiv(chunk, SCAN_BLK)):
                    if cast_ids:
                        T.copy(Ids[tbase + b * SCAN_BLK : tbase + b * SCAN_BLK + SCAN_BLK], id_src)
                        with T.SimtVF(threads=vf_threads):
                            for j in T.Parallel(SCAN_BLK):
                                id_blk[j] = T.cast(id_src[j], "int32")
                    else:
                        T.copy(Ids[tbase + b * SCAN_BLK : tbase + b * SCAN_BLK + SCAN_BLK], id_blk)
                    for j in T.serial(T.min(SCAN_BLK, chunk - b * SCAN_BLK)):
                        t = tbase + b * SCAN_BLK + j
                        idv = id_blk[j]
                        if (t < n) and (vocab_start <= idv) and (idv < vocab_end) and (idv != padding_idx):
                            tok_list[pos] = T.cast(t, "uint32")
                            row_list[pos] = T.cast(idv - vocab_start, "uint32")
                            pos += 1
                T.set_atomic("add", acc_dtype)
                for i in T.Pipelined(pos, num_stages=num_stages):
                    tok = T.cast(tok_list[i], "int32")
                    scale = Scales[tok]
                    T.copy(GradOut[tok, :], src_ub)
                    with T.SimtVF(threads=vf_threads):
                        for j in T.Parallel(H):
                            acc_ub[j] = T.cast(src_ub[j], acc_dtype) * scale
                    T.copy(acc_ub, GradWeight[T.cast(row_list[i], "int32"), :])
                T.set_atomic_none()

    return fused_scaled if scale_grad_by_freq else fused


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
    are dropped inside the kernel (not by the host).

    ``out`` is an optional fp32 ``[num_weights, H]`` destination table and must
    be **zeroed on entry**: the kernel writes this call's gradient into the hit
    rows (row-partitioned path: one plain store per row) or adds into them
    (token-partitioned path), which are the same thing on a zeroed table, and
    is not an accumulator. Omit it and a zeroed table is allocated and
    returned; accumulate across calls on the host. Returned dtype is fp32
    (internal accumulation precision), not ``grad_output``'s dtype.

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

    if scale_grad_by_freq:
        scales = _inv_counts(idx, vocab_start_index, vocab_end_index, padding_idx, num_weights)
        embedding_grad_reduce_kernel(num_weights, H, in_dtype, "float32", True, id_dtype)(
            idx, scales, go, out, vocab_start_index, vocab_end_index, padding_idx
        )
    else:
        embedding_grad_reduce_kernel(num_weights, H, in_dtype, "float32", False, id_dtype)(
            idx, go, out, vocab_start_index, vocab_end_index, padding_idx
        )
    return out


def run_regression_perf(num_tokens=32768, H=7168, dtype="bfloat16", vocab_size=129280):
    """Full-table bandwidth regression (read each grad row, write each row once)."""
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
    # the destination row per token; ids are negligible.
    num_bytes = num_tokens * H * (go.element_size() + 2 * gw.element_size())
    bandwidth = num_bytes / (latency * 1e6)
    print(
        f"embedding_grad_reduce (num_tokens={num_tokens}, H={H}, {dtype}): "
        f"{latency * 1000:.2f} us | {bandwidth:.2f} GB/s"
    )
    return latency


def run_sharded_perf(num_tokens=32768, H=7168, dtype="bfloat16", vocab_size=129280, tp=8):
    """TP=``tp`` simulation: end-to-end wrapper (on-device filter + atomic reduce)."""
    shard = vocab_size // tp
    ids = torch.randint(0, vocab_size, (num_tokens,), dtype=torch.int32).npu()
    go = torch.randn(num_tokens, H, device="npu").to(_STR_TO_TORCH[dtype])
    gw = torch.zeros(shard, H, dtype=torch.float32, device="npu")
    start, end = 3 * shard, 4 * shard  # arbitrary rank 3 of 8
    latency = do_bench(
        lambda: embedding_grad_reduce(
            go, ids, vocab_end_index=end, vocab_start_index=start, out=gw
        ),
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
