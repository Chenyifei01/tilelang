"""Vocab-parallel embedding gradient reduce (scatter-add) for Ascend NPUs.

Design: ``README.md`` (Embedding gradient reduce). ``embedding_lookup`` gathers rows;
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
  streamed in ``ID_CHUNK`` blocks and compacted in SIMD into a ``(token, row)``
  list; each drain assigns the first token of a row a plain store, folds one
  repeat of that row into the same store (read both grad rows, add in UB, store
  once) and keeps any further repeat as an atomic add. Streaming (rather than
  keeping the index array in UB, 128 KiB at n=32768) is what lets the row
  pipeline stay 3-4 buffers deep.
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

Duplicate ids are resolved in UB where the row partition can fold a repeat into
the same store, and otherwise by the hardware store-mode atomic add
(``T.set_atomic("add", ...)``); the order is therefore not fixed, so repeated
runs may differ in the last fp32 ulp.

Folding matters because the atomic add is the expensive way to touch a row
twice: it reads and writes the row (2x) instead of just writing it (1x). Only a
duplicate that is the *second* hit of a row in the same drain folds -- a third
hit, or a hit whose first occurrence was stored by an earlier drain, still needs
the atomic add.

``out=`` must be zeroed on entry: the operator produces the gradient of this
call, it does not accumulate into a pre-existing table.
"""

import argparse

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S
from tilelang.profiler import do_bench

# Measured AIV count on Ascend 950DT.
NUM_AIV = 56
# UB envelope used for the row pipeline; the hardware limit is ~216 KiB with a
# VF region, and the scheduler needs headroom for flags/reservations.
UB_PLAN = 208 * 1024
# Candidate ids per id DMA, largest first. The row partition re-reads every id on
# every core, so each round trip exposes its full latency on the scalar pipe
# before the SIMD scan can start. msopprof PipeTimeline (950DT, n=32768) puts the
# prelude before the first row store at 106 us, ~14 us of it id MTE2 across 32
# round trips; the chunks below cut those trips by 1.5-3x. The chunk cannot grow
# freely: one chunk can add up to ID_CHUNK hits before the drain runs, so
# tok_list/row_list/fold_tok all scale with it, and a wide fp32 row pipeline
# needs that UB for its own buffers -- _id_chunk walks this list down until the
# compaction lists still leave room for the minimum row pipeline.
ID_CHUNK = {"int32": (1536, 1024, 512), "int64": (1024, 512)}
# ids compacted per SIMD scan block; also the granularity the token-partition
# path streams at (it scans this core's own slice, so a small block is right).
SCAN_BLK = 1024
# Hits buffered before a drain runs the row pipeline.
DRAIN_CAP = 1024
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


def _row_depth(H, esize_src, esize_acc, list_bytes):
    """(src_versions, src2_versions) that fit ``UB_PLAN``, or ``None``.

    Both source depths stay at least two so a read can be issued while the
    previous row is cast and stored; the fold source is sacrificed first.
    """
    acc_versions = 2
    base = list_bytes + acc_versions * H * esize_acc
    src2_versions = 2
    while base + (src2_versions + 2) * H * esize_src > UB_PLAN and src2_versions > 1:
        src2_versions -= 1
    room = UB_PLAN - base - src2_versions * H * esize_src
    src_versions = max(2, min(4, room // (H * esize_src)))
    if base + (src2_versions + src_versions) * H * esize_src > UB_PLAN:
        return None
    return src_versions, src2_versions


def _id_chunk(id_dtype, H, esize_src, esize_acc, words):
    """Widest id DMA chunk whose compaction lists still fit the row pipeline."""
    for chunk in ID_CHUNK[id_dtype]:
        if _row_depth(H, esize_src, esize_acc, _list_bytes(id_dtype, words, chunk)) is not None:
            return chunk
    return ID_CHUNK[id_dtype][-1]


def _pipeline_config(H, esize_src, esize_acc, list_bytes=0):
    """(src_versions, acc_versions, src2_versions, num_stages) for the row pipeline.

    Each in-flight token owns one source row (``in_dtype``), one accumulator row
    (``acc_dtype``) and, on a folded repeat, a second source row. The source
    depth is the largest value that keeps
    ``src_versions * src_row + src2_versions * src_row + acc_versions * acc_row
    + list_bytes`` inside ``UB_PLAN`` (``list_bytes`` is the fused kernel's
    per-core id stream, compaction lists and fold bookkeeping). Rows are copied
    one at a time, so a row byte count that is not a multiple of 32 B is fine
    (the lookup's multi-row pitch constraint does not apply).

    The accumulator depth is fixed at two. msopprof shows the store pipe
    saturated with a single store in flight (allowing four, by deleting the
    per-iteration MTE3 drain, measured no change), so a deeper accumulator
    rotation only spends UB that the fold path needs for its second source row.
    The second source row starts at two versions and drops to one only when the
    plan cannot pay for both it and the two-deep main source.
    """
    depth = _row_depth(H, esize_src, esize_acc, list_bytes)
    assert depth is not None, (
        f"hidden={H} needs {H * esize_src} B per source version and "
        f"{H * esize_acc} B per accumulator version plus the compacted lists; "
        f"two source versions, one fold-source version, two accumulator "
        f"versions and the lists do not fit the UB plan"
    )
    src_versions, src2_versions = depth
    return src_versions, 2, src2_versions, src_versions


def _list_bytes(id_dtype, words, id_chunk=None):
    """UB held by the scan buffers and the compaction/split lists.

    Everything except the row pipeline: the streamed ids chunk (int64 gets an
    extra staging buffer), the ``(token, row)`` compaction pair, the four
    plain/atom lists, the running-count vector, the row -> plain-index table and
    the per-plain-row fold token.
    """
    if id_chunk is None:
        id_chunk = ID_CHUNK[id_dtype][0]
    list_cap = id_chunk + DRAIN_CAP
    esize_id = 8 if id_dtype == "int64" else 4
    b = id_chunk * 4                                   # int32 ids chunk
    if esize_id == 8:
        b += id_chunk * 8                              # int64 staging chunk
    b += 2 * list_cap * 4                              # tok_list, row_list
    b += 4 * DRAIN_CAP * 4                             # plain/atom tok/row
    b += 2 * LANES * 4                                 # cnt_ub, sample count
    b += words * 32 * 4                                # row -> absolute plain index
    b += list_cap * 4                                  # per-plain-row fold token
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
      ``[c*R, (c+1)*R)``, so a row has exactly one writer core: its first token
      is a plain store, one repeat is folded into that store, and any further
      repeat atomic-adds. The ids are streamed in ``ID_CHUNK`` blocks (compacted
      in SIMD into a ``(token, row)`` list) -- keeping them resident costs
      128 KiB and pins the row pipeline to a single buffer version, measured
      14-29% slower. Each list is drained when full and once more on the last
      block.
    * **token partition** (low hit ratio, e.g. one rank of a vocab-parallel
      job). Each core scans only its own token slice, so the scan cost follows
      the hits instead of ``n``; every store is an atomic add.

    ``scale_grad_by_freq`` builds a second, otherwise identical prim_func
    (``fused_scaled``) that additionally takes ``Scales`` and multiplies each
    token's gradient by it: an optional input is a fixed-signature
    specialization, and the default path must not have to pass a dummy
    ``float32`` tensor (that would be host-side tensor work). Only the
    selected variant is defined and traced per build -- exactly one binary is
    produced and launched.

    Duplicate ids accumulate; the order across cores and tokens is not fixed
    (see the module docstring).
    """
    assert num_rows > 0 and H > 0
    assert id_dtype in ("int32", "int64")
    esize_src = 2 if in_dtype in ("bfloat16", "float16") else 4
    esize_acc = 2 if acc_dtype in ("bfloat16", "float16") else 4
    vf_threads = _vf_threads(H)
    R = (num_rows + NUM_AIV - 1) // NUM_AIV
    words = (R + 31) // 32
    id_chunk = _id_chunk(id_dtype, H, esize_src, esize_acc, words)
    list_cap = id_chunk + DRAIN_CAP
    src_versions, acc_versions, src2_versions, num_stages = _pipeline_config(
        H, esize_src, esize_acc, _list_bytes(id_dtype, words, id_chunk)
    )
    idx_cap = ((words * 32 + LANES - 1) // LANES) * LANES
    n = T.dynamic("n")
    cast_ids = id_dtype != "int32"

    if not scale_grad_by_freq:
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
                id_src = T.alloc_shared((id_chunk,), id_dtype)
                id_blk = T.alloc_shared((id_chunk,), "int32")
                tok_list = T.alloc_shared((list_cap,), "uint32")
                row_list = T.alloc_shared((list_cap,), "uint32")
                first_idx_ub = T.alloc_shared((idx_cap,), "int32")
                fold_tok = T.alloc_shared((list_cap,), "int32")
                cnt_ub = T.alloc_shared((LANES,), "int32")
                smp_ub = T.alloc_shared((LANES,), "int32")
                plain_tok = T.alloc_shared((DRAIN_CAP,), "int32")
                plain_row = T.alloc_shared((DRAIN_CAP,), "int32")
                atom_tok = T.alloc_shared((DRAIN_CAP,), "int32")
                atom_row = T.alloc_shared((DRAIN_CAP,), "int32")
                src_ub = T.alloc_shared((H,), in_dtype)
                src2_ub = T.alloc_shared((H,), in_dtype)
                acc_ub = T.alloc_shared((H,), acc_dtype)
                T.annotate_buffer_versions(
                    {src_ub: src_versions, acc_ub: acc_versions, src2_ub: src2_versions}
                )

                lo = vocab_start + core * R
                hi = T.min(vocab_start + (core + 1) * R, vocab_end)
                nblk = T.ceildiv(n, id_chunk)
                plain_base = T.alloc_var("int32", init=0)

                # first_idx_ub holds the *absolute* plain-list index of a row's
                # single plain store, or -1; plain_base below tells whether that
                # store belongs to the current drain. That replaces a persistent
                # first-writer bitmap plus a per-drain reset: one UB access fewer
                # per compacted entry.
                with T.SimdVF():
                    full0 = S.pset(32, 'PAT_ALL')
                    S.vsts(cnt_ub[0], S.vdup(0, T.int32, full0), full0)
                    neg0 = S.vdup(-1, T.int32, full0)
                    for c in range(idx_cap // LANES):
                        S.vsts(first_idx_ub[c * LANES], neg0, full0)
                T.ascend_pipe_barrier("PIPE_V")

                # Which strategy? Sample the first block for the *shard* hit ratio.
                # Every core sees the same value, so all cores agree -- mixing the
                # two strategies would let a plain store and an atomic add race on
                # the same row.
                if cast_ids:
                    T.copy(Ids[0 : SCAN_BLK], id_src[0:SCAN_BLK])
                    with T.SimtVF(threads=vf_threads):
                        for j in T.Parallel(SCAN_BLK):
                            id_blk[j] = T.cast(id_src[j], "int32")
                else:
                    T.copy(Ids[0 : SCAN_BLK], id_blk[0:SCAN_BLK])
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
                too_big = T.if_then_else(list_cap < T.ceildiv(n, NUM_AIV), 1, 0)
                row_mode = T.if_then_else(T.max(high, too_big) != 0, 1, 0)

                # ---- row partition: one writer per row, plain store then adds ----
                # One extra block past the end so the tail drain always runs; the
                # overflow read is masked by `tok < n` inside the scan.
                for blk in T.serial(nblk + 1):
                    if row_mode == 1:
                        if cast_ids:
                            T.copy(Ids[blk * id_chunk : blk * id_chunk + id_chunk], id_src)
                            with T.SimtVF(threads=vf_threads):
                                for j in T.Parallel(id_chunk):
                                    id_blk[j] = T.cast(id_src[j], "int32")
                        else:
                            T.copy(Ids[blk * id_chunk : blk * id_chunk + id_chunk], id_blk)
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
                            for c in range(id_chunk // LANES):
                                v = S.vld(id_blk[c * LANES])
                                tok = S.vci(T.int32(blk * id_chunk + c * LANES), "int32")
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
                            cnt = T.min(cnt_ub[0], list_cap)
                            for i in T.serial(cnt):
                                grow = T.cast(row_list[i], "int32")
                                row = grow - core * R
                                tgt = T.cast(first_idx_ub[row], "int32")
                                if tgt < 0:
                                    first_idx_ub[row] = plain_base + np_
                                    plain_tok[np_] = T.cast(tok_list[i], "int32")
                                    plain_row[np_] = grow
                                    fold_tok[np_] = -1
                                    np_ += 1
                                else:
                                    # One repeat folds into its row's single plain
                                    # store. A repeat of a row stored by an earlier
                                    # drain (tgt < plain_base) or a second repeat of
                                    # this one (fold_tok already taken) needs the
                                    # atomic add.
                                    slot = T.max(tgt - plain_base, 0)
                                    free = fold_tok[slot] < 0
                                    if (tgt >= plain_base) and free:
                                        fold_tok[slot] = T.cast(tok_list[i], "int32")
                                    else:
                                        atom_tok[na_] = T.cast(tok_list[i], "int32")
                                        atom_row[na_] = grow
                                        na_ += 1
                            for i in T.Pipelined(np_, num_stages=num_stages):
                                T.copy(GradOut[plain_tok[i], :], src_ub)
                                with T.SimtVF(threads=vf_threads):
                                    for j in T.Parallel(H):
                                        acc_ub[j] = T.cast(src_ub[j], acc_dtype)
                                if fold_tok[i] >= 0:
                                    T.copy(GradOut[fold_tok[i], :], src2_ub)
                                    with T.SimtVF(threads=vf_threads):
                                        for j in T.Parallel(H):
                                            acc_ub[j] = acc_ub[j] + T.cast(src2_ub[j], acc_dtype)
                                T.copy(acc_ub, GradWeight[plain_row[i], :])
                            T.set_atomic("add", acc_dtype)
                            for i in T.Pipelined(na_, num_stages=num_stages):
                                T.copy(GradOut[atom_tok[i], :], src_ub)
                                with T.SimtVF(threads=vf_threads):
                                    for j in T.Parallel(H):
                                        acc_ub[j] = T.cast(src_ub[j], acc_dtype)
                                T.copy(acc_ub, GradWeight[atom_row[i], :])
                            T.set_atomic_none()
                            plain_base += np_
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
                            T.copy(Ids[tbase + b * SCAN_BLK : tbase + b * SCAN_BLK + SCAN_BLK], id_src[0:SCAN_BLK])
                            with T.SimtVF(threads=vf_threads):
                                for j in T.Parallel(SCAN_BLK):
                                    id_blk[j] = T.cast(id_src[j], "int32")
                        else:
                            T.copy(Ids[tbase + b * SCAN_BLK : tbase + b * SCAN_BLK + SCAN_BLK], id_blk[0:SCAN_BLK])
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

        return fused

    else:
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
                id_src = T.alloc_shared((id_chunk,), id_dtype)
                id_blk = T.alloc_shared((id_chunk,), "int32")
                tok_list = T.alloc_shared((list_cap,), "uint32")
                row_list = T.alloc_shared((list_cap,), "uint32")
                first_idx_ub = T.alloc_shared((idx_cap,), "int32")
                fold_tok = T.alloc_shared((list_cap,), "int32")
                cnt_ub = T.alloc_shared((LANES,), "int32")
                smp_ub = T.alloc_shared((LANES,), "int32")
                plain_tok = T.alloc_shared((DRAIN_CAP,), "int32")
                plain_row = T.alloc_shared((DRAIN_CAP,), "int32")
                atom_tok = T.alloc_shared((DRAIN_CAP,), "int32")
                atom_row = T.alloc_shared((DRAIN_CAP,), "int32")
                src_ub = T.alloc_shared((H,), in_dtype)
                src2_ub = T.alloc_shared((H,), in_dtype)
                acc_ub = T.alloc_shared((H,), acc_dtype)
                T.annotate_buffer_versions(
                    {src_ub: src_versions, acc_ub: acc_versions, src2_ub: src2_versions}
                )

                lo = vocab_start + core * R
                hi = T.min(vocab_start + (core + 1) * R, vocab_end)
                nblk = T.ceildiv(n, id_chunk)
                plain_base = T.alloc_var("int32", init=0)

                # first_idx_ub holds the *absolute* plain-list index of a row's
                # single plain store, or -1; plain_base below tells whether that
                # store belongs to the current drain. That replaces a persistent
                # first-writer bitmap plus a per-drain reset: one UB access fewer
                # per compacted entry.
                with T.SimdVF():
                    full0 = S.pset(32, 'PAT_ALL')
                    S.vsts(cnt_ub[0], S.vdup(0, T.int32, full0), full0)
                    neg0 = S.vdup(-1, T.int32, full0)
                    for c in range(idx_cap // LANES):
                        S.vsts(first_idx_ub[c * LANES], neg0, full0)
                T.ascend_pipe_barrier("PIPE_V")

                # Which strategy? Sample the first block for the *shard* hit ratio.
                # Every core sees the same value, so all cores agree -- mixing the
                # two strategies would let a plain store and an atomic add race on
                # the same row.
                if cast_ids:
                    T.copy(Ids[0 : SCAN_BLK], id_src[0:SCAN_BLK])
                    with T.SimtVF(threads=vf_threads):
                        for j in T.Parallel(SCAN_BLK):
                            id_blk[j] = T.cast(id_src[j], "int32")
                else:
                    T.copy(Ids[0 : SCAN_BLK], id_blk[0:SCAN_BLK])
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
                too_big = T.if_then_else(list_cap < T.ceildiv(n, NUM_AIV), 1, 0)
                row_mode = T.if_then_else(T.max(high, too_big) != 0, 1, 0)

                # ---- row partition: one writer per row, plain store then adds ----
                # One extra block past the end so the tail drain always runs; the
                # overflow read is masked by `tok < n` inside the scan.
                for blk in T.serial(nblk + 1):
                    if row_mode == 1:
                        if cast_ids:
                            T.copy(Ids[blk * id_chunk : blk * id_chunk + id_chunk], id_src)
                            with T.SimtVF(threads=vf_threads):
                                for j in T.Parallel(id_chunk):
                                    id_blk[j] = T.cast(id_src[j], "int32")
                        else:
                            T.copy(Ids[blk * id_chunk : blk * id_chunk + id_chunk], id_blk)
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
                            for c in range(id_chunk // LANES):
                                v = S.vld(id_blk[c * LANES])
                                tok = S.vci(T.int32(blk * id_chunk + c * LANES), "int32")
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
                            cnt = T.min(cnt_ub[0], list_cap)
                            for i in T.serial(cnt):
                                grow = T.cast(row_list[i], "int32")
                                row = grow - core * R
                                tgt = T.cast(first_idx_ub[row], "int32")
                                if tgt < 0:
                                    first_idx_ub[row] = plain_base + np_
                                    plain_tok[np_] = T.cast(tok_list[i], "int32")
                                    plain_row[np_] = grow
                                    fold_tok[np_] = -1
                                    np_ += 1
                                else:
                                    # One repeat folds into its row's single plain
                                    # store. A repeat of a row stored by an earlier
                                    # drain (tgt < plain_base) or a second repeat of
                                    # this one (fold_tok already taken) needs the
                                    # atomic add.
                                    slot = T.max(tgt - plain_base, 0)
                                    free = fold_tok[slot] < 0
                                    if (tgt >= plain_base) and free:
                                        fold_tok[slot] = T.cast(tok_list[i], "int32")
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
                                if fold_tok[i] >= 0:
                                    T.copy(GradOut[fold_tok[i], :], src2_ub)
                                    with T.SimtVF(threads=vf_threads):
                                        for j in T.Parallel(H):
                                            acc_ub[j] = acc_ub[j] + T.cast(src2_ub[j], acc_dtype) * scale
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
                            plain_base += np_
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
                            T.copy(Ids[tbase + b * SCAN_BLK : tbase + b * SCAN_BLK + SCAN_BLK], id_src[0:SCAN_BLK])
                            with T.SimtVF(threads=vf_threads):
                                for j in T.Parallel(SCAN_BLK):
                                    id_blk[j] = T.cast(id_src[j], "int32")
                        else:
                            T.copy(Ids[tbase + b * SCAN_BLK : tbase + b * SCAN_BLK + SCAN_BLK], id_blk[0:SCAN_BLK])
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

        return fused_scaled


def ref_program(grad_output, indices, num_weights, padding_idx=-1, scale_grad_by_freq=False,
                vocab_start_index=0, vocab_end_index=None):
    """Golden: ATen ``embedding_dense_backward`` semantics on CPU, fp32.

    Ids outside ``[vocab_start_index, vocab_end_index)`` and the padding row
    contribute nothing. ``scale_grad_by_freq`` divides by the count of valid
    occurrences (the golden is ATen `embedding_dense_backward`). Duplicate ids
    accumulate.
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


def _row_stats(ids, vocab_start, vocab_num):
    """(hits, distinct rows, fold repeats, atomic tokens) for one shard.

    Mirrors what the kernel does: a row's first hit in its owning core is a
    plain store, its second hit folds into that store, and every further hit is
    an atomic add. Each row belongs to exactly one core, so counting over the
    shard gives the per-core counts.
    """
    local = (ids - vocab_start).to(torch.long)
    local = local[(local >= 0) & (local < vocab_num)]
    hits = int(local.numel())
    counts = torch.bincount(local.cpu(), minlength=vocab_num)
    distinct = int((counts > 0).sum())
    atom = int((counts - 2).clamp(min=0).sum())  # 3rd+ hits stay atomic
    return hits, distinct, hits - distinct, atom


def _scatter_bytes(hits, distinct, repeats, atom, H, src_bytes, acc_bytes, row_partition):
    """Logical GM traffic of this call.

    Row partition: every token reads its grad row once, every folded repeat
    reads a second one, every distinct row is stored once (fp32), and a 3rd+
    hit is an atomic read-modify-write. Token partition: every hit is one grad
    read plus one atomic read-modify-write. Ids are negligible.
    """
    if not row_partition:
        return H * (src_bytes * hits + 2 * acc_bytes * hits)
    return H * (src_bytes * (hits + repeats) + acc_bytes * (distinct + 2 * atom))


def _row_partitioned(hits, num_tokens):
    """Same 1/4-hit-ratio rule the kernel samples from the first id block."""
    return num_tokens > 0 and 4 * hits >= num_tokens


def run_regression_perf(num_tokens=32768, H=7168, dtype="bfloat16", vocab_size=129280):
    """Full-table bandwidth regression (row partition: one store per hit row)."""
    ids = torch.randint(0, vocab_size, (num_tokens,), dtype=torch.int32).npu()
    go = torch.randn(num_tokens, H, device="npu").to(_STR_TO_TORCH[dtype])
    gw = torch.zeros(vocab_size, H, dtype=torch.float32, device="npu")
    latency = do_bench(
        lambda: embedding_grad_reduce(go, ids, vocab_size, out=gw),
        backend="msprof",
        _n_warmup=30,
        _n_repeat=50,
    )
    hits, distinct, repeats, atom = _row_stats(ids.cpu(), 0, vocab_size)
    num_bytes = _scatter_bytes(
        hits, distinct, repeats, atom, H, go.element_size(), gw.element_size(), True
    )
    bandwidth = num_bytes / (latency * 1e6)
    print(
        f"embedding_grad_reduce (num_tokens={num_tokens}, H={H}, {dtype}, "
        f"distinct={distinct}, fold={repeats}, atomic={atom}): "
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
    hits, distinct, repeats, atom = _row_stats(ids.cpu(), start, shard)
    row_mode = _row_partitioned(hits, num_tokens)
    num_bytes = _scatter_bytes(
        hits, distinct, repeats, atom, H, go.element_size(), gw.element_size(), row_mode
    )
    bandwidth = num_bytes / (latency * 1e6)
    print(
        f"embedding_grad_reduce shard (TP={tp}, hits={hits}/{num_tokens}, "
        f"{'row' if row_mode else 'token'} partition, H={H}, {dtype}): "
        f"{latency * 1000:.2f} us | {bandwidth:.2f} GB/s"
    )
    return latency


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bench", action=argparse.BooleanOptionalAction, default=True, help="benchmark after correctness checks"
    )
    args = parser.parse_args()
    # Sharded scenario with duplicate ids and padding.
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
