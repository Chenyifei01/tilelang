# Ascend examples

These examples demonstrate complete kernels on Ascend 950. Use the explicit
`tilelang.ascend.language` dialect and an Ascend-enabled build; see the
[backend guide](../../tilelang/ascend/README.md) for setup.
Inputs, reference computations and correctness checks run on the NPU.

| Example | Purpose |
| --- | --- |
| [GEMM](example_gemm.py) | Persistent matrix multiplication with swizzling, precision modes and a mixed-core epilogue |
| [Split-K GEMM](example_gemm_splitk.py) | Atomic or deterministic reduction across K partitions |
| [Vector addition](example_vecadd.py) | SIMT and explicit SIMD with pipelined data movement |
| [RMSNorm](example_rmsnorm.py) | SIMT fragments, reduction and pipelined row processing |
| [FP8 quantization](example_per_token_cast_to_fp8.py) | The same group-wise quantization in SIMT and explicit SIMD |
| [Embedding gradient reduce](example_embedding_grad_reduce.py) | Vocab-parallel scatter-add: row/token partitions, SIMD compaction, folded duplicate repeats |

For advanced GEMM implementations and tuning, see [deepgemm](deepgemm).
Attention kernels are in [flash_attention](flash_attention/README.md).

Run an example directly to check correctness and report device performance.
GEMM and attention report latency and TFLOPS; vector addition, RMSNorm, FP8
quantization and embedding gradient reduce report latency and effective
bandwidth in GB/s.
Bandwidth counts each logical input and output tensor's bytes once, including
auxiliary outputs such as FP8 scales and RMSNorm RSTD.
GEMM throughput uses `2 * M * N * K` operations, including for Split-K.
Use `--no-bench` on a top-level example to run only correctness checks.

Examples use the configured default target. Set `TILELANG_DEFAULT_TARGET=ascend`
to select AscendC or `TILELANG_DEFAULT_TARGET=pto` to select PTO.

GEMM runs FP8, BF16, FP32, HF32, BF16 output and both accumulation cases.
It uses `M = K = N = 8192`.
Vector addition and FP8 quantization run both SIMT and SIMD by default;
use `--mode simt` or `--mode simd` to select one. Each selected case is timed by default.

```bash
python examples/ascend/example_gemm.py
python examples/ascend/example_gemm_splitk.py --deterministic
python examples/ascend/example_vecadd.py
python examples/ascend/example_per_token_cast_to_fp8.py
python examples/ascend/example_per_token_cast_to_fp8.py --mode simd
python examples/ascend/example_rmsnorm.py --no-bench
python examples/ascend/example_embedding_grad_reduce.py
python -m pytest examples/ascend/test_rmsnorm.py -q
```

## Embedding gradient reduce

[`example_embedding_grad_reduce.py`](example_embedding_grad_reduce.py) is the
reverse of an embedding lookup: it accumulates `grad_output[j]` into row
`indices[j]` of an fp32 gradient table, so repeated ids add up (ATen
`embedding_dense_backward` semantics). Ids outside
`[vocab_start, vocab_end)` and `padding_idx` are dropped inside the kernel; the
table lives in global memory, so no on-chip reduction of the table is possible;
`out=` must be zeroed on entry.

One kernel picks one of two write strategies by sampling the first id block, and
every core picks the same one, so a plain store and an atomic add can never race
on the same row:

* **row partition** (full-table call). Core `c` owns rows `[c*R, (c+1)*R)`, so a
  row has exactly one writer core. The first token of a row is a plain store,
  the second is folded into that store (both grad rows are read and summed in
  UB), and any further token atomic-adds. Folding matters because the atomic add
  is the expensive way to touch a row twice: it reads *and* writes the row.
* **token partition** (one rank of a vocab-parallel job, low hit ratio). Each
  core scans only its own token slice, so the scan cost follows the hits instead
  of the token count; every store is an atomic add.

The ids are streamed and compacted in SIMD into a `(token, row)` list; the row
pipeline keeps four source, two fold-source and two accumulator row versions
inside the UB plan, and degrades those (and the id chunk) on wide fp32 rows.

The reported bandwidth counts real traffic: one grad row read per token, a second
read per folded repeat, one fp32 row store per hit row, and an atomic
read-modify-write only for a third hit of the same row. On Ascend 950DT
(`n=32768`, `H=7168`, `vocab=129280`) that is 1.35 GB in 563 us, i.e. 2.40 TB/s
against a 2.67 TB/s streaming ceiling. The steady state runs read-bound (MTE2 97%
busy); the remaining ~18% is the serialized id scan and classification that runs
before the first row store.

Each top-level `example_<name>.py` has one `test_<name>.py`. Correctness tests use
smaller inputs than performance runs. Feature-level and compiler regressions
live in [testing/ascend](../../testing/ascend/README.md), independently of examples.
For copy padding, buffer annotations and scheduling APIs, consult the
[Ascend programming skill](../../.agents/skills/tilelang-ascend/SKILL.md).

The [performance regression driver](../../maint/scripts/ascend_perf_regression.py)
benchmarks these applications, including both vector programming modes, DeepGEMM and attention.
Language-feature tests are not performance entries.
