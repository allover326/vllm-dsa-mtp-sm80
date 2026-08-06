# vllm-dsa-mtp-sm80

**Run GLM-5.2 / DeepSeek-family DSA (sparse-attention) models with MTP speculative
decoding under pipeline parallelism on Ampere GPUs (sm_80 / sm_86 — A100, CMP 170HX,
RTX 3090).**

Stock vLLM can't do this today: the sparse-MLA backends are Hopper/Blackwell-only, and
MTP speculative decoding is broken under pipeline parallelism. Two open PRs fix the two
halves — this repo composes them onto the official `vllm/vllm-openai:v0.26.0` image and
patches the version skew between them, giving you a working image in ~2 minutes with no
compilation.

| piece | what it adds | status upstream |
|---|---|---|
| [PR #38476](https://github.com/vllm-project/vllm/pull/38476) | `TRITON_MLA_SPARSE` — pure-Triton sparse-MLA backend for SM8x/11x/12x | open |
| [PR #46994](https://github.com/vllm-project/vllm/pull/46994) | MTP speculative decoding under pipeline parallelism (V2 model runner) | open |
| this repo | 16 skew fixes composing both onto v0.26.0 (`fix_compose_skew.py`) | — |

All credit for the heavy lifting goes to the authors of those two PRs. This repo exists
because neither is merged yet and they target different eras of `main`.

## Quick start

```bash
git clone https://github.com/allover326/vllm-dsa-mtp-sm80
cd vllm-dsa-mtp-sm80
./build_compose.sh            # → image vllm-mtp:compose46994-38476  (~2 min)
```

Then serve a GLM-5.2 quant with MTP on 8 GPUs (~64 GB each):

```bash
docker run --rm --runtime=nvidia --shm-size=32g \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_USE_FLASHINFER_SAMPLER=0 \
  -v /models:/models -p 8000:8000 \
  vllm-mtp:compose46994-38476 \
  --model /models/GLM-5.2-Int4-Int8Mix \
  --pipeline-parallel-size 8 --tensor-parallel-size 1 \
  --max-model-len 16384 --gpu-memory-utilization 0.80 \
  --block-size 64 --enforce-eager \
  --kv-cache-dtype auto --trust-remote-code \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
# --block-size 64 is REQUIRED (kernel block negotiation fails at the default).
# Without MTP: 32768 / 0.90 and drop --enforce-eager (28.47 tok/s measured PP8).
# With MTP: eager + 0.80 -- the draft loads extra weight on the last PP rank
# and cudagraph capture OOM'd at 0.90 on 8x64GB.
```

Benchmark baseline-vs-MTP instead with `glm_mtp_bench.py` (see file header).

## Verifying a build

`build_compose.sh` refuses to commit an image that fails verification, and
`verify_patches.sh` runs the same checks against an image you already have:

```bash
./verify_patches.sh                          # or: ./verify_patches.sh my-tag
```

Three checks, in increasing order of what they catch:

| check | catches |
|---|---|
| `py_compile` | syntax only — **passes on code that cannot run** |
| import smoke | import-time breakage |
| **undefined names** (pyflakes) | **a patch that CALLS something no patch DEFINES** |

The third one is the one that matters, and it exists because of a real failure in
the sibling repo. A patch generated against a working tree that already contains a
helper captures the *call sites* and not the *helper*: the diff applies cleanly on
the author's machine and raises `NameError` on everyone else's. That shipped in
[deepseek-v4-cmp170hx#1](https://github.com/allover326/deepseek-v4-cmp170hx/issues/1)
— patch 0006 called two functions that no patch defined, and two people lost time
to it before it was found.

**`py_compile` and the import smoke test both pass on that bug**, because the
missing name is only resolved when the function actually runs — which is why
neither was sufficient and the third check was added.

## Validation status (updated 2026-07-30)

- **GLM-5.2-744B real weights (`QuantTrio/GLM-5.2-Int4-Int8Mix`), 8× CMP 170HX
  (sm_80, PCIe Gen2 x4), PP8:** ✅ **28.47 tok/s decode, coherent output,
  cudagraphs on** — within ~6% of the bespoke 0.20.2-era recipe (30.2), now on a
  stock PyPI wheel. Requires `--block-size 64` and this repo's indexer
  page-padding fix (see below); both ship here.
- **GLM-5.2 + MTP:** the DeepSeek-family draft **loads and initializes under
  PP8** (SupportsPP path exercised on real weights). The first attempt OOM'd on
  the *last PP rank* — the draft layer + its private embed + LM head concentrate
  there — during cudagraph capture at `gpu-memory-utilization 0.90`. The bench
  now runs the MTP config eager at 0.80/16k; a measured speedup number is
  pending the next session. On MiMo-7B this exact stack measured **1.28× at
  70.9% acceptance**, and the tiny-GLM plumbing test is greedy-equivalent, so
  the remaining risk is memory tuning, not correctness.
- **Tiny random-init `glm_moe_dsa` (incl. MTP nextn layer), 2× RTX 3090:** PP1 ✅,
  PP2 ✅, **PP2 + MTP n=1 ✅ — token-for-token identical to the PP2 baseline
  (greedy-equivalent)**. Regenerate with `gen_tiny_glm.py`, run `tiny_glm_test.py`.
  Note: tiny configs with `max_model_len <= index_topk` never build the indexer
  KV cache, so they cannot catch the two real-model boot blockers below.
- **MiMo-7B-RL real weights (PP2, 2× 3090, #46994 alone):** 54.4 → 69.5 tok/s
  (**1.28×**), draft acceptance **70.9%**, no deadlock, cudagraphs on.

### Operational warnings for multi-GPU rigs (learned the expensive way)
- **Never tree-kill or `kill -9` a crashed multi-GPU vLLM run** (`tmux
  kill-session`, `pkill`, etc.). Killed workers strand CUDA contexts in the
  host driver: NVML keeps showing all GPUs while the CUDA runtime drops them
  (`device < num_gpus INTERNAL ASSERT`, wrong count). A container restart does
  NOT clear it — only a **host** reboot does. Send SIGINT to the Python PID and
  wait, or let the process die on its own.
- Put retries *inside* one process (see `glm_mtp_bench.py`'s attempt ladder)
  so external kills are never needed.

## Which GLM-5.2 quants work on sm_80

| repo | verdict |
|---|---|
| `QuantTrio/GLM-5.2-Int4-Int8Mix` | ✅ symmetric; **MTP head unquantized** — best for spec decode |
| `lowbitcoffee/GLM-5.2-W4A16` | ✅ symmetric (no MTP head — baseline only) |
| `c-bf/GLM-5.2-AutoRound-W4G64-MTP` | symmetric w/ MTP, untested |
| `cyankiwi/GLM-5.2-AWQ-INT4` | ❌ asymmetric — MoE assertion failure |
| `canada-quant/GLM-5.2-W4A16-MTP` | ❌ asymmetric |
| any `*-W4AFP8` / FP8-activation | ❌ needs sm_89+ |

Also: keep `--kv-cache-dtype auto` (BF16) — FP8 KV needs sm_89+.

## The skew fixes (why this repo isn't just `patch < *.diff`)

`v0.26.0` sits between the two PRs' base trees. The big ones (`fix_compose_skew.py`
documents all 16, each asserting its exact target):

0. **`indexer.py`: opt the DSA indexer KV cache into page-size padding.** The
   indexer page (block×132 B) never divides the sparse-MLA page (96:11 byte
   ratio — no block size can fix it), so real DSA models cannot boot 0.26.0's
   KV unifier without padding. Safe: all three ops touching that cache
   (`indexer_k_quant_and_cache`, `cp_gather_indexer_k_quant_cache`,
   `fp8_paged_mqa_logits_triton`) address blocks via `kv_cache.stride(0)` —
   verified in csrc before enabling. Only real models trigger this; tiny
   test models with `max_model_len <= index_topk` skip the indexer cache.

1. **`deepseek_v2.py`: fused indexer-q rope-quant gated to sm_89+.** 0.26.0
   unconditionally launches a Triton kernel that converts to `fp8e4nv`, which Triton
   cannot compile on sm_80/86 (`"type fp8e4nv not supported in this architecture"`).
   The unfused fallback uses the sm_80-safe `per_token_group_quant_fp8` CUDA op —
   the same approach current `main` uses. **This is the #1 landmine for anyone
   trying DSA models on Ampere with vLLM ≥ 0.26.**
2. `mla_attention.py`: the XPU/Triton sparse metadata has no decode/prefill split —
   guard `forward_impl` (all tokens go through `forward_mqa`, matching `main`'s
   `is_sparse_impl` branch) and the KV-cache-update field access.
3. `sparse_attn_indexer.py`: PR #38476's DeepGEMM→Triton fallback hunks re-applied
   against 0.26.0's XPU/fp4-era call sites.
4. Fuzz-misplacement repairs for both PR diffs (`patch --fuzz=3` twice produced code
   that *compiled* but was wrong — every fix here was re-verified by diffing the
   patched tree against stock).

A source-tree fork with everything applied is also available:
[allover326/vllm @ `compose-46994-38476-v0.26.0`](https://github.com/allover326/vllm/tree/compose-46994-38476-v0.26.0).

## Files

| file | purpose |
|---|---|
| `build_compose.sh` | one-shot image build from `vllm/vllm-openai:v0.26.0` |
| `pr46994-vllm-only.diff`, `pr38476-vllm-only.diff` | the two PRs, filtered to `vllm/` files |
| `fix_compose_skew.py` | the 15 skew fixes (respects `VLLM_TREE` for source checkouts) |
| `verify_patches.sh` | compile + import smoke + **undefined-name** check against a built image |
| `glm_mtp_bench.py` | baseline-vs-MTP decode benchmark + acceptance metrics |
| `gen_tiny_glm.py`, `tiny_glm_config.json`, `glm_key_manifest.json` | regenerate the tiny random-init `glm_moe_dsa` test checkpoint |
| `tiny_glm_test.py` | PP1 / PP2 / PP2+MTP greedy-equivalence matrix |

## License

Apache-2.0, same as vLLM. The bundled diffs are the respective PR authors' work,
carried unmodified from the vLLM project.
