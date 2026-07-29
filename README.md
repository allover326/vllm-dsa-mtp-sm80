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
| this repo | 15 skew fixes composing both onto v0.26.0 (`fix_compose_skew.py`) | — |

All credit for the heavy lifting goes to the authors of those two PRs. This repo exists
because neither is merged yet and they target different eras of `main`.

## Quick start

```bash
git clone https://github.com/cachenetics/vllm-dsa-mtp-sm80
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
  --max-model-len 32768 --gpu-memory-utilization 0.90 \
  --kv-cache-dtype auto --trust-remote-code \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

Benchmark baseline-vs-MTP instead with `glm_mtp_bench.py` (see file header).

## Validation status (2026-07-29)

- **Tiny random-init `glm_moe_dsa` (incl. MTP nextn layer), 2× RTX 3090:** PP1 ✅,
  PP2 ✅, **PP2 + MTP n=1 ✅ — token-for-token identical to the PP2 baseline
  (greedy-equivalent)**. Regenerate the test checkpoint with `gen_tiny_glm.py`,
  run the matrix with `tiny_glm_test.py`.
- **MiMo-7B-RL real weights (PP2, 2× 3090, #46994 alone):** 54.4 → 69.5 tok/s
  (**1.28×**), draft acceptance **70.9%**, no deadlock, cudagraphs on.
- **GLM-5.2-744B real weights:** in progress — this section will be updated with
  measured PP8 numbers. Reference: the same 8×170HX rig does 30.2 tok/s decode /
  2,675 tok/s prefill @131k without MTP (vLLM 0.20.2 + #38476).

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
documents all 15, each asserting its exact target):

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
[cachenetics/vllm @ `compose-46994-38476-v0.26.0`](https://github.com/cachenetics/vllm/tree/compose-46994-38476-v0.26.0).

## Files

| file | purpose |
|---|---|
| `build_compose.sh` | one-shot image build from `vllm/vllm-openai:v0.26.0` |
| `pr46994-vllm-only.diff`, `pr38476-vllm-only.diff` | the two PRs, filtered to `vllm/` files |
| `fix_compose_skew.py` | the 15 skew fixes (respects `VLLM_TREE` for source checkouts) |
| `glm_mtp_bench.py` | baseline-vs-MTP decode benchmark + acceptance metrics |
| `gen_tiny_glm.py`, `tiny_glm_config.json`, `glm_key_manifest.json` | regenerate the tiny random-init `glm_moe_dsa` test checkpoint |
| `tiny_glm_test.py` | PP1 / PP2 / PP2+MTP greedy-equivalence matrix |

## License

Apache-2.0, same as vLLM. The bundled diffs are the respective PR authors' work,
carried unmodified from the vLLM project.
