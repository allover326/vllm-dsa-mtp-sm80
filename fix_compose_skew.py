"""All hand-fixes for vllm-mtp:compose46994-38476, applied AFTER both PR diffs.

Run inside the container (dist-packages already patched with
pr46994-vllm-only.diff and pr38476-vllm-only.diff, fuzz=3).
Every replacement asserts exactly-one occurrence: a failed assert means the
fuzz application landed differently than expected -- STOP and inspect.

Set VLLM_TREE to point at a different vllm package root (e.g. a source
checkout's vllm/ dir) -- defaults to the official image's dist-packages.
"""
import os

SP = os.environ.get("VLLM_TREE", "/usr/local/lib/python3.12/dist-packages/vllm")


def sub(path, old, new, tag):
    # newline="" disables platform newline translation so running this on
    # Windows (e.g. against a source checkout) cannot rewrite LF as CRLF.
    p = f"{SP}/{path}"
    with open(p, encoding="utf-8", newline="") as f:
        s = f.read()
    n = s.count(old)
    assert n == 1, f"{tag}: pattern count={n} in {path}"
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write(s.replace(old, new))
    print(f"OK {tag}")


# ---- PR #46994 fuzz repairs (autoregressive speculator) ----
sub("v1/worker/gpu/spec_decode/autoregressive/speculator.py",
"""            num_tokens_across_dp,
        )
            intermediate_tensors=intermediate_tensors,
""",
"""            num_tokens_across_dp,
            intermediate_tensors=intermediate_tensors,
        )
""", "46994-speculator-callsite")

sub("v1/worker/gpu/spec_decode/autoregressive/speculator.py",
"""        num_tokens_across_dp: torch.Tensor | None,
    ) -> None:
        intermediate_tensors: IntermediateTensors | None = None,
""",
"""        num_tokens_across_dp: torch.Tensor | None,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> None:
""", "46994-speculator-signature")

# ---- MiMo MTP SupportsPP (PR only covers deepseek/qwen3.5) ----
sub("model_executor/models/mimo_mtp.py",
"from .utils import AutoWeightsLoader, WeightsMapper, maybe_prefix",
"""from .interfaces import SupportsPP
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    make_empty_intermediate_tensors_factory,
    maybe_prefix,
)""", "mimo-imports")

sub("model_executor/models/mimo_mtp.py",
"class MiMoMTP(nn.Module):",
"class MiMoMTP(nn.Module, SupportsPP):", "mimo-class")

sub("model_executor/models/mimo_mtp.py",
"""        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )""",
"""        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        # PP support (mirrors PR #46994 deepseek_mtp.py): the MTP draft runs only
        # on the last PP stage, so it never consumes PP intermediate tensors, but
        # SupportsPP requires this factory.
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], self.config.hidden_size
        )""", "mimo-factory")

# ---- PR #38476 fuzz repair: hunk 3 landed inside a Triton kernel ----
sub("model_executor/layers/sparse_attn_indexer.py",
"""        tl.store(
            out_base + 32 + offs32,
            tl.clamp(r1 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
    # DeepGEMM availability is constant per process; check once for both branches.
    use_deep_gemm = is_deep_gemm_supported()
    if not use_deep_gemm:
        assert not use_fp4_cache, (
            "Triton sparse-MLA fallback does not support FP4 KV cache"
        )
        )
""",
"""        tl.store(
            out_base + 32 + offs32,
            tl.clamp(r1 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
""", "38476-kernel-repair")

# ---- #38476 hunk 3 intent: probe at function level ----
sub("model_executor/layers/sparse_attn_indexer.py",
"""    if not skip_topk_buffer_clear:
        topk_indices_buffer[: hidden_states.shape[0]] = -1
""",
"""    if not skip_topk_buffer_clear:
        topk_indices_buffer[: hidden_states.shape[0]] = -1
    # DeepGEMM availability is constant per process; check once for both branches.
    use_deep_gemm = is_deep_gemm_supported()
    if not use_deep_gemm:
        assert not use_fp4_cache, (
            "Triton sparse-MLA fallback does not support FP4 KV cache"
        )
""", "38476-probe")

# ---- #38476 hunk 4 intent: prefill branch ----
sub("model_executor/layers/sparse_attn_indexer.py",
"""                else:
                    logits = fp8_fp4_mqa_logits(
                        (q_slice_cast, q_scale_slice),
                        (k_quant_cast, k_scale_cast),
                        weights[chunk.token_start : chunk.token_end],
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                        clean_logits=False,
                    )
""",
"""                elif use_deep_gemm:
                    logits = fp8_fp4_mqa_logits(
                        (q_slice_cast, q_scale_slice),
                        (k_quant_cast, k_scale_cast),
                        weights[chunk.token_start : chunk.token_end],
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                        clean_logits=False,
                    )
                else:
                    logits = fp8_mqa_logits_triton(
                        q_slice_cast,
                        (k_quant_cast, k_scale_cast),
                        weights[chunk.token_start : chunk.token_end],
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                        clean_logits=False,
                    )
""", "38476-prefill-branch")

# ---- #38476 hunk 5 intent: decode branch ----
sub("model_executor/layers/sparse_attn_indexer.py",
"""        else:
            logits = fp8_fp4_paged_mqa_logits(
                (padded_q_quant_cast, padded_q_scale),
                kv_cache,
                weights[:num_padded_tokens],
                seq_lens,
                decode_metadata.block_table,
                decode_metadata.schedule_metadata,
                max_model_len=max_model_len,
                clean_logits=False,
            )
""",
"""        elif use_deep_gemm:
            logits = fp8_fp4_paged_mqa_logits(
                (padded_q_quant_cast, padded_q_scale),
                kv_cache,
                weights[:num_padded_tokens],
                seq_lens,
                decode_metadata.block_table,
                decode_metadata.schedule_metadata,
                max_model_len=max_model_len,
                clean_logits=False,
            )
        else:
            # SM80/SM121 Triton fallback. Downstream topk reads only up to
            # `seq_lens`, so size the buffer to the active batch max rather
            # than the configured model max.
            active_max_model_len = attn_metadata_narrowed.max_seq_len
            logits = fp8_paged_mqa_logits_triton(
                padded_q_quant_cast,
                kv_cache,
                weights[:num_padded_tokens],
                seq_lens,
                decode_metadata.block_table,
                max_model_len=active_max_model_len,
                clean_logits=False,
            )
""", "38476-decode-branch")

# ---- #38476 hunk 6 intent: __init__ DeepGEMM error -> warning ----
sub("model_executor/layers/sparse_attn_indexer.py",
"""        if current_platform.is_cuda() and not has_deep_gemm():
            raise RuntimeError(
                "Sparse Attention Indexer CUDA op requires DeepGEMM support in "
                "the current vLLM environment."
            )
""",
"""        if current_platform.is_cuda() and not is_deep_gemm_supported():
            logger.warning_once(
                "DeepGEMM not supported on this platform; "
                "using Triton fallback for sparse attention indexer."
            )
""", "38476-init-warning")

# ---- cuda.py: fuzz put the enum in the SM100 list; move to SM8x/9x else ----
sub("platforms/cuda.py",
"""                AttentionBackendEnum.TRITON_MLA,
                *sparse_backends,
                AttentionBackendEnum.TRITON_MLA_SPARSE,
            ]""",
"""                AttentionBackendEnum.TRITON_MLA,
                *sparse_backends,
            ]""", "cuda-remove-sm100")

sub("platforms/cuda.py",
"""                AttentionBackendEnum.TRITON_MLA,
                AttentionBackendEnum.FLASH_ATTN_MLA_SPARSE,
                AttentionBackendEnum.FLASHMLA_SPARSE,
            ]""",
"""                AttentionBackendEnum.TRITON_MLA,
                AttentionBackendEnum.FLASH_ATTN_MLA_SPARSE,
                AttentionBackendEnum.FLASHMLA_SPARSE,
                AttentionBackendEnum.TRITON_MLA_SPARSE,
            ]""", "cuda-add-sm8x")

# ---- deepseek_v2.py: fused fp8e4nv Triton kernel cannot compile <sm_89 ----
sub("model_executor/models/deepseek_v2.py",
"""        self.use_fused_indexer_q = (
            current_platform.is_cuda()
            and self.quant_block_size == self.head_dim""",
"""        self.use_fused_indexer_q = (
            current_platform.is_cuda()
            # Triton fp8e4nv converts need sm_89+; older archs use the
            # unfused path (per_token_group_quant_fp8 CUDA op).
            and current_platform.has_device_capability(89)
            and self.quant_block_size == self.head_dim""", "dsv2-fused-gate")

# ---- mla_attention.py: XPU/Triton sparse metadata lacks decode-split fields ----
sub("model_executor/layers/attention/mla_attention.py",
"            attn_metadata.num_decode_tokens if attn_metadata is not None else None,",
'            getattr(attn_metadata, "num_decode_tokens", None) if attn_metadata is not None else None,',
"mla-kvupdate-getattr")

sub("model_executor/layers/attention/mla_attention.py",
"""        assert (
            attn_metadata.num_decodes is not None
            and attn_metadata.num_prefills is not None
            and attn_metadata.num_decode_tokens is not None
        )
        num_mqa_tokens = attn_metadata.num_decode_tokens
        num_mha_tokens = q.size(0) - num_mqa_tokens

        if self.impl.is_sparse and num_mha_tokens > 0:
""",
"""        if not hasattr(attn_metadata, "num_decodes"):
            # XPU/Triton sparse-MLA metadata has no decode/prefill split;
            # those impls support only forward_mqa (matches upstream main).
            num_mqa_tokens = q.size(0)
            num_mha_tokens = 0
        else:
            assert (
                attn_metadata.num_decodes is not None
                and attn_metadata.num_prefills is not None
                and attn_metadata.num_decode_tokens is not None
            )
            num_mqa_tokens = attn_metadata.num_decode_tokens
            num_mha_tokens = q.size(0) - num_mqa_tokens

        if self.impl.is_sparse and num_mha_tokens > 0:
""", "mla-forward-guard")

sub("model_executor/layers/attention/mla_attention.py",
"""            and attn_metadata.prefill is not None
            and attn_metadata.prefill.chunked_context is None""",
"""            and getattr(attn_metadata, "prefill", None) is not None
            and attn_metadata.prefill.chunked_context is None""",
"mla-prefill-getattr")

print("ALL_FIXES_APPLIED")
