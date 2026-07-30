"""GLM-5.2 MTP+PP benchmark for the rental rig.

Usage (inside the compose image):
    python3 glm_mtp_bench.py /models/GLM-5.2-Int4-Int8Mix [pp_size]

Runs PP<N> baseline then PP<N>+MTP n=1, reports decode t/s, acceptance,
and greedy-output agreement. Expects the weights pre-downloaded.
"""
import sys, time, gc, torch
from vllm import LLM, SamplingParams

M = sys.argv[1] if len(sys.argv) > 1 else "/models/GLM-5.2-Int4-Int8Mix"
PP = int(sys.argv[2]) if len(sys.argv) > 2 else 8

PROMPTS = ["Explain step by step how a modern CPU pipeline works.\n"]
SP_GEN = SamplingParams(max_tokens=256, temperature=0.0, ignore_eos=True)


def bench(tag, spec):
    print("\n" + "=" * 58 + f"\n{tag}\n" + "=" * 58, flush=True)
    kw = {}
    if spec:
        kw["speculative_config"] = {"method": "mtp", "num_speculative_tokens": 1}
    try:
        t0 = time.time()
        # block_size=64 is REQUIRED for real DSA models (kernel block-size
        # negotiation fails at the engine default of 16). The MTP config runs
        # eager at lower util: the draft concentrates extra weight on the last
        # PP rank and graph capture pushed it OOM at 0.90 on 8x64GB (2026-07-30).
        llm = LLM(model=M, tensor_parallel_size=1, pipeline_parallel_size=PP,
                  max_model_len=32768 if not spec else 16384,
                  gpu_memory_utilization=0.90 if not spec else 0.80,
                  block_size=64,
                  kv_cache_dtype="auto", trust_remote_code=True,
                  disable_log_stats=False, enforce_eager=bool(spec), **kw)
        load = time.time() - t0
        llm.generate(PROMPTS, SamplingParams(max_tokens=8, temperature=0.0), use_tqdm=False)
        t0 = time.time()
        outs = llm.generate(PROMPTS, SP_GEN, use_tqdm=False)
        dt = time.time() - t0
        n = len(outs[0].outputs[0].token_ids)
        txt = outs[0].outputs[0].text
        print(f"RESULT {tag}: load={load:.0f}s decode={n/dt:.2f} tok/s ({n} toks / {dt:.2f}s)", flush=True)
        print(f"TEXT {tag}: {txt[:300]!r}", flush=True)
        try:
            for m in llm.get_metrics():
                nm = m.name.lower()
                if "spec" in nm or "accept" in nm:
                    v = getattr(m, "value", None) or getattr(m, "values", None)
                    print(f"METRIC {m.name} = {v}", flush=True)
        except Exception as e:
            print(f"(metrics: {e})", flush=True)
        del llm; gc.collect(); torch.cuda.empty_cache()
        time.sleep(10)
        return n / dt, txt
    except Exception:
        import traceback; traceback.print_exc()
        return None, None


if __name__ == "__main__":
    base, base_txt = bench(f"A_PP{PP}_baseline", False)
    mtp, mtp_txt = bench(f"B_PP{PP}_MTP_n1", True)
    print("\n" + "=" * 58, flush=True)
    if base and mtp:
        print(f"SPEEDUP: {mtp/base:.2f}x  (baseline {base:.1f} -> MTP {mtp:.1f} tok/s)", flush=True)
        if base_txt and mtp_txt:
            print(f"GREEDY MATCH: {base_txt.strip() == mtp_txt.strip()}", flush=True)
    else:
        print("A config failed - see traceback above.", flush=True)
    print("=" * 58, flush=True)
