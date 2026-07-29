import time, gc, os, torch
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

M = "/models/tiny-glm-dsa"
PROMPT_IDS = list(range(1000, 1064))  # 64 fixed token ids
SP = SamplingParams(max_tokens=64, temperature=0.0, ignore_eos=True)

def bench(tag, pp, spec):
    print("\n" + "=" * 58, flush=True)
    print(tag, flush=True)
    print("=" * 58, flush=True)
    kw = {}
    if spec:
        kw["speculative_config"] = {"method": "mtp", "num_speculative_tokens": 1}
    try:
        llm = LLM(model=M, tensor_parallel_size=1, pipeline_parallel_size=pp,
                  max_model_len=2048, gpu_memory_utilization=0.35,
                  trust_remote_code=True, enforce_eager=False, **kw)
        outs = llm.generate([TokensPrompt(prompt_token_ids=PROMPT_IDS)], SP, use_tqdm=False)
        toks = list(outs[0].outputs[0].token_ids)
        print(f"RESULT {tag}: OK  n={len(toks)}", flush=True)
        print(f"TOKENS {tag}: {toks}", flush=True)
        del llm; gc.collect(); torch.cuda.empty_cache()
        time.sleep(5)
        return toks
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"RESULT {tag}: FAILED -> {type(e).__name__}: {str(e)[:300]}", flush=True)
        gc.collect()
        return None

if __name__ == "__main__":
    print("backend:", os.environ.get("VLLM_ATTENTION_BACKEND"),
          "| v2:", os.environ.get("VLLM_USE_V2_MODEL_RUNNER"), flush=True)
    a = bench("A_PP1_noMTP", 1, False)
    b = bench("B_PP2_noMTP", 2, False)
    c = bench("C_PP2_MTP_n1", 2, True)
    print("\n" + "=" * 58, flush=True)
    if a and b: print(f"EQUIV PP1==PP2 (no MTP): {a == b}", flush=True)
    if b and c: print(f"EQUIV PP2==PP2+MTP:      {b == c}", flush=True)
    if a and c: print(f"EQUIV PP1==PP2+MTP:      {a == c}", flush=True)
    print("=" * 58, flush=True)
