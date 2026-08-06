#!/bin/bash
# Verify a built compose image the way it should have been verified before release.
#
#   ./verify_patches.sh [image-tag]        # default: vllm-mtp:compose46994-38476
#
# Three checks, in increasing order of what they catch:
#   1. py_compile        - syntax only. Passes on code that cannot run.
#   2. import smoke      - catches import-time breakage.
#   3. UNDEFINED NAMES   - catches a patch that CALLS something no patch DEFINES.
#
# Check 3 is the one that matters. A patch generated against a working tree that
# already contains a helper will capture the call sites and not the helper: the
# diff applies on the author's machine and raises NameError on everyone else's.
# That is exactly what happened to the sibling repo deepseek-v4-cmp170hx, where
# patch 0006 called two functions that no patch defined, and it shipped:
#   https://github.com/allover326/deepseek-v4-cmp170hx/issues/1
# py_compile and an import smoke test BOTH pass on that bug, because the missing
# name is only resolved when the function actually runs.
set -u
IMG="${1:-vllm-mtp:compose46994-38476}"
NAME=verify-patches-$$

FILES="
vllm/model_executor/models/deepseek_mtp.py
vllm/model_executor/models/qwen3_5_mtp.py
vllm/model_executor/models/mimo_mtp.py
vllm/model_executor/models/deepseek_v2.py
vllm/model_executor/layers/sparse_attn_indexer.py
vllm/model_executor/layers/attention/mla_attention.py
vllm/platforms/cuda.py
vllm/v1/attention/backends/registry.py
vllm/v1/attention/backends/mla/indexer.py
vllm/v1/attention/backends/mla/triton_mla_sparse.py
vllm/v1/attention/ops/mqa_logits_triton.py
vllm/v1/attention/ops/triton_mla_sparse_kernel.py
vllm/v1/worker/gpu/model_runner.py
vllm/v1/worker/gpu/pp_utils.py
vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py
vllm/v1/worker/gpu/spec_decode/speculator.py
"

docker image inspect "$IMG" >/dev/null 2>&1 || { echo "no such image: $IMG"; exit 1; }
docker rm -f $NAME >/dev/null 2>&1
docker run -d --name $NAME --entrypoint sleep "$IMG" 600 >/dev/null || exit 1
trap 'docker rm -f $NAME >/dev/null 2>&1' EXIT
SP=/usr/local/lib/python3.12/dist-packages
FAIL=0

echo "== 1. py_compile =="
if docker exec $NAME sh -c "cd $SP && python3 -m py_compile $(echo $FILES | tr '\n' ' ')"; then
  echo "   COMPILE_OK"
else
  echo "   COMPILE_FAILED"; FAIL=1
fi

echo "== 2. import smoke =="
if docker exec $NAME python3 -c "
from vllm.v1.attention.backends.registry import AttentionBackendEnum as E
assert hasattr(E, 'TRITON_MLA_SPARSE')
import importlib
for m in ('vllm.v1.attention.backends.mla.triton_mla_sparse',
          'vllm.v1.attention.ops.mqa_logits_triton',
          'vllm.model_executor.layers.sparse_attn_indexer'):
    importlib.import_module(m)
print('   SMOKE_OK')"; then :; else echo "   SMOKE_FAILED"; FAIL=1; fi

echo "== 3. undefined names (the one that catches missing definitions) =="
OUT=$(docker exec $NAME sh -c "
  pip install -q pyflakes >/dev/null 2>&1 || { echo __NOPYFLAKES__; exit 0; }
  cd $SP && python3 -m pyflakes $(echo $FILES | tr '\n' ' ') 2>&1 | grep 'undefined name' || true
")
if echo "$OUT" | grep -q __NOPYFLAKES__; then
  echo "   PYFLAKES UNAVAILABLE - check SKIPPED (this is not a pass)"
  FAIL=1
elif [ -n "$OUT" ]; then
  echo "$OUT" | sed 's/^/   /'
  echo "   UNDEFINED_NAMES_FOUND - a patch calls something no patch defines."
  FAIL=1
else
  echo "   NO_UNDEFINED_NAMES"
fi

echo
if [ "$FAIL" -eq 0 ]; then echo "VERIFY_OK  ($IMG)"; else echo "VERIFY_FAILED  ($IMG)"; fi
exit $FAIL
