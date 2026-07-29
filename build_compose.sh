#!/bin/bash
# Build vllm-mtp:compose46994-38476 from stock vllm/vllm-openai:v0.26.0.
# Usage: ./build_compose.sh [output-tag]
# Needs: docker; this dir must contain pr46994-vllm-only.diff,
#        pr38476-vllm-only.diff, fix_compose_skew.py
set -e
IMG_BASE=vllm/vllm-openai:v0.26.0
IMG_OUT=${1:-vllm-mtp:compose46994-38476}
WORK=$(cd "$(dirname "$0")" && pwd)
NAME=compose-build-auto

docker image inspect $IMG_BASE >/dev/null 2>&1 || docker pull $IMG_BASE
docker rm -f $NAME 2>/dev/null || true
docker run -d --name $NAME --entrypoint sleep -v "$WORK":/work $IMG_BASE 7200

echo "== applying PR #46994 (must apply 13/13) =="
docker exec $NAME bash -c "cd /usr/local/lib/python3.12/dist-packages && patch -p1 --forward --fuzz=3 < /work/pr46994-vllm-only.diff"

echo "== applying PR #38476 (3 sparse_attn_indexer hunks EXPECTED to fail; fixed next) =="
docker exec $NAME bash -c "cd /usr/local/lib/python3.12/dist-packages && patch -p1 --forward --fuzz=3 < /work/pr38476-vllm-only.diff; true"

echo "== applying consolidated skew fixes =="
docker exec $NAME python3 /work/fix_compose_skew.py

echo "== compile + cleanup =="
docker exec $NAME bash -c 'cd /usr/local/lib/python3.12/dist-packages && \
  rm -f vllm/model_executor/layers/sparse_attn_indexer.py.rej && \
  find vllm -name "*.orig" -delete && \
  python3 -m py_compile \
    vllm/model_executor/models/deepseek_mtp.py \
    vllm/model_executor/models/qwen3_5_mtp.py \
    vllm/model_executor/models/mimo_mtp.py \
    vllm/model_executor/models/deepseek_v2.py \
    vllm/model_executor/layers/sparse_attn_indexer.py \
    vllm/model_executor/layers/attention/mla_attention.py \
    vllm/platforms/cuda.py \
    vllm/v1/attention/backends/registry.py \
    vllm/v1/attention/backends/mla/indexer.py \
    vllm/v1/attention/backends/mla/triton_mla_sparse.py \
    vllm/v1/attention/ops/mqa_logits_triton.py \
    vllm/v1/attention/ops/triton_mla_sparse_kernel.py \
    vllm/v1/worker/gpu/model_runner.py \
    vllm/v1/worker/gpu/pp_utils.py \
    vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py \
    vllm/v1/worker/gpu/spec_decode/speculator.py && echo COMPILE_OK'

echo "== import smoke test (needs GPU visible for vllm._C; failure of _C-only is OK on a CPU box) =="
docker exec $NAME python3 -c "
from vllm.v1.attention.backends.registry import AttentionBackendEnum as E
assert hasattr(E, 'TRITON_MLA_SPARSE')
import importlib
importlib.import_module('vllm.v1.attention.backends.mla.triton_mla_sparse')
importlib.import_module('vllm.v1.attention.ops.mqa_logits_triton')
importlib.import_module('vllm.model_executor.layers.sparse_attn_indexer')
print('SMOKE_OK')"

docker commit $NAME "$IMG_OUT"
docker rm -f $NAME
echo "BUILT: $IMG_OUT"
