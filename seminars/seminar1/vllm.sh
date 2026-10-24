HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 \
  uv run --frozen --extra metal vllm serve models/Qwen3.5-9B-8bit \
    --served-model-name qwen3.5-9b \
    --host 127.0.0.1 \
    --port 8001 \
    --language-model-only \
    --max-model-len 4096 \
    --max-num-seqs 1 \
    --max-num-batched-tokens 256 \
    --gpu-memory-utilization 0.62 \
    --no-enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": false}'