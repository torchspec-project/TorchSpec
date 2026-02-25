export SGLANG_DISABLE_CUDNN_CHECK=1

python3 -m sglang.launch_server \
  --model-path moonshotai/Kimi-K2.5 \
  --port 12346 \
  --enable-metrics \
  --served-model-name moonshotai/Kimi-K2.5 \
  --tensor-parallel-size 8 \
  --host 0.0.0.0 \
  --context-length 262144 \
  --enable-multimodal \
  --tool-call-parser kimi_k2 \
  --reasoning-parser kimi_k2 \
  --trust-remote-code \
  --mem-fraction-static 0.85 \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 32}' \
  --disable-flashinfer-autotune
