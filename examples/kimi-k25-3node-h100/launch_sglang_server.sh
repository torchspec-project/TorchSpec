# Example to launch a sglang server for multi-node inference with Kimi-K2.5 Eagle3 model.
export SGLANG_DISABLE_CUDNN_CHECK=1

# Multi-node network settings
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp27s0np0}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-enp27s0np0}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-enp27s0np0}

NNODES=${NNODES:-2}
NODE_RANK=${NODE_RANK:-0}
LOCAL_IP=${LOCAL_IP:-$(ip route get 1 | awk '{for(i=1;i<=NF;i++) if($i=="src") print $(i+1)}')}
# DIST_INIT_ADDR must be node 0's IP on ALL nodes
DIST_INIT_ADDR=${DIST_INIT_ADDR:?'DIST_INIT_ADDR must be set to node 0 IP:port (e.g. 10.0.0.1:50000)'}

python3 -m sglang.launch_server \
  --model-path moonshotai/Kimi-K2.5 \
  --port 12346 \
  --enable-metrics \
  --served-model-name moonshotai/Kimi-K2.5 \
  --tensor-parallel-size 16 \
  --host 0.0.0.0 \
  --context-length 262144 \
  --enable-multimodal \
  --tool-call-parser kimi_k2 \
  --reasoning-parser kimi_k2 \
  --trust-remote-code \
  --mem-fraction-static 0.85 \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 32}' \
  --disable-flashinfer-autotune \
  --nnodes $NNODES \
  --node-rank $NODE_RANK \
  --dist-init-addr $DIST_INIT_ADDR \
  --dist-timeout 600
