#!/bin/bash
# setup_and_run.sh

set -e # エラーが発生した場合は直ちに終了

# TIKTOKEN_ENCODINGS_BASE ディレクトリを作成
mkdir -p "$TIKTOKEN_ENCODINGS_BASE"

# TIKTOKEN エンコーディングファイルをダウンロード
echo "Downloading tiktoken encoding files..."
wget -P "$TIKTOKEN_ENCODINGS_BASE" https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken
wget -P "$TIKTOKEN_ENCODINGS_BASE" https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken

# MODEL_HANDLEが設定されていることを確認
if [ -z "$MODEL_HANDLE" ]; then
    echo "Error: MODEL_HANDLE environment variable is not set." >&2
    exit 1
fi

# Hugging Face モデルをダウンロード
echo "Downloading Hugging Face model: $MODEL_HANDLE..."
hf download "$MODEL_HANDLE"

# LLM API の追加設定ファイルを作成
echo "Creating /tmp/extra-llm-api-config.yml..."
cat > /tmp/extra-llm-api-config.yml <<EOF
print_iter_log: false
kv_cache_config:
  dtype: "auto"
  free_gpu_memory_fraction: 0.9
cuda_graph_config:
  enable_padding: true
disable_overlap_scheduler: true
EOF

# TensorRT-LLM サービスを実行
echo "Starting trtllm-serve for model $MODEL_HANDLE on port 8355..."
trtllm-serve "$MODEL_HANDLE" \
  --max_batch_size 64 \
  --trust_remote_code \
  --port 8355 \
  --host 0.0.0.0 \
  --extra_llm_api_options /tmp/extra-llm-api-config.yml