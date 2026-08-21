#!/bin/bash

# --- 配置区 ---
export MODEL_ID="qwen/Qwen2.5-VL-7B-Instruct"
export LOCAL_MODEL_PATH="/home/zhaoyan/llm/modelscope_models/Qwen3.5-4B-AWQ-4bit"  # 可改成您的实际路径

export VLLM_USE_MODELSCOPE=True
export VLLM_API_BASE="http://localhost:8000/v1"
export VLLM_USE_FLASHINFER_SAMPLER=0
export LD_LIBRARY_PATH="/home/zhaoyan/anaconda3/envs/vllm_cu13/lib:${LD_LIBRARY_PATH}"

# --- 模型路径选择 ---
if [ -d "$LOCAL_MODEL_PATH" ] && [ "$(ls -A $LOCAL_MODEL_PATH)" ]; then
    echo "使用本地模型: $LOCAL_MODEL_PATH"
    FINAL_MODEL_PATH="$LOCAL_MODEL_PATH"
else
    echo "从 ModelScope 加载: $MODEL_ID"
    FINAL_MODEL_PATH="$MODEL_ID"
fi

echo "启动 vLLM 服务器..."
python3 -m vllm.entrypoints.openai.api_server \
    --model "$FINAL_MODEL_PATH" \
    --trust-remote-code \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9 \
    --enforce-eager &

VLLM_PID=$!

echo "等待 vLLM 就绪 (最多5分钟)..."
for i in {1..60}; do
    if curl -s http://localhost:8000/v1/models > /dev/null; then
        echo "vLLM 服务已就绪！"
        READY=1
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "错误：vLLM 进程崩溃。"
        exit 1
    fi
    sleep 5
done

if [ "$READY" != "1" ]; then
    echo "超时，退出。"
    kill $VLLM_PID
    exit 1
fi

export VLLM_MODEL_NAME=$(basename "$FINAL_MODEL_PATH")   # 便于 app.py 获取
echo "启动 Gradio 应用..."
python3 app.py   # 您的 app.py 文件

kill $VLLM_PID