# config.py
import os

# --- 模型与 API 配置 ---
MODEL_PATH = "models/Qwen3.5-VL-9B"  # 已废弃，保留但不再使用

# vLLM 服务地址（与参考脚本一致）
VLLM_API_BASE = os.getenv("VLLM_API_BASE", "http://localhost:8000/v1")

# 自动检测模型名称（通过 /models 端点获取），也可手动指定
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "")  # 若空则自动检测

# 请求超时（秒）
VLLM_TIMEOUT = 300

# --- 原有参数（保持不变） ---
DEFAULT_PROMPT = "你是一名铁路工务巡检工程师。请分析这张图片，判断铁轨扣件（弹条、挡板等）是否被沙尘掩埋。\n输出要点： 简述图像清晰度及扣件可见情况。判断覆盖程度（完全裸露 部分覆盖 完全掩埋）及积沙形态。 \n给出最终结论（存在 轻微 无）及风险等级（安全 注意 危险），若为注意或危险，附一句清沙建议。 若图片看不清，请直接说明 无法准确判断 。回复要简洁专业。"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_TOKENS = 512
DEFAULT_BATCH_SIZE = 4
# 自动并发调节
AUTO_BATCH_SIZE = True          # 是否启用自动调节
MAX_CONCURRENT_REQUESTS = 8     # 并发数上限（防止过大导致服务崩溃）
MEMORY_PER_REQUEST_GB = 2.0     # 每个并发请求预估占用的显存（GB），可根据实际情况调整


# 视频抽帧间隔
FRAME_INTERVAL = 60
