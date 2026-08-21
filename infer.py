"""
infer.py - vLLM API 客户端
"""

import base64
import io
import json
import requests
from typing import List, Optional
from PIL import Image

from config import VLLM_API_BASE, VLLM_MODEL_NAME, VLLM_TIMEOUT
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import subprocess
import re

def get_free_gpu_memory_gb(device=0):
    """返回指定 GPU 的空闲显存（GB），优先使用 torch，失败则回退到 nvidia-smi"""
    try:
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(device).total_memory
            allocated = torch.cuda.memory_allocated(device)
            free = total - allocated
            return free / (1024**3)
    except Exception:
        pass

    # 回退方案：调用 nvidia-smi 解析
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True
        )
        lines = result.stdout.strip().split('\n')
        if device < len(lines):
            free_mb = float(lines[device])
            return free_mb / 1024.0   # 转换为 GB
    except Exception:
        pass
    return None  # 无法获取
# ------------------------------------------------------------------
# 工具函数：图片转 base64（用于 vLLM 的多模态输入）
# ------------------------------------------------------------------
def image_to_base64(image_path: str, max_size: tuple = (512, 512)) -> str:
    """
    将图片缩放并编码为 base64 JPEG，以节省 token 和传输时间。
    """
    try:
        with Image.open(image_path) as img:
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        raise Exception(f"图片处理失败: {str(e)}")

# ------------------------------------------------------------------
# vLLM 推理类（单例模式）
# ------------------------------------------------------------------
class VLLMInference:
    """
    单例推理器，通过 vLLM API 进行多模态推理。
    """

    def __init__(self):
        self.api_base = VLLM_API_BASE
        self.model_name = None  # 自动检测
        self._detect_model()

    def _detect_model(self):
        """从 vLLM 服务获取当前加载的模型 ID"""
        try:
            resp = requests.get(f"{self.api_base}/models", timeout=5)
            if resp.status_code == 200:
                models = resp.json().get("data", [])
                if models:
                    self.model_name = models[0]["id"]
                    print(f"[vLLM] 已检测到模型: {self.model_name}")
                    return
        except Exception as e:
            print(f"[vLLM] 模型检测失败: {e}")
        # 若检测失败，使用配置或 fallback
        if VLLM_MODEL_NAME:
            self.model_name = VLLM_MODEL_NAME
        else:
            self.model_name = "qwen/Qwen2.5-VL-7B-Instruct"  # 默认
        print(f"[vLLM] 使用指定模型: {self.model_name}")

    def _call_api(self, image_path: str, prompt: str, max_tokens: int, temperature: float) -> str:
        """调用 vLLM /chat/completions 接口"""
        # 图片编码
        img_base64 = image_to_base64(image_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                    {"type": "text", "text": prompt},
                ]
            }
        ]

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            # "chat_template_kwargs": {
            #     "enable_thinking": False
            # }
        }

        try:
            response = requests.post(
                f"{self.api_base}/chat/completions",
                json=payload,
                timeout=VLLM_TIMEOUT
            )
            response.raise_for_status()
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            return content
        except requests.exceptions.RequestException as e:
            return f"[vLLM 错误] {str(e)}"
        except Exception as e:
            return f"[处理错误] {str(e)}"

    def infer(
        self,
        image_path: str,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        do_sample: bool = True,   # 保留兼容，vLLM 通过 temperature 控制
    ) -> str:
        """
        单张图片推理（兼容原有接口）
        """
        return self._call_api(image_path, prompt, max_new_tokens, temperature)

    def infer_batch(
        self,
        image_paths: List[str],
        prompt: str,
        batch_size: int = 4,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
    ) -> List[str]:
        """
        批量推理：逐张调用 API（vLLM 本身也支持 batch，但本实现顺序调用）
        """
        results = []
        for path in image_paths:
            results.append(self.infer(path, prompt, max_new_tokens, temperature))
        return results

    def gpu_info(self):
        """
        返回 vLLM 服务状态信息（替代原来的 GPU 显存信息）
        """
        try:
            resp = requests.get(f"{self.api_base}/models", timeout=3)
            if resp.status_code == 200:
                models = resp.json().get("data", [])
                if models:
                    model_info = models[0]
                    return {
                        "status": "vLLM 服务正常运行",
                        "model": model_info.get("id", "未知"),
                        "details": f"已加载模型: {model_info.get('id')}"
                    }
                else:
                    return {"status": "vLLM 服务运行中，但未加载模型"}
            else:
                return {"status": "vLLM 服务异常", "error": f"HTTP {resp.status_code}"}
        except Exception as e:
            return {"status": "vLLM 服务不可用", "error": str(e)}

    def infer_batch_concurrent(
            self,
            image_paths: List[str],
            prompt: str,
            max_new_tokens: int = 512,
            temperature: float = 0.2,
            max_workers: int = 4,  # 并发数，可根据 vLLM 服务能力调整
    ) -> List[str]:
        """并发批量推理，保持顺序"""
        results = [None] * len(image_paths)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.infer, path, prompt, max_new_tokens, temperature): idx
                for idx, path in enumerate(image_paths)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = f"[错误] {str(e)}"
        return results
# ------------------------------------------------------------------
# 全局单例 & 对外接口（与原有保持一致）
# ------------------------------------------------------------------
_model = None

def load_model(model_path: str = None):
    """
    加载模型（实际为初始化 vLLM 客户端），保持接口兼容
    """
    global _model
    if _model is None:
        _model = VLLMInference()
    return _model

def infer(image_path, prompt):
    global _model
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    return _model.infer(image_path, prompt)

def infer_batch(image_paths, prompt):
    global _model
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    return _model.infer_batch(image_paths, prompt)

def gpu_info():
    global _model
    if _model is None:
        return {"status": "未连接 vLLM 服务"}
    return _model.gpu_info()

def get_model():
    global _model
    return _model