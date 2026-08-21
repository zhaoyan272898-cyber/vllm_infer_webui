# app.py
import os
import time
import tempfile
import pandas as pd
import gradio as gr
from typing import List
import torch
from config import (
    MODEL_PATH,
    DEFAULT_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_MAX_TOKENS,
    DEFAULT_BATCH_SIZE,
    FRAME_INTERVAL,
)
from utils import get_image_files, get_images_from_directory, is_image_file, is_video_file, extract_frames, ALLOWED_EXTENSIONS
from pdf_export import export_to_pdf, export_pdf
from infer import load_model, get_model   # 新增 get_model
from config import AUTO_BATCH_SIZE, MAX_CONCURRENT_REQUESTS, MEMORY_PER_REQUEST_GB
from infer import get_free_gpu_memory_gb   # 新增导入
# 全局停止标志
stop_flag = False

# 语言文本配置
LANG = {
    "zh": {
        "title": "Qwen3.5-VL WebUI",
        "upload_label": "拖拽或点击上传图片 / 文件夹",
        "prompt_label": "提示词",
        "temperature_label": "温度 (Temperature)",
        "top_p_label": "Top-p",
        "max_tokens_label": "最大生成 Token 数",
        "batch_size_label": "批处理大小 (仅供显示)",
        "start_btn": "▶ 开始推理",
        "stop_btn": "⏹ 停止",
        "gpu_status": "GPU 状态 (实时)",
        "model_status": "模型状态",
        "time_info": "推理耗时",
        "output_headers": ["图片", "结果", "耗时 (s)"],
        "export_pdf": "📄 导出 PDF（带图片）",
        "export_excel": "📊 导出 Excel",
        "export_md": "📝 导出 Markdown",
        "progress_desc": "处理中",
        "complete": "完成",
        "no_images": "未选择图片",
    },
    "en": {
        "title": "Qwen3.5-VL WebUI",
        "upload_label": "Drag or click to upload images / folder",
        "prompt_label": "Prompt",
        "temperature_label": "Temperature",
        "top_p_label": "Top-p",
        "max_tokens_label": "Max Tokens",
        "batch_size_label": "Batch Size (for display only)",
        "start_btn": "▶ Start Inference",
        "stop_btn": "⏹ Stop",
        "gpu_status": "GPU Status (Real-time)",
        "model_status": "Model Status",
        "time_info": "Inference Time",
        "output_headers": ["Image", "Result", "Time (s)"],
        "export_pdf": "📄 Export PDF (with images)",
        "export_excel": "📊 Export Excel",
        "export_md": "📝 Export Markdown",
        "progress_desc": "Processing",
        "complete": "Complete",
        "no_images": "No images selected",
    }
}

current_lang = "zh"

# ---------- 实时 GPU 状态更新函数 ----------
def update_gpu_info():
    """定时刷新 vLLM 服务状态"""
    model = get_model()
    if model is None:
        return "vLLM 服务未连接"
    info = model.gpu_info()  # 返回字典
    if info.get("status"):
        return f"状态: {info['status']}\n模型: {info.get('model', '未知')}\n详情: {info.get('details', '')}"
    else:
        return f"状态: {info.get('status', '未知')}\n错误: {info.get('error', '')}"

# ---------- 推理生成器（移除 GPU 状态输出） ----------


def inference_process(
    files: List[gr.FileData],
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    batch_size: int,
    progress=gr.Progress()
):
    global stop_flag
    stop_flag = False

    if not files:
        empty_state = {"image_paths": [], "result_texts": [], "question": ""}
        yield gr.update(value=[]), LANG[current_lang]["no_images"], "0s", empty_state
        return

    # 收集所有图片路径
    all_image_paths = []
    for file_data in files:
        path = file_data.name
        if is_image_file(path):
            all_image_paths.append(path)
        elif is_video_file(path):
            try:
                frames = extract_frames(path, interval=FRAME_INTERVAL)
                all_image_paths.extend(frames)
            except Exception as e:
                print(f"视频抽帧失败 {path}: {e}")

    if not all_image_paths:
        empty_state = {"image_paths": [], "result_texts": [], "question": ""}
        yield gr.update(value=[]), "未找到可处理的图片或视频", "0s", empty_state
        return

    # ---------- 自动调节并发数 ----------
    if AUTO_BATCH_SIZE:
        free_gb = get_free_gpu_memory_gb()
        if free_gb is not None:
            auto_concurrency = max(1, int(free_gb / MEMORY_PER_REQUEST_GB))
            auto_concurrency = min(auto_concurrency, MAX_CONCURRENT_REQUESTS)
            batch_size = auto_concurrency
            print(f"[INFO] 自动调节并发数为 {batch_size}（剩余显存 {free_gb:.2f} GB）")
        else:
            batch_size = batch_size or 4
    else:
        batch_size = batch_size or 4

    model = load_model(MODEL_PATH)

    total = len(all_image_paths)
    results = []
    result_texts = []
    total_time = 0.0

    # 分批并发处理
    for start_idx in range(0, total, batch_size):
        if stop_flag:
            break

        batch_paths = all_image_paths[start_idx:start_idx + batch_size]
        batch_size_actual = len(batch_paths)

        t0 = time.time()
        try:
            batch_results = model.infer_batch_concurrent(
                batch_paths,
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                max_workers=batch_size_actual
            )
        except Exception as e:
            batch_results = [f"[批次错误] {str(e)}"] * batch_size_actual

        t1 = time.time()
        elapsed_batch = t1 - t0
        total_time += elapsed_batch

        for idx_in_batch, (path, text) in enumerate(zip(batch_paths, batch_results)):
            elapsed_per = elapsed_batch / batch_size_actual
            filename = os.path.basename(path)
            results.append([filename, text, round(elapsed_per, 3)])
            result_texts.append(text)

        processed = start_idx + batch_size_actual
        progress(processed / total, desc=f"{LANG[current_lang]['progress_desc']} {processed}/{total}")

        df = pd.DataFrame(results, columns=LANG[current_lang]["output_headers"])
        state_dict = {
            "image_paths": all_image_paths[:processed],
            "result_texts": result_texts,
            "question": prompt,
        }
        yield df, f"已处理 {processed}/{total}", f"{total_time:.3f}s", state_dict

    final_status = LANG[current_lang]["complete"] if not stop_flag else "已停止"
    final_state_dict = {
        "image_paths": all_image_paths[:len(results)],
        "result_texts": result_texts,
        "question": prompt,
    }
    yield df, final_status, f"{total_time:.3f}s", final_state_dict

# ---------- 导出函数（带图片） ----------
def export_pdf_with_images(state_dict):
    if not state_dict or not state_dict.get("image_paths"):
        return None
    pdf_path = export_to_pdf(
        state_dict["image_paths"],
        state_dict["result_texts"],
        state_dict.get("question", "")
    )
    return pdf_path

# 纯文本导出（备选）
def export_pdf_text_only_fn(results_df: pd.DataFrame):
    if results_df is None or results_df.empty:
        return None
    records = results_df.values.tolist()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        export_pdf(records, tmp.name)
        return tmp.name

def export_excel_from_state(state_dict):
    if not state_dict or not state_dict.get("image_paths"):
        return None
    df = pd.DataFrame({
        "图片": [os.path.basename(p) for p in state_dict["image_paths"]],
        "结果": state_dict["result_texts"],
    })
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        df.to_excel(tmp.name, index=False)
        return tmp.name

def export_md_from_state(state_dict):
    if not state_dict or not state_dict.get("image_paths"):
        return None
    df = pd.DataFrame({
        "图片": [os.path.basename(p) for p in state_dict["image_paths"]],
        "结果": state_dict["result_texts"],
    })
    md = "# Qwen3.5-VL 推理结果\n\n"
    md += df.to_markdown(index=False)
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tmp:
        tmp.write(md.encode('utf-8'))
        return tmp.name

def stop_inference():
    global stop_flag
    stop_flag = True
    return "停止请求已发送"

def reset_stop():
    global stop_flag
    stop_flag = False

def update_language(lang: str):
    global current_lang
    current_lang = lang
    return (
        gr.update(label=LANG[lang]["upload_label"]),
        gr.update(label=LANG[lang]["prompt_label"]),
        gr.update(label=LANG[lang]["temperature_label"]),
        gr.update(label=LANG[lang]["top_p_label"]),
        gr.update(label=LANG[lang]["max_tokens_label"]),
        gr.update(label=LANG[lang]["batch_size_label"]),
        gr.update(value=LANG[lang]["start_btn"]),
        gr.update(value=LANG[lang]["stop_btn"]),
        gr.update(value=LANG[lang]["gpu_status"]),
        gr.update(value=LANG[lang]["model_status"]),
        gr.update(value=LANG[lang]["time_info"]),
        gr.update(value=LANG[lang]["export_pdf"]),
        gr.update(value=LANG[lang]["export_excel"]),
        gr.update(value=LANG[lang]["export_md"]),
    )

def update_gallery(files):
    if not files:
        return [], None
    file_paths = [f.name for f in files]
    image_paths = get_image_files(file_paths)   # 只取图片文件（视频不显示预览）
    return image_paths, image_paths

def load_model_at_start():
    try:
        model = load_model(MODEL_PATH)
        #return f"已加载: {MODEL_PATH}"
        return f"已加载模型"
    except Exception as e:
        return f"加载失败: {str(e)}"

# ---------- 构建 UI ----------
with gr.Blocks(title="Qwen3.5-VL WebUI", theme=gr.themes.Soft()) as demo:
    with gr.Row():
        lang_dropdown = gr.Dropdown(choices=["zh", "en"], value="zh", label="语言 / Language", interactive=True)

    gr.Markdown("# Qwen3.5-VL WebUI")

    with gr.Row():
        with gr.Column(scale=1):
            upload = gr.File(
                label="拖拽或点击上传图片 / 视频（视频自动抽帧）",
                file_count="multiple",
                file_types=ALLOWED_EXTENSIONS,
                interactive=True,
            )
            gallery = gr.Gallery(label="图片预览", columns=4, rows=2, height="auto", object_fit="contain")

        with gr.Column(scale=1):
            prompt = gr.Textbox(label="提示词", value=DEFAULT_PROMPT, lines=3, placeholder="请输入你的问题或指令...")
            with gr.Row():
                temperature = gr.Slider(minimum=0.0, maximum=1.0, value=DEFAULT_TEMPERATURE, step=0.01, label="温度 (Temperature)")
                top_p = gr.Slider(minimum=0.0, maximum=1.0, value=DEFAULT_TOP_P, step=0.01, label="Top-p")
            max_tokens = gr.Slider(minimum=1, maximum=2048, value=DEFAULT_MAX_TOKENS, step=1, label="最大生成 Token 数")
            batch_size = gr.Number(value=DEFAULT_BATCH_SIZE, label="批处理大小 (仅供显示)", precision=0, interactive=False)
            with gr.Row():
                start_btn = gr.Button("▶ 开始推理", variant="primary")
                stop_btn = gr.Button("⏹ 停止", variant="stop")

        with gr.Column(scale=1):
            gpu_status_label = gr.Label(value="GPU 状态 (实时)")
            # 改为 Textbox 以显示多行信息
            gpu_status = gr.Textbox(value="等待推理...", lines=3, interactive=False, label="GPU 信息")
            model_status_label = gr.Label(value="模型状态")
            model_status = gr.Label(value="未加载")
            time_info_label = gr.Label(value="推理耗时")
            time_info = gr.Label(value="0s")
            progress = gr.Progress()

    with gr.Row():
        output_df = gr.Dataframe(headers=["图片", "结果", "耗时 (s)"], label="推理结果", interactive=False, wrap=True)

    with gr.Row():
        export_pdf_btn = gr.DownloadButton("📄 导出 PDF（带图片）")
        export_excel_btn = gr.DownloadButton("📊 导出 Excel")
        export_md_btn = gr.DownloadButton("📝 导出 Markdown")

    # 状态存储
    state_dict = gr.State(value={"image_paths": [], "result_texts": [], "question": ""})

    # ---------- 定时器（每2秒刷新 GPU 状态） ----------
    timer = gr.Timer(1)
    timer.tick(update_gpu_info, outputs=gpu_status)

    # ---------- 事件绑定 ----------
    upload.change(update_gallery, inputs=[upload], outputs=[gallery, state_dict])

    lang_dropdown.change(
        update_language,
        inputs=[lang_dropdown],
        outputs=[upload, prompt, temperature, top_p, max_tokens, batch_size,
                 start_btn, stop_btn, gpu_status_label, model_status_label, time_info_label,
                 export_pdf_btn, export_excel_btn, export_md_btn],
    )

    start_btn.click(
        reset_stop,
        outputs=[],
    ).then(
        inference_process,
        inputs=[upload, prompt, temperature, top_p, max_tokens, batch_size],
        outputs=[output_df, model_status, time_info, state_dict],   # 移除了 gpu_status
        queue=True,
    )

    stop_btn.click(stop_inference, outputs=[model_status])

    # 导出按钮
    export_pdf_btn.click(export_pdf_with_images, inputs=[state_dict], outputs=[export_pdf_btn])
    export_excel_btn.click(export_excel_from_state, inputs=[state_dict], outputs=[export_excel_btn])
    export_md_btn.click(export_md_from_state, inputs=[state_dict], outputs=[export_md_btn])

    # 启动时加载模型
    demo.load(load_model_at_start, outputs=[model_status])

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(server_name="0.0.0.0", share=False)
