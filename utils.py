# utils.py
import os
import cv2
from typing import List

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
ALLOWED_EXTENSIONS = list(IMAGE_EXTENSIONS | VIDEO_EXTENSIONS)

def is_image_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in IMAGE_EXTENSIONS

def is_video_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in VIDEO_EXTENSIONS

def get_image_files(file_paths: List[str]) -> List[str]:
    return [f for f in file_paths if is_image_file(f)]

def get_images_from_directory(directory: str) -> List[str]:
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if is_image_file(file):
                image_paths.append(os.path.join(root, file))
    return image_paths

def extract_frames(video_path: str, interval: int = 60) -> List[str]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    # 固定保存路径：项目根目录下的 outputs/frames/
    base_dir = os.path.join(os.path.dirname(__file__), "outputs", "frames")
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frames_dir = os.path.join(base_dir, video_name)
    os.makedirs(frames_dir, exist_ok=True)

    frame_count = 0
    saved_count = 0
    frame_paths = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            frame_filename = f"frame_{saved_count:04d}.png"
            frame_path = os.path.join(frames_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            saved_count += 1
        frame_count += 1

    cap.release()
    print(f"[INFO] 已抽取 {len(frame_paths)} 帧，保存至 {frames_dir}")
    return frame_paths
# def extract_frames(video_path: str, interval: int = 60) -> List[str]:
#     """
#     从视频中按固定帧间隔抽取帧，保存到视频所在目录的 frames/ 子目录下。
#     返回帧文件路径列表。
#     """
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise ValueError(f"无法打开视频文件: {video_path}")
#
#     video_dir = os.path.dirname(video_path)
#     video_name = os.path.splitext(os.path.basename(video_path))[0]
#     frames_dir = os.path.join(video_dir, "frames")
#     os.makedirs(frames_dir, exist_ok=True)
#
#     frame_count = 0
#     saved_count = 0
#     frame_paths = []
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         if frame_count % interval == 0:
#             frame_filename = f"{video_name}_frame_{saved_count:04d}.png"
#             frame_path = os.path.join(frames_dir, frame_filename)
#             cv2.imwrite(frame_path, frame)
#             frame_paths.append(frame_path)
#             saved_count += 1
#         frame_count += 1
#
#     cap.release()
#     return frame_paths
