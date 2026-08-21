# -*- coding: utf-8 -*-
"""
skill/image_io.py —— 图像读取与预处理模块
==========================================

职责（只做"图像输入 + 基础预处理"）：
    1. 读取图片（兼容中文路径 / 上传的字节流）
    2. 彩色图转灰度图
    3. 高斯降噪，抑制传感器 / 环境噪声
    4. 边缘背景采样，扣除环境杂光引起的背景基线（踩坑点 #1）

设计约束：
    - 只依赖 numpy + OpenCV，不依赖 streamlit / matplotlib；
    - 所有函数都是纯函数式输入输出，任何项目都可以
      `from skill import image_io` 直接复用。
"""

from __future__ import annotations

import os

import cv2
import numpy as np

__all__ = [
    "imread_unicode", "imread_bytes", "load_image",
    "to_gray", "gaussian_denoise", "estimate_background",
    "subtract_background", "rotate_image",
]


# ---------------------------------------------------------------------------
# 1. 图片读取
# ---------------------------------------------------------------------------
def imread_unicode(path: str) -> np.ndarray:
    """读取图片，兼容中文 / 空格路径。

    踩坑点补充：cv2.imread 遇到中文路径会静默返回 None，
    这里先用 numpy 把文件读成字节流，再交给 cv2.imdecode 解码，
    从而彻底绕开编码问题。
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise FileNotFoundError(f"无法读取图片文件：{path}")
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)  # 统一以 BGR 彩色读入
    if img is None:
        raise ValueError(f"图片解码失败，请确认文件是有效图片：{path}")
    return img


def imread_bytes(buf: bytes) -> np.ndarray:
    """从字节流读取图片（用于 streamlit 上传的 UploadedFile）。"""
    data = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("图片解码失败，请确认文件是有效的 png/jpg/bmp 等图片")
    return img


def load_image(source) -> np.ndarray:
    """统一读取入口：传路径（str / pathlib.Path）或字节流（bytes）均可。"""
    if isinstance(source, (str, os.PathLike)):
        return imread_unicode(str(source))
    if isinstance(source, bytes):
        return imread_bytes(source)
    raise TypeError("source 必须是图片路径（str/Path）或图片字节流（bytes）")


# ---------------------------------------------------------------------------
# 2. 灰度化
# ---------------------------------------------------------------------------
def to_gray(img: np.ndarray) -> np.ndarray:
    """彩色图转灰度图；若输入本身就是灰度图则返回副本。"""
    if img is None:
        raise ValueError("输入图像为空")
    if len(img.shape) == 2:                 # 已经是单通道灰度图
        return img.copy()
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------------------------
# 3. 高斯降噪
# ---------------------------------------------------------------------------
def gaussian_denoise(gray: np.ndarray, ksize: int = 5, sigma: float = 0.0) -> np.ndarray:
    """高斯降噪。

    参数:
        ksize: 高斯核大小，必须是正奇数（3、5、7...），越大越平滑
        sigma: 高斯标准差；0 表示由 OpenCV 根据核大小自动计算
    """
    if ksize % 2 == 0 or ksize < 1:
        raise ValueError("ksize 必须是正奇数，例如 3、5、7")
    return cv2.GaussianBlur(gray, (int(ksize), int(ksize)), sigma)


# ---------------------------------------------------------------------------
# 4. 背景杂光扣除（踩坑点 #1 的解决方案）
# ---------------------------------------------------------------------------
def estimate_background(gray: np.ndarray, margin: int = 20) -> float:
    """边缘背景采样，估计环境杂光引起的背景灰度值。

    原理：
        实验照片里光斑 / 条纹通常位于画面中央，而上下左右四周边框
        区域基本只有环境光、没有信号。取四条边框带内的像素做统计，
        再剔除 5%~95% 之外的离群值（防止边缘意外出现亮点干扰），
        用剩余像素的均值代表背景。

    返回:
        背景灰度值（float，范围 0~255）
    """
    h, w = gray.shape[:2]
    margin = int(max(1, min(margin, min(h, w) // 4)))

    border = np.concatenate([
        gray[:margin, :].ravel(),    # 上边框
        gray[-margin:, :].ravel(),   # 下边框
        gray[:, :margin].ravel(),    # 左边框
        gray[:, -margin:].ravel(),   # 右边框
    ])
    if border.size == 0:
        return 0.0

    # 剔除离群值，防止边缘恰好拍到亮斑 / 脏点
    low, high = np.percentile(border, [5, 95])
    samples = border[(border >= low) & (border <= high)]
    return float(samples.mean()) if samples.size else 0.0


def subtract_background(gray: np.ndarray, margin: int = 20) -> np.ndarray:
    """扣除背景基线：灰度值 - 背景值，负值截断为 0，返回 uint8 图。"""
    bg = estimate_background(gray, margin)
    out = gray.astype(np.float32) - bg
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 5. 旋转校正（配合踩坑点 #2：条纹拍倾斜时的辅助手段）
# ---------------------------------------------------------------------------
def rotate_image(gray: np.ndarray, angle: float = 0.0) -> np.ndarray:
    """按角度旋转图像，用于校正拍摄倾斜的条纹。

    角度为正表示逆时针旋转；旋转后超出画布的区域填黑（0）。
    """
    angle = float(angle)
    if abs(angle) < 1e-6:
        return gray
    h, w = gray.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        gray, matrix, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
