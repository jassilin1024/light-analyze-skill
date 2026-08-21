# -*- coding: utf-8 -*-
"""
skill/light_calc.py —— 光强分析与光学参数计算模块
==================================================

职责（核心算法，全部只依赖 numpy，可脱离网页单独复用）：
    1. 提取一维光强分布曲线（x / y 方向，支持 ROI 区域平均）
    2. 光斑中心坐标（光强加权质心，一阶矩）
    3. 光斑半径（RMS 半径 + 高斯 1/e^2 半径近似）
    4. 干涉 / 衍射条纹峰值检测与平均间距

配套踩坑点：
    - 踩坑点 #2  → 用 ROI 区域平均 + 旋转校正提升倾斜条纹的精度；
    - 踩坑点 #3  → 峰检测采用"先平滑、再阈值筛选极值点"三步法。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "extract_1d_profile", "compute_centroid", "compute_spot_radius",
    "smooth_profile", "find_peaks_simple", "compute_fringe_spacing",
]


# ---------------------------------------------------------------------------
# 1. 一维光强分布提取
# ---------------------------------------------------------------------------
def extract_1d_profile(gray: np.ndarray, axis: str = "x", roi=None):
    """提取一维光强分布曲线。

    参数:
        gray: 灰度图，形状 (H, W)
        axis: "x" —— 返回沿水平方向的光强分布（对每一列取行方向平均，
                    横轴为像素列坐标 x，适合竖直条纹 / 竖直光斑剖面）；
              "y" —— 返回沿垂直方向的光强分布（对每一行取列方向平均，
                    横轴为像素行坐标 y，适合水平条纹）。
        roi: 可选矩形区域 (x0, y0, x1, y1)，只在该区域内做平均。
             对倾斜、有噪的条纹图，用一个条带 ROI 平均多行 / 多列，
             能显著提高信噪比与峰值位置精度（踩坑点 #2 的解决方案）。

    返回:
        (positions, profile)
        positions: 位置数组（像素坐标，float64）
        profile  : 对应位置的光强数组（灰度均值，float64）
    """
    img = gray.astype(np.float64)
    if roi is not None:
        x0, y0, x1, y1 = _normalize_roi(roi, gray.shape)
        img = img[y0:y1, x0:x1]

    if axis == "x":
        profile = img.mean(axis=0)            # 每一列：行方向平均
        positions = np.arange(img.shape[1])
    else:
        profile = img.mean(axis=1)            # 每一行：列方向平均
        positions = np.arange(img.shape[0])
    return positions, profile


def _normalize_roi(roi, shape):
    """把 ROI 规整到图像范围内并做边界保护，防止越界 / 空区域报错。"""
    h, w = shape[:2]
    x0, y0, x1, y1 = (int(v) for v in roi)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("ROI 范围无效：需要 x0 < x1 且 y0 < y1，且不超出图像")
    return x0, y0, x1, y1


# ---------------------------------------------------------------------------
# 2. 光斑中心坐标（光强加权质心）
# ---------------------------------------------------------------------------
def compute_centroid(gray: np.ndarray):
    """光强加权质心，即光斑中心坐标 (cx, cy)，单位像素。

    以灰度（光强）作为权重对整幅图求一阶矩：
        cx = Σ(I · x) / ΣI ,   cy = Σ(I · y) / ΣI
    相比直接找"最亮像素"，加权质心对噪声更稳健，
    也更符合光斑能量中心的物理含义。
    """
    img = gray.astype(np.float64)
    total = img.sum()
    if total <= 0:
        return (0.0, 0.0)
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cx = float((img * xx).sum() / total)
    cy = float((img * yy).sum() / total)
    return (cx, cy)


# ---------------------------------------------------------------------------
# 3. 光斑半径（RMS 半径 + 1/e^2 半径）
# ---------------------------------------------------------------------------
def compute_spot_radius(gray: np.ndarray, center=None):
    """计算光斑半径（单位：像素）。

    以二维高斯光斑 I(r) = A·exp(-r^2 / 2σ^2) 为参考模型：
        1. 先求光强加权二阶矩   E[r^2] = Σ(I·r^2) / ΣI
        2. 对高斯光斑有 E[r^2] = 2σ^2，因此 σ = sqrt(E[r^2] / 2)

    返回:
        rms_radius : RMS 半径，即光斑的等效标准差 σ（像素），
                     对高斯光斑严格等于其宽度参数；
        radius_1e2 : 1/e^2 半径 = 2σ。即光强降到峰值 e^-2 ≈ 13.5%
                     处的半径，是激光光斑最常用的工程定义。

    说明:
        用二阶矩求半径无需曲线拟合，简单稳定；
        受残留噪声 / 背景扣除不彻底影响，结果为近似值（见 README 踩坑点 #1）。
    """
    img = gray.astype(np.float64)
    total = img.sum()
    if total <= 0:
        return (0.0, 0.0)
    if center is None:
        cx, cy = compute_centroid(gray)
    else:
        cx, cy = center
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    moment2 = float((img * r2).sum() / total)   # 二阶矩 E[r^2]
    sigma = np.sqrt(max(moment2 / 2.0, 0.0))    # 高斯光斑 σ = sqrt(E[r^2]/2)
    return (float(sigma), float(2.0 * sigma))


# ---------------------------------------------------------------------------
# 4. 条纹峰值检测与平均间距（踩坑点 #3 的核心实现）
# ---------------------------------------------------------------------------
def smooth_profile(profile, window: int = 5):
    """滑动平均平滑一维曲线，用于抑制毛刺噪声（踩坑点 #3 第一步）。"""
    window = int(window)
    arr = np.asarray(profile, dtype=np.float64)
    if window < 1:
        return arr.copy()
    if window % 2 == 0:          # 强制奇数窗口，保证对称
        window += 1
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def find_peaks_simple(profile, min_distance: int = 5, height_ratio: float = 0.3,
                      smooth_window: int = 5):
    """检测条纹峰值位置（踩坑点 #3 的解决方案）。

    三步走：
        1. 先滑动平均平滑，消除单像素毛刺产生的假极值；
        2. 找局部极大值点（比左右邻居都大）；
        3. 双重筛选剔除伪峰：
           a) 高度阈值：只保留高度 >= height_ratio × 最高峰 的点；
           b) 最小间距：相邻峰距离必须 >= min_distance（贪心保留更强的峰）。

    返回:
        按坐标升序排列的峰位置列表（像素坐标，int）
    """
    arr = smooth_profile(profile, smooth_window)
    n = len(arr)
    if n < 3:
        return []

    # 1) 局部极大值
    peaks = [i for i in range(1, n - 1)
             if arr[i] > arr[i - 1] and arr[i] >= arr[i + 1]]
    if not peaks:
        return []

    # 2) 高度阈值：相对最高峰
    thr = float(height_ratio) * float(arr.max())
    peaks = [p for p in peaks if arr[p] >= thr]

    # 3) 最小间距去重：从强到弱贪心，保留更明显的峰
    peaks.sort(key=lambda p: arr[p], reverse=True)
    selected = []
    for p in peaks:
        if all(abs(p - q) >= min_distance for q in selected):
            selected.append(p)
    selected.sort()
    return selected


def compute_fringe_spacing(gray: np.ndarray, axis: str = "x", roi=None,
                           min_distance: int = 5, height_ratio: float = 0.3):
    """计算干涉条纹平均间距（单位：像素）。

    返回:
        (spacing, peaks, positions, profile)
        spacing : 相邻峰间距的平均值；若有效峰少于 2 个则返回 None
        peaks   : 检测到的峰位置列表
        positions / profile：一维光强曲线，方便直接绘图
    """
    positions, profile = extract_1d_profile(gray, axis=axis, roi=roi)
    peaks = find_peaks_simple(profile, min_distance=min_distance,
                              height_ratio=height_ratio)
    if len(peaks) >= 2:
        diffs = np.diff(np.asarray(peaks, dtype=np.float64))
        spacing = float(np.mean(diffs))
    else:
        spacing = None
    return spacing, peaks, positions, profile

