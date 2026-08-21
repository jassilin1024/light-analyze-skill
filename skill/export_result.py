# -*- coding: utf-8 -*-
"""
skill/export_result.py —— 绘图与导出模块
==========================================

职责：
    1. 绘制"原始图 / 预处理后图 / 一维光强曲线"三联对比图
    2. 把一维光强数据导出为 CSV（含峰值位置与平均间距信息）
    3. 把对比图保存为 PNG，或转成字节流供 streamlit 下载按钮使用

设计约束：
    - 只依赖 numpy + matplotlib，不依赖 streamlit；
    - 使用 Agg 无界面后端，服务器 / 脚本环境下也能正常出图。
"""

from __future__ import annotations

import csv
import io
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")   # 无界面后端：不弹窗、不出错，适合脚本和网页
import matplotlib.pyplot as plt
from matplotlib import font_manager

__all__ = [
    "setup_chinese_font", "plot_comparison", "fig_to_png_bytes",
    "build_csv_bytes", "export_csv",
]

# 常见中文字体候选（Windows / macOS / Linux），按优先级排列
_CHINESE_FONTS = [
    "Microsoft YaHei", "SimHei", "PingFang SC",
    "Noto Sans CJK SC", "WenQuanYi Micro Hei",
]


# ---------------------------------------------------------------------------
# 中文字体配置
# ---------------------------------------------------------------------------
def setup_chinese_font():
    """尽量配置中文字体，避免绘图中文变成方块；找不到则退回默认字体。

    注意：图内文字采用"中文 English"双语标签，即使没有中文字体，
    英文部分也保证可读，不会影响评审展示。
    """
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in _CHINESE_FONTS:
        if name in available:
            current = plt.rcParams.get("font.sans-serif", [])
            plt.rcParams["font.sans-serif"] = [name] + list(current)
            break
    plt.rcParams["axes.unicode_minus"] = False   # 正常显示负号


# ---------------------------------------------------------------------------
# 对比效果图
# ---------------------------------------------------------------------------
def plot_comparison(original, processed, positions, profile,
                    peaks=None, title: str = "", save_path=None, dpi: int = 150):
    """绘制三联对比图：原始图 | 预处理后图 | 一维光强分布。

    参数:
        original : 原始灰度图 (H, W)
        processed: 预处理后灰度图 (H, W)
        positions: 位置数组
        profile  : 一维光强数组
        peaks    : 可选，检测到的峰位置列表，会在曲线上画红色虚线标注
        title    : 图标题
        save_path: 若给出则把图保存为 PNG
    返回:
        matplotlib Figure 对象
    """
    setup_chinese_font()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    if title:
        fig.suptitle(title, fontsize=13)

    # 左：原始图
    axes[0].imshow(original, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("原始图 Original")
    axes[0].axis("off")

    # 中：预处理后
    axes[1].imshow(processed, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("预处理后 Processed")
    axes[1].axis("off")

    # 右：一维光强分布
    axes[2].plot(positions, profile, lw=1.6, color="#1f77b4")
    axes[2].set_xlabel("位置 Position (px)")
    axes[2].set_ylabel("光强 Intensity (灰度值)")
    axes[2].set_title("一维光强分布 1-D Intensity")
    axes[2].grid(alpha=0.3)
    if peaks:
        for p in peaks:
            axes[2].axvline(p, color="red", ls="--", lw=0.8, alpha=0.7)

    fig.tight_layout()
    if save_path:
        save_path = str(save_path)
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def fig_to_png_bytes(fig, dpi: int = 150) -> bytes:
    """把 matplotlib Figure 渲染成 PNG 字节流（供 streamlit 下载按钮）。"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# CSV 导出
# ---------------------------------------------------------------------------
def build_csv_bytes(positions, profile, peaks=None, spacing=None) -> bytes:
    """把一维光强数据构造成 CSV 字节流（utf-8-sig，Excel 打开不乱码）。

    内容结构：
        第 1 行        ：表头"位置, 光强"
        中间若干行      ：位置 -> 光强 数据对
        若检测到峰      ：追加一空行 + "峰值位置" 行
        若算得平均间距  ：追加"平均条纹间距" 行
    """
    positions = np.asarray(positions, dtype=np.float64)
    profile = np.asarray(profile, dtype=np.float64)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["位置(pixel)", "光强(灰度值)"])
    for x, y in zip(positions, profile):
        writer.writerow([f"{x:.3f}", f"{y:.3f}"])
    if peaks:
        writer.writerow([])
        writer.writerow(["峰值位置(pixel)"] + [f"{p:.1f}" for p in peaks])
    if spacing is not None:
        writer.writerow(["平均条纹间距(pixel)", f"{spacing:.3f}"])
    return buf.getvalue().encode("utf-8-sig")


def export_csv(csv_path, positions, profile, peaks=None, spacing=None) -> str:
    """把一维光强数据保存为 CSV 文件，返回文件路径。"""
    csv_path = str(csv_path)
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, "wb") as f:
        f.write(build_csv_bytes(positions, profile, peaks, spacing))
    return csv_path
