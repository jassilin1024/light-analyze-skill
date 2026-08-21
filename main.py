# -*- coding: utf-8 -*-
"""
Light-Analyze 光斑图像光强分析工具 —— Streamlit 网页入口（UI 层）
=================================================================

运行方式（在项目根目录下）：
    1) pip install opencv-python numpy matplotlib streamlit
    2) streamlit run main.py
    3) 浏览器自动打开 http://localhost:8501

分层设计说明（踩坑点 #4）：
    本文件只负责"网页交互 + 参数收集 + 结果展示"，
    所有算法逻辑都封装在 skill/ 中，可脱离网页单独复用。

支持功能（只做 4 项，拒绝堆砌）：
    1. 上传光斑 / 单缝衍射 / 双缝干涉图片，完成灰度化、高斯降噪、背景杂光扣除
    2. 提取一维光强分布曲线并绘图展示
    3. 自动计算光斑中心坐标、光斑半径、干涉条纹平均间距
    4. 导出处理后的光强数据 CSV 与对比效果图
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import streamlit as st

from skill import image_io, light_calc, export_result

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_IMG_DIR = os.path.join(BASE_DIR, "test_img")
DEMO_DIR = os.path.join(BASE_DIR, "demo")


# ===========================================================================
# 页面配置与标题
# ===========================================================================
st.set_page_config(
    page_title="Light-Analyze 光斑光强分析",
    page_icon="🔦",
    layout="wide",
)

st.title("🔦 Light-Analyze 光斑图像光强分析工具")
st.caption(
    "光电信息科学与工程 · 本科实验数据处理 · 完全本地运行，图片与数据不上传任何服务器"
)


# ===========================================================================
# 侧边栏：参数设置
# ===========================================================================
with st.sidebar:
    st.header("⚙️ 参数设置")

    denoise_ksize = st.slider(
        "高斯降噪核大小（奇数）", 3, 15, 5, step=2,
        help="越大越平滑；噪点严重的照片可以调大到 7 或 9",
    )
    bg_margin = st.slider(
        "背景采样边缘宽度（像素）", 5, 60, 20,
        help="用画面四周边框估计环境杂光背景（对应踩坑点 #1）",
    )
    rotate_angle = st.number_input(
        "旋转校正角度（度，正=逆时针）", -45.0, 45.0, 0.0, step=1.0,
        help="条纹拍倾斜时先用它校正（对应踩坑点 #2）",
    )

    axis_label = st.radio(
        "光强提取方向", ["x（水平方向分布）", "y（垂直方向分布）"], index=0,
    )
    profile_axis = "x" if axis_label.startswith("x") else "y"

    st.subheader("📐 ROI 区域（可选，提升精度）")
    use_roi = st.checkbox(
        "启用 ROI 区域平均", value=False,
        help="在条纹上框一个矩形条带做平均，抗噪、抗倾斜（对应踩坑点 #2）",
    )
    roi_text = st.text_input(
        "ROI 格式：x0,y0,x1,y1（留空 = 整图）", "",
        placeholder="例：0,120,640,260",
    )
    st.caption("例：0,120,640,260 表示 x∈[0,640)、y∈[120,260) 的横条区域")

    st.subheader("🔬 条纹检测参数")
    min_dist = st.number_input("相邻峰最小间距（像素）", 1, 100, 5,
                               help="过滤间距过近的伪峰")
    height_ratio = st.slider(
        "峰高阈值（相对最高峰）", 0.05, 0.90, 0.30, 0.05,
        help="只保留高于该比例的点，过滤低矮毛刺（对应踩坑点 #3）",
    )

    st.divider()
    st.subheader("🧪 测试样图快速体验")
    test_files = []
    if os.path.isdir(TEST_IMG_DIR):
        test_files = sorted(
            f for f in os.listdir(TEST_IMG_DIR)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        )
    use_test = st.selectbox("选择 test_img 中的样图", [""] + test_files)


# ===========================================================================
# 工具函数
# ===========================================================================
def parse_roi(text: str):
    """解析 ROI 文本 "x0,y0,x1,y1" 为元组；留空返回 None。"""
    text = (text or "").strip()
    if not text:
        return None
    parts = [p.strip() for p in text.replace("，", ",").split(",")]
    if len(parts) != 4:
        raise ValueError("ROI 需要 4 个数字：x0,y0,x1,y1")
    x0, y0, x1, y1 = (int(p) for p in parts)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("ROI 需要满足 x0 < x1 且 y0 < y1")
    return (x0, y0, x1, y1)


def analyze(gray: np.ndarray, params: dict) -> dict:
    """核心流水线（UI 层只做编排，算法全部在 skill/ 中）。

    流程：旋转校正 -> 高斯降噪 -> 背景扣除 -> 光强提取 -> 参数计算 -> 绘图
    """
    # 1) 旋转校正（倾斜条纹，踩坑点 #2）
    gray_work = image_io.rotate_image(gray, params["rotate_angle"])
    # 2) 高斯降噪
    denoised = image_io.gaussian_denoise(gray_work, ksize=params["denoise_ksize"])
    # 3) 背景杂光扣除（踩坑点 #1）
    processed = image_io.subtract_background(denoised, margin=params["bg_margin"])

    # 4) 一维光强分布
    positions, profile = light_calc.extract_1d_profile(
        processed, axis=params["axis"], roi=params["roi"],
    )

    # 5) 光斑中心与半径
    center = light_calc.compute_centroid(processed)
    rms_r, radius_1e2 = light_calc.compute_spot_radius(processed, center=center)

    # 6) 条纹间距
    spacing, peaks, _, _ = light_calc.compute_fringe_spacing(
        processed, axis=params["axis"], roi=params["roi"],
        min_distance=params["min_dist"], height_ratio=params["height_ratio"],
    )

    # 7) 对比效果图
    figure = export_result.plot_comparison(
        gray_work, processed, positions, profile, peaks=peaks,
        title="预处理对比（高斯降噪核 {} · 背景扣除 {}px）".format(
            params["denoise_ksize"], params["bg_margin"]),
    )

    return {
        "gray_work": gray_work,
        "processed": processed,
        "positions": positions,
        "profile": profile,
        "center": center,
        "rms_radius": rms_r,
        "radius_1e2": radius_1e2,
        "spacing": spacing,
        "peaks": peaks,
        "figure": figure,
    }


# ===========================================================================
# 图片输入：上传 或 测试样图
# ===========================================================================
uploaded = st.file_uploader(
    "📤 上传光斑 / 单缝衍射 / 双缝干涉图片",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
)

source, src_name = None, None
if uploaded is not None:
    source = uploaded.getvalue()      # 字节流
    src_name = uploaded.name
elif use_test:
    with open(os.path.join(TEST_IMG_DIR, use_test), "rb") as f:
        source = f.read()
    src_name = use_test

if source is None:
    st.info("👆 请上传一张实验照片，或从左侧选择一个测试样图开始体验。")
    st.stop()

try:
    img = image_io.load_image(source)
    gray = image_io.to_gray(img)
    roi = parse_roi(roi_text) if use_roi else None
    params = {
        "denoise_ksize": int(denoise_ksize),
        "bg_margin": int(bg_margin),
        "rotate_angle": float(rotate_angle),
        "axis": profile_axis,
        "roi": roi,
        "min_dist": int(min_dist),
        "height_ratio": float(height_ratio),
    }
    result = analyze(gray, params)
except Exception as exc:              # 参数或图片异常时友好提示
    st.error(f"❌ 处理出错：{exc}")
    st.stop()

# 导出内容先生成（下载按钮需要字节流）
stem = os.path.splitext(src_name)[0]
csv_bytes = export_result.build_csv_bytes(
    result["positions"], result["profile"],
    peaks=result["peaks"], spacing=result["spacing"],
)
png_bytes = export_result.fig_to_png_bytes(result["figure"])


# ===========================================================================
# 结果展示
# ===========================================================================
st.subheader("📋 图片与预处理结果")
st.pyplot(result["figure"])
plt.close(result["figure"])           # 及时释放 Figure，避免内存堆积

# ---- 光学参数 ----
st.subheader("📐 光学参数计算结果")
mode_info = (
    f"检测到 **{len(result['peaks'])}** 个条纹峰 → 判定为条纹图"
    if result["spacing"] is not None
    else "未检测到明显条纹 → 按光斑图输出中心与半径"
)
st.info(mode_info)

c1, c2, c3, c4 = st.columns(4)
c1.metric("光斑中心 X", f"{result['center'][0]:.1f} px")
c2.metric("光斑中心 Y", f"{result['center'][1]:.1f} px")
c3.metric("RMS 半径", f"{result['rms_radius']:.2f} px")
c4.metric("1/e² 半径", f"{result['radius_1e2']:.2f} px")

if result["spacing"] is not None:
    c1.metric("条纹平均间距", f"{result['spacing']:.2f} px")
    c2.metric("峰值数量", f"{len(result['peaks'])}")
else:
    c1.metric(
        "条纹平均间距", "未检测到",
        help="条纹峰少于 2 个：可调低峰高阈值、调小最小间距，或先用旋转校正",
    )

# ---- 导出 ----
st.subheader("💾 导出结果")
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.download_button(
        "⬇️ 下载光强数据 CSV",
        data=csv_bytes,
        file_name=f"{stem}_intensity.csv",
        mime="text/csv",
    )
with col_b:
    st.download_button(
        "⬇️ 下载对比效果图 PNG",
        data=png_bytes,
        file_name=f"{stem}_result.png",
        mime="image/png",
    )
with col_c:
    if st.button("保存到项目 demo/ 目录"):
        try:
            csv_path = export_result.export_csv(
                os.path.join(DEMO_DIR, f"{stem}_intensity.csv"),
                result["positions"], result["profile"],
                peaks=result["peaks"], spacing=result["spacing"],
            )
            png_path = os.path.join(DEMO_DIR, f"{stem}_result.png")
            result["figure"].savefig(png_path, dpi=150, bbox_inches="tight")
            st.success(f"已保存：\n{csv_path}\n{png_path}")
        except Exception as exc:
            st.error(f"保存失败：{exc}")

# ---- 原理说明 ----
with st.expander("📖 算法与踩坑说明（点击展开）"):
    st.markdown(
        """
#### 处理流水线
1. **灰度化**：彩色图转灰度，光强 = 灰度值；
2. **高斯降噪**：抑制传感器 / 环境随机噪声；
3. **背景扣除**：取画面四周边框采样均值作为背景基线并减去（踩坑点 #1）；
4. **一维光强提取**：沿 x 或 y 方向做 ROI 区域平均（踩坑点 #2）；
5. **参数计算**：光强加权质心 → 光斑中心；二阶矩 → RMS 半径与 1/e² 半径；
   峰值检测（先平滑、再阈值筛选）→ 条纹平均间距（踩坑点 #3）。

#### 分层架构
- **UI 层**（本文件 main.py）：上传、参数、展示、导出；
- **Skill 层**（skill/ 目录）：image_io / light_calc / export_result，
  纯算法、零 streamlit 依赖，可脱离网页单独调用（踩坑点 #4）。

> 本项目不涉及硬件驱动、不调用任何 AI 识别模型，全部为本科
> 光学实验常用的经典图像处理算法，难度适中、完全自主可复现。
        """
    )

