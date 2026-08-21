# -*- coding: utf-8 -*-
"""
Light-Analyze 核心 Skill 包
==========================

纯算法模块，不依赖 streamlit，可脱离网页单独调用、单独复用。

用法示例（在项目根目录下）:
    from skill import image_io, light_calc, export_result

    gray = image_io.to_gray(image_io.load_image("test_img/gaussian_spot.png"))
    proc = image_io.subtract_background(image_io.gaussian_denoise(gray, 5))
    cx, cy = light_calc.compute_centroid(proc)
    print("光斑中心:", cx, cy)

版本:
    1.0.0
"""

__version__ = "1.0.0"

__all__ = ["image_io", "light_calc", "export_result"]
