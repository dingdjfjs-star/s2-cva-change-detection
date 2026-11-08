# -*- coding: utf-8 -*-
"""
S2-CVA Change Detection Pipeline
================================
基于 Sentinel-2 时序影像的土地覆被变化检测流水线。

模块结构（自底向上）：
    config            —— 全局配置（dataclass + JSON 读写）
    io_utils          —— 清单(manifest)读取 / 时序栈加载 / GeoTIFF·GeoJSON 写出
    cloud_mask        —— 去云：SCL 掩膜(含膨胀) + 4 波段规则云分数兜底
    compositing       —— 年度最优值合成：逐像元 medoid（xarray 向量化实现）
    radiometric       —— 相对辐射归一化：伪不变特征(PIF)迭代回归
    cva               —— CVA 变化向量分析核心（纯 numpy 自研实现）
    threshold         —— 变化幅度阈值分割（自研 Otsu / 分位数 / 均值+2σ）
    pseudo_filter     —— 伪变化过滤：自研连通域标记(Union-Find)、面积阈值、
                         空间一致性、纹理(Sobel/局部标准差)约束、二值形态学
    classify          —— 基于随机森林(sklearn)的土地覆被分类（供转移矩阵使用）
    patch_extract     —— 变化图斑矢量化(rasterio.features) + 属性挂接 → GeoJSON
    transition_matrix —— 土地利用转移矩阵 + 损失/增益/净变化/交换变化分解
    report            —— Markdown 统计报表 + CSV 明细
    viz               —— 示例结果图（概览图 / 变化分布图 / 转移矩阵热力图 等）
    pipeline          —— 流水线编排入口 run(cfg)
"""

from .config import Config
from .pipeline import run

__all__ = ["Config", "run"]
__version__ = "1.0.0"
