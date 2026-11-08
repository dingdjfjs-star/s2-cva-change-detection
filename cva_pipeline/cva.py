# -*- coding: utf-8 -*-
"""CVA（Change Vector Analysis，变化向量分析）核心 —— 自研实现，纯 numpy。

原理
----
对两期同波段数的多光谱合成影像 X1(基期)、X2(目标期)，逐像元构造变化向量：

    ΔX = X2 - X1 = (Δb1, Δb2, ..., ΔbC)      ∈ R^C

- 模长  |ΔX| = sqrt( Σ_c w_c·Δb_c² )  → 变化强度（是否变了、变了多剧烈）
- 方向  ΔX/|ΔX|                        → 变化性质（怎么变的：变亮/变绿/变干…）

模长对"任何方向的光谱变化"都敏感，这是 CVA 相比单波段差值/指数差值的
根本优势：它不预设变化的类型，把"检测"与"定性"解耦 —— 模长做二值检测，
方向再做变化类型归属（本项目由随机森林分类图给出 from→to，CVA 方向用于
复核与展示各转移类型的光谱变化特征）。

实现要点
--------
- 支持波段权重 w_c（如对噪声更敏感的蓝波段降权、对植被变化关键的 NIR 加权）；
- NaN 语义贯穿：任一参与波段缺失 → 模长置 NaN（下游统一处理）；
- direction_cosines 输出各波段方向余弦，供"各转移类型平均变化向量"分析。
"""
from __future__ import annotations

import numpy as np


def change_vector_magnitude(x1: np.ndarray, x2: np.ndarray,
                            weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """计算 CVA 变化幅度与变化向量（差值）。

    参数
    ----
    x1, x2 : (C,H,W) 两期年度合成（float，同辐射域）
    weights: (C,) 波段权重，None = 等权

    返回
    ----
    magnitude : (H,W) 变化幅度，NaN 表示无法计算（nodata）
    delta     : (C,H,W) 变化向量 ΔX = X2 - X1
    """
    x1 = np.asarray(x1, np.float64)
    x2 = np.asarray(x2, np.float64)
    if x1.shape != x2.shape or x1.ndim != 3:
        raise ValueError(f"形状不一致: {x1.shape} vs {x2.shape}")
    delta = x2 - x1                                  # (C,H,W)
    w = np.ones(delta.shape[0], np.float64) if weights is None else np.asarray(weights, np.float64)
    sq = (delta ** 2) * w[:, None, None]
    mag = np.sqrt(sq.sum(axis=0))                    # NaN 会自然传播为 NaN
    return mag.astype(np.float32), delta.astype(np.float32)


def direction_cosines(delta: np.ndarray, mag: np.ndarray) -> np.ndarray:
    """变化方向余弦 Δc/|ΔX|（|Δ|≈0 处置 0，避免除零）。返回 (C,H,W)。"""
    mag_safe = np.where(mag > 1e-9, mag, np.nan)
    cos = np.asarray(delta, np.float64) / mag_safe[None]
    return np.nan_to_num(cos, nan=0.0).astype(np.float32)


def summarize_directions(delta: np.ndarray, labels: np.ndarray, n_classes: int) -> dict:
    """按图斑标签汇总平均变化向量（各变化图斑的"方向指纹"）。

    返回 {label_id: (C,) 平均 ΔX}，供报表/图件展示不同转移类型的
    光谱变化方向差异（如 林地→耕地 呈 NIR 大幅下降 + 红光上升）。
    """
    flat = labels.ravel()
    out = {}
    for lid in np.unique(flat):
        if lid <= 0:
            continue
        m = flat == lid
        out[int(lid)] = np.stack([d.ravel()[m] for d in delta]).mean(axis=1)
    return out
