# -*- coding: utf-8 -*-
"""变化幅度阈值分割：自研 Otsu / 分位数 / 均值+2σ 三种策略。

为什么需要阈值
--------------
CVA 幅度是连续量。典型场景的幅度直方图呈"单峰主体(未变) + 低矮长尾
(变化)"形态，Otsu 通过最小化类内方差自动寻找把主体与长尾分开的切割点；
变化占比极小(<1%)时 Otsu 会失准，此时分位数(p99)或均值+2σ 更稳 ——
提供三种策略并在配置里一行切换。

Otsu 自研实现：灰度统计 + 类内方差遍历，O(n_bins)，无 OpenCV/skimage。
"""
from __future__ import annotations

import numpy as np


def otsu_threshold(values: np.ndarray, valid: np.ndarray | None = None,
                   n_bins: int = 256, value_range: tuple | None = None) -> dict:
    """对有效像元做 Otsu 双峰分割。返回 dict(threshold, separation)。"""
    v = np.asarray(values, np.float64)
    if valid is not None:
        v = v[valid]
    v = v[np.isfinite(v)]
    if v.size < 100:
        return dict(threshold=float("nan"), separation=0.0, n=int(v.size))

    lo, hi = value_range if value_range else (float(v.min()), float(v.max()))
    if hi - lo < 1e-12:
        return dict(threshold=lo, separation=0.0, n=int(v.size))

    hist, edges = np.histogram(v, bins=n_bins, range=(lo, hi))
    centers = (edges[:-1] + edges[1:]) / 2.0
    p = hist.astype(np.float64) / hist.sum()

    w0 = np.cumsum(p)                       # ≤ t 的累计概率（类0 = 未变化主体）
    mu_t = np.cumsum(p * centers)
    mu = mu_t[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        mu0 = mu_t / w0
        w1 = 1.0 - w0
        mu1 = (mu - mu_t) / w1
        sigma_b = w0 * w1 * (mu0 - mu1) ** 2        # 类间方差（σw 最小 ⇔ σb 最大）
    sigma_b[~np.isfinite(sigma_b)] = -1.0
    k = int(np.argmax(sigma_b))
    thr = float(centers[k])
    sep = float(sigma_b[k] / max(np.var(v), 1e-12))  # 类间/总体方差比，衡量双峰性
    return dict(threshold=thr, separation=round(sep, 4), n=int(v.size))


def select_threshold(mag: np.ndarray, valid: np.ndarray, cfg) -> dict:
    """按配置策略选择阈值，含下限保护。返回策略详情 + raw 掩膜统计。"""
    m = mag[valid]
    m = m[np.isfinite(m)]
    info: dict = {"method": cfg.threshold_method, "n_valid": int(m.size)}

    if cfg.threshold_method == "otsu":
        r = otsu_threshold(mag, valid)
        thr = r["threshold"]
        info.update(separation=r["separation"])
    elif cfg.threshold_method == "percentile":
        thr = float(np.percentile(m, cfg.threshold_percentile))
        info.update(percentile=cfg.threshold_percentile)
    elif cfg.threshold_method == "meanstd":
        mu, sd = float(np.mean(m)), float(np.std(m))
        thr = mu + 2.0 * sd
        info.update(mean=round(mu, 5), std=round(sd, 5))
    else:
        raise ValueError(f"未知阈值方法: {cfg.threshold_method}")

    thr_c = max(thr, float(cfg.threshold_min))      # 下限保护：反射率域噪声不该被当变化
    info["threshold_raw"] = round(float(thr), 5)
    info["threshold"] = round(float(thr_c), 5)
    info["clamped"] = bool(thr_c > thr)
    return info
