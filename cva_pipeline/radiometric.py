# -*- coding: utf-8 -*-
"""相对辐射归一化：伪不变特征(PIF) 迭代回归，把 year2 合成拉到 year1 辐射域。

动机
----
两期影像的传感器状态、大气状况、太阳角度不可能完全一致，会带来全局性的
辐射差异（整体偏亮/偏暗、波段间比例漂移）。这类差异不是地表变化，却会让
CVA 幅度普遍抬升、淹没真实变化 —— 归一化是把"变化检测"从"差异检测"里
救出来的关键一步。

方法（Schott et al. 1988 PIF 思想的稳健化）
----
逐波段拟合线性关系  y1 ≈ a·y2 + b  ，然后把 y2' = a·y2 + b 作为归一化输出：
1. 初始用全部有效像元拟合（最小二乘）；
2. 计算残差，仅保留 |残差| ≤ σ·std 的像元作为伪不变特征 PIF ——
   直觉是"两年都没变的地物"才符合一致的辐射关系，变化像元是离群点；
3. 用 PIF 重新拟合，重复 n_iter 轮逐步收紧。

自研实现（闭式最小二乘 + numpy 掩膜迭代），不依赖任何回归库。
"""
from __future__ import annotations

import numpy as np


def _robust_linear_fit(x: np.ndarray, y: np.ndarray, n_iter: int, sigma: float) -> tuple[float, float, np.ndarray]:
    """迭代稳健一元线性回归：y ≈ a·x + b。返回 (a, b, 最终 PIF 掩膜)。"""
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 100:
        return 1.0, 0.0, keep  # 样本过少：不做归一化，退化为恒等映射
    a, b = 1.0, 0.0
    for _ in range(max(1, n_iter)):
        xs, ys = x[keep], y[keep]
        # 闭式最小二乘（带方差护栏）
        mx, my = xs.mean(), ys.mean()
        vx = ((xs - mx) ** 2).sum()
        a = 1.0 if vx < 1e-12 else float(((xs - mx) * (ys - my)).sum() / vx)
        b = float(my - a * mx)
        resid = np.abs(y - (a * x + b))
        std = resid[keep].std()
        new_keep = keep & (resid <= sigma * max(std, 1e-6))
        if new_keep.sum() < 100 or bool((new_keep == keep).all()):
            keep = new_keep if new_keep.sum() >= 100 else keep
            break
        keep = new_keep
    return a, b, keep


def relative_normalization(ref: np.ndarray, sub: np.ndarray, valid: np.ndarray,
                           n_iter: int = 3, sigma: float = 2.0) -> tuple[np.ndarray, list[dict]]:
    """将 sub(year2) 逐波段归一化到 ref(year1) 的辐射域。

    参数
    ----
    ref, sub : (C,H,W) float 两期年度合成
    valid : (H,W) bool 两期共同有效区

    返回
    ----
    sub_norm : (C,H,W)
    report : 每波段 dict(band, a, b, r2, pif_ratio)
    """
    ref = ref.astype(np.float64)
    sub = sub.astype(np.float64)
    out = sub.copy()
    report = []
    for c in range(ref.shape[0]):
        x = sub[c][valid].ravel()
        y = ref[c][valid].ravel()
        a, b, keep = _robust_linear_fit(x, y, n_iter, sigma)
        out[c] = (a * sub[c] + b)
        # 决定系数（在 PIF 上评估）
        xs, ys = x[keep], y[keep]
        pred = a * xs + b
        ss_res = float(((ys - pred) ** 2).sum())
        ss_tot = float(((ys - ys.mean()) ** 2).sum())
        r2 = 0.0 if ss_tot < 1e-12 else max(0.0, 1.0 - ss_res / ss_tot)
        report.append(dict(band=c, a=round(a, 5), b=round(b, 5),
                           r2=round(r2, 4), pif_ratio=round(float(keep.mean()), 4)))
    return out.astype(np.float32), report
