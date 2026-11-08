# -*- coding: utf-8 -*-
"""去云模块：SCL 掩膜（含形态学膨胀）为主，4 波段规则云分数为兜底。

两条路径：
1) SCL（Sentinel-2 L2A 自带场景分类层）：直接按类别码剔除无观测，再按
   配置膨胀 N 像元 —— 云边缘的半透明薄雾在 SCL 里往往标成 "云中概率" 或
   未标记，膨胀是实践中性价比最高的防御。
2) 规则云分数（无 SCL 的 L1C/自定义栈兜底）：借鉴 s2cloudless 的单时相
   思路，用蓝波段亮度、全波段亮度、低 NDVI 三个因子的加权和作为云分数。
   云在蓝波段亮、全谱亮、且不具植被/水体光谱特征，因此该组合对四波段
   栈是可解释的最小特征集。注意它无法识别云影 —— 这是明确的能力边界。
"""
from __future__ import annotations

import numpy as np
import xarray as xr


# ------------------------------------------------------------- 形态学 ----
def binary_dilation(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """3×3 结构元的二值膨胀（手写 np.roll 实现，免依赖）。mask: (...,H,W) bool。"""
    out = mask.copy()
    for _ in range(max(0, iterations)):
        acc = out.copy()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                acc |= np.roll(np.roll(out, dx, axis=-2), dy, axis=-1)
        out = acc
    return out


# ---------------------------------------------------------------- SCL ----
def invalid_from_scl(scl: xr.DataArray, invalid_codes, dilate_px: int = 2) -> xr.DataArray:
    """SCL → 无效观测掩膜 (time,y,x) bool（True=该时相该像元不可用）。"""
    bad = xr.zeros_like(scl, dtype=bool)
    for code in invalid_codes:
        bad |= scl == code
    if dilate_px > 0:
        # 对每个时相膨胀（SCL 尺寸 (T,H,W)，roll 作用在后两维）
        arr = bad.values
        for t in range(arr.shape[0]):
            arr[t] = binary_dilation(arr[t], dilate_px)
        bad = xr.DataArray(arr, coords=bad.coords, dims=bad.dims)
    return bad


# -------------------------------------------------------- 规则云分数 ----
def rule_cloud_score(stack: xr.DataArray) -> xr.DataArray:
    """4 波段规则云分数，返回 (time,y,x)，取值 0~1，越大越像云。

    f1 蓝亮    : clip((B02 - 0.11) / (0.25 - 0.11), 0, 1)
    f2 全亮    : clip((mean(B02,B03,B04) - 0.15) / (0.40 - 0.15), 0, 1)
    f3 低植被  : clip((0.20 - NDVI) / 0.20, 0, 1)
    score      : 0.5*f1 + 0.3*f2 + 0.2*f3
    """
    b2, b3, b4, b8 = (stack.sel(band=b) for b in ("B02", "B03", "B04", "B08"))
    f1 = ((b2 - 0.11) / 0.14).clip(0, 1)
    f2 = (((b2 + b3 + b4) / 3.0 - 0.15) / 0.25).clip(0, 1)
    ndvi = (b8 - b4) / (b8 + b4 + 1e-6)
    f3 = ((0.20 - ndvi) / 0.20).clip(0, 1)
    return (0.5 * f1 + 0.3 * f2 + 0.2 * f3).fillna(0.0)


def build_invalid_mask(stack: xr.DataArray, scl: xr.DataArray | None, cfg) -> xr.DataArray:
    """统一入口：返回 (time,y,x) bool 无效掩膜。优先 SCL，否则规则云分数。"""
    if cfg.use_scl and scl is not None:
        return invalid_from_scl(scl, cfg.scl_invalid_codes, cfg.cloud_dilate_px)
    score = rule_cloud_score(stack)
    bad = score > cfg.fallback_cloud_thr
    if cfg.cloud_dilate_px > 0:
        arr = bad.values.copy()
        for t in range(arr.shape[0]):
            arr[t] = binary_dilation(arr[t], cfg.cloud_dilate_px)
        bad = xr.DataArray(arr, coords=bad.coords, dims=bad.dims)
    return bad


def cloud_stats(invalid: xr.DataArray) -> list[dict]:
    """逐时相无效观测占比（报表/QC 用）。"""
    frac = invalid.mean(dim=("y", "x"))  # (time,)
    return [{"date": str(d.values), "invalid_ratio": float(f)} for d, f in zip(invalid.time, frac)]
