# -*- coding: utf-8 -*-
"""年度最优值合成：逐像元 medoid 合成（xarray 向量化实现）。

原理
----
把一年内该像元的所有**有效**观测视为多维光谱空间中的一组点，取到
"逐波段中位数" 欧氏距离最小的那个观测作为年度合成值：

    m_b = median_t(X[t,b])                     （逐波段中位数，抗离群）
    t*  = argmin_t  Σ_b (X[t,b] - m_b)²        （medoid：真实的某一景观测）
    comp[b] = X[t*, b]

为什么用 medoid 而不是均值/中位数合成：
- 均值/中位数合成的像元在光谱空间里是"虚构"的，可能不满足任何真实地物的
  波段相关结构（例如 NDVI 与红光的关系被破坏），而 medoid 输出的是某一景
  真实观测，光谱自洽；
- medoid 对云残留离群值稳健（中位数距离天然抗污染），且隐式选择了"最接近
  典型状态"的时相，达到"最优可用像元(BAP)"的效果；
- 对双季稻这类强物候地物，中位数基准让合成落在生长旺季附近，抑制了
  移栽泡田期与收割裸地期的干扰。

实现全部用 xarray 命名维度运算：(T,C,H,W) → (C,H,W)，NaN 表示无有效观测。
"""
from __future__ import annotations

import numpy as np
import xarray as xr


def medoid_composite(stack: xr.DataArray, min_valid_obs: int = 4) -> dict:
    """年度 medoid 合成。

    参数
    ----
    stack : (time, band, y, x) 反射率，无效观测已置 NaN（见 cloud_mask.build_invalid_mask）
    min_valid_obs : 有效观测次数下限，低于该值的像元输出 NaN（nodata）

    返回
    ----
    dict(composite=(band,y,x) DataArray, valid_obs=(y,x) int16, best_time=(y,x) int16,
         obs_counts=(y,x) int16)
    """
    if "time" not in stack.dims:
        raise ValueError("stack 需要 time 维")

    # 1) 逐像元有效观测次数：至少一个波段非 NaN 即记一次观测
    valid = stack.notnull().any("band")            # (time,y,x) bool
    obs_counts = valid.sum("time").astype("int16")  # (y,x)

    # 2) 逐波段中位数基准（skipna：NaN 不参与统计）
    med = stack.median("time", skipna=True)        # (band,y,x)

    # 3) 每次观测到中位数的欧氏距离（只在有观测的波段上累加）
    diff = stack - med                              # (time,band,y,x)
    dist2 = (diff ** 2).sum("band", skipna=True)    # (time,y,x)；全 NaN 像元 → 0，需修正
    all_nan = diff.isnull().all("band")             # 该次观测整景缺失
    dist2 = dist2.where(~all_nan, np.inf)           # 缺失观测距离置 +inf，永不入选
    dist2 = dist2.where(obs_counts > 0, np.inf)     # 全年无观测的像元全置 inf

    # 4) argmin 选观测（并列取最早时相）；vectorized isel 逐像元取数
    best_time = dist2.argmin("time")                # (y,x)
    comp = stack.isel(time=best_time)               # (y,x,band) → 转回 (band,y,x)
    comp = comp.transpose("band", "y", "x")

    # 5) 观测次数不足 → nodata
    comp = comp.where(obs_counts >= min_valid_obs)

    return dict(
        composite=comp.astype("float32"),
        valid_obs=obs_counts,
        best_time=best_time.astype("int16"),
        obs_counts=obs_counts,
    )
