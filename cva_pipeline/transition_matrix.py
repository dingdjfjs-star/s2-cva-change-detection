# -*- coding: utf-8 -*-
"""土地利用转移矩阵：全要素矩阵 + 损失/增益/净变化/交换变化分解。

定义（土地变化科学标准口径，Pontius & Shrestha 2005）
-----------------------------------------------------
    转移矩阵 T[i,j] = 基期类 i → 目标期类 j 的像元数
    损失 Loss_i  = T[i,i] - Σ_j T[i,j]           （行内非对角之和，转出）
    增益 Gain_j  = T[j,j] 增量 = Σ_i T[i,j] - T[j,j]（列内非对角之和，转入）
    净变化 Net_c = |终止面积 - 初始面积|
    交换变化 Swap_c = 2·min(Loss_c, Gain_c)      （同类间"换地"的部分）
    总变化 = 净变化 + 交换变化
交换变化刻画了净变化看不到的"空间换位"（如 A 处耕地变林地、B 处林地变
耕地，净变化近似为 0，但真实扰动是双份）—— 这是只看两期面积对比的常见盲区。

另提供 "CVA 确认转移矩阵"：只在变化检测确认的图斑内统计转移，用于对比
"分类差异"与"光谱变化"两路证据的一致性。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def transition_matrix(class1: np.ndarray, class2: np.ndarray, valid: np.ndarray,
                      class_names: dict) -> pd.DataFrame:
    """全要素转移矩阵（像元数）。行=基期，列=目标期，含合计。"""
    c1 = class1[valid & (class1 > 0) & (class2 > 0)]
    c2 = class2[valid & (class1 > 0) & (class2 > 0)]
    codes = sorted(class_names.keys())
    df = pd.crosstab(pd.Categorical(c1, categories=codes), pd.Categorical(c2, categories=codes),
                     dropna=False)
    df.index = [class_names.get(int(i), i) for i in df.index]
    df.columns = [class_names.get(int(c), c) for c in df.columns]
    df["合计"] = df.sum(axis=1)
    df.loc["合计"] = df.sum(axis=0)
    return df


def matrix_metrics(counts: pd.DataFrame, pixel_area_m2: float, class_names: dict) -> pd.DataFrame:
    """逐类损失/增益/净变化/交换变化（公顷）。输入为不含合计行的方阵。"""
    ha = pixel_area_m2 / 1e4
    square = counts.copy()
    for name in ("合计",):
        if name in square.columns:
            square = square.drop(columns=name)
        if name in square.index:
            square = square.drop(index=name)
    rows = []
    for code, name in class_names.items():
        if name not in square.index and name not in square.columns:
            continue
        init = float(square.loc[name].sum()) if name in square.index else 0.0
        final = float(square[name].sum()) if name in square.columns else 0.0
        persist = float(square.loc[name, name]) if (name in square.index and name in square.columns) else 0.0
        loss = init - persist          # 转出
        gain = final - persist         # 转入
        net = abs(final - init)
        swap = 2.0 * min(loss, gain)
        rows.append(dict(类别=name,
                         初始面积_ha=round(init * ha, 2),
                         终止面积_ha=round(final * ha, 2),
                         损失_ha=round(loss * ha, 2),
                         增益_ha=round(gain * ha, 2),
                         净变化_ha=round((final - init) * ha, 2),
                         交换变化_ha=round(swap * ha, 2)))
    return pd.DataFrame(rows)


def cva_confirmed_matrix(class1: np.ndarray, class2: np.ndarray, change_mask: np.ndarray,
                         class_names: dict) -> pd.DataFrame:
    """CVA 确认转移矩阵：仅在检测出的变化像元内统计 from→to。"""
    sel = change_mask & (class1 > 0) & (class2 > 0)
    c1, c2 = class1[sel], class2[sel]
    codes = sorted(class_names.keys())
    df = pd.crosstab(pd.Categorical(c1, categories=codes), pd.Categorical(c2, categories=codes),
                     dropna=False)
    df.index = [class_names.get(int(i), i) for i in df.index]
    df.columns = [class_names.get(int(c), c) for c in df.columns]
    return df


def df_to_markdown(df: pd.DataFrame, index_name: str = "") -> str:
    """手工实现 DataFrame → Markdown 管道表格（避免 tabulate 依赖）。"""
    cols = [index_name] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [f"{v:,}" if isinstance(v, (int, np.integer))
                                                     else (f"{v:,.2f}" if isinstance(v, (float, np.floating)) else str(v))
                                                     for v in row.values]) + " |")
    return "\n".join(lines)
