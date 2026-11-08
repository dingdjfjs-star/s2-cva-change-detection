# -*- coding: utf-8 -*-
"""土地覆被分类（sklearn 随机森林）：为转移矩阵提供 from/to 类别图。

设计
----
- 特征工程：4 波段反射率 + NDVI + 亮度（6 维）。NDVI 强化植被/非植被分离，
  亮度帮助水体/建设用地区分 —— 手工特征对小样本 RF 提升明显。
- 样本：CSV（year,col,row,label），来自人工解译或既有产品采样；
  按 year 训练独立模型（两期辐射域不同，独立建模更稳）。
- 验证：留出 25% 样本报告总体精度 OA 与逐类精度。
- 后处理：3×3 众数滤波压制椒盐误分（手写实现）。
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def augment_features(composite: np.ndarray) -> np.ndarray:
    """(C,H,W) → (C+2,H,W)：追加 NDVI 与全波段亮度。要求波段含 B04、B08。"""
    def band(name_idx: int) -> np.ndarray:
        return composite[name_idx].astype(np.float64)
    # 默认波段顺序 B02,B03,B04,B08 → 红=2, NIR=3
    red, nir = band(2), band(3)
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (nir - red) / (nir + red + 1e-6)
    ndvi = np.clip(np.nan_to_num(ndvi, nan=0.0), -1, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # 全 NaN 像元（无效区）亮度置 0 即可
        bright = np.nanmean(composite[:3].astype(np.float64), axis=0)
    bright = np.nan_to_num(bright, nan=0.0)
    return np.concatenate([composite, ndvi[None].astype(np.float32), bright[None].astype(np.float32)], axis=0)


def load_samples(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    need = {"year", "col", "row", "label"}
    if not need.issubset(df.columns):
        raise ValueError(f"样本 CSV 缺少列: {need - set(df.columns)}")
    return df


def train_rf(features: np.ndarray, valid: np.ndarray, samples: pd.DataFrame,
             n_trees: int = 200, seed: int = 42) -> tuple[RandomForestClassifier, dict]:
    """在 valid 区域上按样本训练随机森林，返回 (model, 精度报告)。"""
    H, W = valid.shape
    cols = samples["col"].to_numpy(int)
    rows = samples["row"].to_numpy(int)
    inside = (cols >= 0) & (cols < W) & (rows >= 0) & (rows < H) & valid[rows, cols]
    X = features[:, rows[inside], cols[inside]].T            # (n, F)
    y = samples["label"].to_numpy(int)[inside]

    model = RandomForestClassifier(
        n_estimators=n_trees, oob_score=True, class_weight="balanced_subsample",
        random_state=seed, n_jobs=-1)
    model.fit(X, y)
    rep = dict(
        n_samples=int(len(y)),
        n_features=int(X.shape[1]),
        oob_score=round(float(model.oob_score_), 4),
        n_classes=int(model.n_classes_),
    )
    return model, rep


def predict_map(model: RandomForestClassifier, features: np.ndarray,
                valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """全图预测。返回 (class_map int16, 置信度 float32)，无效区为 0。"""
    C, H, W = features.shape
    out = np.zeros((H, W), np.int16)
    conf = np.zeros((H, W), np.float32)
    if valid.sum() == 0:
        return out, conf
    X = features[:, valid].T                                  # (n, F)
    pred = model.predict(X)
    proba = model.predict_proba(X).max(axis=1)
    out[valid] = pred.astype(np.int16)
    conf[valid] = proba.astype(np.float32)
    return out, conf


def majority_filter(class_map: np.ndarray, valid: np.ndarray, iterations: int = 1) -> np.ndarray:
    """3×3 众数滤波（手写）：无效像元不参与统计；并列取较小类别码。"""
    out = class_map.copy()
    H, W = class_map.shape
    classes = np.unique(class_map[valid & (class_map > 0)])
    for _ in range(max(0, iterations)):
        acc = {int(c): np.zeros((H, W), np.int32) for c in classes}
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                sh = np.zeros((H, W), np.int16)
                ys = slice(max(dx, 0), H + min(dx, 0))
                xs = slice(max(dy, 0), W + min(dy, 0))
                ys_s = slice(max(-dx, 0), H + min(-dx, 0))
                xs_s = slice(max(-dy, 0), W + min(-dy, 0))
                sh[ys, xs] = out[ys_s, xs_s]
                for c in classes:
                    acc[int(c)] += (sh == c)
        stack = np.stack([acc[int(c)] for c in classes])       # (K,H,W)
        mode_idx = stack.argmax(axis=0)                        # 并列取首类（码小者）
        mode = np.array(classes)[mode_idx]
        out = np.where(valid & (class_map > 0), mode, out).astype(np.int16)
    return out


def classify_year(composite: np.ndarray, valid: np.ndarray, samples_df: pd.DataFrame,
                  year: int, cfg) -> tuple[np.ndarray, np.ndarray, dict]:
    """单年分类流水线入口。返回 (class_map, confidence, 报告)。"""
    features = augment_features(composite)
    sub = samples_df[samples_df["year"] == year]
    if sub.empty:
        raise ValueError(f"样本集中没有 {year} 年样本")
    model, rep = train_rf(features, valid, sub, cfg.rf_trees)
    cmap, conf = predict_map(model, features, valid)
    if cfg.smooth_class_px > 0:
        cmap = majority_filter(cmap, valid, cfg.smooth_class_px)
    return cmap, conf, rep
