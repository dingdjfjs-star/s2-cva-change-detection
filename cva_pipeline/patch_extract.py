# -*- coding: utf-8 -*-
"""变化图斑矢量化：rasterio.features.shapes 一次遍历 + 属性挂接 → GeoJSON。

- 连通域编号图 labels 上调用 features.shapes（connectivity=8），每个图斑
  一步得到多边形边界（免逐图斑反复栅格化）；
- 属性在 stats 阶段已算好（面积/幅度/一致性/纹理），此处再补 from→to 类别
  （图斑内众数），组装为带完整属性的 GeoJSON Feature；
- 坐标按 transform 反投影到地理/投影坐标，精度裁剪到 0.1 m 控制体积。
"""
from __future__ import annotations

import json

import numpy as np
from rasterio import features as rfeatures


def _mode_class(values: np.ndarray) -> int:
    """非零众数；全零返回 0。"""
    v = values[values > 0]
    if v.size == 0:
        return 0
    return int(np.bincount(v).argmax())


def extract_change_patches(labels: np.ndarray, transform, class1: np.ndarray | None,
                           class2: np.ndarray | None, class_names: dict,
                           patch_props: dict, magnitude: np.ndarray,
                           pixel_area_m2: float) -> list[dict]:
    """labels → GeoJSON Feature 列表（1 个图斑 1 个 Feature）。"""
    geoms = rfeatures.shapes(labels.astype(np.int32), mask=labels > 0,
                             connectivity=8, transform=transform)
    feats = []
    for geom, value in geoms:
        lid = int(value)
        area_px = int(patch_props["area"][lid - 1]) if lid - 1 < len(patch_props["area"]) else 0
        mag = float(patch_props["mag_mean"][lid - 1]) if np.isfinite(patch_props["mag_mean"][lid - 1]) else None
        props = {
            "patch_id": lid,
            "area_px": area_px,
            "area_m2": round(area_px * pixel_area_m2, 1),
            "area_ha": round(area_px * pixel_area_m2 / 1e4, 3),
            "mean_magnitude": None if mag is None else round(float(mag), 4),
            "coherence": round(float(patch_props["coherence"][lid - 1]), 3),
            "texture_sigma": round(float(patch_props["tex_mean"][lid - 1]), 4)
            if np.isfinite(patch_props["tex_mean"][lid - 1]) else None,
        }
        if class1 is not None:
            m = labels == lid
            c1 = _mode_class(class1[m])
            c2 = _mode_class(class2[m])
            props["from_class"] = c1
            props["to_class"] = c2
            props["from_name"] = class_names.get(c1, str(c1))
            props["to_name"] = class_names.get(c2, str(c2))
            props["change_type"] = f"{class_names.get(c1, c1)}→{class_names.get(c2, c2)}"
        # 坐标裁剪精度（0.1 m），控制文件体积
        g2 = json.loads(json.dumps(geom))
        for ring in g2["coordinates"][0]:
            ring[0] = round(ring[0], 1)
            ring[1] = round(ring[1], 1)
        feats.append({"type": "Feature", "geometry": g2, "properties": props})
    return feats


def patch_table(feats: list[dict]) -> "list[dict]":
    """图斑属性表（报表/CSV 用），按面积降序。"""
    rows = [dict(f["properties"]) for f in feats]
    rows.sort(key=lambda r: -r["area_px"])
    return rows
