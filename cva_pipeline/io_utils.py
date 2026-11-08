# -*- coding: utf-8 -*-
"""IO 工具：时序影像清单读取、xarray 时序栈构建、GeoTIFF/GeoJSON 写出。

清单(manifest) 是流水线与数据解耦的桥梁 —— 无论是模拟数据还是真实
Sentinel-2 L2A 栅格（用 tools/build_manifest.py 扫描生成），只要给出：

    {
      "scene": "demo",
      "crs": "EPSG:32650",
      "transform": [10.0, 0, 585000, 0, -10, 2860000],   # GDAL geotransform
      "bands": ["B02", "B03", "B04", "B08"],
      "nodata": 0,
      "dates": [
        {"date": "2023-01-15", "stack": "xxx/20230115.tif", "scl": "xxx/20230115_scl.tif"},
        ...
      ]
    }

即可直接进流水线。stack 为按 bands 顺序排列的多波段 float32 反射率
（0~1.5 量级），scl 为单波段 uint8（哨兵 SCL 类别码，可缺省）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from affine import Affine


# ---------------------------------------------------------------- 清单 ----
@dataclass
class Manifest:
    scene: str
    crs: str
    transform: Affine
    bands: list
    nodata: float
    dates: list  # [{"date","stack","scl"}, ...]

    @classmethod
    def load(cls, path: str | Path) -> "Manifest":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            scene=d.get("scene", "scene"),
            crs=d.get("crs", "EPSG:32650"),
            transform=Affine.from_gdal(*d["transform"]),
            bands=list(d["bands"]),
            nodata=float(d.get("nodata", 0.0)),
            dates=sorted(d["dates"], key=lambda x: x["date"]),
        )

    def dates_of_year(self, year: int) -> list:
        return [d for d in self.dates if d["date"].startswith(str(year))]


# ------------------------------------------------------------ 时序栈 ----
def load_year_stack(m: Manifest, bands: list, year: int) -> tuple[xr.DataArray, xr.DataArray | None]:
    """读取某一年全部时相 → (反射率时序栈 (time,band,y,x), SCL 时序栈 (time,y,x)|None)。

    反射率以 float32 装载，nodata 像元填 NaN，交给下游统一 NaN 语义处理。
    """
    entries = m.dates_of_year(year)
    if not entries:
        raise ValueError(f"清单中没有 {year} 年的时相")
    ref = rasterio.open(entries[0]["stack"])
    H, W = ref.height, ref.width
    ref.close()

    stack = np.full((len(entries), len(bands), H, W), np.nan, np.float32)
    scl = np.zeros((len(entries), H, W), np.uint8)
    has_scl = True
    for t, e in enumerate(entries):
        with rasterio.open(e["stack"]) as src:
            arr = src.read().astype(np.float32)  # (C,H,W)
            if arr.shape[0] != len(bands):
                raise ValueError(f"{e['stack']} 波段数 {arr.shape[0]} != 清单 {len(bands)}")
        bad = arr <= m.nodata + 1e-6 if m.nodata != 0 else arr <= 0
        arr[bad] = np.nan
        # 物理合理性护栏：反射率超出 [-0.1, 2] 视为坏值
        arr[(arr < -0.1) | (arr > 2.0)] = np.nan
        stack[t] = arr
        if e.get("scl") and Path(e["scl"]).exists():
            with rasterio.open(e["scl"]) as src:
                scl[t] = src.read(1)
        else:
            has_scl = False

    coords = {
        "time": [e["date"] for e in entries],
        "band": list(bands),
        "y": np.arange(H),
        "x": np.arange(W),
    }
    da = xr.DataArray(stack, coords=coords, dims=("time", "band", "y", "x"), name="reflectance")
    scl_da = xr.DataArray(scl, coords={k: coords[k] for k in ("time", "y", "x")},
                          dims=("time", "y", "x"), name="scl") if has_scl else None
    return da, scl_da


def scene_shape(m: Manifest) -> tuple[int, int]:
    with rasterio.open(m.dates[0]["stack"]) as src:
        return src.height, src.width


# -------------------------------------------------------------- 写出 ----
def write_tif(path, arr, transform: Affine, crs: str, dtype: str,
              nodata=None, names: list | None = None, compress="deflate") -> None:
    """写出单/多波段 GeoTIFF。arr 为 (H,W) 或 (C,H,W)。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.asarray(arr)
    if a.ndim == 2:
        a = a[None, ...]
    if a.ndim != 3:
        raise ValueError(f"write_tif 期望 (H,W) 或 (C,H,W)，收到 {a.shape}")
    profile = dict(driver="GTiff", height=a.shape[1], width=a.shape[2], count=a.shape[0],
                   dtype=dtype, crs=crs, transform=transform, compress=compress,
                   nodata=nodata if nodata is not None else None)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(a.astype(dtype))
        if names:
            dst.descriptions = tuple(names)


def write_geojson(path, features: list, crs_name: str = "EPSG:32650") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fc = {"type": "FeatureCollection",
          "crs": {"type": "name", "properties": {"name": crs_name}},
          "features": features}
    path.write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")
