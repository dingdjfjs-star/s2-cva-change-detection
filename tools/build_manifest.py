# -*- coding: utf-8 -*-
"""把真实 Sentinel-2 栅格目录整理成流水线清单(manifest.json)。

支持两种命名习惯（自动识别）：
1. 每时相一个多波段栈：  20230515.tif           （波段顺序用 --bands 声明）
   可选配 SCL：           20230515_scl.tif
2. 波段分文件：           20230515_B02.tif ...   （--pattern "{date}_{band}.tif"）

用法示例：
    python tools/build_manifest.py --dir D:/s2/scene1 --out manifest.json \
        --bands B02,B03,B04,B08 --crs EPSG:32650 \
        --transform 10,0,585000,0,-10,2860000
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import rasterio


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bands", default="B02,B03,B04,B08")
    ap.add_argument("--crs", default=None, help="缺省取自首个影像")
    ap.add_argument("--transform", default=None,
                    help="GDAL geotransform 六元组，缺省取自首个影像")
    ap.add_argument("--pattern", default="{date}.tif", help="多波段栈命名模板")
    ap.add_argument("--scl-pattern", default="{date}_scl.tif")
    ap.add_argument("--scene", default="scene")
    args = ap.parse_args()

    src = Path(args.dir)
    bands = args.bands.split(",")
    pat = re.compile(r"^(?P<date>\d{8})")  # 日期前缀识别
    by_date: dict[str, dict] = {}

    for f in sorted(src.glob("*.tif")):
        m = pat.match(f.name)
        if not m:
            continue
        d = m.group("date")
        date = f"{d[:4]}-{d[4:6]}-{d[6:]}"
        e = by_date.setdefault(date, {})
        if f.name.endswith("_scl.tif"):
            e["scl"] = str(f.resolve())
        elif f.name == args.pattern.format(date=d):
            e["stack"] = str(f.resolve())
        else:
            # 波段分文件模式：收集 {date}_{band}.tif
            for b in bands:
                if f.name == f"{d}_{b}.tif":
                    e.setdefault("stack_files", {})[b] = str(f.resolve())

    # 波段分文件 → 合并成单栈（写 "<date>_stack.tif"）
    dates = []
    for date, e in sorted(by_date.items()):
        if "stack" not in e and "stack_files" in e:
            files = e["stack_files"]
            missing = [b for b in bands if b not in files]
            if missing:
                print(f"[warn] {date} 缺波段 {missing}，跳过")
                continue
            out_t = src / f"{date.replace('-', '')}_stack.tif"
            with rasterio.open(files[bands[0]]) as ref:
                prof = ref.profile.copy()
                prof.update(count=len(bands), compress="deflate")
            with rasterio.open(out_t, "w", **prof) as dst:
                for i, b in enumerate(bands):
                    dst.write(rasterio.open(files[b]).read(1), i + 1)
                dst.descriptions = tuple(bands)
            e["stack"] = str(out_t.resolve())
        if "stack" in e:
            dates.append(dict(date=date, stack=e["stack"], scl=e.get("scl")))

    if not dates:
        sys.exit("未识别到任何时相影像，请检查命名/目录")

    with rasterio.open(dates[0]["stack"]) as ref:
        crs = args.crs or str(ref.crs)
        tf = [float(x) for x in args.transform.split(",")] if args.transform else \
            list(ref.transform.to_gdal())
        nodata = float(ref.nodata) if ref.nodata is not None else 0.0

    manifest = dict(scene=args.scene, crs=crs, transform=tf, bands=bands,
                    nodata=nodata, dates=dates)
    Path(args.out).write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    print(f"清单已生成：{args.out}（{len(dates)} 个时相）")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
