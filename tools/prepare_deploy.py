# -*- coding: utf-8 -*-
"""准备 GitHub Pages 部署拷贝：镜像 代码+文档+图件+统计 到 ../s2-cva-change-detection-site/

排除项（不进仓库）：
- data/            合成时序影像 ~100MB（README 已说明 run_demo.py --force 可重新生成）
- outputs/rasters/ 中间栅格 ~15MB（可由流水线重新生成）
- .venv / __pycache__ / *.pyc / .git / .zcode
保留项：代码、README、index.html、config.sample.json、outputs/figures、outputs/stats、
        outputs/report.md、outputs/change_patches.geojson

用法：python tools/prepare_deploy.py   （在项目根目录运行）
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT.parent / "s2-cva-change-detection-site"

COPY_ITEMS = [
    "cva_pipeline",
    "tools",
    "scripts",
    "test",
    "outputs/figures",
    "outputs/stats",
    "outputs/report.md",
    "outputs/change_patches.geojson",
    "index.html",
    "README.md",
    "config.sample.json",
    "requirements.txt",
]


def main() -> None:
    if SITE.exists():
        shutil.rmtree(SITE)
    SITE.mkdir(parents=True)

    for item in COPY_ITEMS:
        src = ROOT / item
        dst = SITE / item
        if not src.exists():
            print(f"[warn] 缺少 {item}，跳过（先跑一次 run_demo.py）")
            continue
        if src.is_dir():
            shutil.copytree(
                src, dst,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    (SITE / ".nojekyll").write_text("", encoding="utf-8")  # 纯静态站，跳过 Jekyll

    n = sum(1 for _ in SITE.rglob("*") if _.is_file())
    size = sum(f.stat().st_size for f in SITE.rglob("*") if f.is_file()) / 1e6
    print(f"部署拷贝完成：{SITE}")
    print(f"  文件 {n} 个，共 {size:.1f} MB")


if __name__ == "__main__":
    main()
