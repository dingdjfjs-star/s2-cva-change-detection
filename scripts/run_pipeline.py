# -*- coding: utf-8 -*-
"""真实数据入口：python scripts/run_pipeline.py --config config.json

配置文件由 Config.save_json 生成后按需修改（见 README「接入真实数据」）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from cva_pipeline.config import Config        # noqa: E402
from cva_pipeline.pipeline import run         # noqa: E402
from cva_pipeline import viz                  # noqa: E402
from cva_pipeline.report import write_report  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Sentinel-2 CVA 变化检测流水线")
    ap.add_argument("--config", required=True, help="配置 JSON 路径")
    ap.add_argument("--no-figures", action="store_true", help="跳过图件生成")
    args = ap.parse_args()

    cfg = Config.from_json(args.config)
    if not Path(cfg.manifest_path).exists():
        sys.exit(f"[错误] 清单不存在：{cfg.manifest_path}\n"
                 "请先用 tools/build_manifest.py 从真实影像目录生成清单，"
                 "并确认 config 里的 manifest_path / samples_csv 指向正确路径。")
    if cfg.samples_csv and not Path(cfg.samples_csv).exists():
        print(f"[提示] 样本文件不存在：{cfg.samples_csv}，将跳过分类与转移矩阵")
        cfg.samples_csv = None
    ctx = run(cfg)
    if not args.no_figures and cfg.make_figures:
        for f in viz.make_all_figures(cfg, ctx):
            print(f"  图件 → {f}")
    rep = write_report(cfg, ctx)
    print(f"  报表 → {rep}")


if __name__ == "__main__":
    main()
