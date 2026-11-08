# -*- coding: utf-8 -*-
"""端到端演示入口：
1. 若 data/sim_scene 缺失或 --force，先运行模拟器生成 S2 时序演示数据；
2. 配置并运行完整流水线（去云→合成→归一化→CVA→阈值→伪变化过滤→分类→
   图斑/转移矩阵→落盘）；
3. 生成全部示例结果图与 Markdown 报表；
4. 对照模拟真值输出精度评估（像元级 + 分事件召回）。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from cva_pipeline.config import Config                 # noqa: E402
from cva_pipeline.pipeline import run                  # noqa: E402
from cva_pipeline import viz                           # noqa: E402
from cva_pipeline.report import write_report           # noqa: E402
from tools.simulate_s2 import simulate                 # noqa: E402

DATA = ROOT / "data" / "sim_scene"


def accuracy_vs_truth(cfg: Config, ctx: dict) -> str:
    """对照模拟真值：像元级召回/精度 + 分事件召回 + 转移矩阵一致率。"""
    tdir = DATA / "truth"
    gt = rasterio.open(tdir / "change_mask_truth.tif").read(1).astype(bool)
    H, W = ctx["shape"]
    pred = ctx["change_mask"]
    calc = ctx["calc_mask"]
    valid = np.isfinite(gt)
    gt_v, pred_v, calc_v = gt & valid, pred & valid, calc & valid

    tp = int((pred_v & gt_v).sum())
    fp = int((pred_v & ~gt_v).sum())
    fn = int((~pred_v & gt_v).sum())
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    lines = [
        f"- 像元级：TP {tp:,}｜FP {fp:,}（伪变化残留）｜FN {fn:,}（漏检）",
        f"- **召回率 {recall * 100:.1f}% ｜ 精确率 {precision * 100:.1f}% ｜ F1 {f1 * 100:.1f}%**",
    ]

    # 分事件召回
    lines.append("\n| 事件 | 真值面积(ha) | 召回率 |")
    lines.append("|---|---|---|")
    events = json.loads((tdir / "events.json").read_text(encoding="utf-8"))
    for ev in events:
        em = rasterio.open(tdir / f"event_{ev['id']}.tif").read(1).astype(bool) & valid
        r = (pred & em).sum() / max(em.sum(), 1)
        lines.append(f"| {ev['name']} | {ev['ha']} | {r * 100:.1f}% |")

    # 转移矩阵真值对比
    c1t = rasterio.open(tdir / "class_year1_truth.tif").read(1)
    c2t = rasterio.open(tdir / "class_year2_truth.tif").read(1)
    changed_gt = gt_v
    agree = int((ctx["class1"][changed_gt] == c1t[changed_gt]).mean() * 100) if changed_gt.any() else 0
    agree2 = int((ctx["class2"][changed_gt] == c2t[changed_gt]).mean() * 100) if changed_gt.any() else 0
    lines.append(f"\n- 变化区分类一致率：基期 {agree}% / 目标期 {agree2}%"
                 f"（随机森林 from/to 判定与真值吻合度）")
    return "\n".join(lines)


def main(force: bool = False) -> None:
    t0 = time.time()
    if force or not (DATA / "manifest.json").exists():
        print("=" * 60)
        print("STEP 1/3  生成演示数据（合成 Sentinel-2 时序）")
        print("=" * 60)
        sim = simulate(DATA, w=400, h=400, seed=20260829)
    else:
        sim = dict(manifest=str(DATA / "manifest.json"), samples=str(DATA / "samples.csv"),
                   truth_dir=str(DATA / "truth"))
        print("STEP 1/3  演示数据已存在，跳过模拟（--force 重新生成）")

    print("\n" + "=" * 60)
    print("STEP 2/3  运行变化检测流水线")
    print("=" * 60)
    cfg = Config(
        manifest_path=str(DATA / "manifest.json"),
        samples_csv=str(DATA / "samples.csv"),
        year1=2023, year2=2024,
        out_dir=str(ROOT / "outputs"),
        threshold_method="otsu",
        normalize=True,
        composite_doy_range=(150, 260),   # 生长季峰值窗口（6 月初~9 月中，双季稻稳定生长期）
        min_valid_obs=4,
        min_patch_px=6, strong_patch_px=60,
        min_coherence=0.55, texture_max_sigma=0.06,
    )
    ctx = run(cfg)

    print("\n" + "=" * 60)
    print("STEP 3/3  生成结果图与报表")
    print("=" * 60)
    figs = viz.make_all_figures(cfg, ctx)
    for f in figs:
        print(f"  图件 → {f}")
    eval_block = accuracy_vs_truth(cfg, ctx)
    rep = write_report(cfg, ctx, eval_block)
    print(f"  报表 → {rep}")
    print(f"\n真值评估：\n{eval_block}")
    print(f"\n全部完成，用时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
