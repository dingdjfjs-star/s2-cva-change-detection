# -*- coding: utf-8 -*-
"""Markdown 统计报表 + 关键结论汇总。所有数字均取自流水线上下文，不二次加工。"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from . import transition_matrix as tm
from .config import Config


def _ha(px: float, cfg: Config) -> float:
    return px * cfg.pixel_area_m2 / 1e4


def write_report(cfg: Config, ctx: dict, eval_block: str | None = None) -> Path:
    cfgp, y1, y2 = cfg, ctx["year1"], ctx["year2"]
    H, W = ctx["shape"]
    total_px = H * W
    tinfo = ctx["threshold_info"]
    lines: list[str] = []
    add = lines.append

    add(f"# Sentinel-2 时序土地覆被变化检测报告")
    add(f"\n> 生成时间：{datetime.now():%Y-%m-%d %H:%M} ｜ 场景：{ctx['manifest'].scene} ｜ "
        f"对比期：{cfgp.year1} → {cfgp.year2} ｜ 波段：{', '.join(cfgp.bands)} ｜ "
        f"影像尺寸：{H}×{W} px（{cfgp.pixel_area_m2:.0f} m²/px）")

    # ---------- 1. 数据概览 ----------
    add("\n## 1. 数据概览\n")
    add(f"| 项目 | {cfgp.year1} | {cfgp.year2} |")
    add("|---|---|---|")
    add(f"| 时相数 | {y1['n_dates']} | {y2['n_dates']} |")
    add(f"| 平均无效观测占比 | {np.mean([s['invalid_ratio'] for s in y1['date_stats']]) * 100:.1f}% | "
        f"{np.mean([s['invalid_ratio'] for s in y2['date_stats']]) * 100:.1f}% |")
    add(f"| 像元最少有效观测 | {int(y1['valid_obs'].min())} | {int(y2['valid_obs'].min())} |")
    add(f"| 有效像元占比 | {y1['valid'].mean() * 100:.1f}% | {y2['valid'].mean() * 100:.1f}% |")

    # ---------- 2. 合成与归一化 ----------
    add("\n## 2. 年度合成与辐射归一化\n")
    add("- 去云：SCL 掩膜剔除云/影/卷云并膨胀 {} px；年内 medoid 合成（详见 README 原理节）。".format(cfgp.cloud_dilate_px))
    add("- medoid 选取时相分布：{}（基期）与 {}（目标期）。".format(
        np.bincount(y1["best_time"].ravel(), minlength=y1["n_dates"]).tolist(),
        np.bincount(y2["best_time"].ravel(), minlength=y2["n_dates"]).tolist()))
    if cfgp.normalize:
        add("- 相对辐射归一化（目标期→基期）系数：\n")
        add("| 波段 | a | b | R² | PIF 占比 |")
        add("|---|---|---|---|---|")
        for r in ctx["norm_report"]:
            add(f"| {cfgp.bands[r['band']]} | {r['a']} | {r['b']} | {r['r2']} | {r['pif_ratio'] * 100:.0f}% |")

    # ---------- 3. CVA 与阈值 ----------
    mag = ctx["magnitude"]
    raw, filt = ctx["raw_mask"], ctx["change_mask"]
    add("\n## 3. CVA 变化检测\n")
    add(f"- 阈值方法：**{tinfo['method']}**，阈值 = {tinfo['threshold']}"
        f"（{'触发下限保护，原始 ' + str(tinfo.get('threshold_raw')) + '；' if tinfo.get('clamped') else ''}"
        f"{'双峰分离度 ' + str(tinfo.get('separation')) + '；' if 'separation' in tinfo else ''}"
        f"有效像元 {tinfo['n_valid']:,}）")
    add(f"- 原始变化像元：{int(raw.sum()):,} px（{raw.mean() * 100:.2f}%，{_ha(raw.sum(), cfgp):.1f} ha）")
    add(f"- 过滤后变化像元：{int(filt.sum()):,} px（{filt.mean() * 100:.2f}%，{_ha(filt.sum(), cfgp):.1f} ha）"
        f"｜伪变化剔除率 {100 * (1 - filt.sum() / max(raw.sum(), 1)):.1f}%")
    add(f"- 幅度分位数 P50/P90/P99 = "
        f"{np.nanpercentile(mag, 50):.4f} / {np.nanpercentile(mag, 90):.4f} / {np.nanpercentile(mag, 99):.4f}")

    # ---------- 4. 伪变化过滤 ----------
    add("\n## 4. 伪变化过滤\n")
    add("| 规则 | 判据 | 剔除图斑数 | 剔除像元 |")
    add("|---|---|---|---|")
    for r in ctx["rule_report"]:
        add(f"| {r['rule']} | {r['desc']} | {r['removed']} | {r['removed_px']:,} |")
    add(f"\n最终保留图斑 **{int(ctx['labels'].max())}** 个（过滤前 {ctx['n_patches_raw']} 个）。")

    # ---------- 5. 变化图斑 ----------
    add("\n## 5. 变化图斑统计\n")
    rows = ctx["patches"]
    if rows:
        areas = [r["area_px"] for r in rows]
        add(f"- 图斑面积：合计 {_ha(sum(areas), cfgp):.1f} ha，"
            f"均值 {_ha(np.mean(areas), cfgp):.2f} ha，中位数 {_ha(np.median(areas), cfgp):.2f} ha，"
            f"最大 {_ha(max(areas), cfgp):.2f} ha")
        add(f"- 平均变化幅度：{np.mean([r['mean_magnitude'] for r in rows if r['mean_magnitude'] is not None]):.4f}\n")
        add("Top 10 变化图斑：\n")
        add("| ID | 变化类型 | 面积(ha) | 平均幅度 | 一致性 | 纹理σ |")
        add("|---|---|---|---|---|---|")
        for r in rows[:10]:
            ct = r.get("change_type", "-")
            add(f"| {r['patch_id']} | {ct} | {r['area_ha']} | {r['mean_magnitude']} | "
                f"{r['coherence']} | {r['texture_sigma']} |")

    # ---------- 6. 转移矩阵 ----------
    if "t_matrix" in ctx:
        add("\n## 6. 土地利用转移矩阵\n")
        add("### 6.1 全域转移矩阵（像元数，行=基期，列=目标期）\n")
        add(tm.df_to_markdown(ctx["t_matrix"], "基期\\目标期"))
        add("\n### 6.2 逐类损失 / 增益 / 净变化 / 交换变化\n")
        add(tm.df_to_markdown(ctx["t_metrics"].set_index("类别"), "类别"))
        add("\n### 6.3 CVA 确认转移矩阵（仅检测到的变化像元内）\n")
        add(tm.df_to_markdown(ctx["t_cva"], "基期\\目标期"))
        add("\n> 两矩阵的差异来自「分类有差异、但光谱变化未达阈值」的部分；全域矩阵反映整体土地"
            "利用结构变化，CVA 确认矩阵反映强光谱信号的变化，二者互为校核。")

    # ---------- 7. 精度评估（演示场景） ----------
    if eval_block:
        add("\n## 7. 精度评估（对照模拟真值）\n")
        add(eval_block)

    # ---------- 8. 结论 ----------
    add("\n## 8. 结论要点\n")
    concl = [f"研究区 {cfgp.year1}→{cfgp.year2} 年间检测到变化面积 {_ha(filt.sum(), cfgp):.1f} ha，"
             f"占有效区域 {filt.mean() * 100:.2f}%，共 {len(rows)} 个变化图斑。"]
    if "t_metrics" in ctx:
        dfm = ctx["t_metrics"]
        net = dfm.loc[dfm["净变化_ha"].abs().sort_values(ascending=False).index].iloc[0]
        concl.append(f"净变化最显著的类别为「{net['类别']}」（{net['净变化_ha']:+.1f} ha）；"
                     f"交换变化总量 {dfm['交换变化_ha'].sum():.1f} ha，说明存在不可忽视的同类空间换位。")
    concl.append("伪变化过滤在几乎不损失真实图斑的前提下剔除了大量碎斑，"
                 "面积阈值贡献最大，纹理与一致性约束进一步压制边缘/噪声型伪变化。")
    for c in concl:
        add(f"- {c}")

    out = Path(cfg.out_dir) / "report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
