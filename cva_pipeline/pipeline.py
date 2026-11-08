# -*- coding: utf-8 -*-
"""流水线编排：清单 → 去云 → 年度合成 → 辐射归一化 → CVA → 阈值 → 伪变化
过滤 → 分类 → 图斑/矩阵 → 栅格与矢量落盘。run(cfg) 返回上下文供报表/图件使用。
"""
from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

import numpy as np

from . import cloud_mask, compositing, classify, io_utils
from . import cva as cva_mod
from . import pseudo_filter as pf
from . import patch_extract, radiometric, threshold as thr_mod
from . import transition_matrix as tm
from .config import Config


def _composite_year(m: io_utils.Manifest, cfg: Config, year: int, log) -> dict:
    t0 = time.time()
    stack, scl = io_utils.load_year_stack(m, list(cfg.bands), year)
    # 生长季窗口（可选）：只保留 DOY ∈ [lo, hi] 的时相再合成。
    # 动机：全年 medoid 会被"非生长季状态占多数"的地物（如冬季裸地期水稻）
    # 拉到离峰态，且两年 medoid 在物候抖动下可能跳变 —— 收窄窗口后 medoid
    # 落在稳定的生长旺季平台上，年际可比性显著提升。
    if cfg.composite_doy_range:
        lo, hi = cfg.composite_doy_range
        doys = np.array([date.fromisoformat(str(d)).timetuple().tm_yday for d in stack.time.values])
        sel = (doys >= lo) & (doys <= hi)
        stack = stack.isel(time=sel)
        if scl is not None:
            scl = scl.isel(time=sel)
    invalid = cloud_mask.build_invalid_mask(stack, scl, cfg)
    n_dates = stack.sizes["time"]
    stats = cloud_mask.cloud_stats(invalid)
    clean = stack.where(~invalid)
    res = compositing.medoid_composite(clean, cfg.min_valid_obs)
    comp = res["composite"]
    obs = res["valid_obs"].values
    # 有效像元 = 观测次数达标 且 所有波段均非 NaN（medoid 保留逐波段 NaN 语义）
    valid = (obs >= cfg.min_valid_obs) & np.isfinite(comp.values).all(axis=0)
    log(f"  [{year}] 时相 {n_dates} 景 | 平均无效观测 {np.mean([s['invalid_ratio'] for s in stats]) * 100:.1f}%"
        f" | 有效像元 {valid.mean() * 100:.1f}% | 用时 {time.time() - t0:.1f}s")
    return dict(stack=stack, invalid=invalid, date_stats=stats, composite=comp,
                composite_np=comp.values.astype(np.float32), valid=valid,
                valid_obs=res["valid_obs"].values, best_time=res["best_time"].values,
                n_dates=n_dates)


def run(cfg: Config, log=print) -> dict:
    cfg.ensure_dirs()
    t_start = time.time()
    ctx: dict = {"cfg": cfg}

    # ---------------- 1. 数据加载与年度合成 ----------------
    log("[1/8] 加载时序清单与年度 medoid 合成")
    m = io_utils.Manifest.load(cfg.manifest_path)
    ctx["manifest"] = m
    H, W = io_utils.scene_shape(m)
    y1 = _composite_year(m, cfg, cfg.year1, log)
    y2 = _composite_year(m, cfg, cfg.year2, log)
    ctx.update(year1=y1, year2=y2, shape=(H, W))

    # ---------------- 2. 相对辐射归一化 ----------------
    log("[2/8] 相对辐射归一化（PIF 迭代回归）")
    common_valid = y1["valid"] & y2["valid"]
    if cfg.normalize:
        x2n, norm_rep = radiometric.relative_normalization(
            y1["composite_np"], y2["composite_np"], common_valid,
            n_iter=cfg.norm_iter, sigma=cfg.norm_sigma)
        y2["composite_norm"] = x2n
    else:
        x2n = y2["composite_np"].copy()
        norm_rep = [dict(band=i, a=1.0, b=0.0, r2=1.0, pif_ratio=1.0) for i in range(len(cfg.bands))]
        y2["composite_norm"] = x2n
    ctx["norm_report"] = norm_rep
    for r in norm_rep:
        log(f"  波段{cfg.bands[r['band']]}: a={r['a']}, b={r['b']}, R²={r['r2']}, PIF={r['pif_ratio'] * 100:.0f}%")

    # ---------------- 3. CVA 变化向量分析（自研核心） ----------------
    log("[3/8] CVA 变化向量分析")
    mag, delta = cva_mod.change_vector_magnitude(
        y1["composite_np"], x2n,
        weights=np.asarray(cfg.band_weights) if cfg.band_weights else None)
    calc_ok = np.isfinite(mag) & common_valid
    mag[~calc_ok] = np.nan
    ctx["magnitude"], ctx["delta"], ctx["calc_mask"] = mag, delta, calc_ok
    log(f"  幅度范围 {np.nanpercentile(mag, 50):.4f}(P50) / {np.nanpercentile(mag, 99):.4f}(P99)")

    # ---------------- 4. 阈值分割 ----------------
    log("[4/8] 变化幅度阈值分割")
    tinfo = thr_mod.select_threshold(mag, calc_ok, cfg)
    raw_mask = np.zeros((H, W), bool)
    raw_mask[calc_ok] = mag[calc_ok] > tinfo["threshold"]
    ctx["threshold_info"], ctx["raw_mask"] = tinfo, raw_mask
    log(f"  方法 {tinfo['method']} | 阈值 {tinfo['threshold']} | 原始变化像元占比 {raw_mask.mean() * 100:.2f}%")

    # ---------------- 5. 伪变化过滤 ----------------
    log("[5/8] 伪变化过滤（面积 / 一致性 / 实心宽度 / 纹理）")
    # 形态学预处理：闭运算弥合真实图斑内部碎缝，开运算在标记前剥除 1~2px
    # 条带（配准残差）—— 先开后闭的顺序让条带无法在标记阶段粘连成簇
    work = pf.binary_closing(raw_mask, cfg.morph_close_px) if cfg.morph_close_px > 0 else raw_mask.copy()
    work = pf.binary_opening(work, cfg.morph_open_px) if cfg.morph_open_px > 0 else work
    labels, n_patches = pf.label_patches(work, connectivity=8)
    # 纹理底图：目标期 NIR 合成（年度合成已压制时相噪声，纹理高值处即空间异质区）
    nir = y2["composite_norm"][list(cfg.bands).index("B08")]
    texture = pf.local_std(nir, cfg.texture_window)
    stats = pf.patch_statistics(labels, n_patches, mag, texture)
    keep_ids, rule_report = pf.filter_patches(stats, cfg)
    change_mask, labels_f = pf.apply_filter(labels, keep_ids)

    ctx["texture"] = texture
    ctx["labels_raw"], ctx["n_patches_raw"] = labels, n_patches
    ctx["labels"], ctx["keep_ids"], ctx["rule_report"] = labels_f, keep_ids, rule_report
    ctx["change_mask"] = change_mask
    ctx["patch_props"] = pf.patch_statistics(labels_f, int(labels_f.max()), mag, texture)
    log(f"  图斑 {n_patches} → {int(labels_f.max())} 个 | 像元占比 {raw_mask.mean() * 100:.2f}% → {change_mask.mean() * 100:.2f}%")
    for r in rule_report:
        log(f"  {r['rule']}: 剔除 {r['removed']} 图斑 / {r['removed_px']} px")

    # ---------------- 6. 随机森林分类（转移矩阵 from/to） ----------------
    class1 = class2 = None
    cls_reports = {}
    if cfg.samples_csv and Path(cfg.samples_csv).exists():
        log("[6/8] 随机森林土地覆被分类")
        samples = classify.load_samples(cfg.samples_csv)
        class1, _, rep1 = classify.classify_year(y1["composite_np"], common_valid, samples, cfg.year1, cfg)
        class2, _, rep2 = classify.classify_year(x2n, common_valid, samples, cfg.year2, cfg)
        cls_reports = {str(cfg.year1): rep1, str(cfg.year2): rep2}
        for k, v in cls_reports.items():
            log(f"  {k}: 样本 {v['n_samples']} | OOB={v['oob_score']}")
    else:
        log("[6/8] 跳过分类（未提供训练样本，转移矩阵不可用）")
    ctx["class1"], ctx["class2"], ctx["cls_reports"] = class1, class2, cls_reports

    # ---------------- 7. 转移矩阵与图斑矢量化 ----------------
    log("[7/8] 转移矩阵与变化图斑提取")
    if class1 is not None:
        ctx["t_matrix"] = tm.transition_matrix(class1, class2, common_valid, cfg.class_names)
        ctx["t_metrics"] = tm.matrix_metrics(ctx["t_matrix"], cfg.pixel_area_m2, cfg.class_names)
        ctx["t_cva"] = tm.cva_confirmed_matrix(class1, class2, change_mask, cfg.class_names)
    feats = patch_extract.extract_change_patches(
        labels_f, m.transform, class1, class2, cfg.class_names, ctx["patch_props"], mag, cfg.pixel_area_m2)
    io_utils.write_geojson(Path(cfg.out_dir) / "change_patches.geojson", feats, cfg.scene_crs)
    ctx["patches"] = patch_extract.patch_table(feats)
    ctx["direction_summary"] = cva_mod.summarize_directions(delta, labels_f, 0)
    log(f"  变化图斑 {len(feats)} 个 → change_patches.geojson")

    # ---------------- 8. 结果落盘 ----------------
    log("[8/8] 写出栅格与统计表")
    nd, dt = float(cfg.nodata), None
    out = Path(cfg.out_dir)
    io_utils.write_tif(out / "rasters/cva_magnitude.tif", np.where(np.isfinite(mag), mag, nd),
                       m.transform, cfg.scene_crs, "float32", nodata=nd, names=["cva_magnitude"])
    io_utils.write_tif(out / "rasters/change_mask_filtered.tif", change_mask.astype("uint8"),
                       m.transform, cfg.scene_crs, "uint8", nodata=0, names=["change"])
    if cfg.write_intermediate:
        io_utils.write_tif(out / "rasters/change_mask_raw.tif", raw_mask.astype("uint8"),
                           m.transform, cfg.scene_crs, "uint8", nodata=0, names=["change_raw"])
        io_utils.write_tif(out / "rasters/change_patches_labels.tif", labels_f,
                           m.transform, cfg.scene_crs, "int32", nodata=0, names=["patch_id"])
        io_utils.write_tif(out / "rasters/composite_year1.tif",
                           np.where(np.isfinite(y1["composite_np"]), y1["composite_np"], nd),
                           m.transform, cfg.scene_crs, "float32", nodata=nd, names=list(cfg.bands))
        io_utils.write_tif(out / "rasters/composite_year2_norm.tif",
                           np.where(np.isfinite(x2n), x2n, nd),
                           m.transform, cfg.scene_crs, "float32", nodata=nd, names=list(cfg.bands))
        if class1 is not None:
            io_utils.write_tif(out / "rasters/class_year1.tif", class1, m.transform, cfg.scene_crs,
                               "int16", nodata=0, names=["class"])
            io_utils.write_tif(out / "rasters/class_year2.tif", class2, m.transform, cfg.scene_crs,
                               "int16", nodata=0, names=["class"])
            # 变化类型编码图：from*100 + to（仅变化像元）
            ct = np.zeros((H, W), "int32")
            ct[change_mask & (class1 > 0) & (class2 > 0)] = class1[change_mask & (class1 > 0) & (class2 > 0)] * 100 \
                + class2[change_mask & (class1 > 0) & (class2 > 0)]
            io_utils.write_tif(out / "rasters/change_type.tif", ct, m.transform, cfg.scene_crs,
                               "int32", nodata=0, names=["change_type"])
    _dump_stats(cfg, ctx)
    ctx["elapsed"] = time.time() - t_start
    log(f"完成：总用时 {ctx['elapsed']:.1f}s，输出目录 {out.resolve()}")
    return ctx


def _dump_stats(cfg: Config, ctx: dict) -> None:
    """统计明细 CSV 落盘（图斑表 / 转移矩阵 / 指标）。"""
    sd = cfg.stats_dir
    pd_mod = __import__("pandas")
    pd_mod.DataFrame(ctx["patches"]).to_csv(sd / "change_patches.csv", index=False, encoding="utf-8-sig")
    if "t_matrix" in ctx:
        ctx["t_matrix"].to_csv(sd / "transition_matrix.csv", encoding="utf-8-sig")
        ctx["t_metrics"].to_csv(sd / "class_change_metrics.csv", index=False, encoding="utf-8-sig")
        ctx["t_cva"].to_csv(sd / "transition_matrix_cva_confirmed.csv", encoding="utf-8-sig")
    pd_mod.DataFrame(ctx["rule_report"]).to_csv(sd / "pseudo_filter_report.csv",
                                                index=False, encoding="utf-8-sig")
    (sd / "threshold_info.json").write_text(
        json.dumps(ctx["threshold_info"], ensure_ascii=False, indent=2), encoding="utf-8")
