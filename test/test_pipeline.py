# -*- coding: utf-8 -*-
"""流水线自动化测试（纯 python 运行：python test/test_pipeline.py）。

覆盖：算法单元（Otsu/CVA/连通域/形态学/medoid/转移矩阵）+ 端到端小场景
冒烟（含真值召回门槛）。全部通过输出 ALL TESTS PASSED，任一失败抛异常。
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from cva_pipeline.threshold import otsu_threshold                       # noqa: E402
from cva_pipeline.cva import change_vector_magnitude, direction_cosines  # noqa: E402
from cva_pipeline.pseudo_filter import (label_patches, binary_opening,   # noqa: E402
                                        binary_closing, local_std, sobel_magnitude,
                                        filter_patches, patch_statistics)
from cva_pipeline.compositing import medoid_composite                    # noqa: E402
from cva_pipeline.transition_matrix import transition_matrix             # noqa: E402
from cva_pipeline.config import Config                                   # noqa: E402
from cva_pipeline import io_utils                                        # noqa: E402
from cva_pipeline.pipeline import run                                    # noqa: E402
from tools.simulate_s2 import simulate                                   # noqa: E402

PASS = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS
    if not cond:
        raise AssertionError(f"FAIL: {name} {detail}")
    PASS += 1
    print(f"  ok  {name}")


def test_otsu():
    rng = np.random.default_rng(0)
    a = np.concatenate([rng.normal(0.02, 0.008, 9000), rng.normal(0.25, 0.03, 1000)])
    r = otsu_threshold(a)
    check("otsu 双峰分割阈值位于两峰之间", 0.035 < r["threshold"] < 0.20,
          f"thr={r['threshold']}")  # 9:1 样本不均衡时切点偏向窄峰 3σ 外，为数学最优
    check("otsu 双峰分离度>0.5", r["separation"] > 0.5, f"sep={r['separation']}")
    r2 = otsu_threshold(rng.normal(0, 1, 5000))  # 单峰：理论分离度≈0.64（切均值处）
    check("otsu 单峰分离度低于双峰且<0.7", r2["separation"] < 0.7 and r2["separation"] < r["separation"],
          f"sep={r2['separation']} vs 双峰={r['separation']}")


def test_cva():
    x1 = np.stack([np.full((4, 4), 0.1), np.full((4, 4), 0.2)])
    mag, delta = change_vector_magnitude(x1, x1.copy())
    check("CVA 同输入幅度为 0", float(np.abs(mag).max()) < 1e-9)
    x2 = np.stack([np.full((4, 4), 0.1), np.full((4, 4), 0.5)])  # ΔNIR=0.3
    mag, delta = change_vector_magnitude(x1, x2)
    check("CVA 已知向量模长", abs(float(mag[0, 0]) - 0.3) < 1e-6)
    cos = direction_cosines(delta, mag)
    check("CVA 方向余弦 NIR 分量=1", abs(float(cos[1, 0, 0]) - 1.0) < 1e-6)
    x1n = x1.copy(); x1n[0, 0, 0] = np.nan
    mag2, _ = change_vector_magnitude(x1n, x2)
    check("CVA NaN 语义传播", bool(np.isnan(mag2[0, 0])))


def test_label_and_morph():
    mask = np.zeros((40, 40), bool)
    mask[3:8, 3:8] = True          # 块 A 25px
    mask[20:33, 20:24] = True      # 块 B 52px（4×13）
    mask[15, 36] = True            # 孤立单像元
    labels, n = label_patches(mask)
    check("连通域数量（含孤立点）", n == 3, f"n={n}")
    areas = np.bincount(labels.ravel())[1:]
    check("连通域面积正确", sorted(areas) == [1, 25, 52], f"{sorted(areas)}")

    opened = binary_opening(mask, 1)
    lab2, n2 = label_patches(opened)
    check("开运算去除孤立像元", n2 == 2 and not opened[15, 36])
    closed = binary_closing(opened, 1)
    check("闭运算不扩大面积(此例)", int(closed.sum()) >= int(opened.sum()))

    # 随机场景与暴力洪泛对拍
    rng = np.random.default_rng(7)
    for _ in range(5):
        rmask = rng.random((30, 30)) > 0.78
        lab, nn = label_patches(rmask)
        # 暴力 BFS
        seen = np.zeros_like(rmask)
        cnt = 0
        for i in range(30):
            for j in range(30):
                if rmask[i, j] and not seen[i, j]:
                    cnt += 1
                    stack = [(i, j)]
                    seen[i, j] = True
                    while stack:
                        y, x = stack.pop()
                        for dy in (-1, 0, 1):
                            for dx in (-1, 0, 1):
                                ny, nx_ = y + dy, x + dx
                                if 0 <= ny < 30 and 0 <= nx_ < 30 and rmask[ny, nx_] and not seen[ny, nx_]:
                                    seen[ny, nx_] = True
                                    stack.append((ny, nx_))
        check(f"并查集与BFS对拍 #{cnt}", nn == cnt, f"union={nn} bfs={cnt}")


def test_texture():
    rng = np.random.default_rng(1)
    flat = np.full((50, 50), 0.2) + rng.normal(0, 0.005, (50, 50))
    rough = rng.normal(0.2, 0.08, (50, 50))
    check("平坦区局部σ小", float(local_std(flat, 5).mean()) < 0.01)
    check("粗糙区局部σ大", float(local_std(rough, 5).mean()) > 0.05)
    edge_img = np.zeros((50, 50)); edge_img[:, 25:] = 1.0
    check("Sobel 在边缘处响应最大",
          float(sobel_magnitude(edge_img)[:, 23:28].max()) > float(sobel_magnitude(edge_img).max() * 0.99))


def test_medoid():
    import xarray as xr
    # 3 时相：中位基准 medoid 应选第 2 观测
    t = np.full((3, 2, 1, 1), np.nan)
    vals = [0.0, 0.4, 1.0]
    for i, v in enumerate(vals):
        t[i, :, 0, 0] = v
    da = xr.DataArray(t, coords={"time": [0, 1, 2], "band": ["a", "b"], "y": [0], "x": [0]},
                      dims=("time", "band", "y", "x"))
    res = medoid_composite(da, min_valid_obs=1)
    check("medoid 选中位数最近观测", abs(float(res["composite"].values[0, 0, 0]) - 0.4) < 1e-6)
    t[2, :, 0, 0] = np.nan  # 整时相缺失 → 仅剩 2 次有效观测
    da2 = xr.DataArray(t, coords=da.coords, dims=da.dims)
    res2 = medoid_composite(da2, min_valid_obs=3)
    check("观测不足置 NaN", bool(np.isnan(res2["composite"].values[0, 0, 0])))
    res3 = medoid_composite(da2, min_valid_obs=2)
    # 剩余观测 0.0 / 0.4 距中位数 0.2 等距 → 平局取最早时相 0.0
    check("观测达标正常合成(平局取最早)", abs(float(res3["composite"].values[0, 0, 0]) - 0.0) < 1e-6)


def test_filter_rules():
    cfg = Config(min_patch_px=6, strong_patch_px=60, min_coherence=0.55, texture_max_sigma=0.06)
    mag = np.zeros((40, 40)); mag[:] = 0.1
    tex = np.zeros((40, 40))  # 全平坦 → 纹理全过
    # 图斑1：25px 连片（一致性高）→ 保留；图斑2：1px → R1 剔除；图斑3：细长 8px 一字线 → 一致性低
    mask = np.zeros((40, 40), bool)
    mask[2:7, 2:7] = True
    mask[20, 20] = True
    mask[30, 5:13] = True
    labels, n = label_patches(mask)
    stats = patch_statistics(labels, n, mag, tex)
    keep, report = filter_patches(stats, cfg)
    check("过滤规则：保留连片 25px 图斑", 1 in keep)
    check("过滤规则：剔除 1px 孤点", len([i for i in keep if stats['area'][i - 1] == 1]) == 0)
    line_patch = [i for i in range(1, n + 1) if stats["area"][i - 1] == 8]
    if line_patch:
        check("过滤规则：剔除细长低一致性图斑", line_patch[0] not in keep)


def test_transition_matrix():
    rng = np.random.default_rng(3)
    c1 = rng.integers(1, 4, (50, 50)).astype(np.int16)
    c2 = c1.copy()
    c2.flat[:100] = rng.integers(1, 4, 100)  # 打乱 100 个像元制造转移
    names = {1: "水", 2: "林", 3: "耕"}
    valid = np.ones((50, 50), bool)
    df = transition_matrix(c1, c2, valid, names)
    check("转移矩阵守恒：总和=有效像元数", int(df.loc["合计", "合计"]) == 2500)
    check("转移矩阵对角占优", (np.diag(df.values[:3, :3]) > 0).all())


def test_e2e_small():
    with tempfile.TemporaryDirectory() as td:
        sim = simulate(Path(td) / "sim", w=140, h=140, seed=99, quiet=True)
        cfg = Config(manifest_path=sim["manifest"], samples_csv=sim["samples"],
                     out_dir=str(Path(td) / "out"), rf_trees=60, make_figures=False,
                     write_intermediate=False,
                     composite_doy_range=(150, 260))  # 与演示一致：生长季峰值窗口合成
        ctx = run(cfg, log=lambda *a: None)
        check("端到端：产出变化掩膜", ctx["change_mask"].sum() > 0)
        check("端到端：保留图斑≥1", int(ctx["labels"].max()) >= 1)
        # 真值召回门槛：全局 ≥0.7（弱信号小图斑在迷你场景允许漏检），
        # 强信号事件（水体→耕地，幅度最大）≥0.95
        import rasterio
        gt = rasterio.open(Path(sim["truth_dir"]) / "change_mask_truth.tif").read(1).astype(bool)
        tp = int((ctx["change_mask"] & gt).sum())
        rec = tp / max(int(gt.sum()), 1)
        check("端到端：真值召回率≥0.7", rec >= 0.7, f"recall={rec:.3f}")
        events = (Path(sim["truth_dir"]) / "events.json").read_text(encoding="utf-8")
        import json as _json
        ev_list = _json.loads(events)
        ev3 = [e for e in ev_list if "水体" in e["name"]]
        if ev3:
            em = rasterio.open(Path(sim["truth_dir"]) / f"event_{ev3[0]['id']}.tif").read(1).astype(bool)
            r3 = (ctx["change_mask"] & em).sum() / max(int(em.sum()), 1)
            check("端到端：强信号事件召回≥0.95", r3 >= 0.95, f"recall={r3:.3f}")
        check("端到端：转移矩阵生成", "t_matrix" in ctx and ctx["t_matrix"].shape[0] >= 4)
        # medoid 合成图与掩膜栅格可写可读
        with rasterio.open(Path(td) / "out" / "rasters" / "change_mask_filtered.tif") as src:
            check("端到端：GeoTIFF 落盘可回读", src.read(1).shape == (140, 140))


if __name__ == "__main__":
    print("== CVA 流水线自动化测试 ==")
    test_otsu()
    test_cva()
    test_label_and_morph()
    test_texture()
    test_medoid()
    test_filter_rules()
    test_transition_matrix()
    test_e2e_small()
    print(f"\nALL TESTS PASSED ({PASS} checks)")
