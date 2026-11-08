# -*- coding: utf-8 -*-
"""合成 Sentinel-2 时序影像模拟器（演示/测试数据，非真实卫星数据）。

设计目标：给流水线提供一份"物理上讲得通、真值完全已知"的 S2 时序数据，
覆盖真实数据处理中的全部关键难点：

1. 地物光谱的季节轨迹（植被红边/近红外动态），稻田含移栽泡田期 NIR 骤降；
2. 逐时相大气差异（增益/偏移）与传感器噪声；
3. 云、云影、卷云三类云污染（半透明混合、SCL 类别码）；
4. 地块级物候年际抖动与土壤湿度差异 —— 天然的"伪变化"来源；
5. 五处已知真值变化（采伐、建设、围垦、开垦、撂荒），用于精度评估。

空间框架：赣南某 4×4 km 乡村景观（EPSG:32650，10 m），地块用 Voronoi
剖分 + 地物先验布局（北部林地、河流、水库、村镇、田间道路）。

产物：
    data/sim_scene/
      ├── manifest.json            时序清单（流水线唯一入口）
      ├── YYYY-MM-DD.tif           每时相 4 波段 float32 反射率
      ├── YYYY-MM-DD_scl.tif       每时相 SCL 场景分类（uint8）
      ├── samples.csv              两个年度各 6 类训练样本
      └── truth/                   真值：两期类别图 / 变化掩膜 / 事件掩膜
"""
from __future__ import annotations

import json
from datetime import date, timedelta, timedelta
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine

BANDS = ["B02", "B03", "B04", "B08"]
CLASSES = {1: "水体", 2: "林地", 3: "草地", 4: "耕地", 5: "建设用地", 6: "裸地"}

# ---------------------------------------------------------------- 光谱参数 --
# 每类：物候峰值 DOY / 宽度 / 各波段(植被茂盛值 lo?, 枯季值) —— 记 (裸季lo, 旺季hi)
SIGNS = {
    2: dict(peak=240, width=95, bands={"B02": (0.095, 0.042), "B03": (0.105, 0.052),
                                       "B04": (0.078, 0.028), "B08": (0.28, 0.46)}),   # 常绿林
    3: dict(peak=228, width=72, bands={"B02": (0.105, 0.052), "B03": (0.115, 0.062),
                                       "B04": (0.128, 0.06), "B08": (0.13, 0.25)}),   # 草地
    4: dict(peak=225, width=48, bands={"B02": (0.14, 0.045), "B03": (0.15, 0.055),
                                       "B04": (0.135, 0.03), "B08": (0.16, 0.47)}),    # 水稻为主（冬底 NIR 0.16）
    5: dict(peak=200, width=200, bands={"B02": (0.128, 0.128), "B03": (0.15, 0.15),
                                        "B04": (0.145, 0.145), "B08": (0.185, 0.185)}),  # 建设
    6: dict(peak=180, width=180, bands={"B02": (0.15, 0.155), "B03": (0.17, 0.175),
                                        "B04": (0.18, 0.19), "B08": (0.24, 0.26)}),    # 裸地
    1: dict(peak=180, width=180, bands={"B02": (0.055, 0.07), "B03": (0.045, 0.062),
                                        "B04": (0.03, 0.045), "B08": (0.012, 0.022)}),  # 水体
}
CLOUD_REF = np.array([0.70, 0.72, 0.75, 0.65])  # 云光谱参考


def _smooth(a: np.ndarray, k: int = 9, passes: int = 2) -> np.ndarray:
    """盒式平滑（积分图），生成空间相关随机场用。"""
    p = k // 2
    ap = np.pad(a, p, mode="edge")
    cs = np.zeros((ap.shape[0] + 1, ap.shape[1] + 1))
    cs[1:, 1:] = ap.cumsum(0).cumsum(1)
    H, W = a.shape
    out = cs[k:k + H, k:k + W] - cs[0:H, k:k + W] - cs[k:k + H, 0:W] + cs[0:H, 0:W]
    out = out / float(k * k)
    return out if passes <= 1 else _smooth(out, k, passes - 1)


# ---------------------------------------------------------------- 地表布局 --
def build_landscape(H: int, W: int, rng: np.random.Generator) -> dict:
    """Voronoi 地块 + 地物先验 → (year1 类别图, 地块图, 地块效应)。"""
    yy, xx = np.mgrid[0:H, 0:W]

    # 1) Voronoi 地块剖分
    n_parcel = max(int(H * W / 2200), 12)
    sx = rng.uniform(0, W, n_parcel)
    sy = rng.uniform(0, H, n_parcel)
    d2 = (xx[..., None] - sx[None, None, :]) ** 2 + (yy[..., None] - sy[None, None, :]) ** 2
    parcels = d2.argmin(axis=2).astype(np.int32) + 1  # 1..n

    # 2) 地块类别（北部与东部偏林地，其余以耕地为主）
    forest_field = _smooth(rng.normal(0, 1, (H, W)), 25)
    forest_zone = (yy < 0.22 * H) | (xx > 0.78 * W) | (forest_field > 0.75)
    wet_zone = (yy > 0.62 * H) & (xx < 0.5 * W)  # 南部水田概率更高
    pcls = np.zeros(n_parcel + 1, np.int8)
    for p in range(1, n_parcel + 1):
        cy, cx = sy[p - 1], sx[p - 1]
        if forest_zone[int(cy), int(cx)]:
            pcls[p] = rng.choice([2, 3, 6, 1], p=[0.72, 0.15, 0.08, 0.05])
        else:
            if wet_zone[int(cy), int(cx)]:
                pcls[p] = rng.choice([4, 3, 6, 5, 1], p=[0.60, 0.15, 0.10, 0.08, 0.07])
            else:
                pcls[p] = rng.choice([4, 3, 6, 5, 2, 1], p=[0.55, 0.15, 0.10, 0.08, 0.07, 0.05])
    class_map = pcls[parcels].astype(np.int8)

    # 3) 先验地物：河流 / 水库 / 村镇 / 道路
    river_y = (0.52 * H + 0.09 * H * np.sin(xx / 38.0)).astype(int)
    class_map[np.abs(yy - river_y) < 6] = 1
    ry, rx, rry, rrx = 0.74 * H, 0.22 * W, 0.095 * H, 0.14 * W
    reservoir = (((yy - ry) / rry) ** 2 + ((xx - rx) / rrx) ** 2) < 1.0
    class_map[reservoir] = 1
    village = (((yy - 0.56 * H) / (0.075 * H)) ** 2 + ((xx - 0.52 * W) / (0.07 * W)) ** 2) < 1.0
    class_map[village] = 5
    class_map[np.abs(yy - 0.34 * H) < 2] = 5          # 东西向道路
    class_map[np.abs(xx - 0.6 * W) < 2] = 6           # 南北向机耕道

    # 4) 地块效应：波段乘性系数 + 物候抖动（两期抖动差异 = 物候伪变化源）
    factor = rng.normal(1.0, 0.06, (n_parcel + 1, 4)).clip(0.85, 1.15)
    jitter1 = rng.normal(0, 4, n_parcel + 1)
    jitter2 = jitter1 + rng.normal(0, 3.0, n_parcel + 1)  # 年际物候错位
    return dict(parcels=parcels, class_map=class_map, factor=factor,
                jitter1=jitter1, jitter2=jitter2, n_parcel=n_parcel)


def plant_changes(land: dict, H: int, W: int, rng: np.random.Generator) -> dict:
    """在 year2 类别图上植入 5 处已知变化，返回真值结构。

    事件落点采用"最优窗口搜索"：在目标类别的密度图上滑窗取密度最高处，
    保证每处变化都真实踩中源地物（面积足够、边界清晰）。
    """
    cm1 = land["class_map"].copy()
    cm2 = cm1.copy()
    yy, xx = np.mgrid[0:H, 0:W]
    events = []

    def ragged_rect(mask_area: np.ndarray, edge_noise: float = 0.12) -> np.ndarray:
        return mask_area & (_smooth(rng.normal(0, 1, (H, W)), 6) > -edge_noise * 4)

    def best_window(target: np.ndarray, fh: float, fw: float,
                    region: np.ndarray | None = None, step: int = 8):
        """在 target 密度图上找含量最高的 fh×fw 窗口（可限定 region）。"""
        bh, bw = int(fh * H), int(fw * W)
        best_pos, best_cnt = None, 0
        ys = range(0, H - bh, step)
        xs = range(0, W - bw, step)
        for wy in ys:
            for wx in xs:
                if region is not None and not region[wy + bh // 2, wx + bw // 2]:
                    continue
                cnt = int(target[wy:wy + bh, wx:wx + bw].sum())
                if cnt > best_cnt:
                    best_cnt, best_pos = cnt, (wy, wx)
        if best_pos is None or best_cnt < 0.3 * bh * bw:
            return None
        win = np.zeros_like(target)
        win[best_pos[0]:best_pos[0] + bh, best_pos[1]:best_pos[1] + bw] = True
        return win

    # E1 林地→裸地（采伐）：全图找林地最密的 0.10H×0.16W 窗口
    win = best_window(cm2 == 2, 0.10, 0.16)
    if win is not None:
        e1 = ragged_rect(win) & (cm2 == 2)
        cm2[e1] = 6
        events.append(("E1_林地→裸地(采伐)", e1))

    # E2 耕地→建设用地（村镇周边新建厂房）：限村镇附近搜索
    village = (((yy - 0.56 * H) / (0.16 * H)) ** 2 + ((xx - 0.52 * W) / (0.20 * W)) ** 2) < 1.0
    win = best_window((cm2 == 4) | (cm2 == 3), 0.08, 0.10, region=village)
    if win is not None:
        e2 = ragged_rect(win) & ((cm2 == 4) | (cm2 == 3))
        cm2[e2] = 5
        events.append(("E2_耕地→建设用地(厂房)", e2))

    # E3 水体→耕地（水库东侧围垦）
    ry, rx, rry, rrx = 0.74 * H, 0.22 * W, 0.095 * H, 0.14 * W
    e3 = ((((yy - ry) / rry) ** 2 + ((xx - rx) / rrx) ** 2) < 1.0) & (xx > rx) & (cm2 == 1)
    cm2[e3] = 4
    events.append(("E3_水体→耕地(围垦)", e3))

    # E4 草地→耕地（选取真实草地集中区）
    win = best_window(cm2 == 3, 0.10, 0.12)
    if win is not None:
        e4 = ragged_rect(win) & (cm2 == 3)
        if e4.sum() > 150:
            cm2[e4] = 4
            events.append(("E4_草地→耕地(开垦)", e4))

    # E5 耕地→裸地（撂荒/取土）：在河流以南的耕地区搜索
    win = best_window(cm2 == 4, 0.09, 0.13, region=(yy > 0.36 * H))
    if win is not None:
        e5 = ragged_rect(win) & (cm2 == 4)
        cm2[e5] = 6
        events.append(("E5_耕地→裸地(撂荒)", e5))

    change_gt = cm1 != cm2
    return dict(class1=cm1, class2=cm2, change_gt=change_gt,
                events=[(name, mask) for name, mask in events if mask.sum() > 0])


# ---------------------------------------------------------------- 时序渲染 --
def render_date(cls_map: np.ndarray, land: dict, doy: float, year_key: str,
                rng: np.random.Generator, moisture: np.ndarray | None) -> np.ndarray:
    """单时相 4 波段反射率（含大气/噪声/地块效应），返回 (4,H,W)。"""
    H, W = cls_map.shape
    jit = land["jitter1"] if year_key == "y1" else land["jitter2"]
    doy_eff = doy + jit[land["parcels"]]                 # 地块物候抖动
    out = np.zeros((4, H, W), np.float32)
    for ci, cls in enumerate(CLASSES):
        sig = SIGNS[cls]
        f = np.exp(-0.5 * ((doy_eff - sig["peak"]) / sig["width"]) ** 2)
        for b in range(4):
            lo, hi = sig["bands"][BANDS[b]]
            r = lo + (hi - lo) * f
            if cls == 4:                                  # 稻田移栽泡田期 NIR 骤降（窄凹陷，仅影响单时相）
                dip = np.exp(-0.5 * ((doy_eff - 130) / 12.0) ** 2)
                if b == 3:
                    r -= 0.30 * dip
                elif b == 2:
                    r += 0.03 * dip
                elif b == 1:
                    r += 0.025 * dip
            out[b] += np.where(cls_map == cls, r, 0.0)
    for b in range(4):                                    # 地块乘性系数
        out[b] *= land["factor"][land["parcels"], b]
    if moisture is not None:                              # 土壤湿度伪变化（仅耕地区）
        out[2] -= moisture * 0.012
        out[3] += moisture * 0.018
        out[0] += moisture * 0.006
    # 大气：整体增益 + 蓝波段散射偏移
    gain = 1.0 + rng.normal(0, 0.015)
    offsets = np.abs(rng.normal(0.008, 0.003, 4)) * np.array([1.7, 1.35, 1.0, 0.25])
    out *= gain
    out += offsets[:, None, None]
    out += rng.normal(0, 0.005, out.shape).astype(np.float32)  # 传感器噪声
    return np.clip(out, 0.004, 1.4)


def contaminate(refl: np.ndarray, cls_map: np.ndarray, rng: np.random.Generator):
    """云/影/卷云污染。返回 (污染后反射率, SCL 图)。"""
    H, W = cls_map.shape
    scl = np.zeros((H, W), np.uint8)
    base = {1: 6, 2: 4, 3: 4, 4: 4, 5: 5, 6: 5}
    for c, code in base.items():
        scl[cls_map == c] = code

    cover = rng.uniform(0, 0.15)
    cloud = np.zeros((H, W), bool)
    alpha = np.zeros((H, W), np.float32)
    shadow = np.zeros((H, W), bool)
    veil = np.zeros((H, W), np.float32)
    if cover > 0.05:
        field = _smooth(rng.normal(0, 1, (H, W)), 13)
        thr = np.quantile(field, 1 - cover)
        cloud = field > thr
        cloud = _smooth(cloud.astype(float), 3) > 0.3        # 边缘圆滑
        alpha = np.clip(_smooth(rng.normal(0.85, 0.2, (H, W)), 9), 0.45, 1.0) * cloud
        scl[cloud & (rng.random((H, W)) < 0.4)] = 9
        scl[cloud & (scl < 8)] = 8
        # 云影：向东北方向位移
        sh = np.zeros_like(cloud)
        dy, dx = int(H * 0.045), int(W * 0.055)
        sh[dy:, dx:] = cloud[:H - dy, :W - dx]
        shadow = sh & ~cloud
        scl[shadow] = 3
    if cover > 0.10:                                          # 卷云层（薄、局部，SCL=10 只标明显处）
        vfield = _smooth(rng.normal(0, 1, (H, W)), 31)
        veil = np.clip(vfield * 0.025, 0, 0.045)
        cirrus = veil > 0.028
        scl[cirrus & (scl < 8)] = 10

    out = refl.copy()
    out = out * (1 - alpha) + alpha * CLOUD_REF[:, None, None]     # 云混合
    shade = np.array([0.55, 0.55, 0.52, 0.45])[:, None, None]      # 云影压暗
    out = np.where(shadow[None], out * shade, out)
    out += veil[None] * np.array([1.0, 0.9, 0.8, 0.9])[:, None, None]  # 卷云薄雾
    return out, scl


# ------------------------------------------------------------------ 主流程 --
def simulate(out_dir: str | Path, w: int = 400, h: int = 400, seed: int = 20260829,
             quiet: bool = False) -> dict:
    out = Path(out_dir)
    (out / "truth").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    H, W = h, w
    log = (lambda *a: None) if quiet else print

    log(f"[sim] 场景 {W}×{H} @10m, seed={seed}")
    land = build_landscape(H, W, rng)
    truth = plant_changes(land, H, W, rng)
    log(f"[sim] 地块 {land['n_parcel']} 个，真值变化像元 {int(truth['change_gt'].sum())}"
        f"（{truth['change_gt'].mean() * 100:.1f}%），事件 {len(truth['events'])} 处")

    # 土壤湿度伪变化地块：平滑随机场取高值区（约 8% 耕地），年际轻微光谱漂移
    mf = _smooth(rng.normal(0, 1, (H, W)), 21)
    hi = np.quantile(mf, 0.90)
    moisture = np.clip((mf - hi) / max(mf.max() - hi, 1e-6), 0, 1)
    moisture *= ((truth["class1"] == 4) | (truth["class2"] == 4)).astype(float)

    # 时相列表：每 10 天一景（贴近 Sentinel-2 重访密度的量级）
    transform = Affine(10, 0, 585000, 0, -10, 2860000)
    entries = []
    for year, ykey in ((2023, "y1"), (2024, "y2")):
        d0 = date(year, 1, 10)
        n_dates = 0
        while d0.year == year:
            doy = (d0 - date(year, 1, 1)).days + 1
            cname = truth["class1"] if year == 2023 else truth["class2"]
            if year == 2024:
                # 模拟配准残差：目标期整体 1 px 错位 —— 真实 CVA 生产中最典型的
                # 伪变化来源（所有高对比边缘出现 1px 宽的"变化"条带），
                # 供伪变化过滤环节（R1/R2）实战演示。
                cname = np.roll(cname, 1, axis=0)
            refl = render_date(cname, land, doy, ykey, rng,
                               moisture if year == 2024 else None)
            refl, scl = contaminate(refl, cname, rng)
            fname = f"{d0:%Y-%m-%d}.tif"
            sname = f"{d0:%Y-%m-%d}_scl.tif"
            prof = dict(driver="GTiff", height=H, width=W, count=4, dtype="float32",
                        crs="EPSG:32650", transform=transform, compress="deflate")
            with rasterio.open(out / fname, "w", **prof) as dst:
                dst.write(refl.astype(np.float32))
                dst.descriptions = tuple(BANDS)
            with rasterio.open(out / sname, "w", driver="GTiff", height=H, width=W,
                               count=1, dtype="uint8", crs="EPSG:32650",
                               transform=transform, compress="deflate") as dst:
                dst.write(scl, 1)
            entries.append(dict(date=f"{d0:%Y-%m-%d}", stack=str(out / fname), scl=str(out / sname)))
            n_dates += 1
            d0 = d0 + timedelta(days=10)
        log(f"[sim] {year} 年 {n_dates} 时相完成")

    manifest = dict(scene="sim_gannan_demo", crs="EPSG:32650",
                    transform=[10.0, 0.0, 585000.0, 0.0, -10.0, 2860000.0],
                    bands=BANDS, nodata=-1.0, dates=entries)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                                       encoding="utf-8")

    # ---- 训练样本：两期各 6 类 × 120 点（避开类别边界 3px 与影像边缘） ----
    def edge_mask(cm: np.ndarray) -> np.ndarray:
        e = np.zeros_like(cm, bool)
        for dx in (-3, 0, 3):
            for dy in (-3, 0, 3):
                e |= np.roll(np.roll(cm, dx, 0), dy, 1) != cm
        return e

    rows = []
    for year, cname in ((2023, truth["class1"]), (2024, truth["class2"])):
        ok = ~edge_mask(cname)
        ok[:8, :] = ok[-8:, :] = False
        ok[:, :8] = ok[:, -8:] = False
        for cls, name in CLASSES.items():
            cand = np.argwhere((cname == cls) & ok)
            take = min(120, len(cand))
            sel = cand[rng.choice(len(cand), take, replace=False)]
            for r, c in sel:
                rows.append(dict(year=year, col=int(c), row=int(r), label=cls, class_name=name))
    import pandas as pd
    pd.DataFrame(rows).to_csv(out / "samples.csv", index=False, encoding="utf-8-sig")

    # ---- 真值落盘 ----
    tdir = out / "truth"
    for name, arr, dt in (("class_year1_truth", truth["class1"], "uint8"),
                          ("class_year2_truth", truth["class2"], "uint8"),
                          ("change_mask_truth", truth["change_gt"], "uint8")):
        with rasterio.open(tdir / f"{name}.tif", "w", driver="GTiff", height=H, width=W,
                           count=1, dtype=dt, crs="EPSG:32650", transform=transform,
                           compress="deflate") as dst:
            dst.write(arr.astype(dt), 1)
    events_meta = []
    for i, (name, mask) in enumerate(truth["events"], 1):
        with rasterio.open(tdir / f"event_{i}.tif", "w", driver="GTiff", height=H, width=W,
                           count=1, dtype="uint8", crs="EPSG:32650", transform=transform,
                           compress="deflate") as dst:
            dst.write(mask.astype("uint8"), 1)
        events_meta.append(dict(id=i, name=name, px=int(mask.sum()),
                                ha=round(mask.sum() / 100.0, 1)))
    (tdir / "events.json").write_text(json.dumps(events_meta, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    log(f"[sim] 完成 → {out}")
    return dict(manifest=str(out / "manifest.json"), samples=str(out / "samples.csv"),
                truth_dir=str(tdir), events=events_meta)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    simulate(Path(__file__).resolve().parents[1] / "data" / "sim_scene")
