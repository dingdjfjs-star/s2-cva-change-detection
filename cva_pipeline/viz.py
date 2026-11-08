# -*- coding: utf-8 -*-
"""示例结果图（matplotlib，中文字体，统一纸墨风配色）：

fig1_overview.png          总体概览：两期假彩色 / CVA 幅度 / 阈值直方图 / 过滤前后掩膜
fig2_change_distribution.png  变化分布图（变化图斑按 from→to 类型着色叠 + 类型面积条形图）
fig3_transition_matrix.png    转移矩阵热力图 + 逐类损失/增益/净/交换条形图
fig4_pseudo_filter.png        伪变化过滤：剔除图斑分布 + 图斑面积分布直方图
fig5_change_vectors.png       各主要转移类型的平均变化向量（方向指纹）
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

from .config import Config

# ---------- 全局样式 ----------
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DengXian", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110

INK = "#2b2b2b"
PAPER = "#faf7f0"

# 类别配色（与模拟器一致：水/林/草/耕/建/裸）
CLASS_COLORS = {1: "#3d7ec2", 2: "#2e6e3a", 3: "#8fbc6f", 4: "#d9c27a", 5: "#b0533b", 6: "#c9b8a0"}
# 变化类型配色（top 转移对循环取用）
TRANS_COLORS = ["#c23b22", "#e07b39", "#3b7a57", "#4a6fa5", "#8e4585",
                "#b8860b", "#5f7f80", "#a0522d", "#4682b4", "#6b8e23", "#cd5c5c", "#2f4f4f"]


def _style(ax, title=None, ticks: bool = False):
    """统一面板样式。ticks=False 用于影像面板（去刻度），图表面板传 True 保留刻度。"""
    ax.set_title(title, fontsize=10, color=INK)
    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#999")


def _rgb(composite: np.ndarray, band_order=(3, 2, 1), stretch=(2, 98)) -> np.ndarray:
    """假彩色合成（默认 NIR-Red-Green）+ 2% 线性拉伸。"""
    rgb = np.take(composite, list(band_order), axis=0).astype(np.float64)
    out = np.zeros_like(rgb)
    for i in range(3):
        band = rgb[i]
        lo, hi = np.nanpercentile(band, stretch)
        out[i] = np.clip((band - lo) / max(hi - lo, 1e-6), 0, 1)
    return np.clip(np.transpose(out, (1, 2, 0)), 0, 1)


def _extent_km(H: int, W: int, px: float):
    return [0, W * px / 1000, H * px / 1000, 0]


# ---------------------------------------------------------------- fig1 ----
def fig_overview(cfg: Config, ctx: dict) -> Path:
    mag, raw, filt = ctx["magnitude"], ctx["raw_mask"], ctx["change_mask"]
    tinfo = ctx["threshold_info"]
    H, W = ctx["shape"]
    ext = _extent_km(H, W, cfg.pixel_area_m2)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), facecolor=PAPER)
    axes[0, 0].imshow(_rgb(ctx["year1"]["composite_np"]), extent=ext)
    _style(axes[0, 0], f"{cfg.year1} 年度 medoid 合成（假彩色 NIR-R-G）")
    axes[0, 1].imshow(_rgb(ctx["year2"]["composite_norm"]), extent=ext)
    _style(axes[0, 1], f"{cfg.year2} 年度 medoid 合成（归一化后，假彩色）")

    im = axes[0, 2].imshow(mag, cmap="magma", extent=ext)
    _style(axes[0, 2], f"CVA 变化幅度 ‖ΔX‖（阈值 {tinfo['threshold']}）")
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.02, label="幅度")

    # 阈值直方图
    m = mag[np.isfinite(mag)]
    axes[1, 0].hist(m, bins=200, color="#4a6fa5", alpha=0.85)
    if "threshold" in tinfo and np.isfinite(tinfo["threshold"]):
        axes[1, 0].axvline(tinfo["threshold"], color="#c23b22", lw=1.6,
                           label=f"阈值 {tinfo['threshold']}")
        axes[1, 0].legend(fontsize=8)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("变化幅度", fontsize=9); axes[1, 0].set_ylabel("像元数(log)", fontsize=9)
    _style(axes[1, 0], f"幅度直方图（方法 {tinfo['method']}）", ticks=True)
    axes[1, 0].set_facecolor(PAPER)

    axes[1, 1].imshow(np.where(raw, 1.0, np.nan), cmap="autumn_r", vmin=0, vmax=1, extent=ext)
    _style(axes[1, 1], f"阈值分割原始变化（{raw.mean() * 100:.2f}%）")
    axes[1, 2].imshow(np.where(filt, 1.0, np.nan), cmap="autumn_r", vmin=0, vmax=1, extent=ext)
    _style(axes[1, 2], f"伪变化过滤后（{filt.mean() * 100:.2f}%，剔除率 "
                       f"{100 * (1 - filt.sum() / max(raw.sum(), 1)):.0f}%）")

    fig.suptitle("Sentinel-2 时序变化检测总体概览", fontsize=14, color=INK, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = cfg.fig_dir / "fig1_overview.png"
    fig.savefig(path, facecolor=PAPER, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- fig2 ----
def fig_change_distribution(cfg: Config, ctx: dict) -> Path | None:
    """变化分布图：图斑按 from→to 着色（要求分类可用）。"""
    if ctx.get("class1") is None:
        return None
    H, W = ctx["shape"]
    labels, filt = ctx["labels"], ctx["change_mask"]
    c1, c2 = ctx["class1"], ctx["class2"]
    names = cfg.class_names

    # 每图斑 from→to 众数 → 类型编码 from*100+to
    from_mode = np.zeros(int(labels.max()) + 1, np.int16)
    to_mode = np.zeros(int(labels.max()) + 1, np.int16)
    for lid in range(1, int(labels.max()) + 1):
        m = labels == lid
        if m.any() and (c1[m] > 0).any():
            from_mode[lid] = int(np.bincount(c1[m][c1[m] > 0]).argmax())
            to_mode[lid] = int(np.bincount(c2[m][c2[m] > 0]).argmax())

    type_ids = from_mode[labels] * 100 + to_mode[labels]
    pairs = sorted({(int(from_mode[i]), int(to_mode[i]))
                    for i in range(1, int(labels.max()) + 1) if from_mode[i] > 0 and to_mode[i] > 0})
    pair_color = {p: TRANS_COLORS[i % len(TRANS_COLORS)] for i, p in enumerate(pairs)}
    # 按图斑面积排序决定图例顺序（大→小）
    pair_area = {p: int(((type_ids == p[0] * 100 + p[1]) & filt).sum()) for p in pairs}
    pairs_sorted = sorted(pairs, key=lambda p: -pair_area[p])

    type_rgba = np.zeros((H, W, 4))
    for p in pairs_sorted:
        sel = filt & (type_ids == p[0] * 100 + p[1])
        col = matplotlib.colors.to_rgb(pair_color[p])
        type_rgba[sel] = (*col, 1.0)
    # 变化图斑上再叠加暗色边界感：轻微描边由 labels 边缘实现
    edge = filt & (np.roll(labels, 1, 0) != labels) | (filt & (np.roll(labels, 1, 1) != labels))
    type_rgba[edge] = (0.1, 0.1, 0.1, 0.9)

    ext = _extent_km(H, W, cfg.pixel_area_m2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7.5), facecolor=PAPER,
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    base = _rgb(ctx["year2"]["composite_norm"])
    ax1.imshow(base * 0.35 + 0.65 * np.ones_like(base))  # 灰化底图
    ax1.imshow(np.where(filt[..., None], type_rgba, np.nan))
    _style(ax1, f"{cfg.year1}→{cfg.year2} 变化图斑分布（按转移类型着色）")
    legend = [Patch(facecolor=pair_color[p], edgecolor="#333",
                    label=f"{names.get(p[0], p[0])}→{names.get(p[1], p[1])}  "
                          f"{pair_area[p] * cfg.pixel_area_m2 / 1e4:.1f} ha")
              for p in pairs_sorted]
    ax1.legend(handles=legend, loc="lower left", fontsize=8, framealpha=0.9, title="转移类型 / 面积")

    top = pairs_sorted[:12]
    vals = [pair_area[p] * cfg.pixel_area_m2 / 1e4 for p in top]
    ax2.barh(range(len(top)), vals, color=[pair_color[p] for p in top], edgecolor="#333", lw=0.4)
    ax2.set_yticks(range(len(top)))
    ax2.set_yticklabels([f"{names.get(p[0], p[0])}→{names.get(p[1], p[1])}" for p in top], fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel("面积 (ha)", fontsize=9)
    _style(ax2, "主要转移类型面积排名", ticks=True)
    ax2.set_facecolor(PAPER)
    ax2.tick_params(colors=INK)

    fig.suptitle("土地覆被变化分布图", fontsize=14, color=INK, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = cfg.fig_dir / "fig2_change_distribution.png"
    fig.savefig(path, facecolor=PAPER, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- fig3 ----
def fig_transition_matrix(cfg: Config, ctx: dict) -> Path | None:
    if "t_matrix" not in ctx:
        return None
    counts = ctx["t_matrix"].drop(index="合计", errors="ignore").drop(columns="合计", errors="ignore")
    metrics = ctx["t_metrics"].set_index("类别")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), facecolor=PAPER,
                                   gridspec_kw={"width_ratios": [1, 1.2]})
    arr = counts.to_numpy(float)
    ha = arr * cfg.pixel_area_m2 / 1e4
    im = ax1.imshow(ha, cmap="YlOrRd")
    ax1.set_xticks(range(len(counts.columns)), counts.columns, rotation=35, ha="right", fontsize=9)
    ax1.set_yticks(range(len(counts.index)), counts.index, fontsize=9)
    for i in range(ha.shape[0]):
        for j in range(ha.shape[1]):
            v = ha[i, j]
            if v > 0:
                ax1.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                         color="white" if v > ha.max() * 0.5 else INK)
    ax1.set_xlabel("目标期 (2024)", fontsize=10); ax1.set_ylabel("基期 (2023)", fontsize=10)
    _style(ax1, "转移矩阵（ha，对角线为未变化）", ticks=True)
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.02, label="面积 (ha)")

    x = np.arange(len(metrics))
    w = 0.2
    for k, (col, c) in enumerate(zip(["损失_ha", "增益_ha", "净变化_ha", "交换变化_ha"],
                                     ["#c23b22", "#2e6e3a", "#4a6fa5", "#b8860b"])):
        ax2.bar(x + (k - 1.5) * w, metrics[col], w, label=col.replace("_ha", ""), color=c, edgecolor="#333", lw=0.3)
    ax2.set_xticks(x, metrics.index, fontsize=9)
    ax2.axhline(0, color="#555", lw=0.8)
    ax2.set_ylabel("面积 (ha)", fontsize=10)
    ax2.legend(fontsize=8, ncol=2)
    _style(ax2, "逐类损失 / 增益 / 净变化 / 交换变化", ticks=True)
    ax2.set_facecolor(PAPER)
    ax2.tick_params(colors=INK)

    fig.suptitle("土地利用转移矩阵与变化分解", fontsize=14, color=INK, y=0.98)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    path = cfg.fig_dir / "fig3_transition_matrix.png"
    fig.savefig(path, facecolor=PAPER, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- fig4 ----
def fig_pseudo_filter(cfg: Config, ctx: dict) -> Path:
    labels_f = ctx["labels"]
    raw, final = ctx["raw_mask"], ctx["change_mask"]
    removed = raw & ~final          # 全部被剔除的伪变化（含形态学预处理剥除的条带）
    kept = final
    H, W = ctx["shape"]
    ext = _extent_km(H, W, cfg.pixel_area_m2)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), facecolor=PAPER)
    base = np.full((H, W, 3), 1.0)
    base[kept] = (0.24, 0.43, 0.65)    # 蓝：保留
    base[removed] = (0.76, 0.23, 0.13)  # 红：剔除
    axes[0].imshow(base, extent=ext)
    _style(axes[0], f"伪变化过滤结果（蓝=保留，红=剔除 {int(removed.sum()):,} px）")

    areas_raw = np.bincount(ctx["labels_raw"].ravel())[1:]
    areas_kept = np.bincount(labels_f.ravel())[1:]
    ha = cfg.pixel_area_m2 / 1e4
    a_raw_ha = areas_raw * ha
    a_kept_ha = areas_kept * ha
    lo = max(float(a_raw_ha.min()) if a_raw_ha.size else 0.01, 0.01)
    hi = max(float(a_raw_ha.max()) if a_raw_ha.size else 1.0,
             float(a_kept_ha.max()) if a_kept_ha.size else 1.0) * 1.3
    bins = np.logspace(np.log10(lo), np.log10(hi), 30)
    axes[1].hist(a_raw_ha, bins=bins, histtype="step", color="#b0533b",
                 lw=1.8, label="规则过滤前")
    axes[1].hist(a_kept_ha, bins=bins, color="#2e6e3a", alpha=0.75, label="规则过滤后")
    axes[1].set_xscale("log")
    axes[1].axvline(cfg.min_patch_px * ha, color=INK, ls="--", lw=1,
                    label=f"面积阈值 {cfg.min_patch_px}px={cfg.min_patch_px * ha:.2f}ha")
    axes[1].set_xlabel("图斑面积 (ha, log)", fontsize=9); axes[1].set_ylabel("图斑数", fontsize=9)
    axes[1].legend(fontsize=8)
    _style(axes[1], "图斑面积分布（形态学预处理后，规则过滤前后）", ticks=True)
    axes[1].set_facecolor(PAPER); axes[1].tick_params(colors=INK)

    # 按阶段统计剔除像元数：形态学(开/闭) + 四条规则
    rr = ctx["rule_report"]
    morph_px = int(raw.sum() - (ctx["labels_raw"] > 0).sum())
    names = (["形态学开闭"] if morph_px > 0 else []) + [r["rule"].split("_")[1] for r in rr]
    vals = ([morph_px] if morph_px > 0 else []) + [r["removed_px"] for r in rr]
    palette = ["#8c564b", "#b0533b", "#b8860b", "#5f7f80", "#4a6fa5"]
    bars = axes[2].bar(names, vals, color=[palette[i % len(palette)] for i in range(len(names))],
                       edgecolor="#333", lw=0.4)
    axes[2].bar_label(bars, fontsize=9, fmt="%d")
    axes[2].set_ylabel("剔除像元数", fontsize=9)
    axes[2].tick_params(axis="x", rotation=15, labelsize=8)
    _style(axes[2], "各阶段剔除贡献", ticks=True)
    axes[2].set_facecolor(PAPER); axes[2].tick_params(colors=INK)

    fig.suptitle("伪变化过滤（形态学去条带 + 面积阈值 + 空间一致性 + 实心宽度 + 纹理约束）",
                 fontsize=13, color=INK, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = cfg.fig_dir / "fig4_pseudo_filter.png"
    fig.savefig(path, facecolor=PAPER, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- fig5 ----
def fig_change_vectors(cfg: Config, ctx: dict) -> Path | None:
    """各主要转移类型的平均变化向量（CVA 方向指纹）。"""
    if ctx.get("class1") is None:
        return None
    c1, c2 = ctx["class1"], ctx["class2"]
    delta, filt = ctx["delta"], ctx["change_mask"]
    names = cfg.class_names
    bands = list(cfg.bands) + ["NDVI"]
    d = delta.astype(np.float64)

    def _ndvi(comp: np.ndarray) -> np.ndarray:
        return (comp[3].astype(np.float64) - comp[2]) / (comp[3] + comp[2] + 1e-6)

    ndvi_d = _ndvi(ctx["year2"]["composite_norm"]) - _ndvi(ctx["year1"]["composite_np"])
    feats = np.concatenate([d, ndvi_d[None]], axis=0)  # (5,H,W)：4 波段变化 + NDVI 变化

    sel = filt & (c1 > 0) & (c2 > 0)
    pairs = {}
    key = c1[sel].astype(int) * 100 + c2[sel].astype(int)
    for k in np.unique(key):
        pairs[int(k)] = int((key == k).sum())
    top = sorted(pairs, key=lambda k: -pairs[k])[:8]
    if not top:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=PAPER)
    x = np.arange(len(bands))
    for i, k in enumerate(top):
        c_from, c_to = k // 100, k % 100
        m = sel & (c1.astype(int) * 100 + c2.astype(int) == k)
        mean_v = feats[:, m].mean(axis=1)
        color = TRANS_COLORS[i % len(TRANS_COLORS)]
        ax1.plot(x, mean_v, "-o", ms=4, lw=1.6, color=color,
                 label=f"{names.get(c_from, c_from)}→{names.get(c_to, c_to)} ({pairs[k]} px)")
    ax1.axhline(0, color="#555", lw=0.8)
    ax1.set_xticks(x, bands, fontsize=9)
    ax1.set_ylabel("平均反射率变化 ΔX", fontsize=10)
    ax1.legend(fontsize=7.5, loc="best")
    _style(ax1, "主要转移类型的平均变化向量（方向指纹）", ticks=True)
    ax1.set_facecolor(PAPER); ax1.tick_params(colors=INK)

    # 幅度箱线对比
    mag = ctx["magnitude"]
    data, labels_, colors_ = [], [], []
    for k in top:
        c_from, c_to = k // 100, k % 100
        m = sel & (c1.astype(int) * 100 + c2.astype(int) == k)
        vals = mag[m]
        vals = vals[np.isfinite(vals)]
        data.append(vals); labels_.append(f"{names.get(c_from, c_from)}→{names.get(c_to, c_to)}")
        colors_.append(TRANS_COLORS[(top.index(k)) % len(TRANS_COLORS)])
    bp = ax2.boxplot(data, showfliers=False, patch_artist=True, tick_labels=labels_,
                     medianprops=dict(color=INK))
    for p, c in zip(bp["boxes"], colors_):
        p.set_facecolor(c); p.set_alpha(0.75); p.set_edgecolor("#333")
    ax2.set_ylabel("CVA 幅度", fontsize=10)
    ax2.tick_params(axis="x", rotation=30, labelsize=8)
    _style(ax2, "各转移类型的幅度分布", ticks=True)
    ax2.set_facecolor(PAPER); ax2.tick_params(colors=INK)

    fig.suptitle("变化向量方向分析：不同转移类型的光谱变化特征", fontsize=13, color=INK, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = cfg.fig_dir / "fig5_change_vectors.png"
    fig.savefig(path, facecolor=PAPER, dpi=150)
    plt.close(fig)
    return path


def make_all_figures(cfg: Config, ctx: dict) -> list[str]:
    out = []
    for fn in (fig_overview, fig_change_distribution, fig_transition_matrix,
               fig_pseudo_filter, fig_change_vectors):
        try:
            p = fn(cfg, ctx)
            if p:
                out.append(str(p))
        except Exception as e:  # 单图失败不阻断整体
            import traceback
            print(f"[viz] {fn.__name__} 失败: {e}\n{traceback.format_exc()}")
    return out
