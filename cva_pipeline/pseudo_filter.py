# -*- coding: utf-8 -*-
"""伪变化过滤：连通域标记(自研 Union-Find) + 面积阈值 + 空间一致性 + 纹理约束。

伪变化的来源
------------
1. 物候噪声：两期合成所选时相的物候期不完全同步（如双季稻移栽期早晚差异）；
2. 大气/辐射残留：薄雾、气溶胶未除净造成的整体性小幅偏移；
3. 配准误差：地物边缘处的错位产生细碎"变化"；
4. 传感器噪声：孤立的椒盐状高幅度像元。

这些伪变化的共同特征：**图斑小、形状破碎（空间一致性低）、发生在纹理
剧变处或孤立出现**。真实土地利用变化则是"成片、成块"的（一个地块要么
整体变化要么不变）。过滤即围绕这一先验设计三条规则：

    R1 面积硬阈值   : 图斑像元数 ≥ min_patch_px（<6px≈600m² 直接剔除）
    R2 空间一致性   : 图斑"邻域同类占比"（8 邻域中同图斑像元的比例之均值）
                      ≥ min_coherence，大图斑（≥ strong_patch_px）豁免
    R3 最小实心宽度 : 图斑须含 3×3 实心核（一次腐蚀后非空）—— 条带状伪变化
                      （配准残差、细边缘条带）的典型形态被直接剔除
    R4 纹理约束     : 图斑内 NIR 年度合成的局部标准差均值 ≤ texture_max_sigma，
                      大图斑豁免 —— 在高纹理边缘处的小图斑多为配准/物候噪声

全部算法（连通域、形态学、Sobel、局部标准差）手写实现，核心不借助任何
现成封装，便于面试时逐行讲解。
"""
from __future__ import annotations

import numpy as np


# ================================================================ 形态学 ==
def _shift_or(mask: np.ndarray, dx: int, dy: int, op) -> np.ndarray:
    """带边界处理的邻域移位聚合（膨胀=OR，腐蚀=AND）。mask: (H,W) bool。"""
    H, W = mask.shape
    acc = mask.copy()
    for ddx in range(-dx, dx + 1):
        for ddy in range(-dy, dy + 1):
            if ddx == 0 and ddy == 0:
                continue
            shifted = np.full_like(mask, False if op is np.logical_or else True)
            # 目标(i,j) ← 源(i-ddx, j-ddy)，越界视作背景
            ys = slice(max(ddx, 0), H + min(ddx, 0))
            xs = slice(max(ddy, 0), W + min(ddy, 0))
            ys_src = slice(max(-ddx, 0), H + min(-ddx, 0))
            xs_src = slice(max(-ddy, 0), W + min(-ddy, 0))
            shifted[ys, xs] = mask[ys_src, xs_src]
            acc = op(acc, shifted)
    return acc


def binary_dilation2(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    out = mask.copy()
    for _ in range(max(0, iterations)):
        out = _shift_or(out, 1, 1, np.logical_or)
    return out


def binary_erosion2(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    out = mask.copy()
    for _ in range(max(0, iterations)):
        out = _shift_or(out, 1, 1, np.logical_and)
    return out


def binary_opening(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    return binary_dilation2(binary_erosion2(mask, iterations), iterations)


def binary_closing(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    return binary_erosion2(binary_dilation2(mask, iterations), iterations)


# ============================================================ 连通域标记 ==
def label_patches(mask: np.ndarray, connectivity: int = 8) -> tuple[np.ndarray, int]:
    """两遍扫描连通域标记（8/4 连通，Union-Find 并查集，路径压缩）。

    返回 (labels, n)：labels 为 int32 (H,W)，1..n 为图斑编号，0 为背景。
    时间 O(N·α)，空间 O(N)；纯 Python 循环对 10⁵~10⁶ 像元量级足够快。
    """
    mask = np.asarray(mask, bool)
    H, W = mask.shape
    labels = np.zeros((H, W), np.int32)
    parent = [0]  # parent[i]：并查集父节点；0 号保留给背景

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:          # 路径压缩
            parent[x], x = root, parent[x]
        return root

    offs = [(0, -1), (-1, 0), (-1, -1), (-1, 1)] if connectivity == 8 else [(0, -1), (-1, 0)]

    nxt = 1
    for i in range(H):
        row = mask[i]
        for j in range(W):
            if not row[j]:
                continue
            neigh = []
            for di, dj in offs:
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W and labels[ni, nj]:
                    neigh.append(labels[ni, nj])
            if not neigh:
                parent.append(nxt)
                labels[i, j] = nxt
                nxt += 1
            else:
                roots = [find(n) for n in neigh]
                r = min(roots)
                labels[i, j] = r
                for nr in roots:          # 等价类合并
                    parent[nr] = r

    # 第二遍：解析根 → 连续编号
    remap: dict[int, int] = {}
    out = np.zeros_like(labels)
    for i in range(H):
        li = labels[i]
        for j in range(W):
            v = li[j]
            if v:
                r = find(v)
                idx = remap.setdefault(r, len(remap) + 1)
                out[i, j] = idx
    return out, len(remap)


# ================================================================ 纹理 ====
def box_mean(a: np.ndarray, k: int) -> np.ndarray:
    """盒式均值滤波（积分图实现，O(N)，k 为奇数）。NaN 视为 0 并返回权重图。"""
    a = np.asarray(a, np.float64)
    if k % 2 == 0:
        k += 1
    p = k // 2
    valid = np.isfinite(a)
    a0 = np.where(valid, a, 0.0)
    ap = np.pad(a0, p, mode="edge")
    cs = np.zeros((ap.shape[0] + 1, ap.shape[1] + 1))
    cs[1:, 1:] = ap.cumsum(0).cumsum(1)
    H, W = a.shape
    s = cs[k:k + H, k:k + W] - cs[0:H, k:k + W] - cs[k:k + H, 0:W] + cs[0:H, 0:W]
    return s / float(k * k)


def local_std(a: np.ndarray, k: int = 5) -> np.ndarray:
    """局部标准差：sqrt(E[x²] - E[x]²)（盒窗）。NaN 按缺失处理（以 0 参与统计）。"""
    m1 = box_mean(a, k)
    m2 = box_mean(np.where(np.isfinite(a), a, 0.0) ** 2, k)
    return np.sqrt(np.clip(m2 - m1 ** 2, 0.0, None))


def sobel_magnitude(a: np.ndarray) -> np.ndarray:
    """Sobel 梯度幅值（3×3 卷积手写切片实现）。"""
    a = np.where(np.isfinite(a), a, 0.0)
    ap = np.pad(a, 1, mode="edge")
    gx = (ap[0:-2, 2:] + 2 * ap[1:-1, 2:] + ap[2:, 2:]) - \
         (ap[0:-2, 0:-2] + 2 * ap[1:-1, 0:-2] + ap[2:, 0:-2])
    gy = (ap[2:, 0:-2] + 2 * ap[2:, 1:-1] + ap[2:, 2:]) - \
         (ap[0:-2, 0:-2] + 2 * ap[0:-2, 1:-1] + ap[0:-2, 2:])
    return np.sqrt(gx ** 2 + gy ** 2)


# ============================================================ 图斑统计 ====
def patch_statistics(labels: np.ndarray, n_patches: int, magnitude: np.ndarray,
                     texture: np.ndarray) -> dict:
    """向量化统计每个图斑的属性（np.bincount 一次遍历）。

    coherence（空间一致性）定义为图斑内像元的"8 邻域同类占比"之均值：
    孤立点为 0，细线约 0.2~0.3，成片图斑接近 0.7~1.0 —— 与文献中的
    spatial coherence 滤波一致，小而破碎的伪变化被显著区分。
    """
    H, W = labels.shape
    flat = labels.ravel()
    ids = np.arange(1, n_patches + 1)

    area = np.bincount(flat, minlength=n_patches + 1)[1:].astype(np.int64)

    mflat = magnitude.ravel()
    mag_sum = np.bincount(flat, weights=np.nan_to_num(mflat), minlength=n_patches + 1)[1:]
    mag_cnt = np.bincount(flat, weights=np.isfinite(mflat).astype(np.float64), minlength=n_patches + 1)[1:]
    mag_mean = np.where(mag_cnt > 0, mag_sum / np.maximum(mag_cnt, 1), np.nan)

    tflat = texture.ravel()
    tex_sum = np.bincount(flat, weights=np.nan_to_num(tflat), minlength=n_patches + 1)[1:]
    tex_mean = np.where(area > 0, tex_sum / np.maximum(area, 1), np.nan)

    # 邻域同类占比：8 邻域中同属本图斑的邻居数 / 8（越界按 0 计）
    nb = np.zeros((H, W), np.float32)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            sh = np.zeros((H, W), np.int32)  # 越界为 0（背景），不会误判同类
            ys = slice(max(dx, 0), H + min(dx, 0))
            xs = slice(max(dy, 0), W + min(dy, 0))
            ys_s = slice(max(-dx, 0), H + min(-dx, 0))
            xs_s = slice(max(-dy, 0), W + min(-dy, 0))
            sh[ys, xs] = labels[ys_s, xs_s]
            nb += ((sh == labels) & (labels > 0)).astype(np.float32)
    nb_frac = nb / 8.0
    nb_sum = np.bincount(flat, weights=(nb_frac.ravel() * (flat > 0)),
                         minlength=n_patches + 1)[1:]
    coherence = np.where(area > 0, nb_sum / np.maximum(area, 1), 0.0)

    # 3×3 实心核像元数：8 邻域全部同标签的像元数（=一次腐蚀后的剩余量）
    solid_cnt = np.bincount(flat, weights=((nb.ravel() >= 7.5) & (flat > 0)).astype(np.float64),
                            minlength=n_patches + 1)[1:].astype(np.int64)

    return dict(
        ids=ids, area=area, mag_mean=mag_mean, tex_mean=tex_mean, coherence=coherence,
        solid_cnt=solid_cnt,
    )


def filter_patches(stats: dict, cfg) -> tuple[np.ndarray, list[dict]]:
    """四条规则逐图斑判决。返回 (keep_ids, 规则执行报告)。

    判定顺序（依次淘汰，记录每条规则剔除量）：
        R1 面积硬阈值 → R2 空间一致性 → R3 最小实心宽度 → R4 纹理约束
    R2/R4 对大图斑（面积 ≥ strong_patch_px）豁免 —— 大面积真实变化即便
    发生在高纹理/离散区域也不该被误杀；R3（形状规则）不豁免，条带状
    伪变化无论多长都不可信。
    """
    area, coh, tex, solid = (stats["area"], stats["coherence"],
                             stats["tex_mean"], stats["solid_cnt"])
    n = len(area)
    alive = np.ones(n, bool)
    report = []

    r1 = area < cfg.min_patch_px
    report.append(dict(rule="R1_面积阈值", desc=f"图斑 < {cfg.min_patch_px} px",
                       removed=int((alive & r1).sum()), removed_px=int(area[alive & r1].sum())))
    alive &= ~r1

    weak_coh = ~np.isfinite(coh) | (coh < cfg.min_coherence)
    r2 = alive & weak_coh & (area < cfg.strong_patch_px)
    report.append(dict(rule="R2_空间一致性", desc=f"邻域同类占比 < {cfg.min_coherence} 且 面积 < {cfg.strong_patch_px}px",
                       removed=int(r2.sum()), removed_px=int(area[r2].sum())))
    alive &= ~r2

    # R3 最小实心宽度：图斑必须含 3×3 实心核（一次腐蚀后非空）。
    # 注意本条不受大图斑豁免 —— 配准残差产生的 1~2px 长条带面积可以很大
    # （1px×300px=300px），但在 10m 分辨率下宽度 <3px 的"变化"不可能是
    # 可信的地类转换，条带状伪变化一律剔除。
    r3 = alive & (solid == 0)
    report.append(dict(rule="R3_最小实心宽度", desc="无 3×3 实心核（条带状，不豁免）",
                       removed=int(r3.sum()), removed_px=int(area[r3].sum())))
    alive &= ~r3

    bad_tex = ~np.isfinite(tex) | (tex > cfg.texture_max_sigma)
    r4 = alive & bad_tex & (area < cfg.strong_patch_px)
    report.append(dict(rule="R4_纹理约束", desc=f"局部σ > {cfg.texture_max_sigma} 且 面积 < {cfg.strong_patch_px}px",
                       removed=int(r4.sum()), removed_px=int(area[r4].sum())))
    alive &= ~r4

    return stats["ids"][alive], report


def apply_filter(labels: np.ndarray, keep_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """按保留名单重建过滤后的掩膜与标签（1..k 重编号）。"""
    keep = np.isin(labels, keep_ids)
    new_labels = np.zeros_like(labels)
    for new_id, old_id in enumerate(keep_ids, start=1):
        new_labels[labels == old_id] = new_id
    return keep, new_labels
