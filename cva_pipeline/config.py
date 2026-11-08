# -*- coding: utf-8 -*-
"""全局配置：dataclass + JSON 读写。所有可调参数集中于此，含注释默认值。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# 演示场景的土地覆被类别体系（代码 -> 名称）。
# 若接入真实数据，请按训练样本的类别体系修改此处（1~N 连续编码，0 保留为无数据）。
DEFAULT_CLASS_NAMES = {
    1: "水体",
    2: "林地",
    3: "草地",
    4: "耕地",
    5: "建设用地",
    6: "裸地",
}


@dataclass
class Config:
    # ---------- 数据输入 ----------
    manifest_path: str = "data/sim_scene/manifest.json"  # 时序影像清单（见 io_utils.Manifest）
    bands: tuple = ("B02", "B03", "B04", "B08")          # 参与合成的波段（Sentinel-2 10m 四波段）
    year1: int = 2023                                    # 基期年份
    year2: int = 2024                                    # 目标年份
    min_valid_obs: int = 4                               # 年内最少有效观测次数，低于此值像元置 nodata
    composite_doy_range: Optional[tuple] = None          # 合成 DOY 窗口，如 (120,275)=生长季；None=全年
    nodata: float = -9999.0                              # 输出栅格 nodata 值

    # ---------- 去云 ----------
    use_scl: bool = True                     # 优先使用 SCL 波段做云/影掩膜
    scl_invalid_codes: tuple = (0, 1, 3, 8, 9, 10, 11)  # SCL 无效类：nodata/饱和/云影/中高概率云/卷云/雪
    cloud_dilate_px: int = 2                 # 云掩膜形态学膨胀像素数（防云缘薄雾）
    fallback_cloud_thr: float = 0.55         # 无 SCL 时规则云分数阈值（0~1）

    # ---------- 相对辐射归一化 ----------
    normalize: bool = True                   # 将 year2 合成归一化到 year1 辐射域（PIF 迭代回归）
    norm_iter: int = 3                       # 迭代回归次数（逐步收紧伪不变特征）
    norm_sigma: float = 2.0                  # 残差剔除窗口（±sigma）

    # ---------- CVA 与阈值 ----------
    band_weights: Optional[tuple] = None     # 各波段权重（None=等权）；如加重 NIR 可传 (1,1,1,1.5)
    threshold_method: str = "otsu"           # otsu | percentile | meanstd
    threshold_percentile: float = 99.0       # percentile 法的分位数
    threshold_min: float = 0.03              # 阈值下限保护（反射率域，防止过检）

    # ---------- 伪变化过滤 ----------
    min_patch_px: int = 6                    # R1 面积硬阈值：图斑最少像元数（10m 下 6px≈600m²）
    strong_patch_px: int = 60                # 大图斑豁免线：≥ 此面积跳过 R2/R3/R4 软约束
    min_coherence: float = 0.55              # R2 空间一致性：邻域同类占比下限（R3 实心宽度、R4 纹理另判）
    texture_max_sigma: float = 0.06          # R3 纹理约束：图斑内 NIR 局部标准差均值上限
    texture_window: int = 5                  # 局部标准差窗口大小（奇数）
    morph_close_px: int = 1                  # 标记前二值闭运算迭代数（弥合 1px 碎缝）
    morph_open_px: int = 1                   # 标记前二值开运算迭代数（剥除条带状伪变化）

    # ---------- 分类（转移矩阵用） ----------
    samples_csv: Optional[str] = None        # 训练样本 CSV（year,col,row,label）；None 则跳过分类
    rf_trees: int = 200                      # 随机森林树数
    smooth_class_px: int = 1                 # 分类图众数滤波迭代（1=3×3 一轮；0 关闭）

    # ---------- 类别体系与输出 ----------
    class_names: dict = field(default_factory=lambda: dict(DEFAULT_CLASS_NAMES))
    out_dir: str = "outputs"
    pixel_area_m2: float = 100.0             # 像元面积（10m 分辨率）
    scene_crs: str = "EPSG:32650"            # 输出矢量/栅格坐标系（覆盖 manifest 中的声明）
    make_figures: bool = True                # 是否生成示例结果图
    write_intermediate: bool = True          # 是否写中间栅格（raw 掩膜、幅度图等）

    # ---------- 路径派生 ----------
    @property
    def fig_dir(self) -> Path:
        return Path(self.out_dir) / "figures"

    @property
    def stats_dir(self) -> Path:
        return Path(self.out_dir) / "stats"

    def ensure_dirs(self) -> None:
        for d in (Path(self.out_dir), self.fig_dir, self.stats_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ---------- 序列化 ----------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["bands"] = list(self.bands)
        if d.get("band_weights"):
            d["band_weights"] = list(self.band_weights)
        if d.get("composite_doy_range"):
            d["composite_doy_range"] = list(self.composite_doy_range)
        d["scl_invalid_codes"] = list(self.scl_invalid_codes)
        return d

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "Config":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f for f in cls.__dataclass_fields__}
        def conv(k, v):
            if v is None:
                return v
            if k in ("bands", "scl_invalid_codes", "band_weights", "composite_doy_range"):
                return tuple(v)
            if k == "class_names":          # JSON 键只能为字符串，转回 int 编码
                return {int(kk): vv for kk, vv in v.items()}
            return v
        return cls(**{k: conv(k, v) for k, v in d.items() if k in known})
