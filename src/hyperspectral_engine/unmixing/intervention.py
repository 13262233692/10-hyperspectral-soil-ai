"""地表覆盖物纯度干预模块

耕地质量评价规程核心干预逻辑:
    1. 对多光谱图斑切片执行VCA无监督端元解混
    2. 对提取的端元逐个与内置标准"农用聚乙烯地膜"纯端元计算光谱角(SAM)
    3. 当最小SAM角跌破0.05弧度临界阈值时，判定该图斑被地膜白色污染高度遮蔽
    4. 立即静默阻断该图斑内所有重金属浓度深度反演
    5. 生成干预净荷: 鲜黄色斜十字警示网格 + 端元丰度对比图数据
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

from .vca import vca, fully_constrained_abundance
from .spectral_endmembers import (
    get_endmember,
    spectral_angle,
    spectral_angle_batch,
    best_matching_endmember,
)


MULCH_SAM_THRESHOLD = 0.05


@dataclass
class InterventionPayload:
    """干预净荷数据结构"""
    blocked: bool = False
    mulch_detected: bool = False
    mulch_sam_angle: float = np.pi
    mulch_abundance_mean: float = 0.0
    mulch_abundance_max: float = 0.0
    mulch_endmember_index: int = -1
    num_endmembers: int = 0
    endmember_labels: List[str] = field(default_factory=list)
    endmember_sam_angles: List[float] = field(default_factory=list)
    abundance_means: List[float] = field(default_factory=list)
    abundance_profile: Optional[np.ndarray] = None
    warning_mask: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "blocked": self.blocked,
            "mulch_detected": self.mulch_detected,
            "mulch_sam_angle": round(self.mulch_sam_angle, 6),
            "mulch_abundance_mean": round(float(self.mulch_abundance_mean), 6),
            "mulch_abundance_max": round(float(self.mulch_abundance_max), 6),
            "mulch_endmember_index": self.mulch_endmember_index,
            "num_endmembers": self.num_endmembers,
            "endmember_labels": self.endmember_labels,
            "endmember_sam_angles": [round(a, 6) for a in self.endmember_sam_angles],
            "abundance_means": [round(m, 6) for m in self.abundance_means],
        }
        if self.abundance_profile is not None:
            d["abundance_profile_shape"] = list(self.abundance_profile.shape)
        if self.warning_mask is not None:
            d["warning_mask_shape"] = list(self.warning_mask.shape)
        return d


class PurityInterventionModule:
    """地表覆盖物纯度干预模块

    基于VCA端元解混 + SAM光谱角匹配的地膜白色污染检测与反演阻断。

    Args:
        mulch_sam_threshold: 地膜SAM角度临界阈值(弧度), 默认0.05
        num_endmembers: VCA提取端元数量, 默认4
        min_pixels: 图斑最少像素数, 少于此数跳过解混, 默认16
    """

    def __init__(
        self,
        mulch_sam_threshold: float = MULCH_SAM_THRESHOLD,
        num_endmembers: int = 4,
        min_pixels: int = 16,
    ):
        self.mulch_sam_threshold = mulch_sam_threshold
        self.num_endmembers = num_endmembers
        self.min_pixels = min_pixels
        self._mulch_ref = get_endmember("polyethylene_mulch")

    def analyze_patch(
        self,
        patch_spectra: np.ndarray,
        patch_mask: Optional[np.ndarray] = None,
        patch_shape: Optional[Tuple[int, int]] = None,
    ) -> InterventionPayload:
        """对图斑执行纯度干预分析

        Args:
            patch_spectra: [L, N] 图斑内像素光谱矩阵(L=波段数, N=像素数)
            patch_mask: [H, W] 可选, 图斑在原始空间的布尔掩码
            patch_shape: (H, W) 图斑空间形状

        Returns:
            InterventionPayload 干预净荷
        """
        payload = InterventionPayload()

        L, N = patch_spectra.shape
        if N < self.min_pixels:
            payload.blocked = False
            return payload

        n_em = min(self.num_endmembers, N // 4, L - 1)
        if n_em < 2:
            return payload

        try:
            endmembers, indices = vca(patch_spectra, n_em)
            abundances = fully_constrained_abundance(patch_spectra, endmembers)
        except (np.linalg.LinAlgError, ValueError):
            return payload

        payload.num_endmembers = n_em
        payload.abundance_profile = abundances

        labels = []
        sam_angles = []
        abundance_means = []
        mulch_idx = -1
        mulch_angle = np.pi

        for i in range(n_em):
            em_i = endmembers[:, i]
            name, angle = best_matching_endmember(em_i)
            labels.append(name)
            sam_angles.append(angle)
            abundance_means.append(float(abundances[i, :].mean()))

            if name == "polyethylene_mulch" and angle < self.mulch_sam_threshold:
                if angle < mulch_angle:
                    mulch_angle = angle
                    mulch_idx = i

        payload.endmember_labels = labels
        payload.endmember_sam_angles = sam_angles
        payload.abundance_means = abundance_means

        if mulch_idx >= 0:
            payload.mulch_detected = True
            payload.mulch_sam_angle = mulch_angle
            payload.mulch_endmember_index = mulch_idx
            payload.mulch_abundance_mean = float(abundances[mulch_idx, :].mean())
            payload.mulch_abundance_max = float(abundances[mulch_idx, :].max())
            payload.blocked = True

            if patch_mask is not None and patch_shape is not None:
                payload.warning_mask = self._build_warning_mask(
                    patch_mask, patch_shape, abundances, mulch_idx
                )
        else:
            pixel_sam = spectral_angle_batch(patch_spectra.T, self._mulch_ref)
            n_close = int(np.sum(pixel_sam < self.mulch_sam_threshold))
            close_ratio = n_close / N if N > 0 else 0.0

            if close_ratio > 0.1:
                payload.mulch_detected = True
                payload.mulch_sam_angle = float(np.percentile(pixel_sam, 10))
                payload.mulch_endmember_index = -1
                payload.mulch_abundance_mean = close_ratio
                payload.mulch_abundance_max = float(pixel_sam.min())
                payload.blocked = True
                del pixel_sam

        del endmembers, indices, abundances

        return payload

    def analyze_pixel(
        self,
        spectrum: np.ndarray,
    ) -> Tuple[bool, float]:
        """快速单像素地膜光谱角检测(不需要VCA解混)

        Args:
            spectrum: [L] 单像素光谱向量

        Returns:
            (is_blocked, sam_angle)
        """
        angle = spectral_angle(spectrum, self._mulch_ref)
        return angle < self.mulch_sam_threshold, angle

    def _build_warning_mask(
        self,
        patch_mask: np.ndarray,
        patch_shape: Tuple[int, int],
        abundances: np.ndarray,
        mulch_idx: int,
    ) -> np.ndarray:
        """构建干预掩码: 地膜丰度超过均值的像素标记为1

        Args:
            patch_mask: [H, W] 原始图斑布尔掩码
            patch_shape: (H, W)
            abundances: [p, N]
            mulch_idx: 地膜端元索引

        Returns:
            [H, W] float32 干预掩码(0.0或1.0)
        """
        H, W = patch_shape
        mask = np.zeros((H, W), dtype=np.float32)

        mulch_ab = abundances[mulch_idx, :]
        threshold = float(mulch_ab.mean())
        flagged = mulch_ab >= threshold

        flat_mask = patch_mask.ravel()
        flagged_indices = np.where(flat_mask)[0]
        n_flag = min(len(flagged_indices), flagged.sum())
        flagged_true = np.where(flagged)[0][:n_flag]

        for idx in flagged_true:
            if idx < len(flat_mask) and flat_mask[idx]:
                r, c = divmod(int(idx), W)
                if 0 <= r < H and 0 <= c < W:
                    mask[r, c] = 1.0

        return mask

    @staticmethod
    def build_crosshatch_overlay(
        mask: np.ndarray,
        spacing: int = 4,
        line_width: int = 1,
    ) -> np.ndarray:
        """生成鲜黄色斜十字交叉警示网格RGBA叠加层

        Args:
            mask: [H, W] 干预掩码(>0的区域需要覆盖)
            spacing: 十字线间距(像素), 默认4
            line_width: 线宽(像素), 默认1

        Returns:
            [H, W, 4] RGBA叠加层, 鲜黄色(R=255,G=255,B=0)高反差
        """
        H, W = mask.shape
        overlay = np.zeros((H, W, 4), dtype=np.uint8)

        rows, cols = np.where(mask > 0)
        if len(rows) == 0:
            return overlay

        diag1 = (rows - cols) % spacing < line_width
        diag2 = (rows + cols) % spacing < line_width
        crosshatch = diag1 | diag2

        overlay[rows[crosshatch], cols[crosshatch], 0] = 255
        overlay[rows[crosshatch], cols[crosshatch], 1] = 255
        overlay[rows[crosshatch], cols[crosshatch], 2] = 0
        overlay[rows[crosshatch], cols[crosshatch], 3] = 200

        return overlay
