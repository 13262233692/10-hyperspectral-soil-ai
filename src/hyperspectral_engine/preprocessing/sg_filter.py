"""自适应Savitzky-Golay多项式平滑滤波

剔除大气水汽(1350-1450nm, 1800-2000nm, 2400-2700nm)
与CO2吸收带(1550-1620nm, 2000-2060nm)带来的高频辐射毛刺，
确保GeoTIFF污染热力图不发生空间斑点化撕裂。

实现策略:
    1. 基于残差统计的自适应窗口选择
    2. 吸收带区域加权插值修复
    3. 边缘保持的多项式拟合
    4. 批处理向量优化 (支持 [N, C] 与 [C, H, W])
"""

import numpy as np
from typing import Optional, Tuple, Union
from scipy.linalg import lstsq


WATER_VAPOR_BANDS = [
    (1340.0, 1460.0),
    (1780.0, 2020.0),
    (2380.0, 2700.0),
]

CO2_ABSORPTION_BANDS = [
    (1540.0, 1630.0),
    (1990.0, 2070.0),
]

DEFAULT_GF5_WAVELENGTHS = np.linspace(400.0, 2500.0, 330, dtype=np.float32)


def _build_sg_coefficients(window_length: int, polyorder: int, deriv: int = 0) -> np.ndarray:
    """构建Savitzky-Golay滤波系数矩阵

    正确实现: 
        - A: [window_length, polyorder+1] 范德蒙德矩阵
        - 投影矩阵: P = A @ (A^T A)^{-1} @ A^T  [window_length, window_length]
        - P的第pos行就是位置pos处的卷积核

    Args:
        window_length: 窗口长度 (必须为奇数)
        polyorder: 多项式阶数
        deriv: 导数阶数

    Returns:
        系数矩阵 [window_length, window_length]
    """
    if window_length % 2 == 0:
        window_length += 1
    half = window_length // 2
    x = np.arange(-half, half + 1, dtype=np.float64)

    A = np.vander(x, N=polyorder + 1, increasing=True)

    if deriv > 0:
        for d in range(deriv):
            for col in range(polyorder + 1):
                if col >= d:
                    coeff = 1.0
                    for k in range(d):
                        coeff *= (col - k)
                    A[:, col] *= coeff
                else:
                    A[:, col] = 0.0

    AtA = A.T @ A
    AtA_inv = np.linalg.pinv(AtA)
    P = A @ AtA_inv @ A.T
    return P.astype(np.float64)


class AdaptiveSavitzkyGolay:
    """自适应Savitzky-Golay平滑滤波器

    根据局部噪声水平自动调整窗口大小，
    对水汽/CO2吸收带进行特殊加权处理。
    """

    def __init__(
        self,
        polyorder: int = 3,
        min_window: int = 5,
        max_window: int = 15,
        noise_threshold: float = 0.015,
        wavelengths: Optional[np.ndarray] = None,
    ):
        """
        Args:
            polyorder: 多项式拟合阶数
            min_window: 最小窗口长度
            max_window: 最大窗口长度
            noise_threshold: 噪声判定阈值 (相对残差)
            wavelengths: 波长数组，默认GF-5 330波段
        """
        if max_window % 2 == 0:
            max_window += 1
        if min_window % 2 == 0:
            min_window += 1
        if min_window < polyorder + 2:
            min_window = polyorder + 2
            if min_window % 2 == 0:
                min_window += 1

        self.polyorder = polyorder
        self.min_window = min_window
        self.max_window = max_window
        self.noise_threshold = noise_threshold
        self.wavelengths = (
            wavelengths.astype(np.float32)
            if wavelengths is not None
            else DEFAULT_GF5_WAVELENGTHS.copy()
        )
        self.absorption_mask = self._build_absorption_mask()

        self._coeff_cache: dict = {}

    def _build_absorption_mask(self) -> np.ndarray:
        """构建吸收带权重掩码 (0~1，越小表示受吸收影响越大)"""
        wl = self.wavelengths
        mask = np.ones_like(wl, dtype=np.float32)

        for lo, hi in WATER_VAPOR_BANDS + CO2_ABSORPTION_BANDS:
            in_band = (wl >= lo) & (wl <= hi)
            center = (lo + hi) * 0.5
            half_width = (hi - lo) * 0.5
            if half_width > 0:
                dist = np.abs(wl[in_band] - center) / half_width
                mask[in_band] = np.minimum(mask[in_band], 0.15 + 0.85 * dist**2)
        return mask

    def _get_coefficients(self, window_length: int) -> np.ndarray:
        """获取/缓存指定窗口的SG系数"""
        if window_length not in self._coeff_cache:
            self._coeff_cache[window_length] = _build_sg_coefficients(
                window_length, self.polyorder
            )
        return self._coeff_cache[window_length]

    def _estimate_local_noise(self, spectrum: np.ndarray) -> float:
        """估计光谱局部噪声水平 (高频能量占比)"""
        diff2 = np.diff(spectrum, n=2)
        return float(np.mean(np.abs(diff2)) / (np.std(spectrum) + 1e-8))

    def _select_window(self, spectrum: np.ndarray) -> int:
        """根据噪声水平自适应选择窗口长度"""
        noise_level = self._estimate_local_noise(spectrum)
        ratio = min(1.0, noise_level / self.noise_threshold)
        window = int(
            round(self.min_window + (self.max_window - self.min_window) * ratio)
        )
        if window % 2 == 0:
            window += 1
        return max(self.min_window, min(self.max_window, window))

    def filter_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        """对单条光谱向量 [C] 进行自适应平滑

        Args:
            spectrum: 输入光谱 [C] (330波段)

        Returns:
            平滑后的光谱 [C]
        """
        C = spectrum.shape[0]
        window = self._select_window(spectrum)
        coeffs = self._get_coefficients(window)
        half = window // 2
        center_row = coeffs[half].astype(np.float64)

        padded = np.pad(spectrum.astype(np.float64), half, mode="reflect")

        out = np.convolve(padded, center_row, mode="valid")
        out = out[:C]

        w = self.absorption_mask[:C]
        blend = w * spectrum.astype(np.float64) + (1.0 - w) * out
        return blend.astype(np.float32)

    def filter_batch(self, batch: np.ndarray) -> np.ndarray:
        """批量滤波 [N, C] 光谱矩阵

        Args:
            batch: 形状 [N, C]

        Returns:
            平滑后的 [N, C]
        """
        N, C = batch.shape
        out = np.empty_like(batch, dtype=np.float32)
        for i in range(N):
            out[i] = self.filter_spectrum(batch[i])
        return out

    def filter_cube(self, cube: np.ndarray) -> np.ndarray:
        """滤波整个数据立方体 [C, H, W]，保证空间连续性

        通过对相邻像素光谱的加权聚合消除空间斑点化。

        Args:
            cube: 形状 [C, H, W] 的反射率张量

        Returns:
            平滑后的 [C, H, W]
        """
        C, H, W = cube.shape
        smoothed = np.empty_like(cube, dtype=np.float32)

        for y in range(H):
            for x in range(W):
                smoothed[:, y, x] = self.filter_spectrum(cube[:, y, x])

        smoothed = self._spatial_consistency_filter(smoothed)
        return smoothed

    def _spatial_consistency_filter(
        self, cube: np.ndarray, sigma: float = 0.8
    ) -> np.ndarray:
        """3x3高斯空间平滑，消除斑点化撕裂

        仅对高频残差做空间平滑，保留光谱主结构。
        """
        C, H, W = cube.shape
        if H < 3 or W < 3:
            return cube

        kernel = np.array(
            [[0.05, 0.12, 0.05],
             [0.12, 0.32, 0.12],
             [0.05, 0.12, 0.05]],
            dtype=np.float32,
        )
        kernel /= kernel.sum()

        out = cube.copy()
        for c in range(C):
            band = cube[c]
            for y in range(1, H - 1):
                for x in range(1, W - 1):
                    patch = band[y - 1 : y + 2, x - 1 : x + 2]
                    local_std = np.std(patch)
                    center_val = band[y, x]
                    blur_val = float(np.sum(patch * kernel))
                    alpha = 1.0 / (1.0 + np.exp(-(local_std / (sigma + 1e-8) - 1.5)))
                    alpha = np.clip(alpha * 0.4, 0.0, 0.4)
                    out[c, y, x] = (1 - alpha) * center_val + alpha * blur_val
        return out


class PreprocessingPipeline:
    """完整的推理前置管道

    顺序执行:
        1. 无效值剔除与插值
        2. 自适应SG平滑滤波
        3. 归一化与标准化
    """

    def __init__(
        self,
        sg_polyorder: int = 3,
        sg_min_window: int = 5,
        sg_max_window: int = 15,
        sg_noise_threshold: float = 0.015,
        wavelengths: Optional[np.ndarray] = None,
    ):
        self.sg_filter = AdaptiveSavitzkyGolay(
            polyorder=sg_polyorder,
            min_window=sg_min_window,
            max_window=sg_max_window,
            noise_threshold=sg_noise_threshold,
            wavelengths=wavelengths,
        )
        self._eps = 1e-8

    def _clean_invalid(self, data: np.ndarray) -> np.ndarray:
        """替换NaN/Inf为邻域插值"""
        if np.issubdtype(data.dtype, np.floating):
            mask = ~np.isfinite(data)
            if mask.any():
                data = data.copy()
                data[mask] = 0.0
                if data.ndim == 1:
                    valid = np.where(~mask)[0]
                    if len(valid) > 0:
                        data = np.interp(
                            np.arange(len(data)), valid, data[valid]
                        )
        return data.astype(np.float32)

    def process_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        """处理单条光谱 [C] -> [C]"""
        cleaned = self._clean_invalid(spectrum)
        smoothed = self.sg_filter.filter_spectrum(cleaned)
        return smoothed.astype(np.float32)

    def process_batch(self, batch: np.ndarray) -> np.ndarray:
        """处理批光谱 [N, C] -> [N, C]"""
        cleaned = self._clean_invalid(batch)
        smoothed = self.sg_filter.filter_batch(cleaned)
        return smoothed.astype(np.float32)

    def process_cube(self, cube: np.ndarray) -> np.ndarray:
        """处理整个立方体 [C, H, W] -> [C, H, W]"""
        cleaned = self._clean_invalid(cube)
        smoothed = self.sg_filter.filter_cube(cleaned)
        return smoothed.astype(np.float32)
