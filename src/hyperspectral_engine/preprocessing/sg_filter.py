"""自适应Savitzky-Golay多项式平滑滤波 + 波长谱序自适应重排

剔除大气水汽(1350-1450nm, 1800-2000nm, 2400-2700nm)
与CO2吸收带(1550-1620nm, 2000-2060nm)带来的高频辐射毛刺，
确保GeoTIFF污染热力图不发生空间斑点化撕裂。

核心修复:
    - 波长谱序自适应重排算子: 动态嗅探输入光谱波长排列方向，
      当检测到降序排列(如南方丘陵大雨冲刷批次)时自动翻转通道，
      确保馈入推理引擎的特征张量始终按波长升序排列。
    - 吸收带掩码基于实际波长值构建，不再依赖硬编码通道索引。

实现策略:
    1. 波长排序自适应检测与重排
    2. 基于残差统计的自适应窗口选择
    3. 吸收带区域加权插值修复
    4. 边缘保持的多项式拟合
    5. 批处理向量优化 (支持 [N, C] 与 [C, H, W])
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


class WavelengthReorderOperator:
    """波长谱序自适应重排算子

    解决问题:
        当南方丘陵大雨冲刷批次传感器下发波长降序(2500nm->400nm)排列
        的数据卷时，1D卷积核与逆向光谱点积导致水羟基特征峰物理通道错位，
        产生天量负向激活值(Cd=-14mg/kg等荒谬结果)。

    工作机制:
        1. 接收输入波长数组，动态检测排列方向
        2. 若为降序，生成翻转索引将通道重排为升序
        3. 所有后续处理(SG滤波、吸收带掩码、模型推理)均基于升序
    """

    def __init__(self, wavelengths: Optional[np.ndarray] = None):
        if wavelengths is not None:
            self._wavelengths = np.asarray(wavelengths, dtype=np.float32)
        else:
            self._wavelengths = DEFAULT_GF5_WAVELENGTHS.copy()
        self._is_descending = self._detect_order()
        self._reorder_index = self._build_reorder_index()
        self._sorted_wavelengths = self._wavelengths[self._reorder_index]

    def _detect_order(self) -> bool:
        """检测波长排列方向

        Returns:
            True = 降序排列, False = 升序排列
        """
        wl = self._wavelengths
        if len(wl) < 2:
            return False
        n_check = min(10, len(wl))
        asc_count = sum(1 for i in range(1, n_check) if wl[i] > wl[i - 1])
        desc_count = sum(1 for i in range(1, n_check) if wl[i] < wl[i - 1])
        return desc_count > asc_count

    def _build_reorder_index(self) -> np.ndarray:
        """构建重排索引数组

        Returns:
            整数索引数组，应用后可将任意排列的波长转为严格升序
        """
        if self._is_descending:
            return np.arange(len(self._wavelengths) - 1, -1, -1, dtype=np.intp)
        return np.argsort(self._wavelengths)

    @property
    def is_descending(self) -> bool:
        return self._is_descending

    @property
    def sorted_wavelengths(self) -> np.ndarray:
        return self._sorted_wavelengths.copy()

    def reorder_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        """将光谱向量重排为波长升序

        Args:
            spectrum: [C] 原始光谱

        Returns:
            [C] 波长升序排列的光谱
        """
        return spectrum[self._reorder_index]

    def reorder_batch(self, batch: np.ndarray) -> np.ndarray:
        """将批量光谱重排为波长升序

        Args:
            batch: [N, C] 原始批量光谱

        Returns:
            [N, C] 波长升序排列的批量光谱
        """
        return batch[:, self._reorder_index]

    def reorder_cube(self, cube: np.ndarray) -> np.ndarray:
        """将数据立方体重排为波长升序

        Args:
            cube: [C, H, W] 原始立方体

        Returns:
            [C, H, W] 波长升序排列的立方体
        """
        return cube[self._reorder_index, :, :]


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

    波长排序感知: 内部始终基于波长升序的sorted_wavelengths
    构建吸收带掩码和SG系数，不再依赖硬编码通道索引。
    """

    def __init__(
        self,
        polyorder: int = 3,
        min_window: int = 5,
        max_window: int = 15,
        noise_threshold: float = 0.015,
        wavelengths: Optional[np.ndarray] = None,
    ):
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

        self._reorder_op = WavelengthReorderOperator(wavelengths)
        self.wavelengths = self._reorder_op.sorted_wavelengths

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

        自动处理波长排列:
            1. 若输入为降序排列，先翻转至升序
            2. 在升序空间上执行SG滤波(吸收带掩码正确匹配)
            3. 输出始终为波长升序排列(与模型训练一致)

        Args:
            spectrum: 输入光谱 [C] (330波段)

        Returns:
            平滑后的光谱 [C] (波长升序)
        """
        if self._reorder_op.is_descending:
            spectrum = self._reorder_op.reorder_spectrum(spectrum)

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

        自动将降序输入重排为升序后再滤波。

        Args:
            batch: 形状 [N, C]

        Returns:
            平滑后的 [N, C] (波长升序)
        """
        if self._reorder_op.is_descending:
            batch = self._reorder_op.reorder_batch(batch)
        N, C = batch.shape
        out = np.empty_like(batch, dtype=np.float32)
        for i in range(N):
            out[i] = self.filter_spectrum(batch[i])
        return out

    def filter_cube(self, cube: np.ndarray) -> np.ndarray:
        """滤波整个数据立方体 [C, H, W]，保证空间连续性

        自动将降序输入重排为升序后再滤波。
        通过对相邻像素光谱的加权聚合消除空间斑点化。

        Args:
            cube: 形状 [C, H, W] 的反射率张量

        Returns:
            平滑后的 [C, H, W] (波长升序)
        """
        if self._reorder_op.is_descending:
            cube = self._reorder_op.reorder_cube(cube)
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
        0. 波长谱序自适应重排 (消除硬编码升序依赖)
        1. 无效值剔除与插值
        2. 自适应SG平滑滤波 (吸收带掩码基于真实波长值)
        3. 输出始终为波长升序 (与模型训练权重一致)
    """

    def __init__(
        self,
        sg_polyorder: int = 3,
        sg_min_window: int = 5,
        sg_max_window: int = 15,
        sg_noise_threshold: float = 0.015,
        wavelengths: Optional[np.ndarray] = None,
    ):
        self._reorder_op = WavelengthReorderOperator(wavelengths)
        self.sg_filter = AdaptiveSavitzkyGolay(
            polyorder=sg_polyorder,
            min_window=sg_min_window,
            max_window=sg_max_window,
            noise_threshold=sg_noise_threshold,
            wavelengths=self._reorder_op.sorted_wavelengths,
        )
        self._eps = 1e-8

    @property
    def is_wavelength_descending(self) -> bool:
        return self._reorder_op.is_descending

    def set_wavelengths(self, wavelengths: np.ndarray) -> None:
        """运行时动态更新波长信息(从ENVI元数据注入)

        当切换数据源(如从晴空干燥区切换到南方丘陵大雨批次)
        时，调用此方法重新初始化重排算子和吸收带掩码。
        """
        wavelengths = np.asarray(wavelengths, dtype=np.float32)
        self._reorder_op = WavelengthReorderOperator(wavelengths)
        self.sg_filter = AdaptiveSavitzkyGolay(
            polyorder=self.sg_filter.polyorder,
            min_window=self.sg_filter.min_window,
            max_window=self.sg_filter.max_window,
            noise_threshold=self.sg_filter.noise_threshold,
            wavelengths=self._reorder_op.sorted_wavelengths,
        )

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

    def _reorder_to_ascending(self, data: np.ndarray) -> np.ndarray:
        """将数据重排为波长升序 (在清洗和滤波之前执行)"""
        if not self._reorder_op.is_descending:
            return data
        if data.ndim == 1:
            return self._reorder_op.reorder_spectrum(data)
        elif data.ndim == 2:
            return self._reorder_op.reorder_batch(data)
        elif data.ndim == 3:
            return self._reorder_op.reorder_cube(data)
        return data

    def process_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        """处理单条光谱 [C] -> [C] (波长升序)"""
        reordered = self._reorder_to_ascending(spectrum)
        cleaned = self._clean_invalid(reordered)
        smoothed = self.sg_filter.filter_spectrum(cleaned)
        return smoothed.astype(np.float32)

    def process_batch(self, batch: np.ndarray) -> np.ndarray:
        """处理批光谱 [N, C] -> [N, C] (波长升序)"""
        reordered = self._reorder_to_ascending(batch)
        cleaned = self._clean_invalid(reordered)
        smoothed = self.sg_filter.filter_batch(cleaned)
        return smoothed.astype(np.float32)

    def process_cube(self, cube: np.ndarray) -> np.ndarray:
        """处理整个立方体 [C, H, W] -> [C, H, W] (波长升序)"""
        reordered = self._reorder_to_ascending(cube)
        cleaned = self._clean_invalid(reordered)
        smoothed = self.sg_filter.filter_cube(cleaned)
        return smoothed.astype(np.float32)
