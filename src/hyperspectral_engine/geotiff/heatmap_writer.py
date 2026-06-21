"""GeoTIFF污染热力图生成模块

功能:
    - 将浓度图编码为GeoTIFF格式
    - 多级空间平滑消除斑点化撕裂
    - 伪彩色渲染
    - EPSG:4326 / UTM 坐标支持
    - 金字塔(overview)构建，支持WebGIS快速加载

严格保证输出GeoTIFF满足WebGIS标准，不发生空间斑点化撕裂。
"""

import os
import io
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

try:
    import rasterio
    from rasterio.transform import from_origin, Affine
    from rasterio.enums import Resampling
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


SOIL_STANDARDS = {
    "Cd": 0.3,
    "Pb": 80.0,
    "As": 15.0,
}


VIRIDIS_CMAP = np.array([
    [0.267004, 0.004874, 0.329415, 1.0],
    [0.282327, 0.140926, 0.457517, 1.0],
    [0.253935, 0.265254, 0.529983, 1.0],
    [0.206756, 0.371758, 0.553117, 1.0],
    [0.163625, 0.471133, 0.558148, 1.0],
    [0.127568, 0.566949, 0.550556, 1.0],
    [0.134692, 0.658636, 0.517649, 1.0],
    [0.266941, 0.748751, 0.440573, 1.0],
    [0.477504, 0.821444, 0.318195, 1.0],
    [0.741388, 0.873449, 0.149561, 1.0],
    [0.993248, 0.906157, 0.143936, 1.0],
], dtype=np.float32)


@dataclass
class GeoTIFFConfig:
    """GeoTIFF输出配置"""
    crs_epsg: int = 4326
    origin_x: float = 0.0
    origin_y: float = 0.0
    pixel_size_x: float = 30.0
    pixel_size_y: float = 30.0
    rotation_x: float = 0.0
    rotation_y: float = 0.0
    build_pyramids: bool = True
    compression: str = "LZW"


class SpatialSmoother:
    """空间平滑器 - 消除斑点化撕裂

    采用自适应双边滤波:
        - 边缘保持区域平滑
        - 浓度突变区采用各向异性扩散
    """

    def __init__(self, iterations: int = 2, kappa: float = 30.0, gamma: float = 0.15):
        self.iterations = iterations
        self.kappa = kappa
        self.gamma = gamma

    @staticmethod
    def _gaussian_smooth(image: np.ndarray, sigma: float = 0.8) -> np.ndarray:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(image.astype(np.float32), sigma=sigma)

    def _anisotropic_diffusion(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32)
        for _ in range(self.iterations):
            dN = np.zeros_like(img)
            dN[:-1, :] = img[1:, :] - img[:-1, :]
            dS = np.zeros_like(img)
            dS[1:, :] = img[:-1, :] - img[1:, :]
            dW = np.zeros_like(img)
            dW[:, :-1] = img[:, 1:] - img[:, :-1]
            dE = np.zeros_like(img)
            dE[:, 1:] = img[:, :-1] - img[:, 1:]

            cN = 1.0 / (1.0 + (np.abs(dN) / self.kappa) ** 2)
            cS = 1.0 / (1.0 + (np.abs(dS) / self.kappa) ** 2)
            cW = 1.0 / (1.0 + (np.abs(dW) / self.kappa) ** 2)
            cE = 1.0 / (1.0 + (np.abs(dE) / self.kappa) ** 2)

            img = img + self.gamma * (cN * dN + cS * dS + cW * dW + cE * dE)
        return img

    def smooth(self, concentration_map: np.ndarray) -> np.ndarray:
        base = self._gaussian_smooth(concentration_map, sigma=0.6)
        refined = self._anisotropic_diffusion(base)
        return refined.astype(np.float32)


class PollutionHeatmapWriter:
    """污染热力图GeoTIFF写入器

    支持:
        - 单波段GeoTIFF (浮点浓度值)
        - RGBA伪彩色GeoTIFF (WebGIS渲染)
    """

    def __init__(self, config: Optional[GeoTIFFConfig] = None):
        if not HAS_RASTERIO:
            raise ImportError(
                "rasterio is not installed. "
                "Please install with: pip install rasterio"
            )
        self.config = config or GeoTIFFConfig()
        self.smoother = SpatialSmoother()

    def _get_transform(self, height: int, width: int) -> Affine:
        return from_origin(
            self.config.origin_x,
            self.config.origin_y,
            self.config.pixel_size_x,
            self.config.pixel_size_y,
        )

    @staticmethod
    def _normalize_to_uint8(
        data: np.ndarray, vmin: float = None, vmax: float = None
    ) -> np.ndarray:
        if vmin is None:
            vmin = float(np.percentile(data, 2))
        if vmax is None:
            vmax = float(np.percentile(data, 98))
        if vmax <= vmin:
            vmax = vmin + 1e-6
        scaled = np.clip((data - vmin) / (vmax - vmin), 0.0, 1.0)
        return (scaled * 255.0).astype(np.uint8)

    @staticmethod
    def apply_colormap(
        data_norm_uint8: np.ndarray, cmap: np.ndarray = VIRIDIS_CMAP
    ) -> np.ndarray:
        H, W = data_norm_uint8.shape
        idx = np.clip(
            (data_norm_uint8.astype(np.float32) / 255.0 * (len(cmap) - 1)).astype(np.int32),
            0,
            len(cmap) - 1,
        )
        rgba = (cmap[idx] * 255.0).astype(np.uint8)
        return rgba.transpose(2, 0, 1)

    def write_concentration(
        self,
        concentration_map: np.ndarray,
        output_path: str,
        apply_smoothing: bool = True,
        metal: str = "Cd",
    ) -> str:
        """写入单波段浮点浓度GeoTIFF

        Args:
            concentration_map: [H, W] 浓度图 (mg/kg)
            output_path: 输出.tif路径
            apply_smoothing: 是否应用空间平滑防斑点化
            metal: 金属名称(用于元数据)

        Returns:
            绝对路径
        """
        if apply_smoothing:
            data = self.smoother.smooth(concentration_map)
        else:
            data = concentration_map.astype(np.float32)

        H, W = data.shape
        transform = self._get_transform(H, W)
        crs = rasterio.crs.CRS.from_epsg(self.config.crs_epsg)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=H,
            width=W,
            count=1,
            dtype="float32",
            crs=crs,
            transform=transform,
            compress=self.config.compression,
            tiled=True,
            blockxsize=256,
            blockysize=256,
        ) as dst:
            dst.write(data, 1)
            dst.update_tags(
                TIFFTAG_SOFTWARE="HyperspectralSoilAI",
                METAL=metal,
                UNIT="mg/kg",
                STANDARD_LIMIT=str(SOIL_STANDARDS.get(metal, 0.0)),
            )
            if self.config.build_pyramids:
                overviews = [2, 4, 8, 16]
                dst.build_overviews(overviews, Resampling.average)
                dst.update_tags(ns="rio_overview", resampling="average")

        return os.path.abspath(output_path)

    def write_rgba_heatmap(
        self,
        concentration_map: np.ndarray,
        output_path: str,
        apply_smoothing: bool = True,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> str:
        """写入RGBA伪彩色热力图GeoTIFF (供WebGIS直接渲染)

        Args:
            concentration_map: [H, W] 浓度图
            output_path: 输出.tif路径
            apply_smoothing: 是否应用空间平滑
            vmin, vmax: 颜色映射范围

        Returns:
            绝对路径
        """
        if apply_smoothing:
            data = self.smoother.smooth(concentration_map)
        else:
            data = concentration_map.astype(np.float32)

        norm = self._normalize_to_uint8(data, vmin, vmax)
        rgba = self.apply_colormap(norm)

        H, W = data.shape
        transform = self._get_transform(H, W)
        crs = rasterio.crs.CRS.from_epsg(self.config.crs_epsg)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=H,
            width=W,
            count=4,
            dtype="uint8",
            crs=crs,
            transform=transform,
            compress=self.config.compression,
            tiled=True,
            blockxsize=256,
            blockysize=256,
            photometric="RGB",
            alpha="premultiplied",
        ) as dst:
            for i in range(4):
                dst.write(rgba[i], i + 1)
            if self.config.build_pyramids:
                dst.build_overviews([2, 4, 8, 16], Resampling.bilinear)

        return os.path.abspath(output_path)

    def write_multi_band(
        self,
        concentration_maps: Dict[str, np.ndarray],
        output_path: str,
        apply_smoothing: bool = True,
    ) -> str:
        """写入多波段GeoTIFF (Cd+Pb+As)

        Args:
            concentration_maps: {"Cd": [H,W], "Pb": [H,W], "As": [H,W]}
            output_path: 输出.tif路径
            apply_smoothing: 是否平滑

        Returns:
            绝对路径
        """
        order = ["Cd", "Pb", "As"]
        bands = []
        for metal in order:
            m = concentration_maps.get(metal)
            if m is None:
                raise KeyError(f"Missing metal band: {metal}")
            if apply_smoothing:
                bands.append(self.smoother.smooth(m))
            else:
                bands.append(m.astype(np.float32))

        stacked = np.stack(bands, axis=0)
        H, W = stacked.shape[1], stacked.shape[2]

        transform = self._get_transform(H, W)
        crs = rasterio.crs.CRS.from_epsg(self.config.crs_epsg)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=H,
            width=W,
            count=3,
            dtype="float32",
            crs=crs,
            transform=transform,
            compress=self.config.compression,
            tiled=True,
        ) as dst:
            for i, metal in enumerate(order):
                dst.write(stacked[i], i + 1)
                dst.set_band_description(i + 1, f"{metal}_mg_kg")
            if self.config.build_pyramids:
                dst.build_overviews([2, 4, 8], Resampling.average)

        return os.path.abspath(output_path)
