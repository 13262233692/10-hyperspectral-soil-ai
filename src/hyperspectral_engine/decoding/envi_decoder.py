"""ENVI格式原生字节解码器 - 高分五号(GF-5)卫星高光谱数据解析

严格按字节位移解析ENVI格式原始高光谱数据卷，
单张影像包含330个连续光谱通道的定标反射率张量。

ENVI数据格式规范:
    - 头文件(.hdr): ASCII元数据描述
    - 数据文件(.dat/.raw): 二进制反射率张量
    - 数据类型映射: 1=uint8, 2=int16, 3=int32, 4=float32, 5=float64, 12=uint16
    - 交错方式: BSQ(波段顺序), BIL(波段按行), BIP(波段按像素)
    - 字节序: 0=主机序, 1=小端(LE), 2=大端(BE)
"""

import os
import struct
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field


@dataclass
class ENVIMetadata:
    """ENVI头文件解析后的元数据结构"""
    samples: int = 0
    lines: int = 0
    bands: int = 0
    data_type: int = 4
    interleave: str = "BSQ"
    byte_order: int = 0
    header_offset: int = 0
    wavelength: List[float] = field(default_factory=list)
    fwhm: List[float] = field(default_factory=list)
    gain: List[float] = field(default_factory=list)
    offset: List[float] = field(default_factory=list)
    map_info: Optional[Dict] = None
    raw_fields: Dict[str, str] = field(default_factory=dict)
    wavelength_order: str = "ascending"


class ENVIDecoder:
    """ENVI格式原生解码器 - 基于字节位移的底层解析

    不依赖GDAL等第三方库，纯Python+struct原生实现。
    支持GF-5卫星的330波段高光谱数据卷解析。
    """

    _DTYPE_MAP = {
        1: ("B", np.uint8),
        2: ("h", np.int16),
        3: ("i", np.int32),
        4: ("f", np.float32),
        5: ("d", np.float64),
        12: ("H", np.uint16),
    }

    _BYTE_ORDER_MAP = {
        0: ("", "="),
        1: ("<", "<"),
        2: (">", ">"),
    }

    def __init__(self, hdr_path: str, data_path: Optional[str] = None):
        """
        Args:
            hdr_path: ENVI头文件路径(.hdr)
            data_path: 数据文件路径，默认为同名.dat或.raw
        """
        self.hdr_path = hdr_path
        if data_path is None:
            base = os.path.splitext(hdr_path)[0]
            for ext in (".dat", ".raw", ".img"):
                candidate = base + ext
                if os.path.exists(candidate):
                    data_path = candidate
                    break
        self.data_path = data_path
        self.metadata: Optional[ENVIMetadata] = None
        self._data_buffer: Optional[bytes] = None

    def parse_header(self) -> ENVIMetadata:
        """解析ENVI ASCII头文件

        Returns:
            ENVIMetadata: 解析后的元数据对象

        Raises:
            FileNotFoundError: 头文件不存在
            ValueError: 头文件格式无效
        """
        if not os.path.exists(self.hdr_path):
            raise FileNotFoundError(f"ENVI header not found: {self.hdr_path}")

        with open(self.hdr_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        if not lines or not lines[0].strip().upper().startswith("ENVI"):
            raise ValueError("Invalid ENVI header: missing ENVI magic line")

        meta = ENVIMetadata()
        current_key = None
        current_value_lines: List[str] = []

        def _commit():
            if current_key and current_value_lines:
                raw = " ".join(current_value_lines).strip()
                meta.raw_fields[current_key] = raw
                self._parse_header_field(meta, current_key, raw)

        for line in lines[1:]:
            stripped = line.rstrip("\n").rstrip("\r")
            if not stripped.strip():
                continue

            if "=" in stripped and not stripped.startswith((" ", "\t")):
                _commit()
                key, _, value = stripped.partition("=")
                current_key = key.strip().lower()
                current_value_lines = [value.strip()]
            else:
                current_value_lines.append(stripped.strip())
        _commit()

        self.metadata = meta
        self._sniff_wavelength_order(meta)
        return meta

    @staticmethod
    def _sniff_wavelength_order(meta: ENVIMetadata) -> None:
        """动态嗅探波长排序方向

        当ENVI头文件未显式声明 wavelength order 标志位时，
        通过检查wavelength数组前N个元素的单调性推断排序方向。
        南方丘陵大雨冲刷批次传感器可能下发波长降序排列的数据卷。
        """
        if not meta.wavelength or len(meta.wavelength) < 2:
            return
        wl = meta.wavelength
        n_check = min(10, len(wl))
        asc_count = sum(1 for i in range(1, n_check) if wl[i] > wl[i - 1])
        desc_count = sum(1 for i in range(1, n_check) if wl[i] < wl[i - 1])
        if desc_count > asc_count:
            meta.wavelength_order = "descending"
        else:
            meta.wavelength_order = "ascending"

    @staticmethod
    def _parse_header_field(meta: ENVIMetadata, key: str, value: str) -> None:
        """解析单个头文件字段"""
        if key == "samples":
            meta.samples = int(value)
        elif key == "lines":
            meta.lines = int(value)
        elif key == "bands":
            meta.bands = int(value)
        elif key == "data type" or key == "data_type":
            meta.data_type = int(value)
        elif key == "interleave":
            meta.interleave = value.upper()
        elif key == "byte order" or key == "byte_order":
            meta.byte_order = int(value)
        elif key == "header offset" or key == "header_offset":
            meta.header_offset = int(value)
        elif key in ("wavelength", "fwhm", "gain", "offset"):
            nums = ENVIDecoder._parse_numeric_list(value)
            setattr(meta, key, nums)
        elif key == "map info" or key == "map_info":
            meta.map_info = ENVIDecoder._parse_map_info(value)
        elif key == "wavelength order" or key == "wavelength_order":
            val_lower = value.strip().lower()
            if "descend" in val_lower or "reverse" in val_lower or "decreas" in val_lower:
                meta.wavelength_order = "descending"
            else:
                meta.wavelength_order = "ascending"

    @staticmethod
    def _parse_numeric_list(value: str) -> List[float]:
        """解析头文件中的数值列表 (如 wavelength={...})"""
        s = value.strip()
        if s.startswith("{"):
            s = s[1:]
        if s.endswith("}"):
            s = s[:-1]
        return [float(x.strip()) for x in s.split(",") if x.strip()]

    @staticmethod
    def _parse_map_info(value: str) -> Dict:
        """解析 map info 字段"""
        s = value.strip().strip("{}")
        parts = [p.strip() for p in s.split(",")]
        if len(parts) >= 7:
            return {
                "projection": parts[0],
                "x": float(parts[1]),
                "y": float(parts[2]),
                "easting": float(parts[3]),
                "northing": float(parts[4]),
                "pixel_size_x": float(parts[5]),
                "pixel_size_y": float(parts[6]),
                "projection_zone": parts[7] if len(parts) > 7 else None,
                "datum": parts[8] if len(parts) > 8 else None,
                "units": parts[9] if len(parts) > 9 else None,
            }
        return {"raw": s}

    def _load_data_buffer(self) -> bytes:
        """加载整个数据文件到内存缓冲区"""
        if not self.data_path or not os.path.exists(self.data_path):
            raise FileNotFoundError(f"ENVI data file not found: {self.data_path}")
        with open(self.data_path, "rb") as f:
            self._data_buffer = f.read()
        return self._data_buffer

    def decode(self, reflectance_scale: float = 1.0) -> np.ndarray:
        """解码二进制数据为三维反射率张量 [bands, lines, samples]

        Args:
            reflectance_scale: 反射率缩放系数（如0.0001表示将整数缩放到0-1范围）

        Returns:
            np.ndarray: 形状 [C, H, W] 的float32反射率张量
                       C=bands=330 (GF-5), H=lines, W=samples

        Raises:
            RuntimeError: 元数据未解析或数据不完整
        """
        if self.metadata is None:
            self.parse_header()

        meta = self.metadata
        expected_elements = meta.samples * meta.lines * meta.bands

        if self._data_buffer is None:
            self._load_data_buffer()

        buf = self._data_buffer
        fmt_char, np_dtype = self._DTYPE_MAP.get(
            meta.data_type, self._DTYPE_MAP[4]
        )
        bo_prefix, np_bo = self._BYTE_ORDER_MAP.get(
            meta.byte_order, self._BYTE_ORDER_MAP[0]
        )

        element_size = struct.calcsize(bo_prefix + fmt_char)
        total_offset = meta.header_offset
        expected_bytes = total_offset + expected_elements * element_size

        if len(buf) < expected_bytes:
            raise RuntimeError(
                f"Data buffer too small: expected {expected_bytes}, got {len(buf)}"
            )

        if np_bo == "=":
            dtype_obj = np_dtype
        else:
            dtype_map = {
                np.uint8: "u1", np.int16: "i2", np.int32: "i4",
                np.float32: "f4", np.float64: "f8", np.uint16: "u2",
            }
            dtype_str = np_bo + dtype_map[np_dtype]
            dtype_obj = np.dtype(dtype_str)

        flat = np.frombuffer(
            buf,
            dtype=dtype_obj,
            count=expected_elements,
            offset=total_offset,
        )

        flat = flat.astype(np.float32)
        if reflectance_scale != 1.0:
            flat *= reflectance_scale

        interleave = meta.interleave.upper()
        if interleave == "BSQ":
            cube = flat.reshape(meta.bands, meta.lines, meta.samples)
        elif interleave == "BIL":
            cube = flat.reshape(meta.lines, meta.bands, meta.samples)
            cube = np.transpose(cube, (1, 0, 2))
        elif interleave == "BIP":
            cube = flat.reshape(meta.lines, meta.samples, meta.bands)
            cube = np.transpose(cube, (2, 0, 1))
        else:
            raise ValueError(f"Unsupported interleave: {meta.interleave}")

        if meta.wavelength_order == "descending":
            cube = cube[::-1, :, :].copy()

        return np.ascontiguousarray(cube)

    def decode_pixel(self, line: int, sample: int) -> np.ndarray:
        """按字节位移随机解码单个像素的全光谱向量 (330波段)

        使用底层字节位移，无需加载整个文件，适用于随机采样场景。

        Args:
            line: 行索引 (0-based)
            sample: 列索引 (0-based)

        Returns:
            np.ndarray: 形状 [330] 的float32反射率向量
        """
        if self.metadata is None:
            self.parse_header()
        if self._data_buffer is None:
            self._load_data_buffer()

        meta = self.metadata
        if not (0 <= line < meta.lines and 0 <= sample < meta.samples):
            raise IndexError(f"Pixel ({line},{sample}) out of bounds")

        fmt_char, np_dtype = self._DTYPE_MAP.get(
            meta.data_type, self._DTYPE_MAP[4]
        )
        bo_prefix, np_bo = self._BYTE_ORDER_MAP.get(
            meta.byte_order, self._BYTE_ORDER_MAP[0]
        )
        element_size = struct.calcsize(bo_prefix + fmt_char)

        spectrum = np.empty(meta.bands, dtype=np.float32)
        interleave = meta.interleave.upper()

        if interleave == "BSQ":
            for b in range(meta.bands):
                offset = (
                    meta.header_offset
                    + (b * meta.lines + line) * meta.samples + sample
                ) * element_size
                spectrum[b] = struct.unpack_from(
                    bo_prefix + fmt_char, self._data_buffer, offset
                )[0]
        elif interleave == "BIL":
            for b in range(meta.bands):
                offset = (
                    meta.header_offset
                    + (line * meta.bands + b) * meta.samples + sample
                ) * element_size
                spectrum[b] = struct.unpack_from(
                    bo_prefix + fmt_char, self._data_buffer, offset
                )[0]
        elif interleave == "BIP":
            base = (
                meta.header_offset
                + (line * meta.samples + sample) * meta.bands
            ) * element_size
            for b in range(meta.bands):
                offset = base + b * element_size
                spectrum[b] = struct.unpack_from(
                    bo_prefix + fmt_char, self._data_buffer, offset
                )[0]
        else:
            raise ValueError(f"Unsupported interleave: {meta.interleave}")

        if meta.wavelength_order == "descending":
            spectrum = spectrum[::-1].copy()

        return spectrum

    def get_spatial_shape(self) -> Tuple[int, int, int]:
        """返回 (bands, lines, samples) 形状"""
        if self.metadata is None:
            self.parse_header()
        return (self.metadata.bands, self.metadata.lines, self.metadata.samples)

    def get_wavelengths(self) -> np.ndarray:
        """返回中心波长数组 (nm)，始终按升序排列"""
        if self.metadata is None:
            self.parse_header()
        wl = np.array(self.metadata.wavelength, dtype=np.float32)
        if self.metadata.wavelength_order == "descending":
            wl = wl[::-1].copy()
        return wl


def decode_gf5_cube(
    hdr_path: str,
    data_path: Optional[str] = None,
    reflectance_scale: float = 1.0,
) -> Tuple[np.ndarray, ENVIMetadata]:
    """便捷函数: 一次性解码GF-5高光谱数据卷

    Args:
        hdr_path: .hdr头文件路径
        data_path: 数据文件路径(可选)
        reflectance_scale: 反射率缩放系数

    Returns:
        (cube, metadata): [330, H, W] 浮点张量与元数据
    """
    decoder = ENVIDecoder(hdr_path, data_path)
    meta = decoder.parse_header()
    cube = decoder.decode(reflectance_scale=reflectance_scale)
    return cube, meta
