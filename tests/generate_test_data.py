"""合成GF-5 ENVI格式测试数据生成器

生成符合ENVI规范的.hdr和.dat文件，用于系统集成测试。
"""

import os
import struct
import numpy as np
from typing import Tuple


SPECTRAL_LENGTH = 330
WAVELENGTHS = np.linspace(400.0, 2500.0, SPECTRAL_LENGTH, dtype=np.float32)
FWHM = np.full(SPECTRAL_LENGTH, 5.0, dtype=np.float32)


def write_envi_header(
    hdr_path: str,
    samples: int,
    lines: int,
    bands: int = SPECTRAL_LENGTH,
    interleave: str = "BSQ",
    data_type: int = 4,
    byte_order: int = 0,
) -> None:
    """写入ENVI .hdr头文件"""
    wl_str = ", ".join(f"{w:.4f}" for w in WAVELENGTHS)
    fwhm_str = ", ".join(f"{w:.2f}" for w in FWHM)

    header = f"""ENVI
description = {{ Synthetic GF-5 Hyperspectral Data for Testing }}
samples = {samples}
lines = {lines}
bands = {bands}
header offset = 0
file type = ENVI Standard
data type = {data_type}
interleave = {interleave}
byte order = {byte_order}
wavelength units = Nanometers
wavelength = {{ {wl_str} }}
fwhm = {{ {fwhm_str} }}
map info = {{ Geographic Lat/Lon, 1, 1, 116.391, 39.907, 0.0002695, 0.0002695, WGS-84 }}
coordinate system string = {{ GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]] }}
"""
    with open(hdr_path, "w", encoding="utf-8") as f:
        f.write(header)


def _soil_spectrum(wl: np.ndarray, base_type: int, cd: float, pb: float, as_: float) -> np.ndarray:
    """生成单条土壤反射率光谱"""
    if base_type == 0:
        base = 0.15 + 0.35 * np.exp(-((wl - 600.0) ** 2) / (2.0 * 200.0**2))
        base += 0.1 * np.tanh((wl - 800.0) / 300.0)
    elif base_type == 1:
        base = 0.08 + 0.15 * (1.0 - np.exp(-(wl - 500.0) / 1500.0))
    else:
        base = 0.3 + 0.2 * np.sin((wl - 500.0) / 600.0)
        base = np.clip(base, 0.2, 0.6)

    base += np.exp(-((wl - 580.0) ** 2) / (2.0 * 40.0**2)) * cd * 0.002
    base += np.exp(-((wl - 680.0) ** 2) / (2.0 * 50.0**2)) * pb * 0.0003
    base += np.exp(-((wl - 520.0) ** 2) / (2.0 * 35.0**2)) * as_ * 0.0008

    return base.astype(np.float32)


def generate_synthetic_envi(
    output_dir: str,
    samples: int = 128,
    lines: int = 128,
    seed: int = 42,
    noise_std: float = 0.015,
) -> Tuple[str, str]:
    """生成合成GF-5 ENVI数据集

    Args:
        output_dir: 输出目录
        samples: 图像宽度
        lines: 图像高度
        seed: 随机种子
        noise_std: 噪声标准差

    Returns:
        (hdr_path, dat_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(seed)

    base_hdr = os.path.join(output_dir, "gf5_test.hdr")
    base_dat = os.path.join(output_dir, "gf5_test.dat")
    write_envi_header(base_hdr, samples, lines)

    H, W, C = lines, samples, SPECTRAL_LENGTH
    cube = np.zeros((C, H, W), dtype=np.float32)

    for y in range(H):
        for x in range(W):
            base_type = rng.randint(0, 3)
            cd = rng.uniform(0.05, 6.0)
            pb = rng.uniform(5.0, 250.0)
            as_ = rng.uniform(1.0, 60.0)
            spectrum = _soil_spectrum(WAVELENGTHS, base_type, cd, pb, as_)
            spectrum += rng.normal(0.0, noise_std, C).astype(np.float32)
            cube[:, y, x] = np.clip(spectrum, 0.0, 1.0)

    with open(base_dat, "wb") as f:
        f.write(cube.astype(np.float32).tobytes())

    return base_hdr, base_dat


if __name__ == "__main__":
    hdr, dat = generate_synthetic_envi("./data")
    print(f"Generated: {hdr}")
    print(f"Generated: {dat}")
