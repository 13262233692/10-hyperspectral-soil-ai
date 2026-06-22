"""内置标准端元光谱库与光谱角距离(SAM)计算

标准端元:
    - 农用聚乙烯地膜(Agricultural Polyethylene Mulch Film)
      特征吸收: 1210nm(C-H变形), 1730nm(C-H伸展), 2310nm(C-H合频)
    - 绿色植被(参考USGS植被光谱)
    - 裸土(参考USGS土壤光谱)
"""

import numpy as np
from typing import Dict, Tuple, Optional


GF5_WAVELENGTHS = np.linspace(400.0, 2500.0, 330, dtype=np.float32)


def _polyethylene_mulch_film(wl: np.ndarray) -> np.ndarray:
    """农用聚乙烯(PE)地膜标准反射率光谱

    基于USGS/ASTER光谱库中polyethylene特征:
        - 可见-近红外: 高反射(0.85-0.95)
        - 1210nm: C-H变形振动吸收(深度~0.15)
        - 1400nm: 水汽残留(微弱)
        - 1730nm: C-H伸展振动强吸收(深度~0.25)
        - 2310nm: C-H合频吸收(深度~0.20)
        - 其余: 相对平坦高反射
    """
    base = 0.90 * np.ones_like(wl, dtype=np.float32)
    base -= 0.15 * np.exp(-((wl - 1210.0) ** 2) / (2.0 * 25.0**2))
    base -= 0.25 * np.exp(-((wl - 1730.0) ** 2) / (2.0 * 30.0**2))
    base -= 0.20 * np.exp(-((wl - 2310.0) ** 2) / (2.0 * 35.0**2))
    base -= 0.03 * np.exp(-((wl - 1400.0) ** 2) / (2.0 * 15.0**2))
    base[:np.searchsorted(wl, 450.0)] *= 0.92
    return np.clip(base, 0.0, 1.0).astype(np.float32)


def _green_vegetation(wl: np.ndarray) -> np.ndarray:
    """绿色植被参考光谱"""
    base = 0.05 * np.ones_like(wl, dtype=np.float32)
    base += 0.10 * np.exp(-((wl - 550.0) ** 2) / (2.0 * 30.0**2))
    base += 0.35 * np.tanh((wl - 700.0) / 50.0)
    base += 0.10 * np.exp(-((wl - 1700.0) ** 2) / (2.0 * 200.0**2))
    base -= 0.18 * np.exp(-((wl - 1450.0) ** 2) / (2.0 * 40.0**2))
    base -= 0.15 * np.exp(-((wl - 1940.0) ** 2) / (2.0 * 50.0**2))
    return np.clip(base, 0.0, 1.0).astype(np.float32)


def _bare_soil(wl: np.ndarray) -> np.ndarray:
    """裸土参考光谱"""
    base = 0.12 + 0.25 * np.tanh((wl - 600.0) / 500.0)
    base -= 0.06 * np.exp(-((wl - 1450.0) ** 2) / (2.0 * 40.0**2))
    base -= 0.05 * np.exp(-((wl - 1940.0) ** 2) / (2.0 * 50.0**2))
    base -= 0.04 * np.exp(-((wl - 2200.0) ** 2) / (2.0 * 30.0**2))
    return np.clip(base, 0.0, 1.0).astype(np.float32)


STANDARD_ENDMEMBERS: Dict[str, np.ndarray] = {}


def _init_endmembers():
    global STANDARD_ENDMEMBERS
    if STANDARD_ENDMEMBERS:
        return
    wl = GF5_WAVELENGTHS
    STANDARD_ENDMEMBERS["polyethylene_mulch"] = _polyethylene_mulch_film(wl)
    STANDARD_ENDMEMBERS["green_vegetation"] = _green_vegetation(wl)
    STANDARD_ENDMEMBERS["bare_soil"] = _bare_soil(wl)


def get_endmember(name: str) -> np.ndarray:
    """获取内置标准端元光谱

    Args:
        name: 端元名称("polyethylene_mulch", "green_vegetation", "bare_soil")

    Returns:
        [330] float32 反射率光谱
    """
    _init_endmembers()
    if name not in STANDARD_ENDMEMBERS:
        raise KeyError(f"Unknown endmember: {name}. Available: {list(STANDARD_ENDMEMBERS.keys())}")
    return STANDARD_ENDMEMBERS[name].copy()


def spectral_angle(spectrum_a: np.ndarray, spectrum_b: np.ndarray) -> float:
    """计算两个光谱向量之间的光谱角距离(SAM)

    SAM = arccos( <a,b> / (||a|| * ||b||) )

    Args:
        spectrum_a: [L] 光谱向量A
        spectrum_b: [L] 光谱向量B

    Returns:
        光谱角距离 (弧度), 范围 [0, π]
    """
    a = np.asarray(spectrum_a, dtype=np.float64).ravel()
    b = np.asarray(spectrum_b, dtype=np.float64).ravel()

    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a < 1e-12 or norm_b < 1e-12:
        return np.pi

    cos_val = dot / (norm_a * norm_b)
    cos_val = np.clip(cos_val, -1.0, 1.0)
    return float(np.arccos(cos_val))


def spectral_angle_batch(
    spectra: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """批量计算光谱角距离

    Args:
        spectra: [N, L] N条光谱
        reference: [L] 参考光谱

    Returns:
        [N] 每条光谱与参考的SAM角度(弧度)
    """
    ref = np.asarray(reference, dtype=np.float64).ravel()
    ref_norm = np.linalg.norm(ref)
    if ref_norm < 1e-12:
        return np.full(spectra.shape[0], np.pi, dtype=np.float32)

    sp = np.asarray(spectra, dtype=np.float64)
    dots = sp @ ref
    norms = np.linalg.norm(sp, axis=1)
    denom = norms * ref_norm
    denom = np.maximum(denom, 1e-12)
    cos_vals = np.clip(dots / denom, -1.0, 1.0)
    return np.arccos(cos_vals).astype(np.float32)


def best_matching_endmember(
    extracted_endmember: np.ndarray,
    candidate_names: Optional[list] = None,
) -> Tuple[str, float]:
    """找出与提取端元最匹配的标准端元

    Args:
        extracted_endmember: [L] 提取的端元光谱
        candidate_names: 候选端元名称列表，默认全部

    Returns:
        (name, sam_angle): 最佳匹配名称和光谱角
    """
    _init_endmembers()
    candidates = candidate_names or list(STANDARD_ENDMEMBERS.keys())

    best_name = ""
    best_angle = np.pi
    for name in candidates:
        ref = STANDARD_ENDMEMBERS[name]
        angle = spectral_angle(extracted_endmember, ref)
        if angle < best_angle:
            best_angle = angle
            best_name = name

    return best_name, best_angle
