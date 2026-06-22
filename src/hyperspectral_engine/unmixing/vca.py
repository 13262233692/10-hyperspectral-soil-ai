"""顶点成分分析(Vertex Component Analysis)凸单纯形寻顶端元解混算子

基于Nascimento & Dias(2005)的经典VCA算法，纯NumPy实现。
从高光谱数据中无监督提取纯端元光谱及对应丰度图。

算法核心:
    1. 对数据做SNR加权投影降维
    2. 在降维空间中迭代寻找凸单纯形顶点
    3. 每次迭代选取当前投影方向上极值像素作为新端元
    4. 全约束最小二乘分解各像素的端元丰度
"""

import numpy as np
from typing import Tuple, Optional


def _snr_estimation(Y: np.ndarray) -> float:
    """估计信号噪声比"""
    d = Y - Y.mean(axis=1, keepdims=True)
    r = np.cov(Y)
    if r.ndim < 2:
        return 10.0
    try:
        eigvals = np.sort(np.linalg.eigvalsh(r))[::-1]
        signal_power = float(np.sum(eigvals[:max(1, len(eigvals) // 4)]))
        noise_power = float(np.sum(eigvals[max(1, len(eigvals) // 4):])) + 1e-12
        return max(1.0, 10.0 * np.log10(signal_power / noise_power))
    except np.linalg.LinAlgError:
        return 10.0


def vca(
    data: np.ndarray,
    num_endmembers: int,
    snr_threshold: float = 15.0,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """顶点成分分析(VCA)端元提取

    Args:
        data: 输入光谱矩阵 [L, N], L=波段数, N=像素数
        num_endmembers: 待提取端元数量 p
        snr_threshold: SNR阈值，高于此值使用投影方式，低于此值使用简单方式
        seed: 随机种子

    Returns:
        (endmembers, indices):
            endmembers: [L, p] 提取的端元光谱矩阵
            indices: [p] 对应的像素索引
    """
    L, N = data.shape
    p = num_endmembers
    rng = np.random.RandomState(seed)

    snr = _snr_estimation(data)

    if snr > snr_threshold and L > p:
        Ud = np.linalg.svd(data, full_matrices=False)[0][:, :p]
        Xd = Ud.T @ data
        u = Xd.mean(axis=1, keepdims=True)
        Xd_centered = Xd - u
    else:
        Xd = data.copy()
        u = Xd.mean(axis=1, keepdims=True)
        Xd_centered = Xd - u

    indices = np.zeros(p, dtype=np.intp)
    A = np.zeros((Xd_centered.shape[0], p), dtype=np.float64)
    A[:, 0] = Xd_centered[:, rng.randint(N)]

    for k in range(1, p):
        Q, _ = np.linalg.qr(A[:, :k], mode="reduced")
        q = Q[:, -1] if Q.ndim > 1 else Q

        proj = np.abs(q @ Xd_centered)
        idx = int(np.argmax(proj))
        indices[k] = idx
        A[:, k] = Xd_centered[:, idx]

    proj_first = np.abs(A[:, 0] @ Xd_centered)
    indices[0] = int(np.argmax(proj_first))

    endmembers = data[:, indices].astype(np.float32)

    return endmembers, indices


def fully_constrained_abundance(
    data: np.ndarray,
    endmembers: np.ndarray,
    max_iter: int = 50,
) -> np.ndarray:
    """全约束最小二乘(FCLS)端元丰度分解

    约束: sum(abundances) = 1, abundances >= 0

    Args:
        data: [L, N] 输入光谱
        endmembers: [L, p] 端元光谱
        max_iter: 迭代次数上限

    Returns:
        abundances: [p, N] 丰度矩阵
    """
    L, N = data.shape
    p = endmembers.shape[1]

    E = endmembers.astype(np.float64)
    Y = data.astype(np.float64)

    EtE = E.T @ E
    EtY = E.T @ Y

    ones = np.ones((1, p), dtype=np.float64)
    delta = 1e-3 * np.eye(p + 1, dtype=np.float64)
    delta[p, p] = 0.0

    M_aug = np.block([
        [EtE, ones.T],
        [ones, np.zeros((1, 1), dtype=np.float64)],
    ]) + delta

    try:
        M_inv = np.linalg.inv(M_aug)
    except np.linalg.LinAlgError:
        M_inv = np.linalg.pinv(M_aug)

    S = np.vstack([EtY, np.ones((1, N), dtype=np.float64)])
    W = M_inv @ S
    w = W[:p, :]

    w = np.clip(w, 0.0, None)

    col_sums = w.sum(axis=0, keepdims=True)
    col_sums = np.maximum(col_sums, 1e-12)
    w = w / col_sums

    return w.astype(np.float32)


def unmix_patch(
    patch_spectra: np.ndarray,
    num_endmembers: int = 4,
    snr_threshold: float = 15.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """对局部图斑执行VCA端元提取+丰度分解

    Args:
        patch_spectra: [L, N] 图斑内像素光谱矩阵
        num_endmembers: 提取端元数量

    Returns:
        (endmembers, abundances):
            endmembers: [L, p] 端元
            abundances: [p, N] 丰度
    """
    endmembers, _ = vca(patch_spectra, num_endmembers, snr_threshold=snr_threshold)
    abundances = fully_constrained_abundance(patch_spectra, endmembers)
    return endmembers, abundances
