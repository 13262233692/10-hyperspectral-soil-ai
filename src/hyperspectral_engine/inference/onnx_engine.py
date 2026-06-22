"""ONNX Runtime低延迟推理引擎

设计原则:
    - 严格解耦：本模块不依赖 PyTorch
    - 极低延迟：session级复用、GraphOptimization
    - 线程安全：支持多线程并发推理
    - 波长排序感知: 预处理管道自动检测并重排降序波长
    - 内存安全: predict_cube切片分发无闭包逃逸
    - 纯度干预: 地膜遮蔽图斑静默阻断重金属反演

只依赖 numpy 和 onnxruntime
"""

import os
import gc
import time
import weakref
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from ..preprocessing.sg_filter import PreprocessingPipeline
from ..unmixing.intervention import PurityInterventionModule, InterventionPayload


TARGET_METALS = ["Cd", "Pb", "As"]


@dataclass
class InferenceResult:
    Cd: np.ndarray
    Pb: np.ndarray
    As: np.ndarray
    inference_time_ms: float
    preprocess_time_ms: float
    batch_size: int
    intervention: Optional[InterventionPayload] = None


class ONNXInferenceEngine:
    """ONNX Runtime推理引擎 (与PyTorch完全解耦)

    仅使用 onnxruntime.InferenceSession
    """

    def __init__(
        self,
        onnx_model_path: str,
        use_gpu: bool = True,
        intra_op_num_threads: int = 0,
        inter_op_num_threads: int = 0,
        enable_optimization: bool = True,
        preprocessing_config: Optional[Dict] = None,
        enable_intervention: bool = True,
        mulch_sam_threshold: float = 0.05,
        intervention_num_endmembers: int = 4,
    ):
        if ort is None:
            raise ImportError(
                "onnxruntime is not installed. "
                "Please install with: pip install onnxruntime-gpu or onnxruntime"
            )
        if not os.path.exists(onnx_model_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_model_path}")

        self.model_path = os.path.abspath(onnx_model_path)
        self._session = None
        self._lock = threading.Lock()

        self._providers = self._select_providers(use_gpu)
        self._session_options = ort.SessionOptions()
        if enable_optimization:
            self._session_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
        self._session_options.intra_op_num_threads = intra_op_num_threads
        self._session_options.inter_op_num_threads = inter_op_num_threads
        self._session_options.log_severity_level = 3

        pp_cfg = preprocessing_config or {}
        self._pipeline = PreprocessingPipeline(**pp_cfg)

        self._enable_intervention = enable_intervention
        self._intervention_module: Optional[PurityInterventionModule] = None
        if enable_intervention:
            self._intervention_module = PurityInterventionModule(
                mulch_sam_threshold=mulch_sam_threshold,
                num_endmembers=intervention_num_endmembers,
            )

        self._build_session()

        meta = self._session.get_inputs()[0]
        self._input_name = meta.name
        self._input_shape = meta.shape
        self._output_names = [o.name for o in self._session.get_outputs()]

        self._spectral_length = self._infer_spectral_length()
        self._warmup()

    @staticmethod
    def _select_providers(use_gpu: bool) -> List[str]:
        available = ort.get_available_providers()
        providers = []
        if use_gpu and "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        return providers

    def _build_session(self) -> None:
        self._session = ort.InferenceSession(
            self.model_path,
            sess_options=self._session_options,
            providers=self._providers,
        )

    def _infer_spectral_length(self) -> int:
        shape = self._input_shape
        if len(shape) >= 1 and isinstance(shape[-1], int):
            return int(shape[-1])
        return 330

    def _warmup(self, batch_size: int = 8) -> None:
        dummy = np.random.randn(batch_size, 1, self._spectral_length).astype(np.float32)
        _ = self._session.run(None, {self._input_name: dummy})

    @property
    def spectral_length(self) -> int:
        return self._spectral_length

    @property
    def providers(self) -> List[str]:
        return self._providers

    def set_wavelengths(self, wavelengths: np.ndarray) -> None:
        """运行时动态注入波长信息

        当数据源从晴空干燥区切换到南方丘陵大雨批次时，
        调用此方法将ENVI元数据中的波长数组注入预处理管道，
        触发波长排序嗅探和吸收带掩码重建。

        Args:
            wavelengths: 实际波长数组 (可能为降序排列)
        """
        self._pipeline.set_wavelengths(wavelengths)

    def _preprocess(self, spectra: np.ndarray) -> Tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        x = np.asarray(spectra, dtype=np.float32)

        if x.ndim == 1:
            x = x.reshape(1, -1)
        elif x.ndim == 3:
            B, C, L = x.shape
            if C == 1:
                x = x.reshape(B, L)

        if x.shape[-1] != self._spectral_length:
            raise ValueError(
                f"Expected spectral length {self._spectral_length}, "
                f"got {x.shape[-1]}"
            )

        x = self._pipeline.process_batch(x)
        x = x.reshape(-1, 1, self._spectral_length)

        dt_ms = (time.perf_counter() - t0) * 1000.0
        return x, dt_ms

    @staticmethod
    def _validate_output(x):
        x = np.asarray(x, dtype=np.float32)
        return np.clip(x, a_min=0.0, a_max=None)

    def predict(self, spectra: np.ndarray) -> InferenceResult:
        x, pp_ms = self._preprocess(spectra)
        B = x.shape[0]

        t0 = time.perf_counter()
        with self._lock:
            outputs = self._session.run(self._output_names, {self._input_name: x})
        inf_ms = (time.perf_counter() - t0) * 1000.0

        result = {}
        for name, arr in zip(self._output_names, outputs):
            result[name] = self._validate_output(arr)

        return InferenceResult(
            Cd=result.get("Cd", np.zeros(B, dtype=np.float32)),
            Pb=result.get("Pb", np.zeros(B, dtype=np.float32)),
            As=result.get("As", np.zeros(B, dtype=np.float32)),
            inference_time_ms=inf_ms,
            preprocess_time_ms=pp_ms,
            batch_size=B,
        )

    def predict_single(self, spectrum: np.ndarray) -> Dict[str, float]:
        res = self.predict(spectrum)
        return {
            "Cd": float(res.Cd[0]),
            "Pb": float(res.Pb[0]),
            "As": float(res.As[0]),
        }

    def predict_cube(
        self,
        cube: np.ndarray,
        batch_size: int = 256,
        intervention_patch_size: int = 16,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[InterventionPayload]]:
        """推理整个高光谱立方体 [C, H, W]

        内存安全设计:
            - 逐批推理后立即提取标量结果写入预分配数组
            - 显式删除中间InferenceResult引用
            - 每批次结束触发分代回收标记
            - 无闭包捕获: 输出数组独立于推理结果对象生命周期

        纯度干预旁路:
            - 将立方体切分为 patch_size x patch_size 图斑
            - 对每个图斑执行VCA端元解混 + 地膜SAM判定
            - 地膜遮蔽图斑内像素Cd/Pb/As设为NaN(静默阻断)
            - 返回干预净荷供WebGIS渲染警示网格

        Args:
            cube: [C, H, W] 反射率张量
            batch_size: 推理批大小
            intervention_patch_size: 干预检测图斑大小

        Returns:
            (Cd_map, Pb_map, As_map, intervention_payload):
                Cd/Pb/As: [H, W] 浓度图(mg/kg), 被阻断像素为NaN
                intervention_payload: 干预净荷(无干预时为None)
        """
        C, H, W = cube.shape
        if C != self._spectral_length:
            raise ValueError(
                f"Expected {self._spectral_length} bands, got {C}"
            )

        pixels = np.ascontiguousarray(
            cube.transpose(1, 2, 0).reshape(-1, C)
        )
        N = pixels.shape[0]

        cds = np.empty(N, dtype=np.float32)
        pbs = np.empty(N, dtype=np.float32)
        ass = np.empty(N, dtype=np.float32)

        intervention_payload: Optional[InterventionPayload] = None
        blocked_pixels = np.zeros(N, dtype=bool)

        if self._enable_intervention and self._intervention_module is not None:
            patch_size = intervention_patch_size
            n_patches_h = max(1, H // patch_size)
            n_patches_w = max(1, W // patch_size)
            ph = H // n_patches_h
            pw = W // n_patches_w

            for pi in range(n_patches_h):
                for pj in range(n_patches_w):
                    r0 = pi * ph
                    r1 = min(r0 + ph, H)
                    c0 = pj * pw
                    c1 = min(c0 + pw, W)
                    patch = cube[:, r0:r1, c0:c1]
                    patch_pixels = patch.reshape(C, -1)

                    if patch_pixels.shape[1] < self._intervention_module.min_pixels:
                        continue

                    payload = self._intervention_module.analyze_patch(patch_pixels)

                    if payload.blocked and payload.mulch_detected:
                        intervention_payload = payload
                        mulch_idx = payload.mulch_endmember_index
                        if payload.abundance_profile is not None and mulch_idx >= 0:
                            mulch_ab = payload.abundance_profile[mulch_idx, :]
                            ab_threshold = float(mulch_ab.mean())
                            patch_blocked = mulch_ab >= ab_threshold
                            for k, local_idx in enumerate(range(patch_pixels.shape[1])):
                                if patch_blocked[k]:
                                    global_row = r0 + local_idx // (c1 - c0)
                                    global_col = c0 + local_idx % (c1 - c0)
                                    global_idx = global_row * W + global_col
                                    if 0 <= global_idx < N:
                                        blocked_pixels[global_idx] = True
                        del payload

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_pixels = np.ascontiguousarray(pixels[start:end])
            res = self.predict(batch_pixels)
            cds[start:end] = res.Cd
            pbs[start:end] = res.Pb
            ass[start:end] = res.As
            del res, batch_pixels

            if start % (batch_size * 4) == 0 and start > 0:
                gc.collect(generation=0)

        cds[blocked_pixels] = np.nan
        pbs[blocked_pixels] = np.nan
        ass[blocked_pixels] = np.nan

        del pixels
        gc.collect(generation=0)

        return (
            np.ascontiguousarray(cds.reshape(H, W)),
            np.ascontiguousarray(pbs.reshape(H, W)),
            np.ascontiguousarray(ass.reshape(H, W)),
            intervention_payload,
        )

    def benchmark(
        self,
        batch_size: int = 64,
        num_iters: int = 100,
    ) -> Dict[str, float]:
        dummy = np.random.randn(batch_size, self._spectral_length).astype(np.float32)

        times = []
        for _ in range(num_iters + 10):
            t0 = time.perf_counter()
            self.predict(dummy)
            times.append((time.perf_counter() - t0) * 1000.0)
        times = times[10:]

        return {
            "batch_size": batch_size,
            "mean_ms": float(np.mean(times)),
            "median_ms": float(np.median(times)),
            "p95_ms": float(np.percentile(times, 95)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "fps": float(batch_size * 1000.0 / np.mean(times)),
        }
