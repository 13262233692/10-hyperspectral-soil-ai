"""FastAPI HTTP服务层

暴露核心接口:
    POST /api/v1/inference/spectrum    - 单条光谱推理
    POST /api/v1/inference/batch       - 批量光谱推理
    POST /api/v1/inference/cube        - 数据立方体推理 -> GeoTIFF
    POST /api/v1/decode/envi           - ENVI原始数据解码
    GET  /api/v1/health                - 健康检查
    GET  /api/v1/status                - 引擎状态
    POST /api/v1/train                 - 触发离线训练 (可选)
"""

import os
import io
import uuid
import tempfile
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from pydantic import BaseModel, Field


from ..decoding.envi_decoder import ENVIDecoder, decode_gf5_cube
from ..preprocessing.sg_filter import PreprocessingPipeline
from ..inference.onnx_engine import ONNXInferenceEngine
from ..geotiff.heatmap_writer import (
    PollutionHeatmapWriter,
    GeoTIFFConfig,
    SOIL_STANDARDS,
)


APP_TITLE = "星载高光谱土壤污染智能反演引擎"
APP_VERSION = "1.0.0"


class SpectrumRequest(BaseModel):
    spectrum: List[float] = Field(..., description="330波段反射率向量")


class BatchSpectrumRequest(BaseModel):
    spectra: List[List[float]] = Field(..., description="N x 330 批量反射率矩阵")


class InferenceResponse(BaseModel):
    Cd: float
    Pb: float
    As: float
    unit: str = "mg/kg"
    inference_time_ms: float
    preprocess_time_ms: float


class BatchInferenceResponse(BaseModel):
    Cd: List[float]
    Pb: List[float]
    As: List[float]
    unit: str = "mg/kg"
    batch_size: int
    inference_time_ms: float
    preprocess_time_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: float


class EngineStatusResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: Optional[str]
    spectral_length: int
    providers: List[str]


class DecodeResponse(BaseModel):
    shape: List[int]
    bands: int
    lines: int
    samples: int
    wavelengths: List[float]
    message: str


class CubeInferenceResponse(BaseModel):
    Cd_geotiff_url: str
    Pb_geotiff_url: str
    As_geotiff_url: str
    multi_band_url: str
    rgba_heatmap_url: str
    processing_time_s: float
    spatial_shape: List[int]


class EngineContext:
    """全局引擎上下文"""

    def __init__(self):
        self.engine: Optional[ONNXInferenceEngine] = None
        self.pipeline = PreprocessingPipeline()
        self.writer = PollutionHeatmapWriter()
        self.output_dir = os.path.join(tempfile.gettempdir(), "hyperspectral_outputs")
        os.makedirs(self.output_dir, exist_ok=True)
        self.start_time = time.time()

    def load_model(self, onnx_path: str, use_gpu: bool = True) -> None:
        self.engine = ONNXInferenceEngine(
            onnx_model_path=onnx_path,
            use_gpu=use_gpu,
            enable_optimization=True,
        )

    def is_ready(self) -> bool:
        return self.engine is not None


context = EngineContext()


def create_app(onnx_model_path: Optional[str] = None, use_gpu: bool = True) -> FastAPI:
    """FastAPI应用工厂

    Args:
        onnx_model_path: ONNX模型路径
        use_gpu: 是否使用GPU

    Returns:
        FastAPI实例
    """
    app = FastAPI(
        title=APP_TITLE,
        version=APP_VERSION,
        description="省级农业生态监控中心 - 高分五号(GF-5)星载高光谱土壤重金属污染智能反演引擎",
    )

    if onnx_model_path and os.path.exists(onnx_model_path):
        try:
            context.load_model(onnx_model_path, use_gpu=use_gpu)
        except Exception as e:
            print(f"Warning: failed to load model: {e}")

    @app.get("/api/v1/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(
            status="ok" if context.is_ready() else "degraded",
            version=APP_VERSION,
            timestamp=time.time(),
        )

    @app.get("/api/v1/status", response_model=EngineStatusResponse)
    async def status():
        return EngineStatusResponse(
            status="ready" if context.is_ready() else "model_not_loaded",
            model_loaded=context.is_ready(),
            model_path=context.engine.model_path if context.engine else None,
            spectral_length=context.engine.spectral_length if context.engine else 330,
            providers=context.engine.providers if context.engine else [],
        )

    @app.post("/api/v1/inference/spectrum", response_model=InferenceResponse)
    async def inference_spectrum(req: SpectrumRequest):
        if not context.is_ready():
            raise HTTPException(status_code=503, detail="推理引擎未就绪，请先加载模型")

        spectrum = np.array(req.spectrum, dtype=np.float32)
        if spectrum.shape[0] != 330:
            raise HTTPException(status_code=400, detail=f"光谱长度必须为330，当前为{spectrum.shape[0]}")

        result = context.engine.predict(spectrum)
        return InferenceResponse(
            Cd=float(result.Cd[0]),
            Pb=float(result.Pb[0]),
            As=float(result.As[0]),
            unit="mg/kg",
            inference_time_ms=result.inference_time_ms,
            preprocess_time_ms=result.preprocess_time_ms,
        )

    @app.post("/api/v1/inference/batch", response_model=BatchInferenceResponse)
    async def inference_batch(req: BatchSpectrumRequest):
        if not context.is_ready():
            raise HTTPException(status_code=503, detail="推理引擎未就绪")

        spectra = np.array(req.spectra, dtype=np.float32)
        if spectra.ndim != 2 or spectra.shape[1] != 330:
            raise HTTPException(
                status_code=400,
                detail=f"输入形状必须为 [N, 330]，当前为 {list(spectra.shape)}",
            )

        result = context.engine.predict(spectra)
        return BatchInferenceResponse(
            Cd=result.Cd.tolist(),
            Pb=result.Pb.tolist(),
            As=result.As.tolist(),
            unit="mg/kg",
            batch_size=result.batch_size,
            inference_time_ms=result.inference_time_ms,
            preprocess_time_ms=result.preprocess_time_ms,
        )

    @app.post("/api/v1/decode/envi", response_model=DecodeResponse)
    async def decode_envi(
        hdr_file: UploadFile = File(...),
        data_file: Optional[UploadFile] = File(None),
        reflectance_scale: float = Form(1.0),
    ):
        t0 = time.time()
        tmp_hdr = tempfile.NamedTemporaryFile(suffix=".hdr", delete=False)
        tmp_hdr.write(await hdr_file.read())
        tmp_hdr.close()

        tmp_data_path = None
        if data_file is not None:
            suffix = os.path.splitext(data_file.filename or "")[1] or ".dat"
            tmp_data = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            tmp_data.write(await data_file.read())
            tmp_data.close()
            tmp_data_path = tmp_data.name

        try:
            cube, meta = decode_gf5_cube(
                tmp_hdr.name, tmp_data_path, reflectance_scale=reflectance_scale
            )
        except Exception as e:
            os.unlink(tmp_hdr.name)
            if tmp_data_path:
                os.unlink(tmp_data_path)
            raise HTTPException(status_code=400, detail=f"ENVI解码失败: {str(e)}")

        os.unlink(tmp_hdr.name)
        if tmp_data_path:
            os.unlink(tmp_data_path)

        return DecodeResponse(
            shape=list(cube.shape),
            bands=int(meta.bands),
            lines=int(meta.lines),
            samples=int(meta.samples),
            wavelengths=meta.wavelength,
            message=f"解码成功，耗时 {(time.time() - t0) * 1000:.1f}ms",
        )

    @app.post("/api/v1/inference/cube")
    async def inference_cube(
        hdr_file: UploadFile = File(...),
        data_file: Optional[UploadFile] = File(None),
        reflectance_scale: float = Form(1.0),
        crs_epsg: int = Form(4326),
        pixel_size: float = Form(30.0),
        origin_x: float = Form(0.0),
        origin_y: float = Form(0.0),
        generate_rgba: bool = Form(True),
    ):
        if not context.is_ready():
            raise HTTPException(status_code=503, detail="推理引擎未就绪")

        t0 = time.time()
        tmp_hdr = tempfile.NamedTemporaryFile(suffix=".hdr", delete=False)
        tmp_hdr.write(await hdr_file.read())
        tmp_hdr.close()

        tmp_data_path = None
        if data_file is not None:
            suffix = os.path.splitext(data_file.filename or "")[1] or ".dat"
            tmp_data = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            tmp_data.write(await data_file.read())
            tmp_data.close()
            tmp_data_path = tmp_data.name

        try:
            cube, meta = decode_gf5_cube(
                tmp_hdr.name, tmp_data_path, reflectance_scale=reflectance_scale
            )
        except Exception as e:
            os.unlink(tmp_hdr.name)
            if tmp_data_path:
                os.unlink(tmp_data_path)
            raise HTTPException(status_code=400, detail=f"ENVI解码失败: {str(e)}")

        os.unlink(tmp_hdr.name)
        if tmp_data_path:
            os.unlink(tmp_data_path)

        try:
            cd_map, pb_map, as_map = context.engine.predict_cube(cube)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"推理失败: {str(e)}")

        run_id = uuid.uuid4().hex[:12]
        writer_cfg = GeoTIFFConfig(
            crs_epsg=crs_epsg,
            origin_x=origin_x,
            origin_y=origin_y,
            pixel_size_x=pixel_size,
            pixel_size_y=pixel_size,
        )
        writer = PollutionHeatmapWriter(config=writer_cfg)

        out_paths = {}
        for metal, arr in zip(["Cd", "Pb", "As"], [cd_map, pb_map, as_map]):
            path = os.path.join(context.output_dir, f"{run_id}_{metal}_conc.tif")
            writer.write_concentration(arr, path, metal=metal)
            out_paths[metal] = path

        multi_path = os.path.join(context.output_dir, f"{run_id}_multi.tif")
        writer.write_multi_band(
            {"Cd": cd_map, "Pb": pb_map, "As": as_map}, multi_path
        )

        rgba_path = None
        if generate_rgba:
            rgba_path = os.path.join(context.output_dir, f"{run_id}_Cd_rgba.tif")
            writer.write_rgba_heatmap(cd_map, rgba_path)

        processing_time_s = time.time() - t0

        return CubeInferenceResponse(
            Cd_geotiff_url=f"/api/v1/download/{os.path.basename(out_paths['Cd'])}",
            Pb_geotiff_url=f"/api/v1/download/{os.path.basename(out_paths['Pb'])}",
            As_geotiff_url=f"/api/v1/download/{os.path.basename(out_paths['As'])}",
            multi_band_url=f"/api/v1/download/{os.path.basename(multi_path)}",
            rgba_heatmap_url=(
                f"/api/v1/download/{os.path.basename(rgba_path)}" if rgba_path else ""
            ),
            processing_time_s=processing_time_s,
            spatial_shape=list(cd_map.shape),
        )

    @app.get("/api/v1/download/{filename}")
    async def download_file(filename: str):
        filepath = os.path.join(context.output_dir, filename)
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="文件不存在或已过期")
        return FileResponse(
            filepath,
            media_type="image/tiff",
            filename=filename,
        )

    return app
