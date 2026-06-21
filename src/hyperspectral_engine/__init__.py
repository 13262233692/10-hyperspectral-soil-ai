"""星载高光谱污染智能反演引擎 - 省级农业生态监控中心

模块结构:
    decoding     - ENVI格式原生字节解码器
    preprocessing - 自适应Savitzky-Golay平滑滤波
    models       - 1D CNN + Self-Attention多任务回归网络
    training     - PyTorch离线训练与权重导出
    inference    - ONNX Runtime低延迟推理引擎
    geotiff      - GeoTIFF污染热力图生成
    api          - FastAPI HTTP服务层
"""

__version__ = "1.0.0"
