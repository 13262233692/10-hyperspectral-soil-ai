"""服务启动入口

Usage:
    python main.py --model ./models/hyperspectral_inversion.onnx --host 0.0.0.0 --port 8000
    python main.py --train --output-dir ./models
"""

import os
import sys
import argparse
import uvicorn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def main():
    parser = argparse.ArgumentParser(description="高光谱土壤污染反演引擎")
    parser.add_argument("--model", type=str, default=None, help="ONNX模型路径")
    parser.add_argument("--train", action="store_true", help="启动前先执行训练")
    parser.add_argument("--output-dir", type=str, default="./models", help="模型输出目录")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-samples", type=int, default=8000)
    parser.add_argument("--num-epochs", type=int, default=40)
    parser.add_argument("--no-gpu", action="store_true")

    args = parser.parse_args()

    model_path = args.model
    if args.train or (model_path is None or not os.path.exists(model_path)):
        from hyperspectral_engine.training.train import run_training_pipeline
        print("=" * 60)
        print("  启动离线训练管线 -> 生成ONNX权重")
        print("=" * 60)
        model_path = run_training_pipeline(
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            num_epochs=args.num_epochs,
        )

    from hyperspectral_engine.api.server import create_app
    app = create_app(onnx_model_path=model_path, use_gpu=not args.no_gpu)

    print("\n" + "=" * 60)
    print(f"  启动HTTP服务: http://{args.host}:{args.port}")
    print(f"  模型文件: {model_path}")
    print(f"  API文档: http://{args.host}:{args.port}/docs")
    print("=" * 60 + "\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
