"""PyTorch离线训练模块

支持功能:
    - 数据集封装与加载
    - 多任务联合损失 (L1 + MSE + 相关系数正则)
    - 学习率调度与早停
    - 训练日志记录
    - PyTorch权重 -> ONNX格式导出
    - 训练/验证/测试指标评估
"""

import os
import json
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from ..models.spectral_net import HyperspectralInversionNet, build_model


TARGET_METALS = ["Cd", "Pb", "As"]


class SyntheticHyperspectralDataset(Dataset):
    """合成高光谱数据集（用于演示与集成测试）

    根据土壤重金属浓度特征，合成GF-5 330波段反射率光谱。
    """

    def __init__(
        self,
        num_samples: int = 5000,
        spectral_length: int = 330,
        seed: int = 42,
        noise_std: float = 0.02,
    ):
        super().__init__()
        self.spectral_length = spectral_length
        rng = np.random.RandomState(seed)
        wavelengths = np.linspace(400.0, 2500.0, spectral_length, dtype=np.float32)

        self.spectra = np.zeros((num_samples, spectral_length), dtype=np.float32)
        self.labels = np.zeros((num_samples, 3), dtype=np.float32)

        soil_bases = [
            self._soil_spectrum_red(wavelengths),
            self._soil_spectrum_dark(wavelengths),
            self._soil_spectrum_sandy(wavelengths),
        ]

        for i in range(num_samples):
            base_idx = rng.randint(0, len(soil_bases))
            spectrum = soil_bases[base_idx].copy()

            cd_conc = rng.uniform(0.05, 8.0)
            pb_conc = rng.uniform(5.0, 300.0)
            as_conc = rng.uniform(1.0, 80.0)

            spectrum += self._cd_spectral_feature(wavelengths) * cd_conc * 0.002
            spectrum += self._pb_spectral_feature(wavelengths) * pb_conc * 0.0003
            spectrum += self._as_spectral_feature(wavelengths) * as_conc * 0.0008

            spectrum += rng.normal(0.0, noise_std, size=spectrum.shape).astype(np.float32)
            spectrum = np.clip(spectrum, 0.0, 1.0)

            self.spectra[i] = spectrum.astype(np.float32)
            self.labels[i] = [cd_conc, pb_conc, as_conc]

    @staticmethod
    def _soil_spectrum_red(wl: np.ndarray) -> np.ndarray:
        base = 0.15 + 0.35 * np.exp(-((wl - 600.0) ** 2) / (2.0 * 200.0**2))
        base += 0.1 * np.tanh((wl - 800.0) / 300.0)
        return base.astype(np.float32)

    @staticmethod
    def _soil_spectrum_dark(wl: np.ndarray) -> np.ndarray:
        base = 0.08 + 0.15 * (1.0 - np.exp(-(wl - 500.0) / 1500.0))
        return base.astype(np.float32)

    @staticmethod
    def _soil_spectrum_sandy(wl: np.ndarray) -> np.ndarray:
        base = 0.3 + 0.2 * np.sin((wl - 500.0) / 600.0)
        base = np.clip(base, 0.2, 0.6)
        return base.astype(np.float32)

    @staticmethod
    def _cd_spectral_feature(wl: np.ndarray) -> np.ndarray:
        feat = (
            np.exp(-((wl - 580.0) ** 2) / (2.0 * 40.0**2))
            - 0.5 * np.exp(-((wl - 920.0) ** 2) / (2.0 * 60.0**2))
            + 0.3 * np.exp(-((wl - 1750.0) ** 2) / (2.0 * 80.0**2))
        )
        return feat.astype(np.float32)

    @staticmethod
    def _pb_spectral_feature(wl: np.ndarray) -> np.ndarray:
        feat = (
            np.exp(-((wl - 680.0) ** 2) / (2.0 * 50.0**2))
            - 0.4 * np.exp(-((wl - 1150.0) ** 2) / (2.0 * 70.0**2))
            + 0.25 * np.exp(-((wl - 2200.0) ** 2) / (2.0 * 100.0**2))
        )
        return feat.astype(np.float32)

    @staticmethod
    def _as_spectral_feature(wl: np.ndarray) -> np.ndarray:
        feat = (
            np.exp(-((wl - 520.0) ** 2) / (2.0 * 35.0**2))
            - 0.3 * np.exp(-((wl - 850.0) ** 2) / (2.0 * 55.0**2))
            + 0.2 * np.exp(-((wl - 1650.0) ** 2) / (2.0 * 75.0**2))
        )
        return feat.astype(np.float32)

    def __len__(self) -> int:
        return self.spectra.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        spectrum = torch.from_numpy(self.spectra[idx]).unsqueeze(0)
        label = torch.from_numpy(self.labels[idx])
        return spectrum, label


class MultiTaskRegressionLoss(nn.Module):
    """多任务联合损失函数

    L = w_l1 * L1 + w_mse * MSE + w_cc * (1 - corr)
    """

    def __init__(
        self,
        w_l1: float = 0.5,
        w_mse: float = 0.4,
        w_cc: float = 0.1,
        metal_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.w_l1 = w_l1
        self.w_mse = w_mse
        self.w_cc = w_cc
        self.metal_weights = metal_weights or {"Cd": 1.5, "Pb": 1.0, "As": 1.2}

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total_loss = 0.0
        metric_dict = {}
        for i, metal in enumerate(TARGET_METALS):
            pred = preds[metal]
            tgt = targets[:, i]
            w = self.metal_weights.get(metal, 1.0)

            l1 = F.l1_loss(pred, tgt)
            mse = F.mse_loss(pred, tgt)
            pred_std = pred.std() + 1e-8
            tgt_std = tgt.std() + 1e-8
            corr = ((pred - pred.mean()) * (tgt - tgt.mean())).mean() / (pred_std * tgt_std)
            cc_loss = 1.0 - torch.clamp(corr, -1.0, 1.0)

            loss_m = w * (self.w_l1 * l1 + self.w_mse * mse + self.w_cc * cc_loss)
            total_loss = total_loss + loss_m

            metric_dict[f"{metal}_l1"] = float(l1.item())
            metric_dict[f"{metal}_mse"] = float(mse.item())
            metric_dict[f"{metal}_rmse"] = float(torch.sqrt(mse).item())
            metric_dict[f"{metal}_r2"] = float(
                (1.0 - mse / (tgt.var() + 1e-8)).clamp(0.0, 1.0).item()
            )

        total_loss = total_loss / len(TARGET_METALS)
        metric_dict["total_loss"] = float(total_loss.item())
        return total_loss, metric_dict


import torch.nn.functional as F


class Trainer:
    """模型训练器 - 封装完整训练流程"""

    def __init__(
        self,
        model: HyperspectralInversionNet,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = "./models",
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
    ):
        self.device = device
        self.model = model.to(device)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.criterion = MultiTaskRegressionLoss()
        self.optimizer = AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6
        )
        self.history: List[Dict] = []
        self.best_val_loss = float("inf")

    def _run_epoch(
        self,
        loader: DataLoader,
        training: bool = True,
    ) -> Dict[str, float]:
        self.model.train(mode=training)
        running_metrics: Dict[str, float] = {}
        num_batches = 0

        for spectra, labels in loader:
            spectra = spectra.to(self.device)
            labels = labels.to(self.device)

            if training:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(training):
                preds = self.model(spectra)
                loss, batch_metrics = self.criterion(preds, labels)
                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    self.optimizer.step()

            for k, v in batch_metrics.items():
                running_metrics[k] = running_metrics.get(k, 0.0) + v
            num_batches += 1

        return {k: v / num_batches for k, v in running_metrics.items()}

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 100,
        early_stopping_patience: int = 20,
    ) -> Dict:
        no_improve = 0
        best_path = os.path.join(self.output_dir, "best_model.pt")

        for epoch in range(1, num_epochs + 1):
            t0 = time.time()
            train_metrics = self._run_epoch(train_loader, training=True)
            val_metrics = self._run_epoch(val_loader, training=False)
            dt = time.time() - t0

            self.scheduler.step(val_metrics["total_loss"])

            epoch_record = {
                "epoch": epoch,
                "time_s": round(dt, 2),
                "train": train_metrics,
                "val": val_metrics,
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            self.history.append(epoch_record)

            print(
                f"[Epoch {epoch:3d}] t={dt:.1f}s | "
                f"train_loss={train_metrics['total_loss']:.4f} "
                f"val_loss={val_metrics['total_loss']:.4f} | "
                f"Cd_R2={val_metrics['Cd_r2']:.3f} "
                f"Pb_R2={val_metrics['Pb_r2']:.3f} "
                f"As_R2={val_metrics['As_r2']:.3f}"
            )

            if val_metrics["total_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["total_loss"]
                no_improve = 0
                self._save_checkpoint(best_path, epoch)
            else:
                no_improve += 1
                if no_improve >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        self._save_history()
        return {"best_val_loss": self.best_val_loss, "epochs": epoch}

    def _save_checkpoint(self, path: str, epoch: int) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_loss": self.best_val_loss,
                "config": {
                    "spectral_length": self.model.spectral_length,
                },
            },
            path,
        )
        print(f"  -> saved checkpoint: {path}")

    def _save_history(self) -> None:
        with open(os.path.join(self.output_dir, "training_history.json"), "w") as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)


def export_to_onnx(
    model: HyperspectralInversionNet,
    onnx_path: str,
    spectral_length: int = 330,
    device: str = "cpu",
    opset_version: int = 14,
) -> str:
    """将PyTorch模型导出为ONNX Runtime格式

    Args:
        model: 已加载权重的PyTorch模型
        onnx_path: 输出ONNX文件路径
        spectral_length: 光谱长度
        device: 运行设备
        opset_version: ONNX opset版本

    Returns:
        导出文件的绝对路径
    """
    model.eval()
    model.to(device)
    dummy_input = torch.randn(1, 1, spectral_length, device=device)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["spectrum"],
            output_names=["Cd", "Pb", "As"],
            dynamic_axes={
                "spectrum": {0: "batch_size"},
                "Cd": {0: "batch_size"},
                "Pb": {0: "batch_size"},
                "As": {0: "batch_size"},
            },
            opset_version=opset_version,
            do_constant_folding=True,
            verbose=False,
        )

    abs_path = os.path.abspath(onnx_path)
    print(f"ONNX model exported: {abs_path}")
    return abs_path


def run_training_pipeline(
    output_dir: str = "./models",
    num_samples: int = 8000,
    batch_size: int = 64,
    num_epochs: int = 80,
    spectral_length: int = 330,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """端到端训练管线入口

    生成合成数据 -> 训练 -> 保存权重 -> 导出ONNX
    """
    print(f"Starting training pipeline on {device}...")
    os.makedirs(output_dir, exist_ok=True)

    full_ds = SyntheticHyperspectralDataset(num_samples=num_samples, spectral_length=spectral_length)
    train_size = int(0.8 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model(spectral_length=spectral_length, device=device)
    print(f"Model params: {model.count_parameters():,}")

    trainer = Trainer(model, device=device, output_dir=output_dir)
    result = trainer.train(train_loader, val_loader, num_epochs=num_epochs)

    best_pt = os.path.join(output_dir, "best_model.pt")
    final_model = build_model(spectral_length=spectral_length, pretrained_path=best_pt, device=device)
    onnx_path = os.path.join(output_dir, "hyperspectral_inversion.onnx")
    export_to_onnx(final_model, onnx_path, spectral_length=spectral_length, device=device)

    print(f"Training complete. Best val loss: {result['best_val_loss']:.4f}")
    return onnx_path
