from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _linear_beta_schedule(timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


def _sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half_dim = dim // 2
    if half_dim == 0:
        return timesteps.float().unsqueeze(-1)
    scale = math.log(10000.0) / max(half_dim - 1, 1)
    freq = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -scale)
    angles = timesteps.float().unsqueeze(1) * freq.unsqueeze(0)
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros((emb.shape[0], 1), device=timesteps.device)], dim=1)
    return emb


class _ConditionalDenoiser(nn.Module):
    def __init__(self, x_dim: int, hidden_dim: int, depth: int, time_dim: int = 16):
        super().__init__()
        self.time_dim = time_dim
        input_dim = x_dim + 1 + time_dim
        layers = []
        current_dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.SiLU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, y_noisy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = _sinusoidal_embedding(t, self.time_dim)
        inputs = torch.cat([x, y_noisy, t_emb], dim=1)
        return self.network(inputs)

    def feature_importance(self, x_dim: int) -> np.ndarray:
        first = next(module for module in self.network if isinstance(module, nn.Linear))
        weights = first.weight[:, :x_dim].detach().cpu().numpy()
        return np.abs(weights).mean(axis=0)


@dataclass
class DDPMTrainingHistory:
    final_loss: float
    residual_std: float


class ConditionalTabularDDPMRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        hidden_dim: int = 64,
        depth: int = 2,
        timesteps: int = 16,
        epochs: int = 20,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        ridge_alpha: float = 1e-3,
        gradient_clip: float = 1.0,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        predict_samples: int = 16,
        clip_value: float = 5.0,
        device: str = "cpu",
        random_state: int = 7,
        verbose: bool = False,
    ):
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.timesteps = timesteps
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.ridge_alpha = ridge_alpha
        self.gradient_clip = gradient_clip
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.predict_samples = predict_samples
        self.clip_value = clip_value
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    def _set_seed(self) -> None:
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ConditionalTabularDDPMRegressor":
        self._set_seed()
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        self.x_dim_ = X.shape[1]
        x_aug = np.concatenate([np.ones((X.shape[0], 1), dtype=np.float32), X], axis=1)
        x_aug64 = x_aug.astype(np.float64)
        y64 = y.astype(np.float64)
        xtx = x_aug64.T @ x_aug64
        xty = x_aug64.T @ y64
        alpha = max(float(self.ridge_alpha), 0.0)
        for _ in range(8):
            ridge = alpha * np.eye(x_aug64.shape[1], dtype=np.float64)
            ridge[0, 0] = 0.0
            try:
                self.linear_coef_ = np.linalg.solve(xtx + ridge, xty).astype(np.float32)
                break
            except np.linalg.LinAlgError:
                alpha = 1e-8 if alpha == 0.0 else alpha * 10.0
        else:
            self.linear_coef_ = np.linalg.lstsq(xtx + ridge, xty, rcond=None)[0].astype(np.float32)
        baseline = x_aug @ self.linear_coef_
        residual = y - baseline
        self.residual_mean_ = float(residual.mean())
        self.residual_std_ = float(residual.std()) if float(residual.std()) > 1e-8 else 1.0
        y_scaled = np.clip((residual - self.residual_mean_) / self.residual_std_, -self.clip_value, self.clip_value)

        self.device_ = torch.device(self.device)
        self.model_ = _ConditionalDenoiser(
            x_dim=self.x_dim_,
            hidden_dim=self.hidden_dim,
            depth=self.depth,
        ).to(self.device_)
        self.betas_ = _linear_beta_schedule(self.timesteps, self.beta_start, self.beta_end).to(self.device_)
        self.alphas_ = 1.0 - self.betas_
        self.alpha_bars_ = torch.cumprod(self.alphas_, dim=0)

        dataset = TensorDataset(
            torch.from_numpy(X),
            torch.from_numpy(y_scaled.astype(np.float32)),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        loss_value = float("nan")

        self.model_.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device_)
                batch_y = batch_y.to(self.device_)
                t = torch.randint(0, self.timesteps, (batch_x.shape[0],), device=self.device_)
                noise = torch.randn_like(batch_y)
                alpha_bar = self.alpha_bars_.gather(0, t).unsqueeze(1)
                y_noisy = torch.sqrt(alpha_bar) * batch_y + torch.sqrt(1.0 - alpha_bar) * noise
                noise_pred = self.model_(batch_x, y_noisy, t)
                loss = torch.mean((noise - noise_pred) ** 2)
                optimizer.zero_grad()
                loss.backward()
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model_.parameters(), self.gradient_clip)
                optimizer.step()
                loss_value = float(loss.detach().cpu().item())

        self.training_history_ = DDPMTrainingHistory(final_loss=loss_value, residual_std=self.residual_std_)
        self.feature_importances_ = self.model_.feature_importance(self.x_dim_)
        return self

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        x_tensor = torch.from_numpy(X).to(self.device_)
        x_aug = np.concatenate([np.ones((X.shape[0], 1), dtype=np.float32), X], axis=1)
        baseline = (x_aug @ self.linear_coef_).reshape(-1)
        self.model_.eval()
        paths = []

        for _ in range(self.predict_samples):
            y_t = torch.randn((x_tensor.shape[0], 1), device=self.device_)
            for step in reversed(range(self.timesteps)):
                t = torch.full((x_tensor.shape[0],), step, device=self.device_, dtype=torch.long)
                noise_pred = self.model_(x_tensor, y_t, t)
                alpha_t = self.alphas_[step]
                alpha_bar_t = self.alpha_bars_[step]
                beta_t = self.betas_[step]
                mean = (y_t - beta_t / torch.sqrt(1.0 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_t)
                if step > 0:
                    y_t = mean + torch.sqrt(beta_t) * torch.randn_like(y_t)
                else:
                    y_t = mean
            paths.append(y_t.detach().cpu().numpy())

        pred_scaled = np.mean(np.stack(paths, axis=0), axis=0).reshape(-1)
        residual_pred = pred_scaled * self.residual_std_ + self.residual_mean_
        return (baseline + residual_pred).astype(float)
