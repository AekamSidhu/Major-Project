"""A small 1-D convolutional network that reads the gaze window directly.

The feature-based detectors look for return sweeps by rule. This network
gets the raw 10 fps window instead (horizontal and vertical gaze plus a
"frame was valid" flag) and learns its own cues. It is tiny (about 150k
weights) and trains on a laptop CPU in about ten minutes.

Input handling keeps the properties the features have:

- unit-free: each window is centred and divided by its own spread, so
  degrees, pixels and MediaPipe iris ratios all look the same;
- direction-free: training mirrors half the windows left-to-right, so a
  mirrored video or a right-to-left script is not a different case;
- noise-aware: training adds fresh webcam noise to clean eye-tracker windows
  every step, over a range of noise levels and text sizes.

Needs PyTorch (`pip install torch`), which the rest of `readgaze` does not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


def to_input(raw: np.ndarray) -> np.ndarray:
    """`(n, frames, 2)` gaze windows with NaN gaps -> `(n, 3, frames)` network input.

    Gaps are filled by linear interpolation (nearest value at the ends), then
    each window is centred on its median and divided by its overall spread.
    The third channel is 1 where the frame was valid.
    """
    raw = np.asarray(raw, dtype=np.float32)
    n, frames, _ = raw.shape
    valid = np.isfinite(raw).all(axis=2)
    filled = raw.copy()
    grid = np.arange(frames)
    for i in np.flatnonzero(~valid.all(axis=1)):
        ok = valid[i]
        if not ok.any():
            filled[i] = 0.0
            continue
        for c in range(2):
            filled[i, :, c] = np.interp(grid, grid[ok], raw[i, ok, c])
    centred = filled - np.median(filled, axis=1, keepdims=True)
    spread = np.sqrt((centred**2).sum(axis=2).mean(axis=1, keepdims=True))[:, :, None]
    scaled = np.clip(centred / np.maximum(spread, 1e-6), -6.0, 6.0)
    out = np.empty((n, 3, frames), dtype=np.float32)
    out[:, 0] = scaled[:, :, 0]
    out[:, 1] = scaled[:, :, 1]
    out[:, 2] = valid
    return out


@dataclass(frozen=True)
class Augment:
    """Ranges for the webcam noise added to clean windows during training."""

    noise: tuple[float, float] = (0.3, 3.5)
    """Per-frame jitter, in the units of the clean data (degrees for GazeBase)."""
    drift: tuple[float, float] = (0.0, 2.0)
    drift_seconds: float = 4.0
    amplitude: tuple[float, float] = (0.4, 1.3)
    """Eye-movement size: below 1 is text close to the camera or a narrow column."""
    blinks_per_minute: float = 15.0
    blink_seconds: float = 0.25
    dropout_probability: float = 0.02
    mirror: bool = True


def augment(clean: np.ndarray, fps: float, config: Augment, rng: np.random.Generator) -> np.ndarray:
    """Fresh webcam degradation for a batch of clean windows `(n, frames, 2)`."""
    n, frames, _ = clean.shape
    out = clean.astype(np.float32, copy=True)
    centre = np.nanmedian(out, axis=1, keepdims=True)
    amplitude = np.exp(rng.uniform(np.log(config.amplitude[0]), np.log(config.amplitude[1]), (n, 1, 1)))
    out = centre + amplitude * (out - centre)

    width = max(int(round(config.drift_seconds * fps)), 1)
    # Slow drift: white noise through a moving average (as simulate._smooth_noise).
    total = np.cumsum(rng.normal(0.0, 1.0, (n, frames + width, 2)), axis=1, dtype=np.float32)
    smooth = total[:, width:] - total[:, :-width]
    smooth /= np.maximum(smooth.std(axis=1, keepdims=True), 1e-9)
    out += smooth * rng.uniform(*config.drift, (n, 1, 1))
    out += rng.normal(0.0, 1.0, out.shape) * rng.uniform(*config.noise, (n, 1, 1))

    lost = rng.random((n, frames)) < config.dropout_probability
    blink_frames = max(int(round(config.blink_seconds * fps)), 1)
    expected = config.blinks_per_minute * frames / fps / 60.0
    for i, count in enumerate(rng.poisson(expected, n)):
        for start in rng.integers(0, frames, count):
            lost[i, start : start + blink_frames] = True
    out[lost] = np.nan
    if config.mirror:
        flip = rng.random(n) < 0.5
        out[flip, :, 0] *= -1.0
    return out


class _Block(nn.Module):
    """Two convolutions with a skip connection."""

    def __init__(self, c_in: int, c_out: int, kernel: int, dropout: float) -> None:
        super().__init__()
        pad = (0, kernel // 2)
        self.body = nn.Sequential(
            nn.Conv2d(c_in, c_out, (1, kernel), padding=pad),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(c_out, c_out, (1, kernel), padding=pad),
            nn.BatchNorm2d(c_out),
        )
        self.skip = nn.Identity() if c_in == c_out else nn.Conv2d(c_in, c_out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.skip(x) + self.body(x))


class ReadingNet(nn.Module):
    """Residual 1-D CNN: each stage halves the time resolution and widens,
    then mean and max pooling over time. The last stage sees about 12 s at
    10 fps, so one window fits in its view.

    The convolutions are written as 1 x k 2-D convolutions in channels-last
    layout only because that is 2-3x faster on a laptop CPU than `Conv1d`.
    """

    def __init__(self, widths: tuple[int, ...] = (32, 48, 64, 96), kernel: int = 5, dropout: float = 0.1) -> None:
        super().__init__()
        # 3 input channels plus the frame-to-frame change of x and y.
        layers: list[nn.Module] = [nn.Conv2d(5, widths[0], (1, 7), padding=(0, 3)), nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True)]
        c = widths[0]
        for i, width in enumerate(widths):
            if i:
                layers.append(nn.AvgPool2d((1, 2)))
            layers.append(_Block(c, width, kernel, dropout))
            c = width
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(2 * c, c), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(c, 1))
        self.to(memory_format=torch.channels_last)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Frame-to-frame change (0 for the first frame). Same as torch.diff
        # with the first frame prepended, written out so ONNX can export it.
        xy = x[:, :2]
        step = xy - torch.cat([xy[:, :, :1], xy[:, :, :-1]], dim=2)
        h = torch.cat([x, step], dim=1).unsqueeze(2).contiguous(memory_format=torch.channels_last)
        h = self.features(h)
        pooled = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)
        return self.head(pooled).squeeze(1)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 12
    batch: int = 256
    lr: float = 2e-3
    weight_decay: float = 1e-4
    widths: tuple[int, ...] = (32, 48, 64, 96)
    kernel: int = 5
    dropout: float = 0.1
    seed: int = 0
    threads: int = 4
    device: str = "auto"
    """"auto" uses a GPU when there is one."""


def train(clean: np.ndarray, labels: np.ndarray, fps: float, config: TrainConfig = TrainConfig(), aug: Augment = Augment(), log=None) -> ReadingNet:
    """Fit on clean windows `(n, frames, 2)`; noise is added fresh every step."""
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.threads)
    rng = np.random.default_rng(config.seed)
    device = _device(config.device)
    model = ReadingNet(config.widths, config.kernel, config.dropout).to(device)
    labels = np.asarray(labels, dtype=np.float32)
    positive = float(labels.mean())
    # Weight the classes equally, as class_weight="balanced" does for the other models.
    pos_weight = torch.tensor((1.0 - positive) / max(positive, 1e-6), device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    steps = config.epochs * int(np.ceil(len(clean) / config.batch))
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=config.lr, total_steps=steps, pct_start=0.15)

    for epoch in range(config.epochs):
        model.train()
        order = rng.permutation(len(clean))
        total, seen = 0.0, 0
        for start in range(0, len(order), config.batch):
            index = np.sort(order[start : start + config.batch])
            x = torch.from_numpy(to_input(augment(clean[index], fps, aug, rng))).to(device, non_blocking=True)
            y = torch.from_numpy(labels[index]).to(device, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimiser.step()
            schedule.step()
            total += float(loss.detach()) * len(index)
            seen += len(index)
        if log is not None:
            log(f"    epoch {epoch + 1}/{config.epochs}  loss {total / seen:.4f}")
    return model.eval()


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@torch.no_grad()
def predict(model: ReadingNet, raw: np.ndarray, batch: int = 2048) -> np.ndarray:
    """Reading probability for each (already noisy) window `(n, frames, 2)`."""
    model.eval()
    device = next(model.parameters()).device
    out = []
    for start in range(0, len(raw), batch):
        x = torch.from_numpy(to_input(np.asarray(raw[start : start + batch]))).to(device)
        out.append(torch.sigmoid(model(x)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)
