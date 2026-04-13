"""
train.py — PyTorch Image Classification Trainer
================================================
Supports: MobileNetV3, EfficientNetV2, ConvNeXt-Tiny, ResNet (via timm)
Features: TensorBoard, Checkpoint, Early Stopping, LR Scheduler
"""

import os
import sys
import time
import json
import shutil
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

try:
    import timm
except ImportError:
    sys.exit("❌  請先安裝 timm：pip install timm")

try:
    from tqdm import tqdm
except ImportError:
    sys.exit("❌  請先安裝 tqdm：pip install tqdm")

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Model aliases → timm names
# ──────────────────────────────────────────────
MODEL_ALIASES: Dict[str, str] = {
    # MobileNetV3
    "mobilenetv3":           "mobilenetv3_large_100",
    "mobilenetv3_large":     "mobilenetv3_large_100",
    "mobilenetv3_small":     "mobilenetv3_small_100",
    # EfficientNetV2
    "efficientnetv2":        "efficientnetv2_s",
    "efficientnetv2_s":      "efficientnetv2_s",
    "efficientnetv2_m":      "efficientnetv2_m",
    "efficientnetv2_l":      "efficientnetv2_l",
    # EfficientViT (MSRA)
    "efficientvit_m0":       "efficientvit_m0",
    "efficientvit_m1":       "efficientvit_m1",
    "efficientvit_m2":       "efficientvit_m2",
    "efficientvit_m3":       "efficientvit_m3",
    "efficientvit_m4":       "efficientvit_m4",
    "efficientvit_m5":       "efficientvit_m5",
    # ConvNeXt
    "convnext":              "convnext_tiny",
    "convnext_tiny":         "convnext_tiny",
    "convnext_small":        "convnext_small",
    "convnext_base":         "convnext_base",
    # ResNet
    "resnet":                "resnet50",
    "resnet18":              "resnet18",
    "resnet34":              "resnet34",
    "resnet50":              "resnet50",
    "resnet101":             "resnet101",
    "resnet152":             "resnet152",
}


def resolve_model_name(name: str) -> str:
    key = name.lower().replace("-", "_")
    return MODEL_ALIASES.get(key, name)   # fallback: pass directly to timm


# ──────────────────────────────────────────────
# Early Stopping
# ──────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "min"):
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        self.counter   = 0
        self.best      = float("inf") if mode == "min" else -float("inf")
        self.triggered = False

    def step(self, metric: float) -> bool:
        improved = (
            metric < self.best - self.min_delta if self.mode == "min"
            else metric > self.best + self.min_delta
        )
        if improved:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered


# ──────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────
def save_checkpoint(state: dict, path: Path, is_best: bool) -> None:
    torch.save(state, path / "last.pth")
    if is_best:
        shutil.copyfile(path / "last.pth", path / "best.pth")
        log.info("  ✅  Best checkpoint updated")


def load_checkpoint(path: Path, model: nn.Module, optimizer=None, scheduler=None) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_val    = ckpt.get("best_val", float("inf"))
    log.info(f"  📂  Resumed from epoch {ckpt['epoch']}  |  best_val={best_val:.4f}")
    return start_epoch, best_val


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
def build_dataloaders(
    data_dir:   str,
    img_size:   int,
    batch_size: int,
    val_split:  float,
    num_workers: int,
    augment:    bool,
) -> Tuple[DataLoader, DataLoader, int]:

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        *(
            [
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                transforms.RandomGrayscale(p=0.2),
            ] if augment else []
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    full_dataset = datasets.ImageFolder(data_dir, transform=train_tf)
    num_classes  = len(full_dataset.classes)
    log.info(f"  📁  Dataset: {len(full_dataset)} images, {num_classes} classes")

    n_val   = int(len(full_dataset) * val_split)
    n_train = len(full_dataset) - n_val
    if n_train <= 0:
        raise ValueError(
            f"訓練樣本數為 0（總樣本={len(full_dataset)}, val_split={val_split}）。"
            "請增加資料或調低 --val-split。"
        )
    if n_val <= 0:
        raise ValueError(
            f"驗證樣本數為 0（總樣本={len(full_dataset)}, val_split={val_split}）。"
            "請增加資料或調高 --val-split。"
        )

    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    # Apply val transform to val subset
    val_ds.dataset = datasets.ImageFolder(data_dir, transform=val_tf)

    # batch_size 不得大於訓練集樣本數（否則 drop_last=True 會產生空 DataLoader）
    if batch_size > n_train:
        log.warning(
            f"  ⚠️  batch_size ({batch_size}) 大於訓練樣本數 ({n_train})，"
            f"自動縮減為 {n_train}"
        )
        batch_size = n_train

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, num_classes


# ──────────────────────────────────────────────
# Auto batch size
# ──────────────────────────────────────────────
def auto_batch_size(
    model: nn.Module,
    img_size: int,
    device: torch.device,
    start: int = 8,
    max_bs: int = 1024,
) -> int:
    """
    Forward + backward 測試，從 start 開始倍增 batch size，
    直到 OOM 為止，回傳最後成功的值。
    CPU 環境直接回傳保守預設值 8。
    """
    if device.type == "cpu":
        log.info("  ℹ️   CPU 環境 — 自動 batch size 設為 8")
        return 8

    log.info("  🔍  自動偵測 batch size（forward + backward 壓力測試）...")
    _criterion = nn.CrossEntropyLoss()
    model.train()

    bs      = start
    last_ok = start

    while bs <= max_bs:
        try:
            dummy_x = torch.zeros(bs, 3, img_size, img_size, device=device)
            dummy_y = torch.zeros(bs, dtype=torch.long, device=device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                logits = model(dummy_x)
                loss   = _criterion(logits, dummy_y)
            loss.backward()
            model.zero_grad(set_to_none=True)
            del dummy_x, dummy_y, logits, loss
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()
            last_ok = bs
            bs *= 2
        except RuntimeError as exc:
            model.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()
            if "out of memory" in str(exc).lower():
                break
            raise  # 非 OOM 錯誤則往上拋

    log.info(f"  ✅  自動 batch size = {last_ok}  (測試上限: {bs // 2})")
    return last_ok


# ──────────────────────────────────────────────
# One epoch helpers
# ──────────────────────────────────────────────
def run_epoch(
    model, loader, criterion, optimizer, device, is_train: bool,
    epoch: int = 0, num_epochs: int = 0,
) -> Tuple[float, float]:
    model.train() if is_train else model.eval()
    total_loss = correct = total = 0
    phase_tag  = "Train" if is_train else "Val  "
    lr_now     = optimizer.param_groups[0]["lr"] if optimizer else None

    pbar = tqdm(
        loader,
        desc=f"  Epoch [{epoch+1:>{len(str(num_epochs))}}/{num_epochs}] {phase_tag}",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
    )

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model(imgs)
                loss   = criterion(logits, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += labels.size(0)

            # 即時更新進度條後綴
            postfix = {
                "loss": f"{total_loss / total:.4f}",
                "acc":  f"{correct / total * 100:.2f}%",
            }
            if is_train and lr_now is not None:
                postfix["lr"] = f"{lr_now:.2e}"
            pbar.set_postfix(postfix)

    pbar.close()

    if total == 0:
        phase = "訓練" if is_train else "驗證"
        raise RuntimeError(
            f"{phase} DataLoader 沒有產生任何批次！\n"
            "  常見原因：batch_size 仍大於樣本數，或資料集資料夾結構不正確。\n"
            "  請確認 --data-dir 下有正確的 class 子資料夾，並考慮縮小 --batch-size。"
        )
    return total_loss / total, correct / total


# ──────────────────────────────────────────────
# Build scheduler
# ──────────────────────────────────────────────
def build_scheduler(name: str, optimizer, epochs: int, steps_per_epoch: int):
    name = name.lower()
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif name == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1)
    elif name == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    elif name == "onecycle":
        return optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=optimizer.param_groups[0]["lr"] * 10,
            epochs=epochs, steps_per_epoch=steps_per_epoch,
        )
    elif name == "warmup_cosine":
        from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
        warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=5)
        cosine = CosineAnnealingLR(optimizer, T_max=epochs - 5)
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])
    else:
        raise ValueError(f"Unknown scheduler: {name}")


# ──────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────
def train(args):
    # ── device ──
    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else
              "mps"  if torch.backends.mps.is_available() else "cpu")
    )
    log.info(f"  🖥   Device: {device}")

    # ── output dir ──
    run_name = f"{args.model}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir  = Path(args.output_dir) / run_name
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"  📂  Run dir: {out_dir}")

    # ── num_classes（快速掃描，供建立模型使用）──
    num_classes = len(datasets.ImageFolder(args.data_dir).classes)
    log.info(f"  📁  Dataset: {args.data_dir}，共 {num_classes} 類別")

    # ── model ──
    timm_name = resolve_model_name(args.model)
    log.info(f"  🧠  Model: {args.model} → timm:{timm_name}")

    if args.pretrained_path:
        # 使用本機權重：先建立隨機初始化的模型，再載入本機檔案
        pretrained_path = Path(args.pretrained_path)
        if not pretrained_path.is_file():
            raise FileNotFoundError(f"--pretrained-path 檔案不存在: {pretrained_path}")
        log.info(f"  📂  載入本機預訓練權重: {pretrained_path}")
        model = timm.create_model(
            timm_name,
            pretrained=False,
            checkpoint_path=str(pretrained_path),
            num_classes=num_classes,
        ).to(device)
        log.info("  ✅  本機權重載入完成")
    else:
        use_pretrain = not args.no_pretrain
        try:
            model = timm.create_model(
                timm_name,
                pretrained=use_pretrain,
                num_classes=num_classes,
            ).to(device)
        except RuntimeError as exc:
            if use_pretrain and any(k in str(exc).lower() for k in
                                    ("403", "forbidden", "download", "reconstruct", "out of memory")):
                log.warning(
                    "  ⚠️  預訓練權重下載失敗，已改用隨機初始化。\n"
                    "      解決方式：\n"
                    "        1. huggingface-cli login\n"
                    "        2. 手動下載後用 --pretrained-path 指定本機路徑"
                )
                model = timm.create_model(
                    timm_name, pretrained=False, num_classes=num_classes
                ).to(device)
            else:
                raise

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  📐  Trainable params: {n_params:,}")

    # ── auto batch size ──
    if args.batch_size == -1:
        args.batch_size = auto_batch_size(model, args.img_size, device)
    log.info(f"  📦  Batch size: {args.batch_size}")

    # ── data ──
    train_loader, val_loader, _ = build_dataloaders(
        args.data_dir, args.img_size, args.batch_size,
        args.val_split, args.num_workers, not args.no_augment,
    )

    # ── loss / optimizer ──
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = {
        "adamw": optim.AdamW,
        "adam":  optim.Adam,
        "sgd":   lambda p, **kw: optim.SGD(p, momentum=0.9, nesterov=True, **kw),
    }[args.optimizer](model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ── scheduler ──
    scheduler = build_scheduler(args.scheduler, optimizer, args.epochs, len(train_loader))
    plateau_mode = isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau)

    # ── early stopping ──
    early_stop = EarlyStopping(patience=args.patience, mode="min")

    # ── tensorboard ──
    writer = SummaryWriter(log_dir=str(out_dir / "tb_logs"))

    # ── resume ──
    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume:
        start_epoch, best_val_loss = load_checkpoint(
            Path(args.resume), model, optimizer, scheduler
        )

    # ── save config ──
    cfg_path = out_dir / "config.json"
    cfg_path.write_text(json.dumps(vars(args), indent=2))

    # ──────────────────────────────────────────
    # Training loop
    # ──────────────────────────────────────────
    log.info("=" * 60)
    log.info(f"  🚀  Training for {args.epochs} epochs")
    log.info("=" * 60)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device,
            is_train=True, epoch=epoch, num_epochs=args.epochs,
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, None, device,
            is_train=False, epoch=epoch, num_epochs=args.epochs,
        )

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]

        # ── scheduler step ──
        if plateau_mode:
            scheduler.step(val_loss)
        elif not isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
            # OneCycleLR steps per batch inside run_epoch — handled separately below
            scheduler.step()

        # ── tensorboard ──
        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": train_acc, "val": val_acc}, epoch)
        writer.add_scalar("LR", lr_now, epoch)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        # ── checkpoint ──
        save_checkpoint(
            {
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if not plateau_mode else {},
                "best_val":        best_val_loss,
                "args":            vars(args),
            },
            ckpt_dir,
            is_best,
        )

        best_mark = " ★ best" if is_best else ""
        log.info(
            f"  ✦ Epoch [{epoch+1:>{len(str(args.epochs))}}/{args.epochs}]"
            f"  train loss={train_loss:.4f}  acc={train_acc*100:.2f}%"
            f"  │  val loss={val_loss:.4f}  acc={val_acc*100:.2f}%"
            f"  │  lr={lr_now:.2e}  ⏱{elapsed:.1f}s"
            f"{best_mark}"
        )

        # ── early stopping ──
        if early_stop.step(val_loss):
            log.info(f"  🛑  Early stopping triggered at epoch {epoch+1}")
            break

    writer.close()
    log.info("=" * 60)
    log.info(f"  ✅  Training complete — best val_loss: {best_val_loss:.4f}")
    log.info(f"  📂  Checkpoints → {ckpt_dir}")
    log.info(f"  📊  TensorBoard  → tensorboard --logdir {out_dir / 'tb_logs'}")
    log.info("=" * 60)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="PyTorch Image Classification Trainer (timm)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── required ──
    p.add_argument("--data-dir",    required=True,
                   help="ImageFolder 根目錄 (含 class 子資料夾)")
    p.add_argument("--model",       required=True,
                   help=f"模型名稱，支援別名: {', '.join(MODEL_ALIASES.keys())}")

    # ── training ──
    g = p.add_argument_group("Training")
    g.add_argument("--epochs",          type=int,   default=50)
    g.add_argument("--batch-size",      type=int,   default=32,
                   help="Batch size；設為 -1 可依可用記憶體自動偵測")
    g.add_argument("--lr",              type=float, default=1e-3)
    g.add_argument("--weight-decay",    type=float, default=1e-4)
    g.add_argument("--label-smoothing", type=float, default=0.1)
    g.add_argument("--optimizer",       choices=["adamw", "adam", "sgd"], default="adamw")
    g.add_argument("--no-pretrain",      action="store_true",
                   help="不使用 ImageNet pretrained 權重")
    g.add_argument("--pretrained-path",  default=None,
                   help="本機預訓練權重路徑（.pth / .pt / .safetensors），"
                        "可取代從 HuggingFace 下載；設定此項時 --no-pretrain 自動忽略")
    g.add_argument("--no-augment",       action="store_true",
                   help="停用 ColorJitter / Grayscale 增強")

    # ── scheduler ──
    g2 = p.add_argument_group("LR Scheduler")
    g2.add_argument("--scheduler", default="cosine",
                    choices=["cosine", "step", "plateau", "onecycle", "warmup_cosine"])

    # ── early stopping ──
    g3 = p.add_argument_group("Early Stopping")
    g3.add_argument("--patience", type=int, default=10,
                    help="連續 N 個 epoch val_loss 無改善即停止")

    # ── data ──
    g4 = p.add_argument_group("Data")
    g4.add_argument("--img-size",    type=int,   default=224)
    g4.add_argument("--val-split",   type=float, default=0.2,
                    help="驗證集比例 (0~1)")
    g4.add_argument("--num-workers", type=int,   default=4)

    # ── misc ──
    g5 = p.add_argument_group("Misc")
    g5.add_argument("--device",     default="auto",
                    help="cuda / mps / cpu / auto")
    g5.add_argument("--output-dir", default="runs",
                    help="輸出根目錄")
    g5.add_argument("--resume",     default=None,
                    help="從 checkpoint .pth 恢復訓練")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
