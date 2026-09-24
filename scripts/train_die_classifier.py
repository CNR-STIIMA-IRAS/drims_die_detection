"""
Train MobileNetV3-Small for Die Orientation & Double Face Detection with W&B Logging
===================================================================================
Predicts:
  - Head Top    : Classes 0-5  (Top Face 1-6)
  - Head Front  : Classes 0-6  (0: Single Face / None, 1-6: Front Face 1-6)
  - Head Double : Classes 0-1  (0: Single Face, 1: Double Face)
  - Head Joint  : Classes 0-29 (All 30 discrete physical orientations: 24 double + 6 single)

Logs live training metrics to Weights & Biases (wandb).
Exports CPU-portable TorchScript model to weights/die_mobilenet_v3.pt for deployment.
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.models import mobilenet_v3_small
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class Die30ClassDataset(Dataset):
    def __init__(self, data_dir, is_train=True):
        self.data_dir = data_dir
        self.is_train = is_train
        labels_file = os.path.join(data_dir, "labels.json")
        with open(labels_file, "r") as f:
            self.records = json.load(f)

        if is_train:
            self.transforms = T.Compose([
                T.ToPILImage(),
                T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
                T.RandomAffine(degrees=10, translate=(0.04, 0.04), scale=(0.95, 1.05)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transforms = T.Compose([
                T.ToPILImage(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_path = os.path.join(self.data_dir, rec["image_path"])
        bgr = cv2.imread(img_path)
        if bgr is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transforms(rgb)

        # Labels:
        # top_face: 1..6 -> 0..5
        top_label = torch.tensor(rec["top_face"] - 1, dtype=torch.long)
        # front_face: 0..6
        front_label = torch.tensor(rec["front_face"], dtype=torch.long)
        # double_face: 0 or 1
        double_label = torch.tensor(1 if rec.get("is_double_face", rec["front_face"] != 0) else 0, dtype=torch.long)
        # joint_class: 0..29
        joint_label = torch.tensor(rec.get("class_id", 0), dtype=torch.long)

        return tensor, top_label, front_label, double_label, joint_label


class MultiHeadMobileNetV3(nn.Module):
    """MobileNetV3-Small with decoupled heads for Top, Front, Double-face, and Joint class."""
    def __init__(self, pretrained=True):
        super().__init__()
        base = mobilenet_v3_small(pretrained=pretrained)
        self.features = base.features
        self.avgpool = base.avgpool
        feat_dim = 576

        self.head_top = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 6)
        )
        self.head_front = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 7)
        )
        self.head_double = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, 2)
        )
        self.head_joint = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 30)
        )

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = self.avgpool(x)
        feat = torch.flatten(x, 1)

        out_top = self.head_top(feat)
        out_front = self.head_front(feat)
        out_double = self.head_double(feat)
        out_joint = self.head_joint(feat)

        return out_top, out_front, out_double, out_joint


def train_model(
    dataset_dir="/home/ws/src/drims_die_detection/dataset",
    output_weights="/home/ws/src/drims_die_detection/weights/die_mobilenet_v3.pt",
    epochs=12,
    batch_size=64,
    lr=1e-3,
    device_str="cuda",
    use_wandb=False,
    wandb_project="die_orientation_detection",
    wandb_run_name="mobilenet_v3_double_face",
    wandb_mode="online",
):
    print("=" * 80)
    print("Training Multi-Head MobileNetV3-Small for Die Orientation & Double Face Detection")
    print(f"Dataset dir    : {dataset_dir}")
    print(f"Output weights : {output_weights}")
    print(f"Epochs         : {epochs}")
    print(f"Batch size     : {batch_size}")
    print(f"Learning rate  : {lr}")
    print(f"Device target  : {device_str}")
    print(f"W&B Logging    : {use_wandb} (Project: {wandb_project}, Mode: {wandb_mode})")
    print("=" * 80)

    # Device selection
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print(f"[Warning] CUDA requested ('{device_str}') but CUDA is not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    pin_mem = (device.type == "cuda")
    print(f"Using compute device: {device} (pin_memory={pin_mem})")

    # W&B Initialization
    wandb_run = None
    if use_wandb:
        if not HAS_WANDB:
            print("[Warning] wandb is not installed. Skipping W&B logging.")
        else:
            try:
                wandb_run = wandb.init(
                    project=wandb_project,
                    name=wandb_run_name,
                    mode=wandb_mode,
                    dir=PACKAGE_ROOT,  # run logs land in <pkg>/wandb/, not the CWD
                    config={
                        "architecture": "MultiHead-MobileNetV3-Small",
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "lr": lr,
                        "device": str(device),
                        "num_classes_joint": 30,
                        "num_double_classes": 24,
                        "num_single_classes": 6,
                    }
                )
                # Explicitly configure all metrics as line plots tracked across epochs
                wandb.define_metric("epoch")
                wandb.define_metric("train/*", step_metric="epoch")
                wandb.define_metric("val/*", step_metric="epoch")
                wandb.define_metric("learning_rate", step_metric="epoch")

                print(f"Initialized W&B Run: {wandb.run.name} ({wandb.run.url if wandb.run.url else 'local'})")
            except Exception as e:
                print(f"[Warning] Could not initialize W&B ({e}). Running in offline/no-wandb mode.")
                wandb_run = None

    train_set = Die30ClassDataset(os.path.join(dataset_dir, "train"), is_train=True)
    val_set = Die30ClassDataset(os.path.join(dataset_dir, "val"), is_train=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=pin_mem)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=pin_mem)

    print(f"Train samples: {len(train_set)} | Val samples: {len(val_set)}")

    model = MultiHeadMobileNetV3(pretrained=True).to(device)

    # Optimizer with differential learning rate
    optimizer = optim.AdamW([
        {"params": model.features.parameters(), "lr": lr * 0.2},
        {"params": model.head_top.parameters(), "lr": lr},
        {"params": model.head_front.parameters(), "lr": lr},
        {"params": model.head_double.parameters(), "lr": lr},
        {"params": model.head_joint.parameters(), "lr": lr},
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion_top = nn.CrossEntropyLoss()
    criterion_front = nn.CrossEntropyLoss()
    criterion_double = nn.CrossEntropyLoss()
    criterion_joint = nn.CrossEntropyLoss()

    best_val_score = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        train_top_correct = 0
        train_front_correct = 0
        train_double_correct = 0
        train_joint_correct = 0
        total_samples = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch:02d}/{epochs:02d}] Train", unit="batch", leave=False)
        for imgs, t_lbls, f_lbls, d_lbls, j_lbls in train_pbar:
            imgs = imgs.to(device)
            t_lbls = t_lbls.to(device)
            f_lbls = f_lbls.to(device)
            d_lbls = d_lbls.to(device)
            j_lbls = j_lbls.to(device)

            optimizer.zero_grad()
            out_top, out_front, out_double, out_joint = model(imgs)

            loss_t = criterion_top(out_top, t_lbls)
            loss_f = criterion_front(out_front, f_lbls)
            loss_d = criterion_double(out_double, d_lbls)
            loss_j = criterion_joint(out_joint, j_lbls)

            # Combined multi-task loss
            loss = loss_t + loss_f + 0.5 * loss_d + 0.5 * loss_j
            loss.backward()
            optimizer.step()

            bs = imgs.size(0)
            train_loss += loss.item() * bs
            train_top_correct += (out_top.argmax(dim=1) == t_lbls).sum().item()
            train_front_correct += (out_front.argmax(dim=1) == f_lbls).sum().item()
            train_double_correct += (out_double.argmax(dim=1) == d_lbls).sum().item()
            train_joint_correct += (out_joint.argmax(dim=1) == j_lbls).sum().item()
            total_samples += bs

            train_pbar.set_postfix({
                "loss": f"{loss.item():.3f}",
                "top": f"{train_top_correct / total_samples * 100:.1f}%",
                "dbl": f"{train_double_correct / total_samples * 100:.1f}%"
            })

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_top_correct = 0
        val_front_correct = 0
        val_double_correct = 0
        val_joint_correct = 0
        val_total = 0

        val_pbar = tqdm(val_loader, desc=f"Epoch [{epoch:02d}/{epochs:02d}] Val  ", unit="batch", leave=False)
        with torch.no_grad():
            for imgs, t_lbls, f_lbls, d_lbls, j_lbls in val_pbar:
                imgs = imgs.to(device)
                t_lbls = t_lbls.to(device)
                f_lbls = f_lbls.to(device)
                d_lbls = d_lbls.to(device)
                j_lbls = j_lbls.to(device)

                out_top, out_front, out_double, out_joint = model(imgs)

                loss_t = criterion_top(out_top, t_lbls)
                loss_f = criterion_front(out_front, f_lbls)
                loss_d = criterion_double(out_double, d_lbls)
                loss_j = criterion_joint(out_joint, j_lbls)
                v_loss = loss_t + loss_f + 0.5 * loss_d + 0.5 * loss_j

                bs = imgs.size(0)
                val_loss += v_loss.item() * bs
                val_top_correct += (out_top.argmax(dim=1) == t_lbls).sum().item()
                val_front_correct += (out_front.argmax(dim=1) == f_lbls).sum().item()
                val_double_correct += (out_double.argmax(dim=1) == d_lbls).sum().item()
                val_joint_correct += (out_joint.argmax(dim=1) == j_lbls).sum().item()
                val_total += bs

                val_pbar.set_postfix({
                    "val_loss": f"{v_loss.item():.3f}",
                    "top": f"{val_top_correct / val_total * 100:.1f}%"
                })

        elapsed = time.time() - t0
        top_acc = val_top_correct / val_total
        front_acc = val_front_correct / val_total
        double_acc = val_double_correct / val_total
        joint_acc = val_joint_correct / val_total
        val_loss_avg = val_loss / val_total
        train_loss_avg = train_loss / total_samples

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
            f"Loss: {train_loss_avg:.3f}/{val_loss_avg:.3f} | "
            f"Top Acc: {top_acc*100:.1f}% | "
            f"Front Acc: {front_acc*100:.1f}% | "
            f"Double Acc: {double_acc*100:.1f}% | "
            f"Joint 30-Cls: {joint_acc*100:.1f}%",
            flush=True
        )

        # Log to Weights & Biases (continuous line plots over epoch step)
        if wandb_run is not None:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss_avg,
                "val/loss": val_loss_avg,
                "val/top_accuracy": top_acc,
                "val/front_accuracy": front_acc,
                "val/double_face_accuracy": double_acc,
                "val/joint_pair_accuracy": joint_acc,
                "learning_rate": scheduler.get_last_lr()[0],
            }, step=epoch)

        # Save best model
        val_score = top_acc + front_acc + double_acc
        if val_score > best_val_score or epoch == epochs:
            best_val_score = val_score
            os.makedirs(os.path.dirname(output_weights), exist_ok=True)

            # Export CPU-portable TorchScript model for deployment
            model_export = MultiHeadMobileNetV3(pretrained=False)
            model_export.load_state_dict(model.state_dict())
            model_export.eval()
            example_cpu = torch.randn(1, 3, 224, 224, device="cpu")
            traced_model = torch.jit.trace(model_export, example_cpu)
            traced_model.save(output_weights)
            print(
                f"  -> Exported Best Model (Top: {top_acc*100:.1f}%, Front: {front_acc*100:.1f}%, Double: {double_acc*100:.1f}%) to {output_weights}",
                flush=True
            )

    if wandb_run is not None:
        wandb.finish()
        print("W&B Run completed and closed.")

    print("\nTraining completed successfully!")
    print(f"Final Model saved at: {output_weights}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MobileNetV3-Small for Die Orientation and Double Face Detection")
    parser.add_argument("--dataset_dir", default="/home/ws/src/drims_die_detection/dataset")
    parser.add_argument("--output_weights", default="/home/ws/src/drims_die_detection/weights/die_mobilenet_v3.pt")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda", help="Target device: 'cuda', 'cuda:0', or 'cpu'")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", default="die_orientation_detection", help="W&B project name")
    parser.add_argument("--wandb_run_name", default="mobilenet_v3_double_face", help="W&B run display name")
    parser.add_argument("--wandb_mode", default="online", choices=["online", "offline", "disabled"], help="W&B syncing mode")

    args = parser.parse_args()

    train_model(
        dataset_dir=args.dataset_dir,
        output_weights=args.output_weights,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_str=args.device,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_mode=args.wandb_mode,
    )
