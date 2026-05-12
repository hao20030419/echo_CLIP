import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Train a ResNet binary classifier on extracted echo frames.")
    parser.add_argument("--data-root", type=str, default="data_frames", help="Root with train/val/test subfolders.")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--model-name", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--freeze-backbone", action="store_true", help="Only train final FC layer.")
    return parser.parse_args()


def build_model(model_name: str, num_classes: int, freeze_backbone: bool):
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif model_name == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def accuracy_from_logits(logits, targets):
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


def format_metric(value):
    return "N/A" if value is None else f"{value:.4f}"


def compute_auc_f1(labels, preds, pos_probs):
    if len(labels) == 0 or len(preds) == 0:
        return None, None

    f1 = f1_score(labels, preds, zero_division=0)
    auc = None
    if len(pos_probs) == len(labels) and len(set(labels)) > 1:
        auc = roc_auc_score(labels, pos_probs)
    return auc, f1


def plot_and_save_confusion_matrix(cm, class_names, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    max_val = max(max(row) for row in cm) if cm else 0
    threshold = max_val / 2.0 if max_val > 0 else 0.0
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            ax.text(
                j,
                i,
                str(cm[i][j]),
                ha="center",
                va="center",
                color="white" if cm[i][j] > threshold else "black",
            )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_one_train_epoch(model, loader, criterion, optimizer, device):
    model.train()

    epoch_loss = 0.0
    epoch_acc = 0.0
    total = 0
    all_labels = []
    all_preds = []
    all_pos_probs = []

    progress = tqdm(loader, desc="train", leave=False)

    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)
        probs = F.softmax(logits, dim=1)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)
        batch_size = labels.size(0)
        epoch_loss += loss.item() * batch_size
        epoch_acc += accuracy_from_logits(logits, labels) * batch_size
        total += batch_size
        all_labels.extend(labels.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())
        if probs.shape[1] == 2:
            all_pos_probs.extend(probs[:, 1].detach().cpu().tolist())

    if total == 0:
        return {
            "loss": 0.0,
            "acc": 0.0,
            "labels": [],
            "preds": [],
            "pos_probs": [],
        }

    return {
        "loss": epoch_loss / total,
        "acc": epoch_acc / total,
        "labels": all_labels,
        "preds": all_preds,
        "pos_probs": all_pos_probs,
    }


def evaluate_loader(model, loader, criterion, device, desc="eval"):
    model.eval()

    epoch_loss = 0.0
    epoch_acc = 0.0
    total = 0
    all_labels = []
    all_preds = []
    all_pos_probs = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            preds = torch.argmax(logits, dim=1)
            probs = F.softmax(logits, dim=1)

            batch_size = labels.size(0)
            epoch_loss += loss.item() * batch_size
            epoch_acc += (preds == labels).float().mean().item() * batch_size
            total += batch_size

            all_labels.extend(labels.detach().cpu().tolist())
            all_preds.extend(preds.detach().cpu().tolist())

            # For binary classification: probability of class index 1.
            if probs.shape[1] == 2:
                all_pos_probs.extend(probs[:, 1].detach().cpu().tolist())

    if total == 0:
        return {
            "loss": 0.0,
            "acc": 0.0,
            "labels": [],
            "preds": [],
            "pos_probs": [],
        }

    return {
        "loss": epoch_loss / total,
        "acc": epoch_acc / total,
        "labels": all_labels,
        "preds": all_preds,
        "pos_probs": all_pos_probs,
    }


def compute_final_metrics(labels, preds, pos_probs):
    metrics = {}
    metrics["f1"] = f1_score(labels, preds, zero_division=0)
    metrics["confusion_matrix"] = confusion_matrix(labels, preds).tolist()

    if len(pos_probs) == len(labels) and len(set(labels)) > 1:
        metrics["auc"] = roc_auc_score(labels, pos_probs)
    else:
        metrics["auc"] = None

    return metrics


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_root)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_dir = data_root / "train"
    val_dir = data_root / "val"
    test_dir = data_root / "test"

    if not train_dir.exists():
        raise FileNotFoundError(f"Missing train folder: {train_dir}")

    train_tfms = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=7),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tfms)
    val_ds = datasets.ImageFolder(str(val_dir), transform=eval_tfms) if val_dir.exists() else None
    test_ds = datasets.ImageFolder(str(test_dir), transform=eval_tfms) if test_dir.exists() else None

    if len(train_ds) == 0:
        raise RuntimeError("No training images found in train split")

    print(f"Classes: {train_ds.classes}")
    print(f"Train images: {len(train_ds)}")
    print(f"Val images: {len(val_ds) if val_ds is not None else 0}")
    print(f"Test images: {len(test_ds) if test_ds is not None else 0}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        if val_ds is not None and len(val_ds) > 0
        else None
    )
    test_loader = (
        DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        if test_ds is not None and len(test_ds) > 0
        else None
    )

    model = build_model(args.model_name, num_classes=len(train_ds.classes), freeze_backbone=args.freeze_backbone)
    model = model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    best_val_acc = -1.0
    best_model_path = save_dir / f"{args.model_name}_best.pt"

    for epoch in range(1, args.epochs + 1):
        train_result = run_one_train_epoch(model, train_loader, criterion, optimizer, device)
        train_loss = train_result["loss"]
        train_acc = train_result["acc"]
        train_auc, train_f1 = compute_auc_f1(
            train_result["labels"],
            train_result["preds"],
            train_result["pos_probs"],
        )

        if val_loader is not None:
            val_result = evaluate_loader(model, val_loader, criterion, device, desc="val")
            val_loss = val_result["loss"]
            val_acc = val_result["acc"]
            val_auc, val_f1 = compute_auc_f1(
                val_result["labels"],
                val_result["preds"],
                val_result["pos_probs"],
            )
        else:
            val_loss, val_acc = 0.0, train_acc
            val_auc, val_f1 = None, None

        if test_loader is not None:
            test_result_epoch = evaluate_loader(model, test_loader, criterion, device, desc="test")
            test_loss = test_result_epoch["loss"]
            test_acc = test_result_epoch["acc"]
            test_auc, test_f1 = compute_auc_f1(
                test_result_epoch["labels"],
                test_result_epoch["preds"],
                test_result_epoch["pos_probs"],
            )
        else:
            test_loss, test_acc = 0.0, 0.0
            test_auc, test_f1 = None, None

        scheduler.step()

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_auc={format_metric(train_auc)} train_f1={format_metric(train_f1)} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_auc={format_metric(val_auc)} val_f1={format_metric(val_f1)} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} test_auc={format_metric(test_auc)} test_f1={format_metric(test_f1)}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "classes": train_ds.classes,
                    "best_val_acc": best_val_acc,
                    "args": vars(args),
                },
                best_model_path,
            )

    print(f"Best val acc: {best_val_acc:.4f}")
    print(f"Saved best checkpoint: {best_model_path}")

    if test_loader is not None:
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        final_test_result = evaluate_loader(model, test_loader, criterion, device, desc="test-final")
        final_metrics = compute_final_metrics(
            labels=final_test_result["labels"],
            preds=final_test_result["preds"],
            pos_probs=final_test_result["pos_probs"],
        )

        print(f"Final test loss: {final_test_result['loss']:.4f}")
        print(f"Final test accuracy: {final_test_result['acc']:.4f}")

        if final_metrics["auc"] is None:
            print("Final test AUC: N/A (need binary labels in both classes and valid probability scores)")
        else:
            print(f"Final test AUC: {final_metrics['auc']:.4f}")

        print(f"Final test F1: {final_metrics['f1']:.4f}")
        print("Final test confusion matrix [ [TN, FP], [FN, TP] ]:")
        print(final_metrics["confusion_matrix"])

        cm_png_path = save_dir / "confusion_matrix_test.png"
        plot_and_save_confusion_matrix(final_metrics["confusion_matrix"], train_ds.classes, cm_png_path)
        print(f"Saved confusion matrix PNG: {cm_png_path}")

    elif val_loader is not None:
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        final_val_result = evaluate_loader(model, val_loader, criterion, device, desc="val-final")
        val_metrics = compute_final_metrics(
            labels=final_val_result["labels"],
            preds=final_val_result["preds"],
            pos_probs=final_val_result["pos_probs"],
        )
        cm_png_path = save_dir / "confusion_matrix_val.png"
        plot_and_save_confusion_matrix(val_metrics["confusion_matrix"], train_ds.classes, cm_png_path)
        print(f"Saved confusion matrix PNG: {cm_png_path}")


if __name__ == "__main__":
    main()
