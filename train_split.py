import os
import glob
import math
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import torchvision.transforms as T
from open_clip import create_model_and_transforms
from template_tokenizer import template_tokenize
from utils import read_avi
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class EchoFrameDataset(Dataset):
    def __init__(
        self,
        file_list,
        label_list,
        preprocess_val,
        frame_stride=1,
        max_frames_per_video=None,
    ):
        self.preprocess_val = preprocess_val
        self.frame_stride = max(1, frame_stride)
        self.max_frames_per_video = max_frames_per_video
        self.samples = []

        pos_text = "An echocardiogram showing aortic dissection."
        neg_text = "A normal echocardiogram without aortic dissection."
        self.pos_tokens = torch.tensor(template_tokenize(pos_text), dtype=torch.long)
        self.neg_tokens = torch.tensor(template_tokenize(neg_text), dtype=torch.long)

        print(f"Building frame-level samples from {len(file_list)} videos...")
        for file_path, label in tqdm(
            list(zip(file_list, label_list)), desc="Extract frames", leave=False
        ):
            try:
                frames = read_avi(file_path, (224, 224))
                frames = frames[:: self.frame_stride]
                if self.max_frames_per_video is not None:
                    frames = frames[: self.max_frames_per_video]

                for frame in frames:
                    self.samples.append((frame, int(label)))
            except Exception as e:
                print(f"[{file_path}] Read warning: {e}")

        pos_count = sum(1 for _, lbl in self.samples if lbl == 1)
        neg_count = len(self.samples) - pos_count
        print(
            f"Frame samples ready: total={len(self.samples)} | positive={pos_count} | negative={neg_count}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame, label = self.samples[idx]
        image = self.preprocess_val(T.ToPILImage()(frame))
        tokens = self.pos_tokens if label == 1 else self.neg_tokens
        return image, tokens, torch.tensor(label, dtype=torch.long)


class BalancedBinaryBatchSampler(Sampler):
    def __init__(self, labels, batch_size):
        if batch_size < 2:
            raise ValueError("batch_size must be >= 2 for balanced sampling")

        self.batch_size = batch_size if batch_size % 2 == 0 else batch_size - 1
        self.half = self.batch_size // 2

        self.pos_indices = [i for i, y in enumerate(labels) if y == 1]
        self.neg_indices = [i for i, y in enumerate(labels) if y == 0]

        if len(self.pos_indices) == 0 or len(self.neg_indices) == 0:
            raise ValueError("Both positive and negative samples are required for balanced batches")

        # Reuse smaller class with replacement to match the larger class.
        max_class_count = max(len(self.pos_indices), len(self.neg_indices))
        self.num_batches = math.ceil(max_class_count / self.half)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        pos_perm = torch.randperm(len(self.pos_indices)).tolist()
        neg_perm = torch.randperm(len(self.neg_indices)).tolist()

        for b in range(self.num_batches):
            start = b * self.half
            end = (b + 1) * self.half

            # Wrap-around so we can oversample minority class.
            pos_batch = [
                self.pos_indices[pos_perm[i % len(pos_perm)]]
                for i in range(start, end)
            ]
            neg_batch = [
                self.neg_indices[neg_perm[i % len(neg_perm)]]
                for i in range(start, end)
            ]

            batch = pos_batch + neg_batch
            batch_tensor = torch.tensor(batch)
            shuffle_order = torch.randperm(len(batch_tensor))
            yield batch_tensor[shuffle_order].tolist()


def contrastive_loss(image_embeddings, text_embeddings, logit_scale, batch_labels):
    """
    DPR-style in-batch objective for aortic dissection retrieval.
    - Query: dissection image embeddings only (label==1)
    - Positive passages: dissection text embeddings in the same batch
    - Negative passages: all non-dissection text embeddings in the same batch
    """
    image_embeddings = F.normalize(image_embeddings, dim=-1)
    text_embeddings = F.normalize(text_embeddings, dim=-1)

    logits = logit_scale * (image_embeddings @ text_embeddings.T)  # (B, B)

    # Only use dissection samples as queries.
    query_mask = batch_labels == 1
    if query_mask.sum() == 0:
        # No positive query in this batch; skip contribution safely.
        return logits.new_zeros((), requires_grad=True)

    query_logits = logits[query_mask]  # (Q, B)

    # Positives are all dissection texts; non-dissection texts are negatives.
    pos_mask = (batch_labels == 1).unsqueeze(0).expand(query_logits.size(0), -1)

    # log p(pos|query) = logsumexp(pos) - logsumexp(all)
    pos_logits = query_logits.masked_fill(~pos_mask, float("-inf"))
    log_pos = torch.logsumexp(pos_logits, dim=1)
    log_all = torch.logsumexp(query_logits, dim=1)
    loss = -(log_pos - log_all).mean()

    return loss


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading EchoCLIP-R model...")
    model, _, preprocess_val = create_model_and_transforms(
        "hf-hub:mkaichristensen/echo-clip-r", precision="bf16", device=device
    )

    logit_scale = torch.nn.Parameter(torch.ones([], device=device) * 2.6592)

    pos_dir = os.path.join("data", "positive")
    neg_dir = os.path.join("data", "negative")

    pos_files = glob.glob(os.path.join(pos_dir, "*.mp4")) + glob.glob(os.path.join(pos_dir, "*.avi"))
    neg_files = glob.glob(os.path.join(neg_dir, "*.mp4")) + glob.glob(os.path.join(neg_dir, "*.avi"))

    all_files = pos_files + neg_files
    all_labels = [1] * len(pos_files) + [0] * len(neg_files)

    if len(all_files) == 0:
        print("找不到影片檔！請先將影片準備好再執行。")
        return

    print(f"Videos found: positive={len(pos_files)} | negative={len(neg_files)}")

    # Split at video-level first to avoid leakage between sets.
    try:
        train_files, temp_files, train_labels, temp_labels = train_test_split(
            all_files, all_labels, test_size=0.2, random_state=42, stratify=all_labels
        )
        val_files, test_files, val_labels, test_labels = train_test_split(
            temp_files, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
        )
    except ValueError:
        print("警告：資料量過少，無法確保資料類別比例 (stratify)，改為一般隨機切分...")
        train_files, temp_files, train_labels, temp_labels = train_test_split(
            all_files, all_labels, test_size=0.2, random_state=42
        )
        if len(temp_files) >= 2:
            val_files, test_files, val_labels, test_labels = train_test_split(
                temp_files, temp_labels, test_size=0.5, random_state=42
            )
        else:
            val_files, val_labels = temp_files, temp_labels
            test_files, test_labels = [], []

    print(f"Dataset split (video-level): Train={len(train_files)} Val={len(val_files)} Test={len(test_files)}")

    frame_stride = 1
    max_frames_per_video = None

    train_dataset = EchoFrameDataset(
        train_files,
        train_labels,
        preprocess_val,
        frame_stride=frame_stride,
        max_frames_per_video=max_frames_per_video,
    )
    val_dataset = EchoFrameDataset(
        val_files,
        val_labels,
        preprocess_val,
        frame_stride=frame_stride,
        max_frames_per_video=max_frames_per_video,
    )

    if len(train_dataset) == 0:
        print("訓練集沒有可用 frame，請檢查影片是否可讀取。")
        return

    train_bs = min(32, len(train_dataset))
    if train_bs % 2 != 0:
        train_bs = max(2, train_bs - 1)
    val_bs = min(32, len(val_dataset)) if len(val_dataset) > 0 else 1

    train_labels_for_sampling = [lbl for _, lbl in train_dataset.samples]
    try:
        train_batch_sampler = BalancedBinaryBatchSampler(train_labels_for_sampling, train_bs)
        train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler)
        print(f"Using balanced batch sampling: batch_size={train_batch_sampler.batch_size}")
    except ValueError as e:
        print(f"警告：{e}，改用一般隨機抽樣。")
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_bs,
            shuffle=True,
            drop_last=(len(train_dataset) > train_bs),
        )

    if len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_bs,
            shuffle=False,
            drop_last=(len(val_dataset) > val_bs),
        )
    else:
        val_loader = None

    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": 1e-5},
            {"params": [logit_scale], "lr": 1e-3},
        ],
        weight_decay=0.01,
    )

    epochs = 10
    train_losses = []
    val_losses = []

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")

        for batch_images, batch_texts, batch_labels in train_pbar:
            optimizer.zero_grad()

            batch_images = batch_images.to(device, dtype=torch.bfloat16)
            batch_texts = batch_texts.to(device)
            batch_labels = batch_labels.to(device)

            image_embeds = model.encode_image(batch_images)
            text_embeds = model.encode_text(batch_texts)

            # Prevent unstable temperature growth.
            logit_scale.data.clamp_(0, 4.6052)
            loss = contrastive_loss(
                image_embeds,
                text_embeds,
                torch.exp(logit_scale),
                batch_labels,
            )

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_train_loss / max(1, len(train_loader))
        train_losses.append(avg_train_loss)

        avg_val_loss = None
        if val_loader is not None and len(val_loader) > 0:
            model.eval()
            total_val_loss = 0.0
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]")

            with torch.no_grad():
                for batch_images, batch_texts, batch_labels in val_pbar:
                    batch_images = batch_images.to(device, dtype=torch.bfloat16)
                    batch_texts = batch_texts.to(device)
                    batch_labels = batch_labels.to(device)

                    image_embeds = model.encode_image(batch_images)
                    text_embeds = model.encode_text(batch_texts)

                    loss = contrastive_loss(
                        image_embeds,
                        text_embeds,
                        torch.exp(logit_scale),
                        batch_labels,
                    )

                    total_val_loss += loss.item()
                    val_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            val_msg = f"{avg_val_loss:.4f}"
        else:
            val_losses.append(float("nan"))
            val_msg = "N/A (no validation samples)"

        print(
            f"Epoch {epoch + 1} Summary -> Train Loss: {avg_train_loss:.4f} | Val Loss: {val_msg}"
        )

    os.makedirs("checkpoints", exist_ok=True)
    save_path = "checkpoints/echo_clip_finetuned.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Training complete. 模型學習成果已存至 {save_path}")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss", marker="o")
    plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss", marker="o")
    plt.title("Training and Validation Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.xticks(range(1, epochs + 1))
    plt.legend()
    plt.grid(True)
    curve_path = "checkpoints/loss_curve.png"
    plt.savefig(curve_path)
    print(f"Loss curve 已儲存在 {curve_path}")


if __name__ == "__main__":
    train()
