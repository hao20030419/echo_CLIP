import os
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from open_clip import create_model_and_transforms
from template_tokenizer import template_tokenize
from utils import read_avi
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class EchoDataset(Dataset):
    def __init__(self, file_list, label_list, preprocess_val):
        self.preprocess_val = preprocess_val
        self.files = file_list
        self.labels = label_list
        
        pos_text = "An echocardiogram showing aortic dissection."
        neg_text = "A normal echocardiogram without aortic dissection."
        
        self.pos_tokens = torch.tensor(template_tokenize(pos_text), dtype=torch.long)
        self.neg_tokens = torch.tensor(template_tokenize(neg_text), dtype=torch.long)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]
        
        try:
            frames = read_avi(file_path, (224, 224))
            
            # 從所有影格中均勻抽取 10 個 frame，讓每支影片的時間維長度都一致且適應記憶體
            num_frames = len(frames)
            if num_frames > 0:
                import numpy as np
                indices = np.linspace(0, num_frames - 1, 10, dtype=int)
                sampled_frames = [frames[i] for i in indices]
            else:
                sampled_frames = []
                
            video_tensor = torch.stack(
                [self.preprocess_val(T.ToPILImage()(frame)) for frame in sampled_frames], dim=0
            )
            # 確保最後出來一定是 10 frames，如果原始影片有問題則報錯走 except
            if video_tensor.shape[0] != 10:
                raise ValueError("Frames count is not 10")
                
        except Exception as e:
            # 發生錯誤或者影片毀損時，給定全零的 10 frames (10, 3, 224, 224) 確保訓練不中斷
            print(f"[{file_path}] Read warning: {e}")
            video_tensor = torch.zeros((10, 3, 224, 224))
        
        tokens = self.pos_tokens if label == 1 else self.neg_tokens
        
        return video_tensor, tokens

def contrastive_loss(video_embeddings, text_embeddings, logit_scale, batch_labels):
    """
    Supervised Soft-Label Contrastive Loss.
    因為我們只有兩種文字 Prompt，同一個 Batch 裡會有多個 video 共享相同的 text embedding。
    標準 InfoNCE 會把這些相同的 text 當成 Negative，導致梯度互相打架。
    這裡改用 Soft Label：同一類別的所有 (video, text) pair 都算 Positive，
    並且對每一列做 normalization，讓 loss 正確引導模型學習。
    """
    video_embeddings = F.normalize(video_embeddings, dim=-1)
    text_embeddings = F.normalize(text_embeddings, dim=-1)
    
    logits = logit_scale * (video_embeddings @ text_embeddings.T)  # (B, B)
    
    # 建立 Soft Target 矩陣：對 video_i，所有與它 label 相同的 text_j 都是正確答案
    labels_t = batch_labels.unsqueeze(0)  # (1, B)
    labels_v = batch_labels.unsqueeze(1)  # (B, 1)
    soft_targets = (labels_v == labels_t).float()  # (B, B)
    # 每列做正規化，讓每個 video 的正確答案比例加總為 1
    soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True)
    
    # Soft Cross Entropy (Video -> Text)
    loss_v = -(soft_targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    # Soft Cross Entropy (Text -> Video)
    loss_t = -(soft_targets.T * F.log_softmax(logits.T, dim=1)).sum(dim=1).mean()
    
    return (loss_v + loss_t) / 2

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading EchoCLIP-R model...")
    model, _, preprocess_val = create_model_and_transforms(
        "hf-hub:mkaichristensen/echo-clip-r", precision="bf16", device=device
    )
    
    logit_scale = torch.nn.Parameter(torch.ones([], device=device) * 2.6592)

    pos_dir = os.path.join("data", "positive")
    neg_dir = os.path.join("data", "negative")
    
    pos_files = glob.glob(os.path.join(pos_dir, "*.mp4"))
    neg_files = glob.glob(os.path.join(neg_dir, "*.mp4"))
    
    all_files = pos_files + neg_files
    all_labels = [1] * len(pos_files) + [0] * len(neg_files)
    
    if len(all_files) == 0:
        print("找不到影片檔！請先將影片準備好再執行。")
        return

    # 嘗試進行分層切分 Train (80%) / Temp (20%)，再由 Temp 均分給 Val (10%) 與 Test (10%)
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
            # 如果資料真的極少，把剩下的所有檔案都塞給 val 當作測試用
            val_files, val_labels = temp_files, temp_labels
            test_files, test_labels = [], []

    print(f"Dataset split: Train={len(train_files)} Val={len(val_files)} Test={len(test_files)}")

    train_dataset = EchoDataset(train_files, train_labels, preprocess_val)
    val_dataset = EchoDataset(val_files, val_labels, preprocess_val)
    
    # 防止當 batch 數量比 batch_size 小時報錯，若是資料太少，把 drop_last 關掉或換 batch_size
    train_bs = min(4, len(train_dataset)) if len(train_dataset) > 0 else 1
    val_bs = min(4, len(val_dataset)) if len(val_dataset) > 0 else 1

    train_loader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True, drop_last=(len(train_dataset)>train_bs))
    if len(val_dataset) > 0:
        val_loader = DataLoader(val_dataset, batch_size=val_bs, shuffle=False, drop_last=(len(val_dataset)>val_bs))
    else:
        val_loader = []
    
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': 1e-6},
        {'params': [logit_scale], 'lr': 1e-3}
    ], weight_decay=0.01)

    epochs = 5
    train_losses = []
    val_losses = []
    
    print("Starting training...")
    for epoch in range(epochs):
        # --- Training Loop ---
        model.train()
        total_train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for batch_videos, batch_texts in train_pbar:
            optimizer.zero_grad()
            
            batch_videos = batch_videos.to(device, dtype=torch.bfloat16)
            batch_texts = batch_texts.to(device)
            
            B, F_count, C, H, W = batch_videos.shape
            flat_videos = batch_videos.view(B * F_count, C, H, W)
            
            frame_embeds = model.encode_image(flat_videos).view(B, F_count, -1)
            video_embeds = frame_embeds.mean(dim=1)
            text_embeds = model.encode_text(batch_texts)
            
            # clamp 防止 logit_scale 爆炸 (CLIP 論文建議上限為 ln(100) ≈ 4.6)
            logit_scale.data.clamp_(0, 4.6052)

            # 透過比對 token 來還原當前 batch 每個樣本的 label
            pos_tokens_ref = train_dataset.pos_tokens.to(device)
            batch_labels = torch.tensor(
                [1 if torch.equal(batch_texts[i], pos_tokens_ref) else 0
                 for i in range(batch_texts.size(0))],
                device=device
            )
            loss = contrastive_loss(video_embeds, text_embeds, torch.exp(logit_scale), batch_labels)
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_train_loss = total_train_loss / max(1, len(train_loader))
        train_losses.append(avg_train_loss)
        
        # --- Validation Loop ---
        model.eval()
        total_val_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
        
        with torch.no_grad():
            for batch_videos, batch_texts in val_pbar:
                batch_videos = batch_videos.to(device, dtype=torch.bfloat16)
                batch_texts = batch_texts.to(device)
                
                B, F_count, C, H, W = batch_videos.shape
                flat_videos = batch_videos.view(B * F_count, C, H, W)
                
                frame_embeds = model.encode_image(flat_videos).view(B, F_count, -1)
                video_embeds = frame_embeds.mean(dim=1)
                text_embeds = model.encode_text(batch_texts)
                
                pos_tokens_ref = train_dataset.pos_tokens.to(device)
                batch_labels = torch.tensor(
                    [1 if torch.equal(batch_texts[i], pos_tokens_ref) else 0
                     for i in range(batch_texts.size(0))],
                    device=device
                )
                loss = contrastive_loss(video_embeds, text_embeds, torch.exp(logit_scale), batch_labels)
                
                total_val_loss += loss.item()
                val_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
               
        avg_val_loss = total_val_loss / max(1, len(val_loader))
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1} Summary -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # === 5. 儲存模型與圖表 ===
    os.makedirs("checkpoints", exist_ok=True)
    save_path = "checkpoints/echo_clip_finetuned.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Training complete. 模型學習成果已存至 {save_path}")

    # 繪製 Loss Curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs+1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, epochs+1), val_losses, label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(range(1, epochs+1))
    plt.legend()
    plt.grid(True)
    curve_path = 'checkpoints/loss_curve.png'
    plt.savefig(curve_path)
    print(f"Loss curve 已儲存在 {curve_path}")

if __name__ == '__main__':
    train()