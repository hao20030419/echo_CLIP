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
            video_tensor = torch.stack(
                [self.preprocess_val(T.ToPILImage()(frame)) for frame in frames], dim=0
            )
        except Exception as e:
            video_tensor = torch.zeros((10, 3, 224, 224))
        
        tokens = self.pos_tokens if label == 1 else self.neg_tokens
        
        return video_tensor, tokens

def contrastive_loss(video_embeddings, text_embeddings, logit_scale):
    video_embeddings = F.normalize(video_embeddings, dim=-1)
    text_embeddings = F.normalize(text_embeddings, dim=-1)
    
    logits_per_video = logit_scale * (video_embeddings @ text_embeddings.T)
    logits_per_text = logits_per_video.T
    
    labels = torch.arange(video_embeddings.size(0), device=video_embeddings.device)
    
    loss_v = F.cross_entropy(logits_per_video, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    
    return (loss_v + loss_t) / 2

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading EchoCLIP-R model...")
    model, _, preprocess_val = create_model_and_transforms(
        "hf-hub:mkaichristensen/echo-clip-r", precision="bf16", device=device
    )
    
    logit_scale = torch.nn.Parameter(torch.ones([]) * 2.6592)
    logit_scale = logit_scale.to(device)

    pos_dir = os.path.join("data", "positive")
    neg_dir = os.path.join("data", "negative")
    
    pos_files = glob.glob(os.path.join(pos_dir, "*.avi"))
    neg_files = glob.glob(os.path.join(neg_dir, "*.avi"))
    
    all_files = pos_files + neg_files
    all_labels = [1] * len(pos_files) + [0] * len(neg_files)
    
    if len(all_files) == 0:
        print("找不到影片檔！請先將影片準備好再執行。")
        return

    # 切分 Train (80%) / Temp (20%)，再從 Temp 切出 Val(10%) 跟 Test(10%) -> test_size=0.5
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        all_files, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )

    print(f"Dataset split: Train={len(train_files)} Val={len(val_files)} Test={len(test_files)}")

    train_dataset = EchoDataset(train_files, train_labels, preprocess_val)
    val_dataset = EchoDataset(val_files, val_labels, preprocess_val)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, drop_last=True)
    
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
            
            loss = contrastive_loss(video_embeds, text_embeds, torch.exp(logit_scale))
            
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
                
                loss = contrastive_loss(video_embeds, text_embeds, torch.exp(logit_scale))
                
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