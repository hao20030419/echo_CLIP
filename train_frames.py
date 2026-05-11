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

class EchoFrameDataset(Dataset):
    def __init__(self, file_list, label_list, preprocess_val):
        self.preprocess_val = preprocess_val
        self.samples = []
        
        pos_text = "An echocardiogram showing aortic dissection."
        neg_text = "A normal echocardiogram without aortic dissection."
        
        self.pos_tokens = torch.tensor(template_tokenize(pos_text), dtype=torch.long)
        self.neg_tokens = torch.tensor(template_tokenize(neg_text), dtype=torch.long)

        print(f"正在將 {len(file_list)} 支影片拆解為獨立影格...")
        for file_path, label in tqdm(zip(file_list, label_list), total=len(file_list), desc="Extracting frames"):
            try:
                frames = read_avi(file_path, (224, 224))
                for frame in frames:
                    self.samples.append((frame, label))
            except Exception as e:
                print(f"\n[{file_path}] Read error: {e}")
                
        print(f"提取完成！總共獲得 {len(self.samples)} 個影格樣本。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame, label = self.samples[idx]
        img_tensor = self.preprocess_val(T.ToPILImage()(frame))
        tokens = self.pos_tokens if label == 1 else self.neg_tokens
        return img_tensor, tokens

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

    # 這裡非常重要：我們先切分「影片 (Files)」，再去提取「影格 (Frames)」!
    # 為什麼？因為如果在影格層級去切 Train/Test，同一個影片的相近影格會同時出現在訓練集與測試集，
    # 導致「Data Leakage (資料外洩)」，模型會作弊只利用背景來辨認病人。
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

    print(f"\nVideo split: Train={len(train_files)} Val={len(val_files)} Test={len(test_files)}")

    print("\n--- 準備訓練集 ---")
    train_dataset = EchoFrameDataset(train_files, train_labels, preprocess_val)
    print("\n--- 準備驗證集 ---")
    val_dataset = EchoFrameDataset(val_files, val_labels, preprocess_val)
    
    # 資料量變大了，Batch Size 也可以提高了！
    train_bs = min(32, len(train_dataset)) if len(train_dataset) > 0 else 1
    val_bs = min(32, len(val_dataset)) if len(val_dataset) > 0 else 1

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
    
    print("\nStarting training...")
    for epoch in range(epochs):
        # --- Training Loop ---
        model.train()
        total_train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for batch_images, batch_texts in train_pbar:
            optimizer.zero_grad()
            
            batch_images = batch_images.to(device, dtype=torch.bfloat16)
            batch_texts = batch_texts.to(device)
            
            # 因為已經是 Independent Frames，所以 shape 直接是 (B, 3, 224, 224)
            # 直接 Encode_image 即可，不需要再做 view 攤平或 mean 平均！
            video_embeds = model.encode_image(batch_images)
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
            for batch_images, batch_texts in val_pbar:
                batch_images = batch_images.to(device, dtype=torch.bfloat16)
                batch_texts = batch_texts.to(device)
                
                video_embeds = model.encode_image(batch_images)
                text_embeds = model.encode_text(batch_texts)
                
                loss = contrastive_loss(video_embeds, text_embeds, torch.exp(logit_scale))
                
                total_val_loss += loss.item()
                val_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
               
        avg_val_loss = total_val_loss / max(1, len(val_loader))
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1} Summary -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    save_path = "checkpoints/echo_clip_finetuned.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Training complete. 模型學習成果已存至 {save_path}")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs+1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, epochs+1), val_losses, label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(range(1, epochs+1))
    plt.legend()
    plt.grid(True)
    plt.savefig('checkpoints/loss_curve.png')
    print("Loss curve 已儲存在 checkpoints/loss_curve.png")

if __name__ == '__main__':
    train()