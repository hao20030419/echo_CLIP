import os
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from open_clip import create_model_and_transforms
from template_tokenizer import template_tokenize
from utils import read_avi
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm
import numpy as np

class TestDataset(Dataset):
    def __init__(self, file_list, label_list, preprocess_val):
        self.preprocess_val = preprocess_val
        self.files = file_list
        self.labels = label_list

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]
        
        try:
            frames = read_avi(file_path, (224, 224))
            num_frames = len(frames)
            if num_frames > 0:
                indices = np.linspace(0, num_frames - 1, 10, dtype=int)
                sampled_frames = [frames[i] for i in indices]
            else:
                sampled_frames = []
                
            video_tensor = torch.stack(
                [self.preprocess_val(T.ToPILImage()(frame)) for frame in sampled_frames], dim=0
            )
            if video_tensor.shape[0] != 10:
                raise ValueError("Frames count is not 10")
        except Exception as e:
            print(f"[{file_path}] Read warning: {e}")
            video_tensor = torch.zeros((10, 3, 224, 224))
            
        return video_tensor, label

def evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading EchoCLIP-R model...")
    model, _, preprocess_val = create_model_and_transforms(
        "hf-hub:mkaichristensen/echo-clip-r", precision="bf16", device=device
    )
    
    weights_path = "checkpoints/echo_clip_finetuned.pt"
    if os.path.exists(weights_path):
        print(f"找到微調權重！！正在載入 {weights_path} ...")
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        print(f"警告：找不到 {weights_path}，將使用原始未訓練的預設權重進行測試。")
        
    model.eval()

    # === 重建資料切分邏輯，確保拿到與訓練時一模一樣的 Test Set ===
    pos_dir = os.path.join("data", "positive")
    neg_dir = os.path.join("data", "negative")
    
    # 支援 .mp4 與 .avi
    pos_files = glob.glob(os.path.join(pos_dir, "*.mp4")) + glob.glob(os.path.join(pos_dir, "*.avi"))
    neg_files = glob.glob(os.path.join(neg_dir, "*.mp4")) + glob.glob(os.path.join(neg_dir, "*.avi"))
    
    all_files = pos_files + neg_files
    all_labels = [1] * len(pos_files) + [0] * len(neg_files)

    if len(all_files) == 0:
        print("找不到任何影片檔！")
        return

    try:
        _, temp_files, _, temp_labels = train_test_split(
            all_files, all_labels, test_size=0.2, random_state=42, stratify=all_labels
        )
        _, test_files, _, test_labels = train_test_split(
            temp_files, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
        )
    except ValueError:
        _, temp_files, _, temp_labels = train_test_split(
            all_files, all_labels, test_size=0.2, random_state=42
        )
        if len(temp_files) >= 2:
            _, test_files, _, test_labels = train_test_split(
                temp_files, temp_labels, test_size=0.5, random_state=42
            )
        else:
            test_files, test_labels = [], []

    if len(test_files) == 0:
        print("目前資料量不足以分出測試集 (Test Set)，請準備更多資料。")
        return
        
    print(f"使用 {len(test_files)} 筆 Test Data 進行獨立驗證。")
    test_dataset = TestDataset(test_files, test_labels, preprocess_val)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # === 準備正例與反例的文字 Prompt，用來當作分類依據 ===
    # 標籤設定: Index 0 代表 Negative, Index 1 代表 Positive
    neg_text = "A normal echocardiogram without aortic dissection."
    pos_text = "An echocardiogram showing aortic dissection."
    
    tokens_neg = torch.tensor(template_tokenize(neg_text), dtype=torch.long)
    tokens_pos = torch.tensor(template_tokenize(pos_text), dtype=torch.long)
    text_tokens = torch.stack([tokens_neg, tokens_pos]).to(device)

    all_preds = []
    all_targets = []

    print("開始驗證 (評估模型)...")
    with torch.no_grad():
        # 先將文字轉化出兩條標準參考向量
        text_embeds = F.normalize(model.encode_text(text_tokens), dim=-1) # (2, D)

        for batch_videos, batch_labels in tqdm(test_loader, desc="[Test Set]"):
            batch_videos = batch_videos.to(device, dtype=torch.bfloat16)
            
            # Video 降維與特徵萃取
            B, F_count, C, H, W = batch_videos.shape
            flat_videos = batch_videos.view(B * F_count, C, H, W)
            frame_embeds = model.encode_image(flat_videos).view(B, F_count, -1)
            video_embeds = frame_embeds.mean(dim=1)
            video_embeds = F.normalize(video_embeds, dim=-1)
            
            # 【核心辨識機制】：將每部影片與 "正例文字" 和 "反例文字" 比較相似度
            similarity = video_embeds @ text_embeds.T # Output shape: (B, 2)
            
            # 模型選擇分數最高的文字作為預測結果 (argmax: 0=正常, 1=生病)
            preds = similarity.argmax(dim=-1).cpu().numpy()
            
            all_preds.extend(preds)
            all_targets.extend(batch_labels.numpy())

    # === 印出評估指標 ===
    print("\n\n" + "="*50)
    print("                測試集驗證結果 (Test Metrics) ")
    print("="*50)
    print(f"Overall Accuracy 模型準確率: {accuracy_score(all_targets, all_preds)*100:.2f}%")
    print("-" * 50)
    print("混淆矩陣 (Confusion Matrix):")
    print("(row=真實答案, col=模型預測)")
    print(confusion_matrix(all_targets, all_preds))
    print("-" * 50)
    print("分類詳細報告 (Classification Report):")
    target_names = ["Negative (Normal)", "Positive (Dissection)"]
    print(classification_report(all_targets, all_preds, target_names=target_names))
    print("="*50)

if __name__ == "__main__":
    evaluate()