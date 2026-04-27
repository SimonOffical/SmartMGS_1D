import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import matplotlib.pyplot as plt # 新增导入
# 导入刚才写的模块
from dataset import FDEM_Dataset
from resnet1d import ResNet1D_Evaluator

def train_model():
    # 1. 基本参数设置
    EPOCHS = 100
    BATCH_SIZE = 512
    LR = 1e-4
    DATA_PATH = "./data/inv_training_dataset_1d.npy"
    SAVE_DIR = "./models/weights"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 自动选择设备 (Mac M系芯片用 mps，N卡用 cuda，否则用 cpu)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"🔥 训练设备分配: {device}")

    # 2. 准备数据
    full_dataset = FDEM_Dataset(DATA_PATH)
    # 按 8:2 划分训练集和验证集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3. 初始化网络、损失函数、优化器
    model = ResNet1D_Evaluator().to(device)
    criterion = nn.MSELoss() # 均方误差，适合回归任务
    optimizer = Adam(model.parameters(), lr=LR)

    # 4. 开始训练循环
    best_val_loss = float('inf')
    # 🌟 新增：用于记录 Loss 历史的列表
    history_train_loss = []
    history_val_loss = []
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        # 加上 tqdm 进度条，看着舒服
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for batch_x, batch_y in pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # 前向传播
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            
            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # 5. 验证模型 (不计算梯度)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        print(f"📊 Epoch {epoch+1} 总结 -> Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")
        history_train_loss.append(avg_train_loss)
        history_val_loss.append(avg_val_loss)
        # 6. 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(SAVE_DIR, "best_resnet1d.pth")
            torch.save(model.state_dict(), save_path)
            print(f"💾 发现更低的 Val Loss，模型已保存至: {save_path}")
    # 🌟 新增：训练结束后绘制并保存 Loss 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS+1), history_train_loss, 'b-', label='Train Loss', linewidth=2)
    plt.plot(range(1, EPOCHS+1), history_val_loss, 'r-', label='Validation Loss', linewidth=2)
    plt.title('Training and Validation Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.savefig(os.path.join(SAVE_DIR, "loss_curve.png"), dpi=300)
    plt.show()
if __name__ == "__main__":
    train_model()