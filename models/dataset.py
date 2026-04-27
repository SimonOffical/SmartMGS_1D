import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class FDEM_Dataset(Dataset):
    def __init__(self, npy_path):
        """
        加载 1D FDEM 反演数据集
        """
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"找不到数据集文件: {npy_path}")
            
        print(f"正在加载数据: {npy_path}")
        data = np.load(npy_path, allow_pickle=True).item()
        
        # 提取输入特征 (X)
        # 1. 物理反演结果 (通常取 log10 压缩数值范围，方便网络学习)
        m_inv = np.log10(data['input_m_fdem'])
        # 2. 灵敏度特征 (归一化后的对数值)
        sens = data['input_sensitivity']
        
        # 将两个特征在通道维度 (Channel) 拼接
        # 形状变化: (Samples, Layers) -> (Samples, 2, Layers)
        x_combined = np.stack([m_inv, sens], axis=1)
        self.x = torch.tensor(x_combined, dtype=torch.float32)
        
        # 提取目标标签 (Y)
        # 置信度 r_true (范围 0~1)
        self.y = torch.tensor(data['label_r_true'], dtype=torch.float32)
        
        print(f"✅ 数据加载完成! X shape: {self.x.shape}, Y shape: {self.y.shape}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# 测试代码
if __name__ == "__main__":
    # 在根目录下运行时的相对路径
    ds = FDEM_Dataset("./data/inv_training_dataset_1d.npy")
    dl = DataLoader(ds, batch_size=32, shuffle=True)
    x_batch, y_batch = next(iter(dl))
    print(f"Batch X: {x_batch.shape}, Batch Y: {y_batch.shape}")