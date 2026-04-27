import torch
import numpy as np
import os

# 导入您的网络结构
from resnet1d import ResNet1D_Evaluator

def predict_confidence(m_inv_linear, sensitivity, model_path="./models/weights/best_resnet1d.pth"):
    """
    使用训练好的 1D-ResNet 模型评估单条 FDEM 反演结果的置信度。
    
    参数:
    m_inv_linear (numpy.ndarray): 一维反演得到的线性电阻率数组，形状 (n_layers,)
    sensitivity (numpy.ndarray): 对应的物理灵敏度特征数组，形状 (n_layers,)
    model_path (str): 训练好的权重文件路径
    
    返回:
    r_pred (numpy.ndarray): 预测的各层置信度得分 (0~1)
    """
    # ==========================================
    # 1. 环境与模型准备
    # ==========================================
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ 找不到模型权重文件: {model_path}，请确认是否已完成训练！")
        
    # 自动分配设备
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    # 实例化网络骨架
    model = ResNet1D_Evaluator(in_channels=2, hidden_channels=64, num_blocks=4).to(device)
    
    # 加载训练好的“灵魂”（权重参数）
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # ⚠️ 极其重要：切断 Dropout 和 BatchNorm 的训练特性，进入评估模式
    model.eval() 

    # ==========================================
    # 2. 数据预处理 (与训练时必须保持绝对一致)
    # ==========================================
    # 电阻率取对数
    m_inv_log = np.log10(m_inv_linear)
    
    # 拼接成 (2, n_layers) 的二维数组
    x_numpy = np.stack([m_inv_log, sensitivity], axis=0)
    
    # 转换为 Tensor，并增加 Batch 维度，变成 (1, 2, n_layers)
    x_tensor = torch.tensor(x_numpy, dtype=torch.float32).unsqueeze(0).to(device)

    # ==========================================
    # 3. 执行前向推理
    # ==========================================
    # ⚠️ 极其重要：推理时不需要算梯度，省内存且加速
    with torch.no_grad():
        y_pred_tensor = model(x_tensor)
        
    # 提取结果：去掉 Batch 维度，移回 CPU，转回 Numpy 数组
    r_pred = y_pred_tensor.squeeze().cpu().numpy()
    
    return r_pred


# ==========================================
# 极简测试用法演示
# ==========================================
# if __name__ == "__main__":
if False:
    import numpy as np
    
    # ==========================================
    # 构造一个复杂的“地质+物理”测试模型 (20层)
    # ==========================================
    n_layers = 20
    
    # 1. 构造反演电阻率 (薄互层 + 高低阻交替)
    dummy_m_inv = np.ones(n_layers) * 100.0  # 背景是 100 欧姆米的普通地层
    dummy_m_inv[4:7] = 10.0      # 【陷阱 1】浅部良导层 (如含水黏土，极低阻 10)
    dummy_m_inv[12:16] = 2000.0  # 【陷阱 2】深部致密基底 (极高阻 2000)

    # 2. 构造物理灵敏度 (指数衰减 + 突变)
    # 正常情况下随深度指数衰减 (从 1.0 衰减到 0.05 左右)
    depths = np.linspace(0, 3, n_layers)
    dummy_sens = np.exp(-depths)
    
    # 模拟趋肤效应：因为第 5~7 层是低阻，电磁波被大量吸收，导致第 7 层以下的灵敏度断崖式下跌！
    dummy_sens[7:] = dummy_sens[7:] * 0.3 
    
    # 底部加一点极其微弱的随机扰动，模拟仪器噪声
    dummy_sens[-4:] += np.random.uniform(-0.01, 0.01, 4)
    dummy_sens = np.clip(dummy_sens, 0.0001, 1.0) # 保证物理量不为负数

    # ==========================================
    # 开始呼叫网络进行评估
    # ==========================================
    print("🔮 正在呼叫 1D-ResNet 进行置信度评估...")
    
    try:
        r_scores = predict_confidence(dummy_m_inv, dummy_sens)
        
        print("\n✅ 评估完成！复杂的电磁测深置信度报告如下:\n")
        print("-" * 65)
        print(f"{'地层序号':<8} | {'反演电阻率 (Ω·m)':<15} | {'物理灵敏度':<10} | {'网络预测置信度 (r)'}")
        print("-" * 65)
        
        for i, score in enumerate(r_scores):
            res = dummy_m_inv[i]
            sens = dummy_sens[i]
            
            # 简单做个可视化：高置信度打绿勾，中等打黄圈，极低打红叉
            mark = "🟢 (可靠)" if score > 0.75 else "🟡 (存疑)" if score > 0.4 else "🔴 (盲区)"
            
            # 高亮特殊地层方便查看
            if res == 10.0:
                layer_str = f"浅层低阻🎯"
            elif res == 2000.0:
                layer_str = f"深部高阻🧱"
            else:
                layer_str = f"第 {i+1:02d} 层"
                
            print(f"{layer_str:<10} | {res:>12.1f}    |  {sens:>8.4f}  |  {score:.4f} {mark}")
            
        print("-" * 65)
        
    except Exception as e:
        print(f"❌ 预测出错: {e}")