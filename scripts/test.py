import numpy as np
import matplotlib.pyplot as plt

def generate_1d_grf_resistivity(n_layers, alpha, log_rho_range):
    """
    使用一维高斯随机场(1D GRF)生成具有空间相关性的多层电阻率模型。
    
    参数:
    -----------
    n_layers : int
        一维模型的层数。
    alpha : float
        比例因子 (Scaling factor)，控制地层的平滑度/相关长度。
        - alpha 较大 (如 2.0 - 4.0): 地层电性变化平缓，呈现大套地层特征。
        - alpha 较小 (如 0.5 - 1.5): 地层电性变化剧烈，呈现薄互层特征。
    log_rho_range : tuple
        以10为底的对数电阻率范围，例如 (0.0, 3.0) 对应 1 到 1000 Ohm·m。
        
    返回:
    --------
    resistivities : np.ndarray
        生成的各层电阻率数组 (长度为 n_layers)。
    """
    # 1. 在深度域生成标准白噪声
    noise = np.random.normal(0, 1, n_layers)
    
    # 2. 傅里叶变换到频率/波数域
    noise_fft = np.fft.fft(noise)
    
    # 3. 计算一维波数 (k)
    k = np.fft.fftfreq(n_layers)
    k_abs = np.abs(k)
    k_abs[0] = 1e-10  # 防止直流分量除零
    
    # 4. 应用高斯随机场能量谱衰减 (|k|^-alpha)
    amplitude = k_abs ** (-alpha / 2.0)
    amplitude[0] = 0  # 消除直流分量，使均值为0
    
    filtered_fft = noise_fft * amplitude
    
    # 5. 逆傅里叶变换回到深度域
    grf = np.fft.ifft(filtered_fft).real
    
    # 6. 数据归一化与物理量映射
    grf_min, grf_max = grf.min(), grf.max()
    grf_norm = (grf - grf_min) / (grf_max - grf_min + 1e-8)
    
    log_min, log_max = log_rho_range
    log_rho = grf_norm * (log_max - log_min) + log_min
    
    # 还原为实际电阻率
    resistivities = 10 ** log_rho
    return resistivities


# ==========================================
# 测试与地球物理测井曲线风格可视化
# ==========================================
if __name__ == "__main__":
    n_layers = 100         # 细分 100 层以展示多尺度特征
    depth_max = 300.0      # 总深度 300 米
    rho_range = (-2, 5) # 10^1 到 10^3 (10 到 1000 欧姆米)
    
    # 生成等厚度的深度网格用于绘图
    thicknesses = np.ones(n_layers) * (depth_max / n_layers)
    depth_nodes = np.insert(np.cumsum(thicknesses), 0, 0.0) # 深度节点 [0, h1, h1+h2, ...]

    # 使用不同尺度因子生成 3 组模型
    alphas = [0.5, 1.5, 2.5]
    titles = ["Thin Interbeds (alpha=0.5)", "Moderate Scale (alpha=1.5)", "Massive Strata (alpha=2.5)"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    fig, axes = plt.subplots(1, 3, figsize=(12, 8), sharey=True)
    
    for i, alpha in enumerate(alphas):
        # 核心：调用 GRF 生成 1D 电阻率
        res = generate_1d_grf_resistivity(n_layers, alpha, rho_range)
        
        # 为了画出“阶梯状”测井曲线，我们需要重复电阻率值以匹配深度节点的首尾
        res_plot = np.repeat(res, 2)
        depth_plot = np.repeat(depth_nodes[1:-1], 2)
        depth_plot = np.insert(depth_plot, 0, depth_nodes[0])
        depth_plot = np.append(depth_plot, depth_nodes[-1])

        axes[i].plot(res_plot, depth_plot, color=colors[i], lw=2)
        axes[i].set_xscale('log')
        axes[i].set_xlim(10**rho_range[0], 10**rho_range[1])
        axes[i].set_ylim(depth_max, 0) # 深度轴向下翻转
        axes[i].set_title(titles[i], fontsize=12)
        axes[i].grid(True, which='both', linestyle='--', alpha=0.5)
        if i == 0:
            axes[i].set_ylabel("Depth (m)", fontsize=12)
        axes[i].set_xlabel("Resistivity (Ohm.m)", fontsize=12)

    plt.suptitle("1D Multi-scale Geoelectrical Models via Gaussian Random Field", fontsize=16, y=0.98)
    plt.tight_layout()
    plt.show()