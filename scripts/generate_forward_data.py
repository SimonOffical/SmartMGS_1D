import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
# print(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from physics.physic_fdem import forward_process
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
def forward_calc(rho, hh):
    """
    针对给定的一维地电模型进行 FDEM 正演计算。
    
    参数:
    rho: 列表或一维数组，各层的电阻率
    hh:  列表或一维数组，各层的厚度 (最后一层通常填 0)
    
    返回:
    hz1: 对应频点的二次场复数数组
    hp1: 对应频点的一次场复数数组
    """
    # 1. 设定的观测频率 (从 1kHz 到 30kHz 的 20 个频点)
    fq = np.linspace(1e3, 30e3, 20)
    
    # 2. 仪器装置参数
    nturn_tx = 10      # 发射线圈匝数
    nturn_rx = 90      # 接收线圈匝数
    xr = 0.2           # 水平收发距 (m)
    hr = 0.1           # 线圈离地高度 (m)
    rt = 1.0           # 发射线圈半径/边长 (m)
    rr = 0.2           # 接收线圈半径 (m)
    
    # 3. 电流设置
    # forward_process 要求 current 是一个与 fq 长度相同的数组
    current_scalar = 60.0
    current_array = np.full_like(fq, current_scalar)

    # 4. 调用正演核心函数
    hz1, hp1 = forward_process(
        rho=rho, 
        hh=hh, 
        fq=fq, 
        nturn=nturn_tx, 
        nturn1=nturn_rx, 
        xr=xr, 
        hr=hr, 
        rt=rt, 
        rr=rr, 
        current=current_array,
        shape='rectangle'  # 或者保留默认的 'rectangle'，取决于您的实际线圈形状
    )
    
    return hz1, hp1

#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
# @njit(nogil=True, fastmath=True)
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
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
from tqdm import tqdm
def generate_synthetic_data(n_samples, n_layers, alpha_range, log_rho_range, max_depth = 300.0):
    """
    生成一批具有空间相关性的合成电阻率模型及其对应的 FDEM 响应数据。
    """
    # 1. 预分配 NumPy 矩阵以提升海量数据生成性能
    all_resistivities = np.zeros((n_samples, n_layers))
    
    # 预分配 20 个频点的实部和虚部矩阵
    n_freqs = 20
    all_hz_real = np.zeros((n_samples, n_freqs))
    all_hz_imag = np.zeros((n_samples, n_freqs))
    all_hp_real = np.zeros((n_samples, n_freqs))
    all_hp_imag = np.zeros((n_samples, n_freqs))
    
    frequencies = np.linspace(1e3, 30e3, n_freqs)
    
    # 2. 正确处理地层厚度 (前 n_layers-1 层平分 20 米，最后一层无限厚)
    hh = np.ones(n_layers) * (max_depth / (n_layers - 1))
    hh[-1] = 0.0  # [关键修正]：最后一层厚度强制为 0
    
    # 3. 批量生成
    for i in tqdm(range(n_samples), desc="Generating Synthetic Data"):
        # 随机选择一个 alpha 值
        alpha = np.random.uniform(*alpha_range)
        
        # 生成电阻率模型
        resistivity_model = generate_1d_grf_resistivity(n_layers, alpha, log_rho_range)
        
        # 计算 FDEM 响应
        hz1, hp1 = forward_calc(resistivity_model, hh)
        
        # 存储到预分配的矩阵中，分离实部和虚部
        all_resistivities[i, :] = resistivity_model
        all_hz_real[i, :] = hz1.real
        all_hz_imag[i, :] = hz1.imag
        all_hp_real[i, :] = hp1.real
        all_hp_imag[i, :] = hp1.imag
        
    # 4. 返回结构化的字典，非常适合直接 np.save() 存为 .npy 文件
    dataset = {
        'n_layers': n_layers,
        'frequencies': frequencies,
        'thicknesses': hh,  # 所有样本共用同一套厚度网格
        'resistivities': all_resistivities,
        'hz_real': all_hz_real,
        'hz_imag': all_hz_imag,
        'hp_real': all_hp_real,
        'hp_imag': all_hp_imag
    }
    
    return dataset
import numpy as np
import matplotlib.pyplot as plt
import os
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
def plot_1d_models(dataset_path, num_samples_to_plot=3):
    """
    加载生成的 .npy 数据集，并可视化其中几个一维电阻率模型
    """
    # 1. 加载数据
    if not os.path.exists(dataset_path):
        print(f"❌ 找不到文件: {dataset_path}")
        return
        
    print(f"📂 正在加载数据: {dataset_path}")
    dataset = np.load(dataset_path, allow_pickle=True).item()
    
    resistivities = dataset['resistivities']
    thicknesses = dataset['thicknesses']
    n_layers = dataset['n_layers']
    total_samples = resistivities.shape[0]
    
    print(f"✅ 成功加载！共有 {total_samples} 个样本。")

    # 2. 将层厚度 (Thicknesses) 转换为深度节点 (Depths)
    # 因为 matplotlib 画阶梯图需要知道每一层的顶界和底界
    depths = np.zeros(n_layers + 1)
    for i in range(n_layers):
        if i == n_layers - 1 and thicknesses[i] == 0:
            # 最后一层是无限厚，为了画图好看，我们往下多延伸 50 米
            depths[i+1] = depths[i] + 50.0 
        else:
            depths[i+1] = depths[i] + thicknesses[i]

    # 3. 随机抽取几个样本进行画图
    # 如果总样本数少于想要画的数量，就画全部
    num_plots = min(num_samples_to_plot, total_samples)
    sample_indices = np.random.choice(total_samples, num_plots, replace=False)

    # 4. 开始画图
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 7), sharey=True)
    if num_plots == 1:
        axes = [axes] # 兼容只有1个子图的情况

    for ax, idx in zip(axes, sample_indices):
        rho = resistivities[idx]
        
        # 为了画出标准的阶梯图，我们需要将每一层的电阻率值与两个深度节点对应
        # np.repeat 可以把 [R1, R2] 变成 [R1, R1, R2, R2] 配合深度界线
        plot_depths = np.repeat(depths, 2)[1:-1]
        plot_rho = np.repeat(rho, 2)
        
        # 绘制阶梯曲线
        ax.plot(plot_rho, plot_depths, 'b-', linewidth=2)
        
        # 给曲线和 y 轴之间填色，增加高级感
        # ax.fill_betweenx(plot_depths, plot_rho, 1, color='blue', alpha=0.1)

        # 设置坐标轴和格式
        ax.set_xscale('log') # 电阻率必须用对数坐标
        ax.set_xlim(0.1, 10000) # 根据您的 LOG_RHO_RANGE (1.0~3.0) 设为 1~1000
        ax.set_ylim(depths[-1], 0) # Y 轴反转，0 在最上面 (地表)
        
        ax.set_title(f"Sample #{idx}\n1D GRF Resistivity Model", fontsize=12, fontweight='bold')
        ax.set_xlabel('Resistivity (Ohm·m)', fontsize=12)
        if ax == axes[0]:
            ax.set_ylabel('Depth (m)', fontsize=12)
            
        ax.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    plt.show()
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################

if __name__ == "__main__":
    NUM_SAMPLES = 5000       # 生成的样本总数 (测试可设为 500，正式训练建议 20000+)
    N_LAYERS = 50               # 深度方向划分为 20 层
    # ALPHA_RANGE = (0.5, 3.5)    # GRF 尺度因子范围：涵盖薄互层(0.5)到大套厚地层(3.5)
    ALPHA_RANGE = (3.0, 3.5)
    LOG_RHO_RANGE = (0.0, 3.0)  # 电阻率对数范围：10^1 到 10^3 (即 10 ~ 1000 Ohm·m)
    MAX_DEPTH = 30.0           # 模型总深度 (米)，用于计算层厚度
    print(f"=== 开始生成 {NUM_SAMPLES} 个多尺度 GRF 样本 ===")
    dataset = generate_synthetic_data(
        n_samples=NUM_SAMPLES,
        n_layers=N_LAYERS,
        alpha_range=ALPHA_RANGE,
        log_rho_range=LOG_RHO_RANGE,
        max_depth= MAX_DEPTH
    )
    
    # ==========================================
    # 3. 检查生成的数据维度
    # ==========================================
    print("\n=== 数据集生成完毕，维度检查 ===")
    print(f"电阻率矩阵 (Input) 维度:   {dataset['resistivities'].shape}")
    print(f"厚度向量 (Mesh) 维度:      {dataset['thicknesses'].shape}")
    print(f"二次场实部 (Label) 维度:   {dataset['hz_real'].shape}")
    print(f"二次场虚部 (Label) 维度:   {dataset['hz_imag'].shape}")
    
    # ==========================================
    # 4. 保存为 .npy 文件供 1D-ResNet 读取
    # ==========================================
    save_dir = "./data"
    os.makedirs(save_dir, exist_ok=True)  # 如果 data 文件夹不存在则自动创建
    save_path = os.path.join(save_dir, "grf_fdem_dataset_1d.npy")
    
    # 使用 allow_pickle=True 保存字典格式
    np.save(save_path, dataset)
    print(f"\n✅ 数据集已成功保存至: {save_path}")

#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
# if __name__ == "__main__":
if False:
    # 指定刚才生成保存的数据集路径
    DATA_PATH = "./data/grf_fdem_dataset_1d.npy"
    
    # 画出 3 个随机样本的地层结构
    plot_1d_models(DATA_PATH, num_samples_to_plot=3)