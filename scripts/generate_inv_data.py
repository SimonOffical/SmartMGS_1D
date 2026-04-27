import numpy as np
from joblib import Parallel, delayed
from generate_forward_data import forward_calc
import matplotlib.pyplot as plt
import numpy as np
# https://gemini.google.com/app/06ce96c606887c20
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
def compute_column(j, log_rho_guess, hh, step=1e-6):
    """
    使用中心差分法 (Central Difference) 计算对数域雅可比矩阵的第 j 列。
    对应 MATLAB 代码的逻辑。
    
    参数:
    j: 当前施加微扰的地层索引
    log_rho_guess: 当前模型的对数电阻率 log10(rho)
    hh: 模型各层厚度
    step: 步长系数 (对应 MATLAB 中的 step)
    
    返回:
    column_j: 雅可比矩阵的第 j 列
    """
    # 复制当前的对数模型
    log_rho_plus = log_rho_guess.copy()
    log_rho_minus = log_rho_guess.copy()
    
    # 计算微扰量 dm (与 MATLAB 保持一致: dm = step * m_inv(j))
    # 注意: 为了防止 log_rho 为 0 时 dm 为 0，通常会加一个绝对下限
    dm = step * np.abs(log_rho_guess[j]) 
    
    # 1. 在对数域进行正负微扰
    log_rho_plus[j] += dm
    log_rho_minus[j] -= dm
    
    # 2. 还原为真实电阻率
    rho_plus = 10 ** log_rho_plus
    rho_minus = 10 ** log_rho_minus
    
    # 3. 运行两次正演 (对应 MATLAB 中的 Fwd_new 和 Fwd_new2)
    hz_plus, hp_plus = forward_calc(rho_plus, hh)
    hz_minus, hp_minus = forward_calc(rho_minus, hh)
    
    # 提取线性数据 (您之前是将实部和虚部拼接，MATLAB 中是算比值，这里保持您的格式)

    V_real = 1e2*hz_plus.real/hp_plus.real
    V_imag = 1e2*hz_plus.imag/hp_plus.real
    data_plus = np.concatenate([V_real, V_imag])


    data_minus = np.concatenate([1e2*hz_minus.real/hp_minus.real, 1e2*hz_minus.imag/hp_minus.real])
    
    # 4. 中心差分公式求导: (f(x+h) - f(x-h)) / 2h
    column_j = (data_plus - data_minus) / (2 * dm)

    return column_j
from joblib import Parallel, delayed
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
def calculate_jacobian(log_rho, hh, n_jobs=-1, step=1e-4):
    """
    并行计算对数域的完整雅可比矩阵 (Jacobian Matrix) J。
    
    参数:
    log_rho: 待求模型的对数电阻率数组 log10(rho)
    hh: 地层厚度数组
    n_jobs: 并行调用的 CPU 核心数 (-1 为全部核心)
    step: 差分步长系数
    
    返回:
    J: 完整的对数雅可比矩阵，形状为 (2 * n_freqs, n_layers)
    """
    n_layers = len(log_rho)
    
    # 使用 Joblib 的 Parallel 分发各列的独立计算任务
    # tqdm 可以加在外面，但由于这里已经是底层引擎，为了速度通常不加进度条
    J_columns = Parallel(n_jobs=n_jobs)(
        delayed(compute_column)(j, log_rho, hh, step) 
        for j in range(n_layers)
    )
    
    # 将列表中的一维数组按列 (column) 拼接成标准的 2D 矩阵
    J = np.column_stack(J_columns)
    
    return J
def get_forward_data_percent(log_rho, hh):
    """
    辅助函数：输入对数模型，输出与 MATLAB 对齐的百分比相对观测数据 (%)。
    """
    rho_linear = 10 ** log_rho
    hz, hp = forward_calc(rho_linear, hh)
    
    v_real = 1e2 * hz.real / hp.real
    v_imag = 1e2 * hz.imag / hp.real
    
    return np.concatenate([v_real, v_imag])


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






def fdem_inv_log_dls(d_obs, hh, log_rho0, max_iter=50, lambda_init=0.05, lambda_decay=0.8, n_jobs=1):
    """
    基于阻尼最小二乘法 (DLS / Levenberg-Marquardt) 和空间平滑的对数域 1D FDEM 反演。
    
    参数:
    d_obs: 观测数据 (相对一次场的百分比 %)
    hh: 地层厚度
    log_rho0: 初始模型的对数电阻率 log10(rho)
    max_iter: 最大外部迭代次数
    lambda_init: 初始阻尼系数
    lambda_decay: 阻尼系数衰减因子
    n_jobs: 雅可比矩阵计算时的并行核心数 (-1 为全部核心)
    """
    m_k = log_rho0.copy()
    n_layers = len(log_rho0)
    lambda_k = lambda_init
    for i in range( max_iter):
        print(f"第 {i+1} 次迭代，当前 lambda: {lambda_k:.4e}")
        lambda_k = lambda_k * lambda_decay
        We = np.eye(n_layers)
        hz1_iter, hp1_iter = forward_calc(10 ** m_k, hh)
        v_real_iter = 1e2 * hz1_iter.real / hp1_iter.real
        v_imag_iter = 1e2 * hz1_iter.imag / hp1_iter.real
        d_pred_iter = np.concatenate([v_real_iter, v_imag_iter])
        # 计算雅可比矩阵
        delta_d = d_obs - d_pred_iter
        r_error = np.sqrt(np.mean(delta_d**2))
        print(f"当前 RMS 误差: {r_error:.4e}")
        J = calculate_jacobian(m_k, hh, n_jobs=n_jobs, step=1e-6)
        A = (J.T@J) + (lambda_k * We.T@We)
        g = J.T @ delta_d
        dm = np.linalg.solve(A, g)
        m_k += dm
        



    return 10 ** m_k


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

import numpy as np
import os

def load_npy_dataset(file_path):
    """
    读取 .npy 文件的通用函数。
    自动处理 numpy 数组和保存为 .npy 的字典格式。
    
    参数:
    file_path (str): .npy 文件的完整路径
    
    返回:
    data: 读取到的 numpy 数组或字典数据
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"报错: 找不到文件 '{file_path}'。请检查路径是否正确。")
        
    print(f"正在加载数据集: {file_path} ...")
    
    # allow_pickle=True 是必选项，因为您保存的数据包含了字典对象
    data = np.load(file_path, allow_pickle=True)
    
    # 判断是否为字典封装。如果是，使用 .item() 将其解包为真实的 Python 字典
    if data.ndim == 0 and data.dtype.names is None:
        data = data.item()
        
    return data


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
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# =======================================================
# 批量反演所需的辅助函数
# =======================================================
def calculate_confidence_label(m_true_linear, m_inv_linear, gamma=2.0):
    """计算神经网络的拟合目标：置信度标签 r"""
    relative_error = (m_true_linear - m_inv_linear) / m_true_linear
    # r_label = np.exp(-gamma * (relative_error ** 2))
    r_label = np.exp(-gamma * np.abs(relative_error))
    return r_label
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
def calculate_sensitivity_feature(log_rho, hh):
    """提取对数灵敏度特征 diag(J^T J)"""
    # 再次强调：底层计算必须是 n_jobs=1
    J = calculate_jacobian(log_rho, hh, n_jobs=1, step=1e-6)
    sensitivity_diag = np.sum(J ** 2, axis=0)
    max_sens = np.max(sensitivity_diag) + 1e-30 
    sensitivity_norm = sensitivity_diag / max_sens
    return np.log10(sensitivity_norm + 1e-8)
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
def process_single_sample(args):
    """单个样本的多进程执行单元"""
    i, rho_true_linear, hh, hz_real, hz_imag, hp_real, m0_log = args
    try:
        # 提取数据并转为百分比
        Vi = 1e2 * hz_real / hp_real
        Vj = 1e2 * hz_imag / hp_real
        d_obs_clean = np.concatenate([Vi, Vj])
        
       
        noise = 0
        d_obs_noisy = d_obs_clean + noise
        
        # 执行反演 (迭代15-20次通常足够)
        m_inv_linear = fdem_inv_log_dls(
            d_obs_noisy, hh, m0_log, 
            max_iter=50, 
            lambda_init=0.01, 
            lambda_decay=0.8,
            n_jobs=1  # 根据您的服务器核心数调整
        )
        
        # 计算供网络训练的特征和标签
        m_inv_log = np.log10(m_inv_linear)
        sens_feat = calculate_sensitivity_feature(m_inv_log, hh)
        r_label = calculate_confidence_label(rho_true_linear, m_inv_linear)
        
        return (i, m_inv_linear, sens_feat, r_label, True, None)
    except Exception as e:
        return (i, None, None, None, False, str(e))

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
# =======================================================
# 主力批量跑批函数
# =======================================================
def generate_synthetic_data(input_path="./data/grf_fdem_dataset_1d.npy", 
                            output_path="./data/inv_training_dataset_1d.npy", 
                            max_workers=30):
    """
    静默批量处理正演数据集，生成供 1D-ResNet 训练的最终数据集。
    默认使用 30 个核心榨干服务器性能。
    """
    print(f"🚀 开始加载数据集: {input_path}")
    try:
        dataset = load_npy_dataset(input_path)
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    n_total = dataset['resistivities'].shape[0]
    n_layers = dataset['resistivities'].shape[1]
    thicknesses = dataset['thicknesses']
    
    print(f"📦 准备多进程批量处理 {n_total} 个样本，动用核心数: {max_workers}...")

    # 预分配大矩阵内存
    all_m_fdem = np.zeros((n_total, n_layers))
    all_sensitivity = np.zeros((n_total, n_layers))
    all_r_labels = np.zeros((n_total, n_layers))
    
    # 初始盲猜模型: 100 欧姆米
    m0_log = np.ones(n_layers) * 1.0
    
    # 组装任务包
    tasks = []
    for i in range(n_total):
        rho_true = dataset['resistivities'][i]
        hh = thicknesses[i] if thicknesses.ndim > 1 else thicknesses
        hz_r = dataset['hz_real'][i]
        hz_i = dataset['hz_imag'][i]
        hp_r = dataset['hp_real'][i]
        tasks.append((i, rho_true, hh, hz_r, hz_i, hp_r, m0_log))

    success_count = 0
    failed_indices = []

    # 开启多进程池
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_sample, task): task for task in tasks}
        
        # 配合 tqdm 实时显示进度
        with tqdm(total=n_total, desc="反演进度", unit="样本") as pbar:
            for future in as_completed(futures):
                i, m_inv_linear, sens_feat, r_label, success, err_msg = future.result()
                if success:
                    # 检查反演出的电阻率是否有任何一层超出了 [1e-4, 1e6] 的范围
                    if np.any(m_inv_linear < 1e-4) or np.any(m_inv_linear > 1e6):
                        success = False
                        err_msg = "反演电阻率数值发散，超出合理物理范围 [1e-4, 1e6]"
                if success:
                    all_m_fdem[i, :] = m_inv_linear
                    all_sensitivity[i, :] = sens_feat
                    all_r_labels[i, :] = r_label
                    success_count += 1
                else:
                    failed_indices.append((i, err_msg))
                pbar.update(1)

    print(f"\n✅ 批量处理完成！成功: {success_count}/{n_total}，失败: {len(failed_indices)}")
    
    # 将输入和标签打包保存
    final_dataset = {
        'n_layers': n_layers,
        'thicknesses': thicknesses,
        'input_m_fdem': all_m_fdem,
        'input_sensitivity': all_sensitivity,
        'label_r_true': all_r_labels,
        'ground_truth_m': dataset['resistivities']
    }
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, final_dataset)
    print(f"💾 训练集已成功保存至: {output_path}")


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
import numpy as np
import pandas as pd
import os

def convert_npy_to_csv(npy_path="./data/inv_training_dataset_1d.npy", csv_path="./data/dataset_inspection.csv"):
    """
    将 1D FDEM 反演数据集从 npy 格式转换为平坦化的 CSV 格式，方便在 Excel 中查看。
    """
    print(f"正在读取文件: {npy_path} ...")
    if not os.path.exists(npy_path):
        print(f"❌ 错误: 找不到文件 {npy_path}")
        return

    # 1. 加载数据字典
    data = np.load(npy_path, allow_pickle=True).item()
    
    # 2. 提取各项数据
    m_inv = data['input_m_fdem']       # 反演电阻率 (N, n_layers)
    sens = data['input_sensitivity']   # 灵敏度特征 (N, n_layers)
    r_true = data['label_r_true']      # 置信度标签 (N, n_layers)
    m_true = data['ground_truth_m']    # 真实电阻率 (N, n_layers)
    
    n_samples, n_layers = m_inv.shape
    
    # 3. 处理地层厚度和深度坐标
    thicknesses = data['thicknesses']
    # 如果厚度是对所有样本通用的 (1D 数组)，则进行广播处理
    if thicknesses.ndim == 1:
        thicknesses = np.tile(thicknesses, (n_samples, 1))
        
    # 计算每层的中心深度
    z_top = np.cumsum(thicknesses, axis=1) - thicknesses
    z_center = z_top + thicknesses / 2
    # 对于最底层（通常假设为无限半空间，厚度设为0），中心深度可微调显示
    z_center[:, -1] = z_top[:, -1] + 5.0 

    # 4. 构建平坦化的 DataFrame
    print("🧠 正在进行数据平坦化处理...")
    rows = []
    
    # 为了防止 CSV 过大（如 5000 样本 * 20 层 = 10万行），这里可以循环构建
    # 或者使用 numpy 的 reshape 配合 pd.DataFrame
    indices = np.indices((n_samples, n_layers))
    sample_ids = indices[0].flatten()
    layer_ids = indices[1].flatten() + 1 # 从 1 开始计数
    
    df = pd.DataFrame({
        'Sample_ID': sample_ids,
        'Layer_ID': layer_ids,
        'Thickness_m': thicknesses.flatten(),
        'Depth_Center_m': z_center.flatten(),
        'True_Resistivity': m_true.flatten(),
        'Inverted_Resistivity': m_inv.flatten(),
        'Sensitivity_Log': sens.flatten(),
        'Confidence_Label_r': r_true.flatten()
    })

    # 5. 保存文件
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"✅ 转换完成！总计 {len(df)} 行数据已保存至: {csv_path}")



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
import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_training_sample(data_path="./data/inv_training_dataset_1d.npy", sample_index=None):
    """
    可视化生成的深度学习训练集中的单个样本。
    展示网络输入（反演模型、灵敏度）和网络目标（置信度标签 r）。
    
    参数:
    data_path (str): 数据集 .npy 文件的路径。
    sample_index (int, optional): 指定要查看的样本索引。如果不指定，则随机抽取。
    """
    print(f"=== 开始加载并可视化数据集: {data_path} ===")
    
    # 1. 检查文件与加载数据
    if not os.path.exists(data_path):
        print(f"❌ 找不到文件！请检查路径是否正确: {data_path}")
        return
        
    try:
        dataset = np.load(data_path, allow_pickle=True).item()
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    n_samples = dataset['input_m_fdem'].shape[0]
    
    # 2. 确定可视化的样本索引
    if sample_index is None:
        sample_index = np.random.randint(0, n_samples)
    elif sample_index >= n_samples or sample_index < 0:
        print(f"❌ 错误: sample_index 必须在 0 到 {n_samples-1} 之间。")
        return
        
    print(f"🎨 正在绘制第 {sample_index}/{n_samples} 个样本的特征分析图...")
    
    # 3. 提取当前样本的各项数据
    m_inv = dataset['input_m_fdem'][sample_index]
    m_true = dataset['ground_truth_m'][sample_index]
    sens = dataset['input_sensitivity'][sample_index]
    r_label = dataset['label_r_true'][sample_index]
    
    # 处理厚度数据 (兼顾统一厚度和随机厚度两种保存格式)
    if dataset['thicknesses'].ndim > 1:
        hh = dataset['thicknesses'][sample_index]
    else:
        hh = dataset['thicknesses']
        
    # 4. 准备画图所需的深度坐标
    z_top = np.concatenate(([0], np.cumsum(hh[:-1])))
    z_center = z_top + np.concatenate((hh[:-1] / 2, [hh[-2]]))
    
    # 辅助函数：将离散的分层数据转换为连续的阶梯图 (Step Plot)
    def prep_step_plot(res, depths):
        bottom_depth = depths[-1] + 1.0 # 为底层加一个展示厚度 (10米)
        depths_full = np.append(depths, bottom_depth)
        res_plot = np.repeat(res, 2)
        depth_plot = np.repeat(depths_full, 2)[1:-1]
        return res_plot, depth_plot

    m_true_p, d_true_p = prep_step_plot(m_true, z_top)
    m_inv_p, d_inv_p = prep_step_plot(m_inv, z_top)
    
    # ==========================================
    # 5. 开始绘制三合一面板图
    # ==========================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 8), sharey=True)
    
    # --- 子图 1: 电阻率剖面 (网络输入 1 & 真实模型) ---
    axes[0].plot(m_true_p, d_true_p, 'k-', linewidth=3, label='Ground Truth')
    axes[0].plot(m_inv_p, d_inv_p, 'r-', linewidth=2.5, label='DLS Inverted (Input 1)')
    axes[0].set_xscale('log')
    axes[0].invert_yaxis() # 深度轴向下
    axes[0].set_title('Resistivity Profile\n(Physics & Prediction)', fontsize=13, pad=10)
    axes[0].set_xlabel('Resistivity ($\Omega\cdot m$)', fontsize=12)
    axes[0].set_ylabel('Depth (m)', fontsize=12)
    axes[0].legend(loc='lower left')
    axes[0].grid(True, which='both', linestyle=':', alpha=0.7)
    
    # --- 子图 2: 物理灵敏度特征 (网络输入 2) ---
    axes[1].plot(sens, z_center, 'b-o', linewidth=2, markersize=6)
    axes[1].set_title('Sensitivity Feature\n(Network Input 2)', fontsize=13, pad=10)
    axes[1].set_xlabel('Log Normalized Sensitivity', fontsize=12)
    axes[1].grid(True, linestyle=':', alpha=0.7)
    
    # --- 子图 3: 置信度标签 (网络拟合目标 Y) ---
    # 使用颜色填充来直观显示置信度高低 (绿色高置信，红色低置信)
    axes[2].plot(r_label, z_center, 'g-o', linewidth=2, markersize=6)
    axes[2].fill_betweenx(z_center, 0, r_label, color='green', alpha=0.2)
    axes[2].fill_betweenx(z_center, r_label, 1, color='red', alpha=0.1)
    
    axes[2].set_title('Confidence Label $r$\n(Network Training Target)', fontsize=13, pad=10)
    axes[2].set_xlabel('Confidence Score (0 to 1)', fontsize=12)
    axes[2].set_xlim(-0.05, 1.05)
    axes[2].grid(True, linestyle=':', alpha=0.7)
    
    # 整体排版
    plt.suptitle(f'1D-ResNet Training Sample #{sample_index} Inspection', fontsize=16, y=1.02, fontweight='bold')
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


import numpy as np
import os

def clean_zero_samples(input_path="./data/inv_training_dataset_1d.npy", 
                       output_path="./data/inv_training_dataset_1d.npy"):
    print(f"🧹 开始加载原始数据集: {input_path}")
    data = np.load(input_path, allow_pickle=True).item()
    
    m_inv = data['input_m_fdem']
    original_count = m_inv.shape[0]
    
    # 核心逻辑：找出所有不是全 0 的行 (按行求绝对值之和，大于0说明有数据)
    # 返回一个布尔数组，例如 [True, False, True...]
    valid_mask = np.sum(np.abs(m_inv), axis=1) > 0
    
    clean_count = np.sum(valid_mask)
    print(f"📊 原始样本数: {original_count}")
    print(f"🗑️ 发现并剔除全0废弃样本: {original_count - clean_count} 个")
    
    if original_count == clean_count:
        print("✅ 数据集很干净，无需处理！")
        return

    # 使用 valid_mask 同步切割所有特征矩阵，保证数据对齐
    data['input_m_fdem'] = data['input_m_fdem'][valid_mask]
    data['input_sensitivity'] = data['input_sensitivity'][valid_mask]
    data['label_r_true'] = data['label_r_true'][valid_mask]
    data['ground_truth_m'] = data['ground_truth_m'][valid_mask]
    
    # 检查厚度矩阵如果是 2D 的，也需要一并切割
    if data['thicknesses'].ndim > 1:
        data['thicknesses'] = data['thicknesses'][valid_mask]
        
    # 保存干净的数据集
    np.save(output_path, data)
    print(f"💾 纯净版数据集已保存至: {output_path}")
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
# ==========================================
# 测试调用
# ==========================================
# if __name__ == "__main__":
if False:
    # 为了防止批量运行时打印太多刷屏信息，建议把 fdem_inv_log_dls 里面的 print 注释掉
    
    # 调用生成函数开始炼丹准备！
    generate_synthetic_data(
        input_path="./data/grf_fdem_dataset_1d.npy", 
        output_path="./data/inv_training_dataset_1d.npy",
        max_workers=10  # 根据您的 32 核服务器设置
    )
    clean_zero_samples()
    convert_npy_to_csv()




if __name__ == "__main__":
# if False:
    # 随机抽查一个样本
    for i in range(0 , 10):
        print(f"\n\n=== 可视化第 {i} 个样本 ===")
        visualize_training_sample(sample_index= i)
    
    # 如果您想查看特定索引的样本，可以这样调用：
    # visualize_training_sample(sample_index=5)


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
    import matplotlib.pyplot as plt
    import pandas as pd  # 引入 pandas 用于保存 CSV
    import os
    # 测试load_npy_dataset函数
    test_file_path = "./data/grf_fdem_dataset_1d.npy"
    try:
        dataset = load_npy_dataset(test_file_path)
        print("数据集加载成功！")
        print(f"数据集包含的键: {list(dataset.keys())}")
    except Exception as e:
        print(str(e))   
        exit() # 如果加载失败，直接退出
        
    # 随机挑选一个数据进行反演测试
    sample_index = np.random.randint(0, dataset['resistivities'].shape[0])
    # sample_index = 3  # 固定索引，方便调试和结果复现
    print(f"\n=== 选取第 {sample_index} 个样本进行反演测试 ===")
    
    # 提取真实模型和网格
    rho_true_linear = dataset['resistivities'][sample_index]
    
    # 注意：如果您的 thicknesses 是对所有样本一样的 1D 数组，就不用加 [sample_index]
    # 如果是二维数组，需要加 hh_sample = dataset['thicknesses'][sample_index]
    hh_sample = dataset['thicknesses'] 
    n_layers = len(rho_true_linear)
    
    # 提取二次场并计算百分比
    Vi = 1e2 * dataset['hz_real'][sample_index] / dataset['hp_real'][sample_index]
    Vj = 1e2 * dataset['hz_imag'][sample_index] / dataset['hp_real'][sample_index]
    d_obs_sample = np.concatenate([Vi, Vj])

    # 【修复 2】初始模型：盲猜 100 欧姆米 (不要用真实数据)
    m0_log = np.ones(n_layers) * 1.0  # log10(100) = 2.0
    
    # 【修复 1】反演接收：直接接收线性电阻率，千万别再取指数了！
    m_inv_linear = fdem_inv_log_dls(d_obs_sample, hh_sample, m0_log, max_iter=50, lambda_init=0.01, lambda_decay=0.8, n_jobs=-1)

    # 【修复 3】可视化：使用阶梯状折线图
    print("\n=== 绘制反演剖面对比图 ===")
    z_top = np.concatenate(([0], np.cumsum(hh_sample[:-1])))
    z_center = z_top + np.concatenate((hh_sample[:-1] / 2, [hh_sample[-2]]))
    print("\n=== 保存反演结果到 CSV ===")
    # 构建数据字典
    result_data = {
        'Layer_Index': np.arange(1, n_layers + 1),
        'Thickness_m': hh_sample,
        'Depth_Top_m': z_top,
        'Depth_Center_m': z_center,
        'True_Resistivity_Ohm_m': rho_true_linear,
        'Initial_Resistivity_Ohm_m': 10**m0_log,
        'Inverted_Resistivity_Ohm_m': m_inv_linear
    }
    # 转换为 DataFrame
    df_result = pd.DataFrame(result_data)
    
    # 确保输出目录存在
    os.makedirs("./data/results", exist_ok=True)
    
    # 保存 CSV 文件
    csv_filename = f"./data/results/inversion_result_sample_{sample_index}.csv"
    df_result.to_csv(csv_filename, index=False)
    print(f"✅ 结果已成功保存至: {csv_filename}")
    def prep_step_plot(res, depths):
        bottom_depth = depths[-1] + 1.0 # 为底层加一个展示厚度
        depths_full = np.append(depths, bottom_depth)
        res_plot = np.repeat(res, 2)
        depth_plot = np.repeat(depths_full, 2)[1:-1]
        return res_plot, depth_plot

    res_true_p, d_true_p = prep_step_plot(rho_true_linear, z_top)
    res_inv_p, d_inv_p = prep_step_plot(m_inv_linear, z_top)
    
    plt.figure(figsize=(6, 8))
    plt.plot(res_true_p, d_true_p, 'k-', linewidth=3, label='True Model (GRF)')
    plt.plot(10**m0_log, z_center, 'g--', linewidth=2, label='Initial Model')
    plt.plot(res_inv_p, d_inv_p, 'r-', linewidth=2.5, label='Inverted Model (DLS)')
    
    plt.gca().invert_yaxis()
    plt.xscale('log')
    plt.title(f'1D FDEM Inversion Test (Sample #{sample_index})', fontsize=14)
    plt.xlabel('Resistivity ($\Omega\cdot m$)', fontsize=12)
    plt.ylabel('Depth (m)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, which='both', linestyle=':', alpha=0.8)
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




# if __name__ == "__main__":
if False:
    import matplotlib.pyplot as plt
    np.random.seed(42) # 固定随机种子，方便复现结果
    
    n_layers = 20
    
    # 1. 设定等厚度地层网格
    depth_max = 20.0
    hh = np.ones(n_layers) * (depth_max / (n_layers - 1))
    hh[-1] = 0.0 
    
    # 2. 构造极其挑战的反演模型 (薄互层 + 强弱电阻率对比)
    # 背景为 100 欧姆米，包含 10 欧姆米的良导层和 1000 欧姆米的高阻层
    m_true_linear = np.ones(n_layers) * 100.0
    m_true_linear[4:7] = 10.0    # 浅部极低阻 (极其敏感)
    m_true_linear[12:16] = 1000.0 # 深部极高阻 (电磁法的盲区，极难反演)
    
    print("=== 1. 生成合成观测数据 ===")
    m_true_log = np.log10(m_true_linear)
    d_obs_clean = get_forward_data_percent(m_true_log, hh)
    
    # 【工业级测试】: 给数据添加 1% 的高斯随机噪声
    # 测试 DLS 算法的抗噪能力和正则化平滑效果
    
   
    
    # 3. 设定初始模型 (均匀半空间 100)
    m0_linear = np.ones(n_layers) * 100.0
    m0_log = np.log10(m0_linear)
    
    # 4. 执行对数域 DLS 反演
    print("\n=== 2. 开始执行反演 ===")
    # DLS 非常稳健，可以适当调大 max_iter，lambda_init 设为 0.1
    m_inv_linear = fdem_inv_log_dls(
        d_obs_clean, hh, m0_log, 
        max_iter=200, 
        lambda_init=0.01,
        lambda_decay=0.8,
        n_jobs=-1

    )
    
    # 5. 可视化结果
    print("\n=== 3. 绘制反演剖面对比图 ===")
    z_top = np.concatenate(([0], np.cumsum(hh[:-1])))
    z_center = z_top + np.concatenate((hh[:-1] / 2, [hh[-2]]))
    
    # 稳健的阶梯图处理函数（已修复画图维度错误）
    def prep_step_plot(res, depths):
        bottom_depth = depths[-1] + 1.0 # 为无限厚底层增加一个虚拟显示底界
        depths_full = np.append(depths, bottom_depth)
        res_plot = np.repeat(res, 2)
        depth_plot = np.repeat(depths_full, 2)[1:-1]
        return res_plot, depth_plot

    res_true_p, d_true_p = prep_step_plot(m_true_linear, z_top)
    res_inv_p, d_inv_p = prep_step_plot(m_inv_linear, z_top)
    
    plt.figure(figsize=(7, 9))
    plt.plot(res_true_p, d_true_p, 'k-', linewidth=3, label='True Model')
    plt.plot(m0_linear, z_center, 'g--', linewidth=2, label='Initial Model')
    plt.plot(res_inv_p, d_inv_p, 'r-', linewidth=2.5, label='Inverted Model (DLS + 1% Noise)')
    
    # 突出显示地下异常体的位置
    plt.axhspan(z_top[4], z_top[7], color='blue', alpha=0.1, label='Conductive Target')
    plt.axhspan(z_top[12], z_top[16], color='red', alpha=0.1, label='Resistive Target')
    
    plt.gca().invert_yaxis()
    plt.xscale('log')
    plt.title('Log-Domain DLS FDEM Inversion', fontsize=14, pad=15)
    plt.xlabel('Resistivity ($\Omega\cdot m$)', fontsize=12)
    plt.ylabel('Depth (m)', fontsize=12)
    
    # 设置坐标轴范围让图形更美观
    plt.xlim(1, 10000)
    plt.ylim(depth_max + 20, 0)
    
    plt.legend(fontsize=11, loc='lower left')
    plt.grid(True, which='both', linestyle=':', alpha=0.8)
    plt.tight_layout()
    plt.show()

    pass
    
    