# %% [markdown]
# # 瞬变电磁现场数据正演实现

# %% [markdown]
# ## hankel系数

# %%
# hankel系数函数
import numpy as np
from numba import njit
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
from numba import njit
@njit(nogil=True, fastmath=True)
def get_hankel_filter_coeffs():
    """
    初始化汉克尔变换(Hankel Transform)的滤波系数。
    对应原Fortran程序的 Subroutine filter。
    
    Returns:
        nc (int): 滤波系数总数
        ncnull (int): 零值位移或相关参数
        h0 (np.ndarray): J0 贝塞尔函数的滤波系数
        h1 (np.ndarray): J1 贝塞尔函数的滤波系数
    """
    
    nc = 100
    ncnull = 60

    # h0 系数 (对应 J0)
    # Fortran中的 data(h0(i), i=1, 48) 和 data(h0(i), i=49, 100)
    h0 = np.array([
        2.89878288e-07, 3.64935144e-07, 4.59426126e-07, 5.78383226e-07,
        7.28141338e-07, 9.16675639e-07, 1.15402625e-06, 1.45283298e-06,
        1.82900834e-06, 2.30258511e-06, 2.89878286e-06, 3.64935148e-06,
        4.59426119e-06, 5.78383236e-06, 7.28141322e-06, 9.16675664e-06,
        1.15402621e-05, 1.45283305e-05, 1.82900824e-05, 2.30258527e-05,
        2.89878259e-05, 3.64935186e-05, 4.59426051e-05, 5.78383329e-05,
        7.28141144e-05, 9.16675882e-05, 1.15402573e-04, 1.45283354e-04,
        1.82900694e-04, 2.30258630e-04, 2.89877891e-04, 3.64935362e-04,
        4.59424960e-04, 5.78383437e-04, 7.28137738e-04, 9.16674828e-04,
        1.15401453e-03, 1.45282561e-03, 1.82896826e-03, 2.30254535e-03,
        2.89863979e-03, 3.64916703e-03, 4.59373308e-03, 5.78303238e-03,
        7.27941497e-03, 9.16340705e-03, 1.15325691e-02, 1.45145832e-02,
        # Part 2 (i=49 to 100)
        1.82601199e-02, 2.29701042e-02, 2.88702619e-02, 3.62691810e-02,
        4.54794031e-02, 5.69408192e-02, 7.09873072e-02, 8.80995426e-02,
        1.08223889e-01, 1.31250483e-01, 1.55055715e-01, 1.76371506e-01,
        1.85627738e-01, 1.69778044e-01, 1.03405245e-01, -3.02583233e-02,
        -2.27574393e-01, -3.62173217e-01, -2.05500446e-01, 3.37394873e-01,
        3.17689897e-01, -5.13762160e-01, 3.09130264e-01, -1.26757592e-01,
        4.61967890e-02, -1.80968674e-02, 8.35426050e-03, -4.47368304e-03,
        2.61974783e-03, -1.60171357e-03, 9.97717882e-04, -6.26275815e-04,
        3.94338818e-04, -2.48606354e-04, 1.56808604e-04, -9.89266288e-05,
        6.24152398e-05, -3.93805393e-05, 2.48472358e-05, -1.56774945e-05,
        9.89181741e-06, -6.24131160e-06, 3.93800058e-06, -2.48471018e-06,
        1.56774609e-06, -9.89180896e-07, 6.24130948e-07, -3.93800005e-07,
        2.48471005e-07, -1.56774605e-07, 9.89180888e-08, -6.24130946e-08
    ])

    # h1 系数 (对应 J1)
    # Fortran中的 data(h1(i), i=1, 48) 和 data(h1(i), i=49, 100)
    h1 = np.array([
        1.84909557e-13, 2.85321327e-13, 4.64471808e-13, 7.16694771e-13,
        1.16670043e-12, 1.80025587e-12, 2.93061898e-12, 4.52203829e-12,
        7.36138206e-12, 1.13588466e-11, 1.84909557e-11, 2.85321327e-11,
        4.64471808e-11, 7.166694771e-11, 1.16670043e-10, 1.80025587e-10, # 注意: 源代码中第14个数据似乎有排版问题，这里根据上下文修正为 7.16694771e-11
        2.93061898e-10, 4.52203829e-10, 7.36138206e-10, 1.13588466e-09,
        1.84909557e-09, 2.85321326e-09, 4.64471806e-09, 7.16694765e-09,
        1.16670042e-08, 1.80025583e-08, 2.93061889e-08, 4.52203807e-08,
        7.36138149e-08, 1.13588452e-07, 1.84909521e-07, 2.85321237e-07,
        4.64471580e-07, 7.16694198e-07, 1.16669899e-06, 1.80025226e-06,
        2.93060990e-06, 4.52201549e-06, 7.36132477e-06, 1.13587027e-05,
        1.84905942e-05, 2.85312247e-05, 4.64449000e-05, 7.16637480e-05,
        1.16655653e-04, 1.79989440e-04, 2.92971106e-04, 4.51975783e-04,
        # Part 2 (i=49 to 100)
        7.35565435e-04, 1.13444615e-03, 1.84548306e-03, 2.84414257e-03,
        4.62194743e-03, 7.10980590e-03, 1.15236911e-02, 1.76434485e-02,
        2.84076233e-02, 4.29770596e-02, 6.80332569e-02, 9.97845929e-02,
        1.51070544e-01, 2.03540581e-01, 2.71235377e-01, 2.76073871e-01,
        2.16691977e-01, -7.83723737e-02, -3.40675627e-01, -3.60693673e-01,
        5.13024526e-01, -5.94724729e-02, -1.95117123e-01, 1.99235600e-01,
        -1.38521553e-01, 8.79320859e-02, -5.50697146e-02, 3.45637848e-02,
        -2.17527180e-02, 1.37100291e-02, -8.64656417e-03, 5.45462758e-03,
        -3.44138864e-03, 2.17130686e-03, -1.36998628e-03, 8.64398952e-04,
        -5.45397874e-04, 3.44122545e-04, -2.17126585e-04, 1.36997597e-04,
        -8.64396364e-05, 5.45397224e-05, -3.44122382e-05, 2.17126544e-05,
        -1.36997587e-05, 8.64396338e-06, -5.45397218e-06, 3.44122380e-06,
        -2.17126543e-06, 1.36997587e-06, -8.64396337e-07, 5.45397218e-07
    ])

    return nc, ncnull, h0, h1



# %%
# 使用示例
if __name__ == "__main__":
    nc, ncnull, h0, h1 = get_hankel_filter_coeffs()
    print(f"Loaded {len(h0)} coefficients for J0")
    print(f"First element of h0: {h0[0]}")
    print(f"Last element of h0: {h0[-1]}")

# %% [markdown]
# ## 指数运算

# %%
import math
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
def expc(x):
    """
    计算安全的指数函数 exp(x)，防止数值溢出。
    对应原 Fortran 程序的 Function expc。
    
    Args:
        x (float): 输入值
        
    Returns:
        float: e^x，其中 x 被限制在 [-650, 650] 之间。
    """
    # 使用 clip 函数限制范围，然后计算 exp
    # 注意: x 可以是单个数值也可以是 numpy 数组
    x_safe = np.clip(x, -650.0, 650.0)
    return np.exp(x_safe)



# %%
# 使用示例
if __name__ == "__main__":
    print(f"expc(10)   = {expc(10)}")      # 正常计算
    print(f"expc(800)  = {expc(800)}")     # 被截断为 exp(650)
    print(f"expc(-900) = {expc(-900)}")    # 被截断为 exp(-650) -> 接近0

# %% [markdown]
# ## 层状介质阻抗递推公式
# 对于第 $i$ 层介质，其垂直方向的传播常数 $\alpha_i$ 定义为：
# $$\alpha_i = \sqrt{\lambda^2 + k_i^2} = \sqrt{\lambda^2 + j\omega\mu_0\sigma_i}$$
# - $\lambda$ 是积分变量（水平波数，代码中的 u）。
# - $\omega = 2\pi f$ 是角频率。
# - $\sigma_i = 1/\rho_i$ 是第 $i$ 层的电导率。
# - $\mu_0 = 4\pi \times 10^{-7} \text{ H/m}$ 是真空磁导率。
# 假设第 $n$ 层（底层）为均匀半空间，向下无限延伸，其初始等效值 $B_n$（即代码中的 b_val 初始值）为：
# $$B_n = \alpha_n = \sqrt{\lambda^2 + j\omega\mu_0\sigma_n}$$
# 从底层（第 $n$ 层）开始，逐层向上递推至地表。对于第 $i$ 层（厚度为 $h_i$），其顶部的等效响应 $B_i$ 与其底部（即第 $i+1$ 层顶部）的响应 $B_{i+1}$ 的关系为：
# $$B_i = \alpha_i \frac{B_{i+1} + \alpha_i \tanh(\alpha_i h_i)}{\alpha_i + B_{i+1} \tanh(\alpha_i h_i)}$$
# 为了防止计算 $\tanh(x)$ 时发生数值溢出（当实部很大时），代码利用了恒等式 $\tanh(x) = \frac{1-e^{-2x}}{1+e^{-2x}}$ 进行变形：计算衰减因子：$S_{1,i} = e^{-2\alpha_i h_i}$计算辅助项： $ S_{2,i} = \frac{1 - S_{1,i}}{1 + S_{1,i}}$ （即 $\tanh(\alpha_i h_i)$）代入递推：
# $$B_i = \alpha_i \frac{B_{i+1} + \alpha_i S_{2,i}}{\alpha_i + B_{i+1} S_{2,i}}$$
# 
# 经过从 $i = n-1$ 到 $i = 1$ 的循环递推，函数最终返回 $B_1$。
# 
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
# %%
import numpy as np
@njit(nogil=True, fastmath=True)
def calc_layer_response(rho, hh, f, u, nlayer):
    """
    计算层状大地的电磁响应核函数 (Kernel Function)。
    对应原 Fortran 程序的 Function b。
    
    实现了从底层向上层的递推计算 (Wait's Recursion)。
    
    Args:
        rho (list or np.ndarray): 各层电阻率数组 [Ohm-m]
        hh (list or np.ndarray): 各层厚度数组 [m]
        f (float): 频率 [Hz]
        u (float): 积分变量 (波数/lambda)
        nlayer (int): 层数
        
    Returns:
        complex: 地表的等效复波数/响应值
    """
    
    # 物理常数
    PI = np.pi
    MU0 = 4e-7 * PI
    
    # 角频率
    omega = 2.0 * PI * f
    
    # ---------------------------------------------------------
    # 1. 初始底层 (Bottom Layer / Half-space)
    # ---------------------------------------------------------
    # Fortran索引为 nlayer (1-based)，Python为 nlayer-1 (0-based)
    # 计算底层的垂直复波数 alpha_n = sqrt(u^2 + i*omega*mu/rho_n)
    
    rho_bottom = rho[nlayer - 1]
    val_bottom = u**2 + 1j * MU0 * omega / rho_bottom
    b_val = np.sqrt(val_bottom)
    
    # 如果只有一层，直接返回
    if nlayer == 1:
        return b_val

    # ---------------------------------------------------------
    # 2. 向上递推 (Recursion Loop)
    # ---------------------------------------------------------
    # Fortran: Do i = nlayer-1, 1, -1
    # Python: range(nlayer-2, -1, -1) -> 对应层索引 nlayer-2 到 0
    
    for i in range(nlayer - 2, -1, -1):
        rho_curr = rho[i]
        h_curr = hh[i]
        
        # 当前层的垂直波数 alpha
        alpha_sq = u**2 + 1j * MU0 * omega / rho_curr
        alpha = np.sqrt(alpha_sq)
        
        # 计算 s1 = exp(-2 * alpha * h)
        # 对应 Fortran 中的溢出保护: If (dreal(2.D0*alpha*hh4(i))<400.D0)
        arg = 2.0 * alpha * h_curr
        
        if np.real(arg) < 400.0:
            s1 = np.exp(-arg)
        else:
            s1 = 0.0 + 0.0j  # 指数极小，近似为0
            
        # 计算 s2 = tanh(alpha * h)
        # 利用恒等式: tanh(x) = (1 - e^-2x) / (1 + e^-2x)
        s2 = (1.0 - s1) / (1.0 + s1)
        
        # Wait 递推公式
        # b = alpha * (b + alpha * s2) / (alpha + b * s2)
        # 这里 gam=1.0 被省略了
        numerator = b_val + alpha * s2
        denominator = alpha + b_val * s2
        
        b_val = alpha * (numerator / denominator)
        
    return b_val



# %%
# 使用示例
if __name__ == "__main__":
    # 3层模型示例: 
    # rho = [100, 10, 100] (欧姆米)
    # hh = [100, 50] (米, 最后一层厚度不需要)
    rho_test = np.array([100.0, 10.0, 100.0], dtype=np.float64)
    hh_test = np.array([100.0, 50.0, 0.0], dtype=np.float64) # 最后一个厚度在算法中不使用
    freq = 1000.0
    u_val = 0.001
    
    result = calc_layer_response(rho_test, hh_test, freq, u_val, 3)
    print(f"Result: {result}")

# %% [markdown]
# ## t3核函数
# 函数 t3 实际上是在计算如下形式的无穷积分：
# $$T_3(r, z) = \frac{1}{r} \int_{0}^{\infty} \left[ R_{\text{TE}}(\lambda) \cdot e^{-\lambda z} \cdot \lambda^2 \right] J_0(\lambda r) d\lambda$$
# - $\lambda$：空间波数（对应代码中的 u）。
# - $R_{\text{TE}}(\lambda)$：反射系数，由层状介质阻抗递推得到（对应代码中的 reflection）。
# $$
# R_{\text{TE}}(\lambda) = \frac{\hat{b}(\lambda) - \lambda}{\hat{b}(\lambda) + \lambda}
# $$ 
# 
# 其中 $\hat{b}(\lambda)$ 是由 calc_layer_response 计算的地表等效输入阻抗。
# - $e^{-\lambda z}$：垂直方向的指数衰减项（对应代码中的 decay）。
# - $\lambda^2$：核函数中的权重项（对应代码中的 u**2）。
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
# %%
@njit(nogil=True, fastmath=True)
def t3(rho, hh, f, z, nlayer, r):
    """
    计算 T3 电磁场积分项。
    
    Args:
        rho (list/array): 各层电阻率
        hh (list/array): 各层厚度
        f (float): 频率
        z (float): 垂直距离
        nlayer (int): 层数
        r (float): 收发距
    
    Returns:
        complex: 积分结果
    """
    
    # [修改处] 直接在函数内部获取滤波系数
    nc, ncnull, h0, h1 = get_hankel_filter_coeffs()
    
    # 常数定义
    fac = 0.1 * np.log(10.0)
    
    s = 0.0 + 0.0j
    
    # 滤波卷积循环
    for nn in range(1, nc + 1):
        nu = nn
        
        # 索引转换 (mn 是 Fortran 的索引)
        mn = nc - nn + 1
        h0_idx = mn - 1 
        
        nnn = ncnull - nc + nu
        exponent = -(nnn - 1) * fac
        
        if exponent < -700:
            u = 0.0
        else:
            u = np.exp(exponent) / r
            
        if u == 0: continue 

        # 计算 Kernel b
        b_val = calc_layer_response(rho, hh, f, u, nlayer)
        
        # 计算被积函数
        reflection = (b_val - u) / (b_val + u)
        
        decay_arg = -u * z
        if decay_arg < -650.0:
            decay = 0.0
        else:
            decay = np.exp(decay_arg)
            
        s1 = reflection * decay * (u**2)
        
        s += s1 * h0[h0_idx]
        
    return s / r

# %%
# ==========================================
# 4. 测试代码
# ==========================================
if __name__ == "__main__":
    # 参数设置
    rho_test = np.array([100.0, 10.0], dtype=np.float64) # 两层电阻率
    hh_test = np.array([50.0, 0.0], dtype=np.float64)   # 第一层厚50m (第二层无限厚)
    freq = 1000.0             # 1000 Hz
    z_dist = 10.0             # 收发高度差/深度
    r_dist = 100.0            # 收发距
    n_lay = 2
    
    print("开始计算 T3 (内嵌滤波系数调用)...")
    result = t3(rho_test, hh_test, freq, z_dist, n_lay, r_dist)
    print(f"Result t3: {result}")

# %% [markdown]
# ## forward函数
# 根据传入参数计算一次场或者二次场
# 
# - A. 一次场 (Primary Field, item == 8)
# 
# 一次场是指磁偶极子在自由空间（忽略大地存在）中产生的直接场。代码使用的是垂直磁偶极子磁场分量的解析公式：$$H_{f,\text{primary}} = \frac{1}{4\pi} \left[ \frac{3(z_{\text{plus}})^2 - (r_{\text{plus}})^2}{(r_{\text{plus}})^5} \right]$$
# 参数对应：$z_{\text{plus}}$：代码中传入的参数，物理上代表接收点与发射源的垂直距离 $|z_r - z_t|$。$r_{\text{plus}}$：接收点到发射源的直线距离，即 $\sqrt{x^2 + y^2 + z_{\text{plus}}^2}$。
# 
# - 二次场 (Secondary Field, item == 2)
# 
# 二次场是指由大地（层状介质）反射产生的磁场。代码通过调用数值积分函数 t3 来实现：
# $$H_{f,\text{secondary}} = -\frac{1}{4\pi} T_3(\rho, z_{\text{minus}})$$
# 参数对应：$T_3$：即之前讨论的汉克尔积分项。$z_{\text{minus}}$：代码中用于控制指数衰减项的参数。在电磁感应正演中，若 $z$ 轴向下为正，地面为 $0$，发射器在 $-h_t$，接收器在 $-h_r$，则对于地面的镜像反射距离为 $z_{\text{minus}} = h_t + h_r$。
# 

# %%
import numpy as np
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
# 注意：此函数依赖于之前定义的 t3 函数
# 如果 t3 在另一个文件中，请使用 from ... import t3

def forward(rho, hh, f, item, zplus, zminus, nlayer, r):
    """
    计算频率域电磁场响应。
    对应原 Fortran 的 Subroutine forward (仅 item=2 和 8).
    
    Args:
        rho (list/array): 各层电阻率
        hh (list/array): 各层厚度
        f (float): 频率
        item (int): 计算选项 (2=二次场, 8=一次场)
        zplus (float): ht + hr (用于一次场计算)
        zminus (float): ht - hr (用于二次场计算)
        nlayer (int): 层数
        r (float): 收发距 (对应原 Common /para/r，用于 item=2)
        
    Returns:
        complex: 计算结果 fun (通常是磁感应强度 B)
    """
    
    # 物理常数
    PI = np.pi
    MU0 = 4e-7 * PI
    
    # --- 1. 原代码内部定义的几何参数 ---
    # 注意：Fortran代码中在这里硬编码了 x=0.59 用于计算 rplus
    x = 0.59
    y = 0.0
    r1 = np.sqrt(x**2 + y**2)
    
    # rplus 是源到接收点的直线距离（用于解析解）
    rplus = np.sqrt(r1**2 + zplus**2)
    
    hf = 0.0 + 0.0j

    # --- 2. 根据 item 选择计算公式 ---
    
    if item == 2:
        # ==========================================
        # Item 2: 计算二次场 (Secondary Field)
        # ==========================================
        # 公式: hf = -t3(...) / (4 * pi)
        # 依赖于数值积分函数 t3
        
        # 调用 t3 (注意：t3 需要 r 作为参数)
        val_t3 = t3(rho, hh, f, zminus, nlayer, r)
        
        hf = -val_t3 / (4.0 * PI)
        
    elif item == 8:
        # ==========================================
        # Item 8: 计算一次场 (Primary Field)
        # ==========================================
        # 公式: hf = (3*zplus^2 - rplus^2) / rplus^5 / (4 * pi)
        # 这是垂直磁偶极子在自由空间的解析解
        
        numerator = 3 * zplus**2 - rplus**2
        denominator = (rplus**5) * (4.0 * PI)
        
        hf = numerator / denominator
        
    else:
        # 其他情况未实现
        raise ValueError(f"Item {item} is not supported. Only 2 and 8 are allowed.")

    # --- 3. 最终单位转换 ---
    # Fortran: fun = hf * 4.D-7 * pi
    # 即: fun = hf * MU0
    # 将磁场强度 H 转换为 磁感应强度 B
    
    fun = hf * MU0
    
    return fun

# %%
if __name__ == "__main__":
    # 假设你已经定义了 t3, calc_layer_response, get_hankel_filter_coeffs
    
    # 模拟数据
    rho = [100, 100]
    hh = [50, 0]
    freq = 10000.0
    nlayer = 2
    r_offset = 50.0 # 收发距
    z_plus = 2.0    # h_tx + h_rx
    z_minus = 0.0   # h_tx - h_rx (共面)
    
    # 1. 计算二次场 (Item 2)
    try:
        res2 = forward(rho, hh, freq, 2, z_plus, z_minus, nlayer, r_offset)
        print(f"Item 2 (Secondary): {res2}")
    except NameError:
        print("缺少依赖函数 t3，无法计算 item 2")

    # 2. 计算一次场 (Item 8) - 不依赖 t3，可直接计算
    res8 = forward(rho, hh, freq, 8, z_plus, z_minus, nlayer, r_offset)
    print(f"Item 8 (Primary):   {res8}")

# %% [markdown]
# ## frt函数
# 
# 将 forward 函数计算得到的频率域响应（通常是磁感应强度 $B(\omega)$），根据发射电流的波形进行频谱加权，从而得到特定电流波形激发的频率域响应。
# 
# 函数返回的复数响应 $\text{fun}(\omega)$ 可以表示为：
# 
# $$\text{fun}(\omega) = B(\omega) \cdot I(\omega)$$
# 
# 其中：$B(\omega)$：由 forward 函数计算的单位电流下的频率域磁感应强度。$I(\omega)$：波形转换因子（频谱修正项），取决于发射电流波形 ic。
# 
# - 脉冲波响应 (Impulse Response, ic == 0)
# 在数学上，脉冲波对应于单位脉冲信号（Dirac Delta），其频谱为常数 1。
# 
# $$I(\omega) = 1$$
# 
# 代码实现：iomega = 1.0 + 0.0j。此时 fun 直接等于 forward 的结果。
# 
# - 阶跃波响应 (Step Wave Response, ic == 1)
# 在数学上，阶跃信号的频率域变换包含 $1/j\omega$ 项。由于磁感应强度 $B$ 的产生与磁通量的变化率（导数）有关，而阶跃电流在断开瞬间相当于一个 $\delta$ 函数的变化，因此需要对频率域响应进行积分处理：
# 
# $$I(\omega) = \frac{1}{-j\omega} = \frac{j}{\omega}$$
# 
# 代码实现：iomega = 1.0 / (-1j * omega)。
# 这里的 $\omega = 2\pi f$ 是角频率。
# 

# %%
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

# 假设 forward 函数已经在同一环境或模块中定义
# from your_module import forward

def frt(rho, hh, item, zplus, zminus, ic, nlayer, frq, r):
    """
    计算频率域电磁响应（包含波形转换）。
    对应原 Fortran 的 Subroutine frt。
    
    Args:
        rho (list/array): 各层电阻率
        hh (list/array): 各层厚度
        item (int): 计算项 (2=二次场, 8=一次场)
        zplus (float): 几何参数 ht + hr
        zminus (float): 几何参数 ht - hr
        ic (int): 源波形类型 (0=脉冲 Impulse, 1=阶跃 Step wave)
        nlayer (int): 层数
        frq (list/array): 频率列表 [Hz]
        r (float): 收发距 (原 Fortran 通过 Common 块传递，这里需显式传入)
        
    Returns:
        np.ndarray: 复数结果数组 (对应原程序的 fun)
    """
    
    # 确保频率是 numpy 数组
    frq_arr = np.array(frq, dtype=np.float64)
    nfrq = len(frq_arr)
    
    # 初始化输出数组 (复数)
    fun = np.zeros(nfrq, dtype=np.complex128)
    
    # 常量
    PI = np.pi
    
    # 循环计算每个频率点
    for i in range(nfrq):
        f = frq_arr[i]
        
        # 1. 调用 forward 计算基础场值 (B场)
        # 注意：这里需要传入 r，因为 forward 内部的 t3 需要它
        # func(item, i) 在 Fortran 中是存储单元，这里直接获取返回值
        raw_field = forward(rho, hh, f, item, zplus, zminus, nlayer, r)
        
        # 2. 计算角频率
        omega = 2.0 * PI * f
        
        # 3. 根据波形计算转换因子 iomega
        if ic == 0:
            # Impulse (脉冲波)
            iomega = 1.0 + 0.0j
        elif ic == 1:
            # Step wave (阶跃波)
            # Fortran: 1.D0/((0.,-1.D0)*omega) -> 1 / (-j * omega)
            # 1 / (-j * w) = j / w
            iomega = 1.0 / (-1j * omega)
        else:
            # 默认情况 (防止未定义行为)
            iomega = 1.0 + 0.0j

        # 4. 应用波形因子
        fun[i] = raw_field * iomega
        
    return fun

# ==========================================
# 依赖函数 (前序步骤中已定义，为了完整性在此简略保留)
# ==========================================

# # 为了让代码可运行，这里提供一个简单的 forward 桩函数(Stub)
# # 实际使用时，请确保你包含了之前转换好的完整 forward 函数
# def forward(rho, hh, f, item, zplus, zminus, nlayer, r):
#     # 这里只是为了演示 frt 能跑通，实际逻辑请使用之前生成的 forward 代码
#     # 模拟返回一个复数场值
#     mu0 = 4e-7 * np.pi
#     fake_response = (1.0 + 1j) * mu0 / (r + 1e-3) 
#     return fake_response



# %%
# ==========================================
# 测试代码
# ==========================================
if __name__ == "__main__":
    # 模拟输入数据
    rho_test = [100.0, 10.0]
    hh_test = [50.0, 0.0]
    n_lay = 2
    
    # 频率扫描范围
    freqs = [100.0, 1000.0, 10000.0]
    
    # 几何参数
    ht = 30.0
    hr = 30.0
    z_plus = ht + hr
    z_minus = ht - hr
    r_dist = 100.0
    
    # 1. 计算脉冲响应 (ic=0), 二次场 (item=2)
    print("--- Impulse Response (ic=0) ---")
    res_impulse = frt(rho_test, hh_test, 2, z_plus, z_minus, 0, n_lay, freqs, r_dist)
    print(res_impulse)
    
    # 2. 计算阶跃响应 (ic=1), 二次场 (item=2)
    print("\n--- Step Response (ic=1) ---")
    res_step = frt(rho_test, hh_test, 2, z_plus, z_minus, 1, n_lay, freqs, r_dist)
    print(res_step)

# %% [markdown]
# ## forwardprocess函数
# 首先确定了发射线圈与接收线圈的相对空间位置：
# - 水平收发距 $\rho$：$\rho = \sqrt{x_r^2 + y_r^2}$
# - 一次场垂直距离 $z_{primary}$：$z_{plus} = |h_t - h_r|$ （源与接收器的直接高差）
# - 二次场垂直距离 $z_{secondary}$：$z_{minus} = h_t + h_r$ （源到地面再回到接收器的总垂直路程，基于镜像原理）
# 
# 通过调用 frt 函数，计算单位电流下的基础响应：
# - 二次场基础值：$\tilde{H}_{zs}(\omega) = \text{frt}(\rho, z_{minus}, \text{item}=2)$
# - 一次场基础值：$\tilde{H}_{zp}(\omega) = \text{frt}(\rho, z_{plus}, \text{item}=8)$
# 
# 装置参数补偿公式将物理场转化为具体仪器的测量值。
# $$V(\omega) = - \mathcal{M} \cdot \tilde{H}(\omega) \cdot A_{eff}$$
# 代码中使用了一个特定的近似系数来描述发射源的强度 $\mathcal{M}$：
# $$\mathcal{M} = 4 \cdot a^2 \cdot n_t \cdot I(\omega)$$
# $a$: 发射线圈边长 (rt)$n_t$: 发射线圈匝数 (nturn)$I(\omega)$: 发射电流 (current)
# 
# 描述接收线圈截获磁通量的能力 $A_{eff}$：
# $$A_{eff} = \pi \cdot r_r^2 \cdot n_r$$
# $r_r$: 接收线圈半径 (rr)，$n_r$: 接收线圈匝数 (nturn1)
# 
# 最终返回的二次场 $H_{z1}$ 和一次场 $H_{p1}$ 分别为：
# $$H_{z1}(\omega) = - (4 a^2 n_t I) \cdot \tilde{H}_{zs}(\omega) \cdot (\pi r_r^2 n_r)$$
# 
# $$H_{p1}(\omega) = - (4 a^2 n_t I) \cdot \tilde{H}_{zp}(\omega) \cdot (\pi r_r^2 n_r)$$
# 
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
# %%
import numpy as np

# 假设 frt 函数已经在同一个文件中定义
# def frt(rho, hh, item, zplus, zminus, ic, nlayer, frq, r): ...

def forward_process(rho, hh, fq, nturn, nturn1, xr, hr, rt, rr, current, shape = 'rectangle'):
    """
    频率域电磁法正演主流程。
    对应原 Fortran 的 Subroutine forwardprocess。
    
    Args:
        rho (list/array): 电阻率向量
        hh (list/array): 厚度向量 (最后一层厚度通常无效)
        fq (list/array): 频率扫描向量 [Hz]
        nturn (int): 发射线圈匝数
        nturn1 (int): 接收线圈匝数
        xr (float): 收发距 (偏心距)
        hr (float): 接收线圈高度 (代码中设定 ht=hr)
        rt (float): 发射线圈半径
        rr (float): 接收线圈半径
        current (list/array): 发射电流向量 (对应每个频率)
        
    Returns:
        tuple: (hz1, hp1)
            hz1 (np.ndarray): 二次场 (复数数组)
            hp1 (np.ndarray): 一次场 (复数数组)
    """
    
    # --- 1. 基础数据准备 ---
    # 确保输入是 numpy 数组
    rho = np.array(rho)
    hh = np.array(hh)
    fq = np.array(fq)
    current = np.array(current)
    
    nlayer = len(rho)
    nfreq = len(fq)
    
    # 常数
    PI = np.pi
    
    # --- 2. 几何参数设置 (Geometry) ---
    # 对应 Fortran: yr=0, ht=hr
    yr = 0.0
    ht = hr
    
    # 对应 Fortran: zplus = ht - hr, zminus = ht + hr
    # zplus: 发射与接收的高差 (对于一次场)
    # zminus: 发射与接收的高度和 (对于二次场/镜像)
    zplus = ht - hr
    zminus = ht + hr
    
    # 收发距 r (对应 Common /para/r)
    r = np.sqrt(xr**2 + yr**2)
    # rplus 实际上在 internal forward 函数中可能会被用到，
    # 但在这里 python 版本的 frt/forward 会内部处理，或者通过参数传递 r
    
    # --- 3. 计算场值 (Field Calculation) ---
    ic = 0  # 对应 Fortran: impulse response
    
    # 计算二次场 (hz1) -> item = 2
    # 注意: Python 的 frt 返回数组，而不是像 Fortran 那样原地修改
    hz1_raw = frt(rho, hh, 2, zplus, zminus, ic, nlayer, fq, r)
    
    # 计算一次场 (hp1) -> item = 8
    hp1_raw = frt(rho, hh, 8, zplus, zminus, ic, nlayer, fq, r)
    
    # --- 4. 物理量转换与校正 (Post-processing) ---
    # 对应 Fortran 中的 Do i = 1, nfreq 循环
    
    # 发射矩因子 Calculation
    # Fortran: ti0 = current(i); tm = 4*rt*rt*nturn*ti0
    # 注意: 这里严格照搬 Fortran 代码公式 (4 * rt^2)，
    # 虽然物理上磁矩通常是 pi*rt^2，但我们保持与原代码一致。
    if shape == 'rectangle':
        tm_arr = 4.0 * (rt**2) * nturn * current
    elif shape == 'circle':
        tm_arr = PI*rt^2*nturn*current
    # 接收线圈有效面积因子
    # factor = pi * rr**2 * nturn1
    rx_factor = PI * (rr**2) * nturn1
    
    # 应用公式:
    # hz1(i) = -tm * hz1(i) * rx_factor
    # hp1(i) = -tm * hp1(i) * rx_factor
    
    hz1 = -tm_arr * hz1_raw * rx_factor
    hp1 = -tm_arr * hp1_raw * rx_factor
    
    return hz1, hp1

# %%
# ==========================================
# 简单的调用示例 (前提: 其他函数已定义)
# ==========================================
if __name__ == "__main__":
    # 模拟输入参数
    rho_input = [100.0, 50.0]  # 两层地
    hh_input = [30.0, 0.0]     # 第一层厚30m
    fq_input = [1000.0, 10000.0, 50000.0] # 频率
    current_input = [10.0, 10.0, 10.0]    # 电流
    
    # 装置参数
    nturn_tx = 3
    nturn_rx = 1600
    xr_val = 50.0   # 收发距
    hr_val = 1.0    # 高度
    rt_val = 0.5    # Tx 半径
    rr_val = 0.2    # Rx 半径
    
    print("正在计算 forwardprocess...")
    
    # 尝试调用 (需要 ensure frt, forward 等函数存在)
    try:
        hz_res, hp_res = forward_process(
            rho_input, hh_input, fq_input, 
            nturn_tx, nturn_rx, xr_val, hr_val, rt_val, rr_val, 
            current_input
        )
        
        print(f"频率: {fq_input}")
        print(f"二次场 (Hz): {hz_res}")
        print(f"一次场 (Hp): {hp_res}")
        
    except NameError as e:
        print(f"运行失败: {e}")
        print("提示: 请确保将之前生成的 t3, forward, frt, get_hankel_filter_coeffs 等函数放在同一个文件中。")

# %%

import numpy as np
import matplotlib.pyplot as plt

def reconstruct_time_domain_signals(freqs, hz_complex, num_periods=1, points_per_period=100):
    """
    将频域复数场值还原为时域正弦波形
    """
    time_signals = []
    
    for i, f in enumerate(freqs):
        # 1. 提取物理量
        omega = 2 * np.pi * f
        amplitude = np.abs(hz_complex[i])  # 振幅
        phase = np.angle(hz_complex[i])    # 相位 (弧度)
        
        # 2. 生成时间向量 (计算 num_periods 个周期)
        T = 1 / f
        t = np.linspace(0, num_periods * T, points_per_period)
        
        # 3. 还原波形: A * sin(wt + phi)
        # 注意：电磁学中常使用 e^{jwt}，对应实部即为 cos，这里用 sin 表达波动
        signal = amplitude * np.sin(omega * t + phase)
        
        time_signals.append((t, signal, amplitude, phase))
        
    return time_signals

if __name__ == "__main__":
    # --- 1. 模拟正演输出的数据 (基于你的输入结果) ---
    fq_input = np.array([1000.0, 10000.0, 50000.0])
    # 选取的二次场复数结果 (Hz)
    hz_res = np.array([
        1.00979923e-10 + 2.06967572e-10j, 
        9.26232023e-10 + 2.67600062e-10j, 
        1.03385014e-09 - 2.07247506e-09j
    ])

    # --- 2. 还原时域波形 ---
    signals = reconstruct_time_domain_signals(fq_input, hz_res)

    # --- 3. 绘图展示 ---
    plt.figure(figsize=(12, 8))
    plt.style.use('seaborn-v0_8-whitegrid')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i in range(len(fq_input)):
        t, sig, amp, phi = signals[i]
        
        # 归一化时间轴以便在同一个图对比 (t/T)
        t_normalized = t * fq_input[i] 
        
        plt.plot(t_normalized, sig, label=f'Freq: {fq_input[i]/1000:.0f} kHz', color=colors[i], lw=2)
        
        # 打印物理参数
        print(f"频率 {fq_input[i]:>7} Hz | 振幅: {amp:.4e} | 相位(度): {np.degrees(phi):.2f}°")

    plt.axhline(0, color='black', lw=1, alpha=0.3)
    plt.title("Time Domain Reconstruction of Secondary Field ($H_z$)", fontsize=14)
    plt.xlabel("Normalized Time (Periods $t/T$)", fontsize=12)
    plt.ylabel("Magnetic Field Intensity (A/m)", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.show()



# %% [markdown]
# # 对比py和fortran结果

# %%
import numpy as np
import matplotlib.pyplot as plt
def verify_results():
    # ================================
    # 1. 设置完全相同的参数 (Python侧)
    # ================================
    rho = [100.0, 50.0]
    hh = [30.0, 0.0]
    fq = [1000.0, 10000.0, 50000.0]
    current = [10.0, 10.0, 10.0]
    
    nturn = 3
    nturn1 = 1600
    xr = 50.0
    hr = 1.0
    rt = 0.5
    rr = 0.2

    # ================================
    # 2. 运行 Python 代码
    # ================================
    hz_py, hp_py = forward_process(
        rho, hh, fq, nturn, nturn1, xr, hr, rt, rr, current
    )

    # ================================
    # 3. 读取 Fortran 结果
    # ================================
    try:
        fort_data = np.loadtxt('fortran_result.txt')
    except IOError:
        print("错误：未找到 fortran_result.txt。请先运行 Fortran 程序。")
        return

    # Fortran 文件列结构: [Freq, Hz_Real, Hz_Imag, Hp_Real, Hp_Imag]
    hz_fort_real = fort_data[:, 1]
    hz_fort_imag = fort_data[:, 2]
    hp_fort_real = fort_data[:, 3]
    hp_fort_imag = fort_data[:, 4]

    # 组合成复数
    hz_fort = hz_fort_real + 1j * hz_fort_imag
    hp_fort = hp_fort_real + 1j * hp_fort_imag

    # ================================
    # 4. 计算误差并打印
    # ================================
    print(f"{'Freq(Hz)':<10} | {'Component':<5} | {'Fortran (Real)':<15} | {'Python (Real)':<15} | {'Rel Error':<12}")
    print("-" * 80)

    # 定义相对误差函数 (避免除以0)
    def rel_err(v1, v2):
        return np.abs(v1 - v2) / (np.abs(v2) + 1e-20)

    for i in range(len(fq)):
        # 对比 Hz 实部
        err_hz_r = rel_err(hz_py[i].real, hz_fort[i].real)
        print(f"{fq[i]:<10.0f} | Hz_Re | {hz_fort[i].real:<15.10e} | {hz_py[i].real:<15.10e} | {err_hz_r:.2e}")
        
        # 对比 Hz 虚部
        err_hz_i = rel_err(hz_py[i].imag, hz_fort[i].imag)
        print(f"{'':<10} | Hz_Im | {hz_fort[i].imag:<15.5e} | {hz_py[i].imag:<15.5e} | {err_hz_i:.2e}")

    print("-" * 80)
    
    # ================================
    # 5. 自动断言 (阈值判定)
    # ================================
    # 通常双精度浮点数误差在 1e-12 到 1e-15 之间是正常的
    # 考虑到算法中的累加误差，1e-10 以下通常都认为是一致的
    
    match_hz = np.allclose(hz_py, hz_fort, rtol=1e-10, atol=1e-15)
    match_hp = np.allclose(hp_py, hp_fort, rtol=1e-10, atol=1e-15)

    if match_hz and match_hp:
        print("\n✅ 验证成功：Python 与 Fortran 结果高度一致！")
    else:
        print("\n❌ 验证失败：结果存在显著差异，请检查参数或公式实现。")

if __name__ == "__main__":
    verify_results()

# %% [markdown]
# 以发射线圈中心为原点，$z$ 轴向下为正，二次场的计算公式为：
# $$H_{zs} = \frac{Ia}{2} \int_{0}^{\infty} R_{\text{TE}}(\lambda) e^{\lambda(z-h)} \lambda J_1(\lambda a) J_0(\lambda \rho) d\lambda $$
# 其中，$I$ 为发射电流强度，$a$ 为发射线圈半径， $R_{\text{TE}}(\lambda)$ 为反射系数，与地下电阻率分布相关，$\lambda$ 为积分变量，$J_1, J_0$ 为 第一类一阶和零阶贝塞尔函数，$h$ 为接收线圈相对发射线圈的高度，$\rho$ 为接收线圈圆心相对发射线圈圆心的径向距离
# 
# $$V(\omega) = \underbrace{\left( \frac{1}{-j\omega} \right)}_{\text{波形因子}} \cdot \mu_0 \cdot \underbrace{\left[ \frac{Ia}{2} \int_{0}^{\infty} \dots d\lambda \right]}_{H_{zs} \text{ 理论值}} \cdot \underbrace{(n_t \cdot n_r \cdot \pi r_r^2)}_{\text{装置增益}}$$

# %% [markdown]
# 

# %% [markdown]
# $$H_{zs} = \frac{Ia}{2} \int_{0}^{\infty} R_{\text{TE}}(\lambda) e^{\lambda(z-h)} \lambda J_1(\lambda a) J_0(\lambda \rho) d\lambda $$

# %% [markdown]
# 符号约定：
# 发射线圈半径：$R_T$
# 发射线圈匝数：$N_T$
# 
# 补偿线圈半径：$R_B$
# 补偿线圈匝数：$N_B$
# 
# 接收线圈半径：$R_R$
# 接收线圈匝数：$N_R$
# 接收线圈偏心距：$d$
# 

# %% [markdown]
# 

# %%



