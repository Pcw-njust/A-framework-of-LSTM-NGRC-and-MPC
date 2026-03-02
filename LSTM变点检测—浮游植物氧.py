import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import warnings
# 新增：导入mat文件保存库
import scipy.io as sio

warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)

# -------------------------- 湖泊生态系统全局参数 --------------------------
# 正常自净阶段参数（阶段1：t < SWITCH_TIME）
PARAMS_NORMAL = {
    "a1": 0.6, "k1": 5, "b1": 0.1, "c1": 0.1,  # 浮游植物增长参数
    "d1": 0.25, "e1": 0.15, "f1": 0.8  # 溶解氧变化参数
}
# 富营养化阶段参数（阶段2：t ≥ SWITCH_TIME）
PARAMS_EUTROPHIC = {
    "a1": 1.2, "k1": 25, "b1": 0.03, "c1": 1.0,  # 浮游植物疯长参数
    "d1": 0.05, "e1": 0.8, "f1": 0.1  # 溶解氧快速消耗参数
}

# 基础仿真配置（适配湖泊生态系统）
T_MAX = 200  # 总模拟时长 (s)，对应湖泊生态的200s
SWITCH_TIME = 100  # 阶段切换/变点时刻 (s)，100s切换为富营养化
T_STEP = 0.05  # 采样步长 (s)，200/0.05=4000个点，保证平滑
INITIAL_STATE = [4.0, 8.0]  # 系统初始状态：[浮游植物2.0, 溶解氧8.0] (mg/L)

# 噪声配置
ADD_NOISE = True
NOISE_AMP_X = 0.2  # 浮游植物噪声幅值
NOISE_AMP_Y = 0.2  # 溶解氧噪声幅值


# -------------------------- 平衡点计算函数（适配湖泊生态） --------------------------
def calculate_equilibrium(params):
    """计算湖泊生态系统稳态平衡点 (x*, y*)：dx/dt=0, dy/dt=0"""
    a1, k1, b1, c1, d1, e1, f1 = params.values()

    # 定义平衡点方程组
    def equations(vars):
        x, y = vars
        eq1 = a1 * x * (1 - x / k1) - b1 * x * y + c1  # dx/dt=0
        eq2 = d1 * x - e1 * y + f1  # dy/dt=0
        return [eq1, eq2]

    # 数值求解稳态值
    x_eq, y_eq = fsolve(equations, INITIAL_STATE)
    return round(x_eq, 4), round(y_eq, 4)


# 预计算两个阶段的平衡点
x_eq_normal, y_eq_normal = calculate_equilibrium(PARAMS_NORMAL)
x_eq_eutro, y_eq_eutro = calculate_equilibrium(PARAMS_EUTROPHIC)
print("=" * 60)
print(f"正常自净阶段平衡点：浮游植物 x* = {x_eq_normal}, 溶解氧 y* = {y_eq_normal}")
print(f"富营养化阶段平衡点：浮游植物 x* = {x_eq_eutro}, 溶解氧 y* = {y_eq_eutro}")
print("=" * 60)


# -------------------------- 湖泊生态系统微分方程模型 --------------------------
def lake_ecosystem_model(t, state):
    """湖泊生态双变量动力学模型，分阶段切换参数"""
    x, y = state  # x=浮游植物生物量，y=溶解氧浓度
    # 阶段切换逻辑
    params = PARAMS_NORMAL if t < SWITCH_TIME else PARAMS_EUTROPHIC
    a1, k1, b1, c1, d1, e1, f1 = params.values()

    # 核心微分方程
    dx_dt = a1 * x * (1 - x / k1) - b1 * x * y + c1
    dy_dt = d1 * x - e1 * y + f1
    return [dx_dt, dy_dt]


# ===================== 1. 数据生成（替换为湖泊生态数据） =====================
def generate_lake_data():
    """生成带变点的湖泊生态时序数据：正常自净→富营养化"""
    t_eval = np.linspace(0, T_MAX, int(T_MAX / T_STEP))
    # 数值求解微分方程
    solution = solve_ivp(
        fun=lake_ecosystem_model,
        t_span=(0, T_MAX),
        y0=INITIAL_STATE,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-6
    )
    data = solution.y.T  # 形状：[n_samples, 2]，列分别为x(浮游植物), y(溶解氧)

    # 添加独立噪声
    if ADD_NOISE:
        noise_x = np.random.normal(0, NOISE_AMP_X, data[:, 0].shape)
        noise_y = np.random.normal(0, NOISE_AMP_Y, data[:, 1].shape)
        data[:, 0] += noise_x
        data[:, 1] += noise_y

    return t_eval, data, SWITCH_TIME, (x_eq_normal, y_eq_normal), (x_eq_eutro, y_eq_eutro)


# ========== 生成正常阶段（阶段1）训练数据 ==========
def generate_phase1_train_data(phase_params, scale_factor, t_train_end=None, n_points=None):
    """
    生成正常自净阶段训练数据（无变点）
    :param phase_params: 正常阶段参数
    :param scale_factor: 初始值缩放因子
    :param t_train_end: 训练数据时长
    :param n_points: 采样点数
    """
    # 默认时长为变点前80%
    if t_train_end is None:
        t_train_end = SWITCH_TIME * 0.9
    if n_points is None:
        n_points = int(t_train_end / T_STEP)

    a1, k1, b1, c1, d1, e1, f1 = phase_params.values()

    def phase1_dynamics(t, state):
        """纯正常自净阶段动力学方程"""
        x, y = state
        dx_dt = a1 * x * (1 - x / k1) - b1 * x * y + c1
        dy_dt = d1 * x - e1 * y + f1
        return [dx_dt, dy_dt]

    # 基于平衡点缩放生成初始值
    x_eq, y_eq = calculate_equilibrium(phase_params)
    x0 = [x_eq * scale_factor, y_eq * scale_factor]
    t_eval = np.linspace(0, t_train_end, n_points)

    # 求解微分方程
    solution = solve_ivp(
        fun=phase1_dynamics,
        t_span=(0, t_train_end),
        y0=x0,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-6
    )
    phase1_data = solution.y.T

    # 添加标准噪声
    if ADD_NOISE:
        noise_x = np.random.normal(0, NOISE_AMP_X, phase1_data[:, 0].shape)
        noise_y = np.random.normal(0, NOISE_AMP_Y, phase1_data[:, 1].shape)
        phase1_data[:, 0] += noise_x
        phase1_data[:, 1] += noise_y

    return t_eval, phase1_data


# ========== 生成多组不同初始状态的正常阶段训练数据 ==========
def generate_multi_phase1_data(phase_params, start_scale=0.7, end_scale=1.2, step=0.1):
    """生成多组缩放初始值的正常阶段数据，合并为训练集"""
    all_phase1_data = []
    all_t = []
    scale_factors = np.arange(start_scale, end_scale + step, step)

    for idx, scale in enumerate(scale_factors):
        t, data = generate_phase1_train_data(phase_params, scale)
        all_phase1_data.append(data)
        all_t.append(t)
        print(f"生成第{idx + 1}组正常阶段数据（缩放倍数={scale:.1f}）：{len(data)}个样本")

    # 合并数据集
    merged_data = np.concatenate(all_phase1_data, axis=0)
    merged_t = np.concatenate(all_t, axis=0)
    print(f"\n合并后总训练数据量：{len(merged_data)}个样本")
    return merged_t, merged_data


# 1. 生成带变点的完整湖泊生态时序数据
t_time, lake_data_L1, true_tau, (x_eq1, y_eq1), (x_eq2, y_eq2) = generate_lake_data()

# 2. 生成多组正常阶段训练数据（模型仅学习正常自净状态规律）
phase1_params = PARAMS_NORMAL
phase1_t, phase1_train_data = generate_multi_phase1_data(
    phase1_params,
    start_scale=0.7,
    end_scale=1.2,
    step=0.1
)

# ===================== 2. 数据预处理（框架保持不变） =====================
data_norm_L1 = lake_data_L1  # 标准化可根据需求开启，此处保持原始值
phase1_data_norm = phase1_train_data

WINDOW_SIZE = 40


def create_sequences(data, window_size):
    """滑动窗口构建时序样本"""
    xs, ys = [], []
    for i in range(len(data) - window_size):
        x = data[i:i + window_size]
        y = data[i + window_size]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


# 构建训练/测试序列
X_train, y_train = create_sequences(phase1_data_norm, WINDOW_SIZE)
X_all_L1, y_all_L1 = create_sequences(data_norm_L1, WINDOW_SIZE)

batch_size = 64
train_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
    batch_size=batch_size, shuffle=True
)


# ===================== 3. LSTM预测模型（无修改） =====================
class EcoPredictor(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=2):
        super(EcoPredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        prediction = self.fc(last_out)
        return prediction


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EcoPredictor().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.005)
criterion = nn.MSELoss()

# ===================== 4. 模型训练（无修改） =====================
print("\n开始在正常自净阶段数据上训练LSTM模型...")
epochs = 200
model.train()
for epoch in range(epochs):
    epoch_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        preds = model(batch_x)
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.6f}")

# ===================== 5. 变点检测（框架保持不变） =====================
model.eval()
all_losses = []
inputs = torch.FloatTensor(X_all_L1).to(device)
targets = torch.FloatTensor(y_all_L1).to(device)

with torch.no_grad():
    predictions = model(inputs)
    errors = torch.mean((predictions - targets) ** 2, dim=1).cpu().numpy()

# 误差平滑
smooth_window = 1
errors_smooth = np.convolve(errors, np.ones(smooth_window) / smooth_window, mode='same')

# 计算正常阶段预测误差，设定阈值
X_phase1_test, y_phase1_test = create_sequences(phase1_data_norm, WINDOW_SIZE)
phase1_inputs = torch.FloatTensor(X_phase1_test).to(device)
phase1_targets = torch.FloatTensor(y_phase1_test).to(device)
with torch.no_grad():
    phase1_preds = model(phase1_inputs)
    phase1_errors = torch.mean((phase1_preds - phase1_targets) ** 2, dim=1).cpu().numpy()
phase1_errors_smooth = np.convolve(phase1_errors, np.ones(smooth_window) / smooth_window, mode='same')

error_max = max(phase1_errors)
error_std = np.std(phase1_errors)
sigma = 1
threshold = sigma * error_max + 3 * error_std

# 变点验证窗口
VERIFY_WINDOW = 40
candidate_indices = np.where(errors_smooth > threshold)[0]
detected_idx = None

for idx in candidate_indices:
    if idx + VERIFY_WINDOW > len(errors_smooth):
        continue
    window_errors_mean = np.mean(errors_smooth[idx:idx + VERIFY_WINDOW])
    if window_errors_mean >= threshold:
        detected_idx = idx
        break

# 对齐时间轴
time_axis = t_time[WINDOW_SIZE:]

# ===================== 6. 可视化（替换为湖泊生态相关标签） =====================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(12, 12))

# 子图1: 多组正常阶段训练数据
plt.subplot(3, 1, 1)
n_points_per_group = int(SWITCH_TIME * 0.8 / T_STEP)
colors_x = plt.cm.Reds(np.linspace(0.3, 0.9, 10))  # 浮游植物用红色系
colors_y = plt.cm.Blues(np.linspace(0.3, 0.9, 10))  # 溶解氧用蓝色系

for i in range(7):
    start_idx = i * n_points_per_group
    end_idx = (i + 1) * n_points_per_group
    group_t = phase1_t[start_idx:end_idx]
    group_data = phase1_train_data[start_idx:end_idx]
    plt.plot(group_t, group_data[:, 0], color=colors_x[i], alpha=0.6, linewidth=1)
    plt.plot(group_t, group_data[:, 1], color=colors_y[i], alpha=0.6, linewidth=1)

# 绘制正常阶段平衡点
plt.axhline(y=x_eq1, color='darkred', linestyle=':', linewidth=2.5, label=f'正常阶段 浮游植物平衡点 ({x_eq1:.4f})')
plt.axhline(y=y_eq1, color='darkblue', linestyle=':', linewidth=2.5, label=f'正常阶段 溶解氧平衡点 ({y_eq1:.4f})')
plt.plot([], [], color='red', alpha=0.6, linewidth=1, label='浮游植物 - 10组初始值')
plt.plot([], [], color='blue', alpha=0.6, linewidth=1, label='溶解氧 - 10组初始值')
plt.title('正常自净阶段训练数据（0.7~1.3倍平衡点初始值）')
plt.xlabel('时间 (s)')
plt.ylabel('浓度 (mg/L)')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

# 子图2: 完整湖泊生态时序数据+双阶段平衡点+真实变点
plt.subplot(3, 1, 2)
plt.plot(t_time, lake_data_L1[:, 0], label='浮游植物生物量 (mg/L)', alpha=0.7, color='#E74C3C')
plt.plot(t_time, lake_data_L1[:, 1], label='溶解氧浓度 (mg/L)', alpha=0.7, color='#3498DB')
plt.axhline(y=x_eq1, color='#E74C3C', linestyle=':', linewidth=1.5, label=f'正常阶段 浮游植物平衡点 ({x_eq1:.4f})')
plt.axhline(y=y_eq1, color='#3498DB', linestyle=':', linewidth=1.5, label=f'正常阶段 溶解氧平衡点 ({y_eq1:.4f})')
plt.axhline(y=x_eq2, color='#E74C3C', linestyle='-.', linewidth=1.5, label=f'富营养化 浮游植物平衡点 ({x_eq2:.4f})')
plt.axhline(y=y_eq2, color='#3498DB', linestyle='-.', linewidth=1.5, label=f'富营养化 溶解氧平衡点 ({y_eq2:.4f})')
plt.axvline(x=true_tau, color='k', linestyle='--', linewidth=2, label='真实变点 τ=100s')
plt.text(50, lake_data_L1[:, 0].max() * 0.9, '正常自净阶段', ha='center', color='black')
plt.text(150, lake_data_L1[:, 0].max() * 0.9, '富营养化阶段', ha='center', color='darkred')
plt.title('湖泊生态时序数据（正常自净→富营养化变点）')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

# 子图3: 预测误差与变点检测结果
plt.subplot(3, 1, 3)
plt.plot(time_axis, errors_smooth, color='purple', linewidth=1.5, label='预测误差')
plt.axhline(y=threshold, color='orange', linestyle='--', label=f'阈值 ({threshold:.4f})')
plt.axvline(x=true_tau, color='k', linestyle='--', linewidth=2, label='真实变点(100s)')

if detected_idx is not None:
    detected_time = time_axis[detected_idx]
    plt.axvline(x=detected_time, color='red', linewidth=2, label=f'检测变点 t={detected_time:.2f}s')
    print(f"\n检测到的变点时刻: {detected_time:.2f}s | 真实变点: {true_tau}s")
    print(f"检测误差: {abs(detected_time - true_tau):.2f}s")
    plt.axvspan(time_axis[detected_idx], time_axis[detected_idx + VERIFY_WINDOW],
                alpha=0.2, color='red', label=f'验证窗口({VERIFY_WINDOW}点)')
else:
    print("\n未检测到满足条件的变点")
    plt.text(0.5, 0.5, '未检测到满足条件的变点', ha='center', va='center', transform=plt.gca().transAxes)

plt.title(f'变点检测结果（验证窗口={VERIFY_WINDOW}个点）')
plt.ylabel('预测MSE误差')
plt.xlabel('时间 (s)')
plt.legend()
plt.grid(True, alpha=0.3)

# ===================== 新增：数据保存逻辑 =====================
# 整理所有需要保存的绘图数据
save_data = {
    # 子图1：正常自净阶段训练数据
    'phase1_t': phase1_t,  # 训练数据时间轴 (s)
    'phase1_train_data_x': phase1_train_data[:, 0],  # 浮游植物训练数据 (mg/L)
    'phase1_train_data_y': phase1_train_data[:, 1],  # 溶解氧训练数据 (mg/L)
    'x_eq1': x_eq1,  # 正常阶段浮游植物平衡点
    'y_eq1': y_eq1,  # 正常阶段溶解氧平衡点

    # 子图2：完整湖泊生态时序数据
    't_time': t_time,  # 完整数据时间轴 (s)
    'lake_data_x': lake_data_L1[:, 0],  # 浮游植物完整数据 (mg/L)
    'lake_data_y': lake_data_L1[:, 1],  # 溶解氧完整数据 (mg/L)
    'x_eq2': x_eq2,  # 富营养化阶段浮游植物平衡点
    'y_eq2': y_eq2,  # 富营养化阶段溶解氧平衡点
    'true_tau': true_tau,  # 真实变点时间 (100s)

    # 子图3：变点检测误差数据
    'time_axis': time_axis,  # 误差时间轴 (s)
    'errors_smooth': errors_smooth,  # 平滑后的预测MSE误差
    'threshold': threshold,  # 变点检测阈值
    'VERIFY_WINDOW': VERIFY_WINDOW,  # 验证窗口大小
    'detected_time': detected_time if detected_idx is not None else np.nan  # 检测到的变点时间
}

# 保存为MATLAB可读取的.mat文件
save_path = 'lake_ecosystem_plot_data.mat'
sio.savemat(save_path, save_data)
print(f"\n✅ 所有绘图数据已保存至：{save_path}")
print(f"📊 保存的变量清单：")
for key in save_data.keys():
    print(f"   - {key}: {type(save_data[key])}, 形状: {np.shape(save_data[key])}")

plt.tight_layout()
plt.show()