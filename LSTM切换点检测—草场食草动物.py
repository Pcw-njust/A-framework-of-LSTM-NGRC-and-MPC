import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.integrate import solve_ivp
from scipy.io import savemat
import warnings

warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)


# ===================== 1. 数据生成 =====================
def generate_eco_data():
    r1 = 0.5
    K = 50.0
    alpha1 = 0.02
    beta = 0.8
    d1 = 0.15
    r2 = 0.3
    alpha2 = 0.03
    d2 = 0.3
    tau, t_end = 100.0, 200.0

    x1_eq1 = d1 / (beta * alpha1)
    x2_eq1 = r1 * (1 - x1_eq1 / K) / alpha1
    print(f"阶段1理论平衡点：x1={x1_eq1:.2f}, x2={x2_eq1:.2f}")

    x1_eq2 = d2 / (beta * alpha2)
    x2_eq2 = r2 * (1 - x1_eq2 / K) / alpha2
    print(f"阶段2理论平衡点：x1={x1_eq2:.2f}, x2={x2_eq2:.2f}")
    assert x1_eq2 > 0 and x2_eq2 > 0, "阶段2平衡点必须为正数！"

    def system_dynamics(t, x):
        x1, x2 = x
        if t < tau:
            r, alpha, d = r1, alpha1, d1
        else:
            r, alpha, d = r2, alpha2, d2

        dx1dt = r * x1 * (1 - x1 / K) - alpha * x1 * x2
        dx2dt = beta * alpha * x1 * x2 - d * x2
        return [dx1dt, dx2dt]

    x0 = [x1_eq1 * 1.3, x2_eq1 * 1.2]
    print(x0)
    t_eval = np.linspace(0, t_end, 2000)

    solution = solve_ivp(
        fun=system_dynamics,
        t_span=(0, t_end),
        y0=x0,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-6
    )

    data = solution.y.T
    noise = np.random.normal(0, 0.5, data.shape)
    return t_eval, data + noise, tau, (x1_eq1, x2_eq1), (x1_eq2, x2_eq2)


def generate_phase1_train_data(phase1_params, scale_factor, t_train_end=80.0, n_points=800):
    r1, K, alpha1, beta, d1 = phase1_params
    x1_eq1 = d1 / (beta * alpha1)
    x2_eq1 = r1 * (1 - x1_eq1 / K) / alpha1

    def phase1_dynamics(t, x):
        x1, x2 = x
        dx1dt = r1 * x1 * (1 - x1 / K) - alpha1 * x1 * x2
        dx2dt = beta * alpha1 * x1 * x2 - d1 * x2
        return [dx1dt, dx2dt]

    x0 = [x1_eq1 * scale_factor, x2_eq1 * scale_factor]
    t_eval = np.linspace(0, t_train_end, n_points)
    solution = solve_ivp(
        fun=phase1_dynamics,
        t_span=(0, t_train_end),
        y0=x0,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-6
    )
    phase1_data = solution.y.T

    noise = np.random.normal(0, 0.5, phase1_data.shape)
    phase1_data_noise = phase1_data + noise

    return t_eval, phase1_data_noise


def generate_multi_phase1_data(phase1_params, start_scale=1.1, end_scale=2.0, step=0.1, t_train_end=80.0, n_points=800):
    all_phase1_data = []
    all_t = []
    scale_factors = np.arange(start_scale, end_scale + step, step)

    for idx, scale in enumerate(scale_factors):
        t, data = generate_phase1_train_data(phase1_params, scale, t_train_end, n_points)
        all_phase1_data.append(data)
        all_t.append(t)
        print(f"生成第{idx + 1}组阶段1数据（缩放倍数={scale:.1f}）：{len(data)}个样本")

    merged_data = np.concatenate(all_phase1_data, axis=0)
    merged_t = np.concatenate(all_t, axis=0)
    print(f"\n合并后总训练数据量：{len(merged_data)}个样本")
    return merged_t, merged_data


# 1. 生成完整时间序列L1
t_time, eco_data_L1, true_tau, (x1_eq1, x2_eq1), (x1_eq2, x2_eq2) = generate_eco_data()

# 2. 生成多组阶段1训练数据
phase1_params = (0.5, 50.0, 0.02, 0.8, 0.15)
phase1_t, phase1_train_data = generate_multi_phase1_data(
    phase1_params,
    start_scale=1.1,
    end_scale=2.0,
    step=0.1,
    t_train_end=80.0,
    n_points=800
)

# ===================== 2. 数据预处理 =====================
scaler = StandardScaler()
scaler.fit(phase1_train_data)
data_norm_L1 = scaler.transform(eco_data_L1)
phase1_data_norm = scaler.transform(phase1_train_data)

WINDOW_SIZE = 40


def create_sequences(data, window_size):
    xs, ys = [], []
    for i in range(len(data) - window_size):
        x = data[i:i + window_size]
        y = data[i + window_size]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


X_train, y_train = create_sequences(phase1_data_norm, WINDOW_SIZE)
X_all_L1, y_all_L1 = create_sequences(data_norm_L1, WINDOW_SIZE)

batch_size = 64
train_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
    batch_size=batch_size, shuffle=True
)


# ===================== 3. LSTM预测模型 =====================
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

# ===================== 4. 模型训练 =====================
print("\n开始在多组阶段1数据上训练LSTM...")
epochs = 300
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

# ===================== 5. 变点检测 =====================
model.eval()
all_losses = []
inputs = torch.FloatTensor(X_all_L1).to(device)
targets = torch.FloatTensor(y_all_L1).to(device)

with torch.no_grad():
    predictions = model(inputs)
    errors = torch.mean((predictions - targets) ** 2, dim=1).cpu().numpy()

smooth_window = 1
errors_smooth = np.convolve(errors, np.ones(smooth_window) / smooth_window, mode='same')

X_phase1_test, y_phase1_test = create_sequences(phase1_data_norm, WINDOW_SIZE)
phase1_inputs = torch.FloatTensor(X_phase1_test).to(device)
phase1_targets = torch.FloatTensor(y_phase1_test).to(device)
with torch.no_grad():
    phase1_preds = model(phase1_inputs)
    phase1_errors = torch.mean((phase1_preds - phase1_targets) ** 2, dim=1).cpu().numpy()
phase1_errors_smooth = np.convolve(phase1_errors, np.ones(smooth_window) / smooth_window, mode='same')

error_max = max(phase1_errors)
errors_standardized = errors_smooth

sigma = 1
threshold = sigma * error_max

VERIFY_WINDOW = 40
candidate_indices = np.where(errors_standardized > threshold)[0]
detected_idx = None

for idx in candidate_indices:
    if idx + VERIFY_WINDOW > len(errors_standardized):
        continue
    window_errors_mean = np.mean(errors_standardized[idx:idx + VERIFY_WINDOW])
    if window_errors_mean >= threshold:
        detected_idx = idx
        break

time_axis = t_time[WINDOW_SIZE:]

# ===================== 【新增】数据保存为MATLAB格式 =====================
print("\n正在整理并保存数据...")

# 1. 整理子图1数据 (10组阶段1训练数据)
n_points_per_group = 800
# 初始化矩阵 (800行 x 10列)
phase1_t_mat = np.zeros((n_points_per_group, 10))
phase1_x1_mat = np.zeros((n_points_per_group, 10))
phase1_x2_mat = np.zeros((n_points_per_group, 10))

for i in range(10):
    start_idx = i * n_points_per_group
    end_idx = (i + 1) * n_points_per_group
    phase1_t_mat[:, i] = phase1_t[start_idx:end_idx]
    phase1_x1_mat[:, i] = phase1_train_data[start_idx:end_idx, 0]
    phase1_x2_mat[:, i] = phase1_train_data[start_idx:end_idx, 1]

# 2. 整理子图3数据 (变点检测)
if detected_idx is not None:
    detected_time = time_axis[detected_idx]
    verify_win_start = time_axis[detected_idx]
    verify_win_end = time_axis[detected_idx + VERIFY_WINDOW]
else:
    detected_time = np.nan
    verify_win_start = np.nan
    verify_win_end = np.nan

# 3. 构建保存字典
data_dict = {
    # --- 子图1相关 ---
    'phase1_t': phase1_t_mat,  # 10组数据的时间轴 (800x10)
    'phase1_x1': phase1_x1_mat,  # 10组食草动物数据 (800x10)
    'phase1_x2': phase1_x2_mat,  # 10组食肉动物数据 (800x10)
    'eq1_x1': x1_eq1,  # 阶段1 x1平衡点 (标量)
    'eq1_x2': x2_eq1,  # 阶段1 x2平衡点 (标量)

    # --- 子图2相关 ---
    'L1_t': t_time,  # 原始序列时间轴 (2000x1)
    'L1_x1': eco_data_L1[:, 0],  # 原始序列x1 (2000x1)
    'L1_x2': eco_data_L1[:, 1],  # 原始序列x2 (2000x1)
    'eq2_x1': x1_eq2,  # 阶段2 x1平衡点 (标量)
    'eq2_x2': x2_eq2,  # 阶段2 x2平衡点 (标量)
    'true_tau': true_tau,  # 真实变点时刻 (标量)

    # --- 子图3相关 ---
    'sub3_t': time_axis,  # 检测图时间轴 (1960x1)
    'sub3_error': errors_standardized,  # 预测误差 (1960x1)
    'threshold': threshold,  # 检测阈值 (标量)
    'detected_tau': detected_time,  # 检测到的变点 (标量/NaN)
    'verify_win_start': verify_win_start,  # 验证窗口开始 (标量/NaN)
    'verify_win_end': verify_win_end  # 验证窗口结束 (标量/NaN)
}

# 4. 保存文件
savemat('eco_data_for_matlab.mat', data_dict)
print("数据已成功保存至：eco_data_for_matlab.mat")

# ===================== 6. 绘图 (保持原样) =====================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(12, 12))

# 子图1: 绘制所有10组阶段1训练数据
plt.subplot(3, 1, 1)
n_points_per_group = 800
colors_x1 = plt.cm.Greens(np.linspace(0.3, 0.9, 10))
colors_x2 = plt.cm.Blues(np.linspace(0.3, 0.9, 10))
for i in range(10):
    start_idx = i * n_points_per_group
    end_idx = (i + 1) * n_points_per_group
    group_t = phase1_t[start_idx:end_idx]
    group_data = phase1_train_data[start_idx:end_idx]
    plt.plot(group_t, group_data[:, 0], color=colors_x1[i], alpha=0.6, linewidth=1)
    plt.plot(group_t, group_data[:, 1], color=colors_x2[i], alpha=0.6, linewidth=1)

plt.axhline(y=x1_eq1, color='darkgreen', linestyle=':', linewidth=2.5, label=f'阶段1 x1平衡点 ({x1_eq1:.2f})')
plt.axhline(y=x2_eq1, color='darkblue', linestyle=':', linewidth=2.5, label=f'阶段1 x2平衡点 ({x2_eq1:.2f})')
plt.plot([], [], color='green', alpha=0.6, linewidth=1, label='草场 (x1) - 10组初始值')
plt.plot([], [], color='blue', alpha=0.6, linewidth=1, label='食草动物 (x2) - 10组初始值')
plt.title('阶段1训练数据（10组不同初始值，1.1~2.0倍平衡点）')
plt.xlabel('时间')
plt.ylabel('种群数量')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

# 子图2: 原始L1数据+双阶段平衡点
plt.subplot(3, 1, 2)
plt.plot(t_time, eco_data_L1[:, 0], label='食草动物 (x1)', alpha=0.7, color='green')
plt.plot(t_time, eco_data_L1[:, 1], label='食肉动物 (x2)', alpha=0.7, color='blue')
plt.axhline(y=x1_eq1, color='green', linestyle=':', linewidth=1.5, label=f'阶段1 x1平衡点 ({x1_eq1:.2f})')
plt.axhline(y=x2_eq1, color='blue', linestyle=':', linewidth=1.5, label=f'阶段1 x2平衡点 ({x2_eq1:.2f})')
plt.axhline(y=x1_eq2, color='green', linestyle='-.', linewidth=1.5, label=f'阶段2 x1平衡点 ({x1_eq2:.2f})')
plt.axhline(y=x2_eq2, color='blue', linestyle='-.', linewidth=1.5, label=f'阶段2 x2平衡点 ({x2_eq2:.2f})')
plt.axvline(x=true_tau, color='k', linestyle='--', linewidth=2, label='变点 τ')
plt.text(50, eco_data_L1[:, 0].max() * 0.9, '阶段1', ha='center', color='black')
plt.text(150, eco_data_L1[:, 0].max() * 0.9, '阶段2', ha='center', color='darkgreen')
plt.title('原时间序列L1（含变点+双阶段平衡点）')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

# 子图3: 预测误差与变点检测
plt.subplot(3, 1, 3)
plt.plot(time_axis, errors_standardized, color='purple', linewidth=1.5, label='预测误差')
plt.axhline(y=threshold, color='orange', linestyle='--', label=f'阈值 ({threshold:.4f})')
plt.axvline(x=true_tau, color='k', linestyle='--', linewidth=2, label='真实变点')

if detected_idx is not None:
    detected_time = time_axis[detected_idx]
    plt.axvline(x=detected_time, color='red', linewidth=2, label=f'检测到变点 t={detected_time:.2f}')
    print(f"\n检测到的变点时刻: {detected_time:.2f} (真实: {true_tau})")
    print(f"检测误差: {abs(detected_time - true_tau):.2f}")
    plt.axvspan(time_axis[detected_idx], time_axis[detected_idx + VERIFY_WINDOW],
                alpha=0.2, color='red', label=f'验证窗口({VERIFY_WINDOW}个点)')
else:
    print("\n未检测到满足条件的变点")
    plt.text(0.5, 0.5, '未检测到满足条件的变点', ha='center', va='center', transform=plt.gca().transAxes)

plt.title(f'变点检测（验证窗口={VERIFY_WINDOW}个点）')
plt.ylabel('预测误差')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()