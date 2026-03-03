import numpy as np
from scipy.integrate import odeint
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from scipy.io import savemat

# ---------------------------------------------------------
# 1. 数据生成 (Van der Pol 模型)
# ---------------------------------------------------------
def vdp(z, t, mu):
    """Van der Pol 振荡器，状态变量 z = [x, y]"""
    x, y = z
    dxdt = y
    dydt = -x + mu * (1 - x**2) * y
    return [dxdt, dydt]

# 参数设置
params = (0.1,)
z0 = [1.0, 0.8]          # 初始条件
dt = 0.001
t_total = np.arange(0, 60, dt)
sol = odeint(vdp, z0, t_total, args=params)

split = int(len(sol) * 0.2)
train_data = sol[:split]
test_data = sol[split:]

t_train = t_total[:split]
t_test = t_total[split:]

# ---------------------------------------------------------
# 2. 构造特征
# ---------------------------------------------------------
def get_ngrc_features_with_names(data_window, k, s, dim_names=['x', 'y']):
    dim = data_window.shape[1]
    names = ["常数项"]
    vals = [1.0]

    # 1. 线性项
    for i in range(k):
        idx = -1 - (i * s)
        time_label = f"(t-{i}dt)"
        for d in range(dim):
            vals.append(data_window[idx, d])
            names.append(f"{dim_names[d]}{time_label}")

    # 2. 二阶非线性项
    linear_vals = vals[1:]
    linear_names = names[1:]
    vals_quad = []
    names_quad = []
    for i in range(len(linear_vals)):
        for j in range(i, len(linear_vals)):
            vals_quad.append(linear_vals[i] * linear_vals[j])
            names_quad.append(f"{linear_names[i]}×{linear_names[j]}")

    return np.array(vals + vals_quad), names + names_quad

# ---------------------------------------------------------
# 3. 准备训练
# ---------------------------------------------------------
k, s = 4, 1
X_list = []
Y_list = []
feature_names = []

for i in range((k - 1) * s, len(train_data) - 1):
    window = train_data[i - (k - 1) * s: i + 1]
    feats, names = get_ngrc_features_with_names(window, k, s)

    delta = train_data[i + 1] - train_data[i]
    Y_list.append(delta)

    X_list.append(feats)
    if i == (k - 1) * s:
        feature_names = names

X_train = np.array(X_list)
Y_train = np.array(Y_list)

model = Ridge(alpha=1e-6, fit_intercept=False)
model.fit(X_train, Y_train)

# ---------------------------------------------------------
# 4. 预测部分
# ---------------------------------------------------------
# 测试集闭环预测
history = list(train_data[-(k - 1) * s - 1:])
predictions = []
for i in range(len(test_data)):
    current_window = np.array(history[-(k - 1) * s - 1:])
    feat, _ = get_ngrc_features_with_names(current_window, k, s)
    delta_pred = model.predict(feat.reshape(1, -1))[0]
    pred = history[-1] + delta_pred
    predictions.append(pred)
    history.append(pred)
predictions = np.array(predictions)

# 训练集预测
train_predictions = []
train_history = list(train_data[:(k - 1) * s + 1])
for i in range((k - 1) * s, len(train_data) - 1):
    current_window = np.array(train_history[-(k - 1) * s - 1:])
    feat, _ = get_ngrc_features_with_names(current_window, k, s)
    delta_pred = model.predict(feat.reshape(1, -1))[0]
    pred = train_history[-1] + delta_pred
    train_predictions.append(pred)
    train_history.append(pred)
train_predictions = np.array(train_predictions)

# 对齐时间轴
t_train_plot = t_train[(k-1)*s : -1]

# ---------------------------------------------------------
# 绘图设置
# ---------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
split_time = t_total[split]

# =========================================================
# 图 1: 真实相图 (仅真实数据)
# =========================================================
plt.figure(figsize=(8, 8))
plt.plot(sol[:, 0], sol[:, 1], 'k-', label='真实轨迹', linewidth=1.5)
plt.scatter(z0[0], z0[1], c='red', s=100, marker='*', label='初始点', zorder=5)
plt.title("Van der Pol 振荡器真实相图", fontsize=14)
plt.xlabel('$x$', fontsize=12)
plt.ylabel('$y$', fontsize=12)
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()

# =========================================================
# 图 2: 真实相图 vs 预测相图 (对比)
# =========================================================
plt.figure(figsize=(8, 8))
plt.plot(sol[:, 0], sol[:, 1], 'k-', label='真实轨迹 (完整)', alpha=0.2, linewidth=1)
plt.plot(train_data[:, 0], train_data[:, 1], 'b-', label='真实轨迹 (训练)', linewidth=2)
plt.plot(train_predictions[:, 0], train_predictions[:, 1], 'r--', label='NGRC 预测 (训练)', linewidth=1.5)
plt.plot(test_data[:, 0], test_data[:, 1], 'b-', label='真实轨迹 (测试)', linewidth=2)
plt.plot(predictions[:, 0], predictions[:, 1], 'g--', label='NGRC 预测 (测试)', linewidth=1.5)
plt.scatter(z0[0], z0[1], c='black', s=100, marker='*', label='初始点', zorder=5)
plt.scatter(test_data[0, 0], test_data[0, 1], c='purple', s=100, marker='s', label='预测起点', zorder=5)
plt.title("相图对比：真实 vs NGRC 预测", fontsize=14)
plt.xlabel('$x$', fontsize=12)
plt.ylabel('$y$', fontsize=12)
plt.legend(loc='best', fontsize=9)
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()

# =========================================================
# 图 3: 双变量时间序列 (单图合并版)
# =========================================================
plt.figure(figsize=(14, 6))

# --- 绘制变量 x (黑色系) ---
# 真实值
plt.plot(t_train, train_data[:, 0], 'k-', label='真实 $x$', linewidth=1.5)
plt.plot(t_test, test_data[:, 0], 'k-', linewidth=1.5)
# 预测值
plt.plot(t_train_plot, train_predictions[:, 0], 'r:', label='NGRC 预测 $x$ (训练)', alpha=0.9, linewidth=1.5)
plt.plot(t_test, predictions[:, 0], 'r--', label='NGRC 预测 $x$ (测试)', linewidth=1.5)

# --- 绘制变量 y (蓝色系) ---
# 真实值
plt.plot(t_train, train_data[:, 1], 'b-', label='真实 $y$', linewidth=1.5)
plt.plot(t_test, test_data[:, 1], 'b-', linewidth=1.5)
# 预测值
plt.plot(t_train_plot, train_predictions[:, 1], 'g:', label='NGRC 预测 $y$ (训练)', alpha=0.9, linewidth=1.5)
plt.plot(t_test, predictions[:, 1], 'g--', label='NGRC 预测 $y$ (测试)', linewidth=1.5)

# 分界线
plt.axvline(x=split_time, color='purple', linestyle='-.', linewidth=2, label='训练/测试分界')

plt.xlabel('Time $t$', fontsize=12)
plt.ylabel('State $x, y$', fontsize=12)
plt.title('状态变量时间序列对比 ($x$ 和 $y$ 同图)', fontsize=14)
plt.legend(loc='upper right', ncol=3, fontsize=9) # 图例分成3列防止遮挡
plt.grid(True, alpha=0.3)
plt.xlim(t_total[0], t_total[-1])

plt.tight_layout()

# ---------------------------------------------------------
# 5. 保存数据为 MAT 文件 (新增部分)
# ---------------------------------------------------------
# 输出误差指标
rmse = np.sqrt(np.mean((test_data - predictions) ** 2))
print(f"测试集均方根误差 (RMSE): {rmse:.6f}")

# 准备要保存的数据字典
# 注意：MATLAB 不支持空格和特殊字符作为变量名，故使用下划线命名
data_to_save = {
    't_total': t_total,
    'sol': sol,                # 完整真实轨迹
    't_train': t_train,
    'train_data': train_data,  # 训练集真实值
    't_test': t_test,
    'test_data': test_data,    # 测试集真实值
    't_train_plot': t_train_plot,
    'train_pred': train_predictions, # 训练集预测值
    'test_pred': predictions,        # 测试集预测值
    'z0': z0,                  # 初始条件
    'split_time': split_time,  # 训练/测试分界时间
    'mu': params[0],           # 模型参数
    'rmse_test': rmse          # 测试误差
}

# 保存文件
savemat('vdp_ngrc_results.mat', data_to_save)
print("数据已保存至 vdp_ngrc_results.mat")

plt.show()