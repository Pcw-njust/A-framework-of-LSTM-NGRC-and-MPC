import numpy as np
from scipy.integrate import odeint
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from scipy.io import savemat

# ---------------------------------------------------------
# 1. 数据生成
# ---------------------------------------------------------
def lotka_volterra(z, t, a, b, c, d):
    x, y = z
    return [a * x * (1 - x / d) - b * x * y, 0.8 * b * x * y - c * y]

params = (0.3, 0.03, 0.3, 50.0)
z0 = [7.5, 12.5]
dt = 0.001
t_total = np.arange(0, 100, dt)
sol = odeint(lotka_volterra, z0, t_total, args=params)

split = int(len(sol) * 0.02)
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
k, s = 2, 1
X_list = []
Y_list = []
feature_names = []

for i in range((k - 1) * s, len(train_data) - 1):
    window = train_data[i - (k - 1) * s: i + 1]
    feats, names = get_ngrc_features_with_names(window, k, s)

    delta = train_data[i + 1] - train_data[i]
    Y_list.append(delta)

    X_list.append(feats)
    if i == (k - 1) * s: feature_names = names

X_train = np.array(X_list)
Y_train = np.array(Y_list)

model = Ridge(alpha=1e-6, fit_intercept=False)
model.fit(X_train, Y_train)

# ---------------------------------------------------------
# 4. 特征重要性
# ---------------------------------------------------------
weights = model.coef_
importance = np.sum(np.abs(weights), axis=0)
indices = np.argsort(importance)[::-1]

print(f"{'特征重要性 (对状态变化的贡献, 前10个):'}")
for i in range(min(10, len(indices))):
    idx = indices[i]
    name = feature_names[idx]
    imp_val = importance[idx]
    w_x = weights[0, idx]
    w_y = weights[1, idx]
    print(f"{i + 1:>4}. 特征 {idx:>3} ({name:<30})")
    print(f"       重要性: {imp_val:.6f}, 权重(dx): {w_x:+.6f}, 权重(dy): {w_y:+.6f}")

# ---------------------------------------------------------
# 5. 测试集闭环预测
# ---------------------------------------------------------
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

# 生成训练集的预测值
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

# ---------------------------------------------------------
# 6. 保存数据为 MATLAB 格式
# ---------------------------------------------------------
# 注意：train_predictions 因为有初始窗口，长度比 t_train 短
# 我们需要对齐时间轴
t_train_plot = t_train[(k-1)*s : -1]  # 对应 train_predictions 的时间

# 构建字典
data_dict = {
    't_train': t_train,           # 完整训练集时间
    't_train_plot': t_train_plot, # 训练集预测对应的时间
    't_test': t_test,             # 测试集时间
    'train_data': train_data,     # 真实训练数据 (x, y)
    'test_data': test_data,       # 真实测试数据 (x, y)
    'train_pred': train_predictions, # NGRC 训练集预测
    'test_pred': predictions,     # NGRC 测试集预测
    'split_time': t_total[split]  # 分界时间点
}

# 保存文件
savemat('ngrc_data.mat', data_dict)
print("数据已保存至 ngrc_data.mat")

# ---------------------------------------------------------
# 7. 绘图
# ---------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(12, 6))

# 训练集
plt.plot(t_train_plot, train_predictions[:, 0], 'r:', label='NGRC x (训练集)', alpha=0.8)
plt.plot(t_train, train_data[:, 0], 'k-', label='True x (Grass)', alpha=0.8)
plt.plot(t_train_plot, train_predictions[:, 1], 'g:', label='NGRC y (训练集)', alpha=0.8)
plt.plot(t_train, train_data[:, 1], 'b-', label='True y (Animal)', alpha=0.8)

# 测试集
plt.plot(t_test, predictions[:, 0], 'r--', label='NGRC x (测试集)', linewidth=2)
plt.plot(t_test, test_data[:, 0], 'k-', alpha=0.8)
plt.plot(t_test, predictions[:, 1], 'g--', label='NGRC y (测试集)', linewidth=2)
plt.plot(t_test, test_data[:, 1], 'b-', alpha=0.8)

# 分界线
split_time = t_total[split]
plt.axvline(x=split_time, color='purple', linestyle='-', linewidth=2, label='训练/测试分界', alpha=0.8)

plt.title("NGRC Prediction (Delta Formulation: $X_{i+1} - X_i = f(X_i)$)", fontsize=12)
plt.xlabel("Time", fontsize=10)
plt.ylabel("Population", fontsize=10)
plt.legend(loc='best', fontsize=9)
plt.grid(True, alpha=0.3)
plt.xlim(t_total[0], t_total[-1])

plt.show()

# 输出误差指标
rmse = np.sqrt(np.mean((test_data - predictions) ** 2))
print(f"测试集均方根误差 (RMSE): {rmse:.6f}")