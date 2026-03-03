import numpy as np
from scipy.integrate import odeint
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.io import savemat


# ---------------------------------------------------------
# 1. 数据生成 (Van der Pol 模型)
# ---------------------------------------------------------
def vdp(z, t, mu):
    """Van der Pol 振荡器，状态变量 z = [x, y]"""
    x, y = z
    dxdt = y
    dydt = -x + mu * (1 - x ** 2) * y
    return [dxdt, dydt]


# 参数设置
params = (0.1,)
z0 = [1.0, 0.8]  # 初始条件
dt = 0.001
t_total = np.arange(0, 60, dt)
sol = odeint(vdp, z0, t_total, args=params)

split = int(len(sol) * 0.2)
train_data = sol[:split]
test_data = sol[split:]

t_train = t_total[:split]
t_test = t_total[split:]


# ---------------------------------------------------------
# 2. SINDY 核心模块 (仅含常数项+线性项+二次项，无三次项)
# ---------------------------------------------------------
def get_sindy_features(z):
    """构造SINDy基函数库：常数项 + 线性项 + 二次项（已移除所有三次项）"""
    x, y = z
    return np.array([
        1,  # 常数项
        x, y,  # 线性项
        x ** 2, x * y, y ** 2  # 二次项（无三次项）
    ])


def sindy_stlsq(X, Y, threshold=0.05, max_iter=20):
    """
    顺序阈值最小二乘 (STLSQ) 算法
    用于稀疏回归，识别控制方程的稀疏系数
    """
    n_features = X.shape[1]
    n_outputs = Y.shape[1]
    Xi = np.zeros((n_features, n_outputs))  # 系数矩阵 [特征数, 状态数]

    for i in range(n_outputs):
        y = Y[:, i]
        # 初始：普通最小二乘
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X, y)
        xi = reg.coef_.copy()

        # 迭代：阈值化 -> 重新拟合
        for _ in range(max_iter):
            # 保留绝对值大于阈值的系数
            idx = np.abs(xi) > threshold
            if np.sum(idx) == 0:
                break  # 无有效项，退出

            # 仅用非零项重新拟合
            reg.fit(X[:, idx], y)
            xi_new = np.zeros(n_features)
            xi_new[idx] = reg.coef_

            # 收敛判断
            if np.allclose(xi, xi_new, atol=1e-8):
                break
            xi = xi_new

        Xi[:, i] = xi
    return Xi


# ---------------------------------------------------------
# 3. 训练 SINDY 模型
# ---------------------------------------------------------
# 构造训练数据矩阵 (特征矩阵 X 和 导数矩阵 Y)
X_train = np.array([get_sindy_features(z) for z in train_data])
# 计算真实导数 (也可用有限差分 np.gradient(train_data, dt, axis=0))
Y_train = np.array([vdp(z, 0, params[0]) for z in train_data])

# 运行 STLSQ 稀疏回归
threshold = 0.02  # 稀疏阈值，可根据拟合效果调整
Xi = sindy_stlsq(X_train, Y_train, threshold=threshold)

# 打印识别出的方程（同步更新特征名，无三次项）
feature_names = ['1', 'x', 'y', 'x²', 'xy', 'y²']
print("=" * 50)
print("识别出的 SINDY 控制方程（仅二次项，无三次项）：")
for i, state in enumerate(['x', 'y']):
    eq_str = f"d{state}/dt = "
    terms = []
    for j in range(len(feature_names)):
        coef = Xi[j, i]
        if np.abs(coef) > 1e-6:
            terms.append(f"{coef:.4f}*{feature_names[j]}")
    eq_str += " + ".join(terms)
    print(eq_str)
print("=" * 50)


# ---------------------------------------------------------
# 4. 预测部分 (使用识别出的 SINDY 模型积分)
# ---------------------------------------------------------
def sindy_model(z, t, Xi):
    """SINDY 识别出的动力学模型"""
    features = get_sindy_features(z)
    dzdt = features @ Xi  # 矩阵乘法：特征 × 系数
    return dzdt


# 训练集预测 (从初始点开始积分)
train_pred = odeint(sindy_model, z0, t_train, args=(Xi,))

# 测试集预测 (从训练集最后一个点开始闭环积分)
test_pred = odeint(sindy_model, train_data[-1], t_test, args=(Xi,))

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
# 图 2: 真实相图 vs SINDY 预测相图 (对比)
# =========================================================
plt.figure(figsize=(8, 8))
plt.plot(sol[:, 0], sol[:, 1], 'k-', label='真实轨迹 (完整)', alpha=0.2, linewidth=1)
plt.plot(train_data[:, 0], train_data[:, 1], 'b-', label='真实轨迹 (训练)', linewidth=2)
plt.plot(train_pred[:, 0], train_pred[:, 1], 'r--', label='SINDY 预测 (训练)', linewidth=1.5)
plt.plot(test_data[:, 0], test_data[:, 1], 'b-', label='真实轨迹 (测试)', linewidth=2)
plt.plot(test_pred[:, 0], test_pred[:, 1], 'g--', label='SINDY 预测 (测试)', linewidth=1.5)
plt.scatter(z0[0], z0[1], c='black', s=100, marker='*', label='初始点', zorder=5)
plt.scatter(test_data[0, 0], test_data[0, 1], c='purple', s=100, marker='s', label='预测起点', zorder=5)
plt.title("相图对比：真实 vs SINDY 预测（无三次项）", fontsize=14)
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
plt.plot(t_train, train_data[:, 0], 'k-', label='真实 $x$', linewidth=1.5)
plt.plot(t_test, test_data[:, 0], 'k-', linewidth=1.5)
plt.plot(t_train, train_pred[:, 0], 'r:', label='SINDY 预测 $x$ (训练)', alpha=0.9, linewidth=1.5)
plt.plot(t_test, test_pred[:, 0], 'r--', label='SINDY 预测 $x$ (测试)', linewidth=1.5)

# --- 绘制变量 y (蓝色系) ---
plt.plot(t_train, train_data[:, 1], 'b-', label='真实 $y$', linewidth=1.5)
plt.plot(t_test, test_data[:, 1], 'b-', linewidth=1.5)
plt.plot(t_train, train_pred[:, 1], 'g:', label='SINDY 预测 $y$ (训练)', alpha=0.9, linewidth=1.5)
plt.plot(t_test, test_pred[:, 1], 'g--', label='SINDY 预测 $y$ (测试)', linewidth=1.5)

# 分界线
plt.axvline(x=split_time, color='purple', linestyle='-.', linewidth=2, label='训练/测试分界')

plt.xlabel('Time $t$', fontsize=12)
plt.ylabel('State $x, y$', fontsize=12)
plt.title('状态变量时间序列对比（无三次项）', fontsize=14)
plt.legend(loc='upper right', ncol=3, fontsize=9)
plt.grid(True, alpha=0.3)
plt.xlim(t_total[0], t_total[-1])

plt.tight_layout()

# ---------------------------------------------------------
# 5. 保存数据为 MAT 文件 (已调整格式)
# ---------------------------------------------------------
# 输出误差指标
rmse_train = np.sqrt(np.mean((train_data - train_pred) ** 2))
rmse_test = np.sqrt(np.mean((test_data - test_pred) ** 2))
print(f"训练集均方根误差 (RMSE): {rmse_train:.6f}")
print(f"测试集均方根误差 (RMSE): {rmse_test:.6f}")

# --- 变量名映射以匹配目标格式 ---
t_train_plot = t_train          # 映射训练时间数组
train_predictions = train_pred  # 映射训练集预测值
predictions = test_pred         # 映射测试集预测值
rmse = rmse_test                # 映射测试误差

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
savemat('vdp_sindy_no_cubic.mat', data_to_save)
print("数据已保存至 vdp_sindy_no_cubic.mat")

plt.show()