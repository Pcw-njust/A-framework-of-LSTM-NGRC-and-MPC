import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from scipy.io import savemat


# ---------------------------------------------------------
# 1. 系统定义
# ---------------------------------------------------------
def custom_system(z, t, r=0.3, alpha=0.03, d=0.3, K=50, u_x=0, u_y=0):
    x1, x2 = z
    dxdt = r * x1 * (1 - x1 / K) - alpha * x1 * x2 + u_x
    dydt = 0.8 * alpha * x1 * x2 - d * x2 + u_y
    return [dxdt, dydt]


# ---------------------------------------------------------
# 2. 生成新的时间序列数据
# ---------------------------------------------------------
z0 = np.array([9.38, 20.31])
dt = 0.01

# 时间阶段划分
t_hold = np.arange(0, 5, dt)  # 0-5秒
t_evolve = np.arange(5, 6, dt)  # 5-6秒
t_ctrl = np.arange(6, 10, dt)  # 6-10秒
t_total = np.concatenate([t_hold, t_evolve, t_ctrl])

# 1. 0-5秒：保持目标值不动
sol_hold = np.tile(z0, (len(t_hold), 1))

# 2. 5-6秒：无控制，自由演化 (作为NGRC的训练数据)
sol_evolve = odeint(custom_system, sol_hold[-1], t_evolve, args=(0.3, 0.03, 0.3, 50, 0, 0))

# 用于训练的数据是 5-6秒 的演化数据
sol_train = sol_evolve

# ---------------------------------------------------------
# 3. NGRC 特征构造与训练 (修改点1：预测导数)
# ---------------------------------------------------------
k, s = 3, 1

feature_names_base = []
for tau in range(k):
    feature_names_base.append(f'x(t-{tau}dt)')
    feature_names_base.append(f'y(t-{tau}dt)')
poly = PolynomialFeatures(degree=2, include_bias=True)


def prepare_input_data(time_series, k, s, dt):
    X_linear = []
    Y_target = []
    for i in range(k - 1, len(time_series) - 1):
        window = time_series[i - (k - 1): i + 1][::-1]
        X_linear.append(window.flatten())

        current_state = time_series[i]
        next_state = time_series[i + 1]
        delta_state = next_state - current_state

        # [修改] 这里除以 dt，训练目标变为导数 dX/dt
        Y_target.append(delta_state / dt)

    return np.array(X_linear), np.array(Y_target)


# 构造训练特征与标签 (注意传入 dt)
X_linear_train, Y_train = prepare_input_data(sol_train, k, s, dt)
X_poly_train = poly.fit_transform(X_linear_train)

# 训练岭回归模型
model = Ridge(alpha=1e-6, fit_intercept=False)
model.fit(X_poly_train, Y_train)

# ---------------------------------------------------------
# 4. MPC 控制设置 (修改点2：积分器替换)
# ---------------------------------------------------------
target = np.array([9.38, 20.31])
horizon = 15
u_amplitude_min = -15
u_amplitude_max = 15
delta_u_max = 1.0
delta_u_penalty = 0.1

# 初始化：控制开始于 t=6s
initial_ctrl_state = sol_evolve[-1]
current_window = sol_evolve[-k:].copy()[::-1]

state_history = []
u_x_history = []
u_y_history = []
u_prev = np.array([0.0, 0.0])


def mpc_cost(u_seq, window_in, target, model, poly_transformer, h, delta_u_penalty, u_prev, dt):
    cost = 0
    win = window_in.copy()
    u_seq = u_seq.reshape(h, 2)
    last_u = u_prev.copy()

    for i in range(h):
        linear_input = win.flatten().reshape(1, -1)
        feat = poly_transformer.transform(linear_input)

        current_s = win[0]

        # [修改] 1. 模型预测导数 dX/dt
        dXdt_pred = model.predict(feat)[0]

        # 获取当前步控制量
        u_x_step, u_y_step = u_seq[i]

        # [修改] 2. 定义带控制的导数函数 (用于积分)
        # 假设在这个微小的时间步 dt 内，控制量 u 是恒定的
        def dynamics_approx(z, t):
            # NGRC 预测的无控导数 + 控制量
            return dXdt_pred + np.array([u_x_step, u_y_step])

        # [修改] 3. 使用 odeint 积分得到下一时刻状态
        # 积分区间：0 到 dt
        next_s = odeint(dynamics_approx, current_s, [0, dt])[-1]

        # 计算代价
        cost += np.sum((next_s - target) ** 2) * 30
        cost += (u_x_step ** 2 + u_y_step ** 2) * 0.01
        delta_u_x = abs(u_x_step - last_u[0])
        delta_u_y = abs(u_y_step - last_u[1])
        cost += (delta_u_x ** 2 + delta_u_y ** 2) * delta_u_penalty

        # 更新循环变量
        last_u = np.array([u_x_step, u_y_step])
        win = np.vstack([next_s, win[:-1]])
    return cost


def rate_constraint(u_seq, h, u_prev, delta_u_max):
    u_seq = u_seq.reshape(h, 2)
    constraints = []
    last_u = u_prev.copy()
    for i in range(h):
        delta_x = u_seq[i, 0] - last_u[0]
        delta_y = u_seq[i, 1] - last_u[1]
        constraints.append(delta_u_max ** 2 - delta_x ** 2)
        constraints.append(delta_u_max ** 2 - delta_y ** 2)
        last_u = u_seq[i].copy()
    return np.array(constraints)


print("开始MPC控制 (t=6s 至 t=10s)...")
for t in t_ctrl:
    n_ctrl_vars = 2 * horizon
    u_bounds = [(u_amplitude_min, u_amplitude_max)] * n_ctrl_vars
    constraint_dict = {
        'type': 'ineq',
        'fun': rate_constraint,
        'args': (horizon, u_prev, delta_u_max)
    }

    # 注意：cost function 现在需要传入 dt 作为参数
    res = minimize(
        mpc_cost,
        np.zeros(n_ctrl_vars),
        args=(current_window, target, model, poly, horizon, delta_u_penalty, u_prev, dt),
        method='SLSQP',
        bounds=u_bounds,
        constraints=[constraint_dict],
        options={'maxiter': 100, 'ftol': 1e-4}
    )

    u_opt = res.x.reshape(horizon, 2)[0]
    u_x_opt, u_y_opt = u_opt

    # [修改] 3. 真实系统演化
    # 控制量 u 现在直接传入，不再除以 dt，因为它现在是物理意义上的力/导数项
    actual_now = current_window[0]
    next_actual = odeint(
        custom_system,
        actual_now,
        [0, dt],
        args=(0.3, 0.03, 0.3, 50, u_x_opt, u_y_opt)  # 这里 u 直接传
    )[-1]

    state_history.append(actual_now)
    u_x_history.append(u_x_opt)
    u_y_history.append(u_y_opt)

    current_window = np.vstack([next_actual, current_window[:-1]])
    u_prev = np.array([u_x_opt, u_y_opt])

state_history = np.array(state_history)

# ---------------------------------------------------------
# 5. 可视化 & 数据保存
# ---------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 拼接完整数据以便绘图
full_t = np.concatenate([t_hold, t_evolve, t_ctrl])
full_x = np.concatenate([sol_hold[:, 0], sol_evolve[:, 0], state_history[:, 0]])
full_y = np.concatenate([sol_hold[:, 1], sol_evolve[:, 1], state_history[:, 1]])
u_x_plot = np.concatenate([np.zeros(len(t_hold) + len(t_evolve)), u_x_history])
u_y_plot = np.concatenate([np.zeros(len(t_hold) + len(t_evolve)), u_y_history])

# 保存数据到 MATLAB
data_to_save = {
    't': full_t,
    'x': full_x,
    'y': full_y,
    'ux': u_x_plot,
    'uy': u_y_plot,
    'target': target,
    'u_lim': [u_amplitude_min, u_amplitude_max],
    't_markers': [0, 5, 6, 10]
}

filename = f'ngrc_mpc_data_odeint_{horizon}.mat'
savemat(filename, data_to_save)
print(f"数据已保存至 {filename}")

# 绘图
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

# 子图1：状态
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(full_t, full_x, 'r-', linewidth=2, label='x')
ax1.plot(full_t, full_y, 'b-', linewidth=2, label='y')
ax1.axhline(target[0], color='red', linestyle='--', alpha=0.5, linewidth=1)
ax1.axhline(target[1], color='blue', linestyle='--', alpha=0.5, linewidth=1)
ax1.axvspan(0, 5, color='gray', alpha=0.1, label='保持阶段')
ax1.axvspan(5, 6, color='yellow', alpha=0.1, label='演化/训练阶段')
ax1.axvspan(6, 10, color='green', alpha=0.1, label='MPC控制阶段')
ax1.set_title("系统状态演化与控制 (使用 odeint 积分 NGRC 导数)", fontsize=14)
ax1.set_ylabel("种群数量")
ax1.set_xlabel("时间 (s)")
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

# 子图2：u_x
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(full_t, u_x_plot, 'orange', linewidth=2)
ax2.axhline(0, color='k', linestyle=':', linewidth=1)
ax2.axvspan(6, 10, color='green', alpha=0.1)
ax2.set_title(r"控制输入 $u_x$", fontsize=12)
ax2.set_ylim(u_amplitude_min * 1.2, u_amplitude_max * 1.2)
ax2.grid(True, alpha=0.3)

# 子图3：u_y
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(full_t, u_y_plot, 'purple', linewidth=2)
ax3.axhline(0, color='k', linestyle=':', linewidth=1)
ax3.axvspan(6, 10, color='green', alpha=0.1)
ax3.set_title(r"控制输入 $u_y$", fontsize=12)
ax3.set_ylim(u_amplitude_min * 1.2, u_amplitude_max * 1.2)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()