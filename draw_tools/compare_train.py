import matplotlib.pyplot as plt
import pickle
import os
import numpy as np

def load_data(folder_path):
    """从指定文件夹加载 pickle 数据文件"""
    with open(os.path.join(folder_path, 'train_loss.pickle'), 'rb') as f:
        train_loss = pickle.load(f)
    with open(os.path.join(folder_path, 'train_scores.pickle'), 'rb') as f:
        train_scores = pickle.load(f)
    # 注意：只读取训练数据，不读 dev
    return train_loss, train_scores

def smooth_envelope(data, window=10):
    """
    计算滑动窗口内的平均值、最小值、最大值（绘制中心线和包络）
    data: 一维列表或数组
    window: 窗口大小（奇数）
    返回: (smoothed, min_env, max_env)
    """
    if window % 2 == 0:
        window += 1
    half = window // 2
    data_pad = np.pad(data, (half, half), mode='edge')
    smoothed = []
    min_env = []
    max_env = []
    for i in range(len(data)):
        seg = data_pad[i:i+window]
        smoothed.append(np.mean(seg))
        min_env.append(np.min(seg))
        max_env.append(np.max(seg))
    return np.array(smoothed), np.array(min_env), np.array(max_env)

# ================== 配置 ==================
#cnn_folder = './MLP'                # 标准 MLP 结果文件夹
cnn_folder = './CNN_1epoch'                # 标准 CNN 结果文件夹
#cnn_msgd_folder = './MLP_MSGD'      # 带动量 MLP 结果文件夹
cnn_msgd_folder = './CNN_MSGD_1epoch'      # 带动量 CNN 结果文件夹
window_size = 15                    # 滑动窗口大小（奇数），控制平滑程度

# 加载训练数据（不加载验证集）
cnn_train_loss, cnn_train_acc = load_data(cnn_folder)
msgd_train_loss, msgd_train_acc = load_data(cnn_msgd_folder)

# 生成 x 轴（迭代次数）
iters = np.arange(len(cnn_train_loss))

# 对每条训练曲线进行平滑与包络计算
cnn_train_loss_s, cnn_train_loss_min, cnn_train_loss_max = smooth_envelope(cnn_train_loss, window_size)
cnn_train_acc_s,  cnn_train_acc_min,  cnn_train_acc_max  = smooth_envelope(cnn_train_acc, window_size)

msgd_train_loss_s, msgd_train_loss_min, msgd_train_loss_max = smooth_envelope(msgd_train_loss, window_size)
msgd_train_acc_s,  msgd_train_acc_min,  msgd_train_acc_max  = smooth_envelope(msgd_train_acc, window_size)

# ================== 绘图（完全按照您提供的布局风格，仅训练集） ==================
_, axes = plt.subplots(1, 2)          # 左右两个子图，默认大小
axes = axes.reshape(-1)               # 确保为一维数组
_.set_tight_layout(1)                 # 自动紧凑布局

# ------------------ 左图：训练损失曲线 ------------------
ax_loss = axes[0]
# CNN
ax_loss.plot(iters, cnn_train_loss_s, color='#1f77b4', label='CNN Train loss')
ax_loss.fill_between(iters, cnn_train_loss_min, cnn_train_loss_max, color='#1f77b4', alpha=0.2)
# CNN_MSGD
ax_loss.plot(iters, msgd_train_loss_s, color='#ff7f0e', label='CNN_MSGD Train loss')
ax_loss.fill_between(iters, msgd_train_loss_min, msgd_train_loss_max, color='#ff7f0e', alpha=0.2)

ax_loss.set_ylabel('loss')
ax_loss.set_xlabel('iteration')
ax_loss.set_title('')                # 无标题，与参考代码一致
ax_loss.legend(loc='upper right')
ax_loss.grid(True, linestyle=':', alpha=0.6)

# ------------------ 右图：训练准确率曲线 ------------------
ax_acc = axes[1]
# CNN
ax_acc.plot(iters, cnn_train_acc_s, color='#1f77b4', label='CNN Train accuracy')
ax_acc.fill_between(iters, cnn_train_acc_min, cnn_train_acc_max, color='#1f77b4', alpha=0.2)
# CNN_MSGD
ax_acc.plot(iters, msgd_train_acc_s, color='#ff7f0e', label='CNN_MSGD Train accuracy')
ax_acc.fill_between(iters, msgd_train_acc_min, msgd_train_acc_max, color='#ff7f0e', alpha=0.2)

ax_acc.set_ylabel('score')
ax_acc.set_xlabel('iteration')
ax_acc.set_title('')
ax_acc.legend(loc='lower right')
ax_acc.grid(True, linestyle=':', alpha=0.6)

plt.show()