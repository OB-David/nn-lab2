import numpy as np
import matplotlib.pyplot as plt
from mynn.op import softmax

def plot_top_loss_misclassified(logits, y_true, images, top_k=12, img_shape=(28, 28),
                                ncols=4, subplot_size=(2, 2),
                                save_path=None, show=True, title_prefix = None):
    # 1. 概率与交叉熵损失
    probs = softmax(logits)
    correct_class_probs = probs[np.arange(len(y_true)), y_true]
    losses = -np.log(correct_class_probs + 1e-12)

    # 2. 预测与错误筛选
    preds = np.argmax(logits, axis=1)
    mis_mask = (preds != y_true)
    mis_indices_all = np.where(mis_mask)[0]

    if len(mis_indices_all) == 0:
        print("没有错误分类样本，无法绘制。")
        return np.array([]), np.array([])

    # 3. 取 loss 最高的 top_k 个
    mis_losses = losses[mis_mask]
    top_k_actual = min(top_k, len(mis_indices_all))
    sorted_loss_idx = np.argsort(mis_losses)[-top_k_actual:][::-1]
    selected_indices = mis_indices_all[sorted_loss_idx]
    selected_losses = mis_losses[sorted_loss_idx]

    # 4. 图像重塑
    imgs_reshaped = images.reshape(-1, *img_shape)

    # 5. 动态布局
    rows = int(np.ceil(top_k_actual / ncols))
    cols = min(ncols, top_k_actual)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * subplot_size[0], rows * subplot_size[1]))
    # 统一 axes 为一维数组
    if top_k_actual == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (idx, loss_val) in enumerate(zip(selected_indices, selected_losses)):
        img = imgs_reshaped[idx]
        ax = axes[i]
        ax.imshow(img, cmap='gray')
        ax.set_title(f"True:{y_true[idx]} | Pred:{preds[idx]}\nLoss:{loss_val:.4f}", fontsize=10)
        ax.axis('off')

    # 隐藏多余子图
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle(f"{title_prefix} (Top-{top_k_actual})", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)

    return selected_indices, selected_losses