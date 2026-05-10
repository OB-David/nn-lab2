# 热力图可视化权重
# draw_tools/heatmap.py
import numpy as np
import matplotlib.pyplot as plt

def plot_weights_heatmap(model, figsize_per_layer=(10, 6), save_dir=None):
    """
    遍历模型的所有可优化层，绘制权重热力图。

    参数:
        model: 模型实例（Model_MLP 或 Model_CNN），包含 self.layers 属性
        figsize_per_layer: 每个层热力图的基本尺寸 (宽, 高)
        save_dir: 若提供目录路径，则将图片保存到该目录下，否则显示
    """
    for idx, layer in enumerate(model.layers):
        # 只处理可优化的层（有 optimizable 属性且为 True，并且包含 'W' 权重）
        if not getattr(layer, 'optimizable', False):
            continue
        if not hasattr(layer, 'params') or 'W' not in layer.params:
            continue

        weights = layer.params['W']   # numpy 数组
        layer_name = f"{type(layer).__name__}_{idx}"

        # 根据权重维度选择绘图方式
        if weights.ndim == 2:
            # 全连接层：直接绘制矩阵热力图
            _plot_2d_heatmap(weights, layer_name, figsize_per_layer, save_dir)
        elif weights.ndim == 4:
            # 卷积层：将每个卷积核展平为一行，构成 (out_channels, in_channels*kH*kW) 矩阵
            _plot_conv_weights_heatmap(weights, layer_name, figsize_per_layer, save_dir)
        else:
            print(f"Warning: Layer {layer_name} has unsupported weight shape {weights.shape}, skip.")
            continue


def _plot_2d_heatmap(weights, layer_name, figsize, save_dir):
    """绘制 2D 权重矩阵热力图"""
    plt.figure(figsize=figsize)
    plt.imshow(weights, cmap='viridis', aspect='auto', interpolation='none')
    plt.colorbar(label='Weight Value')
    plt.xlabel(f'Output dimension (out_features={weights.shape[1]})')
    plt.ylabel(f'Input dimension (in_features={weights.shape[0]})')
    plt.title(f'Weight Heatmap - {layer_name} (shape {weights.shape})')
    plt.tight_layout()
    _save_or_show(save_dir, f"{layer_name}_heatmap.png")


def _plot_conv_weights_heatmap(weights, layer_name, figsize, save_dir):
    """
    卷积层权重形状：(out_channels, in_channels, kernel_h, kernel_w)
    将每个卷积核展平为特征向量，绘制热力图矩阵 (out_channels, in_channels * kernel_h * kernel_w)
    """
    out_c, in_c, kh, kw = weights.shape
    flattened = weights.reshape(out_c, -1)   # (out_c, in_c*kh*kw)
    plt.figure(figsize=figsize)
    plt.imshow(flattened, cmap='viridis', aspect='auto', interpolation='none')
    plt.colorbar(label='Weight Value')
    plt.xlabel(f'Flattened kernel features (total = {in_c*kh*kw})')
    plt.ylabel(f'Output channel (out_channels = {out_c})')
    plt.title(f'Conv Layer Weight Matrix - {layer_name} (each row = one kernel flattened)')
    plt.tight_layout()
    _save_or_show(save_dir, f"{layer_name}_conv_heatmap.png")


def _save_or_show(save_dir, filename):
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()