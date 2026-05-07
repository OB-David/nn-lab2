from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        # Y = XW +b
        # 保存输入以供反向传播使用
        self.input = X
        # 计算线性变换，使用 params 中的权重以确保使用已更新的参数
        return X.dot(self.params['W']) + self.params['b']

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        # ∂L/∂W = X^T * grad, ∂L/∂b = sum(grad, axis=0), ∂L/∂X = grad * W^T
        self.grads['W'] = self.input.T.dot(grad)
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)
        return grad.dot(self.params['W'].T)
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}


class conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

        self.W = initialize_method(size=(out_channels, in_channels, kernel_size, kernel_size))
        self.b = initialize_method(size=(1, out_channels, 1, 1))
        self.params = {'W': self.W, 'b': self.b}
        self.grads = {'W': None, 'b': None}

        self.X = None
        self.X_pad = None          # 保存填充后的输入（用于反向传播）
        self.out_h = None
        self.out_w = None

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.X = X
        N, C, H, W = X.shape
        k = self.kernel_size
        s = self.stride
        p = self.padding

        # 输出尺寸
        H_out = (H + 2*p - k) // s + 1
        W_out = (W + 2*p - k) // s + 1
        self.out_h, self.out_w = H_out, W_out

        # 填充
        if p > 0:
            X_pad = np.pad(X, ((0,0), (0,0), (p,p), (p,p)), mode='constant')
        else:
            X_pad = X
        self.X_pad = X_pad   # 保存用于反向传播

        # 使用 as_strided 构建窗口视图（零拷贝）
        # 所需步幅: (N_stride, C_stride, H_stride, W_stride)
        # 窗口形状: (N, C, H_out, W_out, k, k)
        # 内存布局: 最后一个维度是窗口内的行和列
        n, c, h, w = X_pad.strides
        window_shape = (N, C, H_out, W_out, k, k)
        window_strides = (n, c, s * h, s * w, h, w)
        windows = np.lib.stride_tricks.as_strided(X_pad, shape=window_shape, strides=window_strides)
        # windows 形状: (N, C, H_out, W_out, k, k)

        # 转换为 (N*H_out*W_out, C*k*k) 的矩阵
        # 先合并 H_out,W_out -> 一维，再合并 C,k,k -> 一维
        X_col = windows.reshape(N, C, H_out*W_out, k*k).transpose(0, 2, 1, 3).reshape(N*H_out*W_out, -1)
        self.X_col = X_col

        # 卷积核展平: (out_channels, C*k*k)
        W_col = self.W.reshape(self.out_channels, -1)

        # 矩阵乘法
        out_col = W_col @ X_col.T   # (out_channels, N*H_out*W_out)
        # 重排为 (N, H_out, W_out, out_channels) 再加偏置
        out = out_col.T.reshape(N, H_out, W_out, self.out_channels)
        out = out.transpose(0, 3, 1, 2) + self.b
        return out

    def backward(self, grads):
        N, C, H, W = self.X.shape
        k = self.kernel_size
        s = self.stride
        p = self.padding
        H_out, W_out = self.out_h, self.out_w

        # grads: [N, out_channels, H_out, W_out]
        # 转换为 (N*H_out*W_out, out_channels)
        dout = grads.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)

        # 计算 dW, db
        dW_col = dout.T @ self.X_col   # (out_channels, C*k*k)
        dW = dW_col.reshape(self.W.shape)
        db = np.sum(grads, axis=(0,2,3), keepdims=True)
        if self.weight_decay:
            dW += self.weight_decay_lambda * self.W
        self.grads['W'] = dW
        self.grads['b'] = db

        # 计算 dX (col2im)
        # dX_col = dout @ W_col: (N*H_out*W_out, C*k*k)
        W_col = self.W.reshape(self.out_channels, -1)
        dX_col = dout @ W_col

        # 将 dX_col 还原为窗口视图 (N, H_out, W_out, C, k, k)
        dX_windows = dX_col.reshape(N, H_out, W_out, C, k, k).transpose(0, 3, 1, 2, 4, 5)
        # dX_windows 形状: (N, C, H_out, W_out, k, k)

        # 创建填充梯度数组
        dX_pad = np.zeros((N, C, H + 2*p, W + 2*p), dtype=self.X.dtype)

        # 使用 as_strided 获得 dX_pad 的窗口视图，然后累加
        # 注意：dX_pad 的窗口视图应与 forward 中的窗口位置完全一致
        n, c, h_pad, w_pad = dX_pad.strides
        pad_window_strides = (n, c, s * h_pad, s * w_pad, h_pad, w_pad)
        pad_windows = np.lib.stride_tricks.as_strided(
            dX_pad,
            shape=(N, C, H_out, W_out, k, k),
            strides=pad_window_strides,
            writeable=True
        )
        # 将 dX_windows 累加到 pad_windows
        pad_windows += dX_windows

        if p > 0:
            dX = dX_pad[:, :, p:-p, p:-p]
        else:
            dX = dX_pad
        return dX

    def clear_grad(self):
        self.grads = {'W': None, 'b': None}


class ReLU(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.input = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        return np.where(X < 0, 0, X)

    def backward(self, grads):
        return np.where(self.input < 0, 0, grads)


class MultiCrossEntropyLoss(Layer):
    def __init__(self, model=None, max_classes=10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.predicts = None
        self.labels = None
        self.has_softmax = True
        self.optimizable = False

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)

    def forward(self, predicts, labels):
        self.predicts = predicts
        self.labels = labels
        if self.has_softmax:
            probs = softmax(predicts)
        else:
            probs = predicts
        batch_size = predicts.shape[0]
        loss = -np.mean(np.log(probs[np.arange(batch_size), labels] + 1e-12))
        return loss

    def backward(self):
        batch_size = self.predicts.shape[0]
        if self.has_softmax:
            probs = softmax(self.predicts)
            grad = probs.copy()
            grad[np.arange(batch_size), self.labels] -= 1
            grad /= batch_size
        else:
            grad = np.zeros_like(self.predicts)
            grad[np.arange(batch_size), self.labels] = -1.0 / (self.predicts[np.arange(batch_size), self.labels] + 1e-12)
            grad /= batch_size
        if self.model is not None:
            self.model.backward(grad)
        return grad

    def cancel_soft_max(self):
        self.has_softmax = False
        return self


class L2Regularization(Layer):
    pass


def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition