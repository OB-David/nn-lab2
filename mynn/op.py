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

        # 对W做He初始化，约束输入输出的方差相等，避免梯度消失或爆炸（在CNN里必要）
        he_scale = np.sqrt(2.0 / in_dim)
        self.W = initialize_method(size=(in_dim, out_dim)) * he_scale
        # b初始化为0
        self.b = np.zeros((1, out_dim))
        self.params = {'W' : self.W, 'b' : self.b}  # params是优化器修改对象，之后不再使用self.W和self.b
        self.grads = {'W' : None, 'b' : None}   # grads是优化器读取对象，之后不再使用self.grads['W']和self.grads['b']

        self.input = None
        self.weight_decay = weight_decay # 固定不用weiht decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        no padding
        """
        # y = XW + b
        self.input = X
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
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.grads = {'W' : None, 'b' : None}
        self.input = None
        fan_in = in_channels * kernel_size * kernel_size    # 输入维度 = 输入通道数 * 卷积核大小
        he_scale = np.sqrt(2.0 / fan_in)
        self.params = {
            'W' : initialize_method(size=(out_channels, in_channels, kernel_size, kernel_size)) * he_scale, # 对卷积核做He初始化
            'b' : np.zeros((1, out_channels))
        }
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [out, in, k, k]
        """
        self.input = X

        batch_size, in_channels, H, W = X.shape
        W_param = self.params['W']
        b_param = self.params['b']
        out_H = (H - self.kernel_size) // self.stride + 1
        out_W = (W - self.kernel_size) // self.stride + 1
        out = np.zeros(
            (batch_size, self.out_channels, out_H, out_W)
        )
        for i in range(out_H):
            # 遍历H方向的每个位置
            for j in range(out_W):
                # 遍历W方向的每个位置
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size
                x_slice = X[:, :, h_start:h_end, w_start:w_end]
                out[:, :, i, j] = np.sum(
                    x_slice[:, np.newaxis, :, :, :] * W_param,
                    axis=(2,3,4)
                ) + b_param

        return out

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, out_H, out_W]
        """
        X = self.input
        batch_size, in_channels, H, W = X.shape
        _, _, out_H, out_W = grads.shape
        W_param = self.params['W']
        # 初始化梯度
        dW = np.zeros_like(W_param)
        db = np.zeros_like(self.params['b'])
        dX = np.zeros_like(X)
        # ∂L/∂b = sum(grad, axis=(0,2,3))
        db = np.sum(grads, axis=(0,2,3))
        db = db.reshape(1, -1)  # 保持与b_param的形状一致
        # ∂L/∂W = sum(grad * x_slice)， ∂L/∂X = sum(grad * W)
        for i in range(out_H):
            for j in range(out_W):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size
                x_slice = X[:, :, h_start:h_end, w_start:w_end]

                # ∂L/∂W = sum(grad * x_slice)
                dW += np.sum(grads[:, :, i, j][:, :, np.newaxis, np.newaxis, np.newaxis] * x_slice[:, np.newaxis, :, :, :], axis=0)
                # ∂L/∂X = sum(grad * W)
                dX[:, :, h_start:h_end, w_start:w_end] += np.sum(grads[:, :, i, j][:, :, np.newaxis, np.newaxis, np.newaxis] * W_param[np.newaxis, :, :, :, :], axis=1)

        self.grads['W'] = dW
        self.grads['b'] = db

        return dX
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
        

class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output


class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True


    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        # 保存标签
        self.labels = labels
        # 保存预测值
        self.predicts = predicts
        if self.has_softmax:
            self.predicts = softmax(predicts)
        batch_size = self.predicts.shape[0]
        # 计算交叉熵损失
        eps = 1e-12  # 防止log(0)导致数值不稳定
        loss = -np.sum(
            np.log(
                self.predicts[np.arange(batch_size), labels] + eps
            )
        )/batch_size
        return loss
    
    def backward(self):
        """
        Compute gradients and back propagation
        """
        batch_size = self.predicts.shape[0]
        # softmax + cross entropy gradient
        self.grads = self.predicts.copy()
        # ∂L/∂z = y_pred - I{y_true}
        self.grads[np.arange(batch_size), self.labels] -= 1
        # 首先在交叉熵损失中取梯度平均
        # 反向传播其他地方不再除以batch_size
        self.grads /= batch_size
        self.model.backward(self.grads)

    '''def cancel_soft_max(self):
        self.has_softmax = False
        return self'''
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    def __init__(self, model, lambda_):
        self.model = model
        self.lambda_ = lambda_

    def forward(self):
        reg_loss = 0
        for layer in self.model.layers:
            if layer.optimizable == True:
                reg_loss += 0.5 * self.lambda_ * np.sum(layer.params['W'] ** 2)
        return reg_loss
    
    def backward(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                layer.grads['W'] += self.lambda_ * layer.params['W']
    
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition