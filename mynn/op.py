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

        # Initialize weights and biases
        self.W = initialize_method(size=(out_channels, in_channels, kernel_size, kernel_size))
        self.b = initialize_method(size=(out_channels,))

        self.grads = {'W' : None, 'b' : None}
        self.input = None

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
        self.params = {'W': self.W, 'b': self.b}

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [out, in, k, k]
        no padding
        """
        self.input = X

        batch_size, _, H, W = X.shape
        out_H = (H - self.kernel_size) // self.stride + 1
        out_W = (W - self.kernel_size) // self.stride + 1

        output = np.zeros((batch_size, self.out_channels, out_H, out_W), dtype=X.dtype)

        for i in range(out_H):
            for j in range(out_W):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size

                input_slice = X[:, :, h_start:h_end, w_start:w_end]
                for oc in range(self.out_channels):
                    output[:, oc, i, j] = np.sum(input_slice * self.params['W'][oc], axis=(1, 2, 3)) + self.params['b'][oc]

        return output

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        batch_size, _, out_H, out_W = grads.shape

        self.grads['W'] = np.zeros_like(self.W)
        self.grads['b'] = np.zeros_like(self.b)
        grad_input = np.zeros_like(self.input)

        for i in range(out_H):
            for j in range(out_W):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size

                input_slice = self.input[:, :, h_start:h_end, w_start:w_end]
                for oc in range(self.out_channels):
                    grad_val = grads[:, oc, i, j][:, None, None, None]
                    self.grads['W'][oc] += np.sum(input_slice * grad_val, axis=0)
                    self.grads['b'][oc] += np.sum(grads[:, oc, i, j])
                    grad_input[:, :, h_start:h_end, w_start:w_end] += self.params['W'][oc][None, :, :, :] * grad_val

        if self.weight_decay:
            # weight decay disabled: no-op
            pass

        return grad_input
    
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
        # 初始化模型参数
        self.model = model
        self.has_softmax = True
        self.max_classes = max_classes
        self.grads = None

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        if self.has_softmax:
            predicts = softmax(predicts)
        self.predicts = predicts
        self.labels = labels
        # 计算交叉熵损失，加epsilon防止log(0)
        batch_size = predicts.shape[0]
        eps = 1e-10
        loss = -np.sum(np.log(predicts[np.arange(batch_size), labels] + eps)) / batch_size
        return loss
    
    def backward(self):
        # first compute the grads from the loss to the input
        batch_size = self.predicts.shape[0]
        self.grads = self.predicts.copy()
        self.grads[np.arange(batch_size), self.labels] -= 1
        self.grads /= batch_size

        # Pass the upstream gradient into the model's backward()
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition