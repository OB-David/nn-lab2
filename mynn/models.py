from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(size_list[i], size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        for i in range(len(self.size_list) - 1):
            self.layers = []
            for i in range(len(self.size_list) - 1):
                layer = Linear(self.size_list[i], self.size_list[i + 1])
                layer.W = param_list[i + 2]['W']
                layer.b = param_list[i + 2]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[i + 2]['weight_decay']
                layer.weight_decay_lambda = param_list[i+2]['lambda']
                if self.act_func == 'Logistic':
                    raise NotImplemented
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(self.size_list) - 2:
                    self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

import numpy as np
from mynn.op import Layer, conv2D, ReLU, Linear

class Model_CNN(Layer):
    def __init__(self, out_channels=4, kernel_size=3, image_size=28, num_classes=10):
        super().__init__()
        self.out_channels = out_channels
        self.out_channels2 = out_channels * 4
        self.kernel_size = kernel_size
        self.image_size = image_size
        self.num_classes = num_classes

        # 计算两次卷积后的空间尺寸
        h1 = image_size - kernel_size + 1
        h2 = h1 - kernel_size + 1
        self.h2 = h2
        flatten_dim = h2 * h2 * self.out_channels2

        # 定义层
        self.conv1 = conv2D(1, out_channels, kernel_size, stride=1, padding=0)
        self.relu1 = ReLU()
        self.conv2 = conv2D(out_channels, self.out_channels2, kernel_size, stride=1, padding=0)
        self.relu2 = ReLU()
        self.fc = Linear(flatten_dim, num_classes)

        self.layers = [self.conv1, self.relu1, self.conv2, self.relu2, self.fc]

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        batch = X.shape[0]
        # 将 [batch, 784] reshape 为 [batch, 1, 28, 28]
        X = X.reshape(batch, 1, self.image_size, self.image_size)

        out = X
        for layer in self.layers:
            out = layer.forward(out)
            # 在 relu2 之后展平（为全连接做准备）
            if layer is self.relu2:
                out = out.reshape(batch, -1)
        return out

    def backward(self, loss_grad):
        grad = loss_grad
        # 反向遍历各层
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            # 全连接层返回的是 [batch, flatten_dim]，需要 reshape 回四维
            if layer is self.fc:
                grad = grad.reshape(-1, self.out_channels2, self.h2, self.h2)
        return grad

    def load_model(self, file_path):
        with open(file_path, 'rb') as f:
            param_list = pickle.load(f)   # 这里加载的就是保存时的参数列表
        
        idx = 0
        for layer in self.layers:
            if hasattr(layer, 'params') and layer.optimizable:
                # 将参数列表中的对应字典赋值给当前层
                if idx < len(param_list):
                    for name in layer.params:
                        if name in param_list[idx]:
                            layer.params[name] = param_list[idx][name]
                idx += 1

    def save_model(self, save_path):
        param_list = []
        for layer in self.layers:
            if hasattr(layer, 'params') and layer.optimizable:
                param_list.append(layer.params)
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)