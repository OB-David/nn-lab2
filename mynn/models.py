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
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
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
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
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



class Model_CNN(Layer):
    """
    CNN:
    Conv(1->4) -> ReLU ->
    Conv(4->16) -> ReLU ->
    FC(16*24*24 -> 10)
    """
    def __init__(self):
        super().__init__()
        self.layers = [
            conv2D(in_channels=1, out_channels=4, kernel_size=3),
            ReLU(),
            conv2D(in_channels=4, out_channels=16, kernel_size=3),
            ReLU(),
            Linear(16 * 24 * 24, 10)
        ]

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        
        # 处理[batch, 784]的输入，reshape成[batch, 1, 28, 28]   
        if X.ndim == 2:
            X = X.reshape(X.shape[0], 1, 28, 28)
        outputs = X
        # Conv1
        outputs = self.layers[0](outputs)
        # ReLU1
        outputs = self.layers[1](outputs)
        # Conv2
        outputs = self.layers[2](outputs)
        # ReLU2
        outputs = self.layers[3](outputs)
        # flatten
        self.conv_output_shape = outputs.shape
        outputs = outputs.reshape(
            outputs.shape[0],
            -1
        )
        # FC (直接到10分类)
        outputs = self.layers[4](outputs)

        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        # FC (直接到10分类)
        grads = self.layers[4].backward(grads)
        # reshape back to conv feature map
        grads = grads.reshape(self.conv_output_shape)
        # ReLU2
        grads = self.layers[3].backward(grads)
        # Conv2
        grads = self.layers[2].backward(grads)
        # ReLU1
        grads = self.layers[1].backward(grads)
        # Conv1
        grads = self.layers[0].backward(grads)

        return grads

    def save_model(self, save_path):

        param_list = []
        for layer in self.layers:
            if hasattr(layer, 'optimizable') and layer.optimizable:
                param_list.append({
                    'W': layer.params['W'],
                    'b': layer.params['b']
                })
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)

    def load_model(self, param_path):

        with open(param_path, 'rb') as f:
            param_list = pickle.load(f)
        param_idx = 0
        for layer in self.layers:
            if hasattr(layer, 'optimizable') and layer.optimizable:
                layer.params['W'] = param_list[param_idx]['W']
                layer.params['b'] = param_list[param_idx]['b']
                layer.W = layer.params['W']
                layer.b = layer.params['b']
                param_idx += 1