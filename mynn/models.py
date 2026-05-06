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
        

class Model_CNN(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, image_size=28, num_classes=10, lambda_list=None):
        # 架构配置
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.image_size = image_size
        self.num_classes = num_classes

        # 计算两层卷积后的特征图尺寸
        conv_h = image_size - kernel_size + 1
        conv_h = conv_h - kernel_size + 1
        conv_w = image_size - kernel_size + 1
        conv_w = conv_w - kernel_size + 1

        # 双层卷积（ 1-> 4 -> 16) +全连接结构，卷积层无padding，步长为1
        self.conv1 = conv2D(in_channels= 1, out_channels= 4, kernel_size=kernel_size, stride=1, padding=0)
        self.relu1 = ReLU()
        self.conv2 = conv2D(in_channels= 4, out_channels= 16, kernel_size=kernel_size, stride=1, padding=0)
        self.relu2 = ReLU()
        # 全连接层： 16*conv_h*conv_w -> num_classes
        self.fc = Linear(16 * conv_h * conv_w, num_classes)
        
        self.layers = [self.conv1, self.relu1, self.conv2, self.relu2, self.fc]
        self.feature_shape3 = None

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        # Accept both flattened MNIST inputs [N, 784] and image tensors [N, C, H, W].
        if X.ndim == 2:
            # 将展平的输入重塑为图像张量，统一按照[N, C, H, W]格式处理
            X = X.reshape(X.shape[0], self.in_channels, self.image_size, self.image_size)
        elif X.ndim == 4:
            # 本身就是图像张量，无需重塑
            pass

        outputs = self.conv1(X)
        outputs = self.relu1(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu2(outputs)

        # Cache feature-map shape for backward reshape.
        self.feature_shape3 = outputs.shape
        outputs = outputs.reshape(outputs.shape[0], -1)
        outputs = self.fc(outputs)
        return outputs

    def backward(self, loss_grad):
        # Reverse order of forward: fc -> unflatten -> relu -> conv.
        grads = self.fc.backward(loss_grad)
        grads = grads.reshape(self.feature_shape3)
        grads = self.relu2.backward(grads)
        grads = self.conv2.backward(grads)
        grads = self.relu1.backward(grads)
        grads = self.conv1.backward(grads)

        return grads
    
    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)

        # 重建架构配置
        config = param_list[0]
        self.__init__(
            in_channels=config['in_channels'],
            out_channels=config['out_channels'],
            kernel_size=config['kernel_size'],
            image_size=config['image_size'],
            num_classes=config['num_classes']
        )

        self.conv1.W = param_list[1]['W']
        self.conv1.b = param_list[1]['b']
        self.conv1.params['W'] = self.conv1.W
        self.conv1.params['b'] = self.conv1.b
        self.conv1.weight_decay = param_list[1]['weight_decay']
        self.conv1.weight_decay_lambda = param_list[1]['lambda']

        self.conv2.W = param_list[2]['W']
        self.conv2.b = param_list[2]['b']
        self.conv2.params['W'] = self.conv2.W
        self.conv2.params['b'] = self.conv2.b
        self.conv2.weight_decay = param_list[2]['weight_decay']
        self.conv2.weight_decay_lambda = param_list[2]['lambda']

        self.fc.W = param_list[3]['W']
        self.fc.b = param_list[3]['b']
        self.fc.params['W'] = self.fc.W
        self.fc.params['b'] = self.fc.b
        self.fc.weight_decay = param_list[3]['weight_decay']
        self.fc.weight_decay_lambda = param_list[3]['lambda']

    def save_model(self, save_path):
        # Save architecture config + trainable params.
        param_list = [
            {
                'in_channels': self.in_channels,
                'out_channels': self.out_channels,
                'kernel_size': self.kernel_size,
                'image_size': self.image_size,
                'num_classes': self.num_classes,
            },
            {
                'W': self.conv1.params['W'],
                'b': self.conv1.params['b'],
                'weight_decay': self.conv1.weight_decay,
                'lambda': self.conv1.weight_decay_lambda,
            },
            {
                'W': self.conv2.params['W'],
                'b': self.conv2.params['b'],
                'weight_decay': self.conv2.weight_decay,
                'lambda': self.conv2.weight_decay_lambda,
            },
            {
                'W': self.fc.params['W'],
                'b': self.fc.params['b'],
                'weight_decay': self.fc.weight_decay,
                'lambda': self.fc.weight_decay_lambda,
            },
        ]

        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)