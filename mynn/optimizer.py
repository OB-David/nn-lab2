from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key] = layer.params[key] - self.init_lr * layer.grads[key]


class MomentGD(Optimizer):
    '''
    带动量的SGD优化器
    '''
    def __init__(self, init_lr, model, mu):
        super().__init__(init_lr, model)
        self.mu = mu
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    # 用字典来存储每一层的动量
                    if not hasattr(layer, 'velocity'):
                        layer.velocity = {} 
                    if key not in layer.velocity:
                        layer.velocity[key] = np.zeros_like(layer.params[key])
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    # v = mu * v - lr * dw
                    layer.velocity[key] = self.mu * layer.velocity[key] - self.init_lr * layer.grads[key]
                    # w = w + v
                    layer.params[key] += layer.velocity[key]