import sys, inspect
import numpy as np
from NN.activation import functions as fn

activation_fn = {
    'linear': fn.Linear(), 
    'relu': fn.ReLu(), 
    'tanh': fn.Tanh(), 
    'sigmoid': fn.Tanh(), 
    'softmax': fn.Softmax()
    }

class Linear:
    def __init__(self, units, activation=None, use_bias=True):
        self.units = int(units)
        # self.activations = fn.get(activation)
        activation = activation.lower()
        if activation in activation_fn:
            self.activations = activation_fn[activation]
        else:
            raise ValueError(f'Not support this actionvation function %s'%(activation))
        if self.units < 0:
            raise ValueError(f'Invalid value for units %s'%(self.units))
        self.use_bias = use_bias
        self.w = None
        self.b = None       

    def __call__(self, inputs):
        self.w =  np.random.rand(inputs.shape[-1], self.units)
        # self.w = self.add_weight(shape=(input_shape[-1], self.units))
        if self.use_bias:
            self.b =  np.random.rand(self.units,)

        outputs = inputs @ self.w
        if self.use_bias:
            outputs += self.b
        if self.activations is not None:
            outputs = self.activations(outputs)
        self._output = outputs
        return outputs 
    
    def update(self, lr: float):
        self.w = self.w - lr.self._dw
        self.b = self.b - lr.self._db

    @property
    def grad_weights(self):
        return self._dw

    @grad_weights.setter
    def grad_weights(self, gradients: np.ndarray):
        self._db = gradients
    
    @property
    def grad_bias(self):
        return self.db

    @grad_bias.setter
    def grad_bias(self, gradients: np.ndarray):
        self._db = gradients


if __name__ == "__main__":
    model = Linear(units=5, activation='relu') 
    model.build([1,7])
    print(model([1,3.2, 1, 2 ,4,5,6]))
