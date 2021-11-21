import sys, inspect
import numpy as np
from NN.activation import functions as fn
# from activation import functions as fn
activation_fn = {
    'linear': fn.Linear(), 
    'relu': fn.ReLu(), 
    'tanh': fn.Tanh(), 
    'sigmoid': fn.Sigmoid(), 
    'softmax': fn.Softmax()
    }

class Linear:
    def __init__(self, units, activation=None, use_bias=True):
        self.units = int(units)
        # self.activations = fn.get(activation)
        self.a = activation
        activation = activation.lower()
        if activation in activation_fn:
            # print(activation_fn)
            self.activations = activation_fn[activation]
        else:
            raise ValueError(f'Not support this actionvation function %s'%(activation))
        if self.units < 0:
            raise ValueError(f'Invalid value for units %s'%(self.units))
        self.use_bias = use_bias
        self.w = None
        self.b = None       
    
    def build(self, inputs_shape):
        # self.w =  np.random.rand(inputs_shape[-1], self.units)* np.sqrt(2. / inputs_shape[-1])
        # if self.use_bias:
        #     self.b = np.zeros(self.units,)
        self.w = np.random.randn(self.units, inputs_shape[0]) * np.sqrt(2. / inputs_shape[0])
        self.b = np.zeros((self.units, 1))

    def __call__(self, inputs):
        
        if self.w is None:
            self.build(inputs.shape)
        # 
        # self.w = np.random.randn(self.units, inputs.shape[0]) * np.sqrt(2. / inputs.shape[0])
        # self.w = self.add_weight(shape=(input_shape[-1], self.units))
        
        outputs = np.dot(self.w, inputs)

        if self.use_bias:
            outputs += self.b
        if self.activations is not None:
            outputs = self.activations(outputs)
        self._output = outputs
        # print(self.a, outputs[0])
        return outputs
    
    def update(self, lr: float):
        self.w = self.w - lr*self._dw
        self.b = self.b - lr*self._db

    @property
    def grad_weights(self):
        return self._dw

    @grad_weights.setter
    def grad_weights(self, gradients: np.ndarray):
        self._dw = gradients
    
    @property
    def grad_bias(self):
        return self.db

    @grad_bias.setter
    def grad_bias(self, gradients: np.ndarray):
        self._db = gradients

    @property
    def weights(self):
        return self.w

    @property
    def bias(self):
        return self.b

    @property
    def output(self):
        return self._output

if __name__ == "__main__":
    model = Linear(units=5, activation='relu') 
    y = np.array([[1., 1., 1., 1.], [1., 1., 1., 1.]])
    print(y.shape)
    y = model(y)
    print(y.shape)
    model = Linear(units=2, activation='sigmoid') 
    print(model(y))