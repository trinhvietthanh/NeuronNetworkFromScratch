from NN.activation import Activations
# import Activations
import numpy as np

class Linear(Activations.Activation):

    def __call__(self, x: np.ndarray):
        return x

    def gradient(self, output: np.ndarray):
        '''
        We are using the linear transfer function, the derivative of which can be calculated as follows:
        derivative = 1.0
        '''
        return np.ones_like(output)

class ReLu(Activations.Activation):
    def __call__(self, x: np.ndarray):
        return np.maximum(x, 0)      

    def gradient(self, output: np.ndarray):
        _result = output.copy()
        _result[output >= 0] = 1
        _result[output < 0] = 0
        return _result
    
class Sigmoid(Activations.Activation):
    def __call__(self, x: np.ndarray):
        return 1.0 / (1.0 + np.exp(-1*x))
    
    def gradient(self, output: np.ndarray):
        return self(output) * (1 - self(output))

class Tanh(Activations.Activation):
    def __call__(self, x: np.ndarray):
        return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

    def gradient(self, output: np.ndarray):
        return 1 - (self(output) * self(output))

class Softmax(Activations.Activation):
    def __call__(self, x: np.ndarray):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def gradient(self, output):
        # Reshape the 1-d softmax to 2-d so that np.dot will do the matrix multiplication
        s = output.reshape(-1,1)
        return np.diagflat(s) - np.dot(s, s.T)
