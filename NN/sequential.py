import inspect
import numpy as np
from NN.loss import Loss

class Sequential:
    """Sequential groups a linear stack of layers into model
    """
    def __init__(self, layers=None):
        if layers:
            if not isinstance(layers, (list, tuple)):
                layers = [layers]
            for layer in layers:
                self.add(layer)
        self.list_layer = []
        self.layers = {}
        self._inputs = None
        self._output = None

    def __call__(self, inputs):
        outputs = inputs
        for layer in self.layers:
            
            outputs = layer(inputs)

            inputs = outputs
        self._output = outputs
        return outputs

    def add(self, layer):
        self.list_layer.append(layer)
        self.layers[layer] = layer
    
    def compile(self, lr: float, loss: Loss):
        self.lr = lr
        self._loss = loss

    def backward_step(self, labels:np.ndarray):
        da = self._loss.gradient(self._output, labels)
        self._num_layers = len(self.list_layer)
        # for i in reversed(range(0, self._num_layers)):
        #     layer = self.layers[self.list_layer[i]]
        #     print(layer.units)
        for index in reversed(range(0, self._num_layers)):
            layer = self.layers[self.list_layer[index]]
            activation = layer.activations
            if index == 0:
                prev_layer_output = self._inputs
            else:
                prev_layer = self.layers[self.list_layer[index-1]]
                prev_activation = prev_layer.activations
                prev_layer_output = prev_activation(prev_layer._output)
            
            dz = np.multiply(da, activation.gradient(layer._output))
            layer.grad_weights = (dz @ np.transpose(prev_layer_output)) / self._num_examples
            layer.grad_weights = layer.grad_weights + (self._regularization_factor / self._num_examples) * layer.weights
            layer.grad_bias = np.mean(dz, axis=1, keepdims=True)
            da = np.dot(np.transpose(layer.grad_weights), dz)
        
    def fit(self, x=None, y=None, epochs=1, verbose=False, callbacks=()):
        
        for epoch in range(1, epochs + 1):
            self._inputs = x
            _ = self(self._inputs)
            loss = self._loss(self._output, y)
            self.backward_step(y)
            self.update()

            for callback in callbacks:
                loss_scalar = float(np.squeeze(loss))
                callback.on_epoch_end(epoch, loss_scalar)
            if verbose:
                print(f"Epoch: {epoch:03d}, Loss {loss:0.4f}")
        
    def predict(self, x):
        outputs = self(x)
        return (outputs > 0.5).astype("uint8")
    
    def evaluate(self, x, y):
        _ = self(x)
        return self._loss(self._output, y)

    def update(self):
        for layer in self.layers:
            layer.update(self.lr)
