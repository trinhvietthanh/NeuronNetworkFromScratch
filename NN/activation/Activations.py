from abc import ABC, abstractmethod
import numpy as np

class Activation(ABC):
    @abstractmethod
    def __call__(self, x: np.ndarray):
        raise NotImplementedError()
    
    @abstractmethod
    def gradient(self, output):
        raise NotImplementedError()