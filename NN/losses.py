import numpy as np
from abc import abstractmethod, ABC
from math import log2

class Loss(ABC):
    """
    This protocol must be implemented by Loss classes
    """
    @abstractmethod
    def __call__(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class BinaryCrossEntropy(Loss):
    def __call__(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        predictions += np.finfo(np.float32).eps
        loss = np.mean(np.multiply(labels, np.log(predictions)) + np.multiply(1 - labels, np.log(1 - predictions)))
        return -1 * loss

    def gradient(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        predictions += np.finfo(np.float32).eps
        return -1 * (np.divide(labels, predictions) - np.divide(1 - labels, 1 - predictions))


if __name__ == "__main__":
    y = BinaryCrossEntropy()
    # p =np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    p = np.array([1,0,0])
    # q =  np.array([0.8, 0.9, 0.9, 0.6, 0.8, 0.1, 0.4, 0.2, 0.1, 0.3])
    q = np.array(
    [[8.94985434, 6.27792454, 8.85854436],
    [2.11564754, 1.59541812, 1.55138302],
    [2.05987611, 1.49378658, 1.51647058],
    [2.44710899, 1.74698073, 1.69747657]])
    results = list()
    for i in range(len(p)):
        # create the distribution for each event {0, 1}
        expected = [1.0 - p[i], p[i]]
        predicted = [1.0 - q[i], q[i]]
        # calculate cross entropy for the two events
        ce = y(expected, predicted)
        print('>[y=%.1f, yhat=%.1f] ce: %.3f nats' % (p[i], q[i], ce))
        results.append(ce)
    
    # calculate the average cross entropy
    mean_ce = np.mean(results)
    print('Average Cross Entropy: %.3f nats' % mean_ce)