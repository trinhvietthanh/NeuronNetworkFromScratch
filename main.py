from NN import layers
import numpy as np
from NN.sequential import Sequential
from NN.loss import BinaryCrossEntropy
from NN.callback import CSVLogger

x = np.array([[2, 2], [1, -2], [-2, 2], [-1, 1]])
y = np.array([[0, 1, 0, 1]])
np.random.seed(20)# reproduce value
model = Sequential()
model.add(layers.Linear(5, activation="relu"))
model.add(layers.Linear(3, activation="relu"))
model.compile(lr=0.01, loss=BinaryCrossEntropy())
model.fit(x, y, epochs=10, callbacks=CSVLogger(file_path="/logs/logs.csv", overwrite=True))