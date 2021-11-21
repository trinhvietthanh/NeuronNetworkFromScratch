from NN import layers
import numpy as np
from NN.sequential import Sequential
from NN.losses import BinaryCrossEntropy
from NN.callback import CSVLogger
import mnist

x_train, y_train, x_test, y_test = mnist.create_dataset(data_dir="data")
model = Sequential()
model.add(layers.Linear(128, activation="relu"))
model.add(layers.Linear(128, activation="relu"))
model.add(layers.Linear(1, activation="sigmoid"))

model.compile(lr=0.1, loss=BinaryCrossEntropy())
model.fit(
    x_train, y_train,
    epochs=100, callbacks=(
        CSVLogger(file_path="logs.csv", overwrite=True),
    ),
    verbose=True
)

predictions = model.predict(x_test)

accuracy = np.mean((predictions == y_test))

print(y_train.shape)
print("First 5 predictions")
print(predictions[..., :5])
print(y_test[..., :5])

print(f"Test accuracy: {accuracy * 100: .2f}%")
