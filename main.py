from NN import layers
import numpy as np
from NN.sequential import Sequential
from NN.losses import BinaryCrossEntropy
from NN.callback import CSVLogger
import pandas as pd
import matplotlib.pyplot as plt
# import preprocessing
from sklearn.model_selection import KFold

train = np.load('save/train.npy')
print(train.shape)
test = np.load('save/test.npy')
train_poly_fea = np.load('save/train_poly_fea.npy')
test_poly_fea = np.load('save/test_poly_fea.npy')
TARGET = np.load('save/target.npy')

model = Sequential()
model.add(layers.Linear(128, activation="relu"))
model.add(layers.Linear(64, activation="relu"))
model.add(layers.Linear(1, activation="sigmoid"))

model.compile(lr=0.1, loss=BinaryCrossEntropy(), regularization_factor=10.)

count = 0
kfold = KFold(n_splits = 10, shuffle = True, random_state = 12)
valid_scores = []
train_scores = []

for train_idx, valid_idx in kfold.split(train_poly_fea):
    while count < 1:
        count += 1
        # Split train, valid
        train_features, train_labels = train[train_idx], TARGET[train_idx]
        valid_features, valid_labels = train[valid_idx], TARGET[valid_idx]
        data = np.array([train_features[:, i] for i in range(train.shape[1])])
        model.fit(
            data,
            train_labels,
            epochs=2,
            callbacks=(
                CSVLogger(file_path="logs/logs1.csv", overwrite=True),
            ), verbose=True
        )

# model.fit(x_train, y_train, epochs=80, callbacks=(
#         CSVLogger(file_path="logs/logs.csv", overwrite=True),
#     ), verbose=True)
train_prob_nn = model.predict(train.T)
print(TARGET.shape)
from sklearn.metrics import roc_curve


def _plot_roc_curve(fpr, tpr, thres):
    roc = plt.figure(figsize = (10, 8))
    plt.plot(fpr, tpr, 'b-', label = 'ROC')
    plt.plot([0, 1], [0, 1], '--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    return roc

fpr4, tpr4, thres4 = roc_curve(TARGET, train_prob_nn.T)
_plot_roc_curve(fpr4, tpr4, thres4)
from sklearn.metrics import auc, precision_recall_curve, accuracy_score

print(auc(fpr4, tpr4))

# def _plot_prec_rec_curve(prec, rec, thres):
#     plot_pr = plt.figure(figsize = (10, 8))
#     plt.plot(thres, prec[:-1], 'b--', label = 'Precision')
#     plt.plot(thres, rec[:-1], 'g-', label = 'Recall')
#     plt.xlabel('Threshold')
#     plt.ylabel('Probability')
#     plt.title('Precsion vs Recall Curve')
#     plt.show()
#     return plot_pr

# prec, rec, thres = precision_recall_curve(TARGET, train_prob_nn.T)
# _plot_prec_rec_curve(prec, rec, thres)
from sklearn.metrics import classification_report
print(classification_report(TARGET, train_prob_nn.T))