from sklearn.model_selection import KFold

count = 0
kfold = KFold(n_splits=10, shuffle=True, random_state=12)
valid_scores = []
train_scores = []

for train_idx, valid_idx in kfold.split(train_poly_fea):
    while count < 1:
        count += 1
        

class Model:
    def __init__(self, n_inputs, n_hidden, n_outputs):
        network = list()

        