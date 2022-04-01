import numpy as np
import sklearn
from sklearn import datasets
from sklearn import model_selection


# Load the data from scikit-learn.
digits = datasets.load_digits()

# Load the targets.
# Note that the targets are stored as digits, these need to be 
#  converted to one-hot-encoding for the output sofmax layer.
T = np.zeros((digits.target.shape[0], 10))
T[np.arange(len(T)), digits.target] += 1

# Divide the data into a train and test set.
(X_train, X_test, T_train, T_test) = model_selection.train_test_split(digits.data, T, test_size=0.4)
# Divide the test set into a validation set and final test set.
(X_validation, X_test, T_validation, T_test) = model_selection.train_test_split(X_test, T_test, test_size=0.5)