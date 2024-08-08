import numpy as np

def MSE(y_true, y_pred):
    return np.average(np.average((y_pred - y_true) ** 2, axis = 0))