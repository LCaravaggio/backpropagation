import numpy as np

"""
Debería resolver esta práctica sin agregar más librerías externas
"""

def NotImplemented_message():
    print('###################################')
    print('Tienen que implementar esta función')
    print('###################################')
    return np.array([1, 1])

def densa_forward(X, W, b):
    return np.matmul(X,W)+b

def MSE(X_true, X_pred):
    return ((X_pred - X_true)**2).mean(1)

def MSE_grad(X_true, X_pred):
    return NotImplemented_message()

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def sigmoid_jac(Xin):
    return NotImplemented_message()

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))

def softmax_jac(Xin):
    return NotImplemented_message()

def forward(X, P_true, weights):
    d1=densa_forward(X, weights[0], weights[1])
    s1=sigmoid(d1)
    d2=densa_forward(s1, weights[2], weights[3])
    s2=sigmoid(d2)
    d3=densa_forward(s2, weights[4], weights[5])
    salida=softmax(d3)
    mse=MSE(P_true, salida)
    return  salida, mse, X, s1, s2

def get_gradients(X, P_true, weights):
    return NotImplemented_message()