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
    return 2*((X_pred - X_true))/X_pred.shape[1]

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def sigmoid_jac(Xin):
    return np.diag((sigmoid(Xin)*(1-sigmoid(Xin))).reshape(-1))

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))

def softmax_jac(Xin):
    softmax_out = softmax(Xin)
    m1=np.diag(softmax_out.reshape(-1))
    m2=softmax_out.T.dot(softmax_out)
    return m1-m2

def forward(X, P_true, weights):
    d1=densa_forward(X, weights[0], weights[1])
    s1=sigmoid(d1)
    d2=densa_forward(s1, weights[2], weights[3])
    s2=sigmoid(d2)
    d3=densa_forward(s2, weights[4], weights[5])
    salida=softmax(d3)
    mse=MSE(P_true, salida)
    return  salida, mse, X, s1, s2

def get_gradients(X, P_true, w):
       
    D1_out = densa_forward(X, w[0], w[1])
    A1_out = sigmoid(D1_out)
    
    D2_out = densa_forward(A1_out, w[2], w[3])
    A2_out = sigmoid(D2_out)
    
    D3_out = densa_forward(A2_out, w[4], w[5])
    P_est = softmax(D3_out)
        
    softmax_out = softmax(D3_out)
    
    MSE_grad_out = MSE_grad(P_true, P_est)
    out = forward(X, P_true, w)
        
    mse=MSE(P_est, P_true)
    
    error_D3 = softmax_jac(D3_out).dot(MSE_grad_out.T)
    error_A2 = w[4].dot(error_D3)
    error_D2 = sigmoid_jac(D2_out).dot(error_A2)
    
    error_A1 = w[2].dot(error_D2)
    error_D1 = sigmoid_jac(D1_out).dot(error_A1)
    
    g1, l1=np.matmul(error_D1 , X), error_D1
    g2, l2=np.matmul(error_D2 , A1_out), error_D2
    g3, l3=np.matmul(error_D3 , A2_out), error_D3
    grads=g1.T, l1, g2.T, l2, g3.T, l3
    loss=mse
    return grads, loss