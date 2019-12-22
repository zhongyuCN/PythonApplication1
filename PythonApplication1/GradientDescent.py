import numpy as np

def lrCostFunction(X,y,theta):
    m,feature=X.shape

    H=sigmoid(X,theta)


def sigmoid(X,theta):
    z=np.dot(theta,X)
    g=1/(1+np.exp(-z))
    return g

def sigmoid2(z):   
    g=1/(1+np.exp(-z))
    return g
def sigmoidgradient(z):
    return sigmoid2(z)*(1-sigmoid2(z))