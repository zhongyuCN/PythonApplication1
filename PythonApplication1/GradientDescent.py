import numpy as np
import NeuralNetworkDemo as nn

def lrCostFunction(X,y,theta):
    m,feature=X.shape

    H=sigmoid(X,theta)
def nngradientDescent(X,y,theta,grad):
    _lambda=0.1
    step=100
    J=np.zeros(step)
    temptheta=theta-_lambda*grad
    for i in np.arange(step):        
        tempgrad,J[i]=nn.NNCostFunction(temptheta,X,y)
        
        temptheta=temptheta-_lambda*tempgrad
        print("第%d次迭代,J=%f"%(i,J[i]))
        if i>0:
            if J[i]>J[i-1]:
                print("梯度上升!..............")
            else:
                print("梯度下降")
    
    return temptheta


def sigmoid(X,theta):
    z=np.dot(theta,X)
    g=1/(1+np.exp(-z))
    return g

def sigmoid2(z):   
    g=1/(1+np.exp(-z))
    return g
def sigmoidgradient(z):
    return sigmoid2(z)*(1-sigmoid2(z))