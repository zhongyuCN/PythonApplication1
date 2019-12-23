'''
神经网络
'''
import numpy as np
import GradientDescent as gd

def NNCostFunction(thetas,X,y):
    #取出神经网络层数
    K=thetas.size
    #取出样本数量
    m,feature=X.shape
    #给样本添加偏置项
    tempX=np.hstack((np.ones((m,1)),X))
    #给theta添加偏置项
    tempthetas=np.empty(K,dtype='O')
    for i in np.arange(K):
        #theta添加偏置项
        innum,outnum=thetas[i].shape
        tempthetas[i]=np.vstack((np.ones((1,outnum)),thetas[i]))
    A=np.empty(K+1,dtype='O')
    A[0]=X
    for k in np.arange(K):
        if k==0:
            #第一层            
            a=gd.sigmoid2(np.dot(tempX,tempthetas[k]))
        else:
            
            #添加偏置项
            a=np.hstack((np.ones((a.shape[0],1)),a))
            a=gd.sigmoid2(np.dot(a,tempthetas[k]))
        #每层的计算值
        A[k+1]=a
    
    print(A)
    #计算成本
    #反向传播算法
    delta=np.empty(K,dtype='O')
    for k in np.arange(K-1,-1,-1,np.uint8):
        if k==K-1:
            delta[k]=A[k+1]-y
        else:
            delta[k]=np.dot(delta[k+1],thetas[k+1].T)*gd.sigmoidgradient(np.dot(A[k],thetas[k]))
    print(delta)
    print("delta求和")
    D=np.empty(delta.size,dtype='O')
    for i in np.arange(delta.size):
        D[i]=np.sum(delta[i])
    print(D)



if __name__   =='__main__':
    theta1=np.ones((3,4))
    theta2=np.ones((4,2))
    thetas=np.array([theta1,theta2])
    X=np.array([0.4,0.5,0.6]).reshape((1,3))
    y=np.array([0,1]).reshape((1,2))
    
    NNCostFunction(thetas,X,y)
    print('1');
