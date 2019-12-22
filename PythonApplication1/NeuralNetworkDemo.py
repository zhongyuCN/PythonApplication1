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
    for i in np.arange(K):
        #theta添加偏置项
        innum,outnum=thetas[i].shape
        thetas[i]=np.vstack((np.ones((1,outnum)),thetas[i]))
   
    for k in np.arange(K):
        if k==0:
            #第一层            
            a=gd.sigmoid2(np.dot(tempX,thetas[k]))
        else:
            
            #添加偏置项
            a=np.hstack((np.ones((a.shape[0],1)),a))
            a=gd.sigmoid2(np.dot(a,thetas[k]))
    predict=a
    print(a)
    #计算成本
    #反向传播算法



        


if __name__   =='__main__':
    theta1=np.ones((3,4))
    theta2=np.ones((4,2))
    thetas=np.array([theta1,theta2])
    X=np.array([1,2,3]).reshape((1,3))
    
    NNCostFunction(thetas,X,0)
    print('1');
