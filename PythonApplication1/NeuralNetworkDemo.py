'''
神经网络
'''
import numpy as np
import GradientDescent as gd
import LoadTrainingSet as ld


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
    
    #print(A)
    #计算误差
    #反向传播算法
    delta=np.empty(K,dtype='O')
    D=np.empty(K,dtype='O')
    for k in np.arange(K-1,-1,-1,np.uint8):
        if k==K-1:
            delta[k]=A[k+1]-y
        else:
            delta[k]=np.dot(delta[k+1],thetas[k+1].T)*gd.sigmoidgradient(np.dot(A[k],thetas[k]))
        #计算每个神经元所有样本的总偏导数
        D[k]=np.sum(delta[k]*A[k+1],axis=0)
    #print("delta:%s"%delta)
    #print("D:%s"%D)
    total=y*np.log10(A[K])+ (1-y)*np.log10(1-A[K])
    J=-(np.sum(y*np.log10(A[K])+ (1-y)*np.log10(1-A[K])))/m

    #返回每个神经元的偏导数和成本，进行梯度下降训练
    return D,J




if __name__   =='__main__':
    theta1=np.ones((3,4))
    theta2=np.ones((4,2))
    thetas=np.array([theta1,theta2])
    X=np.array([[0.4,0.5,0.6],
                [0.1,0.2,0.3]]).reshape((2,3))
    y=np.array([[0,1],
                [1,0]]).reshape((2,2))

    images= ld.loadImage()
    labels=ld.loadLabel()

    m=len(labels)
    
    X_train=images.reshape(m,784)
    eye=np.eye(10)
    
    y_train=np.empty((m,10))
    for i in np.arange(m):
        y_train[i]=eye[labels[i]]
    #均值归一化
    X_train=(X_train-np.mean(X_train,axis=0))/255

    tempX=X_train[0:100]
    tempy=y_train[0:100]


    theta1=np.random.random((784,785))/100
    theta2=np.random.random((785,10))/100
    thetas=np.array([theta1,theta2])

    grad,J=NNCostFunction(thetas,tempX,tempy)
    print("J:%f"%J)
    v=gd.nngradientDescent(tempX,tempy,thetas,grad)

    #print('1');
