'''
加载训练集
'''

import struct
import numpy as np

trainingImagesPath="Sample\\train-images.idx3-ubyte"
trainingLabelsPath="Sample\\train-labels.idx1-ubyte"
testImagesPath="Sample\\t10k-images.idx3-ubyte"
testLabelsPath="Sample\\t10k-labels.idx1-ubyte"

def loadImage(filePath=trainingImagesPath):
    with open(filePath,'rb') as imgPath: #以二进制读取
       magic,num,row,col =struct.unpack('>IIII',imgPath.read(16))
       images=np.fromfile(imgPath,np.uint8)
       return images
def loadLabel(filePath=trainingLabelsPath):
    with open(filePath,'rb') as lbPath: #以二进制读取
       magic,n =struct.unpack('>II',lbPath.read(8))
       labels=np.fromfile(lbPath,np.uint8)
       return labels
if __name__   =='__main__':
   images= loadImage(trainingImagesPath)
   labels=loadLabel(trainingLabelsPath)

   X_train=images.reshape(len(labels),784)
   y_train=labels

   print('OK')