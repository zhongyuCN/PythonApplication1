import LoadTrainingSet as ls
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
import time


def test1():
    trainingImagesPath="Sample\\train-images.idx3-ubyte"
    trainingLabelsPath="Sample\\train-labels.idx1-ubyte"
    testImagesPath="Sample\\t10k-images.idx3-ubyte"
    testLabelsPath="Sample\\t10k-labels.idx1-ubyte"

    x_train=ls.loadImage().reshape(60000,28,28)
    y_train=ls.loadLabel().reshape((60000,1))
    x_test=ls.loadImage(testImagesPath).reshape(10000,28,28)
    y_test=ls.loadLabel(testLabelsPath).reshape((10000,1))


    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128, activation='sigmoid',kernel_regularizer=tf.keras.regularizers.l1(0.01)),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    #使用tensorboard
    NAME = 'DigiRecognizer-CNN-{}'.format(int(time.time()))
    tensorboard = TensorBoard()

    model.fit(x_train,
             y_train, 
             batch_size=200,
             epochs=100,
             callbacks=[tensorboard])

    model.evaluate(x_test,  y_test, verbose=2)
    model.save('D://my_model.h5')

def test2():
    saved_model_path="D://my_model.h5"
    model=tf.keras.models.load_model(saved_model_path)
    # 显示网络结构
    #model.summary()
    
    img=cv2.imread("Sample\\one.png")

    #灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    height,width=img.shape[0:2]
    new_img=cv2.resize(gray,(int(height*0.5),int(width*0.5)))
    

    predict=model.predict(new_img.reshape(-1,28,28))
    print(np.argmax(predict))
    cv2.imshow('new_img',new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def displayimage():
    x_train=ls.loadImage().reshape(60000,28,28)
    y_train=ls.loadLabel().reshape((60000,1))
    plt.figure(figsize=(20,20))
    for i in np.arange(100):
        plt.subplot(10,10,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[i], cmap=plt.cm.binary)
        plt.xlabel(y_train[i])
    plt.show()
        


if __name__   =='__main__':
    #test1()
    #test2()
    displayimage()