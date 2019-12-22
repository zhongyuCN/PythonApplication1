import LoadTrainingSet as ls
import tensorflow as tf

def test1():
    trainingImagesPath="Sample\\train-images.idx3-ubyte"
    trainingLabelsPath="Sample\\train-labels.idx1-ubyte"
    testImagesPath="Sample\\t10k-images.idx3-ubyte"
    testLabelsPath="Sample\\t10k-labels.idx1-ubyte"

    x_train=ls.loadImage().reshape((60000,784))
    y_train=ls.loadLabel()
    x_test=ls.loadImage(testImagesPath).reshape((10000,784))
    y_test=ls.loadLabel(testLabelsPath)


    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5)

    model.evaluate(x_test,  y_test, verbose=2)

if __name__   =='__main__':
    test1()