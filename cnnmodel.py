# -*- coding: utf-8 -*-
"""cnnmodel.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1vArhz33x0PA4YUTp8sMrfnc3eAI2GkCB
"""

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt

(X_train,y_train),(X_test,y_test)=datasets.cifar10.load_data()

X_train.shape
X_test.shape

y_train[:5]

y_train=y_train.reshape(-1)

y_train

classes=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

def plot_sample(X,y,index):
  plt.figure(figsize=(10,5))
  plt.imshow(X[index])
  plt.xlabel(classes[int(y_test[index])])

plot_sample(X_train,y_train,0)

plot_sample(X_train,y_train,2)

X_train=X_train/255
X_test=X_test/255

cnn=models.Sequential([
    #cnn
    layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    #DENSE
    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(10,activation='softmax')
])

cnn.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

cnn.fit(X_train,y_train,epochs=10)

cnn.evaluate(X_test,y_test)

from sklearn.metrics import confusion_matrix,classification_report
import numpy as np
y_predict=cnn.predict(X_test)
y_predict_classes=[np.argmax(element) for element in y_predict]
print(classification_report(y_test,y_predict_classes))

y_predict=cnn.predict(X_test)
y_predict[:5]

y_classes=[np.argmax(element) for element in y_predict]
y_classes[:5]

classes

y_test=y_test.reshape(-1)
y_test

classes[y_classes[100]]

plot_sample(X_test,y_test,100)