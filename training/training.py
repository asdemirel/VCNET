import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential,Model,model_from_json
from keras.layers import Conv2D,Conv1D,MaxPooling2D,MaxPooling1D, AveragePooling2D,BatchNormalization
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import visualkeras

data = np.loadtxt('csvset/17Mart.csv', dtype=int)

train_data = data[:,0:-1]
y_train = data[:, -1]

train_data, x_test,y_train,y_test = train_test_split(train_data ,y_train ,test_size=0.20 ,random_state=3)

train_data = train_data.astype(np.float)
x_test = x_test.astype(np.float)

train_labels_flat = y_train.ravel()
train_labels_count = np.unique(train_labels_flat).shape[0]  
                                                           
def dense_to_one_hot (labels_dense , num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels , num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


y_train = dense_to_one_hot(train_labels_flat, train_labels_count)
y_train = y_train.astype(np.uint8)

test_labels_flat = y_test.ravel()
test_labels_count = np.unique(test_labels_flat).shape[0]

y_test = dense_to_one_hot(test_labels_flat,test_labels_count)
y_test = y_test.astype(np.uint8)

x_train = train_data.reshape(-1,128,1)
x_test = x_test.reshape(-1,128,1)

#MODEL OLUŞTURMA.
print(x_train.shape[1])

model = Sequential()

model.add(Conv1D(64,kernel_size=3 ,activation='relu',input_shape=(128,1)))
model.add(Conv1D(64,kernel_size=10 ,activation='relu'))
model.add(MaxPooling1D(pool_size =2))
model.add(Dropout(0.3))
model.add(Conv1D(32,kernel_size=1 ,activation='relu'))
model.add(Conv1D(32,kernel_size=1 , strides=1,activation='relu'))
model.add(MaxPooling1D(pool_size =2))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation("relu"))
model.add(Dropout(0.4))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',optimizer ='adam' ,metrics=['accuracy'])
model.summary()


checkpointer =ModelCheckpoint(filepath ='model.h5',verbose =1 ,save_best_only =True)   

model.fit(x_train , y_train , epochs=20, batch_size=50 ,validation_data=(x_test, y_test) ,callbacks = [checkpointer] , verbose=2)

y_pred_test = model.predict(x_test)
max_y_pred_test = np.argmax(y_pred_test,axis=1)
max_y_test = np.argmax(y_test,axis=1)

from sklearn.metrics import confusion_matrix
print('Confusion matrix = ',confusion_matrix(max_y_test,max_y_pred_test))

from sklearn.metrics import classification_report
print('\n\n\n',classification_report(max_y_test,max_y_pred_test))
