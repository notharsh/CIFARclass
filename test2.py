from __future__ import print_function
import pandas as pd
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D , ZeroPadding2D

batch_size = 16
num_classes = 10
epochs = 10
data_augmentation = True

img_rows, img_cols = 32, 32

img_channels = 3

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_train.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
model = Sequential()
 
model.add(Conv2D(32, 3, 3,
                          input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
 
model.add(Conv2D(64, 3, 3))
model.add(Activation('relu'))
model.add(ZeroPadding2D((1, 1)))
 
model.add(Conv2D(128, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(ZeroPadding2D((1, 1)))
 
model.add(Conv2D(256, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(ZeroPadding2D((1, 1)))

model.add(Conv2D(512, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(ZeroPadding2D((1, 1)))
 
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128))
model.add(Dropout(0.2))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 2550

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
    model.save_weights('nweights.h5')
    model.save('model.h5')
    predict_classes = model.predict_classes(x_train,
    			  batch_size=batch_size,
    			  verbose = 1
    			  )
    df = pd.DataFrame(predict_classes)
    df.to_csv("Prediction.csv")
    """i=0
    while(i<100):
    	print(predict_classes[i],' ')
    	i = i+1"""
else:
    print('Using real-time data augmentation.')
   
    datagen = ImageDataGenerator(
        featurewise_center=False,  
        samplewise_center=False, 
        featurewise_std_normalization=False, 
        samplewise_std_normalization=False, 
        zca_whitening=False, 
        rotation_range=0, 
        width_shift_range=0.1, 
        height_shift_range=0.1,  
        horizontal_flip=True,  
        vertical_flip=False)  

  
    datagen.fit(x_train)

 
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        			 steps_per_epoch=x_train.shape[0] // batch_size,
                      				 epochs=epochs,
									 validation_data=(x_test, y_test))
    model.save_weights('aweights.h5')
    model.save('Model.h5')
    predict_classes = model.predict_classes(x_train,
    			  batch_size=batch_size,
    			  verbose = 1
    			  )
    df = pd.DataFrame(predict_classes)
    df.to_csv("Prediction.csv")
    """i=0
    while(i<100):
    	print(predict_classes[i],' ')
    	i = i+1"""