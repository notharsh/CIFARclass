import pandas as pd
import keras
from keras.models import load_model
from keras.datasets import cifar10
num_classes = 10
model = load_model('Model.h5')
img_rows, img_cols = 32, 32
img_channels = 3
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape[0], 'test samples')
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
predict_classes = model.predict_classes(x_train,
    			  batch_size=16,
    			  verbose = 1
    			  )
df = pd.DataFrame(predict_classes)
df.to_csv("Prediction2.csv")