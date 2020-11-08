#import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from fastai.vision.all import *
import numpy as np
import pathlib
#import tensorflow as tf
import sys
import matplotlib.pyplot as plt



#CLASS_NAMES = np.array([item.name for item in data_dir_train.glob('*') if item.name != "LICENSE.txt"])
BATCH_SIZE = 32
IMG_SIZE = (48, 48)
path=sys.argv[1]
mode=sys.argv[2]

def prepare_dataset(path):
    data_dir = pathlib.Path(path)
    train_dir = data_dir / "train"
    val_dir = data_dir / "test"
    class_names = np.array([item.name for item in train_dir.glob('*') if item.name != "LICENSE.txt"])

    datagen = ImageDataGenerator(rescale=1./255)

    train_it = datagen.flow_from_directory(train_dir, class_mode='categorical',color_mode="grayscale",batch_size=64, target_size=(48,48),classes = list(class_names))
    val_it = datagen.flow_from_directory(val_dir, class_mode='categorical',color_mode="grayscale",batch_size=64, target_size=(48,48),classes = list(class_names))

    # confirm the iterator works
    batchX, batchy = train_it.next()
    print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))
    return train_it,val_it,class_names


def show_batch(image_batch, label_batch,class_names):
    plt.figure(figsize=(10,10))
    for n in range(10):
        ax = plt.subplot(5,5,n+1)
        plt.imshow(image_batch[n])
        plt.title(class_names[label_batch[n]==1][0].title())
        plt.axis('off')
    plt.show()


def convolutional_model(mode,path):

    train_it,val_it,class_names = prepare_dataset(path)
    
    # create model
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    if(mode == 'train'):
        model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])
        model.fit(train_it,epochs=50, validation_data=val_it)
        model.save_weights('fruits.h5')
    else:
        model.load_weights("fruits.h5")

    return model
                








                
        






