
# coding: utf-8

# In[1]:

#import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation,Dropout,Flatten,Dense
from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output



# In[3]:

from keras import backend as K
K.set_image_dim_ordering('th')


# In[4]:

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(3, 150, 150)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# the model so far outputs 3D feature maps (height, width, features)
model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])


# In[ ]:

batch_size = 20
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(
        rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        'train',
        target_size=(150,150),
        batch_size=batch_size,
        class_mode='binary')
validation_generator = test_datagen.flow_from_directory(
        'validation',
        target_size=(150,150),
        batch_size=batch_size,
        class_mode='binary')

model.fit_generator(train_generator,
                    steps_per_epoch=20000//batch_size,
                    epochs=50,
                    validation_data=validation_generator,
                    validation_steps=5000//batch_size)
model.save_weights("dog_vs_cat.h5")
validation_score = model.evaluate_generator(validation_generator,
                        steps=5000//batch_size)


# In[ ]:

test_generator = test_datagen.flow_from_directory(
        'test',
        target_size=(150,150),
        batch_size = batch_size,
        class_mode = None,
        shuffle=False)
test_prediction = model.predict_generator(test_generator,
                       steps=12500//batch_size)
test_filename = [re.split("/|\.",t)[-2] for t in test_generator.filenames]
results = pd.DataFrame({'id':test_filename,'label':test_prediction.reshape(-1).tolist()})
results.to_csv('submission.csv')