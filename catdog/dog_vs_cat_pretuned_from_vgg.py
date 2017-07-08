
# coding: utf-8

# In[1]:

#import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation,Dropout,Flatten,Dense
from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img
from keras.applications.vgg16 import VGG16
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

# In[3]:

from keras import backend as K
K.set_image_dim_ordering('th')

model_vgg16_tmp = VGG16(weights='imagenet',include_top=False)
model_vgg16 = VGG16(weights=None,include_top=False,input_shape=(3, 150, 150))
for layer, layer_weights in zip(model_vgg16.layers, model_vgg16_tmp.layers):
    weights = layer_weights.get_weights()
    if not len(weights):
        continue
    if type(layer).__name__ == 'Conv2D':
        weights[0] = weights[0].transpose(2,3,0,1)
    layer.set_weights(weights)


batch_size = 20
train_datasize = 20000
val_datasize = 5000
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
        class_mode=None,
        shuffle=False)
bottleneck_features_train = model_vgg16.predict_generator(train_generator,train_datasize//batch_size)
# save the output as a Numpy array
np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)


validation_generator = test_datagen.flow_from_directory(
        'validation',
        target_size=(150,150),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False
        )

bottleneck_features_validation = model_vgg16.predict_generator(validation_generator,val_datasize//batch_size)
np.save(open('bottleneck_features_validation','w'),bottleneck_features_validation)

train_labels = np.array([0]*train_datasize/2+[1]*train_datasize/2)
val_labels = np.array([0]*val_datasize/2+[1]*val_datasize/2)

model = Sequential()
model.add(Flatten(input_shape=bottleneck_features_train.shape[1:]))  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

model.fit(bottleneck_features_train,train_labels,
                    batch_size=batch_size,
                    epochs=50,
                    validation_data=(bottleneck_features_validation,val_labels))
model.save_weights('bottleneck_fc_model.h5')

validation_score = model.evaluate(validation_data,val_labels,
                        batch_size=batch_size)

model_vgg16.add(model)

for layer in model_vgg16.layers[:25]:
    layer.trainable = False

model_vgg16.compile(loss='binary_crossentropy',optimizer=optimizers.SGD(lr=1e-4,momentum=0.9),metrics=['accuracy'])

train_generator_ft = train_datagen.flow_from_directory(
    'train',
    target_size=(150,150),
    batch_size=batch_size,
    class_mode='binary')
validation_generator_ft = test_datagen.flow_from_directory(
    'validation',
    target_size=(150,150),
    batch_size=batch_size,
    class_mode='binary')

model_vgg16.fit_generator(train_generator,
    steps_per_epoch=train_datasize//batch_size,
    epochs=50,
    validation_data=validation_generator_ft,
    validation_steps=val_datasize//batch_size)

test_generator_ft = test_datagen.flow_from_directory(
        'test',
        target_size=(150,150),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
test_prediction = model_vgg.predict_generator(test_generator,
                       steps=12500//batch_size)
test_filename = [re.split("/|\.",t)[-2] for t in test_generator.filenames]
results = pd.DataFrame({'id':test_filename,'label':test_prediction.reshape(-1).tolist()})
results.to_csv('submission_vggft.csv')