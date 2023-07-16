#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 18:35:16 2023

@author: fahimahghazali
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil



batch_size = 32
img_height = 224
img_width = 224

train_dataset = tf.keras.utils.image_dataset_from_directory(
  './rs_streetviews/trainval',
  validation_split=0.18,
  subset = "training",
  label_mode = 'categorical',
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
  './rs_streetviews/trainval',
  validation_split=0.18,
  subset = "validation",
  label_mode = 'categorical',
  seed = 123,
  image_size = (img_height, img_width),
  batch_size = batch_size)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    './rs_streetviews/test',
    image_size = (img_height, img_width),
    batch_size = batch_size)


AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)


base_model = tf.keras.applications.MobileNet(input_shape = (img_height, img_width, 3), 
                                include_top = False, 
                                weights = 'imagenet')

for layer in base_model.layers:
    layer.trainable = False

data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1) # zooming in 10%,
  ]
)

rescale = tf.keras.layers.Rescaling(1./255)

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()


inputs = tf.keras.Input(shape=(img_height, img_width, 3))
x = data_augmentation(inputs)
x = rescale(x)
x = base_model(x, training=False)
x = global_average_layer(x)

# Add a final sigmoid layer with 1 node for classification output
outputs = layers.Dense(4, activation='softmax')(x)

model = tf.keras.models.Model(inputs, outputs)

base_learning_rate = 0.0001
model.compile(optimizer = optimizers.Adam(learning_rate=base_learning_rate), 
              loss = tf.keras.losses.CategoricalCrossentropy(),
              metrics = ['acc'])

model.summary()

initial_epochs = 15
mobnet_history = model.fit(train_dataset, 
                                validation_data = validation_dataset, 
                                epochs = initial_epochs,
                                verbose = 2)


model.save('models/modelmobnet.h5')
saved_model = tf.keras.models.load_model('models/modelmobnet.h5')
saved_model.summary()

train_loss, train_acc = saved_model.evaluate(train_dataset, verbose=2)
print('Restored model, train accuracy: {:5.2f}%'.format(100 * train_acc))
print('Restored model, train loss: {}'.format(train_loss))

val_loss, val_acc = saved_model.evaluate(validation_dataset, verbose=2)
print('Restored model, val accuracy: {:5.2f}%'.format(100 * val_acc))
print('Restored model, val loss: {}'.format(val_loss))

acc = mobnet_history.history['acc']
val_acc = mobnet_history.history['val_acc']

loss = mobnet_history.history['loss']
val_loss = mobnet_history.history['val_loss']

epochs_range = range(initial_epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('figures/modelmobnet.png', bbox_inches='tight')
plt.close()




print("Number of layers in the base model: ", len(base_model.layers))

base_model.trainable = True
for layer in base_model.layers[:len(base_model.layers)-29]:
    layer.trainable = False


model.compile(optimizer = optimizers.Adam(learning_rate=base_learning_rate/10), 
              loss = tf.keras.losses.CategoricalCrossentropy(),
              metrics = ['acc'])

model.summary()

fine_tune_epochs = 15
total_epochs = initial_epochs + fine_tune_epochs
mobnetfine_history = model.fit(train_dataset, 
                                validation_data = validation_dataset, 
                                epochs = total_epochs,
                                initial_epoch = mobnet_history.epoch[-1],
                                verbose = 2)

acc += mobnetfine_history.history['acc']
val_acc += mobnetfine_history.history['val_acc']

loss += mobnetfine_history.history['loss']
val_loss += mobnetfine_history.history['val_loss']

epochs_range = range(total_epochs)

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.savefig('figures/modelftmobnet.png', bbox_inches='tight')
plt.close()

model.save('models/modelftmobnet.h5')
saved_model = tf.keras.models.load_model('models/modelftmobnet.h5')
saved_model.summary()

train_loss, train_acc = saved_model.evaluate(train_dataset, verbose=2)
print('Restored model, train accuracy: {:5.2f}%'.format(100 * train_acc))
print('Restored model, train loss: {}'.format(train_loss))

val_loss, val_acc = saved_model.evaluate(validation_dataset, verbose=2)
print('Restored model, val accuracy: {:5.2f}%'.format(100 * val_acc))
print('Restored model, val loss: {}'.format(val_loss))

