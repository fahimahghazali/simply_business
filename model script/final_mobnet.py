#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 20:37:46 2023

@author: fahimahghazali
"""

# General setup

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

# reading in data
batch_size = 32
img_height = 224
img_width = 224

# roughly 70% , 15%, 15% train validation test split

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
    label_mode = 'categorical',
    batch_size = batch_size)

# saving class names
val_class_names = validation_dataset.class_names
test_class_names = test_dataset.class_names

class_names = train_dataset.class_names
num_classes = len(class_names)

# preprocessing for performance
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# calling in weights of mobilenet parameters on imagenet
base_model = tf.keras.applications.MobileNet(input_shape = (img_height, img_width, 3), 
                                include_top = False, 
                                weights = 'imagenet')

# freezing the base model to obtain better initialization when fine-tuning
for layer in base_model.layers:
    layer.trainable = False

# preprocessing - data augmentation
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

# building the model by chaining
inputs = tf.keras.Input(shape=(img_height, img_width, 3))
x = data_augmentation(inputs)
x = rescale(x)
x = base_model(x, training=False)
x = global_average_layer(x)

# Add a final softmax layer with 4 nodes for classification output
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.models.Model(inputs, outputs)

# compile model
base_learning_rate = 0.0001
model.compile(optimizer = optimizers.Adam(learning_rate=base_learning_rate), 
              loss = tf.keras.losses.CategoricalCrossentropy(),
              metrics = ['acc'])

model.summary()

# early stopping to avoid overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_acc', 
    patience = 8, 
    min_delta = 0.001, 
    mode='max'
)

# model fitting
initial_epochs = 50
mobnet_history = model.fit(train_dataset, 
                                validation_data = validation_dataset, 
                                epochs = initial_epochs,
                                verbose = 2,
                                callbacks = early_stopping)

# saving the whole model
model.save('modelmobnet_update2.h5')
saved_model = tf.keras.models.load_model('modelmobnet_update2.h5')
saved_model.summary()

# train and validation metrics
train_loss, train_acc = saved_model.evaluate(train_dataset, verbose=2)
print('Restored model, train accuracy: {:5.2f}%'.format(100 * train_acc))
print('Restored model, train loss: {}'.format(train_loss))

val_loss, val_acc = saved_model.evaluate(validation_dataset, verbose=2)
print('Restored model, val accuracy: {:5.2f}%'.format(100 * val_acc))
print('Restored model, val loss: {}'.format(val_loss))

# training progress
acc = mobnet_history.history['acc']
val_acc = mobnet_history.history['val_acc']

loss = mobnet_history.history['loss']
val_loss = mobnet_history.history['val_loss']

init_epoch = len(acc)

epochs_range = range(len(acc))

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
plt.savefig('modelmobnet_update2.png', bbox_inches='tight')
plt.close()

# start fine-tuning

print("Number of layers in the base model: ", len(base_model.layers))

# unfreezing the last 35 layers
base_model.trainable = True
for layer in base_model.layers[:len(base_model.layers)-34]:
    layer.trainable = False

# training with lower learning rate
model.compile(optimizer = optimizers.Adam(learning_rate=base_learning_rate/10), 
              loss = tf.keras.losses.CategoricalCrossentropy(),
              metrics = ['acc'])

model.summary()

# also adopting early stopping
fine_tune_epochs = 75
total_epochs = init_epoch + fine_tune_epochs
mobnetfine_history = model.fit(train_dataset, 
                                validation_data = validation_dataset, 
                                epochs = total_epochs,
                                initial_epoch = mobnet_history.epoch[-1],
                                callbacks = early_stopping)

# fine-tuned training progress
acc += mobnetfine_history.history['acc']
val_acc += mobnetfine_history.history['val_acc']

loss += mobnetfine_history.history['loss']
val_loss += mobnetfine_history.history['val_loss']


plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.plot([init_epoch-1,init_epoch-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.plot([init_epoch-1,init_epoch-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.savefig('modelftmobnet_update2.png', bbox_inches='tight')
plt.close()

# saving fine-tuned model
model.save('modelftmobnet_update2.h5')
saved_model = tf.keras.models.load_model('modelftmobnet_update2.h5')
saved_model.summary()


# train and validation metrics
train_loss, train_acc = saved_model.evaluate(train_dataset, verbose=2)
print('Restored model, train accuracy: {:5.2f}%'.format(100 * train_acc))
print('Restored model, train loss: {}'.format(train_loss))

val_loss, val_acc = saved_model.evaluate(validation_dataset, verbose=2)
print('Restored model, val accuracy: {:5.2f}%'.format(100 * val_acc))
print('Restored model, val loss: {}'.format(val_loss))





# to make predictions, and decode one hot encoding to create confusion matrix
def get_actual_predicted_labels(dataset):
  """
    Create a list of actual ground truth values and the predictions from the model.

    Args:
      dataset: An iterable data structure, such as a TensorFlow Dataset, with features and labels.

    Return:
      Ground truth and predicted values for a particular dataset.
  """
  actual = [labels for _, labels in dataset.unbatch()]
  predicted = saved_model.predict(dataset)

  actual = tf.stack(actual, axis=0)
  actual = tf.argmax(actual, axis=1)
  predicted = tf.concat(predicted, axis=0)
  predicted = tf.argmax(predicted, axis=1)

  return actual, predicted

# returns and plots confusion matrix heatmap
def plot_confusion_matrix(actual, predicted, labels):
  cm = tf.math.confusion_matrix(actual, predicted)
  confmat = cm.numpy()
  print(confmat)
  fig, ax = plt.subplots()
  im = ax.imshow(confmat)
  fig.colorbar(im, ticks=list(np.arange(stop = np.max(confmat), step = 50)))
  # plt.set(font_scale=1.4)
  for i in range(len(labels)):
    for j in range(len(labels)):
        plt.annotate(str(confmat[i][j]), xy=(j, i),
                     ha='center', va='center', color='red')
 
  ax.set_xlabel('Predicted property type')
  ax.set_ylabel('Actual property type')
  ax.set_xticks(np.arange(confmat.shape[0]), labels = labels)
  ax.set_yticks(np.arange(confmat.shape[1]), labels = labels)
  plt.xticks(rotation=90)
  plt.yticks(rotation=0)
  plt.show()
  return confmat

# validation confusion matrix
actual, predicted = get_actual_predicted_labels(validation_dataset)
plot_confusion_matrix(actual, predicted, val_class_names)


# evaluation on test dataset
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)



test_loss, test_acc = saved_model.evaluate(test_dataset, verbose=2)
print('Restored model, test accuracy: {:5.2f}%'.format(100 * test_acc))
print('Restored model, test loss: {}'.format(test_loss))

test_actual, test_predicted = get_actual_predicted_labels(test_dataset)
confmat = plot_confusion_matrix(test_actual, test_predicted, test_class_names)

# class specific metrics to obtain precision, recall and f1-score
def class_metrics(confmat, label, label_names):
    ind = label_names.index(label)
    tp = confmat[ind][ind]
    total  = np.sum(confmat)
    col_total = np.sum(confmat, axis=0)[ind]
    row_total = np.sum(confmat, axis=1)[ind]
    fp = row_total - tp
    fn = col_total - tp
    tn = total - (tp + fn + fp)
    print([tp, fp, fn, tn])
    prec = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2*(prec*recall/(prec + recall))
    support = row_total
    metrics = [prec, recall, f1, support]
    return [round(met, 3) for met in metrics]


class_metrics(confmat, 'Terraced', test_class_names)    