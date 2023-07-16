#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 12:51:49 2023

@author: fahimahghazali
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil


try:
    shutil.rmtree('./rs_streetviews')
except OSError as e:
    print("Error: %s - %s." % (e.filename, e.strerror))


try:
    shutil.rmtree('./houses')
except OSError as e:
    print("Error: %s - %s." % (e.filename, e.strerror))


shutil.copytree('uni_project_data/street_view_juny12', 'houses')

flist = os.listdir('./houses')

prop = pd.read_csv("uni_project_data/properties_juny12.csv")
# print(len(prop))
prop['id'] = "gsv_" + prop["property_id"] + ".jpg"
prop.head()

im = pd.DataFrame(flist, columns = ['dir'])
im['id'] = im['dir'].apply(os.path.basename)
im.head()

props = prop.merge(im, on = 'id', how='inner')
props.head()
# len(props)

len(props[props.duplicated() == True])

unknown_props = props[props['propertyType']== 'Unknown']
# len(unknown_props)
known_props = props[props['propertyType'] != 'Unknown']
# len(known_props)

from sklearn.model_selection import train_test_split
trainval_set, test_set = train_test_split(known_props,
                                          train_size = 0.85,
                                          shuffle = True,
                                          stratify = known_props['propertyType'],
                                          random_state = 42)

trainval_set = trainval_set.reset_index(drop = True)
test_set = test_set.reset_index(drop = True)


class_names = list(known_props['propertyType'].unique())

trainval_set = trainval_set.sort_values('propertyType')
test_set = test_set.sort_values('propertyType')



# creating subfolders

os.makedirs('rs_streetviews/trainval')
for c in class_names:
    dest = 'rs_streetviews/trainval/' + c
    os.makedirs(dest)
    for i in list(trainval_set[trainval_set['propertyType'] == c]['id']): # Image Id
        get_image = os.path.join('houses', i) # Path to Images
        shutil.move(get_image, dest)

os.makedirs('rs_streetviews/test')
for c in class_names:
    dest = 'rs_streetviews/test/' + c
    os.makedirs(dest)
    for i in list(test_set[test_set['propertyType'] == c]['id']): # Image Id
        get_image = os.path.join('houses', i) # Path to Images
        move_image_to_cat = shutil.move(get_image, dest)


batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
  './rs_streetviews/trainval',
  validation_split=0.18,
  subset = "training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names

plt.figure(figsize=(12, 12))
for images, labels in train_ds.take(1):
  for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
plt.savefig('figures/classplot.png', bbox_inches='tight')
plt.close()

from tensorflow.keras.models import Sequential

data_augmentation = Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1) # zooming in 10%,
  ]
)

for image, _ in train_ds.take(1):
  plt.figure(figsize=(10, 10))
  first_image = image[5]
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
    plt.imshow(augmented_image[0] / 255)
    plt.axis('off')
plt.savefig('figures/augmentation.png', bbox_inches='tight')
plt.close()

























