#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 12:38:59 2023

@author: fahimahghazali
"""

import pandas as pd
import matplotlib.pyplot as plt


# model comparison: data read in manually
model_list = ['EfficientNetB0', 'InceptionV3', 'InceptionResnetV2', 'MobileNet']
img_dim = [224, 150, 299, 224]
size = [29, 92, 215, 16]
validation_accuracy = [53.43, 48.46, 53.03, 53.55]
speed = [17,5,60,11]

models = {'model': model_list, 
          'image_size': img_dim, 
          'val_acc': validation_accuracy, 
          'MB': size, 
          'speed': speed}

models = pd.DataFrame(models)

# bubble plots for model comparison
plt.scatter('speed', 'val_acc', s='MB', data = models)
plt.ylim((45,55))
plt.xlim((0,65))
plt.xlabel('validation speed (s)')
plt.ylabel('validation accuracy (%)')

# labeling bubble plots
plt.text(x=models.speed[0]+1,y=models.val_acc[0]-0.8,s=models.model[0], 
      fontdict=dict(color='black',size=8),
      bbox=dict(facecolor='white',alpha=0.5))

plt.text(x=models.speed[1]+1,y=models.val_acc[1]-0.8,s=models.model[1], 
      fontdict=dict(color='black',size=8),
      bbox=dict(facecolor='white',alpha=0.5))

plt.text(x=models.speed[2]-15,y=models.val_acc[2]-1.1,s=models.model[2], 
      fontdict=dict(color='black',size=8),
      bbox=dict(facecolor='white',alpha=0.5))

plt.text(x=models.speed[3]+1,y=models.val_acc[3]+0.5,s=models.model[3], 
      fontdict=dict(color='black',size=8),
      bbox=dict(facecolor='white',alpha=0.5))
plt.show()
