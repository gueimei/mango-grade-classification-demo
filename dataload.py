# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 10:30:18 2020

@author: Owner
"""
import os
from PIL import Image


data_folder = 'C:/Users/Owner/Desktop/keras_spp/images/org'
data_dir = list()

for img in os.listdir(data_folder):
  data_dir.append(os.path.join(data_folder, img))



from collections import defaultdict
import numpy as np
size_cluster = defaultdict(list)

i = 0
while i < len(data_dir):
    img = Image.open(data_dir[i])
    im = np.array(img)#image to numpy
    im = im.astype('float32')
    im /=255
    size_cluster[i] = im
    i+=1


from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

label = np.ones((len(data_dir),), dtype = int)
#label[0:2035] = 0
#label[2035:4396] = 1
#label[4396:] = 2
label[0:3] = 0
label[3:9] = 1
label[9:] = 2

Data, Label = shuffle(size_cluster, label, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(Data, Label, test_size=0.125, random_state = 4)
Y_train = np_utils.to_categorical(y_train, 3)
Y_test = np_utils.to_categorical(y_test, 3)






'''    
s = [('red', 1), ('blue', 2), ('red', 3), ('blue', 4), ('red', 1), ('blue', 4)]
d = defaultdict(set)
for k, v in s:
   d[k].add(v)
print(d)


from six import BytesIO
#from PIL import Image

#import glob
#import matplotlib
#import matplotlib.pyplot as plt

def load_image_into_numpy_array(path):
    img_data = open(path,'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height)=image.size
    return np.array(image.getdata()).reshape((im_height, im_width,3)).astype(np.uint8)

image_path='C:/Users/Owner/Desktop/keras_spp/images/org'
images_np = []
for iname in os.listdir(image_path):
    images_np.append(load_image_into_numpy_array(iname))
'''