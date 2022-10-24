import os
import sys
import skimage.io
from mrcnn.config import Config
from datetime import datetime
# Root directory of the project
ROOT_DIR = os.getcwd()
 
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
# sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
# from samples.coco import coco
 
 
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
 
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(MODEL_DIR ,"shapes20200813T1158\\mask_rcnn_shapes_0010.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
    print("cuiwei***********************")
 
# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
 
class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"
 
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
 
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 3 shapes
 
    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024
 
    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  # anchor side in pixels
 
    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE =100
 
    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100
 
    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50
 
#import train_tongue
#class InferenceConfig(coco.CocoConfig):
class InferenceConfig(ShapesConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
 
config = InferenceConfig()
 
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
 
 
# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
 
# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)
 
# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'mango']
# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]

##########################################################辨識移到下面迴圈
#image = skimage.io.imread("./images/A50tra_org.jpg")
 
#a=datetime.now()
## Run detection
#results = model.detect([image], verbose=1)
#b=datetime.now()
## Visualize results
#print("shijian",(b-a).seconds)
#r = results[0]
#visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
#                            class_names, r['scores'])
###########################################################




#####cut the mango

##find biggest 框框
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2

def get_width(xy):
    width = abs(xy[1] - xy[3])
    return width

def get_height(xy):
    height = abs(xy[0] - xy[2])
    return height

def get_area(xy):
    width = get_width(xy)
    height = get_height(xy)
    area = width * height
    return area

def get_biggest_box(xy_list):
    biggest_area = 0
    for i, xy in enumerate(xy_list):
        area = get_area(xy)
        if area > biggest_area:
            biggest_area = area
            biggest_xy = xy
            ix = i
    return biggest_xy, ix

def overlay_box(image, xy): 
    position = (xy[1], xy[0])
    width = get_width(xy)
    height = get_height(xy)
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    rect = patches.Rectangle(position, 
                             width, 
                             height,
                             linewidth=1,
                             edgecolor='r',
                             facecolor='none')
    ax.add_patch(rect)
    plt.show()
    
#big_box, big_ix = get_biggest_box(r['rois'])
#overlay_box(image, big_box)



##crop image   
def crop_box_mask(image, xy):    
    #target = image[xy[0]:xy[2], xy[1]:xy[3], :]
    #img = np.zeros_like(image)
    #img[xy[0]:xy[2], xy[1]:xy[3], :] = target
    crop_img = image[xy[0]:xy[2], xy[1]:xy[3]]#crop image
    #plt.imshow(img)
    return crop_img

#crop_box_mask(image, big_box)



##overlay background
def make_segmentation_mask(image, mask):#mask = array of bool ,in mask=true,out mask=false
    img = image.copy()
    img[:,:,0] *= mask
    img[:,:,1] *= mask
    img[:,:,2] *= mask
    #make a blue mask
    maskblue = mask*(-255)
    maskblue +=255
    maskblue = maskblue.astype('uint8')
    img[:,:,2] +=maskblue
    plt.imshow(img)
    return img

#mask = r['masks'][:,:,big_ix]
#make_segmentation_mask(image, mask)



#######匯入資料

import os
from PIL import Image, ImageOps


path1 = 'C:/Users/Owner/Desktop/pyqT/Images/ForModel'
path2 = 'C:/Users/Owner/Desktop/pyqT/Images/ForUI'
'''
import random
random.seed('foobar')

#preprocessing
imageList = os.listdir(path1)
for file in imageList:
    ###MRCNN 辨識
    image = skimage.io.imread(path1 + '//' + file)
    a=datetime.now()
    # Run detection
    results = model.detect([image], verbose=1)
    b=datetime.now()
    # Visualize results
    print("shijian",(b-a).seconds)
    r = results[0]
    ###find biggest 框框
    big_box, big_ix = get_biggest_box(r['rois'])
    ###cut background
    mask = r['masks'][:,:,big_ix]
    img_cut_background = make_segmentation_mask(image, mask)#img_cut_background is numpy
    img_crop = crop_box_mask(img_cut_background, big_box)#img_crop is numpy
    img = Image.fromarray(img_crop) # numpy to image
    #img = img.resize((224, 224))
    ##save image
    #img.show()
    img.save(path2 +'//' +  file, "JPEG")

import tensorflow as tf
print(tf.__version__)
'''

















##########################################################################################################
#model
##########################################################################################################
###SPP
from keras.engine.topology import Layer
import keras.backend as K


class SpatialPyramidPooling(Layer):
    """Spatial pyramid pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_list: list of int
            List of pooling regions to use. The length of the list is the number of pooling regions,
            each int in the list is the number of regions in that pool. For example [1,2,4] would be 3
            regions with 1, 2x2 and 4x4 max pools, so 21 outputs per feature map
    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.
    # Output shape
        2D tensor with shape:
        `(samples, channels * sum([i * i for i in pool_list])`
    """

    def __init__(self, pool_list, **kwargs):

        self.dim_ordering = K.image_data_format()
        #assert self.data_format in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.pool_list = pool_list

        self.num_outputs_per_channel = sum([i * i for i in pool_list])

        super(SpatialPyramidPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        #if self.dim_ordering == 'th':
            #self.nb_channels = input_shape[1]
        #elif self.dim_ordering == 'tf':
        self.nb_channels = input_shape[3]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.nb_channels * self.num_outputs_per_channel)

    def get_config(self):
        config = {'pool_list': self.pool_list}
        base_config = super(SpatialPyramidPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x, mask=None):

        input_shape = K.shape(x)#take x's row col into input_shape
        #judge using tensorflow or theano 
        #if self.dim_ordering == 'th':
         #   num_rows = input_shape[2]
         #   num_cols = input_shape[3]
      #  elif self.dim_ordering == 'tf':
        num_rows = input_shape[1]
        num_cols = input_shape[2]
        #k.cast() change num_rows to datetype 'float32'
        row_length = [K.cast(num_rows, 'float32') / i for i in self.pool_list]#creat a list:[k.cast()/i1 ,k.cast()/i2,......]
        col_length = [K.cast(num_cols, 'float32') / i for i in self.pool_list]

        outputs = []

        #elif self.dim_ordering == 'tf':
        for pool_num, num_pool_regions in enumerate(self.pool_list):
            for jy in range(num_pool_regions):
                for ix in range(num_pool_regions):
                    x1 = ix * col_length[pool_num]
                    x2 = ix * col_length[pool_num] + col_length[pool_num]
                    y1 = jy * row_length[pool_num]
                    y2 = jy * row_length[pool_num] + row_length[pool_num]

                    x1 = K.cast(K.round(x1), 'int32')
                    x2 = K.cast(K.round(x2), 'int32')
                    y1 = K.cast(K.round(y1), 'int32')
                    y2 = K.cast(K.round(y2), 'int32')

                    new_shape = [input_shape[0], y2 - y1,
                                     x2 - x1, input_shape[3]]

                    x_crop = x[:, y1:y2, x1:x2, :]
                    xm = K.reshape(x_crop, new_shape)
                    pooled_val = K.max(xm, axis=(1, 2))
                    outputs.append(pooled_val)

        #if self.dim_ordering == 'th':
        #    outputs = K.concatenate(outputs)
        #elif self.dim_ordering == 'tf':
            #outputs = K.concatenate(outputs,axis = 1)
        outputs = K.concatenate(outputs)
            #outputs = K.reshape(outputs,(len(self.pool_list),self.num_outputs_per_channel,input_shape[0],input_shape[1]))
            #outputs = K.permute_dimensions(outputs,(3,1,0,2))
            #outputs = K.reshape(outputs,(input_shape[0], self.num_outputs_per_channel * self.nb_channels))

        return outputs
    
#####################################################################################################################
#model_v3    
import numpy as np
#import keras
#from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout#,Flatten
from keras.optimizers import Adam
#from spp import SpatialPyramidPooling
SPPmodel = Sequential()
    
    #First Convolution and Pooling layer
SPPmodel.add(Conv2D(96,(11,11),strides=(4,4),input_shape=(None,None,3),padding='valid',activation='relu'))
SPPmodel.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    
    #Second Convolution and Pooling layer
SPPmodel.add(Conv2D(256,(5,5),strides=(1,1),padding='same',activation='relu'))
SPPmodel.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    
    #Three Convolution layer and Pooling Layer
SPPmodel.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu'))
SPPmodel.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu'))
SPPmodel.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu'))
#model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
SPPmodel.add(SpatialPyramidPooling([4, 2, 1]))
    
    #Fully connection layer
#model.add(Flatten())
SPPmodel.add(Dense(4096,activation='relu'))
SPPmodel.add(Dropout(0.5))
SPPmodel.add(Dense(4096,activation='relu'))
SPPmodel.add(Dropout(0.5))
SPPmodel.add(Dense(1000,activation='relu'))
SPPmodel.add(Dropout(0.5)) 
    
    #Classfication layer
SPPmodel.add(Dense(3,activation='softmax'))

SPPmodel.compile(optimizer=Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),loss = 'categorical_crossentropy',metrics=['accuracy'])

SPP_AlexNet_weights = os.path.join(MODEL_DIR ,"SPP_AlexNet\\SPP_AlexNet_v2_modelweight.h5")
#from keras.models import load_weights
SPPmodel.load_weights(SPP_AlexNet_weights)


###############################################################################################
#UI
###############################################################################################
from PIL import Image
import csv
import numpy as np
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QHBoxLayout, QFileDialog
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon, QPixmap
from PyQt5 import QtGui, QtWidgets, QtCore
from PIL import ImageQt



blue = [225,0,0]
        


class AppDemo(QWidget):
    def __init__(self):
        super(AppDemo,self).__init__()
        self.fileName = ''
        self.filepath = ''
        self.count = 0
        self.imageList = list()
        
        self.setWindowIcon(QIcon('Images/ForUI/A812tra.ico'))
        self.setWindowTitle("Mango Judge")
        
        
        self.checkButton = QtWidgets.QPushButton(self)
        self.checkButton.setGeometry(QtCore.QRect(430, 180, 300, 60))
        self.checkButton.setFont(QFont("Roman times",30,QFont.Bold))
        self.checkButton.setText("Check")
        self.checkButton.clicked.connect(self.Model_run)
        
        self.myButton = QtWidgets.QPushButton(self)
        self.myButton.setGeometry(QtCore.QRect(200, 80, 200, 60))
        self.myButton.setFont(QFont("Roman times",30,QFont.Bold))
        self.myButton.setText("Chose file")
        self.myButton.clicked.connect(self.msg)
        
        self.showpathLabel = QtWidgets.QLabel(self)
        self.showpathLabel.setGeometry(QtCore.QRect(420, 100, 540, 40))
        self.showpathLabel.setFont(QFont("Roman times",9,QFont.Bold))
        self.showpathLabel.setStyleSheet('''
            QLabel{
                color: black;
                background-color : white;
                border: 1px solid  black;
            }
        ''')#point the ImageLabel edge
        
        self.showImgButton = QtWidgets.QPushButton(self)
        self.showImgButton.setGeometry(QtCore.QRect(1390, 120, 120, 60))
        self.showImgButton.setFont(QFont("Roman times",30,QFont.Bold))
        self.showImgButton.setText("Image")
        self.showImgButton.clicked.connect(self.showImg)
        
        self.showImgLabel = QtWidgets.QLabel(self)
        self.showImgLabel.setGeometry(QtCore.QRect(1200, 200, 500, 500))
        self.showImgLabel.setScaledContents (True)#make input image's size = ImageLabel's size
        self.showImgLabel.setFont(QFont("Roman times",9,QFont.Bold))
        self.showImgLabel.setStyleSheet('''
            QLabel{
                color: black;
                background-color : white;
                border: 1px solid  black;
            }
        ''')#point the ImageLabel edge
        
        self.showImgClassLabel = QtWidgets.QLabel(self)
        self.showImgClassLabel.setGeometry(QtCore.QRect(1200, 750, 500, 40))
        self.showImgClassLabel.setFont(QFont("Roman times",15,QFont.Bold))
        self.showImgClassLabel.setText('Image message:')
        self.showImgClassLabel.setStyleSheet('''
            QLabel{
                color: black;
            }
        ''')#point the ImageLabel edge
        
        self.board = QtGui.QStandardItemModel(self)
        self.board.setHorizontalHeaderLabels(['Image Name', 'Image Class'])
        self.tableView = QtWidgets.QTableView(self)
        self.tableView.setModel(self.board)
        self.tableView.horizontalHeader().setStretchLastSection(True)
        self.tableView.setGeometry(QtCore.QRect(200, 250, 760, 700))
        self.tableView.setStyleSheet('''
            QTableView{
                background-color : white;
                border: 1px solid  black;
            }
        ''')#point the ImageLabel edge
    

    def showImg(self):
        if self.count < len(self.imageList):
            self.showImgLabel.setPixmap(QPixmap(self.filepath + '//' + self.imageList[self.count]))
            self.showImgClassLabel.setText(self.imageList[self.count])
            self.count +=1
        else:
            self.count = 0
            self.showImgLabel.setPixmap(QPixmap(self.filepath + '//' + self.imageList[self.count]))
            self.showImgClassLabel.setText(self.imageList[self.count])
    
    def loadCsv(self, fileName):
        with open(fileName, "r") as fileInput:
            for row in csv.reader(fileInput):    
                items = [
                    QtGui.QStandardItem(field)
                    for field in row
                ]
                self.board.appendRow(items)
    

    def msg(self):
        self.board.clear()
        self.board.setHorizontalHeaderLabels(['Image Name', 'Image Class'])
        self.filepath = QFileDialog.getExistingDirectory(self,
                  "選取資料夾",
                  "./")                 #起始路徑
        self.showpathLabel.setText(self.filepath)
        self.count = 0
        self.showImgClassLabel.setText('Image message:')
    
    
    def Model_run(self):
        classlist = list()
        self.imageList = os.listdir(self.filepath)
        for file in self.imageList:
            #self.showImgLabel.setText(file)
            ###MRCNN 辨識
            image = skimage.io.imread(self.filepath + '//' + file)
            # Run detection
            results = model.detect([image], verbose=1)
            # Visualize results
            r = results[0]
            ###find biggest 框框
            big_box, big_ix = get_biggest_box(r['rois'])
            ###cut background
            mask = r['masks'][:,:,big_ix]
            img_cut_background = make_segmentation_mask(image, mask)#img_cut_background is numpy
            img_crop = crop_box_mask(img_cut_background, big_box)#img_crop is numpy
            image = Image.fromarray(img_crop) # numpy to image
            cimg = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)# PIL img to Opcv img
            cimg_re = cv2.resize(cimg, (178, 178))#cv2 resize
            cimg_pad = cv2.copyMakeBorder(cimg_re,1,1,1,1,cv2.BORDER_CONSTANT,value=blue)#img_pad is OpenCV array
            image = Image.fromarray(cv2.cvtColor(cimg_pad,cv2.COLOR_BGR2RGB))#OpenCV array to PIL image
            ##save image
            #img.show()
            #image.save(path2 +'//' +  file, "JPEG")
            immatrix = np.array(image,'f')
            immatrix = np.expand_dims(immatrix, axis=0)
            immatrix /= 255
            
        
            ###prepare model
            result_class = SPPmodel.predict_classes(immatrix)
            
            if result_class[0] == 0:
                Mclass = 'A'
            if result_class[0] == 1:
                Mclass = 'B'
            if result_class[0] == 2:
                Mclass = 'C'
            classlist.append(Mclass)
                
        import pandas as pd
        dict = { "no_header": self.imageList,
                 "no_heade": classlist
        }
        select_df = pd.DataFrame(dict)
        select_df.to_csv('csv_test.txt',sep=',',index=False,header=0)
        self.loadCsv('csv_test.txt')
        


##############Main############################
app = QApplication(sys.argv)
demo = AppDemo()
demo.showMaximized()
sys.exit(app.exec_())
