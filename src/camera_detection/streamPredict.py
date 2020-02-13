#! /usr/bin/env python3

# This script opens a webcam stream using its URL.
# It then reads a frame at a time and runs the frame
# through the GunBlock neural net to see if a gun
# is present.

import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage
import cv2
import requests

# Turn off Tensorflow deprecation warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Set Matplotlib backend
# Do this on the Raspberry Pi
#plt.switch_backend('agg')

# Prevent large white screen Matplotlib likes to open
# Problem when using osx backend
matplotlib.use('TkAgg')

# Root directory of the of the Mask_RCNN software code
# GunBlockModel must be a subdir to this directory
# on the R-Pi
#MRCNN_ROOT_DIR = os.path.abspath("/home/pi/ML/Mask_RCNN/")
# On the Macbook
MRCNN_ROOT_DIR = os.path.abspath("/Users/dog/ml_gun_detection/Mask_RCNN/")

GUNMODELDIR = os.path.abspath(MRCNN_ROOT_DIR+"/GunBlockModel")

# Import Mask RCNN python libraries
sys.path.append(MRCNN_ROOT_DIR)
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

# Load our specific gun detection model
# guns.py defines our class, so its name is "guns" for
# import purposes
from GunBlockModel import guns

# Directory to save logs and trained model
# We trained and saved our logs under the GunBlockModel directory
MODEL_DIR = os.path.join(GUNMODELDIR, "logs")

# Path to trained weights
#GUN_WEIGHTS_FILE = GUNMODELDIR+"/logs/gun20200129T0654/mask_rcnn_gun_0030.h5"
GUN_WEIGHTS_FILE = GUNMODELDIR+"/logs/gw_500_30E_50SPE.h5"

# Configurations
# The configuration defined inside our model (see guns.py)
config = guns.gunConfig() 
# The image and region data used to train the model
GUNS_DATA_DIR = GUNMODELDIR+"/gundata"

# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1  # The Raspberry Pi does not have a GPU
    IMAGES_PER_GPU = 1

config = InferenceConfig()

# Device to use for inferencing/prediction
DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0
TEST_MODE = "inference" # As opposed to training, which we already did

# Load validation dataset
dataset = guns.gunDataset()
dataset.load_guns(GUNS_DATA_DIR, "val")

# Must call before using the dataset
dataset.prepare()

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load trained weights
model.load_weights(GUN_WEIGHTS_FILE, by_name=True)

# Load and predict our captured images
CAPTURED_IMG='/Users/dog/ml_gun_detection/videotest/capture.jpg'
RESULT_IMG='/Users/dog/ml_gun_detection/videotest/predicted.jpg'

print("Running preduction framework...")
n_frames = 0
# Create a figure for plotting
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
# Open a URL stream
stream = cv2.VideoCapture('http://192.168.50.1:8080/stream.mjpg')
# Only store 2 frames at a time in the buffer
stream.set(cv2.CAP_PROP_BUFFERSIZE, 2) # Supposedly doesn't work in CV2
# Read a frame from the stream
ret, img = stream.read()
timeCheck = time.time()
# It takes roughly 10 seconds to process a detection phase
futureDelay = 10
if stream.isOpened():
    cv2.namedWindow('Video Stream Monitor', cv2.WINDOW_AUTOSIZE)
    while ret: # ret == True if stream.read() was successful
        if time.time() >= timeCheck:
            ret, img = stream.read()
            n_frames = n_frames + 1
            cv2.imshow('Video Stream Monitor', img)
            key = cv2.waitKey(30) & 0xff
            if key == 27:
                break
            # Run object detection
            results = model.detect([img])
            score = results[0]['scores']
            print(score)
            if score > 0.90:
                # Display results
                r = results[0]
                a = visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'],
                                dataset.class_names, r['scores'], ax=ax, title="Predictions")
                #plt.savefig(RESULT_IMG)
                plt.show()
            timeCheck = timeCheck + futureDelay
        else: # read from buffer but do nothing
            # stream.grab() only returns a ret value
            ret = stream.grab()
else:
    print("Unable to open image stream.")
