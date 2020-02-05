# This script will process are VIA annotation file and split up the images
# into a training set and a validation set for use with the
# Mask_RCNN neural network model
#
# Run it from this directory as
#   python3 data_setup.py

from random import random
import os
from pathlib import Path
from glob import glob
import json

# Path to our images
image_paths = glob("../images/*")

# Path to JSON annotation file from the VIA tool
annotation_file = '../GunBlock-polygons-500.json'

# Clean up the annotations a little:
# - remove all the VIA app settings and attribute definition bits from 
#   the attribution editor settings
annotations = json.load(open(annotation_file))
cleaned_annotations = {}
for k,v in annotations['_via_img_metadata'].items():
    cleaned_annotations[v['filename']] = v

# Create train and validation directories
Path("gundata/val").mkdir(parents=True, exist_ok=True)
Path("gundata/train").mkdir(parents=True, exist_ok=True)

train_annotations = {}
valid_annotations = {}
# Put 20% of images in validation folder, based on random selection
for img in image_paths:
    # Image goes to Validation folder
    if random()<0.2:
        os.system("cp "+ img + " gundata/val/")
        img = img.split("/")[-1]
        valid_annotations[img] = cleaned_annotations[img]
    else:
        os.system("cp "+ img + " gundata/train/")
        img = img.split("/")[-1]
        train_annotations[img] = cleaned_annotations[img]

# Split the annotation data into the appropriate folders,
# i.e. keep the training annotation data in the train folder
# and the validation annotation data in the val folder
with open('gundata/val/region_data.json', 'w') as fp:
    json.dump(valid_annotations, fp)
with open('gundata/train/region_data.json', 'w') as fp:
    json.dump(train_annotations, fp)
