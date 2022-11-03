## for data
import cv2
import os
import numpy as np
import pandas as pd

## for plotting
import matplotlib.pyplot as plt

## for object detection
#from imageai import Detection
#from imageai.Detection.Custom import DetectionModelTrainer, CustomObjectDetection

## for deep learning
#from tensorflow.keras import models, layers, applications

## for ocr
#import pytesseract


###############################################################################
#                   IMG ANALYSIS                                              #
###############################################################################


'''
Load a single image with opencv
'''
def load_img(file, ext=['.png','.jpg','.jpeg','.JPG']):
    if file.endswith(tuple(ext)):
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    else:
        print("file extension unknown")




'''
Plot n images in (1 row) x (n columns).
'''
def plot_imgs(imgs, titles=[]):
    ## single image
    if (len(imgs) == 1) or (type(imgs) not in [list,pd.core.series.Series]):
        img = imgs if type(imgs) is not list else imgs[0]
        title = None if len(titles) == 0 else (titles[0] if type(titles) is list else titles)
        fig, ax = plt.subplots(figsize=(5,3))
        fig.suptitle(title, fontsize=15)
        if len(img.shape) > 2:
            plt.imshow(img)
        else:
            plt.imshow(img, cmap=plt.cm.binary)

    ## multiple images
    else:
        fig, ax = plt.subplots(nrows=1, ncols=len(imgs), sharex=False, sharey=False, figsize=(4*len(imgs),10))
        if len(titles) == 1:
            fig.suptitle(titles[0], fontsize=15)
        for i,img in enumerate(imgs):
            ax[i].imshow(img)
            if len(titles) > 1:
                ax[i].set(title=titles[i])
    plt.show()


