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
def plot_imgs(imgs, titles=[], dim=None):
    if dim == "yes":
        ## Info about image
        dimensions = imgs.shape
        height = imgs.shape[0]
        width = imgs.shape[1]
        channels = imgs.shape[2]
        print('The dimension of the input image is : ', dimensions)
        print('The height of the input image is : ', height)
        print('The width of the input image is : ', width)
        print('The Number of Channels in the input image are : ', channels)
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


'''
Plot a single image in RGB
'''
def rgb_imgs(imgs):
    lst = []
    for i in range(3):
        tmp = np.zeros(imgs.shape, dtype = "uint8")
        tmp[:,:,i] = imgs[:,:,i]
        lst.append(tmp)
    plot_imgs(lst,["r","g","b"])


'''
Load a folder of imgs.
'''
def load_imgs(dirpath, ext=['.png','.jpg','.jpeg','.JPG'], plot=False, figsize=(20,13)):
    lst_imgs =[]
    errors = 0
    for file in os.listdir(dirpath):
        try:
            if file.endswith(tuple(ext)):
                img = load_img(file=dirpath+file, ext=ext)
                lst_imgs.append(img)
        except Exception as e:
            print("failed on:", file, "| error:", e)
            errors += 1
            lst_imgs.append(np.nan)
            pass
    if plot is True:
        plot_imgs(lst_imgs[0:5], lst_titles=[], figsize=figsize)
    return lst_imgs,errors

def analysis_imgs(df, target_name):
    #Find freq of every class
    df[target_name].value_counts().sort_index(ascending=False).plot(kind="barh", title="Y", figsize=(5,3)).grid(axis='x')
    plt.show()
    #Find number of classes
    n_classes = df[target_name].nunique()
    print("Nr of classes in target : ", n_classes)
    #Scatterplot of widt and height
    width = [img.shape[0] for img in df["img"]]
    height = [img.shape[1] for img in df["img"]]
    print("Mean of width :", np.mean(width))
    print("Mean of height :", np.mean(height))
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
    #All
    ax[0].scatter(x=width, y=height, color="black")
    ax[0].set(xlabel='width', ylabel="height", title="Size distribution")
    ax[0].grid()
    #Zoom
    ax[1].scatter(x=width, y=height, color="black")
    ax[1].set(xlabel='width', ylabel="height", xlim=[100,700], ylim=[100,700], title="Zoom")
    ax[1].grid()
    plt.show()
