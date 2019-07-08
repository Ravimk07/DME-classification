# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 18:30:22 2019

@author: E75849
"""

import os
from os.path import join 
import cv2
import numpy as np
from matplotlib import pylab, pyplot
import matplotlib.pyplot as plt
import time as time
import scipy.spatial.distance as dist

directory = "E:/sample data/croped data/" 
searchImage= directory + "BRT_18_003NG_TN_105_1-1_2018-02-22 10_18_17.jpg"

index={}
def find(directory):
      for (dirname,dirs,files) in os.walk(directory):
          for filename in files:
              if (filename.endswith(".jpg")):
                  fullpath=join(dirname,filename)
                  index[fullpath]=features(fullpath)
 
      print("total number of photos in this directory %s"%len(index))
      return index
  
def features(imageDirectory):
       img=cv2.imread(imageDirectory)
       histogram=cv2.calcHist([img],[0],None,[256],[0,256])
       Nhistogram=cv2.normalize(histogram,None)
       return Nhistogram.flatten()
   
    
def search(SearchImage,SearchDir):
     histim=features(SearchImage)
     allimages=find(SearchDir)
     match=top(histim,allimages)
     return match

def top(histim,allimages):
      correlation={}
      for (address,value) in allimages.items():
          correlation[address]=cv2.compareHist(histim,value,cv2.HISTCMP_BHATTACHARYYA)
          ranked=sorted(correlation.items() ,key=lambda tup:       float(tup[1]))
      return ranked[0:6]

finalOutput=search(searchImage,directory)

fig=plt.figure()
for imageAdd,Histvalue in finalOutput:
    for i in range(1,2):
        img = cv2.imread(imageAdd)
        fig.add_subplot(i,2,1)
        plt.imshow(img)
        plt.show()
        
        