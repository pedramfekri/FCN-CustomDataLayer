import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import time
import math

#below might be useful link
#https://www.edureka.co/community/30685/this-valueerror-invalid-literal-for-with-base-error-python

#reading from a .csv file
path_to_csv= "/home/reach/caffe1/models/FCN-CustomDataLayer/dataset.csv"
df= pd.read_csv(path_to_csv)

plt.ion()

#closing all plots
plt.close('all')

for i in range (150):

  input_img = df.values[i][1] # retrieve input from first colum in dataframe
  label = df.values[i][2] # # retrieve label from first colum in dataframe 
   
  #read input and label image
  img_input = mpimg.imread(input_img)
  img_label = mpimg.imread(label)
  
  #creating subplots
  f, axarr = plt.subplots(2)
  
  #displaying images to plots
  axarr[1].imshow(img_label)
  axarr[0].imshow(img_input)
   
  plt.draw()
  plt.show()

  time.sleep(2)
 
