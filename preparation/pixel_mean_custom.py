#---------------------------------------------------------
#Calculate mean pixel from dataset
#---------------------------------------------------------

from __future__ import division
import math
import os
import numpy as np
import numpy as np
from PIL import Image
import csv

dataset_path='/home/pedram/PycharmProjects/FCN-Cityscapes/dataset.csv'

with open(dataset_path) as f:
    reader = csv.reader(f)
    csv_file = list(reader)

n=0
mean_sum=np.zeros(3)
print(len(csv_file))
for idx in range(1, len(csv_file)-1):
    name = csv_file[idx][1]
    print(name)
    im = Image.open(name)
    im = im.resize((int(round(im.size[0] / 2)), int(round(im.size[1] / 2))))
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    mean_sum += in_.mean((0,1))
    n += 1

mean=mean_sum/n
print('evaluated ' + str(n) + ' images')
print('mean: ' + str(mean))
