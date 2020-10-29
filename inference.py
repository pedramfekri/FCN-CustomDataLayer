caffe_root = '/home/pedram/caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
import sys
import matplotlib.pyplot as plt
sys.path.insert(0, caffe_root + 'python')
import caffe

from PIL import Image
import numpy as np
import os
import setproctitle
import numpy as np
from PIL import Image
import csv


csv_dir = '/home/pedram/PycharmProjects/FCN-CustomDataLayerInf/dataset.csv'
with open(csv_dir) as f:
    reader = csv.reader(f)
    csv_file = list(reader)



n = 50
addr = csv_file[n][1]
im = Image.open(addr)
im = im.resize((int(round(im.size[0] / 3)), int(round(im.size[1] / 3))))
plt.imshow(im)
plt.show()
in_ = np.array(im, dtype=np.float32)
in_ = in_[:, :, ::-1]
in_ -= np.array((72.44887872, 82.96755211, 73.2131611 ))

in_ = in_.transpose((2, 0, 1))
# in_ = in_[np.newaxis, ...]
print(in_.shape)
"""
im = Image.open('image.jpg')
in_ = np.array(im, dtype=np.float32)
in_ = in_[:,:,::-1]
in_ -= np.array((10.00698793,11.66876762,12.67891434))
in_ = in_.transpose((2,0,1))
"""

addr = csv_file[n][2]
print(addr)
im = Image.open(addr)
im = im.resize((int(round(im.size[0] / 3)), int(round(im.size[1] / 3))))
label = np.array(im, dtype=np.uint8)
# label = label[np.newaxis, ...]
print(label.shape)
print(label)
label[label > 10] = 10
"""
model_def = 'deploy.prototxt'
model_weights = 'solver_iter_1000.caffemodel'


net = caffe.Net(model_def,      # defines the structure of the model
               model_weights,  # contains the trained weights
               caffe.TEST)
net.blobs['data'].reshape(1, *in_.shape)
net.blobs['data'].data[...] = in_
# blob = net.blobs['upscore8'].data[...]
# print(type(blob))
# print(blob.shape)
# out =  net.forward_all(**{"data": in_})

net.forward()

# print(type(blob))
# output_prob = output['score'][0]
out = net.blobs['score_c'].data[0].argmax(axis=0)
# for key in out:
#      print(key)

# out = out["prob"][0]
# out = output_prob
# print(type(out))
print(out.shape)

# out = out[0, :, :].reshape(128, 256, 33)
# print(out.shape)
# out = out[0,...]
"""
color = [[128,128,128]
        ,[128,0,0]
        ,[192,192,128]
        ,[255,69,0]
        ,[128,64,128]
        ,[60,40,222]
        ,[128,128,0]
        ,[192,128,128]
        ,[64,64,128]
        ,[64,0,128]
        ,[64,64,0]
        ,[0,128,192]
        ,[0,0,0]
        ,[40,40,40]
        ,[163, 105, 69]
        ,[163, 132, 69]
        ,[138, 163, 69]
        ,[69, 79, 40]
        ,[183, 245, 5]
        ,[193, 232, 162]
        ,[177, 250, 214]
        ,[4, 97, 51]
        ,[184, 212, 198]
        ,[224, 20, 214]
        ,[191, 103, 187]
        ,[235, 209, 233]
        ,[84, 38, 81]
        ,[123, 122, 163]
        ,[67, 65, 135]
        ,[176, 88, 58]
        ,[71, 22, 6]
        ,[133, 100, 38]
        ,[27, 102, 140]
        ,[6, 134, 199]]

# print("here", label[0,0,2])
im = np.zeros([341, 682, 3])

for i in range(im.shape[0]):
    for j in range(im.shape[1]):
        # im[i, j, :] = color[out[i, j]]
        # print(out[:, i, j])
        # print("next")
        im[i, j, :] = color[label[i, j]]
        # print(i, "   ",j)

im = im / 255
# print(out)
plt.imshow(im)
plt.show()

