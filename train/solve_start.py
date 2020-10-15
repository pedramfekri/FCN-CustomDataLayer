#---------------------------------------------------------
# Start training a model for the cityscapes dataset, initializing with the VGG 16-layer net

# Adapted from: Fully Convolutional Networks for Semantic Segmentation by Jonathan Long*, Evan Shelhamer*, and Trevor Darrell. CVPR 2015 and PAMI 2016. http://fcn.berkeleyvision.org


#Parameters to be set by the user:

caffe_root = '/home/pedram/caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
import sys
sys.path.insert(0, caffe_root + 'python')
fcn_root='/home/pedram/PycharmProjects/FCN-Cityscapes/train/'
sys.path.insert(0, fcn_root)      # path containing cityscapes_layers.py and surgery.py
# weights = '/export/home/sguist/caffe/models/vgg_16/VGG_ILSVRC_16_layers_conv.caffemodel'	#weights for initialization: VGG 16-layer net
n_steps=50000		# number of training iteration (one iteration = one random image from the training set)
final_model_name='customlayer-fcn32s-2x.caffemodel'

#---------------------------------------------------------


import sys
sys.path.insert(0, fcn_root)

import caffe
import surgery
import os
import setproctitle
setproctitle.setproctitle(os.path.basename(os.getcwd()))


# init
caffe.set_mode_gpu()
caffe.set_device(0)

# initialize with VGG net
solver = caffe.SGDSolver('solver.prototxt')
# solver.net.copy_from(weights)
"""
# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)
#train
solver.step(n_steps)
solver.net.save(final_model_name)
"""