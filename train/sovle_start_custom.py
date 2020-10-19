caffe_root = '/home/reach/caffe1/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
import sys
sys.path.insert(0, caffe_root + 'python')
fcn_root='/home/reach/caffe1/models/FCN-CustomDataLayer/train/'
sys.path.insert(0, fcn_root)      # path containing cityscapes_layers.py and surgery.py
	#weights for initialization: VGG 16-layer net
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

weights = '/home/reach/caffe1/models/FCN-customDataLayer/fcn8s-heavy-pascal.caffemodel'

# initialize with VGG net
solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from(weights)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

#train
for _ in range(25):
    solver.step(4000)

solver.net.save(final_model_name)

