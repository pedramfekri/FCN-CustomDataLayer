# this module needs Python 3.x.
import pandas as pd
import pathlib
from natsort import natsorted

data_dir_train = pathlib.Path("/home/pedram/caffe/models/testing-fcn-for-cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train/aachen/")
data_dir_train_label = pathlib.Path("/home/pedram/caffe/models/testing-fcn-for-cityscapes/gtFine_trainvaltest/gtFine/train/aachen/")

image_count_train = len(list(data_dir_train.glob('*.png')))
image_count_train_label = len(list(data_dir_train_label.glob('*'+'_gtFine_instanceIds.png')))

print(image_count_train_label)
print(image_count_train)

image_train = list(data_dir_train.glob('*.png'))
image_train_label = list(data_dir_train_label.glob('*'+'_gtFine_instanceIds.png'))

image_train = natsorted(image_train)
image_train_label = natsorted(image_train_label)
for i in range(50):
    print('train: ' , image_train[i] , ' test: ', image_train_label[i])

df = {'input': image_train, 'label': image_train_label}
df = pd.DataFrame(data=df)
print(df.head())

df.to_csv('dataset.csv')