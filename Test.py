
import numpy as np
from PIL import Image
import csv


path = "/home/pedram/PycharmProjects/FCN-Cityscapes/dataset.csv"

with open(path) as f:
    reader = csv.reader(f)
    csv_file = list(reader)

idx = 1
c = 0
a = np.zeros((len(csv_file)))
print("max = ", a.shape)
for i in range(len(csv_file)-1):
    r = csv_file[idx]
    im = Image.open(r[2])
    im = im.resize((int(round(im.size[0] / 8)), int(round(im.size[1] / 8))))
    label = np.array(im, dtype=np.uint8)
    label = label[np.newaxis, ...]
    a[c] = np.max(label)
    print(np.min(label))
    print(np.max(label))
    idx += 1
    c += 1

print("max = ", np.max(a))