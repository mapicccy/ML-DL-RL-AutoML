import numpy as np
import os
from scipy import misc
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('data/mnist_data/', one_hot=True)
save_dir = 'data/mnist_data/'
if os.path.exists(save_dir) is False:
    os.mkdir(save_dir)

print(mnist.train.images.shape, mnist.train.labels.shape)

for i in range(1):
    image_raw = mnist.train.images[i, :]
    image = np.reshape(image_raw, [28, 28])
    filename = save_dir + 'image_minist_%d.jpg' % i
    misc.toimage(image, cmin=.0, cmax=1.).save(filename)
