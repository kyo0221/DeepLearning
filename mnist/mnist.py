import sys
import os
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

def image_show(img):
    pil_image = Image.fromarray(np.uint8(img))
    pil_image.show()

(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=False)

image = x_train[0]
label = t_train[0]
print(label)

print(image.shape)
image = image.reshape(28, 28)
print(image.shape)

image_show(image)