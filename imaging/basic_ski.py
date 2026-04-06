import skimage as ski
import numpy as np
from tqdm import tqdm

img = ski.io.imread("example.JPG")

ski.io.imshow(img)
ski.io.show()

img_grey = ski.color.rgb2gray(img)
ski.io.imshow(img_grey)
ski.io.show()
