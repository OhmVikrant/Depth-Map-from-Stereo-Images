import numpy as np
import cv2
from matplotlib import pyplot as plt

imgL = cv2.imread('input_images/L1.jpg',0)
imgR = cv2.imread('input_images/R1.jpg',0)

stereo = cv2.StereoBM_create(numDisparities=96, blockSize=15)
disparity = stereo.compute(imgL,imgR)
plt.figure(figsize = (20,10))
plt.imshow(disparity,'disparity')
plt.xticks([])
plt.yticks([])

