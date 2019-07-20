import cv2
import numpy as np

#H = np.array([[1.38592897e-03,8.66205605e-03,-2.34207693e+00],
#              [0.00000000e+00,1.73241121e-02,-3.99454207e+00],
#              [0.00000000e+00,3.46482242e-05,-5.98908415e-03]])
H = np.array([[ 3.36202259e-03,  1.26885519e-02, -2.34205816e+00],
              [-0.00000000e+00, 2.53771038e-02, -3.99449824e+00],
              [-0.00000000e+00,  5.07542075e-05, -5.98899649e-03]])

img=cv2.imread("000410.png")
ratio=img.shape[0]/img.shape[1]
w=512
h=int((img.shape[0]/img.shape[1])*512)
print(h)
img=cv2.resize(img, (w, h))
img=cv2.copyMakeBorder(img,51,51,0,0,cv2.BORDER_CONSTANT, value=0)
img=img[51:205,:]
print(img.shape)
#img=cv2.warpPerspective(img, H, (500, 500))
#img=img[150:350,150:350]
cv2.namedWindow("show")
cv2.imshow("show",img)
cv2.waitKey(0)
