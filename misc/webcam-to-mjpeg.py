#!/usr/bin/env python3
import time
import cv2

name="video"+str(time.time()).split(".")[0]+".avi"

# camera
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cam.set(cv2.CAP_PROP_FOCUS, 0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 448)
cam.set(cv2.CAP_PROP_FPS, 10)

# video writer
fourcc=cv2.VideoWriter_fourcc(*"MJPG")
out = cv2.VideoWriter(name,fourcc, 10.0, (800,448))

while (cam.isOpened()):
  ret, img = cam.read()
  if ret!=True:
    break
  out.write(img)
cam.release()
out.release()
