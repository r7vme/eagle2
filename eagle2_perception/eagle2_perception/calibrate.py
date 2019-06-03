import cv2
import numpy as np

def rot_from_euler_4D(theta):
    R_x = np.array([[ 1, 0,                0,                 0 ],
                    [ 0, np.cos(theta[0]), -np.sin(theta[0]), 0 ],
                    [ 0, np.sin(theta[0]), np.cos(theta[0]),  0 ],
                    [ 0, 0,                0,                 1 ]])

    R_y = np.array([[ np.cos(theta[1]),  0, -np.sin(theta[1]), 0 ],
                    [ 0,                 1, 0               , 0 ],
                    [ np.sin(theta[1]), 0, np.cos(theta[1]), 0 ],
                    [ 0,                 0, 0,                1 ]])

    R_z = np.array([[ np.cos(theta[2]), -np.sin(theta[2]), 0, 0 ],
                    [ np.sin(theta[2]), np.cos(theta[2]),  0, 0 ],
                    [ 0,                0,                 1, 0 ],
                    [ 0,                0,                 0, 1 ]])
    return np.dot(R_z, np.dot(R_y, R_x))


K = np.array([[7.215377000000e+02, 0.000000000000e+00, 6.095593000000e+02],
              [0.000000000000e+00, 7.215377000000e+02, 1.728540000000e+02],
              [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]])
fx = K[0][0]
fy = K[1][1]
u0 = K[0][2]
v0 = K[1][2]

def n(x):
    pass
cv2.namedWindow("top", cv2.WINDOW_NORMAL)
cv2.namedWindow("top", cv2.WINDOW_NORMAL)
cv2.createTrackbar("C0", "top" , 250, 500, n)
cv2.createTrackbar("C1", "top" , 250, 500, n)
cv2.createTrackbar("T0", "top" , 0, 100, n)
cv2.createTrackbar("T1", "top" , 40, 100, n)
cv2.createTrackbar("T2", "top" , 750, 1000, n)
cv2.createTrackbar("pitch", "top" , 0, 30, n)
while True:
    C0 = cv2.getTrackbarPos('C0','top')
    C1 = cv2.getTrackbarPos('C1','top')
    T0 = cv2.getTrackbarPos('T0','top')
    T1 = cv2.getTrackbarPos('T1','top')
    T2 = cv2.getTrackbarPos('T2','top')
    p0 = cv2.getTrackbarPos('pitch','top')
    pitch = np.deg2rad(p0/10)
    # Top down transformation (homography) matrix
    # H = K * (T * (R * A1))
    # Projection 2D->3D
    A1 = np.array([[1, 0, -C0],
                   [0, 1, C1],
                   [0, 0, 0  ],
                   [0, 0, 1  ]])
    R = rot_from_euler_4D([pitch-np.deg2rad(90), 0, 0])
    T = np.array([[1, 0, 0, T0],
                  [0, 1, 0, T1],
                  [0, 0, 1, T2],
                  [0, 0, 0, 1]])
    K2 = np.array([[fx, 0,  u0, 0],
                   [0,  fy, v0, 0],
                   [0,  0,  1 , 0]])
    H = np.dot(K2, np.dot(T, np.dot(R, A1)))
    # Take inverted transformation
    H = np.linalg.inv(H)

    frame = cv2.imread('kitti.png')
    dst = cv2.warpPerspective(frame,H,(500, 500))
    cv2.line(dst,(250, 0),(250, 500),(255,0,0),1)
    cv2.imshow("top", dst)
    if cv2.waitKey(100) & 0xFF == ord('q'): break

print(H)
