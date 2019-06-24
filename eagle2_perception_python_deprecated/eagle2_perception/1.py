import numpy as np

points3D=np.array([[ 0.77822761,  0.72811497,  1.99804894],
 [ 0.77822761, -0.72811497,  1.99804894],
 [ 0.77822761,  0.72811497, -1.99804894],
 [ 0.77822761, -0.72811497, -1.99804894],
 [-0.77822761,  0.72811497, -1.99804894],
 [-0.77822761, -0.72811497, -1.99804894],
 [-0.77822761,  0.72811497,  1.99804894],
 [-0.77822761, -0.72811497,  1.99804894]])
rot_M=np.array([[ 0.99929528,  0., -0.03753585],
 [ 0.,  1.,  0.],
 [ 0.03753585,  0.,  0.99929528]])
cam_to_img=np.array([[7.070912000000e+02,0.000000000000e+00,6.018873000000e+02,0.],
[0.000000000000e+00,7.070912000000e+02,1.831104000000e+02,0.],
[0.000000000000e+00,0.000000000000e+00,1.000000000000e+00,0.]])
box_2D=np.array([[494.],[182.],[518.],[200.]]) 
inds=np.array([[4, 0, 1, 2], [4, 0, 1, 4], [4, 0, 5, 4], [4, 2, 1, 4], [6, 2, 1, 0], [6, 2, 1, 2], [6, 2, 1, 4], [6, 2, 1, 6]])


def points3D_to_2D(points3D,center,rot_M,cam_to_img):
    points2D = []
    for point3D in points3D:
        point3D = point3D.reshape((-1,1))
        point = center + np.dot(rot_M, point3D)
        point = np.append(point, 1)
        point = np.dot(cam_to_img, point)
        point2D = point[:2] / point[2]
        points2D.append(point2D)
    points2D = np.asarray(points2D)

    return points2D

def compute_error(points3D,center,rot_M, cam_to_img,box_2D):
    points2D = points3D_to_2D(points3D, center, rot_M, cam_to_img)
    new_box_2D = np.asarray([np.min(points2D[:,0]),
                  np.max(points2D[:,0]),
                  np.min(points2D[:,1]),
                  np.max(points2D[:,1])]).reshape((-1,1))
    error = np.sum(np.abs(new_box_2D - box_2D))
    return error

fx = cam_to_img[0][0]
fy = cam_to_img[1][1]
u0 = cam_to_img[0][2]
v0 = cam_to_img[1][2]
A = np.array([[fx, 0, float(u0 - box_2D[0])],
              [fx, 0, float(u0 - box_2D[2])],
              [0, fy, float(v0 - box_2D[1])],
              [0, fy, float(v0 - box_2D[3])]])
U, Sigma, VT = np.linalg.svd(A)

center =None
error_min = 1e10

# Ax=b
# Ax => center of bbox point x projected by four different cameras to the same point
# b => rotated (and translated) 3D point projected
# left side 
for ind in inds:
    y = np.zeros((4, 1))
    for i in range(len(ind)):
        RP = np.dot(rot_M, (points3D[ind[i]]).reshape((-1, 1)))
        # XXX: i'm not 100% understand what is Y
        #y[i] = box_2D[i] * cam_to_img[2, 3] - np.dot(W[i], RP) - cam_to_img[i // 2, 3]
        y[i] = -np.dot(A[i], RP)
    print(y)
    print(box_2D)
    result = np.dot(np.dot(np.dot(VT.T, np.linalg.pinv(np.eye(4, 3) * Sigma)), U.T), y)
    break
    error = compute_error(points3D, result, rot_M, cam_to_img, box_2D)
    if error < error_min and result[2,0]>0:
        center = result
        error_min = error
