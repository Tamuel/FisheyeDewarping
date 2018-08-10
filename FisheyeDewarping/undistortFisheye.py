import cv2
import numpy as np
from matplotlib import pyplot as plt

calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW

#  We need at least 12 points for fisheye camera calibration

_img_shape = None
# 3d point in real world space
obj_point = [
    [0, 3600, 0],
    [0, 0, 0],
    [-3000, 2550, 50],
    [-3000, 1050, 50]
]
obj_point = np.array(obj_point, np.float32).reshape([1, 4, 3])

# 2d points in image plane.
img_point = [
    [605, 1797],
    [1396, 1750],
    [806, 1816],
    [994, 1826]
]
img_point = np.array(img_point, np.float32).reshape([4, 1, 2])

# 3d point in real world space
obj_points = []
obj_points.append(obj_point)

# 2d points in image plane.
img_points = []
img_points.append(img_point)

fname = '0001.jpg'
img = cv2.imread(fname)
# for idx, p in enumerate(img_points[0]):
#     cv2.circle(
#         img=img,
#         center=tuple(p),
#         radius=10,
#         color=(idx * 50 + 50, idx * 50 + 50, idx * 50 + 50),
#         thickness=-1
#     )
#
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.imshow(img)
# plt.show()
if _img_shape is None:
    _img_shape = img.shape[:2]
else:
    assert _img_shape == img.shape[:2], "All images must share the same size."

N_OK = 1
obj_points = np.array(obj_points).astype('float32')
img_points = np.array(img_points).astype('float32')
# obj_points = obj_points.reshape([1, 1, 4, 3])
# img_points = img_points.reshape([1, 4, 1, 2])

print(np.shape(obj_points), obj_points)
print(np.shape(img_points), img_points)

K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
print(_img_shape, np.shape(rvecs), np.shape(tvecs))
rms, _, _, _, _ = \
    cv2.fisheye.calibrate(
        objectPoints=obj_points,
        imagePoints=img_points,
        image_size=_img_shape,
        K=K,
        D=D,
        rvecs=rvecs,
        tvecs=tvecs,
        flags=calibration_flags,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )
print("Found " + str(N_OK) + " valid images for calibration")
print("DIM=" + str(_img_shape[::-1]))
print("K=np.array(" + str(K.tolist()) + ")")
print("D=np.array(" + str(D.tolist()) + ")")

# import cv2
# assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'
# import numpy as np
# import os
# import glob
# CHECKERBOARD = (6,9)
# subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
# calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
# objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
# objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
# _img_shape = None
# objpoints = [] # 3d point in real world space
# imgpoints = [] # 2d points in image plane.
# images = []
# images.append('fish.png')
# for fname in images:
#     img = cv2.imread(fname)
#     if _img_shape == None:
#         _img_shape = img.shape[:2]
#     else:
#         assert _img_shape == img.shape[:2], "All images must share the same size."
#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     # Find the chess board corners
#     ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
#     # If found, add object points, image points (after refining them)
#     if ret == True:
#         objpoints.append(objp)
#         cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
#         imgpoints.append(corners)
# N_OK = len(objpoints)
# K = np.zeros((3, 3))
# D = np.zeros((4, 1))
# rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
# tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
# objpoints = np.array(objpoints)
# imgpoints = np.array(imgpoints)
# n_points = 20
# objpoints = np.reshape(objpoints[:, :, :n_points, :], [1, 1, n_points, 3])
# imgpoints = np.reshape(imgpoints[:, :n_points, :, :], [1, n_points, 1, 2])
# print(np.shape(objpoints), objpoints)
# print(np.shape(imgpoints), imgpoints)
# rms, _, _, _, _ = \
#     cv2.fisheye.calibrate(
#         objpoints,
#         imgpoints,
#         gray.shape[::-1],
#         K,
#         D,
#         rvecs,
#         tvecs,
#         calibration_flags,
#         (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
#     )
# print("Found " + str(N_OK) + " valid images for calibration")
# print("DIM=" + str(_img_shape[::-1]))
# print("K=np.array(" + str(K.tolist()) + ")")
# print("D=np.array(" + str(D.tolist()) + ")")

# You should replace these 3 lines with the output in calibration step
DIM = _img_shape[::-1]


def undistort(img_path):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imshow("undistorted", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

undistort(fname)
