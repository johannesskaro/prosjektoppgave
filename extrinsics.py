import numpy as np

PIREN_LAT = 63.43892 #63.4389029083
PIREN_LON = 10.39904 #10.39908278
PIREN_ALT = 39.923 #39.923

def invert_transformation(H):
    R = H[:3,:3]
    T = H[:3,3]
    H_transformed = np.block([ # ROT_FLOOR_TO_CAM.T, -ROT_FLOOR_TO_CAM.T.dot(TRANS_FLOOR_TO_CAM)[:,np.newaxis]
        [R.T, -R.T.dot(T)[:,np.newaxis]],
        [np.zeros((1,3)), np.ones((1,1))]
    ])
    return H_transformed


# Using calibration board. Original extrinsics. 
# Transforms points from the left frame to the right frame
R_BOARD = np.array([
    [ 0.99966116, -0.02563086, 0.00454098],
    [ 0.02569622, 0.99955764, -0.01497426],
    [-0.00415517, 0.01508587, 0.99987757],
])
T_BOARD = np.array([
    -1.92037729,
    -0.01943822,
     0.02775805,
])
H_POINTS_RIGHT_ZED_FROM_LEFT_ZED_BOARD = np.block([
    [R_BOARD, T_BOARD[:,np.newaxis]], 
    [np.zeros((1,3)), np.ones((1,1))]
])

# # Other extrinsics, from only farther away points
# ROT_EXTRINSIC = np.array([[ 0.99940448, -0.0263474 ,  0.02228227],
#        [ 0.02664859,  0.99955599, -0.01332975],
#        [-0.02192117,  0.01391561,  0.99966285]])
# TRANS_EXTRINSIC = np.array([[-1.94891761],
#        [-0.02335371],
#        [ 0.04565223]])




# From tweaker:
H_POINTS_RIGHT_ZED_FROM_LIDAR = np.array([[-0.54803768, -0.83638881,  0.01041464,  0.27549763],
       [-0.100261  ,  0.05332393, -0.99353123,  0.02734051],
       [ 0.83042305, -0.54553672, -0.11308071, -0.03806797],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])

# From tweaker:
H_POINTS_LEFT_ZED_FROM_LEFT_CAM = np.array([[ 0.99996133,  0.00223492,  0.00850553, -0.08994646],
       [-0.00141977,  0.99550815, -0.09466524,  0.32318367],
       [-0.0086789 ,  0.0946495 ,  0.99547283,  0.08694378],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])

# From MA2
ROT_FLOOR_TO_LIDAR = np.array([[-8.27535228e-01,  5.61392452e-01,  4.89505779e-03],
       [ 5.61413072e-01,  8.27516685e-01,  5.61236993e-03],
       [-8.99999879e-04,  7.39258326e-03, -9.99972269e-01]])
TRANS_FLOOR_TO_LIDAR = np.array([-4.1091, -1.1602, -1.015 ])
H_POINTS_FLOOR_FROM_LIDAR = np.block([
    [ROT_FLOOR_TO_LIDAR, TRANS_FLOOR_TO_LIDAR[:,np.newaxis]], 
    [np.zeros((1,3)), np.ones((1,1))]
])
# Using left MA2 cam pose
ROT_FLOOR_TO_LEFT_CAM = np.array([[-3.20510335e-09, -3.20510335e-09, -1.00000000e+00],
       [-1.00000000e+00, -5.55111512e-17,  3.20510335e-09],
       [-5.55111512e-17,  1.00000000e+00, -3.20510335e-09]])
TRANS_FLOOR_TO_LEFT_CAM = np.array([-4.1358,  1.0967, -0.702 ])
H_POINTS_LEFT_CAM_FROM_FLOOR = np.block([
    [ROT_FLOOR_TO_LEFT_CAM.T, -ROT_FLOOR_TO_LEFT_CAM.T.dot(TRANS_FLOOR_TO_LEFT_CAM)[:,np.newaxis]], 
    [np.zeros((1,3)), np.ones((1,1))]
])
H_POINTS_LEFT_ZED_FROM_LIDAR = H_POINTS_LEFT_ZED_FROM_LEFT_CAM @ H_POINTS_LEFT_CAM_FROM_FLOOR @ H_POINTS_FLOOR_FROM_LIDAR

ROT_VESSEL_TO_FLOOR = np.array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]])
TRANS_VESSEL_TO_FLOOR = np.array([ 0. ,  0. , -0.3])
H_POINTS_VESSEL_FROM_FLOOR = np.block([
    [ROT_VESSEL_TO_FLOOR, TRANS_VESSEL_TO_FLOOR[:,np.newaxis]], 
    [np.zeros((1,3)), np.ones((1,1))]
])

# Piren and Piren ENU
ROT_PIREN_TO_PIREN_ENU = np.array([[ 4.89658314e-12,  1.00000000e+00, -2.06823107e-13],
       [ 1.00000000e+00, -4.89658314e-12,  1.01273332e-24],
       [-1.26217745e-29, -2.06823107e-13, -1.00000000e+00]])
TRANS_PIREN_TO_PIREN_ENU = np.array([0., 0., 0.])
H_POINTS_PIREN_FROM_PIREN_ENU = np.block([
    [ROT_PIREN_TO_PIREN_ENU, TRANS_PIREN_TO_PIREN_ENU[:,np.newaxis]], 
    [np.zeros((1,3)), np.ones((1,1))]
])
H_POINTS_PIREN_ENU_FROM_PIREN = invert_transformation(H_POINTS_PIREN_FROM_PIREN_ENU)

# From other extrinsic parameters
H_POINTS_LIDAR_FROM_LEFT_ZED = invert_transformation(H_POINTS_LEFT_ZED_FROM_LIDAR)
H_POINTS_RIGHT_ZED_FROM_LEFT_ZED = H_POINTS_RIGHT_ZED_FROM_LIDAR @ H_POINTS_LIDAR_FROM_LEFT_ZED
R_WIDE = H_POINTS_RIGHT_ZED_FROM_LEFT_ZED[:3,:3]
T_WIDE = H_POINTS_RIGHT_ZED_FROM_LEFT_ZED[:3,3]