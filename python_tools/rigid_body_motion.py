import numpy as np
from scipy.spatial.transform import Rotation


def calc_rigid_body(Q, P, W):
    """
    the R and t rotates and translates the points in P to become like the ones in Q. 
    W is the diagonal weight matrix. 
    They are in the world frame. Rotate in the world frame, then translate. 
    See here: https://igl.ethz.ch/projects/ARAP/svd_rot.pdf
    """

    p_mean = np.sum(P * W.diagonal(), axis=1) / np.sum(W)
    q_mean = np.sum(Q * W.diagonal(), axis=1) / np.sum(W)

    X = P - p_mean[:,np.newaxis]
    Y = Q - q_mean[:,np.newaxis]

    S = X @ W @ Y.T

    U, Sigma, VT = np.linalg.svd(S)
    V = VT.T

    eye_shape = V.shape[1]-1
    O = np.block([
        [np.eye(eye_shape), np.zeros((eye_shape,1))],
        [np.zeros((1,eye_shape)), np.linalg.det(V@U.T)]
    ])
    R = V @ O @ U.T
    t = q_mean - R @ p_mean

    return R, t

def rot_matrix_2d_to_euler(R):
    cos = R[0,0]
    sin = R[1,0]
    angle = np.arctan2(sin, cos)
    return angle

if __name__ == "__main__":
    P1 = np.array([
        [2, 4, 3, 3],
        [0, 0, 1, -1]
    ])
    P2 = np.array([
        [3, 3, 2, 4],
        [2, 4, 3, 3]
    ])
    W = np.eye(4)

    R, t = calc_rigid_body(P2, P1, W)

    assert np.allclose(t, np.array([3, 0]))
    assert rot_matrix_2d_to_euler(R) == np.pi/2

def calc_vel(P1, P2, dt):
    """"
    Calcualte the angular and linear velocity in P2 when points move from P1 to P2. 
    P1 and P2 are 2xN matrices. 
    Around the mean. 
    return vel_lin and vel_ang
    """
    N = P1.shape[1]
    W = np.eye(N)
    R, t = calc_rigid_body(P2, P1, W)
    # This t is after rotating the points in P1 around the world origin. We want the rotation just around the mean. 
    p1_mean = np.mean(P1, axis=1)
    p2_mean = np.mean(P2, axis=1)
    t = p2_mean - p1_mean
    ang = rot_matrix_2d_to_euler(R)

    vel_lin = t/dt
    vel_ang = ang/dt

    return vel_lin, vel_ang

if __name__ == "__main__":
    vel_lin, vel_ang = calc_vel(P1, P2, 1)
    assert np.allclose(vel_lin, np.array([0, 3]))
    assert vel_ang == np.pi/2
