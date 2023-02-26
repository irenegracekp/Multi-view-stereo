from weakref import ref
import numpy as np
import cv2
from scipy.fft import dst
from tqdm import tqdm


EPS = 1e-8


def backproject_corners(K, width, height, depth, Rt):
    """
    Backproject 4 corner points in image plane to the imaginary depth plane

    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        width -- width of the image
        heigh -- height of the image
        depth -- depth value of the imaginary plane
    Output:
        points -- 2 x 2 x 3 array of 3D coordinates of backprojected points
    """

    points = np.array((
        (0, 0, 1),
        (width, 0, 1),
        (0, height, 1),
        (width, height, 1),
    ), dtype=np.float32).reshape(2, 2, 3)
    points = points.reshape(4, 3).T


    """ YOUR CODE HERE
    """
    R = Rt[: 3, : 3]
    T = Rt[: 3, -1].reshape((-1, 1))
    points_cam =  ((np.linalg.inv(K))  @ points)
    points_cam = points_cam / points_cam[-1, :]
    points_cam = depth * points_cam
    points_world = (np.linalg.inv(R) @ (points_cam - T)).T
    points = points_world.reshape(2, 2, 3)
    """ END YOUR CODE
    """
    return points

def project_points(K, Rt, points):
    """
    Project 3D points into a calibrated camera.
    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- points_height x points_width x 3 array of 3D points
    Output:
        projections -- points_height x points_width x 2 array of 2D projections
    """
    """ YOUR CODE HERE
    """
    height, width = points.shape[0], points.shape[1]
    points = points.reshape(height * width, 3)
    one_ = np.ones((height * width, 1))
    points = np.hstack((points, one_))
    points_new = K @ Rt @ (points.T)
    points = points_new.T
    points = points / (points[:, -1].reshape((-1 ,1)))
    points = points[:, : -1]
    points = points.reshape(height, width, 2)
    # points = points[:-1].T.reshape((points.shape[0], points.shape[1], 2))


    """ END YOUR CODE
    """
    return points

def warp_neighbor_to_ref(backproject_fn, project_fn, depth, neighbor_rgb, K_ref, Rt_ref, K_neighbor, Rt_neighbor):
    """ 
    Warp the neighbor view into the reference view 
    via the homography induced by the imaginary depth plane at the specified depth

    Make use of the functions you've implemented in the same file (which are passed in as arguments):
    - backproject_corners
    - project_points

    Also make use of the cv2 functions:
    - cv2.findHomography
    - cv2.warpPerspective

    Input:
        backproject_fn -- backproject_corners function
        project_fn -- project_points function
        depth -- scalar value of the depth at the imaginary depth plane
        neighbor_rgb -- height x width x 3 array of neighbor rgb image
        K_ref -- 3 x 3 camera intrinsics calibration matrix of reference view
        Rt_ref -- 3 x 4 camera extrinsics calibration matrix of reference view
        K_neighbor -- 3 x 3 camera intrinsics calibration matrix of neighbor view
        Rt_neighbor -- 3 x 4 camera extrinsics calibration matrix of neighbor view
    Output:
        warped_neighbor -- height x width x 3 array of the warped neighbor RGB image
    """

    height, width = neighbor_rgb.shape[:2]

    """ YOUR CODE HERE
    """

    points = np.array([[0, 0], [width, 0], [0, height], [width, height]], dtype=np.float32)
    # points = points.T
    srcPoints = backproject_fn(K_ref, width, height, depth, Rt_ref)
    dstPoints = project_fn(K_neighbor, Rt_neighbor, srcPoints)

    dstPoints = dstPoints.reshape((dstPoints.shape[0]*dstPoints.shape[1], 2))
    H, mask = cv2.findHomography(points, dstPoints, cv2.RANSAC)
    warped_neighbor = cv2.warpPerspective(neighbor_rgb, np.linalg.inv(H), dsize = (width, height))
    # print(warped_neighbor.shape)
    """ END YOUR CODE
    """
    return warped_neighbor


def zncc_kernel_2D(src, dst):
    """ 
    Compute the cost map between src and dst patchified images via the ZNCC metric
    
    IMPORTANT NOTES:
    - Treat each RGB channel separately but sum the 3 different zncc scores at each pixel

    - When normalizing by the standard deviation, add the provided small epsilon value, 
    EPS which is included in this file, to both sigma_src and sigma_dst to avoid divide-by-zero issues

    Input:
        src -- height x width x K**2 x 3
        dst -- height x width x K**2 x 3
    Output:
        zncc -- height x width array of zncc metric computed at each pixel
    """
    assert src.ndim == 4 and dst.ndim == 4
    assert src.shape[:] == dst.shape[:]

    """ YOUR CODE HERE
    """
    # mean_src = np.mean(src, axis = 2)
    # # print(mean_src.shape)
    # mean_dst = np.mean(dst, axis = 2)
    # zncc_new = np.empty((src.shape[0], src.shape[1], 3))
    # for r in range(3):
    #     for i in range(src.shape[0]):
    #         for j in range(src.shape[1]):
    #             zncc_new[i, j, r] = np.sum((src[i, j, :, r] - mean_src[i, j, r]) * (dst[i, j, :, r] - mean_dst[i, j, r]))/ ((np.std(src, axis = 2)[i, j, r] * np.std(dst, axis =2)[i, j, r]) + EPS)
    # zncc = np.sum(zncc_new, axis = 2)
    #vectorized
    mean_src = np.mean(src, axis = 2)
    mean_dst = np.mean(dst, axis = 2)
    mean_src = mean_src[:, :, np.newaxis, :]
    mean_dst = mean_dst[:, :, np.newaxis, :]
    zncc = np.sum((src - mean_src) * (dst - mean_dst), axis = 2)/(np.std(src, axis = 2) * np.std(dst, axis = 2) + EPS)
    zncc = np.sum(zncc, axis = 2)

    """ END YOUR CODE
    """

    return zncc  # height x width


def backproject(dep_map, K):
    """ 
    Backproject image points to 3D coordinates wrt the camera frame according to the depth map

    Input:
        K -- camera intrinsics calibration matrix
        dep_map -- height x width array of depth values
    Output:
        points -- height x width x 3 array of 3D coordinates of backprojected points
    """
    u, v = np.meshgrid(np.arange(dep_map.shape[1]), np.arange(dep_map.shape[0]))

    """ YOUR CODE HERE
    """
    u_0 = K[0, -1]
    v_0 = K[1, -1]
    f_x = K[0, 0]
    f_y = K[1, 1]

    x_cam = (u - u_0) * dep_map/f_x
    y_cam = (v - v_0) * dep_map/f_y
    z_cam = dep_map
    xyz_cam = np.dstack((x_cam, y_cam, z_cam))
    """ END YOUR CODE
    """
    return xyz_cam

