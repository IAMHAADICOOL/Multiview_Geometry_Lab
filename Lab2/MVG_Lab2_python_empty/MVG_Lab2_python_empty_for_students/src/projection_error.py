import numpy as np

def projection_error(H12, CL1uv, CL2uv):
    """Given two list of coordinates (CL1uv and CL2uv) on two images and the
    homography that relates them (H12) this function will compute an error vector
    (errorVec).  This vector will contain the Euclidean distance between each
    point in CL1uv and its corresponding point in CL2uv after applying the
    homography  H12.

    Args:
        H12 (numpy.ndarray): Homography relating image #1 and image #2. 3x3 matrix.
        CL1uv (numpy.ndarray): Set of points on image #1. Each row represents a 2-D point (u,v). Size: Nx2, with N number of points.
        CL2uv (numpy.ndarray): Set of points on image #2. Each row represents a 2-D point (u,v). Size: Nx2, with N number of points.

    Returns:
        numpy.ndarray: Set of L2 norm's calculated between the original and projected points. Size: Nx1, with N number of points.
    """
    number_of_points = CL2uv.shape[0]
    # print(f"This is the shape of CL2uv {CL2uv.shape}")
    ones = np.ones((number_of_points,1)) 
    CL2uv_homogenous = np.concatenate((CL2uv, ones), axis=1).T
    # print(f"This is the shape of CL2uv_homogenous {CL2uv_homogenous.shape}")
    predicted_CL1uv_homogenous = H12 @ CL2uv_homogenous
    # print(f"This is the shape of predicted_CL1uv_homogenous {predicted_CL1uv_homogenous.shape}")

    predicted_CL1uv_cartesian = predicted_CL1uv_homogenous[0:2,]/predicted_CL1uv_homogenous[2:3,]
    # print(f"This is the shape of predicted_CL1uv_cartesian.T {predicted_CL1uv_cartesian.T.shape}")

    error_per_point = np.linalg.norm(predicted_CL1uv_cartesian.T - CL1uv, axis=1)

    # print(f"This is error_per_point of per point {error_per_point}")
    # Project coordinates of second image onto first one
    return error_per_point

    # Calculate the projection error vector
# The following code is for checking the consistency of 
if __name__ == "__main__":
    H12 = np.ones((3,3))
    CL1uv = np.ones((73,2))
    CL2uv = np.ones((73,2))
    projection_error(H12, CL1uv, CL2uv)
