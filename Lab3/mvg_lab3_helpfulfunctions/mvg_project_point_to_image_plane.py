import numpy as np

def project_points_to_image_plane(points_3d, P):
    """
    Project 3D points to the image plane using projection matrix P.
    Parameters
    ----------
    points_3d : (3,N) or (4,N) ndarray
        3D points in world (non-homogeneous or homogeneous). If (3,N), a row of ones is appended.
    P : (3,4) ndarray
        Projection matrix.
    Returns
    -------
    points_2d : (2,N) ndarray
        Image coordinates (non-homogeneous).
    """
    if points_3d.shape[0] == 3:
        points_3d = np.vstack([points_3d, np.ones((1, points_3d.shape[1]))])
    p2d_h = P @ points_3d
    # Avoid divide-by-zero
    z = np.where(np.abs(p2d_h[2]) < 1e-12, 1e-12, p2d_h[2])
    points_2d = np.vstack([p2d_h[0]/z, p2d_h[1]/z])
    return points_2d
