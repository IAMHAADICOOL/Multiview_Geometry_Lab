import numpy as np

def compute_epipolar_geometry(pts1, pts2, F):
    """
    Compute epipolar lines and line coefficients on each image.
    Inputs are 2xN or 3xN; we will homogenize if needed.
    Returns
    -------
    lm1, lm2 : (3,N) ndarray
        Epipolar lines on image 1 and image 2 (ax + by + c = 0).
    l_coef_1, l_coef_2 : (2,N) ndarray
        Slope-intercept form y = m x + q for each line in each image,
        with rows [m; q].
    """
    if pts1.shape[0] == 2:
        pts1 = np.vstack([pts1, np.ones((1, pts1.shape[1]))])
    if pts2.shape[0] == 2:
        pts2 = np.vstack([pts2, np.ones((1, pts2.shape[1]))])
    lm2 = F @ pts1
    lm1 = F.T @ pts2
    # Convert to y = m x + q; beware division by zero on b
    eps = 1e-12
    b1 = np.where(np.abs(lm1[1]) < eps, np.sign(lm1[1]) * eps + eps, lm1[1])
    b2 = np.where(np.abs(lm2[1]) < eps, np.sign(lm2[1]) * eps + eps, lm2[1])
    m1 = -lm1[0] / b1
    q1 = -lm1[2] / b1
    m2 = -lm2[0] / b2
    q2 = -lm2[2] / b2
    l_coef_1 = np.vstack([m1, q1])
    l_coef_2 = np.vstack([m2, q2])
    return lm1, lm2, l_coef_1, l_coef_2

def compute_epipoles_from_epipolar_lines(pts1, pts2, F):
    """
    Method 2: Find epipoles by intersecting epipolar lines
    All epipolar lines cross at the epipole.
    """
    # Get epipolar lines
    lm1, lm2, _, _ = compute_epipolar_geometry(pts1, pts2, F)
    
    # lm1 = [a, b, c]^T represents line: a*u + b*v + c = 0
    # Intersect first two lines in image 1 to find epipole
    # Line 1: lm1[:, 0]
    # Line 2: lm1[:, 1]
    
    # Form matrix and solve: [lm1[:, 0] lm1[:, 1]]^T @ ep1 = 0
    A1 = lm1[:, :2].T  # 2x3 matrix
    U1, S1, Vt1 = np.linalg.svd(A1)
    ep1_lines = Vt1[-1, :2] / (Vt1[-1, 2] + 1e-12)
    
    # Same for image 2
    A2 = lm2[:, :2].T  # 2x3 matrix
    U2, S2, Vt2 = np.linalg.svd(A2)
    ep2_lines = Vt2[-1, :2] / (Vt2[-1, 2] + 1e-12)
    
    return ep1_lines, ep2_lines

def compute_epipoles_from_camera_centers(P1, P2):
    """
    Method 3: Project camera centers to image planes
    Epipole in image 1 = P1 @ camera_center_2
    Epipole in image 2 = P2 @ camera_center_1
    """
    # Camera center is in the null space of projection matrix
    U1, S1, Vt1 = np.linalg.svd(P1)
    C1_homog = Vt1[-1, :]  # Camera 1 center in world coords
    
    U2, S2, Vt2 = np.linalg.svd(P2)
    C2_homog = Vt2[-1, :]  # Camera 2 center in world coords
    
    # Project camera centers
    ep1_homog = P1 @ C2_homog
    ep1_cam = ep1_homog[:2] / (ep1_homog[2] + 1e-12)
    
    ep2_homog = P2 @ C1_homog
    ep2_cam = ep2_homog[:2] / (ep2_homog[2] + 1e-12)
    
    return ep1_cam, ep2_cam

def compute_epipoles_from_F_svd(F):
    """
    Method 1: Extract epipoles from SVD of the final fundamental matrix F
    """
    U_f, S_f, Vt_f = np.linalg.svd(F)
    
    # Epipole in image 1: null space of F (last column of V)
    e1_homog = Vt_f[-1, :]
    ep1_svd = e1_homog[:2] / (e1_homog[2] + 1e-12)
    
    # Epipole in image 2: null space of F^T (last column of U)
    e2_homog = U_f[:, -1]
    ep2_svd = e2_homog[:2] / (e2_homog[2] + 1e-12)
    
    return ep1_svd, ep2_svd