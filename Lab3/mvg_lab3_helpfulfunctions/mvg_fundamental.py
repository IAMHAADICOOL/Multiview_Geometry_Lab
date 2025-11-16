import numpy as np

def normalize_points(pts):
    """
    Normalize 2D points (2xN) to zero mean and sqrt(2) average distance.
    Returns T, pts_norm (homogeneous 3xN).
    """
    assert pts.shape[0] == 2
    mean = np.mean(pts, axis=1, keepdims=True)
    pts_c = pts - mean
    rms = np.sqrt(np.mean(np.sum(pts_c**2, axis=0)))
    s = np.sqrt(2) / (rms + 1e-12)
    T = np.array([[s, 0, -s*mean[0,0]],
                  [0, s, -s*mean[1,0]],
                  [0, 0, 1]], dtype=float)
    pts_h = np.vstack([pts, np.ones((1, pts.shape[1]))])
    pts_n = T @ pts_h
    return T, pts_n

def enforce_rank2(F):
    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0.0
    return U @ np.diag(S) @ Vt

def eight_point_fundamental(pts1, pts2, enforce_rank2_flag, normalize=True):
    """
    Compute the fundamental matrix using the (normalized) 8-point algorithm.
    Inputs
    ------
    pts1, pts2 : (2,N) arrays with N >= 8 corresponding points.
    Returns
    -------
    F : (3,3) ndarray normalized such that ||F||_F = 1.
    """
    assert pts1.shape[1] >= 8 and pts1.shape == pts2.shape
    if normalize:
        T1, x1 = normalize_points(pts1)
        T2, x2 = normalize_points(pts2)
    else:
        x1 = np.vstack([pts1, np.ones((1, pts1.shape[1]))])
        x2 = np.vstack([pts2, np.ones((1, pts2.shape[1]))])
        T1 = np.eye(3); T2 = np.eye(3)

    X = []
    for i in range(x1.shape[1]):
        u1, v1, w1 = x1[:, i]
        u2, v2, w2 = x2[:, i]
        X.append([u2*u1, u2*v1, u2*w1,
                  v2*u1, v2*v1, v2*w1,
                  w2*u1, w2*v1, w2*w1])
    X = np.asarray(X)
    if normalize:
        condition_number = np.linalg.cond(X)
        print("Condition number of normalized design matrix X:", condition_number)
    else:
        condition_number = np.linalg.cond(X)
        print("Condition number of unnormalized design matrix X:", condition_number)
    # Solve Af = 0 via SVD of X
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    F_n = Vt[-1].reshape(3, 3)


    # Enforce rank-2
    if enforce_rank2_flag:
        print("Enforcing rank 2 on F")
        F_n = enforce_rank2(F_n)

    # Denormalize
    F = T2.T @ F_n @ T1
    # Normalize scale
    F = F / (np.linalg.norm(F) + 1e-12)

    
    return F

def sampson_distance(F, pts1, pts2):
    """
    Sampson approximation of geometric error for correspondences.
    Inputs: pts1, pts2 are (2,N)
    Returns: (N,) distances
    """
    x1 = np.vstack([pts1, np.ones((1, pts1.shape[1]))])
    x2 = np.vstack([pts2, np.ones((1, pts2.shape[1]))])
    Fx1 = F @ x1
    Ftx2 = F.T @ x2
    denom = Fx1[0]**2 + Fx1[1]**2 + Ftx2[0]**2 + Ftx2[1]**2 + 1e-18
    num = np.sum(x2 * (F @ x1), axis=0)**2
    return num / denom
