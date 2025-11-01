import numpy as np
from scipy.linalg import lstsq

def compute_homography(img1, img2, CL1uv, CL2uv, model):
    """Estimate the Homography between two images according to model.

    Args:
        CL1uv (numpy.ndarray): Set of points on image #1. Each row represents a 2-D point (u,v). Size: Nx2, with N number of points.
        CL2uv (numpy.ndarray): Set of points on image #2. Each row represents a 2-D point (u,v). Size: Nx2, with N number of points.
        model (str): Type of Homography to estimate. It has to be equal to one of the following strings: 'Translation', 'Similarity', 'Affine', 'Projective'.

    Returns:
        numpy.ndarray: Estimated Homography of Model type. 3x3 matrix.
    """
    # Calculate the matrices for coordinate normalisation  

    # Dummy normalization matrices (identity) - You should replace these with actual normalization
    #TODO Get the loction of the mid point in both the images
    # For image 1
    tx_norm_1 = img1.shape[1] / 2  # width / 2 (center in u/x direction)
    ty_norm_1 = img1.shape[0] / 2  # height / 2 (center in v/y direction)
    scale_1 = max(img1.shape[0], img1.shape[1]) / 2  # scale factor

    T1 = np.array([[1/scale_1, 0, -tx_norm_1/scale_1],
                [0, 1/scale_1, -ty_norm_1/scale_1],
                [0, 0, 1]], dtype=np.float64)

    # For image 2 (same process)
    tx_norm_2 = img2.shape[1] / 2
    ty_norm_2 = img2.shape[0] / 2
    scale_2 = max(img2.shape[0], img2.shape[1]) / 2

    T2 = np.array([[1/scale_2, 0, -tx_norm_2/scale_2],
                [0, 1/scale_2, -ty_norm_2/scale_2],
                [0, 0, 1]], dtype=np.float64)
        
    # T1 = np.matrix([[1, 0, 1],
    #                 [0, 1, 1],
    #                 [0,0,1]])

    
    # T2 = np.matrix([[1, 0, 1],
    #                 [0, 1, 1],
    #                 [0,0,1]])
        
    # Normalise the coordinates

    
    N = CL1uv.shape[0]
    CL1uv_hom = np.hstack([CL1uv, np.ones((N,1))]) 
    CL2uv_hom = np.hstack([CL2uv, np.ones((N,1))]) 
    CL1uv = np.array((T1 @  CL1uv_hom.T ).T)
    CL2uv = np.array((T2 @ CL2uv_hom.T ).T)

    if model == "Translation":
        
        # Create Q matrix 
        Q = np.zeros((2*N,2), dtype=np.float64)
        Q[::2,0] = 1.0    
        Q[1::2,1] = 1.0   

        # Create b vector
        b = np.zeros((2*N), dtype=np.float64)
        b[::2] = CL2uv[:,0] - CL1uv[:,0]   
        b[1::2] = CL2uv[:,1] - CL1uv[:,1] 

        # Calculate translation coefficients and build Homography
        x,_,_,_ = lstsq(Q, b)
        # print(f"This is the shape of x {x.shape} before squeeze")
        # print(f"This is the shape of x.squeeze {np.squeeze(x).shape} before squeeze")
        (tx, ty) = np.squeeze(x)
        A = np.eye(3)
        A[:2,2] = [tx, ty]

    elif model == "Similarity":
        
        # Create Q matrix 
        Q = np.zeros((2*N,4), dtype=np.float64)
        Q[::2,2] = 1.0
        Q[1::2,3] = 1.0 
        for i in range(N):
            u1, v1 = CL1uv[i,0], CL1uv[i,1]
            Q[2*i,0] = u1
            Q[2*i,1] = -v1
            Q[2*i+1,0] = v1
            Q[2*i+1,1] = u1
        # TODO Calculate the Q matrix and the b matrix in a similar way that you computed for
        # Q matrix for projective transformation in the homework given by the professor Rafael
        # Create b vector
        b = np.zeros((2*N), dtype=np.float64)
        b[::2] = CL2uv[:,0]
        b[1::2] = CL2uv[:,1]
        # Calculate similarity coefficients and build Homography
        x,_,_,_ = lstsq(Q, b)
        (a, b, tx, ty) = np.squeeze(x)
        A = np.array([[a, -b, tx],
                      [b,  a, ty],
                      [0,  0, 1]])
        # A = np.linalg.inv(T2) @ A @ T1

    elif model == "Affine":
        
    #     # Create Q matrix 
        Q = np.zeros((2*N,6), dtype=np.float64)
        for i in range(N):
            u1, v1 = CL1uv[i,0], CL1uv[i,1]
            Q[2*i,0] = u1
            Q[2*i,1] = v1
            Q[2*i,2] = 1.0
            Q[2*i+1,3] = u1
            Q[2*i+1,4] = v1
            Q[2*i+1,5] = 1.0
    #     # Create b vector
        b = np.zeros((2*N), dtype=np.float64)
        b[::2] = CL2uv[:,0]
        b[1::2] = CL2uv[:,1]
    #     # Calculate affine coefficients and build Homography
        x,_,_,_ = lstsq(Q, b)
        (a11, a12, tx, a21, a22, ty) = np.squeeze(x)
        A = np.array([[a11, a12, tx],
                      [a21, a22, ty],
                      [0,   0,   1]])
        # A = np.linalg.inv(T2) @ A @ T1

    elif model == "Projective":
        
    #     # Create Q matrix 
        Q = np.zeros((2*N,8), dtype=np.float64)
        for i in range(N):
            u1, v1 = CL1uv[i,0], CL1uv[i,1]
            Q[2*i,0] = u1
            Q[2*i,1] = v1
            Q[2*i,2] = 1.0
            Q[2*i,6] = -u1 * CL2uv[i,0]
            Q[2*i,7] = -v1 * CL2uv[i,0]
            Q[2*i+1,3] = u1
            Q[2*i+1,4] = v1
            Q[2*i+1,5] = 1.0
            Q[2*i+1,6] = -u1 * CL2uv[i,1]
            Q[2*i+1,7] = -v1 * CL2uv[i,1]

    #     # Create b vector
        b = np.zeros((2*N), dtype=np.float64)
        b[::2] = CL2uv[:,0]
        b[1::2] = CL2uv[:,1]

    #     # Calculate projective coefficients and build Homography
        x,_,_,_ = lstsq(Q, b)
        (a11, a12, tx, a21, a22, ty, a31, a32) = np.squeeze(x)
        A = np.array([[a11, a12, tx],
                      [a21, a22, ty],
                      [a31, a32, 1]])
        # A = np.linalg.inv(T2) @ A @ T1
    else:
        print("Invalid model, returning identity homography")
        H12 = np.eye(3)

    # Un-normalise Homography 
    H12 = np.linalg.inv(T2) @ A @ T1
    return H12

if __name__ == "__main__":
    # Dummy data for testing
    img1 = np.zeros((480,640))
    img2 = np.zeros((480,640))
    CL1uv = np.array([[100,150],[200,250],[300,350]])
    CL2uv = np.array([[110,160],[210,260],[310,360]])
    model = "Translation"
    H12 = compute_homography(img1, img2, CL1uv, CL2uv, model)
    print("Computed Homography:\n", H12)