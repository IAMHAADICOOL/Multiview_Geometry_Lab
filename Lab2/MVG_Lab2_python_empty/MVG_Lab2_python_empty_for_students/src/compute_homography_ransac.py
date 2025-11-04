import numpy as np 
from math import log 

from src.compute_homography import compute_homography
from src.projection_error import projection_error


def compute_homography_ransac(img1, img2, CL1uv, CL2uv, model):
    """Estimate the Homography between two images according to model using the RANSAC algorithm.

    Args:
        CL1uv (numpy.ndarray): Set of points on image #1. Each row represents a 2-D point (u,v). Size: Nx2, with N number of points.
        CL2uv (numpy.ndarray): Set of points on image #2. Each row represents a 2-D point (u,v). Size: Nx2, with N number of points.
        model (str): Type of Homography to estimate. It has to be equal to one of the following strings: 'Translation', 'Similarity', 'Affine', 'Projective'.

    Returns:
        numpy.ndarray: Estimated Homography of Model type. 3x3 matrix.
    """
    
    t = 1e1     # RANSAC threshold
    p = 0.98    # probability that at least one random sample is free from outliers

    num_matches = CL1uv.shape[0]   # Number of matching points 

    outlier_percent = 0.5   # Outliers percentage

    if model == "Translation":
        dof = 2
        num_samples = dof//2
    elif model == "Similarity":
        dof = 4
        num_samples = dof//2    
    elif model == "Affine":
        dof = 6
        num_samples = dof//2
    elif model == "Projective":
        dof = 8
        num_samples = dof//2
    else:
        print("Invalid model")
        return None 
    
    if dof/2 > num_matches:
        print("Not enough matching points..")
        return None 
    
    best_inlier_idxs = None 
    best_consensus_percent = 0.0
    inlier_indxs = []
    # Calculate the number of iterations according to the current number of estimated outliers and the target outlier percentage.
    num_iter = abs((log(1-p)) / (log(1-(1-outlier_percent)**(num_samples)) + 1e-6))

    for i in range(int(num_iter)):

        # Select a number of random point indices 
        
        rand_pt_indices = np.random.choice(num_matches, num_samples, replace=False) 

        # Estimate the Homography with the selected points
        H21 = compute_homography(img1, img2, CL1uv[rand_pt_indices], CL2uv[rand_pt_indices], model)

        if np.any(np.isnan(H21)):
            #If there are problems with H estimation the consensus is valued as null
            consensus_percent = None

        else:
            # Compute the consesus related to estimated H
            # errors = projection_error(H, CL1uv, CL2uv)
            # inlier_indxs = np.where(errors < t)[0]
            error_vec = projection_error(H21, CL2uv, CL1uv)
            # print(f"This mean error {np.mean(error_vec)} for model {model}")
            for i in range(len(error_vec)):
                if error_vec[i] < t:
                    inlier_indxs.append(i)
            # print(f"This is type of inlier_indxs {type(inlier_indxs)}")
            # print(f"This is shape of inlier_indxs {inlier_indxs.shape}")
            consensus_percent = len(inlier_indxs) / num_matches
            # Update best Homography found
            if consensus_percent >= best_consensus_percent:
                best_inlier_idxs = inlier_indxs
                best_consensus_percent = consensus_percent

        # Exit condition
        if consensus_percent >= p:
            break

    # Estimate the Homography with the best inliers 
    inliers1uv = CL1uv[best_inlier_idxs]
    inliers2uv = CL2uv[best_inlier_idxs]
    H_best = compute_homography(img1, img2, CL1uv[best_inlier_idxs], CL2uv[best_inlier_idxs], model)
    # print(f'This is the shape of CL1uv {CL1uv.shape} and type of CL1uv {type(CL1uv)}')
    # print(f'This is the shape of CL1uv[best_inlier_idxs] {CL1uv[best_inlier_idxs].shape} and type of CL1uv {type(CL1uv[best_inlier_idxs])}')
    return H_best, inliers1uv, inliers2uv