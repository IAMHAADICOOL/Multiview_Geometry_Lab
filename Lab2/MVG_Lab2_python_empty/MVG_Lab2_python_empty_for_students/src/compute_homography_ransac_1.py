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
    
    t = 1e1   # RANSAC threshold
    p = 0.98    # probability that at least one random sample is free from outliers

    num_matches = CL1uv.shape[0]   # Number of matching points 

    outlier_percent = 0.5   # Outliers percentage

    if model == "Translation":
        dof = 2
    elif model == "Similarity":
        dof = 4
    elif model == "Affine":
        dof = 6
    elif model == "Projective":
        dof = 8        
    else:
        print("Invalid model")
        return None 
    
    if dof/2 > num_matches:
        print("Not enough matching points..")
        return None 
    
    inlier_idxs = []
    best_inlier_idxs = []
    best_consensus_percent = 0.0

    # Calculate the number of iterations according to the current number of estimated outliers and the target outlier percentage.
    num_iter = abs((log(1-p)) / (log(1-(1-outlier_percent)**dof) + 1e-6))

    for _ in range(int(num_iter)):

        # Select a number of random point indices  
        random_points_idx = np.random.choice(num_matches, int(dof/2), replace=False)


        # Estimate the Homography with the selected points
        H = compute_homography(img1, img2, CL1uv[random_points_idx], CL2uv[random_points_idx], model=model)


        if np.any(np.isnan(H)):
            #If there are problems with H estimation the consensus is valued as null
            best_consensus_percent = None

        else:
            # Compute the consesus related to estimated H
            print(f'This is the shape of CL1uv {CL1uv.shape} and type of CL1uv {type(CL1uv)}')
            error_vec = projection_error(H, CL2uv, CL1uv)
            for i in range(len(error_vec)):
                if error_vec[i] < t:
                    inlier_idxs.append(i)
            
            # print(len(inlier_idxs), "inliers found for this iteration.")

            consensus_percent = len(inlier_idxs) / num_matches
            if(best_consensus_percent <= consensus_percent):
                best_consensus_percent = consensus_percent
                best_inlier_idxs = inlier_idxs.copy()
                # print("best_inlier_idxs:", len(best_inlier_idxs))

            # Update best Homography found

        # Exit condition
        if best_consensus_percent >= p:
            break

        inlier_idxs.clear()
    # Estimate the Homography with the best inliers
    H = compute_homography(img1, img2, CL1uv[best_inlier_idxs], CL2uv[best_inlier_idxs], model=model)
    print(f'This is the shape of CL1uv {CL1uv.shape} and type of CL1uv {type(CL1uv)}')
    print(f'This is the shape of CL1uv {CL1uv[best_inlier_idxs].shape} and type of CL1uv {type(CL1uv[best_inlier_idxs])}')
    return H, CL1uv[best_inlier_idxs], CL2uv[best_inlier_idxs]
    
        