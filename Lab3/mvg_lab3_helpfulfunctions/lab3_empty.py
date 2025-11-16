#!/usr/bin/env python3
"""
Lab 3 - Multi-View Geometry - Epipolar Geometry and Stereo
"""

import numpy as np
import matplotlib.pyplot as plt

from mvg_project_point_to_image_plane import project_points_to_image_plane
from mvg_show_projected_points import show_projected_points
from mvg_compute_epipolar_geom import compute_epipolar_geometry, compute_epipoles_from_epipolar_lines, compute_epipoles_from_camera_centers, compute_epipoles_from_F_svd
from mvg_show_epipolar_lines import show_epipolar_lines
from mvg_show_epipoles import show_epipoles
from mvg_fundamental import eight_point_fundamental

def main():
    # Step 1 - Camera 1
    au1, av1, uo1, vo1 = 100, 120, 128, 128
    image_size = (256, 256)

    # Step 2 - Camera 2
    au2, av2, uo2, vo2 = 90, 110, 128, 128
    ax, by, cz = 0.1, np.pi/4, 0.2
    tx, ty, tz = -1000, 190, 230

    # Step 3 - Intrinsics and projection matrices
    K1 = np.array([[au1, 0, uo1], [0, av1, vo1], [0, 0, 1]], float)
    wR1c = np.eye(3)
    wt1c = np.zeros(3)

    # Note: ************** You have to add your own code from here onward ************
    K2 = np.array([[au2, 0, uo2], [0, av2, vo2], [0, 0, 1]], float)
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(ax), -np.sin(ax)],
                    [0, np.sin(ax), np.cos(ax)]])
    R_y = np.array([[np.cos(by), 0, np.sin(by)],
                    [0, 1, 0],
                    [-np.sin(by), 0, np.cos(by)]])
    R_z = np.array([[np.cos(cz), -np.sin(cz), 0],
                    [np.sin(cz), np.cos(cz), 0],
                    [0, 0, 1]])
    wR2c = R_x @ R_y @ R_z #1R2
    wt2c = np.array([[tx],[ty],[tz]]) #1t2

    P1 = K1 @ np.hstack([wR1c.T, -wR1c.T @ wt1c.reshape(3, 1)])
    P2 = K2 @ np.hstack([wR2c.T, -wR2c.T @ wt2c.reshape(3, 1)])
    t12_x = np.array([[0, -tz, ty],
                      [tz, 0, -tx],
                      [-ty, tx, 0]])
    # Step 4 
    # Attention: This is an invented matrix just to have some input for the drawing functions. You have to compute it properly
    F = np.linalg.inv(K2).T @ wR2c.T @ t12_x @ np.linalg.inv(K1)
    F /= F[2, 2]
    print("Step 4: Analytically obtained F:\n", F)

    # Step 5 - 3D points
    V = np.array([[100,300,500,700,900,100,300,500,700,900,100,300,500,700,900,100,300,500,700,900],
                  [-400,-400,-400,-400,-400,-40,-40,-40,-40,-40,40,40,40,40,40,400,400,400,400,400],
                  [2000,3000,4000,2000,3000,4000,2000,3000,4000,2000,3000,4000,2000,3000,4000,2000,3000,4000,2000,3000]], float)

    # Step 6 - Projection and visualization
    cam1_p2d = project_points_to_image_plane(V, P1)
    cam2_p2d = project_points_to_image_plane(V, P2)
    

    # print("This is type of cam1_p2d:", type(cam1_p2d))
    # print("This is shape of cam1_p2d:", cam1_p2d.shape)
    
    F_computed = eight_point_fundamental(cam1_p2d, cam2_p2d, enforce_rank2_flag=False, normalize=False)
    print("Step 7: Computed F using 8-point algorithm:\n", F_computed)
    fro_norm = np.linalg.norm(F - F_computed, 'fro')
    print("Step 8: Frobenius norm between analytical and computed F:", fro_norm)
    
    # Method 1: From SVD of F
    ep1_svd, ep2_svd = compute_epipoles_from_F_svd(F)

    # Method 2: From intersecting epipolar lines
    ep1_lines, ep2_lines = compute_epipoles_from_epipolar_lines(cam1_p2d, cam2_p2d, F)

    # Method 3: From camera centers (if you have P1, P2)
    ep1_cam, ep2_cam = compute_epipoles_from_camera_centers(P1, P2)


    # Compare all three methods
    print("Method 1 (SVD of F):")
    print(f"  ep1: {ep1_svd}")
    print(f"  ep2: {ep2_svd}")
    print("\nMethod 2 (Epipolar lines intersection):")
    print(f"  ep1: {ep1_lines}")
    print(f"  ep2: {ep2_lines}")
    print("\nMethod 3 (Camera centers projection):")
    print(f"  ep1: {ep1_cam}")
    print(f"  ep2: {ep2_cam}")
    
    cam1_fig = show_projected_points(cam1_p2d, image_size, "Projected points on image plane 1")
    cam2_fig = show_projected_points(cam2_p2d, image_size, "Projected points on image plane 2")
    ax1, ax2 = cam1_fig.axes[0], cam2_fig.axes[0]
    # For image 1
    ax1.set_xlim(min(0, ep1_svd[0] - 50), max(image_size[0], ep1_svd[0] + 50))
    ax1.set_ylim(min(0, ep1_svd[1] - 50), max(image_size[1], ep1_svd[1] + 50))

    # For image 2
    ax2.set_xlim(min(0, ep2_svd[0] - 50), max(image_size[0], ep2_svd[0] + 50))
    ax2.set_ylim(min(0, ep2_svd[1] - 50), max(image_size[1], ep2_svd[1] + 50))
    _, _, c1, c2 = compute_epipolar_geometry(cam1_p2d, cam2_p2d, F)
    margins = ((-400, 300), (1, 400))
    show_epipolar_lines(ax1, ax2, c1, c2, margins, color='b')
    # ep1, ep2 = np.array([-300,200,1]), np.array([200,50,1]) # These are dummy values for the epipoles just for illustrating. You have to compute them
    show_epipoles(ax1, ax2, ep1_svd, ep2_svd)
    print("Epipole in image 1 (left):", ep1_svd)
    print("Epipole in image 2 (right):", ep2_svd)
    plt.show()
    exit()
    

    

    
if __name__ == "__main__":
    main()
