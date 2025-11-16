import numpy as np
from mvg_compute_epipolar_geom import compute_epipolar_geometry
from mvg_compute_distance_point_line import compute_distance_point_line

def compute_distances_to_epipolar_lines(cam1_pts2d, cam2_pts2d, F):
    """
    Return absolute distance vectors from points to corresponding epipolar lines
    in both images.
    """
    lm1, lm2, lcoef1, lcoef2 = compute_epipolar_geometry(cam1_pts2d, cam2_pts2d, F)
    _, d1 = compute_distance_point_line(lm1, cam1_pts2d)
    _, d2 = compute_distance_point_line(lm2, cam2_pts2d)
    return d1, d2
