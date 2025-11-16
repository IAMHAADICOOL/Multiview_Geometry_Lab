import numpy as np

def compute_distance_point_line(lines, points):
    """
    Compute distances from points to corresponding lines.
    Parameters
    ----------
    lines : (3,N) ndarray
        Line parameters for each point (a,b,c) such that a x + b y + c = 0.
    points : (2,N) or (3,N)
        Point coordinates. If (2,N), homogenize with 1s.
    Returns
    -------
    res_d : tuple(sum, mean, std)
    d : (N,) ndarray
        Absolute distances for each point.
    """
    if points.shape[0] == 2:
        points = np.vstack([points, np.ones((1, points.shape[1]))])
    # Distance formula: |l^T p| / sqrt(a^2 + b^2)
    num = np.abs(np.sum(lines * points, axis=0))
    denom = np.sqrt(lines[0]**2 + lines[1]**2 + 1e-18)
    d = num / denom
    return (float(np.sum(d)), float(np.mean(d)), float(np.std(d))), d
