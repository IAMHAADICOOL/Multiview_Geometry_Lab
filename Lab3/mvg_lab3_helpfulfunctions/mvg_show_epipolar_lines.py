import numpy as np
import matplotlib.pyplot as plt

def show_epipolar_lines(cam1_ax, cam2_ax, lcoef1, lcoef2, margins, color=None):
    """
    Draw epipolar lines on both images.
    margins: ((x0_img1, x1_img1), (x0_img2, x1_img2))
    lcoef: (2,N) with rows [m; q]  -> y = m*x + q
    """
    x0_1, x1_1 = margins[0]
    x0_2, x1_2 = margins[1]

    m1, q1 = lcoef1[0], lcoef1[1]
    m2, q2 = lcoef2[0], lcoef2[1]

    x1_vals = np.vstack([np.full_like(m1, x0_1), np.full_like(m1, x1_1)])
    y1_vals = x1_vals * m1 + q1

    x2_vals = np.vstack([np.full_like(m2, x0_2), np.full_like(m2, x1_2)])
    y2_vals = x2_vals * m2 + q2

    for i in range(m1.size):
        cam1_ax.plot(x1_vals[:, i], y1_vals[:, i], color=color)
    for i in range(m2.size):
        cam2_ax.plot(x2_vals[:, i], y2_vals[:, i], color=color)
