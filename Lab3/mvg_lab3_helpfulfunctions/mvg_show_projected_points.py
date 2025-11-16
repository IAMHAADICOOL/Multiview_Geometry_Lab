import numpy as np
import matplotlib.pyplot as plt

def show_projected_points(points_2d, image_size, title_str="Projected points"):
    """
    Display projected points and the image border.
    image_size: (W, H)
    Returns matplotlib figure handle.
    """
    W, H = image_size
    pts = np.asarray(points_2d)
    assert pts.shape[0] == 2, "points_2d must be 2xN"
    # Factor to include all points even if they are outside the image
    if pts.size:
        factors = np.vstack([pts[0] / W, pts[1] / H])
        max_f = max(1.0, np.max(factors)) - 1.0
        min_f = min(0.0, np.min(factors))
    else:
        max_f, min_f = 0.0, 0.0
    factor = 1.25 * max(abs(min_f), max_f)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title_str)
    # Draw image rectangle
    ax.add_patch(plt.Rectangle((0, 0), W, H, fill=False))
    # Plot points
    ax.scatter(pts[0], pts[1], s=30, marker='x')
    ax.scatter(pts[0], pts[1], s=30, marker='o', alpha=0.5)
    # Axes
    ax.set_xlim(-factor*W, (1+factor)*W)
    ax.set_ylim((1+factor)*H, -factor*H)  # axis ij
    ax.set_aspect('equal', adjustable='box')
    return fig
