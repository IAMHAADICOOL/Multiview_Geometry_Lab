import matplotlib.pyplot as plt

def show_epipoles(cam1_ax, cam2_ax, ep1, ep2):
    cam1_ax.scatter([ep1[0]], [ep1[1]], s=30)
    cam2_ax.scatter([ep2[0]], [ep2[1]], s=30)
