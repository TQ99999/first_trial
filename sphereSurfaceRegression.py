import numpy as np
import numpy.random as rd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def get_shells(center, radius, N):
    # from http://stats.stackexchange.com/questions/7977
    r = radius
    zs = rd.uniform(-1, 1, N)
    theta = rd.uniform(-np.pi, np.pi, N)
    xs = np.sin(theta)*np.sqrt(1-zs**2)
    ys = np.cos(theta)*np.sqrt(1-zs**2)
    sh0 = np.vstack((xs, ys, zs)).T
    return sh0*r + center


def plot_plt(pts):
    xs, ys, zs = pts.T.tolist()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs)
    plt.show()


def ball_regress(pts):
    ctr = pts.sum(axis=0)/N_dots
    pts_n = pts - ctr
    ri_sqr_n = (pts_n*pts_n).sum(axis=1)
    rhs = 0.5 * (pts_n.T * ri_sqr_n).sum(axis=1)
    rij_sum = np.dot(pts_n.T, pts_n)
    r0_n = np.linalg.solve(rij_sum, rhs)
    r0 = r0_n + ctr
    ra = np.sqrt(ri_sqr_n.sum()/N_dots + np.inner(r0_n, r0_n))
    return r0, ra


def randomize_all(mat):
    return mat + rd.standard_normal(mat.shape)


N_dots = 50
center = np.array([20,2,2])
radius = 200
pts = get_shells(center, radius, N_dots)  # points, N*3
pts = randomize_all(pts)
plot_plt(pts)
center_r, radius_r = ball_regress(pts)
print(center)
print(center_r)
print(radius)
print(radius_r)