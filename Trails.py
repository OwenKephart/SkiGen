import numpy as np
import Curves

def gen_trail_map(size, n_points=10, extrude_method=Curves.extrude_linear):
    kps = gen_trail_points(size, n_points, offset=np.array([0,size/2,0]))
    ts = np.arange(size-1)
    curve = Curves.curve_lwr(kps, ts, plot=True)

    return extrude_method(size, ts, curve)


# generate a set of n keypoints starting from p0
def gen_trail_points(size, n_points, offset=np.array([0,0,0])):
    # array to store generated points
    pts = np.empty((n_points,3)) 
    scale = size/float(n_points-1)

    pts[0] = offset
    # generate each point from the last one
    for i in range(1,n_points):
        if i == 1 or i == n_points-1:
            pts[i] = pts[i-1] + straight_vec(scale)
        else:
            pts[i] = pts[i-1]+trail_vec(scale)

    return pts 

# generates a random vector representing the displacement from previous point
# to the current one
def trail_vec(scale, mu_y=0, sigma_y=1., mu_z=-.25, sigma_z=.15):
    # scale makes it so points are evenly spaced in x direction
    dx = scale
    dy = scale*np.random.normal(mu_y, sigma_y)
    dz = scale*np.random.normal(mu_z, sigma_z)
    v = np.array([dx, dy, dz])
    return v

def straight_vec(scale):
    return np.array([scale, 0, 0])
