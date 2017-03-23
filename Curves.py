import matplotlib.pyplot as plt
import numpy as np
import cv2

# basic curve extrusion algorithm
def extrude_linear(size, ts, curve, path_radius=30):
    xs, ys, zs = curve
    img = np.zeros((size,size))
    for t in ts:
        x = xs[t]
        y = ys[t]
        z = zs[t]
        img[x][y-path_radius:y+path_radius] = z
    return img

# diffusion-based extrusion algorithm
def extrude_diffusion(size, ts, curve, iters=100, sigma=10):
    xs, ys, zs = curve
    img = np.zeros((size,size))
    for i in range(iters):
        # fill in curve
        for t in ts:
            x = xs[t]
            y = ys[t]
            z = zs[t]
            img[x][y] = z
        # blur it
        img = cv2.GaussianBlur(img, (0,0), sigma)
    return img

# nearest-neighbor based extrusion algorithm
def extrude_nn(size, ts, curve, spacing=16, path_radius=30):
    # select about npts evenly spaced points
    ts_sel = ts[::spacing]
    X, Y = np.meshgrid(np.arange(0,size,1),np.arange(0,size,1))
    P = np.array([X, Y]).transpose()
    img = _segment_height(P, curve, ts_sel, size, path_radius)

    #return _fill_mountain(img)
    return img

# generate a curve from the keypoints using lwr
def curve_lwr(kps, ts, plot=False):
    xs = ts
    ys = _lwr_array(kps[:,0], kps[:,1], ts, plot)
    zs = _lwr_array(kps[:,0], kps[:,2], ts, plot)
    return np.array([xs, ys, zs])

# ************************* USED FOR NNE *************************

def _segment_height(p, curve, ts_sel, size, path_radius):

    # init heightmap
    z = np.zeros((size,size))

    # larger than maximum possible
    min_d = np.tile(-1., (size,size))
    min_idx = np.tile(0, (size,size))
    min_alpha = np.tile(0.0, (size,size))

    # find closest line segment in curve
    for i in range(0,ts_sel.size-1):
        # get line segment points
        pa = curve[0:2,ts_sel[i]]
        pb = curve[0:2,ts_sel[i+1]]

        # get pct of segment
        alpha = np.dot(p-pa,pb-pa)/float(np.dot(pb-pa,pb-pa))
        alpha = np.clip(alpha,0,1)

        # closest point on the line pb-pa
        # the einsum thing just kinda works for multiplying weirdly shaped
        # things
        pc = pa + np.einsum('ij,k->ijk',alpha,pb-pa)

        d = np.linalg.norm(pc-p, axis=2)

        # update values
        mask = (d < min_d) | (min_d < 0)
        min_d[mask] = d[mask]
        min_alpha[mask] = alpha[mask]
        min_idx[mask] = i

    tcs = ts_sel[min_idx]
    za = curve[2, ts_sel[min_idx]]
    zb = curve[2, ts_sel[min_idx+1]]
    zinterp = za + min_alpha*(zb-za)

    #mask = min_d > -1
    mask = min_d < path_radius
    logbump = 4./(1. + np.exp((-min_d[mask]+path_radius/2))/10.)
    z[mask] = (zinterp[mask] + logbump) - np.min(zinterp)

    bwmap = ((min_idx % 3)*127).astype(np.uint8)
    bwmap[~mask] = 64

    cv2.imwrite('bwmap.png', bwmap)

    return z

def _fill_mountain(z, iters=400, sigma=10):
    # get non-zero values
    mask = z > 0
    img = np.zeros_like(z)
    img[mask] = z[mask]
    for i in range(iters):
        img = cv2.GaussianBlur(img, (0,0), sigma)
        img[mask] = z[mask]
    return img


# ************************* USED FOR LWR *************************

def _weighted_regression(x, y, w):
    # x, y, and w all have length n
    # let's make them all column vectors
    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))
    w = w.reshape((-1, 1))
    
    d = np.sqrt(w)
    
    A = d * np.hstack( (x, np.ones_like(x)) )
    b = d * y

    results = np.linalg.lstsq(A, b)
    return results[0] # solution

def _lwr(x, y, sigma, x0):

    # compute weights based on x0
    w = np.exp( -(x - x0)**2 / (2*sigma**2) )

    # compute coefficients with weighted regression
    coeffs = _weighted_regression(x, y, w)

    # compute fit
    return coeffs[0]*x0 + coeffs[1]

def _lwr_array(kps_a, kps_b, ts, plot, sigma=25):

    # store the interpolated value
    cs = np.empty(ts.size)
    
    # for each t value, generate an interpolated c value based on kps
    for (i, t) in enumerate(ts):
        cs[i] = _lwr(kps_a, kps_b, sigma, t)

    if plot:
        plt.plot(kps_a,kps_b, 'ro')
        plt.plot(ts,cs, 'g-')
        plt.axis('equal')

    return cs
