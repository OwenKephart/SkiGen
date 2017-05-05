import matplotlib.pyplot as plt
import numpy as np
import cv2
import Noise

PATH_RADIUS_DEFAULT=30

# basic curve extrusion algorithm
def extrude_linear(size, ts, curve, path_radius=PATH_RADIUS_DEFAULT):
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
def extrude_nn(size, ts, curve, spacing=1, path_radius=PATH_RADIUS_DEFAULT):
    # select about npts evenly spaced points
    ts_sel = ts[::spacing]
    X, Y = np.meshgrid(np.arange(0,size,1),np.arange(0,size,1))
    P = np.array([X, Y]).transpose()
    print "Extruding Curve..."
    img = _segment_height(P, curve, ts_sel, size, path_radius)

    return _fill_lwr(img)

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

    mask = min_d < path_radius
    logbump = 1./(1. + np.exp(1.5*(-min_d[mask]+3*path_radius/4)))
    z[mask] = (zinterp[mask] + logbump) - np.min(zinterp)

    return z

def _fill_mountain(zs, iters=400, sigma=10):
    # get non-zero values
    mask = zs > 0
    img = np.zeros_like(zs)
    img[mask] = zs[mask]
    for i in range(iters):
        img = cv2.GaussianBlur(img, (0,0), sigma)
        img[mask] = zs[mask]
    return img

def _fill_lwr(zs):
    tree_pts = _get_tree_pts(zs)
    kps_x, kps_y, kps_z = _get_lwr_kps(zs)

    print "Interpolating using LWR..."
    zs_interp = _lwr_array2d(kps_x, kps_y, kps_z, zs.shape)

    print "Generating Noise"
    noise = 0 #5*Noise.GradientNoise(zs.shape[0], 16)

    # add in some gradient noise
    zs_interp += noise

    mask = (zs == 0)
    zs[mask] = zs_interp[mask]

    return zs, tree_pts

def _get_tree_pts(zs):

    kps = []
    # selected points along the trail
    for x in np.arange(0,zs.shape[0],1):
        nz = np.nonzero(zs[x])
        if len(nz[0]) == 0:
            continue
        x_c = x + (.5 - np.random.random()/2)
        ymin = nz[0][0] + 4*np.random.random()
        ymax = nz[0][-1] - 4*np.random.random()
        kps.append((x_c, ymin))
        kps.append((x_c, ymax))

    return np.array(kps)

def _get_lwr_kps(zs):

    zmax = np.max(zs)
    zmin = np.min(zs)

    kps = []
    # selected points along the trail
    for x in np.arange(0,zs.shape[0],4):
        nz = np.nonzero(zs[x])
        if len(nz[0]) == 0:
            continue
        ymin = nz[0][0]
        ymax = nz[0][-1]
        kps.append((x, ymin, zs[x][ymin]))
        kps.append((x, ymax, zs[x][ymax]))
        
    curve_pts = np.array(kps)

    # corner points should be zero or zmax depending
    kps.append((0, 0, zmax))
    kps.append((0, zs.shape[1]-1, zmax))
    kps.append((zs.shape[0]-1, 0, 0))
    kps.append((zs.shape[0]-1, zs.shape[1]-1, 0))
    kps = np.array(kps)

    # random variation
    NRAND = 60

    # generate some random points that generally follow the slope
    randxs = zs.shape[0]*np.random.random(NRAND)
    randys = zs.shape[1]*np.random.random(NRAND)
    all_x = np.arange(0, zs.shape[0]-1, 20)
    mins = np.repeat(0, all_x.size)
    maxs = np.repeat(zs.shape[1]-1, all_x.size)
    randxs = np.hstack((randxs, all_x, all_x))
    randys = np.hstack((randys, mins, maxs))
    randzs = zmax*(1 - np.random.random(randxs.size)/3. - randxs/zs.shape[0])

    # only grab the ones that will not intersect the trail
    print randxs.astype(np.int32)
    good_inds = (zs[randxs.astype(np.int32), randys.astype(np.int32)] == 0)
    randxs = randxs[good_inds]
    randys = randys[good_inds]
    randzs = randzs[good_inds]

    xs = np.hstack((randxs,kps[:,0]))
    ys = np.hstack((randys,kps[:,1]))
    zs = np.hstack((randzs,kps[:,2]))

    plt.figure()
    plt.scatter(xs, ys)

    return xs, ys, zs



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

def _weighted_regression2d(x, y, z, w):

    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))
    z = z.reshape((-1, 1))
    w = w.reshape((-1, 1))

    d = np.sqrt(w)

    A = d * np.hstack((x, y, np.ones_like(x)))
    b = d * z

    results = np.linalg.lstsq(A, b)

    return results[0]

def _lwr(x, y, sigma, x0):

    # compute weights based on x0
    w = np.exp( -(x - x0)**2 / (2*sigma**2) )

    # compute coefficients with weighted regression
    coeffs = _weighted_regression(x, y, w)

    # compute fit
    return coeffs[0]*x0 + coeffs[1]

def _lwr2d(x0, y0, x, y, z, sigma):

    w = np.exp( -((x-x0)**2 + (y-y0)**2)/ (2*sigma**2))

    coeffs = _weighted_regression2d(x, y, z, w)

    return coeffs[0]*x0 + coeffs[1]*y0 + coeffs[2]

def _lwr_array(kps_a, kps_b, ts, plot, sigma=40):

    # store the interpolated value
    cs = np.empty(ts.size)
    
    # for each t value, generate an interpolated c value based on kps
    #for (i, t) in enumerate(ts):
    for pt in ts:
        cs[pt] = _lwr(kps_a, kps_b, sigma, pt)

    if plot:
        plt.plot(kps_a,kps_b, 'ro')
        plt.plot(ts,cs, 'g-')
        plt.axis('equal')

    return cs

def _lwr_array2d(kps_x, kps_y, kps_z, shape, sigma=40):

    # store the interpolated value
    zs = np.empty(shape)
    
    # for each t value, generate an interpolated c value based on kps
    for x in np.arange(shape[0]):
        for y in np.arange(shape[1]):
            zs[x][y] = _lwr2d(x, y, kps_x, kps_y, kps_z, sigma)

    return zs
