import cv2
import numpy as np

# smoothing function
def _smooth(t):
    ret = np.empty_like(t)
    t = np.abs(t)

    mask = (ret >= 1.0)
    ret[mask] = 0.
    ret[~mask] = 1.0-t*t*(3.0-2.0*t)
    return ret

# interpolation func
def _surflet(p, grad):
    return _smooth(p[0])*_smooth(p[1])*np.dot(p,grad)
    
# get the noise function at each point in a grid
def _noise(x, y, size, perm, grads):

    x_cell = int(x)
    y_cell = int(y)

    total = 0.0
    # look at nearby corners
    for x_grid in range(x_cell,x_cell+2):
        for y_grid in range(y_cell,y_cell+2):
            # "random" hash
            h = perm[ (perm[(x_grid+size)%size] + y_grid + size) % size]
            # smooth this gradient in
            pt = np.array([x-x_grid, y-y_grid])
            total += _surflet(pt, grads[h])
    return total

def FractalNoise(w, h, sizes):
    vals = np.zeros((w,h))
    for size in sizes:
       vals += GradientNoise(w,h,size,size)/float(size)
    return vals

def GradientNoise(size, grid_size):
    # number of pixels
    npix = size**2
    # used for hashing
    perm = np.random.permutation(npix)
    # gradients evenly distributed around the circle
    marks = 2.*np.pi*np.arange(0,npix,1)/npix
    grads = np.stack((np.cos(marks), np.sin(marks)), axis=1)

    vals = np.empty((size, size))

    # noise value at each point 
    for x in range(size):
        for y in range(size):
            vals[x,y] = _noise(
                    (x/float(size))*grid_size, 
                    (y/float(size))*grid_size, size, perm, grads)
    return vals
