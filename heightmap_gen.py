import cv2
import numpy as np

# smoothing function
def smooth(t):
    t = abs(t)
    if t >= 1.0:
        return 0.0
    return 1.0-t*t*(3.0-2.0*t)

# interpolation func
def surflet(p, grad):
    return smooth(p[0])*smooth(p[1])*np.dot(p,grad)
    
# get the noise at a given point
def noise(x, y, perm, size, grads):
    x_cell = int(x)
    y_cell = int(y)
    # look at nearby corners
    total = 0.
    for x_grid in range(x_cell,x_cell+2):
        for y_grid in range(y_cell,y_cell+2):
            # "random" hash
            h = perm[ (perm[(x_grid+size)%size] + y_grid + size) % size]
            # smooth this gradient in
            #print x-x_grid, y-y_grid, grads[h]
            pt = np.array([x-x_grid, y-y_grid])
            total += surflet(pt, grads[h])
    return total

# normalize an image so that all values are between 0 and 1
def normalize(img):
    maxval = np.max(img)
    minval = np.min(img)
    return (img-minval)/(maxval-minval)

def Gaussian(w, h, height, sigma):

    # get center
    x0 = w//2
    y0 = h//2

    # get grid
    x = np.arange(0, w, 1)
    y = np.arange(0, h, 1)
    xx, yy = np.meshgrid(x,y)

    # gaussian function
    return height*np.exp(-((xx-x0)**2 + (yy-y0)**2)/(2.*sigma**2))

def Slope(w, h, height):
    # get grid
    x = np.arange(0, w, 1)
    y = np.arange(0, h, 1)
    xx, yy = np.meshgrid(x,y)

    return height*(.5*xx)/float(w)
    

def FractalNoise(w, h, sizes):
    vals = np.zeros((w,h))
    for size in sizes:
       vals += GradientNoise(w,h,size,size)/float(size)
    return vals

def GradientNoise(w, h, grid_w, grid_h):

    # used for hashing
    size = grid_w*grid_h
    perm = np.random.permutation(size)
    # gradients evenly distributed around the circle
    marks = 2.*np.pi*np.arange(0,size,1)/size
    grads = np.stack((np.cos(marks), np.sin(marks)), axis=1)

    # store noise values
    vals = np.empty((w,h))

    # noise value at each point 
    for x in range(w):
        for y in range(h):
            vals[x,y] = noise(
                    (x/float(w))*grid_w, 
                    (y/float(h))*grid_h, perm, size, grads)

    return vals

def GenMountains(size, n, maxheight):
    for i in range(n):
        grad = normalize(FractalNoise(size,size,[20,30,40]))
        #gauss = Gaussian(size,size,maxheight,40)
        total = Slope(size, size, maxheight) + (maxheight/100.)*grad
        #total = (maxheight/100.)*grad + gauss
        total = np.uint16(total)
        total[:,-1] = 0
        total[:,0] = 0
        total[-1,:] = 0
        total[0,:] = 0
        print total
        print np.max(total), np.min(total)
        cv2.imwrite("tmap"+str(i)+".tiff", total)

def DiamondSquare(detail, roughness):
    max_loc = 2**detail
    size = max_loc+1
    heights = np.empty((size,size))
    # set the corners to be random heights
    heights[0,0] = 0 #np.random.random()
    heights[0,-1] = 0 #np.random.random()
    heights[-1,0] = 0 #np.random.random()
    heights[-1,-1] = 0 #np.random.random()

    divide(heights, size, max_loc, roughness) 
    return heights

def divide(heights, size, max_loc, roughness):
    half = size/2
    scale = roughness*size
    if (half < 1):
        return
    
    for y in range(half, max_loc, size):
        for x in range(half, max_loc, size):
            square(heights, x, y, half, scale*(2*np.random.random()-1))

    for y in range(0, max_loc, half):
        for x in range((y+half)%size, max_loc, size):
            diamond(heights, x, y, half, scale*(2*np.random.random()-1))

    divide(heights, size/2, max_loc, roughness)

# average edge points and add random offset
def diamond(heights, x, y, size, offset):
    avg = .25*(heights[x,y-size]+ # top
           heights[x+size,y]+ # right
           heights[x,y+size]+ # bottom
           heights[x-size,y]) # left
    heights[x,y] = avg + offset

# average corner points and add random offset
def square(heights, x, y, size, offset):
    avg = .25*(heights[x-size,y-size]+ # top
           heights[x-size,y+size]+ # right
           heights[x+size,y-size]+ # bottom
           heights[x+size,y+size]) # left
    heights[x,y] = avg + offset

# cleans up a given height array and saves it
def save(heights, maxheight, name):
    # make all the heights correct
    total = maxheight*normalize(heights)
    # correct data type
    total = np.uint16(total)
    # zero out the edges
    total[:,-1] = 0
    total[:,0] = 0
    total[-1,:] = 0
    total[0,:] = 0
    cv2.imwrite(name + ".tiff", total)

save(DiamondSquare(9,.1),65530,"B9.1")
save(DiamondSquare(10,.1),65530,"B10.1")
save(DiamondSquare(9,.05),65530,"B9.05")
save(DiamondSquare(10,.05),65530,"B10.05")
#save(DiamondSquare(9,.7),65530,"UNDS1.7")
#save(DiamondSquare(9,.5),65530,"UNDS1.5")
#save(DiamondSquare(9,.3),65530,"UNDS1.3")
#save(DiamondSquare(9,.1),65530,"UNDS1.1")


