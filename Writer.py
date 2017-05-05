import cv2
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# normalize an image so that all values are between 0 and 1
def normalize(img):
    maxval = np.max(img)
    minval = np.min(img)
    return (img-minval)/(maxval-minval)

# cleans up a given height array and saves it
def save_trail(heights, maxheight, name):
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

# saves a map of all the tree locations in x,y space
def save_trees(tree_list, name):
    f = open(name+".txt", "w")
    for loc in tree_list:
        x = loc[0]
        y = loc[1]
        f.write(str(x) + " " + str(y) + "\n")
    f.close()

# displays a heightmap
def plot(heights): 
    size = heights[0].size
    # generate axes
    X, Y = np.meshgrid(np.arange(0,size,1),np.arange(0,size,1))
    # plot it
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, heights, cmap=cm.coolwarm)

# really just so I don't have to import in other places
def show():
    plt.show()
