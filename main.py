import argparse 
import numpy as np 
import matplotlib.pyplot as plt
from kmeans import KMeans

# init parser
parser = argparse.ArgumentParser(description = 'Use Kmeans to cluster color in the image')
# add argument to parser
parser.add_argument('-i','--img',type = str, help = 'directory to image', required = True)
parser.add_argument('-c','--compare',action = 'store_true', help = 'option to compare size by size')
# create arguments
args = parser.parse_args()

def image_to_matrix(image_file,grays = False):
    '''
    Convert .png image to matrix
    Arguments:
        image_file --- (str) image path
        grays --- Boolean
    Return:
        img (numpy array) color or grayscale
    '''
    img = plt.imread(image_file)
    # in case of transparency values
    if len(img.shape) == 3 and img.shape[2] > 3:
        height,width,depth = img.shape
        new_img = np.zeros([height,width,3])
        for r in range(height):
            for c in range(width):
                new_img[r,c,:] = img[r,c,0:3]
        img = np.copy(new_img)
    if grays and len(img.shape) == 3:
        height,width = img.shape[0:2]
        new_img = np.zeros([height,width])
        for r in range(height):
            for c in range(width):
                new_img[r,c] = img[r,c,0]
        img = new_img
    return img

# init model
model = KMeans()

if __name__  == '__main__':
    print('done!')
