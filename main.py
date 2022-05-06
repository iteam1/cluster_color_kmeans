import argparse 
import numpy as np 
import matplotlib.pyplot as plt
from kmeans import KMeans
import time

# init parser
parser = argparse.ArgumentParser(description = 'Use Kmeans to cluster color in the image')
# add argument to parser
parser.add_argument('-i','--img',type = str, help = 'directory to image', required = True)
parser.add_argument('-k','--Knumber',type = int,help = 'The number of color clustered',default = 5)
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

if __name__  == '__main__':
    
    # start counting processing time
    t_start  = time.time()
    
    # get the name of image
    img_name = args.img
    #print(img_name.split('.'))
    origin_name = img_name.split('.')[-2]
    save_name = origin_name +  '_clustered.jpg' 
    # convert image to numpy array
    img_array =  image_to_matrix(args.img)
     
    h = img_array.shape[0]
    w = img_array.shape[1]
    ch = img_array.shape[2]
    
    # reshape the image to flatten
    img_flatted = img_array.reshape(h*w,ch)
    
    # cluster
    cluster_idx,centers,loss = KMeans()(img_flatted,args.Knumber,verbose = False)
        
    # predict
    img_clustered =KMeans().predict(img_flatted,cluster_idx,centers)
    print(type(img_clustered))
    
    # stop counting processing time
    t_done = time.time()
    
    print(f"done! Timing: {t_done - t_start}")
