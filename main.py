import argparse 
import numpy as np 
import matplotlib.pyplot as plt
from kmeans import KMeans

# init parser
parser = argparse.ArgumentParser(description = 'Use Kmeans to cluster color in the image')
# add argument to parser

# create arguments

# init model
model = KMeans()

if __name__  == '__main__':
    print('done!')
