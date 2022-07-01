from functions import *
import numpy as np
import random
import math
import matplotlib.pyplot as plt

def nearPSD_testing():
    pass

# This function compares the distance numbers computed by the distance() 
# function and the distance_dot() function. 
# The distance_dot() function computes great circle distances by computing the 
# angle between two vectors using a dot product.
def distance_testing():
    distance_list = []
    distance_dot_list = []
    stepSize = 100
    lon1 = 0 #without loss of generality, by rotational symmetry
    for lat1 in range(-90, 89, stepSize):
        for lat2 in range(lat1 + stepSize, 89, stepSize):
            for lon2 in range(lon1 + stepSize, 359, stepSize):
                origin = lat1, lon1
                destination = lat2, lon2
                distance_list.append(distance(origin, destination))
                distance_dot_list.append(distance_dot(origin, destination))
    fig, ax = plt.subplots()
    ax.scatter(distance_list, distance_dot_list)
    #plt.show()
    lat1 = 50
    lat2 = 50
    lon1 = 0
    lon2 = 0
    origin = lat1, lon1
    destination = lat2, lon2
    print(distance(origin, destination))
    print(distance_dot(origin, destination))

def kronmult_testing():
    pass

def kronmult2_testing():
    pass

if __name__ == '__main__':
    distance_testing()
    pass