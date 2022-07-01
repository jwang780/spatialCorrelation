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
# According to this test, the distance() function is computing distances around
# 50 times too small. 
def distance_testing():
    numberOfCases = 100
    maxDiff = 0.01
    centerLat = random.random() * (math.pi - maxDiff) - (math.pi - maxDiff) / 2
    centerLon = random.random() * (math.pi - maxDiff) * 2 + maxDiff
    distance_list = []
    distance_dot_list = []
    for i in range(numberOfCases):
        lat1 = random.random() * maxDiff * 2 - maxDiff + centerLat
        lon1 = random.random() * maxDiff * 2 - maxDiff + centerLon
        origin = lat1, lon1
        lat2 = random.random() * maxDiff * 2 - maxDiff + centerLat
        lon2 = random.random() * maxDiff * 2 - maxDiff + centerLon
        destination = lat2, lon2
        distance_list.append(distance(origin, destination))
        distance_dot_list.append(distance_dot(origin, destination))
    fig, ax = plt.subplots()
    ax.scatter(distance_list, distance_dot_list)
    plt.show()

def kronmult_testing():
    pass

def kronmult2_testing():
    pass

if __name__ == '__main__':
    distance_testing()
