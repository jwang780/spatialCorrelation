from functions import *
import numpy as np
import random
import math

def nearPSD_testing():
    pass

def distance_testing():
    numberOfCases = 10
    maxDiff = 0.1
    centerLat = random.random() * (math.pi - maxDiff) - (math.pi - maxDiff) / 2
    centerLon = random.random() * (math.pi - maxDiff) * 2 + maxDiff
    for i in range(numberOfCases):
        lat1 = random.random() * maxDiff * 2 - maxDiff + centerLat
        lon1 = random.random() * maxDiff * 2 - maxDiff + centerLon
        origin = lat1, lon1
        lat2 = random.random() * maxDiff * 2 - maxDiff + centerLat
        lon2 = random.random() * maxDiff * 2 - maxDiff + centerLon
        destination = lat2, lon2
        print("distance:" + str(distance(origin, destination)))
        print("distance dot:" + str(distance_dot(origin, destination)))

def kronmult_testing():
    pass

def kronmult2_testing():
    pass

if __name__ == '__main__':
    distance_testing()
