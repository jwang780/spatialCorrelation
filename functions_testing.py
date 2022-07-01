from functions import *
import numpy as np
import random
import math

def nearPSD_testing():
    pass

def distance_testing():
    numberOfCases = 10
    for i in range(numberOfCases):
        lat1 = random.random() * math.pi - math.pi / 2
        lon1 = random.random() * math.pi * 2
        origin = lat1, lon1
        lat2 = random.random() * math.pi - math.pi / 2
        lon2 = random.random() * math.pi * 2
        destination = lat2, lon2
        print("distance:" + str(distance(origin, destination)))
        print("distance dot:" + str(distance_dot(origin, destination)))

def kronmult_testing():
    pass

def kronmult2_testing():
    pass

if __name__ == '__main__':
    distance_testing()
