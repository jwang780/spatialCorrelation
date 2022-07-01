import os, math, sys
import numpy as np
import scipy as sp
import pandas as pd
import scipy.stats as sps
import scipy.spatial.distance as spsd
from sys import stdin

# Input: A is a square matrix
# Output: out is a positive semidefinite matrix. 
# 
# I think that this is an attempt to create a positive definite matrix with the
# same eigen vectors as A. 
def nearPSD(A, epsilon = 0):
    n = A.shape[0]
    eigval, eigvec = np.linalg.eig(A)
    val = np.matrix(np.maximum(eigval, epsilon))
    vec = np.matrix(eigvec)
    T = 1 / (np.multiply(vec, vec) * val.T)
    T = np.matrix(np.sqrt(np.diag(np.array(T).reshape((n)))))
    B = T * vec * np.diag(np.array(np.sqrt(val)).reshape((n)))
    out = B * B.T
    return (np.array(out))

# Input: A is a square matrix
# epsilon is a positive number.
# Output: A is a square matrix that is positive definite. 
# 
# Some multiple of the identity matrix is added to make all eigenvalues greater
# than or equal to epsilon. (i.e., the output is a positive definite matrix)
def nearPSD_eigen(A, epsilon=1e-4):
    eigen = sp.linalg.eigvalsh(A)
    if np.any(eigen < 0):
        d = min(eigen)
        A += (abs(d) + epsilon) * np.identity(len(A))
    return A

radius = 6371  # km

# Input: origin and destination are two pairs of coordinates
# Output: d the distance between the two coordinates. 
# 
# This is an approximation of distance on the surface of a sphere. 
# c should be angle between the lines from the origin and destination to the
# center of the sphere. 
# This angle is probably easiest to compute using dot products (inner products)
# instead of all this trigonometry. 
def distance(origin, destination):
    lat1, lon1 = origin
    lat2, lon2 = destination
    global radius
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    sin_lat = math.sin(dlat / 2)
    sin_lon = math.sin(dlon / 2)
    a = sin_lat * sin_lat + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * sin_lon * sin_lon
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c
    return d

# Input: origin and destination are two pairs of coordinates
# Output: d is the distance between the two coordinates. 
def distance_dot(origin, destination):  
    lat1, lon1 = origin
    lat2, lon2 = destination
    # For computing the angle between the vectors, the length of the vectors is 
    # not important. 
    x1, y1, z1 = polar_to_cartesian(1, lon1, math.pi - lat1)
    x2, y2, z2 = polar_to_cartesian(1, lon2, math.pi - lat2)
    dot_product = x1 * x2 + y1 * y2 + z1 * z2
    angle = math.acos(dot_product) # magnitude of both vector is 1
    global radius
    d = radius * angle # arclength
    return d

# Input: A location in spherical coordinates
# Output: A location in cartesian coordinates
def polar_to_cartesian(r, theta, phi):
    x = r * math.sin(phi) * math.cos(theta)
    y = r * math.sin(phi) * math.sin(theta)
    z = r * math.cos(phi)
    return x, y, z

# Inputs: origin is a list of locations
# destination is a list of locations
# N_orig is the size of the input origin
# N_dest is the size of the input destination
# Output: distance is a matrix of distances between locations in origin and 
# locations in destination
# 
# Instead of calling the distance() function, the code in the distance function
# is rewritten here, in order to compute cos_lat2 only once per iterations
# This is at best a 25% decrease in runtime, as math.sin and math.cos functions
# are used 3 more times per iteration anyways. 
def distance2(origin, destination, N_orig, N_dest):
    global radius
    distance = np.zeros([N_orig, N_dest])
    for j in xrange(N_dest):
        lat2, lon2 = destination[j]
        cos_lat2 = math.cos(math.radians(lat2))
        for i in xrange(N_orig):
            lat1, lon1 = origin[i]
            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)
            sin_lat = math.sin(dlat / 2)
            sin_lon = math.sin(dlon / 2)
            a = sin_lat * sin_lat + math.cos(math.radians(lat1)) * cos_lat2 * sin_lon * sin_lon
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            distance[i, j] = radius * c

    return distance

# Inputs: coords is a list of locations
# Outputs: A symmetric distance matrix
# 
# distance_compact is a vector/1-dim array, squareform transforms to a square
# symmetric distance matrix. 
def distance_square(coords):
    N_loc = len(coords)
    distance_compact = []
    for i in xrange(N_loc):
        for j in xrange(i + 1, N_loc):
            distance_compact.append(distance(coords[i], coords[j]))

    return spsd.squareform(np.array(distance_compact))

def kronmult2(Q, X, mB, nX, B_type):
    mA, nA = Q[0].shape
    nB = mB
    Y = np.empty((mA * mB, nX))
    if B_type == 1:  # if identity matrix
        for i in xrange(0, nX):
            x = X[:, i].reshape(nA, nB)
            Y[:, i] = np.dot(Q[0], x).reshape(1, -1)
    elif B_type == 2:  # if symmetric
        for i in xrange(0, nX):
            x = X[:, i].reshape(nA, nB)
            Y[:, i] = np.dot(np.dot(Q[0], x), Q[1]).reshape(1, -1)
    else:
        for i in xrange(0, nX):
            x = X[:, i].reshape(nA, nB)
            Y[:, i] = np.dot(np.dot(Q[0], x), Q[1].T).reshape(1, -1)
    return Y

# This function is unused. 
def kronmult(Q, X):
    N = len(Q)
    n = [0] * N
    nright = 1
    nleft = 1
    for i in xrange(0, N - 1):
        n[i] = Q[i].shape[0]
        nleft = nleft * n[i]
    n[N - 1] = Q[N - 1].shape[0]
    for i in xrange(N - 1, -1, -1):
        base = 0
        jump = n[i] * nright
        for k in range(0, nleft):
            for j in range(0, nright):
                index1 = base + j
                index2 = index1 + nright * (n[i] - 1) + 1
                X[index1:index2:nright] = np.dot(Q[i], X[index1:index2:nright])
            base = base + jump
        nleft = nleft / n[max(i - 1, 0)]
        nright = nright * n[i]

def corr_sampling(corr_spatial_cholesky, corr_vul_cholesky, N_event, N_loc):
    sample_normal = sps.norm.rvs(size=N_loc * N_event).reshape(N_loc, -1)
    return kronmult2([corr_spatial_cholesky, corr_vul_cholesky], sample_normal, 1, N_event, 0)

def SD_kriging(B, scale_B, N_sub):
    N_loc = int(N_sub + 0.0)
    sd_kriging = np.zeros((N_loc, 1))
    for i in xrange(N_loc):
        sd_kriging[i] = (sum(B[:, i] ** 2)) ** 0.5 * scale_B[i]
    sd_kriging = np.tile(sd_kriging, (1, 1))
    sd_kriging = sd_kriging.reshape((-1, 1))
    return sd_kriging

def kriging(sample_normal_loc, A, sd_kriging, N_sub, N_event):
    sample_normal_kriging = kronmult2([A], sample_normal_loc, 1, N_event, 1)
    sample_normal_kriging = sample_normal_kriging / np.tile(sd_kriging, (1, N_event))
    return sample_normal_kriging

def spatial_sampling(occID_set, eventID, locID, spatial_decay, N_MaxSamplingSize=1000):
    occIDList = list(occID_set)
    locIDList = list(locID)
    N_loc = len(locIDList)
    N_Sampling = min(N_MaxSamplingSize, N_loc)
    N_Kriging = max(0, N_loc - N_MaxSamplingSize)
    dfLocIDLatLong = pd.read_csv("JPEQ_GridID_LatLong.txt", sep='\t')
    dfLocIDs = pd.DataFrame(locIDList, columns=['gridID'])
    dfLoc = pd.merge(dfLocIDs, dfLocIDLatLong, left_on='gridID', right_on='grid.id', how='left')[['gridID', 'Latitude', 'Longitude']]

    while True:
        flag2 = True
        counter = 0
        print("A new run of eventID %s." % str(eventID))
        counter1 = 0
        while True:
            counter1 += 1
            print("The %i th Cholesky run for eventID %s." %(counter1, str(eventID)))
            dfLoc = dfLoc.reindex(np.random.permutation(dfLoc.index))
            lat = np.array(dfLoc['Latitude'])
            lon = np.array(dfLoc['Longitude'])
            coords = np.array(zip(lat[:N_MaxSamplingSize], lon[:N_MaxSamplingSize]))
            Do = distance_square(coords)
            corr_spatial = np.exp(spatial_decay * Do)
            try:
                # corr_spatial_cholesky = sp.linalg.cholesky(corr_spatial, lower=True)
                corr_spatial2 = nearPSD_eigen(corr_spatial)
                corr_spatial_cholesky = sp.linalg.cholesky(corr_spatial2, lower=True)
                break
            except:
                # This should never happen. 
                # If this block is executed, this means that the matrix 
                # corr_spatial has non-real eigenvalues
                # which means that corr_spatial is not a symmetric real matrix
                try:
                    corr_spatial2 = nearPSD(corr_spatial)
                    corr_spatial_cholesky = sp.linalg.cholesky(corr_spatial2, lower=True)
                    break
                except:
                    pass

        locIDs = np.array(dfLoc['gridID'])
        Sigma_vul = np.array([1])
        corr_vul_cholesky = np.array([1])

        if N_Kriging > 0:
            pcoords = np.array(zip(lat[N_MaxSamplingSize:], lon[N_MaxSamplingSize:]))
            Dop = distance2(coords, pcoords, N_MaxSamplingSize, N_Kriging)
            corr_spatial_cross = np.exp(spatial_decay * Dop)
            corr_spatial_cholesky_inv = sp.linalg.solve_triangular(corr_spatial_cholesky, np.identity(N_Sampling), lower=True)
            corr_spatial_inv = np.dot((corr_spatial_cholesky_inv).T, corr_spatial_cholesky_inv)
            A = np.dot(corr_spatial_cross.T, corr_spatial_inv)
            B = np.dot(corr_spatial_cholesky_inv, corr_spatial_cross)
            scale_B = np.abs(B).max(axis=0)
            B = (B + 0.0) / scale_B
            sd_kriging = SD_kriging(B, scale_B, N_Kriging)

        flag = True
        quantiles = []
        N_occ_sample = 1

        for occID in occIDList:
            sample_sev_kriging = []
            sample_normal_loc = corr_sampling(corr_spatial_cholesky, corr_vul_cholesky, N_occ_sample, N_Sampling)
            sample_sev_loc = sps.norm.cdf(sample_normal_loc)
            sample_sev_loc = ["%.8f"%round(sample_sev_loc[i][0], 8) for i in xrange(len(sample_sev_loc))]
            sample_sev_total = sample_sev_loc

            if N_Kriging > 0:
                sample_sev_kriging = kriging(sample_normal_loc, A, sd_kriging, N_Kriging, N_occ_sample)
                sample_sev_kriging = sps.norm.cdf(sample_sev_kriging)
                v = np.array(sample_sev_kriging)
                counter += np.count_nonzero(v == 1)
                counter += np.count_nonzero(v == 0)
                # If the cholesky decomposition is calculated correctly, there shouldn't be any nans.
                # counter += np.count_nonzero(np.isnan([i[0] for i in sample_sev_kriging]))
                if counter > 10:
                    flag2 = False
                    break
                tmp_kriging = ["%.8f"%round(sample_sev_kriging[i][0], 8) for i in xrange(len(sample_sev_kriging))]
                sample_sev_kriging = ["%.8f"%round(sample_sev_kriging[i][0], 8) for i in xrange(len(sample_sev_kriging))]
                sample_sev_total += sample_sev_kriging
            if flag:
                quantiles = sample_sev_total
                flag = False
            else:
                quantiles += list(sample_sev_total)
        if flag2:
            nOccIDs = len(occIDList)
            occIDListLong = list(np.repeat(occIDList, N_loc))
            eventIDListLong = [eventID] * (N_loc * nOccIDs)
            locIDListLong = list(locIDs) * nOccIDs
            df = pd.DataFrame({'occID': occIDListLong, 'eventID': eventIDListLong, 'locID': locIDListLong, 'quantile': quantiles})
            outputFileName = 'df_' + eventID + '.csv'
            df.to_csv('output\\' + outputFileName, index=False, header=False)
            print("EventID %s is done." % str(eventID))
            break
        else:
            pass
