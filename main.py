import pandas as pd
import numpy as np
import pickle, pprint, time, os
from functions import spatial_sampling

## Step 1: get the occID set and locID set and put them into a dictionary
path1 = 'C:\\Users\\Xu.Tian\\Documents\\11_JPEQ\\Data\\'

df_locID = pd.read_csv(path1 + "df_locID", header=None, dtype=object)  # dtype
df_occID = pd.read_csv(path1 + "df_occID", header=None, dtype=object)
df_locID.columns = ['eventID', 'locID']
df_occID.columns = ['eventID', 'occID']

print(df_occID.head(10))
print(df_locID.head(10))

df_grouped_locID = df_locID.groupby('eventID')
df_grouped_occID = df_occID.groupby('eventID')

occIDDict = {}
locIDDict = {}

for name, group in df_grouped_locID:
    if name not in locIDDict:
        locIDDict[name] = set(group['locID'].unique())

# print(locIDDict)
for name, group in df_grouped_occID:
    if name not in occIDDict:
        occIDDict[name] = set(group['occID'].unique())

print(occIDDict.keys())
print(locIDDict.keys())

output = open('dictData_JPEQ.pkl', 'wb')
pickle.dump(occIDDict, output)
pickle.dump(locIDDict, output)
output.close()

print("Step 1 is finished.")

# Step 2: read in the occID sets and locID sets from the dictionary

# import pprint, pickle
#
# pkl_file = open('dictData_JPEQ.pkl', 'rb')
#
# occIDDict = pickle.load(pkl_file)
#
# # print(eventIDs)
# # print(len(eventIDs))
# # pprint.pprint(occIDDict['3063800'])
#
# locIDDict = pickle.load(pkl_file)
# # eventIDs = [i for i in locIDDict.keys() if len(locIDDict[i]) > 10000]
# # print(eventIDs)
# # pprint.pprint(locIDDict['9339710'])
#
# # for key, val in locIDDict.items():
# #     print("EventID %s has %i locIDs." %(key, len(val)))
#
# pkl_file.close()

## Step 3: join locID with lat lon?

from multiprocessing import Pool
import pprint, pickle
import os.path

def fun(x):
    try:
        outputFileName = 'df_' + x + '.csv'
        if not os.path.exists('output\\' + outputFileName):
            print(x)
            filename = 'dictData_JPEQ_' + x + '.pkl'
            pkl_file = open('sets\\' + filename, 'rb')
            occIDSet = pickle.load(pkl_file)
            locIDSet = pickle.load(pkl_file)
            pkl_file.close()
            # print(occIDSet)
            # print(len(locIDSet))
            # print(locIDSet)
            # print(len(occIDSet))
            eventID = x
            spatial_decay = -0.051
            print(occIDSet)
            print(locIDSet)
            try:
                if len(locIDSet) > 2000:
                    spatial_sampling(occIDSet, eventID, locIDSet, spatial_decay, N_MaxSamplingSize=1000)
                elif len(locIDSet) > 800 and len(locIDSet) <= 2000:
                    spatial_sampling(occIDSet, eventID, locIDSet, spatial_decay, N_MaxSamplingSize=500)
                elif len(locIDSet) > 400 and len(locIDSet) <= 800:
                    spatial_sampling(occIDSet, eventID, locIDSet, spatial_decay, N_MaxSamplingSize=200)
                else:
                    spatial_sampling(occIDSet, eventID, locIDSet, spatial_decay, N_MaxSamplingSize=200)
            except:
                print("!!!!!EventID %s failed!!!!!" %str(eventID))
    except:
        print("EventID %s failed!!!!!" %x)

# if __name__ == '__main__':
#     list0 = os.listdir('sets//')
#     eventIDs = [i.lstrip('dictData_JPEQ_').rstrip('.pkl') for i in list0]
#     nEvents = len(eventIDs)
#     p = Pool(min(nEvents, 20))  # Open at most 10 processes
#     try:
#         p.map(fun, eventIDs)
#         time.sleep(10)
#     except WindowsError:
#         print("WindowsError: [Error 5] Access is denied.")
#         print("The program should continue.")
#     except Exception as e:
#         print(str(e))
#         pass

if __name__ == '__main__':
    eventIDs = ['9348511']
    nEvents = len(eventIDs)
    p = Pool(min(nEvents, 10))  # Open at most 10 processes
    try:
        p.map(fun, eventIDs)
        time.sleep(30)
    except WindowsError:
        print("WindowsError: [Error 5] Access is denied.")
        print("The program should continue.")

## Step 4: checking results
# from multiprocessing import Pool
#
# def fun(x):
#     eventID = x
#     filename = 'df_' + eventID + '.csv'
#     df = pd.read_csv(filename)
#


