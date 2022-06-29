import pandas as pd
import numpy as np
import pickle, time
import os.path
from multiprocessing import Pool

df = pd.read_csv("locID", header=None)
df.columns = ['eventID', 'locID']
df2 = pd.read_csv("occID", header=None)
df2.columns = ['eventID', 'occID']

eventIDsloc = set(df['eventID'].unique())
# eventIDsocc = set(df['eventID'].unique())

eventIDs = list(eventIDsloc)

def f(x):
    filename = 'dictData_JPEQ_' + str(x) + '.pkl'
    if not os.path.exists('sets\\' + filename):
        occSet = set(df2[df2['eventID'] == x]['occID'])
        locSet = set(df[df['eventID'] == x]['locID'])
        filename = 'dictData_JPEQ_' + str(x) + '.pkl'
        output = open('sets\\' + filename, 'wb')
        pickle.dump(occSet, output)
        pickle.dump(locSet, output)
        output.close()

if __name__ == '__main__':
    p = Pool(min(len(eventIDs), 10))  # Open at most 10 processes
    try:
        p.map(f, eventIDs)
        time.sleep(10)
    except WindowsError:
        print("WindowsError: [Error 5] Access is denied.")
        print("The program should continue.")

#
# if eventIDsloc == eventIDsocc:
#     for eventID in eventIDsloc:
#         print(eventID)
#         occSet = set(df2[df2['eventID'] == eventID]['occID'])
#         locSet = set(df[df['eventID'] == eventID]['locID'])
#         filename = 'dictData_JPEQ_' + str(eventID) + '.pkl'
#         output = open('sets\\' + filename, 'wb')
#         pickle.dump(occSet, output)
#         pickle.dump(locSet, output)
#         output.close()
# else:
#     print(len(eventIDsloc))
#     print(len(eventIDsocc))
