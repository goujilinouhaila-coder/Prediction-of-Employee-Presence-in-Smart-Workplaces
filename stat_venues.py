import pandas as pd
import numpy as np
import pickle

df = pd.read_csv("data/df_venues_processed.csv",sep=";")
print(df.info())

print(df.head())
print(df.tail())
print(df.describe())


X = np.zeros((50,3,5))
for i in range(0,250,5):
    X[i//5,0,:] = df.iloc[i:(i+5),1] # venues
    X[i//5,1,:] = df.iloc[i:(i+5),13]  # temp
    # X[i//5,2,:] = df.iloc[i:(i+5),14] / 15  # precip.
    X[i//5,2,:] = df.iloc[i:(i+5),19] /100# resa

def audessus100(X):
    n = X.shape[0]
    ligne_a_enlever=[]
    for i in range(n):
        if True in [X[i,0,j]<100 for j in range(5)]:
            ligne_a_enlever.append(i)
    k = len(ligne_a_enlever)
    Xaudessus100 = np.delete(X,ligne_a_enlever,axis=0)
    return (n-k,Xaudessus100)

n,X = audessus100(X)
print(n)

moy_semaine = np.array([np.mean(X[:,0,0]),np.mean(X[:,0,1]),np.mean(X[:,0,2]),np.mean(X[:,0,3]),np.mean(X[:,0,4])])


Xapp = np.zeros((n-2,2*44,5))
Yapp = np.zeros((n-2,5))
for i in range(n-2):
    for j in range(i):
        Xapp[i,2*j,:] = X[j,0,:] #venue lagged (order2)
        # Xapp[i,4*j+3,:] = X[j+1,0,:] # venue lagged (order1)
        # Xapp[i,4*j+1,:] = X[j,1,:] #temp
        Xapp[i,2*j+1,:] = X[j,2,:] #resa
    Yapp[i,:] = X[i+2,0,:]

Xapp = Xapp.swapaxes(1,-1)


with open('data/venues.dat', 'wb') as f:
    pickle.dump((Xapp,Yapp), f)