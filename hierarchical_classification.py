import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Mall_Customers.csv")

x= dataset.iloc[: , [3,4]].values


import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x, method="ward"))
plt.title("Dendrogram")
plt.show()


from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity="euclidean", linkage="ward")
y_means = hc.fit_predict(x)


plt.scatter(x[y_means==0,0],x[y_means==0,1],s=100,c='blue',label='C1')
plt.scatter(x[y_means==1,0],x[y_means==1,1],s=100,c='red',label='C2')
plt.scatter(x[y_means==2,0],x[y_means==2,1],s=100,c='black',label='C3')
plt.scatter(x[y_means==3,0],x[y_means==3,1],s=100,c='cyan',label='C4')
plt.scatter(x[y_means==4,0],x[y_means==4,1],s=100,c='green',label='C5')


plt.title("Cluster formation ")
plt.xlabel("k clusters of clients")
plt.ylabel('Spending Score')
plt.show()
