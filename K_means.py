import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

dataset = pd.read_csv('Mall_Customers.csv')
x =dataset.iloc[:,[3,4]].values



from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init= "k-means++" , random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title("elbow_method")
plt.xlabel("k_cluster")
plt.ylabel  ("wcss")  
    

kmeans = KMeans(n_clusters=5,init="k-means++",random_state=0)
y_means = kmeans.fit_predict(x)

plt.scatter(x[y_means==0,0],x[y_means==0,1],s=100,c='blue',label='C1')
plt.scatter(x[y_means==1,0],x[y_means==1,1],s=100,c='red',label='C2')
plt.scatter(x[y_means==2,0],x[y_means==2,1],s=100,c='black',label='C3')
plt.scatter(x[y_means==3,0],x[y_means==3,1],s=100,c='cyan',label='C4')
plt.scatter(x[y_means==4,0],x[y_means==4,1],s=100,c='green',label='C5')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='yellow',label='centroids')

plt.title("Cluster formation ")
plt.xlabel("k clusters of clients")
plt.ylabel('Spending Score')
plt.show()