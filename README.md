# Exp :08 Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Pick customer segment quantity (k)

2.Seed cluster centers with random data points.

3.Assign customers to closest centers.

4.Re-center clusters and repeat until stable.
 

## Program:
```py

Program to implement the K Means Clustering for Customer Segmentation.
Developed by: KANISHKA V S
RegisterNumber: 212222230061 

```
```py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

data=pd.read_csv('/content/Mall_Customers_EX8.csv')
data

x=data[['Annual Income (k$)', 'Spending Score (1-100)']]
x

plt.figure(figsize=(4,4))
plt.scatter(data['Annual Income (k$)'],data['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending score (1-100)')
plt.show()

k=5
kmeans = KMeans(n_clusters=k)
kmeans.fit(x)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_
print("Centroid:")
print(centroids)
print("Labels:")
print(labels)

colors =['r','g','b','c','m']
for i in range(k):
  cluster_points =x[labels == i]
  plt.scatter(cluster_points['Annual Income (k$)'], cluster_points['Spending Score (1-100)'],color=colors[i], label = f'Cluster {i+1}')
  distances = euclidean_distances(cluster_points,[centroids[i]])
  radius = np.max(distances)
  circle = plt.Circle(centroids[i],radius,color=colors[i],fill=False)
  plt.gca().add_patch(circle)
  
plt.scatter(centroids[:,0], centroids[:,1],marker='*',s=200,color='k',label='Centroids')
plt.title('k-means Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show
```
## Output:
### data:
![image](https://github.com/kanishka2305/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/113497357/c920d5aa-5f36-4682-81f6-4efed9dfdb51)

![image](https://github.com/kanishka2305/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/113497357/64b4c7e3-4899-46d3-b046-a398c5adf1e9)

### Scatter Plot:
![image](https://github.com/kanishka2305/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/113497357/330b708a-027c-428e-aaf7-1a1ba7c8fb0f)

### Centroids:
![image](https://github.com/kanishka2305/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/113497357/af14e891-b8aa-4c70-8236-b9caf415a376)

### KMeans Clustering:
![image](https://github.com/kanishka2305/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/113497357/d9e184d8-c221-4bbe-be30-4378d0eeb88b)

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
