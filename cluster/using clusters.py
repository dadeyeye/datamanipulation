#!/usr/bin/env python
# coding: utf-8



# In[32]:


#imports 
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


# In[34]:


#make random dataset
data = make_blobs(n_samples=100,centers=4,n_features=3,cluster_std=1.6,random_state=50)
data


# In[41]:


x = data[0]
x


# In[44]:


def kmean(x,k, total_iterations):
    idx = np.random.choice(len(x), k, replace=False)
    
    #random choice centroid
    centroids = x[idx, :]
    
    distances = cdist(x, centroids,'euclidean')
    
    points = np.array([np.argmax(i) for i in distances])
    
    for _ in range(total_iterations):
        centroids = []
        for idx in range(k):
            temp_cent = x[points==idx].mean(axis=0)
            centroids.append(temp_cent)
            
        centroids = np.vstack(centroids)
        
        distances = cdist(x, centroids, 'euclidean')
        points = np.array([np.argmax(i) for i in distances])
        
    return points


# In[45]:


#using k mean implementation
label = kmean(x, 10, 100)


# In[47]:


u_labels = np.unique(label)
for i in u_labels:
    plt.scatter(x[label == i , 0] , x[label == i , 1] , label = i)
    plt.legend()
    plt.show()


# In[36]:


kmeans = KMeans(n_clusters=5)
kmeans.fit(x)


# In[ ]:





# In[38]:


clusters = kmeans.cluster_centers_
print(clusters)


# In[39]:


y_km = kmeans.fit_predict(x)


# In[40]:


y_km


# In[21]:


plt.scatter(x[y_km == 0,0], x[y_km == 0,1], s=50, color='green')
plt.scatter(x[y_km == 1,0], x[y_km == 1,1], s=50, color='red')
plt.scatter(x[y_km == 2,0], x[y_km == 2,1], s=50, color='blue')
plt.scatter(x[y_km == 3,0], x[y_km == 3,1], s=50, color='black')



plt.scatter(clusters[0][0], clusters[0][1], marker='*', s=300, color='purple')
plt.scatter(clusters[1][0], clusters[1][1], marker='*', s=300, color='purple')
plt.scatter(clusters[2][0], clusters[2][1], marker='*', s=300, color='purple')
plt.scatter(clusters[3][0], clusters[3][1], marker='*', s=300, color='purple')
plt.show()


# In[49]:


x


# In[56]:


pd.DataFrame(x).to_csv('data.csv', index_label = "Index", header = ['a' ,'b','c'])


# In[58]:


df1 = pd.read_csv('data.csv')
df1.head()


# In[62]:


clusters


# In[ ]:




