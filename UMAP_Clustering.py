#!/usr/bin/env python
# coding: utf-8

#Author: Cong Zhu
import umap
from sklearn.datasets import load_digits
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler




import matplotlib.pyplot as plt
import gower


import os
import seaborn as sns
from pandas import DataFrame


os.chdir("....")


df_baseline = pd.read_excel('....')


df_gower = pd.read_excel('....')





df_index = pd.read_excel('...',header=None, index_col=False)



pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)



import hdbscan
import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import numba


df_baseline2 = df_baseline.iloc[:,1:]
df_baseline2.head()

df_gower.head()

df_gower_np = np.array(df_gower)





def umap_plot(dim, clusterable_embedding,labels):

    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.7, 0.7, 0.7) for x in labels]
    fig =plt.figure(figsize=(12, 8))
    
        
    if dim==3:
        ax_clst = fig.add_subplot(111, projection='3d')
        ax_clst.scatter(clusterable_embedding[:, 0], clusterable_embedding[:, 1],clusterable_embedding[:, 2],
                c=colors, s=20, cmap='Spectral')
    
    elif dim==2:
        ax_clst = fig.add_subplot(111)
        ax_clst.scatter(clusterable_embedding[:, 0],
            clusterable_embedding[:, 1],
            c=colors,
            s=20,
            alpha = 0.5,
            cmap='Spectral')
    else:
        print("Dimension is {}, can't visualize".format(dim))
    
    


# In[26]:


def umap_cluster(dim, n_neighbors,min_samples, min_cluster_size):
    
    clusterable_embedding = umap.UMAP(
    n_neighbors=n_neighbors,
    min_dist=0.1,
    n_components=dim,
    random_state=42,
    metric = "precomputed"
    ).fit_transform(df_gower_np)
    
    
    labels = hdbscan.HDBSCAN(
    min_samples=min_samples,
    min_cluster_size=min_cluster_size,
        
    ).fit_predict(clusterable_embedding)
    
    umap_plot(dim = dim,clusterable_embedding = clusterable_embedding,labels = labels)
    
    
    newvar = "cluster_{dim}D_minsamp{min_samples}_mincluster{min_cluster_size}".format(dim = dim,
                                                                                      min_samples = min_samples,
                                                                                      min_cluster_size = min_cluster_size)
    df_index[newvar] = labels
    if dim<=3:
        plt.savefig('aim2_cluster/output/umap_hdbscan/{}_gower_final_V3.png'.format(newvar), format='png', dpi=700)
        plt.close()
    pass

    return(clusterable_embedding)


# In[29]:


writer = pd.ExcelWriter('.....xlsx')
for i in range(2,4):
    for j in range(70,180,20):
        tab_name = "{}D_{}nb".format(i,j)
        gower_umap_embed = umap_cluster(dim=i, min_cluster_size=j,n_neighbors=30,min_samples=100)
        gower_umap_embed = DataFrame(gower_umap_embed)
        gower_umap_embed.to_excel(writer, sheet_name=tab_name)
writer.save()
df_index.to_excel(".....xlsx")



