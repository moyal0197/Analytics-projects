#!/usr/bin/env python
# coding: utf-8

# # Clustering

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import scipy.stats as stats
import pandas_profiling   #need to install using anaconda prompt (pip install pandas_profiling)

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = 10, 7.5
plt.rcParams['axes.grid'] = True

from matplotlib.backends.backend_pdf import PdfPages


# In[2]:


from sklearn.cluster import KMeans

# center and scale the data
from sklearn.preprocessing import StandardScaler


# In[3]:


# reading data into dataframe

telco= pd.read_csv("C:\\Users\\ADMIN\\Desktop\\Analytix Labs\\PYTHON\\Class 15-16 files\\Segmentation - Telecom\\telco_csv.csv")


# In[4]:


telco.head()


# In[5]:


telco.info()


# In[6]:


telco.describe()


# In[7]:


#Detailed profiling using pandas profiling

pandas_profiling.ProfileReport(telco)


# In[8]:


numeric_var_names=[key for key in dict(telco.dtypes) if dict(telco.dtypes)[key] in ['float64', 'int64', 'float32', 'int32']]
cat_var_names=[key for key in dict(telco.dtypes) if dict(telco.dtypes)[key] in ['object']]
print(numeric_var_names)
print(cat_var_names)


# In[9]:


telco_num=telco[numeric_var_names]
telco_num.head(5)


# In[10]:


telco_cat = telco[cat_var_names]
telco_cat.head(5)


# In[11]:


# Creating Data audit Report
# Use a general function that returns multiple values
def var_summary(x):
    return pd.Series([x.count(), x.isnull().sum(), x.sum(), x.mean(), x.median(),  x.std(), x.var(), x.min(), x.dropna().quantile(0.01), x.dropna().quantile(0.05),x.dropna().quantile(0.10),x.dropna().quantile(0.25),x.dropna().quantile(0.50),x.dropna().quantile(0.75), x.dropna().quantile(0.90),x.dropna().quantile(0.95), x.dropna().quantile(0.99),x.max()], 
                  index=['N', 'NMISS', 'SUM', 'MEAN','MEDIAN', 'STD', 'VAR', 'MIN', 'P1' , 'P5' ,'P10' ,'P25' ,'P50' ,'P75' ,'P90' ,'P95' ,'P99' ,'MAX'])

num_summary=telco_num.apply(lambda x: var_summary(x)).T


# In[12]:


num_summary


# In[13]:


#Handling Outliers - Method2
def outlier_capping(x):
    x = x.clip_upper(x.quantile(0.99))
    x = x.clip_lower(x.quantile(0.01))
    return x

telco_num=telco_num.apply(lambda x: outlier_capping(x))


# In[14]:


telco.isnull().any()


# In[15]:


#Handling missings - Method2
def Missing_imputation(x):
    x = x.fillna(x.mean())
    return x

telco_num=telco_num.apply(lambda x: Missing_imputation(x))


# In[16]:


telco_num.corr()


# In[17]:


# visualize correlation matrix in Seaborn using a heatmap
sns.heatmap(telco_num.corr())


# ### Standardrizing data 
# - To put data on the same scale 

# In[18]:


sc=StandardScaler()

telco_scaled=sc.fit_transform(telco_num)


# In[19]:


pd.DataFrame(telco_scaled).describe()


# ### Applyting PCA

# In[20]:


from sklearn.decomposition import PCA


# In[21]:


pc = PCA(n_components=30)


# In[22]:


pc.fit(telco_scaled)

#The amount of variance that each PC explains
var= pc.explained_variance_ratio_

#Cumulative Variance explains
var1=np.cumsum(np.round(pc.explained_variance_ratio_, decimals=4)*100)


# In[23]:


var


# In[24]:


var1


# In[25]:


#Alternative method

var_ratio={}
for n in range(2,30):
    pc=PCA(n_components=n)
    telco_pca=pc.fit(telco_scaled)
    var_ratio[n]=sum(telco_pca.explained_variance_ratio_)
    
var_ratio


# In[26]:


plt.plot(var1)


# In[27]:



pc_final=PCA(n_components=12).fit(telco_scaled)

reduced_cr=pc_final.fit_transform(telco_scaled)


# In[28]:


pd.DataFrame(reduced_cr).head(5)


# In[29]:


col_list=telco_num.columns


# In[30]:


col_list


# #### Loading Matrix
# 
# Loadings=Eigenvectors * sqrt(Eigenvalues)
# 
# loadings are the covariances/correlations between the original variables and the unit-scaled components.

# In[31]:


#pc_final.components_

#print pd.DataFrame(pc_final.components_,columns=telco_num.columns).T

Loadings =  pd.DataFrame((pc_final.components_.T * np.sqrt(pc_final.explained_variance_)).T,columns=telco_num.columns).T


# In[32]:


Loadings.to_csv("Loadings.csv")


# ### Clustering 

# In[33]:


list_var = ['custcat','pager','wiremon','equipmon','ebill','multline','marital','income','retire','gender','region']


# In[34]:


telco_scaled1=pd.DataFrame(telco_scaled, columns=telco_num.columns)
telco_scaled1.head(5)

telco_scaled1.dtypes


# In[35]:


telco_scaled2=telco_scaled1[list_var]
telco_scaled2.head(5)


# In[36]:


from sklearn.cluster import KMeans


# In[37]:


km_3=KMeans(n_clusters=3,random_state=123)


# In[38]:


km_3.fit(telco_scaled2)
#km_4.labels_


# In[39]:


km_3.labels_


# In[40]:


pd.Series(km_3.labels_).value_counts()


# In[41]:


km_4=KMeans(n_clusters=4,random_state=123).fit(telco_scaled2)
#km_5.labels_a

km_5=KMeans(n_clusters=5,random_state=123).fit(telco_scaled2)
#km_5.labels_

km_6=KMeans(n_clusters=6,random_state=123).fit(telco_scaled2)
#km_6.labels_

km_7=KMeans(n_clusters=7,random_state=123).fit(telco_scaled2)
#km_7.labels_

km_8=KMeans(n_clusters=8,random_state=123).fit(telco_scaled2)
#km_5.labels_


# In[42]:


km_4.labels_


# In[43]:


# Conactenating labels found through Kmeans with data 
#cluster_df_4=pd.concat([telco_num,pd.Series(km_4.labels_,name='Cluster_4')],axis=1)

# save the cluster labels and sort by cluster
telco_num['cluster_3'] = km_3.labels_
telco_num['cluster_4'] = km_4.labels_
telco_num['cluster_5'] = km_5.labels_
telco_num['cluster_6'] = km_6.labels_
telco_num['cluster_7'] = km_7.labels_
telco_num['cluster_8'] = km_8.labels_


# In[44]:


telco_num.head(5)


# In[45]:


pd.Series.sort_index(telco_num.cluster_3.value_counts())


# In[46]:


pd.Series(telco_num.cluster_3.size)


# In[47]:


size=pd.concat([pd.Series(telco_num.cluster_3.size), pd.Series.sort_index(telco_num.cluster_3.value_counts()), pd.Series.sort_index(telco_num.cluster_4.value_counts()),
           pd.Series.sort_index(telco_num.cluster_5.value_counts()), pd.Series.sort_index(telco_num.cluster_6.value_counts()),
           pd.Series.sort_index(telco_num.cluster_7.value_counts()), pd.Series.sort_index(telco_num.cluster_8.value_counts())])


# In[48]:


size


# In[49]:


Seg_size=pd.DataFrame(size, columns=['Seg_size'])
Seg_Pct = pd.DataFrame(size/telco_num.cluster_3.size, columns=['Seg_Pct'])
Seg_size.T


# In[50]:


Seg_Pct.T


# In[51]:


telco_num.head(5)


# In[52]:


# Mean value gives a good indication of the distribution of data. So we are finding mean value for each variable for each cluster
Profling_output = pd.concat([telco_num.apply(lambda x: x.mean()).T, telco_num.groupby('cluster_3').apply(lambda x: x.mean()).T, telco_num.groupby('cluster_4').apply(lambda x: x.mean()).T,
          telco_num.groupby('cluster_5').apply(lambda x: x.mean()).T, telco_num.groupby('cluster_6').apply(lambda x: x.mean()).T,
          telco_num.groupby('cluster_7').apply(lambda x: x.mean()).T, telco_num.groupby('cluster_8').apply(lambda x: x.mean()).T], axis=1)

Profling_output_final=pd.concat([Seg_size.T, Seg_Pct.T, Profling_output], axis=0)
#Profling_output_final.columns = ['Seg_' + str(i) for i in Profling_output_final.columns]
Profling_output_final.columns = ['Overall', 'KM3_1', 'KM3_2', 'KM3_3',
                                'KM4_1', 'KM4_2', 'KM4_3', 'KM4_4',
                                'KM5_1', 'KM5_2', 'KM5_3', 'KM5_4', 'KM5_5',
                                'KM6_1', 'KM6_2', 'KM6_3', 'KM6_4', 'KM6_5','KM6_6',
                                'KM7_1', 'KM7_2', 'KM7_3', 'KM7_4', 'KM7_5','KM7_6','KM7_7',
                                'KM8_1', 'KM8_2', 'KM8_3', 'KM8_4', 'KM8_5','KM8_6','KM8_7','KM8_8',]


# In[53]:


Profling_output_final


# In[54]:


Profling_output_final.to_csv('Profiling_output.csv')


# ### Finding Optimal number of clusters

# In[55]:


# Dendogram

cmap = sns.cubehelix_palette(as_cmap=True, rot=-.3, light=1)
g = sns.clustermap(telco_scaled, cmap=cmap, linewidths=.5)


# <b> Note: </b>
# 
# - The dendogram shows there are 5 disctinct clusters. 

# ### Elbow Analysis 

# In[56]:


cluster_range = range( 1, 20 )
cluster_errors = []

for num_clusters in cluster_range:
    clusters = KMeans( num_clusters )
    clusters.fit( telco_scaled2 )
    cluster_errors.append( clusters.inertia_ )


# In[57]:


clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors } )

clusters_df[0:10]


# In[58]:


# allow plots to appear in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))
plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" )


# ### Note:
# - The elbow diagram shows that the gain in explained variance reduces significantly from 3 to 4 to 5. So, optimal number of clusters could either 4 or 5. 
# - The actual number of clusters chosen can be finally based on business context and convenience of dealing with number of segments or clusters.

# ### Silhouette Coefficient

# In[59]:


# calculate SC for K=3
from sklearn import metrics
metrics.silhouette_score(telco_scaled2, km_3.labels_)


# In[60]:


# calculate SC for K=3 through K=12
k_range = range(3, 8)
scores = []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=1)
    km.fit(telco_scaled2)
    scores.append(metrics.silhouette_score(telco_scaled2, km.labels_))


# In[61]:


scores


# In[62]:


# plot the results
plt.plot(k_range, scores)
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Coefficient')
plt.grid(True)


# ### Note
# 
# The solution can be 4 or 5 or 6

# #  DBSCAN clustering
# ### Density-based spatial clustering of applications with noise (DBSCAN) 

# In[ ]:


# DBSCAN with eps=1 and min_samples=3
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=2.05, min_samples=10)
db.fit(telco_scaled2)


# In[ ]:


pd.Series(db.labels_).value_counts()


# In[ ]:


# review the cluster labels
db.labels_


# In[ ]:


# save the cluster labels and sort by cluster
telco_num['DB_cluster'] = db.labels_


# In[ ]:


# review the cluster centers
telco_num.groupby('DB_cluster').mean()


# In[ ]:




