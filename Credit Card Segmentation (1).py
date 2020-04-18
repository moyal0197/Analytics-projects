#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Data pre-Preparation


# In[3]:


data= pd.read_csv("C:\\Users\\ADMIN\\Desktop\\Python ASsignments\\Questions\\Advance\\4. CREDIT CARD CASE STUDY - SEGMENTATION\\CC GENERAL.csv")


# In[4]:


data.head()


# In[5]:


data.info()


# In[6]:


data.describe()


# In[7]:


data['CREDIT_LIMIT'].notnull().value_counts()


# In[8]:


data[data['CREDIT_LIMIT'].isnull()]


# In[9]:


data['MINIMUM_PAYMENTS'].notnull().value_counts()


# In[10]:


#Missing value treatment
#replacing them with mean


# In[11]:


data['CREDIT_LIMIT'].fillna(data['CREDIT_LIMIT'].mean(),inplace=True)
data['MINIMUM_PAYMENTS'].fillna(data['MINIMUM_PAYMENTS'].mean(),inplace=True)
print( data.isnull().sum())


# # KPI

# In[12]:


# 1 Monthly average purchase and cash advance amount


# In[13]:


data['Monthly_avg_purchase']=data['PURCHASES']/data['TENURE']
data['Monthly_cash_advance']=data['CASH_ADVANCE']/data['TENURE']


# In[14]:


data.head()


# In[15]:


# 2 Purchase type (one off, installments)


# In[16]:


data[data['ONEOFF_PURCHASES']==0]['ONEOFF_PURCHASES'].count()


# In[17]:


data.loc[:,['ONEOFF_PURCHASES','INSTALLMENTS_PURCHASES']].head(10)


# In[18]:


#There are 4 sets of people. 
#1 who do one off payments
#2 who do installment payments
#3 who do both
#4 who do neither


# In[19]:


#Making a new Variable


# In[20]:


def purchase(data):   
    if (data['ONEOFF_PURCHASES']==0) & (data['INSTALLMENTS_PURCHASES']==0):
        return 'neither'
    if (data['ONEOFF_PURCHASES']>0) & (data['INSTALLMENTS_PURCHASES']>0):
         return 'both'
    if (data['ONEOFF_PURCHASES']>0) & (data['INSTALLMENTS_PURCHASES']==0):
        return 'one_off'
    if (data['ONEOFF_PURCHASES']==0) & (data['INSTALLMENTS_PURCHASES']>0):
        return 'installment'


# In[21]:


data['purchase_type']=data.apply(purchase,axis=1)


# In[22]:


data.head()


# In[23]:


data['purchase_type'].value_counts()


# In[24]:


# 3 Average Amount per purchase and average amount per cash advance are already stated in the table


# In[25]:


# 4  Limit Usage


# In[26]:


data['limit_usage']=data.apply(lambda x: x['BALANCE']/x['CREDIT_LIMIT'], axis=1)


# In[27]:


data.head()


# In[28]:


#5 Payment to minimum payments Ratio


# In[29]:


data['payment_minpay']=data.apply(lambda x:x['PAYMENTS']/x['MINIMUM_PAYMENTS'],axis=1)


# In[30]:


data.head()


# In[31]:


#Log treatment for outliers


# In[32]:


temp=data.drop(['CUST_ID','purchase_type'],axis=1).applymap(lambda x: np.log(x+1))


# In[33]:


col=['BALANCE','PURCHASES','CASH_ADVANCE','TENURE','PAYMENTS','MINIMUM_PAYMENTS','PRC_FULL_PAYMENT','CREDIT_LIMIT']
datanew=temp[[x for x in temp.columns if x not in col ]]


# In[34]:


datanew


# # Analyzing the KPI's

# In[85]:


# Average payment to minpayment ratio for each purchse type.
z=data.groupby('purchase_type').apply(lambda x: np.mean(x['payment_minpay']))
type(z)
z


# In[87]:


data.describe()


# In[88]:


# Insight 1
#Above data signifies that customers that purchased on installments are paying their dues 


# In[89]:


data.groupby('purchase_type').apply(lambda x: np.mean(x['limit_usage'])).plot.bar()


# In[91]:


# Insight 2
# People with installment purchase have good balance to limit ratio which means good credit score


# In[93]:


data.groupby('purchase_type').apply(lambda x: np.mean(x['Monthly_cash_advance'])).plot.bar()


# In[95]:


#Insight 3
# People who belong to neither category have taken the highest average monthly cash advance


# In[98]:


#Creating an original dataset
data_original=pd.concat([data,pd.get_dummies(data['purchase_type'])],axis=1)


# ### Data preparation for modelling

# In[99]:


# creating Dummies for categorical variable
datanew['purchase_type']=data.loc[:,'purchase_type']
pd.get_dummies(datanew['purchase_type']).head()


# In[100]:


data_dummy=pd.concat([datanew,pd.get_dummies(datanew['purchase_type'])],axis=1)


# In[101]:


temp1=['purchase_type']


# In[102]:


data_dummy=data_dummy.drop(temp1,axis=1)
data_dummy.isnull().sum()


# In[103]:


sns.heatmap(data_dummy.corr())


# In[104]:


#Standardizing the data to avoid skewness .


# In[105]:


from sklearn.preprocessing import  StandardScaler
sc=StandardScaler()
data_scaled=sc.fit_transform(data_dummy)


# In[106]:


#PCA


# In[107]:


from sklearn.decomposition import PCA
var_ratio={}
for n in range(4,15):
    pc=PCA(n_components=n)
    data_pca=pc.fit(data_scaled)
    var_ratio[n]=sum(data_pca.explained_variance_ratio_)


# In[108]:


pc=PCA(n_components=5)
p=pc.fit(data_scaled)


# In[109]:


data_scaled.shape


# In[110]:


p.explained_variance_


# In[112]:


pd.Series(var_ratio).plot()


# In[113]:


var_ratio


# In[114]:


#Since 5 components are explaining about 87.5% variance so we select 5 components


# In[115]:


pc_final=PCA(n_components=5).fit(data_scaled)

reduced_data=pc_final.fit_transform(data_scaled)


# In[117]:


df1=pd.DataFrame(reduced_data)
df1.shape


# In[118]:


col_list=data_dummy.columns
col_list


# In[122]:


temp1=pd.DataFrame(pc_final.components_.T, columns=['PC_' +str(i) for i in range(5)],index=col_list)
temp1


# In[123]:


#Factor analysis
pd.Series(pc_final.explained_variance_ratio_,index=['PC_'+ str(i) for i in range(5)])


# ### Clustering

# In[124]:


# Using K means


# In[ ]:


# Creating 4 clusters


# In[128]:


from sklearn.cluster import KMeans
cl4=KMeans(n_clusters=4,random_state=123)
cl4.fit(reduced_data)
cl4.labels_


# In[129]:


pd.Series(cl4.labels_).value_counts()


# In[133]:


color_map={0:'b',1:'r',2:'y',3:'m'}
label_color=[color_map[l] for l in cl4.labels_]
plt.figure(figsize=(7,7))
plt.scatter(reduced_data[:,0],reduced_data[:,1],c=label_color,cmap='Spectral',alpha=0.5)


# In[134]:


pair=pd.DataFrame(reduced_data,columns=['PC_' +str(i) for i in range(5)])


# In[135]:


pair['Cluster']=cl4.labels_


# In[142]:


# Key performace variable selection 
col_kpi=['PURCHASES_TRX','Monthly_avg_purchase','Monthly_cash_advance','limit_usage','CASH_ADVANCE_TRX',
         'payment_minpay','both','installment','one_off','neither','CREDIT_LIMIT']


# In[143]:


#concatenating labels


# In[144]:


clus4=pd.concat([data_original[col_kpi],pd.Series(cl4.labels_,name='Cluster_4')],axis=1)


# In[145]:


clus4.head()


# In[147]:


#Using mean values to find distribution of data
cluster4=clus4.groupby('Cluster_4').apply(lambda x: x[col_kpi].mean()).T
cluster4


# In[152]:


# Percentage of each cluster in the total customer base
s=clus4.groupby('Cluster_4').apply(lambda x: x['Cluster_4'].value_counts())
per=pd.Series((s.values.astype('float')/ clus4.shape[0])*100,name='Percentage')
print (pd.concat([pd.Series(s.values,name='Size'),per],axis=1),'\n')


# In[149]:


#It shows that there is distinguishing distribution among clusters


# In[153]:


#5 Clusters


# In[156]:


cl5=KMeans(n_clusters=5,random_state=123)
cl5=cl5.fit(reduced_data)
cl5.labels_


# In[157]:


pd.Series(cl5.labels_).value_counts()


# In[159]:


plt.figure(figsize=(10,10))
plt.scatter(reduced_data[:,0],reduced_data[:,1],c=cl5.labels_,cmap='Spectral',alpha=0.5)
plt.xlabel('PC_0')
plt.ylabel('PC_1')


# In[160]:


clus5=pd.concat([data_original[col_kpi],pd.Series(cl5.labels_,name='Cluster_5')],axis=1)


# In[163]:


# Finding Mean of features for each cluster to determine distribution
clus5.groupby('Cluster_5').apply(lambda x: x[col_kpi].mean()).T


# In[164]:


# Cluster 2 and 4 are complementing each other


# In[167]:


#Percentage of each cluster
s1=clus5.groupby('Cluster_5').apply(lambda x: x['Cluster_5'].value_counts())
per_5=pd.Series((s1.values.astype('float')/ clus5.shape[0])*100,name='Percentage')
print (pd.concat([pd.Series(s1.values,name='Size'),per_5],axis=1))


# In[168]:


# 6 Clusters


# In[170]:


cl6=KMeans(n_clusters=6,random_state=123)
cl6=cl6.fit(reduced_data)
cl6.labels_


# In[171]:


color_map={0:'y',1:'b',2:'g',3:'c',4:'m',5:'k'}
label_color=[color_map[l] for l in cl6.labels_]
plt.figure(figsize=(10,10))
plt.scatter(reduced_data[:,0],reduced_data[:,1],c=label_color,cmap='Spectral',alpha=0.5)


# In[172]:


clus6=pd.concat([data_original[col_kpi],pd.Series(cl6.labels_,name='Cluster_6')],axis=1)


# In[173]:


# Finding Mean of features for each cluster to determine distribution
clus6.groupby('Cluster_6').apply(lambda x: x[col_kpi].mean()).T


# In[174]:


#Clusters are overlapping
#Similar behavior observed within clusters


# ## Checking performance using silhoutte score

# In[175]:


from sklearn.metrics import silhouette_score


# In[176]:


score={}
for n in range(3,10):
    km_score=KMeans(n_clusters=n)
    km_score.fit(reduced_data)   
    score[n]=silhouette_score(reduced_data,km_score.labels_)


# In[177]:


pd.Series(score).plot()


# In[178]:


#Silhoutte Score clearly suggests that cluster 4 is the most optimal solution


# In[181]:


cluster4


# ## Strategies for each cluster

# group 0 : the first cluster has the highest payment to min payment ratio. 
#           This group is performing best among all as cutomers are maintaining good credit score and paying dues on time.This group should be given maximum attention and incentives
# 

# group 1: the second cluster has poor credit score and take only cash in advance. They should be given less attention and can be targetted by giving lowered interest rates

# group 2 : They are potential customers who are paying dues and doing purchases and maintaining good credit score. They can be targetted by increasing their bank limits or lowering their interest rates 
# 

# group 3 :The final cluster  has minimum paying ratio and is using card for just oneoff transactions. This is the highest risk group and should be given least importance

# In[ ]:




