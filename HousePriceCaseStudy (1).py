#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling  


# In[6]:



from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[7]:


HousePrice = pd.read_csv('C:/Users/ADMIN/Desktop/Analytix Labs/PYTHON/Assignments/Regress/Case Study - Housing Example/House_Prices.csv')


# In[8]:


numeric_var_names=[key for key in dict(HousePrice.dtypes) if dict(HousePrice.dtypes)[key] in ['float64', 'int64', 'float32', 'int32']]
cat_var_names=[key for key in dict(HousePrice.dtypes) if dict(HousePrice.dtypes)[key] in ['object']]
print(numeric_var_names)
print(cat_var_names)


# In[15]:


HousePrice_num=HousePrice._get_numeric_data()


# In[37]:


HousePrice_cat = HousePrice[cat_var_names]


# In[16]:


def var_summary(x):
    return pd.Series([x.count(), x.isnull().sum(), x.sum(), x.mean(), x.median(),  x.std(), x.var(), x.min(), x.dropna().quantile(0.01), x.dropna().quantile(0.05),x.dropna().quantile(0.10),x.dropna().quantile(0.25),x.dropna().quantile(0.50),x.dropna().quantile(0.75), x.dropna().quantile(0.90),x.dropna().quantile(0.95), x.dropna().quantile(0.99),x.max()], 
                  index=['N', 'NMISS', 'SUM', 'MEAN','MEDIAN', 'STD', 'VAR', 'MIN', 'P1' , 'P5' ,'P10' ,'P25' ,'P50' ,'P75' ,'P90' ,'P95' ,'P99' ,'MAX'])

num_summary=HousePrice_num.apply(lambda x: var_summary(x)).T


# In[17]:


num_summary


# In[18]:


def outlier_capping(x):
    x = x.clip_upper(x.quantile(0.99))
    x = x.clip_lower(x.quantile(0.01))
    return x

HousePrice_num=HousePrice_num.apply(lambda x: outlier_capping(x))


# In[20]:


num_summary=HousePrice_num.apply(lambda x: var_summary(x)).T
num_summary


# In[21]:


pandas_profiling.ProfileReport(HousePrice)


# In[35]:


def create_dummies( df, colname ):
    col_dummies = pd.get_dummies(df[colname], prefix=colname)
    col_dummies.drop(col_dummies.columns[0], axis=1, inplace=True)
    df = pd.concat([df, col_dummies], axis=1)
    df.drop( colname, axis = 1, inplace = True )
    return df


# In[36]:


HousePrice.info()


# In[38]:


HousePrice_cat=HousePrice[['Brick', 'Neighborhood']]

for c_feature in ['Brick', 'Neighborhood']:
    HousePrice_cat[c_feature] = HousePrice_cat[c_feature].astype('category')
    HousePrice_cat= create_dummies(HousePrice_cat, c_feature )


# In[45]:


HousePrice_new = pd.concat([HousePrice_num, HousePrice_cat], axis=1)


# In[42]:





# In[46]:



feature_columns = HousePrice_new.columns.difference( ['Price'] )
feature_columns


# In[47]:


from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split( HousePrice_new[feature_columns],
                                                  HousePrice_new['Price'],
                                                  test_size = 0.3,
                                                  random_state = 123 )


# In[48]:


import statsmodels.api as sm


# In[49]:


train_X = sm.add_constant(train_X)
lm=sm.OLS(train_y,train_X.astype(float)).fit()


# In[50]:


print(lm.summary())


# In[51]:


print('Parameters: ', lm.params)
print('R2: ', lm.rsquared)


# In[52]:


test_X = sm.add_constant(test_X)
y_pred = lm.predict(test_X)
# calculate these metrics by hand!
from sklearn import metrics
import numpy as np
print('MAE:', metrics.mean_absolute_error(test_y, y_pred))
print('MSE:', metrics.mean_squared_error(test_y, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(test_y, y_pred)))


# In[ ]:




