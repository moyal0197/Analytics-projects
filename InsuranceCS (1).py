#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import datetime as dt   
import datetime
from dateutil.relativedelta import relativedelta
from datetime import date
import matplotlib.pyplot as plt
import plotly.express as px


# In[2]:


x=pd.read_csv('C:\\Users\\ADMIN\\Desktop\\Python ASsignments\\Questions\\Basic\\5. Pandas Case Study 3 - Insurance Claims Case Study\\claims.csv')
y=pd.read_csv('C:\\Users\\ADMIN\\Desktop\\Python ASsignments\\Questions\\Basic\\5. Pandas Case Study 3 - Insurance Claims Case Study\\cust_demographics.csv')


# In[3]:


y=y.rename(columns={"CUST_ID": "customer_id"})


# In[4]:


final=pd.merge(y,x,on='customer_id',how='inner')
final.head()


# In[5]:


#Data preparation Question 2 
final.DateOfBirth=pd.to_datetime(final.DateOfBirth)
final.claim_date=pd.to_datetime(final.claim_date)


# In[6]:


#Question 3
final['claim_amount'] = final['claim_amount'].str.replace('$', '')
final['claim_amount'] = final['claim_amount'] .astype(float)


# In[7]:


#Question 4
final['alert'] = np.where((final.claim_type == 'Injury only') & (final.police_report == 'Unknown'), '1', 
                           np.where((final.claim_type == 'Injury only') & (final.police_report == 'Yes'),'0','0'))
final.head()


# In[8]:


#question5


# In[9]:


#Question6
x=final.isnull().sum()


# In[10]:


final['claim_amount'] = final['claim_amount'].fillna(final.claim_amount.mean())
final['total_policy_claims'] = final['total_policy_claims'].fillna(final.total_policy_claims.mean())


# In[11]:


#Question7

tod=dt.datetime.today().strftime("%y-%m-%d")
tod=pd.to_datetime(tod)
final['age'] =  tod-(final.DateOfBirth) 
final['age']=final['age']/np.timedelta64(1,'Y')
final['age']=final['age'].astype(int)
final['age'].abs()
final.head()


# In[12]:


final.loc[final.age <18, 'Age Group'] = 'Children'  
final.loc[(final.age >=18) & (final.age <30), 'Age Group'] = 'Youth'  
final.loc[(final.age >=30) & (final.age <60), 'Age Group'] = 'Adult' 
final.loc[final.age >=60, 'Age Group'] = 'Senior'  


# In[173]:


final.head()


# In[174]:


#Question 8


# In[13]:


x=final.groupby('Segment').agg({'claim_amount':'mean'})
x


# In[14]:


#Question 9
temp=final.sort_values(by='claim_date',ascending=False)
temp.groupby('incident_cause').agg({'claim_amount':'sum'})


# In[15]:


#Question 10  5 youths

temp2=final[(final.State == 'TX') | (final.State == 'AK') | (final.State == 'DE')]
temp2=temp2[(final.incident_cause == 'Driver error')]
temp2.groupby('Age Group').agg({'customer_id':'count'})


# In[16]:


#Question 11
temp=final.groupby('gender').agg({'claim_amount':'sum'})
temp.claim_amount=temp.claim_amount.astype(int)
plot = temp.plot.pie(y='claim_amount', figsize=(5, 5),autopct='%1.1f%%')


# In[17]:


temp=final.groupby('Segment').agg({'claim_amount':'sum'})
temp.claim_amount=temp.claim_amount.astype(int)
plot = temp.plot.pie(y='claim_amount', figsize=(5, 5),autopct='%1.1f%%')


# In[18]:


#Ques 12
temp2=final.groupby(['gender','incident_cause']).agg({'claim_amount':'count'})
temp2=temp2.reset_index()


# In[19]:


temp2=temp2[(temp2.incident_cause == 'Driver error') | (temp2.incident_cause == 'Other driver error')]
temp2=temp2.rename(columns={"claim_amount": "Count"})
temp2


# In[20]:


temp2.plot(kind='bar',x='gender',y='Count')


# In[21]:


#Ques13
temp3=final.groupby(['Age Group','fraudulent']).agg({'claim_id':'count'})

temp3=temp3.reset_index()


# In[22]:


temp3=temp3[(temp3.fraudulent == 'Yes')]
temp3=temp3.rename(columns={"claim_id": "Count"})
temp3.plot(kind='bar',x='Age Group',y='Count')


# In[23]:


#Ques14
final.groupby(pd.Grouper(key='claim_date',freq='M')).agg({'claim_amount':'sum'}).plot(kind='bar',stacked=True)


# In[261]:


temp1=final.groupby(['fraudulent','gender','Age Group']).agg({'claim_amount':'mean'})
temp1=temp1.reset_index()
temp1


# In[ ]:


#Question 16
#Yes the amount claimed by males and females is almost equal with mail claiming 52% while female claiming 48%


# In[266]:


#Question 17.
temp=final.groupby(['Segment','Age Group']).agg({'claim_amount':'sum'})
temp.claim_amount=temp.claim_amount.astype(int)
plot = temp.plot.pie(y='claim_amount', figsize=(15, 15),autopct='%1.1f%%')


# In[ ]:


#Yes the above anlaysis shows that each segment, children have highest claim percentages follwed by youth and follwed by adults


# In[25]:


#Question 18
final.groupby(pd.Grouper(key='claim_date',freq='Y')).agg({'claim_amount':'mean'})

#No, the hypothsis stated is rejected. This is because the average claims of both the years has been almost equal


# In[29]:


#Question 19
final.groupby(['Age Group','total_policy_claims']).agg({'claim_amount':'count'})


# In[ ]:


#Yes, there is a pattern between age groups and number of claims made. 


# In[33]:


#Question 20
final.corr(method='pearson')
#Coefficient index of 0.04 suggests that there is very minor positive correlation between number of claims and claimed amount


# In[ ]:




