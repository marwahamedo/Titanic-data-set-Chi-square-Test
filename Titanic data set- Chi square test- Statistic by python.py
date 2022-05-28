#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')
import scipy 
from scipy import stats
import seaborn as sns
import os
os.getcwd ()


# In[2]:


df = pd.read_csv('titanic_train.csv', encoding ='latin-1')
df.head(5)


# In[3]:


df.info()


# In[4]:


df.isnull().sum()


# In[5]:


df.dropna(inplace = True)


# In[6]:


#adding column named Alone, based on the Parch (Parent or children) and Sibsp (Siblings or spouse) columns. 
#The idea we want to explore is if being alone affects the surviability of the passenger.
#So Alone is 1 if both Parch(parent or children) and Sibsp(Siblings or spouse) are 0.
df['Alone'] = (df['Parch'] + df['SibSp']).apply(
                  lambda x: 1 if x == 0 else 0)
df


# In[29]:


ax = sns.barplot(x='Alone', y='Survived', data=df, ci=None)    
ax.set_xticklabels(['Not Alone','Alone'])


# In[8]:


df1 = df[['Survived','Alone','Sex']]
print(pd.pivot_table(df1, index = 'Alone', columns = 'Survived' ,aggfunc= 'count', margins = True, margins_name='Total'))


# In[9]:


def chi2(df, col1, col2):    
    #---create the contingency table---
    df_cont = pd.crosstab(index = df[col1], columns = df[col2])
    display(df_cont)
    #---calculate degree of freedom---
    degree_f = (df_cont.shape[0]-1) * (df_cont.shape[1]-1)
    #---sum up the totals for row and columns---
    df_cont.loc[:,'Total']= df_cont.sum(axis=1)
    df_cont.loc['Total']= df_cont.sum()
    print('---Observed (O)---')
    display(df_cont)
    #---create the expected value dataframe---
    df_exp = df_cont.copy()    
    df_exp.iloc[:,:] = np.multiply.outer(
        df_cont.sum(1).values,df_cont.sum().values) / df_cont.sum().sum()            
    print('---Expected (E)---')
    display(df_exp)
        
    # calculate chi-square values
    df_chi2 = ((df_cont - df_exp)**2) / df_exp    
    df_chi2.loc[:,'Total']= df_chi2.sum(axis=1)
    df_chi2.loc['Total']= df_chi2.sum()
    
    print('---Chi-Square---')
    display(df_chi2)
    #---get chi-square score---   
    chi_square_score = df_chi2.iloc[:-1,:-1].sum().sum()
    
    

    return chi_square_score, degree_f
  


# In[26]:


chi_score, degree_f = chi2(df,'Sex','Survived')
print(f'Chi2_score: {chi_score}, Degrees of freedom: {degree_f}')


# In[22]:


#---calculate the p-value---
from scipy import stats
p = stats.distributions.chi2.sf(51.8748, 1)
print (p)


# In[ ]:


### p- value > 0.05 you accept the H₀ (Null Hypothesis) 
###This means the two categorical variables are independent


# In[24]:


chi_score, degree_f = chi2(df,'Alone','Survived')
print(f'Chi2_score: {chi_score}, Degrees of freedom: {degree_f}')


# In[25]:


p = stats.distributions.chi2.sf(2.627, 1)
print (p)


# In[ ]:


### p- value > 0.05 you accept the H₀ (Null Hypothesis) 
###This means the two categorical variables are independent

