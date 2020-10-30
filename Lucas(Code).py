
# coding: utf-8

# In[2]:


import pandas as pd#import all of the related python libraries to be able to deal with the data 
import numpy as np
import glob
import sklearn.preprocessing
import struct
import ctypes
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt


# In[2]:


# get the names of the files in the directory
print(glob.glob("D:\School\Fifth Year\Large Scale Data Analytics\Project\ieee-fraud-detection\*"))


# This makes it so we can imput the data and have it stored 

# In[3]:


#Include the data as a csv for the training identity and test to see if it is inluded properly
train_identity=pd.read_csv("D:\School\Fifth Year\Large Scale Data Analytics\Project\ieee-fraud-detection\\train_identity.csv")
#Create a Data frame to collect and summarise the data 
tr_IDdf=pd.DataFrame(train_identity)

#include the test identity files 
test_identity=pd.read_csv("D:\School\Fifth Year\Large Scale Data Analytics\Project\ieee-fraud-detection\\test_identity.csv")
#Create a Data frame to summarize info  
te_IDdf=pd.DataFrame(test_identity)

#Get the training transaction info 
train_transaction=pd.read_csv("D:\School\Fifth Year\Large Scale Data Analytics\Project\ieee-fraud-detection\\train_transaction.csv")
#Create a Data frame to summarize info 
tr_TR_df=pd.DataFrame(train_transaction)

#Get the testing transaction data 
test_transaction=pd.read_csv("D:\School\Fifth Year\Large Scale Data Analytics\Project\ieee-fraud-detection\\test_transaction.csv")
#Create a Data frame to summarize info 
te_TR_df=pd.DataFrame(test_transaction)


# We will be using this next section to view the information about each data frame 

# In[4]:


print("Training Identity Data\n\n")#print the data frame info for the identity 
print(tr_IDdf.info())
print("\n\nTraining Transaction Data\n\n")#print the training transaction data frame information 
print(tr_TR_df.info())


# For this section we will be looking to find the amount of missing data for the training identity set 

# In[5]:


missing_ident=train_identity.isnull().sum()
missing_ident_sorted=(missing_ident.sort_values(ascending=False))
print('{:15s} {:15s} {:15s} {:15s}'.format("Col Name","# Missing", "% Missing","Data Type"))#print out the columns used 
for y in range(missing_ident.size):#go through each of the columns and get the name, amount missing and the data type and calulate the total amount of the data that is missing 
    print('{:15s} {:15s} {:15s} {:15s}'.format(missing_ident_sorted.index[y],str(tr_IDdf[missing_ident_sorted.index[y]].isnull().sum()),str(round(missing_ident_sorted.data[y]/train_identity.shape[0]*100)),str(tr_IDdf[missing_ident_sorted.index[y]].dtypes)))


# For this section we will be looking for the missing data in the Transaction data 

# In[5]:


missing_trans=train_transaction.isnull().sum()
missing_trans_sorted=(missing_trans.sort_values(ascending=False))
print('{:15s} {:15s} {:15s} {:15s}'.format("Col Name","# Missing", "% Missing","Data Type"))#print out the columns used 
for z in range(missing_trans.size):#go through each of the columns and get the name, amount missing and the data type and calulate the total amount of the data that is missing 
    print('{:15s} {:15s} {:15s} {:15s}'.format(missing_trans_sorted.index[z],str(tr_TR_df[missing_trans_sorted.index[z]].isnull().sum()),str(round(missing_trans_sorted.data[z]/train_transaction.shape[0]*100)),str(tr_TR_df[missing_trans_sorted.index[z]].dtypes)))


# In this next section we will attempt to combine the two data sets based on the transaction ID to get the full information combined we will do this for just the training at the moment

# In[4]:


train_combined=pd.merge(train_identity,train_transaction,on='TransactionID')#combines the two data frames based on the same Transaction ID
train_combinedDF=pd.DataFrame(train_combined)#Convert to a data frame for better information 
print(train_combinedDF.info())#print out the data frame infomation 


# In this next section we will work on starting to look at the data and how much we are missing which is the first step of cleaning the data
# 
# At the end of this section we will print out our total list of columns in the combined data with the percentage of missing data and the data types the columns are sorted by the most missing to the least 

# In[7]:


percent_missing=train_combined.isnull().sum()#Calculate the percentage missing to the closest percentage
missing_sorted=(percent_missing.sort_values(ascending=False))#Sort the data based on the values of missing data in decreasing order 
print('{:15s} {:15s} {:15s} {:15s}'.format("Col Name","# Missing", "% Missing","Data Type"))#print out the columns used 
for x in range(percent_missing.size):#go through each of the columns and get the name, amount missing and the data type and calulate the total amount of the data that is missing 
    print('{:15s} {:15s} {:15s} {:15s}'.format(missing_sorted.index[x],str(train_combinedDF[missing_sorted.index[x]].isnull().sum()),str(round(missing_sorted.data[x]/train_combined.shape[0]*100)),str(train_combinedDF[missing_sorted.index[x]].dtypes)))


# At this point we can start looking at how each variable affects the outcome of the test. This is called feature selection and we can use a few different methods to try and get our most important variables 

# For the first part we will be seperating the data into two groups one being if there was any fraud or not for the answer and the other being the remaining columns. We seperate them so we can have suprovised learning with the two sets and unsupervised with the non answer set. 

# In[5]:


ans=train_combined['isFraud']#make ans the column with the answers in it 
dat=train_combined.drop(columns=('isFraud'))# the data without the answer is seperated into the dat data type


# Once we have our seperated data we can start to perform the selection process with feature importance which gives a score based on how relevent the data is 

# In[6]:


#to use chi squared we have to convert the categorical data into something we can use such as 
enc=LabelEncoder()# create the label encoder
for cols in dat:#loop through all of the columns
    if dat[cols].dtype.name=="object":#if the data is in a string format we will need to convert it to numeric to find the correlation 
        enc.fit(dat[cols].astype(str))#fit the column to the encoder to convert to numeric 
        dat[cols]=enc.fit_transform(dat[cols].astype(str))#transform the data into numeric 
       


# for this part we use the built in correlation function to see the corelation between the variables 

# In[7]:


corrmat=train_combined.corr()#create a correlation matrix 
top_corr_features=corrmat.index#assign the column names from the data to the matirx
print(corrmat)# print out the matrix. This was very large and did not make a good graph 

#Once we have the correlation matrix we need to better understand what is going on and understand which variables are the most related to the answer 
# In[57]:


print("Correlation of variables to the isFraud variable sorted based on absolute value\n")
corr_sort=(corrmat['isFraud'].abs().sort_values(kind="quicksort",ascending=False))#sort the variables decending by their abolute value as some are negativly or positivly correlated
org=corrmat['isFraud']#create a column of just the isFroud correlations
col=[]#make two arrays to gather the sorted data
dat=[]
for t in range(corr_sort.size):#go though all of the data and assign the index to the index and the origional number from the unsorted array to keep the sign 
    col.append(corr_sort.index[t])
    dat.append(org[corr_sort.index[t]])

sorted_corr=pd.Series(dat,index=col)#create a new series with the index and numbers with the origional sign 
for m in range(sorted_corr.size):
    print('{:15s} {:15s}'.format(sorted_corr.index[m],str(sorted_corr[m])))#in order to print the whole thing we need a for loop to not have it concatinated

