# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 21:51:09 2022

@author: patzo
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

 
data=pd.read_csv(r'M:\OMSA\ISYE6740\Final Project\DA_Final.csv')
data=data.dropna()

print(data)

#flitering required columns

df1=data[['location','product_type','PLI','tot_weeks','YOF']]

 

#data cleanup
df1['PLI']=df1['PLI'].str.replace("-","")
df1=df1[df1.PLI.apply(lambda x: x.isnumeric())]
df1.columns = df1.columns.str.lstrip()
df1.columns = df1.columns.str.rstrip()
df1.dtypes
 

#catergorical encoding
df1['location']=df1['location'].astype('category')
df1['loc_cat']=df1['location'].cat.codes
df1['product_type']=df1['product_type'].astype('category')
df1['product_type_cat']=df1['product_type'].cat.codes

 

#converting dtypes
df1['loc_cat']=df1['loc_cat'].astype(float)
df1['product_type_cat']=df1['product_type_cat'].astype(float)
df1['PLI']=df1['PLI'].astype(float)
df1['tot_weeks']=df1['tot_weeks'].astype(float)

 

#clean dataset
data_cln=df1[['loc_cat','product_type_cat','PLI','tot_weeks','YOF']]

 

#check for coorelation
corr=data_cln.corr()
plt.imshow(corr)
plt.show()

 

#since column3 and 4 have heavy correlation we will just use one
final_data=df1[['loc_cat','product_type_cat','PLI','YOF']]
print(final_data)

 

#Scale data
scaled_data=StandardScaler().fit_transform(final_data)

 

#find best K to reduce Sum of Squared errors (SSE)
kmeans_kargs={"init":"random","n_init":10,"max_iter":300}
sse=[]
for k in range(1,30):
    kmeans=KMeans(n_clusters=k,**kmeans_kargs)
    kmeans.fit(scaled_data)
    sse.append(kmeans.inertia_)

 

plt.plot(range(1, 30), sse)
plt.xticks(range(1,30))
plt.show()

 

#Running the best fit K, 23 seems to be the elbow
kmeans=KMeans(init="random",n_clusters=23,n_init=10,max_iter=300)
kmeans.fit(scaled_data)
print(kmeans.inertia_)

 

#Show clusters in current data
label=kmeans.fit_predict(scaled_data)

#add to origianl database
clusters=final_data.copy()
clusters['cluster']=label

 

#keys
keys=df1[['location','product_type','PLI','tot_weeks','YOF']]
print(keys)

 

#merge clusters and keys
#result=pd.merge(clusters,keys,on=['site_cat','component_type_cat'])
#print(result)

 

#export to excel
clusters.to_excel(r"M:\OMSA\ISYE6740\Final Project\clusters.xlsx", index = False)
keys.to_excel(r"M:\OMSA\ISYE6740\Final Project\keys.xlsx", index = False)
