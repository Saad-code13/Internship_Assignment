# -*- coding: utf-8 -*-
"""
Created on Mon May 23 19:51:32 2022

@author: saad
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OrdinalEncoder
import pickle

df = pd.read_csv('DataSets/Clean_DataSet.csv')

df= df.drop(columns={'Unnamed: 0'})

cat_columns = ['airline','source_city','departure_time','stops','arrival_time','destination_city','class',]
num_columns = ['duration','days_left']

oe = OrdinalEncoder()
encoder = oe.fit_transform(df[cat_columns])
encoder = pd.DataFrame(encoder,columns = cat_columns)
pickle.dump(oe,open('Pickle/encoder.pkl','wb'))

X = pd.concat([encoder,df[num_columns]],axis=1)
y= df[['price']]

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2,random_state = 20)

dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)

pickle.dump(dt,open('Pickle/model.pkl','wb'))


