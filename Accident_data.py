'''Project name : AI_Employability
File name : test_resumepar.py
Created on : 20th March,2019
Author : Kshitiz'''

import os
import pandas as pd
import numpy as np
os.chdir("C:\\Users\\91920\\Downloads")
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

#####  Reading data from the csv file

a2017=pd.read_csv("2017.csv")
a2016=pd.read_csv("2016.csv",encoding = "ISO-8859-1")
a2015=pd.read_csv("2015.csv",encoding = "ISO-8859-1")
a2014=pd.read_csv("2014.csv",encoding = "ISO-8859-1")
a2013=pd.read_csv("2013.csv",encoding = "ISO-8859-1")
a2012=pd.read_csv("2012.csv",encoding = "ISO-8859-1")
a2011=pd.read_csv("2011.csv",encoding = "ISO-8859-1")
a2010=pd.read_csv("2010.csv",encoding = "ISO-8859-1")
a2009=pd.read_csv("2009.csv",encoding = "ISO-8859-1")

# Making small changes
a2014['Casualty Class']=a2014['Unnamed: 11']
a2013['Casualty Class']=a2013['Unnamed: 11']
#Removing unnecessary columns
a2017.drop(['Reference Number','Grid Ref: Easting','Grid Ref: Northing'],axis=1,inplace=True)
a2016.drop(['Reference Number','Grid Ref: Easting','Grid Ref: Northing','Expr1'],axis=1,inplace=True)
a2015.drop(['Reference Number','Grid Ref: Easting','Grid Ref: Northing'],axis=1,inplace=True)
a2014.drop(['Reference Number','Grid Ref: Easting','Grid Ref: Northing','Unnamed: 11'],axis=1,inplace=True)
a2013.drop(['Reference Number','Grid Ref: Easting','Grid Ref: Northing','Unnamed: 11'],axis=1,inplace=True)
a2012.drop(['Reference Number','Easting','Northing'],axis=1,inplace=True)
a2011.drop(['Reference Number','Easting','Northing'],axis=1,inplace=True)
a2010.drop(['Reference Number','Easting','Northing'],axis=1,inplace=True)
a2009.drop(['Reference Number','Easting','Northing'],axis=1,inplace=True)



#Correcting column names

a2013.rename(columns={'Unnamed: 15':'Type of Vehicle'},inplace=True)
a2017.rename(columns={'1st Road Class & No':'1st Road Class'},inplace=True)

## Concat all the data from different years

accident_data=pd.concat([a2017,a2016,a2015,a2014,a2013,a2012,a2011,a2010,a2009],axis=0)

## Target is to predict Casualty Severity

target=accident_data['Casualty Severity']

target=target.astype(object)

accident_data['Road Surface'] = accident_data['Road Surface'].astype(object)
accident_data['Casualty Severity'] = accident_data['Casualty Severity'].astype(object)
accident_data['Casualty Class'] = accident_data['Casualty Class'].astype(object)
accident_data['Type of Vehicle'] = accident_data['Type of Vehicle'].astype(object)
accident_data['Weather Conditions'] = accident_data['Weather Conditions'].astype(object)

accident_data1=accident_data.loc[:, accident_data.columns != 'Casualty Severity']
accident_data1.drop(['Accident Date','1st Road Class','Type of Vehicle'],inplace=True,axis=1)
#accident_data1.drop(['Road Surface'],inplace=True,axis=1)
accident_data1.head()

SS=StandardScaler()
accident_data2=accident_data1.loc[:,['Age of Casualty','Number of Vehicles','Time (24hr)']]
accident_data2=SS.fit_transform(accident_data2)
accident_data2=pd.DataFrame(accident_data2,columns=['Age of Casualty','Number of Vehicles','Time (24hr)'])
accident_data1[['Age of Casualty','Number of Vehicles','Time (24hr)']]=accident_data2
accident_data1=accident_data1.replace('Wet / Damp','WetDamp')
accident_data1=accident_data1.replace('Frost / Ice','FrostIce')
accident_data1=accident_data1.replace('Frost/Ice','FrostIce')
accident_data1=accident_data1.replace('Wet/Damp','WetDamp')
accident_data1=accident_data1.replace('5','S')

accident_data1=accident_data1.replace('Frost/ Ice','FrostIce')
accident_data1=accident_data1.replace('Flood (surface water over 3cm deep)','Flood')
accident_data1.head()


accident_data1['Road Surface'].value_counts()

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
for column_name in accident_data1.columns:
    if accident_data1[column_name].dtype == object:
        accident_data1[column_name] = le.fit_transform(accident_data1[column_name])
    else:
        pass


X_train, X_test, y_train, y_test = train_test_split(accident_data1,target,test_size=0.3,random_state=42)

model=RandomForestClassifier()
model.fit(X_train,y_train)

s=model.predict(X_test)

confmat=confusion_matrix(s,Y_test)
        # cross validation score
CV_score=cross_val_score(model,X=X_train,y=y_train,cv=10)
        