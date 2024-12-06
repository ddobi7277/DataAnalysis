#The goal is make a model that could predict for the features the Severity of the Accident
#And analyze the most important features that could lead to an accident.
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sbn
import lazypredict
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV 
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score,roc_auc_score #scores metrics measures
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay #matrix confusion
from xgboost import plot_importance
from lazypredict.supervised import LazyClassifier
import joblib
import shap
import torch
import torch.nn as nn
import torch.utils.data as td
from scipy.stats import shapiro

#Loading the data 2.59 GB
data= pd.read_csv('./US_Accidents_March23.csv')

#EDA
data.info() # The dataset have bool(13), float64(12), int64(1), object(20) and some missing values
data.shape
data.describe()
#There are 7 million+  rows and 46 columns
#there are 2 columns that are not so important for us
#Let's do a copy of our dfset
df= data.copy()
#Let's drop the columns that are not so important for us
df= df.drop(['ID', 'Source'], axis=1)

#Now let's check if there are missing values
df.isna().sum()

# There are a lot of missing values in the dfset, we need to handle them

df= df.dropna()


#After drop all the Null values the dfset now have 3,554,549 rows and 44 columns


#The only rare here was that there was accident with 0 miles of Distance.

#Like we have the Street city and county i think that for this analysis the lat and long and description stats are not to importants
#So we will drop them 
df= df.drop(['Start_Lat', 'Start_Lng','End_Lat', 'End_Lng','Description','Country'], axis=1)

#Now it's time to make the Start_time and End_time to datetime
#df['Start_Time'] = pd.to_datetime(df['Start_Time'])
#df['End_Time'] = pd.to_datetime(df['End_Time'])
#At the time to make Start_Time to datetime i get an error that i have a format "%Y-%m-%d %H:%M:%S": ".000000000", at position 9262.
#So let's go to fixe it.

#First get the rows with the probelm

print(df[df['Start_Time'].str.contains('.000000000')]['Start_Time'])
print(df[df['End_Time'].str.contains('.000000000')]['End_Time'])
#The problem is that the row have a value of 0.000000000 at the end they're more than one so it's time to fix it.

df['Start_Time']= df['Start_Time'].astype(str).str.split('.').str[0]
df['End_Time']= df['End_Time'].astype(str).str.split('.').str[0]

#Now let's check if the problem was solved

print(df[df['Start_Time'].str.contains('.000000000')]['Start_Time'])
print(df[df['End_Time'].str.contains('.000000000')]['End_Time'])

#We already fix it so now we can proced to convert to datetime

df['Start_Time'] = pd.to_datetime(df['Start_Time'])
df['End_Time'] = pd.to_datetime(df['End_Time'])

#Let's check if it worked

df.info()#It worked

#Convert those columns to string
for i in ['County', 'State', 'Zipcode', 'Country','Street','City','Timezone','Airport_Code']:
   df[i]= df[i].astype('string')
   
#Let's check if it worked

df.info() #It worked

#Now let's check the datetimes columns but we gonna add the Weather_Condition feature 

df[['Start_Time','End_Time','Weather_Condition','Weather_Timestamp']]
#All have the same format but sometimes the Weather_Timestamp don't have
# the same time of when the accident started.
#But I don't think that the Weather_Timestamp will be relevant 
df= df.drop('Weather_Timestamp',axis=1)

#Now let's encode the boolean features and the others features

def encodeBooleans(d,value):
    df[d]= np.where(df[d] == value,1,0)
    
for i in df.columns:
    if ((df[i].dtype) == bool):
        encodeBooleans(i,True)
        
#Now let's check the object types of the dfset

for i in df.columns:
   if df[i].dtype == object:
       print(i+'\n')
       print(df[i].unique())
       print('-------------------------------')
       
#We can see that Sunrise_Sunset Civil_Twilight ,Nautical_Twilight and Astronomical_Twilight
#have the same values 'Night' and 'Day'
#So let's encode dem with day =1 and night = 0

for i in df.columns:
   if df[i].dtype == object and len(df[i].unique()) == 2:
       encodeBooleans(i,'Day')
       

#Now I noticed that the name of the columns are a littel bit complicate so let's change 
#the name of the columns to something more simple and clear

df= df.rename(columns={'Distance(mi)':'Distance','Temperature(F)':'Temperature',
                           'Wind_Chill(F)':'Wind_Chill','Humidity(%)':'Humidity',
                           'Pressure(in)':'Pressure','Visibility(mi)':'Visibility',
                           'Wind_Speed(mph)':'Wind_Speed','Precipitation(in)':'Precipitation'})

#now let's convert to string Weather_condition and Wind_direction 

df['Wind_Direction']= df['Wind_Direction'].astype('string')
df['Weather_Condition']= df['Weather_Condition'].astype('string')  
                     
#Now let's get the dumies for Weather_condition

dumies= pd.get_dummies(df['Weather_Condition'])
#Now let's add the dumies to the df
df = pd.concat([df,dumies],axis=1)

#And now we can drop the Weather_COndition feature
df = df.drop('Weather_Condition',axis=1)

#Now let's make the same procedure with the new boolean values
for i in df.columns:
    if ((df[i].dtype) == bool):
        encodeBooleans(i,True)
    
df.info()#To check if it work, it did

#Now I'm gonna check for Wind_direction feature 

df['Wind_Direction'].unique()
#This is the result

'''
array(['SW', 'WSW', 'West', 'NNW', 'WNW', 'NW', 'W', 'SSW', 'East', 'SE',
       'North', 'ENE', 'NNE', 'NE', 'SSE', 'CALM', 'South', 'ESE', 'S',
       'Variable', 'VAR', 'N', 'E'], dtype=object)
'''
#There are the same direction with different names so lets format in only one mode like N or SW
#I gonna  create a mapping dictionary to efficiently replace the values in the 'Wind_Direction' column.

wind_direction_mapping = {
    'Variable': 'VAR',
    'South':'S',
    'North':'N',
    'East':'E',
    'West':'W',
    'North-Northeast': 'NNE',
    'North Northeast': 'NNE',
    'Northeast': 'NE',
    'East-Northeast': 'ENE',
    'East Northeast': 'ENE',
    'East-Southeast': 'ESE',
    'East Southeast': 'ESE',
    'Southeast': 'SE',
    'South-Southeast': 'SSE',
    'South Southeast': 'SSE',
    'South-Southwest': 'SSW',
    'South Southwest': 'SSW',
    'West-Southwest': 'WSW',
    'West Southwest': 'WSW',
    'West-Northwest': 'WNW',
    'West Northwest': 'WNW',
    'North-Northwest': 'NNW',
    'North Northwest': 'NNW',
    'CALM': 'CALM',
}        
df['Wind_Direction'] = df['Wind_Direction'].replace(wind_direction_mapping)

#Now let's check the result
df['Wind_Direction'].unique()
'''
['SW', 'WSW', 'W', 'NNW', 'WNW', 'NW', 'SSW', 'E', 'SE', 'N', 'ENE',
       'NNE', 'NE', 'SSE', 'CALM', 'S', 'ESE', 'VAR']
'''
#It works, 


#Now let's get dumies,encode and concat Wind direction

dumies_direction= pd.get_dummies(df['Wind_Direction'])

for i in dumies_direction.columns:
    if ((dumies_direction[i].dtype) == bool):
        dumies_direction[i]= np.where(dumies_direction[i] == True,1,0)

df= pd.concat([df,dumies_direction],axis=1)

#And now we can drop Wind_direction
df= df.drop('Wind_Direction',axis=1)


#Then let's check the Zipcode feature
df['Zipcode'].unique()
#There are a lot unique zipcodes, but there are two format xxxxx or xxxxx-xxxx so I'll let 
#the format xxxxx 
df['Zipcode']= df['Zipcode'].str.split('-').str[0]

#Let's check if it worked

df['Zipcode'][df['Zipcode'].str.contains('-')].str.split('-').head(10)

#It returned Series([], Name: Zipcode, dtype: object) so It works
#Now it's time to convert to string 
df['Zipcode']= df['Zipcode'].astype('string')

#Now it's time for state feature first lets create a dictionary with the code of state and the value 1-49 

states= {}
for i, state in enumerate(df['State'].unique(), start=1):
   states[state]= i

#Now let's replace or encode


