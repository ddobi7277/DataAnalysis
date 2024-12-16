# %% [markdown]
# The goal is make a model that could predict for the features the Severity of the Accident
# and analyze the most important features that could lead to an accident.

# %% [markdown]
# Let's start importing everuthing that we need

# %%
import pandas as pd
import numpy as np 
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV 
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score,roc_auc_score,classification_report #scores metrics measures
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay #matrix confusion
from xgboost import plot_importance
from sklearn.linear_model import LogisticRegression
from lazypredict.supervised import LazyClassifier
import joblib
import shap
import torch
import torch.nn as nn
import torch.utils.data as td
from scipy.stats import shapiro

# %%
#Loading the data 2.59 GB
data= pd.read_csv('./US_Accidents_March23.csv')

# %%
#EDA
data.info()

# %% [markdown]
# The dataset have bool(13), float64(12), int64(1), object(20) and some missing values
# 

# %%
data.shape

# %%
data.describe()

# %% [markdown]
# There are 7 million+  rows and 46 columns and there are 2 columns that are not so important for us ID and Source but first let's make a copy of our data to work with the copy .

# %%
df= data.copy()

# %% [markdown]
# #Let's drop the columns that are not so important for us

# %%
df= df.drop(['ID', 'Source'], axis=1)

# %% [markdown]
# Now let's check if there are missing values

# %%
df.isna().sum()

# %% [markdown]
# There are a lot of missing values in the dfset, we need to handle them

# %%
df= df.dropna()

# %%
df.info()

# %% [markdown]
# After drop all the Null values the dfset now have 3,554,549 rows and 44 columns the only rare here was that there was accident with 0 miles of Distance but I'll touch that later,now like I have the Street, city,county and State I think that for this analysis the lat and long and description stats are not to importants so we will drop them, and like all the accidents are in USA I'll drop Country too 

# %%
df= df.drop(['Start_Lat', 'Start_Lng','End_Lat', 'End_Lng','Description','Country'], axis=1)

# %% [markdown]
# Now it's time to make the Start_time and End_time to datetime

# %%
#df['Start_Time'] = pd.to_datetime(df['Start_Time'])
#df['End_Time'] = pd.to_datetime(df['End_Time'])

# %% [markdown]
# At the time to make Start_Time to datetime i get an error that i have a format "%Y-%m-%d %H:%M:%S": ".000000000", at position 9262. So let's go to fixe it.

# %%

#The problem is that the row have a value of 0.000000000 at the end they're more than one so it's time to fix it.

df['Start_Time']= df['Start_Time'].astype(str).str.split('.').str[0]
df['End_Time']= df['End_Time'].astype(str).str.split('.').str[0]

#Now let's check if the problem was solved

print(df[df['Start_Time'].str.contains('.000000000')]['Start_Time'].head(5))
print(df[df['End_Time'].str.contains('.000000000')]['End_Time'].head(5))


# %% [markdown]
# We already fix it so now we can proced to convert to datetime

# %%
df['Start_Time'] = pd.to_datetime(df['Start_Time'])
df['End_Time'] = pd.to_datetime(df['End_Time'])

# %%
#Let's check if it worked

df.info()#It worked

# %% [markdown]
# Now I noticed that the name of the columns are a littel bit complicate so let's change the name of the columns to something more simple and clear

# %%
df= df.rename(columns={'Distance(mi)':'Distance','Temperature(F)':'Temperature',
                           'Wind_Chill(F)':'Wind_Chill','Humidity(%)':'Humidity',
                           'Pressure(in)':'Pressure','Visibility(mi)':'Visibility',
                           'Wind_Speed(mph)':'Wind_Speed','Precipitation(in)':'Precipitation'})

# %% [markdown]
# Now I got 'County', 'State', 'Zipcode','Street','City','Timezone','Airport_Code' as a object so I'll convert to string

# %%
for i in ['County', 'State', 'Zipcode','Street','City','Timezone','Airport_Code']:
   df[i]= df[i].astype('string')

# %% [markdown]
# Let's check if it worked
# 

# %%
df.info()

# %% [markdown]
# It worked and now let's check the datetimes columns but we gonna add the Weather_Condition feature 

# %%
df[['Start_Time','End_Time','Weather_Condition','Weather_Timestamp']]

# %% [markdown]
# All have the same format but sometimes the Weather_Timestamp don't have the same time of when the accident started.
# But I don't think that the Weather_Timestamp will be relevant so I'm gonna drop it.

# %%
df= df.drop('Weather_Timestamp',axis=1)

# %% [markdown]
# Now It's time to encode the boolean and string features

# %%
def encodeBooleans(d,value):
    df[d]= np.where(df[d] == value,1,0)
    
for i in df.columns:
    if ((df[i].dtype) == bool):
        encodeBooleans(i,True)

# %% [markdown]
# Now let's check the object types of the dataset

# %%
for i in df.columns:
   if df[i].dtype == object:
       print(i+'\n')
       print(df[i].unique())
       print('-------------------------------')

# %% [markdown]
# We can see that **Sunrise_Sunset Civil_Twilight ,Nautical_Twilight and Astronomical_Twilight** have the same values 'Night' and 'Day'. Since all four features (Sunrise_Sunset, Civil_Twilight, Nautical_Twilight, and Astronomical_Twilight) have the same values (Night and Day), it's likely that they are capturing the same information.
# 
# In this case, I will make a new feature called Moment_of_Day that could be Morning, Afternoon, Evening, Night, as the others would be redundant. This is an example of feature redundancy, where multiple features are highly correlated or capture the same information.
# 

# %%
df= df.drop(['Civil_Twilight','Nautical_Twilight','Astronomical_Twilight','Sunrise_Sunset'],axis=1)

#And now I'll create the new feature

df['Moment_of_Day'] = pd.cut(
    df['Start_Time'].dt.hour,
    bins= [0,6,12,18,24],
    labels=['Night', 'Morning', 'Afternoon', 'Evening'],
    include_lowest=True
    )

# %% [markdown]
# Now I'll create the feature Moment_of_Day_Encoded to work with the model that will be as Night = 0 , Morning = 1 , Afternoon = 2, Evening = 3, and I'll let the Moment_of_Day to work with vizualitations.

# %%
df['Moment_of_Day_Encoded']= df['Moment_of_Day'].map({'Night':0, 'Morning':1, 'Afternoon':2, 'Evening':3})
df['Moment_of_Day_Encoded']= df['Moment_of_Day_Encoded'].astype('int64')

# %% [markdown]
# Now let's convert to string Weather_condition and Wind_direction 
# 

# %%
df['Wind_Direction']= df['Wind_Direction'].astype('string')
df['Weather_Condition']= df['Weather_Condition'].astype('string')  


# %% [markdown]
# Now I have to many Weather conditions so I will classify into 3 types :
# - Low: Favorable conditions, clear skies or light cloud cover or no precipitacion or light rain
# - Normal: Moderate Conditions like partially clody skies or moderate clouds cover.
# - High: Unfavorable conditions, heavy rain, thunderstorms, strong winds, etc.

# %% [markdown]
# Now I gonna code all the values with Heavy , Thunderstorms,Fog, Haze , T-Storm , Tornado as a high too.

# %%

df["Weather_Condition"] = np.where(df['Weather_Condition'].str.contains('Heavy'),'High',df['Weather_Condition'])
df["Weather_Condition"] = np.where(df['Weather_Condition'].str.contains('Thunderstorms'),'High',df['Weather_Condition'])
df["Weather_Condition"] = np.where(df['Weather_Condition'].str.contains('Thunderstorm'),'High',df['Weather_Condition'])
df["Weather_Condition"] = np.where(df['Weather_Condition'].str.contains('Fog'),'High',df['Weather_Condition'])
df["Weather_Condition"] = np.where(df['Weather_Condition'].str.contains('Haze'),'High',df['Weather_Condition'])
df["Weather_Condition"] = np.where(df['Weather_Condition'].str.contains('T-Storm'),'High',df['Weather_Condition'])
df["Weather_Condition"] = np.where(df['Weather_Condition'].str.contains('Tornado'),'High',df['Weather_Condition'])
df["Weather_Condition"] = np.where(df['Weather_Condition'].str.contains('Funnel Cloud'),'High',df['Weather_Condition'])


print(df['Weather_Condition'][df['Weather_Condition'].str.contains('Heavy')].unique())
print(df['Weather_Condition'][df['Weather_Condition'].str.contains('Thunderstorms')].unique())
print(df['Weather_Condition'][df['Weather_Condition'].str.contains('Thunderstorm')].unique())
print(df['Weather_Condition'][df['Weather_Condition'].str.contains('Fog')].unique())
print(df['Weather_Condition'][df['Weather_Condition'].str.contains('Haze')].unique())
print(df['Weather_Condition'][df['Weather_Condition'].str.contains('T-Storm')].unique())
print(df['Weather_Condition'][df['Weather_Condition'].str.contains('Tornado')].unique())
print(df['Weather_Condition'][df['Weather_Condition'].str.contains('Funnel Cloud')].unique())

# %% [markdown]
# Now considering that all the weather conditions are "light" versions, it's also reasonable to classify them as "low risk"

# %%
print(df['Weather_Condition'][df['Weather_Condition'].str.contains('Light')].unique())

df["Weather_Condition"] = np.where(df['Weather_Condition'].str.contains('Light'),'Low',df['Weather_Condition'])

print(df['Weather_Condition'][df['Weather_Condition'].str.contains('Light')].unique())

# %% [markdown]
# And now I'll code all the values with cloudy or clouds or Overcast as Normal risks.
# 

# %%
df["Weather_Condition"] = np.where(df['Weather_Condition'].str.contains('Cloudy'),'Normal',df['Weather_Condition'])
df["Weather_Condition"] = np.where(df['Weather_Condition'].str.contains('Clouds'),'Normal',df['Weather_Condition'])
df["Weather_Condition"] = np.where(df['Weather_Condition'].str.contains('Overcast'),'Normal',df['Weather_Condition'])



print(df['Weather_Condition'][df['Weather_Condition'].str.contains('Cloudy')].unique())
print(df['Weather_Condition'][df['Weather_Condition'].str.contains('Clouds')].unique())
print(df['Weather_Condition'][df['Weather_Condition'].str.contains('Overcast')].unique())


# %% [markdown]
# While Snow conditions can sometimes be hazardous, the conditions of Snow  are generally considered "normal" winter weather conditions.
# 
# In the context of my analysis, classifying these snow-related conditions as "normal" is reasonable, especially because I'm comparing them to more severe weather conditions.
# 
# So, I'll classify those snow-related conditions as "normal"

# %%
df["Weather_Condition"] = np.where(df['Weather_Condition'].str.contains('Snow'),'Normal',df['Weather_Condition'])

print(df['Weather_Condition'][df['Weather_Condition'].str.contains('Snow')].unique())

# %% [markdown]
#  I will classify thunder conditions as "High Risk". Here's why:
# 
# - Thunderstorms and thunder-related events can be hazardous due to the risk of lightning strikes, strong winds, and heavy precipitation.
# - The presence of thunder or thunderstorms often indicates unstable weather conditions, which can increase the risk of accidents.
# 
# While some of these conditions might not always lead to severe consequences, it's generally safer to err on the side of caution and classify them as "High Risk".

# %%
df["Weather_Condition"] = np.where(df['Weather_Condition'].str.contains('Thunder'),'High',df['Weather_Condition'])

print(df['Weather_Condition'][df['Weather_Condition'].str.contains('Thunder')].unique())

# %% [markdown]
# Considering the conditions ['Blowing Dust / Windy' 'Blowing Dust' 'Blowing Sand']
# ['Sand / Dust Whirls Nearby' 'Sand / Dust Whirlwinds' 'Sand / Windy'
#  'Sand / Dust Whirlwinds / Windy' 'Blowing Sand'], I'll classify them as "High Risk". Here's why:
# 
# - Blowing dust, sand, or debris can reduce visibility, making it difficult for drivers to navigate safely.
# - Strong winds can also make vehicles more difficult to control, increasing the risk of accidents.
# 
# While some of these conditions might not always lead to severe consequences, the potential for reduced visibility and increased difficulty in controlling vehicles makes it safer to classify them as "High Risk".

# %%
df["Weather_Condition"] = np.where(df['Weather_Condition'].str.contains('Blowing'),'High',df['Weather_Condition'])
df["Weather_Condition"] = np.where(df['Weather_Condition'].str.contains('Sand'),'High',df['Weather_Condition'])

print(df['Weather_Condition'][df['Weather_Condition'].str.contains('Blowing')].unique())

print(df['Weather_Condition'][df['Weather_Condition'].str.contains('Sand')].unique())


# %% [markdown]
# Then I'll classify the rest as low risk because these conditions are generally considered typical or common weather phenomena that don't usually pose a significant risk to road safety.
# 
# Some of these conditions, like rain or drizzle, might require drivers to exercise caution, but they are not typically associated with a high risk of accidents.
# 

# %%
df.loc[~df['Weather_Condition'].isin(['High', 'Low', 'Normal']), 'Weather_Condition']= 'Low'


# %%
df['Weather_Condition'].unique()

# %% [markdown]
# Now I'll encode Low = 0 , Normal = 1 , High = 2 using the method .map()

# %%
df['Weather_Condition']= df['Weather_Condition'].map({'Low':0, 'Normal':1, 'High':2})

# %%
df['Weather_Condition'].unique()

# %%
df.info()#To check if it work, it did


# %%
for i in df.columns:
    if df[i].dtype == 'int32':
        print(i)
        print(df[i].unique())
        print('-----------------------------')

# %% [markdown]
# As We can see here all the values of turning loop are 0, so it's mean that there are 0 accident that was occurred doing in a turning loop so I think I can drop  Turning_Loop feature, also I think that the feature Airport_Code won't be useful here so I gonna drop it too.

# %%
df= df.drop(['Turning_Loop','Airport_Code'],axis=1)

# %% [markdown]
# Now I'll convert all those values int32 into int64.

# %%
for i in df.columns:
    if df[i].dtype == 'int32':
        df[i] = df[i].astype('int64')

# %%
df.info()

# %% [markdown]
# Now I'm gonna check for Wind_direction feature 

# %%
df['Wind_Direction'].unique()

# %% [markdown]
# There are the same direction with different names so lets format in only one mode like N or SW
# I'm gonna create a mapping dictionary to efficiently replace the values in the 'Wind_Direction' column because I can do that with string if not I could use the map() method.

# %%
wind_direction_mapping = {
    'Variable':'VAR',
    'South':'S',
    'North':'N',
    'East':'E',
    'West':'W',
    'North-Northeast':'NNE',
    'North Northeast':'NNE',
    'Northeast':'NE',
    'East-Northeast':'ENE',
    'East Northeast':'ENE',
    'East-Southeast':'ESE',
    'East Southeast':'ESE',
    'Southeast':'SE',
    'South-Southeast':'SSE',
    'South Southeast':'SSE',
    'South-Southwest':'SSW',
    'South Southwest':'SSW',
    'West-Southwest':'WSW',
    'West Southwest':'WSW',
    'West-Northwest':'WNW',
    'West Northwest':'WNW',
    'North-Northwest':'NNW',
    'North Northwest':'NNW',
    'CALM':'CALM',
}        
df['Wind_Direction'] = df['Wind_Direction'].replace(wind_direction_mapping)

# %% [markdown]
# Now let's check the result

# %%
df['Wind_Direction'].unique()

# %% [markdown]
# Now let's encode Wind direction

# %%
wind_direction_encoded= {'SW':1,'WSW':2,'W':3,'NNW':4,'WNW':5,'NW':6,'SSW':7,'E':8,'SE':9,
    'N':10,'ENE':11,'NNE':12,'NE':13,'SSE':14,'CALM':15,'S':16,'ESE':17,'VAR':18}

df['Wind_Direction']= df['Wind_Direction'].map(wind_direction_encoded)




# %%
df['Wind_Direction'].unique()

# %% [markdown]
# Then let's check the Zipcode feature

# %%
df['Zipcode'].unique()
#There are a lot unique zipcodes, but there are two format xxxxx or xxxxx-xxxx so I'll let 
#the format xxxxx 
df['Zipcode']= df['Zipcode'].str.split('-').str[0]


# %% [markdown]
# Let's check if it worked

# %%
df['Zipcode'][df['Zipcode'].str.contains('-')].str.split('-').head(10)

# %% [markdown]
# Now it's time to convert to string 

# %%
df['Zipcode']= df['Zipcode'].astype('string')

# %% [markdown]
# Now I'll work with State feateure, first lets create a dictionary with the code of state and the value 1-49 

# %%
states= {}
for i, state in enumerate(df['State'].unique(), start=1):
   states[state]= i
   

# %%
states

# %% [markdown]
# Next I'll encode with map() function because theyrn't string

# %%
df['State']= df['State'].map(states)

# %%
#To check if it worked
df['State'].unique()

# %% [markdown]
# The next feature that I'll work with is Timezone

# %%
df['Timezone'].unique()

# %% [markdown]
# I'll encode with the map() -- 'US/Eastern':1, 'US/Pacific':2, 'US/Central':3, 'US/Mountain':4

# %%
df['Timezone'] = df['Timezone'].map({'US/Eastern':1, 'US/Pacific':2, 'US/Central':3, 'US/Mountain':4})

# %%
#To check if It worked
df['Timezone'].unique()

# %% [markdown]
# Now I'll make the new features Duration = The total duration of an  accident in minutes,Date for vizualizations but I'll break down Date into ( Day = day when It happend,Month = month when it happend,Year year when it happend ) all numeric in oreder of the models , and Time the time when it occured that I'll break into hour and minuts and Time will be for visulization analysis. Also I'll create a feature called Holyday that will have values 1 if is or 0 if no.

# %%

df['Duration']= (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60 
df['Date']= df['Start_Time'].dt.date
df['Time']= df['Start_Time'].dt.time
df['Day'] = (df['Start_Time'].dt.day).astype('int64')
df['Month'] = (df['Start_Time'].dt.month).astype('int64')
df['Year'] = (df['Start_Time'].dt.year).astype('int64')
df['Hour'] = (df['Start_Time'].dt.hour).astype('int64')
df['Minute'] = (df['Start_Time'].dt.minute).astype('int64')


# %%
import holidays

us_holidays = holidays.UnitedStates()

def is_holiday(date):
    return date in us_holidays

df['Is_Holiday'] = df['Date'].apply(is_holiday)

df['Is_Holiday']= np.where(df['Is_Holiday'] == True, 1,0)

df['Is_Holiday'] = df['Is_Holiday'].astype('int64')

df.info()


# %% [markdown]
# I think with all we have is enough information of dates so I can drop Start_Time and End_Time features

# %%
df= df.drop(['Start_Time','End_Time'],axis=1)

# %% [markdown]
# Now I'll convert Time feature to string.

# %%
df['Time']= df['Time'].astype('string')
df['Date']= df['Date'].astype('string')

# %% [markdown]
# Now let's check for outliers for variables int64 or float64

# %%
for i in df.columns:
    if df[i].dtype == 'int64' or df[i].dtype == 'float64':
        print(i)
        print(len(df[i].unique()))
        print('----------------------')

# %%
df['Severity'].value_counts(normalize=True)*100

# %%
df.info()

# %% [markdown]
# Here we can see that there are more accident type 2 (94.20 % of the data) than the sum of the rest of the others accident, so we have here a classic problem that is imbalanced classess first let's analyze the features and the relation that have with the target variable that is Severity, after analyze then I'm gonna oversample the dataset to avoid a model with bias of overfitting.

# %%
plt.figure(figsize=(10,5))
sbn.barplot(y=df['Severity'],x= df['Moment_of_Day'],hue=df['Severity'])
plt.title('Severity vs Moment of the day')
plt.show()

# %%
plt.figure(figsize=(10,5))
sbn.barplot(x=df['Severity'],y= df['Duration'])
plt.title('Severity vs Duration')
plt.ylabel('Duration')
plt.xlabel('Severity by Duration')
plt.show()

# %% [markdown]
# We can see here that the accidents with more duration are the accident with high level of severity, definitly Duration could be a good feature to predict the severity, now let's check the others features, but like we have to many I'll only check the strings and float64 and int64 features

# %%
plt.figure(figsize=(10,5))
sbn.barplot(x=df['Severity'],y= df['Distance'],hue=df['Distance'])
plt.title('Severity vs Distance')
plt.ylabel('Distance')
plt.xlabel('Severity by Distance')
plt.show()

# %%
plt.figure(figsize=(10,5))
sbn.histplot(x=df['Severity'],y= df['Distance'])
plt.title('Severity vs Distance')
plt.ylabel('Distance')
plt.xlabel('Severity by Distance')
plt.show()

# %%
df.groupby('Severity')['Distance'].value_counts().sort_values()

# %%
(df.groupby('Severity')['Distance'].mean())

# %% [markdown]
# It's seem that we have more accidents with severity type2 and those are the accidents with more distance but thoso have a lot of accident with 0 distance too and maby for that the highest average of distance is for the accidents type 4 so there are a lot of outliers because the accidents to be an accident have to ve at least 0.1 mi of distance. Let's try now with Temperature feature.

# %%
plt.figure(figsize=(10,5))
sbn.barplot(x=df['Severity'],y= df['Temperature'],hue=df['Temperature'])
plt.title('Severity vs Temperature')
plt.show()

# %% [markdown]
# Here we can see that should be a some outliers in temperature with values that reach even more than 150 F that no human could survive, and It's seems that most accident of lelv 2 on severity occurs in values of temp < 0, now I'll check Wind_chill

# %%
plt.figure(figsize=(10,5))
sbn.histplot(x=df['Severity'],y= df['Wind_Chill'])
plt.title('Severity vs Wind Chill')
plt.ylabel('Wind Chill')
plt.xlabel('Severity')
plt.show()

# %% [markdown]
# Everything seems ok here. Now let's check Humidity feature

# %%
plt.figure(figsize=(10,5))
sbn.histplot(x=df['Severity'],y= df['Humidity'])
plt.title('Severity vs Humidity')
plt.ylabel('Humidity')
plt.xlabel('Severity')
plt.show()

# %%
plt.figure(figsize=(10,5))
sbn.histplot(x=df['Severity'],y= df['Pressure'])
plt.title('Severity vs Pressure')
plt.ylabel('Pressure')
plt.xlabel('Severity')
plt.show()

# %%
plt.figure(figsize=(10,5))
sbn.histplot(x=df['Severity'],y= df['Visibility'])
plt.title('Severity vs Visibility')
plt.ylabel('Visibility')
plt.xlabel('Severity')
plt.show()

# %%
plt.figure(figsize=(10,5))
sbn.scatterplot(x=df['Severity'],y= df['Visibility'])
plt.title('Severity vs Visibility')
plt.ylabel('Visibility')
plt.xlabel('Severity')
plt.show()

# %%
df.groupby('Severity')['Visibility'].value_counts().sort_values()

# %%
df.groupby('Severity')['Visibility'].mean().sort_values()

# %% [markdown]
# There are a lot of accident in the level 2 and 4 with low visibility and the lower average of visibility is for the accidents lvl 4 and then the lvl 2.

# %%
plt.figure(figsize=(10,5))
sbn.barplot(x=df['Severity'],y= df['Wind_Speed'])
plt.title('Severity vs Wind speed')
plt.ylabel('Wind Speed')
plt.xlabel('Severity')
plt.show()

# %%
plt.figure(figsize=(10,5))
sbn.barplot(x=df['Severity'],y= df['Precipitation'])
plt.title('Severity vs Precipitation')
plt.ylabel('Precipitation')
plt.xlabel('Severity')
plt.show()

# %% [markdown]
#  Here I can see that there are a lot of accident Severity lvl 3. Now I'll proced to check for outliers.

# %%
for i in df.columns:
    if df[i].dtype == 'int64' or df[i].dtype == 'float64':
        plt.figure(figsize=(10,5))
        sbn.boxplot(x= df[i])
        plt.title(f"Box plot of {i}")
        plt.xlabel(i)
        plt.show()

# %% [markdown]
# As we can see in the boxplots I have a lot of outliers like:
# - Distance: I don't think that an accident has more than 1 or 2 milles even mega accident will be like 6 to 20 milles but no 60 miles.
# - Duration: I don't think that an accident has more than 1 or 2 hours even the mefa accident could be like 8 hours that is in an extreme situation.
# - Temperature and Wind_Child: The coldest state in USA is Alaska with 26 F averag and the hottest State is Florida araund 74 F average so I have values here of less than 26 and more than 74.
# - Pressure: The average pressure is 29 to 30.
# - Wind_Speed: The average wind speed of a Hurricane Category 1 is 74 - 95 mph and we can cleary see in the boxplot for Wind Speed that it have some values of more than that.
# 
# So, I have outliers in a few variables. I don't want to remove them. I gonna try to replacing those outliers with average values.

# %%
df['Distance'].describe()

# %% [markdown]
# Removing outliers from Distance, the 0 values with the mean, then the I'll set the lower and upper limit and I'll replace the upper values with the upper limit

# %%
df.loc[df['Distance'] == 0,'Distance']= df['Distance'].mean()

q1= df['Distance'].quantile(0.25)
q3= df['Distance'].quantile(0.75)
iqr = q3 - q1 #Interquantile

#Now I'll set the upper limit
upper_limit= q3 + (1.5 * iqr)

df.loc[df['Distance'] > upper_limit,'Distance']= upper_limit

#Now we chechk for outliers again
plt.figure(figsize=(5,2))
sbn.boxplot(x= df['Distance'])
plt.title('Distance Boxplot')
plt.show()

# %% [markdown]
# There are no more outliers in Distance now is time for Duration.
# 

# %%
q1= df['Duration'].quantile(0.25)
q3= df['Duration'].quantile(0.75)
iqr = q3 - q1 #Interquantile

#Now I'll set the upper limit
upper_limit= q3 + (1.5 * iqr)
#lower_limit= q1 - (1.5 * iqr)

#Now replace all values grater than the upper limit 

df.loc[df['Duration'] > upper_limit, 'Duration'] = upper_limit

#Now we check again for outliers
plt.figure(figsize=(5,2))
sbn.boxplot(x= df['Duration'])
plt.title('Duration Boxplot')
plt.show()

#print(f"upper: {upper_limit} -- lower: {lower_limit}")

# %% [markdown]
# Distance fixed, now it's time for Temperature and then Wind Child.

# %%
print(df['Temperature'].describe())
q1= df['Temperature'].quantile(0.25)
q3= df['Temperature'].quantile(0.75)
iqr = q3 - q1 #Interquantile

#Now I'll set the upper limit
upper_limit= q3 + (1.5 * iqr)
lower_limit= q1 - (1.5 * iqr)

print(f"upper: {upper_limit} -- lower: {lower_limit}")
#The coldest temperature in habitable zone in USA in 2024 was -18 F and the hottest was 129 
# and the lower and upper limits are pretty close to this so I'll replace all values that are
#above or below the limits and I'll replace with the limits
#Now replace all values grater than the upper limit 

df.loc[df['Temperature'] > upper_limit, 'Temperature'] = upper_limit
df.loc[df['Temperature'] < lower_limit, 'Temperature'] = lower_limit

#Now we check again for outliers
plt.figure(figsize=(5,2))
sbn.boxplot(x= df['Temperature'])
plt.title('Temperature Boxplot')
plt.show()


# %% [markdown]
# Now temperature is fixed with Wind_Chill should be similar

# %%
print(df['Wind_Chill'].describe())
q1= df['Wind_Chill'].quantile(0.25)
q3= df['Wind_Chill'].quantile(0.75)
iqr = q3 - q1 #Interquantile

#Now I'll set the upper limit
upper_limit= q3 + (1.5 * iqr)
lower_limit= q1 - (1.5 * iqr)

print(f"upper: {upper_limit} -- lower: {lower_limit}")

df.loc[df['Wind_Chill'] > upper_limit, 'Wind_Chill'] = upper_limit
df.loc[df['Wind_Chill'] < lower_limit, 'Wind_Chill'] = lower_limit

#Now we check again for outliers
plt.figure(figsize=(5,2))
sbn.boxplot(x= df['Wind_Chill'])
plt.title('Wind Chill Boxplot')
plt.show()

# %% [markdown]
# Now Wind Child is fixed too. Now it's time for Pressure variable.

# %%
print(df['Pressure'].describe())
q1= df['Pressure'].quantile(0.25)
q3= df['Pressure'].quantile(0.75)
iqr = q3 - q1 #Interquantile
#Now I'll set the upper limit
upper_limit= q3 + (1.5 * iqr)
lower_limit= q1 - (1.5 * iqr)

print(f"upper: {round(upper_limit)} -- lower: {round(lower_limit)}")
#The normal range of air pressure at sea level is considered to be between 29.5 and 30.2 inches of mercury
#We will use this to filter out the data that is outside of this range and
#like the upper and lower limits are pretty close to those values 
# I'll replace the values outside of this range for the upper and lower limits

df.loc[df['Pressure'] > upper_limit, 'Pressure'] = upper_limit
df.loc[df['Pressure'] < lower_limit, 'Pressure'] = lower_limit

#Now we check again for outliers
plt.figure(figsize=(5,2))
sbn.boxplot(x= df['Pressure'])
plt.title('Pressure Boxplot')
plt.show()


# %% [markdown]
# Now the Pressure is fixed then I'll work with Wind_Speed. The problem here is the winds of more than 46 mph

# %%
print(df['Wind_Speed'].describe())

#Now I'll filter the data to remove outliers with the winds of more than 46 mph and I'll replace by the limit that is 46 mph.

df.loc[df['Wind_Speed'] > 46, 'Wind_Speed'] = 46

#Now we check again for outliers
plt.figure(figsize=(5,2))
sbn.boxplot(x= df['Wind_Speed'])
plt.title('Wind_Speed Boxplot')
plt.show()

# %% [markdown]
# Now It's fixed with some some outliers in the box plot but those winds are probables so they're not outliers at all. Now that I already fix the outliers problems I will check the balance of the dataset again.

# %%
df['Severity'].value_counts()

# %% [markdown]
# I gonna make a Label encoding to Street, City, County, and Zipcode

# %%
df['Street'] = pd.factorize(df['Street'])[0]
df['City'] = pd.factorize(df['City'])[0]
df['County'] = pd.factorize(df['County'])[0]
df['Zipcode'] = pd.factorize(df['Zipcode'])[0]


# %%
df.info()

# %% [markdown]
# So as I noticed early there is a classic problem of imbalance because the severity #2 is the 94 % of the dataset, so we have to fix that, to fix it I'll  duplicate the minority class samples randomly or known as Random Oversampling
# 

# %%
level1Severity= pd.concat([df[df['Severity'] == 1]]*10,ignore_index=True)
level3Severity= pd.concat([df[df['Severity'] == 3]]*9,ignore_index=True)
level4Severity= pd.concat([df[df['Severity'] == 4]]*9,ignore_index=True)

oversampled_df= pd.concat([df,level1Severity,level3Severity,level4Severity],ignore_index=True)

# %%
oversampled_df['Severity'].value_counts(normalize=True)

# %% [markdown]
# So here we fixed the problem of imbalance. Now I gonna start to the proccess of creating a model, first I will split the data into train data, test data and validation data with a 60/20/20 of the total of the data. But first I'll do a random sampling to get only 100000 random rows of the data, 25000 of each wind of severity.

# %%
sampled_df1= oversampled_df[oversampled_df['Severity'] == 1].sample(n=25000,random_state=42)
sampled_df2= oversampled_df[oversampled_df['Severity'] == 2].sample(n=25000,random_state=42)
sampled_df3= oversampled_df[oversampled_df['Severity'] == 3].sample(n=25000,random_state=42)
sampled_df4= oversampled_df[oversampled_df['Severity'] == 4].sample(n=25000,random_state=42)

balanced_sampled_data= pd.concat([sampled_df1,sampled_df2,sampled_df3,sampled_df4])

balanced_sampled_data= balanced_sampled_data.drop(['Moment_of_Day','Date','Time'],axis=1)

# %%
X= balanced_sampled_data.drop(['Severity'],axis=1)
y= balanced_sampled_data['Severity']

X_tr,X_test,y_tr,y_test= train_test_split(X,y,random_state=42,test_size=0.2,stratify=y)

X_train,X_validation,y_train,y_validation = train_test_split(X_tr,y_tr,random_state=42,test_size=0.20,stratify=y_tr)


# %% [markdown]
# Now that the data is already splitted then it's time to initialize the model and then fit with the data. I'll do first a LazzyClassifier that include severals models and I'll check which have better performance.

# %%
lazy_clf = LazyClassifier(verbose=1,ignore_warnings=True, custom_metric=None,random_state=42)

models,predictions = lazy_clf.fit(X_train, X_validation, y_train, y_validation)



# %%
models

# %% [markdown]
# We can see here that RandomForestClassifier has a better performance than the others so I'll make a tunning with gridSearchCV.

# %%
   
# Define the hyperparameter grid for GridSearchCV


cv_rf_params = {'max_depth': [2,3,4,5, None], 
             'min_samples_leaf': [1,2,3],
             'min_samples_split': [2,3,4],
             'max_features': [2,3,4],
             'n_estimators': [75, 100, 125, 150]
             }  

randomForest_clf_model = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(estimator=randomForest_clf_model, param_grid=cv_rf_params, cv=5, scoring=['accuracy', 'precision', 'recall', 'f1'], refit='f1')

grid_search.fit(X_train,y_train)


# %%
grid_search.best_params_

# %% [markdown]
# Now I gonna validate the model I'll create two functions one to get the confusion matrix and evaluate the model with the validation set and the other to get the scores.

# %%
def get_test_scores(model_name:str, preds, y_test_data,model):
    '''
    Generate a table of test scores.

    In:
        model_name (string): Your choice: how the model will be named in the output table
        preds: numpy array of test predictions
        y_test_data: numpy array of y_test data

    Out:
        table: a pandas df of precision, recall, f1, and accuracy scores for your model
    '''
    #aucu= round(roc_auc_score(y_test_data,preds),5)
    accuracy = round(accuracy_score(y_test_data, preds),5)
    precision = round(precision_score(y_test_data, preds,average='macro'),5)
    recall = round(recall_score(y_test_data, preds,average='macro'),5)
    f1 = round(f1_score(y_test_data, preds,average='macro'),5)          
    

    table = pd.DataFrame({'Model': [model_name],
                          'Precision': [precision*100],
                          'Recall': [recall*100],
                          'F1': [f1*100],
                          'Accuracy': [accuracy*100]
                          #'Auc': [aucu*100]
                        })

    return table

def  confuMatrix_plot(model,x_data_test,y_data_test):
                    
                     pred= model.predict(x_data_test)

                     cm = confusion_matrix(y_data_test, pred)

                     # Create the display for your confusion matrix.

                     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

                     # Plot the visual in-line.
                     
                     disp.plot(values_format='')  # `values_format=''` suppresses scientific notation

                     


# %%
confusion_matrix(y_test, grid_search.best_estimator_.predict(X_test))

# %%
confuMatrix_plot(grid_search.best_estimator_,X_test,y_test)

# %%
predict_randomForest_clf= grid_search.best_estimator_.predict(X_test)


# %%
results= get_test_scores('RandomForestClassifierCV',predict_randomForest_clf,y_test,grid_search.best_estimator_)

# %%
results

# %% [markdown]
# We can see here that there are low score numbers so I'll compare the results with the randomForestClssifier without gridSearchCV.

# %%
randomForest_clf= randomForest_clf_model.fit(X_train,y_train)

# %%
randomForest_predict_validation= randomForest_clf.predict(X_validation)
randomForest_predict_test= randomForest_clf.predict(X_test)


# %%
confuMatrix_plot(randomForest_clf,X_test,y_test)

# %%
results= pd.concat([results,get_test_scores('RandomForestClassifier',randomForest_predict_test,y_test,randomForest_clf)])

# %%
results.sort_values(by='F1',ascending=False)

# %% [markdown]
# As we can see here the  RandomForestClassifier has better performance than gridSearchCv so I'll save the model and I'll continue the project and the analysis with that.

# %%
path= './Models/'

joblib.dump(randomForest_clf,path+'RandomForestClassifier.pkl')

loaded_RandomForestClassifier= joblib.load(path+'RandomForestClassifier.pkl')


# %% [markdown]
# Now I'll analyse the feature importances.
# 

# %%
importance= loaded_RandomForestClassifier.feature_importances_

df_importance= pd.DataFrame(data=importance,index=X.columns, columns=['Importance']).sort_values(by='Importance',ascending=False)

df_importance= df_importance.rename_axis('Features').reset_index()

# %%
df_importance

# %%
df_importance= df_importance[df_importance['Importance'] > 0.01]
df_importance

# %%
plt.figure(figsize=(10,5))
sbn.barplot(df_importance,x='Features',y='Importance',hue='Features')
plt.title('Feature Importances')
plt.xticks(rotation= 30)
plt.show()

# %% [markdown]
# We can see here that the 3 main characteristics with the most importance are Year, Duration and Distance, so we could think that, if the accident is from recent years then it could be less serious, due to the new car models that include many new technologies, but we can also think that an accident that lasts longer could be more serious than others with a shorter duration or that an accident with a greater distance is due to the fact that there are roads closed due to the severity of the accident.

# %%
shap.initjs()

explainer_randomForest_CV= shap.Explainer(loaded_RandomForestClassifier.predict,X_train)

shap_values_xgb= explainer_randomForest_CV(X_test)

shap.summary_plot(shap_values_xgb, X_test,feature_names=X.columns)


