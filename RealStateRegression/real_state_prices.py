import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sbn 
#from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
#load the data

data= pd.read_csv('real_estate.csv')

data.info()
data.describe()

# it shows that there are not null values or missing values

df= data.copy()
#Create the feature year and month
df['year'] = df['transaction_date'].astype('int')
df['month'] = ((df['transaction_date'] - df['year']) * 12).astype('int') + 1

df= df.drop('transaction_date',axis=1)

#Now check the outliers for everthing but we only going to focus in the variable that we need predict

def plotbox(e):
    for i in ((e.columns.values)):
        plt.figure(figsize=(10,6))
        sbn.boxplot(x=e[i])
        

sbn.histplot(df['price_per_unit'])    

plotbox(df)
    
#The boxplots and histplot show that there are some outliers in transit_distance and in price_per_unit features 
#But let's focus in price_per_unit feature
#We will remove the outliers using the IQR method
#First we need to calculate the IQR
q1= df['price_per_unit'].quantile(0.25)
q3 = df['price_per_unit'].quantile(0.75)

iqr= q3 - q1
#Then set the upper and lower limit
upper_limit= q3 + (1.5*iqr)
lower_limit = q1 - (1.5*iqr)
#Then remove the outliers
df.loc[df['price_per_unit'] > upper_limit, 'price_per_unit'] = upper_limit
df.loc[df['price_per_unit'] < 0, 'price_per_unit'] = 0

sbn.histplot(df['price_per_unit'])    

plotbox(df)

#Now we have removed the outliers and we have a normal distribution in price_per_unit feature
#Now let's check the numeric correlations
plt.figure(figsize=(10,5))
plt.title('Correlation Graph')
sbn.pairplot(df)

#Like year and month doesn't have a great correlation let's ignore this
df= df.drop(['year','month'],axis=1)
#Now that we have the possible corelation between features let's split the data

#We will use the train_test_split function from sklearn library

X= df.drop('price_per_unit',axis=1)
y= df['price_per_unit']

x_train,x_test,y_train,y_test = train_test_split(X,y, test_size=0.30 , random_state=42)

#Like we want to predict a numeric features but in the correlation map it shows that the features don't have to significant correlation then let's try a Random Forest Regressor
#We will use the RandomForestRegressor from sklearn library

#But first let's preproccess the data like scale the date.

#Normal way
'''
scaler= StandardScaler()

scaler.fit(x_train)

x_train_scaled = scaler.transform(x_train)

x_test_scaled = scaler.transform(x_train)

randomForest=  RandomForestRegressor(random_state=42)

randomForest.fit(x_train_scaled,y_train)

'''

#Using pipeline

# Define preprocessing for numeric columns (scale them)
numeric_features = [0,1,2,3,4]
numeric_transformer= Pipeline(steps=[('scaler',StandardScaler())])

#Combining preprocessing steps

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ])

modeltype= RandomForestRegressor(random_state=42)

# Create preprocessing and training pipeline
pipeline=  Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', modeltype)])

# fit the pipeline to train a linear regression model on the training set

modelRandomForest= pipeline.fit(x_train,y_train)

predictRF=  modelRandomForest.predict(x_test)

#To evaluate the metrics

def scoresRegressor(pred,d_test):
         mse = mean_squared_error(d_test, pred)
         print("MSE: {0}".format(round(mse,2)))
         rmse = np.sqrt(mse)
         print("RMSE: {0}".format(round(rmse,2)))
         r2 = r2_score(y_test, pred)
         print("R2: {0}".format(round(r2,2)))
         
scoresRegressor(predictRF,y_test)
         
#And now is time to plot the predict values vs the true values


plt.figure(figsize=(10,5))
sbn.regplot(x=y_test,y= predictRF,line_kws=dict(color="r"))
plt.title('Predicted  vs True Values')
plt.xlabel('predicted values')
plt.ylabel('actual values')

# Save the model as a pickle file
filename = './real_estate_model.pkl'
joblib.dump(modelRandomForest, filename)

# Load the model from the file
loaded_model = joblib.load(filename)

X_new = np.array([[16.2,289.3248,5,24.98203,121.54348],
                  [13.6,4082.015,0,24.94155,121.5038]])

results = loaded_model.predict(X_new)
print('Predictions:')
for prediction in results:
    print(round(prediction,2))
    
