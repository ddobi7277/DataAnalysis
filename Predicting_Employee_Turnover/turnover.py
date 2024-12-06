'''
I've assigned a new project: Predicting Employee Turnover. Your task is to develop a predictive model to identify employees likely to leave the company.

'''
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



data= pd.read_csv('./HR_dataset.csv')

# Inspecting the data and starting the EDA

data.head()

data.info()

#Let's check for missing values
data.isna().sum()
#There is 0 missing values

data.describe()


#In the describe I could see the montly average hours is a very high , the normal full-time person should work 40 hours by week , so 40*4= 160 , and the mean in the average_montly_hours 
#and the mean in the average_montly_hours is  201,05 which is higher than 160 , so this feature should be important in the analysis.
# Let's take a look of average montly hours closer.

data.groupby('left')['average_montly_hours'].mean()

# We here can see that the mean of the average montly hours of the people that stay in the company is lower than the those who left, so the average montly hours 
# could be a good feature to predict if the employee will leave the company.
#Here we can see also that the average_montly_hours is the feature with the highest difference between
plt.figure(figsize=(6,3))
fig= sbn.barplot(data,x='average_montly_hours',hue='left')
plt.title('Average montly hours by employee who stay and left.')
plt.xlabel('Average Montly Hours')
plt.show()

#To check the others feature order by left
for i in data.drop(['Department','salary','left','average_montly_hours'],axis= 1).columns:
   print(data.groupby('left')[i].mean())
   
#Here we can see that the employees who stay has a mean() of promotion_last_5_years,satisfaction_level,work_accident higer than those who left but this is logically normal
   
data.groupby('left')['number_project'].value_counts()


plt.figure(figsize=(6,3))
sbn.barplot(data, x= 'number_project',y='satisfaction_level', hue='left')
plt.title('Satisfaction level by number of projects')
plt.xlabel('Number of Projects')
plt.show()

#Here we can see that there is not a single employee who stayed that has 7 projects and by the other hand there are 256 employees that have 7 projects
#Also we can see that the satisfaction lvl start to decrease with the 6 projects

data[data['number_project']>5].groupby('left')['satisfaction_level'].mean()

#You can notice the obvius difference between the satisfaction lvl of those who stayed and left with more than 5 projects
#So the number_project feature could be an important feature for further analysis. 

#Let's check the salary and Department features

data[['salary']].value_counts() # There're 5 times more employees with low and medium salary thatn employees with high salary

data[['Department']].value_counts()#the Department with more employees is sales and the less employees is management

round(data.groupby('salary')['left'].value_counts(normalize=True)*100,3)
#Only the 6.62% of employees with high salary left there is a big difference with low (29.68 left) and medium (20.43 left)

plt.figure(figsize=(10,5))
sbn.barplot(data,x='salary', y= 'satisfaction_level')
plt.title('Satisfaction level by salary')
plt.xlabel('Number of Projects')
plt.show()

plt.figure(figsize=(10,5))
sbn.barplot(data,x='left', y= 'satisfaction_level',hue= 'salary')
plt.title('Satisfaction level by salary (stayed vs left employees)')
plt.xlabel('Number of Projects')
plt.show()

round(data.groupby(['salary','left'])['Department'].value_counts(normalize=True)*100,3)

plt.figure(figsize=(16,8))
sbn.barplot(data,x= 'Department',y= 'satisfaction_level',hue= 'salary')
plt.title('Satisfaction level by salary by departmens')
plt.xlabel('Departments')
plt.xticks(rotation= 30)
plt.show()
# 6 of 10 have employees with low salary  that the satisfaction level is lower than medium and high salaries  
plt.figure(figsize=(10,5))
sbn.lineplot(data.groupby(['average_montly_hours'])['satisfaction_level'].mean())
plt.title('Satisfaction level by average montly hours')
plt.xlabel('average montly hours')
plt.show()
#We can see here that more hours will have less satisfaction lvl

#Now is time to encode before check the correlations

df= data.copy()

df= df.rename(columns={'Department':'department'})

df['salary'].unique()

df['salary']= df['salary'].map({'low':1,'medium':2,'high':3})

df['department'].unique()

df['department']= df['department'].map({'sales':1, 'accounting':2, 'hr':3, 'technical':4, 'support':5, 'management':6,
       'IT':7, 'product_mng':8, 'marketing':9, 'RandD':10})
#Now it's time for check for outliers and normal distribution
#Then we define the function get_boxPlot that get the feature name and the show a histplot to
#check how is distributed
def get_boxPlot(d):
    feature= str(d).replace("_"," ")
    plt.figure(figsize=(5,2))
    sbn.boxplot(x=df[d])
    plt.title(f"Box Plot of :{feature}")
    plt.xlabel(feature)
    plt.show()
    
#First we check for outliers

for i in df.columns:
    get_boxPlot(i)
    
#We can see that there are outliers in the time spend company but that could be some of the older employee so we gonna let pass that

#Now we check for normal distribution
#I'll perform Shapiro-Wilk test for normality

def get_normality(d):
    #If the p-value is less than 0.05, we reject the null hypothesis of normality
    normality= shapiro(df[d])
    if normality.pvalue < 0.05:
        print("Is normally distributed!")
    else:
        print("Is not normally distributed!")
        
for i in df.columns:
    get_normality(i)
    
#We get that all the features are normally distributed




#Let's check the correlation between the features

sbn.pairplot(df)

#Left don't have a strong relation with work accident or promotion_last_5years 

#We can start now to prepare the data for the modeling

X= df.drop('left',axis= 1)
y= df['left']

X_tr,X_test,y_tr,y_test= train_test_split(X.values,y.values,random_state=42,test_size=0.20, stratify=y)

X_train,X_validation,y_train,y_validation= train_test_split(X_tr,y_tr,random_state=42,test_size=0.20,stratify=y_tr)

clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)

models,predictions = clf.fit(X_train, X_validation, y_train, y_validation)

#After make a lazyClassifier that involves some of the models 
#We could see here that the best models with the highest metrics (accuracy, precision, recall, F1-score, AUC-ROC)  are RandomForestClassifier and XGBClassifier now we gonna check again those scores

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
    aucu= round(roc_auc_score(y_test_data,preds),5)
    accuracy = round(accuracy_score(y_test_data, preds),5)
    precision = round(precision_score(y_test_data, preds),5)
    recall = round(recall_score(y_test_data, preds),5)
    f1 = round(f1_score(y_test_data, preds),5)          
    

    table = pd.DataFrame({'Model': [model_name],
                          'Precision': [precision*100],
                          'Recall': [recall*100],
                          'F1': [f1*100],
                          'Accuracy': [accuracy*100],
                          'Auc': [aucu*100]
                        })

    return table

def  confuMatrix_plot(model,x_data_test,y_data_test):
                    
                     pred= model.predict(x_data_test)

                     cm = confusion_matrix(y_data_test, pred)

                     # Create the display for your confusion matrix.

                     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

                     # Plot the visual in-line.
                     
                     disp.plot(values_format='')  # `values_format=''` suppresses scientific notation

                     disp.plot()


randomForest_clf= RandomForestClassifier(random_state=42)

randomForest_clf.fit(X_train,y_train)

predicts_randomForest_clf_validation= randomForest_clf.predict(X_validation)

results_validation= get_test_scores('RandomForest_clf',predicts_randomForest_clf_validation,y_validation,randomForest_clf)

#We can see here a precisiont and accuracy of 99 , recall 96 , f1 and Auc 98 that this means that the model is was very good with the validation set
#Now let's see whit the test set.
results_test= get_test_scores('RandomForest_clf',randomForest_clf.predict(X_test),y_test,randomForest_clf)
#We can see the same metrics for the model so this could be a champion model

#Now let's try with the XGB
xgb_clf = XGBClassifier(objective='binary:logistic', random_state=42)

xgb_clf.fit(X_train,y_train)

results_validation= pd.concat([results_validation,get_test_scores('XGBoost_clf',xgb_clf.predict(X_validation),y_validation,xgb_clf)])
#We can see here a precisiont and accuracy of 98 , recall 96 , f1 97 and Auc 98 that this means that the model is was very good with the validation set but the metrics are little bit lower than RandomForest
#Now let's see whit the test set.

results_test= pd.concat([results_test,get_test_scores('XGboos_clf_clf',xgb_clf.predict(X_test),y_test,xgb_clf)])

#The result is similar there are a little bit lower than RandomForest Classifier

#Now lets try with hypertunning and cv for both

cv_rf_params = {'max_depth': [2,3,4,5, None], 
             'min_samples_leaf': [1,2,3],
             'min_samples_split': [2,3,4],
             'max_features': [2,3,4],
             'n_estimators': [75, 100, 125, 150]
             }  

scoring = ['accuracy', 'precision', 'recall', 'f1']

RandomForest_CV= GridSearchCV(randomForest_clf,cv_rf_params,scoring= scoring, cv= 5, refit= 'f1')

RandomForest_CV.fit(X_train,y_train)

pred_rf_cv= RandomForest_CV.best_estimator_.predict(X_validation)

results_validation= pd.concat([results_validation,get_test_scores('RandomForestCV',pred_rf_cv,y_validation,RandomForest_CV)],axis=0)

path= './Models/'

joblib.dump(RandomForest_CV,path+'modelTree.pkl')

#To load use:
loaded_RandomForest_CV= joblib.load(path+'RandomForest_CV.pkl')

#Now It's time to get the features with more importance at the time of predict if they will stay or left


importance= loaded_RandomForest_CV.best_estimator_.feature_importances_ #Get the feature importance values
#Make a DataFrame with the values and the names of each feature to see the importance
df_importance= pd.DataFrame(data=importance,index=X.columns, columns=['Importance']).sort_values(by='Importance',ascending=False)

df_importance= df_importance.rename_axis('Features').reset_index()

#As we can see and we already analyzed early with the graphs the features that can have a significanse are:
# satisfaction_level, number_project,time_spend_company adn average_montly_hours.

#But if we analyze a detail the results:

shap.initjs()

explainer_randomForest_CV= shap.Explainer(loaded_RandomForest_CV.best_estimator_.predict,X_train)

shap_values_xgb= explainer_randomForest_CV(X_test)

shap.summary_plot(shap_values_xgb, X_test,feature_names=X.columns)

#Here we can see cleary that it they are the same 5 important features but in satisfaction_level , number_project and average_hours are coorelated
#and that with high values of number_project and average_montly_hours we get low values of satisfaction_level
#and then we can start a teory that with more projects and more hours in the week the satisfaction_level 
#will drop in the employees and will grow up the posibilities of left, so the recomends should be try to not overcharge the employees with
#projects and hours in the week.



