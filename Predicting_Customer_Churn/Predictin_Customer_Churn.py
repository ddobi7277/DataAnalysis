import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV 
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score,roc_auc_score #scores metrics measures
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay #matrix confusion
from xgboost import plot_importance
import joblib
import shap


data= pd.read_csv('customer_data.csv')

#Starting realizing a EDA

data.head()

data.dtypes

data.info()

data.describe()

#Checking the balance of the data

data['churn'].value_counts()

plt.figure(figsize=(4,2)) 
sbn.histplot(data['churn'],shrink=0.4)
plt.title('Churn Customer')
plt.show()

#We can see cleary that there are more customer that don't churn so the data is imbalanced

#Checking for duplicates

data.duplicated().sum() #there are not duplicates

#Checking for missing values
data.isnull().sum() #There are no missing values

#Creating a copy of the data but dropping the columns recordID and customer_id

df= data.drop(['customer_id','recordID'],axis=1)


#Checking the diferences of customer who churn and not churn

def showS():
    for i in data.columns:
        print('Column:'+i+'\n')
        print(data.groupby('churn')[i].describe())
        print('\n---------------------------------------------------------------------------------------')
        
showS()


#Feature Engineering

#I will create those new columns to be able to drop all those columns , and like in the function showS() we can see that they're close each other so if I sum and delete those columns
#will be more legible and easy to analyse.

df['total_charge']= df['total_day_charge']+df['total_eve_charge']+df['total_night_charge']+df['total_intl_charge'] 

df.groupby('churn')['total_charge'].describe() #The customers that churned has the highest mean of total charge

df['total_minutes']= df['total_day_minutes']+df['total_eve_minutes']+df['total_night_minutes']+df['total_intl_minutes']

df.groupby('churn')['total_minutes'].describe()  #The customers that churned has the highest mean of total minutes

df['total_calls']= df['total_day_calls']+df['total_eve_calls']+df['total_night_calls']+df['total_intl_calls']

df.groupby('churn')['total_calls'].describe() #They're pretty close esach others in the total of calls

#Now we drop the columns that we don't need anymore

df= df.drop(['total_day_minutes','total_day_calls',
             'total_day_charge','total_eve_minutes',
             'total_eve_calls','total_eve_charge',
             'total_night_minutes','total_night_calls',
             'total_night_charge','total_intl_minutes',
             'total_intl_calls','total_intl_charge','state'],axis=1)


df.groupby('churn')['number_customer_service_calls'].describe()  #The customers that churned has the highest mean of customer service calls

df.groupby('churn')['number_vmail_messages'].describe()   #The customers that churned has the lowest mean of video mail messages

# Now it's time to encode the variables 

# Encoding categorical variables

df['international_plan'] = np.where(df['international_plan'] == 'no',0,1)
df['voice_mail_plan'] = np.where(df['voice_mail_plan'] == 'no',0,1)
df['churn']= np.where(df['churn'] == 'no',0,1)


#Checking the relationships between the features


sbn.pairplot(df)
plt.title('Pairplot')
plt.show()


plt.figure(figsize=(16, 9))
heatmap = sbn.heatmap(df.corr(), vmin=-1, vmax=1, annot=True, cmap=sbn.color_palette("vlag", as_cmap=True))
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':14}, pad=12)


from sklearn.feature_selection import RFE


# We separate the targeted variable in y and the predictive features in X, like the decision tree and the random forest are robust against clase imbalance i don have to do anything else

X= df.drop('churn',axis=1)
y= df['churn']

modelTree = DecisionTreeClassifier(random_state=42)


X_tr,X_test,y_tr,y_test = train_test_split(X,y,test_size= 0.2 , random_state= 0) # here we split the data in 80/20 , we've got the 20 of test


X_train,X_validation,y_train,y_validation = train_test_split(X_tr,y_tr,random_state= 0, stratify= y_tr,test_size= 0.2) # here we used the restant data and split it in 60/20

#1 Using DecisionTreeClassifier

modelTree.fit(X_train,y_train)

#We get the predictions 

pred = modelTree.predict(X_validation)

#Function to get the scores
def scoresClassifier(pred,d_test):

         print('Accuracy: {0}%'.format(round(accuracy_score(d_test,pred)*100,2)))

         # Print your precision score.

         ### YOUR CODE HERE ###
         print('Precision: {0}%'.format(round(precision_score(d_test,pred)*100,2)))

         # Print your recall score.
         print('Recall: {0}%'.format(round(recall_score(d_test,pred)*100,2)))

         # Print your f1 score.
         print('F1: {0}%'.format(round(f1_score(d_test,pred)*100,2)))



scoresClassifier(pred,y_validation)

#Function to see the confussion matrix
def  confuMatrix_plot(model,x_data_test,y_data_test):
                    
                     pred= model.predict(x_data_test)

                     cm = confusion_matrix(y_data_test, pred)

                     # Create the display for your confusion matrix.

                     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

                     # Plot the visual in-line.                     
                     disp.plot(values_format='')  # `values_format=''` suppresses scientific notation


confuMatrix_plot(modelTree,X_validation,y_validation)

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
                          'Precision': [precision],
                          'Recall': [recall],
                          'F1': [f1],
                          'Accuracy': [accuracy],
                          'Auc': [aucu]
                        })

    return table

#Now we validate the results

results= get_test_scores('DecissionTree',pred,y_validation,modelTree)
results

#2 Random Forest Classifier

modelRandomForest= RandomForestClassifier(random_state=0)

modelRandomForest.fit(X_train, y_train)

results= pd.concat([results,get_test_scores('RandomForestClassifier',modelRandomForest.predict(X_validation),y_validation,modelRandomForest)],axis= 0)

confuMatrix_plot(modelRandomForest,X_validation,y_validation)

#3 Models with Cross Validation and Tuning

#Decission Tree CV

cv_dt_params = {'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50],
             'min_samples_leaf': [2, 5, 10, 20, 50]}

scoring = ['accuracy', 'precision', 'recall', 'f1']


DecisionTreeModel_cv= GridSearchCV(modelTree,cv_dt_params,scoring= scoring,cv= 5,refit='f1') #Tunning the model

DecisionTreeModel_cv.fit(X_train,y_train) 

pred_df_cv= DecisionTreeModel_cv.best_estimator_.predict(X_validation) #Getting the predictions with the best estimator

results= pd.concat([results,get_test_scores('DecissionTree_CV',pred_df_cv,y_validation,DecisionTreeModel_cv)],axis= 0)

confuMatrix_plot(DecisionTreeModel_cv,X_validation,y_validation)

#Random Forest CV

cv_rf_params = {'max_depth': [2,3,4,5, None], 
             'min_samples_leaf': [1,2,3],
             'min_samples_split': [2,3,4],
             'max_features': [2,3,4],
             'n_estimators': [75, 100, 125, 150]
             }  

RandomForest_CV= GridSearchCV(modelRandomForest,cv_rf_params,scoring= scoring, cv= 5, refit= 'f1')

RandomForest_CV.fit(X_train,y_train)

pred_rf_cv= RandomForest_CV.best_estimator_.predict(X_validation)

results= pd.concat([results,get_test_scores('RandomForestCV',pred_rf_cv,y_validation,RandomForest_CV)],axis=0)

confuMatrix_plot(RandomForest_CV,X_validation,y_validation)
#4 Models with Hyperparameter Tuning using XGBoost

xgb = XGBClassifier(objective= 'binary:logistic',random_state= 0)

cv_xgb_params = {'max_depth': [4,5,6,7,8], 
             'min_child_weight': [1,2,3,4,5],
             'learning_rate': [0.1, 0.2, 0.3],
             'n_estimators': [75, 100, 125]
             } 

xgb_cv= GridSearchCV(xgb,cv_xgb_params,scoring= scoring, cv= 5, refit='f1')

xgb_cv.fit(X_train,y_train)

plot_importance(xgb_cv.best_estimator_)

confuMatrix_plot(xgb_cv,X_validation,y_validation)

results= pd.concat([results,get_test_scores('XGB_CV',pred_rf_cv,y_validation,xgb_cv)],axis=0)

#Now I'll search with a randomized search 
# Define hyperparameter space


# As we can see the most efficient models are the Random Forest Cross-validated and the XGBoost Cross-validate 

# We'll now go to get the most important features

importance= xgb_cv.best_estimator_.get_booster().get_score(importance_type='gain')
indexImp= list(xgb_cv.best_estimator_.get_booster().get_score(importance_type='gain').keys())[0:9]
valImp= list(xgb_cv.best_estimator_.get_booster().get_score(importance_type='gain').values())[0:9]
importanceDF= pd.DataFrame(data= valImp, index=indexImp, columns=['importance']).sort_values(by='importance',ascending=False)
importanceDF= importanceDF.rename_axis('Features').reset_index()

plt.figure(figsize=(10,5))
fig= sbn.barplot(data= importanceDF, x= 'Features', y= 'importance')
plt.xticks(rotation= 80)
plt.bar_label(fig.containers[0])
plt.title('Importance of Features')
plt.show()

# models. We can now use these models to make predictions on the test set.

test_result= []

decissionTree_test= get_test_scores('DecissionTree_test',modelTree.predict(X_test),y_test,modelTree)

randomForest_test= get_test_scores('RandomForest_test',modelRandomForest.predict(X_test),y_test,modelRandomForest)

decissionTree_CV_test= get_test_scores('DecissionTree_CV_test',DecisionTreeModel_cv.predict(X_test),y_test,DecisionTreeModel_cv)

randomForest_CV_test= get_test_scores('RandomForest_CV_test',RandomForest_CV.predict(X_test),y_test,RandomForest_CV)

xgb_test= get_test_scores("XGBoost_CV_test",xgb_cv.predict(X_test),y_test,xgb_cv)


test_result= decissionTree_test

test_result= pd.concat([test_result,randomForest_test,decissionTree_CV_test,randomForest_CV_test,xgb_test],axis= 0)

#Now we save the models just in case that we have to use it again.

path= './Models/'

joblib.dump(modelTree,path+'modelTree.pkl')

joblib.dump(modelRandomForest,path+'modelRandomForest.pkl')

joblib.dump(DecisionTreeModel_cv,path+'DecissionTree_CV.pkl')

joblib.dump(RandomForest_CV,path+'RandomForest_CV.pkl')

joblib.dump(xgb_cv,path+'XGBoost_CV.pkl')

'''
In case that you want to load again do this:

loaded_modelTree= joblib.load(path+'modelTree.pkl')

loaded_modelRandomForest= joblib.load(path+'modelRandomForest.pkl')

loaded_DecissionTree_CV= joblib.load(path+'DecissionTree_CV.pkl')

loaded_RandomForest_CV= joblib.load(path+'RandomForest_CV.pkl')

loaded_xgb_cv= joblib.load(path+'XGBoost_CV.pkl')

shap.initjs()

explainer= shap.Explainer(modelRandomForest.predict,X_train)

shap_values= explainer.shap_values(X_test)


shap.summary_plot(shap_values, X_test)

'''


shap.initjs()

explainer_xgb= shap.Explainer(xgb_cv.best_estimator_.predict,X_train)

shap_values_xgb= explainer_xgb(X_test)

shap.summary_plot(shap_values_xgb, X_test)

#In the summary we can see cleary that like in the feature importance the 








