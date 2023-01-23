#!/usr/bin/env python
# coding: utf-8

# Name: David Adeyeye
# ID:0680640
# Assignment 3 

# In[248]:


#import packages used through the entire program
import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import ensemble
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
get_ipython().run_line_magic('matplotlib', 'inline')


# In[235]:


#using panda to read the csv file 
df1 = pd.read_csv("breast-cancer.csv")
df1


# In[236]:


#encoding diagnosis for ease of use with ml model 
le = preprocessing.LabelEncoder()
df1['diagnosis'] = le.fit_transform(df1['diagnosis'])


# In[237]:


df1.shape


# In[186]:


# checking for missing values 
df1.isnull().sum()


# In[187]:


df1.head()


# In[254]:


#splitting our data in attribute and target
y = df1['diagnosis']
x = df1.drop(['diagnosis'], axis = 1)


# In[255]:


#splitting data into test and train partitions 
x_train, x_test, y_train, y_test = train_test_split(x, y,  train_size=0.8, test_size=0.2,random_state=0)


# In[256]:


y_train.value_counts(normalize=True)


# In[257]:


y_test.value_counts(normalize=True)


# In[258]:


x_train.shape, y_train.shape


# In[259]:


x_train.shape, y_train.shape


# DECISION TREE

# In[260]:


#appling decision tree on our data
decision_tree = DecisionTreeClassifier(random_state=10)
decision_tree.fit(x_train,y_train)


# In[261]:


#score on the training data
decision_tree.score(x_train,y_train)


# In[262]:


decision_tree.score(x_test,y_test)


# In[263]:


#predicting our data with decision tree
y_decision = decision_tree.predict(x_test)


# In[264]:


y_decision


# In[306]:


#visually looking at predicted data and actual values
pred_tree=pd.DataFrame({'Actual Value':y_test,'Predicted value':y_decision, 'Difference': y_test-y_decision})
pred_tree[0:20]


# In[266]:


plt.figure(figsize=(15,10))
plt.scatter(y_test,y_decision)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')


# In[267]:


#using confusion matrix to determine: sensitivity and specitivity 
TP, FN, FP, TN = confusion_matrix(y_test, y_decision).ravel()


# In[268]:


#report showing accuray and precision
print(classification_report(y_test, y_decision))


# In[269]:


Sensitivity = TP / (TP + FN)
Specificity = TN / (TN + FP)


# In[270]:


print({'Sensitivity is' :Sensitivity, 'Specificity is':Specificity})


# In[271]:


clf = svm.SVC(random_state=0)
clf.fit(x_train, y_train)


# In[272]:


#ROC curve for decision tree
metrics.plot_roc_curve(clf, x_test, y_decision) 


# In[273]:


#determining auc score
roc_auc_score(y_test, y_decision, multi_class = "ovo" )


# RANDOM FOREST

# In[274]:


#appling random forest on our data
random_forest = ensemble.RandomForestClassifier(n_estimators=100)
random_forest.fit(x_train, y_train)
random_forest.score(x_test, y_test)


# In[275]:


random_forest.score(x_train, y_train)


# In[276]:


#predicting our data with random forest
y_random_forest = random_forest.predict(x_test)


# In[277]:


y_random_forest


# In[278]:


pred_random_forest =pd.DataFrame({'Actual Value':y_test,'Predicted value':y_random_forest, 'Difference': y_test-y_random_forest})
pred_random_forest[0:20]


# In[279]:


plt.figure(figsize=(15,10))
plt.scatter(y_test,y_random_forest)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')


# In[280]:


#using confusion matrix to determine: sensitivity and specitivity 
tP, fN, fP, tN = confusion_matrix(y_test, y_random_forest).ravel()
print(classification_report(y_test, y_random_forest))


# In[281]:


Sensitivity = tP / (tP + fN)
Specificity = tN / (tN + fP)
print({'Sensitivity is' :Sensitivity, 'Specificity is':Specificity})


# In[282]:


#ROC curve for random forest
metrics.plot_roc_curve(clf, x_test, y_random_forest) 


# In[283]:


#determining auc score
roc_auc_score(y_test, y_random_forest, multi_class = "ovo" )


# NAIVE BAYES 

# In[284]:


#appling naive bayes on our data
naive_bayes = GaussianNB()
naive_bayes.fit(x_train, y_train)
naive_bayes.score(x_test, y_test)


# In[285]:


naive_bayes.score(x_train, y_train)


# In[286]:


#predicting our data with naive bayes
y_naive_bayes = naive_bayes.predict(x_test)
y_naive_bayes


# In[287]:


pred_naive_bayes =pd.DataFrame({'Actual Value':y_test,'Predicted value':y_naive_bayes, 'Difference': y_test-y_naive_bayes})
pred_naive_bayes[0:20]


# In[288]:


plt.figure(figsize=(15,10))
plt.scatter(y_test,y_naive_bayes)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')


# In[289]:


#using confusion matrix to determine: sensitivity and specitivity 
tp, fn, fp, tn = confusion_matrix(y_test, y_naive_bayes).ravel()
print(classification_report(y_test, y_naive_bayes))


# In[369]:


y1_naive_bayes = naive_bayes.predict(x1_test)
y1_naive_bayes


# In[370]:


#ROC curve for naive bayes
metrics.plot_roc_curve(clf, x_test, y_naive_bayes) 


# In[371]:


#determining auc score
roc_auc_score(y_test, y_naive_bayes, multi_class = "ovo" )


# USING GRID SEARCH TO TUNE HYPERPARAMETERS

# Grid search on Random Forest

# In[293]:


#making parameter dictionary for random forest
param_forest = {'n_estimators': [10,20], 'max_depth' : [2,3]}


# In[294]:


#using grid search on random forest 
gs_random_forest = GridSearchCV(estimator=random_forest, param_grid=param_forest, cv=5, n_jobs=-1)


# In[295]:


gs_random_forest.fit(x_train, y_train)


# Grid search on Decision Tree

# In[296]:


std_scal = StandardScaler()
pca = decomposition.PCA()


# In[297]:


#making parameter dictionary for decision tree
n_components = list(range(1,x_test.shape[1]+1,1))
criterion = ['gini', 'entropy']
max_depth = [2,4,6,8]

#parameters
param_tree = dict(pca__n_components=n_components,
                 decision_tree__criterion=criterion,
                 decision_tree__max_depth=max_depth)

#pipeline
pipe = Pipeline(steps=[('std_scal', std_scal),
                      ('pca', pca),
                      ('decision_tree', decision_tree)])


# In[298]:


gs_decision_tree = GridSearchCV(pipe, param_tree)
gs_decision_tree.fit(x_train, y_train)


# IMPLEMENTING FEATURE SELECTION USING THE OBJECTIVE APPROACH WITH UNIVARIATE SELECTION

# In[320]:


#using object approach feature selection on our attributes 
best_attributes = SelectKBest(score_func=chi2, k=15)
fit = best_attributes.fit(x,y)

df1score = pd.DataFrame(fit.scores_)


# In[316]:


Columns = pd.DataFrame(df1.columns)


# In[317]:


#combine the 2 dataframes to visualise features
attribute_scores = pd.concat([Columns, df1score], axis=1)
attribute_scores.columns = ['Attributes', 'Score']
attribute_scores


# In[319]:


#using feature selection to find the top 15 important attributes
print(attribute_scores.nlargest(15, 'Score'))


# In[323]:


#picking new attributes based on the selection
y1 = df1['diagnosis']
x1 = df1.drop(['diagnosis', 'smoothness_mean','compactness_mean','concave points_mean','symmetry_mean','fractal_dimension_mean','texture_se','smoothness_se','compactness_se','concavity_se','concave points_se','symmetry_se','fractal_dimension_se','concave points_worst','fractal_dimension_worst','smoothness_worst','concave points_worst','symmetry_worst'], axis = 1)


# In[324]:


#new split after feature selection 
x1_train, x1_test, y1_train, y1_test = train_test_split(x, y,  train_size=0.8, test_size=0.2,random_state=0)


# In[326]:


y1_train.value_counts(normalize=True)


# In[328]:


y1_test.value_counts(normalize=True)


# In[329]:


x1_train.shape, y1_train.shape


# In[330]:


x1_test.shape, y1_test.shape


# Decision Tree After feature Selection

# In[332]:


decision_tree.fit(x1_train,y1_train)


# In[333]:


decision_tree.score(x1_train,y1_train)


# In[334]:


decision_tree.score(x1_test,y1_test)


# In[335]:


y1_decision = decision_tree.predict(x1_test)


# In[336]:


y1_decision


# In[339]:


TP1, FN1, FP1, TN1 = confusion_matrix(y1_test, y1_decision).ravel()
print(classification_report(y1_test, y1_decision))


# In[341]:


Sensitivity = TP1 / (TP1 + FN1)
Specificity = TN1 / (TN1 + FP1)
print({'Sensitivity is' :Sensitivity, 'Specificity is':Specificity})


# In[343]:


clf = svm.SVC(random_state=0)
clf.fit(x1_train, y1_train)


# In[344]:


#after feature seletion roc decision tree 
metrics.plot_roc_curve(clf, x1_test, y1_decision) 


# In[345]:


#after feature selection auc score for decision tree
roc_auc_score(y1_test, y1_decision, multi_class = "ovo" )


# Random forest after feature selection 

# In[346]:


random_forest.fit(x1_train,y1_train)


# In[347]:


random_forest.score(x1_train,y1_train)


# In[348]:


random_forest.score(x1_test,y1_test)


# In[350]:


y1_random_forest = random_forest.predict(x1_test)
y1_random_forest


# In[351]:


tP1, fN1, fP1, tN1 = confusion_matrix(y_test, y_random_forest).ravel()
print(classification_report(y1_test, y1_random_forest))


# In[352]:


Sensitivity = tP1 / (tP1 + fN1)
Specificity = tN1 / (tN1 + fP1)
print({'Sensitivity is' :Sensitivity, 'Specificity is':Specificity})


# In[355]:


#after feature selection roc random forest 
clf = svm.SVC(random_state=0)
>>> clf.fit(x1_train, y1_train)
metrics.plot_roc_curve(clf, x1_test, y1_random_forest) 


# In[356]:


#after feature selection auc score for random forest
roc_auc_score(y1_test, y1_random_forest, multi_class = "ovo" )


# Naive Bayes after Feature Selection

# In[357]:


naive_bayes.fit(x1_train,y1_train)


# In[361]:


naive_bayes.score(x1_train,y1_train)


# In[360]:


naive_bayes.score(x1_test,y1_test)


# In[364]:


y1_naive_bayes =naive_bayes.predict(x1_test)
y1_naive_bayes


# In[365]:


tp1, fn1, fp1, tn1 = confusion_matrix(y_test, y_naive_bayes).ravel()
print(classification_report(y_test, y_naive_bayes))


# In[366]:


#after feature selection roc naive bayes 
clf = svm.SVC(random_state=0)
>>> clf.fit(x1_train, y1_train)
metrics.plot_roc_curve(clf, x1_test, y1_naive_bayes) 


# In[367]:


#after feature selection auc score for naive bayes 
roc_auc_score(y1_test, y1_naive_bayes, multi_class = "ovo" )


# In[ ]:




