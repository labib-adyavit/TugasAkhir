#!/usr/bin/env python
# coding: utf-8

# In[9]:


# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Reading and converting csv file to Pandas DataFrame object
df_path = 'Data TA newscore.csv'
df = pd.read_csv(df_path)
df = df.drop(['Sumber','No', 'dop_el', 'init_cap', '%_cr'], axis=1)

# Showing the dataset
df

# Describing dataset
df.describe()

# Dataset info
df.info()

# EDA
## Pearson Correlation Matrix
### Calculating Pearson correlation coefficients 
corr = df.corr()

### Creating Pearson correlation matrix
plt.subplots(figsize=(13,13))
ax = sns.heatmap(
    corr,
    vmin = -1, vmax = 1, center = 0,
    linewidths = .5, annot = True,
    cmap = sns.diverging_palette(20, 220, n = 200),
    square = True)

## Parallel Coordinates Plot
### Importing modules
from plotly import graph_objects as go

### Creating parallel coordinate plot for Target 1 
cols = ['Ni_sto', 'Co_sto', 'Mn_sto', 'sint_temp', 'sint_time', 'dop_ionrad', 'dop_eln', 'dop_sto', 'xrd_a', 'xrd_c', 'del_V', 'init_cap_score', '%cr_inc_score']
fig1 = go.Figure(data=
        go.Parcoords(
        line = dict(color = df['init_cap_score'],
                   colorscale = [[0, 'white'], [0.5, 'white'], [1.0, 'red']]),
        dimensions = [dict(label=col, values=df[col]) for col in cols]   
        )
    )
fig1.update_layout(
    plot_bgcolor = 'white',
    paper_bgcolor = 'white'
)

### Showing parallel coordinate plot for Target 1
fig1.show()

### Creating parallel coordinate plot for Target 2
cols = ['Ni_sto', 'Co_sto', 'Mn_sto', 'sint_temp', 'sint_time', 'dop_ionrad', 'dop_eln', 'dop_sto', 'xrd_a', 'xrd_c', 'del_V', 'init_cap_score', '%cr_inc_score']
fig2 = go.Figure(data=
        go.Parcoords(
        line = dict(color = df['%cr_inc_score'],
                   colorscale = [[0, 'white'], [0.5, 'white'], [1.0, 'green']]),
        dimensions = [dict(label=col, values=df[col]) for col in cols]   
        )
    )
fig2.update_layout(
    plot_bgcolor = 'white',
    paper_bgcolor = 'white'
)

### Showing parallel coordinate plot for Target 2
fig2.show()



# Defining unimputed X
X_raw = df.iloc[:,:-2]

# Defining y1 (Target 1)
y1 = df.iloc[:,-2]

# Defining y2 (Target 2)
y2 = df.iloc[:,-1]

# Imputing empty values
## Importing modules
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Imputing X
my_imputer = SimpleImputer(strategy = 'mean')
X = pd.DataFrame(my_imputer.fit_transform(X_raw))
X.columns = X_raw.columns
X.index = X_raw.index

# Creating scaled X variable 
sc = StandardScaler()
X_scaled = X
X_scaled.iloc[:, [0,1,2,3,4,8,9,10]] = sc.fit_transform(X_scaled.iloc[:, [0,1,2,3,4,8,9,10]])



# Importing LOOCV modules
from sklearn.model_selection import cross_val_score, LeaveOneOut

# Defining LOOCV accuracy score function 
def LOO_cv (X, y, model):
    LOOCVscore = cross_val_score(model, X, y, scoring="accuracy", cv=LeaveOneOut())
    return LOOCVscore.mean()

# Searching the most accurate models for Target 1 (X,y1)

from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier

## Candidate algorithms
candidates1 = {'SGD' : SGDClassifier(random_state=123),  
              'kNN': KNeighborsClassifier(), 
              'NearestCentroid': NearestCentroid(),
              'Logistic': LogisticRegression(),
             }
candidates1

candidates2 = {'DecisionTree' : DecisionTreeClassifier(random_state=123), 
              'RF': RandomForestClassifier(random_state=123),
              'ET': ExtraTreesClassifier(random_state=123),
              'GB': GradientBoostingClassifier(random_state=123),
              'AdaBoost': AdaBoostClassifier(random_state=123),
              'Bagging': BaggingClassifier(random_state=123)
             }
candidates2

## Printing the accuracy score of each models for Target 1
for candidate1, model in candidates1.items():
    print('Model {} memiliki skor LOOCV sebesar {}'.format(candidate1, LOO_cv(X_scaled, y1, model)))
for candidate2, model in candidates2.items():
    print('Model {} memiliki skor LOOCV sebesar {}'.format(candidate2, LOO_cv(X, y1, model)))

    
# Searching the most accurate models for Target 2 (X,y2)

## Printing the accuracy score of each models for Target 2
for candidate1, model in candidates1.items():
    print('Model {} memiliki skor LOOCV sebesar {}'.format(candidate1, LOO_cv(X_scaled, y2, model)))
for candidate2, model in candidates2.items():
    print('Model {} memiliki skor LOOCV sebesar {}'.format(candidate2, LOO_cv(X, y2, model)))
    

# Hyperparameter Optimization (Target 1)

# Importing Bayesian optimization (BO) library
from bayes_opt import BayesianOptimization

# Defining BO function for GradientBoostingClassifier
def gb_cl_bo(max_depth, max_features, learning_rate, n_estimators, subsample):
    params_gb = {}
    params_gb['max_depth'] = round(max_depth)
    params_gb['max_features'] = max_features
    params_gb['learning_rate'] = learning_rate
    params_gb['n_estimators'] = round(n_estimators)
    params_gb['subsample'] = subsample
    scores = cross_val_score(GradientBoostingClassifier(random_state=123, **params_gb),
                             X, y1,
                             scoring='accuracy', 
                             cv=LeaveOneOut())
    score = scores.mean()
    return score

## Exploration range
params_gb ={
    'max_depth':(3, 10),
    'max_features':(0.8, 1),
    'learning_rate':(0.01, 1),
    'n_estimators':(80, 150),
    'subsample': (0.8, 1)

}
# Defining BO function for ExtraTreesClassifier
def et_cl_bo(max_features, n_estimators, min_samples_split, min_samples_leaf):
    params_et = {}
    params_et['max_features'] = max_features
    params_et['n_estimators'] = round(n_estimators)
    params_et['min_samples_split'] = int(min_samples_split)
    params_et['min_samples_leaf'] = int(min_samples_leaf)
    
    scores = cross_val_score(ExtraTreesClassifier(random_state=123, **params_et),
                             X, y1,
                             scoring='accuracy', 
                             cv=LeaveOneOut())
    score = scores.mean()
    return score

## Exploration range
params_et ={
    'max_features':(0.15, 1),
    'n_estimators':(25, 251),
    'min_samples_split': (2,14),
    'min_samples_leaf' : (1,14)
}

# Defining BO function for RandomForestClassifier
def rf_cl_bo(max_depth, max_features, n_estimators, min_samples_split, min_samples_leaf):
    params_rf = {}
    params_rf['max_depth'] = round(max_depth)
    params_rf['max_features'] = max_features
    params_rf['n_estimators'] = round(n_estimators)
    params_rf['min_samples_split'] = int(min_samples_split)
    params_rf['min_samples_leaf'] = int(min_samples_leaf)
    scores = cross_val_score(RandomForestClassifier(random_state=123, **params_rf),
                             X, y1,
                             scoring='accuracy', 
                             cv=LeaveOneOut())
    score = scores.mean()
    return score

## Exploration range
params_rf ={
    'max_depth':(3, 10),
    'max_features':(0.8, 1),
    'n_estimators':(80, 150),
    'min_samples_split': (2,15),
    'min_samples_leaf' : (2,15)
}

# Optimizing GB
gb_bo = BayesianOptimization(gb_cl_bo, params_gb, random_state=111)
gb_bo.maximize(init_points=5, n_iter=20)

print("Hasil optimasi GradientBoostingClassifier:", gb_bo.max)

# Optimizing ET
et_bo = BayesianOptimization(et_cl_bo, params_et, random_state=111)
et_bo.maximize(init_points=5, n_iter=20)

print("Hasil optimasi ExtraTreesClassifier:", et_bo.max)

# Optimizing RF
rf_bo = BayesianOptimization(rf_cl_bo, params_rf, random_state=111)
rf_bo.maximize(init_points=5, n_iter=20)

print("Hasil optimasi RandomForestClassifier:", rf_bo.max)


# Hyperparameter Optimization (Target 2)
# Importing Bayesian optimization (BO) library
from bayes_opt import BayesianOptimization

# Defining BO function for GradientBoostingClassifier
def gb_cl_bo2(max_depth, max_features, learning_rate, n_estimators, subsample, min_samples_split, min_samples_leaf):
    params_gb = {}
    params_gb['max_depth'] = round(max_depth)
    params_gb['max_features'] = max_features
    params_gb['learning_rate'] = learning_rate
    params_gb['n_estimators'] = round(n_estimators)
    params_gb['subsample'] = subsample
    params_gb['min_samples_split'] = int(min_samples_split)
    params_gb['min_samples_leaf'] = int(min_samples_leaf)
    scores = cross_val_score(GradientBoostingClassifier(random_state=123, **params_gb),
                             X, y2,
                             scoring='accuracy', 
                             cv=LeaveOneOut())
    score = scores.mean()
    return score

## Exploration range
params_gb ={
    'max_depth':(3, 10),
    'max_features':(0.8, 1),
    'learning_rate':(0.01, 1),
    'n_estimators':(80, 150),
    'subsample': (0.8, 1),
    'min_samples_split': (2,15),
    'min_samples_leaf' : (2,15)
}
# Defining BO function for ExtraTreesClassifier
def et_cl_bo2(max_features, n_estimators, min_samples_split, min_samples_leaf):
    params_et = {}
    params_et['max_features'] = max_features
    params_et['n_estimators'] = round(n_estimators)
    params_et['min_samples_split'] = int(min_samples_split)
    params_et['min_samples_leaf'] = int(min_samples_leaf)
    
    scores = cross_val_score(ExtraTreesClassifier(random_state=123, **params_et),
                             X, y2,
                             scoring='accuracy', 
                             cv=LeaveOneOut())
    score = scores.mean()
    return score

## Exploration range
params_et ={
    'max_features':(0.15, 1),
    'n_estimators':(25, 251),
    'min_samples_split': (2,14),
    'min_samples_leaf' : (1,14)
}

# Defining BO function for RandomForestClassifier
def rf_cl_bo2(max_depth, max_features, n_estimators, min_samples_split, min_samples_leaf):
    params_rf = {}
    params_rf['max_depth'] = round(max_depth)
    params_rf['max_features'] = max_features
    params_rf['n_estimators'] = round(n_estimators)
    params_rf['min_samples_split'] = int(min_samples_split)
    params_rf['min_samples_leaf'] = int(min_samples_leaf)
    scores = cross_val_score(RandomForestClassifier(random_state=123, **params_rf),
                             X, y2,
                             scoring='accuracy', 
                             cv=LeaveOneOut())
    score = scores.mean()
    return score

## Exploration range
params_rf ={
    'max_depth':(3, 10),
    'max_features':(0.8, 1),
    'n_estimators':(80, 150),
    'min_samples_split': (2,15),
    'min_samples_leaf' : (2,15)
}

# Optimizing GB
gb_bo2 = BayesianOptimization(gb_cl_bo2, params_gb, random_state=111)
gb_bo2.maximize(init_points=10, n_iter=40)

print("Hasil optimasi GradientBoostingClassifier:", gb_bo2.max)

# Optimizing ET
et_bo2 = BayesianOptimization(et_cl_bo2, params_et, random_state=111)
et_bo2.maximize(init_points=10, n_iter=40)

print("Hasil optimasi ExtraTreesClassifier:", et_bo2.max)

# Optimizing RF
rf_bo2 = BayesianOptimization(rf_cl_bo2, params_rf, random_state=111)
rf_bo2.maximize(init_points=15, n_iter=60)

print("Hasil optimasi RandomForestClassifier:", rf_bo2.max)



# Defining optimum models
## Target 1 (Initial capacity)
y1_opt = ExtraTreesClassifier(random_state=123, 
                              max_features=0.6, 
                              min_samples_leaf=1, 
                              min_samples_split=2,
                              n_estimators=228
                             )
LOO_cv(X,y1,y1_opt)

## Target 2 (Capacity retention increase)
y2_opt = ExtraTreesClassifier(random_state=123, 
                              max_features=0.8, 
                              min_samples_leaf=1, 
                              min_samples_split=2,
                              n_estimators=237
                             )
LOO_cv(X,y2,y2_opt)



# Creating feature importance diagram for Target 1

y1_fit = y1_opt.fit(X, y1)
feat_importance = y1_fit.feature_importances_
sorted_idx = np.argsort(feat_importance)
pos = np.arange(sorted_idx.shape[0])
fig = plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
plt.barh(pos, feat_importance[sorted_idx], align='center')
plt.yticks(pos, np.array(X.columns)[sorted_idx])
plt.title('Feature Importance of "Initial Specific Capacity"')
sorted_feat = feat_importance[sorted_idx]
sort_fe = []
for feat in sorted_feat:
    fe = round(feat, 4)
    sort_fe.append(fe)
for index, value in enumerate(sort_fe):
    plt.text(value, index, str(value))
    
# Creating feature importance diagram for Target 2

y2_fit = y2_opt.fit(X, y2)
feat_importance = y2_fit.feature_importances_
sorted_idx = np.argsort(feat_importance)
pos = np.arange(sorted_idx.shape[0])
fig = plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
plt.barh(pos, feat_importance[sorted_idx], align='center')
plt.yticks(pos, np.array(X.columns)[sorted_idx])
plt.title('Feature Importance of "Capacity Retention Increase"')
sorted_feat = feat_importance[sorted_idx]
sort_fe = []
for feat in sorted_feat:
    fe = round(feat, 4)
    sort_fe.append(fe)
for index, value in enumerate(sort_fe):
    plt.text(value, index, str(value))
    

    
# Evaluating LOOCV

## Defining function
def LOOCV_eval (X, y, model):
    cvpred = np.zeros([len(X)]) # Membuat array kosong
    Xnp = X.to_numpy() #Konversi X ke array NumPy
    ynp = y.to_numpy() #Konversi y ke array NumPy
    
    LOOCVscore = cross_val_score(model, X, y, scoring="accuracy", cv=LeaveOneOut())
    
    for i in range(0,len(X)):
        xpred = Xnp[i,:].reshape(1,-1) 
        XLOO = np.delete(Xnp,i,axis=0) 
        yLOO = np.delete(ynp,i).reshape(-1,1) 
        modelLOO = model 
        modelLOO.fit(XLOO, yLOO.ravel()) 
        cvpred[i] = modelLOO.predict(xpred) 
    #Comparing values of train sets and test sets
    comparison=np.concatenate([cvpred.reshape(-1,1), ynp.reshape(-1,1), LOOCVscore.reshape(-1,1)], axis=1)
    print(comparison)
    
## Evaluating LOOCV of Target 1
LOOCV_eval (X, y1, y1_opt)

## Evaluating LOOCV of Target 2
LOOCV_eval (X, y2, y2_opt)

