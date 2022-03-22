# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
import random
from evidently.dashboard import Dashboard
from evidently.tabs import DataDriftTab, ClassificationPerformanceTab
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.tabs import DataDriftTab, CatTargetDriftTab
from sklearn.model_selection import train_test_split
from alibi_detect.cd import ChiSquareDrift, TabularDrift
from alibi_detect.utils.saving import save_detector, load_detector
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support, f1_score, recall_score
from sklearn.preprocessing import LabelEncoder


warnings.filterwarnings('ignore')


df = pd.read_csv("Data/predictive_maintenance.csv")
df.columns = [x.lower() for x in df.columns]
df.columns = [x.replace(" ","_") for x in df.columns]

id_var = df.iloc[:,[0,1]]
num_var = df.iloc[:,[3,4,5,6,7]]
cat_var = df.iloc[:,[2,8,9]]

print(f"""Variable type distribution:
identifying variables:{id_var.shape[1]} ({", ".join(id_var.columns)})
categorical variables:{cat_var.shape[1]} ({", ".join(cat_var.columns)})
    numeric variables:{num_var.shape[1]} ({", ".join(num_var.columns)})""")
    
print(f'''Target distribution:
no failures: {df['target'].value_counts()[0]}
   failures:  {df['target'].value_counts()[1]}''')
   
df[df['target'] == 0]['failure_type'].value_counts()

#Preprocessing step - Lable encoding

#from sklearn import preprocessing
#label_encoder = preprocessing.LabelEncoder()
enc_dict = {'L':0,
            'M':1,
            'H':2}
# Create the mapped values in a new column
df['type'] = df['type'].map(enc_dict)

#Split data into training and test datasets


X = df.drop(columns = ['failure_type', 'target', 'udi', 'product_id'])
y = df['failure_type']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=2022)

print(f'''Shape of train and test data:
training data:{X_train.shape[0]} obs
testing data :{X_test.shape[0]} obs''')

#Model drift data preparation

X_drift = X_test.copy()
y_drift = y_test.copy()

# drift in numeric variable - rotational speed

arr = np.array(X_drift['rotational_speed_[rpm]'])
for i in range(len(X_drift)):
    arr[i] = random.randint(1600, 2900)
X_drift['rotational_speed_[rpm]'] = arr

# drift in categorical variable - type
X_drift = X_drift.sort_values(by = 'type', ascending = True)

X_drift['type'].iloc[0:1000] = [2 for i in (X_drift['type'].iloc[0:1000]) ]
X_drift['type'].iloc[1000:2000] = [0 for i in (X_drift['type'].iloc[1000:2000]) ]
X_drift['type'].iloc[2000:3000] = [1 for i in (X_drift['type'].iloc[2000:3000]) ]

print(f'''Shape of drift data:
drift data:{X_drift.shape[0]} obs''')

df.describe()

print(f"""Rotation speed distribution in historical dataset:
      avg: {df['rotational_speed_[rpm]'].describe()[1]}
      min: {df['rotational_speed_[rpm]'].describe()[3]}
   median: {df['rotational_speed_[rpm]'].describe()[5]}
      max: {df['rotational_speed_[rpm]'].describe()[-1]}\n""")

print(f"""Rotation speed distribution in drift dataset:
      avg: {X_drift['rotational_speed_[rpm]'].describe()[1]}
      min: {X_drift['rotational_speed_[rpm]'].describe()[3]}
   median: {X_drift['rotational_speed_[rpm]'].describe()[5]}
      max: {X_drift['rotational_speed_[rpm]'].describe()[-1]}""")
      
      
hist_avg = df['rotational_speed_[rpm]'].describe()[1]
hist_min = df['rotational_speed_[rpm]'].describe()[3]
hist_median = df['rotational_speed_[rpm]'].describe()[5]
hist_max = df['rotational_speed_[rpm]'].describe()[-1]
hist_rotational_speed = [hist_min, hist_median, hist_avg, hist_max ]

drift_avg = X_drift['rotational_speed_[rpm]'].describe()[1]
drift_min = X_drift['rotational_speed_[rpm]'].describe()[3]
drift_median = X_drift['rotational_speed_[rpm]'].describe()[5]
drift_max = X_drift['rotational_speed_[rpm]'].describe()[-1]
drift_rotational_speed = [drift_min, drift_median, drift_avg, drift_max ]

labels = ['min', ' median', 'avg', 'max']
plt.plot(hist_rotational_speed, label ='historical')
plt.plot(drift_rotational_speed, label = 'drift')
plt.xticks([0.0, 1.0, 2.0, 3.0], ['min', 'median', 'avg', 'max'])
plt.grid(axis = 'x')
plt.legend()
plt.title('Rotational speed dist - historical v/s drift')

plt.savefig('historical_drift.png')

hist_type = X_test['type'].value_counts().reset_index().rename(columns = {'type' : 'historical'})

drift_type = X_drift['type'].value_counts().reset_index().rename(columns = {'type' : 'drift'})

hist_type = hist_type.merge(drift_type, how = 'inner', on = ['index'])

plt.rcParams['figure.figsize'] = (8,3)
hist_type.plot(x='index',
        kind='bar',
        stacked=False, color={"historical": "red", "drift": "green"},
        title='Type dist: historical v/s drift data')
plt.xticks([0, 1, 2], ['Low', 'Medium', 'High'], rotation = 0)

plt.savefig('historical_data_type.png')

X_train['target'] = df['failure_type']
X_test['target'] = df['failure_type']
X_drift['target'] = df['failure_type']

X_drift1 = X_drift.copy()
X_drift1['target'] = np.where(X_drift1['target'] == 'Random Failures', 'Overstrain Failure', X_drift1['target'])

d1 = X_drift1['target'].value_counts().reset_index().rename(columns = {'target':'drift'})

d = X_drift['target'].value_counts().reset_index()
d = d.merge(d1, how = 'left', on = 'index')
d = d.fillna(0)
d.drift = d.drift.astype('int')

drift = pd.concat([X_drift.loc[:,['target']].rename(columns = {'target':'historical'}), 
                   X_drift1.loc[:,['target']]], axis =1)
                   
plt.rcParams['figure.figsize'] = (10,4)
d.plot(x='index',
        kind='bar',
        stacked=False,
        title='Failure type dist: historical v/s drift')
plt.ylim([0,50])
plt.xticks(rotation = 0)
plt.savefig('failures.png')

X_drift['target'].value_counts()

X_drift['target'] = np.where(X_drift['target'] == 'Random Failures', 'Overstrain Failure', X_drift['target'])
X_drift['target'].value_counts()

#Evidently
#Data drift - test and drift

drift_column_mapping = ColumnMapping()
drift_column_mapping.categorical_features = ['type']
drift_column_mapping.numerical_features = ['air_temperature_[k]', 'process_temperature_[k]'
                                          ,'rotational_speed_[rpm]', 'torque_[nm]', 'tool_wear_[min]' ]

data_drift_report = Dashboard(tabs=[DataDriftTab()])
data_drift_report.calculate(X_test,X_drift,column_mapping = drift_column_mapping)

data_drift_report.save('data_drift_evidently.html')

#Concept drift - test and drift


column_mapping = ColumnMapping()

column_mapping.categorical_features = ['type']
column_mapping.numerical_features = ['air_temperature_[k]', 'process_temperature_[k]'
                                          ,'rotational_speed_[rpm]', 'torque_[nm]', 'tool_wear_[min]' ]
column_mapping.target = 'target'

concept_drift_report = Dashboard(tabs=[DataDriftTab(),CatTargetDriftTab()])
concept_drift_report.calculate(X_test,X_drift,column_mapping = column_mapping)

concept_drift_report.save('concept_drift_evidently.html')


#Alibi Detect
#Data drift (also called co-variate drift)


type_conv = {0:'L',
            1:'M',
            2:'H'}
df['type'] = df['type'].map(type_conv)
X = df.drop(columns = ['failure_type', 'target', 'udi', 'product_id'])
y = df['failure_type']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=2022)

print(f'''Shape of train and test data:
training data:{X_train.shape[0]} obs
testing data :{X_test.shape[0]} obs''')

#import alibi


##### initialize the detector
category_map = {0: None}
categories_per_feature = {f: None for f in list(category_map.keys())}
cd = TabularDrift(X_test.values, p_val=.05, categories_per_feature=categories_per_feature)

dataset = {'X_test': X_test, 'X_drift': X_drift}

out_dfs = pd.DataFrame()
for k, v in dataset.items():
    print(k)
    preds = cd.predict(dataset[k].values)
    
    out_df = pd.DataFrame()
    for f in range(cd.n_features):
        #print(preds)
        stat = 'Chi2' if f in list(categories_per_feature.keys()) else 'K-S'
        fname = list(v.columns)[f]
        stat_val = preds['data']['distance'][f]
        p_val = preds['data']['p_val'][f]
        #print(f'{fname} -- {stat} {stat_val:.3f} -- p-value {p_val:.3f}')
        drift = ''
        if p_val <= 0.05:
            drift = 'yes'
            print(f'{fname} -- {stat} {stat_val:.3f} -- p-value {p_val:.3f} **{drift}**')
        else:
            print(f'{fname} -- {stat} {stat_val:.3f} -- p-value {p_val:.3f}')
        drift_list = [k, fname, stat, p_val , drift]
        k_df  = pd.DataFrame([drift_list], columns = ['dataset','attr','stat','p-value','drift'])
        #print(f'{fname} -- {stat} {stat_val:.3f} -- p-value {p_val:.3f}')
        out_df = out_df.append(k_df)
    out_dfs = out_dfs.append(out_df)
    print(f'\n')
    
#Concept drift

print(f'''Failure % of Total = {(1-(9652/10000)):.2f}%''')


le = LabelEncoder()
df['failure_type'] = le.fit_transform(df['failure_type'])
df['type'] = X['type'].map(enc_dict)

X = df.drop(columns = ['failure_type', 'target', 'udi', 'product_id'])
y = df['failure_type']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=2022)

classifier = lgb.LGBMClassifier()
classifier.fit(X_train.values,y_train.values)
predictions = classifier.predict(X_test.values)
print(f"Accuracy: {recall_score(y_test,predictions, average ='micro'):.2f}%")

try:
    X_drift = X_drift.drop(['target'], axis = 1)
except:
    pass
y_drift = y_test.copy()
predictions = classifier.predict(X_drift.values)
print(f"Accuracy: {recall_score(y_drift,predictions, average='micro'):.2f}%")