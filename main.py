#Importing Libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,make_scorer,classification_report, confusion_matrix, roc_curve, roc_auc_score,auc, accuracy_score,adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn import ensemble
import xgboost as XGB
from sklearn.preprocessing import StandardScaler
import warnings

#Ignoring warnings

warnings.filterwarnings("ignore")

# Set to display all the columns in dataset

pd.set_option("display.max_columns", None)

# Load and display the dataset

df = pd.read_csv(r"E:\aiml and ds\01-project\music_genre.csv", header=0)
df.head()

#Creating a backup file

df_BK=df.copy()

#Displaying info of the dataset
df.info()

#Droping the non influencing columns

df=df.drop(['instance_id','artist_name','duration_ms','obtained_date'],axis=1)
df.head()

df.info()

#differentiating dicrete and continuous data sets

discr_feat = ['track_name', 'popularity','key', 'mode','music_genre','tempo']
cont_feat = ['popularity','acousticness', 'energy','instrumentalness','liveness','loudness','speechiness','valence']

#checking the no.of rows against columns

df.shape

#chekcing the description of the dataset

df.describe()

#chekcing for null values

df.isnull().sum()

df.info()

#use labelencoder for target variables

from sklearn.preprocessing import LabelEncoder

LE=LabelEncoder()

df['key']=LE.fit_transform(df['key'])
df['mode']=LE.fit_transform(df['mode'])
df['tempo']=LE.fit_transform(df['tempo'])
df['track_name']=LE.fit_transform(df['track_name'])

# Use SimpleImputer to address missing values

from sklearn.impute import SimpleImputer

imputer_str = SimpleImputer(missing_values=np.nan, strategy='most_frequent', fill_value=None, verbose=0,
                            copy=True, add_indicator=False)

df['track_name'] = imputer_str.fit_transform(df[['track_name']])
df['key'] = imputer_str.fit_transform(df[['key']])
df['mode'] = imputer_str.fit_transform(df[['mode']])

df.isnull().sum()

# Use KNNImputer to address missing values

from sklearn.impute import KNNImputer

imputer_int = KNNImputer(missing_values=np.nan, n_neighbors=5, weights='uniform', metric='nan_euclidean',
                         copy=True, add_indicator=False)

df['popularity'] = imputer_int.fit_transform(df[['popularity']])
df['acousticness'] = imputer_int.fit_transform(df[['acousticness']])
df['danceability'] = imputer_int.fit_transform(df[['danceability']])
df['energy'] = imputer_int.fit_transform(df[['energy']])
df['instrumentalness'] = imputer_int.fit_transform(df[['instrumentalness']])
df['liveness'] = imputer_int.fit_transform(df[['liveness']])
df['loudness'] = imputer_int.fit_transform(df[['loudness']])
df['tempo'] = imputer_str.fit_transform(df[['tempo']])
df['speechiness'] = imputer_int.fit_transform(df[['speechiness']])
df['valence'] = imputer_int.fit_transform(df[['valence']])

df.isnull().sum()

df.music_genre.value_counts()

#labelling music genre type

Electronic = df.music_genre=='Electronic'
Anime = df.music_genre=='Anime'
Jazz = df.music_genre=='Jazz'
Alternative = df.music_genre=='Alternative'
Country = df.music_genre=='Country'
Rap = df.music_genre=='Rap'
Blues = df.music_genre=='Blues'
Rock = df.music_genre=='Rock'
Classical = df.music_genre=='Classical'
HipHop = df.music_genre=='Hip-Hop'

cols1=['popularity','loudness','tempo']

# Scaling the features by using MinMaxScaler

from sklearn.preprocessing import MinMaxScaler

mmscaler = MinMaxScaler(feature_range=(0, 1))

df[cols1] = mmscaler.fit_transform(df[cols1])
df = pd.DataFrame(df)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

df = df.drop(['track_name','key', 'mode','tempo'], axis=1)
df.head()

df.index[df.isnull().any(axis=1)]

for i in df.index[df.isnull().any(axis=1)]:
    print("In row",i)
    print(    df.iloc[i        ,:]  )
for i in df.index[df.isnull().any(axis=1)]:
    df=df.drop(i)
df.isnull().sum()

# Identify the independent and Target (dependent) variables

IndepVar = []
for col in df.columns:
    if col != 'music_genre':
        IndepVar.append(col)

TargetVar = 'music_genre'

x = df[IndepVar]
y = df[TargetVar]
# Splitting the dataset into train and test 

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 42)

# Display the shape 

x_train.shape, x_test.shape, y_train.shape, y_test.shape

lr = LogisticRegression()
lr.fit(x_train,y_train)

y_lr = lr.predict(x_test)
print(y_lr)

lr_score = accuracy_score(y_test, y_lr)
print(lr_score)

rf = ensemble.RandomForestClassifier()
rf.fit(x_train, y_train)
y_rf = rf.predict(x_test)
print(classification_report(y_test, y_rf))

rf_score = accuracy_score(y_test, y_rf)
print(rf_score)
cm = confusion_matrix(y_test, y_rf)
print(cm)
importances = rf.feature_importances_
indices = np.argsort(importances)
plt.figure(figsize=(12,8))
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), df.columns[indices])
plt.title('Importance of characteristics')

#use labelencoder for target variables

from sklearn.preprocessing import LabelEncoder

LE=LabelEncoder()

df['music_genre']=LE.fit_transform(df['music_genre'])
df.head()

# Identify the independent and Target (dependent) variables

IndepVar = []
for col in df.columns:
    if col != 'music_genre':
        IndepVar.append(col)

TargetVar = 'music_genre'

xxgb = df[IndepVar]
yxgb = df[TargetVar]
# Splitting the dataset into train and test 

from sklearn.model_selection import train_test_split

xxgb_train, xxgb_test, yxgb_train, yxgb_test = train_test_split(xxgb, yxgb, test_size = 0.30, random_state = 42)

# Display the shape 

xxgb_train.shape, xxgb_test.shape, yxgb_train.shape, yxgb_test.shape

yxgb.sample(10)
xgb  = XGB.XGBClassifier()
xgb.fit(xxgb_train, yxgb_train)
y_xgb = xgb.predict(xxgb_test)
cm = confusion_matrix(yxgb_test, y_xgb)
print(cm)
print(classification_report(yxgb_test, y_xgb))
music_features = df.drop("music_genre", axis = 1)
music_labels = df["music_genre"]
scaler = StandardScaler()
music_features_scaled = scaler.fit_transform(music_features)
music_features_scaled.mean(), music_features_scaled.std()
tr_val_f, test_features, tr_val_l, test_labels = train_test_split(music_features_scaled, music_labels, test_size = 0.1, stratify = music_labels)
train_features, val_features, train_labels, val_labels = train_test_split(
    tr_val_f, tr_val_l, test_size = len(test_labels), stratify = tr_val_l)
train_features.shape, train_labels.shape, val_features.shape, val_labels.shape, test_features.shape,   test_labels.shape
f1 = make_scorer(f1_score, average = "weighted")
model = RandomForestClassifier(n_estimators = 35, max_depth = 15, min_samples_leaf = 4)
def classification_task(estimator, features, labels):
    estimator.fit(features, labels)
    predictions = estimator.predict(features)
    
    print(f"Accuracy: {accuracy_score(labels, predictions)}")
    print(f"F1 score: {f1_score(labels, predictions, average = 'weighted')}")
classification_task(model, train_features, train_labels)
classification_task(model, val_features, val_labels)
classification_task(model, test_features, test_labels)
