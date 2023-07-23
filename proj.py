# -*- coding: utf-8 -*-

!pip install opendatasets

"""menggunakan opendatasets untuk mengambil data"""

import opendatasets as od
od.download("https://www.kaggle.com/anmolkumar/health-insurance-cross-sell-prediction")

"""# Impor data

import library yang dibutuhkan
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# %matplotlib inline
import seaborn as sns

"""ambil data dengan pd.read_csv"""

df_train = pd.read_csv('/content/health-insurance-cross-sell-prediction/train.csv')
df_test = pd.read_csv('/content/health-insurance-cross-sell-prediction/test.csv')
df_sample = pd.read_csv('/content/health-insurance-cross-sell-prediction/sample_submission.csv')

"""Training data """

df_train.head()

"""Test data"""

df_test.head()

"""Sample yang disubmit """

df_sample.head()

"""# Data Information

*   id : Unique ID for the customer
*   Gender : Gender of the customer
*   Age : Age of the customer
*   Driving_License :
      * 0 : Customer does not have DL
      * 1 : Customer already has DL


*   Region_Code : Unique code for the region of the customer
*   Previously_Insured : 
    * 1 : Customer already has Vehicle Insurance, 
    * 0 : Customer doesn't have Vehicle Insurance
*   Vehicle_Age : Age of the Vehicle
*   Vehicle_Damage : 
    * 1 : Customer got his/her vehicle damaged in the past. 
    * 0 : Customer didn't get his/her vehicle damaged in the past.
*   Annual_Premium : The amount customer needs to pay as premium in the year
*   PolicySalesChannel : Anonymized Code for the channel of outreaching to the customer ie. Different Agents, Over Mail, Over Phone, In Person, etc.
*   Vintage : Number of Days, Customer has been associated with the company
*   Response :
    * 1 : Customer is interested 
    * 0 : Customer is not interested
"""

df_train.info()

"""Cek null data"""

df_train.isnull().sum()

"""Statistical data """

df_train.describe()

"""Drop kolom yang tidak digunakan """

df_train1 = df_train.drop('id',axis=1)

df_train1.head()

"""cek shape data"""

df_train1.shape

"""# Data Analysis

Ambil kolom numerik
"""

numeric_columns = []
for i in df_train1.columns:
  if(df_train1[i].dtypes == 'int64' or df_train[i].dtypes == 'float64'):
    numeric_columns.append(i)
print(numeric_columns)
numeric_columns.pop(1)
numeric_columns.pop(-1)
print(numeric_columns)

"""Ambil kolom kategori"""

categorical_columns = []
for i in df_train1.columns:
  if(df_train1[i].dtypes != 'int64' and df_train[i].dtypes != 'float64'):
    categorical_columns.append(i)
print(categorical_columns)

"""Cek outliers"""

for col in numeric_columns :
  sns.boxplot(x=df_train1[col])
  plt.show()

"""Mari kita cek korelasi dari data tersebut dengan heatmap"""

plt.figure(figsize = (12,5))
sns.heatmap(df_train1.corr(),annot=True)
plt.show()

"""Dari hasil boxplot dan heatmap, kita bisa drop row yang berada di outliers tersebut

Code dibawah merupakan menghapusan outliers
"""

Q1 = df_train1[numeric_columns].quantile(0.25)
Q3 = df_train1[numeric_columns].quantile(0.75)
IQR=Q3-Q1
other_df=df_train1[~((df_train1<(Q1-1.5*IQR))|(df_train1>(Q3+1.5*IQR))).any(axis=1)]
other_df.shape

"""hasil dari penghapusan outliers"""

other_df.head()

"""Sekarang kita akan cek berapa hasil response yang dibagi dengan gender"""

sns.countplot(x='Response',hue='Gender', data=other_df,palette='Set3')
plt.show()

"""Menariknya, banyak sekali yang menolak ketimbang menerima tawaran asuransi. Hal ini dibuktikan dari lebih banyak yang memilih angka 0 daripada angka 1. Terlebih lagi, pria sangat mendominasi dalam grafik ini.

Sekarang kita akan cek siapa yang punya driving license
"""

sns.countplot(x='Driving_License',hue='Gender', data=other_df,palette='Set2')
plt.show()
other_df.Driving_License.value_counts()

"""Dari grafik ini, dibuktikan bahwa banyak pria yang memiliki driving lisence ketimbang wanita"""

sns.countplot(x='Driving_License',hue='Response', data=other_df,palette='Set2')
plt.show()

"""Seperti yang terduga, banyak sekali orang yang berkata tidak pada asuransi dan memiliki driving license. Tetapi yang menarik adalah ada juga yang tidak memiliki driving license dan tidak juga mengambil asuransi. Tentu itu hal yang wajar karena mereka belum punya driving license

Code dibawah ini adalah code untuk histogram setiap kolom
"""

other_df.hist(bins=30,figsize=(15,10))
plt.show()

"""Code dibawah ini adalah code untuk violinplot yang akan dijelaskan tujuannya dibawah grafik ini."""

plt.figure(figsize=(12,5))
sns.violinplot(x = 'Response',y='Age',hue='Gender',data = other_df,palette = 'mako')
plt.show()

"""Grafik ini menandakan bahwa orang dewasa berumur 20-30 tahun lebih banyak berkata tidak pada asuransi. Hal ini didukung dari grafik histogram sebelumnya yang menunjukan bahwa orang berumur 20- 40 memiliki angka yang lebih besar jika dibandingkan dengan orang berumur 40 keatas. Sehingga ini menandakan bahwa besarnya respon tidak pada asuransi disebabkan karena banyak yang menawarkan asuransi kepada orang berumur 20-30 tahun. Sedangkan orang yang menerima asuransi adalah orang berumur 40-50 tahun.

Selanjutnya, mari kita cek apakah data ini adalah data yang cocok untuk klasifikasi linear atau tidak.
"""

sns.pairplot(other_df,hue='Response',diag_kind = 'kde')

"""Dari grafik *pair plot* ini, kita bisa simpulkan bahwa data tersebut tidak linear atau *non-linear* sehingga kita akan menggunakan model yang dapat digunakan untuk klasifikasi *non-linear*.

# Data Preparation

Ini merupakan salah satu teknik data preparation yaitu *one-hot encoding data*. Data yang dimasukan ke dalam ini adalah data yang mengandung kolom kategori. Fungsi dari *one-hot encoding* adalah untuk membuat kolom data kategori dibagi menjadi beberapa kolom sesuai dengan jumlah fitur yang terdapat dalam kolom tersebut dan mengubah fitur dari setiap kolom yang sudah dibagi menjadi bilangan binari.
"""

for col in categorical_columns:
  other_df = pd.concat([other_df, pd.get_dummies(other_df[col], prefix=col, drop_first=True)],axis=1)
  other_df = other_df.drop(col,axis=1)

other_df

"""Mengubah nama kolom agar diterima oleh machine learning model"""

other_df.rename(columns = {'Vehicle_Age_< 1 Year':'Vehicle_Age_less_1_Year','Vehicle_Age_> 2 Years':'Vehicle_Age_more_2_Years'},inplace = True)
other_df

"""Data di split menjadi train dan test untuk melatih model dan membuktikan bahwa model yang sudah dilatih tidak overfitting."""

from sklearn.model_selection import train_test_split
X = other_df.drop('Response',axis=1)
y = other_df['Response']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,stratify = y, random_state = 42)

"""Ini merupakan salah satu teknik data preparation yaitu *standard scaler*. Metode ini digunakan untuk membuat dataset bisa lebih diterima oleh model dan menghasilkan akurasi yang lebih baik."""

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

"""# Model Training and Evaluation

Import library yang dibutuhkan
"""

import time

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

"""penentuan model"""

dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
xg = XGBClassifier()

"""Pembuatan fungsi untuk train, prediksi, dan evaluation. Di dalam fungsi ini, terdapat time dimana time tersebut akan menghitung waktu training model. Kemudian untuk metrik terdapat *accuracy score* , *confusion matrix* dan *classification report*. Hasil dari fungsi ini adalah waktu yang dipakai selama training, akurasi model, *confussion matrix* dan *classification report*."""

def model_test(model):
  start = time.time()
  model.fit(X_train,y_train)
  end = time.time()
  time_spent = end-start
  pred = model.predict(X_test)
  acc = accuracy_score(y_test,pred)
  print(confusion_matrix(y_test,pred))
  print("Training time spend : {} seconds".format(round(time_spent,3)))
  print("Accuracy            : {}".format(round(acc,3)))
  print(classification_report(y_test,pred))

"""# Hasil Decision Tree"""

print("Decision Tree")
model_test(dt)

"""# Hasil Random Forest Tree"""

print("Random Forest Tree")
model_test(rf)

"""# Hasil XGBoost"""

print('XGB')
model_test(xg)