import zipfile
import pandas as pd
import numpy as np
from sklearn import preprocessing,linear_model,cross_validation
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn import decomposition

train_data = pd.read_csv("train_data_fe.csv") #train_data is already transformed
del train_data['Unnamed: 0']
test_data = pd.read_csv("test_data_fe.csv")
del test_data['Unnamed: 0']

def read_csv_zip(filename):
    z = zipfile.ZipFile(filename+'.zip')
    df = pd.read_csv(z.open(filename))
    return df
train = read_csv_zip("train.csv")
test = read_csv_zip("test.csv")
category_encoded = preprocessing.LabelEncoder()
category_encoded.fit(train["Category"])
train["category_encoded"] = category_encoded.transform(train["Category"])

new_pca=decomposition.PCA(n_components=35)
new_pca.fit(train_data)
train_data_pca=new_pca.transform(train_data)
train_data_pca=pd.DataFrame(train_data_pca)

crime_rf = RandomForestClassifier(n_estimators=150,oob_score = True, min_samples_split=20, min_samples_leaf=10)
crime_rf.fit(train_data_pca,train['category_encoded'])
print "oob score:", crime_rf.oob_score_
train_error = log_loss(train["category_encoded"],np.array(crime_rf.predict_proba(train_data_pca)))
print "training error:", train_error
rf_score = cross_validation.cross_val_score(crime_rf,train_data_pca,train['category_encoded'],cv=5,scoring='log_loss')
print rf_score
test_data_pca = pd.DataFrame(new_pca.transform(test_data))
rf_predict = crime_rf.predict_proba(test_data_pca)
rf_result = pd.DataFrame(rf_predict, columns = category_encoded.classes_)
rf_result.to_csv("randomForest.csv",index=True,index_label ='Id' )

