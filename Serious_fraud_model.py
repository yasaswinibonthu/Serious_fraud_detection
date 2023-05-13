import pandas as pd
import numpy as np
import pickle

df = pd.read_csv('Serious_fraud.data')

X = np.array(df.iloc[:, 0:4])
y = np.array(df.iloc[:, 4:])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y.reshape(-1))

from sklearn.model_selection import train_test_split
train_X,val_X,train_Y,val_y = train_test_split(X, y, test_size=0.2)

from sklearn.svm import SVC
sv = SVC(kernel='linear').fit(X_train,y_train)


pickle.dump(sv, open('Serious_fraud_model.pkl', 'wb'))