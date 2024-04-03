# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import zero_one_loss
from PIL import Image  
import warnings
warnings.filterwarnings("ignore")

# read the data        
data = pd.read_csv("train.csv")
data.head()


X = data.iloc[:, 1:]
y = data['label']

# print image
tmp = np.array(X.iloc[10].values.reshape((28,28))).astype(np.uint8)
img = Image.fromarray(tmp)
img

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.25, random_state=1179)


##############################################################    
# Logistic regression
##############################################################
    
reg = LogisticRegression(solver = "lbfgs") 
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
#
print("Logistic Regression 0/1 loss = ", zero_one_loss(y_pred, y_test))



##############################################################    
# Random forest    
##############################################################
rfc = RandomForestClassifier(n_jobs=-1, n_estimators=100)    
rfc.fit(X_train,y_train )
#
y_pred = rfc.predict(X_test)
#
print("Random Forest 0/1 loss = ", zero_one_loss(y_pred, y_test))
#


