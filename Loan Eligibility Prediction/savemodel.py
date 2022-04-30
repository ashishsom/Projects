from gettext import npgettext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
ssr=StandardScaler()
from sklearn.naive_bayes import GaussianNB
nbc=GaussianNB()
from sklearn import metrics
import joblib
#print('Libraries imported successfully')

link="S:\Data Science\Projects\My Projects\Loan Eligibility Prediction\Loan Eligibility Prediction.csv"
data=pd.read_csv(link)
#print(data)

# Split the data into train and test
array=data.values
X=array[:,0:9]
y=array[:,9:]

X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=0)

# Let's scale the data
X_train=ssr.fit_transform(X_train)
X_test=ssr.fit_transform(X_test)

# Modelling
nbc.fit(X_train,Y_train)

# Prediction
pred=nbc.predict(X_test)
#print(pred)

# Accuracy check 
acc=metrics.accuracy_score(pred,Y_test)
#print(f'The accuracy of the model is {acc}')

# Saving the model
joblib.dump(nbc,'loanpredictor.pkl')

print('Model is saved successfully')