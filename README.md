# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
## Algorithm
1. Import the required libraries and read the dataframe.
2. Assign hours to X and scores to Y.
3.  Implement training set and test set of the dataframe
4.  Plot the required graph both for test data and training data.

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: M SANJAY
RegisterNumber:  212222240090
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
#displaying the content in datafile
df.head()

df.tail()

x=df.iloc[:,:-1].values
print(x)

y=df.iloc[:,1].values
y

#splitting train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

#displaying predicted values
y_pred

#displaying actual values
y_test

#graph plot for training data
plt.scatter(x_train,y_train,color="blue")
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(y_test,y_pred)
print("MAE = ",mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```

## Output:
![image](https://github.com/Sanjay22006832/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119830477/55f44897-cc59-4d27-b464-5069bc2d61d1)
![image](https://github.com/Sanjay22006832/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119830477/4e633e72-8954-4826-8f6c-a1952529dc04)
![image](https://github.com/Sanjay22006832/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119830477/ad91fb98-98ab-4665-95f0-180a37ad78b8)
![image](https://github.com/Sanjay22006832/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119830477/b310bf5b-b55d-4a13-b39a-a314ad617ea3)
![image](https://github.com/Sanjay22006832/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119830477/a9582a08-6a40-42c3-821d-a8004461c024)
![image](https://github.com/Sanjay22006832/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119830477/787c9545-76b4-4fd8-8827-c3410185fb9e)
![image](https://github.com/Sanjay22006832/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119830477/8959f3c6-6e42-4d5b-af7f-b0fae18e306d)
![image](https://github.com/Sanjay22006832/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119830477/0ed54c30-7707-4e83-98f9-518fb9371394)
![image](https://github.com/Sanjay22006832/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119830477/95af4196-e3b6-4c7d-b52b-9166cb23482c)
![image](https://github.com/Sanjay22006832/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119830477/482a1f58-b45b-448c-a4f7-c3fee85d6149)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.



