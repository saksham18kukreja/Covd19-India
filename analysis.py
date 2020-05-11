# importing the libraries needed for analysis
import pandas as pd
import numpy as np

# making a database
df = pd.read_csv('case_time_series.csv')

# sort out the columns needed
Y = df['Daily Confirmed']
X = df[['Total Confirmed']]

# make a linear regression model 
from sklearn.linear_model import LinearRegression
model1 = LinearRegression()

model1.fit(X,Y)

# random values for the prediction
X1 = np.array(range(0,80000,500)).reshape(-1,1)
Y_pred = model1.predict(X1)

#plotting the linear regression curve
import matplotlib.pyplot as plt
plt.show()
plt.scatter(X,Y,label='actual values')
plt.plot(X1,Y_pred,'r',label='predicted values')
plt.xlabel('total cases')
plt.ylabel('no. of new cases per day')
plt.title('Corona Cases in india')
plt.legend()

#calculate the accuracy of the model (0.9625)
model1.score(X,Y)

# make a polynomial regression model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
polynomial = PolynomialFeatures(degree=4)
X_poly = polynomial.fit_transform(X)

polynomial.fit(X_poly,Y)
model2 = Ridge(0.5)
model2.fit(X_poly,Y)

Y_poly = model2.predict(polynomial.fit_transform(X1))

#plot the curve for the poly-regression
plt.scatter(X,Y,label='actual cases')
plt.plot(X1,Y_poly,'r',label='predictions')
plt.xlabel('total cases')
plt.ylabel('new cases per day')
plt.title('Covid Cases in India')
plt.legend()

#calculate the accuracy of the model (0.9814)
model2.score(X_poly,Y)
