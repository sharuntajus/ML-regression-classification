

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"

data = pd.read_csv(url)
print(data.head())
print(data.sample(55))

print(data.describe())

columnData = data[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
print(columnData.sample(9))

visualize=columnData[['CYLINDERS','ENGINESIZE','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
visualize.hist()
plt.show()

plt.scatter(columnData.FUELCONSUMPTION_COMB, columnData.CO2EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()

plt.scatter(columnData.ENGINESIZE, columnData.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.xlim(0,27)
plt.show()

plt.scatter(columnData.CYLINDERS, columnData.CO2EMISSIONS,  color='blue')
plt.xlabel("CYLINDERS")
plt.ylabel("Emission")
plt.show()

X = columnData.ENGINESIZE.to_numpy()
y = columnData.CO2EMISSIONS.to_numpy()


from sklearn.model_selection import train_test_split
#took .2 of test size means 20% for testing and 80 for training and randomstate to fixed
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

print("Type:", type(X_train))#Type: <class 'numpy.ndarray'>
print("Shape:", X_train.shape) #Shape: (853,)
#kindofObj and Dimensions
type(X_train), np.shape(X_train), np.shape(X_train)

from sklearn import linear_model
regressor = linear_model.LinearRegression()
#(853, 1) — a column vector - reshaped
regressor.fit(X_train.reshape(-1, 1), y_train)
print ('Coefficients: ', regressor.coef_[0]) # with simple linear regression there is only one coefficient, here we extract it from the 1 by 1 array.
print ('Intercept: ',regressor.intercept_)

plt.scatter(X_train, y_train,  color='green')
plt.plot(X_train, regressor.coef_ * X_train + regressor.intercept_, '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Use the predict method to make test predictions
y_test_ = regressor.predict(X_test.reshape(-1,1))

# Evaluation
print("Mean absolute error: %.2f" % mean_absolute_error(y_test, y_test_))
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_test_))
print("Root mean squared error: %.2f" % np.sqrt(mean_squared_error(y_test, y_test_)))
print("R2-score: %.2f" % r2_score(y_test, y_test_))