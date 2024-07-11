import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Creating the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# There no missing value or categorical feature in the dataset

# Splitting the data into training and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# Feature scaling
scalar = StandardScaler()
x_train = scalar.fit_transform(x_train)
x_test = scalar.transform(x_test)

# Training the model on our training and test sets
classifier = LogisticRegression()
classifier.fit(x_train, y_train)

print(classifier.predict(scalar.transform([[30, 87000]])))

y_pred = classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
