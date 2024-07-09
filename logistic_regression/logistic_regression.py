import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Creating the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
