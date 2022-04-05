# ca05

### Packages Used:

import pandas as pd

import numpy as np 
import sklearn
from sklearn import linear_model
import sklearn.metrics as metrics
import sklearn.linear_model as lm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import random
import seaborn as sns
from scipy.stats import uniform
import matplotlib.pyplot as plt


## Description

Using logistic regression this project attempts to predict cardiovascular disease using various parameters.  In this repository we can see the usage of 4 different models each with varying accuracy in order to determine which one is best. 

## Datasets

There is no requirement to download a dataset or mount google drive.  The code in the repository should directly extract the dataset from this code:

pd.read_csv('https://github.com/ArinB/CA05-B-Logistic-Regression/raw/master/cvd_data.csv')

## Additional Comments

If you have any additional comments or improvements please send to the author!
