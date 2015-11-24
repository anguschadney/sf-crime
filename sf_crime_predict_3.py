import pandas as pd
import numpy as np
import zipfile
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

#####################################################################
# Author: Gus Chadney
# Date: 11/11/15
#
# Title: Kaggle San Fransisco Crime Classification
#
# Description: In this challenge a data set was made available which
#              had a list of recorded crimes in the San Fransisco
# area, along with metadata which recorded the datetime and location
# that the crime occured.  After loading the data into a data frame,
# the latitude and longitude variables were digitised into n bins, and
# the numerical, day of week and hour of day were extracted from the
# datetime variable.
# A number of models were fitted, focusing on Random Forests and
# K nearest neighbours (more details below).
#
#####################################################################

#####################################################################
# Model/feature selection:
# Algo 1:  'Xbin', 'Ybin', 'day' - RF(n=20) - score = 8.22
#          increased lat/long bin size from 1000 to 120000
# Algo 2:  'Xbin', 'Ybin', 'day', 'hour', 'dayofyear' - RF(n=50) - score = 10.32
# Algo 3:  'Xbin', 'Ybin', 'day', 'hour' - RF(n=30, max_features=4) - score = 13.10
# Algo 4:  'Xbin', 'Ybin', 'day', 'hour' - RF(n=20) - score = 7.19
#          reduced lat/long bin size from 12000 to 1200
# Algo 5:  'Xbin', 'Ybin', 'day', 'hour' - KNN(n=5) - score = 20.83
# Algo 6:  'Xbin', 'Ybin' - KNN(n=5) - score = 21.76
# Algo 7:  Average of the following 2 algos (bin size 10000)
#          'Xbin', 'Ybin', 'day', 'hour' - RF(n=20)
#                                        - KNN(n=5)
#                                          score = 11.24
# Algo 8:  Average of the following 2 algos (bin size 1000)
#          'Xbin', 'Ybin', 'day', 'hour' - RF(n=20)
#                                        - KNN(n=10)
#                                          score = 6.00
# Algo 9:  Average of the following 2 algos (bin size 1000)
#          'Xbin', 'Ybin', 'day', 'hour' - RF(n=50)
#                                        - KNN(n=15)
#                                          score = 5.74
#####################################################################

# Dict to map day of week to number
dayOfWeekMap = {'Monday': 0,
                'Tuesday': 1,
                'Wednesday': 2,
                'Thursday': 3,
                'Friday': 4,
                'Saturday': 5,
                'Sunday': 6}

# Open csv files and store them as panda data frames
print('Loading data...\n')
with zipfile.ZipFile('./input/train.csv.zip') as z:
    train = pd.read_csv(z.open('train.csv'))
with zipfile.ZipFile('./input/test.csv.zip') as z:
    test = pd.read_csv(z.open('test.csv'))

# Cut down the columns to those of interest
#  Prediction class: 'Category'
#  Prediction features: 'Dates', 'DayOfWeek', 'X' (lat), 'Y' (long)
print('Extracting relevant features...\n')
train_df = train[['Category', 'Dates', 'DayOfWeek', 'X', 'Y']]
test_df = test[['Id', 'Dates', 'DayOfWeek', 'X', 'Y']]

# Digitize lat/long values into bins to make the number of unique values
# much smaller, store in new columns 'Xbin' 'Ybin'
print('Processing features - digitizing latitude/longitude...\n')
num_bins = 1000
X_bins = np.linspace(train.X.min(), train.X.max(), num_bins)
Y_bins = np.linspace(train.Y.min(), train.Y.max(), num_bins)

train_df['Xbin'] = pd.Series(np.digitize(train_df.X, X_bins),
                             index=train_df.index)
train_df['Ybin'] = pd.Series(np.digitize(train_df.Y, Y_bins),
                             index=train_df.index)

test_df['Xbin'] = pd.Series(np.digitize(test_df.X, X_bins),
                            index=test_df.index)
test_df['Ybin'] = pd.Series(np.digitize(test_df.Y, Y_bins),
                            index=test_df.index)

# Extract the hour from the datetime string 'Dates' and store in a new column 'hour'
# Should expect to see more crime in the evening
print('Processing features - Extracting hour from date...\n')
train_df['hour'] = pd.to_datetime(train_df.Dates).dt.hour
test_df['hour'] = pd.to_datetime(test_df.Dates).dt.hour

# Map the DayOfWeek to a number and store in new column 'day'
# Likely to see more crime on the weekend
print('Processing features - Extracting day of week from date...\n')
train_df['day'] = pd.to_datetime(train_df.Dates).dt.dayofweek
test_df['day'] = pd.to_datetime(test_df.Dates).dt.dayofweek

# Extract the day of year from the datetime string 'Dates' and store in a new column 'dayofyear'
# Could indicate crime because of significant dates (new years, christmas etc.).  Also, payday
# could see a spike in crime (alcohol, violence, drug, etc)
# print('Processing features - Extracting day of year from date...\n')
train_df['dayofyear'] = pd.to_datetime(train_df.Dates).dt.dayofyear
test_df['dayofyear'] = pd.to_datetime(test_df.Dates).dt.dayofyear

# Cut DFs down again to the class and processed features
train_df = train_df[['Category', 'Xbin', 'Ybin', 'hour', 'day']]
test_df = test_df[['Id', 'Xbin', 'Ybin', 'hour', 'day']]

# Cut the train DF into classes and features
train_classes = train_df.iloc[:, 0].values
train_features = train_df.iloc[:, 1:].values

# Initialise prediction algo. As this is a classification problem
# I am using a random forest classifier.
print('Starting Algo''s...\n')
forest = RandomForestClassifier(n_estimators=40,
                                random_state=123)

knn = KNeighborsClassifier(n_neighbors=15)

# Fit the algo to the training set
print('Fitting training set to RF...')
model_rf = forest.fit(train_features, train_classes)
print('Fitting training set to KNN...\n')
model_knn = knn.fit(train_features, train_classes)

# Calculate the accuracy on the training set
score_rf = model_rf.score(train_features, train_classes)
print('Accuracy on training set (RF) = {0}%'.format(score_rf*100))
score_knn = model_knn.score(train_features, train_classes)
print('Accuracy on training set (KNN) = {0}%\n'.format(score_knn*100))

# Fit the model to the test features, the output is a grid
# of probabilities of the likelihood of each class
test_features = test_df.iloc[:, 1:].values
print('Fitting test set (RF)...')
predict_rf = model_rf.predict_proba(test_features)
predict_df_rf = pd.DataFrame(predict_rf, columns=model_rf.classes_)
print('Fitting test set (KNN)...\n')
predict_knn = model_knn.predict_proba(test_features)
predict_df_knn = pd.DataFrame(predict_knn, columns=model_knn.classes_)

# Average the results
print('Averaging results...\n')
predict_df = (predict_df_rf + predict_df_knn) / 2

# Insert the test Ids so we can easily print to csv
predict_df.insert(0, 'Id', test_df.Id.values)

print('Writing to csv...\n')
predict_df.to_csv('sf_crime_pred.csv', index=False)

print('Finished!')
