import pandas as pd
import numpy as np
import string
import zipfile
from sklearn.ensemble import RandomForestClassifier

# Open csv files and store them as panda data frames
with zipfile.ZipFile('./input/train.csv.zip') as z:
    train = pd.read_csv(z.open('train.csv'))

with zipfile.ZipFile('./input/test.csv.zip') as z:
    test = pd.read_csv(z.open('test.csv'))

# Initialise variables
common_words = ['FROM', 'OF', 'WITH', 'A', 'OR', 'ON', 'THE', 'AND', 'IN',
                'TO', 'W', '&', 'INTO', 'FOR', 'BY', 'USE', 'WHERE', 'ARE',
                'AT', 'AN', 'AS', 'NOT', 'ATT', 'WITHOUT', 'NON', 'MADE',
                'S', 'U', 'FT', 'BB', 'AID', 'P', 'B']
unique_words = []
word_counts = {}
num_sample = 10000
words_to_keep = 250

np.random.seed(123)
train = train.sample(num_sample)
test = test.sample(num_sample)
replace = string.maketrans(string.punctuation, ' '*len(string.punctuation))

# Function to get the unique words from the crime classification descriptions
# and store in the unique_words list
def get_unique_words(description):
    for word in description.translate(replace).split():
        if word not in unique_words:
            if word not in common_words:
                unique_words.append(word)

# Apply 'get_unique_words' to the Descript column of the training data frame
train.Descript.apply(get_unique_words)

# Build up a dict which stores the count of unique words across the 'Descript'
# column in the training data frame
for word in unique_words:
    word_counts[word] = train.Descript.apply(lambda x: word in x).sum()

# Build a new data frame of the unique words and their count, so we can sort it
# and keep the most popular values in order to use as predictors
word_df = pd.DataFrame.from_dict(word_counts, orient='index')
word_df.columns = ['count']
word_df = word_df.sort_values('count', ascending=False).head(words_to_keep)
predictors = list(word_df.index.values)

# Build a new training matrix which has the crime 'Category', 'Descript' and
# a column for each relevant unique word which indicates whether it is included
# in the Descript or not
train_matrix = train[['Category', 'Descript']]

for pred in predictors:
    train_matrix[pred] = train_matrix.Descript.apply(
        lambda x: 1 if pred in x.translate(replace).split() else 0
    )

# Extract the class and features, then fit a Random Forest model
train_classes = train_matrix.iloc[:, 0].values
train_features = train_matrix.iloc[:, 2:].values
forest = RandomForestClassifier(n_estimators=100)
model = forest.fit(train_features, train_classes)

# Print the score
score = model.score(train_features, train_classes)
print(score)

# Build up a test matrix in the same manner as train (obviously we dont
# have the 'Category' column because that is what we are trying to predict
# N.B. This is where it all went wrong!!  The test matrix does not have a
# 'Descript' column - DOH!
test_matrix = test[['Descript']]

for pred in predictors:
    test_matrix[pred] = test_matrix.Descript.apply(
        lambda x: 1 if pred in x.translate(replace).split() else 0
    )

test_descriptions = test_matrix.iloc[:, 0].values
test_features = test_matrix.iloc[:, 1:].values

test_predict = model.predict(test_features)

compare = zip(test_descriptions, test_predict)
print(compare[:100])
