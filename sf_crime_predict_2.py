import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import zipfile
from collections import Counter
from sklearn.ensemble import RandomForestClassifier

with zipfile.ZipFile('./input/train.csv.zip') as z:
    train = pd.read_csv(z.open('train.csv'))

with zipfile.ZipFile('./input/test.csv.zip') as z:
    test = pd.read_csv(z.open('test.csv'))

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


def get_unique_words(description):
    for word in description.translate(replace).split():
        if word not in unique_words:
            if word not in common_words:
                unique_words.append(word)

train.Descript.apply(get_unique_words)

for word in unique_words:
    word_counts[word] = train.Descript.apply(lambda x: word in x).sum()

word_df = pd.DataFrame.from_dict(word_counts, orient='index')
word_df.columns = ['count']
word_df = word_df.sort_values('count', ascending=False).head(words_to_keep)
predictors = list(word_df.index.values)

train_matrix = train[['Category', 'Descript']]

for pred in predictors:
    train_matrix[pred] = train_matrix.Descript.apply(
        lambda x: 1 if pred in x.translate(replace).split() else 0
    )

# word_counts = train_matrix.iloc[:, 3:52].sum(axis=1)
# print(Counter(word_counts.values).keys())  # equals to list(set(words))
# print(Counter(word_counts.values).values())  # counts the elements' frequency

train_classes = train_matrix.iloc[:, 0].values
train_features = train_matrix.iloc[:, 2:].values

forest = RandomForestClassifier(n_estimators=100)
model = forest.fit(train_features, train_classes)

score = model.score(train_features, train_classes)
print(score)

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
