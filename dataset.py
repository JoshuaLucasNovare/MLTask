import pandas as pd

dataset = pd.read_csv('data.csv')
dataset = dataset.drop('Unnamed: 32', axis =1)
dataset = dataset.drop('id', axis =1)

dataset.loc[(dataset['diagnosis'] == 'M'), 'diagnosis'] = 'Malignant'
dataset.loc[(dataset['diagnosis'] == 'B'), 'diagnosis'] = 'Benign'