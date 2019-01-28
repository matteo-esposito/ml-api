# general purpose packages
import pandas as pd 
import numpy as np 

# ml packages
from sklearn.linear_model import LogisticRegression

# Read-in data
url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
df = pd.read_csv(url)
include = ['Age', 'Sex', 'Embarked', 'Survived']
df_ = df[include]

# Quick preprocessing
categoricals = []
for col, col_type in df_.dtypes.iteritems():
    if col_type == 'O':
        categoricals.append(col)
    else:
        df_[col].fillna(0, inplace=True)

df_ohe = pd.get_dummies(df_, columns=categoricals, dummy_na=True)

# Split data
response = 'Survived'
X = df_ohe[df_ohe.columns.difference([response])]
y = df_ohe[response]

# Logreg classifier
lr = LogisticRegression()
lr.fit(X,y)

# Save model
from sklearn.externals import joblib
joblib.dump(lr, 'model.pkl')

# Load model
lr = joblib.load('model.pkl')

# Save training data
model_cols = list(X.columns)
joblib.dump(model_cols, 'model_columns.pkl')
print('Model columns dumped.')


