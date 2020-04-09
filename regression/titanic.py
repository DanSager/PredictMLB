# Import statements
from pydataset import data
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

sample = data('titanic')

# Feature Engineering (one hot encoding)
titanic = pd.get_dummies(sample, drop_first=True)

# Test train split
x_train, x_test, y_train, y_test = train_test_split(titanic.drop('survived_yes', axis=1), titanic['survived_yes'])

# Train the model using the training data
LogReg = LogisticRegression(solver='lbfgs')
LogReg.fit(x_train, y_train)

# Predict if a class-1 child girl survived 1 = survived
print(LogReg.predict(np.array([[0,0,1,1]]))[0])

# Predict if a class-3 adult male survived
print(LogReg.predict(np.array([[0,1,0,0]]))[0])

# Scoring the model
print(LogReg.score(x_test, y_test))

# Understanding the score
prediction = (LogReg.predict(x_test) > .5).astype(int)
print(np.sum(prediction == y_test) / len(y_test))

print("Exit")
