# Import statements
from pydataset import data
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# data preprocessing
# import pandas as pd
# produces a prediction model in the form of an ensemble of weak prediction models, typically decision tree
import xgboost as xgb
# the outcome (dependent variable) has only a limited number of possible values.
# Logistic Regression is used when response variable is categorical in nature.
# from sklearn.linear_model import LogisticRegression
# A random forest is a meta estimator that fits a number of decision tree classifiers
# on various sub-samples of the dataset and use averaging to improve the predictive
# accuracy and control over-fitting.
from sklearn.ensemble import RandomForestClassifier
# a discriminative classifier formally defined by a separating hyperplane.
from sklearn.svm import SVC
# displayd data
from IPython.display import display

#for measuring training time
from time import time

# Database related imports
from teamdata.seasonstats import *
import sqlite3 as sql

teams = ['ATL', 'ARI', 'BAL', 'BOS', 'CHC', 'CHW', 'CIN', 'CLE', 'COL', 'DET', 'HOU', 'KCR', 'LAA', 'LAD', 'MIA',
         'MIL', 'MIN', 'NYM', 'NYY', 'OAK', 'PHI', 'PIT', 'SDP', 'SEA', 'SFG', 'STL', 'TBR', 'TEX', 'TOR', 'WSN']
years = ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']

from sklearn.metrics import f1_score


def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''

    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()

    # Print the results
    print("Trained model in {:.4f} seconds".format(end - start))


def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''

    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)

    end = time()
    # Print and return results
    print("Made predictions in {:.4f} seconds.".format(end - start))

    return f1_score(target, y_pred), sum(target == y_pred) / float(len(y_pred))

def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''

    # Indicate the classifier and the training set size
    print("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train)))

    # Train the classifier
    train_classifier(clf, X_train, y_train)

    # Print the results of prediction for both training and testing
    f1, acc = predict_labels(clf, X_train, y_train)
    print(f1, acc)
    print("F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1, acc))

    f1, acc = predict_labels(clf, X_test, y_test)
    print("F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1, acc))


def main():
    teams2 = ['NYM']
    train_year = ['2012', '2013', '2014', '2015', '2016', '2017', '2018']
    test_year = ['2019']

    for team in teams:
        # Create / Connect to db
        directory = '../teamdata/'

        train_location, train_opp, train_outcome = [], [], []
        for year in train_year:

            dbname = directory + 'teamstats_' + year + '.db'
            statsdb = sql.connect(dbname)

            # Create a cursor to navigate the db
            statscursor = statsdb.cursor()

            table = team + 'Schedule'

            schedule = get_team_schedule(statscursor, table)

            for game in schedule:
                train_location.append(game[2])
                train_opp.append(game[3])
                train_outcome.append(game[4])

        test_location, test_opp, test_outcome = [], [], []
        for year in test_year:

            dbname = directory + 'teamstats_' + year + '.db'
            statsdb = sql.connect(dbname)

            # Create a cursor to navigate the db
            statscursor = statsdb.cursor()

            table = team + 'Schedule'

            schedule = get_team_schedule(statscursor, table)

            for game in schedule:
                test_location.append(game[2])
                test_opp.append(game[3])
                test_outcome.append(game[4])

        # Calling DataFrame constructor after zipping
        # both lists, with columns specified
        train_data = pd.DataFrame(list(zip(train_location, train_opp, train_outcome)), columns=['location', 'opp', 'outcome'])
        test_data = pd.DataFrame(list(zip(test_location, test_opp, test_outcome)), columns=['location', 'opp', 'outcome'])

        # Feature Engineering (one hot encoding)
        columns = ['location_home', 'opp_ATL', 'outcome_W']
        train_data = pd.get_dummies(train_data, drop_first=True)
        test_data = pd.get_dummies(test_data, drop_first=True)

        # Test train split
        x_train, x_test, y_train, y_test = train_test_split(train_data.drop('outcome_W', axis=1), train_data['outcome_W'])
        # x_train = train_data.drop('outcome_W', axis=1)
        # y_train = train_data['outcome_W']
        # x_test = test_data.drop('outcome_W', axis=1)
        # y_test = test_data['outcome_W']
        predict = test_data.drop('outcome_W', axis=1)
        predict_w = test_data['outcome_W']
        # x_test, y_test, = train_test_split(test_data.drop('outcome_W', axis=1), test_data['outcome_W'])

        # Train the model using the training data
        LogReg = LogisticRegression(solver='lbfgs', random_state=45)
        LogReg.fit(x_train, y_train)

        # Predict if a class-3 adult male survived
        # prediction = LogReg.predict(np.array([[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))[0]
        # print(LogReg.predict(np.array([[0, 1, 0, 0]]))[0])
        #if prediction:
        #    print(team + " wins!")
        #else:
        #    print(team + " loses :(")

        # Scoring the model
        #print(LogReg.score(x_test, y_test))

        # prediction = (LogReg.predict(x_test) > .5).astype(int)
        #print(np.sum(prediction == y_test) / len(y_test))

        ##############################


        # Initialize the three models (XGBoost is initialized later)
        clf_A = LogisticRegression(random_state=42)
        clf_B = SVC(random_state=912, kernel='rbf')
        # Boosting refers to this general problem of producing a very accurate prediction rule
        # by combining rough and moderately inaccurate rules-of-thumb
        clf_C = xgb.XGBClassifier(seed=82)

        train_predict(clf_A, x_train, y_train, x_test, y_test)
        print('')
        train_predict(clf_B, x_train, y_train, x_test, y_test)
        print('')
        train_predict(clf_C, x_train, y_train, x_test, y_test)
        print('')

        return

# main
main()