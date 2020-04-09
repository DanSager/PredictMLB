# Import statements
import itertools
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from IPython.display import display
from time import time

# Database related imports
from teamdata.seasonstats import *
import sqlite3 as sql

teams = ['ARI', 'ATL', 'BAL', 'BOS', 'CHC', 'CHW', 'CIN', 'CLE', 'COL', 'DET', 'HOU', 'KCR', 'LAA', 'LAD', 'MIA',
         'MIL', 'MIN', 'NYM', 'NYY', 'OAK', 'PHI', 'PIT', 'SDP', 'SEA', 'SFG', 'STL', 'TBR', 'TEX', 'TOR', 'WSN']
years = ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']

filenameA = 'finalized_clf_A.sav'
filenameB = 'finalized_clf_B.sav'
filenameC = 'finalized_clf_C.sav'


def train_classifier(clf, x_train, y_train):
    """ Fits a classifier to the training data. """

    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(x_train, y_train)
    end = time()

    # Print the results
    print("Trained model in {:.4f} seconds".format(end - start))


def predict_labels(clf, features, target):
    """ Makes predictions using a fit classifier based on F1 score. """

    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    end = time()
    # Print and return results
    print("Made predictions in {:.4f} seconds.".format(end - start))

    return f1_score(target, y_pred), sum(target == y_pred) / float(len(y_pred))


def train_predict(clf, x_train, x_test, y_train, y_test):
    """ Train and predict using a classifer based on F1 score. """

    # Indicate the classifier and the training set size
    print("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(x_train)))

    # Train the classifier
    train_classifier(clf, x_train, y_train)

    # Print the results of prediction for both training and testing
    f1, acc = predict_labels(clf, x_train, y_train)
    print(f1, acc)
    print("F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1, acc))

    f1, acc = predict_labels(clf, x_test, y_test)
    print("F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1, acc))


def build_data(gamelog):  # x_train, x_test, y_train, y_test
    """ Build the data used to fit the model, x_train, x_test, y_train, y_test """
    # Date, Home Team, Away Team, Winner, Winning Pitcher, Losing Pitcher
    df = pd.DataFrame(gamelog, columns=['date', 'home', 'away', 'winpitcher', 'losepitcher', 'winner'])
    del df['date']
    df = pd.get_dummies(df, drop_first=True)

    # Remove the test game from the training data
    if 'winner_NA' in df.columns:
        df = df.drop('winner_NA', axis=1)  # axis 0 is rows, axis 1 is columns
    df_train = df.iloc[1:]
    df_test = df.iloc[:1]

    x_train, x_test, y_train, y_test = train_test_split(df_train.drop('winner_H', axis=1), df_train['winner_H'])
    return [x_train, x_test, y_train, y_test], df_test


def build_model(data):
    """ Build different regression models """

    clf_A = LogisticRegression(solver='lbfgs', random_state=42)
    clf_B = SVC(random_state=912, kernel='rbf')
    clf_C = xgb.XGBClassifier(seed=82)

    # train_predict(clf_A, x_train, y_train, x_test, y_test)
    train_predict(clf_A, data[0], data[1], data[2], data[3])
    print('')
    train_predict(clf_B, data[0], data[1], data[2], data[3])
    print('')
    train_predict(clf_C, data[0], data[1], data[2], data[3])
    print('')

    pickle.dump(clf_A, open(filenameA, 'wb'))
    pickle.dump(clf_B, open(filenameB, 'wb'))
    pickle.dump(clf_C, open(filenameC, 'wb'))
    return clf_A, clf_B, clf_C


def insert_gamelog(game, gamelog):
    """ Insert front a game into the gamelog """
    gamelog.insert(0, [game[0], game[1], game[2], game[3], game[4], 'NA'])
    return gamelog


def load_model():
    """ Load clf models """
    clf_A = pickle.load(open(filenameA, 'rb'))
    clf_B = pickle.load(open(filenameB, 'rb'))
    clf_C = pickle.load(open(filenameC, 'rb'))
    return clf_A, clf_B, clf_C


def yearly_gamelog_builder(year, included_teams):
    """ Build gamelog dataset """
    gamelog = []
    for team in included_teams:
        # Create / Connect to db
        directory = '../teamdata/'

        dbname = directory + 'teamstats_' + year + '.db'
        statsdb = sql.connect(dbname)

        # Create a cursor to navigate the db
        statscursor = statsdb.cursor()

        table = team + 'Schedule'

        schedule = get_team_schedule(statscursor, table)

        for game in schedule:  # Date, Home Team, Away Team, Winning Pitcher, Losing Pitcher, Winner
            if game[2] == 'home':
                if game[4] == 'W':
                    game = [game[1], team, game[3], game[5].replace(u'\xa0', u' '), game[7].replace(u'\xa0', u' '), 'H']
                    # game = [game[1], team, game[3], 'H']
                else:
                    game = [game[1], team, game[3], game[5].replace(u'\xa0', u' '), game[7].replace(u'\xa0', u' '), 'A']
                    # game = [game[1], team, game[3], 'A']
            else:
                if game[4] == 'W':
                    game = [game[1], game[3], team, game[5].replace(u'\xa0', u' '), game[7].replace(u'\xa0', u' '), 'A']
                    # game = [game[1], game[3], team, 'A']
                else:
                    game = [game[1], game[3], team, game[5].replace(u'\xa0', u' '), game[7].replace(u'\xa0', u' '), 'H']
                    # game = [game[1], game[3], team, 'H']

            gamelog.append(game)

    # Remove duplicates
    gamelog.sort()
    gamelog = list(k for k, _ in itertools.groupby(gamelog))
    return gamelog


def execute(rebuild, predict_game, training_years):
    """ Execute model and predict winner """
    gamelog = []
    for year in training_years:
        gamelog = gamelog + yearly_gamelog_builder(year, teams)

    if predict_game != '':
        gamelog = insert_gamelog(predict_game, gamelog)

    home = gamelog[0][1]
    away = gamelog[0][2]

    data, df = build_data(gamelog)  # x_train, x_test, y_train, y_test

    if rebuild:
        clf_A, clf_B, clf_C = build_model(data)
    else:
        clf_A, clf_B, clf_C = load_model()

    # Test Prediction
    game = df.iloc[0].drop('winner_H')

    try:
        pred = clf_B.predict([game])[:1]
    except ValueError:  # There is missing feature, rebuild the model
        print("ValueError. Rebuilding model")
        clf_A, clf_B, clf_C = build_model(data)
        pred = clf_B.predict([game])[:1]

    if pred:
        print("Prediction that " + home + " will win against " + away)
    else:
        print("Prediction that " + away + " will win against " + home)


def main():
    """ Main Function """

    training_years = ['2012', '2013', '2014', '2015', '2016']

    # game = 'date', 'home', 'away', 'winpitcher', 'losepitcher', 'winner'
    # predict = ['NA', 'NYM', 'TBR', 'Johan Santana', 'Mike Minor', 'NA']
    # predict = yearly_gamelog_builder('2013', teams)[0]
    predict = yearly_gamelog_builder('2012', ['NYM'])[0]
    print(predict)
    execute(True, predict, training_years)



main()
