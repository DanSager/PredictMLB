""" Support functions used to setup machine learning models as well as executing them """

# Import statements
import itertools
import pickle
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from time import time

# Database related imports
from teamdata.seasonstats import *
import sqlite3 as sql

teams = ['ARI', 'ATL', 'BAL', 'BOS', 'CHC', 'CHW', 'CIN', 'CLE', 'COL', 'DET', 'HOU', 'KCR', 'LAA', 'LAD', 'MIA',
         'MIL', 'MIN', 'NYM', 'NYY', 'OAK', 'PHI', 'PIT', 'SDP', 'SEA', 'SFG', 'STL', 'TBR', 'TEX', 'TOR', 'WSN']
years = ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']

sav_directory = 'regression/models/'
filenameLR = sav_directory + 'clf_LR.sav'
filenameLR_p = sav_directory + 'clf_LR_p.sav'
filenameSVC = sav_directory + 'clf_SVC.sav'
filenameXGB = sav_directory + 'clf_XGB.sav'


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
    print("F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1, acc))

    f1, acc = predict_labels(clf, x_test, y_test)
    print("F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1, acc))


def build_data(predict_gamelog, simplifed_predict_gamelog, training_gamelog):  # x_train, x_test, y_train, y_test
    """ Build the data used to fit the model, x_train, x_test, y_train, y_test """
    training_size = len(training_gamelog)
    predict_size = len(predict_gamelog)
    temp = insert_gamelog(simplifed_predict_gamelog, training_gamelog)
    temp = insert_gamelog(predict_gamelog, temp)

    # num, date, hometeam, awayteam, runshome, runsaway, innings, day, homepitcher, awaypitcher, winner
    df = pd.DataFrame(temp, columns=['num', 'date', 'hometeam', 'awayteam', 'runshome', 'runsaway',
                                     'innings', 'day', 'homepitcher', 'homepitcher_era', 'homepitcher_whip',
                                     'awaypitcher', 'awaypitcher_era', 'awaypitcher_whip', 'winner'])
    del df['num']
    del df['date']
    # TODO test the effects of removing 'del df['date']
    df = pd.get_dummies(df, drop_first=True)

    # Remove the test game from the training data
    # if 'winner_NA' in df.columns:
    #     df = df.drop('winner_NA', axis=1)  # axis 0 is rows, axis 1 is columns
    df_train = df.iloc[:training_size]
    df_predict = df.iloc[training_size:training_size+predict_size]

    x_train, x_test, y_train, y_test = train_test_split(df_train.drop('winner_home', axis=1), df_train['winner_home'])
    return [x_train, x_test, y_train, y_test], df_predict


def build_model_LR(data, file):
    """ Build different regression models """
    clf_LR = LogisticRegression(solver='lbfgs', random_state=42)

    train_predict(clf_LR, data[0], data[1], data[2], data[3])
    print('')

    pickle.dump(clf_LR, open(file, 'wb'))
    return clf_LR


def build_model_SVC(data, file):
    """ Build different regression models """
    clf_SVC = SVC(random_state=912, kernel='rbf')

    train_predict(clf_SVC, data[0], data[1], data[2], data[3])
    print('')

    pickle.dump(clf_SVC, open(file, 'wb'))
    return clf_SVC


def build_model_SVC_proba(data, file):
    """ Build different regression models """
    clf_SVC = SVC(random_state=912, kernel='rbf', probability=True)

    train_predict(clf_SVC, data[0], data[1], data[2], data[3])
    print('')

    pickle.dump(clf_SVC, open(file, 'wb'))
    return clf_SVC


def build_model_XGB(data, file):
    """ Build different regression models """
    clf_XGB = xgb.XGBClassifier(max_depth=6)

    train_predict(clf_XGB, data[0], data[1], data[2], data[3])
    print('')

    pickle.dump(clf_XGB, open(file, 'wb'))
    return clf_XGB


def load_model():
    """ Load clf models """
    try:
        clf_LR = pickle.load(open(filenameLR, 'rb'))
        clf_LR_p = pickle.load(open(filenameLR_p, 'rb'))
        clf_SVC = pickle.load(open(filenameSVC, 'rb'))
        clf_XGB = pickle.load(open(filenameXGB, 'rb'))
    except FileNotFoundError:
        print(FileNotFoundError)
        return [], [], [], []
    return clf_LR, clf_LR_p, clf_SVC, clf_XGB


def insert_gamelog(predict_gamelog, training_gamelog):
    """ Insert front a game into the gamelog """
    temp = []
    for game in training_gamelog:
        temp.append(game)
    # num, date, hometeam, awayteam, runshome, runsaway, innings, day, homepitcher, homepitcher_era,
    # homepitcher_whip, awaypitcher, awaypitcher_era, awaypitcher_whip, winner
    for game in predict_gamelog:
        temp.append(game)

    return temp


def gamelog_builder(gamelog_years, included_teams):
    """ Build gamelog dataset """
    gamelog = []
    included_teams_size = len(included_teams)
    for year in gamelog_years:
        for team in included_teams:
            # Create / Connect to db
            directory = 'teamdata/'

            dbname = directory + 'teamstats_' + year + '.db'
            statsdb = sql.connect(dbname)

            # Create a cursor to navigate the db
            statscursor = statsdb.cursor()

            table = team + 'Schedule'
            schedule = get_team_schedule(statscursor, table)

            # num, date, hometeam, awayteam, runshome, runsaway, innings, day, homepitcher, homepitcher_era,
            # homepitcher_whip, awaypitcher, awaypitcher_era, awaypitcher_whip, winner
            for game in schedule:
                game = [game[0], game[1], game[2], game[3], game[4], game[5], game[6], game[7],
                        game[8].replace(u'\xa0', u' '), game[9], game[10], game[11].replace(u'\xa0', u' '), game[12],
                        game[13], game[14]]
                # game = [game[0], game[1], game[2], game[3], game[4], game[5], game[6], game[7],
                #         game[8].replace(u'\xa0', u' '), game[9].replace(u'\xa0', u' '), game[10]]

                gamelog.append(game)

    if included_teams_size > 1:
        # Remove duplicates
        # gamelog.sort()
        gamelog = sorted(gamelog, key=lambda x: x[1])
        gamelog = list(k for k, _ in itertools.groupby(gamelog))
    return gamelog


def simplifed_gamelog_builder(gamelog_years, included_teams):
    """ Build gamelog dataset """
    gamelog = []
    included_teams_size = len(included_teams)
    for year in gamelog_years:
        for team in included_teams:
            # Create / Connect to db
            db_directory = 'teamdata/'

            dbname = db_directory + 'teamstats_' + year + '.db'
            statsdb = sql.connect(dbname)

            # Create a cursor to navigate the db
            statscursor = statsdb.cursor()

            table = team + 'Schedule'
            schedule = get_team_schedule(statscursor, table)

            # num, date, hometeam, awayteam, runshome, runsaway, innings, day, homepitcher, homepitcher_era,
            # homepitcher_whip, awaypitcher, awaypitcher_era, awaypitcher_whip, winner
            for game in schedule:
                game = [None, None, game[2], game[3], None, None, None, game[7], game[8].replace(u'\xa0', u' '),
                        game[9], game[10], game[11].replace(u'\xa0', u' '), game[12], game[13], None]
                # game = [None, None, game[2], game[3], None, None, None, game[7], None, None, None]

                gamelog.append(game)

    if included_teams_size > 1:
        # Remove duplicates
        gamelog = sorted(gamelog, key=lambda x: x[1])
        gamelog = list(k for k, _ in itertools.groupby(gamelog))
    return gamelog


def assess_prediction(actual_results, predictions):
    """ Access if our prediction is correct or not """
    correct = incorrect = 0
    for i in range(len(predictions)):
        if actual_results[i]:
            if predictions[i]:
                correct = correct + 1
            else:
                incorrect = incorrect + 1
        else:
            if predictions[i]:
                incorrect = incorrect + 1
            else:
                correct = correct + 1

    print(correct / (correct + incorrect))
    print(correct, incorrect)
    print(' ')
    return [correct, incorrect]
