""" Support functions used to setup machine learning models as well as executing them """

# Import statements
import itertools
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC as SupportVectorClassification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from time import time

# Database related imports
from teamdata.seasonstats import *
import sqlite3 as sql

# Variables
teams = ['ARI', 'ATL', 'BAL', 'BOS', 'CHC', 'CHW', 'CIN', 'CLE', 'COL', 'DET', 'HOU', 'KCR', 'LAA', 'LAD', 'MIA',
         'MIL', 'MIN', 'NYM', 'NYY', 'OAK', 'PHI', 'PIT', 'SDP', 'SEA', 'SFG', 'STL', 'TBR', 'TEX', 'TOR', 'WSN']
years = ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']

sav_directory = 'regression/models/'
filenameLR = sav_directory + 'LR.sav'
filenameSVC = sav_directory + 'SVC.sav'
filenameKNC = sav_directory + 'KNC.sav'
filenameRFC = sav_directory + 'RFC.sav'
filenameXGB = sav_directory + 'XGB.sav'
filenameLR_p = sav_directory + 'LR_p.sav'

features = ['num', 'date', 'season', 'team', 'opponent', 'home', 'runs', 'runsallowed', 'innings', 'day', 'pitcher',
            'pitcher_wlp', 'pitcher_era', 'pitcher_whip', 'pitcher_fip', 'opp_pitcher', 'opp_pitcher_wlp',
            'opp_pitcher_era', 'opp_pitcher_whip', 'opp_pitcher_fip', 'team_loc_wlp', 'opp_loc_wlp', 'win']


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
    # f1, acc = predict_labels(clf, x_train, y_train)
    # print("F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1, acc))

    # f1, acc = predict_labels(clf, x_test, y_test)
    # print("F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1, acc))


def build_data(predict_gamelog, simplifed_predict_gamelog, training_gamelog):  # x_train, x_test, y_train, y_test
    """ Build the data used to fit the model, x_train, x_test, y_train, y_test """
    training_size = len(training_gamelog)
    predict_size = len(predict_gamelog)
    temp = insert_gamelog(simplifed_predict_gamelog, training_gamelog)
    temp = insert_gamelog(predict_gamelog, temp)

    df = pd.DataFrame(temp, columns=features)
    df = df.fillna(df.mean())
    del df['num']
    del df['date']
    #del df['day']
    #del df['home']
    del df['innings']
    del df['runs']
    del df['runsallowed']
    del df['pitcher_wlp']
    del df['pitcher_era']
    #del df['pitcher_whip']
    #del df['pitcher_fip']
    del df['opp_pitcher_wlp']
    del df['opp_pitcher_era']
    #del df['opp_pitcher_whip']
    #del df['opp_pitcher_fip']
    # TODO test the effects of removing 'del df['date']
    df = pd.get_dummies(df, drop_first=True)

    # Remove the test game from the training data
    # if 'winner_NA' in df.columns:
    #     df = df.drop('winner_NA', axis=1)  # axis 0 is rows, axis 1 is columns
    df_train = df.iloc[:training_size]
    df_predict = df.iloc[training_size:training_size+predict_size]

    x_train, x_test, y_train, y_test = train_test_split(df_train.drop('win_1', axis=1), df_train['win_1'],
                                                        random_state=42)
    return [x_train, x_test, y_train, y_test], df_predict


def build_LR(data, file):
    """ Build different regression models """
    LR = LogisticRegression(solver='lbfgs', random_state=42, max_iter=25000)

    train_predict(LR, data[0], data[1], data[2], data[3])
    print('')

    pickle.dump(LR, open(file, 'wb'))
    return LR


def build_SVC(data, file):
    """ Build different regression models """
    SVC_m = SupportVectorClassification(random_state=42, kernel='rbf')

    train_predict(SVC_m, data[0], data[1], data[2], data[3])
    print('')

    pickle.dump(SVC_m, open(file, 'wb'))
    return SVC_m


def build_KNC(data, file):
    """ Build different regression models """
    KNC = KNeighborsClassifier(n_neighbors=8)

    train_predict(KNC, data[0], data[1], data[2], data[3])
    print('')

    pickle.dump(KNC, open(file, 'wb'))
    return KNC


def build_RFC(data, file):
    """ Build different regression models """
    RFC = RandomForestClassifier(random_state=42)

    train_predict(RFC, data[0], data[1], data[2], data[3])
    print('')

    pickle.dump(RFC, open(file, 'wb'))
    return RFC


def build_XGB(data, file):
    """ Build different regression models """
    XGB = xgb.XGBClassifier(random_state=42, max_depth=6)

    train_predict(XGB, data[0], data[1], data[2], data[3])
    print('')

    pickle.dump(XGB, open(file, 'wb'))
    return XGB


def load_model():
    """ Load clf models """
    try:
        LR = pickle.load(open(filenameLR, 'rb'))
        SVC = pickle.load(open(filenameSVC, 'rb'))
        KNC = pickle.load(open(filenameKNC, 'rb'))
        RFC = pickle.load(open(filenameRFC, 'rb'))
        XGB = pickle.load(open(filenameXGB, 'rb'))
        LR_p = pickle.load(open(filenameLR_p, 'rb'))
    except FileNotFoundError:
        print(FileNotFoundError)
        return [], [], [], [], [], []
    return LR, SVC, KNC, RFC, XGB, LR_p


def insert_gamelog(predict_gamelog, training_gamelog):
    """ Insert front a game into the gamelog """
    temp = []
    for game in training_gamelog:
        temp.append(game)
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

            winlosssplit_tname = 'WinLossSplit'
            winlosssplit_table = get_team_schedule(statscursor, winlosssplit_tname)

            for game in schedule:
                team_location_wlp = opponent_location_wlp = 0.000
                for stat in winlosssplit_table:
                    if stat[0] == team:
                        if game[4] == '1':
                            team_location_wlp = stat[2]
                        else:
                            team_location_wlp = stat[3]
                    if stat[0] == game[3]:
                        if game[4] == '0':
                            opponent_location_wlp = stat[2]
                        else:
                            opponent_location_wlp = stat[3]

                game = [game[0], game[1], year, game[2], game[3], game[4], game[5], game[6], game[7], game[8], game[9], game[10],
                        game[11], game[12], game[13], game[14], game[15], game[16], game[17], game[18], team_location_wlp, opponent_location_wlp, game[19]]

                gamelog.append(game)

    if included_teams_size > 1:
        # Remove duplicates
        # gamelog.sort()
        gamelog = sorted(gamelog, key=lambda x: x[1])
        gamelog = list(k for k, _ in itertools.groupby(gamelog))
    return gamelog


def testing_gamelog_builder(prior_year, year, included_teams):
    """ Build gamelog dataset """
    gamelog = []
    included_teams_size = len(included_teams)
    for team in included_teams:
        # Create / Connect to db
        db_directory = 'teamdata/'

        dbname = db_directory + 'teamstats_' + year[0] + '.db'
        statsdb = sql.connect(dbname)

        # Create a cursor to navigate the db
        statscursor = statsdb.cursor()

        table = team + 'Schedule'
        schedule = get_team_schedule(statscursor, table)

        prior_dbname = db_directory + 'teamstats_' + prior_year[0] + '.db'
        prior_statsdb = sql.connect(prior_dbname)

        prior_statscursor = prior_statsdb.cursor()

        prior_table = team + 'Schedule'
        prior_schedule = get_team_schedule(prior_statscursor, prior_table)

        winlosssplit_tname = 'WinLossSplit'
        winlosssplit_table = get_team_schedule(prior_statscursor, winlosssplit_tname)

        # num, date, hometeam, awayteam, runshome, runsaway, innings, day, homepitcher, homepitcher_era,
        # homepitcher_whip, awaypitcher, awaypitcher_era, awaypitcher_whip, winner
        for game in schedule:
            pitcher_wlp = opp_pitcher_wlp = 0.500
            pitcher_era = opp_pitcher_era = 4.5
            pitcher_whip = opp_pitcher_whip = 1.300
            pitcher_fip = opp_pitcher_fip = 4.2
            pitcher_name = game[9]
            for game_p in prior_schedule:
                if game_p[9] == pitcher_name:
                    pitcher_wlp = game_p[10]
                    pitcher_era = game_p[11]
                    pitcher_whip = game_p[12]
                    pitcher_fip = game_p[13]
            awaypitcher_name = game[14]
            prior_table_opp = game[3] + 'Schedule'
            prior_schedule_opp = get_team_schedule(prior_statscursor, prior_table_opp)
            for game_p in prior_schedule_opp:
                if game_p[9] == awaypitcher_name:
                    opp_pitcher_wlp = game_p[10]
                    opp_pitcher_era = game_p[11]
                    opp_pitcher_whip = game_p[12]
                    opp_pitcher_fip = game_p[13]

            team_location_wlp = opponent_location_wlp = 0.000
            for stat in winlosssplit_table:
                if stat[0] == team:
                    if game[4] == '1':
                        team_location_wlp = stat[2]
                    else:
                        team_location_wlp = stat[3]
                if stat[0] == game[3]:
                    if game[4] == '0':
                        opponent_location_wlp = stat[2]
                    else:
                        opponent_location_wlp = stat[3]

            game = [game[0], game[1], year[0], game[2], game[3], game[4], None, None, None, game[8], game[9],
                    pitcher_wlp, pitcher_era, pitcher_whip, pitcher_fip, game[14], opp_pitcher_wlp, opp_pitcher_era,
                    opp_pitcher_whip, opp_pitcher_fip, team_location_wlp, opponent_location_wlp, None]

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
