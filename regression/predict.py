""" Support functions used to setup machine learning models as well as executing them """

# Import statements
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC as SupportVectorClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from time import time

# Database related imports
from teamdata.seasonstats import *
import sqlite3 as sql

# Variables
teams = ['ARI', 'ATL', 'BAL', 'BOS', 'CHC', 'CHW', 'CIN', 'CLE', 'COL', 'DET', 'HOU', 'KCR', 'LAA', 'LAD', 'MIA',
         'MIL', 'MIN', 'NYM', 'NYY', 'OAK', 'PHI', 'PIT', 'SDP', 'SEA', 'SFG', 'STL', 'TBR', 'TEX', 'TOR', 'WSN']
years = ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']

# Classifier saving
sav_directory = 'regression/classifiers/'
filenameLR = sav_directory + 'LR.sav'
filenameSVC = sav_directory + 'SVC.sav'
filenameKNC = sav_directory + 'KNC.sav'
filenameRFC = sav_directory + 'RFC.sav'
filenameXGB = sav_directory + 'XGB.sav'
filenameLR_p = sav_directory + 'LR_p.sav'

features = ['season', 'team', 'opponent', 'home', 'runs', 'runsallowed', 'innings', 'day', 'pitcher',
            'pitcher_wlp', 'pitcher_era', 'pitcher_whip', 'pitcher_fip', 'opp_pitcher', 'opp_pitcher_wlp',
            'opp_pitcher_era', 'opp_pitcher_whip', 'opp_pitcher_fip', 'team_loc_wlp', 'opp_loc_wlp', 'win']


def train_classifier(clf, x_train, y_train):
    """ Train the classifer. """
    print("Training a " + clf.__class__.__name__ + " using a training set size of " + str(len(x_train)))

    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(x_train, y_train)
    end = time()

    print("Trained classifier in {:.4f} seconds".format(end - start) + '\n')


def build_data(prediction_gamelog, training_gamelog):
    """
    Build the data used to fit the model, x_train, x_test, y_train, y_test.
    :param prediction_gamelog: list of list - Used to input feature values from test data. From build_testing_gamelog().
    :param training_gamelog: list of list - Used to fit the data. From build_gamelog().
    """
    training_size = len(training_gamelog)
    temp = insert_gamelog(prediction_gamelog, training_gamelog)

    # Create dataframe from training_gamelog and predicition gamelog
    df = pd.DataFrame(temp, columns=features)
    df = df.fillna(df.mean())

    # Remove unwanted feature. They lower accuracy. Overfitting?
    del df['innings']
    del df['runs']
    del df['runsallowed']
    del df['pitcher_wlp']
    del df['pitcher_era']
    del df['opp_pitcher_wlp']
    del df['opp_pitcher_era']

    df = pd.get_dummies(df, drop_first=True)

    df_train = df.iloc[:training_size]
    df_test = df.iloc[training_size:]

    # x_train, x_test, y_train, y_test =
    # train_test_split(df_train.drop('win_1', axis=1), df_train['win_1'], random_state=42)

    x_train = df_train.drop('win_1', axis=1)
    y_train = df_train['win_1']
    x_test = df_test.drop('win_1', axis=1)
    y_test = df_test['win_1']
    return [x_train, x_test, y_train, y_test], df_test


def build_LR(data, file):
    """ Build logistic regression model """
    clf = LogisticRegression(solver='lbfgs', random_state=42, max_iter=25000)

    train_classifier(clf, data[0], data[2])

    pickle.dump(clf, open(file, 'wb'))
    return clf


def build_SVC(data, file):
    """ Build support vector classifier """
    clf = SupportVectorClassifier(random_state=42, kernel='rbf')

    train_classifier(clf, data[0], data[2])

    pickle.dump(clf, open(file, 'wb'))
    return clf


def build_KNC(data, file):
    """ Build kneighbor classifier """
    clf = KNeighborsClassifier(n_neighbors=8)

    train_classifier(clf, data[0], data[2])

    pickle.dump(clf, open(file, 'wb'))
    return clf


def build_RFC(data, file):
    """ Build random forest classifier """
    clf = RandomForestClassifier(random_state=42)

    train_classifier(clf, data[0], data[2])

    pickle.dump(clf, open(file, 'wb'))
    return clf


def build_XGB(data, file):
    """ Build XGBoost classifier """
    clf = xgb.XGBClassifier(random_state=42, max_depth=6)

    train_classifier(clf, data[0], data[2])

    pickle.dump(clf, open(file, 'wb'))
    return clf


def load_clfs():
    """ Load clfs from memory """
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


def build_gamelog(gamelog_years, gamelog_teams):
    """ Build testing dataset. Only supplies data that is availible after a ball game is played. """
    gamelog = []
    for season in gamelog_years:
        for team in gamelog_teams:
            # Connect to db
            directory = 'teamdata/'
            dbname = directory + 'teamstats_' + season + '.db'
            statsdb = sql.connect(dbname)

            # Create a cursor to navigate the db
            statscursor = statsdb.cursor()

            # Specified team schedule table
            schedule_table = team + 'Schedule'
            schedule = get_team_schedule(statscursor, schedule_table)

            # Win/loss split table
            wls = get_team_schedule(statscursor, 'WinLossSplit')

            for game in schedule:
                opponent = game[3]
                home = game[4]  # Either '1', meaning home, or '0', meaning away

                team_wlp = opponent_wlp = 0.000
                for value in wls:
                    # team, overall, home, away
                    if value[0] == team:
                        if home == '1':
                            team_wlp = value[2]
                        else:
                            team_wlp = value[3]
                    if value[0] == opponent:
                        if home == '0':
                            opponent_wlp = value[2]
                        else:
                            opponent_wlp = value[3]

                # Features
                game = [season, game[2], game[3], game[4], game[5], game[6], game[7], game[8], game[9], game[10],
                        game[11], game[12], game[13], game[14], game[15], game[16], game[17], game[18], team_wlp,
                        opponent_wlp, game[19]]

                gamelog.append(game)

    # if gamelog_teams > 1:
        # Remove duplicates
        # gamelog.sort()
        # gamelog = sorted(gamelog, key=lambda x: x[1])
        # gamelog = list(k for k, _ in itertools.groupby(gamelog))
    return gamelog


def build_testing_gamelog(previous_season, season, gamelog_teams):
    """ Build testing dataset. Only supplies data that is availible before a ball game is played. """
    gamelog = []
    for team in gamelog_teams:
        # Connect to testing year db
        directory = 'teamdata/'
        dbname = directory + 'teamstats_' + season + '.db'
        statsdb = sql.connect(dbname)

        # Create a cursor to navigate the db
        statscursor = statsdb.cursor()

        # Specified team schedule table
        schedule_table = team + 'Schedule'
        schedule = get_team_schedule(statscursor, schedule_table)

        # Win/loss split table
        wls = get_team_schedule(statscursor, 'WinLossSplit')

        # Connect to previous year db
        previous_dbname = directory + 'teamstats_' + previous_season + '.db'
        previous_statsdb = sql.connect(previous_dbname)

        # Create a cursor to navigate the db
        previous_statscursor = previous_statsdb.cursor()

        for game in schedule:
            pitcher_wlp = opp_pitcher_wlp = 0.500
            pitcher_era = opp_pitcher_era = 4.5
            pitcher_whip = opp_pitcher_whip = 1.300
            pitcher_fip = opp_pitcher_fip = 4.2

            pitcher_name = game[9]
            opp_pitcher_name = game[14]

            # Team's previous season's schedule table
            previous_schedule_table = team + 'Schedule'
            previous_schedule = get_team_schedule(previous_statscursor, previous_schedule_table)
            for game_p in previous_schedule:
                if game_p[9] == pitcher_name:
                    pitcher_wlp = game_p[10]
                    pitcher_era = game_p[11]
                    pitcher_whip = game_p[12]
                    pitcher_fip = game_p[13]

            # Opponent's previous season's schedule table
            previous_schedule_table = game[3] + 'Schedule'
            previous_schedule = get_team_schedule(previous_statscursor, previous_schedule_table)
            for game_p in previous_schedule:
                if game_p[9] == opp_pitcher_name:
                    opp_pitcher_wlp = game_p[10]
                    opp_pitcher_era = game_p[11]
                    opp_pitcher_whip = game_p[12]
                    opp_pitcher_fip = game_p[13]

            opponent = game[3]
            home = game[4]  # Either '1', meaning home, or '0', meaning away

            team_wlp = opponent_wlp = 0.000
            for value in wls:
                # team, overall, home, away
                if value[0] == team:
                    if home == '1':
                        team_wlp = value[2]
                    else:
                        team_wlp = value[3]
                if value[0] == opponent:
                    if home == '0':
                        opponent_wlp = value[2]
                    else:
                        opponent_wlp = value[3]

            game = [season, game[2], game[3], game[4], None, None, None, game[8], game[9],
                    pitcher_wlp, pitcher_era, pitcher_whip, pitcher_fip, game[14], opp_pitcher_wlp, opp_pitcher_era,
                    opp_pitcher_whip, opp_pitcher_fip, team_wlp, opponent_wlp, None]

            gamelog.append(game)

    return gamelog


def assess_prediction(actual_results, predictions):
    """ Assess the accuracy of the predictions """
    correct = incorrect = 0
    for i in range(len(predictions)):
        if actual_results[i] == predictions[i]:
            correct = correct + 1
        else:
            incorrect = incorrect + 1

    print(correct / (correct + incorrect))
    print(correct, incorrect)
    print(' ')
    return [correct, incorrect]
