""" Setup prediction testing using predict.py """
from sklearn.metrics import classification_report

from regression.predict import *
from time import time
import datetime

ML_algorithms = ['LR', 'SVC', 'KNC', 'RFC', 'XGB', 'LR_Update', 'census']
now = datetime.datetime.now()
directory = 'regression/testing/'
filename = directory + now.strftime('test-results_%d-%m-%Y--%H-%M-%S.log')
minimum_year = 2012
num_of_training_years = 2


def write_evalutated_results(logfile, results):
    """
    Print the overall results of all the machine learning classifiers
    :param logfile: file - Used as a log file for past execution records.
    :param results: list of list of int - contains assessed data for each classifier.
           ex, [[LR_correct, LR_incorrect][SVC_correct, SVC_incorrect] etc..]
    """
    LR_correct = 0
    LR_incorrect = 0
    SVC_correct = 0
    SVC_incorrect = 0
    KNC_correct = 0
    KNC_incorrect = 0
    RFC_correct = 0
    RFC_incorrect = 0
    XGB_correct = 0
    XGB_incorrect = 0
    LR_p_correct = 0
    LR_p_incorrect = 0
    census_correct = 0
    census_incorrect = 0
    for value in results:
        LR_correct = LR_correct + value[0][0]
        LR_incorrect = LR_incorrect + value[0][1]
        SVC_correct = SVC_correct + value[1][0]
        SVC_incorrect = SVC_incorrect + value[1][1]
        KNC_correct = KNC_correct + value[2][0]
        KNC_incorrect = KNC_incorrect + value[2][1]
        RFC_correct = RFC_correct + value[3][0]
        RFC_incorrect = RFC_incorrect + value[3][1]
        XGB_correct = XGB_correct + value[4][0]
        XGB_incorrect = XGB_incorrect + value[4][1]
        LR_p_correct = LR_p_correct + value[5][0]
        LR_p_incorrect = LR_p_incorrect + value[6][1]
        census_correct = census_correct + value[6][0]
        census_incorrect = census_incorrect + value[6][1]

    logfile.write("LogisticRegression, correct: " + str(LR_correct) + ", incorrect: " + str(LR_incorrect) +
                  ", percentage: " + str(LR_correct / (LR_correct + LR_incorrect)) + '\n')
    logfile.write("State Vector Machine, correct: " + str(SVC_correct) + ", incorrect: " + str(SVC_incorrect) +
                  ", percentage: " + str(SVC_correct / (SVC_correct + SVC_incorrect)) + '\n')
    logfile.write("KNeighbor Classifier, correct: " + str(KNC_correct) + ", incorrect: " + str(KNC_incorrect) +
                  ", percentage: " + str(KNC_correct / (KNC_correct + KNC_incorrect)) + '\n')
    logfile.write("Random Forest Classifier, correct: " + str(RFC_correct) + ", incorrect: " + str(RFC_incorrect) +
                  ", percentage: " + str(RFC_correct / (RFC_correct + RFC_incorrect)) + '\n')
    logfile.write("XGBoost, correct: " + str(XGB_correct) + ", incorrect: " + str(XGB_incorrect) +
                  ", percentage: " + str(XGB_correct / (XGB_correct + XGB_incorrect)) + '\n')
    logfile.write("LogisticRegression_prior, correct: " + str(LR_p_correct) + ", incorrect: " + str(LR_p_incorrect) +
                  ", percentage: " + str(LR_p_correct / (LR_p_correct + LR_p_incorrect)) + '\n')
    logfile.write("Census, correct: " + str(census_correct) + ", incorrect: " + str(census_incorrect) +
                  ", percentage: " + str(census_correct / (census_correct + census_incorrect)) + '\n')


def predict_season_bo1(logfile, team, training_years, previous_year, test_year, clf_name):
    """
    Predict the yearly win/loss for a team and check accuracy.
    Using LR, SVC, KNC, RFC, XGB, and LR_p which updates after every game.
    :param logfile: file - Used as a log file for past execution records.
    :param team: str - which team's season is being simulated. ex, 'NYM'
    :param training_years: str list - list of every year to be used to train the classifiers. ex, ['2017']
    :param previous_year: str - indicates the previous year to the test year. For gamelog purposes. ex, '2017'
    :param test_year: str - indicates the season being simulated. ex, '2018'
    :param clf_name: str - specifies which ML_algorithm is being used. ex, 'LR'
    """
    logfile.write("Executing predict_season_bo1\n")

    # Obtain gamelogs from the season databases
    training_gamelog = build_gamelog(training_years, teams)
    previous_gamelog = build_gamelog([previous_year], [team])
    prediction_gamelog_full = build_gamelog([test_year], [team])
    prediction_gamelog_reduced = build_testing_gamelog(previous_year, test_year, [team])

    # Obtain training and test data based on the game logs
    if clf_name == ML_algorithms[5]:
        data, df = build_data(prediction_gamelog_reduced, previous_gamelog)
    else:
        data, df = build_data(prediction_gamelog_reduced, training_gamelog)

    # Load saved classifiers
    # LR, SVC, KNC, RFC, XGB, LR_p, = load_clfs()

    # Build classifier
    if clf_name == ML_algorithms[0]:
        clf = build_LR(data, filenameLR)
    elif clf_name == ML_algorithms[1]:
        clf = build_SVC(data, filenameSVC)
    elif clf_name == ML_algorithms[2]:
        clf = build_KNC(data, filenameKNC)
    elif clf_name == ML_algorithms[3]:
        clf = build_RFC(data, filenameRFC)
    elif clf_name == ML_algorithms[4]:
        clf = build_XGB(data, filenameXGB)
    else:
        clf = build_LR(data, filenameLR_p)

    # print(classification_report(data[3], clf.predict(data[1])))

    # Gather real results - for comparison
    actual_results = []
    for game in prediction_gamelog_full:
        if game[len(features) - 1] == '1':
            actual_results.append(1)
        else:
            actual_results.append(0)

    print("Actual results: " + ('[%s]' % ', '.join(map(str, actual_results))))

    # Test Prediction
    games_count = len(prediction_gamelog_full)
    predictions = []
    if clf_name == ML_algorithms[4]:
        games = df.drop('win_1', axis=1)
        predictions = clf.predict(games)
    else:
        for i in range(games_count):

            game = df.iloc[i].drop('win_1')

            try:
                prediction_SVC = clf.predict([game])[:1]
                predictions.append(prediction_SVC[0])
            except ValueError as err:
                print("ValueError with game: " + str(prediction_gamelog_full[i]))
                print("ValueError: {0}".format(err))
                if clf_name == ML_algorithms[0]:
                    clf = build_LR(data, filenameLR)
                elif clf_name == ML_algorithms[1]:
                    clf = build_SVC(data, filenameSVC)
                elif clf_name == ML_algorithms[2]:
                    clf = build_KNC(data, filenameKNC)
                elif clf_name == ML_algorithms[3]:
                    clf = build_RFC(data, filenameRFC)
                elif clf_name == ML_algorithms[5]:
                    clf = build_LR(data, filenameLR_p)
                continue
            except TypeError as err:
                print("TypeError with game: " + str(prediction_gamelog_full[i]))
                print("TypeError: {0}".format(err))

            # Update data
            previous_gamelog.append(prediction_gamelog_reduced[i])
            if clf_name == ML_algorithms[5]:
                data, df = build_data(prediction_gamelog_reduced, previous_gamelog)
                clf = build_LR(data, filenameLR_p)

    print("Predictions: " + ('[%s]' % ', '.join(map(str, predictions))))
    logfile.write(('[%s]' % ', '.join(map(str, predictions))) + '\n')
    assess = assess_prediction(actual_results, predictions)
    logfile.write(clf_name + " , correct: " + str(assess[0]) + ", incorrect: " + str(assess[1]) +
                  ", percentage: " + str(assess[0] / (assess[0] + assess[1])) + '\n')
    return assess


def predict_season_bo5(logfile, team, training_years, previous_year, test_year):
    """
    Predict the yearly win/loss for a team and check accuracy.
    Using LR, SVC, KNC, RFC, XGB, and LR_p which updates after every game.
    :param logfile: file - Used as a log file for past execution records.
    :param team: str - which team's season is being simulated. ex, 'NYM'
    :param training_years: str list - list of every year to be used to train the classifiers. ex, ['2017']
    :param previous_year: str - indicates the previous year to the test year. For gamelog purposes. ex, '2017'
    :param test_year: str - indicates the season being simulated. ex, '2018'
    """
    logfile.write("Executing predict_season_bo5\n")

    # Obtain gamelogs from the season databases
    training_gamelog = build_gamelog(training_years, teams)
    previous_gamelog = build_gamelog([previous_year], [team])
    prediction_gamelog_full = build_gamelog([test_year], [team])
    prediction_gamelog_reduced = build_testing_gamelog(previous_year, test_year, [team])

    # Obtain training and test data based on the game logs
    data, df = build_data(prediction_gamelog_reduced, training_gamelog)
    data_p, df_p = build_data(prediction_gamelog_reduced, previous_gamelog)

    # Load saved classifiers
    # LR, SVC, KNC, RFC, XGB, LR_p, = load_clfs()

    # Build classifiers
    clf_LR = build_LR(data, filenameLR)
    clf_SVC = build_SVC(data, filenameSVC)
    clf_KNC = build_KNC(data, filenameKNC)
    clf_RFC = build_RFC(data, filenameRFC)
    clf_XGB = build_XGB(data, filenameXGB)
    clf_LR_p = build_LR(data_p, filenameLR_p)

    # Gather real results - for comparison
    actual_results = []
    for game in prediction_gamelog_full:
        if game[len(features) - 1] == '1':
            actual_results.append(1)
        else:
            actual_results.append(0)

    print("Actual results: " + ('[%s]' % ', '.join(map(str, actual_results))))

    # Test Prediction
    games_count = len(prediction_gamelog_full)

    predictions_LR = []
    predictions_LR_p = []
    predictions_SVC = []
    predictions_KNC = []
    predictions_RFC = []

    games = df.drop('win_1', axis=1)
    predictions_XGB = clf_XGB.predict(games)

    for i in range(games_count):

        game = df.iloc[i].drop('win_1')
        game_p = df_p.iloc[i].drop('win_1')

        try:
            prediction_LR = clf_LR.predict([game])[:1]
            predictions_LR.append(prediction_LR[0])
            prediction_SVC = clf_SVC.predict([game])[:1]
            predictions_SVC.append(prediction_SVC[0])
            prediction_KNC = clf_KNC.predict([game])[:1]
            predictions_KNC.append(prediction_KNC[0])
            prediction_RFC = clf_RFC.predict([game])[:1]
            predictions_RFC.append(prediction_RFC[0])
            prediction_LR_p = clf_LR_p.predict([game_p])[:1]
            predictions_LR_p.append(prediction_LR_p[0])
        except ValueError as err:
            print("ValueError with game: " + str(prediction_gamelog_full[i]))
            print("ValueError: {0}".format(err))
            clf_LR = build_LR(data, filenameLR)
            clf_SVC = build_SVC(data, filenameSVC)
            clf_KNC = build_KNC(data, filenameKNC)
            clf_RFC = build_RFC(data, filenameRFC)
            clf_LR_p = build_LR(data_p, filenameLR_p)
            continue
        except TypeError as err:
            print("TypeError with game: " + str(prediction_gamelog_full[i]))
            print("TypeError: {0}".format(err))

        # Update data
        previous_gamelog.append(prediction_gamelog_reduced[i])
        data_p, df_p = build_data(prediction_gamelog_reduced, previous_gamelog)
        clf_LR_p = build_LR(data_p, filenameLR_p)

    census = []
    for i in range(len(predictions_LR)):
        sums = predictions_LR[i] + predictions_SVC[i] + predictions_KNC[i] + predictions_RFC[i] + predictions_XGB[i]
        if sums > 2:
            census.append(1)
        else:
            census.append(0)

    print(('[%s]' % ', '.join(map(str, predictions_LR))))
    assess_LR = assess_prediction(actual_results, predictions_LR)
    logfile.write(ML_algorithms[0] + ", correct: " + str(assess_LR[0]) + ", incorrect: " + str(assess_LR[1]) +
                  ", percentage: " + str(assess_LR[0] / (assess_LR[0] + assess_LR[1])) + '\n')
    logfile.write(('[%s]' % ', '.join(map(str, predictions_LR))) + "\n")
    print(('[%s]' % ', '.join(map(str, predictions_SVC))))
    assess_SVC = assess_prediction(actual_results, predictions_SVC)
    logfile.write(ML_algorithms[1] + ", correct: " + str(assess_SVC[0]) + ", incorrect: " + str(assess_SVC[1]) +
                  ", percentage: " + str(assess_SVC[0] / (assess_SVC[0] + assess_SVC[1])) + '\n')
    logfile.write(('[%s]' % ', '.join(map(str, predictions_SVC))) + "\n")
    print(('[%s]' % ', '.join(map(str, predictions_KNC))))
    assess_KNC = assess_prediction(actual_results, predictions_KNC)
    logfile.write(ML_algorithms[2] + ", correct: " + str(assess_KNC[0]) + ", incorrect: " + str(assess_KNC[1]) +
                  ", percentage: " + str(assess_KNC[0] / (assess_KNC[0] + assess_KNC[1])) + '\n')
    logfile.write(('[%s]' % ', '.join(map(str, predictions_KNC))) + "\n")
    print(('[%s]' % ', '.join(map(str, predictions_RFC))))
    assess_RFC = assess_prediction(actual_results, predictions_RFC)
    logfile.write(ML_algorithms[3] + ", correct: " + str(assess_RFC[0]) + ", incorrect: " + str(assess_RFC[1]) +
                  ", percentage: " + str(assess_RFC[0] / (assess_RFC[0] + assess_RFC[1])) + '\n')
    logfile.write(('[%s]' % ', '.join(map(str, predictions_RFC))) + "\n")
    print(('[%s]' % ', '.join(map(str, predictions_XGB))))
    assess_XGB = assess_prediction(actual_results, predictions_XGB)
    logfile.write(ML_algorithms[4] + ", correct: " + str(assess_XGB[0]) + ", incorrect: " + str(assess_XGB[1]) +
                  ", percentage: " + str(assess_XGB[0] / (assess_XGB[0] + assess_XGB[1])) + '\n')
    logfile.write(('[%s]' % ', '.join(map(str, predictions_XGB))) + "\n")
    print(('[%s]' % ', '.join(map(str, predictions_LR_p))))
    assess_LR_p = assess_prediction(actual_results, predictions_LR_p)
    logfile.write(ML_algorithms[5] + ", correct: " + str(assess_LR_p[0]) + ", incorrect: " + str(assess_LR_p[1]) +
                  ", percentage: " + str(assess_LR_p[0] / (assess_LR_p[0] + assess_LR_p[1])) + '\n')
    logfile.write(('[%s]' % ', '.join(map(str, predictions_LR_p))) + "\n")
    print(census)
    assess_census = assess_prediction(actual_results, census)
    logfile.write("Census, correct: " + str(assess_census[0]) + ", incorrect: " + str(assess_census[1]) +
                  ", percentage: " + str(assess_census[0] / (assess_census[0] + assess_census[1])) + '\n')
    logfile.write(('[%s]' % ', '.join(map(str, census))) + "\n")
    results = [assess_LR, assess_SVC, assess_KNC, assess_RFC, assess_XGB, assess_LR_p, assess_census]
    return results


def execute_season_bo1(year, forecast_teams, clf_name):
    """
    Automate testing on an entire season, analyze predicted results against the real data.
    bo1 meaning best of 1, or in other words, only using one desiginated classifier.
    :param year: int - which year to execute for. ex, 2019
    :param forecast_teams: str list - which team's seasons to simulate. ex, ['NYM']
    :param clf_name: str - which classifier to use to predict with. Based of ML_algorithms list. ex, 'LR'
    """
    # years = ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']
    training_years = []
    for i in range(num_of_training_years):
        if year - i - 1 >= minimum_year:
            training_years.insert(0, str(year - i - 1))

    prior_year = str(year - 1)
    prediction_year = str(year)
    results = []

    for team in forecast_teams:
        file = open(filename, "a")

        start = time()
        result = predict_season_bo1(file, team, training_years, prior_year, prediction_year, clf_name)
        end = time()

        file.write("Predicting for team: " + team + '\n')
        file.write("Predicting with prediction year: " + prediction_year + '\n')
        file.write("Predicting with training years: " + ('[%s]' % ', '.join(map(str, training_years))) + '\n')
        file.write("Predicting with prior year: " + prior_year + '\n')

        results.append(result)

        # Print the results
        print("Predicted " + team + " season in {:.4f} seconds".format(end - start) + '\n')
        file.write("Predicted " + team + " season in {:.4f} seconds".format(end - start) + '\n\n')
        file.close()

    file = open(filename, "a")
    if forecast_teams == teams:
        file.write(
            "Overall stats for " + clf_name + ' ' + prediction_year + " season\n")
    else:
        file.write("Overall stats for " + clf_name + ' ' + ('%s' % ', '.join(map(str, forecast_teams))) + ' ' +
                   prediction_year + " season\n")

    correct = 0
    incorrect = 0
    for value in results:
        correct = correct + value[0]
        incorrect = incorrect + value[1]

    file.write(clf_name + ", correct: " + str(correct) + ", incorrect: " + str(incorrect) +
               ", percentage: " + str(correct / (correct + incorrect)) + '\n\n')

    file.close()


def execute_season_bo5(year, forecast_teams):
    """
    Automate testing on an entire season analyze predicted results against the real data.
    bo5 meaning best of 5, or in other words, only using ML_algorithm[0-5] to come up with an overall census.
    :param year: int - which year to execute for. ex, 2019
    :param forecast_teams: str list - which team's seasons to simulate. ex, ['NYM'] or teams
    """
    training_years = []
    for i in range(num_of_training_years):
        if year - i - 1 >= minimum_year:
            training_years.insert(0, str(year - i - 1))
    prior_year = str(year - 1)
    prediction_year = str(year)
    results = []

    for team in forecast_teams:
        file = open(filename, "a")

        start = time()
        result = predict_season_bo5(file, team, training_years, prior_year, prediction_year)
        end = time()

        file.write("Predicting for team: " + team + '\n')
        file.write("Predicting with prediction year: " + prediction_year + '\n')
        file.write("Predicting with training years: " + ('[%s]' % ', '.join(map(str, training_years))) + '\n')
        file.write("Predicting with prior year: " + prior_year + '\n')

        results.append(result)

        # Print the results
        print("Predicted " + team + " season in {:.4f} seconds".format(end - start))
        file.write("Predicted " + team + " season in {:.4f} seconds".format(end - start) + '\n\n')
        file.close()

    file = open(filename, "a")
    if forecast_teams == teams:
        file.write("Overall stats for " + prediction_year + " season\n")
    else:
        file.write("Overall stats for " + ('%s' % ', '.join(map(str, forecast_teams))) + ' ' +
                   prediction_year + " season\n")

    write_evalutated_results(file, results)

    file.close()


def main():
    """ Main """
    # execute_season_bo1(2019, ['ARI'], ML_algorithms[5])

    execute_season_bo5(2017, teams)
    execute_season_bo5(2018, teams)
    execute_season_bo5(2019, teams)


# Call main
main()
