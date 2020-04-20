""" Setup prediction testing using predict.py """
from regression.predict import *
from time import time
import datetime

ML_algorithms = ['LR', 'SVC', 'KNC', 'RFC', 'XGB', 'LR_Update', 'census']
now = datetime.datetime.now()
directory = 'regression/testing/'
filename = directory + now.strftime('test-results_%d-%m-%Y--%H-%M-%S.log')
minimum_year = 2012
num_of_training_years = 3


def write_evalutated_results(file, results):
    """ Print the results of the machine learning test """
    for value in results:
        for i in range(len(value)):
            correct = incorrect = 0
            correct = correct + value[i][0]
            incorrect = incorrect + value[i][1]
            print(ML_algorithms[i] + ", correct: " + str(correct) + ", incorrect: " + str(incorrect) +
                  ", percentage: " + str(correct / (correct + incorrect)))
            file.write(ML_algorithms[i] + ", correct: " + str(correct) + ", incorrect: " + str(incorrect) +
                       ", percentage: " + str(correct / (correct + incorrect)) + '\n')


def predict_team_season_bo1(file, team, testing_year, training_years, prior_year, model):
    """ Predict the yearly win/loss for a team and check accuracy """
    file.write("Executing predict_team_season_bo1\n")
    training_predict_gamelog = gamelog_builder(testing_year, [team])
    # simplifed_predict_gamelog = simplifed_gamelog_builder(testing_year, [team])
    testing_predict_gamelog = testing_gamelog_builder(prior_year, testing_year, [team])
    training_gamelog = gamelog_builder(training_years, teams)
    prior_gamelog = gamelog_builder(prior_year, teams)

    # BUILD DATA WITHOUT UPDATING AFTER EVERY GAME
    data, df = build_data(training_predict_gamelog, testing_predict_gamelog, training_gamelog)
    data_p, df_p = build_data(training_predict_gamelog, testing_predict_gamelog, prior_gamelog)

    # Load saved models
    # LR, SVC, KNC, RFC, XGB, LR_p, = load_model()

    # Build new models
    if model == ML_algorithms[0]:
        LR = build_LR(data, filenameLR)
    elif model == ML_algorithms[1]:
        SVC = build_SVC(data, filenameSVC)
    elif model == ML_algorithms[2]:
        KNC = build_KNC(data, filenameKNC)
    elif model == ML_algorithms[3]:
        RFC = build_RFC(data, filenameRFC)
    elif model == ML_algorithms[4]:
        XGB = build_XGB(data, filenameXGB)
    elif model == ML_algorithms[5]:
        LR_p = build_LR(data_p, filenameLR_p)

    # Gather real results - for comparison
    actual_results = []
    for game in training_predict_gamelog:
        if game[len(features) - 1] == 'home':
            actual_results.append(1)
        else:
            actual_results.append(0)

    print("Actual results: " + ('[%s]' % ', '.join(map(str, actual_results))))

    # Test Prediction
    games_count = len(training_predict_gamelog)
    predictions = []
    if model == ML_algorithms[4]:
        games = df.drop('winner_home', axis=1)
        predictions = XGB.predict(games)
    else:
        for i in range(games_count):

            game = df.iloc[i].drop('winner_home')
            game_p = df_p.iloc[i].drop('winner_home')

            try:
                if model == ML_algorithms[0]:
                    prediction_LR = LR.predict([game])[:1]
                    predictions.append(prediction_LR)
                elif model == ML_algorithms[1]:
                    prediction_SVC = SVC.predict([game])[:1]
                    predictions.append(prediction_SVC)
                elif model == ML_algorithms[2]:
                    prediction_KNC = KNC.predict([game])[:1]
                    predictions.append(prediction_KNC)
                elif model == ML_algorithms[3]:
                    prediction_RFC = RFC.predict([game])[:1]
                    predictions.append(prediction_RFC)
                elif model == ML_algorithms[5]:
                    prediction_LR_p = LR_p.predict([game_p])[:1]
                    predictions.append(prediction_LR_p)
            except ValueError as err:
                print("ValueError with game: " + str(training_predict_gamelog[i]))
                print("ValueError: {0}".format(err))
                if model == ML_algorithms[0]:
                    LR = build_LR(data, filenameLR)
                elif model == ML_algorithms[1]:
                    SVC = build_SVC(data, filenameSVC)
                elif model == ML_algorithms[2]:
                    KNC = build_KNC(data, filenameKNC)
                elif model == ML_algorithms[3]:
                    RFC = build_RFC(data, filenameRFC)
                elif model == ML_algorithms[5]:
                    LR_p = build_LR(data_p, filenameLR_p)
                continue
            except TypeError as err:
                print("TypeError with game: " + str(training_predict_gamelog[i]))
                print("TypeError: {0}".format(err))

            # UPDATE DATA AFTER EVERY GAME
            prior_gamelog.append(training_predict_gamelog[i])
            if model == ML_algorithms[5]:
                data_p, df_p = build_data(training_predict_gamelog, testing_predict_gamelog, prior_gamelog)
                LR_p = build_LR(data_p, filenameLR_p)

    print(predictions)
    assess = assess_prediction(actual_results, predictions)
    file.write(model + " , correct: " + str(assess[0]) + ", incorrect: " + str(assess[1]) +
               ", percentage: " + str(assess[0] / (assess[0] + assess[1])) + '\n')
    return assess


def predict_team_season_bo5(file, team, testing_year, training_years, prior_year):
    """ Predict the yearly win/loss for a team and check accuracy """
    file.write("Executing predict_team_season_bo5\n")
    predict_gamelog = gamelog_builder(testing_year, [team])
    # simplifed_predict_gamelog = simplifed_gamelog_builder(testing_year, [team])
    testing_predict_gamelog = testing_gamelog_builder(prior_year, testing_year, [team])
    training_gamelog = gamelog_builder(training_years, teams)
    prior_gamelog = gamelog_builder(prior_year, teams)

    # BUILD DATA WITHOUT UPDATING AFTER EVERY GAME
    data, df = build_data(predict_gamelog, testing_predict_gamelog, training_gamelog)
    data_p, df_p = build_data(predict_gamelog, testing_predict_gamelog, prior_gamelog)

    clf_LR = build_LR(data, filenameLR)
    clf_SVC = build_SVC(data, filenameSVC)
    clf_KNC = build_KNC(data, filenameKNC)
    clf_RFC = build_RFC(data, filenameRFC)
    clf_XGB = build_XGB(data, filenameXGB)
    clf_LR_p = build_LR(data_p, filenameLR_p)

    # Gather real results - for comparison
    actual_results = []
    for game in predict_gamelog:
        if game[len(features) - 1] == 'home':
            actual_results.append(1)
        else:
            actual_results.append(0)

    # Test Prediction
    games_count = len(predict_gamelog)
    predictions_LR = []
    predictions_LR_p = []
    predictions_SVC = []
    predictions_KNC = []
    predictions_RFC = []

    games = df.drop('winner_home', axis=1)
    predictions_XGB = clf_XGB.predict(games)

    for i in range(games_count):

        game = df.iloc[i].drop('winner_home')
        game_p = df_p.iloc[i].drop('winner_home')

        try:
            prediction_LR = clf_LR.predict([game])[:1]
            predictions_LR.append(prediction_LR)
            prediction_SVC = clf_SVC.predict([game])[:1]
            predictions_SVC.append(prediction_SVC)
            prediction_KNC = clf_KNC.predict([game])[:1]
            predictions_KNC.append(prediction_KNC)
            prediction_RFC = clf_RFC.predict([game])[:1]
            predictions_RFC.append(prediction_RFC)
            prediction_LR_p = clf_LR_p.predict([game_p])[:1]
            predictions_LR_p.append(prediction_LR_p)
        except ValueError as err:
            print("ValueError with game: " + str(predict_gamelog[i]))
            print("ValueError: {0}".format(err))
            clf_LR = build_LR(data, filenameLR)
            clf_SVC = build_SVC(data, filenameSVC)
            clf_KNC = build_KNC(data, filenameKNC)
            clf_RFC = build_RFC(data, filenameRFC)
            clf_LR_p = build_LR(data_p, filenameLR_p)
            continue
        except TypeError as err:
            print("TypeError with game: " + str(predict_gamelog[i]))
            print("TypeError: {0}".format(err))

        # UPDATE DATA AFTER EVERY GAME
        prior_gamelog.append(predict_gamelog[i])
        data_p, df_p = build_data(predict_gamelog, testing_predict_gamelog, prior_gamelog)
        clf_LR_p = build_LR(data_p, filenameLR_p)

    census = []
    for i in range(len(predictions_LR)):
        sums = predictions_LR[i] + predictions_SVC[i] + predictions_KNC[i] + predictions_RFC[i] + predictions_XGB[i]
        if sums > 2:
            census.append(1)
        else:
            census.append(0)

    print(predictions_LR)
    assess_LR = assess_prediction(actual_results, predictions_LR)
    file.write(ML_algorithms[0] + ", correct: " + str(assess_LR[0]) + ", incorrect: " + str(assess_LR[1]) +
               ", percentage: " + str(assess_LR[0] / (assess_LR[0] + assess_LR[1])) + '\n')
    print(predictions_SVC)
    assess_SVC = assess_prediction(actual_results, predictions_SVC)
    file.write(ML_algorithms[1] + ", correct: " + str(assess_SVC[0]) + ", incorrect: " + str(assess_SVC[1]) +
               ", percentage: " + str(assess_SVC[0] / (assess_SVC[0] + assess_SVC[1])) + '\n')
    print(predictions_KNC)
    assess_KNC = assess_prediction(actual_results, predictions_KNC)
    file.write(ML_algorithms[2] + ", correct: " + str(assess_KNC[0]) + ", incorrect: " + str(assess_KNC[1]) +
               ", percentage: " + str(assess_KNC[0] / (assess_KNC[0] + assess_KNC[1])) + '\n')
    print(predictions_RFC)
    assess_RFC = assess_prediction(actual_results, predictions_RFC)
    file.write(ML_algorithms[3] + ", correct: " + str(assess_RFC[0]) + ", incorrect: " + str(assess_RFC[1]) +
               ", percentage: " + str(assess_RFC[0] / (assess_RFC[0] + assess_RFC[1])) + '\n')
    print(predictions_XGB)
    assess_XGB = assess_prediction(actual_results, predictions_XGB)
    file.write(ML_algorithms[4] + ", correct: " + str(assess_XGB[0]) + ", incorrect: " + str(assess_XGB[1]) +
               ", percentage: " + str(assess_XGB[0] / (assess_XGB[0] + assess_XGB[1])) + '\n')
    print(predictions_LR_p)
    assess_LR_p = assess_prediction(actual_results, predictions_LR_p)
    file.write(ML_algorithms[5] + ", correct: " + str(assess_LR_p[0]) + ", incorrect: " + str(assess_LR_p[1]) +
               ", percentage: " + str(assess_LR_p[0] / (assess_LR_p[0] + assess_LR_p[1])) + '\n')
    print(census)
    assess_census = assess_prediction(actual_results, census)
    file.write("Census, correct: " + str(assess_census[0]) + ", incorrect: " + str(assess_census[1]) +
               ", percentage: " + str(assess_census[0] / (assess_census[0] + assess_census[1])) + '\n')
    results = [assess_LR, assess_SVC, assess_KNC, assess_RFC, assess_XGB, assess_LR_p, assess_census]
    return results


def execute_season_bo1(year, execute_teams, model_name):
    """ Automate testing an entire past season to compare predicted results against the actual results """
    # years = ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']
    training_years = []
    for i in range(num_of_training_years):
        if year - i - 1 >= minimum_year:
            training_years.insert(0, str(year - i - 1))
    prior_year = [str(year - 1)]
    prediction_year = [str(year)]
    results = []

    for team in execute_teams:
        file = open(filename, "a")

        start = time()
        result = predict_team_season_bo1(file, team, prediction_year, training_years, prior_year, model_name)
        end = time()

        file.write("Predicting for team: " + team + '\n')
        file.write("Predicting with prediction year: " + ('[%s]' % ', '.join(map(str, prediction_year))) + '\n')
        file.write("Predicting with training years: " + ('[%s]' % ', '.join(map(str, training_years))) + '\n')
        file.write("Predicting with prior year: " + ('[%s]' % ', '.join(map(str, prior_year))) + '\n')

        results.append(result)

        # Print the results
        print("Predicted " + team + " season in {:.4f} seconds".format(end - start))
        file.write("Predicted " + team + " season in {:.4f} seconds".format(end - start) + '\n\n')
        file.close()

    file = open(filename, "a")
    if execute_teams == teams:
        file.write(
            "Overall stats for " + model_name + ' ' + ('%s' % ', '.join(map(str, prediction_year))) + " season\n")
    else:
        file.write("Overall stats for " + model_name + ' ' + ('%s' % ', '.join(map(str, execute_teams))) + ' ' +
                   ('%s' % ', '.join(map(str, prediction_year))) + " season\n")

    correct = 0
    incorrect = 0
    for value in results:
        correct = correct + value[0]
        incorrect = incorrect + value[1]

    file.write("LogisticRegression, correct: " + str(correct) + ", incorrect: " + str(incorrect) +
               ", percentage: " + str(correct / (correct + incorrect)) + '\n')

    file.close()


def execute_season_bo5(year, execute_teams):
    """ Automate testing an entire past season to compare predicted results against the actual results """
    # years = ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']
    training_years = []
    for i in range(num_of_training_years):
        if year - i - 1 >= minimum_year:
            training_years.insert(0, str(year - i - 1))
    prior_year = [str(year - 1)]
    prediction_year = [str(year)]
    results = []

    for team in execute_teams:
        file = open(filename, "a")

        start = time()
        result = predict_team_season_bo5(file, team, prediction_year, training_years, prior_year)
        end = time()

        file.write("Predicting for team: " + team + '\n')
        file.write("Predicting with prediction year: " + ('[%s]' % ', '.join(map(str, prediction_year))) + '\n')
        file.write("Predicting with training years: " + ('[%s]' % ', '.join(map(str, training_years))) + '\n')
        file.write("Predicting with prior year: " + ('[%s]' % ', '.join(map(str, prior_year))) + '\n')

        results.append(result)

        # Print the results
        print("Predicted " + team + " season in {:.4f} seconds".format(end - start))
        file.write("Predicted " + team + " season in {:.4f} seconds".format(end - start) + '\n\n')
        file.close()

    file = open(filename, "a")
    if execute_teams == teams:
        file.write("Overall stats for " + ('%s' % ', '.join(map(str, prediction_year))) + " season\n")
    else:
        file.write("Overall stats for " + ('%s' % ', '.join(map(str, execute_teams))) + ' ' +
                   ('%s' % ', '.join(map(str, prediction_year))) + " season\n")

    write_evalutated_results(file, results)

    file.close()


def test():
    """ Test """
    # print(classification_report(y_test, knc.predict(x_test)))
    # print(roc_auc_score(y_test, knc.predict(x_test)))


def main():
    """ Main """
    # int 'year', list [str 'names']
    # execute_season_bo5(2015, ['NYM'])
    execute_season_bo1(2018, ['NYM'], ML_algorithms[2])


# Call main
main()
# test()
