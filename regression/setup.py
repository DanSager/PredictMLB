""" Setup prediction testing using predict.py """
from regression.predict import *
from time import time
import datetime

now = datetime.datetime.now()
directory = 'regression/testing/'
filename = directory + now.strftime('test-results_%d-%m-%Y--%H-%M-%S.log')


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


def predict_team_season_binary(file, team, testing_year, training_years, prior_year):
    """ Predict the yearly win/loss for a team and check accuracy """
    file.write("Executing predict_team_season_binary\n")
    predict_gamelog = gamelog_builder(testing_year, [team])
    simplifed_predict_gamelog = simplifed_gamelog_builder(testing_year, [team])
    training_gamelog = gamelog_builder(training_years, teams)
    prior_gamelog = gamelog_builder(prior_year, [team])

    # BUILD DATA WITHOUT UPDATING AFTER EVERY GAME
    data, df = build_data(predict_gamelog, simplifed_predict_gamelog, training_gamelog)
    data_p, df_p = build_data(predict_gamelog, simplifed_predict_gamelog, prior_gamelog)
    # games = df.drop('winner_home', axis=1)
    # data = x_train, x_test, y_train, y_test

    # Load saved models
    clf_LR, clf_LR_p, clf_SVC, clf_XGB = load_model()

    # Test loaded models
    sample_game = df.iloc[0].drop('winner_home')
    sample_game_p = df_p.iloc[0].drop('winner_home')

    try:
        p = clf_LR.predict([sample_game])[:1]
        p = clf_LR_p.predict([sample_game_p])[:1]
        p = clf_SVC.predict([sample_game])[:1]
    except ValueError as err:
        print("ValueError loaded model: " + str(predict_gamelog[0]))
        print("ValueError: {0}\n".format(err))
        # Build new models
        clf_LR = build_model_LR(data, filenameLR)
        clf_LR_p = build_model_LR(data_p, filenameLR_p)
        clf_SVC = build_model_SVC(data, filenameSVC)
        # clf_XGB = build_model_XGB(data, filenameXGB)
        print("Rebuilt models\n")

    # Gather real results - for comparison
    actual_results = []
    for game in predict_gamelog:
        if game[10] == 'home':
            actual_results.append(1)
        else:
            actual_results.append(0)

    # Test Prediction
    games_count = len(predict_gamelog)
    predictions_LR = []
    predictions_LR_p = []
    predictions_SVC = []
    # predictions_XGB = []
    for i in range(games_count):

        game = df.iloc[i].drop('winner_home')
        game_p = df_p.iloc[i].drop('winner_home')
        # game_xgb = games.iloc[i:i+1]

        try:
            prediction_LR = clf_LR.predict([game])[:1]
            predictions_LR.append(prediction_LR)
            prediction_LR_p = clf_LR_p.predict([game_p])[:1]
            predictions_LR_p.append(prediction_LR_p)
            prediction_SVC = clf_SVC.predict([game])[:1]
            predictions_SVC.append(prediction_SVC)
            # prediction_XGB = clf_XGB.predict_proba(game_xgb)
            # predictions_XGB.append(prediction_XGB)

        except ValueError as err:
            print("ValueError with game: " + str(predict_gamelog[i]))
            print("ValueError: {0}".format(err))
            clf_LR = build_model_LR(data, filenameLR)
            clf_LR_p = build_model_LR(data_p, filenameLR_p)
            clf_SVC = build_model_SVC(data, filenameSVC)
            continue
        except TypeError as err:
            print("TypeError with game: " + str(predict_gamelog[i]))
            print("TypeError: {0}".format(err))

        # UPDATE DATA AFTER EVERY GAME
        prior_gamelog.append(predict_gamelog[i])
        data_p, df_p = build_data(predict_gamelog, simplifed_predict_gamelog, prior_gamelog)
        clf_LR_p = build_model_LR(data_p, filenameLR_p)

    # predictions_XGB2 = clf_XGB.predict(games)

    census = []
    for i in range(len(predictions_LR)):
        sums = predictions_LR[i] + predictions_LR_p[i] + predictions_SVC[i]
        if sums < 2:
            census.append(0)
        else:
            census.append(1)

    print(predictions_LR)
    assess_LR = assess_prediction(actual_results, predictions_LR)
    file.write("LogisticRegression, correct: " + str(assess_LR[0]) + ", incorrect: " + str(assess_LR[1]) +
               ", percentage: " + str(assess_LR[0]/(assess_LR[0]+assess_LR[1])) + '\n')
    print(predictions_LR_p)
    assess_LR_p = assess_prediction(actual_results, predictions_LR_p)
    file.write("LogisticRegression_prior, correct: " + str(assess_LR_p[0]) + ", incorrect: " + str(assess_LR_p[1]) +
               ", percentage: " + str(assess_LR_p[0]/(assess_LR_p[0]+assess_LR_p[1])) + '\n')
    print(predictions_SVC)
    assess_SVC = assess_prediction(actual_results, predictions_SVC)
    file.write("State Vector Machine, correct: " + str(assess_SVC[0]) + ", incorrect: " + str(assess_SVC[1]) +
               ", percentage: " + str(assess_SVC[0]/(assess_SVC[0]+assess_SVC[1])) + '\n')
    # print(predictions_XGB)
    # assess_prediction(actual_results, predictions_XGB)
    # print(predictions_XGB2)
    # assess_prediction(actual_results, predictions_XGB2)
    print(census)
    assess_census = assess_prediction(actual_results, census)
    file.write("Census, correct: " + str(assess_census[0]) + ", incorrect: " + str(assess_census[1]) +
               ", percentage: " + str(assess_census[0]/(assess_census[0]+assess_census[1])) + '\n')
    results = [assess_LR, assess_LR_p, assess_SVC, assess_census]
    return results


def predict_team_season_proba(file, team, testing_year, training_years, prior_year):
    """ Predict the yearly win/loss for a team and check accuracy """
    file.write("Executing predict_team_season_proba\n")
    predict_gamelog = gamelog_builder(testing_year, [team])
    simplifed_predict_gamelog = simplifed_gamelog_builder(testing_year, [team])
    training_gamelog = gamelog_builder(training_years, teams)
    prior_gamelog = gamelog_builder(prior_year, [team])

    # BUILD DATA WITHOUT UPDATING AFTER EVERY GAME
    data, df = build_data(predict_gamelog, simplifed_predict_gamelog, training_gamelog)
    data_p, df_p = build_data(predict_gamelog, simplifed_predict_gamelog, prior_gamelog)
    # games = df.drop('winner_home', axis=1)
    # data = x_train, x_test, y_train, y_test

    # Load saved models
    clf_LR, clf_LR_p, clf_SVC, clf_XGB = load_model()

    # Test loaded models
    sample_game = df.iloc[0].drop('winner_home')
    sample_game_p = df_p.iloc[0].drop('winner_home')

    try:
        p = clf_LR.predict_proba([sample_game])[:1]
        p = clf_LR_p.predict_proba([sample_game_p])[:1]
        p = clf_SVC.predict_proba([sample_game])[:1]
    except ValueError as err:
        print("ValueError loaded model: " + str(predict_gamelog[0]))
        print("ValueError: {0}\n".format(err))
        # Build new models
        clf_LR = build_model_LR(data, filenameLR)
        clf_LR_p = build_model_LR(data_p, filenameLR_p)
        clf_SVC = build_model_SVC_proba(data, filenameSVC)
        # clf_XGB = build_model_XGB(data, filenameXGB)
        print("Rebuilt models\n")
    except AttributeError as err:
        print("AttributeError loaded model: " + str(predict_gamelog[0]))
        print("AttributeError: {0}\n".format(err))
        # Build new models
        clf_LR = build_model_LR(data, filenameLR)
        clf_LR_p = build_model_LR(data_p, filenameLR_p)
        clf_SVC = build_model_SVC_proba(data, filenameSVC)
        # clf_XGB = build_model_XGB(data, filenameXGB)
        print("Rebuilt models\n")

    # Gather real results - for comparison
    actual_results = []
    for game in predict_gamelog:
        if game[10] == 'home':
            actual_results.append(1)
        else:
            actual_results.append(0)

    # Test Prediction
    games_count = len(predict_gamelog)
    predictions_LR = []
    predictions_LR_p = []
    predictions_SVC = []
    predictions_LR_proba = []
    predictions_LR_p_proba = []
    predictions_SVC_proba = []
    # predictions_XGB = []
    for i in range(games_count):

        game = df.iloc[i].drop('winner_home')
        game_p = df_p.iloc[i].drop('winner_home')
        # game_xgb = games.iloc[i:i+1]

        try:
            prediction_LR = clf_LR.predict_proba([game])[:1]
            predictions_LR_proba.append(prediction_LR)
            prediction_LR_p = clf_LR_p.predict_proba([game_p])[:1]
            predictions_LR_p_proba.append(prediction_LR_p)
            prediction_SVC = clf_SVC.predict_proba([game])[:1]
            predictions_SVC_proba.append(prediction_SVC)
            # prediction_XGB = clf_XGB.predict_proba(game_xgb)
            # predictions_XGB.append(prediction_XGB)

        except ValueError as err:
            print("ValueError with game: " + str(predict_gamelog[i]))
            print("ValueError: {0}".format(err))
            clf_LR = build_model_LR(data, filenameLR)
            clf_LR_p = build_model_LR(data_p, filenameLR_p)
            clf_SVC = build_model_SVC_proba(data, filenameSVC)
            continue
        except AttributeError as err:
            print("AttributeError with game: " + str(predict_gamelog[i]))
            print("AttributeError: {0}".format(err))
            clf_LR = build_model_LR(data, filenameLR)
            clf_LR_p = build_model_LR(data_p, filenameLR_p)
            clf_SVC = build_model_SVC_proba(data, filenameSVC)
            continue
        except TypeError as err:
            print("TypeError with game: " + str(predict_gamelog[i]))
            print("TypeError: {0}".format(err))

        # UPDATE DATA AFTER EVERY GAME
        prior_gamelog.append(predict_gamelog[i])
        data_p, df_p = build_data(predict_gamelog, simplifed_predict_gamelog, prior_gamelog)
        clf_LR_p = build_model_LR(data_p, filenameLR_p)

    # predictions_XGB2 = clf_XGB.predict(games)

    census = []
    for i in range(len(predictions_LR_proba)):
        sums = predictions_LR_proba[i][0][1] + predictions_LR_p_proba[i][0][1] + predictions_SVC_proba[i][0][1]
        if sums > 1.5:
            census.append(1)
        else:
            census.append(0)

    for value in predictions_LR_proba:
        if value[0][1] > .5:
            predictions_LR.append(1)
        else:
            predictions_LR.append(0)

    for value in predictions_LR_p_proba:
        if value[0][1] > .5:
            predictions_LR_p.append(1)
        else:
            predictions_LR_p.append(0)

    for value in predictions_SVC_proba:
        if value[0][1] > .5:
            predictions_SVC.append(1)
        else:
            predictions_SVC.append(0)

    print(predictions_LR)
    assess_LR = assess_prediction(actual_results, predictions_LR)
    file.write("LogisticRegression, correct: " + str(assess_LR[0]) + ", incorrect: " + str(assess_LR[1]) +
               ", percentage: " + str(assess_LR[0] / (assess_LR[0] + assess_LR[1])) + '\n')
    print(predictions_LR_p)
    assess_LR_p = assess_prediction(actual_results, predictions_LR_p)
    file.write("LogisticRegression_prior, correct: " + str(assess_LR_p[0]) + ", incorrect: " + str(assess_LR_p[1]) +
               ", percentage: " + str(assess_LR_p[0] / (assess_LR_p[0] + assess_LR_p[1])) + '\n')
    print(predictions_SVC)
    assess_SVC = assess_prediction(actual_results, predictions_SVC)
    file.write("State Vector Machine, correct: " + str(assess_SVC[0]) + ", incorrect: " + str(assess_SVC[1]) +
               ", percentage: " + str(assess_SVC[0] / (assess_SVC[0] + assess_SVC[1])) + '\n')
    # print(predictions_XGB)
    # assess_prediction(actual_results, predictions_XGB)
    # print(predictions_XGB2)
    # assess_prediction(actual_results, predictions_XGB2)
    print(census)
    assess_census = assess_prediction(actual_results, census)
    file.write("Census, correct: " + str(assess_census[0]) + ", incorrect: " + str(assess_census[1]) +
               ", percentage: " + str(assess_census[0] / (assess_census[0] + assess_census[1])) + '\n')
    results = [assess_LR, assess_LR_p, assess_SVC, assess_census]
    return results


def execute_season(year, execute_teams):
    """ Automate testing an entire past season to compare predicted results against the actual results """
    # years = ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']
    training_years = [str(year-3), str(year-2), str(year-1)]
    prior_year = [str(year-1)]
    prediction_year = [str(year)]
    results = []

    for team in execute_teams:
        file = open(filename, "a")
        file.write("Predicting for team: " + team + '\n')
        file.write("Predicting with prediction year: " + ('[%s]' % ', '.join(map(str, prediction_year))) + '\n')
        file.write("Predicting with training years: " + ('[%s]' % ', '.join(map(str, training_years))) + '\n')
        file.write("Predicting with prior year: " + ('[%s]' % ', '.join(map(str, prior_year))) + '\n')
        start = time()
        result = predict_team_season_binary(file, team, prediction_year, training_years, prior_year)
        end = time()

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

    # LR Total Results
    LR_correct = 0
    LR_incorrect = 0
    LR_p_correct = 0
    LR_p_incorrect = 0
    SVC_correct = 0
    SVC_incorrect = 0
    census_correct = 0
    census_incorrect = 0
    for value in results:
        LR_correct = LR_correct + value[0][0]
        LR_incorrect = LR_incorrect + value[0][1]
        LR_p_correct = LR_p_correct + value[1][0]
        LR_p_incorrect = LR_p_incorrect + value[1][1]
        SVC_correct = SVC_correct + value[2][0]
        SVC_incorrect = SVC_incorrect + value[2][1]
        census_correct = census_correct + value[3][0]
        census_incorrect = census_incorrect + value[3][1]

    file.write("LogisticRegression, correct: " + str(LR_correct) + ", incorrect: " + str(LR_incorrect) +
               ", percentage: " + str(LR_correct / (LR_correct + LR_incorrect)) + '\n')
    file.write("LogisticRegression_prior, correct: " + str(LR_p_correct) + ", incorrect: " + str(LR_p_incorrect) +
               ", percentage: " + str(LR_p_correct / (LR_p_correct + LR_p_incorrect)) + '\n')
    file.write("State Vector Machine, correct: " + str(SVC_correct) + ", incorrect: " + str(SVC_incorrect) +
               ", percentage: " + str(SVC_correct / (SVC_correct + SVC_incorrect)) + '\n')
    file.write("Census, correct: " + str(census_correct) + ", incorrect: " + str(census_incorrect) +
               ", percentage: " + str(census_correct / (census_correct + census_incorrect)) + '\n')

    file.close()


def main():
    """ Main """
    # int 'year', list [str 'names']
    execute_season(2019, ['NYM'])


# Call main
main()
