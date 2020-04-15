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

    print(correct / 162)
    print(correct, incorrect)
    print(' ')
    return correct, incorrect


def predict_team_season(file, team, testing_year, training_years, prior_year):
    """ Predict the yearly win/loss for a team and check accuracy """
    predict_gamelog = gamelog_builder(testing_year, team)
    simplifed_predict_gamelog = simplifed_gamelog_builder(testing_year, team)
    training_gamelog = gamelog_builder(training_years, teams)
    prior_gamelog = gamelog_builder(prior_year, team)

    # BUILD DATA WITHOUT UPDATING AFTER EVERY GAME
    data, df = build_data(predict_gamelog, simplifed_predict_gamelog, training_gamelog)
    data_p, df_p = build_data(predict_gamelog, simplifed_predict_gamelog, prior_gamelog)
    games = df.drop('winner_home', axis=1)
    # data = x_train, x_test, y_train, y_test

    # Load saved models
    clf_LR, clf_LR_p, clf_SVC, clf_XGB = load_model()

    # Build new models
    clf_LR = build_model_LR(data, filenameLR)
    clf_LR_p = build_model_LR(data_p, filenameLR_p)
    clf_SVC = build_model_SVC(data, filenameSVC)
    # clf_XGB = build_model_XGB(data, filenameXGB)

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
    predictions_XGB = []
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
            # clf_LR = build_model_LR(data, filenameLR)
            # clf_LR_p = build_model_LR(data_p, filenameLR_p)
            # clf_SVC = build_model_SVC(data)
            i = i - 1
        except TypeError:
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
    c, i = assess_prediction(actual_results, predictions_LR)
    file.write("LogisticRegression, correct: " + str(c) + ", incorrect: " + str(i) + ", percentage: " + str(c/162) + '\n')
    print(predictions_LR_p)
    c, i = assess_prediction(actual_results, predictions_LR_p)
    file.write("LogisticRegression_prior, correct: " + str(c) + ", incorrect: " + str(i) + ", percentage: " + str(c/162) + '\n')
    print(predictions_SVC)
    c, i = assess_prediction(actual_results, predictions_SVC)
    file.write("State Vector Machine, correct: " + str(c) + ", incorrect: " + str(i) + ", percentage: " + str(c/162) + '\n')
    # print(predictions_XGB)
    # assess_prediction(actual_results, predictions_XGB)
    # print(predictions_XGB2)
    # assess_prediction(actual_results, predictions_XGB2)
    print(census)
    c, i = assess_prediction(actual_results, census)
    file.write("Census, correct: " + str(c) + ", incorrect: " + str(i) + ", percentage: " + str(c/162) + '\n')


def main():
    """ Main Function """

    predict_team = ['LAD']
    training_years = ['2016', '2017', '2018']
    prior_year = ['2018']
    prediction_year = ['2019']

    for prediction_team in teams:
        file = open(filename, "a")
        file.write("Predicting for team: " + prediction_team + '\n')
        file.write("Predicting with prediction year: " + ('[%s]' % ', '.join(map(str, prediction_year))) + '\n')
        file.write("Predicting with training years: " + ('[%s]' % ', '.join(map(str, training_years))) + '\n')
        file.write("Predicting with prior year: " + ('[%s]' % ', '.join(map(str, prior_year))) + '\n')
        start = time()
        predict_team_season(file, predict_team, prediction_year, training_years, prior_year)
        end = time()

        # Print the results
        print("Predicted " + prediction_team + "season in {:.4f} seconds".format(end - start))
        file.write("Predicted " + prediction_team + "season in {:.4f} seconds".format(end - start) + '\n\n')
        file.close()


# Call main
main()
