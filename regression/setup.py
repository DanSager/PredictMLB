""" Setup prediction testing using predict.py """
from regression.predict import *
from time import time
import datetime

now = datetime.datetime.now()
directory = 'regression/testing/'
filename = directory + now.strftime('test-results_%d-%m-%Y--%H-%M-%S.log')
minimum_year = 2012
num_of_training_years = 3


def write_evalutated_results(file, results):
    """ Print the results of the machine learning test """
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


def predict_team_season_bo1(file, team, testing_year, training_years, prior_year, model_name):
    """ Predict the yearly win/loss for a team and check accuracy """
    file.write("Executing predict_team_season_bo1\n")
    predict_gamelog = gamelog_builder(testing_year, [team])
    simplifed_predict_gamelog = simplifed_gamelog_builder(testing_year, [team])
    # TODO uncomment
    # training_gamelog = gamelog_builder(training_years, teams)
    training_gamelog = gamelog_builder(training_years, [team])
    prior_gamelog = gamelog_builder(prior_year, [team])

    # BUILD DATA WITHOUT UPDATING AFTER EVERY GAME
    data, df = build_data(predict_gamelog, simplifed_predict_gamelog, training_gamelog)
    # data, df = build_data(predict_gamelog, simplifed_predict_gamelog, prior_gamelog)
    data_p, df_p = build_data(predict_gamelog, simplifed_predict_gamelog, prior_gamelog)
    # games = df.drop('winner_home', axis=1)
    # data = x_train, x_test, y_train, y_test

    # Load saved models
    # clf_LR, clf_LR_p, clf_SVC = load_model()

    if model_name == 'clf_LR':
        clf_LR = build_model_LR(data, filenameLR)
    elif model_name == 'clf_LR_p':
        clf_LR_p = build_model_LR(data_p, filenameLR_p)
    elif model_name == 'clf_SVC':
        clf_SVC = build_model_SVC(data, filenameSVC)

    # Gather real results - for comparison
    actual_results = []
    for game in predict_gamelog:
        if game[len(data_columns)-1] == 'home':
            actual_results.append(1)
        else:
            actual_results.append(0)

    # Test Prediction
    games_count = len(predict_gamelog)
    predictions = []
    for i in range(games_count):

        game = df.iloc[i].drop('winner_home')
        game_p = df_p.iloc[i].drop('winner_home')
        # game_xgb = games.iloc[i:i+1]

        try:
            if model_name == 'clf_LR':
                prediction_LR = clf_LR.predict([game])[:1]
                predictions.append(prediction_LR)
            elif model_name == 'clf_LR_p':
                prediction_LR_p = clf_LR_p.predict([game_p])[:1]
                predictions.append(prediction_LR_p)
            elif model_name == 'clf_SVC':
                prediction_SVC = clf_SVC.predict([game])[:1]
                predictions.append(prediction_SVC)
        except ValueError as err:
            print("ValueError with game: " + str(predict_gamelog[i]))
            print("ValueError: {0}".format(err))
            if model_name == 'clf_LR':
                clf_LR = build_model_LR(data, filenameLR)
            elif model_name == 'clf_LR_p':
                clf_LR_p = build_model_LR(data_p, filenameLR_p)
            elif model_name == 'clf_SVC':
                clf_SVC = build_model_SVC(data, filenameSVC)
            continue
        except TypeError as err:
            print("TypeError with game: " + str(predict_gamelog[i]))
            print("TypeError: {0}".format(err))

        # UPDATE DATA AFTER EVERY GAME
        prior_gamelog.append(predict_gamelog[i])
        # if model_name == 'clf_LR':
        # data, df = build_data(predict_gamelog, simplifed_predict_gamelog, training_gamelog)
        # clf_LR = build_model_LR(data, filenameLR)
        if model_name == 'clf_LR_p':
            data_p, df_p = build_data(predict_gamelog, simplifed_predict_gamelog, prior_gamelog)
            clf_LR_p = build_model_LR(data_p, filenameLR_p)

    print(predictions)
    assess = assess_prediction(actual_results, predictions)
    file.write(model_name + " , correct: " + str(assess[0]) + ", incorrect: " + str(assess[1]) +
               ", percentage: " + str(assess[0] / (assess[0] + assess[1])) + '\n')
    return assess


def predict_team_season_bo3(file, team, testing_year, training_years, prior_year):
    """ Predict the yearly win/loss for a team and check accuracy """
    file.write("Executing predict_team_season_bo3\n")
    predict_gamelog = gamelog_builder(testing_year, [team])
    simplifed_predict_gamelog = simplifed_gamelog_builder(testing_year, [team])
    # training_gamelog = gamelog_builder(training_years, teams)
    training_gamelog = gamelog_builder(training_years, [team])
    prior_gamelog = gamelog_builder(prior_year, [team])

    # BUILD DATA WITHOUT UPDATING AFTER EVERY GAME
    data, df = build_data(predict_gamelog, simplifed_predict_gamelog, training_gamelog)
    data_p, df_p = build_data(predict_gamelog, simplifed_predict_gamelog, prior_gamelog)
    # games = df.drop('winner_home', axis=1)
    # data = x_train, x_test, y_train, y_test

    clf_LR = build_model_LR(data, filenameLR)
    clf_LR_p = build_model_LR(data_p, filenameLR_p)
    clf_SVC = build_model_SVC(data, filenameSVC)

    # Gather real results - for comparison
    actual_results = []
    for game in predict_gamelog:
        if game[len(data_columns)-1] == 'home':
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
        # data, df = build_data(predict_gamelog, simplifed_predict_gamelog, training_gamelog)
        data_p, df_p = build_data(predict_gamelog, simplifed_predict_gamelog, prior_gamelog)
        # clf_LR = build_model_LR(data, filenameLR)
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


def execute_season_bo1(year, execute_teams, model_name):
    """ Automate testing an entire past season to compare predicted results against the actual results """
    # years = ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']
    training_years = []
    for i in range(num_of_training_years):
        if year - i - 1 >= minimum_year:
            training_years.insert(0, str(year - i - 1))
            # training_years.append(str(year - i - 1))
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


def execute_season_bo3(year, execute_teams):
    """ Automate testing an entire past season to compare predicted results against the actual results """
    # years = ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']
    training_years = []
    for i in range(num_of_training_years):
        if year - i - 1 >= minimum_year:
            training_years.append(str(year - i - 1))
    prior_year = [str(year - 1)]
    prediction_year = [str(year)]
    results = []

    for team in execute_teams:
        file = open(filename, "a")

        start = time()
        result = predict_team_season_bo3(file, team, prediction_year, training_years, prior_year)
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
    # predict_gamelog = gamelog_builder_full(['2016', '2017', '2018'], ['NYM'])

    columns = ['num', 'date', 'hometeam', 'awayteam', 'day', 'homepitcher',
               'homepitcher_wlp', 'homepitcher_era', 'homepitcher_whip', 'homepitcher_fip', 'awaypitcher',
               'awaypitcher_wlp', 'awaypitcher_era', 'awaypitcher_whip', 'awaypitcher_fip', 'winner']
    # df_scaled = pd.DataFrame(predict_gamelog, columns=columns)

    # df_all = pd.DataFrame(predict_gamelog, columns=columns)

    # df_ints = df_all.select_dtypes(include=['float'])

    # scaler = MinMaxScaler()

    # df_transformed = scaler.fit_transform(df_ints)

    # df = pd.DataFrame(predict_gamelog, columns=columns)
    # df = pd.DataFrame(temp, columns=['num', 'date', 'hometeam', 'awayteam', 'runshome', 'runsaway',
    #                                  'innings', 'day', 'homepitcher', 'awaypitcher', 'winner'])
    # df = pd.DataFrame(temp, columns=['num', 'date', 'hometeam', 'awayteam', 'runshome', 'runsaway',
    #                                  'innings', 'day', 'homepitcher', 'homepitcher_wlp', 'homepitcher_era',
    #                                  'homepitcher_whip', 'homepitcher_fip', 'awaypitcher', 'awaypitcher_wlp',
    #                                  'awaypitcher_era', 'awaypitcher_whip', 'awaypitcher_fip', 'winner'])
    # del df['num']
    # del df['date']
    # TODO test the effects of removing 'del df['date']
    # df = pd.get_dummies(df, drop_first=True)

    # Remove the test game from the training data
    # if 'winner_NA' in df.columns:
    #     df = df.drop('winner_NA', axis=1)  # axis 0 is rows, axis 1 is columns

    # x_train, x_test, y_train, y_test = train_test_split(df.drop('winner_home', axis=1), df['winner_home'])

    # knc = KNeighborsClassifier()
    # knc.fit(x_train, y_train)

    # print(classification_report(y_test, knc.predict(x_test)))
    # print(roc_auc_score(y_test, knc.predict(x_test)))

    # df_scaled = df_scaled.append(pd.DataFrame(df_transformed))
    # print('a')


def main():
    """ Main """
    # int 'year', list [str 'names']
    # execute_season_bo3(2015, teams)
    execute_season_bo1(2019, ['NYM'], 'clf_LR_p')


# Call main
main()
# test()
