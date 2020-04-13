from regression.predict import *

# TODO change winning/losing pitcher to home/away pitcher


def assess_prediction(actual_results, predictions):
    """ Access if our prediction is correct or not """
    correct = incorrect = 0
    for i in range(len(actual_results)):
        if actual_results[i] == "home":
            if predictions[i]:
                correct = correct + 1
            else:
                incorrect = incorrect + 1
        else:
            if predictions[i]:
                incorrect = incorrect + 1
            else:
                correct = correct + 1
    return correct, incorrect


def predict_team_season(team, testing_year, training_year):
    """ Predict the yearly win/loss for a team and check accuracy """
    predict_gamelog = gamelog_builder(testing_year, team)
    simplifed_predict_gamelog = simplifed_gamelog_builder(testing_year, team)
    training_gamelog = gamelog_builder(training_year, teams)

    data, df = build_data(predict_gamelog, simplifed_predict_gamelog, training_gamelog)  # x_train, x_test, y_train, y_test

    models = load_model()

    # Gather real results - for comparison
    actual_results = []
    for game in predict_gamelog:
        actual_results.append(game[10])

    # Test Prediction
    games_count = len(df)
    model_count = len(models)
    predictions = [[] for x in range(model_count)]
    total_correct = total_incorrect = total_errors = 0
    for i in range(games_count):
        game = df.iloc[i].drop('winner_home')

        try:
            sample_prediction = models[0].predict([game])[:1]
        except ValueError:
            print("ValueError. With game: " + str(predict_gamelog[i]))
            models = build_model(data)

        for j in range(model_count):
            prediction = None
            try:
                prediction = models[j].predict([game])[:1]
                print(prediction)
                predictions[j].append(prediction)

            except ValueError:
                print("ValueError with game: " + str(predict_gamelog[i]))
                total_errors = total_errors + 1
            except TypeError:
                print("TypeError with game: " + str(predict_gamelog[i]))
                total_errors = total_errors + 1

    print(assess_prediction(actual_results, predictions[0]))
    print(assess_prediction(actual_results, predictions[1]))
    print("a")


def main():
    """ Main Function """

    training_years = ['2012']
    predict_years = ['2013']
    predict_team = ['NYM']
    predict_team_season(predict_team, predict_years, training_years)

    # num, date, hometeam, awayteam, runshome, runsaway, innings, day, winningpitcher, losingpitcher, winner
    # predict = [None, None, 'NYM', 'SDP', None, None, None, 1, None, None, None]
    # print(predict)
    # execute(True, predict, training_years)


# Call main
main()
