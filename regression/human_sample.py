""" Gather human predictions based on the same info the algorithm has """
from regression.predict import *

directory = 'regression/human_sample/'


def main(year, team, name):
    """ Main """
    filename_s = directory + name + '.txt'
    h_file = open(filename_s, "a")

    predictions = []
    gamelog = gamelog_builder([str(year)], [team])

    i = 1
    print("Gathering human sample predictions, " + team + ' ' + str(year) + ', file: ' + filename_s)
    for game in gamelog:
        print(str(i) + ": Home team = " + game[2] + ", Away team = " + game[3])
        print("Home Pitcher = " + game[8] + ", Home Pitcher ERA = " + game[9] + ", Home Pitcher WHIP = " + game[10])
        print("Away Pitcher = " + game[11] + ", Away Pitcher ERA = " + game[12] + ", Away Pitcher WHIP = " + game[13])
        console = input()
        if console == 'h':
            console = '1'
        if console == 'a':
            console = '0'
        print('')
        predictions.append(int(console))
        h_file.write(console + '\n')
        i = i + 1

    h_file.write('\n')

    actual_results = []
    for game in gamelog:
        if game[14] == 'home':
            actual_results.append(1)
        else:
            actual_results.append(0)

    data = assess_prediction(actual_results, predictions)
    h_file.write("Human prediction accuracy, correct: " + str(data[0]) + ", incorrect: " + str(data[1]) +
                 ", percentage: " + str(data[0] / (data[0] + data[1])) + '\n')

    h_file.close()


# main
main(2012, 'ARI', 'danny-test1')
