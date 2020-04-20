""" Gather human predictions based on the same info the algorithm has """
from regression.predict import *

directory = 'regression/control/'


def main(year, team, name):
    """ Main """
    filename_s = directory + name + '.txt'
    h_file = open(filename_s, "a")

    predictions = []
    gamelog = testing_gamelog_builder([str(year-1)], [str(year)], [team])

    i = 1
    print("Gathering human sample predictions, " + team + ' ' + str(year) + ', file: ' + filename_s)
    for game in gamelog:
        print(str(i) + ": Team = " + str(game[3]) + ", Opponent = " + str(game[4]) + ", Home (Yes = 1) = " + str(game[5]))
        print("Pitcher = " + str(game[10].replace(u'\xa0', u' ')) + ", Pitcher ERA = " + str(game[12]) + ", Pitcher WHIP = " + str(game[13]))
        print("Opp Pitcher = " + str(game[15].replace(u'\xa0', u' ')) + ", Opp Pitcher ERA = " + str(game[17]) + ", Opp Pitcher WHIP = " + str(game[18]))
        print("Team W/L percentage based on location = " + str(game[20]) + ", Opponent W/L percentage based on location = " + str(game[21]))
        console = input()
        if console == 't':
            console = '1'
        elif console == 'o':
            console = '0'
        else:
            while console != '1' or console != '0':
                print("try again")
                console = input()
                if console == 't':
                    console = '1'
                    break
                elif console == 'o':
                    console = '0'
                    break
        print('')
        predictions.append(int(console))
        h_file.write(console + '\n')
        i = i + 1

    h_file.write('\n')

    actual_results = []
    gamelog_results = gamelog_builder([str(year)], [team])
    for game in gamelog_results:
        if game[22] == '1':
            actual_results.append(1)
        else:
            actual_results.append(0)

    data = assess_prediction(actual_results, predictions)
    h_file.write("Human prediction accuracy, correct: " + str(data[0]) + ", incorrect: " + str(data[1]) +
                 ", percentage: " + str(data[0] / (data[0] + data[1])) + '\n')
    h_file.write(team + ' ' + str(year))

    h_file.close()


def assess(name):
    predictions = []
    filename_s = directory + name + '.txt'
    with open(filename_s) as file_in:
        for line in file_in:
            predictions.append(int(line[0]))

    actual_results = []
    gamelog_results = gamelog_builder(['2019'], ['ARI'])
    for game in gamelog_results:
        if game[22] == '1':
            actual_results.append(1)
        else:
            actual_results.append(0)

    data = assess_prediction(actual_results, predictions)
    print(data)


# main
main(2019, 'ARI', 'test2')
# assess('test2')
