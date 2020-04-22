""" Gather human predictions based on the same info the algorithm has """

from regression.predict import *

directory = 'regression/control/'


def main(year, team, name):
    """
    Present a human an entire teams season where they respond with either t or o to assess how they would predict a
    given season.
    :param year: int - indicates the season being simulated. ex, 2018
    :param team: str - which team's season is being simulated. ex, 'NYM'
    :param name: str - name of file being assessed. ex. 'test2'
    """
    filename_s = directory + name + '.txt'
    h_file = open(filename_s, "a")

    predictions = []
    gamelog = build_testing_gamelog(str(year-1), str(year), [team])

    i = 1
    print("Gathering human sample predictions, " + team + ' ' + str(year) + ', file: ' + filename_s)
    print("Type t or 1 for the given team or type o or 0 for the opponent.")
    for game in gamelog:
        print(str(i) + ": Team = " + str(game[1]) + ", Opponent = " + str(game[2]) + ", Home (Yes = 1) = " +
              str(game[3]))
        print("Pitcher = " + str(game[8].replace(u'\xa0', u' ')) + ", Pitcher ERA = " + str(game[10]) +
              ", Pitcher WHIP = " + str(game[11]))
        print("Opp Pitcher = " + str(game[13].replace(u'\xa0', u' ')) + ", Opp Pitcher ERA = " + str(game[15]) +
              ", Opp Pitcher WHIP = " + str(game[16]))
        print("Team W/L percentage based on location = " + str(game[18]) +
              ", Opponent W/L percentage based on location = " + str(game[19]))
        console = input()
        if console == 't':
            console = '1'
        elif console == 'o':
            console = '0'
        else:
            while True:
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
    gamelog_results = build_gamelog([str(year)], [team])
    for game in gamelog_results:
        if game[20] == '1':
            actual_results.append(1)
        else:
            actual_results.append(0)

    data = assess_prediction(actual_results, predictions)
    h_file.write("Human prediction accuracy, correct: " + str(data[0]) + ", incorrect: " + str(data[1]) +
                 ", percentage: " + str(data[0] / (data[0] + data[1])) + '\n')
    h_file.write(team + ' ' + str(year))

    h_file.close()


def assess_file(name):
    """
    Determine the accuracy of a text file where every line is a 0 or 1.
    :param name: str - name of file being assessed. ex. 'test2'
    """
    predictions = []
    filename_s = directory + name + '.txt'
    with open(filename_s) as file_in:
        for line in file_in:
            predictions.append(int(line[0]))

    actual_results = []
    gamelog_results = build_gamelog(['2019'], ['ARI'])
    for game in gamelog_results:
        if game[22] == '1':
            actual_results.append(1)
        else:
            actual_results.append(0)

    data = assess_prediction(actual_results, predictions)
    print(data)


# main
main(2019, 'ARI', 'test3')
# assess_file('test2')
