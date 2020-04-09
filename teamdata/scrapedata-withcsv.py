from urllib.request import urlopen as ureq
# pip install bs4
from bs4 import BeautifulSoup as Soup
from bs4 import Comment as Com
from teamdata.seasonstats import SeasonStats

import sqlite3 as sql


def insert(conn, c, stat):
    with conn:
        c.execute("INSERT INTO employee VALUES (:first, :last, :pay)",
                  {'first': stat.team, 'last': stat.year, 'pay': .532})


def get_by_name(c, team):
    c.execute("SELECT * FROM employee WHERE first=:first", {'first': team})
    return c.fetchone()


def update_pay(conn, c, stat, pay):
    with conn:
        c.execute("""UPDATE employee SET pay = :pay WHERE first = :first AND last = :last""",
                  {'first': stat.team, 'last': stat.year, 'pay': pay})


def remove(conn, c, stat):
    with conn:
        c.execute("DELETE from employee WHERE first = :first AND last = :last",
                  {'first': stat.team, 'last': stat.year})


def extract_data():
    teams = {'LAA', 'CHC', 'SDP', 'MIA', 'ATL', 'BAL', 'ARI', 'BOS', 'WSN', 'PHI', 'NYM', 'MIN', 'TOR', 'OAK', 'SFG',
             'COL', 'LAD', 'MIL', 'SEA', 'CIN', 'PIT', 'DET', 'HOU', 'KCR', 'CHW', 'CLE', 'TBR', 'NYY', 'STL', 'TEX'}
    years = {'2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019'}
    # teams = {'NYM'}

    for year in years:
        for team in teams:
            myurl = 'https://www.baseball-reference.com/teams/' + team + "/" + year + '-schedule-scores.shtml'

            # opening connection, grabbing page
            uClient = ureq(myurl)
            page_html = uClient.read()
            uClient.close()

            # html parsing
            page_soup = Soup(page_html, "html.parser")

            # sqlite
            conn = sql.connect('teamstats.db')
            # conn = sql.connect(':memory:')

            c = conn.cursor()

            c.execute("""CREATE TABLE employee (
                        First text,
                        last text,
                        pay real
                        )""")

            team_1 = SeasonStats(team, year)

            insert(conn, c, team_1)

            person = get_by_name(c, 'NYM')
            print(person)

            update_pay(conn, c, team_1, .753)

            person = get_by_name(c, 'NYM')
            print(person)

            conn.close()

            # open up team file
            filename = team + "-" + year + ".csv"
            f = open(filename, "w")

            # read 'Team Win/Loss Splits Table'
            year_container = page_soup.find("div", {"id": "all_win_loss"})
            commentsoup = Soup(year_container.find(text=lambda text: isinstance(text, Com)), "html.parser")

            # read year win/loss splits
            year_header = "Overall W-L%, Home W-L%, Away W-L%\n"
            f.write(year_header)
            column_one = commentsoup.find("div", {"id": "win_loss_1"})
            overall_win_loss = column_one.findAll("tr")[2].findAll("td")[5].text
            home_win_loss = column_one.findAll("tr")[5].findAll("td")[5].text
            away_win_loss = column_one.findAll("tr")[6].findAll("td")[5].text
            f.write(overall_win_loss.replace(" ", "") + "," + home_win_loss.replace(" ", "") + ","
                    + away_win_loss.replace(" ", "") + "\n\n")

            # read opponent win/loss split
            opponent_header = "Opponent, W-L%\n"
            f.write(opponent_header)
            column_three = commentsoup.find("div", {"id": "win_loss_3"})
            opponent_stat_container = column_three.findAll("tr")
            for opponent_stat in opponent_stat_container[2:]:
                opponent_name = opponent_stat.findAll("td")[0].text
                opponent_win_loss = opponent_stat.findAll("td")[5].text
                f.write(opponent_name + "," + opponent_win_loss.replace(" ", "") + "\n")
            f.write("\n")

            # grab each game
            game_header = "gm#, date, location, opp, w/l, win, win ref, loss, loss ref\n"
            f.write(game_header)
            games_container = page_soup.findAll("table", {"id": "team_schedule"})
            games = games_container[0].tbody.findAll("tr", {"class": ""})

            for game in games:
                # set defaults
                num = date = location = opp = outcome = win = win_ref = loss = loss_ref = ""

                try:
                    num_container = game.findAll("th", {"data-stat": "team_game"})
                    num = num_container[0].text

                    date_container = game.findAll("td", {"data-stat": "date_game"})
                    date = date_container[0]["csk"]

                    location_container = game.findAll("td", {"data-stat": "homeORvis"})
                    if location_container[0].text == "@":
                        location = "away"
                    else:
                        location = "home"

                    opp_container = game.findAll("td", {"data-stat": "opp_ID"})
                    opp = opp_container[0].a.text

                    outcome_container = game.findAll("td", {"data-stat": "win_loss_result"})
                    outcome = outcome_container[0].text[0]

                    win_container = game.findAll("td", {"data-stat": "winning_pitcher"})
                    win = win_container[0].a["title"]
                    win_ref = win_container[0].a["href"]

                    loss_container = game.findAll("td", {"data-stat": "losing_pitcher"})
                    loss = loss_container[0].a["title"]
                    loss_ref = loss_container[0].a["href"]

                except:
                    print("There was an error for team " + team + " with year " + year + ", game number " + num)

                # writing essential data
                f.write(num + "," + date + "," + location + "," + opp + "," + outcome + "," + win +
                        "," + win_ref + "," + loss + "," + loss_ref + "\n")

            f.close()


# main
extract_data()
