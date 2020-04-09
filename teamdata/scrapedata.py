from urllib.request import urlopen as ureq
# pip install bs4
from bs4 import BeautifulSoup as Soup
from bs4 import Comment as Com
from teamdata.seasonstats import *

import sqlite3 as sql

# TODO: add season schedules to the db via https://www.baseball-reference.com/leagues/MLB/2019-schedule.shtml


def extract_data():
    teams = ['ATL', 'ARI', 'BAL', 'BOS', 'CHC', 'CHW', 'CIN', 'CLE', 'COL', 'DET', 'HOU', 'KCR', 'LAA', 'LAD', 'MIA',
             'MIL', 'MIN', 'NYM', 'NYY', 'OAK', 'PHI', 'PIT', 'SDP', 'SEA', 'SFG', 'STL', 'TBR', 'TEX', 'TOR', 'WSN']
    years = ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']
    # teams = {'NYM'}

    for year in years:

        # Create / Connect to db
        dbname = 'teamstats_' + year + '.db'
        statsdb = sql.connect(dbname)

        # Create a cursor to navigate the db
        statscursor = statsdb.cursor()

        # Create table for 'WinLossSplit'
        wlsplittable = "WinLossSplit"
        query = """CREATE TABLE IF NOT EXISTS """ + wlsplittable + """ (
                                team text,
                                overall real,
                                home real,
                                away real
                                )"""
        statscursor.execute(query)

        # Create table for 'TeamRivals'
        rivalstable = "TeamRivals"
        query = """CREATE TABLE IF NOT EXISTS """ + rivalstable + """ (
                                        team text,
                                        opp text,
                                        overall real
                                        )"""
        statscursor.execute(query)

        for team in teams:
            myurl = 'https://www.baseball-reference.com/teams/' + team + "/" + year + '-schedule-scores.shtml'

            # opening statsdbection, grabbing page
            uClient = ureq(myurl)
            page_html = uClient.read()
            uClient.close()

            # html parsing
            page_soup = Soup(page_html, "html.parser")

            # Create table for 'Team-Season'
            scheduletable = team + "Schedule"
            query = """CREATE TABLE IF NOT EXISTS """ + scheduletable + """ (
                        num text,
                        date text,
                        location text,
                        opp text,
                        outcome text,
                        win text,
                        win_ref text,
                        loss text,
                        loss_ref text
                        )"""
            statscursor.execute(query)

            # read 'Team Win/Loss Splits Table'
            year_container = page_soup.find("div", {"id": "all_win_loss"})
            commentsoup = Soup(year_container.find(text=lambda text: isinstance(text, Com)), "html.parser")

            # read year win/loss splits
            column_one = commentsoup.find("div", {"id": "win_loss_1"})
            overall_win_loss = column_one.findAll("tr")[2].findAll("td")[5].text
            home_win_loss = column_one.findAll("tr")[5].findAll("td")[5].text
            away_win_loss = column_one.findAll("tr")[6].findAll("td")[5].text
            wlsplit = WinLossSplit(team, overall_win_loss, home_win_loss, away_win_loss)
            if get_split_by_team(statscursor, wlsplittable, wlsplit):
                update_split(statsdb, statscursor, wlsplittable, wlsplit)
            else:
                insert_split(statsdb, statscursor, wlsplittable, wlsplit)

            # read opponent win/loss split
            column_three = commentsoup.find("div", {"id": "win_loss_3"})
            opponent_stat_container = column_three.findAll("tr")
            for opponent_stat in opponent_stat_container[2:]:
                opponent_name = opponent_stat.findAll("td")[0].text
                opponent_win_loss = opponent_stat.findAll("td")[5].text
                rival = Rival(team, opponent_name, opponent_win_loss)
                if get_rival_by_team(statscursor, rivalstable, rival):
                    update_rival(statsdb, statscursor, rivalstable, rival)
                else:
                    insert_rival(statsdb, statscursor, rivalstable, rival)

            # grab each game
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

                # -- writing essential game data -- #
                game_data = GameSchedule(num, date, location, opp, outcome, win, win_ref, loss, loss_ref)
                # If game already exists: update data, else: create
                if get_game_by_index(statscursor, scheduletable, game_data.num):
                    update_game(statsdb, statscursor, scheduletable, game_data)
                else:
                    insert_game(statsdb, statscursor, scheduletable, game_data)

    statsdb.close()


# main
extract_data()