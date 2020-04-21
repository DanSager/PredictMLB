""" Scrape webdata about various mlb teams via baseball-reference.com """

from urllib.request import urlopen as ureq
# pip install bs4
from bs4 import BeautifulSoup as Soup
from bs4 import Comment as Com
from teamdata.seasonstats import *

import sqlite3 as sql

# https://www.baseball-reference.com/teams/NYM/2012-schedule-scores.shtml
# https://www.baseball-reference.com/players/s/syndeno01.shtml


def extract_data():
    """
    Extract data from the baseball-reference.com reguarding team stats in addition to their pitchers.
    """
    teams = ['ARI', 'ATL', 'BAL', 'BOS', 'CHC', 'CHW', 'CIN', 'CLE', 'COL', 'DET', 'HOU', 'KCR', 'LAA', 'LAD', 'MIA',
             'MIL', 'MIN', 'NYM', 'NYY', 'OAK', 'PHI', 'PIT', 'SDP', 'SEA', 'SFG', 'STL', 'TBR', 'TEX', 'TOR', 'WSN']
    years = ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']

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
            # https://www.baseball-reference.com/teams/NYM/2012-schedule-scores.shtml

            # opening statsdbection, grabbing page
            uClient = ureq(myurl)
            page_html = uClient.read()
            uClient.close()

            # html parsing
            page_soup = Soup(page_html, "html.parser")

            # Create table for 'Team-Season'
            scheduletable = team + "Schedule"
            query = """CREATE TABLE IF NOT EXISTS """ + scheduletable + """ (
                        num INTEGER,
                        date TEXT,
                        team TEXT,
                        opponent TEXT,
                        home TEXT,
                        runs INTEGER,
                        runsallowed INTEGER,
                        innings INTEGER,
                        day INTEGER,
                        pitcher TEXT,
                        pitcher_wlp REAL,
                        pitcher_era REAL,
                        pitcher_whip REAL,
                        pitcher_fip REAL,
                        opp_pitcher TEXT,
                        opp_pitcher_wlp REAL,
                        opp_pitcher_era REAL,
                        opp_pitcher_whip REAL,
                        opp_pitcher_fip REAL,
                        win TEXT
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
                home = None
                num = winner = win = ""
                date = team = opponent = home = runs = runsallowed = innings = day = pitcher = opp_pitcher = ""
                pitcher_ref = opp_pitcher_ref = ""
                pitcher_wlp = opp_pitcher_wlp = 0.500
                pitcher_era = opp_pitcher_era = 4.5
                pitcher_whip = opp_pitcher_whip = 1.300
                pitcher_fip = opp_pitcher_fip = 4.5

                try:
                    num_container = game.findAll("th", {"data-stat": "team_game"})
                    num = num_container[0].text

                    date_container = game.findAll("td", {"data-stat": "date_game"})
                    date = date_container[0]["csk"]

                    team_container = game.findAll("td", {"data-stat": "team_ID"})
                    team = team_container[0].text

                    opp_container = game.findAll("td", {"data-stat": "opp_ID"})
                    opponent = opp_container[0].a.text

                    location_container = game.findAll("td", {"data-stat": "homeORvis"})
                    location = location_container[0].text
                    if location == "@":
                        home = 0
                    else:
                        home = 1

                    runs_container = game.findAll("td", {"data-stat": "R"})
                    runs = runs_container[0].text

                    runsallowed_container = game.findAll("td", {"data-stat": "RA"})
                    runsallowed = runsallowed_container[0].text

                    innings_container = game.findAll("td", {"data-stat": "extra_innings"})
                    innings = innings_container[0].text
                    if innings == "":
                        innings = "9"

                    dayornight_container = game.findAll("td", {"data-stat": "day_or_night"})
                    dayornight = dayornight_container[0].text
                    if dayornight == 'D':
                        day = "1"
                    else:
                        day = "0"

                    win_container = game.findAll("td", {"data-stat": "win_loss_result"})
                    win = win_container[0].text[0]
                    if win == "W" or win == "W-wo":
                        win = 1
                    else:
                        win = 0

                    winningpitcher_container = game.findAll("td", {"data-stat": "winning_pitcher"})
                    winningpitcher = winningpitcher_container[0].a["title"]
                    winningpitcher_ref = winningpitcher_container[0].a["href"]

                    losingpitcher_container = game.findAll("td", {"data-stat": "losing_pitcher"})
                    losingpitcher = losingpitcher_container[0].a["title"]
                    losingpitcher_ref = losingpitcher_container[0].a["href"]

                    if win:
                        pitcher = winningpitcher
                        pitcher_ref = winningpitcher_ref
                        opp_pitcher = losingpitcher
                        opp_pitcher_ref = losingpitcher_ref
                    else:
                        pitcher = losingpitcher
                        pitcher_ref = losingpitcher_ref
                        opp_pitcher = winningpitcher
                        opp_pitcher_ref = winningpitcher_ref

                    # https://www.baseball-reference.com/players/s/syndeno01.shtml
                    previous_year = str((int(year))-1)
                    pitcher_url = 'https://www.baseball-reference.com' + pitcher_ref
                    opp_pitcher_url = 'https://www.baseball-reference.com' + opp_pitcher_ref

                    # opening statsdbection, grabbing page
                    pitcher_Client = ureq(pitcher_url)
                    pitcher_page_html = pitcher_Client.read()
                    pitcher_Client.close()
                    opp_pitcher_Client = ureq(opp_pitcher_url)
                    opp_pitcher_page_html = opp_pitcher_Client.read()
                    opp_pitcher_Client.close()

                    # html parsing
                    pitcher_page_soup = Soup(pitcher_page_html, "html.parser")
                    opp_pitcher_page_soup = Soup(opp_pitcher_page_html, "html.parser")

                    pitcher_stats_container = pitcher_page_soup.findAll("table", {"id": "pitching_standard"})
                    pitcher_seasons = pitcher_stats_container[0].tbody.findAll("tr", {"class": "full"})

                    for season in pitcher_seasons:
                        season_num_container = season.findAll("th", {"data-stat": "year_ID"})
                        season_num = season_num_container[0].text
                        if season_num == year:
                            wlp_container = season.findAll("td", {"data-stat": "win_loss_perc"})
                            wlp = wlp_container[0].text
                            if wlp == "" or wlp == "inf":
                                wlp = pitcher_wlp
                            era_container = season.findAll("td", {"data-stat": "earned_run_avg"})
                            era = era_container[0].text
                            if era == "" or era == "inf":
                                era = pitcher_era
                            whip_container = season.findAll("td", {"data-stat": "whip"})
                            whip = whip_container[0].text
                            if whip == "" or whip == "inf":
                                whip = pitcher_whip
                            fip_container = season.findAll("td", {"data-stat": "fip"})
                            fip = fip_container[0].text
                            if fip == "" or fip == "inf":
                                fip = pitcher_fip
                            pitcher_wlp = wlp
                            pitcher_era = era
                            pitcher_whip = whip
                            pitcher_fip = fip

                    opp_pitcher_stats_container = opp_pitcher_page_soup.findAll("table", {"id": "pitching_standard"})
                    opp_pitcher_seasons = opp_pitcher_stats_container[0].tbody.findAll("tr", {"class": "full"})

                    for season in opp_pitcher_seasons:
                        season_num_container = season.findAll("th", {"data-stat": "year_ID"})
                        season_num = season_num_container[0].text
                        if season_num == year:
                            wlp_container = season.findAll("td", {"data-stat": "win_loss_perc"})
                            wlp = wlp_container[0].text
                            if wlp == "" or wlp == "inf":
                                wlp = opp_pitcher_wlp
                            era_container = season.findAll("td", {"data-stat": "earned_run_avg"})
                            era = era_container[0].text
                            if era == "" or era == "inf":
                                era = opp_pitcher_era
                            whip_container = season.findAll("td", {"data-stat": "whip"})
                            whip = whip_container[0].text
                            if whip == "" or whip == "inf":
                                whip = opp_pitcher_whip
                            fip_container = season.findAll("td", {"data-stat": "fip"})
                            fip = fip_container[0].text
                            if fip == "" or fip == "inf":
                                fip = opp_pitcher_fip
                            opp_pitcher_wlp = wlp
                            opp_pitcher_era = era
                            opp_pitcher_whip = whip
                            opp_pitcher_fip = fip

                except:
                    print("There was an error for team " + team + " with year " + year + ", game number " + num)

                # Writing essential game data
                game_data = GameSchedule(num, date, team, opponent, home, runs, runsallowed, innings, day,
                                         pitcher, pitcher_wlp, pitcher_era, pitcher_whip,
                                         pitcher_fip, opp_pitcher, opp_pitcher_wlp, opp_pitcher_era,
                                         opp_pitcher_whip, opp_pitcher_fip, win)
                # If game already exists: update data, else: create
                if get_game_by_index(statscursor, scheduletable, game_data.num):
                    update_game(statsdb, statscursor, scheduletable, game_data)
                else:
                    insert_game(statsdb, statscursor, scheduletable, game_data)

    statsdb.close()


# main
extract_data()
