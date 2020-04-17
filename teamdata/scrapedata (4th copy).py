from urllib.request import urlopen as ureq
# pip install bs4
from bs4 import BeautifulSoup as Soup
from bs4 import Comment as Com
from teamdata.seasonstats import *

import sqlite3 as sql


def extract_data():
    teams = ['ARI', 'ATL', 'BAL', 'BOS', 'CHC', 'CHW', 'CIN', 'CLE', 'COL', 'DET', 'HOU', 'KCR', 'LAA', 'LAD', 'MIA',
             'MIL', 'MIN', 'NYM', 'NYY', 'OAK', 'PHI', 'PIT', 'SDP', 'SEA', 'SFG', 'STL', 'TBR', 'TEX', 'TOR', 'WSN']
    years = ['2015']
    # teams = ['ARI']

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
            # https://www.baseball-reference.com/teams/tgl.cgi?team=NYM&t=p&year=2019#all_team_pitching_gamelogs

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
                        hometeam text,
                        awayteam text,
                        runshome text,
                        runsaway text,
                        innings text,
                        day text,
                        homepitcher text,
                        homepitcher_wlp text,
                        homepitcher_era text,
                        homepitcher_whip text,
                        homepitcher_fip text,
                        awaypitcher text,
                        awaypitcher_wlp text,
                        awaypitcher_era text,
                        awaypitcher_whip text,
                        awaypitcher_fip text,
                        winner text
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
                num = date = hometeam = awayteam = runshome = runsaway = innings = day = homepitcher = awaypitcher = winner = ""
                homepitcher_ref = awaypitcher_ref = ""
                homepitcher_wlp = awaypitcher_wlp = "0.500"
                homepitcher_era = awaypitcher_era = '4.5'
                homepitcher_whip = awaypitcher_whip = '1.300'
                homepitcher_fip = awaypitcher_fip = '4.5'

                try:
                    num_container = game.findAll("th", {"data-stat": "team_game"})
                    num = num_container[0].text

                    date_container = game.findAll("td", {"data-stat": "date_game"})
                    date = date_container[0]["csk"]

                    team_container = game.findAll("td", {"data-stat": "team_ID"})
                    team = team_container[0].text

                    opp_container = game.findAll("td", {"data-stat": "opp_ID"})
                    opp = opp_container[0].a.text

                    location_container = game.findAll("td", {"data-stat": "homeORvis"})
                    if location_container[0].text == "@":
                        home = False
                        hometeam = opp
                        awayteam = team
                    else:
                        home = True
                        hometeam = team
                        awayteam = opp

                    runsscored_container = game.findAll("td", {"data-stat": "R"})
                    runsscored = runsscored_container[0].text

                    runsallowed_container = game.findAll("td", {"data-stat": "RA"})
                    runsallowed = runsallowed_container[0].text

                    if home:
                        runshome = runsscored
                        runsaway = runsallowed
                    else:
                        runshome = runsallowed
                        runsaway = runsscored

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

                    outcome_container = game.findAll("td", {"data-stat": "win_loss_result"})
                    outcome = outcome_container[0].text[0]
                    if outcome == "W":
                        if home:
                            winner = "home"
                        else:
                            winner = "away"
                    else:
                        if home:
                            winner = "away"
                        else:
                            winner = "home"

                    winningpitcher_container = game.findAll("td", {"data-stat": "winning_pitcher"})
                    winningpitcher = winningpitcher_container[0].a["title"]
                    winningpitcher_ref = winningpitcher_container[0].a["href"]

                    losingpitcher_container = game.findAll("td", {"data-stat": "losing_pitcher"})
                    losingpitcher = losingpitcher_container[0].a["title"]
                    losingpitcher_ref = losingpitcher_container[0].a["href"]

                    if winner == "home":
                        homepitcher = winningpitcher
                        homepitcher_ref = winningpitcher_ref
                        awaypitcher = losingpitcher
                        awaypitcher_ref = losingpitcher_ref
                    else:
                        homepitcher = losingpitcher
                        homepitcher_ref = losingpitcher_ref
                        awaypitcher = winningpitcher
                        awaypitcher_ref = winningpitcher_ref

                    # https://www.baseball-reference.com/players/s/syndeno01.shtml
                    previous_year = str((int(year))-1)
                    homepitcher_url = 'https://www.baseball-reference.com' + homepitcher_ref
                    awaypitcher_url = 'https://www.baseball-reference.com' + awaypitcher_ref

                    # opening statsdbection, grabbing page
                    homepitcher_Client = ureq(homepitcher_url)
                    homepitcher_page_html = homepitcher_Client.read()
                    homepitcher_Client.close()
                    awaypitcher_Client = ureq(awaypitcher_url)
                    awaypitcher_page_html = awaypitcher_Client.read()
                    awaypitcher_Client.close()

                    # html parsing
                    homepitcher_page_soup = Soup(homepitcher_page_html, "html.parser")
                    awaypitcher_page_soup = Soup(awaypitcher_page_html, "html.parser")

                    homepitcher_stats_container = homepitcher_page_soup.findAll("table", {"id": "pitching_standard"})
                    homepitcher_seasons = homepitcher_stats_container[0].tbody.findAll("tr", {"class": "full"})

                    for season in homepitcher_seasons:
                        season_num_container = season.findAll("th", {"data-stat": "year_ID"})
                        season_num = season_num_container[0].text
                        if season_num == previous_year:
                            wlp_container = season.findAll("td", {"data-stat": "win_loss_perc"})
                            wlp = wlp_container[0].text
                            if wlp == "":
                                wlp = homepitcher_wlp
                            era_container = season.findAll("td", {"data-stat": "earned_run_avg"})
                            era = era_container[0].text
                            whip_container = season.findAll("td", {"data-stat": "whip"})
                            whip = whip_container[0].text
                            fip_container = season.findAll("td", {"data-stat": "fip"})
                            fip = fip_container[0].text
                            homepitcher_wlp = wlp
                            homepitcher_era = era
                            homepitcher_whip = whip
                            homepitcher_fip = fip

                    awaypitcher_stats_container = awaypitcher_page_soup.findAll("table", {"id": "pitching_standard"})
                    awaypitcher_seasons = awaypitcher_stats_container[0].tbody.findAll("tr", {"class": "full"})

                    for season in awaypitcher_seasons:
                        season_num_container = season.findAll("th", {"data-stat": "year_ID"})
                        season_num = season_num_container[0].text
                        if season_num == previous_year:
                            wlp_container = season.findAll("td", {"data-stat": "win_loss_perc"})
                            wlp = wlp_container[0].text
                            if wlp == "":
                                wlp = awaypitcher_wlp
                            era_container = season.findAll("td", {"data-stat": "earned_run_avg"})
                            era = era_container[0].text
                            whip_container = season.findAll("td", {"data-stat": "whip"})
                            whip = whip_container[0].text
                            fip_container = season.findAll("td", {"data-stat": "fip"})
                            fip = fip_container[0].text
                            awaypitcher_wlp = wlp
                            awaypitcher_era = era
                            awaypitcher_whip = whip
                            awaypitcher_fip = fip

                except:
                    print("There was an error for team " + team + " with year " + year + ", game number " + num)

                # -- writing essential game data -- #
                game_data = GameSchedule(num, date, hometeam, awayteam, runshome, runsaway, innings, day,
                                         homepitcher, homepitcher_wlp, homepitcher_era, homepitcher_whip,
                                         homepitcher_fip, awaypitcher, awaypitcher_wlp, awaypitcher_era,
                                         awaypitcher_whip, awaypitcher_fip, winner)
                # If game already exists: update data, else: create
                if get_game_by_index(statscursor, scheduletable, game_data.num):
                    update_game(statsdb, statscursor, scheduletable, game_data)
                else:
                    insert_game(statsdb, statscursor, scheduletable, game_data)

    statsdb.close()


# main
extract_data()
