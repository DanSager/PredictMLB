# Import statements
# from pydataset import data
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Database related imports
from teamdata.seasonstats import *
import sqlite3 as sql

# Global Variables
teams = ['ATL', 'ARI', 'BAL', 'BOS', 'CHC', 'CHW', 'CIN', 'CLE', 'COL', 'DET', 'HOU', 'KCR', 'LAA', 'LAD', 'MIA',
         'MIL', 'MIN', 'NYM', 'NYY', 'OAK', 'PHI', 'PIT', 'SDP', 'SEA', 'SFG', 'STL', 'TBR', 'TEX', 'TOR', 'WSN']
years = ['2016', '2017', '2018', '2019']
tables = ['WinLossSplit', 'TeamRivals']

homeWLS = None
awayWLS = None


def load_team_schedule(teamName, year):
    #teamOne = Team(team, year)

    # Create / Connect to db
    directory = '../teamdata/'

    dbname = directory + 'teamstats_' + year + '.db'
    statsdb = sql.connect(dbname)

    # Create a cursor to navigate the db
    statscursor = statsdb.cursor()

    table = teamName + 'Schedule'

    teamSchedule = get_team_schedule(statscursor, table)

    sched = pd.get_dummies(teamSchedule, drop_first=True)

    for game in teamSchedule: # num, date, location, opp, outcome, win, win_ref, loss, loss_ref
        location = game[2]
        oppName = game[3]
        local = Team(teamName, year)
        opp = Team(oppName, year)

        titanic = pd.get_dummies(game, drop_first=True)

        wls_table = tables[0]

        localWLS = get_win_loss_split(statscursor, wls_table, local)
        oppWLS = get_win_loss_split(statscursor, wls_table, opp)
        localWP = None
        oppWP = None

        if location == 'home':
            localWP = localWLS[2]
            oppWP = oppWLS[3]
        else:
            localWP = localWLS[3]
            oppWP = oppWLS[2]

        print(oppName + " " + location + " " + str(localWP) + " " + str(oppWP))

        # Logistic regression time



def retrieve_home_away_wls(team_one, team_two, year):
    # Create / Connect to db
    directory = '../teamdata/'

    dbname = directory + 'teamstats_' + year + '.db'
    statsdb = sql.connect(dbname)

    # Create a cursor to navigate the db
    statscursor = statsdb.cursor()

    table = tables[0]

    global homeWLS, awayWLS
    homeWLS = get_win_loss_split(statscursor, table, team_one)
    awayWLS = get_win_loss_split(statscursor, table, team_two)


def initalize(team, year):
    # Setup
    load_team_schedule(team, year)

# main
team = 'NYM'
year = years[0]
initalize(team, year)
