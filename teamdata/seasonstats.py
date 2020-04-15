class SeasonStats:
    """A sample stats class"""

    def __init__(self, team, year):
        self.team = team
        self.year = year

    @property
    def email(self):
        return '{}-{}@gmail.com'.format(self.team, self.year)

    def __repr__(self):
        return "SeasonStats('{}', '{}')".format(self.team, self.year)


class GameSchedule:
    def __init__(self, num, date, hometeam, awayteam, runshome, runsaway, innings, day, homepitcher, homepitcher_era,
                 homepitcher_whip, awaypitcher, awaypitcher_era, awaypitcher_whip, winner):
        self.num = num
        self.date = date
        self.hometeam = hometeam
        self.awayteam = awayteam
        self.runshome = runshome
        self.runsaway = runsaway
        self.innings = innings
        self.day = day
        self.homepitcher = homepitcher
        self.homepitcher_era = homepitcher_era
        self.homepitcher_whip = homepitcher_whip
        self.awaypitcher = awaypitcher
        self.awaypitcher_era = awaypitcher_era
        self.awaypitcher_whip = awaypitcher_whip
        self.winner = winner


class WinLossSplit:
    def __init__(self, team, overall, home, away):
        self.team = team
        self.overall = overall
        self.home = home
        self.away = away


class Rival:
    def __init__(self, team, opp, overall):
        self.team = team
        self.opp = opp
        self.overall = overall


class Team:
    def __init__(self, team, year):
        self.team = team
        self.year = year


def insert_game(statsdb, statscursor, table, stat):
    query = "INSERT INTO " + table + " VALUES (:num, :date, :hometeam, :awayteam, :runshome, :runsaway, :innings, " \
                                     ":day, :homepitcher, :homepitcher_era, :homepitcher_whip, :awaypitcher, " \
                                     ":awaypitcher_era, :awaypitcher_whip, :winner)"
    with statsdb:
        statscursor.execute(query,
                            {'num': stat.num, 'date': stat.date, 'hometeam': stat.hometeam, 'awayteam': stat.awayteam,
                             'runshome': stat.runshome, 'runsaway': stat.runsaway,
                             'innings': stat.innings, 'day': stat.day, 'homepitcher': stat.homepitcher,
                             'homepitcher_era': stat.homepitcher_era, 'homepitcher_whip': stat.homepitcher_whip,
                             'awaypitcher': stat.awaypitcher, 'awaypitcher_era': stat.awaypitcher_era,
                             'awaypitcher_whip': stat.awaypitcher_whip, 'winner': stat.winner})


def insert_split(statsdb, statscursor, table, stat):
    query = "INSERT INTO " + table + " VALUES (:team, :overall, :home, :away)"
    with statsdb:
        statscursor.execute(query, {'team': stat.team, 'overall': stat.overall, 'home': stat.home, 'away': stat.away})


def insert_rival(statsdb, statscursor, table, stat):
    query = "INSERT INTO " + table + " VALUES (:team, :opp, :overall)"
    with statsdb:
        statscursor.execute(query, {'team': stat.team, 'opp': stat.opp, 'overall': stat.overall})


def update_game(statsdb, statscursor, table, stat):
    query = "UPDATE " + table + " SET date = :date, hometeam = :hometeam, awayteam = :awayteam, " \
                                "runshome = :runshome, runsaway = :runsaway, innings = :innings, " \
                                "day = :day, homepitcher = :homepitcher, homepitcher_era = :homepitcher_era, " \
                                "homepitcher_whip = :homepitcher_whip, awaypitcher = :awaypitcher , " \
                                "awaypitcher_era = :awaypitcher_era, awaypitcher_whip = :awaypitcher_whip, " \
                                "winner = :winner WHERE num = :num"
    with statsdb:
        statscursor.execute(query,
                            {'num': stat.num, 'date': stat.date, 'hometeam': stat.hometeam, 'awayteam': stat.awayteam,
                             'runshome': stat.runshome, 'runsaway': stat.runsaway, 'innings': stat.innings,
                             'day': stat.day, 'homepitcher': stat.homepitcher, 'homepitcher_era': stat.homepitcher_era,
                             'homepitcher_whip': stat.homepitcher_whip, 'awaypitcher': stat.awaypitcher,
                             'awaypitcher_era': stat.awaypitcher_era, 'awaypitcher_whip': stat.awaypitcher_whip,
                             'winner': stat.winner})


def update_split(statsdb, statscursor, table, stat):
    query = "UPDATE " + table + " SET overall = :overall, home = :home, away = :away WHERE team = :team"
    with statsdb:
        statscursor.execute(query, {'team': stat.team, 'overall': stat.overall, 'home': stat.home, 'away': stat.away})


def update_rival(statsdb, statscursor, table, stat):
    query = "UPDATE " + table + " SET overall = :overall WHERE team = :team AND opp = :opp"
    with statsdb:
        statscursor.execute(query, {'team': stat.team, 'opp': stat.opp, 'overall': stat.overall})


def get_game_by_index(statscursor, table, index):
    query = "SELECT * FROM " + table + " WHERE num=:num"
    statscursor.execute(query, {'num': index})
    return statscursor.fetchone()


def get_split_by_team(statscursor, table, stat):
    query = "SELECT * FROM " + table + " WHERE team=:team"
    statscursor.execute(query, {'team': stat.team})
    return statscursor.fetchone()


def get_rival_by_team(statscursor, table, stat):
    query = "SELECT * FROM " + table + " WHERE team=:team AND opp=:opp"
    statscursor.execute(query, {'team': stat.team, 'opp': stat.opp})
    return statscursor.fetchone()


def get_game_by_opp(statscursor, opp):
    # Bugged, needs work
    statscursor.execute("SELECT * FROM employee WHERE opp=:opp", {'opp': opp})
    return statscursor.fetchall()


def get_win_loss_split(statscursor, table, stat):
    query = "SELECT * FROM " + table + " WHERE team=:team"
    statscursor.execute(query, {'team': stat.team})
    return statscursor.fetchone()


def get_team_schedule(statscursor, table):
    query = "SELECT * FROM " + table
    statscursor.execute(query)
    return statscursor.fetchall()


def remove_game(statsdb, statscursor, table, index):
    query = "DELETE from " + table + " WHERE num = :num"
    with statsdb:
        statscursor.execute(query, {'num': index})
