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
    def __init__(self, num, date, team, opponent, home, runs, runsallowed, innings, day, pitcher, pitcher_wlp,
                 pitcher_era, pitcher_whip, pitcher_fip, opp_pitcher, opp_pitcher_wlp, opp_pitcher_era,
                 opp_pitcher_whip, opp_pitcher_fip, win):
        self.num = num
        self.date = date
        self.team = team
        self.opponent = opponent
        self.home = home
        self.runs = runs
        self.runsallowed = runsallowed
        self.innings = innings
        self.day = day
        self.pitcher = pitcher
        self.pitcher_wlp = pitcher_wlp
        self.pitcher_era = pitcher_era
        self.pitcher_whip = pitcher_whip
        self.pitcher_fip = pitcher_fip
        self.opp_pitcher = opp_pitcher
        self.opp_pitcher_wlp = opp_pitcher_wlp
        self.opp_pitcher_era = opp_pitcher_era
        self.opp_pitcher_whip = opp_pitcher_whip
        self.opp_pitcher_fip = opp_pitcher_fip
        self.win = win


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
    query = "INSERT INTO " + table + " VALUES (:num, :date, :team, :opponent, :home, :runs, :runsallowed, " \
                                     ":innings, " \
                                     ":day, :pitcher, :pitcher_wlp, :pitcher_era, :pitcher_whip, " \
                                     ":pitcher_fip, :opp_pitcher, :opp_pitcher_wlp, :opp_pitcher_era, " \
                                     ":opp_pitcher_whip, :opp_pitcher_fip, :win)"
    with statsdb:
        statscursor.execute(query,
                            {'num': stat.num, 'date': stat.date, 'team': stat.team, 'opponent': stat.opponent,
                             'home': stat.home, 'runs': stat.runs, 'runsallowed': stat.runsallowed,
                             'innings': stat.innings, 'day': stat.day, 'pitcher': stat.pitcher,
                             'pitcher_wlp': stat.pitcher_wlp, 'pitcher_era': stat.pitcher_era,
                             'pitcher_whip': stat.pitcher_whip, 'pitcher_fip': stat.pitcher_fip,
                             'opp_pitcher': stat.opp_pitcher, 'opp_pitcher_wlp': stat.opp_pitcher_wlp,
                             'opp_pitcher_era': stat.opp_pitcher_era, 'opp_pitcher_whip': stat.opp_pitcher_whip,
                             'opp_pitcher_fip': stat.opp_pitcher_fip, 'win': stat.win})


def insert_split(statsdb, statscursor, table, stat):
    query = "INSERT INTO " + table + " VALUES (:team, :overall, :home, :away)"
    with statsdb:
        statscursor.execute(query, {'team': stat.team, 'overall': stat.overall, 'home': stat.home, 'away': stat.away})


def insert_rival(statsdb, statscursor, table, stat):
    query = "INSERT INTO " + table + " VALUES (:team, :opp, :overall)"
    with statsdb:
        statscursor.execute(query, {'team': stat.team, 'opp': stat.opp, 'overall': stat.overall})


def update_game(statsdb, statscursor, table, stat):
    query = "UPDATE " + table + " SET date = :date, team = :team, opponent = :opponent, home = :home, " \
                                "runs = :runs, runsallowed = :runsallowed, innings = :innings, " \
                                "day = :day, pitcher = :pitcher, pitcher_wlp = :pitcher_wlp, " \
                                "pitcher_era = :pitcher_era, pitcher_fip = :pitcher_fip, " \
                                "pitcher_whip = :pitcher_whip, opp_pitcher = :opp_pitcher , " \
                                "opp_pitcher_wlp = :opp_pitcher_wlp, opp_pitcher_era = :opp_pitcher_era, " \
                                "opp_pitcher_whip = :opp_pitcher_whip, opp_pitcher_fip = :opp_pitcher_fip, " \
                                "win = :win WHERE num = :num"
    with statsdb:
        statscursor.execute(query,
                            {'num': stat.num, 'date': stat.date, 'team': stat.team, 'opponent': stat.opponent,
                             'home': stat.home, 'runs': stat.runs, 'runsallowed': stat.runsallowed,
                             'innings': stat.innings,
                             'day': stat.day, 'pitcher': stat.pitcher, 'pitcher_wlp': stat.pitcher_wlp,
                             'pitcher_era': stat.pitcher_era, 'pitcher_whip': stat.pitcher_whip,
                             'pitcher_fip': stat.pitcher_fip, 'opp_pitcher': stat.opp_pitcher,
                             'opp_pitcher_wlp': stat.opp_pitcher_wlp, 'opp_pitcher_era': stat.opp_pitcher_era,
                             'opp_pitcher_whip': stat.opp_pitcher_whip, 'opp_pitcher_fip': stat.opp_pitcher_fip,
                             'win': stat.win})


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
