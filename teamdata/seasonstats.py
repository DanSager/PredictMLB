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
    def __init__(self, num, date, location, opp, outcome, win, win_ref, loss, loss_ref):
        self.num = num
        self.date = date
        self.location = location
        self.opp = opp
        self.outcome = outcome
        self.win = win
        self.win_ref = win_ref
        self.loss = loss
        self.loss_ref = loss_ref


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


def insert_game(statsdb, statscursor, table, stat):
    query = "INSERT INTO " + table + " VALUES (:num, :date, :location, :opp, :outcome, :win, :win_ref, :loss, :loss_ref)"
    with statsdb:
        statscursor.execute(query,
                            {'num': stat.num, 'date': stat.date, 'location': stat.location, 'opp': stat.opp,
                             'outcome': stat.outcome,
                             'win': stat.win, 'win_ref': stat.win_ref, 'loss': stat.loss, 'loss_ref': stat.loss_ref})


def insert_split(statsdb, statscursor, table, stat):
    query = "INSERT INTO " + table + " VALUES (:team, :overall, :home, :away)"
    with statsdb:
        statscursor.execute(query, {'team': stat.team, 'overall': stat.overall, 'home': stat.home, 'away': stat.away})


def insert_rival(statsdb, statscursor, table, stat):
    query = "INSERT INTO " + table + " VALUES (:team, :opp, :overall)"
    with statsdb:
        statscursor.execute(query, {'team': stat.team, 'opp': stat.opp, 'overall': stat.overall})


def update_game(statsdb, statscursor, table, stat):
    query = "UPDATE " + table + " SET date = :date, location = :location, opp = :opp, outcome = :outcome, " \
                                "win = :win, win_ref = :win_ref, loss = :loss, loss_ref = :loss_ref WHERE num = :num"
    with statsdb:
        statscursor.execute(query,
                            {'num': stat.num, 'date': stat.date, 'location': stat.location, 'opp': stat.opp,
                             'outcome': stat.outcome,
                             'win': stat.win, 'win_ref': stat.win_ref, 'loss': stat.loss, 'loss_ref': stat.loss_ref})


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
    statscursor.execute("SELECT * FROM employee WHERE opp=:opp", {'opp': opp})
    return statscursor.fetchall()


def remove_game(statsdb, statscursor, table, index):
    query = "DELETE from " + table + " WHERE num = :num"
    with statsdb:
        statscursor.execute(query, {'num': index})