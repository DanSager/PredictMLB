from urllib.request import urlopen as uReq
#pip install bs4
from bs4 import BeautifulSoup as soup

teams = set(['LAA', 'CHC', 'SDP', 'MIA', 'ATL', 'BAL', 'ARI', 'BOS', 'WSN', 'PHI', 'NYM', 'MIN', 'TOR', 'OAK', 'SFG', 'COL', 'LAD', 'MIL', 'SEA', 'CIN', 'PIT', 'DET', 'HOU', 'KCR', 'CHW', 'CLE', 'TBR', 'NYY', 'STL', 'TEX'])
years = set(['2019'])

#team = "NYM"
#year = "2019"

for year in years:
    for team in teams:
        myurl = 'https://www.baseball-reference.com/teams/'+ team + "/" + year + '-schedule-scores.shtml'

        filename = team + "-" + year + ".csv"
        f = open(filename, "w")
        header = "gm#, date, location, opp, w/l\n"
        f.write(header)

        # opening connection, grabbing page
        uClient = uReq(myurl)
        page_html = uClient.read()
        uClient.close()

        # html parsing
        page_soup = soup(page_html, "html.parser")

        # grab each product
        games_container = page_soup.findAll("table",{"id":"team_schedule"})

        games = games_container[0].tbody.findAll("tr",{"class":""})

        for game in games:
            try:
                print(game)
                gm_num_container = game.findAll("th",{"data-stat":"team_game"})
                gm_num = gm_num_container[0].text
                print(gm_num)

                date_container = game.findAll("td",{"data-stat":"date_game"})
                date = date_container[0]["csk"]
                print(date)

                location_container = game.findAll("td",{"data-stat":"homeORvis"})
                if (location_container[0].text == "@"):
                    location = "away"
                else:
                    location = "home"
                print(location)

                opp_container = game.findAll("td",{"data-stat":"opp_ID"})
                opp = opp_container[0].a.text
                print(opp)

                outcome_container = game.findAll("td",{"data-stat":"win_loss_result"})
                outcome = (outcome_container[0].text)[0]
                print(outcome)

                f.write(gm_num  + "," + date + "," + location + "," + opp + "," + outcome + "\n")

                teams.add(opp)
            except:
                print("There was an error on game")

f.close()