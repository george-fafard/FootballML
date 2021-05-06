"""
    This file contains all necessary methods and data for working with the 
    dataset. 

    The methods to load the data and save it to files will be used with the
    raw data from the dataset. We can pre-load the raw data and save them to 
    files and upload them to the repo so we just have to clean the data and prepare
    it for training, instead of loading the raw data for every year every time which 
    can be time consuming. 

    To clean the data and prepare it for training, see the clean_data and
    get_training methods below. 

    Also, see the example at the bottom for loading the data and preparing it for training.
"""
# Required libraries
import numpy as np
import os
import pandas as pd

# Stats from dataset
from sportsipy.nfl.boxscore import Boxscores, Boxscore


# NFL team abbreviations
TEAMS = ['NOR', 'MIN', 'CHI', 'DET', 'MIA', 'BUF', 'TAM', 'CLE',
         'PIT', 'ATL', 'OTI', 'RAI', 'NWE', 'CIN', 'HTX', 'CLT',
         'JAX', 'DEN', 'NYG', 'CAR', 'CRD', 'RAM', 'SEA', 'SFO',
         'GNB', 'PHI', 'WAS', 'DAL', 'RAV', 'NYJ', 'KAN', 'SDG']


def game_data_from_year(year):
    """Load the data from a certain year. 
    
    Parameters
    ----------
    year : int
        The year to get the data from
    
    Returns
    -------
    DataFrame
        A DataFrame containing data for all the games in the year
    """
    # Weeks
    FIRST_WEEK = 1
    LAST_WEEK  = 21

    # Game data
    game_data = pd.DataFrame()
    
    try: 
        # Retrieve data for each week in the year
        for week in range(FIRST_WEEK, LAST_WEEK + 1):
            # Week data
            week_data = pd.DataFrame()

            # Retrieve data for each game in the week
            for game in range(len(Boxscores(week, year).games[str(week)+"-"+str(year)])):
                # Box score stats
                box_score = Boxscore(Boxscores(week, year).games[str(week)+"-"+str(year)][game]['boxscore']).dataframe

                # Add box score stats for the game to the data for
                # the week
                week_data = pd.concat([week_data, box_score])

            # Add a column for week number to the data for the week
            # and store the number of the current week
            week_data["week"] = [week]*len(Boxscores(week,year).games[str(week)+"-"+str(year)])

            # Add the data for the week to the game data for 
            # the year
            game_data = pd.concat([game_data, week_data])
    except:
        print("ERROR: Data loading failed")
        return -1
    else:
        return game_data


def save_game_data_to_files(start_year, end_year=None):
    """Save the game data for a range of years to files.

    This method serves as a general purpose method for both saving
    data for years in a range and simply saving data for one year. To 
    save data for just one year, only pass a value for the start year 
    and leave the end year as default.

    This helps save time for retrieving game data as it can
    take a while to run. This way, retrieval from the dataset only 
    has to occur once and then the data can be imported from
    the files in the future.

    Parameters
    ----------
    start_year : int
        The start year to be used as the lower bound for the range [Included]
    end_year : int, optional
        The end year to be used as the upper bound for the range [Included]. 
        If no value is passed, data will be saved for just the start year

    Returns
    ------
    None
    """
    # Error codes
    DATA_ERROR = -1

    # If it is desired to save data for only one year,
    # set the end year to the start year so the loop
    # only runs once
    if end_year is None:
        end_year = start_year

    # Save data
    for year in range(start_year, end_year + 1):
        # Game data for next year
        game_data = game_data_from_year(year)

        # Safety check to ensure data is loaded correctly before writing
        # to file
        if game_data == DATA_ERROR:
            print('ERROR: Loading the data for year {} failed'.format(year))
        else:
            # Filename to write
            data_filename = 'data{}.csv'.format(year)

            # Write data
            game_data.to_csv(data_filename)


def read_game_data_from_files(start_year, end_year=None):
    """Read the game data for a range of years from files.

    This method serves as a general purpose method for both reading
    data for years in a range and simply reading data for one year. To 
    read data for just one year, only pass a value for the start year 
    and leave the end year as default.

    Parameters
    ----------
    start_year : int
        The start year to be used as the lower bound for the range [Included]
    end_year : int, optional
        The end year to be used as the upper bound for the range [Included]. 
        If no value is passed, data will be read for just the start year

    Returns
    -------
    None
    """
    # If it is desired to read data for only one year,
    # set the end year to the start year so the loop
    # only runs once
    if end_year is None:
        end_year = start_year

    # Read data
    for year in range(start_year, end_year + 1):
        try:
            # Game data for next year
            game_data = []

            # Filename to read
            year_filename = os.getcwd() + '/data{}.csv'.format(year)

            # Extract game data from file
            year_file = pd.read_csv(year_filename, header=0)
            game_data.append(year_file)
        except FileNotFoundError:
            print('ERROR: Cannot find file for year {}'.format(year))
        except:
            print('ERROR: Other error occurred')


def clean_data(game_data):
    """Split the stats for all the games into stats for the away team and home team. 

    These are the columns produced:
    -----------------------------------------------------------------------------------------------------------------
        | attendance      | first_downs    | fourth_down_attempts | fourth_down_conversions | fumbles               |         
        | fumbles_lost    | interceptions  | net_pass_yards       | pass_attempts           | pass_completions      |   
        | pass_touchdowns | pass_yards     | penalties            | points                  | rush_attempts         |
        | rush_touchdowns | rush_yards     | third_down_attempts  | third_down_conversions  | time_of_possession    | 
        | times_sacked    | total_yards    | turnovers            | yards_from_penalties    | yards_lost_from_sacks |
        | duration        | roof           | surface              | time                    | temperature           |  
        | humidity        | wind           | week                 | win                     |                       |
        -------------------------------------------------------------------------------------------------------------

    Parameters
    ----------
    game_data : DataFrame 
        A DataFrame containing data for the games

    Returns
    -------
    list 
        A list of lists of lists containing game data for each team, with each team's game
        data split into home (first column) and away (second column) team data.

        
    """
    # Indices
    # [1]      attendance
    # [2:26]   all away team game stats
    # [28]     duration
    # [29:53]  ADD HERE
    # [56]     roof
    # [58:60]  surface, time
    # [61]     weather
    # [-1]     week

    # Set up team data for home and away games. Each index in the list
    # will represent a team, and for each team, there will be two lists,
    # with the first representing home games, and the second representing
    # away games
    teamdata = []
    for team in TEAMS:
        teamdata.append([[],[]])

    # Iterate through all games in the game data 
    for game in game_data:
        g=list(game)

        # Winning team for game
        winner = g[62]

        # If away team won, add a 1 to represent a win to the end of the away
        # team data and a 0 to represent a loss to the end of the home team data 
        if winner == 'Away':
            # Base for next calculation. Put here, it doesn't have to be defined
            # twice. 
            away_calculation_base = g[1:20] + [int(g[20][0:2]) + float(g[20][3:5])/60] + g[21:26]
            home_calculation_base = [g[1]]+g[29:47]+[int(g[47][0:2])+float(g[47][3:5])/60]+g[48:53]

            # Perform conversions if game duration is not a float
            if not isinstance(g[28], float):
                # Calculation including float conversion.
                # Broken up into arbitrary parts to reduce length and improve readability
                conv_part1 = [int(g[28].split(":")[0]) + float(g[28].split(":")[1])/60] + [g[56]]
                conv_part2 = g[58:60] + [g[61]] + [g[-1]]

                # Win identifiers
                away_win = [1]
                home_win = [0]
                
                # Add win identifiers
                home_team = home_calculation_base + conv_part1 + conv_part2 + home_win
                away_team = away_calculation_base + conv_part1 + conv_part2 + away_win
                
            else:
                # Calculation for if game duration is a float.
                # Broken up into arbitrary parts to reduce length and improve readability
                float_part = [3] + [g[56]]+g[58:60]+[g[61]]+[g[-1]]

                # Win identifiers
                away_win = [1]
                home_win = [0]

                # Add win identifiers
                home_team = home_calculation_base + float_part + home_win
                away_team = away_calculation_base + float_part + away_win
            
            
            # Convert all strings to ints so the data consists of only numbers
            if away_team[26]=='Outdoors' :
                away_team[26] = 1
                home_team[26] = 1
            else:
                away_team[26] = 0
                home_team[26] = 0
                
            if away_team[27]=='Grass' :
                away_team[27] = 1
                home_team[27] = 1
            else:
                away_team[27] = 0
                home_team[27] = 0

            
            if away_team[28].split(":")[1][2:4]=="pm":
                away_team[28] = (int(away_team[28].split(":")[0])+12) / 60
                home_team[28] = (int(home_team[28].split(":")[0])+12) / 60
            else:
                away_team[28] = (int(away_team[28].split(":")[0])) + float(away_team[28].split(":")[1][0:2])/60
                home_team[28] = (int(home_team[28].split(":")[0])) + float(home_team[28].split(":")[1][0:2])/60
                

            # Break up weather into it's component values, dealing with nan, a float constant that is provided if there is a missing value 
            # by replacing them with averages calculated from some year
            if isinstance(away_team[29],float):
                temp=55
                humidity=0.5
                wind=9
            elif len(away_team[29].split(" "))>6:
                temp = int(away_team[29].split(" ")[0])
                humidity = float(away_team[29].split(" ")[4][0:-2])/100
                if away_team[29].split(" ")[6] == 'wind,' or away_team[29].split(" ")[6] == 'wind':
                    wind = 0
                else:
                    wind = int(away_team[29].split(" ")[6])
            else: 
                temp = int(away_team[29].split(" ")[0])
                humidity = 0.5
                if away_team[29].split(" ")[3] == 'wind,' or away_team[29].split(" ")[3]=='wind':
                    wind = 0
                else:
                    wind = int(away_team[29].split(" ")[3])

            
            weather = [temp, humidity, wind]

            away_team = away_team[0:29] + weather + away_team[30:32]
            home_team = home_team[0:29] + weather + home_team[30:32]

            teamdata[TEAMS.index(g[63])][1].append(away_team)
            teamdata[TEAMS.index(g[53])][0].append(home_team)
        # Do the opposite if the home team won
        elif winner == 'Home':
            # Base for next calculation. Put here, it doesn't have to be defined
            # twice. 
            away_calculation_base = g[1:20] + [int(g[20][0:2]) + float(g[20][3:5])/60] + g[21:26]
            home_calculation_base = [g[1]]+g[29:47]+[int(g[47][0:2])+float(g[47][3:5])/60]+g[48:53]

            # Perform conversions if game duration is not a float
            if not isinstance(g[28], float):
                # Calculation including float conversion.
                # Broken up into arbitrary parts to reduce length and improve readability
                conv_part1 = [int(g[28].split(":")[0]) + float(g[28].split(":")[1])/60] + [g[56]]
                conv_part2 = g[58:60] + [g[61]] + [g[-1]]

                # Win identifiers
                away_win = [0]
                home_win = [1]
                
                # Add win identifiers
                home_team =  home_calculation_base + conv_part1 + conv_part2 + home_win
                away_team =  away_calculation_base + conv_part1 + conv_part2 + away_win
                
            else:
                # Calculation for if game duration is a float.
                # Broken up into arbitrary parts to reduce length and improve readability
                float_part = [3] + [g[56]]+g[58:60]+[g[61]]+[g[-1]]

                # Win identifiers
                away_win = [0]
                home_win = [1]

                # Add win identifiers
                home_team = home_calculation_base + float_part + home_win
                away_team = away_calculation_base + float_part + away_win
            

            # Convert all strings to ints so the data consists of only numbers
            if away_team[26]=='Outdoors' :
                away_team[26] = 1
                home_team[26] = 1
            else:
                away_team[26] = 0
                home_team[26] = 0
                
            if away_team[27]=='Grass' :
                away_team[27] = 1
                home_team[27] = 1
            else:
                away_team[27] = 0
                home_team[27] = 0

            if away_team[28].split(":")[1][2:4]=="pm":
                away_team[28] = (int(away_team[28].split(":")[0])+12) / 60
                home_team[28] = (int(home_team[28].split(":")[0])+12) / 60
            else:
                away_team[28] = (int(away_team[28].split(":")[0])) + float(away_team[28].split(":")[1][0:2])/60
                home_team[28] = (int(home_team[28].split(":")[0])) + float(home_team[28].split(":")[1][0:2])/60
                

            # Break up weather into it's component values, dealing with nan, a float constant that is provided if there is a missing value 
            # by replacing them with averages calculated from some year
            if isinstance(away_team[29],float):
                temp=55
                humidity=0.5
                wind=9
            elif len(away_team[29].split(" "))>6:
                temp = int(away_team[29].split(" ")[0])
                humidity = float(away_team[29].split(" ")[4][0:-2])/100
                if away_team[29].split(" ")[6] == 'wind,' or away_team[29].split(" ")[6] == 'wind':
                    wind = 0
                else:
                    wind = int(away_team[29].split(" ")[6])
            else: 
                temp = int(away_team[29].split(" ")[0])
                humidity = 0.5
                if away_team[29].split(" ")[3] == 'wind,' or away_team[29].split(" ")[3]=='wind':
                    wind = 0
                else:
                    wind = int(away_team[29].split(" ")[3])

            
            weather = [temp, humidity, wind]


            away_team = away_team[0:29] + weather + away_team[30:32]
            home_team = home_team[0:29] + weather + home_team[30:32]


            teamdata[TEAMS.index(g[63])][0].append(home_team)
            teamdata[TEAMS.index(g[53])][1].append(away_team)
    return teamdata


def get_training(previous_year,current_year,games,year):
    """Gets data to be trained on by the models
        
    Takes data from games and calculates the totals for the games before that game for each team in that 
    season + 3 games from the previous season, taking into account home vs away, and also takes the stats about the
    game that we could know about the game before it happens like the weather and location

    These are the columns produced:
    ---------------------------------------------------------------------------------------------------------------------------------------------------
        | away_attendance            | away_first_downs     | away_fourth_down_attempts | away_fourth_down_conversions | away_fumbles                 |         
        | away_fumbles_lost          | away_interceptions   | away_net_pass_yards       | away_pass_attempts           | away_pass_completions        |   
        | away_pass_touchdowns       | away_pass_yards      | away_penalties            | away_points                  | away_rush_attempts           |
        | away_rush_touchdowns       | away_rush_yards      | away_third_down_attempts  | away_third_down_conversions  | away_time_of_possession      | 
        | away_times_sacked          | away_total_yards     | away_turnovers            | away_yards_from_penalties    | away_yards_lost_from_sacks   |
        | away_duration              | away_roof            | away_surface              | away_time                    | away_temperature             |  
        | away_humidity              | away_wind            | away_week                 | away_win                     | away_games_played            |
        | away_made_playoffs         | home_attendance      | home_first_downs          | home_fourth_down_attempts    | home_fourth_down_conversions |
        | home_fumbles               | home_fumbles_lost    | home_interceptions        | home_net_pass_yards          | home_pass_attempts           |
        | home_pass_completions      | home_pass_touchdowns | home_pass_yards           | home_penalties               | home_points                  |
        | home_rush_attempts         | home_rush_touchdowns | home_rush_yards           | home_third_down_attempts     | home_third_down_conversions  |
        | home_time_of_possession    | home_times_sacked    | home_total_yards          | home_turnovers               | home_yards_from_penalties    |
        | home_yards_lost_from_sacks | home_duration        | home_roof                 | home_surface                 | home_time                    |
        | home_temperature           | home_humidity        | home_wind                 | home_week                    | home_win                     |
        | home_games_played          | home_made_playoffs   | roof                      | surface                      | time                         |
        | temperature                | humidtity            | wind                      |                              |                              |
        -----------------------------------------------------------------------------------------------------------------------------------------------

        Parameters
        ----------
        previous_year : List of lists of lists
            data for the previous year, gotten from clean_data
        current_year : List of lists of lists
            data for the year in question, gotten from clean_data
        games : np array
            The raw data for the year in question

        Returns
        ------- 
        list, list
            x,y where x has the columns listed above and y has 1 for home team win and 0 for away team win.
    """
    counters = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
    trainingX = []
    trainingY = []

    
    # Figure out which teams made the playoffs
    made_playoffs = []
    for i in TEAMS:
        made_playoffs.append(0)
        
    for i in range(len(TEAMS)):
        for j in range(2):
            for g in previous_year[i][j]:
                if g[-2] >= 18:
                    made_playoffs[i] = 1
        
    # Next we cycle through the games for a specific year
    for game in range(len(games)):
        g = list(games[game])
        
        if(int(g[-1])<17):
            # Get the teams playing for that game
            if g[62] == 'Away':
                awayabbr = g[63]
                homeabbr = g[53]
                win=0
            else:
                awayabbr = g[53]
                homeabbr = g[63]
                win = 1

            
            Ai = TEAMS.index(awayabbr)
            Hi = TEAMS.index(homeabbr)


            # Get the averages of the three games before the game
            if counters[Ai][1] <= 3:
                averages_away = previous_year[Ai][1][-1]
                for i in range(counters[Ai][1]-3,-1):
                    for j in range(len(previous_year[Ai][1][i])):
                        averages_away[j] += previous_year[Ai][1][i][j]
                for i in range(counters[Ai][1]):
                    for j in range(len(current_year[Ai][1][i])):
                        averages_away[j] += current_year[Ai][1][i][j]
            else:
                averages_away = current_year[Ai][1][counters[Ai][1]-3]
                for i in range(counters[Ai][1]-2,counters[Ai][1]):
                    for j in range(len(current_year[Ai][1][i])):
                        averages_away[j] += current_year[Ai][1][i][j]
            
            for i in range(len(averages_away)):
                averages_away[i]=averages_away[i]/3
                
                
            if counters[Hi][0] <= 3:
                averages_home = previous_year[Hi][0][-1]
                for i in range(counters[Hi][0]-3,-1):
                    for j in range(len(previous_year[Hi][0][i])):
                        averages_home[j] += previous_year[Hi][0][i][j]
                for i in range(counters[Hi][0]):
                    for j in range(len(current_year[Hi][0][i])):
                        averages_home[j] += current_year[Hi][0][i][j]
            else:
                averages_home = current_year[Hi][0][counters[Hi][0]-3]
                for i in range(counters[Hi][0]-2,counters[Hi][0]):
                    for j in range(len(current_year[Hi][0][i])):
                        averages_home[j] += current_year[Hi][0][i][j]
            
            for i in range(len(averages_home)):
                averages_home[i]=averages_home[i]/3

            # Update counters for these teams
            counters[Ai][1] += 1
            counters[Hi][0] += 1

            # Get the info about the game we could know before hand by getting the info for the game as we did in cleaned data
            home_calculation_base = [g[1]] + g[29:47] + [int(g[47][0:2]) + float(g[47][3:5])/60] + g[48:53]

            # Perform conversions if game duration is not a float
            if not isinstance(g[28], float):
                # Calculation including float conversion.
                # Broken up into arbitrary parts to reduce length and improve readability
                conv_part1 = [int(g[28].split(":")[0]) + float(g[28].split(":")[1])/60] + [g[56]]
                conv_part2 = g[58:60] + [g[61]] + [g[-1]]

                # Win identifiers
                home_win = [1]
                
                # Add win identifiers
                home_team =  home_calculation_base + conv_part1 + conv_part2 + home_win
            else:
                # Calculation for if game duration is a float.
                # Broken up into arbitrary parts to reduce length and improve readability
                float_part = [3] + [g[56]] + g[58:60] + [g[61]] + [g[-1]]

                # Win identifiers
                home_win = [1]

                # Add win identifiers
                home_team = home_calculation_base + float_part + home_win
            

            # Convert all strings to ints so the data consists of only numbers
            if home_team[26]=='Outdoors' :
                venue = 1
            else:
                venue = 0
                
            if home_team[27]=='Grass' :
                field = 1
            else:
                field = 0

            
            if home_team[28].split(":")[1][2:4]=="pm":
                time = (int(home_team[28].split(":")[0])+12) / 60
            else:
                time = (int(home_team[28].split(":")[0])) + float(home_team[28].split(":")[1][0:2]) / 60
                

            # Break up weather into it's component values, dealing with nan, a float constant that is provided if there is a missing value 
            # by replacing them with averages calculated from some year
            if isinstance(home_team[29],float):
                temp=55
                humidity=0.5
                wind=9
            elif len(home_team[29].split(" "))>6:
                temp = int(home_team[29].split(" ")[0])
                humidity = float(home_team[29].split(" ")[4][0:-2])/100
                if home_team[29].split(" ")[6] == 'wind,' or home_team[29].split(" ")[6] == 'wind':
                    wind = 0
                else:
                    wind = int(home_team[29].split(" ")[6])
            else: 
                temp = int(home_team[29].split(" ")[0])
                humidity = 0.5
                if home_team[29].split(" ")[3] == 'wind,' or home_team[29].split(" ")[3]=='wind':
                    wind = 0
                else:
                    wind = int(home_team[29].split(" ")[3])

            
            weather = [temp, humidity, wind]

            # Add everything to the list of training datas
            trainingX.append(averages_home + [counters[Hi][0]] + [made_playoffs[Hi]] + averages_away + [counters[Ai][1]] + [made_playoffs[Ai]] + [venue] + [field] + [time] + weather + [year])
            trainingY.append(win)

    return trainingX, trainingY


"""
Example for loading the data and getting training set:

xtraining=[]
ytraining=[]
aveAway=[]
aveHome=[]
columns=[]
for i in range(1,10):
    try:
        xtraintemp,ytraintemp=get_training(clean_data(np.array(data201[i-1])),clean_data(np.array(data201[i])),np.array(data201[i]))
        xtraining+=xtraintemp
        ytraining+=ytraintemp
    except:
        print(i)
        
    try:
        xtraintemp,ytraintemp=get_training(clean_data(np.array(data200[i-1])),clean_data(np.array(data200[i])),np.array(data200[i]))
        xtraining+=xtraintemp
        ytraining+=ytraintemp
    except:
        print(i)
"""
