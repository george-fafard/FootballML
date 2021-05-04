"""
    This file contains all necessary methods and data for working with the 
    dataset.

    If we decide to pre-load the data to files and upload the files to the 
    repo, we can simply import those files and don't need to use the code to
    load the dataset below. However, if we would prefer to manually load the
    data into the notebooks for our individual classifiers, simply import the 
    required methods into the notebook and run.
"""
# Required libraries
import numpy as np
import os
import pandas as pd

# Stats from dataset
from sportsipy.nfl.boxscore import Boxscores, Boxscore


############################################################################
# If everything runs correctly with these commented out, then we can       #
# remove them                                                              #
############################################################################
#import math

# sklearn utilities
#from sklearn import datasets
#from sklearn.metrics import confusion_matrix, classification_report
#from sklearn import preprocessing

# sklearn models
#from sklearn.linear_model import LogisticRegression
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.svm import SVC
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.decomposition import PCA                          
############################################################################


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
                boxscore = Boxscore(Boxscores(week, year).games[str(week)+"-"+str(year)][game]['boxscore']).dataframe

                # Add box score stats for the game to the data for
                # the week
                week_data = pd.concat([week_data, boxscore])

            # NOT SURE WHAT THIS IS FOR
            week_data["week"] = [week]*len(Boxscores(week,year).games[str(week)+"-"+str(year)])

            # Store the game data for the week
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

        TODO: THIS WAS BASED ON WHAT I INTERPRETED YOU HAD THE RETURN DATA AS BUT I AM STILL
        A LITTLE CONFUSED ON THE LIST STRUCTURE SO WE CAN EDIT THIS AND MAKE IT MORE ACCURATE/
        DETAILED
    """
    # Set up team data for home and away games
    teamdata = []
    for team in TEAMS:
        teamdata.append([[],[]])

    # I have no idea what any of this means lol
    # What are all these magic numbers?
    # What are all these crazy uses of ternary operators lol?
    # It seems there may be some omitted parentheses and stuff that could cause
    # errors in here so we should really clean it up and refactor it.
    for game in game_data:
        g=list(game)
        if g[62]=='Away':
            team1=g[1:20]+[int(g[20][0:2])+float(g[20][3:5])/60]+g[21:26]+([int(g[28].split(":")[0])+float(g[28].split(":")[1])/60] if not isinstance(g[28],float) else [3])+[g[56]]+g[58:60]+[g[61]]+[g[-1]]+[1]
            team2=[g[1]]+g[29:47]+[int(g[47][0:2])+float(g[47][3:5])/60]+g[48:53]+([int(g[28].split(":")[0])+float(g[28].split(":")[1])/60] if not isinstance(g[28],float) else [3])+[g[56]]+g[58:60]+[g[61]]+[g[-1]]+[0]



            team1[26] = (1 if team1[26]=='Outdoors' else 0)
            team2[26] = (1 if team2[26]=='Outdoors' else 0)
            team1[27] = (1 if team1[27]=='Grass' else 0)
            team2[27] = (1 if team2[27]=='Grass' else 0)
            team1[28] = (int(team1[28].split(":")[0])+12 if team1[28].split(":")[1][2:4]=="pm" else int(team1[28].split(":")[0])) + float(team1[28].split(":")[1][0:2])/60
            team2[28]=  (int(team2[28].split(":")[0])+12 if team2[28].split(":")[1][2:4]=="pm" else int(team2[28].split(":")[0])) + float(team2[28].split(":")[1][0:2])/60

            
            
            weather1=[]
            if isinstance(team1[29],float):
                weather1=[55,0.5,9]
            elif len(team1[29].split(" "))>6:
                weather1=[int(team1[29].split(" ")[0]),float(team1[29].split(" ")[4][0:-2])/100, (0 if team1[29].split(" ")[6]=='wind,' or team1[29].split(" ")[6]=='wind' else  int(team1[29].split(" ")[6]))]
            else: 
                weather1=[int(team1[29].split(" ")[0]),.50, (0 if team1[29].split(" ")[3]=='wind,' or team1[29].split(" ")[3]=='wind' else  int(team1[29].split(" ")[3]))]


            weather2=[]
            if isinstance(team2[29],float):
                weather2=[55,0.5,9]
            elif len(team2[29].split(" "))>6:
                weather2=[int(team2[29].split(" ")[0]),float(team2[29].split(" ")[4][0:-2])/100, (0 if team2[29].split(" ")[6]=='wind,' or team2[29].split(" ")[6]=='wind' else  int(team2[29].split(" ")[6]))]
            else: 
                weather2=[int(team2[29].split(" ")[0]),.50, (0 if team2[29].split(" ")[3]=='wind,' or team2[29].split(" ")[3]=='wind' else  int(team2[29].split(" ")[3]))]



            team1=team1[0:29]+weather1+team1[30:32]
            team2=team2[0:29]+weather2+team2[30:32]

            teamdata[TEAMS.index(g[63])][1].append(team1)
            teamdata[TEAMS.index(g[53])][0].append(team2)
        else:
            
            team1=g[1:20]+[int(g[20][0:2])+float(g[20][3:5])/60]+g[21:26]+([int(g[28].split(":")[0])+float(g[28].split(":")[1])/60] if not isinstance(g[28],float) else [3])+[g[56]]+g[58:60]+[g[61]]+[g[-1]]+[0]
            team2=[g[1]]+g[29:47]+[int(g[47][0:2])+float(g[47][3:5])/60]+g[48:53]+([int(g[28].split(":")[0])+float(g[28].split(":")[1])/60] if not isinstance(g[28],float) else [3])+[g[56]]+g[58:60]+[g[61]]+[g[-1]]+[1]



            team1[26] = (1 if team1[26]=='Outdoors' else 0)
            team2[26]=  (1 if team2[26]=='Outdoors' else 0)
            team1[27]=  (1 if team1[27]=='Grass' else 0)
            team2[27]=  (1 if team2[27]=='Grass' else 0)
            team1[28]=  (int(team1[28].split(":")[0])+12 if team1[28].split(":")[1][2:4]=="pm" else int(team1[28].split(":")[0])) + float(team1[28].split(":")[1][0:2])/60
            team2[28]=  (int(team2[28].split(":")[0])+12 if team2[28].split(":")[1][2:4]=="pm" else int(team2[28].split(":")[0])) + float(team2[28].split(":")[1][0:2])/60

            
            
            weather1=[]
            if isinstance(team1[29],float):
                weather1=[55,0.5,9]
            elif len(team1[29].split(" "))>6:
                weather1=[int(team1[29].split(" ")[0]),float(team1[29].split(" ")[4][0:-2])/100, (0 if team1[29].split(" ")[6]=='wind,' or team1[29].split(" ")[6]=='wind' else  int(team1[29].split(" ")[6]))]
            else: 
                weather1=[int(team1[29].split(" ")[0]),.50, (0 if team1[29].split(" ")[3]=='wind,' or team1[29].split(" ")[3]=='wind' else  int(team1[29].split(" ")[3]))]


            weather2=[]
            if isinstance(team2[29],float):
                weather2=[55,0.5,9]
            elif len(team2[29].split(" "))>6:
                weather2=[int(team2[29].split(" ")[0]),float(team2[29].split(" ")[4][0:-2])/100, (0 if team2[29].split(" ")[6]=='wind,' or team2[29].split(" ")[6]=='wind' else  int(team2[29].split(" ")[6]))]
            else: 
                weather2=[int(team2[29].split(" ")[0]),.50, (0 if team2[29].split(" ")[3]=='wind,' or team2[29].split(" ")[3]=='wind' else  int(team2[29].split(" ")[3]))]

    
                
            team1=team1[0:29]+weather1+team1[30:32]
            team2=team2[0:29]+weather2+team2[30:32]



            teamdata[TEAMS.index(g[63])][0].append(team2)
            teamdata[TEAMS.index(g[53])][1].append(team1)
    return teamdata


#These are the columns produced by getTraining
#'away_average_attendance', 'away_average_first_downs', 'away_average_fourth_down_attempts', 'away_average_fourth_down_conversions', 'away_average_fumbles', 'away_average_fumbles_lost', 'away_average_interceptions', 'away_average_net_pass_yards', 'away_average_pass_attempts', 'away_average_pass_completions', 'away_average_pass_touchdowns', 'away_average_pass_yards', 'away_average_penalties', 'away_average_points', 'away_average_rush_attempts', 'away_average_rush_touchdowns', 'away_average_rush_yards', 'away_average_third_down_attempts', 'away_average_third_down_conversions', 'away_average_time_of_possession', 'away_average_times_sacked', 'away_average_total_yards', 'away_average_turnovers', 'away_average_yards_from_penalties', 'away_average_yards_lost_from_sacks', 'away_average_duration', 'away_average_roof', 'away_average_surface', 'away_average_time', 'away_average_temperature', 'away_average_humidity', 'away_average_wind', 'away_average_week', 'away_average_win'
#'home_average_attendance', 'home_average_first_downs', 'home_average_fourth_down_attempts', 'home_average_fourth_down_conversions', 'home_average_fumbles', 'home_average_fumbles_lost', 'home_average_interceptions', 'home_average_net_pass_yards', 'home_average_pass_attempts', 'home_average_pass_completions', 'home_average_pass_touchdowns', 'home_average_pass_yards', 'home_average_penalties', 'home_average_points', 'home_average_rush_attempts', 'home_average_rush_touchdowns', 'home_average_rush_yards', 'home_average_third_down_attempts', 'home_average_third_down_conversions', 'home_average_time_of_possession', 'home_average_times_sacked', 'home_average_total_yards', 'home_average_turnovers', 'home_average_yards_from_penalties', 'home_average_yards_lost_from_sacks', 'home_average_duration', 'home_average_roof', 'home_average_surface', 'home_average_time', 'home_average_temperature', 'home_average_humidity', 'home_average_wind', 'home_average_week', 'home_average_win'
#'roof', 'surface', 'time', 'temperature', 'humidity', 'wind'


def getTraining(data2014,data2015,games):
    """Get data

        Takes data from games and calculates the averages for the games before that game for each team in that 
       season + 3 games from the previous season, taking into account
        #home vs away, and also takes the stats about the game that we could know about the game before it happens like
        #the weather and location
        #@param data2014 data for the previous year, gotten from clean_data
        #@param data2015 data for the year in question, gotten from clean_data
        #@param games The raw data for the year in question

        Returns
        ------- 
        list, list
            x,y where x has the columns listed above and y has 1 for home team win and 0 for away team win.

        TODO: EDIT THIS TO MAKE SURE IT IS ACCURATE
    """
    counters=[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
    trainingX=[]
    trainingY=[]
    for game in range(len(games)):
        g=list(games[game])
        if g[62]=='Away':
            awayabbr=g[63]
            homeabbr=g[53]
            win=0
        else:
            awayabbr=g[53]
            homeabbr=g[63]
            win=1



        averagesA=[]+data2014[TEAMS.index(awayabbr)][1][-3]
        for i in range(len(averagesA)):
            averagesA[i]+=data2014[TEAMS.index(awayabbr)][1][-2][i]+data2014[TEAMS.index(awayabbr)][1][-1][i]
        for i in range(counters[TEAMS.index(awayabbr)][1]):
            for j in range(len(data2015[TEAMS.index(awayabbr)][1][i])):
                averagesA[i]+=data2015[TEAMS.index(awayabbr)][1][i][j]
        for i in range(len(averagesA)):
            averagesA[i] = averagesA[i]/float(counters[TEAMS.index(awayabbr)][1]+3)



        averagesH=[]+data2014[TEAMS.index(homeabbr)][0][-3]
        for i in range(len(averagesH)):
            averagesH[i]+=data2014[TEAMS.index(homeabbr)][0][-2][i]+data2014[TEAMS.index(homeabbr)][0][-1][i]
        for i in range(counters[TEAMS.index(homeabbr)][0]):
            for j in range(len(data2015[TEAMS.index(homeabbr)][0][i])):
                averagesH[i]+=data2015[TEAMS.index(homeabbr)][0][i][j]
        for i in range(len(averagesH)):
            averagesH[i] = averagesH[i]/float(counters[TEAMS.index(homeabbr)][0]+3)


        Ai=TEAMS.index(awayabbr)
        Hi=TEAMS.index(homeabbr)
        counters[Ai][1]+=1
        counters[Hi][0]+=1


        team1=g[1:20]+[int(g[20][0:2])+float(g[20][3:5])/60]+g[21:26]+[g[28]]+[g[56]]+g[58:60]+[g[61]]+[g[-1]]+[1]



        venue=  (1 if team1[26]=='Outdoors' else 0)
        field=  (1 if team1[27]=='Grass' else 0)
        time=  (int(team1[28].split(":")[0])+12 if team1[28].split(":")[1][2:4]=="pm" else int(team1[28].split(":")[0])) + float(team1[28].split(":")[1][0:2])/60



        weather1=[]
        if isinstance(team1[29],float):
            weather1=[55,0.5,9]
        elif len(team1[29].split(" "))>6:
            weather1=[int(team1[29].split(" ")[0]),float(team1[29].split(" ")[4][0:-2])/100, (0 if team1[29].split(" ")[6]=='wind,' or team1[29].split(" ")[6]=='wind' else  int(team1[29].split(" ")[6]))]
        else: 
            weather1=[int(team1[29].split(" ")[0]),.50, (0 if team1[29].split(" ")[3]=='wind,' or team1[29].split(" ")[3]=='wind' else  int(team1[29].split(" ")[3]))]



        trainingX.append(averagesH+averagesA+[venue]+[field]+[time]+weather1)
        trainingY.append(win)

    return trainingX,trainingY


#getting the data that we might train models on.

xtraining=[]
ytraining=[]
aveAway=[]
aveHome=[]
columns=[]
for i in range(1,10):
    try:
        xtraintemp,ytraintemp=getTraining(clean_data(np.array(data201[i-1])),clean_data(np.array(data201[i])),np.array(data201[i]))
        xtraining+=xtraintemp
        ytraining+=ytraintemp
    except:
        print(i)
        
    try:
        xtraintemp,ytraintemp=getTraining(clean_data(np.array(data200[i-1])),clean_data(np.array(data200[i])),np.array(data200[i]))
        xtraining+=xtraintemp
        ytraining+=ytraintemp
    except:
        print(i)
