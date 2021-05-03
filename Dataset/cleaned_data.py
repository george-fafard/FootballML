# Stats from dataset
from sportsipy.nfl.boxscore import Boxscores, Boxscore

# Required libraries
import pandas as pd
import numpy as np
import os
import math

# sklearn utilities
from sklearn import datasets
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import preprocessing

# sklearn models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA


# A function to get the data from a certain year and returns a DataFrame
# @param int year representing the year you wan't to get the data from
# @return A dataframe containing the data.
def game_data_from_year(year):
    returnDF=pd.DataFrame()
    try: 
        for week in range(1,22):
            weekDF=pd.DataFrame()
            for game in range(len(Boxscores(week,year).games[str(week)+"-"+str(year)])):
                weekDF=pd.concat([weekDF,Boxscore(Boxscores(week,year).games[str(week)+"-"+str(year)][game]['boxscore']).dataframe])
            weekDF["week"] = [week]*len(Boxscores(week,year).games[str(week)+"-"+str(year)])
            returnDF=pd.concat([returnDF,weekDF])
    except:
        print(week)
    return returnDF


# Gets the data from the 2000's and saves it to files so you don't have to rerun this code in the future
# because this code takes a while to run
data2000=game_data_from_year(2000)
data2000.to_csv("data2000.csv")
data2001=game_data_from_year(2001)
data2001.to_csv("data2001.csv")
data2002=game_data_from_year(2002)
data2002.to_csv("data2002.csv")
data2003=game_data_from_year(2003)
data2003.to_csv("data2003.csv")
data2004=game_data_from_year(2004)
data2004.to_csv("data2004.csv")
data2005=game_data_from_year(2005)
data2005.to_csv("data2005.csv")
data2006=game_data_from_year(2006)
data2006.to_csv("data2006.csv")
data2007=game_data_from_year(2007)
data2007.to_csv("data2007.csv")
data2008=game_data_from_year(2008)
data2008.to_csv("data2008.csv")
data2009=game_data_from_year(2009)
data2009.to_csv("data2009.csv")


# Gets the data from the 2010's and saves it to files again
data2010=game_data_from_year(2010)
data2010.to_csv("data2010.csv")
data2011=game_data_from_year(2011)
data2011.to_csv("data2011.csv")
data2012=game_data_from_year(2012)
data2012.to_csv("data2012.csv")
data2013=game_data_from_year(2013)
data2013.to_csv("data2013.csv")
data2014=game_data_from_year(2014)
data2014.to_csv("data2014.csv")
data2015=game_data_from_year(2015)
data2015.to_csv("data2015.csv")
data2016=game_data_from_year(2016)
data2016.to_csv("data2016.csv")
data2017=game_data_from_year(2017)
data2017.to_csv("data2017.csv")
data2018=game_data_from_year(2018)
data2018.to_csv("data2018.csv")
data2019=game_data_from_year(2019)
data2019.to_csv("data2019.csv")


# Reads in the data from the files for the 2000's and stores it in a list of dataframes
data200 = []
for i in range(10):
    data200.append(pd.read_csv(os.getcwd() + '/data200'+str(i)+'.csv', header=0))


# Reads in the data from the files for the 2010's and stores it in a list of dataframes
data201 = []
for i in range(10):
    data201.append(pd.read_csv(os.getcwd() + '/data201'+str(i)+'.csv', header=0))


# NFL team abbreviations
teams = ['NOR', 'MIN', 'CHI', 'DET', 'MIA', 'BUF', 'TAM', 'CLE', 'PIT', 'ATL', 'OTI', 'RAI', 'NWE', 'CIN', 'HTX',
         'CLT', 'JAX', 'DEN', 'NYG', 'CAR', 'CRD', 'RAM', 'SEA', 'SFO', 'GNB', 'PHI', 'WAS', 'DAL', 'RAV', 'NYJ', 
         'KAN', 'SDG']

#These are the columns that are produced by clean data: 
#'attendance','first_downs','fourth_down_attempts','fourth_down_conversions','fumbles','fumbles_lost','interceptions',
#'net_pass_yards','pass_attempts','pass_completions','pass_touchdowns','pass_yards','penalties','points','rush_attempts',
#'rush_touchdowns','rush_yards','third_down_attempts','third_down_conversions','time_of_possession','times_sacked',
#'total_yards','turnovers','yards_from_penalties','yards_lost_from_sacks','duration','roof','surface',
#'time','temperature','humidity','wind','week','win'

#splits all the games into stats for the away team and home team and saves them in the list at the index designated for that team
#with the first column in each team representing home games and the 2nd column representing away
#@param newdata being a dataframe containing games
#@return a list of      lists     of lists
#          team      home vs away    stats for the game for that team
def cleandata(newdata):
    teamdata = []
    for team in teams:
        teamdata.append([[],[]])

    for game in newdata:
        g=list(game)
        if g[62]=='Away':
            team1=g[1:20]+[int(g[20][0:2])+float(g[20][3:5])/60]+g[21:26]+([int(g[28].split(":")[0])+float(g[28].split(":")[1])/60] if not isinstance(g[28],float) else [3])+[g[56]]+g[58:60]+[g[61]]+[g[-1]]+[1]
            team2=[g[1]]+g[29:47]+[int(g[47][0:2])+float(g[47][3:5])/60]+g[48:53]+([int(g[28].split(":")[0])+float(g[28].split(":")[1])/60] if not isinstance(g[28],float) else [3])+[g[56]]+g[58:60]+[g[61]]+[g[-1]]+[0]



            team1[26]=  (1 if team1[26]=='Outdoors' else 0)
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

            teamdata[teams.index(g[63])][1].append(team1)
            teamdata[teams.index(g[53])][0].append(team2)
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



            teamdata[teams.index(g[63])][0].append(team2)
            teamdata[teams.index(g[53])][1].append(team1)
    return teamdata


#These are the columns produced by getTraining
#'away_average_attendance', 'away_average_first_downs', 'away_average_fourth_down_attempts', 'away_average_fourth_down_conversions', 'away_average_fumbles', 'away_average_fumbles_lost', 'away_average_interceptions', 'away_average_net_pass_yards', 'away_average_pass_attempts', 'away_average_pass_completions', 'away_average_pass_touchdowns', 'away_average_pass_yards', 'away_average_penalties', 'away_average_points', 'away_average_rush_attempts', 'away_average_rush_touchdowns', 'away_average_rush_yards', 'away_average_third_down_attempts', 'away_average_third_down_conversions', 'away_average_time_of_possession', 'away_average_times_sacked', 'away_average_total_yards', 'away_average_turnovers', 'away_average_yards_from_penalties', 'away_average_yards_lost_from_sacks', 'away_average_duration', 'away_average_roof', 'away_average_surface', 'away_average_time', 'away_average_temperature', 'away_average_humidity', 'away_average_wind', 'away_average_week', 'away_average_win'
#'home_average_attendance', 'home_average_first_downs', 'home_average_fourth_down_attempts', 'home_average_fourth_down_conversions', 'home_average_fumbles', 'home_average_fumbles_lost', 'home_average_interceptions', 'home_average_net_pass_yards', 'home_average_pass_attempts', 'home_average_pass_completions', 'home_average_pass_touchdowns', 'home_average_pass_yards', 'home_average_penalties', 'home_average_points', 'home_average_rush_attempts', 'home_average_rush_touchdowns', 'home_average_rush_yards', 'home_average_third_down_attempts', 'home_average_third_down_conversions', 'home_average_time_of_possession', 'home_average_times_sacked', 'home_average_total_yards', 'home_average_turnovers', 'home_average_yards_from_penalties', 'home_average_yards_lost_from_sacks', 'home_average_duration', 'home_average_roof', 'home_average_surface', 'home_average_time', 'home_average_temperature', 'home_average_humidity', 'home_average_wind', 'home_average_week', 'home_average_win'
#'roof', 'surface', 'time', 'temperature', 'humidity', 'wind'


#takes data from games and calculates the averages for the games before that game for each team in that 
#season+3games from the previous season, taking into account
#home vs away, and also takes the stats about the game that we could know about the game before it happens like
#the weather and location
#@param data2014 data for the previous year, gotten from cleandata
#@param data2015 data for the year in question, gotten from cleandata
#@param games The raw data for the year in question
#@return x,y where x has the columns listed above and y has 1 for home team win and 0 for away team win.
def getTraining(data2014,data2015,games):
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



        averagesA=[]+data2014[teams.index(awayabbr)][1][-3]
        for i in range(len(averagesA)):
            averagesA[i]+=data2014[teams.index(awayabbr)][1][-2][i]+data2014[teams.index(awayabbr)][1][-1][i]
        for i in range(counters[teams.index(awayabbr)][1]):
            for j in range(len(data2015[teams.index(awayabbr)][1][i])):
                averagesA[i]+=data2015[teams.index(awayabbr)][1][i][j]
        for i in range(len(averagesA)):
            averagesA[i] = averagesA[i]/float(counters[teams.index(awayabbr)][1]+3)



        averagesH=[]+data2014[teams.index(homeabbr)][0][-3]
        for i in range(len(averagesH)):
            averagesH[i]+=data2014[teams.index(homeabbr)][0][-2][i]+data2014[teams.index(homeabbr)][0][-1][i]
        for i in range(counters[teams.index(homeabbr)][0]):
            for j in range(len(data2015[teams.index(homeabbr)][0][i])):
                averagesH[i]+=data2015[teams.index(homeabbr)][0][i][j]
        for i in range(len(averagesH)):
            averagesH[i] = averagesH[i]/float(counters[teams.index(homeabbr)][0]+3)


        Ai=teams.index(awayabbr)
        Hi=teams.index(homeabbr)
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
        xtraintemp,ytraintemp=getTraining(cleandata(np.array(data201[i-1])),cleandata(np.array(data201[i])),np.array(data201[i]))
        xtraining+=xtraintemp
        ytraining+=ytraintemp
    except:
        print(i)
        
    try:
        xtraintemp,ytraintemp=getTraining(cleandata(np.array(data200[i-1])),cleandata(np.array(data200[i])),np.array(data200[i]))
        xtraining+=xtraintemp
        ytraining+=ytraintemp
    except:
        print(i)
