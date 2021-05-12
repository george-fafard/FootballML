""" 
    This will be where the actual code for the neural network classifier goes
"""
# THIS IS AN EXAMPLE YOU CAN DELETE THIS WHEN YOU START WORKING ON IT.
import FootballML.Dataset.cleaned_data as cd
import numpy as np

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential 
from tensorflow.keras import layers
from tensorflow.keras import optimizers

# Helper libraries
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Get test data to import in the notebook
def test_data(y1,y2):
    gamedata = cd.read_game_data_from_files(y1,y2)
    
    
    
    x=[]
    y=[]
    for i in range(len(gamedata)-1):
        xtemp,ytemp=cd.get_training(cd.clean_data(np.array(gamedata[i])),cd.clean_data(np.array(gamedata[i+1])),np.array(gamedata[i+1]),2000+i)
        x+=xtemp
        y+=ytemp
    return x,y

def prep_data(x,y):
    
    newx=[]
    for t in x:
        newx.append([t[0]]+[t[1]]+[t[4]]+[t[6]]+[t[7]]+[t[12]]+[t[13]]+[t[16]]+[t[19]]+[t[21]]+[t[22]]+t[26:36]+[t[36+0]]+[t[36+1]]+[t[36+4]]+[t[36+6]]+[t[36+7]]+[t[36+12]]+[t[36+13]]+[t[36+16]]+[t[36+19]]+[t[36+21]]+[t[36+22]]+t[36+26:36+36]+t[-6:-1]+[t[-1]])
    
    xtrain,xtestA,ytrain,ytestB= train_test_split(x,y, test_size=0.3)
    xtest,xvalid,ytest,yvalid=train_test_split(xtestA,ytestB, test_size=0.75)

    
    ytest_1hot=np.zeros((len(ytest),2))
    for i in range(len(ytest)):
        ytest_1hot[i][ytest[i]]=1

    ytrain_1hot=np.zeros((len(ytrain),2))
    for i in range(len(ytrain)):
        ytrain_1hot[i][ytrain[i]]=1

    yvalid_1hot=np.zeros((len(yvalid),2))
    for i in range(len(yvalid)):
        yvalid_1hot[i][yvalid[i]]=1
        
    
    
    averages = xtest[1]
    for g in xtrain:
        for i in range(len(g)):
            averages[i]=averages[i]+g[i]

    for g in xvalid:
        for i in range(len(g)):
            averages[i]+=g[i]

    for g in xtest:
        for i in range(len(g)):
            averages[i]+=g[i]

    for i in range(len(averages)):
            averages[i]/=(len(xtrain)+len(xtest)+len(xvalid)+1)

    for g in xtrain:
        for i in range(len(g)):
            g[i]/=averages[i]  

    for g in xtest:
        for i in range(len(g)):
            g[i]/=averages[i]   

    for g in xvalid:
        for i in range(len(g)):
            g[i]/=averages[i]
            
    
    
    
            
    xtest=np.array(xtest)
    xtrain=np.array(xtrain)
    xvalid=np.array(xvalid)
    '''
    xtrain = np.vstack((xtrain,xtrain*(np.ones(xtrain.shape)-(np.random.rand(xtrain.shape[0],xtrain.shape[1])-np.ones(xtrain.shape)/2)*0.001)))
    xtrain = np.vstack((xtrain,xtrain*(np.ones(xtrain.shape)-(np.random.rand(xtrain.shape[0],xtrain.shape[1])-np.ones(xtrain.shape)/2)*0.001)))
    ytrain_1hot = np.vstack((ytrain_1hot,ytrain_1hot))
    ytrain_1hot = np.vstack((ytrain_1hot,ytrain_1hot))
    '''
    return xtrain, xvalid, xtest, ytrain_1hot, yvalid_1hot, ytest_1hot