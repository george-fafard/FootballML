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
from sklearn.metrics         import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


from sklearn import preprocessing as p



#helper function to get the data
def test_data(y1,y2):
    gamedata = cd.read_game_data_from_files(y1,y2)
    
    
    
    x=[]
    y=[]
    for i in range(len(gamedata)-2):
        xtemp,ytemp=cd.get_training(cd.clean_data(np.array(gamedata[i])),cd.clean_data(np.array(gamedata[i+1])),np.array(gamedata[i+1]),y1+i)
        x+=xtemp
        y+=ytemp
    lastx,lasty=cd.get_training(cd.clean_data(np.array(gamedata[-2])),cd.clean_data(np.array(gamedata[-1])),np.array(gamedata[-1]),y2)
    return x,y,lastx,lasty


#helper function to prep the data for the neural network
def prep_data(x,y,lastx,lasty):
    xtrain=x
    ytrain=y
    
    xvalid=lastx
    xtest=lastx
    
    yvalid=lasty
    ytest=lasty
    
    ytest_1hot=np.zeros((len(ytest),2))
    for i in range(len(ytest)):
        ytest_1hot[i][ytest[i]]=1

    ytrain_1hot=np.zeros((len(ytrain),2))
    for i in range(len(ytrain)):
        ytrain_1hot[i][ytrain[i]]=1

    yvalid_1hot=np.zeros((len(yvalid),2))
    for i in range(len(yvalid)):
        yvalid_1hot[i][yvalid[i]]=1
        
    
    
    xtest=np.array(xtest)
    xtrain=np.array(xtrain)
    xvalid=np.array(xvalid)
    
    
    scaler = p.MinMaxScaler()
    
    scaler.fit(xtrain)
    xtest=scaler.transform(xtest)
    xtrain=scaler.transform(xtrain)
    xvalid=scaler.transform(xvalid)
    
    xtest,xvalid,ytest_1hot,yvalid_1hot=train_test_split(xtest,ytest_1hot, test_size=0.5)
    
    return xtrain, xvalid, xtest, ytrain_1hot, yvalid_1hot, ytest_1hot

#runs the neural network for 100 epochs, the number of epochs that results in the best accuracy
#param : year the year we are predicting for.
def run_neural_network(year):
    
    #hyperparams for model
    l=0.0001
    b=128
    d=0.5
    s=128

    #model
    model = Sequential(
        [

            layers.Dense(2*s),
            layers.LeakyReLU(alpha=0.25),

            layers.Dropout(d),

            layers.Dense(s, activation='relu'),

            layers.Dense(2, activation='softmax')
        ]
    )

    model.compile(optimizer=optimizers.Adam(learning_rate=l),
                          loss='categorical_crossentropy', 
                          metrics=['accuracy'])

    x,y,xl,yl = test_data(year-9,year) 
    xtrain, xvalid, xtest, ytrain, yvalid, ytest = prep_data(x,y,xl,yl)
    
    results=model.fit(xtrain, ytrain, validation_data=(xvalid,yvalid), batch_size=b, epochs=100, verbose=0)
    '''
    plt.plot(results.history['accuracy'])
    plt.plot(results.history['val_accuracy'])
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()

    test_results=model.evaluate(xtest,ytest)
    print('Test accuracy:',test_results[1])
    '''
    return model
    
#function to evaluate the model on predicting some set that is passed in
#params: x:np array of data equivalent to what would be gotten from get_training
def predict(x,model):
    
    y_pred=model.predict(x)
    
    ynew=[]
            
    for i in y_pred:
        ynew.append(i[1])
    
    
    return np.array(ynew)