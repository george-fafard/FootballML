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

l=0.0001
b=128
d=0.5
s=128

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

# Get test data to import in the notebook
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

def prep_data(x,y,lastx,lasty):
    '''
    newx=[]
    newlx=[]
    for t in x:
        newx.append([t[0]]+[t[1]]+[t[4]]+[t[6]]+[t[7]]+[t[12]]+[t[13]]+[t[16]]+[t[19]]+[t[21]]+[t[22]]+t[26:36]+[t[36+0]]+[t[36+1]]+[t[36+4]]+[t[36+6]]+[t[36+7]]+[t[36+12]]+[t[36+13]]+[t[36+16]]+[t[36+19]]+[t[36+21]]+[t[36+22]]+t[36+26:36+36]+t[-6:-1]+[t[-1]])
    for t in lastx:
        newlx.append([t[0]]+[t[1]]+[t[4]]+[t[6]]+[t[7]]+[t[12]]+[t[13]]+[t[16]]+[t[19]]+[t[21]]+[t[22]]+t[26:36]+[t[36+0]]+[t[36+1]]+[t[36+4]]+[t[36+6]]+[t[36+7]]+[t[36+12]]+[t[36+13]]+[t[36+16]]+[t[36+19]]+[t[36+21]]+[t[36+22]]+t[36+26:36+36]+t[-6:-1]+[t[-1]])
    
    #xtrain,xtestA,ytrain,ytestB= train_test_split(x,y, test_size=0.3)
    #xtest,xvalid,ytest,yvalid=train_test_split(xtestA,ytestB, test_size=0.75)
    '''
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
    #scaler = p.QuantileTransformer()
    #scaler = p.PowerTransformer()
    #scaler = p.StandardScaler()
    
    scaler.fit(xtrain)
    xtest=scaler.transform(xtest)
    xtrain=scaler.transform(xtrain)
    xvalid=scaler.transform(xvalid)
    
    xtest,xvalid,ytest_1hot,yvalid_1hot=train_test_split(xtest,ytest_1hot, test_size=0.5)
    
    return xtrain, xvalid, xtest, ytrain_1hot, yvalid_1hot, ytest_1hot

def reset():
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
    

def run_neural_network(year):
    reset()
    
    
    
    x,y,xl,yl = test_data(year-9,year) 
    xtrain, xvalid, xtest, ytrain, yvalid, ytest = prep_data(x,y,xl,yl)
    
    results=model.fit(xtrain, ytrain, validation_data=(xvalid,yvalid), batch_size=b, epochs=100, verbose=0)
    
    plt.plot(results.history['accuracy'])
    plt.plot(results.history['val_accuracy'])
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()

    test_results=model.evaluate(xtest,ytest)
    print('Test accuracy:',test_results[1])
    
    
    
    
def predict(x,y):
    #x,trasha,trashb,y,trashc,trashd = prep_data(x,y,x,y)
    
    y_pred=model.predict(x)
    
    yold=[]
    ynew=[]
    for i in y:
        if i[0]>i[1]:
            yold.append([0])
        else:
            yold.append([1])
            
    for i in y_pred:
        if i[0]>i[1]:
            ynew.append([0])
        else:
            ynew.append([1])
    
    conf_matrix = confusion_matrix(yold, ynew)
    
    print(conf_matrix)