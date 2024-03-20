# Format: 
# 1) Imports
# 2) Load Training Set and CSV Files
# 3) Train Model
# 4) Test Model
# 5) Create Kaggle Submission

############################## IMPORTS ##############################

from __future__ import division
import sklearn
import pandas as pd
import numpy as np
import collections
import os.path
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import sys
from sklearn.ensemble import GradientBoostingRegressor
import math
import csv
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report
from sklearn.calibration import CalibratedClassifierCV
import urllib
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from datetime import datetime
import random

############################## LOAD TRAINING SET ##############################

if os.path.exists("Data/PrecomputedMatrices/xTrain.npy") and os.path.exists("Data/PrecomputedMatrices/yTrain.npy"):
	xTrain = np.load("Data/PrecomputedMatrices/xTrain.npy")
	yTrain = np.load("Data/PrecomputedMatrices/yTrain.npy")
	print ("Shape of xTrain:", xTrain.shape)
	print ("Shape of yTrain:", yTrain.shape)
else:
	print ('We need a training set! Run dataPreprocessing.py')
	sys.exit()

# In case you want to run with Python 2
try:
    input = raw_input
except NameError:
    pass

curYear = int(input('What year are these predictions for?\n'))

############################## LOAD CSV FILES ##############################

sample_sub_pd = pd.read_csv('Data/KaggleData/SampleSubmissionStage1.csv')
sample_sub_pd2 = pd.read_csv('Data/KaggleData/SampleSubmissionStage2.csv')
teams_pd = pd.read_csv('Data/KaggleData/MTeams.csv')

############################## TEST MODEL ##############################

def predictGame(team_1_vector, team_2_vector, home, modelUsed):
    diff = [a - b for a, b in zip(team_1_vector, team_2_vector)]
    diff.append(home)
    if hasattr(modelUsed, 'predict_proba'):
	    return modelUsed.predict_proba([diff])[0][1]
    elif hasattr(modelUsed, 'predict'):
        return modelUsed.predict([diff])[0]
    else:
        raise AttributeError("Model does not have expected prediction method")

############################## PREDICTING THIS YEAR'S BRACKET ##############################

def trainModel():
	model = GradientBoostingRegressor(n_estimators=100, max_depth=5)
	model.fit(xTrain, yTrain)
	return model

def randomWinner(team1, team2, modelUsed, numTrials=10):
	year = [curYear]
	teamVectors = loadTeamVectors(year)[0]
	team1Vector = teamVectors[int(teams_pd[teams_pd['TeamName'] == team1].values[0][0])]
	team2Vector = teamVectors[int(teams_pd[teams_pd['TeamName'] == team2].values[0][0])]
	prediction = predictGame(team1Vector, team2Vector, 0, modelUsed)
	team1Wins = 0
	for i in range(numTrials):
		if (prediction > random.random()):
			team1Wins = team1Wins + 1
		print ("{0} Won {1} times".format(team1, team1Wins))


def findWinner(team1, team2, modelUsed):
	year = [curYear]
	teamVectors = loadTeamVectors(year)[0]
	team1Vector = teamVectors[int(teams_pd[teams_pd['TeamName'] == team1].values[0][0])]
	team2Vector = teamVectors[int(teams_pd[teams_pd['TeamName'] == team2].values[0][0])]
	prediction = predictGame(team1Vector, team2Vector, 0, modelUsed)
	if (prediction < 0.5):
		print ("Probability that {0} wins: {1}".format(team2, 1 - prediction))
	else:
		print ("Probability that {0} wins: {1}".format(team1, prediction))
	return prediction


############################## TRAIN MODEL ##############################

model = GradientBoostingRegressor(n_estimators=100, max_depth=5)

categories=['Wins','PPG','PPGA','PowerConf','3PG', 'APG','TOP','Conference Champ','Tourney Conference Champ',
           'Seed','SOS','SRS', 'RPG', 'SPG', 'Tourney Appearances','National Championships','Location']
accuracy=[]
numTrials = 1

for i in range(numTrials):
    X_train, X_test, Y_train, Y_test = train_test_split(xTrain, yTrain)
    startTime = datetime.now() # For some timing stuff
    trainedModel = trainModel()
    preds = trainedModel.predict(X_test)
    preds[preds < .5] = 0
    preds[preds >= .5] = 1
    localAccuracy = np.mean(preds == Y_test)
    accuracy.append(localAccuracy)
    print ("Finished run #" + str(i) + ". Accuracy = " + str(localAccuracy))
    print ("Time taken: " + str(datetime.now() - startTime))
if numTrials != 0:
	print ("The average accuracy is", sum(accuracy)/len(accuracy))



# First round games in the East for example
	
matchupFile = open('2024_Matchups.csv')
reader = csv.reader(matchupFile)

for row in reader:
	pred = findWinner(str(row[0]),str(row[1]),trainedModel)
	with open("output.txt", "a") as f:	
    		f.write(pred)
    		f.write("\n")
