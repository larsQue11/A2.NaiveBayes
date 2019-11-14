# Course: CS7267
# Student name: William Stone
# Student ID: 000-272-306
# Assignment #: 2
# Due Date: October 2, 2019
# Signature:
# Score:

import numpy as np
from math import sqrt
from math import pi
from math import exp
from statistics import variance as var
import csv
from prettytable import PrettyTable

#retrieve data from CSV file
def getData(dataPath):

    with open(dataPath) as dataFile:
        data = np.array(list(csv.reader(dataFile)))

    return data

def PriorProbability(data,n):

    return n / len(data)
    

def CalculateAverages(data,classification):
    
    count = 0
    averages = [0,0,0,0]
    for scale in data:
        if scale.classification == classification:
            count = count + 1
            averages[0] = averages[0] + scale.left_weight
            averages[1] = averages[1] + scale.left_distance
            averages[2] = averages[2] + scale.right_weight
            averages[3] = averages[3] + scale.right_distance

    for i in range(len(averages)):
        averages[i] = averages[i] / count

    return count, averages
    
def CalculateVariances(data,classification,averages,count):

    variancesLW = []
    variancesLD = []
    variancesRW = []
    variancesRD = []
    for scale in data:
        if scale.classification == classification:
            variancesLW.append(scale.left_weight)
            variancesLD.append(scale.left_distance)
            variancesRW.append(scale.right_weight)
            variancesRD.append(scale.right_distance)

    return [var(variancesLW),var(variancesLD),var(variancesRW),var(variancesRD)]

def GaussianProbability(x,mean,variance):

    a = 1 / sqrt(2 * pi * variance)
    b = (-0.5 * (x - mean) ** 2) / variance

    return a * exp(b)

#a function that returns the index of the row for the corresponding classification being tested
def UpdateTable(classification):

    if classification == "Left":
        dim = 0
    elif classification == "Balanced":
        dim = 1
    else:
        dim = 2

    return dim

#a custom object to represent the scale's data values
class Scale():

    def __init__(self, scaleData):
        if scaleData[0] == 'L':
            self.classification = 'Left'
        elif scaleData[0] == 'B':
            self.classification = 'Balanced'
        elif scaleData[0] == 'R':
            self.classification = 'Right'
        self.left_weight = float (scaleData[1])
        self.left_distance = float (scaleData[2])
        self.right_weight = float (scaleData[3])
        self.right_distance = float (scaleData[4])
   

def main():
    dataPath = './data/balance_dataset.csv'
    retrievedData = getData(dataPath)
    
    #create lists of Scale objects: half for training, half for testing
    trainingSet = []
    testingSet = []
    count = 0
    for scale in retrievedData:
        myScale = Scale(scale)
        if count % 2 == 0:
            trainingSet.append(myScale)
        else:
            testingSet.append(myScale)
        count = count + 1


    #calculate averages
    numLefts, leftAverages = CalculateAverages(trainingSet,'Left')
    numBalanced, balancedAverages = CalculateAverages(trainingSet,'Balanced')
    numRights, rightAverages = CalculateAverages(trainingSet,'Right')

    leftVariances = CalculateVariances(trainingSet,'Left',leftAverages,numLefts)
    balancedVariances = CalculateVariances(trainingSet,'Balanced',balancedAverages,numBalanced)
    rightVariances = CalculateVariances(trainingSet,'Right',rightAverages,numRights)
    
    classProbabilities = [PriorProbability(trainingSet,numLefts), PriorProbability(trainingSet,numBalanced), PriorProbability(trainingSet,numRights)]

     #build confusion matrix
    '''
    +------------+---------+------------+-----------+
    |            | Left    | Balanced   | Right     |
    +------------+---------+------------+-----------+
    | Left       |  TP(S)  |   E(S,Ve)  |  E(S,Vi)  |
    | Balanced   | E(Ve,S) |   TP(Ve)   |  E(Ve,Vi) |
    | Right      | E(Vi,S) |   E(Vi,Ve) |   TP(Vi)  |
    +------------|---------+------------+-----------|
    '''
    confusionMatrix = np.zeros(shape=(3,3)) #creates a 2D array (3x3 table) containing all zeros

    for scale in trainingSet:

        data = [scale.left_weight,scale.left_distance,scale.right_weight,scale.right_distance]
        probabilityLeft = 1
        probabilityBalanced = 1
        probabilityRight = 1
        for i in range(len(data)):
            probabilityLeft = probabilityLeft * GaussianProbability(data[i],leftAverages[i],leftVariances[i])
            probabilityBalanced = probabilityBalanced * GaussianProbability(data[i],balancedAverages[i],balancedVariances[i])
            probabilityRight = probabilityRight * GaussianProbability(data[i],rightAverages[i],rightVariances[i])
        
        probabilityOfClasses = [(probabilityLeft, classProbabilities[0]), (probabilityBalanced, classProbabilities[1]), (probabilityRight, classProbabilities[2])]
        
        sumOfProbabilities = 0
        for x in probabilityOfClasses:
            sumOfProbabilities = sumOfProbabilities + (x[0] * x[1])

        leftProbability = (probabilityOfClasses[0][0] * probabilityOfClasses[0][1]) / sumOfProbabilities
        balancedProbability = (probabilityOfClasses[1][0] * probabilityOfClasses[1][1]) / sumOfProbabilities
        rightProbability = (probabilityOfClasses[2][0] * probabilityOfClasses[2][1]) / sumOfProbabilities

        mostProbableClass = leftProbability
        prediction = 'Left'
        if balancedProbability > mostProbableClass:
            mostProbableClass = balancedProbability
            prediction = 'Balanced'
        if rightProbability > mostProbableClass:
            mostProbableClass = rightProbability
            prediction = 'Right'

        row = UpdateTable(scale.classification)
        col = UpdateTable(prediction)
        confusionMatrix[row,col] = confusionMatrix[row,col] + 1

    #TP: the cell where the actual and predicted values intersect
    #TN: sum of all cells not in the classifier's row or column
    #FP: sum of all cells in the current column excluding TP
    #FN: sum of all cells in the current row excluding TP

    # #calculate overall accuracy of model: (TP+TN)/(TP+TN+FP+FN)
    accuracy = (confusionMatrix[0,0] + confusionMatrix[1,1] + confusionMatrix[2,2]) / np.sum(confusionMatrix)

    # #calculate precision: TP/(TP+FP)
    precisionLeft = confusionMatrix[0,0] / np.sum(confusionMatrix,axis=0)[0]
    precisionBalanced = confusionMatrix[1,1] / np.sum(confusionMatrix,axis=0)[1]
    precisionRight = confusionMatrix[2,2] / np.sum(confusionMatrix,axis=0)[2]

    meanTable = PrettyTable()
    meanTable.field_names = ['classification', 'left_weight', 'left_distance', 'right_weight', 'right_distance']
    meanTable.add_row(['Left', leftAverages[0], leftAverages[1], leftAverages[2], leftAverages[3]])
    meanTable.add_row(['Balanced', balancedAverages[0], balancedAverages[1], balancedAverages[2], balancedAverages[3]])
    meanTable.add_row(['Right', rightAverages[0], rightAverages[1], rightAverages[2], rightAverages[3]])
    print("Average values for Scale features")
    print(meanTable)

    varTable = PrettyTable()
    varTable.field_names = ['classification', 'left_weight', 'left_distance', 'right_weight', 'right_distance']
    varTable.add_row(['Left', leftVariances[0], leftVariances[1], leftVariances[2], leftVariances[3]])
    varTable.add_row(['Balanced', balancedVariances[0], balancedVariances[1], balancedVariances[2], balancedVariances[3]])
    varTable.add_row(['Right', rightVariances[0], rightVariances[1], rightVariances[2], rightVariances[3]])
    print("\nVariance values for Scale features")
    print(varTable)

    matrixTable = PrettyTable()
    matrixTable.field_names = ['', 'Left', 'Balanced', 'Right']
    matrixTable.add_row(['Left', confusionMatrix[0,0], confusionMatrix[0,1], confusionMatrix[0,2]])
    matrixTable.add_row(['Balanced', confusionMatrix[1,0], confusionMatrix[1,1], confusionMatrix[1,2]])
    matrixTable.add_row(['Right', confusionMatrix[2,0], confusionMatrix[2,1], confusionMatrix[2,2]])
    print("\nConfusion Matrix")
    print(matrixTable)

    print(f"\nAccuracy of model = {accuracy}")
    print(f"Precision of Left = {precisionLeft}")
    print(f"Precision of Balanced = {precisionBalanced}")
    print(f"Precision of Right = {precisionRight}")


if __name__ ==  '__main__':
    main()