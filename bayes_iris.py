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

def PriorProbability(data,numOfSpecies):

    return numOfSpecies / len(data)
    

def CalculateAverages(data,species):
    
    count = 0
    averages = [0,0,0,0]
    for iris in data:
        if iris.species == species:
            count = count + 1
            averages[0] = averages[0] + iris.sepal_length
            averages[1] = averages[1] + iris.sepal_width
            averages[2] = averages[2] + iris.petal_length
            averages[3] = averages[3] + iris.petal_width

    for i in range(len(averages)):
        averages[i] = averages[i] / count

    return count, averages
    
def CalculateVariances(data,species,averages,count):

    variancesSL = []
    variancesSW = []
    variancesPL = []
    variancesPW = []
    for iris in data:
        if iris.species == species:
            variancesSL.append(iris.sepal_length)
            variancesSW.append(iris.sepal_width)
            variancesPL.append(iris.petal_length)
            variancesPW.append(iris.petal_width)

    return [var(variancesSL),var(variancesSW),var(variancesPL),var(variancesPW)]

def GaussianProbability(x,mean,variance):

    a = 1 / sqrt(2 * pi * variance)
    b = (-0.5 * (x - mean) ** 2) / variance

    return a * exp(b)

#a function that returns the index of the row for the corresponding species being tested
def UpdateTable(species):

    if species == "Setosa":
        dim = 0
    elif species == "Versicolor":
        dim = 1
    else:
        dim = 2

    return dim

#a custom object to represent the Iris's data values
class Iris():

    def __init__(self, irisData):
        self.sepal_length = float (irisData[0])
        self.sepal_width = float (irisData[1])
        self.petal_length = float (irisData[2])
        self.petal_width = float (irisData[3])
        if len(irisData) > 4:
            if irisData[4] == 'Iris-setosa':
                self.species = 'Setosa'
            elif irisData[4] == 'Iris-versicolor':
                self.species = 'Versicolor'
            elif irisData[4] == 'Iris-virginica':
                self.species = 'Virginica'
        else:
            pass

def main():
    dataPath = './data/iris_dataset.csv'
    retrievedData = getData(dataPath)
    
    #create lists of Iris objects: half for training, half for testing
    trainingSet = []
    testingSet = []
    count = 0
    for iris in retrievedData:
        myIris = Iris(iris)
        if count % 2 == 0:
            trainingSet.append(myIris)
        else:
            testingSet.append(myIris)
        count = count + 1

    #calculate averages
    numSetosas, setosaAverages = CalculateAverages(trainingSet,'Setosa')
    numVersicolors, versicolorAverages = CalculateAverages(trainingSet,'Versicolor')
    numVirginicas, virginicaAverages = CalculateAverages(trainingSet,'Virginica')

    setosaVariances = CalculateVariances(trainingSet,'Setosa',setosaAverages,numSetosas)
    versicolorVariances = CalculateVariances(trainingSet,'Versicolor',versicolorAverages,numVersicolors)
    virginicaVariances = CalculateVariances(trainingSet,'Virginica',virginicaAverages,numVirginicas)
    
    classProbabilities = [PriorProbability(trainingSet,numSetosas), PriorProbability(trainingSet,numVersicolors), PriorProbability(trainingSet,numVirginicas)]

     #build confusion matrix
    '''
    +------------+---------+------------+-----------+
    |            | Setosa  | Versicolor | Virginica |
    +------------+---------+------------+-----------+
    | Setosa     |  TP(S)  |   E(S,Ve)  |  E(S,Vi)  |
    | Versicolor | E(Ve,S) |   TP(Ve)   |  E(Ve,Vi) |
    | Virginica  | E(Vi,S) |   E(Vi,Ve) |   TP(Vi)  |
    +------------|---------+------------+-----------|
    '''
    confusionMatrix = np.zeros(shape=(3,3)) #creates a 2D array (3x3 table) containing all zeros

    for iris in trainingSet:

        data = [iris.sepal_length,iris.sepal_width,iris.petal_length,iris.petal_width]
        probabilitySetosa = 1
        probabilityVersicolor = 1
        probabilityVirginica = 1
        for i in range(len(data)):
            probabilitySetosa = probabilitySetosa * GaussianProbability(data[i],setosaAverages[i],setosaVariances[i])
            probabilityVersicolor = probabilityVersicolor * GaussianProbability(data[i],versicolorAverages[i],versicolorVariances[i])
            probabilityVirginica = probabilityVirginica * GaussianProbability(data[i],virginicaAverages[i],virginicaVariances[i])
        
        probabilityOfClasses = [(probabilitySetosa, classProbabilities[0]), (probabilityVersicolor, classProbabilities[1]), (probabilityVirginica, classProbabilities[2])]
        
        sumOfProbabilities = 0
        for x in probabilityOfClasses:
            sumOfProbabilities = sumOfProbabilities + (x[0] * x[1])

        setosaProbability = (probabilityOfClasses[0][0] * probabilityOfClasses[0][1]) / sumOfProbabilities
        versicolorProbability = (probabilityOfClasses[1][0] * probabilityOfClasses[1][1]) / sumOfProbabilities
        virginicaProbability = (probabilityOfClasses[2][0] * probabilityOfClasses[2][1]) / sumOfProbabilities

        mostProbableClass = setosaProbability
        prediction = 'Setosa'
        if versicolorProbability > mostProbableClass:
            mostProbableClass = versicolorProbability
            prediction = 'Versicolor'
        if virginicaProbability > mostProbableClass:
            mostProbableClass = virginicaProbability
            prediction = 'Virginica'

        row = UpdateTable(iris.species)
        col = UpdateTable(prediction)
        confusionMatrix[row,col] = confusionMatrix[row,col] + 1

    #TP: the cell where the actual and predicted values intersect
    #TN: sum of all cells not in the classifier's row or column
    #FP: sum of all cells in the current column excluding TP
    #FN: sum of all cells in the current row excluding TP

    # #calculate overall accuracy of model: (TP+TN)/(TP+TN+FP+FN)
    accuracy = (confusionMatrix[0,0] + confusionMatrix[1,1] + confusionMatrix[2,2]) / np.sum(confusionMatrix)

    # #calculate precision: TP/(TP+FP)
    precisionSetosa = confusionMatrix[0,0] / np.sum(confusionMatrix,axis=0)[0]
    precisionVersicolor = confusionMatrix[1,1] / np.sum(confusionMatrix,axis=0)[1]
    precisionVirginica = confusionMatrix[2,2] / np.sum(confusionMatrix,axis=0)[2]

    meanTable = PrettyTable()
    meanTable.field_names = ['Species', 'sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    meanTable.add_row(['Setosa', setosaAverages[0], setosaAverages[1], setosaAverages[2], setosaAverages[3]])
    meanTable.add_row(['Versicolor', versicolorAverages[0], versicolorAverages[1], versicolorAverages[2], versicolorAverages[3]])
    meanTable.add_row(['Virginica', virginicaAverages[0], virginicaAverages[1], virginicaAverages[2], virginicaAverages[3]])
    print("Average values for Iris features")
    print(meanTable)

    varTable = PrettyTable()
    varTable.field_names = ['Species', 'sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    varTable.add_row(['Setosa', setosaVariances[0], setosaVariances[1], setosaVariances[2], setosaVariances[3]])
    varTable.add_row(['Versicolor', versicolorVariances[0], versicolorVariances[1], versicolorVariances[2], versicolorVariances[3]])
    varTable.add_row(['Virginica', virginicaVariances[0], virginicaVariances[1], virginicaVariances[2], virginicaVariances[3]])
    print("\nVariance values for Iris features")
    print(varTable)

    matrixTable = PrettyTable()
    matrixTable.field_names = ['', 'Setosa', 'Versicolor', 'Virginica']
    matrixTable.add_row(['Setosa', confusionMatrix[0,0], confusionMatrix[0,1], confusionMatrix[0,2]])
    matrixTable.add_row(['Versicolor', confusionMatrix[1,0], confusionMatrix[1,1], confusionMatrix[1,2]])
    matrixTable.add_row(['Virginica', confusionMatrix[2,0], confusionMatrix[2,1], confusionMatrix[2,2]])
    print("\nConfusion Matrix")
    print(matrixTable)

    print(f"\nAccuracy of model = {accuracy}")
    print(f"Precision of Setosa = {precisionSetosa}")
    print(f"Precision of Versicolor = {precisionVersicolor}")
    print(f"Precision of Virginica = {precisionVirginica}")


if __name__ ==  '__main__':
    main()