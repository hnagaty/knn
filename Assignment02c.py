# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 10:02:53 2017

@author: hnagaty

Assignment #2, ML
KNN Means
"""

import numpy as np
#import copy
from PIL import Image
import string
#import scipy.spatial as sp
import random
import math
import matplotlib.pyplot as plt


mypath='C:\\Users\\hnaga\\Dropbox\\Online Courses\\Nu, Machine Learning\\Assigments\\Assignment02\\Problem 2 Dataset\\Noise Train\\'

#read the 182 letters into a numpy array
allVec=np.zeros([1,144],dtype=int)
oneVec=np.zeros([1,144],dtype=int)
labVec=np.zeros([1])
for i in range(97,123,1): # loop through the asci charaters from "a" to "z"
  for j in range(1,8,1): # loop for 7 variants in each letter
    loc=chr(i)+str(j)
    oneImage = Image.open(mypath+"A1"+loc+".jpg")
    oneVec=np.array(oneImage).flatten().reshape((1,144))
    allVec=np.concatenate((allVec,oneVec))
    labVec=np.concatenate((labVec,[chr(i)]))
allVec=np.delete(allVec,0,axis=0) #2d numpy array of the 182 letters
labVec=np.delete(labVec,0) #numpy array for the labels 


def euclideanDist (vec1,vec2,v1,v2):
    diff=vec1[v1]-vec2[v2]
    dist=math.sqrt(diff@diff.T)
    return dist

# The distance matrix of the 182 letters. 
distMatrix=np.zeros((182,182))
for i in range (182):
    for j in range(182):
        distMatrix[i,j]=euclideanDist(allVec,allVec,i,j)

#sortedMatIndex=np.argsort(distMatrix,axis=1)
#chrnp=np.vectorize(chr) # vectorize the chr() function, to be able to run it over an array
#sortedMatByLabels=chrnp(sortedMatIndex//7+97) # sorted distance matrix, by label

def maxOccur (lst):
    mydict   = {}
    cnt, itm = 0, ''
    for item in reversed(lst):
        mydict[item] = mydict.get(item, 0) + 1
        if mydict[item] >= cnt :
            cnt, itm = mydict[item], item
    return itm

def KNearestClass (matrix,k,v):
    lst=matrix[v,:k]
    return (maxOccur(lst))

KMatrix=np.zeros([100],dtype=int)

for g in range(10):
    allIndices=np.arange(182)
    random.shuffle(allIndices)
    vCnt=int(182*20/100)
    trainingInd=allIndices[vCnt:]
    validationInd=allIndices[:vCnt]
    validationLab=labVec[validationInd]
    trainingLab=labVec[trainingInd]
    distMatReduced=distMatrix[validationInd][:,trainingInd]
    sortedMatReducedIndex=np.argsort(distMatReduced,axis=1)
    sortedRedMatByLabel=trainingLab[sortedMatReducedIndex] # I can't beleive this worked as needed
    for k in range(1,101):
        for i in range(36):
            if KNearestClass (sortedRedMatByLabel,k,i) != validationLab[i]:  KMatrix[k-1]+=1

# Plot the K Erros
x=range(1,101)
KError=KMatrix/(36*10)
plt.bar(x,KError,align="center")
plt.xlabel("K Value",color="green")
plt.ylabel("Error",color="green")
plt.title("Error vs different K values",fontsize=16,color="blue")
plt.show()


# The K with least error is
k=np.argmin(KMatrix)+1


# read test data
mypath='C:\\Users\\hnaga\\Dropbox\\Online Courses\\Nu, Machine Learning\\Assigments\\Assignment02\\Problem 2 Dataset\\Noise Test\\'
allTestVec=np.zeros([1,144],dtype=int)
oneTestVec=np.zeros([1,144],dtype=int)
labTestVec=np.zeros([1])
for i in range(97,123,1): # loop through the asci charaters from "a" to "z"
  for j in range(8,10,1): # loop for 7 variants in each letter
    loc=chr(i)+str(j)
    oneImage = Image.open(mypath+"A1"+loc+".jpg")
    oneTestVec=np.array(oneImage).flatten().reshape((1,144))
    allTestVec=np.concatenate((allTestVec,oneTestVec))
    labTestVec=np.concatenate((labTestVec,[chr(i)]))
allTestVec=np.delete(allTestVec,0,axis=0) #2d numpy array of the 182 letters
labTestVec=np.delete(labTestVec,0) #numpy array for the labels

distMatrixTest=np.zeros((52,182))
for i in range (52):
    for j in range(182):
        distMatrixTest[i,j]=euclideanDist(allTestVec,allVec,i,j)
sortedDistMatTestIndex=np.argsort(distMatrixTest,axis=1)
sortedDistMatTestByLabel=labVec[sortedDistMatTestIndex]


testResult=np.zeros([26],dtype=int)
for i in range(0,52,2):
    for j in range(2):
        if labTestVec[i+j]==KNearestClass(sortedDistMatTestByLabel,k,i+j): testResult[int(i/2)]+=1

x=range(26)
xlab=list(string.ascii_lowercase)
plt.bar(x, testResult,align="center")
plt.xticks(x, xlab)
plt.yticks([0,1,2])
plt.show()
        
        
