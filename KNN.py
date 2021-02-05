from numpy import *
import numpy as np
import operator
import matplotlib
import matplotlib.pyplot as plt
from os import listdir
import os

#创建数据函数
def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

'''
    tile函数，tile([1,2],(x,y))
    表示[1,2]在行方向上重复x次，列方向上重复y次数，
    结果为二维数组
    当第二个参数只有一个const时候，表示为列重复，默认行重复次数为1，
    结果为一维数组
    第一个参数复制，第二个参数拉长
'''
#一个简单的分类器,实质就是矩阵距离计算并排序
def classify0(inX, dataSet, labels, k):
    #返回dataSet的行数，列数。返回值格式为元组：(行，列)
    #此处表示dataSet的行数
    dataSetSize = dataSet.shape[0]
    #计算距离
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis = 1)
    distances = sqDistance ** 0.5
    #选择距离最小的k个点
    sortedDisIndicies = distances.argsort()
    #print(type(sortedDisIndicies))
    classCount = {}
    #原来classCount是空的，经过循环可以将label加入到dict的索引值中
    for i in range(k):
        voteIlabel = labels[sortedDisIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    #排序，将classCount字典分解为元组
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    
    return sortedClassCount[0][0]

#将数据库中的数据转化为可识别处理的数据
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

#归一化，平均权重
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

#测试分类器
def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m],datingLabels[numTestVecs:m],3)
        print('The classifier came back with: %d, the real answer is %d'%(classifierResult, datingLabels[i]))
        if(classifierResult != datingLabels[i]):
            errorCount += 1
    print('The total error rate is %f' %(errorCount / float(numTestVecs)))

#使用分类器
def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of icecream comsumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print("You will probably like this person: ",resultList[classifierResult - 1])

#将数据库中的数字转化为向量形式1*1024向量
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect

#手写识别
def handwritingClassTest():
    hwLabels = []
    currentFilePath = os.path.abspath(__file__)
    fatherFilePath = os.path.abspath(os.path.dirname(currentFilePath)+os.path.sep+".")
    trainingFileList = listdir(r'{}\trainingDigits'.format(fatherFilePath))
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileNameStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector(r'{}\trainingDigits\{}'.format(fatherFilePath, fileNameStr))
    testFileList = listdir(r'{}\testDigits'.format(fatherFilePath))
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector(r'{}\testDigits\{}'.format(fatherFilePath, fileNameStr))
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
        #print('The classifier came back with: %d' %classifierResult, classNumStr)
        if (classifierResult != classNumStr):
            errorCount += 1.0
    print('\n The total number of errors is %d '%errorCount)
    print('\n The total error rate is: %f' % (errorCount / float(mTest)))


if __name__ == "__main__":
    # datingDataMat, datingLabels = file2matrix(r'D:\engeering lib\python\Machine_learningFight\123.txt')
    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1)
    # ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 1.0*array(datingLabels), 1.0*array(datingLabels))
    # plt.show()
    handwritingClassTest()

