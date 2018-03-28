import sys
sys.path.append("F:/pythonTest")
from svm_optimization.svmMLiA import *

dataArr,labelArr = loadDataSet('testSet.txt')
#print(labelArr)
b,alphas = smoP(dataArr, labelArr, 0.6, 0.001, 40)
#print(b)
#print(alphas[alphas>0])
#print(shape(alphas[alphas>0]))
#for i in range(100):
#    if alphas[i]>0.0:
#        print(dataArr[i],labelArr[i])

ws = calcWs(alphas, dataArr, labelArr)
#print(ws)

#datMat = mat(dataArr)
#print(datMat[2]*mat(ws)+b)
#print(labelArr[2])

testDigits(('rbf', 20))