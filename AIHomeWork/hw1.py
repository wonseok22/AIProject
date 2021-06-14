import sys
import sklearn
import matplotlib.pyplot as plt
import numpy as np
from myKNN import myKNN
from sklearn.datasets import load_iris

iris = load_iris()


if __name__ == "__main__":

    # Learning Data Set , Test Data Set 구별작업
    # Index 순서 ['data number', 'data', 'target'] 으로 구분
    # X_train,y_train = trainData[1],trainData[2]
    # X_test,y_test = testData[1],trainData[2]

    trainData = [[],iris['data'][:14],iris['target'][:14]]
    testData = [[],[iris['data'][14]],[iris['target'][14]]]
    target_name = iris['target_names']
    for i in range(10):
        for j in range(15):
            if j != 14:
                trainData[0].append(i*15+j)
            else:
                testData[0].append((i*15+j))

    for i in range(1,10):
        trainData[2] = np.concatenate([trainData[2],iris['target'][i*15:i*15+14]])
        trainData[1] = np.concatenate([trainData[1],iris['data'][i*15:i*15+14]],axis = 0)
        testData[2] = np.concatenate([testData[2],[iris['target'][i*15+14]]])
        testData[1] = np.concatenate([testData[1],[iris['data'][i*15+14]]],axis = 0)


    # KNN 호출
    # Computed class : predict(X_test), True class : y_test 비교

    X_train,y_train = trainData[1],trainData[2]
    X_test,y_test = testData[1],testData[2]
    K = int(input("Please enter K"))
    classes = myKNN(K)
    classes.training(X_train,y_train)
    #predictClass = classes.weightedPredict(X_test)
    predictClass = classes.predict(X_test)

    predictName = []
    trueName = []
    print("Case K =",K)
    for i in range(10):
        predictName.append(target_name[predictClass[i]])
        trueName.append(target_name[y_test[i]])
        print("Test Data Index: ",i,"Computed Class:", target_name[predictClass[i]],", True class:",target_name[y_test[i]])




