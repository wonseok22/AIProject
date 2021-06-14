import csv
import numpy as np
import pandas as pd
from CNN import myCNN
from DNN import myDNN
from KNN import myKNN
from sklearn.model_selection import train_test_split


def FirstProcess():
    path = open('framingham.csv', 'r', encoding='utf-8')
    file = csv.reader(path)
    Train = []
    Test = []
    for idx, value in enumerate(file):
        if idx != 0:
            for i in range(len(value)):
                if value[i] == 'NA':
                    value[i] = 0
            Train.append(value[:15])
            Test.append(int(value[-1]))
    path.close()
    X_test, T_test, X_target, T_target = train_test_split(np.array(Train,dtype=float), np.array(Test,dtype=int), test_size=0.14)
    print('FirstProcessing으로 전처리한 Data')
    print('input feature의 체중, 몸무게를 BMI지수로, Day 단위의 age를 year_age로 변환\n')

    # CNN Architecture 를 기반으로 학습한 모델
    print('Running CNN...')
    CNN = myCNN(X_test, X_target)
    CNN.CNNmodeling(T_test, T_target,X_test.shape,T_test.shape)
    print('\n\n\n')

    # DNN Architecture 를 기반으로 학습한 모델
    print('Running DNN...')
    DNN = myDNN(X_test, X_target)
    DNN.DNNmodeling(T_test, T_target,X_test.shape,T_test.shape)
    print('\n\n\n')

    # DataSet의 정확도 비교를 위해 KNN Algorithm을 사용하여 학습한 모델
    print('Running KNN...')
    print('It takes about 2 minutes')
    KNN = myKNN(10)
    KNN.training(X_test, X_target)
    KNN.KNNmodeling(T_test, T_target)
    print('\n\n\n')


if __name__ == "__main__":

    # X_test = Train input feature
    # X_target = Train output result
    # T_test = Test input feature
    # T_target = Test output result

    FirstProcess()
