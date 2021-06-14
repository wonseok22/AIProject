import csv
import numpy as np
import pandas as pd
from CNN import myCNN
from DNN import myDNN
from KNN import myKNN
from sklearn.model_selection import train_test_split


def FirstProcess(): #첫번째 전처리 실행
    path = open('cardio_train.csv', 'r', encoding='utf-8')
    file = csv.reader(path)
    Train = np.zeros((70000, 10))
    Test = np.zeros((70000, ))

    for idx, value in enumerate(file):
        if idx != 0:
            dataline = value[0].split(';')
            age = int(dataline[1]) // 365
            BMI = float(dataline[4]) / ((int(dataline[3])/100)**2)
            Train[idx-1][0] = age
            Train[idx-1][1] = BMI
            Train[idx-1][2] = int(dataline[2])
            for i in range(3,10):
                Train[idx-1][i] = int(dataline[i+2])
            Test[idx-1] = int(dataline[12])
    path.close()
    X_test, T_test, X_target, T_target = train_test_split(Train, Test, test_size=10000)

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


def SecProcess():
    # input feature의 체중, 몸무게를 BMI지수로, Day 단위의 age를 year_age로 변환
    path = open('cardio_train.csv', 'r', encoding='utf-8')
    file = csv.reader(path)
    Train = np.zeros((70000, 10))
    Test = np.zeros((70000, ))

    for idx, value in enumerate(file):
        if idx != 0:
            dataline = value[0].split(';')
            age = int(dataline[1]) // 365
            BMI = float(dataline[4]) / ((int(dataline[3])/100)**2)
            Train[idx-1][0] = age
            Train[idx-1][1] = BMI
            Train[idx-1][2] = int(dataline[2])
            for i in range(3,10):
                Train[idx-1][i] = int(dataline[i+2])
            Test[idx-1] = int(dataline[12])

    # Age, BMI 를 각각 input feature 의 평균값으로 나누어 feature끼리의 편차를 줄이는 feature scaling 작업
    p = pd.DataFrame(Train).describe(percentiles = [.50])
    avg = []
    for i in range(10):
        avg.append(p[i][1])
    for idx,value in enumerate(Train):
        for id, key in enumerate(Train[idx]):
            if id == 0: # age
                Train[idx][id] /= avg[0]
            elif id == 1: # BMI
                Train[idx][id] /= avg[1]
    path.close()
    X_test, T_test, X_target, T_target = train_test_split(Train, Test, test_size=10000)

    print('SecondProcessing으로 전처리한 Data')
    print('input feature의 체중, 몸무게를 BMI지수로, Day 단위의 age를 year_age로 변환')
    print('BMI 지수와 year_age를 각 feature의 평균값으로 나누어 편차를 줄임\n')

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



def ThirdProcess():
    path = open('cardio_train.csv', 'r', encoding='utf-8')
    file = csv.reader(path)
    Train = np.zeros((70000, 10))
    Test = np.zeros((70000, ))

    for idx, value in enumerate(file):
        if idx != 0:
            dataline = value[0].split(';')
            age = int(dataline[1]) // 365
            BMI = float(dataline[4]) / ((int(dataline[3]) / 100) ** 2)
            Train[idx - 1][0] = age
            Train[idx - 1][1] = BMI
            Train[idx - 1][2] = int(dataline[2])
            for i in range(3, 10):
                Train[idx - 1][i] = int(dataline[i + 2])
            Test[idx - 1] = int(dataline[12])

    # 각 input feature를 해당 feature의 max값으로 나누어 0-1 정규화를 실시
    p = pd.DataFrame(Train).describe(percentiles=[.50])
    M = []
    for i in range(10):
        M.append(p[i][5])
    for idx, value in enumerate(Train):
        for id, key in enumerate(Train[idx]):
            Train[idx][id] /= M[id]
    path.close()
    X_test, T_test, X_target, T_target = train_test_split(Train, Test, test_size=10000)

    print('ThirdProcessing으로 전처리한 Data')
    print('input feature의 체중, 몸무게를 BMI지수로, Day 단위의 age를 year_age로 변환')
    print('모든 input feature에 대해 각 feature를 max값으로 나누어 0-1 정규화 실시\n')

    #CNN Architecture 를 기반으로 학습한 모델
    print('Running CNN...')
    CNN = myCNN(X_test,X_target)
    CNN.CNNmodeling(T_test,T_target,X_test.shape,T_test.shape)
    print('\n\n\n')

    #DNN Architecture 를 기반으로 학습한 모델
    print('Running DNN...')
    DNN = myDNN(X_test, X_target)
    DNN.DNNmodeling(T_test,T_target,X_test.shape,T_test.shape)
    print('\n\n\n')

    #DataSet의 정확도 비교를 위해 KNN Algorithm을 사용하여 학습한 모델
    print('Running KNN...')
    print('It takes about 2 minutes')
    KNN = myKNN(10)
    KNN.training(X_test,X_target)
    KNN.KNNmodeling(T_test,T_target)
    print('\n\n\n')



if __name__ == "__main__":


    # X_test = Train input feature
    # X_target = Train output result
    # T_test = Test input feature
    # T_target = Test output result

    # First Processing
    # input feature의 체중, 몸무게를 BMI지수로, Day 단위의 age를 year_age로 변환
    FirstProcess()

    # Second Processing
    # input feature의 체중, 몸무게를 BMI지수로, Day 단위의 age를 year_age로 변환
    # BMI 지수와 year_age를 각 feature의 평균값으로 나누어 편차를 줄임
    SecProcess()

    # Third Processing
    # input feature의 체중, 몸무게를 BMI지수로, Day 단위의 age를 year_age로 변환
    # 모든 input feature에 대해 각 feature를 max값으로 나누어 0-1 정규화 실시
    ThirdProcess()
