import csv
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


# X_test = Train input feature
# X_target = Train output result
# T_test = Test input feature
# T_target = Test output result
# 남 녀 : 1 = 0.5 2 = 0.4  스모커 0.477 논스모커 0.502

def Process():
    path = open('cardio_train.csv','r',encoding='utf-8')
    file = csv.reader(path)
    Train = np.zeros((70000,10))
    Test = np.zeros((70000,1))

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

    # 평균으로 나누는거
    p = pd.DataFrame(Train).describe(percentiles = [.50])
    c = []
    for i in range(10):
        c.append(p[i][1])
    for idx,value in enumerate(Train):
        for id, key in enumerate(Train[idx]):
            if id == 0: # age
                Train[idx][id] /= c[0]
            elif id == 1: # BMI
                Train[idx][id] /= c[1]
            elif id == 2: # gender
                if Train[idx][id] == 1:
                    Train[idx][id] = 0.5
                else:
                    Train[idx][id] = 0.4
            elif id == 3: # aphi
                Train[idx][id] /= c[3]
            elif id == 4: # aplo
                Train[idx][id] /= c[4]
            elif id == 5: # chol
                if Train[idx][id] == 1:
                    Train[idx][id] = 0.44
                elif Train[idx][id] == 2:
                    Train[idx][id] = 0.605
                else:
                    Train[idx][id] = 0.766
            elif id == 6: # glus
                if Train[idx][id] == 1:
                    Train[idx][id] = 0.480
                elif Train[idx][id] == 2:
                    Train[idx][id] = 0.595
                else:
                    Train[idx][id] = 0.624
            elif id == 7: # smoke
                if Train[idx][id] == 1:
                    Train[idx][id] = 0.477
                else:
                    Train[idx][id] = 0.502
            elif id == 8: # alco
                if Train[idx][id] == 1:
                    Train[idx][id] = 0.484
                else:
                    Train[idx][id] = 0.501
            elif id == 9: # active
                if Train[idx][id] == 1:
                    Train[idx][id] = 0.491
                else:
                    Train[idx][id] = 0.537

    path.close()
    return(train_test_split(Train,Test,test_size=10000))
# age bmi gender aphi aplo chol gluc smoke alco active

X_test,T_test,X_target,T_target = Process()
X_test = np.array(X_test)
X_target = np.array(X_target)
T_test = np.array(T_test)
T_target = np.array(T_target)
