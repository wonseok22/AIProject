import sys,os
import matplotlib.pyplot as plt
import numpy as np
import time
from dataset.mnist import load_mnist
from hw2KNN import myKNN


(x_train,t_train),(x_test,t_test) = load_mnist(flatten=True, normalize=False)

def boxing(boxsize,data):
    tmp = np.array([0 for _ in range(784//(boxsize*boxsize))])
    for i in range(28):
        for j in range(0,28,boxsize):
            for k in range(boxsize):
                tmp[(i//boxsize)*(28//boxsize) + j//boxsize] += data[(i*28)+j+k]
    return tmp



if __name__ == "__main__":
    print(len(x_train),len(x_test))
    x_train = x_train.astype('int32')
    t_train = t_train.astype('int32')

    # hand-craft
    # input feature (28,28)를 28의 약수 2,4,7로 나누어 2*2, 4*4, 7*7 씩 묶는다
    # 이 때, input feature 의 값은 묶은 1*1 feature 들의 합으로 한다.
    # 2*2씩 묶었을 때 input feature 의 수 : 196
    # 4*4씩 묶었을 때 input feature 의 수 : 49
    # 7*7씩 묶었을 때 input feature 의 수 : 16
    # 세 가지 방법에 대해 정확도, 소요시간을 측정한다.

    boxsize = 14  # 묶을 크기
    size = 1000  # testData 의 개수
    cnt = 0 # 정확도 측정을 위한 correct label counter
    sample = np.random.randint(0,t_test.shape[0],size)
    classes = myKNN(10)
    handCraftTrain = np.array([[0]*(784//(boxsize*boxsize)) for _ in range(60000)])
    startTime = time.time()

    # 모든 trainData 에 대해 boxing 을 수행한다.
    for idx,data in enumerate(x_train):
        handCraftTrain[idx] = boxing(boxsize,data)
    classes.training(handCraftTrain, t_train)

    dataTime = time.time() - startTime
    startTime = time.time()
    zz = 0
    # 시간 단축을 위해 모든 testData 가 아닌 sample testData 에 대해서만 boxing 을 수행한 후 KNN 을 적용한다.
    for i in sample:
        print(zz)
        zz += 1
        predicts = classes.weightedPredict(boxing(boxsize,x_test[i]))
        print(str(i) + ' th data result',predicts[0],' label',t_test[i])
        if predicts[0] == t_test[i]:
            cnt +=1
    print('accuracy =',(cnt/size)*100,'%')
    print('Data Processing Time :',dataTime)
    print('Predict Time :',time.time()-startTime)
    print('Total Time :',dataTime+(time.time()-startTime))
