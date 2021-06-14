import numpy as np
class myKNN:
    def __init__(self,K):
        self.K = K

    def training(self, x_train, t_train):
        self.train = x_train
        self.label = t_train

    def weightedPredict(self,testData):
        distances = []
        print(self.train.shape,testData.shape)
        distance = np.sqrt(np.sum((self.train - testData) ** 2, axis=1))
        neighbor = np.argsort(distance)
        KN = []

        for i in range(self.K):
            KN.append([self.label[neighbor[i]],distance[neighbor[i]]])

        # class 에 대한 가중치 계산
        weighted = [0,0,0,0,0,0,0,0,0,0]
        for i,dis in KN:
            if dis != 0:
                weighted[i] += 1/dis
        distances.append(weighted.index(max(weighted)))
        return distances