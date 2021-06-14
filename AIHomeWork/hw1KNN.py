import numpy as np

class myKNN:
    def __init__(self,K):
        self.K = K

    def training(self, X_train, y_train):
        self.train = X_train
        self.label = y_train

    def predict(self,testData):
        distances = []
        for point in testData:
            distance = np.sqrt(np.sum((self.train - point)**2 ,axis = 1))
            neighbor = np.argsort(distance)
            KN = []
            for i in range(self.K):
                KN.append(self.label[neighbor[i]])

            # class counting

            cnt = [0,0,0]
            for i in KN:
                if i == 0:
                    cnt[0] += 1
                elif i == 1:
                    cnt[1] += 1
                else:
                    cnt[2] += 1
            distances.append(cnt.index(max(cnt)))
        return distances

    def weightedPredict(self,testData):
        distances = []
        for point in testData:
            distance = np.sqrt(np.sum((self.train - point) ** 2, axis=1))
            neighbor = np.argsort(distance)
            KN = []
            for i in range(self.K):
                KN.append([self.label[neighbor[i]],distance[neighbor[i]]])

            # class 0, 1, 2에 대한 가중치 계산
            weighted = [0,0,0]
            for i,dis in KN:
                if i == 0:
                    weighted[0] += 1/dis
                elif i == 1:
                    weighted[1] += 1/dis
                else:
                    weighted[2] += 1/dis
            distances.append(weighted.index(max(weighted)))

        return distances