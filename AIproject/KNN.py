import numpy as np

class myKNN:
    def __init__(self,K):
        self.K = K      # K값 받음

    def training(self, X_test, X_target): #데이터 받음
        self.X_test = X_test
        self.X_target = X_target

    def KNNmodeling(self,T_test,T_target):
        self.distances = [] #테스트데이터와 학습용데이터들의 거리 저장할 리스트
        for point in T_test:
            distance = np.sqrt(np.sum((self.X_test - point) ** 2, axis=1))  #유클리디안으로 계산
            neighbor = np.argsort(distance)  # 정렬하여 index 서장
            KN = []  # 가까운 이웃들 저장할 리스트
            for i in range(self.K):  #가장 가까운 이웃 k개만 저장
                KN.append([self.X_target[neighbor[i]],distance[neighbor[i]]])
            weighted = [0,0]
            for i,dis in KN:
                if dis:
                    if dis != 0 :
                        weighted[int(i)] += 1/dis  #거리가 d이면 1/d로 가중치 부여
            self.distances.append(weighted.index(max(weighted)))
        self.printResult(T_target)

    def printResult(self,T_target):  #정확성 출력 함수
        cnt = 0
        size = len(T_target)  #전체 테스트케이스 수
        for idx,value in enumerate(self.distances):
            if value == T_target[idx]:  #예측한 결과가 target값과 일치할 경우
                cnt += 1  #맞춘 갯수 증가
        print('KNN Algorithm accuracy =', (cnt / size), '%')  #최종적으로 정확도 계산후 출력


