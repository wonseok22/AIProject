import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class myCNN:

    def __init__(self,X_test,X_target):
        self.X_test = X_test
        self.X_target = X_target

    def DataReshape(self,T_test,T_target,Xtestshape,Ttestshape):
        # CNN 합성곱 이용하기 위해서 2차원 배열인 데이터를 3차원 numpy 배열로 변경함
        self.X_test = np.reshape((np.array(self.X_test, dtype=np.float32)), (Xtestshape[0], Xtestshape[1], 1))
        self.T_test = np.reshape((np.array(T_test, dtype=np.float32)), (Ttestshape[0], Ttestshape[1], 1))
        self.X_target = np.array(self.X_target)
        self.T_target = np.array(T_target)

    def CNNmodeling(self, T_test, T_target,Xtestshape,Ttestshape):
        self.DataReshape(T_test,T_target,Xtestshape,Ttestshape) # 데이터를 3차원 numpy 배열로 변경
        model = Sequential() # 순차적으로 레이어 더해주는 순차모델 생성
        model.add(Dense(Xtestshape[1], activation='relu'))# input값 받는 레이어
        # 합성곱 레이어 추가
        model.add(Conv1D(256, 3, padding='valid', activation='relu'))
        # 풀링 레이어 추가
        model.add(GlobalMaxPooling1D())
        # 히든 레이어 추가
        model.add(Dense(128, activation = 'relu'))
        # sigmoid함수 이용하여 최종적으로 target값 예측
        model.add(Dense(1, activation='sigmoid'))
        # 일정 에포크횟수 이후 손실함수값의 변화가 더이상 없다 판단하면 에포크 멈춤
        #Early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
        #Model_Checkpoint = ModelCheckpoint('model_binarycross_rmsprop.h5', monitor = 'val_accuracy', mode = 'max', verbose = 1, save_best_only = True)
        # 모델컴파일, 인자로는 손실함수, 옵티마이저, 출력할 값 지정
        model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=['accuracy'])
        # 컴파일된 모델 학습 시작, 인자로는 학습용데이터, 에포크 수, 배치사이즈, 테스트할 데이터
        history = model.fit(self.X_test, self.X_target, epochs=10, batch_size=50,validation_data=(self.T_test, self.T_target))



        # 그래프 출력부
        '''
        history_dict = history.history
        loss = history_dict['loss']
        val_loss = history_dict['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.clf()  # 그래프를 초기화합니다.
        acc = history_dict['accuracy']
        val_acc = history_dict['val_accuracy']

        plt.plot(epochs, loss, 'bo', label='Training loss')  # ‘bo’는 파란색 점을 의미합니다.
        plt.plot(epochs, val_loss, 'b', label='Validation loss')  # ‘b’는 파란색 실선을 의미합니다.
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.ylim([0, 1])
        plt.legend()
        plt.show()


        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.legend()

        plt.show()
        '''
