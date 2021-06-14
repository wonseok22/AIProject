import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class myDNN:
    def __init__(self, X_test, X_target):
        self.X_test = X_test        #학습용 데이터 feature
        self.X_target = X_target    #학습용 데이터 타겟

    def DNNmodeling(self, T_test,T_target,X,Y):
        # 순차적으로 레이어 더해주는 순차모델 생성
        model = Sequential()
        # input값 받는 레이어
        model.add(Dense(X[1], activation = 'relu', input_shape=(X[1],)))
        # 히든 레이어, binary target이므로 relu 이용
        model.add(Dense(256, activation='relu'))
        # 히든 레이어
        model.add(Dense(256, activation='relu'))
        # 히든 레이어, 실행결과 1개일 때 정확성이높았음
        model.add(Dense(256, activation='relu'))
        # sigmoid함수 이용하여 최종적으로 target값 예측
        model.add(Dense(1, activation = 'sigmoid'))
        # 일정 에포크횟수 이후 손실함수값의 변화가 더이상 없다 판단하면 에포크 멈춤
        # early_stopping = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 3)
        # Model_Checkpoint = ModelCheckpoint('model_binarycross_rmsprop.h5', monitor = 'val_accuracy', mode = 'max', verbose = 1, save_best_only = True)
        # monitor = 'val_accuracy', mode = 'max', verbose = 1, save_best_only = True)

        # 이전까지의 학습모델중 가장 정확성이 높은 모델을
        # model_binarycross_rmsprop.h5 로 저장
        # 모델컴파일, 인자로는 손실함수, 옵티마이저, 출력할 값 지정
        model.compile(loss = "binary_crossentropy", optimizer="adam", metrics=['accuracy'])
        # 컴파일된 모델 학습 시작, 인자로는 학습용데이터, 에포크 수, 배치사이즈, 테스트할 데이터
        history = model.fit(self.X_test, self.X_target, epochs=10, batch_size=50, validation_data=(T_test, T_target))

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
