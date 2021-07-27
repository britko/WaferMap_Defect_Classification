import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # 그래프
import seaborn as sns # matplotlib을 기반으로 다양한 색상 테마와 통계용 차트 등의 기능을 추가한 시각화 패키지
import os, cv2
from matplotlib.image import imread # 이미지 파일 읽기

from sklearn.metrics import classification_report,confusion_matrix # 머신러닝 분석모듈

from tensorflow.keras.preprocessing.image import ImageDataGenerator # 데이터 증강
from tensorflow.keras.models import Sequential # sequential 모델: 계층을 선형으로 쌓음
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPool2D, InputLayer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam # 매개변수 갱신 방법 Adam
from tensorflow.keras.preprocessing import image

import warnings
warnings.filterwarnings('ignore') # 경고 메시지를 무시하고 숨김

### 이미지 load하기
train_data=os.listdir(r'C:/Users/pc/Jupyter Notebook/WaferMap/balanced') # 훈련 데이터 디렉토리
# for i in b_d:
#    print( i,len(os.listdir(r'C:/Users/pc/Jupyter Notebook/WaferMap/balanced/'+i)))
# print("==================================================================")
test_data=os.listdir(r'C:/Users/pc/Jupyter Notebook/WaferMap/imbalanced') # 테스트 데이터 디렉토리
# for i in test_d:
#    print( i,len(os.listdir(r'C:/Users/pc/Jupyter Notebook/WaferMap/imbalanced/'+i)))

train_path=r'C:/Users/pc/Jupyter Notebook/WaferMap/balanced/'
sample_wafer=[]
for i in train_data:
    sample_wafer.append(train_path+i+'/'+os.listdir(train_path+i+'/')[0])
# sample_wafer

# # plt.figure(figsize=(24,12))
# f, axarr = plt.subplots(3,3,figsize=(24,12))
# m=0
# for i in range(3):
#     for j in range(3):
#         axarr[i,j].imshow(imread(sample_wafer[m]))
#         axarr[i,j].set_title(os.path.basename(sample_wafer[m])) 
#         m+=1

### 이미지 shape을 받아서 2차원 데이터 생성
def dimension(path,dim1,dim2):
    for image_filename in os.listdir(path): 
        image=imread(path+image_filename)
        d1,d2,channels=image.shape
        dim1.append(d1)
        dim2.append(d2)
#         print(channels)
    return dim1,dim2

loc_dim1=[]
loc_dim2=[]
loc_dim1,loc_dim2=dimension(r'C:/Users/pc/Jupyter Notebook/WaferMap/balanced/Loc/',loc_dim1,loc_dim2)

edgeRing_dim1=[]
edgeRing_dim2=[]
edgeRing_dim1,edgeRing_dim2=dimension(r'C:/Users/pc/Jupyter Notebook/WaferMap/balanced/Edge-ring/',edgeRing_dim1,edgeRing_dim2)

edgeLoc_dim1=[]
edgeLoc_dim2=[]
edgeLoc_dim1,edgeLoc_dim2=dimension(r'C:/Users/pc/Jupyter Notebook/WaferMap/balanced/Edge-loc/',edgeLoc_dim1,edgeLoc_dim2)

center_dim1=[]
center_dim2=[]
center_dim1,center_dim2=dimension(r'C:/Users/pc/Jupyter Notebook/WaferMap/balanced/Center/',center_dim1,center_dim2)

random_dim1=[]
random_dim2=[]
random_dim1,random_dim2=dimension(r'C:/Users/pc/Jupyter Notebook/WaferMap/balanced/Random/',random_dim1,random_dim2)

scratch_dim1=[]
scratch_dim2=[]
scratch_dim1,scratch_dim2=dimension(r'C:/Users/pc/Jupyter Notebook/WaferMap/balanced/Scratch/',scratch_dim1,scratch_dim2)

nearFull_dim1=[]
nearFull_dim2=[]
nearFull_dim1,nearFull_dim2=dimension(r'C:/Users/pc/Jupyter Notebook/WaferMap/balanced/Near-Full/',nearFull_dim1,nearFull_dim2)

donut_dim1=[]
donut_dim2=[]
donut_dim1,donut_dim2=dimension(r'C:/Users/pc/Jupyter Notebook/WaferMap/balanced/Donut/',donut_dim1,donut_dim2)

none_dim1=[]
none_dim2=[]
none_dim1,none_dim2=dimension(r'C:/Users/pc/Jupyter Notebook/WaferMap/balanced/None/',none_dim1,none_dim2)

## 이미지 증강 함수
image_gen = ImageDataGenerator(rotation_range=20, # rotate the image 20 degrees
                               width_shift_range=0.10, # Shift the pic width by a max of 5%
                               height_shift_range=0.10, # Shift the pic height by a max of 5%
                               rescale=1/255, # Rescale the image by normalzing it.
                               shear_range=0.1, # Shear means cutting away part of the image (max 10%)
                               zoom_range=0.1, # Zoom in by 10% max
                               horizontal_flip=True, # Allo horizontal flipping
                               fill_mode='nearest' # Fill in missing pixels with the nearest filled value
                              )


train_path=r'C:/Users/pc/Jupyter Notebook/WaferMap/balanced'
test_path=r'C:/Users/pc/Jupyter Notebook/WaferMap/imbalanced'
# image_gen.flow_from_directory(train_path)
# image_gen.flow_from_directory(test_path)

batch_size = 16 # 16개의 샘플마다 가중치 업데이트
img_shape=(64,65,4)

## 학습 모델 CNN
model = Sequential()

model.add(InputLayer(input_shape=(64, 65, 4)))
model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=img_shape, activation='relu',))
model.add(MaxPool2D(pool_size=(2, 2)))
## FIRST SET OF LAYERS
# CONVOLUTIONAL LAYER
# POOLING LAYER
model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=img_shape, activation='relu',))
model.add(MaxPool2D(pool_size=(2, 2)))

## SECOND SET OF LAYERS
# CONVOLUTIONAL LAYER
# POOLING LAYER
model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=img_shape, activation='relu',))
model.add(MaxPool2D(pool_size=(2, 2)))

# FLATTEN IMAGES FROM 64 by 65 to 4160 BEFORE FINAL LAYER >>> 2차원 배열을 1차원 배열로 바꿔줌
model.add(Flatten())

# 256 NEURONS IN DENSE HIDDEN LAYER (YOU CAN CHANGE THIS NUMBER OF NEURONS)
model.add(Dense(256, activation='relu')) # 입력 4160 > 출력 256
model.add(Dropout(0.3)) # 오버피팅을 억제하기 위해 dropout 사용

# LAST LAYER IS THE CLASSIFIER, THUS 9 POSSIBLE CLASSES
model.add(Dense(9, activation='softmax')) # 입력 256 > 출력 9 (defect종류 개수)

## compile 메서드를 호출해 학습 과정 설정
model.compile(loss='categorical_crossentropy', # 손실함수 설정
              optimizer='adam',
              metrics=['accuracy']) # 훈련 모니터링


# model.summary() # 모델 구조 확인

#####################################################################################################################
# import tensorflow as tf
from keras.callbacks import ModelCheckpoint

# my_callbacks = [
#     tf.keras.callbacks.EarlyStopping(patience=2),
#     tf.keras.callbacks.ModelCheckpoint(filepath=r'C:/Users/pc/Jupyter Notebook/model.{epoch:02d}-{val_loss:.2f}.h5'),
#     tf.keras.callbacks.TensorBoard(log_dir=r'C:/Users/pc/Jupyter Notebook/logs'),
# ]

filename = r'C:/Users/pc/Jupyter Notebook/savemodel/checkpoint-{epoch:02d}-loss{val_loss:.2f}-acc{val_accuracy:.2f}.h5' # 가중치를 저장할 파일
checkpoint = ModelCheckpoint(filename,             # file명을 지정합니다
                             monitor='val_loss',   # val_loss 값이 개선되었을때 호출됩니다
                             verbose=1,            # 로그를 출력합니다
                             save_best_only=True,  # 가장 best 값만 저장합니다
                             mode='auto'           # auto는 알아서 best를 찾습니다. min/max
                            )

## flow_from_directory: 제너레이팅한 이미지를 폴더명에 맞춰 자동으로 레이블링
train_image_gen = image_gen.flow_from_directory(train_path,
                                               target_size=img_shape[:2], # 패치 이미지 크기를 지정
                                               color_mode='rgba', # 적 녹 청 투명도
                                               batch_size=batch_size,
                                               class_mode='categorical') # 분류방식 - 2D one-hot coding된 라벨이 반환됨

test_image_gen = image_gen.flow_from_directory(test_path,
                                               target_size=img_shape[:2],
                                               color_mode='rgba',
                                               batch_size=batch_size,
                                               class_mode='categorical',
                                               shuffle=False)


results = model.fit_generator( # fit_generator: generator를 사용해 데이터를 계속 주면서 학습
    train_image_gen, # 훈련 데이터셋
    validation_data=test_image_gen, # 검증 데이터셋
    epochs=50, # 전체 훈련 데이터셋에 대한 학습 반복 횟수.
    callbacks=[checkpoint] # 모델, 가중치 저장
)