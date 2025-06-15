import pandas as pd
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models, layers

# 경로 설정
csv_path = "/home/yoon/Downloads/ccccnnn/Train.csv"
dataset_path = "/home/yoon/Downloads/ccccnnn"
IMG_SIZE = 32

# 사용할 클래스: 30km/h(1), 70km/h(4)
selected_classes = {1: 0, 4: 1}  # 새 클래스 인덱스로 매핑

# 데이터 로딩 및 필터링
data = pd.read_csv(csv_path)
data = data[data['ClassId'].isin(selected_classes.keys())]

images = []
labels = []

for _, row in data.iterrows():
    img_path = os.path.join(dataset_path, row['Path'])
    img = cv2.imread(img_path)
    if img is None:
        continue
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    images.append(img)
    labels.append(selected_classes[row['ClassId']])

X = np.array(images) / 255.0
y = to_categorical(np.array(labels), num_classes=2)

# 데이터 분할
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 구성
model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(2, activation='softmax')  # 클래스 2개로 변경
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# 모델 저장
model.save("/home/yoon/turtlebot3_ws/src/find_maze/find_maze/model/gtsrb_model_2class2.keras")
