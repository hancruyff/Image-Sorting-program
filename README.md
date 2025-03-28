# 🚘 자동차 이미지 브랜드 분류기

사전 학습된 CNN 모델(MobileNetV2)을 기반으로 자동차 이미지로부터  
해당 차량이 어떤 브랜드인지 예측하는 머신러닝 기반 분류 프로그램입니다.

---

## 📌 프로젝트 개요

- 자동차 이미지 분류 문제를 해결하기 위한 **딥러닝 분류기**입니다.
- 학습된 모델은 **이미지 입력을 받아 해당 브랜드를 예측**합니다.
- **TensorFlow/Keras**를 활용하여 모델 구조 구성 및 전이학습을 수행하였고,  
  `MobileNetV2` 모델을 사용하여 가벼우면서도 정확한 분류 성능을 구현했습니다.

---

## 🧠 주요 기능

| 기능               | 설명 |
|--------------------|------|
| 📂 데이터 불러오기  | 디렉토리 구조 기반 학습/검증 이미지 로딩
| 🧼 데이터 전처리    | 이미지 크기, 배치 구성, 정규화 처리
| 🔍 모델 구성 및 학습| MobileNetV2 기반 전이 학습 구조 사용
| 🎯 정확도 평가      | 검증 데이터에 대한 예측 정확도 출력
| 🖼 이미지 예측      | 테스트 이미지에 대한 브랜드 예측 수행

---

## 🧾 코드 설명

### 1. 데이터셋 불러오기

```python
from tensorflow.keras.preprocessing.image import image_dataset_from_directory

train_dataset = image_dataset_from_directory(
    "dataset/train",
    image_size=(224, 224),
    batch_size=32,
    label_mode="categorical"
)

val_dataset = image_dataset_from_directory(
    "dataset/val",
    image_size=(224, 224),
    batch_size=32,
    label_mode="categorical"
)
```
---

### 2. 🔍 모델 구성 – MobileNetV2 전이 학습 기반

이 프로젝트는 **사전 학습된 CNN 모델인 MobileNetV2**를 기반으로 자동차 브랜드 분류기를 학습했습니다.

---

### ✅ 전이 학습(Transfer Learning)이란?

> 기존에 대규모 데이터셋(ImageNet 등)에서 학습된 모델의 **특징 추출 능력(feature extractor)**을 활용하여,  
> 새로운 문제에 맞춰 일부 레이어만 재학습하거나 새로운 분류기를 붙여 학습하는 방식입니다.

- 학습 데이터가 부족한 상황에서도 높은 정확도를 얻을 수 있습니다.
- 기존 모델의 강력한 특징 추출 성능을 재사용할 수 있습니다.

---

### 🧱 적용된 모델 구조 (코드 기반 설명)

```python
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# 1. 사전 학습된 MobileNetV2 로드 (top 제거)
base_model = MobileNetV2(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
base_model.trainable = False  # 특징 추출기로만 사용

# 2. 분류기(head) 구성
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),        # 특징 벡터 평균 풀링
    Dense(128, activation='relu'),   # 은닉층
    Dense(5, activation='softmax')   # 자동차 브랜드 수 (5개 클래스)
])
```
---

### 📌 핵심 구성 요소

| 구성 요소                      | 설명                                                                           |
|-------------------------------|--------------------------------------------------------------------------------|
| `MobileNetV2`                 | 경량화된 고성능 CNN 구조, 전이학습에 자주 사용됨                              |
| `include_top=False`           | 기존 최상위 분류기 제거 → 사용자 정의 분류기(head) 연결 가능                   |
| `weights='imagenet'`          | ImageNet에서 학습된 가중치 불러와 특징 추출기로 활용                           |
| `base_model.trainable = False`| 기존 가중치 학습을 중단하고, 특징 추출기 역할로만 사용                         |
| `GlobalAveragePooling2D()`    | CNN의 출력 특징맵을 1D 벡터로 변환 → FC(Dense) 레이어 연결 가능                 |
| `Dense(5, activation='softmax')` | 5개 자동차 브랜드 중 하나로 분류하는 최종 출력층 (다중 클래스 분류기)      |

### 3. 모델 학습
```python
history = model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```
---

### 🎯 왜 MobileNetV2를 선택했나요?

✅ 빠른 연산 속도와 낮은 파라미터 수 덕분에 경량 모델로 적합

✅ 사전 학습된 이미지 특징을 그대로 사용할 수 있어 학습 시간이 짧음

✅ 전이 학습(Transfer Learning) 구조와 완벽하게 호환

---

### 🏆 결과
이 구조를 통해 적은 학습 데이터셋으로도
자동차 브랜드 분류 정확도를 높게 유지할 수 있었으며,
학습 속도와 성능 간 균형을 효과적으로 맞출 수 있었습니다

---

### 4. 정확도 및 시각화
```python
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```
---

### 5. 이미지 예측
```python
image = load_img(image_path, target_size=(img_height, img_width))
image_array = img_to_array(image) / 255.0
image_array = np.expand_dims(image_array, axis=0)

# 모델 로드
model = load_model(model_path)

# 이미지 예측
prediction = model.predict(image_array)

# 예측 결과 출력
predicted_class_index = np.argmax(prediction)
predicted_class_label = class_labels[predicted_class_index]
image.show()
print('Predicted Class:', predicted_class_label)
```
![image](https://github.com/user-attachments/assets/82de94eb-e835-41b8-86c8-7062446decc1)

---

### 🛠 기술 스택
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/> <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/> <img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white"/> <img src="https://img.shields.io/badge/MobileNetV2-009688?style=for-the-badge"/>

Python: 전체 코드 구현

TensorFlow / Keras: 딥러닝 프레임워크

MobileNetV2: 가벼운 전이 학습 모델로 정확도와 속도 균형

Matplotlib: 정확도 시각화

### 📚 학습 포인트

✔️ 사전 학습 모델(MobileNetV2) 기반 전이 학습 이해 및 구현

✔️ 이미지 폴더 구조를 활용한 다중 클래스 분류 학습 방법 습득

✔️ 모델 구성, 평가, 예측 과정 전반 구현 경험

✔️ Jupyter Notebook을 통한 실험 기반 개발 프로세스 경험
