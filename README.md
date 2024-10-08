## 🗉목차
[1. 개요](#1-개요)

[2. 데이터 전처리](#2-데이터-전처리)

[3. 모델 구성](#3-모델-구성)

[4. 결과](#4-결과)

## 1. 〽개요
![image (11)](https://github.com/user-attachments/assets/96dc5a44-61d8-42d5-a8f4-65ed14b4dea5)

자동차 이미지를 보고 브랜드 별로 분류하는 프로그램

<img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white"> <img src="https://img.shields.io/badge/googlecolab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white"> <img src="https://img.shields.io/badge/tensorflow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white">

## 2. ⚙데이터 전처리
- 이미지 리사이즈: 150x150 픽셀로 조정.
- 데이터 증강: 회전, 이동, 비틀기, 확대/축소, 수평 뒤집기를 통해 데이터 다양성 증가.
- 정규화: 픽셀 값을 0과 1 사이로 조정.

## 3. 🏗모델 구성
- 합성곱층(Conv2D): 여러 층을 쌓아 이미지 특징 추출.
- 풀링층(MaxPooling2D): 차원 축소 및 특징 요약.
- 완전 연결층(Dense): 이미지 분류를 위한 최종 출력층.
- 활성화 함수 : Relu

## 4. 📊결과
- 에포크: 200
- 배치 크기: 32
- 손실 함수: categorical_crossentropy
![image](https://github.com/user-attachments/assets/82de94eb-e835-41b8-86c8-7062446decc1)

약 91%의 정확도를 나타냄
