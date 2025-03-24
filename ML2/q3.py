import matplotlib.pyplot as plt

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# boston 데이터를 위한 모듈을 불러옵니다. 
from sklearn.datasets import fetch_california_housing

"""
1. 사이킷런에 존재하는 데이터를 불러오고, 
   불러온 데이터를 학습용 데이터와 테스트용 데이터로
   분리하여 반환하는 함수를 구현합니다.
"""

def load_data():
    
    housing = fetch_california_housing()
    X, y = housing.data, housing.target  # 입력 데이터, 타겟 분리

    print("데이터의 입력값(X)의 개수 :", X.shape[1])
    
    # 80% 훈련 데이터, 20% 테스트 데이터 분리
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return train_X, test_X, train_y, test_y
"""
2. 다중 선형회귀 모델을 불러오고, 
   불러온 모델을 학습용 데이터에 맞추어 학습시킨 후
   해당 모델을 반환하는 함수를 구현합니다.

"""
def Multi_Regression(train_X,train_y):
    
    model = LinearRegression()
    
    model.fit(train_X, train_y)
    
    return model
    
"""
3. 모델 학습 및 예측 결과 확인을 위한 main 함수를 완성합니다.
"""
def main():
    
    train_X, test_X, train_y, test_y = load_data()
    
    model = Multi_Regression(train_X,train_y)
    
    predicted = model.predict(test_X)
    
    model_score = model.score(test_X, test_y)
    
    print("\n> 모델 평가 점수 :", model_score)
     
    beta_0 = model.intercept_
    beta_i_list = model.coef_
    
    print("\n> beta_0 : ",beta_0)
    print("> beta_i_list : ",beta_i_list)
    
    return predicted, beta_0, beta_i_list, model_score
    
if __name__ == "__main__":
    main()