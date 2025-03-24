import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes  # boston 대신 diabetes 사용 (boston 데이터셋은 제거됨)

# 데이터 로드 함수
def load_data():
    X, y = load_diabetes(return_X_y=True)
    feature_names = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
    return X, y, feature_names

# ElasticNet 회귀 모델 학습 함수
def ElasticNet_regression(train_X, train_y):
    elasticnet = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
    elasticnet.fit(train_X, train_y)
    return elasticnet

# 그래프 출력 함수
def plot_graph(coef, feature_names):
    plt.figure(figsize=(8, 5))
    coef_series = pd.Series(coef, index=feature_names).sort_values()
    coef_series.plot(kind="bar")
    plt.title("ElasticNet Coefficients")
    plt.savefig("elasticnet_coefficients.png")
    plt.show()

# main 함수
def main():
    X, y, feature_names = load_data()
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=100)

    elasticnet_reg = ElasticNet_regression(train_X, train_y)

    # 모델 평가
    score = elasticnet_reg.score(test_X, test_y)
    print("ElasticNet 회귀의 평가 점수:", score)

    # 회귀 계수 시각화
    plot_graph(elasticnet_reg.coef_, feature_names)

if __name__ == "__main__":
    main()
