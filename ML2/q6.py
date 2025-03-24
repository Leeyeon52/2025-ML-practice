import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge, Lasso
from sklearn.datasets import fetch_california_housing  # Boston 데이터 대체
from sklearn.preprocessing import StandardScaler

"""
1. 사이킷런에서 데이터를 불러오고, 
   데이터를 학습용 데이터와 테스트 데이터로 변환하는 함수
"""
def load_data():
    data = fetch_california_housing()  # 새로운 데이터셋 사용
    X, y = data.data, data.target
    feature_names = data.feature_names  # 올바른 속성명 사용

    # Ridge & Lasso는 정규화가 필요하므로 스케일링 적용
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, feature_names

"""
2. Ridge 회귀를 학습시키는 함수
"""
def Ridge_regression(X, y):
    ridge_reg = Ridge(alpha=1.0)  # alpha 값 조절 가능
    ridge_reg.fit(X, y)
    return ridge_reg

"""
3. Lasso 회귀를 학습시키는 함수
"""
def Lasso_regression(X, y):
    lasso_reg = Lasso(alpha=0.1)  # alpha 값 조절 가능
    lasso_reg.fit(X, y)
    return lasso_reg

"""
4. 회귀 계수를 시각화하는 함수
"""
def plot_graph(coef, title):
    plt.figure(figsize=(10, 6))
    plt.ylim(-1, 1)
    plt.title(title)
    coef.plot(kind='bar')
    plt.show()

"""
5. 모델 학습 및 실행
"""
def main():
    X, y, feature_names = load_data()

    ridge_reg = Ridge_regression(X, y)
    lasso_reg = Lasso_regression(X, y)

    # Ridge 회귀 계수 출력
    ridge_coef = pd.Series(ridge_reg.coef_, index=feature_names).sort_values()
    print("\nRidge 회귀의 beta_i\n", ridge_coef)

    # Lasso 회귀 계수 출력
    lasso_coef = pd.Series(lasso_reg.coef_, index=feature_names).sort_values()
    print("\nLasso 회귀의 beta_i\n", lasso_coef)

    # 그래프 출력
    plot_graph(ridge_coef, 'Ridge Regression Coefficients')
    plot_graph(lasso_coef, 'Lasso Regression Coefficients')

if __name__ == "__main__":
    main()
