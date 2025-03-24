import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 데이터 생성 및 분할
def load_data():
    np.random.seed(0)
    X = 5 * np.random.rand(100, 1)
    y = 3 * X + 5 * np.random.rand(100, 1)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=0)
    return train_X, train_y, test_X, test_y

# 선형 회귀 모델 학습
def Linear_Regression(train_X, train_y):
    lr = LinearRegression()
    lr.fit(train_X, train_y)
    return lr

# RSS 계산 함수
def return_RSS(test_y, predicted):
    RSS = np.sum((test_y - predicted) ** 2)
    return RSS

# 그래프 시각화 함수
def plotting_graph(test_X, test_y, predicted):
    plt.scatter(test_X, test_y, label="Actual Data")
    
    sorted_idx = np.argsort(test_X[:, 0])  # X 값을 정렬
    plt.plot(test_X[sorted_idx], predicted[sorted_idx], color='r', label="Predicted Line")

    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.title("Linear Regression Fit")
    plt.savefig("result.png")
    plt.show()

# 메인 함수
def main():
    train_X, train_y, test_X, test_y = load_data()
    lr = Linear_Regression(train_X, train_y)
    predicted = lr.predict(test_X)

    RSS = return_RSS(test_y, predicted)
    print("> RSS:", RSS)

    plotting_graph(test_X, test_y, predicted)

if __name__ == "__main__":
    main()
