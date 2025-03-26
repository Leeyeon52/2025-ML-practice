import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt

"""
1. 사이킷런에 저장된 데이터를 불러오고, 
   2개의 변수만을 가질 수 있도록 
   고정하여 반환하는 함수를 구현합니다.
"""
def load_data():
    # 사이킷런의 wine 데이터셋을 불러옵니다.
    data = load_wine()
    X, y = data.data, data.target
    
    # 6번째 열부터 2개의 변수를 선택합니다.
    column_start = 6
    X = X[:, column_start : column_start+2]
    print(X.shape)  # X의 크기를 출력
    return X
    
"""
2. 주성분 분석(PCA)을 수행하여 
   2차원 데이터를 1차원으로 축소하는 함수를 완성합니다.
"""
def pca_data(X):
    pca = PCA(n_components=1)  # 1차원으로 축소
    X_pca = pca.fit_transform(X)  # PCA 변환 수행
    
    return pca, X_pca

# 축소된 주성분 축과 데이터 산점도를 그려주는 함수입니다.
def visualize(pca, X, X_pca):
    X_new = pca.inverse_transform(X_pca)  # PCA로 축소된 데이터를 원래 차원으로 복원
    
    # 원본 데이터와 복원된 데이터를 시각화합니다.
    plt.scatter(X[:, 0], X[:, 1], alpha=0.2, label='Original Data')
    plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8, label='Reconstructed Data')
    plt.axis('equal')
    plt.legend()
    
    plt.savefig('PCA.png')  # 결과를 'PCA.png'로 저장
    plt.show()  # 시각화된 그래프를 화면에 출력

def main():
    # 데이터 불러오기
    X = load_data()
    
    # PCA 수행
    pca, X_pca = pca_data(X)
    
    # 원본 데이터와 PCA 후 데이터를 출력
    print("- original shape:   ", X.shape)
    print("- transformed shape:", X_pca.shape)
    
    # 결과 시각화
    visualize(pca, X, X_pca)
    
if __name__ == '__main__':
    main()
