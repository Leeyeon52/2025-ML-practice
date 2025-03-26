import preprocess as pre
from sklearn.decomposition import PCA  # sklearn에서 PCA를 임포트합니다.
import numpy as np


def variance_calculator(model, X):
    # 원본 데이터의 분산을 구합니다.
    original_var = np.var(X, axis=0).sum()
    
    # PCA로 차원 축소 후의 분산을 구합니다.
    shrinked_var = model.explained_variance_.sum()
    
    # 차원 축소 후의 분산 유지율을 계산하여 반환합니다.
    variance_ratio = shrinked_var / original_var
    return variance_ratio


# 사용자가 원하는 차원 수로 PCA 모델을 생성합니다.
def your_choice(X, n_components):
    # sklearn의 PCA를 사용합니다.
    myPCA = PCA(n_components=n_components)
    myPCA.fit_transform(X)
    
    return myPCA


def main():
    X = pre.load_data()
    
    # 사용자에게 차원 수를 입력받습니다.
    n_components = int(input("원하는 차원 수를 입력하세요: "))
    
    # PCA 모델을 학습합니다.
    myPCA = your_choice(X, n_components)
    
    # 원본 대비 축소 후의 분산 유지율을 계산합니다.
    percentage = variance_calculator(myPCA, X)
    
    # 축소 후 차원의 수와 원본 대비 축소 후의 분산 유지율을 출력합니다.
    print('원본 데이터의 차원 = {}, n_components = {}, 원본 대비 축소 후의 분산의 비율: {:.4f}'.format(X.shape[1], n_components, percentage))
    

if __name__ == "__main__":
    main()
3