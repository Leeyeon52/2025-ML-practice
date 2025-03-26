import preprocess as pre
from sklearn.cluster import KMeans  # sklearn에서 KMeans를 직접 임포트
import numpy as np
from sklearn.metrics import silhouette_score

def silhouette_calculator(X, cluster):
    # silhouette_score() 함수를 사용하여 silhouette 값을 계산
    silhouette = silhouette_score(X, cluster, metric='euclidean')
    return silhouette

def your_choice(X, K):
    # 주어진 K 값으로 KMeans 모델을 학습
    return KMeans(n_clusters=K).fit(X)

def main():
    # 데이터 불러오기
    X = pre.load_data()
    
    # 여러 K 값에 대해 실험하여 silhouette 값을 계산
    for K in range(2, 11):  # 예시로 K를 2에서 10까지 실험
        Kmeans = your_choice(X, K)
        
        # KMeans 모델에서 군집화 결과의 labels_ 속성 추출
        cluster_labels = Kmeans.labels_
        
        # silhouette 값을 계산
        silhouette = silhouette_calculator(X, cluster_labels)
        
        # 결과 출력
        print('K = {}, silhouette 값: {:.4f}'.format(K, silhouette))
    
if __name__ == "__main__":
    main()
