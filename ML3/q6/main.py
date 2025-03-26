import preprocess as pre
import model as md
import numpy as np
from sklearn.metrics import silhouette_score

def silhouette_calculator(X, cluster):
    # 군집화 결과에서 label을 추출하여 silhouette score 계산
    labels = list(cluster.values())  # 군집화된 클러스터를 리스트로 변환
    silhouette = silhouette_score(X, labels, metric='euclidean')
    return silhouette

# 군집의 개수를 지정하는 함수
def your_choice(X):
    # 여기서 K값을 설정해야 합니다. 예를 들어 K=3으로 설정했다고 가정합니다.
    K = 3
    return md.KMeans(K=K).fit(X)

def main():
    X = pre.load_data()
    
    # 군집화 모델을 학습
    Kmeans = your_choice(X)
    
    # 학습된 KMeans 모델의 labels를 cluster 형태로 가져옵니다.
    cluster = {i: label for i, label in enumerate(Kmeans.labels_)}
    
    # silhouette 값 계산
    silhouette = silhouette_calculator(X, cluster)
    
    # 군집의 개수와 silhouette 값을 출력합니다. 
    K = len(set(Kmeans.labels_))  # 군집의 개수
    print('K = {}, silhouette 값: {}'.format(K, silhouette))
                

if __name__ == "__main__":
    main()
