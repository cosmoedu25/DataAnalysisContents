# 6장 Part 2: 차원 축소와 군집화
## 고차원 데이터의 숨겨진 패턴 찾기

### 학습 목표
이번 파트를 완료하면 다음과 같은 능력을 갖게 됩니다:
- 차원의 저주 문제를 이해하고 차원 축소의 필요성을 설명할 수 있습니다
- 주성분 분석(PCA)의 원리를 이해하고 구현할 수 있습니다
- t-SNE를 활용하여 고차원 데이터를 효과적으로 시각화할 수 있습니다
- K-평균 군집화와 계층적 군집화의 차이점을 이해하고 적용할 수 있습니다
- 차원 축소와 군집화를 결합하여 고객 세분화 프로젝트를 수행할 수 있습니다

---

### 6.7 차원의 저주와 차원 축소의 필요성

#### 🌌 차원의 저주란 무엇인가?

상상해보세요. 1차원 공간에서는 선 위의 점들 사이의 거리를 쉽게 측정할 수 있습니다. 2차원에서는 평면상의 거리, 3차원에서는 공간상의 거리를 계산할 수 있죠. 하지만 100차원, 1000차원 공간에서는 어떨까요?

**차원의 저주(Curse of Dimensionality)**는 차원이 증가할수록 데이터가 희박해지고, 거리 개념이 무의미해지며, 머신러닝 알고리즘의 성능이 급격히 떨어지는 현상을 말합니다.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

# 시각화를 위한 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 차원의 저주 시연 - 차원에 따른 거리 분포 변화
def demonstrate_curse_of_dimensionality():
    """차원이 증가할 때 점들 간 거리가 어떻게 변하는지 보여주는 함수"""
    dimensions = [2, 10, 50, 100]
    n_points = 1000
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for i, dim in enumerate(dimensions):
        # 각 차원에서 랜덤 점들 생성
        points = np.random.normal(0, 1, (n_points, dim))
        
        # 첫 번째 점과 나머지 점들 간의 거리 계산
        distances = np.sqrt(np.sum((points[1:] - points[0])**2, axis=1))
        
        # 히스토그램 그리기
        axes[i].hist(distances, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[i].set_title(f'{dim}차원 공간에서의 거리 분포')
        axes[i].set_xlabel('거리')
        axes[i].set_ylabel('빈도')
        
        # 통계 정보 표시
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        axes[i].axvline(mean_dist, color='red', linestyle='--', 
                       label=f'평균: {mean_dist:.2f}')
        axes[i].legend()
    
    plt.tight_layout()
    plt.suptitle('차원의 저주: 차원이 증가할수록 거리 차이가 줄어듭니다', 
                fontsize=14, y=1.02)
    plt.show()
    
    return dimensions

# 차원의 저주 시연 실행
print("🌌 차원의 저주 현상 관찰")
print("=" * 50)
dimensions = demonstrate_curse_of_dimensionality()
```

**왜 이 현상이 문제인가?**
- **거리 기반 알고리즘 무력화**: KNN, K-평균 등이 제대로 작동하지 않습니다
- **데이터 희소성**: 고차원에서는 모든 점들이 서로 멀리 떨어져 있습니다
- **과적합 위험 증가**: 특성이 많을수록 노이즈에 민감해집니다
- **계산 복잡도 증가**: 메모리 사용량과 계산 시간이 기하급수적으로 증가합니다

#### 🎯 차원 축소가 해결하는 문제들

```python
# 실제 고차원 데이터 예시 - 손글씨 숫자 데이터
digits = load_digits()
X_digits = digits.data  # 8x8 = 64차원
y_digits = digits.target

print(f"📊 손글씨 숫자 데이터 분석")
print("=" * 30)
print(f"원본 데이터 차원: {X_digits.shape}")
print(f"각 이미지 크기: 8x8 = {X_digits.shape[1]}픽셀")
print(f"클래스 수: {len(np.unique(y_digits))}개 (0~9)")

# 원본 이미지 몇 개 시각화
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
for i in range(10):
    row, col = i // 5, i % 5
    axes[row, col].imshow(X_digits[i].reshape(8, 8), cmap='gray')
    axes[row, col].set_title(f'숫자: {y_digits[i]}')
    axes[row, col].axis('off')

plt.suptitle('원본 손글씨 숫자 데이터 (64차원)', fontsize=14)
plt.tight_layout()
plt.show()
```

**차원 축소의 주요 목적**
1. **시각화**: 고차원 데이터를 2D/3D로 표현하여 패턴 발견
2. **노이즈 제거**: 중요하지 않은 특성들을 제거하여 신호 강화
3. **저장 공간 절약**: 데이터 압축으로 메모리 효율성 향상
4. **계산 속도 향상**: 차원이 줄어들면 알고리즘이 빨라집니다
5. **과적합 방지**: 불필요한 특성 제거로 일반화 성능 향상

---

### 6.8 주성분 분석(PCA) - 분산을 최대한 보존하는 압축

#### 🎯 PCA의 핵심 아이디어

PCA는 마치 그림자 게임과 같습니다. 3차원 물체를 벽에 비춘 그림자는 2차원이지만, 적절한 각도에서 비추면 원본의 특징을 최대한 많이 보존할 수 있습니다.

**PCA의 원리**
1. **분산이 가장 큰 방향**을 첫 번째 주성분으로 선택
2. 첫 번째 주성분과 **직교하면서** 분산이 가장 큰 방향을 두 번째 주성분으로 선택
3. 이 과정을 반복하여 필요한 만큼의 주성분을 찾습니다

```python
# PCA 적용 전 원본 데이터의 분산 분포 확인
print("🎯 PCA 적용 과정 단계별 분석")
print("=" * 40)

# 표준화 (PCA에서 매우 중요!)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_digits_scaled = scaler.fit_transform(X_digits)

print(f"표준화 전 특성별 분산 범위: {X_digits.var(axis=0).min():.2f} ~ {X_digits.var(axis=0).max():.2f}")
print(f"표준화 후 특성별 분산 범위: {X_digits_scaled.var(axis=0).min():.2f} ~ {X_digits_scaled.var(axis=0).max():.2f}")

# PCA 적용
pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_digits_scaled)

# 주성분별 설명 가능한 분산 비율
explained_variance_ratio = pca_full.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

print(f"\n첫 10개 주성분의 설명 분산 비율:")
for i in range(10):
    print(f"  PC{i+1}: {explained_variance_ratio[i]:.4f} ({explained_variance_ratio[i]*100:.2f}%)")

print(f"\n누적 분산 비율:")
print(f"  PC1~5: {cumulative_variance_ratio[4]:.4f} ({cumulative_variance_ratio[4]*100:.2f}%)")
print(f"  PC1~10: {cumulative_variance_ratio[9]:.4f} ({cumulative_variance_ratio[9]*100:.2f}%)")
print(f"  PC1~20: {cumulative_variance_ratio[19]:.4f} ({cumulative_variance_ratio[19]*100:.2f}%)")
```

**왜 표준화가 중요한가?**
- 각 특성의 스케일이 다르면 분산이 큰 특성이 PCA를 지배합니다
- 표준화를 통해 모든 특성을 동등하게 취급할 수 있습니다
- 픽셀 데이터의 경우 이미 0~16 범위로 정규화되어 있어 영향이 적지만, 일반적으로는 필수입니다

#### 📊 PCA 결과 시각화 및 해석

```python
# 분산 비율 시각화
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 1. 주성분별 설명 분산 비율 (처음 20개)
ax1.bar(range(1, 21), explained_variance_ratio[:20], alpha=0.7, color='skyblue')
ax1.set_xlabel('주성분 번호')
ax1.set_ylabel('설명 분산 비율')
ax1.set_title('주성분별 설명 분산 비율 (상위 20개)')
ax1.grid(True, alpha=0.3)

# 2. 누적 설명 분산 비율
ax2.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, 
         'o-', linewidth=2, markersize=4)
ax2.axhline(y=0.8, color='red', linestyle='--', label='80% 분산 보존')
ax2.axhline(y=0.9, color='orange', linestyle='--', label='90% 분산 보존')
ax2.axhline(y=0.95, color='green', linestyle='--', label='95% 분산 보존')
ax2.set_xlabel('주성분 개수')
ax2.set_ylabel('누적 설명 분산 비율')
ax2.set_title('누적 설명 분산 비율')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 적절한 주성분 개수 결정
n_components_80 = np.argmax(cumulative_variance_ratio >= 0.8) + 1
n_components_90 = np.argmax(cumulative_variance_ratio >= 0.9) + 1
n_components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1

print(f"📈 적절한 주성분 개수 선택 가이드")
print(f"  80% 분산 보존: {n_components_80}개 주성분 ({n_components_80/64*100:.1f}% 압축)")
print(f"  90% 분산 보존: {n_components_90}개 주성분 ({n_components_90/64*100:.1f}% 압축)")
print(f"  95% 분산 보존: {n_components_95}개 주성분 ({n_components_95/64*100:.1f}% 압축)")
```

**주성분 개수 선택 기준**
- **80% 분산 보존**: 빠른 탐색적 분석용
- **90% 분산 보존**: 일반적인 분석 및 시각화용
- **95% 분산 보존**: 정확성이 중요한 분석용

#### 🖼️ 2차원 PCA로 데이터 시각화

```python
# 2차원 PCA 적용
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_digits_scaled)

print(f"🖼️ 2차원 PCA 결과")
print(f"  설명 분산 비율: PC1={pca_2d.explained_variance_ratio_[0]:.3f}, PC2={pca_2d.explained_variance_ratio_[1]:.3f}")
print(f"  총 보존 분산: {sum(pca_2d.explained_variance_ratio_):.3f} ({sum(pca_2d.explained_variance_ratio_)*100:.1f}%)")

# 2차원 시각화
plt.figure(figsize=(12, 10))
colors = plt.cm.tab10(np.linspace(0, 1, 10))

for digit in range(10):
    mask = y_digits == digit
    plt.scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1], 
               c=[colors[digit]], label=f'숫자 {digit}', 
               alpha=0.6, s=30)

plt.xlabel(f'첫 번째 주성분 (분산: {pca_2d.explained_variance_ratio_[0]:.3f})')
plt.ylabel(f'두 번째 주성분 (분산: {pca_2d.explained_variance_ratio_[1]:.3f})')
plt.title('PCA를 이용한 손글씨 숫자 데이터 2차원 시각화')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**2차원 PCA 시각화에서 관찰할 점**
- 비슷한 숫자들끼리 군집을 형성하는지 확인
- 겹치는 영역이 많다면 2차원으로는 구분이 어려운 데이터임을 의미
- 각 축이 나타내는 분산의 비율 확인

#### 🔄 PCA 역변환 - 압축된 데이터 복원

```python
# 다양한 주성분 개수로 데이터 복원해보기
n_components_list = [5, 10, 20, 30]

fig, axes = plt.subplots(2, len(n_components_list), figsize=(16, 8))

# 원본 이미지 (첫 번째 행)
original_image = X_digits[0].reshape(8, 8)
for i, n_comp in enumerate(n_components_list):
    axes[0, i].imshow(original_image, cmap='gray')
    axes[0, i].set_title(f'원본 (64차원)')
    axes[0, i].axis('off')

# 복원된 이미지들 (두 번째 행)
for i, n_comp in enumerate(n_components_list):
    # PCA 적용 및 역변환
    pca_temp = PCA(n_components=n_comp)
    X_reduced = pca_temp.fit_transform(X_digits_scaled)
    X_restored = pca_temp.inverse_transform(X_reduced)
    
    # 표준화 역변환
    X_restored = scaler.inverse_transform(X_restored)
    
    # 이미지 복원
    restored_image = X_restored[0].reshape(8, 8)
    axes[1, i].imshow(restored_image, cmap='gray')
    
    # 복원 품질 계산
    mse = np.mean((X_digits[0] - X_restored[0])**2)
    variance_preserved = np.sum(pca_temp.explained_variance_ratio_)
    
    axes[1, i].set_title(f'{n_comp}차원 복원\n(MSE: {mse:.2f}, 분산: {variance_preserved:.2f})')
    axes[1, i].axis('off')

plt.suptitle('PCA 차원에 따른 이미지 복원 품질 비교', fontsize=14)
plt.tight_layout()
plt.show()
```

**복원 품질 해석**
- **MSE(평균제곱오차)**: 낮을수록 원본과 유사
- **분산 보존 비율**: 높을수록 정보 손실이 적음
- 적절한 균형점을 찾는 것이 중요합니다

---

### 6.9 t-SNE - 비선형 관계를 포착하는 시각화의 마법사

#### 🌀 t-SNE의 특별한 능력

PCA는 선형 변환만 가능하지만, **t-SNE(t-distributed Stochastic Neighbor Embedding)**는 비선형 관계도 포착할 수 있습니다. 마치 구겨진 종이를 펼치듯이, 복잡하게 얽힌 고차원 데이터의 구조를 2차원에서 보여줍니다.

```python
# t-SNE 적용 (시간이 오래 걸리므로 데이터 일부만 사용)
print("🌀 t-SNE vs PCA 비교 분석")
print("=" * 40)

# 계산 시간 단축을 위해 샘플 일부만 사용
n_samples = 1000
sample_indices = np.random.choice(len(X_digits), n_samples, replace=False)
X_sample = X_digits_scaled[sample_indices]
y_sample = y_digits[sample_indices]

# PCA 2차원 변환
pca_2d = PCA(n_components=2)
X_pca_sample = pca_2d.fit_transform(X_sample)

# t-SNE 2차원 변환 (다양한 perplexity 값으로 실험)
tsne_params = [
    {'perplexity': 5, 'learning_rate': 200},
    {'perplexity': 30, 'learning_rate': 200},
    {'perplexity': 50, 'learning_rate': 200}
]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# PCA 결과 시각화
ax = axes[0, 0]
for digit in range(10):
    mask = y_sample == digit
    ax.scatter(X_pca_sample[mask, 0], X_pca_sample[mask, 1], 
               c=[colors[digit]], label=f'{digit}', alpha=0.6, s=20)
ax.set_title('PCA (선형 차원 축소)')
ax.set_xlabel('첫 번째 주성분')
ax.set_ylabel('두 번째 주성분')
ax.legend()
ax.grid(True, alpha=0.3)

# t-SNE 결과들 시각화
for i, params in enumerate(tsne_params):
    print(f"t-SNE 계산 중... (perplexity={params['perplexity']})")
    
    tsne = TSNE(n_components=2, random_state=42, 
                perplexity=params['perplexity'], 
                learning_rate=params['learning_rate'],
                n_iter=1000)
    X_tsne = tsne.fit_transform(X_sample)
    
    row = (i + 1) // 2
    col = (i + 1) % 2
    ax = axes[row, col]
    
    for digit in range(10):
        mask = y_sample == digit
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                   c=[colors[digit]], label=f'{digit}', alpha=0.6, s=20)
    
    ax.set_title(f't-SNE (perplexity={params["perplexity"]})')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend()

plt.tight_layout()
plt.show()
```

**t-SNE의 핵심 특징**
- **비선형 변환**: 곡선이나 복잡한 구조도 포착 가능
- **지역 구조 보존**: 가까운 점들은 가깝게, 먼 점들은 멀리 배치
- **클러스터 강조**: 같은 그룹끼리 더욱 명확하게 분리
- **확률적 방법**: 실행할 때마다 결과가 약간씩 다를 수 있음

#### ⚙️ t-SNE 하이퍼파라미터 이해하기

```python
# t-SNE 하이퍼파라미터의 영향 분석
print("⚙️ t-SNE 하이퍼파라미터 효과 분석")
print("=" * 40)

# 작은 데이터셋으로 빠른 실험
X_small = X_sample[:300]
y_small = y_sample[:300]

# 다양한 하이퍼파라미터 조합 테스트
param_combinations = [
    {'perplexity': 5, 'learning_rate': 10, 'title': '낮은 perplexity + 낮은 학습률'},
    {'perplexity': 5, 'learning_rate': 1000, 'title': '낮은 perplexity + 높은 학습률'},
    {'perplexity': 50, 'learning_rate': 10, 'title': '높은 perplexity + 낮은 학습률'},
    {'perplexity': 50, 'learning_rate': 1000, 'title': '높은 perplexity + 높은 학습률'}
]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

for i, params in enumerate(param_combinations):
    print(f"실험 {i+1}: {params['title']}")
    
    tsne = TSNE(n_components=2, random_state=42,
                perplexity=params['perplexity'],
                learning_rate=params['learning_rate'],
                n_iter=500)  # 빠른 실험을 위해 반복 수 줄임
    
    X_tsne = tsne.fit_transform(X_small)
    
    # 시각화
    ax = axes[i]
    for digit in range(10):
        mask = y_small == digit
        if np.any(mask):  # 해당 숫자가 있는 경우만
            ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                      c=[colors[digit]], label=f'{digit}', alpha=0.7, s=30)
    
    ax.set_title(params['title'])
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend()

plt.tight_layout()
plt.show()

print("\n📊 하이퍼파라미터 효과 정리:")
print("  • Perplexity가 낮으면: 지역적 구조에 집중, 작은 클러스터 형성")
print("  • Perplexity가 높으면: 전역적 구조 고려, 큰 그림 파악")
print("  • Learning rate가 낮으면: 안정적이지만 느린 수렴")
print("  • Learning rate가 높으면: 빠르지만 불안정할 수 있음")
```

**t-SNE 사용 시 주의사항**
1. **계산 시간**: PCA보다 훨씬 오래 걸립니다 (O(n²))
2. **재현성**: 확률적 알고리즘이므로 random_state 설정 필요
3. **거리 해석 주의**: t-SNE에서의 거리는 원본 공간의 거리와 다를 수 있습니다
4. **전역 구조**: 전체적인 구조보다는 지역적 클러스터에 집중합니다

---

### 6.10 군집화 알고리즘 - 비슷한 것들끼리 묶기

#### 🎯 K-평균 군집화 - 중심점 기반 클러스터링

군집화는 라벨이 없는 데이터에서 비슷한 특성을 가진 그룹들을 찾는 비지도 학습 방법입니다. K-평균은 가장 널리 사용되는 군집화 알고리즘입니다.

```python
# K-평균 군집화 적용
print("🎯 K-평균 군집화 분석")
print("=" * 30)

# PCA로 차원 축소된 데이터 사용 (2차원)
X_for_clustering = X_pca_2d

# 다양한 K 값으로 군집화 수행
k_values = range(2, 12)
inertias = []
silhouette_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_for_clustering)
    
    # WCSS (Within-Cluster Sum of Squares) 계산
    inertias.append(kmeans.inertia_)
    
    # 실루엣 점수 계산
    sil_score = silhouette_score(X_for_clustering, cluster_labels)
    silhouette_scores.append(sil_score)
    
    print(f"K={k}: WCSS={kmeans.inertia_:.2f}, 실루엣 점수={sil_score:.3f}")

# 엘보우 방법과 실루엣 분석 시각화
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 1. 엘보우 방법
ax1.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('클러스터 수 (K)')
ax1.set_ylabel('WCSS (Within-Cluster Sum of Squares)')
ax1.set_title('엘보우 방법 - 최적 K 찾기')
ax1.grid(True, alpha=0.3)

# 엘보우 포인트 표시 (간단한 방법)
# 기울기 변화가 가장 큰 지점 찾기
deltas = np.diff(inertias)
second_deltas = np.diff(deltas)
elbow_k = k_values[np.argmax(second_deltas) + 1]
ax1.axvline(x=elbow_k, color='red', linestyle='--', 
           label=f'엘보우 포인트 (K={elbow_k})')
ax1.legend()

# 2. 실루엣 분석
ax2.plot(k_values, silhouette_scores, 'ro-', linewidth=2, markersize=8)
ax2.set_xlabel('클러스터 수 (K)')
ax2.set_ylabel('실루엣 점수')
ax2.set_title('실루엣 분석 - 최적 K 찾기')
ax2.grid(True, alpha=0.3)

# 최고 실루엣 점수 표시
best_k = k_values[np.argmax(silhouette_scores)]
ax2.axvline(x=best_k, color='red', linestyle='--',
           label=f'최고 실루엣 점수 (K={best_k})')
ax2.legend()

plt.tight_layout()
plt.show()

print(f"\n📊 최적 클러스터 수 추천:")
print(f"  엘보우 방법: K = {elbow_k}")
print(f"  실루엣 분석: K = {best_k} (점수: {max(silhouette_scores):.3f})")
```

**클러스터 수 결정 방법**
- **엘보우 방법**: WCSS가 급격히 감소하다가 완만해지는 지점
- **실루엣 분석**: 클러스터 내 응집도와 클러스터 간 분리도를 종합 평가
- **도메인 지식**: 비즈니스 관점에서 의미 있는 그룹 수

#### 🌳 계층적 군집화 - 덴드로그램으로 구조 파악

```python
# 계층적 군집화 수행
print("🌳 계층적 군집화 분석")
print("=" * 30)

# 계산 시간을 위해 샘플 일부만 사용
n_samples_hier = 100
sample_idx = np.random.choice(len(X_for_clustering), n_samples_hier, replace=False)
X_hier = X_for_clustering[sample_idx]
y_hier = y_digits[sample_idx]

# 연결 방식별 계층적 군집화
linkage_methods = ['ward', 'complete', 'average', 'single']

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

for i, method in enumerate(linkage_methods):
    print(f"연결 방식: {method}")
    
    # 덴드로그램 생성
    linkage_matrix = linkage(X_hier, method=method)
    
    ax = axes[i]
    dendrogram(linkage_matrix, ax=ax, truncate_mode='level', p=5)
    ax.set_title(f'{method.title()} 연결법 덴드로그램')
    ax.set_xlabel('샘플 인덱스 또는 클러스터 크기')
    ax.set_ylabel('거리')

plt.tight_layout()
plt.show()

# 계층적 군집화로 실제 클러스터링 수행
print("\n📊 계층적 군집화 결과 비교:")
for method in linkage_methods:
    agg_clustering = AgglomerativeClustering(n_clusters=10, linkage=method)
    hier_labels = agg_clustering.fit_predict(X_hier)
    
    # 실루엣 점수 계산
    sil_score = silhouette_score(X_hier, hier_labels)
    print(f"  {method.title()} 연결법: 실루엣 점수 = {sil_score:.3f}")
```

**연결 방식의 특징**
- **Ward**: 클러스터 내 분산을 최소화 (균형 잡힌 클러스터)
- **Complete**: 클러스터 간 최대 거리 최소화 (조밀한 클러스터)
- **Average**: 클러스터 간 평균 거리 최소화 (중간적 특성)
- **Single**: 클러스터 간 최소 거리 최소화 (연결된 구조)

#### 🎨 군집화 결과 시각화 및 해석

```python
# 최적 K값으로 K-평균 군집화 수행 및 시각화
optimal_k = best_k
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X_for_clustering)

# 군집화 결과와 실제 라벨 비교
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# 1. 실제 숫자 라벨
for digit in range(10):
    mask = y_digits == digit
    ax1.scatter(X_for_clustering[mask, 0], X_for_clustering[mask, 1],
               c=[colors[digit]], label=f'숫자 {digit}', alpha=0.6, s=30)

ax1.set_title('실제 숫자 라벨 (정답)')
ax1.set_xlabel('첫 번째 주성분')
ax1.set_ylabel('두 번째 주성분')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# 2. K-평균 군집화 결과
cluster_colors = plt.cm.Set3(np.linspace(0, 1, optimal_k))
for cluster in range(optimal_k):
    mask = cluster_labels == cluster
    ax2.scatter(X_for_clustering[mask, 0], X_for_clustering[mask, 1],
               c=[cluster_colors[cluster]], label=f'클러스터 {cluster}', 
               alpha=0.6, s=30)

# 클러스터 중심점 표시
centers = kmeans_final.cluster_centers_
ax2.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', 
           s=200, linewidths=3, label='중심점')

ax2.set_title(f'K-평균 군집화 결과 (K={optimal_k})')
ax2.set_xlabel('첫 번째 주성분')
ax2.set_ylabel('두 번째 주성분')
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

# 군집화 품질 평가
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ari_score = adjusted_rand_score(y_digits, cluster_labels)
nmi_score = normalized_mutual_info_score(y_digits, cluster_labels)
sil_score = silhouette_score(X_for_clustering, cluster_labels)

print(f"📊 군집화 성능 평가:")
print(f"  조정 랜드 지수 (ARI): {ari_score:.3f}")
print(f"  정규화 상호정보 (NMI): {nmi_score:.3f}")
print(f"  실루엣 점수: {sil_score:.3f}")
print(f"\n해석:")
print(f"  ARI/NMI는 실제 라벨과의 일치도를 측정 (1에 가까울수록 좋음)")
print(f"  실루엣 점수는 클러스터의 응집도와 분리도를 측정 (1에 가까울수록 좋음)")
```

**군집화 성능 지표 해석**
- **ARI (Adjusted Rand Index)**: 우연히 일치할 확률을 보정한 정확도
- **NMI (Normalized Mutual Information)**: 정보 이론 기반 유사도 측정
- **실루엣 점수**: 클러스터 내부 응집도와 외부 분리도의 균형

---

### 6.11 차원 축소 + 군집화 파이프라인 구축

#### 🔗 통합 접근법의 장점

차원 축소와 군집화를 결합하면 시너지 효과를 얻을 수 있습니다:
1. **노이즈 제거**: PCA가 노이즈를 걸러내어 군집화 품질 향상
2. **계산 효율성**: 낮은 차원에서 군집화하여 속도 향상
3. **시각화**: 2D/3D로 축소하여 군집 결과를 쉽게 관찰

```python
# 차원 축소 + 군집화 파이프라인 구축
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

print("🔗 차원 축소 + 군집화 파이프라인")
print("=" * 40)

# 다양한 차원 축소 방법 + K-평균 파이프라인 비교
pipelines = {
    'PCA + K-means': Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95)),  # 95% 분산 보존
        ('kmeans', KMeans(n_clusters=10, random_state=42))
    ]),
    
    'PCA_2D + K-means': Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=2)),
        ('kmeans', KMeans(n_clusters=10, random_state=42))
    ]),
    
    'Raw + K-means': Pipeline([
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(n_clusters=10, random_state=42))
    ])
}

# 파이프라인별 성능 비교
results = {}
for name, pipeline in pipelines.items():
    print(f"\n{name} 수행 중...")
    
    # 파이프라인 실행
    cluster_labels = pipeline.fit_predict(X_digits)
    
    # 성능 평가
    if 'PCA_2D' in name:
        # 2D PCA의 경우 차원 축소된 데이터로 실루엣 점수 계산
        X_transformed = pipeline.named_steps['pca'].transform(
            pipeline.named_steps['scaler'].transform(X_digits)
        )
        sil_score = silhouette_score(X_transformed, cluster_labels)
    else:
        sil_score = silhouette_score(X_digits, cluster_labels)
    
    ari_score = adjusted_rand_score(y_digits, cluster_labels)
    nmi_score = normalized_mutual_info_score(y_digits, cluster_labels)
    
    results[name] = {
        'ARI': ari_score,
        'NMI': nmi_score,
        'Silhouette': sil_score,
        'Labels': cluster_labels
    }
    
    print(f"  ARI: {ari_score:.3f}")
    print(f"  NMI: {nmi_score:.3f}")
    print(f"  실루엣: {sil_score:.3f}")

# 결과 비교 시각화
metrics = ['ARI', 'NMI', 'Silhouette']
methods = list(results.keys())

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, metric in enumerate(metrics):
    scores = [results[method][metric] for method in methods]
    bars = axes[i].bar(methods, scores, alpha=0.7, 
                       color=['skyblue', 'lightgreen', 'salmon'])
    
    axes[i].set_title(f'{metric} 점수 비교')
    axes[i].set_ylabel(f'{metric} 점수')
    axes[i].set_ylim(0, max(scores) * 1.1)
    
    # 막대 위에 점수 표시
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
    
    plt.setp(axes[i].get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.show()

# 최적 방법 추천
best_method = max(results.keys(), key=lambda x: results[x]['ARI'])
print(f"\n🏆 최고 성능 방법: {best_method}")
print(f"   종합 점수 - ARI: {results[best_method]['ARI']:.3f}, "
      f"NMI: {results[best_method]['NMI']:.3f}, "
      f"실루엣: {results[best_method]['Silhouette']:.3f}")
```

#### 🏢 실습 프로젝트: 고객 세분화 시스템 구축

이제 실제 비즈니스 문제인 고객 세분화를 해결해보겠습니다.

```python
# 가상의 고객 데이터 생성 (실제 비즈니스 시나리오 모방)
print("🏢 고객 세분화 프로젝트")
print("=" * 40)

# 고객 데이터 생성
np.random.seed(42)
n_customers = 2000

# 고객 특성 생성
customer_data = {
    'age': np.random.normal(40, 15, n_customers),
    'income': np.random.lognormal(10, 0.5, n_customers),
    'spending_score': np.random.normal(50, 25, n_customers),
    'purchase_frequency': np.random.poisson(5, n_customers),
    'online_activity': np.random.exponential(2, n_customers),
    'loyalty_years': np.random.gamma(2, 2, n_customers),
    'support_calls': np.random.poisson(2, n_customers),
    'review_score': np.random.normal(4, 1, n_customers)
}

# 데이터프레임 생성
import pandas as pd

customer_df = pd.DataFrame(customer_data)

# 현실적인 범위로 조정
customer_df['age'] = np.clip(customer_df['age'], 18, 80)
customer_df['income'] = np.clip(customer_df['income'], 20000, 200000)
customer_df['spending_score'] = np.clip(customer_df['spending_score'], 0, 100)
customer_df['purchase_frequency'] = np.clip(customer_df['purchase_frequency'], 0, 50)
customer_df['online_activity'] = np.clip(customer_df['online_activity'], 0, 20)
customer_df['loyalty_years'] = np.clip(customer_df['loyalty_years'], 0, 20)
customer_df['support_calls'] = np.clip(customer_df['support_calls'], 0, 20)
customer_df['review_score'] = np.clip(customer_df['review_score'], 1, 5)

print("고객 데이터 기본 정보:")
print(customer_df.describe())

# 특성들 간의 상관관계 확인
plt.figure(figsize=(12, 10))
correlation_matrix = customer_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5)
plt.title('고객 특성 간 상관관계')
plt.tight_layout()
plt.show()
```

**고객 데이터 특성 설명**
- **age**: 고객 연령
- **income**: 연간 소득
- **spending_score**: 소비 성향 점수 (0-100)
- **purchase_frequency**: 월평균 구매 횟수
- **online_activity**: 온라인 활동 점수
- **loyalty_years**: 고객 충성도 연수
- **support_calls**: 고객 지원 문의 횟수
- **review_score**: 평균 리뷰 점수 (1-5)

#### 🔍 고객 세분화 파이프라인 구현

```python
# 고객 세분화를 위한 완전한 파이프라인 구현
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

print("🔍 고객 세분화 파이프라인 구현")
print("=" * 40)

# 데이터 준비
X_customers = customer_df.values

# 1단계: 최적 클러스터 수 찾기
# 여러 차원 축소 방법으로 최적 K 탐색
pca_full = PCA()
X_customers_scaled = StandardScaler().fit_transform(X_customers)
X_pca_full = pca_full.fit_transform(X_customers_scaled)

# 95% 분산을 보존하는 주성분 개수 확인
cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
n_components_95 = np.argmax(cumsum_var >= 0.95) + 1
print(f"95% 분산 보존을 위한 주성분 개수: {n_components_95}")

# PCA 차원 축소 후 최적 K 찾기
pca_reduced = PCA(n_components=n_components_95)
X_pca_reduced = pca_reduced.fit_transform(X_customers_scaled)

k_range = range(2, 11)
silhouette_scores_customer = []
inertias_customer = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca_reduced)
    
    sil_score = silhouette_score(X_pca_reduced, labels)
    silhouette_scores_customer.append(sil_score)
    inertias_customer.append(kmeans.inertia_)

# 최적 K 선택
optimal_k_customer = k_range[np.argmax(silhouette_scores_customer)]
print(f"실루엣 분석 기반 최적 K: {optimal_k_customer}")

# 2단계: 최종 고객 세분화 모델 구축
customer_segmentation_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=n_components_95)),
    ('kmeans', KMeans(n_clusters=optimal_k_customer, random_state=42))
])

# 고객 세분화 수행
customer_segments = customer_segmentation_pipeline.fit_predict(X_customers)

# 세분화 결과를 원본 데이터에 추가
customer_df['segment'] = customer_segments

print(f"\n📊 고객 세분화 결과:")
print(f"총 고객 수: {len(customer_df)}")
for segment in range(optimal_k_customer):
    count = np.sum(customer_segments == segment)
    percentage = count / len(customer_df) * 100
    print(f"  세그먼트 {segment}: {count}명 ({percentage:.1f}%)")
```

#### 📈 고객 세그먼트 특성 분석

```python
# 각 세그먼트의 특성 분석
print("📈 고객 세그먼트별 특성 분석")
print("=" * 40)

# 세그먼트별 평균 특성
segment_analysis = customer_df.groupby('segment').agg({
    'age': 'mean',
    'income': 'mean', 
    'spending_score': 'mean',
    'purchase_frequency': 'mean',
    'online_activity': 'mean',
    'loyalty_years': 'mean',
    'support_calls': 'mean',
    'review_score': 'mean'
}).round(2)

print("\n세그먼트별 평균 특성:")
print(segment_analysis)

# 세그먼트별 특성을 히트맵으로 시각화
plt.figure(figsize=(12, 8))

# 정규화를 위해 각 특성을 0-1 스케일로 변환
segment_analysis_normalized = segment_analysis.copy()
for col in segment_analysis.columns:
    min_val = segment_analysis[col].min()
    max_val = segment_analysis[col].max()
    segment_analysis_normalized[col] = (segment_analysis[col] - min_val) / (max_val - min_val)

sns.heatmap(segment_analysis_normalized.T, annot=True, cmap='RdYlBu_r', 
            center=0.5, square=True, linewidths=0.5,
            xticklabels=[f'세그먼트 {i}' for i in range(optimal_k_customer)],
            yticklabels=segment_analysis.columns)
plt.title('고객 세그먼트별 특성 프로필 (정규화된 값)')
plt.tight_layout()
plt.show()

# 세그먼트 해석 및 네이밍
segment_names = {}
segment_descriptions = {}

for i in range(optimal_k_customer):
    segment_data = segment_analysis.iloc[i]
    
    # 간단한 휴리스틱으로 세그먼트 특성 파악
    if segment_data['income'] > segment_analysis['income'].mean():
        if segment_data['spending_score'] > segment_analysis['spending_score'].mean():
            segment_names[i] = "VIP 고객"
            segment_descriptions[i] = "고소득, 고소비 성향의 우수 고객"
        else:
            segment_names[i] = "잠재 고객"
            segment_descriptions[i] = "고소득이지만 소비 성향이 낮은 개발 가능 고객"
    else:
        if segment_data['spending_score'] > segment_analysis['spending_score'].mean():
            segment_names[i] = "가치 고객"
            segment_descriptions[i] = "소득 대비 높은 소비 성향의 충성 고객"
        else:
            segment_names[i] = "일반 고객"
            segment_descriptions[i] = "평균적인 소득과 소비 패턴의 일반 고객"

print("\n🏷️ 세그먼트 해석 및 네이밍:")
for i in range(optimal_k_customer):
    print(f"  세그먼트 {i}: {segment_names[i]}")
    print(f"    → {segment_descriptions[i]}")
    print()

# 2D 시각화를 위한 PCA 적용
pca_2d_customer = PCA(n_components=2)
X_customer_2d = pca_2d_customer.fit_transform(X_customers_scaled)

plt.figure(figsize=(12, 8))
colors_customer = plt.cm.Set1(np.linspace(0, 1, optimal_k_customer))

for i in range(optimal_k_customer):
    mask = customer_segments == i
    plt.scatter(X_customer_2d[mask, 0], X_customer_2d[mask, 1],
                c=[colors_customer[i]], label=f'{segment_names[i]} (세그먼트 {i})',
                alpha=0.6, s=30)

plt.xlabel(f'첫 번째 주성분 (분산: {pca_2d_customer.explained_variance_ratio_[0]:.3f})')
plt.ylabel(f'두 번째 주성분 (분산: {pca_2d_customer.explained_variance_ratio_[1]:.3f})')
plt.title('고객 세분화 결과 (PCA 2D 시각화)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

#### 💼 비즈니스 인사이트 및 액션 플랜

```python
# 세그먼트별 비즈니스 전략 수립
print("💼 세그먼트별 비즈니스 전략")
print("=" * 40)

# 각 세그먼트의 수익성 추정 (간단한 모델)
customer_df['estimated_clv'] = (
    customer_df['income'] * 0.001 *  # 소득의 0.1%를 연간 구매로 가정
    customer_df['spending_score'] * 0.01 *  # 소비 성향 반영
    customer_df['loyalty_years'] * 0.5  # 충성도 반영
)

# 세그먼트별 평균 CLV 계산
segment_clv = customer_df.groupby('segment')['estimated_clv'].agg(['mean', 'sum', 'count'])
segment_clv.columns = ['평균_CLV', '총_CLV', '고객수']
segment_clv = segment_clv.round(2)

print("세그먼트별 고객 생애 가치 (CLV) 분석:")
print(segment_clv)

# 비즈니스 전략 추천
print("\n🎯 세그먼트별 추천 전략:")
for i in range(optimal_k_customer):
    name = segment_names[i]
    clv = segment_clv.loc[i, '평균_CLV']
    count = int(segment_clv.loc[i, '고객수'])
    
    print(f"\n{name} (세그먼트 {i}) - {count}명, 평균 CLV: ${clv:,.0f}")
    
    if "VIP" in name:
        print("  🌟 전략: VIP 프로그램, 개인 맞춤 서비스, 프리미엄 혜택")
        print("  📈 목표: 충성도 유지, 추천 확산, 업셀링")
    elif "잠재" in name:
        print("  🎯 전략: 타겟 마케팅, 특별 할인, 체험 이벤트")
        print("  📈 목표: 구매 전환, 소비 패턴 활성화")
    elif "가치" in name:
        print("  💝 전략: 리워드 프로그램, 충성도 포인트, 정기 혜택")
        print("  📈 목표: 장기 관계 유지, 소비 증대")
    else:
        print("  🔄 전략: 기본 서비스 개선, 가격 경쟁력, 접근성 향상")
        print("  📈 목표: 서비스 만족도 향상, 점진적 업그레이드")

# 우선순위 매트릭스 생성
plt.figure(figsize=(10, 8))
for i in range(optimal_k_customer):
    x = segment_clv.loc[i, '평균_CLV']
    y = segment_clv.loc[i, '고객수']
    plt.scatter(x, y, s=200, c=[colors_customer[i]], alpha=0.7)
    plt.annotate(f'{segment_names[i]}\n(세그먼트 {i})', 
                (x, y), xytext=(5, 5), textcoords='offset points')

plt.xlabel('평균 고객 생애 가치 (CLV)')
plt.ylabel('고객 수')
plt.title('고객 세그먼트 우선순위 매트릭스')
plt.grid(True, alpha=0.3)

# 사분면 표시
plt.axhline(y=customer_df.groupby('segment').size().mean(), color='red', 
           linestyle='--', alpha=0.5, label='평균 고객 수')
plt.axvline(x=segment_clv['평균_CLV'].mean(), color='red', 
           linestyle='--', alpha=0.5, label='평균 CLV')
plt.legend()
plt.tight_layout()
plt.show()
```

---

### 💪 직접 해보기 - 연습 문제

#### 🎯 연습 문제 1: PCA 주성분 수 최적화
다음 코드를 완성하여 적절한 주성분 개수를 찾아보세요.

```python
# TODO: 코드를 완성하세요
from sklearn.datasets import load_wine

# 와인 데이터셋 로드
wine = load_wine()
X_wine, y_wine = wine.data, wine.target

# TODO: 데이터 표준화

# TODO: PCA 적용하여 모든 주성분의 분산 비율 계산

# TODO: 80%, 90%, 95% 분산 보존을 위한 주성분 개수 계산

# TODO: 결과 시각화 (누적 분산 비율 그래프)

# TODO: 각 분산 보존 수준에서 2차원 시각화 및 비교
```

#### 🎯 연습 문제 2: t-SNE vs PCA 시각화 비교
```python
# TODO: 동일한 데이터셋에 대해 PCA와 t-SNE 결과 비교
from sklearn.datasets import make_swiss_roll

# 스위스 롤 데이터 생성 (3D 매니폴드)
X_swiss, color = make_swiss_roll(n_samples=1000, random_state=42)

# TODO: PCA로 2차원 축소

# TODO: t-SNE로 2차원 축소 (여러 perplexity 값 실험)

# TODO: 결과를 subplot으로 비교 시각화

# TODO: 각 방법의 장단점 분석
```

#### 🎯 연습 문제 3: 커스텀 고객 세분화 시스템
실제 비즈니스 시나리오를 가정하고 완전한 고객 세분화 시스템을 구축해보세요.

```python
# TODO: 다음 요구사항을 만족하는 시스템을 구현하세요

# 1. 데이터 생성: 5가지 이상의 고객 특성 포함
# 2. 전처리: 이상치 처리, 표준화, 결측치 처리
# 3. 차원 축소: PCA와 t-SNE 비교
# 4. 군집화: K-평균, 계층적 군집화 비교
# 5. 평가: 실루엣 분석, 엘보우 방법 적용
# 6. 시각화: 세그먼트별 특성 프로필 히트맵
# 7. 비즈니스 해석: 각 세그먼트의 특성과 전략 수립

# 추가 도전 과제:
# - 세그먼트 안정성 검증 (다른 랜덤 시드로 반복 실험)
# - 새로운 고객의 세그먼트 예측 함수 구현
# - 세그먼트 이동 분석 (시간에 따른 고객 세그먼트 변화)
```

---

### 📚 핵심 정리

#### ✨ 이번 파트에서 배운 내용

**1. 차원의 저주와 차원 축소의 필요성**
- 고차원 데이터에서 발생하는 문제점 이해
- 거리 개념의 무의미화와 데이터 희소성 문제
- 시각화, 노이즈 제거, 계산 효율성 향상의 필요성

**2. 주성분 분석(PCA)**
- 분산을 최대한 보존하는 선형 차원 축소
- 주성분별 설명 분산 비율 분석
- 적절한 주성분 개수 선택 방법
- 데이터 압축과 복원의 트레이드오프

**3. t-SNE (t-distributed Stochastic Neighbor Embedding)**
- 비선형 관계를 포착하는 시각화 기법
- perplexity와 learning_rate 하이퍼파라미터 조정
- PCA와의 차이점과 각각의 장단점
- 지역적 구조 보존과 클러스터 강조 효과

**4. 군집화 알고리즘**
- **K-평균**: 중심점 기반 클러스터링
- **계층적 군집화**: 덴드로그램을 통한 구조 파악
- 엘보우 방법과 실루엣 분석으로 최적 K 선택
- 다양한 연결 방식(Ward, Complete, Average, Single)

**5. 차원 축소 + 군집화 통합 파이프라인**
- scikit-learn Pipeline을 활용한 체계적 구현
- 전처리 → 차원 축소 → 군집화의 순차적 적용
- 성능 평가 지표 비교 (ARI, NMI, 실루엣 점수)

**6. 실전 고객 세분화 프로젝트**
- 실제 비즈니스 문제 해결 경험
- 세그먼트별 특성 분석과 프로필링
- 비즈니스 인사이트 도출과 전략 수립
- 고객 생애 가치(CLV) 기반 우선순위 매트릭스

#### 🎯 실무 적용 가이드라인

**언제 어떤 차원 축소 방법을 사용할까?**

- **PCA**: 전처리용, 노이즈 제거, 빠른 연산이 필요한 경우
- **t-SNE**: 탐색적 데이터 분석, 클러스터 시각화, 비선형 패턴 발견
- **두 방법 결합**: PCA로 사전 차원 축소 후 t-SNE 적용

**군집화 방법 선택 기준**

- **K-평균**: 구형 클러스터, 빠른 연산, 클러스터 수를 알고 있는 경우
- **계층적 군집화**: 클러스터 수 미정, 구조 파악 필요, 소규모 데이터
- **DBSCAN**: 임의 모양 클러스터, 노이즈 포함 데이터, 밀도 기반 분석

**주의사항**
- 표준화의 중요성: 특히 PCA에서 필수
- 하이퍼파라미터 조정: t-SNE의 perplexity, K-평균의 K
- 해석의 한계: t-SNE에서 거리는 상대적 의미만 가짐
- 재현성 확보: random_state 설정으로 일관된 결과 보장

---

### 🔮 다음 파트 미리보기

다음 Part 3에서는 **하이퍼파라미터 최적화**에 대해 학습합니다:

- 🎯 **그리드 서치**: 체계적인 파라미터 탐색
- 🚀 **랜덤 서치**: 효율적인 파라미터 샘플링
- 🧠 **베이지안 최적화**: 지능적인 파라미터 탐색
- 🔄 **교차 검증**: 신뢰할 수 있는 성능 평가
- 🛠️ **실습**: 자동화된 하이퍼파라미터 튜닝 시스템 구축

차원 축소와 군집화로 데이터의 숨겨진 패턴을 발견했다면, 이제 모델의 성능을 극대화하는 방법을 배워보겠습니다!

---

*"데이터는 고차원 공간에 숨어있지만, 지혜로운 차원 축소는 그 본질을 드러낸다." - 데이터 과학자의 명언*