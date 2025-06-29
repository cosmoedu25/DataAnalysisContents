# 5장 Part 1: 지도학습과 비지도학습의 개념
*머신러닝의 세계로 첫발을 내딛다*

## 학습 목표
이번 파트를 완료하면 다음을 할 수 있습니다:
- 머신러닝이 무엇인지, 왜 중요한지 설명할 수 있다
- 지도학습과 비지도학습의 근본적인 차이를 이해하고 구분할 수 있다
- 실제 문제 상황에서 어떤 학습 방식을 선택해야 할지 판단할 수 있다
- 각 학습 방식의 대표적인 알고리즘과 활용 사례를 알 수 있다

## 이번 파트 미리보기

여러분은 이미 머신러닝을 매일 경험하고 있습니다. 유튜브가 여러분이 좋아할 만한 영상을 추천하고, 스마트폰이 얼굴을 인식해 잠금을 해제하며, 온라인 쇼핑몰이 관심 있을 만한 상품을 보여주는 것 모두 머신러닝의 결과입니다.

이번 파트에서는 머신러닝의 가장 기본적이면서도 중요한 두 가지 접근 방식인 **지도학습**과 **비지도학습**을 탐험합니다. 마치 요리를 배우는 두 가지 방법과 같습니다. 레시피(정답)를 보며 따라하는 것이 지도학습이라면, 재료만 주어진 상태에서 창의적으로 요리를 만들어내는 것이 비지도학습입니다.

## 5.1.1 머신러닝이란 무엇인가?

### 전통적 프로그래밍 vs 머신러닝

먼저 간단한 예시로 시작해보겠습니다. 스팸 메일을 걸러내는 두 가지 방법을 비교해볼까요?

```python
# 전통적 프로그래밍 방식
def is_spam_traditional(email):
    """규칙 기반 스팸 필터"""
    spam_words = ['무료', '당첨', '클릭하세요', '대출', '비아그라']
    
    for word in spam_words:
        if word in email:
            return True
    return False

# 예시 이메일
email1 = "축하합니다! 무료 아이폰에 당첨되셨습니다!"
email2 = "안녕하세요, 내일 회의 시간 확인 부탁드립니다."

print(f"이메일 1: {email1}")
print(f"스팸 여부: {is_spam_traditional(email1)}")
print(f"\n이메일 2: {email2}")
print(f"스팸 여부: {is_spam_traditional(email2)}")
```

이 방식의 문제점은 무엇일까요? 새로운 유형의 스팸이 나타날 때마다 규칙을 계속 추가해야 합니다. 또한 정상 메일에 '무료'라는 단어가 들어가도 스팸으로 분류될 수 있습니다.

### 머신러닝 접근법

```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# 머신러닝을 위한 예시 데이터
data = pd.DataFrame({
    'email': [
        "무료 쿠폰을 받으세요",
        "회의 일정 확인 부탁드립니다",
        "대출 상담 무료",
        "프로젝트 진행 상황 공유",
        "축하합니다 당첨되셨습니다",
        "점심 메뉴 추천해주세요",
        "클릭하면 선물 증정",
        "내일 약속 시간 변경 가능한가요"
    ],
    'is_spam': [1, 0, 1, 0, 1, 0, 1, 0]  # 1: 스팸, 0: 정상
})

print("머신러닝을 위한 학습 데이터:")
print(data)
print("\n머신러닝의 핵심: 데이터로부터 패턴을 스스로 학습합니다!")
```

### 머신러닝의 정의와 특징

```python
# 머신러닝의 핵심 특징을 시각화
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 전통적 프로그래밍
ax1.text(0.5, 0.8, '입력 데이터', ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
ax1.text(0.5, 0.6, '+', ha='center', fontsize=16)
ax1.text(0.5, 0.4, '규칙(코드)', ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
ax1.text(0.5, 0.2, '↓', ha='center', fontsize=20)
ax1.text(0.5, 0.0, '결과', ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
ax1.set_title('전통적 프로그래밍', fontsize=14, fontweight='bold')
ax1.axis('off')

# 머신러닝
ax2.text(0.5, 0.8, '입력 데이터', ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
ax2.text(0.5, 0.6, '+', ha='center', fontsize=16)
ax2.text(0.5, 0.4, '정답(레이블)', ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
ax2.text(0.5, 0.2, '↓', ha='center', fontsize=20)
ax2.text(0.5, 0.0, '모델(패턴)', ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
ax2.set_title('머신러닝', fontsize=14, fontweight='bold')
ax2.axis('off')

plt.tight_layout()
plt.show()
```

## 5.1.2 머신러닝의 분류: 학습 방식에 따른 구분

### 머신러닝의 세 가지 주요 패러다임

```python
# 머신러닝 유형별 특징 정리
ml_types = pd.DataFrame({
    '학습 유형': ['지도학습', '비지도학습', '강화학습'],
    '데이터 특징': ['정답이 있음', '정답이 없음', '보상/벌칙이 있음'],
    '목표': ['예측/분류', '패턴 발견/그룹화', '최적 행동 학습'],
    '예시': ['스팸 분류, 집값 예측', '고객 그룹화, 추천 시스템', '게임 AI, 로봇 제어']
})

print("머신러닝의 주요 학습 방식:")
print(ml_types.to_string(index=False))
```

### 지도학습과 비지도학습의 직관적 이해

```python
# 실생활 비유로 이해하기
learning_analogies = {
    "지도학습": {
        "비유": "선생님과 함께하는 수학 문제 풀이",
        "과정": [
            "1. 문제와 정답을 함께 봅니다",
            "2. 풀이 패턴을 학습합니다",
            "3. 새로운 문제를 스스로 풉니다"
        ],
        "실제 예시": "사진 속 고양이/개 구분하기"
    },
    "비지도학습": {
        "비유": "처음 보는 과일들을 분류하기",
        "과정": [
            "1. 여러 과일이 섞여 있습니다",
            "2. 비슷한 특징끼리 그룹을 만듭니다",
            "3. 각 그룹의 특징을 파악합니다"
        ],
        "실제 예시": "고객들을 구매 패턴으로 그룹화"
    }
}

for learning_type, info in learning_analogies.items():
    print(f"\n{'='*50}")
    print(f"{learning_type}")
    print(f"비유: {info['비유']}")
    print("과정:")
    for step in info['과정']:
        print(f"  {step}")
    print(f"실제 예시: {info['실제 예시']}")

## 5.1.3 지도학습 깊이 알아보기

### 지도학습의 두 가지 주요 과제

```python
# 지도학습의 분류: 분류 vs 회귀
supervised_tasks = {
    "분류(Classification)": {
        "정의": "데이터를 미리 정의된 카테고리로 분류",
        "출력": "이산적인 클래스 (예: A, B, C 또는 0, 1)",
        "예시": ["스팸/정상 메일", "암 진단(양성/악성)", "붓꽃 종류 분류"],
        "평가지표": ["정확도", "정밀도", "재현율"]
    },
    "회귀(Regression)": {
        "정의": "연속적인 수치를 예측",
        "출력": "연속적인 값 (예: 153.7, 2,345,000)",
        "예시": ["주택 가격", "주식 가격", "온도 예측"],
        "평가지표": ["평균제곱오차(MSE)", "결정계수(R²)"]
    }
}

# 시각적으로 표현
for task_type, details in supervised_tasks.items():
    print(f"\n📊 {task_type}")
    print(f"정의: {details['정의']}")
    print(f"출력 형태: {details['출력']}")
    print("예시:")
    for example in details['예시']:
        print(f"  • {example}")
```

### 지도학습 실습: 간단한 분류 문제

```python
# 과일 분류 예제: 크기와 무게로 사과/오렌지 구분하기
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# 데이터 생성
np.random.seed(42)
n_samples = 100

# 사과 데이터 (크기는 작고 무게는 가벼움)
apple_size = np.random.normal(7, 0.5, n_samples//2)
apple_weight = np.random.normal(150, 20, n_samples//2)

# 오렌지 데이터 (크기는 크고 무게는 무거움)
orange_size = np.random.normal(9, 0.5, n_samples//2)
orange_weight = np.random.normal(200, 20, n_samples//2)

# 데이터 합치기
X = np.column_stack([
    np.concatenate([apple_size, orange_size]),
    np.concatenate([apple_weight, orange_weight])
])
y = np.array(['사과']*50 + ['오렌지']*50)

# 시각화
plt.figure(figsize=(10, 6))
plt.scatter(apple_size, apple_weight, c='red', label='사과', alpha=0.6, s=100)
plt.scatter(orange_size, orange_weight, c='orange', label='오렌지', alpha=0.6, s=100)
plt.xlabel('크기 (cm)', fontsize=12)
plt.ylabel('무게 (g)', fontsize=12)
plt.title('과일 데이터: 지도학습을 위한 레이블이 있는 데이터', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 모델 학습
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# 예측 정확도
accuracy = model.score(X_test, y_test)
print(f"\n모델의 예측 정확도: {accuracy:.2%}")

# 새로운 과일 예측
new_fruit = np.array([[8, 170]])  # 크기 8cm, 무게 170g
prediction = model.predict(new_fruit)
print(f"새로운 과일 (크기: 8cm, 무게: 170g)의 예측: {prediction[0]}")
```

### 지도학습의 장단점 분석

```python
# 지도학습의 실무적 고려사항
supervised_pros_cons = pd.DataFrame({
    '장점': [
        '명확한 성능 측정 가능',
        '비즈니스 목표와 직접 연결',
        '결과 해석이 명확함',
        '검증된 알고리즘 다수 존재'
    ],
    '단점': [
        '레이블 데이터 수집 비용',
        '레이블의 품질이 중요',
        '편향된 레이블의 위험',
        '새로운 패턴 발견 제한'
    ],
    '실무 팁': [
        '초기에는 작은 데이터로 시작',
        '레이블 품질 검증 필수',
        '다양한 관점의 레이블러 활용',
        '주기적인 모델 업데이트'
    ]
})

print("지도학습 실무 가이드:")
for col in supervised_pros_cons.columns:
    print(f"\n{col}:")
    for item in supervised_pros_cons[col]:
        print(f"  ✓ {item}")
```

## 5.1.4 비지도학습 깊이 알아보기

### 비지도학습의 주요 과제들

```python
# 비지도학습의 대표적인 문제 유형
unsupervised_tasks = {
    "군집화(Clustering)": {
        "목적": "비슷한 데이터끼리 그룹으로 묶기",
        "알고리즘": ["K-means", "DBSCAN", "계층적 군집화"],
        "활용": "고객 세분화, 이미지 분할"
    },
    "차원 축소(Dimensionality Reduction)": {
        "목적": "데이터의 중요한 특징만 추출",
        "알고리즘": ["PCA", "t-SNE", "Autoencoder"],
        "활용": "데이터 시각화, 노이즈 제거"
    },
    "이상치 탐지(Anomaly Detection)": {
        "목적": "정상 패턴과 다른 데이터 찾기",
        "알고리즘": ["Isolation Forest", "One-Class SVM"],
        "활용": "사기 거래 탐지, 장비 고장 예측"
    }
}

for task, details in unsupervised_tasks.items():
    print(f"\n🔍 {task}")
    print(f"목적: {details['목적']}")
    print(f"대표 알고리즘: {', '.join(details['알고리즘'])}")
    print(f"활용 사례: {details['활용']}")
```

### 비지도학습 실습: 고객 군집화

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 온라인 쇼핑몰 고객 데이터 생성
np.random.seed(42)

# 3가지 고객 그룹 시뮬레이션
# 그룹 1: 자주 구매, 적은 금액 (일상용품 구매자)
group1_frequency = np.random.normal(20, 3, 30)
group1_amount = np.random.normal(50000, 10000, 30)

# 그룹 2: 가끔 구매, 큰 금액 (명품 구매자)
group2_frequency = np.random.normal(5, 2, 30)
group2_amount = np.random.normal(200000, 30000, 30)

# 그룹 3: 보통 구매, 보통 금액 (일반 고객)
group3_frequency = np.random.normal(12, 3, 40)
group3_amount = np.random.normal(100000, 20000, 40)

# 데이터 합치기
customer_data = np.vstack([
    np.column_stack([group1_frequency, group1_amount]),
    np.column_stack([group2_frequency, group2_amount]),
    np.column_stack([group3_frequency, group3_amount])
])

# 데이터 정규화
scaler = StandardScaler()
customer_data_scaled = scaler.fit_transform(customer_data)

# K-means 군집화
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(customer_data_scaled)

# 시각화
plt.figure(figsize=(10, 6))
scatter = plt.scatter(customer_data[:, 0], customer_data[:, 1], 
                     c=clusters, cmap='viridis', alpha=0.6, s=100)
plt.xlabel('월 평균 구매 횟수', fontsize=12)
plt.ylabel('월 평균 구매 금액 (원)', fontsize=12)
plt.title('비지도학습: 고객 군집화 결과', fontsize=14)
plt.colorbar(scatter, label='클러스터')

# 각 클러스터 중심점 표시
centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='*', 
           edgecolors='black', linewidth=2, label='클러스터 중심')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 각 클러스터 특성 분석
for i in range(3):
    cluster_data = customer_data[clusters == i]
    print(f"\n클러스터 {i} 특성:")
    print(f"  평균 구매 횟수: {cluster_data[:, 0].mean():.1f}회")
    print(f"  평균 구매 금액: {cluster_data[:, 1].mean():,.0f}원")
    print(f"  고객 수: {len(cluster_data)}명")
```

### 비지도학습의 실무 활용과 주의점

```python
# 비지도학습 활용 시 체크리스트
unsupervised_checklist = {
    "데이터 준비": [
        "스케일링이 필요한가? (거리 기반 알고리즘)",
        "이상치 처리를 했는가?",
        "특성 선택이 적절한가?"
    ],
    "모델 선택": [
        "클러스터 개수를 어떻게 정할 것인가?",
        "어떤 거리 측정 방법을 사용할 것인가?",
        "결과를 어떻게 검증할 것인가?"
    ],
    "결과 해석": [
        "각 그룹의 특성이 명확한가?",
        "비즈니스적으로 의미가 있는가?",
        "실행 가능한 인사이트인가?"
    ]
}

print("비지도학습 실무 체크리스트:")
for category, items in unsupervised_checklist.items():
    print(f"\n📋 {category}")
    for item in items:
        print(f"  □ {item}")
```

## 5.1.5 지도학습 vs 비지도학습: 어떤 것을 선택할까?

### 의사결정 트리로 접근법 선택하기

```python
# 머신러닝 접근법 선택 가이드
def choose_ml_approach(has_labels, goal, data_size, domain_knowledge):
    """
    머신러닝 접근법을 선택하는 의사결정 함수
    """
    if has_labels:
        if goal == "예측":
            if data_size >= 1000:
                return "지도학습 추천: 충분한 레이블 데이터로 예측 모델 구축"
            else:
                return "지도학습 가능하나 데이터 증강 고려"
        elif goal == "탐색":
            return "지도학습 + 비지도학습 병행: 레이블을 활용하되 새로운 패턴도 탐색"
    else:
        if domain_knowledge == "높음":
            return "비지도학습 추천: 도메인 지식으로 결과 해석 가능"
        else:
            return "탐색적 분석 먼저: 데이터 이해 후 접근법 재검토"
    
# 예시 시나리오
scenarios = [
    {"has_labels": True, "goal": "예측", "data_size": 5000, "domain_knowledge": "보통"},
    {"has_labels": False, "goal": "탐색", "data_size": 10000, "domain_knowledge": "높음"},
    {"has_labels": True, "goal": "탐색", "data_size": 500, "domain_knowledge": "낮음"}
]

print("머신러닝 접근법 선택 시뮬레이션:")
for i, scenario in enumerate(scenarios, 1):
    print(f"\n시나리오 {i}:")
    print(f"  • 레이블 유무: {'있음' if scenario['has_labels'] else '없음'}")
    print(f"  • 목표: {scenario['goal']}")
    print(f"  • 데이터 크기: {scenario['data_size']:,}개")
    print(f"  • 도메인 지식: {scenario['domain_knowledge']}")
    recommendation = choose_ml_approach(**scenario)
    print(f"  → 추천: {recommendation}")
```

### 실제 사례로 보는 선택 기준

```python
# 실제 비즈니스 사례별 접근법
real_cases = pd.DataFrame({
    '사례': [
        '신용카드 사기 탐지',
        '신규 고객 세분화',
        '제품 불량 검사',
        '웹사이트 사용자 행동 분석',
        '주식 가격 예측'
    ],
    '레이블': ['있음(사기/정상)', '없음', '있음(불량/정상)', '부분적', '있음(과거 가격)'],
    '선택한 방법': ['지도학습', '비지도학습', '지도학습', '혼합', '지도학습'],
    '이유': [
        '과거 사기 데이터 존재',
        '고객 그룹 미리 알 수 없음',
        '불량품 기준 명확',
        '일부만 전환 여부 알 수 있음',
        '과거 가격 데이터 풍부'
    ]
})

print("실제 비즈니스 사례 분석:")
print(real_cases.to_string(index=False))
```

## 5.1.6 미니 프로젝트: 학습 방식별 비교 실험

### 동일한 데이터로 두 가지 접근법 비교

```python
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier

# 붓꽃 데이터 로드
iris = load_iris()
X = iris.data
y = iris.target

# 프로젝트: 동일 데이터로 지도/비지도 학습 비교
print("🌸 붓꽃 데이터 분석 프로젝트")
print(f"데이터 shape: {X.shape}")
print(f"클래스: {iris.target_names}")

# 1. 지도학습 접근
print("\n1️⃣ 지도학습 접근법:")
# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 분류 모델 학습
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
supervised_score = clf.score(X_test, y_test)
print(f"정확도: {supervised_score:.2%}")

# 가장 중요한 특성
feature_importance = pd.DataFrame({
    'feature': iris.feature_names,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False)
print("\n특성 중요도:")
print(feature_importance)

# 2. 비지도학습 접근
print("\n2️⃣ 비지도학습 접근법:")
# 레이블 없이 군집화
kmeans_iris = KMeans(n_clusters=3, random_state=42)
clusters = kmeans_iris.fit_predict(X)

# 실제 레이블과 비교 (평가용)
from sklearn.metrics import adjusted_rand_score
ari = adjusted_rand_score(y, clusters)
print(f"군집화 품질 (ARI): {ari:.3f}")

# PCA로 시각화
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 결과 비교 시각화
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 지도학습 결과 (실제 레이블)
for i, name in enumerate(iris.target_names):
    mask = y == i
    ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                label=name, alpha=0.7, s=100)
ax1.set_title('지도학습: 실제 붓꽃 종류', fontsize=14)
ax1.set_xlabel('첫 번째 주성분')
ax1.set_ylabel('두 번째 주성분')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 비지도학습 결과 (군집)
scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], 
                     c=clusters, cmap='viridis', alpha=0.7, s=100)
ax2.set_title('비지도학습: K-means 군집화', fontsize=14)
ax2.set_xlabel('첫 번째 주성분')
ax2.set_ylabel('두 번째 주성분')
plt.colorbar(scatter, ax=ax2, label='클러스터')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 인사이트 정리
print("\n📊 프로젝트 인사이트:")
print("• 지도학습: 정확한 분류가 가능하지만 레이블이 필요")
print("• 비지도학습: 레이블 없이도 자연스러운 그룹 발견 가능")
print("• 두 방법 모두 유사한 패턴을 찾아냄")
```

## 5.1.7 직접 해보기: 연습 문제

### 연습 문제 1: 학습 유형 분류하기

```python
# 연습 문제: 다음 문제들을 지도/비지도학습으로 분류하세요
exercises = [
    {
        "문제": "병원에서 환자의 증상으로 질병을 진단하는 AI",
        "힌트": "과거 진단 기록이 있습니다"
    },
    {
        "문제": "유사한 뉴스 기사들을 자동으로 묶어주는 시스템",
        "힌트": "어떤 카테고리가 있을지 미리 알 수 없습니다"
    },
    {
        "문제": "학생의 공부 시간으로 시험 점수를 예측하는 모델",
        "힌트": "과거 학생들의 공부 시간과 점수 데이터가 있습니다"
    },
    {
        "문제": "유전자 데이터에서 숨겨진 패턴을 찾는 연구",
        "힌트": "어떤 패턴이 있을지 모르는 탐색적 연구입니다"
    }
]

print("📝 연습 문제: 지도학습 vs 비지도학습")
for i, ex in enumerate(exercises, 1):
    print(f"\n문제 {i}: {ex['문제']}")
    print(f"힌트: {ex['힌트']}")
    print("당신의 답: ________")

# 정답은 아래에
print("\n" + "="*50)
print("정답:")
answers = ["지도학습", "비지도학습", "지도학습", "비지도학습"]
for i, (ex, ans) in enumerate(zip(exercises, answers), 1):
    print(f"{i}. {ans} - {ex['문제']}")
```

### 연습 문제 2: 실습 코드 완성하기

```python
# 코드 완성 문제
print("\n💻 코드 완성 문제:")
print("""
# 아래 코드를 완성하여 간단한 분류 모델을 만드세요
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 데이터 생성
X, y = make_classification(n_samples=200, n_features=2, n_classes=2, random_state=42)

# TODO 1: 데이터를 학습용과 테스트용으로 분할 (테스트 크기 20%)
X_train, X_test, y_train, y_test = ______________________

# TODO 2: 로지스틱 회귀 모델 생성
model = ______________________

# TODO 3: 모델 학습
______________________

# TODO 4: 테스트 정확도 계산
accuracy = ______________________
print(f"모델 정확도: {accuracy:.2%}")
""")

# 정답 제시
print("\n정답 코드:")
print("""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
""")
```

## 5.1.8 요약 및 핵심 정리

### 이번 파트의 핵심 내용

```python
# 핵심 내용 요약
summary = {
    "머신러닝 정의": "데이터로부터 패턴을 학습하여 예측이나 결정을 내리는 기술",
    
    "지도학습": {
        "특징": "정답(레이블)이 있는 데이터로 학습",
        "목적": "새로운 데이터에 대한 예측",
        "유형": ["분류(카테고리 예측)", "회귀(수치 예측)"],
        "예시": "스팸 필터, 주가 예측, 질병 진단"
    },
    
    "비지도학습": {
        "특징": "정답 없이 데이터의 구조 파악",
        "목적": "숨겨진 패턴 발견",
        "유형": ["군집화", "차원축소", "이상치탐지"],
        "예시": "고객 세분화, 추천 시스템, 사기 탐지"
    },
    
    "선택 기준": [
        "레이블 데이터 유무",
        "문제의 목적(예측 vs 탐색)",
        "도메인 지식 수준",
        "결과의 해석 가능성"
    ]
}

print("📚 5.1 핵심 정리\n")
print(f"🤖 머신러닝: {summary['머신러닝 정의']}\n")

for learning_type in ['지도학습', '비지도학습']:
    info = summary[learning_type]
    print(f"📊 {learning_type}")
    print(f"   • 특징: {info['특징']}")
    print(f"   • 목적: {info['목적']}")
    print(f"   • 유형: {', '.join(info['유형'])}")
    print(f"   • 예시: {info['예시']}\n")

print("🎯 선택 기준:")
for criterion in summary['선택 기준']:
    print(f"   • {criterion}")
```

### 체크리스트: 학습 확인

```python
# 학습 확인 체크리스트
checklist = [
    "머신러닝과 전통적 프로그래밍의 차이를 설명할 수 있다",
    "지도학습과 비지도학습의 차이점을 명확히 이해했다",
    "각 학습 방식의 장단점을 알고 있다",
    "주어진 문제에 적합한 학습 방식을 선택할 수 있다",
    "간단한 분류와 군집화 예제를 구현할 수 있다"
]

print("\n✅ 학습 확인 체크리스트:")
for item in checklist:
    print(f"□ {item}")
```

## 5.1.9 생각해보기 및 다음 파트 예고

### 생각해보기 질문

```python
# 심화 학습을 위한 질문들
thought_questions = [
    "지도학습이 항상 비지도학습보다 좋은 결과를 낼까요?",
    "레이블이 부분적으로만 있을 때는 어떤 접근을 해야 할까요?",
    "비지도학습의 결과를 어떻게 평가하고 검증할 수 있을까요?",
    "실제 비즈니스에서 두 방법을 어떻게 함께 활용할 수 있을까요?"
]

print("🤔 생각해보기:")
for i, question in enumerate(thought_questions, 1):
    print(f"{i}. {question}")
```

### 다음 파트 예고

```python
next_preview = {
    "제목": "5.2 분류 알고리즘의 이해와 구현",
    "주요 내용": [
        "로지스틱 회귀: 확률로 분류하기",
        "의사결정나무: 규칙으로 분류하기",
        "랜덤 포레스트: 여러 나무의 지혜",
        "SVM: 최적의 경계선 찾기"
    ],
    "실습 프로젝트": "타이타닉 생존자 예측 - 실제 데이터로 분류 모델 만들기",
    "학습 포인트": "각 알고리즘의 원리를 이해하고 상황에 맞게 선택하는 능력 기르기"
}

print("\n📖 다음 파트 미리보기")
print(f"제목: {next_preview['제목']}\n")
print("학습할 내용:")
for content in next_preview['주요 내용']:
    print(f"  • {content}")
print(f"\n🛠 실습: {next_preview['실습 프로젝트']}")
print(f"🎯 목표: {next_preview['학습 포인트']}")
```

---

**🎉 축하합니다!** 
머신러닝의 가장 기본이 되는 지도학습과 비지도학습의 개념을 마스터했습니다. 이제 여러분은 주어진 문제에 어떤 머신러닝 접근법을 사용해야 할지 판단할 수 있게 되었습니다. 다음 파트에서는 지도학습의 핵심인 분류 알고리즘들을 실제로 구현하며 더 깊이 있게 학습해보겠습니다!
```
