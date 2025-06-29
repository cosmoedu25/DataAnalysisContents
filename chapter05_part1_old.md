# 5장 Part 1: 지도학습과 비지도학습의 개념

## 학습 목표
이번 파트를 완료하면 다음을 할 수 있습니다:
- 머신러닝의 기본 개념과 유형을 설명할 수 있다
- 지도학습과 비지도학습의 차이점을 이해하고 적절한 상황에서 활용할 수 있다
- 각 학습 패러다임의 장단점을 비교 분석할 수 있다
- 실제 비즈니스 문제에 적합한 머신러닝 접근법을 선택할 수 있다

## 5.1.1 머신러닝의 기본 개념

### 머신러닝이란?
머신러닝은 컴퓨터가 **명시적으로 프로그래밍되지 않아도** 데이터로부터 패턴을 찾아 학습하고 예측하는 기술입니다. 

전통적인 프로그래밍과 머신러닝의 차이를 살펴보겠습니다:

**전통적인 프로그래밍:**
```
입력 데이터 + 규칙 → 결과
```

**머신러닝:**
```
입력 데이터 + 결과 → 규칙(모델)
```

### 일상 속 머신러닝 예시

```python
# 머신러닝이 활용되는 일상 예시들
examples = {
    "추천 시스템": "넷플릭스 영화 추천, 유튜브 동영상 추천",
    "음성 인식": "시리, 알렉사, 구글 어시스턴트",
    "이미지 인식": "사진 앱의 얼굴 태그, 자율주행차",
    "번역": "구글 번역, 파파고",
    "검색": "구글 검색 결과 순위",
    "금융": "신용카드 부정거래 탐지, 대출 심사"
}

for category, example in examples.items():
    print(f"{category}: {example}")
```

출력:
```
추천 시스템: 넷플릭스 영화 추천, 유튜브 동영상 추천
음성 인식: 시리, 알렉사, 구글 어시스턴트
이미지 인식: 사진 앱의 얼굴 태그, 자율주행차
번역: 구글 번역, 파파고
검색: 구글 검색 결과 순위
금융: 신용카드 부정거래 탐지, 대출 심사
```

## 5.1.2 머신러닝의 주요 유형

머신러닝은 크게 세 가지 유형으로 분류됩니다:

### 1. 지도학습 (Supervised Learning)
**정답이 있는 데이터**로 학습하는 방법입니다.

```python
import pandas as pd
import numpy as np

# 지도학습 예시 데이터
supervised_data = pd.DataFrame({
    '키': [165, 170, 180, 160, 175],
    '몸무게': [55, 65, 80, 50, 70],
    '성별': ['여성', '남성', '남성', '여성', '남성']  # 정답 레이블
})

print("지도학습 데이터 예시:")
print(supervised_data)
print("\n목표: 키와 몸무게로 성별을 예측하는 모델 만들기")
```

### 2. 비지도학습 (Unsupervised Learning)
**정답이 없는 데이터**에서 숨겨진 패턴을 찾는 방법입니다.

```python
# 비지도학습 예시 데이터
unsupervised_data = pd.DataFrame({
    '키': [165, 170, 180, 160, 175, 168, 172, 178],
    '몸무게': [55, 65, 80, 50, 70, 58, 68, 75]
    # 성별 정보 없음
})

print("비지도학습 데이터 예시:")
print(unsupervised_data)
print("\n목표: 키와 몸무게 데이터에서 자연스러운 그룹 찾기")
```

### 3. 강화학습 (Reinforcement Learning)
**보상과 벌칙**을 통해 최적의 행동을 학습하는 방법입니다.

```python
# 강화학습 개념 예시
reinforcement_concept = {
    "환경": "게임 또는 현실 세계",
    "에이전트": "학습하는 주체 (AI)",
    "행동": "에이전트가 취할 수 있는 선택들",
    "보상": "행동에 대한 피드백 (+점수 또는 -점수)",
    "목표": "누적 보상을 최대화하는 전략 학습"
}

print("강화학습의 핵심 요소:")
for key, value in reinforcement_concept.items():
    print(f"{key}: {value}")
```

## 5.1.3 지도학습 vs 비지도학습 상세 비교

### 지도학습의 특징

```python
# 지도학습의 장단점 분석
supervised_analysis = {
    "장점": [
        "명확한 목표가 있어 성능 측정이 쉬움",
        "예측 정확도가 높음",
        "비즈니스 문제 해결에 직접적으로 활용 가능",
        "결과 해석이 상대적으로 용이"
    ],
    "단점": [
        "레이블이 있는 데이터가 필요 (비용과 시간 소요)",
        "레이블의 품질에 성능이 크게 좌우됨",
        "새로운 패턴 발견에는 한계가 있음",
        "과적합 위험이 높음"
    ],
    "적용 분야": [
        "이메일 스팸 분류",
        "부동산 가격 예측",
        "의료 진단 보조",
        "고객 이탈 예측"
    ]
}

print("=== 지도학습 분석 ===")
for category, items in supervised_analysis.items():
    print(f"\n{category}:")
    for item in items:
        print(f"  • {item}")
```

### 비지도학습의 특징

```python
# 비지도학습의 장단점 분석
unsupervised_analysis = {
    "장점": [
        "레이블이 없는 데이터로도 학습 가능",
        "숨겨진 패턴과 구조 발견 가능",
        "데이터 수집 비용이 상대적으로 낮음",
        "새로운 인사이트 발견에 유리"
    ],
    "단점": [
        "성능 평가가 어려움 (정답이 없음)",
        "결과 해석이 주관적일 수 있음",
        "비즈니스 목표와 직접 연결이 어려울 수 있음",
        "결과의 유의미성 판단이 어려움"
    ],
    "적용 분야": [
        "고객 세분화 (Customer Segmentation)",
        "추천 시스템",
        "이상 거래 탐지",
        "데이터 압축 및 차원 축소"
    ]
}

print("\n=== 비지도학습 분석 ===")
for category, items in unsupervised_analysis.items():
    print(f"\n{category}:")
    for item in items:
        print(f"  • {item}")
```

## 5.1.4 실제 비즈니스 문제와 머신러닝 유형 매칭

### 문제 유형별 접근법 선택 가이드

```python
# 비즈니스 문제별 머신러닝 접근법
problem_solution_map = {
    "고객 이탈 예측": {
        "유형": "지도학습 - 분류",
        "이유": "이탈한 고객 데이터(정답)가 있음",
        "알고리즘 예시": "로지스틱 회귀, 랜덤 포레스트"
    },
    "주택 가격 예측": {
        "유형": "지도학습 - 회귀",
        "이유": "실제 판매 가격 데이터(정답)가 있음",
        "알고리즘 예시": "선형 회귀, XGBoost"
    },
    "고객 그룹 나누기": {
        "유형": "비지도학습 - 군집화",
        "이유": "어떤 그룹이 있는지 모르는 상태",
        "알고리즘 예시": "K-means, 계층적 클러스터링"
    },
    "제품 추천": {
        "유형": "지도학습 + 비지도학습",
        "이유": "구매 패턴(지도) + 유사 고객 찾기(비지도)",
        "알고리즘 예시": "협업 필터링, 행렬 분해"
    }
}

print("=== 비즈니스 문제별 머신러닝 접근법 ===")
for problem, solution in problem_solution_map.items():
    print(f"\n📋 문제: {problem}")
    print(f"   🎯 유형: {solution['유형']}")
    print(f"   💡 이유: {solution['이유']}")
    print(f"   🔧 알고리즘: {solution['알고리즘 예시']}")
```

## 5.1.5 실습: 머신러닝 유형 판별하기

### 실습 1: 문제 분류 연습

```python
# 다양한 비즈니스 문제들
practice_problems = [
    "은행에서 신용카드 신청자의 신용도를 평가하고 싶다",
    "온라인 쇼핑몰에서 고객들을 특성에 따라 그룹으로 나누고 싶다",
    "부동산 투자를 위해 아파트 가격을 예측하고 싶다",
    "제조업체에서 불량품을 자동으로 분류하고 싶다",
    "소셜미디어에서 비슷한 관심사를 가진 사용자들을 찾고 싶다"
]

# 정답
answers = [
    "지도학습 - 분류 (신용도 등급이라는 정답이 있음)",
    "비지도학습 - 군집화 (어떤 그룹이 있는지 모름)",
    "지도학습 - 회귀 (실제 아파트 가격이라는 정답이 있음)",
    "지도학습 - 분류 (불량품/정상품이라는 정답이 있음)",
    "비지도학습 - 군집화 (유사성 기준이 명확하지 않음)"
]

print("=== 머신러닝 유형 판별 연습 ===")
print("\n다음 문제들을 지도학습/비지도학습으로 분류해보세요:")
for i, problem in enumerate(practice_problems, 1):
    print(f"\n{i}. {problem}")

print("\n" + "="*50)
print("정답:")
for i, (problem, answer) in enumerate(zip(practice_problems, answers), 1):
    print(f"\n{i}. {problem}")
    print(f"   → {answer}")
```

### 실습 2: 데이터셋으로 학습 유형 이해하기

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_blobs

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 1. 지도학습용 분류 데이터 생성
X_supervised, y_supervised = make_classification(
    n_samples=200, n_features=2, n_redundant=0, 
    n_informative=2, n_clusters_per_class=1, random_state=42
)

# 2. 비지도학습용 데이터 생성 (레이블 없음)
X_unsupervised, _ = make_blobs(
    n_samples=200, centers=3, n_features=2, 
    random_state=42, cluster_std=1.5
)

# 시각화
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 지도학습 데이터 (색깔로 클래스 구분)
scatter1 = ax1.scatter(X_supervised[:, 0], X_supervised[:, 1], 
                      c=y_supervised, cmap='viridis', alpha=0.7)
ax1.set_title('지도학습: 레이블이 있는 데이터\n(색깔 = 정답 클래스)', fontsize=12)
ax1.set_xlabel('특성 1')
ax1.set_ylabel('특성 2')
plt.colorbar(scatter1, ax=ax1)

# 비지도학습 데이터 (모든 점이 같은 색)
ax2.scatter(X_unsupervised[:, 0], X_unsupervised[:, 1], 
           c='blue', alpha=0.7)
ax2.set_title('비지도학습: 레이블이 없는 데이터\n(숨겨진 패턴을 찾아야 함)', fontsize=12)
ax2.set_xlabel('특성 1')
ax2.set_ylabel('특성 2')

plt.tight_layout()
plt.show()

print("✅ 지도학습: 왼쪽 그래프처럼 정답(색깔)이 주어진 상태에서 학습")
print("✅ 비지도학습: 오른쪽 그래프처럼 정답 없이 데이터의 패턴을 찾아야 함")
```

## 5.1.6 요약 및 핵심 포인트

### 핵심 개념 정리

```python
# 이번 파트의 핵심 개념들
key_concepts = {
    "머신러닝": "데이터로부터 패턴을 학습하여 예측하는 기술",
    
    "지도학습": {
        "정의": "정답(레이블)이 있는 데이터로 학습",
        "목적": "새로운 데이터에 대한 정확한 예측",
        "예시": "스팸 메일 분류, 주택 가격 예측"
    },
    
    "비지도학습": {
        "정의": "정답 없이 데이터의 숨겨진 패턴 발견",
        "목적": "데이터 구조 이해, 그룹화, 차원 축소",
        "예시": "고객 세분화, 이상 거래 탐지"
    },
    
    "선택 기준": "목표가 명확하면 지도학습, 탐색이 목적이면 비지도학습"
}

print("=== 5.1 핵심 개념 정리 ===")
print(f"\n🤖 {key_concepts['머신러닝']}")
print(f"\n📚 지도학습: {key_concepts['지도학습']['정의']}")
print(f"   목적: {key_concepts['지도학습']['목적']}")
print(f"   예시: {key_concepts['지도학습']['예시']}")
print(f"\n🔍 비지도학습: {key_concepts['비지도학습']['정의']}")
print(f"   목적: {key_concepts['비지도학습']['목적']}")
print(f"   예시: {key_concepts['비지도학습']['예시']}")
print(f"\n🎯 {key_concepts['선택 기준']}")
```

### 다음 파트 예고

```python
next_part_preview = {
    "파트": "5.2 분류 알고리즘의 이해와 구현",
    "학습 내용": [
        "로지스틱 회귀로 확률 기반 분류하기",
        "의사결정나무로 규칙 기반 분류하기", 
        "랜덤 포레스트로 성능 향상시키기",
        "분류 성능 평가 지표 완전 정복"
    ],
    "실습": "타이타닉 데이터로 생존자 예측 모델 만들기"
}

print("\n=== 다음 파트 미리보기 ===")
print(f"📖 {next_part_preview['파트']}")
print("\n학습할 내용:")
for content in next_part_preview['학습 내용']:
    print(f"  • {content}")
print(f"\n🛠 실습: {next_part_preview['실습']}")
```

---

**🎯 이번 파트에서 배운 것**
- 머신러닝의 기본 개념과 일상 속 활용 사례
- 지도학습과 비지도학습의 차이점과 특징
- 비즈니스 문제에 적합한 머신러닝 접근법 선택 방법
- 실제 데이터를 통한 학습 유형별 특성 이해

**🚀 다음 파트에서는**
지도학습의 핵심인 **분류 알고리즘**을 깊이 있게 학습하고, 타이타닉 데이터를 활용한 실전 프로젝트를 진행합니다!
```
