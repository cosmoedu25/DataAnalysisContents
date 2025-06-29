# 6장 Part 1: 앙상블 학습과 투표 방식
## 집단 지혜로 더 나은 예측하기

### 학습 목표
이번 파트를 완료하면 다음과 같은 능력을 갖게 됩니다:
- 앙상블 학습의 핵심 개념과 원리를 이해하고 설명할 수 있습니다
- 배깅(Bagging)과 부스팅(Boosting)의 차이점을 구분하고 적용할 수 있습니다
- 다양한 투표 방식(하드 투표, 소프트 투표)을 구현하고 활용할 수 있습니다
- 스태킹 앙상블을 설계하고 구현할 수 있습니다
- 신용카드 사기 탐지 문제에 앙상블 기법을 적용할 수 있습니다

---

### 6.1 앙상블 학습이란? - 집단 지혜의 힘

#### 🎯 실생활 속 앙상블의 예
당신이 어려운 수학 문제를 풀고 있다고 상상해보세요. 혼자서는 확신이 서지 않아 친구 3명에게 도움을 요청했습니다. 한 친구는 기하를 잘하고, 다른 친구는 대수를 잘하며, 마지막 친구는 전체적인 문제 해결 능력이 뛰어납니다. 

이때 세 가지 방법으로 답을 정할 수 있습니다:
1. **다수결 투표**: 세 명 중 가장 많이 나온 답을 선택
2. **가중 투표**: 각 친구의 실력에 따라 가중치를 주어 결정
3. **단계별 활용**: 첫 번째 친구들이 후보 답안을 만들고, 마지막 친구가 최종 결정

**앙상블 학습**도 이와 같은 원리입니다! 여러 개의 서로 다른 모델(학습자)을 결합하여 단일 모델보다 더 나은 성능을 달성하는 기법입니다.

#### 🔍 앙상블 학습의 핵심 원리

**1. 다양성(Diversity)의 중요성**
```python
# 앙상블이 효과적인 이유를 간단한 예로 살펴보기
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 시각화를 위한 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 간단한 데이터셋 생성 (신용카드 사기 탐지와 유사한 특성)
X, y = make_classification(
    n_samples=1000,      # 1000개의 샘플
    n_features=20,       # 20개의 특성
    n_informative=10,    # 실제로 유용한 특성 10개
    n_redundant=10,      # 중복된 특성 10개  
    n_clusters_per_class=1,  # 클래스당 클러스터 수
    random_state=42
)

# 데이터를 훈련용과 테스트용으로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("🎯 앙상블 학습의 효과 실험")
print("=" * 50)
print(f"훈련 데이터: {X_train.shape[0]}개 샘플")
print(f"테스트 데이터: {X_test.shape[0]}개 샘플")
print(f"특성 수: {X_train.shape[1]}개")
```

**왜 이 코드가 중요한가?**
- `make_classification`: 실제 신용카드 사기 탐지와 비슷한 특성을 가진 가상 데이터를 생성합니다
- `n_informative=10`: 실제로 유용한 정보를 담고 있는 특성의 수를 지정합니다
- `n_redundant=10`: 중복된 정보를 가진 특성들도 포함하여 현실적인 데이터를 만듭니다

**2. 개별 모델들의 성능 비교**
```python
# 서로 다른 특성을 가진 세 개의 모델 생성
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'SVM': SVC(random_state=42, probability=True)  # probability=True로 확률 예측 가능하게 설정
}

# 각 모델을 개별적으로 훈련하고 성능 평가
individual_scores = {}
for name, model in models.items():
    # 모델 훈련
    model.fit(X_train, y_train)
    
    # 예측 및 성능 평가
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    individual_scores[name] = accuracy
    
    print(f"{name}: {accuracy:.4f}")

print(f"\n개별 모델 평균 성능: {np.mean(list(individual_scores.values())):.4f}")
```

**왜 이 코드가 중요한가?**
- **의사결정나무**: 규칙 기반으로 분류하며, 비선형 관계를 잘 포착합니다
- **로지스틱 회귀**: 선형 관계를 기반으로 하며, 해석이 용이합니다  
- **SVM**: 복잡한 경계를 만들 수 있으며, 고차원 데이터에 강합니다
- `probability=True`: SVM에서 확률 예측을 가능하게 하여 소프트 투표에 활용할 수 있습니다

---

### 6.2 투표 기반 앙상블 - 민주주의 원리를 AI에 적용

#### 🗳️ 하드 투표 (Hard Voting) - 다수결의 원리

하드 투표는 가장 직관적인 앙상블 방법입니다. 각 모델이 예측한 클래스 중에서 가장 많이 선택된 클래스를 최종 예측으로 선택합니다.

```python
# 하드 투표 분류기 생성 및 훈련
hard_voting_clf = VotingClassifier(
    estimators=[
        ('dt', DecisionTreeClassifier(random_state=42)),
        ('lr', LogisticRegression(random_state=42)), 
        ('svm', SVC(random_state=42))
    ],
    voting='hard'  # 하드 투표 방식 선택
)

# 앙상블 모델 훈련
hard_voting_clf.fit(X_train, y_train)

# 예측 및 성능 평가
y_pred_hard = hard_voting_clf.predict(X_test)
hard_voting_accuracy = accuracy_score(y_test, y_pred_hard)

print("🗳️ 하드 투표 결과")
print("=" * 30)
print(f"하드 투표 앙상블 성능: {hard_voting_accuracy:.4f}")
print(f"개별 모델 평균 성능: {np.mean(list(individual_scores.values())):.4f}")
print(f"성능 향상: {hard_voting_accuracy - np.mean(list(individual_scores.values())):.4f}")
```

**왜 이 방식이 효과적인가?**
- 각 모델이 서로 다른 실수를 하더라도, 다수결로 올바른 답을 찾을 확률이 높아집니다
- 특히 개별 모델들의 성능이 50% 이상이고 서로 독립적인 실수를 할 때 매우 효과적입니다

#### 🎯 소프트 투표 (Soft Voting) - 확신의 정도를 고려

소프트 투표는 단순히 클래스만 보는 것이 아니라, 각 모델이 예측에 대해 얼마나 확신하는지도 고려합니다.

```python
# 소프트 투표 분류기 생성 및 훈련
soft_voting_clf = VotingClassifier(
    estimators=[
        ('dt', DecisionTreeClassifier(random_state=42)),
        ('lr', LogisticRegression(random_state=42)),
        ('svm', SVC(random_state=42, probability=True))  # SVM에 확률 예측 활성화
    ],
    voting='soft'  # 소프트 투표 방식 선택
)

# 앙상블 모델 훈련
soft_voting_clf.fit(X_train, y_train)

# 예측 및 성능 평가
y_pred_soft = soft_voting_clf.predict(X_test)
soft_voting_accuracy = accuracy_score(y_test, y_pred_soft)

print("\n🎯 소프트 투표 결과")
print("=" * 30)
print(f"소프트 투표 앙상블 성능: {soft_voting_accuracy:.4f}")
print(f"하드 투표 앙상블 성능: {hard_voting_accuracy:.4f}")
print(f"개별 모델 평균 성능: {np.mean(list(individual_scores.values())):.4f}")
```

**소프트 투표의 장점**
- 확신이 높은 모델의 의견에 더 큰 가중치를 줍니다
- 미묘한 차이를 더 잘 포착할 수 있습니다
- 일반적으로 하드 투표보다 성능이 좋습니다

#### 📊 투표 과정 시각화하기

```python
# 개별 예측 확률 확인 (처음 5개 샘플)
sample_idx = range(5)
X_sample = X_test[sample_idx]
y_sample = y_test[sample_idx]

print("\n📊 개별 모델의 예측 과정 분석")
print("=" * 50)

for i, (name, model) in enumerate(models.items()):
    model.fit(X_train, y_train)  # 모델 재훈련
    
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_sample)
        pred = model.predict(X_sample)
        
        print(f"\n{name} 예측:")
        for j in range(len(sample_idx)):
            print(f"  샘플 {j+1}: 클래스 {pred[j]} (확률: {proba[j][pred[j]]:.3f})")
    else:
        pred = model.predict(X_sample)
        print(f"\n{name} 예측:")
        for j in range(len(sample_idx)):
            print(f"  샘플 {j+1}: 클래스 {pred[j]}")

# 실제 정답
print(f"\n실제 정답: {y_sample}")
```

**왜 이런 분석이 중요한가?**
- 각 모델이 어떤 근거로 예측하는지 이해할 수 있습니다
- 모델들 간의 의견 차이를 확인할 수 있습니다
- 앙상블의 신뢰도를 평가할 수 있습니다

---

### 6.3 배깅(Bagging) - 부트스트랩으로 다양성 만들기

#### 🎒 배깅의 핵심 아이디어

**배깅**은 "Bootstrap Aggregating"의 줄임말입니다. 마치 같은 반 학생들이 서로 다른 문제집으로 공부한 후 시험을 보는 것과 같습니다.

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# 배깅 분류기 생성
bagging_clf = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(random_state=42),  # 기본 모델로 의사결정나무 사용
    n_estimators=100,    # 100개의 모델을 생성
    max_samples=0.8,     # 각 모델은 전체 데이터의 80%만 사용
    max_features=0.8,    # 각 모델은 전체 특성의 80%만 사용
    bootstrap=True,      # 복원 추출 방식 사용
    random_state=42
)

# 모델 훈련
bagging_clf.fit(X_train, y_train)

# 예측 및 성능 평가
y_pred_bagging = bagging_clf.predict(X_test)
bagging_accuracy = accuracy_score(y_test, y_pred_bagging)

print("🎒 배깅 분류기 결과")
print("=" * 30)
print(f"배깅 분류기 성능: {bagging_accuracy:.4f}")
print(f"단일 의사결정나무 성능: {individual_scores['Decision Tree']:.4f}")
print(f"성능 향상: {bagging_accuracy - individual_scores['Decision Tree']:.4f}")
```

**배깅이 효과적인 이유**
- **부트스트랩 샘플링**: 각 모델이 서로 다른 데이터로 학습하여 다양성을 확보합니다
- **분산 감소**: 여러 모델의 평균을 내면 개별 모델의 불안정성이 줄어듭니다
- **과적합 방지**: 각 모델이 전체 데이터의 일부만 보므로 과적합이 줄어듭니다

#### 🌳 랜덤 포레스트 - 배깅의 발전된 형태

```python
from sklearn.ensemble import RandomForestClassifier

# 랜덤 포레스트 분류기 생성
rf_clf = RandomForestClassifier(
    n_estimators=100,     # 100개의 트리
    max_depth=10,         # 각 트리의 최대 깊이
    min_samples_split=5,  # 노드 분할을 위한 최소 샘플 수
    min_samples_leaf=2,   # 리프 노드의 최소 샘플 수
    random_state=42
)

# 모델 훈련 및 예측
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)

print("\n🌳 랜덤 포레스트 결과")
print("=" * 30)
print(f"랜덤 포레스트 성능: {rf_accuracy:.4f}")
print(f"배깅 분류기 성능: {bagging_accuracy:.4f}")
print(f"단일 의사결정나무 성능: {individual_scores['Decision Tree']:.4f}")

# 특성 중요도 확인
feature_importance = rf_clf.feature_importances_
print(f"\n가장 중요한 특성 상위 5개:")
for i in np.argsort(feature_importance)[-5:][::-1]:
    print(f"  특성 {i}: {feature_importance[i]:.4f}")
```

**랜덤 포레스트의 추가 장점**
- **특성 무작위성**: 각 노드에서 일부 특성만 고려하여 더욱 다양한 트리를 만듭니다
- **특성 중요도**: 어떤 특성이 예측에 중요한지 알 수 있습니다
- **빠른 훈련**: 병렬 처리가 가능하여 훈련 속도가 빠릅니다

---

### 6.4 부스팅(Boosting) - 약한 학습자들의 협력

#### 🚀 부스팅의 핵심 아이디어

부스팅은 "약한 학습자"들이 서로 협력하여 "강한 학습자"를 만드는 방법입니다. 마치 운동 선수가 코치의 지적을 받아가며 점점 실력을 향상시키는 것과 같습니다.

```python
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

# AdaBoost 분류기 생성
ada_clf = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),  # 약한 학습자 (스텀프)
    n_estimators=100,     # 100개의 약한 학습자
    learning_rate=1.0,    # 학습률
    random_state=42
)

# 모델 훈련 및 예측
ada_clf.fit(X_train, y_train)
y_pred_ada = ada_clf.predict(X_test)
ada_accuracy = accuracy_score(y_test, y_pred_ada)

print("🚀 AdaBoost 결과")
print("=" * 30)
print(f"AdaBoost 성능: {ada_accuracy:.4f}")

# Gradient Boosting 분류기 생성
gb_clf = GradientBoostingClassifier(
    n_estimators=100,     # 100개의 트리
    learning_rate=0.1,    # 학습률 (작을수록 안정적)
    max_depth=3,          # 각 트리의 최대 깊이
    random_state=42
)

# 모델 훈련 및 예측
gb_clf.fit(X_train, y_train)
y_pred_gb = gb_clf.predict(X_test)
gb_accuracy = accuracy_score(y_test, y_pred_gb)

print(f"Gradient Boosting 성능: {gb_accuracy:.4f}")
```

**부스팅의 작동 원리**
- **순차적 학습**: 이전 모델의 실수를 다음 모델이 집중적으로 학습합니다
- **가중치 조정**: 틀린 샘플에 더 큰 가중치를 부여합니다  
- **점진적 개선**: 매 단계마다 조금씩 성능이 향상됩니다

#### ⚡ XGBoost - 부스팅의 최고봉

```python
# XGBoost는 별도 설치가 필요합니다: pip install xgboost
try:
    import xgboost as xgb
    
    # XGBoost 분류기 생성
    xgb_clf = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        eval_metric='logloss'  # 경고 메시지 방지
    )
    
    # 모델 훈련 및 예측
    xgb_clf.fit(X_train, y_train)
    y_pred_xgb = xgb_clf.predict(X_test)
    xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
    
    print(f"\n⚡ XGBoost 결과")
    print("=" * 30)
    print(f"XGBoost 성능: {xgb_accuracy:.4f}")
    
except ImportError:
    print("XGBoost가 설치되지 않았습니다. 'pip install xgboost'로 설치해주세요.")
    xgb_accuracy = 0
```

**XGBoost의 특징**
- **고성능**: 많은 머신러닝 대회에서 우승을 차지한 알고리즘입니다
- **정규화**: 과적합을 방지하는 다양한 기법이 내장되어 있습니다
- **속도**: C++로 구현되어 매우 빠른 실행 속도를 자랑합니다

---

### 6.5 스태킹(Stacking) - 메타 러닝의 힘

#### 🏗️ 스태킹의 개념

스태킹은 1차 모델들의 예측을 입력으로 받아 2차 모델이 최종 예측을 하는 방법입니다. 마치 여러 전문가의 의견을 종합하여 최종 결정을 내리는 중재자 같은 역할입니다.

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# 1차 학습자들 정의
base_learners = [
    ('dt', DecisionTreeClassifier(random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
    ('svm', SVC(probability=True, random_state=42))
]

# 2차 학습자 (메타 학습자) 정의
meta_learner = LogisticRegression(random_state=42)

# 스태킹 분류기 생성
stacking_clf = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5,  # 5-fold 교차 검증 사용
    passthrough=False  # 원본 특성을 메타 학습자에게 전달하지 않음
)

# 모델 훈련 및 예측
stacking_clf.fit(X_train, y_train)
y_pred_stacking = stacking_clf.predict(X_test)
stacking_accuracy = accuracy_score(y_test, y_pred_stacking)

print("🏗️ 스태킹 분류기 결과")
print("=" * 30)
print(f"스태킹 분류기 성능: {stacking_accuracy:.4f}")
```

**스태킹의 장점**
- **높은 성능**: 일반적으로 다른 앙상블 방법보다 성능이 좋습니다
- **유연성**: 다양한 종류의 모델을 조합할 수 있습니다
- **과적합 방지**: 교차 검증을 통해 과적합을 방지합니다

---

### 6.6 종합 성능 비교 및 분석

#### 📊 모든 앙상블 방법 성능 비교

```python
# 모든 방법의 성능을 정리
results = {
    '단일 의사결정나무': individual_scores['Decision Tree'],
    '단일 로지스틱 회귀': individual_scores['Logistic Regression'], 
    '단일 SVM': individual_scores['SVM'],
    '하드 투표': hard_voting_accuracy,
    '소프트 투표': soft_voting_accuracy,
    '배깅': bagging_accuracy,
    '랜덤 포레스트': rf_accuracy,
    'AdaBoost': ada_accuracy,
    'Gradient Boosting': gb_accuracy,
    '스태킹': stacking_accuracy
}

if xgb_accuracy > 0:
    results['XGBoost'] = xgb_accuracy

# 결과를 성능 순으로 정렬
sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

print("📊 전체 성능 비교 (정확도 기준)")
print("=" * 50)
for method, accuracy in sorted_results.items():
    print(f"{method:20}: {accuracy:.4f}")

# 최고 성능과 최저 성능 차이 계산
best_score = max(sorted_results.values())
worst_score = min(sorted_results.values())
improvement = best_score - worst_score

print(f"\n📈 성능 개선 효과")
print(f"최고 성능: {best_score:.4f}")
print(f"최저 성능: {worst_score:.4f}")
print(f"개선 폭: {improvement:.4f} ({improvement*100:.2f}%p)")

#### 🎨 성능 비교 시각화

```python
# 성능 비교 막대 그래프 생성
plt.figure(figsize=(12, 8))

methods = list(sorted_results.keys())
accuracies = list(sorted_results.values())

# 막대 색깔 설정 (앙상블 방법은 다른 색으로)
colors = []
for method in methods:
    if '단일' in method:
        colors.append('#ff7f7f')  # 연한 빨강 (단일 모델)
    else:
        colors.append('#7fbf7f')  # 연한 초록 (앙상블 모델)

bars = plt.bar(range(len(methods)), accuracies, color=colors)

# 그래프 꾸미기
plt.title('앙상블 방법별 성능 비교', fontsize=16, fontweight='bold')
plt.xlabel('방법', fontsize=12)
plt.ylabel('정확도', fontsize=12)
plt.xticks(range(len(methods)), methods, rotation=45, ha='right')
plt.ylim(min(accuracies) - 0.01, max(accuracies) + 0.01)

# 각 막대 위에 정확도 값 표시
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
             f'{height:.3f}', ha='center', va='bottom', fontsize=10)

# 범례 추가
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#ff7f7f', label='단일 모델'),
                   Patch(facecolor='#7fbf7f', label='앙상블 모델')]
plt.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.show()
```

**시각화에서 주목할 점**
- 앙상블 방법들이 단일 모델보다 일반적으로 성능이 좋습니다
- 특히 스태킹과 부스팅 계열이 우수한 성능을 보입니다
- 다양한 방법 중에서 문제에 맞는 최적의 방법을 선택하는 것이 중요합니다

---

### 🛠️ 실습 프로젝트: 신용카드 사기 탐지 앙상블 시스템

이제 실제 신용카드 사기 탐지 데이터셋을 사용하여 앙상블 시스템을 구축해보겠습니다.

#### 📊 데이터 준비

```python
# 실제 Credit Card Fraud Detection 데이터셋 로드
# (Kaggle에서 다운로드: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

# 데이터셋이 없는 경우 유사한 데이터 생성
from sklearn.datasets import make_classification

# 신용카드 사기 탐지와 유사한 불균형 데이터셋 생성
X_fraud, y_fraud = make_classification(
    n_samples=10000,      # 10,000개 거래
    n_features=30,        # 30개 특성 (PCA 변환된 특성들)
    n_informative=20,     # 실제로 유용한 특성 20개
    n_redundant=10,       # 중복 특성 10개
    n_clusters_per_class=1,
    weights=[0.999, 0.001],  # 사기 거래는 0.1% (매우 불균형)
    flip_y=0.01,          # 1% 노이즈 추가
    random_state=42
)

print("💳 신용카드 사기 탐지 데이터")
print("=" * 40)
print(f"전체 거래 수: {X_fraud.shape[0]:,}개")
print(f"특성 수: {X_fraud.shape[1]}개")
print(f"정상 거래: {np.sum(y_fraud == 0):,}개 ({np.sum(y_fraud == 0)/len(y_fraud)*100:.1f}%)")
print(f"사기 거래: {np.sum(y_fraud == 1):,}개 ({np.sum(y_fraud == 1)/len(y_fraud)*100:.1f}%)")
```

#### 🔧 불균형 데이터를 위한 앙상블 전략

```python
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# 불균형 데이터에 특화된 훈련/테스트 분할
X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud = train_test_split(
    X_fraud, y_fraud, test_size=0.3, random_state=42, stratify=y_fraud
)

# 1. 클래스 가중치를 적용한 앙상블 모델들
models_fraud = {
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',  # 클래스 불균형 자동 조정
        random_state=42
    ),
    'XGBoost': None,  # XGBoost 설치 시에만 사용
    'Balanced Voting': VotingClassifier(
        estimators=[
            ('dt', DecisionTreeClassifier(class_weight='balanced', random_state=42)),
            ('lr', LogisticRegression(class_weight='balanced', random_state=42)),
        ],
        voting='soft'
    )
}

# XGBoost가 설치된 경우에만 추가
try:
    import xgboost as xgb
    models_fraud['XGBoost'] = xgb.XGBClassifier(
        n_estimators=100,
        scale_pos_weight=np.sum(y_fraud == 0) / np.sum(y_fraud == 1),  # 클래스 불균형 조정
        random_state=42,
        eval_metric='logloss'
    )
except ImportError:
    del models_fraud['XGBoost']

# 모델 훈련 및 평가
fraud_results = {}
for name, model in models_fraud.items():
    if model is not None:
        print(f"\n🔧 {name} 훈련 중...")
        model.fit(X_train_fraud, y_train_fraud)
        y_pred = model.predict(X_test_fraud)
        y_pred_proba = model.predict_proba(X_test_fraud)[:, 1]
        
        # 다양한 평가 지표 계산
        precision = precision_score(y_test_fraud, y_pred)
        recall = recall_score(y_test_fraud, y_pred)
        f1 = f1_score(y_test_fraud, y_pred)
        auc = roc_auc_score(y_test_fraud, y_pred_proba)
        
        fraud_results[name] = {
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'AUC': auc
        }
        
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")
```

**불균형 데이터에서 중요한 평가 지표들**
- **Precision**: 사기로 예측한 것 중 실제 사기인 비율 (거짓 경보 방지)
- **Recall**: 실제 사기 중 올바르게 탐지한 비율 (사기 놓치지 않기)
- **F1-Score**: Precision과 Recall의 조화평균
- **AUC**: ROC 곡선 아래 면적 (전체적인 분류 성능)

#### 📈 ROC 곡선으로 모델 성능 비교

```python
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))

# 각 모델의 ROC 곡선 그리기
for name, model in models_fraud.items():
    if model is not None:
        y_pred_proba = model.predict_proba(X_test_fraud)[:, 1]
        fpr, tpr, _ = roc_curve(y_test_fraud, y_pred_proba)
        auc_score = roc_auc_score(y_test_fraud, y_pred_proba)
        
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})', linewidth=2)

# 대각선 (랜덤 분류기) 그리기
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.500)', alpha=0.5)

# 그래프 꾸미기
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (거짓 양성률)', fontsize=12)
plt.ylabel('True Positive Rate (참 양성률)', fontsize=12)
plt.title('ROC 곡선 비교 - 신용카드 사기 탐지', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()
```

**ROC 곡선 해석**
- 곡선이 왼쪽 위에 가까울수록 성능이 좋습니다
- AUC가 0.5에 가까우면 랜덤 분류기와 비슷한 성능입니다
- AUC가 1.0에 가까울수록 완벽한 분류기입니다

---

### 💪 직접 해보기 - 연습 문제

#### 🎯 연습 문제 1: 앙상블 방법 비교
다음 코드를 완성하여 서로 다른 3가지 앙상블 방법의 성능을 비교해보세요.

```python
# TODO: 코드를 완성하세요
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# 데이터 준비 (붓꽃 데이터 사용)
from sklearn.datasets import load_iris
iris = load_iris()
X_iris, y_iris = iris.data, iris.target

# 훈련/테스트 분할
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
    X_iris, y_iris, test_size=0.3, random_state=42
)

# 1. 랜덤 포레스트 분류기 생성 및 훈련
rf_clf = RandomForestClassifier(n_estimators=50, random_state=42)
# TODO: 모델 훈련
# TODO: 예측 및 정확도 계산

# 2. AdaBoost 분류기 생성 및 훈련
ada_clf = AdaBoostClassifier(n_estimators=50, random_state=42)
# TODO: 모델 훈련
# TODO: 예측 및 정확도 계산

# 3. 투표 분류기 생성 및 훈련
voting_clf = VotingClassifier(
    estimators=[
        ('dt', DecisionTreeClassifier(random_state=42)),
        ('lr', LogisticRegression(random_state=42))
    ],
    voting='soft'
)
# TODO: 모델 훈련
# TODO: 예측 및 정확도 계산

# TODO: 결과 비교 출력
```

#### 🎯 연습 문제 2: 스태킹 앙상블 구현
```python
# TODO: 스태킹 분류기를 구현하세요
from sklearn.ensemble import StackingClassifier

# 1차 학습자들 정의
base_learners = [
    # TODO: 세 개의 서로 다른 모델을 정의하세요
]

# 2차 학습자 정의
meta_learner = # TODO: 메타 학습자를 정의하세요

# 스태킹 분류기 생성
stacking_clf = StackingClassifier(
    # TODO: 매개변수들을 설정하세요
)

# TODO: 모델 훈련 및 성능 평가
```

#### 🎯 연습 문제 3: 불균형 데이터 처리
불균형한 이진 분류 문제를 위한 앙상블 전략을 구현해보세요.

```python
# 불균형 데이터 생성
X_imbal, y_imbal = make_classification(
    n_samples=1000,
    n_features=10,
    weights=[0.9, 0.1],  # 90:10 비율
    random_state=42
)

# TODO: 다음 요구사항을 만족하는 앙상블 시스템을 구현하세요
# 1. 클래스 가중치를 적용한 랜덤 포레스트
# 2. SMOTE를 적용한 후 일반 모델 훈련
# 3. 임계값 조정을 통한 성능 최적화
# 4. F1-Score 기준으로 성능 비교
```

---

### 📚 핵심 정리

#### ✨ 이번 파트에서 배운 내용

**1. 앙상블 학습의 기본 원리**
- 여러 모델의 예측을 결합하여 단일 모델보다 좋은 성능 달성
- 다양성(Diversity)이 앙상블 성능의 핵심
- 편향-분산 트레이드오프 개선

**2. 투표 기반 앙상블**
- **하드 투표**: 다수결 원리로 클래스 결정
- **소프트 투표**: 예측 확률을 고려한 가중 투표
- 서로 다른 특성을 가진 모델들의 조합이 효과적

**3. 배깅(Bagging)**
- 부트스트랩 샘플링으로 다양한 훈련 데이터 생성
- 분산을 줄여 과적합 방지
- 랜덤 포레스트는 배깅의 대표적인 예

**4. 부스팅(Boosting)**
- 순차적으로 약한 학습자들을 결합
- 이전 모델의 실수를 다음 모델이 보완
- AdaBoost, Gradient Boosting, XGBoost 등

**5. 스태킹(Stacking)**
- 1차 모델들의 예측을 2차 모델의 입력으로 사용
- 메타 러닝을 통한 고성능 달성
- 교차 검증으로 과적합 방지

#### 🎯 실무 적용 가이드라인

**언제 어떤 앙상블 방법을 사용할까?**

- **소규모 데이터 + 빠른 결과**: 투표 기반 앙상블
- **안정성이 중요한 경우**: 배깅, 랜덤 포레스트
- **높은 성능이 필요한 경우**: 부스팅, 스태킹
- **불균형 데이터**: 클래스 가중치 + 앙상블
- **해석 가능성이 중요한 경우**: 랜덤 포레스트 (특성 중요도)

**주의사항**
- 과적합 위험: 특히 스태킹에서 주의 필요
- 계산 비용: 모델 수가 많아질수록 훈련/예측 시간 증가
- 해석 어려움: 앙상블은 블랙박스 특성이 강함

---

### 🔮 다음 파트 미리보기

다음 Part 2에서는 **차원 축소와 군집화**에 대해 학습합니다:

- 🎯 **주성분 분석(PCA)**: 고차원 데이터를 저차원으로 압축
- 🔍 **t-SNE**: 데이터의 구조를 시각화하는 비선형 차원 축소
- 📊 **K-평균 군집화**: 비슷한 데이터들을 그룹으로 묶기
- 🌳 **계층적 군집화**: 덴드로그램으로 군집 구조 파악
- 🚀 **실습**: 고객 세분화를 위한 차원 축소 + 군집화 프로젝트

앙상블 학습으로 예측 성능을 높였다면, 이제 데이터의 숨겨진 패턴과 구조를 발견하는 방법을 배워보겠습니다!

---

*"개별 나무를 보지 말고 숲 전체를 보라. 앙상블의 힘은 다양성에서 나온다." - 데이터 과학자의 지혜*
```