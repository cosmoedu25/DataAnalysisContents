# 5장 Part 2: 분류 알고리즘의 이해와 구현

## 학습 목표
이번 파트를 완료하면 다음을 할 수 있습니다:
- 로지스틱 회귀의 시그모이드 함수와 확률 기반 분류 원리를 이해할 수 있다
- 의사결정나무의 정보 이득과 규칙 기반 분류 과정을 설명할 수 있다
- 랜덤 포레스트의 앙상블 원리와 과적합 방지 효과를 이해할 수 있다
- 서포트 벡터 머신의 마진 최적화 개념을 파악할 수 있다
- 혼동 행렬, 정밀도, 재현율, F1-점수 등 분류 성능 지표를 계산하고 해석할 수 있다
- 타이타닉 데이터를 활용하여 실제 생존 예측 모델을 구축할 수 있다

## 이번 파트 미리보기
분류는 머신러닝에서 가장 흔히 마주치는 문제 유형 중 하나입니다. "이 이메일이 스팸일까?", "이 환자가 병에 걸릴까?", "이 고객이 상품을 구매할까?" 같은 질문들이 모두 분류 문제입니다.

이번 파트에서는 가장 널리 사용되는 4가지 분류 알고리즘을 학습하고, 1912년 타이타닉호 침몰 사건의 실제 데이터를 활용하여 승객의 생존을 예측하는 모델을 구축해보겠습니다. 각 알고리즘의 작동 원리를 이해하고, 언제 어떤 알고리즘을 사용해야 하는지 배우게 됩니다.

또한 모델의 성능을 정확하게 평가하는 방법도 학습합니다. 단순한 정확도만으로는 모델의 진짜 성능을 알 수 없기 때문입니다.

## 5.2.1 분류 문제의 이해와 데이터 준비

### 분류란 무엇인가?

분류(Classification)는 주어진 입력 데이터를 미리 정의된 범주(클래스)로 나누는 것입니다. 

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_score, recall_score, f1_score)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("🔍 분류 문제의 유형과 예시")
print("="*50)

classification_types = {
    "이진 분류 (Binary Classification)": {
        "설명": "두 개의 클래스로 분류하는 문제",
        "예시": [
            "스팸 메일 vs 정상 메일",
            "합격 vs 불합격", 
            "구매 vs 미구매",
            "생존 vs 사망",
            "양성 vs 음성"
        ]
    },
    "다중 분류 (Multi-class Classification)": {
        "설명": "세 개 이상의 클래스로 분류하는 문제",
        "예시": [
            "꽃의 종류 (setosa, versicolor, virginica)",
            "영화 장르 (액션, 코미디, 로맨스, 공포)",
            "제품 카테고리 (전자제품, 의류, 도서, 식품)",
            "학점 (A, B, C, D, F)"
        ]
    }
}

for class_type, info in classification_types.items():
    print(f"\n📊 {class_type}")
    print(f"   {info['설명']}")
    print("   예시:")
    for example in info['예시']:
        print(f"     • {example}")
```

### 타이타닉 데이터셋 소개

타이타닉호는 1912년 4월 15일 빙산과 충돌하여 침몰한 여객선입니다. 총 2,224명의 승객과 승무원 중 1,514명이 사망한 비극적인 사건이었습니다. 우리는 이 역사적 데이터를 활용하여 승객의 특성에 따른 생존 여부를 예측하는 분류 모델을 만들어보겠습니다.

```python
# 타이타닉 데이터셋 로드
titanic = sns.load_dataset('titanic')

print("🚢 타이타닉 데이터셋 기본 정보")
print("="*50)
print(f"데이터 크기: {titanic.shape[0]}명의 승객, {titanic.shape[1]}개의 특성")
print(f"컬럼명: {list(titanic.columns)}")

print("\n📋 데이터 샘플 (첫 5행):")
display_columns = ['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
print(titanic[display_columns].head())

print("\n📊 각 컬럼의 의미:")
column_meanings = {
    'survived': '생존 여부 (0: 사망, 1: 생존)',
    'pclass': '객실 등급 (1: 1등급, 2: 2등급, 3: 3등급)',
    'sex': '성별 (male: 남성, female: 여성)',
    'age': '나이',
    'sibsp': '배우자 + 형제자매 수',
    'parch': '부모 + 자녀 수',
    'fare': '요금',
    'embarked': '탑승 항구 (C: 체르부르, Q: 퀸스타운, S: 사우샘프턴)'
}

for col, meaning in column_meanings.items():
    print(f"   • {col}: {meaning}")

# 생존 현황 기본 분석
print("\n⚰️ 생존 현황 분석")
print("="*30)
survival_counts = titanic['survived'].value_counts().sort_index()
survival_rate = titanic['survived'].mean()

print(f"사망자: {survival_counts[0]}명 ({(1-survival_rate)*100:.1f}%)")
print(f"생존자: {survival_counts[1]}명 ({survival_rate*100:.1f}%)")
print(f"전체 생존율: {survival_rate:.3f} ({survival_rate*100:.1f}%)")
```

### 탐색적 데이터 분석을 통한 생존 패턴 발견

분류 모델을 만들기 전에 데이터를 탐색하여 생존에 영향을 미치는 요인들을 파악해보겠습니다.

```python
# 시각화로 생존 패턴 분석
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('타이타닉 승객 생존 패턴 분석', fontsize=16, fontweight='bold')

# 1. 성별에 따른 생존율
sex_survival = titanic.groupby('sex')['survived'].agg(['count', 'sum', 'mean'])
sex_survival['survival_rate'] = sex_survival['mean'] * 100

ax1 = axes[0, 0]
bars = ax1.bar(sex_survival.index, sex_survival['survival_rate'], 
               color=['lightblue', 'pink'], alpha=0.8)
ax1.set_title('성별 생존율')
ax1.set_ylabel('생존율 (%)')
ax1.set_ylim(0, 100)

# 막대 위에 값 표시
for bar, rate in zip(bars, sex_survival['survival_rate']):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')

# 2. 객실 등급에 따른 생존율
class_survival = titanic.groupby('pclass')['survived'].agg(['count', 'sum', 'mean'])
class_survival['survival_rate'] = class_survival['mean'] * 100

ax2 = axes[0, 1]
bars = ax2.bar(class_survival.index, class_survival['survival_rate'], 
               color=['gold', 'silver', '#CD7F32'], alpha=0.8)
ax2.set_title('객실 등급별 생존율')
ax2.set_xlabel('객실 등급')
ax2.set_ylabel('생존율 (%)')
ax2.set_ylim(0, 100)

for bar, rate in zip(bars, class_survival['survival_rate']):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')

# 3. 나이별 생존율 (나이 구간별)
titanic['age_group'] = pd.cut(titanic['age'], 
                             bins=[0, 12, 18, 35, 60, 100], 
                             labels=['어린이(0-12)', '청소년(13-18)', '청년(19-35)', '중년(36-60)', '노년(61+)'])

age_survival = titanic.groupby('age_group')['survived'].agg(['count', 'sum', 'mean'])
age_survival['survival_rate'] = age_survival['mean'] * 100

ax3 = axes[0, 2]
bars = ax3.bar(range(len(age_survival)), age_survival['survival_rate'], 
               color='lightgreen', alpha=0.8)
ax3.set_title('연령대별 생존율')
ax3.set_xlabel('연령대')
ax3.set_ylabel('생존율 (%)')
ax3.set_xticks(range(len(age_survival)))
ax3.set_xticklabels(age_survival.index, rotation=45)
ax3.set_ylim(0, 100)

for bar, rate in zip(bars, age_survival['survival_rate']):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')

# 4. 성별 + 객실등급 교차 분석
ax4 = axes[1, 0]
cross_survival = titanic.groupby(['pclass', 'sex'])['survived'].mean().unstack()
cross_survival.plot(kind='bar', ax=ax4, color=['lightblue', 'pink'])
ax4.set_title('객실등급 × 성별 생존율')
ax4.set_xlabel('객실 등급')
ax4.set_ylabel('생존율')
ax4.legend(title='성별')
ax4.tick_params(axis='x', rotation=0)

# 5. 요금 분포 (생존자 vs 사망자)
ax5 = axes[1, 1]
titanic_clean = titanic.dropna(subset=['fare'])
survived_fares = titanic_clean[titanic_clean['survived'] == 1]['fare']
died_fares = titanic_clean[titanic_clean['survived'] == 0]['fare']

ax5.hist(died_fares, bins=30, alpha=0.7, label='사망자', color='red', density=True)
ax5.hist(survived_fares, bins=30, alpha=0.7, label='생존자', color='green', density=True)
ax5.set_title('요금별 생존자/사망자 분포')
ax5.set_xlabel('요금')
ax5.set_ylabel('밀도')
ax5.legend()
ax5.set_xlim(0, 200)

# 6. 가족 크기와 생존율
titanic['family_size'] = titanic['sibsp'] + titanic['parch'] + 1
family_survival = titanic.groupby('family_size')['survived'].agg(['count', 'sum', 'mean'])
family_survival = family_survival[family_survival['count'] >= 5]  # 5명 이상인 경우만

ax6 = axes[1, 2]
ax6.plot(family_survival.index, family_survival['mean'], 'o-', linewidth=2, markersize=8, color='purple')
ax6.set_title('가족 크기별 생존율')
ax6.set_xlabel('가족 크기 (본인 포함)')
ax6.set_ylabel('생존율')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 주요 발견사항 요약
print("\n🔍 주요 발견사항")
print("="*50)
print("1. 성별의 강력한 영향:")
print(f"   • 여성 생존율: {sex_survival.loc['female', 'survival_rate']:.1f}%")
print(f"   • 남성 생존율: {sex_survival.loc['male', 'survival_rate']:.1f}%")
print("   → '여성과 어린이 먼저' 원칙이 적용됨")

print("\n2. 객실 등급의 계급적 영향:")
print(f"   • 1등급 생존율: {class_survival.loc[1, 'survival_rate']:.1f}%")
print(f"   • 2등급 생존율: {class_survival.loc[2, 'survival_rate']:.1f}%") 
print(f"   • 3등급 생존율: {class_survival.loc[3, 'survival_rate']:.1f}%")
print("   → 경제적 지위가 생존에 큰 영향")

print("\n3. 연령대별 차이:")
for idx, rate in zip(age_survival.index, age_survival['survival_rate']):
    if not pd.isna(rate):
        print(f"   • {idx}: {rate:.1f}%")
print("   → 어린이의 생존율이 상대적으로 높음"

### 데이터 전처리

머신러닝 모델에 데이터를 입력하기 전에 전처리 과정이 필요합니다.

```python
def preprocess_titanic_data(df):
    """
    타이타닉 데이터를 머신러닝에 적합하도록 전처리하는 함수
    """
    print("🔧 데이터 전처리 시작...")
    data = df.copy()
    
    # 1. 필요한 컬럼만 선택
    features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
    data = data[features + ['survived']].copy()
    
    print(f"선택된 특성: {features}")
    
    # 2. 결측치 확인
    print(f"\n결측치 현황:")
    missing_data = data.isnull().sum()
    for col, missing_count in missing_data.items():
        if missing_count > 0:
            print(f"   • {col}: {missing_count}개 ({missing_count/len(data)*100:.1f}%)")
    
    # 3. 결측치 처리
    # 나이: 중앙값으로 대체
    age_median = data['age'].median()
    data['age'].fillna(age_median, inplace=True)
    print(f"\n나이 결측치를 중앙값 {age_median}으로 대체")
    
    # 요금: 중앙값으로 대체  
    fare_median = data['fare'].median()
    data['fare'].fillna(fare_median, inplace=True)
    print(f"요금 결측치를 중앙값 {fare_median:.2f}으로 대체")
    
    # 탑승항구: 최빈값으로 대체
    embarked_mode = data['embarked'].mode()[0]
    data['embarked'].fillna(embarked_mode, inplace=True)
    print(f"탑승항구 결측치를 최빈값 '{embarked_mode}'으로 대체")
    
    # 4. 범주형 변수를 숫자로 변환
    # 성별: male=0, female=1
    data['sex'] = data['sex'].map({'male': 0, 'female': 1})
    
    # 탑승항구: 원-핫 인코딩
    embarked_dummies = pd.get_dummies(data['embarked'], prefix='embarked')
    data = pd.concat([data, embarked_dummies], axis=1)
    data.drop('embarked', axis=1, inplace=True)
    
    # 5. 새로운 특성 생성 (특성 공학)
    data['family_size'] = data['sibsp'] + data['parch'] + 1
    data['is_alone'] = (data['family_size'] == 1).astype(int)
    data['age_class'] = data['age'] * data['pclass']
    
    print(f"\n새로운 특성 생성:")
    print(f"   • family_size: 가족 크기 (본인 포함)")
    print(f"   • is_alone: 혼자 탑승 여부 (1: 혼자, 0: 가족과 함께)")
    print(f"   • age_class: 나이와 객실등급의 상호작용")
    
    # 6. 최종 결측치 확인
    final_missing = data.isnull().sum().sum()
    print(f"\n전처리 완료! 남은 결측치: {final_missing}개")
    
    return data

# 전처리 실행
processed_data = preprocess_titanic_data(titanic)

print(f"\n📊 전처리된 데이터 정보:")
print(f"   • 데이터 크기: {processed_data.shape}")
print(f"   • 특성 개수: {processed_data.shape[1]-1}개")
print(f"   • 특성명: {list(processed_data.columns[:-1])}")

# 처리된 데이터 샘플 확인
print(f"\n전처리된 데이터 샘플:")
print(processed_data.head())
```

## 5.2.2 로지스틱 회귀 (Logistic Regression)

### 로지스틱 회귀의 핵심 아이디어

로지스틱 회귀는 **확률**을 예측하는 분류 알고리즘입니다. 선형 회귀와 달리 시그모이드 함수를 사용하여 출력값을 0과 1 사이로 제한합니다.

```python
print("🧮 로지스틱 회귀의 핵심 개념")
print("="*50)

# 시그모이드 함수 정의
def sigmoid(x):
    """시그모이드 함수: 임의의 실수를 0과 1 사이의 확률로 변환"""
    return 1 / (1 + np.exp(-x))

# 시그모이드 함수 시각화
x = np.linspace(-10, 10, 100)
y = sigmoid(x)

plt.figure(figsize=(12, 8))

# 첫 번째 서브플롯: 시그모이드 함수
plt.subplot(2, 2, 1)
plt.plot(x, y, 'b-', linewidth=3, label='시그모이드 함수')
plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.8, label='결정 경계 (0.5)')
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
plt.axhline(y=1, color='gray', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
plt.xlabel('입력값 (z)')
plt.ylabel('출력 확률 P(y=1)')
plt.title('시그모이드 함수의 모양')
plt.legend()
plt.grid(True, alpha=0.3)

# 주요 지점 표시
key_points_x = [-5, -2, 0, 2, 5]
key_points_y = [sigmoid(x) for x in key_points_x]
plt.scatter(key_points_x, key_points_y, color='red', s=50, zorder=5)

for x_val, y_val in zip(key_points_x, key_points_y):
    plt.annotate(f'({x_val}, {y_val:.3f})', 
                (x_val, y_val), 
                xytext=(5, 5), 
                textcoords='offset points',
                fontsize=8)

# 두 번째 서브플롯: 선형 회귀 vs 로지스틱 회귀 비교
plt.subplot(2, 2, 2)
x_sample = np.array([-3, -2, -1, 0, 1, 2, 3])
y_binary = np.array([0, 0, 0, 0, 1, 1, 1])  # 실제 이진 라벨

# 선형 회귀 라인 (가상)
linear_y = 0.3 * x_sample + 0.5

# 로지스틱 회귀 라인
logistic_y = sigmoid(2 * x_sample)

plt.scatter(x_sample, y_binary, color='black', s=100, alpha=0.8, label='실제 데이터', zorder=5)
plt.plot(x_sample, linear_y, 'r--', linewidth=2, label='선형 회귀 (부적절)', alpha=0.7)
plt.plot(x_sample, logistic_y, 'b-', linewidth=3, label='로지스틱 회귀 (적절)')
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
plt.xlabel('특성값')
plt.ylabel('예측값')
plt.title('선형 회귀 vs 로지스틱 회귀')
plt.legend()
plt.ylim(-0.2, 1.2)
plt.grid(True, alpha=0.3)

# 세 번째 서브플롯: 로지스틱 회귀의 의사결정 과정
plt.subplot(2, 2, 3)
prob_values = np.linspace(0, 1, 100)
decisions = (prob_values >= 0.5).astype(int)

plt.plot(prob_values, decisions, 'g-', linewidth=3, label='예측 클래스')
plt.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='결정 임계값')
plt.xlabel('예측 확률')
plt.ylabel('예측 클래스')
plt.title('확률에서 클래스로 변환')
plt.legend()
plt.grid(True, alpha=0.3)

# 네 번째 서브플롯: 확률 해석 예시
plt.subplot(2, 2, 4)
probabilities = [0.1, 0.3, 0.5, 0.7, 0.9]
colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
labels = ['매우 낮음\n(10%)', '낮음\n(30%)', '중간\n(50%)', '높음\n(70%)', '매우 높음\n(90%)']

bars = plt.bar(range(len(probabilities)), probabilities, color=colors, alpha=0.8)
plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.8, label='결정 경계')
plt.xlabel('상황')
plt.ylabel('생존 확률')
plt.title('로지스틱 회귀 확률 해석')
plt.xticks(range(len(probabilities)), labels, rotation=45)
plt.legend()

for i, (bar, prob) in enumerate(zip(bars, probabilities)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{prob:.1f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

print("\n🔍 시그모이드 함수의 특징:")
print("   1. S자 모양의 곡선")
print("   2. 출력값이 항상 0과 1 사이 → 확률로 해석 가능")
print("   3. 입력이 0일 때 출력이 0.5 (중립)")
print("   4. 입력이 클수록 1에 가까워짐 (양의 클래스)")
print("   5. 입력이 작을수록 0에 가까워짐 (음의 클래스)")

print("\n🎯 로지스틱 회귀의 장점:")
print("   • 확률을 직접 출력하여 해석이 쉬움")
print("   • 계산이 빠르고 메모리 효율적")
print("   • 선형적으로 분리 가능한 데이터에 효과적")
print("   • 과적합이 잘 발생하지 않음")

print("\n⚠️ 로지스틱 회귀의 한계:")
print("   • 비선형 관계를 잘 포착하지 못함")
print("   • 특성 간 복잡한 상호작용 표현이 어려움")
print("   • 이상치에 민감할 수 있음")
```

### 로지스틱 회귀 모델 구현 및 평가

```python
# 특성(X)과 타겟(y) 분리
X = processed_data.drop('survived', axis=1)
y = processed_data['survived']

print("🔧 로지스틱 회귀 모델 구축")
print("="*50)
print(f"특성 수: {X.shape[1]}개")
print(f"샘플 수: {X.shape[0]}개")
print(f"특성명: {list(X.columns)}")

# 훈련/테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 데이터 분할 결과:")
print(f"   • 훈련 데이터: {X_train.shape[0]}개 ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"   • 테스트 데이터: {X_test.shape[0]}개 ({X_test.shape[0]/len(X)*100:.1f}%)")

# 훈련/테스트 세트의 클래스 분포 확인
train_survival_rate = y_train.mean()
test_survival_rate = y_test.mean()
print(f"   • 훈련 세트 생존율: {train_survival_rate:.3f}")
print(f"   • 테스트 세트 생존율: {test_survival_rate:.3f}")

# 특성 스케일링 (로지스틱 회귀의 성능 향상을 위해)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n🔄 특성 스케일링 완료")
print(f"   • 스케일링 전 평균: {X_train.mean().mean():.3f}")
print(f"   • 스케일링 후 평균: {X_train_scaled.mean():.3f}")
print(f"   • 스케일링 후 표준편차: {X_train_scaled.std():.3f}")

# 로지스틱 회귀 모델 훈련
logistic_model = LogisticRegression(random_state=42, max_iter=1000)
logistic_model.fit(X_train_scaled, y_train)

print(f"\n✅ 로지스틱 회귀 모델 훈련 완료!")

# 예측 수행
y_pred_lr = logistic_model.predict(X_test_scaled)
y_prob_lr = logistic_model.predict_proba(X_test_scaled)[:, 1]  # 생존 확률

# 성능 계산
accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)

print(f"\n📈 로지스틱 회귀 모델 성능:")
print(f"   • 정확도 (Accuracy): {accuracy_lr:.3f}")
print(f"   • 정밀도 (Precision): {precision_lr:.3f}")
print(f"   • 재현율 (Recall): {recall_lr:.3f}")
print(f"   • F1-점수: {f1_lr:.3f}")

# 특성 중요도 (계수) 분석
feature_importance_lr = pd.DataFrame({
    '특성': X.columns,
    '계수': logistic_model.coef_[0]
}).sort_values('계수', key=abs, ascending=False)

print(f"\n🔍 특성 중요도 (로지스틱 회귀 계수):")
for _, row in feature_importance_lr.head(8).iterrows():
    direction = "긍정적" if row['계수'] > 0 else "부정적"
    print(f"   • {row['특성']}: {row['계수']:.3f} ({direction} 영향)")

# 예측 결과 샘플 확인
print(f"\n🎯 예측 결과 샘플 (처음 10개):")
comparison_df = pd.DataFrame({
    '실제값': y_test.iloc[:10].values,
    '예측값': y_pred_lr[:10],
    '생존확률': y_prob_lr[:10].round(3),
    '예측결과': ['생존' if pred == 1 else '사망' for pred in y_pred_lr[:10]]
})
print(comparison_df.to_string(index=False))

# 확률 분포 시각화
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.hist(y_prob_lr[y_test == 0], bins=20, alpha=0.7, label='실제 사망자', color='red', density=True)
plt.hist(y_prob_lr[y_test == 1], bins=20, alpha=0.7, label='실제 생존자', color='green', density=True)
plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='결정 경계')
plt.xlabel('예측된 생존 확률')
plt.ylabel('밀도')
plt.title('로지스틱 회귀: 생존 확률 분포')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.barh(feature_importance_lr.head(8)['특성'], 
         feature_importance_lr.head(8)['계수'], 
         color=['green' if x > 0 else 'red' for x in feature_importance_lr.head(8)['계수']])
plt.xlabel('로지스틱 회귀 계수')
plt.title('특성별 생존에 미치는 영향')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## 5.2.5 서포트 벡터 머신 (Support Vector Machine)

### 서포트 벡터 머신의 마진 최적화

서포트 벡터 머신(SVM)은 **마진을 최대화**하여 가장 안전한 결정 경계를 찾는 알고리즘입니다.

```python
print("⚔️ 서포트 벡터 머신의 핵심 개념")
print("="*50)

# SVM 개념 시각화
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 첫 번째 그래프: 결정 경계와 마진
ax1 = axes[0]

# 가상의 2차원 데이터 생성
np.random.seed(42)
class_0_x = np.random.normal(2, 0.8, 30)
class_0_y = np.random.normal(2, 0.8, 30)
class_1_x = np.random.normal(5, 0.8, 30)
class_1_y = np.random.normal(5, 0.8, 30)

ax1.scatter(class_0_x, class_0_y, c='red', marker='o', s=50, alpha=0.8, label='클래스 0 (사망)')
ax1.scatter(class_1_x, class_1_y, c='green', marker='s', s=50, alpha=0.8, label='클래스 1 (생존)')

# 결정 경계와 마진 시각화
x_line = np.linspace(1, 6, 100)
decision_boundary = x_line  # 대각선 결정 경계
margin_upper = x_line + 0.5
margin_lower = x_line - 0.5

ax1.plot(x_line, decision_boundary, 'b-', linewidth=2, label='결정 경계')
ax1.plot(x_line, margin_upper, 'b--', linewidth=1, alpha=0.7, label='마진 경계')
ax1.plot(x_line, margin_lower, 'b--', linewidth=1, alpha=0.7)
ax1.fill_between(x_line, margin_lower, margin_upper, alpha=0.2, color='blue', label='마진')

# 서포트 벡터 표시 (가상)
support_vectors_x = [2.5, 3.0, 4.5, 5.0]
support_vectors_y = [2.5, 3.0, 4.5, 5.0]
ax1.scatter(support_vectors_x[:2], support_vectors_y[:2], 
           facecolor='none', edgecolor='red', s=100, linewidth=3, label='서포트 벡터')
ax1.scatter(support_vectors_x[2:], support_vectors_y[2:], 
           facecolor='none', edgecolor='green', s=100, linewidth=3)

ax1.set_xlabel('특성 1')
ax1.set_ylabel('특성 2')
ax1.set_title('SVM: 최대 마진 분류기')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 두 번째 그래프: 커널 트릭 개념
ax2 = axes[1]

# 비선형 분리 가능한 데이터 예시
theta = np.linspace(0, 2*np.pi, 30)
inner_x = 2 + 0.5 * np.cos(theta) + np.random.normal(0, 0.1, 30)
inner_y = 2 + 0.5 * np.sin(theta) + np.random.normal(0, 0.1, 30)

outer_x = 2 + 1.5 * np.cos(theta) + np.random.normal(0, 0.2, 30)
outer_y = 2 + 1.5 * np.sin(theta) + np.random.normal(0, 0.2, 30)

ax2.scatter(inner_x, inner_y, c='red', marker='o', s=50, alpha=0.8, label='내부 클래스')
ax2.scatter(outer_x, outer_y, c='green', marker='s', s=50, alpha=0.8, label='외부 클래스')

# 비선형 결정 경계 (원형)
circle_theta = np.linspace(0, 2*np.pi, 100)
boundary_x = 2 + 1.0 * np.cos(circle_theta)
boundary_y = 2 + 1.0 * np.sin(circle_theta)
ax2.plot(boundary_x, boundary_y, 'b-', linewidth=2, label='비선형 결정 경계')

ax2.set_xlabel('특성 1')
ax2.set_ylabel('특성 2')
ax2.set_title('커널 트릭: 비선형 분류')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')

# 세 번째 그래프: 다양한 커널 함수
ax3 = axes[2]

x = np.linspace(-3, 3, 100)
linear_kernel = x
poly_kernel = x**2
rbf_kernel = np.exp(-x**2)

ax3.plot(x, linear_kernel/max(abs(linear_kernel)), 'r-', linewidth=2, label='선형 커널')
ax3.plot(x, poly_kernel/max(poly_kernel), 'g-', linewidth=2, label='다항식 커널')
ax3.plot(x, rbf_kernel, 'b-', linewidth=2, label='RBF 커널')

ax3.set_xlabel('입력값')
ax3.set_ylabel('정규화된 커널 값')
ax3.set_title('다양한 커널 함수')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n🎯 SVM의 핵심 아이디어:")
print("   1. 마진 최대화:")
print("      • 클래스 간 가장 안전한 경계선 찾기")
print("      • 서포트 벡터만 결정 경계에 영향")
print("   2. 커널 트릭:")
print("      • 비선형 문제를 고차원에서 선형으로 변환")
print("      • 다양한 커널 함수 사용 가능")

print("\n🚀 SVM의 장점:")
print("   • 고차원 데이터에서 효과적")
print("   • 메모리 효율적 (서포트 벡터만 저장)")
print("   • 커널을 통한 비선형 문제 해결")
print("   • 과적합에 상대적으로 강함")

print("\n⚠️ SVM의 단점:")
print("   • 대용량 데이터에서 느림")
print("   • 확률 출력을 직접 제공하지 않음")
print("   • 특성 스케일링에 민감함")
print("   • 하이퍼파라미터 튜닝이 중요함")
```

### SVM 모델 구현

```python
# SVM 모델 훈련 (스케일링된 데이터 사용)
svm_model = SVC(
    kernel='rbf',         # RBF 커널 사용
    C=1.0,               # 정규화 파라미터
    gamma='scale',       # 커널 계수
    probability=True,    # 확률 출력 활성화
    random_state=42
)

svm_model.fit(X_train_scaled, y_train)

print("⚔️ SVM 모델 훈련 완료!")
print(f"   • 커널: {svm_model.kernel}")
print(f"   • C 파라미터: {svm_model.C}")
print(f"   • 서포트 벡터 수: {svm_model.n_support_}")
print(f"   • 전체 서포트 벡터 비율: {svm_model.n_support_.sum()/len(X_train)*100:.1f}%")

# 예측 수행
y_pred_svm = svm_model.predict(X_test_scaled)
y_prob_svm = svm_model.predict_proba(X_test_scaled)[:, 1]

# 성능 계산
accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)

print(f"\n📈 SVM 성능:")
print(f"   • 정확도: {accuracy_svm:.3f}")
print(f"   • 정밀도: {precision_svm:.3f}")
print(f"   • 재현율: {recall_svm:.3f}")
print(f"   • F1-점수: {f1_svm:.3f}")
```

## 5.2.6 분류 성능 지표의 완전한 이해

### 혼동 행렬 (Confusion Matrix)

혼동 행렬은 분류 모델의 성능을 종합적으로 보여주는 표입니다.

```python
print("📊 분류 성능 지표 완전 가이드")
print("="*50)

# 모든 모델의 혼동 행렬 계산
models = {
    '로지스틱 회귀': y_pred_lr,
    '의사결정나무': y_pred_tree, 
    '랜덤 포레스트': y_pred_rf,
    'SVM': y_pred_svm
}

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

for idx, (model_name, predictions) in enumerate(models.items()):
    cm = confusion_matrix(y_test, predictions)
    
    # 혼동 행렬 시각화
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['사망', '생존'], 
                yticklabels=['사망', '생존'],
                ax=axes[idx])
    axes[idx].set_title(f'{model_name} 혼동 행렬')
    axes[idx].set_xlabel('예측값')
    axes[idx].set_ylabel('실제값')
    
    # 성능 지표 계산
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # 혼동 행렬 위에 성능 지표 표시
    axes[idx].text(0.02, 0.98, f'정확도: {accuracy:.3f}\n정밀도: {precision:.3f}\n재현율: {recall:.3f}', 
                   transform=axes[idx].transAxes, fontsize=10, 
                   verticalalignment='top', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.show()

# 성능 지표 상세 설명
print("\n📋 성능 지표 상세 설명")
print("="*50)

# 예시를 위한 가상 혼동 행렬
example_cm = np.array([[85, 15], [20, 58]])
tn, fp, fn, tp = example_cm.ravel()

print("혼동 행렬 예시:")
print("              예측값")
print("           사망  생존")
print("실제값 사망   85    15   ← 85명 정확히 예측, 15명 잘못 예측")
print("      생존   20    58   ← 20명 놓침, 58명 정확히 예측")

print(f"\n🔍 성능 지표 계산:")
print(f"   • True Negative (TN): {tn} - 사망을 사망으로 정확히 예측")
print(f"   • False Positive (FP): {fp} - 사망을 생존으로 잘못 예측")
print(f"   • False Negative (FN): {fn} - 생존을 사망으로 잘못 예측") 
print(f"   • True Positive (TP): {tp} - 생존을 생존으로 정확히 예측")

accuracy_ex = (tp + tn) / (tp + tn + fp + fn)
precision_ex = tp / (tp + fp)
recall_ex = tp / (tp + fn)
f1_ex = 2 * (precision_ex * recall_ex) / (precision_ex + recall_ex)

print(f"\n📊 계산된 성능 지표:")
print(f"   • 정확도 (Accuracy) = (TP+TN)/(TP+TN+FP+FN) = {accuracy_ex:.3f}")
print(f"     → 전체 예측 중 맞힌 비율")
print(f"   • 정밀도 (Precision) = TP/(TP+FP) = {precision_ex:.3f}")
print(f"     → '생존'으로 예측한 것 중 실제 생존자 비율")
print(f"   • 재현율 (Recall) = TP/(TP+FN) = {recall_ex:.3f}")
print(f"     → 실제 생존자 중 찾아낸 비율")
print(f"   • F1-점수 = 2×(정밀도×재현율)/(정밀도+재현율) = {f1_ex:.3f}")
print(f"     → 정밀도와 재현율의 조화평균")

# 실제 의미 해석
print(f"\n💡 실무적 해석:")
print(f"   • 정밀도가 높다 = 거짓 경보가 적다")
print(f"   • 재현율이 높다 = 놓치는 경우가 적다")
print(f"   • F1-점수가 높다 = 정밀도와 재현율이 균형적으로 좋다")
```

### 모델 성능 종합 비교

```python
# 모든 모델의 성능 종합 비교
print("\n🏆 모델 성능 종합 비교")
print("="*80)

performance_data = {
    '모델': ['로지스틱 회귀', '의사결정나무', '랜덤 포레스트', 'SVM'],
    '정확도': [accuracy_lr, accuracy_tree, accuracy_rf, accuracy_svm],
    '정밀도': [precision_lr, precision_tree, precision_rf, precision_svm],
    '재현율': [recall_lr, recall_tree, recall_rf, recall_svm],
    'F1-점수': [f1_lr, f1_tree, f1_rf, f1_svm]
}

performance_df = pd.DataFrame(performance_data)
print(performance_df.round(3))

# 성능 시각화
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 1. 성능 지표별 비교
metrics = ['정확도', '정밀도', '재현율', 'F1-점수']
x = np.arange(len(performance_data['모델']))
width = 0.2

for i, metric in enumerate(metrics):
    axes[0].bar(x + i*width, performance_data[metric], width, 
                label=metric, alpha=0.8)

axes[0].set_xlabel('모델')
axes[0].set_ylabel('성능 점수')
axes[0].set_title('모델별 성능 지표 비교')
axes[0].set_xticks(x + width * 1.5)
axes[0].set_xticklabels(performance_data['모델'])
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(0, 1)

# 2. 정밀도-재현율 산점도
axes[1].scatter(performance_data['재현율'], performance_data['정밀도'], 
                s=200, alpha=0.7, c=['red', 'green', 'blue', 'orange'])

for i, model in enumerate(performance_data['모델']):
    axes[1].annotate(model, 
                     (performance_data['재현율'][i], performance_data['정밀도'][i]),
                     xytext=(5, 5), textcoords='offset points')

axes[1].set_xlabel('재현율 (Recall)')
axes[1].set_ylabel('정밀도 (Precision)')
axes[1].set_title('정밀도-재현율 관계')
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(0, 1)
axes[1].set_ylim(0, 1)

# 대각선 (이상적인 균형점)
axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='이상적 균형')
axes[1].legend()

plt.tight_layout()
plt.show()

# 최고 성능 모델 식별
best_accuracy = performance_df.loc[performance_df['정확도'].idxmax()]
best_f1 = performance_df.loc[performance_df['F1-점수'].idxmax()]

print(f"\n🥇 최고 성능 결과:")
print(f"   • 정확도 최고: {best_accuracy['모델']} ({best_accuracy['정확도']:.3f})")
print(f"   • F1-점수 최고: {best_f1['모델']} ({best_f1['F1-점수']:.3f})")

# 모델별 특징 요약
print(f"\n📋 모델별 특징 요약:")
model_characteristics = {
    '로지스틱 회귀': "해석이 쉽고 빠름, 선형 관계에 적합",
    '의사결정나무': "규칙이 명확하나 과적합 위험", 
    '랜덤 포레스트': "안정적이고 정확하나 해석이 어려움",
    'SVM': "고차원에서 강력하나 대용량 데이터에 부적합"
}

for model, char in model_characteristics.items():
    print(f"   • {model}: {char}")
```

## 5.2.7 직접 해보기 - 연습 문제

### 연습 문제 1: 코드 완성하기

```python
print("🏃‍♂️ 연습 문제 1: 새로운 승객의 생존 예측")
print("="*50)

# 새로운 승객 데이터 (가상)
new_passengers = pd.DataFrame({
    'pclass': [1, 3, 2, 3, 1],
    'sex': [1, 0, 1, 1, 0],  # 1: 여성, 0: 남성
    'age': [25, 35, 8, 22, 50],
    'sibsp': [0, 1, 2, 0, 1],
    'parch': [0, 0, 1, 0, 0],
    'fare': [100, 15, 50, 25, 200],
    'embarked_C': [1, 0, 0, 0, 1],
    'embarked_Q': [0, 0, 1, 0, 0],
    'embarked_S': [0, 1, 0, 1, 0],
    'family_size': [1, 2, 4, 1, 2],
    'is_alone': [1, 0, 0, 1, 0],
    'age_class': [25, 105, 16, 66, 50]
})

print("새로운 승객 정보:")
passenger_descriptions = [
    "25세 1등급 여성, 혼자 탑승, 요금 $100",
    "35세 3등급 남성, 배우자와 탑승, 요금 $15", 
    "8세 2등급 여성, 가족 4명과 탑승, 요금 $50",
    "22세 3등급 여성, 혼자 탑승, 요금 $25",
    "50세 1등급 남성, 배우자와 탑승, 요금 $200"
]

for i, desc in enumerate(passenger_descriptions):
    print(f"   승객 {i+1}: {desc}")

# TODO: 학생이 완성해야 할 부분
print(f"\n🎯 연습 과제:")
print(f"   1. 새로운 승객 데이터를 스케일링하세요")
print(f"   2. 4개 모델로 생존 확률을 예측하세요")  
print(f"   3. 각 승객별 생존 가능성을 해석하세요")

# 정답 예시 (주석 처리)
"""
# 1. 데이터 스케일링
new_passengers_scaled = scaler.transform(new_passengers)

# 2. 예측 수행
lr_pred = logistic_model.predict_proba(new_passengers_scaled)[:, 1]
tree_pred = tree_model.predict_proba(new_passengers)[:, 1]
rf_pred = rf_model.predict_proba(new_passengers)[:, 1]
svm_pred = svm_model.predict_proba(new_passengers_scaled)[:, 1]

# 3. 결과 정리
results = pd.DataFrame({
    '승객': [f'승객 {i+1}' for i in range(5)],
    '로지스틱회귀': lr_pred,
    '의사결정나무': tree_pred, 
    '랜덤포레스트': rf_pred,
    'SVM': svm_pred
})
print(results.round(3))
"""
```

### 연습 문제 2: 개념 확인

```python
print("\n📝 연습 문제 2: 개념 확인 문제")
print("="*50)

questions = [
    {
        "질문": "로지스틱 회귀에서 시그모이드 함수를 사용하는 이유는?",
        "선택지": [
            "A. 계산 속도를 높이기 위해",
            "B. 출력값을 0과 1 사이로 제한하여 확률로 해석하기 위해", 
            "C. 비선형 관계를 모델링하기 위해",
            "D. 메모리 사용량을 줄이기 위해"
        ],
        "정답": "B"
    },
    {
        "질문": "랜덤 포레스트가 단일 의사결정나무보다 성능이 좋은 주된 이유는?",
        "선택지": [
            "A. 계산이 더 빠르기 때문",
            "B. 해석이 더 쉽기 때문",
            "C. 여러 모델의 예측을 결합하여 과적합을 줄이기 때문",
            "D. 메모리를 더 적게 사용하기 때문"
        ],
        "정답": "C"
    },
    {
        "질문": "의료 진단에서 암 환자를 놓치는 것이 치명적일 때 가장 중요한 성능 지표는?",
        "선택지": [
            "A. 정확도 (Accuracy)",
            "B. 정밀도 (Precision)",
            "C. 재현율 (Recall)", 
            "D. F1-점수"
        ],
        "정답": "C"
    }
]

for i, q in enumerate(questions, 1):
    print(f"\n질문 {i}: {q['질문']}")
    for choice in q['선택지']:
        print(f"   {choice}")
    print(f"   정답: {q['정답']}")

print(f"\n💡 정답 해설:")
print(f"   1. B - 시그모이드 함수는 모든 실수를 0~1 사이 값으로 변환하여 확률로 해석 가능")
print(f"   2. C - 앙상블 효과로 개별 모델의 오류를 상쇄하고 일반화 성능 향상")
print(f"   3. C - 실제 환자를 놓치지 않는 것(재현율)이 거짓 진단(정밀도)보다 중요")
```

## 요약 및 핵심 정리

```python
print("\n📚 5장 Part 2 핵심 정리")
print("="*50)

key_points = {
    "분류 알고리즘": {
        "로지스틱 회귀": "확률 기반, 해석 용이, 선형 분리",
        "의사결정나무": "규칙 기반, 직관적, 과적합 주의",
        "랜덤 포레스트": "앙상블, 안정적, 높은 성능",
        "SVM": "마진 최대화, 커널 트릭, 고차원 효과적"
    },
    "성능 지표": {
        "정확도": "전체 정확히 예측한 비율",
        "정밀도": "양성으로 예측한 것 중 실제 양성 비율",
        "재현율": "실제 양성 중 찾아낸 비율", 
        "F1-점수": "정밀도와 재현율의 조화평균"
    },
    "실무 활용": {
        "알고리즘 선택": "데이터 크기, 해석 필요성, 성능 요구사항 고려",
        "성능 평가": "비즈니스 목적에 맞는 지표 선택",
        "모델 개선": "특성 공학, 하이퍼파라미터 튜닝, 앙상블"
    }
}

for category, items in key_points.items():
    print(f"\n🎯 {category}:")
    for item, description in items.items():
        print(f"   • {item}: {description}")

print(f"\n🚀 다음 파트 예고:")
print(f"   Part 3에서는 연속적인 수치를 예측하는 '회귀 알고리즘'을 학습합니다.")
print(f"   주택 가격 예측을 통해 선형/다항식 회귀, Ridge/Lasso 정규화를 마스터하게 됩니다!")
```

---

**🎯 이번 파트에서 배운 것**
- 로지스틱 회귀의 시그모이드 함수와 확률 기반 분류 원리
- 의사결정나무의 규칙 기반 분류와 정보 이득 개념  
- 랜덤 포레스트의 앙상블 원리와 과적합 방지 효과
- SVM의 마진 최대화와 커널 트릭 활용법
- 혼동 행렬, 정밀도, 재현율, F1-점수 등 성능 지표 완전 이해
- 타이타닉 생존 예측을 통한 실전 분류 모델링 경험

**🚀 다음 파트에서는**
연속적인 수치를 예측하는 **회귀 알고리즘**을 학습하고, 캘리포니아 주택 가격 예측 프로젝트를 진행합니다!


