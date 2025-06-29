# 2장 Part 3: 기술 통계와 데이터 요약

## 학습 목표
- 중심경향치(평균, 중앙값, 최빈값)의 개념을 이해하고 상황에 맞게 선택할 수 있다.
- 산포도(표준편차, 분산, 범위)를 계산하고 데이터의 특성에 맞게 해석할 수 있다.
- 그룹별 통계량을 계산하고 비교 분석을 통해 인사이트를 도출할 수 있다.
- 피벗 테이블과 교차표를 활용하여 다차원 데이터를 효과적으로 요약할 수 있다.

## 이번 Part 미리보기

기술 통계(Descriptive Statistics)는 복잡한 데이터셋을 몇 개의 숫자로 요약하여 데이터의 전체적인 특성을 파악하는 방법입니다. 
마치 긴 소설을 한 줄로 요약하는 것처럼, 수백 개의 데이터 포인트를 대표하는 값들로 표현할 수 있습니다.

예를 들어, 100명의 학생 성적 데이터가 있다면 "평균 85점, 표준편차 10점"이라는 두 개의 숫자만으로도 전체 성적 분포의 특성을 파악할 수 있습니다.

이번 Part에서는 이전 Part에서 배운 EDA 기법을 바탕으로, 데이터를 수치적으로 요약하는 방법을 단계별로 학습하겠습니다.

---

## 실습 환경 준비

### 1. 필요한 라이브러리 설치

실습을 시작하기 전에 필요한 라이브러리들을 설치해보겠습니다.

```bash
# 터미널 또는 명령 프롬프트에서 실행
pip install pandas numpy matplotlib seaborn

# 만약 Jupyter Notebook을 사용한다면
pip install jupyter
```

### 2. 실습 데이터 준비

이번 실습에서는 타이타닉 데이터셋을 사용합니다. 이 데이터는 인터넷에서 직접 불러올 수 있어 별도의 파일 다운로드가 필요하지 않습니다.

### 3. 작업 환경 설정

- **추천 환경**: Jupyter Notebook 또는 Google Colab
- **대안 환경**: VS Code, PyCharm 등의 IDE
- **실행 방법**: 각 코드 블록을 순서대로 실행

---

## 핵심 개념 설명

### 📊 기술 통계란 무엇인가?

기술 통계는 데이터를 **설명(describe)**하고 **요약(summarize)**하는 통계 방법입니다. 

**🔍 실생활 비유로 이해하기**
- 학급 성적표: "우리 반 평균 점수는 78점입니다"
- 날씨 예보: "오늘 최고기온 25도, 최저기온 15도"
- 쇼핑몰 리뷰: "평점 4.2점 (5점 만점)"

이처럼 복잡한 정보를 간단한 숫자로 요약하는 것이 기술 통계의 핵심입니다.

### 🎯 중심경향치 (어디에 모여있나?)

>중심경향치(central tendency)는 **데이터 집합의 중심 또는 대표값을 나타내는 통계적 지표**입니다. 
일반적으로 **평균, 중앙값, 최빈값** 등이 중심경향치로 사용됩니다. 
이 값들은 데이터가 집중되는 경향을 보여주어 데이터 전체의 특징을 하나의 숫자로 요약할 수 있게 합니다. 

>중심경향치의 활용:
데이터 분석에서 데이터의 분포를 파악하고, 데이터의 특징을 요약하는데 사용됩니다.
데이터의 중심을 나타내는 지표를 통해 데이터의 전반적인 경향을 이해할 수 있습니다.
예를 들어, 평균 소득, 중앙값 소득 등을 통해 특정 지역 또는 집단의 소득 수준을 파악할 수 있습니다.
데이터 분석 결과 보고 시, 중심경향치를 포함하여 데이터의 특징을 설명할 수 있습니다. 

#### 1️⃣ 평균 (Mean) - 가장 일반적인 대표값

**개념**:
>데이터 값들의 합을 데이터 개수로 나눈 값입니다. 데이터의 중심을 나타내는 대표적인 지표이지만, 극단적인 값(이상치)에 영향을 많이 받는 단점이 있습니다.

**계산 공식**: (값1 + 값2 + ... + 값n) ÷ n

**언제 사용하나요?**
- ✅ 데이터가 정규분포를 따를 때
- ✅ 이상치가 많지 않을 때
- ❌ 극값(아주 크거나 작은 값)이 많을 때는 부적절

**실생활 예시**:
```
학생 5명의 키: 160cm, 165cm, 170cm, 175cm, 200cm
평균 키 = (160+165+170+175+200) ÷ 5 = 174cm
→ 하지만 200cm가 평균을 끌어올렸다!
```

#### 2️⃣ 중앙값 (Median) - 이상치에 강한 대표값

**개념**:
>데이터를 크기 순서대로 정렬했을 때 중앙에 위치하는 값입니다. 
평균과 달리 극단적인 값에 영향을 덜 받으므로 데이터 분포가 비대칭적인 경우 유용하게 사용됩니다.

**계산 방법**:
1. 데이터를 오름차순으로 정렬
2. 가운데 값을 선택 (데이터 개수가 홀수일 때)
3. 가운데 두 값의 평균 (데이터 개수가 짝수일 때)

**언제 사용하나요?**
- ✅ 이상치가 많을 때
- ✅ 왜곡된 분포를 가진 데이터
- ✅ 소득, 집값 등 극값이 존재하는 데이터

**동일한 예시로 비교**:
```
학생 5명의 키: 160, 165, 170, 175, 200 (정렬된 상태)
중앙값 = 170cm (가운데 값)
→ 이상치 200cm의 영향을 받지 않음!
```

#### 3️⃣ 최빈값 (Mode) - 가장 자주 나타나는 값

**개념**:
>데이터에서 가장 자주 나타나는 값입니다. 
데이터의 분포가 특정 값에 집중될 때 유용하게 사용됩니다. 

**특징**:
- 범주형 데이터에도 적용 가능한 유일한 중심경향치
- 여러 개의 최빈값이 존재할 수 있음
- 수치형, 범주형 데이터 모두에 사용 가능

**예시**:
```
좋아하는 색깔 조사: 빨강, 파랑, 빨강, 초록, 빨강, 노랑, 파랑, 빨강
최빈값 = 빨강 (4번 등장)
```

### 📏 산포도 (얼마나 퍼져있나?)

>산포도는 **"데이터가 중심값 주변에 얼마나 퍼져 있는가?"**를 알려주는 지표입니다.
즉, 자료들이 평균이나 중앙값과 같은 대표값으로부터 얼마나 떨어져 있는지를 측정하여 데이터의 변동성을 파악하는 데 사용됩니다. 
산포도가 크다는 것은 데이터가 널리 퍼져 있다는 의미이고, 작다는 것은 데이터가 밀집되어 있다는 의미입니다. 

#### 1️⃣ 범위 (Range) - 가장 간단한 산포도

**개념**: 
>최댓값에서 최솟값을 뺀 값으로, 데이터의 흩어진 정도를 간단하게 나타냅니다.

**장점**: 계산이 간단하고 직관적
**단점**: 이상치에 매우 민감

**예시**:
```
시험 점수: 60, 70, 75, 80, 95
범위 = 95 - 60 = 35점
```

#### 2️⃣ 분산 (Variance) - 평균적인 차이의 제곱

**개념**: 
>각 데이터가 평균으로부터 떨어진 거리의 제곱의 평균으로, 데이터의 평균적인 흩어진 정도를 나타냅니다.

**왜 제곱을 사용하나요?**
- 음수와 양수를 더했을 때 상쇄되는 것을 방지
- 큰 차이에 더 많은 가중치를 부여

#### 3️⃣ 표준편차 (Standard Deviation) - 실무에서 가장 많이 사용

**개념**: 
>분산의 제곱근으로, 분산보다 해석이 쉽고 많이 사용됩니다. 표준편차 역시 데이터의 흩어진 정도를 나타냅니다.

**왜 표준편차를 사용하나요?**
- 원래 데이터와 같은 단위를 가짐
- 해석이 직관적임

**68-95-99.7 규칙** (정규분포에서):
- 평균 ± 1 표준편차: 약 68%의 데이터
- 평균 ± 2 표준편차: 약 95%의 데이터
- 평균 ± 3 표준편차: 약 99.7%의 데이터

---

## 핵심 기술 / 코드 구현

### 🚀 1단계: 환경 설정과 데이터 불러오기

실습을 시작하기 전에 필요한 라이브러리를 불러오고 기본 설정을 하겠습니다.

```python
# 1단계: 필요한 라이브러리 불러오기
# 각 라이브러리의 역할을 이해해보세요

import pandas as pd      # 데이터 조작과 분석을 위한 라이브러리
import numpy as np       # 수치 계산을 위한 라이브러리  
import matplotlib.pyplot as plt  # 기본 그래프 그리기 라이브러리
import seaborn as sns    # 통계 시각화를 위한 고급 라이브러리

# 2단계: 한글 폰트 설정 (그래프에서 한글이 깨지는 것을 방지)
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows용 한글 폰트
plt.rcParams['axes.unicode_minus'] = False     # 마이너스 기호 깨짐 방지

# 3단계: 경고 메시지 숨기기 (선택사항)
import warnings
warnings.filterwarnings('ignore')

print("✅ 라이브러리 불러오기 완료!")
print("📊 이제 데이터 분석을 시작할 수 있습니다.")
```

```python
# 4단계: 데이터 불러오기
# 인터넷에서 타이타닉 데이터를 직접 불러옵니다

print("🚢 타이타닉 데이터 불러오는 중...")

# read_csv() 함수로 CSV 파일을 불러와서 DataFrame으로 저장
df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

print(f"✅ 데이터 불러오기 완료!")
print(f"📏 데이터 크기: {df.shape[0]}행(승객 수) × {df.shape[1]}열(정보 종류)")
print(f"💾 메모리 사용량: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
```

### 🔍 2단계: 데이터 구조 이해하기

통계를 계산하기 전에 어떤 데이터를 다루고 있는지 파악해보겠습니다.


**데이터 확인하기**
아래의 코드로 데이터의 전체 구조와 주요 열들의 값을 간략히 파악합니다.
```python
# 데이터의 첫 5행 확인
print("📋 데이터 미리보기:")
print(df.head())

print("\n" + "="*50)
```
**열별 정보 확인**
- 각 열의 데이터 타입 (숫자형, 문자열형 등)
- 결측값(NaN) 존재 여부 확인
- 총 데이터 개수 확인 (행 수)

>해석 포인트:
Age, Cabin, Embarked 등에 결측값 존재 가능성
object 타입은 문자열 또는 범주형 데이터
```python
# 각 컬럼의 정보 확인  
print("📊 컬럼별 정보:")
print(df.info())

print("\n" + "="*50)

# 각 컬럼에 어떤 데이터가 들어있는지 확인
print("🔍 각 컬럼의 의미:")
column_meanings = {
    'PassengerId': '승객 고유 번호',
    'Survived': '생존 여부 (0: 사망, 1: 생존)',
    'Pclass': '객실 등급 (1: 1등석, 2: 2등석, 3: 3등석)',
    'Name': '승객 이름',
    'Sex': '성별 (male: 남성, female: 여성)',
    'Age': '나이 (년)',
    'SibSp': '함께 탑승한 형제자매/배우자 수',
    'Parch': '함께 탑승한 부모/자녀 수',
    'Ticket': '티켓 번호',
    'Fare': '운임 (파운드)',
    'Cabin': '객실 번호',
    'Embarked': '탑승 항구 (C: Cherbourg, Q: Queenstown, S: Southampton)'
}

for col, meaning in column_meanings.items():
    print(f"  • {col}: {meaning}")
```

### 📊 3단계: 기본 통계량 계산하기

이제 본격적으로 통계량을 계산해보겠습니다. Pandas의 `describe()` 함수는 한 번에 여러 통계량을 보여줍니다.

```python
# 전체 데이터의 기본 통계량 확인
print("📈 전체 데이터 기본 통계량:")
print(df.describe())

print("\n" + "="*50)

# describe() 결과 해석하기
print("🔍 describe() 결과 해석:")
print("  • count: 결측치가 아닌 데이터의 개수")
print("  • mean: 평균값")
print("  • std: 표준편차")
print("  • min: 최소값")
print("  • 25%: 1사분위수 (하위 25%)")
print("  • 50%: 2사분위수 (중앙값)")
print("  • 75%: 3사분위수 (상위 25%)")
print("  • max: 최대값")
```

```python
# 특정 컬럼(나이)에 대한 상세 분석
print("👥 나이(Age) 데이터 상세 분석:")

# 1단계: 결측치 제거 (NaN 값이 있는 행 제외)
age_data = df['Age'].dropna()  # dropna()는 NaN 값을 제거하는 함수

print(f"📊 전체 승객 수: {len(df)}명")
print(f"📊 나이 정보가 있는 승객 수: {len(age_data)}명")
print(f"📊 나이 정보가 없는 승객 수: {len(df) - len(age_data)}명")

print("\n" + "-"*30)

# 2단계: 중심경향치 계산
mean_age = age_data.mean()      # 평균
median_age = age_data.median()  # 중앙값
mode_age = age_data.mode()[0]   # 최빈값 (mode()는 Series를 반환하므로 [0]으로 첫 번째 값 선택)

print("🎯 중심경향치:")
print(f"  • 평균 나이: {mean_age:.2f}세")
print(f"  • 중앙값 나이: {median_age:.2f}세") 
print(f"  • 최빈값 나이: {mode_age:.0f}세")

# 3단계: 산포도 계산
std_age = age_data.std()        # 표준편차
var_age = age_data.var()        # 분산
range_age = age_data.max() - age_data.min()  # 범위

print("\n📏 산포도:")
print(f"  • 표준편차: {std_age:.2f}세")
print(f"  • 분산: {var_age:.2f}")
print(f"  • 범위: {range_age:.0f}세 (최소 {age_data.min():.0f}세 ~ 최대 {age_data.max():.0f}세)")

# 4단계: 사분위수 계산
q1 = age_data.quantile(0.25)    # 1사분위수 (25%)
q3 = age_data.quantile(0.75)    # 3사분위수 (75%)
iqr = q3 - q1                   # 사분위수 범위 (IQR)

print("\n📐 사분위수:")
print(f"  • 1사분위수 (Q1): {q1:.1f}세")
print(f"  • 3사분위수 (Q3): {q3:.1f}세") 
print(f"  • 사분위수 범위 (IQR): {iqr:.1f}세")
```

### 🎨 4단계: 통계량 시각화하기

숫자만으로는 이해하기 어려우니 그래프로 시각화해보겠습니다.

```python
# 나이 분포 시각화 (4개의 그래프를 한 번에)
plt.figure(figsize=(15, 10))  # 그래프 크기 설정 (가로 15인치, 세로 10인치)

# 첫 번째 그래프: 히스토그램
plt.subplot(2, 2, 1)  # 2행 2열 중 첫 번째 위치
plt.hist(age_data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(mean_age, color='red', linestyle='--', linewidth=2, label=f'평균: {mean_age:.1f}세')
plt.axvline(median_age, color='green', linestyle='--', linewidth=2, label=f'중앙값: {median_age:.1f}세')
plt.title('나이 분포 히스토그램\n(빨간선: 평균, 초록선: 중앙값)', fontsize=12)
plt.xlabel('나이 (세)')
plt.ylabel('빈도 (명)')
plt.legend()  # 범례 표시
plt.grid(True, alpha=0.3)  # 격자 표시

# 두 번째 그래프: 박스플롯
plt.subplot(2, 2, 2)
box_plot = plt.boxplot(age_data, patch_artist=True)  # 색칠된 박스플롯
box_plot['boxes'][0].set_facecolor('lightcoral')   # 박스 색상 설정
plt.title('나이 분포 박스플롯', fontsize=12)
plt.ylabel('나이 (세)')
plt.grid(True, alpha=0.3)

# 박스플롯 해석 가이드 추가
plt.text(1.3, age_data.max()-5, f'최대값: {age_data.max():.0f}세', fontsize=9)
plt.text(1.3, q3, f'Q3: {q3:.1f}세', fontsize=9)
plt.text(1.3, median_age, f'중앙값: {median_age:.1f}세', fontsize=9)
plt.text(1.3, q1, f'Q1: {q1:.1f}세', fontsize=9)
plt.text(1.3, age_data.min(), f'최소값: {age_data.min():.0f}세', fontsize=9)

# 세 번째 그래프: 밀도 곡선
plt.subplot(2, 2, 3)
age_data.plot(kind='density', color='purple', linewidth=2)  # 밀도 곡선
plt.axvline(mean_age, color='red', linestyle='--', label=f'평균: {mean_age:.1f}세')
plt.axvline(median_age, color='green', linestyle='--', label=f'중앙값: {median_age:.1f}세')
plt.title('나이 분포 밀도 곡선', fontsize=12)
plt.xlabel('나이 (세)')
plt.ylabel('밀도')
plt.legend()
plt.grid(True, alpha=0.3)

# 네 번째 그래프: 누적분포
plt.subplot(2, 2, 4)
age_data.plot(kind='hist', cumulative=True, density=True, alpha=0.7, color='orange')
plt.title('나이 누적분포', fontsize=12)
plt.xlabel('나이 (세)')
plt.ylabel('누적 비율')
plt.grid(True, alpha=0.3)

plt.tight_layout()  # 그래프 간격 자동 조정
plt.show()

# 📊 그래프 해석 가이드
print("\n📊 그래프 해석 가이드:")
print("🔹 히스토그램: 나이별 승객 수를 막대로 표현")
print("🔹 박스플롯: 5개 요약 통계량(최소값, Q1, 중앙값, Q3, 최대값)을 한눈에 확인")
print("🔹 밀도 곡선: 히스토그램을 부드러운 곡선으로 표현")
print("🔹 누적분포: 특정 나이 이하의 승객 비율을 확인")
```

---

## 상세 예제 / 미니 프로젝트

### 🚢 그룹별 통계량 분석: 객실 등급에 따른 차이 발견하기

실제 데이터 분석에서는 전체 통계량보다 **그룹별 비교**가 더 의미 있는 인사이트를 제공합니다. 타이타닉 데이터에서 객실 등급에 따른 차이를 분석해보겠습니다.

#### 🔍 5단계: 그룹별 기본 통계량 계산

```python
# 객실 등급별 나이 통계 분석
print("🎩 객실 등급별 나이 통계 분석:")
print("="*50)

# groupby() 함수를 사용한 그룹별 통계
# agg() 함수로 여러 통계량을 한 번에 계산
group_stats = df.groupby('Pclass')['Age'].agg([
    'count',    # 데이터 개수
    'mean',     # 평균
    'median',   # 중앙값  
    'std',      # 표준편차
    'min',      # 최소값
    'max'       # 최대값
]).round(2)  # 소수점 둘째 자리까지 반올림

print(group_stats)

print("\n🔍 해석:")
for pclass in [1, 2, 3]:
    stats = group_stats.loc[pclass]
    print(f"\n🎫 {pclass}등석:")
    print(f"  • 승객 수: {stats['count']:.0f}명")
    print(f"  • 평균 나이: {stats['mean']:.1f}세")
    print(f"  • 중앙값 나이: {stats['median']:.1f}세")
    print(f"  • 표준편차: {stats['std']:.1f}세")
    print(f"  • 연령대: {stats['min']:.0f}세 ~ {stats['max']:.0f}세")
```

#### 🎨 6단계: 그룹별 시각화 비교

```python
# 객실 등급별 나이 분포 시각화
plt.figure(figsize=(16, 12))

# 첫 번째 그래프: 박스플롯으로 분포 비교
plt.subplot(2, 3, 1)
sns.boxplot(data=df, x='Pclass', y='Age', palette='Set2')
plt.title('객실 등급별 나이 분포 (박스플롯)', fontsize=12, fontweight='bold')
plt.xlabel('객실 등급')
plt.ylabel('나이 (세)')
plt.grid(True, alpha=0.3)

# 두 번째 그래프: 바이올린 플롯 (분포 모양까지 확인)
plt.subplot(2, 3, 2)
sns.violinplot(data=df, x='Pclass', y='Age', palette='Set1')
plt.title('객실 등급별 나이 분포 (바이올린플롯)', fontsize=12, fontweight='bold')
plt.xlabel('객실 등급')
plt.ylabel('나이 (세)')
plt.grid(True, alpha=0.3)

# 세 번째 그래프: 평균 나이 막대 그래프
plt.subplot(2, 3, 3)
mean_ages = group_stats['mean']
bars = plt.bar(['1등석', '2등석', '3등석'], mean_ages.values, 
               color=['gold', 'silver', 'brown'], alpha=0.7)
plt.title('객실 등급별 평균 나이', fontsize=12, fontweight='bold')
plt.ylabel('평균 나이 (세)')

# 막대 위에 값 표시
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{height:.1f}세', ha='center', va='bottom')

plt.grid(True, alpha=0.3)

# 네 번째 그래프: 생존율 분석
plt.subplot(2, 3, 4)
survival_by_class = df.groupby('Pclass')['Survived'].mean()
bars = plt.bar(['1등석', '2등석', '3등석'], survival_by_class.values,
               color=['lightgreen', 'orange', 'lightcoral'], alpha=0.7)
plt.title('객실 등급별 생존율', fontsize=12, fontweight='bold')
plt.ylabel('생존율')

# 막대 위에 백분율 표시
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.1%}', ha='center', va='bottom')

plt.grid(True, alpha=0.3)

# 다섯 번째 그래프: 성별 생존율 비교
plt.subplot(2, 3, 5)
gender_survival = df.groupby('Sex')['Survived'].mean()
bars = plt.bar(['여성', '남성'], gender_survival.values,
               color=['pink', 'lightblue'], alpha=0.7)
plt.title('성별 생존율', fontsize=12, fontweight='bold')
plt.ylabel('생존율')

# 막대 위에 백분율 표시
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.1%}', ha='center', va='bottom')

plt.grid(True, alpha=0.3)

# 여섯 번째 그래프: 운임 분포
plt.subplot(2, 3, 6)
sns.boxplot(data=df, x='Pclass', y='Fare', palette='viridis')
plt.title('객실 등급별 운임 분포', fontsize=12, fontweight='bold')
plt.xlabel('객실 등급')
plt.ylabel('운임 (파운드)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 📊 종합 인사이트
print("\n🔍 발견된 인사이트:")
print("1️⃣ 객실 등급이 높을수록 평균 나이가 높음 (부유한 기성세대)")
print("2️⃣ 1등석 승객의 생존율이 가장 높음 (사회적 지위의 영향)")
print("3️⃣ 여성의 생존율이 남성보다 훨씬 높음 ('여성과 어린이 먼저' 원칙)")
print("4️⃣ 객실 등급별 운임 차이가 매우 큼 (계급 사회 반영)")
```

### 🧮 피벗 테이블: 다차원 데이터 분석의 핵심

피벗 테이블은 **두 개 이상의 범주형 변수**를 조합하여 데이터를 요약하는 강력한 도구입니다.

#### 🔍 7단계: 기본 피벗 테이블 만들기

```python
# 객실 등급별, 성별별 평균 나이 피벗 테이블
print("📊 피벗 테이블 1: 객실 등급별, 성별별 평균 나이")
print("="*50)

# pivot_table() 함수 사용법 설명
# values: 계산할 값 (나이)
# index: 행으로 사용할 변수 (객실 등급)  
# columns: 열로 사용할 변수 (성별)
# aggfunc: 집계 함수 (평균)
pivot_age = pd.pivot_table(df, 
                          values='Age',      # 계산할 값
                          index='Pclass',    # 행 (세로축)
                          columns='Sex',     # 열 (가로축)
                          aggfunc='mean')    # 집계 함수

print(pivot_age.round(2))

print("\n🔍 해석:")
print("💡 1등석 남성이 평균 나이가 가장 높음 (기성 부유층)")
print("💡 3등석에서는 성별에 따른 나이 차이가 적음 (젊은 이민자층)")

print("\n" + "="*60)

# 객실 등급별, 성별별 생존율 피벗 테이블
print("📊 피벗 테이블 2: 객실 등급별, 성별별 생존율")
print("="*50)

pivot_survival = pd.pivot_table(df,
                               values='Survived',
                               index='Pclass',
                               columns='Sex',
                               aggfunc='mean')

print(pivot_survival.round(3))

print("\n🔍 해석:")
print("💡 1등석 여성의 생존율이 가장 높음 (96.8%)")
print("💡 3등석 남성의 생존율이 가장 낮음 (13.5%)")
print("💡 모든 등급에서 여성의 생존율이 남성보다 높음")
```

#### 🎨 8단계: 피벗 테이블 히트맵 시각화

```python
# 피벗 테이블을 히트맵으로 시각화
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 첫 번째 히트맵: 평균 나이
sns.heatmap(pivot_age, 
           annot=True,           # 숫자 표시
           fmt='.1f',           # 소수점 1자리 표시
           cmap='YlOrRd',       # 색상 팔레트 (노랑-주황-빨강)
           cbar_kws={'label': '평균 나이 (세)'},  # 컬러바 라벨
           ax=axes[0,0])
axes[0,0].set_title('객실 등급별, 성별별 평균 나이', fontsize=14, fontweight='bold')

# 두 번째 히트맵: 생존율
sns.heatmap(pivot_survival, 
           annot=True, 
           fmt='.2f',           # 소수점 2자리 표시
           cmap='RdYlGn',       # 색상 팔레트 (빨강-노랑-초록)
           cbar_kws={'label': '생존율'},
           ax=axes[0,1])
axes[0,1].set_title('객실 등급별, 성별별 생존율', fontsize=14, fontweight='bold')

# 세 번째 히트맵: 승객 수
pivot_count = pd.pivot_table(df, values='PassengerId', index='Pclass', 
                            columns='Sex', aggfunc='count')
sns.heatmap(pivot_count, 
           annot=True, 
           fmt='d',             # 정수로 표시
           cmap='Blues',        # 파란색 계열
           cbar_kws={'label': '승객 수 (명)'},
           ax=axes[1,0])
axes[1,0].set_title('객실 등급별, 성별별 승객 수', fontsize=14, fontweight='bold')

# 네 번째 히트맵: 평균 운임
pivot_fare = pd.pivot_table(df, values='Fare', index='Pclass', 
                           columns='Sex', aggfunc='mean')
sns.heatmap(pivot_fare, 
           annot=True, 
           fmt='.1f', 
           cmap='viridis',      # 보라-파랑-초록-노랑
           cbar_kws={'label': '평균 운임 (파운드)'},
           ax=axes[1,1])
axes[1,1].set_title('객실 등급별, 성별별 평균 운임', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

print("\n🎨 히트맵 색상 해석 가이드:")
print("🔹 진한 색상: 높은 값")
print("🔹 연한 색상: 낮은 값") 
print("🔹 색상 차이: 그룹 간 차이의 크기")
print("🔹 색상 패턴: 데이터의 전반적인 경향")
```

---

## 직접 해보기 / 연습 문제

### 연습문제 1: 기본 통계량 이해하기 (난이도: ⭐)

**목표**: 운임(Fare) 데이터의 통계적 특성을 파악해보세요.

**단계별 가이드**:
1. 먼저 운임 데이터의 결측치를 확인하세요
2. 기본 통계량을 계산하세요
3. 평균과 중앙값을 비교해보세요

```python
# 여기에 코드를 작성해보세요
# 힌트: df['Fare'].describe() 를 사용해보세요

# 1단계: 결측치 확인
print("💰 운임 데이터 분석:")
fare_data = df['Fare'].dropna()
print(f"전체 데이터: {len(df)}개")
print(f"운임 정보 있음: {len(fare_data)}개") 
print(f"결측치: {len(df) - len(fare_data)}개")

# 2단계: 기본 통계량 계산
print(f"\n📊 운임 통계량:")
print(f"평균: {fare_data.mean():.2f} 파운드")
print(f"중앙값: {fare_data.median():.2f} 파운드")
print(f"표준편차: {fare_data.std():.2f} 파운드")
print(f"최소값: {fare_data.min():.2f} 파운드")
print(f"최대값: {fare_data.max():.2f} 파운드")

# 3단계: 시각화
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(fare_data, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
plt.axvline(fare_data.mean(), color='red', linestyle='--', label=f'평균: {fare_data.mean():.1f}')
plt.axvline(fare_data.median(), color='blue', linestyle='--', label=f'중앙값: {fare_data.median():.1f}')
plt.title('운임 분포 히스토그램')
plt.xlabel('운임 (파운드)')
plt.ylabel('빈도')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.boxplot(fare_data)
plt.title('운임 박스플롯')
plt.ylabel('운임 (파운드)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**💡 해석 힌트**: 
- 평균이 중앙값보다 훨씬 클 때는 무엇을 의미할까요?
- 박스플롯에서 점들(outlier)이 많이 보이는 이유는 무엇일까요?

**✅ 예상 결과**:
- 평균 운임이 중앙값보다 높음 → 우측 치우침 분포
- 고가 운임의 이상치들이 평균을 끌어올림

---

### 연습문제 2: 그룹별 분석 실습 (난이도: ⭐⭐)

**목표**: 나이대별 생존율을 분석하고 인사이트를 도출해보세요.

**단계별 가이드**:
1. 나이를 4개 구간으로 나누세요 (0-18, 18-35, 35-50, 50+)
2. 각 나이대별 생존율을 계산하세요
3. 결과를 시각화하고 해석하세요

```python
# 여기에 코드를 작성해보세요

# 1단계: 나이대 구간 나누기
df_analysis = df.copy()  # 원본 데이터 보존

# pd.cut() 함수로 연속형 변수를 범주형으로 변환
df_analysis['Age_Group'] = pd.cut(df_analysis['Age'], 
                                 bins=[0, 18, 35, 50, 100],  # 구간 경계값
                                 labels=['어린이\n(0-18세)', '청년\n(18-35세)', 
                                        '중년\n(35-50세)', '장년\n(50세+)'])

# 2단계: 나이대별 통계 계산
age_stats = df_analysis.groupby('Age_Group').agg({
    'Survived': ['count', 'sum', 'mean'],  # 전체 수, 생존자 수, 생존율
    'Age': 'mean'  # 평균 나이
}).round(3)

print("👥 나이대별 생존 통계:")
print(age_stats)

# 3단계: 생존율만 따로 추출하여 시각화
survival_by_age = df_analysis.groupby('Age_Group')['Survived'].mean()

plt.figure(figsize=(10, 6))
bars = plt.bar(range(len(survival_by_age)), survival_by_age.values, 
               color=['lightblue', 'lightgreen', 'orange', 'lightcoral'], alpha=0.8)
plt.title('나이대별 생존율', fontsize=16, fontweight='bold')
plt.xlabel('나이대')
plt.ylabel('생존율')
plt.xticks(range(len(survival_by_age)), survival_by_age.index, rotation=0)

# 막대 위에 백분율 표시
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.1%}', ha='center', va='bottom', fontweight='bold')

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n🔍 발견된 인사이트:")
print(f"• 어린이 생존율: {survival_by_age.iloc[0]:.1%}")
print(f"• 청년층 생존율: {survival_by_age.iloc[1]:.1%}")  
print(f"• 중년층 생존율: {survival_by_age.iloc[2]:.1%}")
print(f"• 장년층 생존율: {survival_by_age.iloc[3]:.1%}")
```

**💡 분석 질문**:
1. 어떤 나이대의 생존율이 가장 높나요?
2. 나이와 생존율 사이에 어떤 관계가 있나요?
3. 이러한 패턴이 나타나는 이유는 무엇일까요?

---

### 연습문제 3: 종합 분석 도전 (난이도: ⭐⭐⭐)

**목표**: 가족 규모가 생존에 미치는 영향을 다각도로 분석해보세요.

**단계별 가이드**:
1. 가족 규모 변수를 만드세요 (SibSp + Parch + 1)
2. 가족 유형을 분류하세요 (혼자, 소가족, 대가족)
3. 가족 유형별 생존율, 평균 나이, 평균 운임을 분석하세요
4. 피벗 테이블로 가족 유형과 객실 등급의 교차 분석을 수행하세요

```python
# 도전 과제: 여기에 코드를 작성해보세요

# 1단계: 가족 규모 변수 생성
df_family = df.copy()
df_family['Family_Size'] = df_family['SibSp'] + df_family['Parch'] + 1

# 2단계: 가족 유형 분류
def classify_family_type(size):
    if size == 1:
        return '혼자'
    elif size <= 4:
        return '소가족\n(2-4명)'
    else:
        return '대가족\n(5명+)'

df_family['Family_Type'] = df_family['Family_Size'].apply(classify_family_type)

# 3단계: 가족 유형별 종합 통계
family_comprehensive_stats = df_family.groupby('Family_Type').agg({
    'Survived': ['count', 'mean'],     # 전체 수, 생존율
    'Age': 'mean',                     # 평균 나이
    'Fare': 'mean',                    # 평균 운임
    'Family_Size': 'mean'              # 평균 가족 규모
}).round(2)

print("👨‍👩‍👧‍👦 가족 유형별 종합 통계:")
print(family_comprehensive_stats)

# 4단계: 가족 유형별 생존율 시각화
plt.figure(figsize=(15, 10))

# 첫 번째 그래프: 생존율
plt.subplot(2, 2, 1)
family_survival = df_family.groupby('Family_Type')['Survived'].mean()
bars = plt.bar(family_survival.index, family_survival.values,
               color=['skyblue', 'lightgreen', 'lightcoral'], alpha=0.8)
plt.title('가족 유형별 생존율', fontsize=14, fontweight='bold')
plt.ylabel('생존율')
plt.xticks(rotation=0)

for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.1%}', ha='center', va='bottom')

# 두 번째 그래프: 평균 나이
plt.subplot(2, 2, 2)
family_age = df_family.groupby('Family_Type')['Age'].mean()
bars = plt.bar(family_age.index, family_age.values,
               color=['orange', 'purple', 'brown'], alpha=0.8)
plt.title('가족 유형별 평균 나이', fontsize=14, fontweight='bold')
plt.ylabel('평균 나이 (세)')
plt.xticks(rotation=0)

for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.1f}세', ha='center', va='bottom')

# 세 번째 그래프: 평균 운임
plt.subplot(2, 2, 3)
family_fare = df_family.groupby('Family_Type')['Fare'].mean()
bars = plt.bar(family_fare.index, family_fare.values,
               color=['gold', 'silver', 'bronze'], alpha=0.8)
plt.title('가족 유형별 평균 운임', fontsize=14, fontweight='bold')
plt.ylabel('평균 운임 (파운드)')
plt.xticks(rotation=0)

for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 2,
             f'{height:.1f}', ha='center', va='bottom')

# 네 번째 그래프: 가족 유형별 객실 등급 분포
plt.subplot(2, 2, 4)
family_class_crosstab = pd.crosstab(df_family['Family_Type'], df_family['Pclass'], 
                                   normalize='index') * 100
family_class_crosstab.plot(kind='bar', ax=plt.gca(), 
                          color=['gold', 'silver', 'brown'], alpha=0.8)
plt.title('가족 유형별 객실 등급 분포', fontsize=14, fontweight='bold')
plt.ylabel('비율 (%)')
plt.xlabel('가족 유형')
plt.legend(title='객실 등급', labels=['1등석', '2등석', '3등석'])
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()

# 5단계: 피벗 테이블로 교차 분석
print("\n📊 가족 유형 × 객실 등급 생존율 피벗 테이블:")
family_class_survival = pd.pivot_table(df_family, 
                                      values='Survived',
                                      index='Family_Type',
                                      columns='Pclass',
                                      aggfunc='mean')
print(family_class_survival.round(3))

# 히트맵으로 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(family_class_survival, 
           annot=True, fmt='.2f', cmap='RdYlGn',
           cbar_kws={'label': '생존율'})
plt.title('가족 유형별, 객실 등급별 생존율', fontsize=14, fontweight='bold')
plt.show()

print("\n🔍 종합 인사이트:")
print("• 소가족의 생존율이 가장 높음 (서로 도움)")
print("• 혼자 탑승한 승객의 생존율이 중간")
print("• 대가족의 생존율이 가장 낮음 (피난의 어려움)")
print("• 모든 가족 유형에서 1등석의 생존율이 높음")
```

---

## 요약 / 핵심 정리

### 🎯 주요 학습 내용

**1. 중심경향치 (데이터의 중심 위치)**
- **평균 (Mean)**: 모든 값의 산술 평균, 이상치에 민감
  - 사용 시기: 정규분포, 이상치가 적을 때
- **중앙값 (Median)**: 순서상 중간값, 이상치에 강건
  - 사용 시기: 왜곡된 분포, 이상치가 많을 때
- **최빈값 (Mode)**: 가장 빈번한 값, 범주형 데이터에도 사용 가능
  - 사용 시기: 범주형 데이터, 다봉분포

**2. 산포도 (데이터의 퍼짐 정도)**
- **범위 (Range)**: 최대값 - 최소값, 간단하지만 이상치에 민감
- **표준편차 (Standard Deviation)**: 평균으로부터의 평균적 거리
- **분산 (Variance)**: 표준편차의 제곱
- **사분위수 범위 (IQR)**: Q3 - Q1, 이상치에 강건

**3. 그룹별 분석의 중요성**
- **groupby()**: 조건별 그룹 생성 및 통계량 계산
- **agg()**: 여러 통계량을 한 번에 계산
- **시각화**: 박스플롯, 바이올린플롯으로 그룹 간 차이 확인

**4. 피벗 테이블과 교차분석**
- **pivot_table()**: 다차원 데이터 요약
- **히트맵**: 피벗 테이블의 시각적 표현
- **교차분석**: 두 범주형 변수 간의 관계 파악

### 💡 실무 적용 팁

**1. 적절한 통계량 선택 기준**
```
데이터 분포 → 추천 통계량
정규분포     → 평균, 표준편차
우측 치우침  → 중앙값, IQR
좌측 치우침  → 중앙값, IQR
범주형 데이터 → 최빈값, 빈도
```

**2. 그룹 비교 시 주의사항**
- ✅ 표본 크기 차이 확인 (큰 차이가 있으면 해석에 주의)
- ✅ 분포의 형태 고려 (정규분포 vs 치우친 분포)
- ✅ 변동성 차이 분석 (표준편차가 크게 다른 그룹들)

**3. 이상치 처리 전략**
- 📊 **탐지**: 박스플롯, IQR 방법 활용
- 🔍 **확인**: 데이터 입력 오류인지 실제 극값인지 검증
- 🛠️ **처리**: 제거, 변환, 또는 별도 분석

**4. 시각화 선택 가이드**
```
목적                → 추천 그래프
분포 확인           → 히스토그램, 박스플롯
그룹 비교           → 박스플롯, 바이올린플롯
관계 파악           → 산점도, 히트맵
시계열 변화         → 선 그래프
범주별 빈도         → 막대 그래프
```

### 🚀 실무 활용 시나리오

**📈 마케팅 분야**
- 고객 연령대별 구매 패턴 분석
- 지역별 매출 비교 분석
- 프로모션 효과의 그룹별 차이 분석

**🏥 의료 분야**
- 치료법별 효과 비교
- 환자군별 증상 정도 분석
- 약물 부작용의 나이대별 차이

**🎓 교육 분야**
- 학급별 성적 분포 분석
- 교수법별 학습 효과 비교
- 학생 배경별 성취도 차이

### 🔍 다음 Part 예고

다음 Part에서는 **AI 도구를 활용한 EDA와 전통적 방식 비교**에 대해 배우겠습니다. 

**🤖 학습 내용 미리보기**:
- AI 기반 자동 EDA 도구 소개 및 활용법
- AI 생성 인사이트의 평가 방법과 한계점
- 전통적 EDA vs AI 기반 EDA의 장단점 비교
- 인간과 AI의 협업을 통한 효과적인 데이터 분석 전략

**💡 준비 사항**:
- 이번 Part에서 배운 기술 통계 개념 복습
- Python 기본 문법과 Pandas 활용법 숙지
- 통계적 사고와 비판적 분석 능력 향상

---

**📚 참고 자료**
- Pandas 공식 문서: https://pandas.pydata.org/docs/
- Seaborn 통계 시각화: https://seaborn.pydata.org/tutorial.html
- Matplotlib 시각화 가이드: https://matplotlib.org/stable/tutorials/index.html
- 통계학 기초 개념: https://www.khanacademy.org/math/statistics-probability
```
