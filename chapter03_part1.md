# 3장 Part 1: 확률 분포와 통계적 개념

## 3.1 확률 분포와 통계적 개념

### 학습 목표
이 섹션을 완료하면 다음을 할 수 있습니다:
- 주요 확률 분포의 특성과 활용 사례를 설명할 수 있다
- 표본과 모집단의 차이점을 이해하고 표본 추출의 중요성을 설명할 수 있다
- 통계적 추정과 신뢰구간의 개념을 이해하고 실제 데이터에 적용할 수 있다
- Python을 활용하여 확률 분포를 시각화하고 통계량을 계산할 수 있다

### 이번 섹션 미리보기
데이터 분석의 핵심은 불확실성을 다루는 것입니다. 우리가 수집한 데이터는 전체 모집단의 일부분(표본)이며, 이 표본을 통해 전체에 대한 결론을 내려야 합니다. 이때 확률 분포와 통계적 개념들이 중요한 역할을 합니다.

AI 시대에도 이러한 기본 개념의 이해는 매우 중요합니다. AI 도구가 생성한 분석 결과를 올바르게 해석하고, 그 한계를 파악하기 위해서는 확률과 통계의 기초가 탄탄해야 합니다.

### 3.1.1 주요 확률 분포 이해하기

#### 확률 분포란?
확률 분포는 어떤 사건이 일어날 가능성을 수학적으로 표현한 것입니다. 데이터 분석에서는 데이터가 어떤 패턴을 따르는지 이해하기 위해 확률 분포를 사용합니다.

#### 정규분포 (Normal Distribution)
정규분포는 자연계에서 자주 관찰되는 연속 확률 분포로, 데이터가 평균을 중심으로 좌우 대칭인 종 모양(bell-shaped)을 가집니다. 키, 몸무게, 시험 점수 등 많은 데이터가 정규분포를 따르는 경향이 있습니다. 

Python을 사용해 정규분포의 특성을 시각화하며, 표준 정규분포, 평균이 다른 정규분포, 표준편차가 다른 정규분포를 각각 보여줍니다.

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd

# 한글 폰트 설정
# - 'Malgun Gothic'은 Windows에서 한글 표시를 위한 폰트
# - 'axes.unicode_minus = False'는 음수 기호(-) 깨짐 방지
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 색상 팔레트 설정 (seaborn의 'husl' 팔레트로 색상 다양화)
colors = sns.color_palette("husl", 3)

# 1x3 서브플롯 생성 (가로로 3개의 그래프 배치, 크기 15x5)
# - plt.subplots(1, 3, figsize=(15, 5)): 1행 3열 서브플롯 생성, 전체 크기 지정
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. 표준 정규분포 (Standard Normal Distribution)
# - np.linspace(start, stop, num): start부터 stop까지 num개의 균등한 값 생성
# - stats.norm.pdf(x, loc, scale): 정규분포의 확률 밀도 함수(PDF) 계산
#   - loc: 분포의 평균 (mu)
#   - scale: 분포의 표준편차 (sigma)
x = np.linspace(-4, 4, 100)
y = stats.norm.pdf(x, 0, 1)
axes[0].plot(x, y, color=colors[0], linewidth=2, label='평균=0, 표준편차=1')
# - fill_between(x, y, alpha): 곡선 아래 영역을 alpha 투명도로 채움
axes[0].fill_between(x, y, color=colors[0], alpha=0.3)
axes[0].set_title('표준 정규분포')
axes[0].set_xlabel('값')
axes[0].set_ylabel('확률 밀도')
axes[0].grid(True, alpha=0.3)
# - legend(loc='upper left'): 범례를 왼쪽 상단에 배치해 곡선과 겹침 방지
axes[0].legend(loc='upper left')
# 주석 추가: 평균값(0)에 포인트 표시
axes[0].annotate('평균=0', xy=(0, stats.norm.pdf(0, 0, 1)), xytext=(1, 0.3),
                 arrowprops=dict(facecolor='black', shrink=0.05))

# 2. 다양한 평균을 가진 정규분포
# - 평균이 0, 2, 4로 변할 때 분포의 위치 이동 시각화
x = np.linspace(-2, 8, 100)
for i, mean in enumerate([0, 2, 4]):
    y = stats.norm.pdf(x, mean, 1)
    axes[1].plot(x, y, color=colors[i], linewidth=2, label=f'평균={mean}')
    # 주석 추가: 각 평균값에 포인트 표시
    axes[1].annotate(f'평균={mean}', xy=(mean, stats.norm.pdf(mean, mean, 1)),
                     xytext=(mean + 1, 0.3), arrowprops=dict(facecolor='black', shrink=0.05))
axes[1].set_title('평균이 다른 정규분포')
axes[1].set_xlabel('값')
axes[1].set_ylabel('확률 밀도')
axes[1].grid(True, alpha=0.3)
axes[1].legend(loc='upper left')

# 3. 다양한 표준편차를 가진 정규분포
# - 표준편차가 0.5, 1, 2로 변할 때 분포의 퍼짐 정도 시각화
x = np.linspace(-6, 6, 100)
for i, std in enumerate([0.5, 1, 2]):
    y = stats.norm.pdf(x, 0, std)
    axes[2].plot(x, y, color=colors[i], linewidth=2, label=f'표준편차={std}')
    # 주석 추가: 각 분포의 최대 높이(평균=0)에 포인트 표시
    axes[2].annotate(f'σ={std}', xy=(0, stats.norm.pdf(0, 0, std)),
                     xytext=(1, stats.norm.pdf(0, 0, std) * 0.8),
                     arrowprops=dict(facecolor='black', shrink=0.05))
axes[2].set_title('표준편차가 다른 정규분포')
axes[2].set_xlabel('값')
axes[2].set_ylabel('확률 밀도')
axes[2].grid(True, alpha=0.3)
axes[2].legend(loc='upper left')

# - tight_layout(): 서브플롯 간 간격을 자동 조정해 겹침 방지
plt.tight_layout()
# - plt.show(): 그래프를 화면에 출력
plt.show()
```

**정규분포의 특징:**
- 평균을 중심으로 좌우 대칭
- 평균 ± 1 표준편차 안에 약 68%의 데이터
- 평균 ± 2 표준편차 안에 약 95%의 데이터
- 평균 ± 3 표준편차 안에 약 99.7%의 데이터

#### 이항분포 (Binomial Distribution)
성공/실패, 참/거짓과 같은 이진 결과를 다룰 때 사용됩니다.

```python
# 이항분포 시각화
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 1. 동전 던지기 예시 (n=10, p=0.5)
n, p = 10, 0.5
x = np.arange(0, n+1)
y = stats.binom.pmf(x, n, p)

axes[0].bar(x, y, alpha=0.7, color='skyblue', edgecolor='black')
axes[0].set_title(f'이항분포 (n={n}, p={p})\n동전 10번 던지기에서 앞면이 나올 횟수')
axes[0].set_xlabel('성공 횟수')
axes[0].set_ylabel('확률')
axes[0].grid(True, alpha=0.3)

# 평균과 표준편차 표시
mean = n * p
std = np.sqrt(n * p * (1-p))
axes[0].axvline(mean, color='red', linestyle='--', linewidth=2, 
                label=f'평균={mean:.1f}')
axes[0].legend()

# 2. 다양한 확률을 가진 이항분포
n = 20
for p in [0.2, 0.5, 0.8]:
    x = np.arange(0, n+1)
    y = stats.binom.pmf(x, n, p)
    axes[1].plot(x, y, 'o-', linewidth=2, label=f'p={p}')

axes[1].set_title(f'다양한 성공확률의 이항분포 (n={n})')
axes[1].set_xlabel('성공 횟수')
axes[1].set_ylabel('확률')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.show()
```

#### 포아송분포 (Poisson Distribution)
단위 시간당 발생하는 사건의 횟수를 모델링할 때 사용됩니다. 예: 1시간당 방문자 수, 하루당 교통사고 건수

```python
# 포아송분포 시각화
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# 다양한 λ 값에 대한 포아송분포
x = np.arange(0, 15)
lambdas = [1, 3, 5, 8]
colors = ['blue', 'green', 'red', 'orange']

for i, lam in enumerate(lambdas):
    y = stats.poisson.pmf(x, lam)
    ax.plot(x, y, 'o-', linewidth=2, color=colors[i], 
            label=f'λ={lam} (평균={lam})')

ax.set_title('포아송분포: 단위 시간당 사건 발생 횟수')
ax.set_xlabel('사건 발생 횟수')
ax.set_ylabel('확률')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()

# 실제 예시: 웹사이트 방문자 수 시뮬레이션
np.random.seed(42)
hourly_visitors = np.random.poisson(5, 24)  # 시간당 평균 5명 방문

plt.figure(figsize=(12, 6))
plt.bar(range(24), hourly_visitors, alpha=0.7, color='lightcoral')
plt.axhline(y=5, color='red', linestyle='--', linewidth=2, label='평균 방문자 수 (5명)')
plt.title('웹사이트 시간대별 방문자 수 (포아송분포 시뮬레이션)')
plt.xlabel('시간 (0-23시)')
plt.ylabel('방문자 수')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

print(f"총 방문자 수: {hourly_visitors.sum()}명")
print(f"평균 시간당 방문자 수: {hourly_visitors.mean():.2f}명")
print(f"표준편차: {hourly_visitors.std():.2f}")
```

### 3.1.2 표본과 모집단의 관계

#### 모집단 vs 표본
- **모집단(Population)**: 연구하고자 하는 전체 대상
- **표본(Sample)**: 모집단에서 실제로 관찰한 일부분

```python
# 모집단과 표본의 개념 시각화
np.random.seed(42)

# 가상의 모집단 생성 (전국 고등학생의 키)
population_size = 100000
population_heights = np.random.normal(170, 8, population_size)

# 여러 표본 추출
sample_sizes = [30, 100, 500, 1000]
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.ravel()

for i, sample_size in enumerate(sample_sizes):
    # 표본 추출
    sample = np.random.choice(population_heights, sample_size, replace=False)
    
    # 히스토그램 그리기
    axes[i].hist(sample, bins=20, alpha=0.7, density=True, 
                 color='lightblue', edgecolor='black', label=f'표본 (n={sample_size})')
    
    # 모집단 분포 곡선 오버레이
    x = np.linspace(140, 200, 100)
    y = stats.norm.pdf(x, 170, 8)
    axes[i].plot(x, y, 'r-', linewidth=2, label='모집단 분포')
    
    axes[i].axvline(sample.mean(), color='blue', linestyle='--', 
                    label=f'표본 평균: {sample.mean():.1f}cm')
    axes[i].axvline(170, color='red', linestyle='--', 
                    label='모집단 평균: 170cm')
    
    axes[i].set_title(f'표본 크기: {sample_size}')
    axes[i].set_xlabel('키 (cm)')
    axes[i].set_ylabel('밀도')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.suptitle('표본 크기에 따른 모집단 추정의 정확도', fontsize=16)
plt.tight_layout()
plt.show()
```

#### 표본 추출 방법
표본을 추출하는 방법에 따라 결과가 달라질 수 있습니다.

```python
# 다양한 표본 추출 방법 비교
np.random.seed(42)

# 가상의 학급 성적 데이터 (편향된 분포)
students = pd.DataFrame({
    'student_id': range(1, 101),
    'math_score': np.concatenate([
        np.random.normal(85, 5, 30),  # 우수 학생들 (30명)
        np.random.normal(70, 8, 50),  # 보통 학생들 (50명)  
        np.random.normal(55, 6, 20)   # 부진 학생들 (20명)
    ]),
    'group': ['우수']*30 + ['보통']*50 + ['부진']*20
})

print("전체 학급 통계:")
print(f"평균 점수: {students['math_score'].mean():.2f}")
print(f"표준편차: {students['math_score'].std():.2f}")
print(f"그룹별 인원: {students['group'].value_counts().to_dict()}")

# 1. 단순 무작위 추출
simple_sample = students.sample(n=20, random_state=42)

# 2. 층화 추출 (그룹별 비율 유지)
stratified_sample = students.groupby('group', group_keys=False).apply(
    lambda x: x.sample(n=max(1, int(len(x) * 0.2)), random_state=42)
)

# 3. 편향된 추출 (우수 학생만)
biased_sample = students[students['group'] == '우수'].sample(n=20, random_state=42)

# 결과 비교
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 전체 모집단
axes[0,0].hist(students['math_score'], bins=15, alpha=0.7, color='gray', edgecolor='black')
axes[0,0].axvline(students['math_score'].mean(), color='red', linestyle='--', linewidth=2,
                  label=f'평균: {students["math_score"].mean():.1f}')
axes[0,0].set_title('전체 모집단 (100명)')
axes[0,0].legend()

# 단순 무작위 추출
axes[0,1].hist(simple_sample['math_score'], bins=10, alpha=0.7, color='blue', edgecolor='black')
axes[0,1].axvline(simple_sample['math_score'].mean(), color='red', linestyle='--', linewidth=2,
                  label=f'평균: {simple_sample["math_score"].mean():.1f}')
axes[0,1].set_title('단순 무작위 추출 (20명)')
axes[0,1].legend()

# 층화 추출
axes[1,0].hist(stratified_sample['math_score'], bins=10, alpha=0.7, color='green', edgecolor='black')
axes[1,0].axvline(stratified_sample['math_score'].mean(), color='red', linestyle='--', linewidth=2,
                  label=f'평균: {stratified_sample["math_score"].mean():.1f}')
axes[1,0].set_title('층화 추출 (그룹 비율 유지)')
axes[1,0].legend()

# 편향된 추출
axes[1,1].hist(biased_sample['math_score'], bins=10, alpha=0.7, color='orange', edgecolor='black')
axes[1,1].axvline(biased_sample['math_score'].mean(), color='red', linestyle='--', linewidth=2,
                  label=f'평균: {biased_sample["math_score"].mean():.1f}')
axes[1,1].set_title('편향된 추출 (우수 학생만)')
axes[1,1].legend()

for ax in axes.flat:
    ax.set_xlabel('수학 점수')
    ax.set_ylabel('빈도')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 표본별 통계 비교
print("\n표본별 통계 비교:")
print(f"전체 모집단 평균: {students['math_score'].mean():.2f}")
print(f"단순 무작위 추출 평균: {simple_sample['math_score'].mean():.2f}")
print(f"층화 추출 평균: {stratified_sample['math_score'].mean():.2f}")
print(f"편향된 추출 평균: {biased_sample['math_score'].mean():.2f}")

### 3.1.3 통계적 추정과 신뢰구간

#### 점 추정 vs 구간 추정
- **점 추정**: 모집단 모수를 하나의 값으로 추정 (예: 표본 평균)
- **구간 추정**: 모집단 모수가 있을 것으로 예상되는 구간을 제시 (신뢰구간)

#### 신뢰구간의 개념
신뢰구간은 "이 구간 안에 참값이 있을 확률이 95%이다"라는 의미가 아닙니다. 올바른 해석은 "같은 방법으로 100번 실험하면, 95번은 이런 구간이 참값을 포함할 것이다"입니다.

```python
# 신뢰구간 개념 시각화
from scipy import stats

def calculate_confidence_interval(data, confidence=0.95):
    """신뢰구간 계산"""
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)  # 표준오차
    
    # t-분포 사용 (표본 크기가 작을 때)
    t_val = stats.t.ppf((1 + confidence) / 2, n-1)
    margin_error = t_val * std_err
    
    ci_lower = mean - margin_error
    ci_upper = mean + margin_error
    
    return mean, ci_lower, ci_upper, margin_error

# 시뮬레이션: 100개의 표본에서 신뢰구간 계산
np.random.seed(42)
true_mean = 170  # 실제 모집단 평균 (키)
true_std = 8     # 실제 모집단 표준편차

n_samples = 100
sample_size = 30
confidence_level = 0.95

# 100개의 표본 추출하고 각각의 신뢰구간 계산
results = []
contains_true_mean = []

for i in range(n_samples):
    sample = np.random.normal(true_mean, true_std, sample_size)
    mean, ci_lower, ci_upper, margin_error = calculate_confidence_interval(sample, confidence_level)
    
    results.append({
        'sample_id': i+1,
        'sample_mean': mean,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'margin_error': margin_error
    })
    
    # 신뢰구간이 참값을 포함하는지 확인
    contains_true_mean.append(ci_lower <= true_mean <= ci_upper)

results_df = pd.DataFrame(results)
coverage_rate = np.mean(contains_true_mean)

print(f"100개 표본 중 신뢰구간이 참값을 포함한 비율: {coverage_rate:.1%}")
print(f"이론적 신뢰도: {confidence_level:.1%}")

# 처음 20개 신뢰구간 시각화
plt.figure(figsize=(12, 8))

for i in range(20):
    color = 'blue' if contains_true_mean[i] else 'red'
    plt.errorbar(results_df.iloc[i]['sample_mean'], i+1, 
                xerr=results_df.iloc[i]['margin_error'],
                fmt='o', color=color, alpha=0.7, capsize=3)

plt.axvline(true_mean, color='black', linestyle='--', linewidth=2, 
            label=f'참값 (모집단 평균): {true_mean}cm')
plt.xlabel('키 (cm)')
plt.ylabel('표본 번호')
plt.title(f'95% 신뢰구간 시각화 (처음 20개 표본)\n파란색: 참값 포함, 빨간색: 참값 미포함')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 실제 예시: Titanic 데이터로 신뢰구간 계산
print("\n=== Titanic 데이터 신뢰구간 예시 ===")

# Titanic 데이터 로드 (시뮬레이션)
np.random.seed(42)
titanic_ages = np.random.normal(29.7, 14.5, 714)  # 실제 Titanic 승객 나이 분포 근사

# 95% 신뢰구간 계산
mean_age, ci_lower, ci_upper, margin_error = calculate_confidence_interval(titanic_ages)

print(f"표본 크기: {len(titanic_ages)}명")
print(f"평균 나이: {mean_age:.2f}세")
print(f"95% 신뢰구간: [{ci_lower:.2f}, {ci_upper:.2f}]")
print(f"오차한계: ±{margin_error:.2f}세")
print(f"\n해석: 타이타닉 승객의 평균 나이는 95% 신뢰도로 {ci_lower:.1f}세에서 {ci_upper:.1f}세 사이에 있습니다.")

# 신뢰구간 시각화
plt.figure(figsize=(10, 6))
plt.hist(titanic_ages, bins=30, alpha=0.7, color='lightblue', edgecolor='black', density=True)
plt.axvline(mean_age, color='red', linestyle='-', linewidth=2, label=f'표본 평균: {mean_age:.1f}세')
plt.axvline(ci_lower, color='orange', linestyle='--', linewidth=2, label=f'95% 신뢰구간')
plt.axvline(ci_upper, color='orange', linestyle='--', linewidth=2)
plt.fill_betweenx([0, 0.03], ci_lower, ci_upper, alpha=0.3, color='orange')
plt.xlabel('나이 (세)')
plt.ylabel('밀도')
plt.title('타이타닉 승객 나이 분포와 95% 신뢰구간')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

#### 신뢰구간의 폭에 영향을 미치는 요인

# 표본 크기와 신뢰구간의 관계
sample_sizes = [10, 30, 50, 100, 200, 500]
confidence_intervals = []

np.random.seed(42)
true_mean = 170
true_std = 8

for n in sample_sizes:
    sample = np.random.normal(true_mean, true_std, n)
    mean, ci_lower, ci_upper, margin_error = calculate_confidence_interval(sample)
    confidence_intervals.append({
        'sample_size': n,
        'margin_error': margin_error,
        'ci_width': ci_upper - ci_lower
    })

ci_df = pd.DataFrame(confidence_intervals)

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 1. 표본 크기 vs 오차한계
axes[0].plot(ci_df['sample_size'], ci_df['margin_error'], 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('표본 크기')
axes[0].set_ylabel('오차한계')
axes[0].set_title('표본 크기와 오차한계의 관계')
axes[0].grid(True, alpha=0.3)

# 이론적 곡선 추가
x_theory = np.linspace(10, 500, 100)
y_theory = true_std / np.sqrt(x_theory) * 1.96  # 근사적 계산
axes[0].plot(x_theory, y_theory, 'r--', linewidth=2, label='이론적 관계 (1/√n)')
axes[0].legend()

# 2. 신뢰구간 폭 비교
axes[1].bar(range(len(ci_df)), ci_df['ci_width'], alpha=0.7, color='lightgreen')
axes[1].set_xlabel('표본 크기')
axes[1].set_ylabel('신뢰구간 폭')
axes[1].set_title('표본 크기별 95% 신뢰구간 폭')
axes[1].set_xticks(range(len(ci_df)))
axes[1].set_xticklabels(ci_df['sample_size'])
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("표본 크기별 신뢰구간 결과:")
for _, row in ci_df.iterrows():
    print(f"n={row['sample_size']:3d}: 오차한계=±{row['margin_error']:.2f}, 구간폭={row['ci_width']:.2f}")
```

### 실습: 중심극한정리 확인하기

중심극한정리는 통계학의 핵심 개념 중 하나입니다. 모집단의 분포가 어떤 모양이든 상관없이, 표본 크기가 충분히 크면 표본 평균의 분포는 정규분포에 가까워집니다.

```python
# 중심극한정리 시뮬레이션
def demonstrate_central_limit_theorem():
    np.random.seed(42)
    
    # 1. 균등분포 모집단
    def uniform_population(size):
        return np.random.uniform(0, 10, size)
    
    # 2. 지수분포 모집단 (매우 비대칭)
    def exponential_population(size):
        return np.random.exponential(2, size)
    
    # 3. 이산 분포 모집단 (주사위)
    def discrete_population(size):
        return np.random.choice([1, 2, 3, 4, 5, 6], size)
    
    populations = [
        ('균등분포', uniform_population),
        ('지수분포', exponential_population),
        ('이산분포(주사위)', discrete_population)
    ]
    
    sample_sizes = [5, 30, 100]
    n_samples = 1000  # 각 표본 크기별로 1000개의 표본 추출
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    for pop_idx, (pop_name, pop_func) in enumerate(populations):
        # 모집단 분포 그리기
        population_data = pop_func(10000)
        axes[pop_idx, 0].hist(population_data, bins=50, alpha=0.7, color='gray', density=True)
        axes[pop_idx, 0].set_title(f'{pop_name} 모집단')
        axes[pop_idx, 0].set_ylabel('밀도')
        
        # 각 표본 크기별로 표본 평균의 분포 확인
        for size_idx, sample_size in enumerate(sample_sizes):
            sample_means = []
            
            for _ in range(n_samples):
                sample = pop_func(sample_size)
                sample_means.append(np.mean(sample))
            
            # 표본 평균의 분포 그리기
            axes[pop_idx, size_idx + 1].hist(sample_means, bins=30, alpha=0.7, 
                                           density=True, color='lightblue', edgecolor='black')
            
            # 정규분포 곡선 오버레이
            mean_of_means = np.mean(sample_means)
            std_of_means = np.std(sample_means)
            x = np.linspace(min(sample_means), max(sample_means), 100)
            y = stats.norm.pdf(x, mean_of_means, std_of_means)
            axes[pop_idx, size_idx + 1].plot(x, y, 'r-', linewidth=2, label='정규분포')
            
            axes[pop_idx, size_idx + 1].set_title(f'표본평균 분포\n(n={sample_size})')
            axes[pop_idx, size_idx + 1].legend()
            axes[pop_idx, size_idx + 1].grid(True, alpha=0.3)
    
    # x축 레이블 추가
    for j in range(4):
        axes[2, j].set_xlabel('값' if j == 0 else '표본 평균')
    
    plt.suptitle('중심극한정리: 다양한 모집단에서 표본 평균의 분포', fontsize=16)
    plt.tight_layout()
    plt.show()

demonstrate_central_limit_theorem()
```

### 요약 및 핵심 정리

**이번 섹션에서 배운 핵심 내용:**

1. **주요 확률 분포**
   - 정규분포: 자연계에서 가장 흔한 분포, 평균과 표준편차로 특성화
   - 이항분포: 성공/실패의 이진 결과를 모델링
   - 포아송분포: 단위 시간당 사건 발생 횟수를 모델링

2. **표본과 모집단**
   - 표본 추출 방법이 결과에 큰 영향을 미침
   - 편향된 표본은 잘못된 결론으로 이어질 수 있음
   - 표본 크기가 클수록 모집단을 더 정확히 추정

3. **통계적 추정과 신뢰구간**
   - 점 추정보다는 구간 추정이 더 유용한 정보 제공
   - 신뢰구간은 불확실성을 정량화하는 도구
   - 표본 크기가 클수록 신뢰구간이 좁아짐

4. **중심극한정리**
   - 모집단 분포와 상관없이 표본 평균은 정규분포에 근사
   - 이는 많은 통계적 추론의 이론적 기반이 됨

**다음 섹션 예고:**
다음 Part 2에서는 가설 검정의 개념을 학습합니다. 귀무가설과 대립가설을 설정하고, p-값을 해석하며, 주요 통계 검정 방법들을 실제 데이터에 적용해보겠습니다.

### 연습문제

1. **개념 확인**
   - 정규분포, 이항분포, 포아송분포의 차이점을 설명하고 각각의 실제 사례를 제시해보세요.

2. **실습 문제**
   - 학급 평균 시험 점수가 75점이고 표준편차가 10점일 때, 30명을 무작위로 선택했을 때 표본 평균이 72점 이상 78점 이하일 확률을 계산해보세요.

3. **응용 문제**
   - 온라인 쇼핑몰의 하루 방문자 수가 평균 100명인 포아송분포를 따른다고 할 때, 하루에 120명 이상 방문할 확률을 구해보세요.

4. **비판적 사고**
   - 여론조사에서 "오차한계 ±3%p, 95% 신뢰도"라는 표현의 정확한 의미를 설명하고, 일반인들이 자주 오해하는 부분을 지적해보세요.
```
