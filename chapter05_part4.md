# 5장 Part 4: 모델 평가와 검증 방법

## 학습 목표
이번 파트를 완료하면 다음을 할 수 있습니다:
- 훈련/검증/테스트 데이터 분할의 중요성을 이해하고 올바르게 적용할 수 있다
- K-Fold, Stratified K-Fold 등 교차 검증 기법을 구현하고 활용할 수 있다
- 학습 곡선을 통해 과적합과 과소적합을 진단하고 해결할 수 있다
- 편향-분산 트레이드오프를 이해하고 모델 복잡도를 조절할 수 있다
- 검증 곡선으로 최적의 하이퍼파라미터를 찾을 수 있다
- 체계적인 모델 선택 프레임워크를 구축할 수 있다

## 이번 파트 미리보기
좋은 모델이란 무엇일까요? 단순히 훈련 데이터에서 높은 정확도를 보이는 모델일까요? 아닙니다! **새로운 데이터에서도 잘 작동하는 모델**이 진짜 좋은 모델입니다. 이번 파트에서는 모델을 올바르게 평가하고 검증하는 다양한 방법을 학습합니다. 시험 문제를 미리 보고 공부하는 것과 실제 시험의 차이처럼, 머신러닝 모델도 제대로 평가해야 합니다!

---

## 5.4.1 왜 모델 평가가 중요한가?

### 시험 공부의 비유로 이해하기

```python
# 필요한 라이브러리 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (train_test_split, KFold, StratifiedKFold, 
                                   cross_val_score, learning_curve, validation_curve)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.datasets import load_iris, load_wine, fetch_california_housing
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 모델 평가의 중요성 시각화
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 왼쪽: 잘못된 평가 (문제집만 풀기)
study_methods = ['문제집 A', '문제집 A', '문제집 A', '실제 시험']
scores_wrong = [95, 98, 100, 65]
colors_wrong = ['green', 'green', 'green', 'red']

ax1.bar(study_methods, scores_wrong, color=colors_wrong, alpha=0.7, edgecolor='black')
ax1.set_ylim(0, 110)
ax1.set_ylabel('점수', fontsize=12)
ax1.set_title('잘못된 학습: 같은 문제만 반복', fontsize=14, fontweight='bold')
ax1.axhline(y=80, color='gray', linestyle='--', alpha=0.5, label='목표 점수')

# 점수 표시
for i, score in enumerate(scores_wrong):
    ax1.text(i, score + 2, str(score), ha='center', fontsize=12, fontweight='bold')

# 오른쪽: 올바른 평가 (다양한 문제 풀기)
study_methods_right = ['문제집 A', '문제집 B', '모의고사', '실제 시험']
scores_right = [85, 82, 80, 78]
colors_right = ['blue', 'blue', 'blue', 'green']

ax2.bar(study_methods_right, scores_right, color=colors_right, alpha=0.7, edgecolor='black')
ax2.set_ylim(0, 110)
ax2.set_ylabel('점수', fontsize=12)
ax2.set_title('올바른 학습: 다양한 문제 연습', fontsize=14, fontweight='bold')
ax2.axhline(y=80, color='gray', linestyle='--', alpha=0.5, label='목표 점수')

# 점수 표시
for i, score in enumerate(scores_right):
    ax2.text(i, score + 2, str(score), ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

print("💡 핵심 교훈:")
print("   • 같은 문제만 반복하면 실전에서 실패합니다")
print("   • 다양한 문제로 연습해야 실력이 늘어납니다")
print("   • 머신러닝도 마찬가지! 다양한 데이터로 검증해야 합니다")
```

### 데이터 분할의 원칙

```python
# 데이터 분할 비율 시각화
fig, ax = plt.subplots(figsize=(10, 4))

# 데이터 분할 비율
sections = [0.6, 0.2, 0.2]
labels = ['훈련 데이터 (60%)', '검증 데이터 (20%)', '테스트 데이터 (20%)']
colors = ['#3498db', '#e74c3c', '#2ecc71']
explode = (0.05, 0.05, 0.05)

# 막대 그래프로 표현
left = 0
for i, (section, label, color) in enumerate(zip(sections, labels, colors)):
    ax.barh(0, section, left=left, height=0.5, color=color, 
            edgecolor='black', linewidth=2)
    ax.text(left + section/2, 0, label, ha='center', va='center', 
            fontsize=12, fontweight='bold', color='white')
    left += section

ax.set_xlim(0, 1)
ax.set_ylim(-0.5, 0.5)
ax.axis('off')
ax.set_title('올바른 데이터 분할 비율', fontsize=16, fontweight='bold', pad=20)

# 설명 추가
descriptions = [
    "모델 학습용\n패턴을 찾는다",
    "하이퍼파라미터 튜닝\n모델 선택",
    "최종 평가용\n절대 건드리지 않는다!"
]

for i, (section, desc) in enumerate(zip(sections, descriptions)):
    ax.text(sum(sections[:i]) + section/2, -0.35, desc, 
           ha='center', va='top', fontsize=10, style='italic')

plt.tight_layout()
plt.show()

# 데이터 분할 예시
print("\n=== 실제 데이터 분할 예시 ===")
X = np.random.randn(1000, 5)
y = np.random.randint(0, 2, 1000)

# 1차 분할: 훈련+검증 vs 테스트
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2차 분할: 훈련 vs 검증
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

print(f"전체 데이터: {len(X)}개")
print(f"훈련 데이터: {len(X_train)}개 ({len(X_train)/len(X)*100:.0f}%)")
print(f"검증 데이터: {len(X_val)}개 ({len(X_val)/len(X)*100:.0f}%)")
print(f"테스트 데이터: {len(X_test)}개 ({len(X_test)/len(X)*100:.0f}%)")
```

## 5.4.2 교차 검증 (Cross-Validation)

### K-Fold 교차 검증의 원리

단 한 번의 검증으로는 운이 좋거나 나쁠 수 있습니다. **여러 번 검증**해서 안정적인 성능을 확인해야 합니다:

```python
# K-Fold 교차 검증 시각화
fig, axes = plt.subplots(5, 1, figsize=(10, 8))
n_samples = 100
n_folds = 5

# 데이터 인덱스
indices = np.arange(n_samples)
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
    ax = axes[fold]
    
    # 전체 데이터를 회색으로
    ax.barh(0, n_samples, height=0.5, color='lightgray', edgecolor='black')
    
    # 훈련 데이터는 파란색
    for idx in train_idx:
        ax.barh(0, 1, left=idx, height=0.5, color='#3498db', edgecolor='none')
    
    # 검증 데이터는 빨간색
    for idx in val_idx:
        ax.barh(0, 1, left=idx, height=0.5, color='#e74c3c', edgecolor='none')
    
    ax.set_xlim(0, n_samples)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'Fold {fold + 1}', fontsize=12, loc='left')
    
    # 범례 추가 (첫 번째 fold에만)
    if fold == 0:
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#3498db', label='훈련 데이터'),
                         Patch(facecolor='#e74c3c', label='검증 데이터')]
        ax.legend(handles=legend_elements, loc='upper right')

plt.suptitle('5-Fold 교차 검증: 데이터를 5번 다르게 나누어 검증', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("💡 K-Fold 교차 검증의 장점:")
print("   • 모든 데이터가 훈련과 검증에 사용됨")
print("   • 5번의 평가로 더 안정적인 성능 측정")
print("   • 데이터가 적을 때 특히 유용!")
```

### 교차 검증 실습

```python
# Iris 데이터로 교차 검증 실습
iris = load_iris()
X, y = iris.data, iris.target

# 여러 모델 비교
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(max_depth=3, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=10, random_state=42)
}

# 교차 검증 수행
cv_results = []
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    cv_results.append({
        'Model': name,
        'Scores': scores,
        'Mean': scores.mean(),
        'Std': scores.std()
    })
    
    print(f"\n{name}:")
    print(f"   각 Fold 점수: {scores.round(3)}")
    print(f"   평균 점수: {scores.mean():.3f} (±{scores.std():.3f})")

# 결과 시각화
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 박스플롯
all_scores = [result['Scores'] for result in cv_results]
model_names = [result['Model'] for result in cv_results]

bp = ax1.boxplot(all_scores, labels=model_names, patch_artist=True)
for patch, color in zip(bp['boxes'], ['#3498db', '#e74c3c', '#2ecc71']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax1.set_ylabel('정확도', fontsize=12)
ax1.set_title('모델별 교차 검증 점수 분포', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# 평균과 표준편차
means = [result['Mean'] for result in cv_results]
stds = [result['Std'] for result in cv_results]

x = np.arange(len(model_names))
ax2.bar(x, means, yerr=stds, capsize=10, alpha=0.7, 
        color=['#3498db', '#e74c3c', '#2ecc71'], edgecolor='black')
ax2.set_xticks(x)
ax2.set_xticklabels(model_names)
ax2.set_ylabel('평균 정확도', fontsize=12)
ax2.set_title('모델별 평균 성능과 표준편차', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# 점수 표시
for i, (mean, std) in enumerate(zip(means, stds)):
    ax2.text(i, mean + std + 0.01, f'{mean:.3f}', ha='center', fontsize=10)

plt.tight_layout()
plt.show()
```

### Stratified K-Fold: 클래스 불균형 해결

```python
# 불균형 데이터 생성
np.random.seed(42)
n_samples = 1000
n_class_0 = 900  # 90%
n_class_1 = 100  # 10%

X_imbalanced = np.vstack([
    np.random.randn(n_class_0, 2),
    np.random.randn(n_class_1, 2) + 2
])
y_imbalanced = np.array([0] * n_class_0 + [1] * n_class_1)

# 일반 K-Fold vs Stratified K-Fold 비교
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

# 일반 K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(kf.split(X_imbalanced)):
    ax = axes[0, fold]
    
    # 검증 세트의 클래스 분포
    val_y = y_imbalanced[val_idx]
    class_counts = np.bincount(val_y)
    
    ax.bar(['Class 0', 'Class 1'], class_counts, color=['blue', 'red'], alpha=0.7)
    ax.set_title(f'Fold {fold + 1}')
    ax.set_ylim(0, 200)
    
    # 클래스 1 비율 표시
    class_1_ratio = class_counts[1] / len(val_y) * 100
    ax.text(0.5, 0.9, f'Class 1: {class_1_ratio:.1f}%', 
           transform=ax.transAxes, ha='center')

axes[0, 0].set_ylabel('일반 K-Fold\n(불균형!)', fontsize=12, fontweight='bold')

# Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(X_imbalanced, y_imbalanced)):
    ax = axes[1, fold]
    
    # 검증 세트의 클래스 분포
    val_y = y_imbalanced[val_idx]
    class_counts = np.bincount(val_y)
    
    ax.bar(['Class 0', 'Class 1'], class_counts, color=['blue', 'red'], alpha=0.7)
    ax.set_title(f'Fold {fold + 1}')
    ax.set_ylim(0, 200)
    
    # 클래스 1 비율 표시
    class_1_ratio = class_counts[1] / len(val_y) * 100
    ax.text(0.5, 0.9, f'Class 1: {class_1_ratio:.1f}%', 
           transform=ax.transAxes, ha='center')

axes[1, 0].set_ylabel('Stratified K-Fold\n(균형 유지!)', fontsize=12, fontweight='bold')

plt.suptitle('클래스 불균형 데이터에서의 교차 검증', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n💡 Stratified K-Fold의 중요성:")
print("   • 원본 데이터의 클래스 비율을 각 Fold에서 유지")
print("   • 불균형 데이터에서 필수적!")
print("   • 더 안정적이고 신뢰할 수 있는 평가 결과")
```

## 5.4.3 과적합과 과소적합 진단

### 학습 곡선으로 문제 진단하기

모델이 너무 복잡하면 **과적합**, 너무 단순하면 **과소적합**이 발생합니다:

```python
# 과적합, 적절한 적합, 과소적합 시뮬레이션
np.random.seed(42)
n_samples = 100
X_sim = np.sort(np.random.uniform(0, 5, n_samples))
y_true = 2 * np.sin(X_sim) + X_sim
y_sim = y_true + np.random.normal(0, 0.5, n_samples)

# 세 가지 모델 준비
X_sim_reshape = X_sim.reshape(-1, 1)

# 1. 과소적합: 1차 다항식
poly1 = PolynomialFeatures(degree=1)
X_poly1 = poly1.fit_transform(X_sim_reshape)
model_under = Ridge(alpha=0)
model_under.fit(X_poly1, y_sim)

# 2. 적절한 적합: 3차 다항식
poly3 = PolynomialFeatures(degree=3)
X_poly3 = poly3.fit_transform(X_sim_reshape)
model_good = Ridge(alpha=0.1)
model_good.fit(X_poly3, y_sim)

# 3. 과적합: 15차 다항식
poly15 = PolynomialFeatures(degree=15)
X_poly15 = poly15.fit_transform(X_sim_reshape)
model_over = Ridge(alpha=0)
model_over.fit(X_poly15, y_sim)

# 예측을 위한 세밀한 X 값
X_plot = np.linspace(0, 5, 300).reshape(-1, 1)

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

models = [
    (model_under, poly1, '과소적합\n(너무 단순)', 'red'),
    (model_good, poly3, '적절한 적합\n(균형)', 'green'),
    (model_over, poly15, '과적합\n(너무 복잡)', 'purple')
]

for ax, (model, poly, title, color) in zip(axes, models):
    # 예측
    X_plot_poly = poly.transform(X_plot)
    y_pred = model.predict(X_plot_poly)
    
    # 훈련 데이터 예측
    X_train_poly = poly.transform(X_sim_reshape)
    y_train_pred = model.predict(X_train_poly)
    train_error = mean_squared_error(y_sim, y_train_pred)
    
    # 플롯
    ax.scatter(X_sim, y_sim, alpha=0.5, s=30, label='데이터')
    ax.plot(X_sim, y_true, 'g--', linewidth=2, alpha=0.7, label='실제 함수')
    ax.plot(X_plot, y_pred, color=color, linewidth=2, label='모델 예측')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.legend()
    ax.set_ylim(-5, 10)
    
    # MSE 표시
    ax.text(0.05, 0.95, f'훈련 MSE: {train_error:.2f}', 
           transform=ax.transAxes, bbox=dict(boxstyle='round', 
           facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.show()

print("\n🔍 진단 결과:")
print("   • 과소적합: 훈련 오차도 크고, 테스트 오차도 큼")
print("   • 적절한 적합: 훈련과 테스트 오차가 모두 적당함")
print("   • 과적합: 훈련 오차는 작지만, 테스트 오차가 큼")
```

### 학습 곡선 (Learning Curves)

학습 곡선은 **데이터 크기에 따른 성능 변화**를 보여줍니다:

```python
# 학습 곡선 그리기 함수
def plot_learning_curves(model, X, y, title):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='neg_mean_squared_error'
    )
    
    # MSE를 양수로 변환
    train_scores = -train_scores
    val_scores = -val_scores
    
    # 평균과 표준편차 계산
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    plt.figure(figsize=(10, 6))
    
    # 평균 곡선
    plt.plot(train_sizes, train_mean, 'o-', color='blue', 
            label='훈련 오차', linewidth=2, markersize=8)
    plt.plot(train_sizes, val_mean, 'o-', color='red', 
            label='검증 오차', linewidth=2, markersize=8)
    
    # 신뢰구간
    plt.fill_between(train_sizes, train_mean - train_std, 
                    train_mean + train_std, alpha=0.2, color='blue')
    plt.fill_between(train_sizes, val_mean - val_std, 
                    val_mean + val_std, alpha=0.2, color='red')
    
    plt.xlabel('훈련 데이터 크기', fontsize=12)
    plt.ylabel('평균 제곱 오차 (MSE)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    return train_mean, val_mean

# 캘리포니아 주택 데이터로 학습 곡선 분석
california = fetch_california_housing()
X_cal, y_cal = california.data[:1000], california.target[:1000]  # 일부만 사용

# 표준화
scaler = StandardScaler()
X_cal_scaled = scaler.fit_transform(X_cal)

# 세 가지 복잡도의 모델
models_complexity = [
    ('과소적합 모델 (선형)', Ridge(alpha=100)),
    ('적절한 모델', Ridge(alpha=1)),
    ('과적합 모델 (복잡)', Ridge(alpha=0.001))
]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (name, model) in zip(axes, models_complexity):
    plt.sca(ax)
    train_mean, val_mean = plot_learning_curves(model, X_cal_scaled, y_cal, name)
    
    # 진단 결과 추가
    gap = val_mean[-1] - train_mean[-1]
    if gap > 0.5:
        diagnosis = "과적합 징후"
        color = 'red'
    elif train_mean[-1] > 0.8:
        diagnosis = "과소적합 징후"
        color = 'orange'
    else:
        diagnosis = "적절한 적합"
        color = 'green'
    
    ax.text(0.5, 0.95, diagnosis, transform=ax.transAxes, 
           ha='center', va='top', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

plt.tight_layout()
plt.show()

print("\n📊 학습 곡선 해석법:")
print("   1. 훈련/검증 오차가 모두 높음 → 과소적합 (모델이 너무 단순)")
print("   2. 훈련 오차는 낮은데 검증 오차가 높음 → 과적합 (모델이 너무 복잡)")
print("   3. 둘 다 낮고 차이가 작음 → 적절한 적합")
print("   4. 데이터를 늘려도 개선이 없으면 → 모델 복잡도를 높여야 함")
```

## 5.4.4 편향-분산 트레이드오프

### 활쏘기 비유로 이해하기

편향(Bias)과 분산(Variance)의 관계를 활쏘기에 비유해봅시다:

```python
# 편향-분산 트레이드오프 시각화
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 과녁 중심
center = np.array([0, 0])

# 4가지 경우의 화살 위치 생성
np.random.seed(42)

# 1. 낮은 편향, 낮은 분산 (이상적)
low_bias_low_var = np.random.normal(center, 0.3, (20, 2))

# 2. 높은 편향, 낮은 분산
high_bias_low_var = np.random.normal(center + [2, 2], 0.3, (20, 2))

# 3. 낮은 편향, 높은 분산
low_bias_high_var = np.random.normal(center, 1.5, (20, 2))

# 4. 높은 편향, 높은 분산 (최악)
high_bias_high_var = np.random.normal(center + [2, 2], 1.5, (20, 2))

cases = [
    (low_bias_low_var, "낮은 편향, 낮은 분산\n(이상적!)", 'green'),
    (high_bias_low_var, "높은 편향, 낮은 분산\n(일관되게 빗나감)", 'orange'),
    (low_bias_high_var, "낮은 편향, 높은 분산\n(불안정함)", 'blue'),
    (high_bias_high_var, "높은 편향, 높은 분산\n(최악의 경우)", 'red')
]

for ax, (points, title, color) in zip(axes.flat, cases):
    # 과녁 그리기
    for radius in [3, 2, 1]:
        circle = plt.Circle(center, radius, fill=False, 
                          edgecolor='gray', linewidth=1)
        ax.add_patch(circle)
    
    # 중심점
    ax.plot(0, 0, 'ko', markersize=10)
    
    # 화살 위치
    ax.scatter(points[:, 0], points[:, 1], s=100, alpha=0.7, 
              color=color, edgecolor='black')
    
    # 평균 위치
    mean_point = points.mean(axis=0)
    ax.plot(mean_point[0], mean_point[1], 'r*', markersize=20, 
           label='평균 위치')
    
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)

plt.tight_layout()
plt.show()

# 모델 복잡도에 따른 편향-분산 변화
complexity = np.linspace(0, 10, 100)
bias_squared = 10 / (complexity + 1)**2
variance = complexity**1.5 / 10
total_error = bias_squared + variance

plt.figure(figsize=(10, 6))
plt.plot(complexity, bias_squared, 'b-', linewidth=2, label='편향²')
plt.plot(complexity, variance, 'r-', linewidth=2, label='분산')
plt.plot(complexity, total_error, 'g-', linewidth=3, label='총 오차')

# 최적점 표시
optimal_idx = np.argmin(total_error)
plt.plot(complexity[optimal_idx], total_error[optimal_idx], 'go', 
        markersize=15, label='최적 복잡도')
plt.axvline(x=complexity[optimal_idx], color='gray', linestyle='--', alpha=0.5)

plt.xlabel('모델 복잡도', fontsize=12)
plt.ylabel('오차', fontsize=12)
plt.title('편향-분산 트레이드오프', fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# 영역별 설명 추가
plt.text(1, 8, '과소적합\n(높은 편향)', fontsize=11, ha='center', 
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
plt.text(8, 8, '과적합\n(높은 분산)', fontsize=11, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
plt.text(complexity[optimal_idx], -1, '최적점', fontsize=11, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

plt.tight_layout()
plt.show()

print("\n💡 편향-분산 트레이드오프 핵심:")
print("   • 편향(Bias): 모델의 예측이 실제값에서 얼마나 벗어나는가")
print("   • 분산(Variance): 다른 데이터에서 예측이 얼마나 달라지는가")
print("   • 단순한 모델: 높은 편향, 낮은 분산")
print("   • 복잡한 모델: 낮은 편향, 높은 분산")
print("   • 목표: 둘 사이의 최적 균형점 찾기!")
```

## 5.4.5 하이퍼파라미터 튜닝

### 검증 곡선으로 최적값 찾기

모델의 하이퍼파라미터를 체계적으로 조정해봅시다:

```python
# 검증 곡선 예시: Ridge 회귀의 alpha 파라미터
param_range = np.logspace(-3, 3, 20)
train_scores, val_scores = validation_curve(
    Ridge(), X_cal_scaled, y_cal, 
    param_name='alpha', param_range=param_range,
    cv=5, scoring='neg_mean_squared_error'
)

# MSE를 양수로 변환하고 RMSE로 계산
train_rmse = np.sqrt(-train_scores)
val_rmse = np.sqrt(-val_scores)

# 평균과 표준편차
train_mean = train_rmse.mean(axis=1)
train_std = train_rmse.std(axis=1)
val_mean = val_rmse.mean(axis=1)
val_std = val_rmse.std(axis=1)

plt.figure(figsize=(10, 6))

# 평균 곡선
plt.semilogx(param_range, train_mean, 'b-', linewidth=2, 
            label='훈련 RMSE', marker='o', markersize=8)
plt.semilogx(param_range, val_mean, 'r-', linewidth=2, 
            label='검증 RMSE', marker='o', markersize=8)

# 신뢰구간
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, 
                alpha=0.2, color='blue')
plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, 
                alpha=0.2, color='red')

# 최적값 표시
best_idx = np.argmin(val_mean)
best_alpha = param_range[best_idx]
plt.axvline(x=best_alpha, color='green', linestyle='--', linewidth=2)
plt.plot(best_alpha, val_mean[best_idx], 'go', markersize=15)

plt.xlabel('정규화 강도 (alpha)', fontsize=12)
plt.ylabel('RMSE', fontsize=12)
plt.title('검증 곡선: Ridge 회귀의 최적 Alpha 찾기', fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# 영역별 설명
plt.text(0.0001, val_mean[0] + 0.1, '과적합\n(alpha 너무 작음)', 
        fontsize=10, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
plt.text(100, val_mean[-1] + 0.1, '과소적합\n(alpha 너무 큼)', 
        fontsize=10, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
plt.text(best_alpha, val_mean[best_idx] - 0.15, f'최적: α={best_alpha:.3f}', 
        fontsize=10, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

plt.tight_layout()
plt.show()

print(f"\n🎯 최적 하이퍼파라미터:")
print(f"   • 최적 alpha 값: {best_alpha:.3f}")
print(f"   • 검증 RMSE: {val_mean[best_idx]:.3f}")
```

### 여러 하이퍼파라미터 동시 튜닝

```python
# Decision Tree의 두 가지 하이퍼파라미터 튜닝
from sklearn.model_selection import GridSearchCV

# Wine 데이터셋 사용
wine = load_wine()
X_wine, y_wine = wine.data, wine.target

# 파라미터 그리드
param_grid = {
    'max_depth': [2, 3, 4, 5, 6],
    'min_samples_split': [2, 5, 10, 20, 30]
}

# GridSearchCV
dt = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_wine, y_wine)

# 결과를 히트맵으로 시각화
results = grid_search.cv_results_
scores = results['mean_test_score'].reshape(5, 5)

plt.figure(figsize=(10, 8))
sns.heatmap(scores, annot=True, fmt='.3f', cmap='YlOrRd',
            xticklabels=param_grid['min_samples_split'],
            yticklabels=param_grid['max_depth'])
plt.xlabel('min_samples_split', fontsize=12)
plt.ylabel('max_depth', fontsize=12)
plt.title('Decision Tree 하이퍼파라미터 그리드 서치', fontsize=14, fontweight='bold')

# 최적 파라미터 표시
best_params = grid_search.best_params_
best_i = param_grid['max_depth'].index(best_params['max_depth'])
best_j = param_grid['min_samples_split'].index(best_params['min_samples_split'])
plt.plot(best_j + 0.5, best_i + 0.5, 'ws', markersize=20, 
        markeredgecolor='black', markeredgewidth=3)

plt.tight_layout()
plt.show()

print(f"\n🏆 그리드 서치 결과:")
print(f"   • 최적 파라미터: {grid_search.best_params_}")
print(f"   • 최고 점수: {grid_search.best_score_:.3f}")
print(f"   • 총 {len(param_grid['max_depth']) * len(param_grid['min_samples_split'])}개 조합 테스트")
```

## 5.4.6 미니 프로젝트: 종합적인 모델 평가

이제 배운 모든 기법을 활용하여 체계적인 모델 평가를 수행해봅시다:

```python
# 프로젝트: Wine 품질 분류 모델 개발 및 평가
print("=== 🍷 Wine 품질 분류 프로젝트 ===\n")
print("목표: 와인의 화학적 특성으로 품질 등급을 예측")
print("과제: 최적의 모델을 체계적으로 선택하기\n")

# 1단계: 데이터 준비
print("1️⃣ 데이터 준비 및 탐색")
print("-" * 50)

X_train, X_test, y_train, y_test = train_test_split(
    X_wine, y_wine, test_size=0.2, random_state=42, stratify=y_wine
)

print(f"전체 데이터: {len(X_wine)}개")
print(f"훈련 데이터: {len(X_train)}개")
print(f"테스트 데이터: {len(X_test)}개 (최종 평가용, 절대 건드리지 않음!)")
print(f"클래스 분포: {np.bincount(y_wine)}")

# 2단계: 기본 모델 비교 (교차 검증)
print("\n2️⃣ 기본 모델 비교 (5-Fold 교차 검증)")
print("-" * 50)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

cv_results = []
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    cv_results.append({
        'Model': name,
        'Mean CV Score': scores.mean(),
        'Std': scores.std(),
        'Scores': scores
    })
    print(f"{name}: {scores.mean():.3f} (±{scores.std():.3f})")

# 3단계: 최고 모델의 학습 곡선 분석
print("\n3️⃣ Random Forest 학습 곡선 분석")
print("-" * 50)

best_model = RandomForestClassifier(n_estimators=100, random_state=42)
train_sizes, train_scores, val_scores = learning_curve(
    best_model, X_train, y_train, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy', n_jobs=-1
)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', 
        color='blue', label='훈련 정확도', linewidth=2)
plt.plot(train_sizes, val_scores.mean(axis=1), 'o-', 
        color='red', label='검증 정확도', linewidth=2)
plt.fill_between(train_sizes, 
                train_scores.mean(axis=1) - train_scores.std(axis=1),
                train_scores.mean(axis=1) + train_scores.std(axis=1), 
                alpha=0.2, color='blue')
plt.fill_between(train_sizes, 
                val_scores.mean(axis=1) - val_scores.std(axis=1),
                val_scores.mean(axis=1) + val_scores.std(axis=1), 
                alpha=0.2, color='red')
plt.xlabel('훈련 데이터 크기', fontsize=12)
plt.ylabel('정확도', fontsize=12)
plt.title('Random Forest 학습 곡선', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 4단계: 하이퍼파라미터 튜닝
print("\n4️⃣ Random Forest 하이퍼파라미터 최적화")
print("-" * 50)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=5, scoring='accuracy', n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"최적 파라미터: {grid_search.best_params_}")
print(f"최고 CV 점수: {grid_search.best_score_:.3f}")

# 5단계: 최종 모델 평가
print("\n5️⃣ 최종 모델 평가 (테스트 세트)")
print("-" * 50)

final_model = grid_search.best_estimator_
test_score = final_model.score(X_test, y_test)
train_score = final_model.score(X_train, y_train)

print(f"훈련 정확도: {train_score:.3f}")
print(f"테스트 정확도: {test_score:.3f}")
print(f"과적합 정도: {train_score - test_score:.3f}")

if train_score - test_score < 0.05:
    print("✅ 과적합이 거의 없는 좋은 모델입니다!")
else:
    print("⚠️ 약간의 과적합이 있습니다. 정규화를 고려해보세요.")

# 6단계: 모델 평가 요약 리포트
print("\n📊 모델 평가 종합 리포트")
print("=" * 50)

report = pd.DataFrame({
    'Metric': ['Initial CV Score', 'Optimized CV Score', 'Final Test Score', 
               'Overfitting Gap', 'Total Evaluation Time'],
    'Value': [
        f"{cv_results[2]['Mean CV Score']:.3f}",
        f"{grid_search.best_score_:.3f}",
        f"{test_score:.3f}",
        f"{train_score - test_score:.3f}",
        "~2분"
    ]
})
print(report.to_string(index=False))

print("\n💡 핵심 인사이트:")
print("   1. Random Forest가 가장 좋은 기본 성능을 보임")
print("   2. 하이퍼파라미터 튜닝으로 성능 개선")
print("   3. 과적합이 적어 일반화 성능이 좋음")
print("   4. 최종 모델은 실제 서비스에 배포 가능한 수준")
```

## 🎯 직접 해보기

### 연습 문제 1: 교차 검증 구현
```python
# 주어진 데이터에서 3-Fold 교차 검증을 수동으로 구현해보세요
X_practice = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
y_practice = np.array([0, 1, 0, 1, 0, 1])

# TODO: KFold를 사용하여 3-Fold 교차 검증 수행
# 1. KFold 객체 생성 (n_splits=3)
# 2. 각 fold에서 훈련/검증 인덱스 출력
# 3. 각 fold의 데이터 크기 확인

# 여기에 코드를 작성하세요
```

### 연습 문제 2: 학습 곡선 해석
```python
# 다음 학습 곡선을 보고 진단해보세요
train_scores = [0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.93, 0.94]
val_scores = [0.58, 0.65, 0.7, 0.72, 0.73, 0.73, 0.73, 0.73]
data_sizes = [50, 100, 200, 400, 600, 800, 1000, 1200]

# TODO: 
# 1. 학습 곡선을 그려보세요
# 2. 과적합인지 과소적합인지 진단하세요
# 3. 개선 방법을 제안하세요

# 여기에 코드를 작성하세요
```

### 연습 문제 3: 하이퍼파라미터 튜닝
```python
# Decision Tree의 max_depth를 검증 곡선으로 찾아보세요
from sklearn.datasets import make_classification

# 가상 데이터 생성
X_tune, y_tune = make_classification(n_samples=200, n_features=10, 
                                    n_informative=5, random_state=42)

# TODO:
# 1. max_depth를 1부터 20까지 변화시키며 검증 곡선 그리기
# 2. 최적의 max_depth 찾기
# 3. 과적합이 시작되는 지점 파악

# 여기에 코드를 작성하세요
```

## 📚 핵심 정리

### ✅ 모델 평가의 핵심 원칙
1. **데이터 분할**
   - 훈련(60%), 검증(20%), 테스트(20%)
   - 테스트 데이터는 최종 평가까지 절대 사용 금지

2. **교차 검증**
   - K-Fold: 데이터를 K개로 나누어 K번 평가
   - Stratified K-Fold: 클래스 비율 유지 (불균형 데이터 필수)
   - 더 안정적이고 신뢰할 수 있는 성능 추정

3. **과적합/과소적합 진단**
   - 학습 곡선으로 문제 파악
   - 과소적합: 훈련/검증 오차 모두 높음
   - 과적합: 훈련 오차는 낮고 검증 오차는 높음

4. **편향-분산 트레이드오프**
   - 편향: 예측의 정확도
   - 분산: 예측의 일관성
   - 최적점: 둘 사이의 균형

5. **하이퍼파라미터 튜닝**
   - 검증 곡선: 단일 파라미터 최적화
   - 그리드 서치: 여러 파라미터 동시 최적화
   - 교차 검증과 함께 사용

### 💡 실무 팁
- 항상 교차 검증으로 모델 평가
- 학습 곡선을 그려 문제 진단
- 단순한 모델부터 시작해서 점진적으로 복잡도 증가
- 테스트 세트는 마지막에 단 한 번만 사용

---

## 🚀 다음 파트에서는

**프로젝트: 예측 모델 개발 및 평가**를 진행합니다:
- 실제 비즈니스 문제 정의
- 데이터 전처리부터 모델 평가까지 전체 워크플로우
- 여러 모델 비교 및 최적 모델 선택
- 결과 해석과 비즈니스 인사이트 도출

지금까지 배운 모든 내용을 종합하여 **완전한 머신러닝 프로젝트**를 수행해봅시다!
