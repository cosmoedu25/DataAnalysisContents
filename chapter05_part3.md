# 5장 Part 3: 회귀 알고리즘의 이해와 구현

## 학습 목표
이번 파트를 완료하면 다음을 할 수 있습니다:
- 회귀 문제의 개념을 이해하고 분류 문제와의 차이점을 설명할 수 있다
- 선형 회귀와 다항 회귀의 원리를 이해하고 구현할 수 있다
- Ridge와 Lasso 정규화 기법을 활용하여 과적합을 방지할 수 있다
- MSE, RMSE, MAE, R² 등 회귀 성능 평가 지표를 계산하고 해석할 수 있다
- 잔차 분석을 통해 모델의 적합성을 진단할 수 있다
- 실제 부동산 가격 예측 프로젝트를 수행할 수 있다

## 이번 파트 미리보기
지금까지 분류 알고리즘을 통해 '범주'를 예측하는 방법을 배웠습니다. 이번에는 **연속적인 수치**를 예측하는 회귀 알고리즘을 학습합니다. 집값이나 온도, 매출액처럼 숫자로 된 값을 예측하는 것이 회귀 문제입니다. 선형 회귀부터 시작해 정규화 기법까지 단계별로 학습하며, 실제 캘리포니아 주택 가격을 예측하는 프로젝트로 마무리합니다.

---

## 5.3.1 회귀 문제의 이해

### 일상 속 회귀 문제

회귀(Regression)는 우리 일상에서 수없이 마주치는 예측 문제입니다:

```python
# 필요한 라이브러리 불러오기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.datasets import fetch_california_housing
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 회귀 문제 예시
regression_examples = {
    "🏠 부동산": ["집값 예측", "월세 예측", "전세가 예측"],
    "📈 경제": ["주가 예측", "매출액 예측", "GDP 성장률 예측"],
    "🌡️ 날씨": ["기온 예측", "강수량 예측", "미세먼지 농도 예측"],
    "🎓 교육": ["시험 점수 예측", "대학 입학률 예측", "학습 시간 예측"],
    "🏥 의료": ["혈압 예측", "치료 비용 예측", "회복 기간 예측"]
}

print("=== 일상 속 회귀 문제들 ===\n")
for category, examples in regression_examples.items():
    print(f"{category}")
    for example in examples:
        print(f"  • {example}")
    print()
```

### 회귀와 분류의 차이점

회귀와 분류의 가장 큰 차이는 **예측값의 성격**입니다:

```python
# 회귀 vs 분류 비교 시각화
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 분류 예시: 이산적 출력
np.random.seed(42)
x_class = np.random.randn(100, 2)
y_class = (x_class[:, 0] + x_class[:, 1] > 0).astype(int)

scatter1 = ax1.scatter(x_class[:, 0], x_class[:, 1], c=y_class, 
                      cmap='RdBu', s=50, alpha=0.7, edgecolor='k')
ax1.set_title('분류: 이산적 출력 (0 또는 1)', fontsize=14, fontweight='bold')
ax1.set_xlabel('특성 1')
ax1.set_ylabel('특성 2')
ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3)

# 범례 추가
handles = [plt.Line2D([0], [0], marker='o', color='w', 
                     markerfacecolor=c, markersize=10) 
          for c in ['#d62728', '#1f77b4']]
ax1.legend(handles, ['클래스 0', '클래스 1'])

# 회귀 예시: 연속적 출력
x_reg = np.linspace(0, 10, 100)
y_reg = 2 * x_reg + 3 + np.random.normal(0, 2, 100)

scatter2 = ax2.scatter(x_reg, y_reg, c=y_reg, cmap='viridis', 
                      s=50, alpha=0.7, edgecolor='k')
ax2.plot(x_reg, 2 * x_reg + 3, 'r-', linewidth=2, label='회귀선')
ax2.set_title('회귀: 연속적 출력 (실수값)', fontsize=14, fontweight='bold')
ax2.set_xlabel('특성')
ax2.set_ylabel('예측값')
ax2.legend()

# 컬러바 추가
cbar = plt.colorbar(scatter2, ax=ax2)
cbar.set_label('예측값')

plt.tight_layout()
plt.show()

# 차이점 정리
print("\n=== 회귀 vs 분류 핵심 차이점 ===\n")
comparison_df = pd.DataFrame({
    '구분': ['예측 대상', '출력값 유형', '예시', '평가 지표'],
    '분류': ['범주(Category)', '이산적(Discrete)', '스팸/정상, 개/고양이', '정확도, F1-점수'],
    '회귀': ['수치(Number)', '연속적(Continuous)', '집값, 온도, 점수', 'MSE, R²']
})
print(comparison_df.to_string(index=False))
```

## 5.3.2 선형 회귀 (Linear Regression)

### 선형 회귀의 직관적 이해

선형 회귀는 **데이터에 가장 잘 맞는 직선**을 찾는 과정입니다:

```python
# 간단한 선형 관계 시뮬레이션
np.random.seed(42)
study_hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
test_scores = 5 * study_hours + 30 + np.random.normal(0, 3, 10)

# 시각화
plt.figure(figsize=(10, 6))
plt.scatter(study_hours, test_scores, s=100, alpha=0.7, 
           edgecolor='k', label='실제 데이터')

# 선형 회귀 적합
model = LinearRegression()
X = study_hours.reshape(-1, 1)
model.fit(X, test_scores)
predictions = model.predict(X)

plt.plot(study_hours, predictions, 'r-', linewidth=2, 
        label=f'회귀선: y = {model.coef_[0]:.1f}x + {model.intercept_:.1f}')

# 예측 오차 표시
for i in range(len(study_hours)):
    plt.plot([study_hours[i], study_hours[i]], 
            [test_scores[i], predictions[i]], 
            'gray', linestyle='--', alpha=0.5)

plt.xlabel('공부 시간 (시간)', fontsize=12)
plt.ylabel('시험 점수', fontsize=12)
plt.title('선형 회귀: 데이터에 가장 잘 맞는 직선 찾기', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"🎯 회귀 방정식: 점수 = {model.coef_[0]:.1f} × 공부시간 + {model.intercept_:.1f}")
print(f"   → 공부를 1시간 더 하면 점수가 약 {model.coef_[0]:.1f}점 오릅니다!")
```

### 최소제곱법의 원리

선형 회귀는 **예측값과 실제값의 차이(잔차)의 제곱합을 최소화**하는 직선을 찾습니다:

```python
# 최소제곱법 시각화
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 왼쪽: 나쁜 회귀선
bad_slope = 3
bad_intercept = 40
bad_predictions = bad_slope * study_hours + bad_intercept

ax1.scatter(study_hours, test_scores, s=100, alpha=0.7, edgecolor='k')
ax1.plot(study_hours, bad_predictions, 'b-', linewidth=2, label='나쁜 회귀선')

# 잔차 표시
for i in range(len(study_hours)):
    ax1.plot([study_hours[i], study_hours[i]], 
            [test_scores[i], bad_predictions[i]], 
            'red', linewidth=2, alpha=0.7)
    
bad_sse = np.sum((test_scores - bad_predictions) ** 2)
ax1.set_title(f'잘못된 회귀선\n잔차 제곱합: {bad_sse:.1f}', fontsize=12)
ax1.set_xlabel('공부 시간')
ax1.set_ylabel('시험 점수')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 오른쪽: 최적 회귀선
ax2.scatter(study_hours, test_scores, s=100, alpha=0.7, edgecolor='k')
ax2.plot(study_hours, predictions, 'g-', linewidth=2, label='최적 회귀선')

# 잔차 표시
for i in range(len(study_hours)):
    ax2.plot([study_hours[i], study_hours[i]], 
            [test_scores[i], predictions[i]], 
            'green', linewidth=2, alpha=0.7)

good_sse = np.sum((test_scores - predictions) ** 2)
ax2.set_title(f'최적 회귀선 (최소제곱법)\n잔차 제곱합: {good_sse:.1f}', fontsize=12)
ax2.set_xlabel('공부 시간')
ax2.set_ylabel('시험 점수')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n💡 최소제곱법의 핵심:")
print(f"   • 나쁜 회귀선의 오차: {bad_sse:.1f}")
print(f"   • 최적 회귀선의 오차: {good_sse:.1f}")
print(f"   • 오차 감소: {bad_sse - good_sse:.1f} ({(bad_sse - good_sse)/bad_sse*100:.1f}%)")
```

### 다변량 선형 회귀

실제 문제에서는 여러 특성을 동시에 고려해야 합니다:

```python
# 캘리포니아 주택 데이터셋 로드
california = fetch_california_housing()
X_cal = pd.DataFrame(california.data, columns=california.feature_names)
y_cal = california.target

print("=== 캘리포니아 주택 데이터셋 ===")
print(f"데이터 크기: {X_cal.shape}")
print(f"\n특성 설명:")
for i, (feature, description) in enumerate(zip(california.feature_names, 
    ['평균 소득', '집 나이', '평균 방 수', '평균 침실 수', 
     '인구', '평균 가구원 수', '위도', '경도'])):
    print(f"  • {feature}: {description}")

# 상관관계 히트맵
plt.figure(figsize=(10, 8))
correlation_matrix = X_cal.copy()
correlation_matrix['Price'] = y_cal
corr = correlation_matrix.corr()

mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', 
            cmap='coolwarm', center=0, vmin=-1, vmax=1,
            square=True, linewidths=1, cbar_kws={"shrink": .8})
plt.title('주택 가격과 특성들 간의 상관관계', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# 다변량 선형 회귀 적합
X_train, X_test, y_train, y_test = train_test_split(
    X_cal, y_cal, test_size=0.2, random_state=42
)

# 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 모델 학습
multi_model = LinearRegression()
multi_model.fit(X_train_scaled, y_train)

# 특성 중요도 시각화
feature_importance = pd.DataFrame({
    'feature': california.feature_names,
    'coefficient': multi_model.coef_
}).sort_values('coefficient', key=abs, ascending=False)

plt.figure(figsize=(10, 6))
colors = ['green' if x > 0 else 'red' for x in feature_importance['coefficient']]
plt.barh(feature_importance['feature'], feature_importance['coefficient'], 
         color=colors, alpha=0.7, edgecolor='black')
plt.xlabel('회귀 계수', fontsize=12)
plt.title('다변량 선형 회귀: 각 특성의 영향력', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

print("\n🔍 해석:")
print("   • 양(+)의 계수: 해당 특성이 증가하면 집값도 증가")
print("   • 음(-)의 계수: 해당 특성이 증가하면 집값은 감소")
print(f"   • 가장 중요한 특성: {feature_importance.iloc[0]['feature']}")
```

## 5.3.3 다항 회귀 (Polynomial Regression)

### 비선형 관계 모델링

데이터가 직선이 아닌 **곡선** 패턴을 보일 때는 다항 회귀를 사용합니다:

```python
# 비선형 데이터 생성
np.random.seed(42)
X_nonlinear = np.linspace(-3, 3, 100)
y_nonlinear = 0.5 * X_nonlinear**3 + X_nonlinear**2 - 2*X_nonlinear + 1 + np.random.normal(0, 1, 100)

# 선형 vs 다항 회귀 비교
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1차 (선형)
ax = axes[0, 0]
linear_model = LinearRegression()
X_reshape = X_nonlinear.reshape(-1, 1)
linear_model.fit(X_reshape, y_nonlinear)
y_linear_pred = linear_model.predict(X_reshape)

ax.scatter(X_nonlinear, y_nonlinear, alpha=0.5, s=30, label='실제 데이터')
ax.plot(X_nonlinear, y_linear_pred, 'r-', linewidth=2, label='1차 (선형)')
ax.set_title('1차 다항식 (선형 회귀)', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# 2차, 3차, 5차 다항 회귀
degrees = [2, 3, 5]
positions = [(0, 1), (1, 0), (1, 1)]

for degree, (i, j) in zip(degrees, positions):
    ax = axes[i, j]
    
    # 다항 특성 생성
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly_features.fit_transform(X_reshape)
    
    # 모델 학습
    poly_model = LinearRegression()
    poly_model.fit(X_poly, y_nonlinear)
    
    # 예측
    X_plot = np.linspace(-3, 3, 300).reshape(-1, 1)
    X_plot_poly = poly_features.transform(X_plot)
    y_poly_pred = poly_model.predict(X_plot_poly)
    
    ax.scatter(X_nonlinear, y_nonlinear, alpha=0.5, s=30, label='실제 데이터')
    ax.plot(X_plot, y_poly_pred, 'g-', linewidth=2, label=f'{degree}차 다항식')
    ax.set_title(f'{degree}차 다항 회귀', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # R² 점수 표시
    y_pred_train = poly_model.predict(X_poly)
    r2 = r2_score(y_nonlinear, y_pred_train)
    ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()

print("\n💡 다항 회귀의 핵심:")
print("   • 차수가 높을수록 복잡한 패턴을 포착할 수 있음")
print("   • 하지만 너무 높은 차수는 과적합을 일으킬 수 있음")
print("   • 적절한 차수 선택이 중요!")
```

## 5.3.4 정규화 기법: Ridge와 Lasso

### 과적합 문제와 정규화

과적합을 방지하기 위해 **회귀 계수에 페널티**를 부여하는 정규화 기법을 사용합니다:

```python
# 과적합 시연을 위한 데이터
np.random.seed(42)
n_samples = 30
X_overfit = np.sort(np.random.uniform(0, 4, n_samples))
y_true = np.sin(X_overfit) + X_overfit/2
y_overfit = y_true + np.random.normal(0, 0.3, n_samples)

# 고차 다항식으로 과적합 유도
X_overfit_reshape = X_overfit.reshape(-1, 1)
poly_features = PolynomialFeatures(degree=15)
X_poly = poly_features.fit_transform(X_overfit_reshape)

# 일반 선형 회귀, Ridge, Lasso 비교
models = {
    'Linear Regression': LinearRegression(),
    'Ridge (α=0.1)': Ridge(alpha=0.1),
    'Ridge (α=1.0)': Ridge(alpha=1.0),
    'Lasso (α=0.01)': Lasso(alpha=0.01),
    'Lasso (α=0.1)': Lasso(alpha=0.1)
}

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for idx, (name, model) in enumerate(models.items()):
    ax = axes[idx]
    
    # 모델 학습
    model.fit(X_poly, y_overfit)
    
    # 예측을 위한 세밀한 X 값
    X_plot = np.linspace(0, 4, 300).reshape(-1, 1)
    X_plot_poly = poly_features.transform(X_plot)
    y_pred = model.predict(X_plot_poly)
    
    # 시각화
    ax.scatter(X_overfit, y_overfit, s=50, alpha=0.7, 
              edgecolor='k', label='학습 데이터')
    ax.plot(X_overfit, y_true, 'g--', linewidth=2, 
           label='실제 함수', alpha=0.7)
    ax.plot(X_plot, y_pred, 'r-', linewidth=2, label='예측')
    
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-2, 4)
    
    # 계수의 크기 표시
    coef_norm = np.linalg.norm(model.coef_)
    ax.text(0.05, 0.95, f'||w|| = {coef_norm:.1f}', 
           transform=ax.transAxes, 
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# 빈 subplot 숨기기
axes[-1].set_visible(False)

plt.tight_layout()
plt.show()

# 정규화 방법 비교
print("\n=== Ridge vs Lasso 정규화 비교 ===")
comparison_df = pd.DataFrame({
    '특징': ['페널티 유형', '계수 처리', '특성 선택', '사용 시기'],
    'Ridge (L2)': ['계수 제곱합', '작게 만듦', '모든 특성 유지', '모든 특성이 중요할 때'],
    'Lasso (L1)': ['계수 절댓값 합', '0으로 만들 수 있음', '자동 특성 선택', '일부 특성만 중요할 때']
})
print(comparison_df.to_string(index=False))
```

### 정규화 강도(α) 선택하기

```python
# 다양한 알파 값에 대한 성능 비교
alphas = np.logspace(-4, 2, 50)
ridge_scores = []
lasso_scores = []

# 데이터 준비
X_train, X_test, y_train, y_test = train_test_split(
    X_cal, y_cal, test_size=0.2, random_state=42
)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 각 알파 값에 대해 모델 학습 및 평가
for alpha in alphas:
    # Ridge
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)
    ridge_scores.append(ridge.score(X_test_scaled, y_test))
    
    # Lasso
    lasso = Lasso(alpha=alpha, max_iter=1000)
    lasso.fit(X_train_scaled, y_train)
    lasso_scores.append(lasso.score(X_test_scaled, y_test))

# 시각화
plt.figure(figsize=(10, 6))
plt.semilogx(alphas, ridge_scores, 'b-', linewidth=2, label='Ridge')
plt.semilogx(alphas, lasso_scores, 'r-', linewidth=2, label='Lasso')
plt.xlabel('정규화 강도 (α)', fontsize=12)
plt.ylabel('R² 점수', fontsize=12)
plt.title('정규화 강도에 따른 모델 성능', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# 최적 알파 표시
best_ridge_alpha = alphas[np.argmax(ridge_scores)]
best_lasso_alpha = alphas[np.argmax(lasso_scores)]
plt.axvline(x=best_ridge_alpha, color='blue', linestyle='--', alpha=0.5)
plt.axvline(x=best_lasso_alpha, color='red', linestyle='--', alpha=0.5)

plt.show()

print(f"\n🎯 최적 정규화 강도:")
print(f"   • Ridge: α = {best_ridge_alpha:.4f} (R² = {max(ridge_scores):.3f})")
print(f"   • Lasso: α = {best_lasso_alpha:.4f} (R² = {max(lasso_scores):.3f})")
```

## 5.3.5 회귀 평가 지표

### 주요 평가 지표 이해하기

회귀 모델의 성능을 평가하는 다양한 지표들을 알아봅시다:

```python
# 예측값 생성 (Ridge 모델 사용)
ridge_best = Ridge(alpha=best_ridge_alpha)
ridge_best.fit(X_train_scaled, y_train)
y_pred = ridge_best.predict(X_test_scaled)

# 평가 지표 계산
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. 예측 vs 실제 산점도
ax = axes[0, 0]
ax.scatter(y_test, y_pred, alpha=0.5, s=30)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
        'r--', linewidth=2)
ax.set_xlabel('실제값', fontsize=12)
ax.set_ylabel('예측값', fontsize=12)
ax.set_title('예측값 vs 실제값', fontsize=12, fontweight='bold')
ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# 2. 잔차 플롯
ax = axes[0, 1]
residuals = y_test - y_pred
ax.scatter(y_pred, residuals, alpha=0.5, s=30)
ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax.set_xlabel('예측값', fontsize=12)
ax.set_ylabel('잔차 (실제 - 예측)', fontsize=12)
ax.set_title('잔차 플롯', fontsize=12, fontweight='bold')

# 3. 잔차 히스토그램
ax = axes[1, 0]
ax.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
ax.set_xlabel('잔차', fontsize=12)
ax.set_ylabel('빈도', fontsize=12)
ax.set_title('잔차 분포', fontsize=12, fontweight='bold')

# 4. 평가 지표 요약
ax = axes[1, 1]
ax.axis('off')
metrics_text = f"""
📊 회귀 평가 지표

MSE (Mean Squared Error): {mse:.3f}
  • 오차의 제곱 평균
  • 큰 오차에 더 큰 페널티

RMSE (Root MSE): {rmse:.3f}
  • MSE의 제곱근
  • 원래 단위와 동일

MAE (Mean Absolute Error): {mae:.3f}
  • 오차의 절댓값 평균
  • 모든 오차를 동일하게 처리

R² (결정계수): {r2:.3f}
  • 모델이 설명하는 분산의 비율
  • 1에 가까울수록 좋음
"""
ax.text(0.1, 0.5, metrics_text, transform=ax.transAxes, 
        fontsize=12, verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.tight_layout()
plt.show()

# 평가 지표 해석
print("\n🔍 평가 지표 해석:")
print(f"   • RMSE = {rmse:.3f}: 평균적으로 예측이 실제값에서 ±{rmse:.3f} 정도 벗어남")
print(f"   • MAE = {mae:.3f}: 평균 절대 오차가 {mae:.3f}")
print(f"   • R² = {r2:.3f}: 모델이 데이터 변동성의 {r2*100:.1f}%를 설명함")
```

## 5.3.6 미니 프로젝트: 캘리포니아 주택 가격 예측

이제 배운 모든 기법을 활용하여 실제 주택 가격을 예측해봅시다:

```python
# 프로젝트 개요
print("=== 🏠 캘리포니아 주택 가격 예측 프로젝트 ===\n")
print("목표: 주택의 특성을 바탕으로 가격을 예측하는 모델 구축")
print("데이터: 캘리포니아 20,640개 지역의 주택 정보")
print("평가: 여러 회귀 모델을 비교하여 최적 모델 선택\n")

# 1. 데이터 탐색
print("1️⃣ 데이터 탐색")
print("-" * 50)
print(f"데이터 크기: {X_cal.shape}")
print(f"타겟 통계: 평균 ${y_cal.mean():.2f}만, 표준편차 ${y_cal.std():.2f}만")

# 주요 특성과 가격의 관계 시각화
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()

for idx, col in enumerate(X_cal.columns):
    ax = axes[idx]
    scatter = ax.scatter(X_cal[col], y_cal, alpha=0.3, s=10, 
                        c=y_cal, cmap='viridis')
    ax.set_xlabel(col)
    ax.set_ylabel('House Price')
    ax.set_title(f'Price vs {col}')

plt.tight_layout()
plt.show()

# 2. 데이터 전처리
print("\n2️⃣ 데이터 전처리")
print("-" * 50)

# 특성 공학: 새로운 특성 생성
X_enhanced = X_cal.copy()
X_enhanced['rooms_per_household'] = X_cal['AveRooms'] / X_cal['AveOccup']
X_enhanced['bedrooms_per_room'] = X_cal['AveBedrms'] / X_cal['AveRooms']
X_enhanced['population_per_household'] = X_cal['Population'] / X_cal['HouseAge']

print("✅ 새로운 특성 생성:")
print("   • rooms_per_household: 가구당 방 수")
print("   • bedrooms_per_room: 방당 침실 비율")
print("   • population_per_household: 가구당 인구")

# 데이터 분할 및 스케일링
X_train, X_test, y_train, y_test = train_test_split(
    X_enhanced, y_cal, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n✅ 학습 데이터: {X_train.shape[0]}개")
print(f"✅ 테스트 데이터: {X_test.shape[0]}개")

# 3. 모델 학습 및 비교
print("\n3️⃣ 모델 학습 및 비교")
print("-" * 50)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge (α=1.0)': Ridge(alpha=1.0),
    'Lasso (α=0.01)': Lasso(alpha=0.01),
    'Polynomial (degree=2)': 'poly'
}

results = []

for name, model in models.items():
    if name == 'Polynomial (degree=2)':
        # 다항 특성 생성
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_train_poly = poly.fit_transform(X_train_scaled)
        X_test_poly = poly.transform(X_test_scaled)
        
        model = Ridge(alpha=1.0)  # 다항 회귀에는 Ridge 적용
        model.fit(X_train_poly, y_train)
        
        train_pred = model.predict(X_train_poly)
        test_pred = model.predict(X_test_poly)
    else:
        model.fit(X_train_scaled, y_train)
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)
    
    # 평가
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_mae = mean_absolute_error(y_test, test_pred)
    
    results.append({
        'Model': name,
        'Train R²': train_r2,
        'Test R²': test_r2,
        'Test RMSE': test_rmse,
        'Test MAE': test_mae
    })
    
    print(f"\n{name}:")
    print(f"   Train R²: {train_r2:.3f}")
    print(f"   Test R²:  {test_r2:.3f}")
    print(f"   Test RMSE: ${test_rmse:.3f}만")

# 결과 시각화
results_df = pd.DataFrame(results)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# R² 비교
x = np.arange(len(results_df))
width = 0.35

ax1.bar(x - width/2, results_df['Train R²'], width, label='Train R²', alpha=0.8)
ax1.bar(x + width/2, results_df['Test R²'], width, label='Test R²', alpha=0.8)
ax1.set_xlabel('Model')
ax1.set_ylabel('R² Score')
ax1.set_title('모델별 R² 점수 비교', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(results_df['Model'], rotation=45, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# RMSE와 MAE 비교
ax2.bar(x - width/2, results_df['Test RMSE'], width, label='RMSE', alpha=0.8)
ax2.bar(x + width/2, results_df['Test MAE'], width, label='MAE', alpha=0.8)
ax2.set_xlabel('Model')
ax2.set_ylabel('Error ($10,000)')
ax2.set_title('모델별 예측 오차 비교', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(results_df['Model'], rotation=45, ha='right')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# 4. 최종 모델 선택 및 분석
print("\n4️⃣ 최종 모델 분석")
print("-" * 50)

# Ridge 모델을 최종 모델로 선택
final_model = Ridge(alpha=1.0)
final_model.fit(X_train_scaled, y_train)
final_pred = final_model.predict(X_test_scaled)

# 예측 오차 분석
errors = y_test - final_pred
error_percentages = (errors / y_test) * 100

print("\n📊 최종 모델 (Ridge) 상세 분석:")
print(f"   • 평균 오차율: {np.mean(np.abs(error_percentages)):.1f}%")
print(f"   • 10% 이내 정확도: {np.sum(np.abs(error_percentages) <= 10) / len(error_percentages) * 100:.1f}%")
print(f"   • 20% 이내 정확도: {np.sum(np.abs(error_percentages) <= 20) / len(error_percentages) * 100:.1f}%")

# 특성 중요도
feature_importance = pd.DataFrame({
    'Feature': X_enhanced.columns,
    'Coefficient': final_model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

print("\n🎯 주택 가격에 가장 큰 영향을 미치는 요인:")
for i in range(5):
    feature = feature_importance.iloc[i]
    direction = "증가" if feature['Coefficient'] > 0 else "감소"
    print(f"   {i+1}. {feature['Feature']}: 계수 {feature['Coefficient']:.3f} ({direction})")
```

## 🎯 직접 해보기

### 연습 문제 1: 간단한 선형 회귀
```python
# 학생들의 수면 시간과 집중력 데이터
sleep_hours = np.array([4, 5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 10])
concentration = np.array([40, 55, 65, 70, 75, 80, 85, 82, 78, 70])

# TODO: 선형 회귀 모델을 학습하고 시각화하세요
# 1. LinearRegression 모델 생성
# 2. 데이터 reshape 및 학습
# 3. 예측선 그리기
# 4. 7.3시간 수면 시 집중력 예측

# 여기에 코드를 작성하세요
```

### 연습 문제 2: 다항 회귀와 정규화
```python
# 온도와 아이스크림 판매량 데이터
temperature = np.array([15, 18, 20, 22, 25, 28, 30, 32, 35, 38])
sales = np.array([20, 35, 50, 80, 120, 150, 160, 155, 140, 100])

# TODO: 2차 다항 회귀와 Ridge 회귀를 비교하세요
# 1. PolynomialFeatures로 2차 특성 생성
# 2. 일반 선형 회귀와 Ridge 회귀 학습
# 3. 두 모델의 예측 곡선 비교
# 4. 어느 모델이 더 적절한지 판단

# 여기에 코드를 작성하세요
```

### 연습 문제 3: 평가 지표 계산
```python
# 실제값과 예측값
y_actual = np.array([100, 120, 140, 160, 180, 200])
y_predicted = np.array([110, 115, 145, 155, 190, 195])

# TODO: 다음 평가 지표를 직접 계산하세요
# 1. MSE (Mean Squared Error)
# 2. RMSE (Root Mean Squared Error)
# 3. MAE (Mean Absolute Error)
# 4. R² (결정계수)

# 힌트: numpy 함수를 활용하세요
# MSE = np.mean((y_actual - y_predicted) ** 2)

# 여기에 코드를 작성하세요
```

## 📚 핵심 정리

### ✅ 회귀 알고리즘의 핵심 개념
1. **회귀 vs 분류**
   - 회귀: 연속적인 수치 예측 (집값, 온도, 점수)
   - 분류: 이산적인 범주 예측 (합격/불합격, 스팸/정상)

2. **선형 회귀**
   - 최소제곱법으로 데이터에 가장 잘 맞는 직선 찾기
   - 다변량 회귀로 여러 특성 동시 고려

3. **다항 회귀**
   - 비선형 관계를 모델링할 때 사용
   - 차수가 높을수록 복잡한 패턴 포착 (과적합 주의)

4. **정규화 기법**
   - Ridge (L2): 모든 계수를 작게 만들어 과적합 방지
   - Lasso (L1): 일부 계수를 0으로 만들어 특성 선택

5. **평가 지표**
   - MSE/RMSE: 큰 오차에 민감
   - MAE: 모든 오차를 동일하게 처리
   - R²: 모델의 설명력 (1에 가까울수록 좋음)

### 💡 실무 팁
- 데이터가 선형 관계를 보이면 단순한 선형 회귀부터 시작
- 비선형 패턴이 있으면 다항 회귀나 다른 알고리즘 고려
- 특성이 많으면 정규화 기법 필수
- 여러 평가 지표를 종합적으로 고려하여 모델 선택

---

## 🚀 다음 파트에서는

**모델 평가와 검증 방법**을 학습합니다:
- 교차 검증으로 안정적인 성능 평가
- 과적합과 과소적합 진단 방법
- 편향-분산 트레이드오프 이해
- 최적의 모델 선택 전략

머신러닝의 핵심인 **"좋은 모델"**을 선택하는 체계적인 방법을 배워봅시다!
