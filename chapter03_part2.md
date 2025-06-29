# 3장 Part 2: 가설 검정의 이해

## 3.2 가설 검정의 이해

### 학습 목표
이 섹션을 완료하면 다음을 할 수 있습니다:
- 귀무가설과 대립가설의 개념을 이해하고 올바르게 설정할 수 있다
- 유의수준과 p-값의 의미를 정확히 해석할 수 있다
- 주요 통계 검정 방법(t-검정, ANOVA, 카이제곱)을 적절히 선택하고 적용할 수 있다
- 1종 오류와 2종 오류의 차이점을 이해하고 실무에서의 의미를 파악할 수 있다

### 이번 섹션 미리보기
가설 검정은 데이터를 바탕으로 어떤 주장이 참인지 거짓인지를 판단하는 통계적 방법입니다. "새로운 약이 효과가 있을까?", "두 그룹의 평균이 정말 다를까?"와 같은 질문에 답하기 위해 사용합니다.

AI 시대에도 가설 검정의 원리를 이해하는 것은 매우 중요합니다. AI가 제시하는 분석 결과가 통계적으로 유의미한지, 실무적으로 의미가 있는지 판단할 수 있어야 합니다.

### 3.2.1 귀무가설과 대립가설 설정

#### 가설 검정의 기본 개념
가설 검정은 두 개의 상반된 가설을 설정하고, 데이터 증거를 바탕으로 어느 쪽이 더 설득력 있는지 판단하는 과정입니다.

- **귀무가설(H₀)**: 기본적으로 참이라고 가정하는 가설 ("변화가 없다", "차이가 없다")
- **대립가설(H₁)**: 우리가 증명하고자 하는 가설 ("변화가 있다", "차이가 있다")

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 가설 검정 예시: 새로운 교육 방법의 효과
print("=== 가설 검정 예시: 새로운 교육 방법의 효과 ===")
print()
print("상황: 새로운 온라인 교육 방법이 기존 방법보다 효과적인지 확인하고 싶습니다.")
print("기존 교육 방법의 평균 점수: 75점")
print()
print("가설 설정:")
print("H₀ (귀무가설): 새로운 교육 방법의 평균 점수 = 75점 (효과 없음)")
print("H₁ (대립가설): 새로운 교육 방법의 평균 점수 > 75점 (효과 있음)")
print()

# 시뮬레이션 데이터 생성
np.random.seed(42)
traditional_scores = np.random.normal(75, 10, 100)  # 기존 방법
new_method_scores = np.random.normal(78, 10, 50)    # 새로운 방법 (실제로 약간 더 효과적)

print(f"기존 방법 평균: {traditional_scores.mean():.2f}점")
print(f"새로운 방법 평균: {new_method_scores.mean():.2f}점")
print(f"관찰된 차이: {new_method_scores.mean() - traditional_scores.mean():.2f}점")

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 1. 점수 분포 비교
axes[0].hist(traditional_scores, bins=20, alpha=0.7, label='기존 방법', color='lightblue', density=True)
axes[0].hist(new_method_scores, bins=20, alpha=0.7, label='새로운 방법', color='lightcoral', density=True)
axes[0].axvline(traditional_scores.mean(), color='blue', linestyle='--', linewidth=2, label=f'기존 방법 평균: {traditional_scores.mean():.1f}')
axes[0].axvline(new_method_scores.mean(), color='red', linestyle='--', linewidth=2, label=f'새로운 방법 평균: {new_method_scores.mean():.1f}')
axes[0].set_xlabel('점수')
axes[0].set_ylabel('밀도')
axes[0].set_title('교육 방법별 점수 분포')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2. 박스플롯으로 비교
data_for_plot = pd.DataFrame({
    'method': ['기존 방법'] * len(traditional_scores) + ['새로운 방법'] * len(new_method_scores),
    'score': np.concatenate([traditional_scores, new_method_scores])
})

sns.boxplot(data=data_for_plot, x='method', y='score', ax=axes[1])
axes[1].set_title('교육 방법별 점수 분포 (박스플롯)')
axes[1].set_ylabel('점수')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#### 가설 설정의 다양한 형태

print("\n=== 가설 설정의 다양한 형태 ===")

examples = [
    {
        "상황": "신약의 효과 검증",
        "H0": "신약의 완치율 = 기존약의 완치율 (차이 없음)",
        "H1": "신약의 완치율 > 기존약의 완치율 (신약이 더 효과적)",
        "검정 유형": "단측 검정 (one-tailed)"
    },
    {
        "상황": "두 그룹의 평균 비교",
        "H0": "그룹 A의 평균 = 그룹 B의 평균",
        "H1": "그룹 A의 평균 ≠ 그룹 B의 평균",
        "검정 유형": "양측 검정 (two-tailed)"
    },
    {
        "상황": "품질 기준 준수 확인",
        "H0": "불량률 ≤ 5% (기준 준수)",
        "H1": "불량률 > 5% (기준 위반)",
        "검정 유형": "단측 검정 (one-tailed)"
    }
]

for i, example in enumerate(examples, 1):
    print(f"{i}. {example['상황']}")
    print(f"   H₀: {example['H0']}")
    print(f"   H₁: {example['H1']}")
    print(f"   검정 유형: {example['검정 유형']}")
    print()

### 3.2.2 유의수준과 p-값 해석

#### 유의수준(α)의 개념
유의수준은 귀무가설이 참인데도 잘못 기각할 확률의 상한선입니다. 일반적으로 α = 0.05 (5%)를 사용합니다.

#### p-값의 정확한 의미
p-값은 "귀무가설이 참이라고 가정했을 때, 관찰된 결과 또는 그보다 극단적인 결과가 나올 확률"입니다.

# p-값 개념 시각화
def demonstrate_p_value():
    # 귀무가설: 모집단 평균 = 100
    null_mean = 100
    population_std = 15
    sample_size = 30
    
    # 관찰된 표본 평균 (예시)
    observed_mean = 107
    
    # 표준오차 계산
    standard_error = population_std / np.sqrt(sample_size)
    
    # z-점수 계산
    z_score = (observed_mean - null_mean) / standard_error
    
    # p-값 계산 (양측 검정)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    print(f"=== p-값 계산 예시 ===")
    print(f"귀무가설: 모집단 평균 = {null_mean}")
    print(f"관찰된 표본 평균: {observed_mean}")
    print(f"표본 크기: {sample_size}")
    print(f"표준오차: {standard_error:.2f}")
    print(f"z-점수: {z_score:.2f}")
    print(f"p-값: {p_value:.4f}")
    print()
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. 표준정규분포와 z-점수
    x = np.linspace(-4, 4, 1000)
    y = stats.norm.pdf(x)
    
    axes[0].plot(x, y, 'b-', linewidth=2, label='표준정규분포')
    axes[0].axvline(z_score, color='red', linestyle='--', linewidth=2, label=f'관찰된 z = {z_score:.2f}')
    axes[0].axvline(-z_score, color='red', linestyle='--', linewidth=2)
    
    # p-값 영역 음영
    x_right = x[x >= z_score]
    y_right = stats.norm.pdf(x_right)
    axes[0].fill_between(x_right, y_right, alpha=0.3, color='red', label=f'p-값/2 = {p_value/2:.4f}')
    
    x_left = x[x <= -z_score]
    y_left = stats.norm.pdf(x_left)
    axes[0].fill_between(x_left, y_left, alpha=0.3, color='red')
    
    axes[0].set_xlabel('z-점수')
    axes[0].set_ylabel('확률밀도')
    axes[0].set_title('p-값의 시각적 표현 (양측 검정)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. 표본평균의 분포
    x_means = np.linspace(90, 110, 1000)
    y_means = stats.norm.pdf(x_means, null_mean, standard_error)
    
    axes[1].plot(x_means, y_means, 'b-', linewidth=2, label='표본평균의 분포 (H₀하에서)')
    axes[1].axvline(observed_mean, color='red', linestyle='--', linewidth=2, label=f'관찰된 평균 = {observed_mean}')
    axes[1].axvline(null_mean, color='green', linestyle='-', linewidth=2, label=f'귀무가설 평균 = {null_mean}')
    
    # 극단적인 값들의 영역
    x_extreme_right = x_means[x_means >= observed_mean]
    y_extreme_right = stats.norm.pdf(x_extreme_right, null_mean, standard_error)
    axes[1].fill_between(x_extreme_right, y_extreme_right, alpha=0.3, color='red')
    
    x_extreme_left = x_means[x_means <= (null_mean - (observed_mean - null_mean))]
    y_extreme_left = stats.norm.pdf(x_extreme_left, null_mean, standard_error)
    axes[1].fill_between(x_extreme_left, y_extreme_left, alpha=0.3, color='red')
    
    axes[1].set_xlabel('표본 평균')
    axes[1].set_ylabel('확률밀도')
    axes[1].set_title('귀무가설 하에서 표본평균의 분포')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return p_value

p_value_example = demonstrate_p_value()

#### p-값 해석의 올바른 방법과 흔한 오해

print("=== p-값 해석 가이드 ===")
print()
print("✅ 올바른 해석:")
print("• p < 0.05: 귀무가설을 기각할 만한 충분한 증거가 있다")
print("• p ≥ 0.05: 귀무가설을 기각할 만한 충분한 증거가 없다")
print("• p-값은 효과의 크기가 아니라 증거의 강도를 나타낸다")
print()
print("❌ 흔한 오해:")
print("• p < 0.05이면 귀무가설이 거짓일 확률이 95%다 (잘못됨)")
print("• p-값이 작을수록 효과가 크다 (잘못됨)")
print("• p ≥ 0.05이면 귀무가설이 참이다 (잘못됨)")
print()

### 3.2.3 1종 오류와 2종 오류

#### 통계적 결정의 4가지 경우

# 오류 유형 시각화
def demonstrate_error_types():
    print("=== 가설 검정에서 발생할 수 있는 4가지 상황 ===")
    print()
    
    # 표로 정리
    situations = pd.DataFrame({
        '실제 상황': ['H₀가 참', 'H₀가 참', 'H₀가 거짓', 'H₀가 거짓'],
        '우리의 결정': ['H₀ 채택', 'H₀ 기각', 'H₀ 채택', 'H₀ 기각'],
        '결과': ['올바른 결정', '1종 오류 (α)', '2종 오류 (β)', '올바른 결정'],
        '설명': ['귀무가설이 참이고 채택함', '귀무가설이 참인데 기각함', 
                '귀무가설이 거짓인데 채택함', '귀무가설이 거짓이고 기각함']
    })
    
    print(situations.to_string(index=False))
    print()
    
    # 실제 사례로 설명
    print("=== 신약 개발 사례로 이해하기 ===")
    print("H₀: 신약은 효과가 없다")
    print("H₁: 신약은 효과가 있다")
    print()
    print("1종 오류 (α): 실제로는 효과가 없는 약을 효과가 있다고 결론")
    print("   → 결과: 무효한 약이 시장에 출시됨 (환자와 의료진이 피해)")
    print()
    print("2종 오류 (β): 실제로는 효과가 있는 약을 효과가 없다고 결론")
    print("   → 결과: 유효한 약이 개발되지 못함 (환자가 치료 기회를 잃음)")
    print()
    
    # 검정력(Power) 설명
    print("검정력(1-β): 실제로 효과가 있을 때 이를 올바르게 탐지할 확률")
    print("• 일반적으로 0.8 이상을 권장")
    print("• 표본 크기가 클수록, 효과 크기가 클수록 검정력이 증가")
    
    return situations

error_table = demonstrate_error_types()

# 1종 오류와 2종 오류의 관계 시각화
def visualize_error_relationship():
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 가상의 두 분포 (H0과 H1)
    x = np.linspace(-3, 6, 1000)
    h0_dist = stats.norm.pdf(x, 0, 1)  # 귀무가설 하의 분포
    h1_dist = stats.norm.pdf(x, 2.5, 1)  # 대립가설 하의 분포
    
    # 임계값들
    critical_values = [1.64, 1.96, 2.33]  # α = 0.05(단측), 0.05(양측), 0.01(단측)
    alpha_levels = [0.05, 0.025, 0.01]
    
    for i, (cv, alpha) in enumerate(zip(critical_values, alpha_levels)):
        row = i // 2
        col = i % 2
        
        if i < 3:
            ax = axes[row, col]
            
            # 분포 그리기
            ax.plot(x, h0_dist, 'b-', linewidth=2, label='H₀ 분포')
            ax.plot(x, h1_dist, 'r-', linewidth=2, label='H₁ 분포')
            
            # 임계값 선
            ax.axvline(cv, color='black', linestyle='--', linewidth=2, label=f'임계값 = {cv}')
            
            # 1종 오류 영역 (α)
            x_alpha = x[x >= cv]
            y_alpha = stats.norm.pdf(x_alpha, 0, 1)
            ax.fill_between(x_alpha, y_alpha, alpha=0.3, color='blue', label=f'1종 오류 (α≈{alpha:.3f})')
            
            # 2종 오류 영역 (β)
            x_beta = x[x <= cv]
            y_beta = stats.norm.pdf(x_beta, 2.5, 1)
            ax.fill_between(x_beta, y_beta, alpha=0.3, color='red', label=f'2종 오류 (β)')
            
            # 검정력 영역
            x_power = x[x >= cv]
            y_power = stats.norm.pdf(x_power, 2.5, 1)
            ax.fill_between(x_power, y_power, alpha=0.5, color='green', label='검정력 (1-β)')
            
            ax.set_xlabel('통계량 값')
            ax.set_ylabel('확률밀도')
            ax.set_title(f'임계값 = {cv} (α ≈ {alpha:.3f})')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # 네 번째 subplot: 오류 관계 요약
    axes[1,1].axis('off')
    summary_text = """
    1종 오류와 2종 오류의 관계:
    
    • 임계값을 낮추면 (왼쪽으로):
      - 1종 오류(α) 감소
      - 2종 오류(β) 증가
      - 검정력(1-β) 감소
    
    • 임계값을 높이면 (오른쪽으로):
      - 1종 오류(α) 증가  
      - 2종 오류(β) 감소
      - 검정력(1-β) 증가
    
    • 균형점 찾기:
      - 일반적으로 α = 0.05 사용
      - 검정력 ≥ 0.8 권장
      - 표본 크기 증가로 두 오류 모두 감소 가능
    """
    axes[1,1].text(0.1, 0.9, summary_text, transform=axes[1,1].transAxes, 
                   fontsize=12, verticalalignment='top', fontfamily='Malgun Gothic')
    
    plt.tight_layout()
    plt.show()

visualize_error_relationship()

### 요약 및 핵심 정리

**이번 섹션에서 배운 핵심 내용:**

1. **가설 설정**
   - 귀무가설(H₀): 현상태 유지, 차이가 없다는 가설
   - 대립가설(H₁): 변화가 있다, 차이가 있다는 가설
   - 단측 vs 양측 검정의 선택이 중요

2. **p-값의 올바른 해석**
   - p-값은 증거의 강도를 나타내는 지표
   - p < 0.05라고 해서 효과가 크다는 의미는 아님
   - 통계적 유의성과 실무적 중요성은 다름

3. **오류 유형 이해**
   - 1종 오류(α): 참인 귀무가설을 잘못 기각
   - 2종 오류(β): 거짓인 귀무가설을 잘못 채택
   - 검정력(1-β): 실제 효과를 올바르게 탐지할 확률

4. **실무적 고려사항**
   - 오류의 비용을 고려한 유의수준 설정
   - 충분한 표본 크기로 검정력 확보
   - 효과 크기와 통계적 유의성의 균형

**다음 섹션 예고:**
다음 Part 3에서는 상관관계와 인과관계의 차이점을 학습하고, 실험 설계를 통한 인과 추론 방법을 알아보겠습니다.

### 연습문제

1. **개념 확인**
   - 다음 상황에서 적절한 귀무가설과 대립가설을 설정해보세요:
     "새로운 광고 캠페인이 매출 증가에 효과가 있는지 확인하고 싶습니다."

2. **p-값 해석**
   - p-값이 0.03일 때와 0.07일 때의 올바른 해석을 설명하고, 흔한 오해를 지적해보세요.

3. **오류 유형 분석**
   - 코로나19 진단키트 개발 상황에서 1종 오류와 2종 오류의 실무적 의미를 설명하고, 어느 쪽 오류가 더 심각한지 논의해보세요.

4. **비판적 사고**
   - "통계적으로 유의미한 결과"와 "실무적으로 중요한 결과"의 차이점을 실제 사례와 함께 설명해보세요.
```