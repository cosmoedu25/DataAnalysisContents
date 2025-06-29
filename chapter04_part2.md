# 4장 Part 2: 데이터 변환과 정규화
## 머신러닝을 위한 데이터 표준화와 인코딩 전문 기법

---

## 📚 학습 목표

이번 Part에서는 다음과 같은 내용을 학습합니다:

✅ **다양한 스케일링 기법의 원리와 적용 시나리오를 이해할 수 있다**
✅ **범주형 변수를 수치형으로 변환하는 인코딩 기법을 마스터할 수 있다**  
✅ **분포 변환을 통해 데이터의 왜도를 제거하고 정규성을 향상시킬 수 있다**
✅ **scikit-learn Pipeline을 활용한 체계적인 전처리 워크플로우를 구축할 수 있다**
✅ **House Prices 데이터셋에 적합한 맞춤형 변환 파이프라인을 설계할 수 있다**

---

## 🎯 이번 Part 미리보기

결측치와 이상치를 처리한 깨끗한 데이터가 있다고 해서 바로 머신러닝 모델에 입력할 수 있는 것은 아닙니다. 데이터의 **크기(Scale)**와 **형태(Format)**를 알고리즘이 이해할 수 있도록 변환하는 과정이 필요합니다.

예를 들어, 주택 면적은 수백~수천 제곱피트 단위인 반면, 방의 개수는 1~10 정도의 작은 숫자입니다. 이런 **크기 차이**는 머신러닝 알고리즘이 큰 값을 가진 변수에만 집중하게 만들어 성능을 저하시킵니다. 마치 시끄러운 소리에 묻혀 작은 소리를 듣지 못하는 것과 같습니다.

또한 "Excellent", "Good", "Fair" 같은 **범주형 데이터**는 컴퓨터가 직접 계산할 수 없으므로 숫자로 변환해야 합니다. 하지만 단순히 1, 2, 3으로 바꾸면 "Excellent가 Fair의 3배 좋다"는 잘못된 의미를 만들 수 있습니다.

이번 Part에서는 이런 문제들을 해결하는 **전문적인 변환 기법**들을 배워보겠습니다. 특히 House Prices 데이터의 특성을 고려한 **맞춤형 전처리 파이프라인**을 구축하여 실무에서 바로 활용할 수 있는 역량을 기르겠습니다.

> **💡 Part 2의 핵심 포인트**  
> "데이터 변환은 단순한 형식 변경이 아니라 알고리즘이 데이터의 패턴을 올바르게 학습할 수 있도록 돕는 '번역' 과정입니다."

---

## 📖 4.2.1 스케일링 기법 (Scaling Techniques)

### 스케일링이 필요한 이유

머신러닝 알고리즘들은 변수 간의 **크기 차이**에 민감합니다. 크기가 큰 변수가 모델의 학습을 지배하게 되어 작은 값을 가진 중요한 변수들의 영향을 무시할 수 있습니다.

> **🔍 주요 용어 해설**
> - **스케일링(Scaling)**: 서로 다른 범위의 데이터를 동일한 범위로 조정하는 과정
> - **표준화(Standardization)**: 평균 0, 표준편차 1이 되도록 변환
> - **정규화(Normalization)**: 최솟값 0, 최댓값 1이 되도록 변환
> - **로버스트 스케일링**: 이상치에 영향을 덜 받는 중앙값 기반 스케일링

### 실제 데이터로 스케일링 필요성 확인

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# House Prices 데이터 로드 (Part 1에서 전처리된 데이터 사용 가정)
try:
    train_data = pd.read_csv('datasets/house_prices/train.csv')
    print("✅ 데이터 로드 성공!")
except FileNotFoundError:
    print("❌ 데이터 파일을 찾을 수 없습니다.")

# 스케일링 필요성 시각화
def demonstrate_scaling_need(df):
    """
    스케일링이 필요한 이유를 시각적으로 보여주는 함수
    """
    # 대표적인 수치형 변수들 선택
    key_columns = ['LotArea', 'GrLivArea', 'TotalBsmtSF', 'BedroomAbvGr', 'FullBath']
    available_columns = [col for col in key_columns if col in df.columns]
    
    if len(available_columns) < 3:
        print("⚠️ 충분한 수치형 변수가 없습니다.")
        return
    
    # 기본 통계 확인
    print("📊 주요 변수들의 크기 차이:")
    stats_df = df[available_columns].describe()
    print(stats_df)
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('🔍 스케일링 필요성 분석', fontsize=16, fontweight='bold')
    
    # 1. 원본 데이터 분포
    df[available_columns].hist(bins=30, ax=axes[0,0])
    axes[0,0].set_title('원본 데이터 분포\n(각 변수의 크기가 크게 다름)')
    
    # 2. 박스플롯으로 범위 비교
    df[available_columns].plot(kind='box', ax=axes[0,1])
    axes[0,1].set_title('변수별 값의 범위 비교\n(세로축 스케일 주목)')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. 상관관계 히트맵
    corr_matrix = df[available_columns].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', 
                center=0, ax=axes[1,0], fmt='.2f')
    axes[1,0].set_title('변수 간 상관관계')
    
    # 4. 평균과 표준편차 비교
    means = df[available_columns].mean()
    stds = df[available_columns].std()
    
    x_pos = range(len(available_columns))
    width = 0.35
    
    axes[1,1].bar([x - width/2 for x in x_pos], means, width, 
                  label='평균', alpha=0.7, color='skyblue')
    axes[1,1].bar([x + width/2 for x in x_pos], stds, width, 
                  label='표준편차', alpha=0.7, color='lightcoral')
    
    axes[1,1].set_title('평균 vs 표준편차 비교\n(크기 차이가 크면 스케일링 필요)')
    axes[1,1].set_xticks(x_pos)
    axes[1,1].set_xticklabels(available_columns, rotation=45)
    axes[1,1].legend()
    axes[1,1].set_yscale('log')  # 로그 스케일로 표시
    
    plt.tight_layout()
    plt.show()
    
    # 수치적 분석
    print(f"\n🔢 크기 차이 분석:")
    max_mean = means.max()
    min_mean = means.min()
    print(f"   최대 평균값: {max_mean:,.0f} ({means.idxmax()})")
    print(f"   최소 평균값: {min_mean:,.0f} ({means.idxmin()})")
    print(f"   평균값 차이: {max_mean/min_mean:.0f}배")
    print(f"   💡 이런 차이는 머신러닝 알고리즘의 성능을 저하시킬 수 있습니다!")

# 스케일링 필요성 시연
demonstrate_scaling_need(train_data)
```

**🔍 코드 해설:**
- `describe()`: 기술통계량(평균, 표준편차, 분위수 등)을 한 번에 계산
- `hist(bins=30)`: 히스토그램으로 데이터 분포 시각화
- `set_yscale('log')`: y축을 로그 스케일로 설정하여 큰 차이를 쉽게 비교

> **📊 이미지 생성 프롬프트:**  
> "Create a 2x2 dashboard showing the need for data scaling in machine learning. Include: 1) Histograms of different numerical variables (LotArea, GrLivArea, BedroomAbvGr) showing vastly different scales and distributions, 2) Box plots comparing the ranges of these variables side by side, 3) A correlation heatmap between the variables, 4) A bar chart comparing means vs standard deviations of variables using log scale. Use professional styling with clear labels showing the dramatic scale differences that would confuse ML algorithms."

### 방법 1: 표준화 (Standardization) - StandardScaler

표준화는 데이터를 **평균 0, 표준편차 1**로 변환하는 방법입니다. 정규분포를 따르는 데이터에 특히 효과적입니다.

**공식**: z = (x - μ) / σ

```python
# 표준화 구현 및 분석
def apply_standardization(df, columns=None):
    """
    표준화를 적용하고 결과를 분석하는 함수
    """
    if columns is None:
        # 수치형 변수 자동 선택
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'SalePrice' in columns:
            columns.remove('SalePrice')  # 타겟 변수 제외
        columns = columns[:5]  # 처음 5개만 사용
    
    print("📊 표준화(StandardScaler) 적용:")
    
    # 원본 데이터 통계
    original_stats = df[columns].describe()
    print(f"\n🔸 원본 데이터 통계:")
    print(original_stats.loc[['mean', 'std']].round(2))
    
    # 표준화 적용
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[columns])
    scaled_df = pd.DataFrame(scaled_data, columns=columns, index=df.index)
    
    # 변환 후 통계
    scaled_stats = scaled_df.describe()
    print(f"\n🔸 표준화 후 통계:")
    print(scaled_stats.loc[['mean', 'std']].round(6))
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('📊 표준화(StandardScaler) 효과 분석', fontsize=16, fontweight='bold')
    
    # 1. 원본 vs 변환 후 분포 비교 (첫 번째 변수)
    first_col = columns[0]
    axes[0,0].hist(df[first_col].dropna(), bins=30, alpha=0.7, 
                   label='원본', color='skyblue', density=True)
    axes[0,0].hist(scaled_df[first_col], bins=30, alpha=0.7, 
                   label='표준화 후', color='lightcoral', density=True)
    axes[0,0].set_title(f'{first_col} 분포 변화')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. 모든 변수의 표준화 후 분포
    scaled_df.hist(bins=30, ax=axes[0,1])
    axes[0,1].set_title('표준화 후 모든 변수 분포\n(평균 0, 표준편차 1 근처)')
    
    # 3. 원본 데이터 박스플롯
    df[columns].plot(kind='box', ax=axes[1,0])
    axes[1,0].set_title('원본 데이터 (크기 차이 큼)')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # 4. 표준화 후 박스플롯
    scaled_df.plot(kind='box', ax=axes[1,1])
    axes[1,1].set_title('표준화 후 (동일한 스케일)')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n✅ 표준화 완료!")
    print(f"   평균: 모두 0에 가까움 (목표: 0)")
    print(f"   표준편차: 모두 1에 가까움 (목표: 1)")
    print(f"   💡 이제 모든 변수가 동일한 스케일을 가집니다!")
    
    return scaled_df, scaler

# 표준화 적용
standardized_data, standard_scaler = apply_standardization(train_data)
```

**🔍 코드 해설:**
- `StandardScaler()`: 표준화를 수행하는 scikit-learn 클래스
- `fit_transform()`: 변환 규칙을 학습하고 동시에 적용
- `density=True`: 히스토그램을 확률밀도로 표시하여 다른 스케일의 분포를 비교 가능

### 방법 2: 정규화 (Normalization) - MinMaxScaler

정규화는 데이터를 **0과 1 사이**로 변환하는 방법입니다. 최솟값과 최댓값이 명확한 데이터에 적합합니다.

**공식**: x_scaled = (x - min) / (max - min)

```python
# 정규화 구현 및 분석
def apply_normalization(df, columns=None):
    """
    정규화를 적용하고 결과를 분석하는 함수
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'SalePrice' in columns:
            columns.remove('SalePrice')
        columns = columns[:5]
    
    print("📊 정규화(MinMaxScaler) 적용:")
    
    # 원본 데이터 범위
    print(f"\n🔸 원본 데이터 범위:")
    for col in columns:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            range_val = max_val - min_val
            print(f"   {col}: {min_val:.1f} ~ {max_val:.1f} (범위: {range_val:.1f})")
    
    # 정규화 적용
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df[columns])
    normalized_df = pd.DataFrame(normalized_data, columns=columns, index=df.index)
    
    # 변환 후 범위
    print(f"\n🔸 정규화 후 범위:")
    for col in columns:
        min_val = normalized_df[col].min()
        max_val = normalized_df[col].max()
        print(f"   {col}: {min_val:.3f} ~ {max_val:.3f}")
    
    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('📊 정규화(MinMaxScaler) 효과 분석', fontsize=16, fontweight='bold')
    
    # 1. 원본 데이터 범위
    ranges_original = [df[col].max() - df[col].min() for col in columns]
    axes[0].bar(columns, ranges_original, color='skyblue', alpha=0.7)
    axes[0].set_title('원본 데이터 범위')
    axes[0].set_ylabel('범위 크기')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].set_yscale('log')
    
    # 2. 정규화 후 범위 (모두 1)
    ranges_normalized = [1.0] * len(columns)  # 모두 0~1 범위
    axes[1].bar(columns, ranges_normalized, color='lightcoral', alpha=0.7)
    axes[1].set_title('정규화 후 범위 (모두 0~1)')
    axes[1].set_ylabel('범위 크기')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].set_ylim(0, 1.2)
    
    # 3. 정규화 후 데이터 분포
    normalized_df.plot(kind='box', ax=axes[2])
    axes[2].set_title('정규화 후 분포 (0~1 범위)')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n✅ 정규화 완료!")
    print(f"   모든 변수가 0~1 범위로 변환되었습니다!")
    print(f"   💡 최솟값 0, 최댓값 1이 명확한 의미를 가집니다!")
    
    return normalized_df, scaler

# 정규화 적용
normalized_data, minmax_scaler = apply_normalization(train_data)
```

**🔍 코드 해설:**
- `MinMaxScaler()`: 0~1 범위로 정규화하는 클래스
- `set_yscale('log')`: 로그 스케일로 y축을 설정하여 큰 차이를 명확히 표시
- `set_ylim()`: y축 범위를 수동으로 설정

### 방법 3: 로버스트 스케일링 (Robust Scaling) - RobustScaler

로버스트 스케일링은 **중앙값과 IQR**을 사용하여 이상치의 영향을 최소화하는 방법입니다.

**공식**: x_scaled = (x - median) / IQR

```python
# 로버스트 스케일링 구현 및 분석
def apply_robust_scaling(df, columns=None):
    """
    로버스트 스케일링을 적용하고 이상치 처리 효과를 분석하는 함수
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'SalePrice' in columns:
            columns.remove('SalePrice')
        columns = columns[:5]
    
    print("📊 로버스트 스케일링(RobustScaler) 적용:")
    
    # 이상치가 포함된 데이터 생성 (시연 목적)
    df_with_outliers = df.copy()
    
    # 첫 번째 변수에 인위적으로 극단값 추가
    if len(columns) > 0:
        first_col = columns[0]
        if first_col in df.columns:
            # 상위 1%에 매우 큰 값 추가
            extreme_value = df[first_col].quantile(0.99) * 5
            outlier_indices = df.sample(n=5).index  # 5개 데이터포인트에 극단값 추가
            df_with_outliers.loc[outlier_indices, first_col] = extreme_value
            
            print(f"\n🔸 {first_col}에 극단값 추가 (시연용):")
            print(f"   극단값: {extreme_value:.0f}")
            print(f"   영향받는 데이터: {len(outlier_indices)}개")
    
    # 3가지 스케일링 방법 비교
    scalers = {
        'Standard': StandardScaler(),
        'MinMax': MinMaxScaler(), 
        'Robust': RobustScaler()
    }
    
    scaled_results = {}
    
    for name, scaler in scalers.items():
        scaled_data = scaler.fit_transform(df_with_outliers[columns])
        scaled_results[name] = pd.DataFrame(scaled_data, columns=columns)
    
    # 첫 번째 변수로 비교 분석
    first_col = columns[0]
    
    print(f"\n🔸 {first_col} 변수의 스케일링 방법별 비교:")
    for name, scaled_df in scaled_results.items():
        q1, q3 = scaled_df[first_col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        outliers = scaled_df[first_col][(scaled_df[first_col] < lower_fence) | 
                                        (scaled_df[first_col] > upper_fence)]
        
        print(f"   {name:8}: 이상치 {len(outliers)}개, 범위 [{scaled_df[first_col].min():.2f}, {scaled_df[first_col].max():.2f}]")
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('📊 로버스트 스케일링 vs 다른 방법들', fontsize=16, fontweight='bold')
    
    # 1. 원본 데이터 (이상치 포함)
    axes[0,0].hist(df_with_outliers[first_col].dropna(), bins=50, 
                   alpha=0.7, color='gray', label='이상치 포함')
    axes[0,0].hist(df[first_col].dropna(), bins=50, 
                   alpha=0.7, color='skyblue', label='원본')
    axes[0,0].set_title(f'{first_col} 원본 vs 이상치 포함')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. 표준화 결과
    axes[0,1].hist(scaled_results['Standard'][first_col], bins=50, 
                   alpha=0.7, color='lightcoral')
    axes[0,1].set_title('표준화 결과\n(이상치에 민감)')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. MinMax 정규화 결과
    axes[1,0].hist(scaled_results['MinMax'][first_col], bins=50, 
                   alpha=0.7, color='lightgreen')
    axes[1,0].set_title('MinMax 정규화 결과\n(이상치에 매우 민감)')
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. 로버스트 스케일링 결과
    axes[1,1].hist(scaled_results['Robust'][first_col], bins=50, 
                   alpha=0.7, color='orange')
    axes[1,1].set_title('로버스트 스케일링 결과\n(이상치에 덜 민감)')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n✅ 로버스트 스케일링의 장점:")
    print(f"   🔹 이상치의 영향을 최소화")
    print(f"   🔹 중앙값과 IQR 사용으로 안정적")
    print(f"   🔹 이상치가 많은 실제 데이터에 적합")
    
    return scaled_results['Robust'], RobustScaler().fit(df[columns])

# 로버스트 스케일링 적용
robust_scaled_data, robust_scaler = apply_robust_scaling(train_data)
```

**🔍 코드 해설:**
- `RobustScaler()`: 중앙값과 IQR을 사용하는 로버스트 스케일링
- `quantile([0.25, 0.75])`: 제1사분위수와 제3사분위수를 동시에 계산
- `sample(n=5)`: 무작위로 5개 행을 선택하여 극단값 추가

### 스케일링 방법 선택 가이드

```python
# 스케일링 방법 선택 가이드
def scaling_method_guide():
    """
    어떤 상황에서 어떤 스케일링 방법을 사용할지 가이드 제공
    """
    print("🎯 스케일링 방법 선택 가이드:")
    print()
    
    guide_data = {
        '방법': ['StandardScaler', 'MinMaxScaler', 'RobustScaler'],
        '적용 시나리오': [
            '정규분포에 가까운 데이터\n신경망, SVM 등',
            '0~1 범위가 의미있는 경우\n이미지 데이터, 확률값',
            '이상치가 많은 데이터\n실제 비즈니스 데이터'
        ],
        '장점': [
            '• 정규분포 가정 알고리즘에 최적\n• 평균 0, 표준편차 1로 해석 용이',
            '• 명확한 0~1 범위\n• 이해하기 쉬움',
            '• 이상치에 강건\n• 실제 데이터에 안정적'
        ],
        '단점': [
            '• 이상치에 민감\n• 정규분포 아닐 때 효과 제한',
            '• 이상치에 매우 민감\n• 새로운 데이터가 범위 벗어날 수 있음',
            '• 해석이 상대적으로 어려움\n• 특정 알고리즘에는 부적합'
        ]
    }
    
    guide_df = pd.DataFrame(guide_data)
    
    for i, row in guide_df.iterrows():
        print(f"📊 {row['방법']}")
        print(f"   🎯 사용 시나리오: {row['적용 시나리오']}")
        print(f"   ✅ 장점: {row['장점']}")
        print(f"   ⚠️ 단점: {row['단점']}")
        print()
    
    print("💡 실무 팁:")
    print("   1️⃣ 먼저 RobustScaler로 시작해보세요 (이상치 많은 실제 데이터)")
    print("   2️⃣ 신경망 사용시 StandardScaler 고려")
    print("   3️⃣ 해석이 중요하면 MinMaxScaler 사용")
    print("   4️⃣ 여러 방법을 실험해보고 성능 비교!")

# 가이드 출력
scaling_method_guide()
```

이제 실제 House Prices 데이터에 적합한 스케일링 방법을 선택해보겠습니다.

```python
# House Prices 데이터 특성 분석 및 스케일링 방법 추천
def recommend_scaling_for_house_data(df):
    """
    House Prices 데이터의 특성을 분석하여 최적 스케일링 방법 추천
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'SalePrice' in numeric_cols:
        numeric_cols.remove('SalePrice')
    
    print("🏡 House Prices 데이터 스케일링 방법 추천:")
    print()
    
    recommendations = {}
    
    for col in numeric_cols[:8]:  # 주요 8개 변수 분석
        if col in df.columns:
            data = df[col].dropna()
            
            # 기본 통계
            mean_val = data.mean()
            median_val = data.median()
            std_val = data.std()
            q1, q3 = data.quantile([0.25, 0.75])
            iqr = q3 - q1
            
            # 이상치 비율
            lower_fence = q1 - 1.5 * iqr
            upper_fence = q3 + 1.5 * iqr
            outlier_ratio = len(data[(data < lower_fence) | (data > upper_fence)]) / len(data)
            
            # 왜도 계산
            from scipy.stats import skew
            skewness = skew(data)
            
            # 추천 로직
            if outlier_ratio > 0.05:  # 이상치 5% 이상
                if abs(skewness) > 1:  # 심한 왜도
                    recommendation = "RobustScaler + 로그변환"
                else:
                    recommendation = "RobustScaler"
            elif abs(skewness) > 1:
                recommendation = "StandardScaler + 로그변환"
            elif (data >= 0).all() and data.max() > 1000:  # 큰 양수값
                recommendation = "MinMaxScaler 또는 StandardScaler"
            else:
                recommendation = "StandardScaler"
            
            recommendations[col] = {
                'method': recommendation,
                'outlier_ratio': outlier_ratio,
                'skewness': skewness,
                'mean': mean_val,
                'median': median_val
            }
            
            print(f"📊 {col}:")
            print(f"   이상치 비율: {outlier_ratio:.2%}")
            print(f"   왜도: {skewness:.2f}")
            print(f"   추천 방법: {recommendation}")
            print()
    
    return recommendations

# House Prices 데이터 스케일링 추천
scaling_recommendations = recommend_scaling_for_house_data(train_data)

---

## 📖 4.2.2 범주형 변수 인코딩 (Categorical Variable Encoding)

### 범주형 변수의 이해

**범주형 변수**는 카테고리나 그룹을 나타내는 데이터로, 머신러닝 알고리즘이 직접 처리할 수 없어 수치형으로 변환해야 합니다. 하지만 단순히 숫자를 할당하면 잘못된 순서나 거리 관계를 만들 수 있습니다.

> **🔍 주요 용어 해설**
> - **명목형(Nominal)**: 순서가 없는 범주 (색깔, 지역 등)
> - **순서형(Ordinal)**: 순서가 있는 범주 (등급, 크기 등)
> - **원-핫 인코딩**: 각 범주를 별도 이진 변수로 변환
> - **레이블 인코딩**: 범주를 연속된 정수로 변환
> - **타겟 인코딩**: 타겟 변수의 평균값으로 범주를 인코딩

### House Prices 데이터의 범주형 변수 분석

```python
# 범주형 변수 분석 함수
def analyze_categorical_variables(df):
    """
    데이터셋의 범주형 변수를 분석하고 인코딩 전략 수립
    """
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    print(f"🏷️ 범주형 변수 분석 ({len(categorical_cols)}개 변수):")
    print()
    
    # 변수별 상세 분석
    for col in categorical_cols[:10]:  # 처음 10개 변수만 분석
        if col in df.columns:
            unique_count = df[col].nunique()
            missing_count = df[col].isnull().sum()
            missing_ratio = missing_count / len(df)
            
            print(f"📊 {col}:")
            print(f"   고유값 수: {unique_count}개")
            print(f"   결측치: {missing_count}개 ({missing_ratio:.1%})")
            
            # 상위 빈도값 표시
            value_counts = df[col].value_counts().head(5)
            print(f"   상위 5개 값:")
            for value, count in value_counts.items():
                percentage = count / len(df) * 100
                print(f"      {value}: {count}개 ({percentage:.1f}%)")
            
            # 인코딩 방법 추천
            if unique_count <= 5:
                encoding_method = "원-핫 인코딩"
            elif unique_count <= 20:
                encoding_method = "원-핫 인코딩 또는 타겟 인코딩"
            else:
                encoding_method = "타겟 인코딩 또는 빈도 인코딩"
            
            print(f"   💡 추천 인코딩: {encoding_method}")
            print()
    
    # 카디널리티별 분류
    low_cardinality = [col for col in categorical_cols if df[col].nunique() <= 5]
    medium_cardinality = [col for col in categorical_cols if 5 < df[col].nunique() <= 20]
    high_cardinality = [col for col in categorical_cols if df[col].nunique() > 20]
    
    print(f"📈 카디널리티별 분류:")
    print(f"   🟢 낮음 (≤5개): {len(low_cardinality)}개 변수")
    print(f"   🟡 중간 (6-20개): {len(medium_cardinality)}개 변수") 
    print(f"   🔴 높음 (>20개): {len(high_cardinality)}개 변수")
    
    return {
        'low_cardinality': low_cardinality,
        'medium_cardinality': medium_cardinality,
        'high_cardinality': high_cardinality
    }

# 범주형 변수 분석 실행
categorical_analysis = analyze_categorical_variables(train_data)
```

**🔍 코드 해설:**
- `nunique()`: 결측치를 제외한 고유값의 개수를 계산
- `value_counts()`: 각 값의 빈도를 계산하여 내림차순으로 정렬
- 카디널리티에 따라 다른 인코딩 전략을 추천

### 방법 1: 원-핫 인코딩 (One-Hot Encoding)

각 범주를 별도의 이진(0/1) 변수로 변환하는 방법입니다. 순서가 없는 명목형 변수에 적합합니다.

```python
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# 원-핫 인코딩 구현
def apply_onehot_encoding(df, columns=None, max_categories=10):
    """
    원-핫 인코딩을 적용하고 결과를 분석하는 함수
    """
    if columns is None:
        # 낮은 카디널리티 변수 선택
        categorical_cols = df.select_dtypes(include=['object']).columns
        columns = [col for col in categorical_cols if df[col].nunique() <= max_categories][:3]
    
    print(f"🎯 원-핫 인코딩 적용 ({len(columns)}개 변수):")
    
    original_shape = df.shape
    encoded_dfs = []
    column_info = {}
    
    for col in columns:
        if col in df.columns:
            print(f"\n📊 {col} 변수:")
            
            # 원본 정보
            unique_values = df[col].dropna().unique()
            print(f"   원본 범주: {list(unique_values)}")
            print(f"   범주 수: {len(unique_values)}개")
            
            # 원-핫 인코딩 적용
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            
            # 결측치 처리 (임시로 'Unknown'으로 대체)
            col_data = df[col].fillna('Unknown').values.reshape(-1, 1)
            encoded_array = encoder.fit_transform(col_data)
            
            # 컬럼명 생성
            feature_names = [f"{col}_{category}" for category in encoder.categories_[0]]
            encoded_df = pd.DataFrame(encoded_array, columns=feature_names, index=df.index)
            
            encoded_dfs.append(encoded_df)
            column_info[col] = {
                'original_categories': list(unique_values),
                'encoded_columns': feature_names,
                'encoder': encoder
            }
            
            print(f"   생성된 컬럼: {len(feature_names)}개")
            print(f"   컬럼명: {feature_names}")
    
    # 결과 결합
    if encoded_dfs:
        final_encoded_df = pd.concat(encoded_dfs, axis=1)
        
        print(f"\n📈 인코딩 결과 요약:")
        print(f"   원본 범주형 변수: {len(columns)}개")
        print(f"   생성된 이진 변수: {final_encoded_df.shape[1]}개")
        print(f"   데이터 크기 변화: {original_shape} → {(df.shape[0], df.shape[1] - len(columns) + final_encoded_df.shape[1])}")
        
        # 샘플 데이터 시각화
        print(f"\n🔍 첫 번째 변수 인코딩 예시:")
        first_col = columns[0]
        print(f"   원본 {first_col} 값들:")
        sample_indices = df.sample(5).index
        for idx in sample_indices:
            original_value = df.loc[idx, first_col]
            encoded_values = final_encoded_df.loc[idx, [col for col in final_encoded_df.columns if col.startswith(first_col)]]
            active_column = encoded_values[encoded_values == 1]
            if len(active_column) > 0:
                print(f"      '{original_value}' → {active_column.index[0]} = 1")
            else:
                print(f"      '{original_value}' → 모든 컬럼 = 0 (결측치)")
        
        return final_encoded_df, column_info
    else:
        print("⚠️ 인코딩할 범주형 변수가 없습니다.")
        return None, None

# 원-핫 인코딩 적용
onehot_encoded_data, onehot_info = apply_onehot_encoding(train_data)
```

**🔍 코드 해설:**
- `OneHotEncoder(sparse_output=False)`: 희소행렬 대신 일반 배열로 출력
- `handle_unknown='ignore'`: 새로운 범주가 나타나면 모든 값을 0으로 설정
- `reshape(-1, 1)`: 1차원 배열을 2차원으로 변환 (scikit-learn 요구사항)

### 방법 2: 순서형 인코딩 (Ordinal Encoding)

순서가 있는 범주형 변수를 순서를 유지하면서 수치로 변환하는 방법입니다.

```python
from sklearn.preprocessing import OrdinalEncoder

# 순서형 인코딩 구현
def apply_ordinal_encoding(df):
    """
    House Prices 데이터의 품질 관련 변수들에 순서형 인코딩 적용
    """
    print("📊 순서형 인코딩 적용:")
    
    # House Prices 데이터의 품질 관련 변수들 (순서가 있음)
    quality_columns = {
        'ExterQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],  # Poor → Excellent
        'ExterCond': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'BsmtQual': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'BsmtCond': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'HeatingQC': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'KitchenQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'FireplaceQu': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex']
    }
    
    # 실제 존재하는 컬럼만 선택
    available_quality_cols = {col: order for col, order in quality_columns.items() 
                             if col in df.columns}
    
    encoded_df = df.copy()
    encoding_info = {}
    
    for col, order_list in available_quality_cols.items():
        print(f"\n🏷️ {col} 인코딩:")
        
        # 원본 데이터 분포 확인
        original_counts = df[col].value_counts()
        print(f"   원본 분포: {dict(original_counts)}")
        
        # 결측치를 'None'으로 대체
        col_data = df[col].fillna('None')
        
        # 순서형 인코딩 적용
        encoder = OrdinalEncoder(categories=[order_list])
        
        try:
            encoded_values = encoder.fit_transform(col_data.values.reshape(-1, 1)).flatten()
            encoded_df[col + '_Encoded'] = encoded_values
            
            # 인코딩 결과 확인
            encoding_map = {category: i for i, category in enumerate(order_list)}
            print(f"   인코딩 맵핑: {encoding_map}")
            
            # 통계 정보
            print(f"   인코딩된 값 범위: {encoded_values.min():.0f} ~ {encoded_values.max():.0f}")
            print(f"   평균 품질 점수: {encoded_values.mean():.2f}")
            
            encoding_info[col] = {
                'order': order_list,
                'mapping': encoding_map,
                'encoder': encoder
            }
            
        except ValueError as e:
            print(f"   ⚠️ 인코딩 실패: {e}")
            print(f"   데이터에 정의되지 않은 범주가 있을 수 있습니다.")
    
    # 품질 점수들 간의 상관관계 분석
    encoded_quality_cols = [col + '_Encoded' for col in available_quality_cols.keys()]
    existing_encoded_cols = [col for col in encoded_quality_cols if col in encoded_df.columns]
    
    if len(existing_encoded_cols) > 1:
        correlation_matrix = encoded_df[existing_encoded_cols].corr()
        
        print(f"\n📊 품질 변수들 간 상관관계:")
        print(correlation_matrix.round(3))
        
        # 시각화
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
                   center=0, square=True, fmt='.3f')
        plt.title('품질 변수들의 순서형 인코딩 후 상관관계')
        plt.tight_layout()
        plt.show()
    
    print(f"\n✅ 순서형 인코딩 완료!")
    print(f"   처리된 변수: {len(encoding_info)}개")
    print(f"   💡 순서가 있는 품질 정보가 수치로 변환되어 모델이 패턴을 학습하기 쉬워졌습니다!")
    
    return encoded_df, encoding_info

# 순서형 인코딩 적용
ordinal_encoded_data, ordinal_info = apply_ordinal_encoding(train_data)
```

**🔍 코드 해설:**
- `OrdinalEncoder(categories=[order_list])`: 미리 정의된 순서로 인코딩
- `flatten()`: 2차원 배열을 1차원으로 변환
- 품질 변수들 간의 상관관계 분석으로 인코딩 결과 검증

### 방법 3: 타겟 인코딩 (Target Encoding)

범주형 변수의 각 범주를 해당 범주에서의 타겟 변수 평균값으로 인코딩하는 방법입니다. 높은 카디널리티 변수에 특히 유용합니다.

```python
# 타겟 인코딩 구현
def apply_target_encoding(df, target_col='SalePrice', categorical_cols=None, smoothing=1.0):
    """
    타겟 인코딩을 적용하는 함수 (과적합 방지 기법 포함)
    
    Parameters:
    smoothing (float): 스무딩 파라미터 (과적합 방지)
    """
    if target_col not in df.columns:
        print(f"⚠️ 타겟 변수 '{target_col}'이 데이터에 없습니다.")
        return df, {}
    
    if categorical_cols is None:
        # 중간~높은 카디널리티 변수 선택
        categorical_all = df.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_all 
                           if 5 < df[col].nunique() <= 50][:3]  # 예시로 3개만
    
    print(f"🎯 타겟 인코딩 적용 ({len(categorical_cols)}개 변수):")
    print(f"   타겟 변수: {target_col}")
    print(f"   스무딩 파라미터: {smoothing}")
    
    encoded_df = df.copy()
    encoding_info = {}
    
    # 전체 타겟 평균 (스무딩에 사용)
    global_mean = df[target_col].mean()
    
    for col in categorical_cols:
        if col in df.columns:
            print(f"\n📊 {col} 타겟 인코딩:")
            
            # 범주별 타겟 통계 계산
            category_stats = df.groupby(col)[target_col].agg(['count', 'mean']).reset_index()
            category_stats.columns = [col, 'count', 'target_mean']
            
            # 스무딩 적용 (과적합 방지)
            # 공식: (count * category_mean + smoothing * global_mean) / (count + smoothing)
            category_stats['smoothed_mean'] = (
                (category_stats['count'] * category_stats['target_mean'] + 
                 smoothing * global_mean) / 
                (category_stats['count'] + smoothing)
            )
            
            print(f"   범주별 타겟 인코딩 결과:")
            for _, row in category_stats.head().iterrows():
                print(f"      {row[col]}: {row['target_mean']:.0f} → {row['smoothed_mean']:.0f} (샘플 {row['count']}개)")
            
            # 인코딩 적용
            encoding_map = dict(zip(category_stats[col], category_stats['smoothed_mean']))
            encoded_df[col + '_TargetEnc'] = df[col].map(encoding_map)
            
            # 결측치 처리 (전체 평균으로)
            encoded_df[col + '_TargetEnc'] = encoded_df[col + '_TargetEnc'].fillna(global_mean)
            
            # 정보 저장
            encoding_info[col] = {
                'encoding_map': encoding_map,
                'global_mean': global_mean,
                'category_stats': category_stats
            }
            
            # 효과 분석
            original_correlation = df[col].astype('category').cat.codes.corr(df[target_col])
            encoded_correlation = encoded_df[col + '_TargetEnc'].corr(df[target_col])
            
            print(f"   상관관계 변화: {original_correlation:.3f} → {encoded_correlation:.3f}")
            
            # 시각화 (첫 번째 변수만)
            if col == categorical_cols[0]:
                fig, axes = plt.subplots(1, 2, figsize=(15, 5))
                
                # 원본 범주별 타겟 분포
                df.boxplot(column=target_col, by=col, ax=axes[0])
                axes[0].set_title(f'{col}별 {target_col} 분포 (원본)')
                axes[0].tick_params(axis='x', rotation=45)
                
                # 타겟 인코딩 결과 vs 실제 타겟
                axes[1].scatter(encoded_df[col + '_TargetEnc'], df[target_col], alpha=0.6)
                axes[1].set_xlabel(f'{col} 타겟 인코딩 값')
                axes[1].set_ylabel(f'실제 {target_col}')
                axes[1].set_title(f'타겟 인코딩 효과 (상관계수: {encoded_correlation:.3f})')
                
                plt.tight_layout()
                plt.show()
    
    print(f"\n✅ 타겟 인코딩 완료!")
    print(f"   처리된 변수: {len(encoding_info)}개")
    print(f"   💡 범주별 타겟 평균이 직접적인 예측 정보로 활용됩니다!")
    
    return encoded_df, encoding_info

# 타겟 인코딩 적용
target_encoded_data, target_info = apply_target_encoding(train_data)
```

**🔍 코드 해설:**
- 스무딩 기법으로 과적합 방지: 샘플이 적은 범주는 전체 평균에 가까워짐
- `groupby().agg(['count', 'mean'])`: 범주별로 개수와 평균을 동시에 계산
- `astype('category').cat.codes`: 범주형 데이터를 임시로 숫자 코드로 변환하여 상관계수 계산

> **📊 이미지 생성 프롬프트:**  
> "Create a comprehensive comparison visualization of categorical encoding methods showing: 1) A bar chart comparing the number of features created by One-Hot vs Ordinal vs Target encoding, 2) A correlation heatmap showing relationships between ordinal-encoded quality variables (ExterQual, BsmtQual, KitchenQual etc.), 3) A scatter plot showing target encoding effectiveness with actual SalePrice vs encoded values, 4) A box plot showing how different categories relate to target variable before encoding. Use professional styling with clear legends and annotations."

### 인코딩 방법 비교 및 선택 가이드

```python
# 인코딩 방법 종합 비교
def compare_encoding_methods(df, target_col='SalePrice'):
    """
    다양한 인코딩 방법의 효과를 비교 분석
    """
    print("📊 인코딩 방법 종합 비교:")
    
    comparison_results = {}
    
    # 1. 데이터 크기 비교
    original_shape = df.shape
    
    # 원-핫 인코딩 시뮬레이션
    categorical_cols = df.select_dtypes(include=['object']).columns
    low_card_cols = [col for col in categorical_cols if df[col].nunique() <= 10]
    
    if low_card_cols:
        total_onehot_features = sum(df[col].nunique() for col in low_card_cols)
        onehot_shape = (original_shape[0], original_shape[1] - len(low_card_cols) + total_onehot_features)
    else:
        onehot_shape = original_shape
    
    # 순서형/타겟 인코딩은 크기 변화 없음
    ordinal_shape = original_shape
    target_shape = original_shape
    
    print(f"\n📈 데이터 크기 변화:")
    print(f"   원본: {original_shape}")
    print(f"   원-핫 인코딩 후: {onehot_shape} (+{onehot_shape[1] - original_shape[1]}개 컬럼)")
    print(f"   순서형 인코딩 후: {ordinal_shape} (변화 없음)")
    print(f"   타겟 인코딩 후: {target_shape} (변화 없음)")
    
    # 2. 장단점 비교표
    methods_comparison = {
        '방법': ['원-핫 인코딩', '순서형 인코딩', '타겟 인코딩'],
        '적용 대상': [
            '명목형 변수\n(낮은 카디널리티)',
            '순서형 변수\n(품질, 등급)',
            '모든 범주형 변수\n(높은 카디널리티)'
        ],
        '장점': [
            '• 순서 관계 없음\n• 해석 명확\n• 안정적',
            '• 순서 정보 보존\n• 효율적\n• 직관적',
            '• 예측력 높음\n• 차원 증가 없음\n• 높은 카디널리티 처리'
        ],
        '단점': [
            '• 차원 폭발\n• 희소성 문제\n• 높은 카디널리티 부적합',
            '• 순서 정보 필요\n• 명목형에 부적합\n• 임의적 순서 위험',
            '• 과적합 위험\n• 데이터 누수 가능\n• 교차검증 복잡'
        ],
        '사용 시나리오': [
            'Neighborhood, GarageType\n등 지역/유형 변수',
            'ExterQual, BsmtQual\n등 품질 변수',
            'Neighborhood (높은 카디널리티)\n새로운 범주 많은 경우'
        ]
    }
    
    comparison_df = pd.DataFrame(methods_comparison)
    
    print(f"\n📋 인코딩 방법별 상세 비교:")
    for i, row in comparison_df.iterrows():
        print(f"\n🔹 {row['방법']}")
        print(f"   적용 대상: {row['적용 대상']}")
        print(f"   ✅ 장점: {row['장점']}")
        print(f"   ⚠️ 단점: {row['단점']}")
        print(f"   🎯 사용 예: {row['사용 시나리오']}")
    
    # 3. 실무 선택 가이드
    print(f"\n💡 실무 인코딩 선택 가이드:")
    print(f"   1️⃣ 범주 수 ≤ 10개 + 명목형 → 원-핫 인코딩")
    print(f"   2️⃣ 순서가 있는 범주 → 순서형 인코딩") 
    print(f"   3️⃣ 범주 수 > 10개 → 타겟 인코딩 (과적합 주의)")
    print(f"   4️⃣ 불확실하면 → 여러 방법 실험 후 성능 비교")
    print(f"   5️⃣ 새로운 범주 많으면 → 타겟 인코딩 + 스무딩")

# 인코딩 방법 비교 실행
compare_encoding_methods(train_data)

---

## 📖 4.2.3 비선형 변환과 분포 정규화

### 왜도와 분포의 중요성

많은 머신러닝 알고리즘은 데이터가 **정규분포**에 가깝다고 가정합니다. 하지만 실제 데이터는 종종 **왜도(Skewness)**를 가지며, 이는 모델 성능을 저하시킬 수 있습니다.

> **🔍 주요 용어 해설**
> - **왜도(Skewness)**: 분포의 비대칭 정도 (0에 가까울수록 대칭적)
> - **첨도(Kurtosis)**: 분포의 뾰족한 정도
> - **로그 변환**: 양의 왜도를 줄이는 대표적인 변환 방법
> - **Box-Cox 변환**: 최적의 변환 파라미터를 자동으로 찾는 방법

### 분포 분석 및 변환 필요성 확인

```python
from scipy import stats
from scipy.stats import boxcox, yeojohnson
import numpy as np

# 분포 분석 함수
def analyze_distribution(df, columns=None):
    """
    변수들의 분포를 분석하고 변환 필요성을 판단하는 함수
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'SalePrice' in numeric_cols:
            numeric_cols.remove('SalePrice')
        columns = numeric_cols[:6]  # 처음 6개 변수
    
    print("📊 분포 분석 및 변환 필요성 평가:")
    print()
    
    distribution_info = {}
    
    for col in columns:
        if col in df.columns:
            data = df[col].dropna()
            
            # 기본 통계
            skewness = stats.skew(data)
            kurtosis = stats.kurtosis(data)
            
            # Shapiro-Wilk 정규성 검정 (샘플 크기 제한)
            if len(data) <= 5000:
                shapiro_stat, shapiro_p = stats.shapiro(data.sample(min(len(data), 1000)))
            else:
                shapiro_stat, shapiro_p = stats.normaltest(data.sample(1000))
            
            # 변환 필요성 판단
            needs_transform = abs(skewness) > 1.0  # 왜도 절댓값이 1 이상
            
            distribution_info[col] = {
                'skewness': skewness,
                'kurtosis': kurtosis,
                'shapiro_p': shapiro_p,
                'needs_transform': needs_transform
            }
            
            print(f"📈 {col}:")
            print(f"   왜도: {skewness:.3f} {'(변환 필요)' if abs(skewness) > 1 else '(양호)'}")
            print(f"   첨도: {kurtosis:.3f}")
            print(f"   정규성 검정 p-값: {shapiro_p:.3e}")
            print(f"   💡 변환 권장: {'예' if needs_transform else '아니오'}")
            print()
    
    # 시각화
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('📊 변수별 분포 분석', fontsize=16, fontweight='bold')
    
    for i, col in enumerate(columns[:6]):
        if i >= 6:
            break
            
        row = i // 3
        col_idx = i % 3
        
        if col in df.columns:
            data = df[col].dropna()
            
            # 히스토그램 + 정규분포 곡선
            axes[row, col_idx].hist(data, bins=50, density=True, alpha=0.7, color='skyblue')
            
            # 정규분포 곡선 추가
            mu, sigma = data.mean(), data.std()
            x = np.linspace(data.min(), data.max(), 100)
            normal_curve = stats.norm.pdf(x, mu, sigma)
            axes[row, col_idx].plot(x, normal_curve, 'r-', linewidth=2, label='정규분포')
            
            # 왜도 정보 추가
            skew_val = distribution_info[col]['skewness']
            axes[row, col_idx].set_title(f'{col}\n왜도: {skew_val:.2f}')
            axes[row, col_idx].legend()
            axes[row, col_idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return distribution_info

# 분포 분석 실행
distribution_analysis = analyze_distribution(train_data)
```

**🔍 코드 해설:**
- `stats.skew()`: 왜도 계산 (양수면 오른쪽 꼬리, 음수면 왼쪽 꼬리)
- `stats.shapiro()`: 소표본용 정규성 검정 (p > 0.05이면 정규분포)
- `stats.norm.pdf()`: 정규분포 확률밀도함수로 이론적 곡선 그리기

### 방법 1: 로그 변환 (Log Transformation)

가장 간단하고 효과적인 변환 방법으로, 양의 왜도를 크게 줄일 수 있습니다.

```python
# 로그 변환 구현
def apply_log_transformation(df, columns=None):
    """
    로그 변환을 적용하고 효과를 분석하는 함수
    """
    if columns is None:
        # 양의 왜도가 큰 변수들 자동 선택
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'SalePrice' in numeric_cols:
            numeric_cols.remove('SalePrice')
        
        columns = []
        for col in numeric_cols:
            if col in df.columns:
                data = df[col].dropna()
                if (data > 0).all() and stats.skew(data) > 1:  # 양수이면서 왜도 > 1
                    columns.append(col)
        
        columns = columns[:4]  # 최대 4개
    
    if not columns:
        print("⚠️ 로그 변환이 적합한 변수가 없습니다.")
        return df, {}
    
    print(f"📊 로그 변환 적용 ({len(columns)}개 변수):")
    
    transformed_df = df.copy()
    transformation_info = {}
    
    # 변환 전후 비교
    fig, axes = plt.subplots(len(columns), 3, figsize=(15, 4*len(columns)))
    if len(columns) == 1:
        axes = axes.reshape(1, -1)
    
    for i, col in enumerate(columns):
        if col in df.columns:
            original_data = df[col].dropna()
            
            # 0이나 음수 처리
            if (original_data <= 0).any():
                # log1p 사용 (log(1+x))
                transformed_data = np.log1p(original_data)
                transform_type = "log1p"
            else:
                # 자연로그 사용
                transformed_data = np.log(original_data)
                transform_type = "log"
            
            # 변환 적용
            transformed_df[col + '_log'] = np.nan
            valid_indices = original_data.index
            if transform_type == "log1p":
                transformed_df.loc[valid_indices, col + '_log'] = np.log1p(df.loc[valid_indices, col])
            else:
                transformed_df.loc[valid_indices, col + '_log'] = np.log(df.loc[valid_indices, col])
            
            # 통계 계산
            original_skew = stats.skew(original_data)
            transformed_skew = stats.skew(transformed_data)
            
            transformation_info[col] = {
                'transform_type': transform_type,
                'original_skew': original_skew,
                'transformed_skew': transformed_skew,
                'improvement': original_skew - transformed_skew
            }
            
            print(f"\n📈 {col}:")
            print(f"   변환 방법: {transform_type}")
            print(f"   왜도 변화: {original_skew:.3f} → {transformed_skew:.3f}")
            print(f"   개선도: {original_skew - transformed_skew:.3f}")
            
            # 시각화
            # 원본 분포
            axes[i, 0].hist(original_data, bins=50, alpha=0.7, color='skyblue')
            axes[i, 0].set_title(f'{col} 원본\n왜도: {original_skew:.2f}')
            axes[i, 0].grid(True, alpha=0.3)
            
            # 변환 후 분포
            axes[i, 1].hist(transformed_data, bins=50, alpha=0.7, color='lightcoral')
            axes[i, 1].set_title(f'{col} {transform_type} 변환\n왜도: {transformed_skew:.2f}')
            axes[i, 1].grid(True, alpha=0.3)
            
            # Q-Q 플롯 (정규성 확인)
            stats.probplot(transformed_data, dist="norm", plot=axes[i, 2])
            axes[i, 2].set_title(f'{col} Q-Q 플롯\n(직선에 가까울수록 정규분포)')
            axes[i, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n✅ 로그 변환 완료!")
    print(f"   변환된 변수: {len(transformation_info)}개")
    print(f"   💡 왜도가 크게 개선되어 정규분포에 가까워졌습니다!")
    
    return transformed_df, transformation_info

# 로그 변환 적용
log_transformed_data, log_info = apply_log_transformation(train_data)
```

**🔍 코드 해설:**
- `np.log1p()`: log(1+x) 변환으로 0값 포함 데이터에 안전하게 적용
- `stats.probplot()`: Q-Q 플롯으로 정규성 시각적 확인
- 변환 전후 왜도 비교로 개선 효과 정량화

### 방법 2: Box-Cox 변환

최적의 변환 파라미터(λ)를 자동으로 찾아주는 고급 변환 방법입니다.

```python
# Box-Cox 변환 구현
def apply_boxcox_transformation(df, columns=None):
    """
    Box-Cox 변환을 적용하고 최적 파라미터를 찾는 함수
    """
    if columns is None:
        # 양수인 변수들 중 왜도가 큰 것 선택
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'SalePrice' in numeric_cols:
            numeric_cols.remove('SalePrice')
        
        columns = []
        for col in numeric_cols[:6]:  # 최대 6개
            if col in df.columns:
                data = df[col].dropna()
                if (data > 0).all() and abs(stats.skew(data)) > 0.5:  # 양수이면서 왜도 > 0.5
                    columns.append(col)
    
    if not columns:
        print("⚠️ Box-Cox 변환이 적합한 변수가 없습니다.")
        return df, {}
    
    print(f"📊 Box-Cox 변환 적용 ({len(columns)}개 변수):")
    
    transformed_df = df.copy()
    transformation_info = {}
    
    for col in columns:
        if col in df.columns:
            data = df[col].dropna()
            
            try:
                # Box-Cox 변환 적용 (최적 λ 자동 계산)
                transformed_data, optimal_lambda = boxcox(data)
                
                # 변환 적용
                transformed_df[col + '_boxcox'] = np.nan
                valid_indices = data.index
                transformed_values, _ = boxcox(df.loc[valid_indices, col])
                transformed_df.loc[valid_indices, col + '_boxcox'] = transformed_values
                
                # 통계 계산
                original_skew = stats.skew(data)
                transformed_skew = stats.skew(transformed_data)
                
                transformation_info[col] = {
                    'lambda': optimal_lambda,
                    'original_skew': original_skew,
                    'transformed_skew': transformed_skew,
                    'improvement': abs(original_skew) - abs(transformed_skew)
                }
                
                print(f"\n📈 {col}:")
                print(f"   최적 λ (람다): {optimal_lambda:.3f}")
                print(f"   왜도 변화: {original_skew:.3f} → {transformed_skew:.3f}")
                print(f"   개선도: {abs(original_skew) - abs(transformed_skew):.3f}")
                
                # 람다 값 해석
                if abs(optimal_lambda) < 0.1:
                    interpretation = "로그 변환과 유사"
                elif abs(optimal_lambda - 0.5) < 0.1:
                    interpretation = "제곱근 변환과 유사"
                elif abs(optimal_lambda - 1.0) < 0.1:
                    interpretation = "변환 불필요"
                elif abs(optimal_lambda - 2.0) < 0.1:
                    interpretation = "제곱 변환과 유사"
                else:
                    interpretation = f"λ={optimal_lambda:.2f} 거듭제곱 변환"
                
                print(f"   해석: {interpretation}")
                
            except Exception as e:
                print(f"   ⚠️ {col} 변환 실패: {e}")
    
    # 변환 효과 시각화
    if transformation_info:
        # 최적 λ 분포
        lambdas = [info['lambda'] for info in transformation_info.values()]
        improvements = [info['improvement'] for info in transformation_info.values()]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # λ 값 분포
        axes[0].bar(range(len(lambdas)), lambdas, color='skyblue', alpha=0.7)
        axes[0].set_title('Box-Cox 최적 λ (람다) 값')
        axes[0].set_xlabel('변수')
        axes[0].set_ylabel('λ 값')
        axes[0].set_xticks(range(len(columns)))
        axes[0].set_xticklabels(columns, rotation=45)
        axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5, label='λ=0 (로그변환)')
        axes[0].axhline(y=1, color='green', linestyle='--', alpha=0.5, label='λ=1 (변환없음)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 개선 효과
        axes[1].bar(range(len(improvements)), improvements, color='lightcoral', alpha=0.7)
        axes[1].set_title('왜도 개선 효과')
        axes[1].set_xlabel('변수')
        axes[1].set_ylabel('왜도 개선 정도')
        axes[1].set_xticks(range(len(columns)))
        axes[1].set_xticklabels(columns, rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    print(f"\n✅ Box-Cox 변환 완료!")
    print(f"   변환된 변수: {len(transformation_info)}개")
    print(f"   💡 각 변수에 최적화된 변환으로 최대 왜도 개선 효과!")
    
    return transformed_df, transformation_info

# Box-Cox 변환 적용
boxcox_transformed_data, boxcox_info = apply_boxcox_transformation(train_data)
```

**🔍 코드 해설:**
- `boxcox()`: 최적의 λ(람다) 파라미터를 자동으로 찾아서 변환 수행
- λ=0이면 로그변환, λ=1이면 변환 불필요, λ=0.5면 제곱근 변환과 유사
- 각 변수별로 최적화된 변환으로 더 나은 결과 기대

---

## 📖 4.2.4 scikit-learn Pipeline을 활용한 통합 전처리

지금까지 배운 다양한 전처리 기법들을 **체계적으로 결합**하여 재사용 가능한 파이프라인을 구축해보겠습니다.

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# 통합 전처리 파이프라인 구축
def create_preprocessing_pipeline(df):
    """
    House Prices 데이터에 특화된 종합 전처리 파이프라인 구축
    """
    print("🔧 House Prices 맞춤형 전처리 파이프라인 구축:")
    
    # 1. 변수 유형별 분류
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'SalePrice' in numeric_features:
        numeric_features.remove('SalePrice')  # 타겟 변수 제외
    
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    
    # 범주형 변수를 카디널리티에 따라 세분화
    low_cardinality_features = [col for col in categorical_features 
                               if df[col].nunique() <= 10]
    high_cardinality_features = [col for col in categorical_features 
                                if df[col].nunique() > 10]
    
    # 순서형 변수 (품질 관련)
    ordinal_features = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
                       'HeatingQC', 'KitchenQual', 'FireplaceQu']
    ordinal_features = [col for col in ordinal_features if col in df.columns]
    
    # 순서형 변수를 다른 범주형에서 제거
    low_cardinality_features = [col for col in low_cardinality_features 
                               if col not in ordinal_features]
    high_cardinality_features = [col for col in high_cardinality_features 
                                if col not in ordinal_features]
    
    print(f"   📊 변수 분류:")
    print(f"      수치형: {len(numeric_features)}개")
    print(f"      낮은 카디널리티 범주형: {len(low_cardinality_features)}개")
    print(f"      높은 카디널리티 범주형: {len(high_cardinality_features)}개")
    print(f"      순서형: {len(ordinal_features)}개")
    
    # 2. 각 변수 유형별 전처리 파이프라인 정의
    
    # 수치형 변수 파이프라인
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # 결측치 중앙값 대체
        ('scaler', RobustScaler())  # 이상치에 강건한 스케일링
    ])
    
    # 낮은 카디널리티 범주형 변수 파이프라인
    low_cardinality_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # 높은 카디널리티 범주형 변수 파이프라인 (단순화)
    high_cardinality_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('label', LabelEncoder())  # 실제로는 타겟 인코딩이 더 좋지만 단순화
    ])
    
    # 순서형 변수 파이프라인
    ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    
    # 3. ColumnTransformer로 통합
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('low_cat', low_cardinality_transformer, low_cardinality_features),
            ('high_cat', high_cardinality_transformer, high_cardinality_features),
            ('ord', ordinal_transformer, ordinal_features)
        ]
    )
    
    print(f"\n🔧 파이프라인 구성:")
    print(f"   1️⃣ 수치형: 중앙값 대체 → 로버스트 스케일링")
    print(f"   2️⃣ 낮은 카디널리티: Unknown 대체 → 원-핫 인코딩")
    print(f"   3️⃣ 높은 카디널리티: 최빈값 대체 → 레이블 인코딩")
    print(f"   4️⃣ 순서형: Unknown 대체 → 순서형 인코딩")
    
    return preprocessor, {
        'numeric_features': numeric_features,
        'low_cardinality_features': low_cardinality_features,
        'high_cardinality_features': high_cardinality_features,
        'ordinal_features': ordinal_features
    }

# 파이프라인 생성
preprocessor, feature_info = create_preprocessing_pipeline(train_data)

# 파이프라인 적용 및 결과 확인
def apply_and_evaluate_pipeline(df, preprocessor, feature_info):
    """
    파이프라인을 적용하고 결과를 평가하는 함수
    """
    print("\n🚀 전처리 파이프라인 적용:")
    
    # 타겟 변수 분리
    if 'SalePrice' in df.columns:
        X = df.drop('SalePrice', axis=1)
        y = df['SalePrice']
    else:
        X = df
        y = None
    
    original_shape = X.shape
    
    # 파이프라인 적용
    try:
        X_transformed = preprocessor.fit_transform(X)
        
        print(f"   ✅ 변환 성공!")
        print(f"   📊 데이터 크기 변화: {original_shape} → {X_transformed.shape}")
        print(f"   📈 특성 개수 변화: {original_shape[1]} → {X_transformed.shape[1]}")
        
        # 결측치 확인
        if hasattr(X_transformed, 'isnan'):
            missing_after = np.isnan(X_transformed).sum()
        else:
            missing_after = pd.DataFrame(X_transformed).isnull().sum().sum()
        
        print(f"   🔍 변환 후 결측치: {missing_after}개")
        
        # 변환된 데이터의 기본 통계
        X_transformed_df = pd.DataFrame(X_transformed)
        print(f"\n📊 변환된 데이터 통계:")
        print(f"   평균 범위: [{X_transformed_df.mean().min():.3f}, {X_transformed_df.mean().max():.3f}]")
        print(f"   표준편차 범위: [{X_transformed_df.std().min():.3f}, {X_transformed_df.std().max():.3f}]")
        
        return X_transformed, y
        
    except Exception as e:
        print(f"   ❌ 변환 실패: {e}")
        return None, y

# 파이프라인 적용
X_processed, y = apply_and_evaluate_pipeline(train_data, preprocessor, feature_info)

print(f"\n✅ 통합 전처리 파이프라인 완성!")
print(f"💡 이제 이 파이프라인을 새로운 데이터에도 동일하게 적용할 수 있습니다!")
```

**🔍 코드 해설:**
- `ColumnTransformer`: 서로 다른 변수 유형에 다른 전처리 방법을 적용
- `Pipeline`: 여러 전처리 단계를 순차적으로 연결
- `fit_transform()`: 파이프라인 학습과 적용을 동시에 수행

---

## 🎯 직접 해보기 - 연습 문제

### 연습 문제 1: 스케일링 방법 비교 ⭐⭐
다음 코드를 완성하여 스케일링 방법들의 효과를 비교해보세요.

```python
# 연습 문제 1: 스케일링 방법 비교
def exercise_compare_scaling(df):
    """
    여러 스케일링 방법을 비교하고 최적 방법을 찾는 함수
    """
    from sklearn.metrics import mean_squared_error
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    
    # TODO: 다음 단계를 구현하세요
    # 1. GrLivArea 변수에 3가지 스케일링 방법 적용
    # 2. 각각으로 SalePrice 예측하는 간단한 선형 회귀 모델 학습
    # 3. 성능(RMSE) 비교하여 최적 방법 선택
    
    scalers = {
        'Standard': StandardScaler(),
        'MinMax': MinMaxScaler(),
        'Robust': RobustScaler()
    }
    
    results = {}
    
    # 여기에 코드를 작성하세요
    
    return results

# 힌트: fit_transform(), train_test_split(), mean_squared_error() 활용
```

### 연습 문제 2: 맞춤형 인코딩 전략 ⭐⭐⭐
부동산 데이터의 특성을 고려한 인코딩 전략을 수립해보세요.

```python
# 연습 문제 2: 맞춤형 인코딩 전략
def exercise_custom_encoding(df):
    """
    변수별 특성을 고려한 맞춤형 인코딩 전략 수립
    """
    # TODO: 다음 규칙에 따라 인코딩 방법을 결정하고 적용하세요
    # 1. Neighborhood: 카디널리티가 높으므로 타겟 인코딩
    # 2. MSSubClass: 명목형이지만 숫자로 되어 있으므로 원-핫 인코딩
    # 3. GarageType: 카디널리티가 낮으므로 원-핫 인코딩
    
    encoding_strategy = {}
    
    # 여기에 코드를 작성하세요
    
    return encoding_strategy
```

### 연습 문제 3: 분포 변환 효과 분석 ⭐⭐⭐⭐
SalePrice의 분포를 개선하는 최적 변환 방법을 찾아보세요.

```python
# 연습 문제 3: 분포 변환 효과 분석
def exercise_transform_target(df):
    """
    타겟 변수(SalePrice)의 분포를 개선하는 최적 변환 찾기
    """
    # TODO: 다음 변환들을 시도하고 효과를 비교하세요
    # 1. 로그 변환: log(SalePrice)
    # 2. 제곱근 변환: sqrt(SalePrice)
    # 3. Box-Cox 변환: 최적 λ 찾기
    # 4. 변환 전후 왜도, 정규성 검정 결과 비교
    
    transformations = {}
    
    # 여기에 코드를 작성하세요
    
    return transformations
```

---

## 📚 핵심 정리

이번 Part에서 배운 핵심 내용을 정리하면 다음과 같습니다:

### ✅ 스케일링 기법 핵심 포인트

1. **StandardScaler**: 평균 0, 표준편차 1 → 정규분포 가정 알고리즘에 최적
2. **MinMaxScaler**: 0~1 범위 → 명확한 최솟값/최댓값이 의미있는 경우
3. **RobustScaler**: 중앙값과 IQR 사용 → 이상치가 많은 실제 데이터에 안정적

### ✅ 범주형 변수 인코딩 핵심 포인트

1. **원-핫 인코딩**: 명목형 + 낮은 카디널리티 → 안전하고 해석하기 쉬움
2. **순서형 인코딩**: 순서가 있는 변수 → 순서 정보 보존하면서 효율적
3. **타겟 인코딩**: 높은 카디널리티 → 예측력 높지만 과적합 주의

### ✅ 분포 변환 핵심 포인트

1. **로그 변환**: 양의 왜도 개선 → 간단하고 해석하기 쉬움
2. **Box-Cox 변환**: 최적 파라미터 자동 탐색 → 더 정교한 변환
3. **정규성 개선**: 많은 ML 알고리즘의 성능 향상

### 💡 실무 적용 팁

- **파이프라인 활용**: 체계적이고 재사용 가능한 전처리 워크플로우 구축
- **변환 순서**: 결측치 처리 → 인코딩 → 스케일링 → 분포 변환
- **검증**: 변환 전후 성능 비교로 효과 확인
- **문서화**: 선택한 방법과 이유를 명확히 기록

---

## 🤔 생각해보기

1. **스케일링의 필요성**: 왜 나이(20-80)와 연봉(2000-8000만원) 같은 변수를 함께 사용할 때 스케일링이 필요할까요? 구체적인 문제 상황을 생각해보세요.

2. **인코딩 방법의 트레이드오프**: 원-핫 인코딩은 안전하지만 차원이 폭발할 수 있고, 타겟 인코딩은 효과적이지만 과적합 위험이 있습니다. 어떤 기준으로 선택해야 할까요?

3. **변환의 해석성**: 로그 변환된 데이터로 학습한 모델의 결과를 어떻게 원래 스케일로 해석할 수 있을까요? 비즈니스 담당자에게 어떻게 설명하시겠습니까?

---

## 🔜 다음 Part 예고: 특성 공학(Feature Engineering)

다음 Part에서는 원본 데이터로부터 **새로운 의미있는 특성**을 생성하는 고급 기법들을 배웁니다:

- **도메인 지식 기반 특성 생성**: 주택 데이터의 비즈니스 로직을 활용한 파생 변수
- **수학적 특성 조합**: 비율, 차이, 상호작용 등 수치적 특성 결합
- **시간 기반 특성**: 연도, 계절성 등 시간 정보 활용
- **특성 선택**: 중요한 특성만 골라내는 체계적 방법
- **자동 특성 공학**: AI 도구를 활용한 특성 생성과 검증

데이터에서 숨겨진 패턴을 발견하고 모델의 예측력을 극대화하는 창의적인 특성 공학 기법들을 마스터해보겠습니다!

---

*"데이터 변환은 알고리즘과 데이터 사이의 '번역' 작업입니다. 올바른 변환은 숨겨진 패턴을 드러내고 모델의 학습을 도와줍니다."*
```
```

