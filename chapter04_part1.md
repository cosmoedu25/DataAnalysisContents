# 4장 Part 1: 결측치와 이상치 처리
## 실제 데이터의 불완전성을 해결하는 전문가 기법

---

## 📚 학습 목표

이번 Part에서는 다음과 같은 내용을 학습합니다:

✅ **결측치의 종류와 발생 원리를 이해하고 적절한 처리 방법을 선택할 수 있다**
✅ **이상치를 탐지하고 비즈니스 맥락에서 적절히 처리할 수 있다**  
✅ **House Prices 데이터셋을 활용한 실전 결측치/이상치 처리 경험을 쌓는다**
✅ **Pandas와 scikit-learn을 활용한 고급 처리 기법을 구현할 수 있다**
✅ **처리 결과의 품질을 평가하고 비즈니스 영향을 분석할 수 있다**

---

## 🎯 이번 Part 미리보기

실제 데이터 분석 프로젝트에서 가장 먼저 마주하게 되는 도전은 바로 **불완전한 데이터**입니다. 설문조사에서 응답하지 않은 문항, 센서 오류로 누락된 측정값, 입력 실수로 발생한 극단값들... 이런 문제들은 데이터 분석가라면 반드시 해결해야 할 현실적인 과제입니다.

이번 Part에서는 Kaggle의 **House Prices Dataset**을 활용하여 실제 부동산 데이터에서 발생하는 결측치와 이상치 문제를 체계적으로 해결해보겠습니다. 단순히 결측치를 삭제하거나 평균값으로 채우는 수준을 넘어서, **비즈니스 맥락을 고려한 전문가 수준의 처리 기법**을 배우게 됩니다.

특히 주택 데이터의 특성상 "지하실이 없으면 지하실 면적이 결측"되는 것처럼 **구조적인 결측치의 의미**를 파악하고, 주택 가격에 큰 영향을 미치는 **이상치의 비즈니스적 해석**을 수행하는 방법을 익히게 됩니다.

> **💡 Part 1의 핵심 포인트**  
> "결측치와 이상치는 단순한 '문제'가 아니라 데이터가 전하는 '정보'일 수 있습니다. 이를 올바르게 해석하고 처리하는 것이 데이터 과학자의 핵심 역량입니다."

---

## 📖 4.1.1 결측치의 이해와 분류

### 결측치란 무엇인가?

**결측치(Missing Value)**는 데이터셋에서 값이 누락된 관측값을 의미합니다. Python의 pandas에서는 주로 `NaN`(Not a Number), `None`, 또는 빈 문자열로 표현됩니다.

> **🔍 주요 용어 해설**
> - **NaN (Not a Number)**: 숫자형 데이터에서 정의되지 않은 값
> - **None**: Python의 null 객체로, 값이 없음을 나타냄
> - **결측률(Missing Rate)**: 전체 데이터 중 결측치가 차지하는 비율

결측치가 발생하는 이유는 다양합니다:

1. **데이터 수집 과정의 오류**: 센서 고장, 네트워크 문제 등
2. **응답자의 의도적/비의도적 누락**: 설문조사에서 민감한 질문 회피
3. **시스템 설계상의 한계**: 특정 조건에서만 수집되는 데이터
4. **데이터 전송/저장 과정의 손실**: 파일 전송 오류, 저장 공간 부족

### 결측치의 3가지 유형 (Missing Data Mechanisms)

통계학에서는 결측치를 발생 메커니즘에 따라 3가지로 분류합니다:

#### 1. MCAR (Missing Completely At Random)
- **완전 무작위 결측**: 결측 여부가 다른 변수와 전혀 관련이 없는 경우
- **예시**: 설문 조사 중 컴퓨터 시스템 오류로 랜덤하게 일부 응답이 누락
- **처리**: 단순 삭제나 평균값 대체가 비교적 안전

#### 2. MAR (Missing At Random)  
- **무작위 결측**: 결측 여부가 관측된 다른 변수와는 관련이 있지만, 해당 변수 자체의 값과는 무관
- **예시**: 나이가 많을수록 소득 정보 제공을 꺼려하는 경우 (나이는 관측됨)
- **처리**: 다른 변수를 활용한 예측 기반 대체 방법 활용

#### 3. MNAR (Missing Not At Random)
- **비무작위 결측**: 결측 여부가 해당 변수 자체의 값과 관련이 있는 경우  
- **예시**: 소득이 매우 높거나 낮은 사람들이 소득 정보 제공을 거부
- **처리**: 가장 복잡하며, 도메인 지식과 고급 통계 기법 필요

### House Prices 데이터에서 결측치 탐색

이제 실제 데이터를 통해 결측치의 패턴을 분석해보겠습니다.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (이미지 생성 시 한글 표시를 위함)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# House Prices 데이터 로드
# 주의: 데이터 파일 경로는 사용자 환경에 맞게 조정하세요
try:
    train_data = pd.read_csv('datasets/house_prices/train.csv')
    print("✅ 데이터 로드 성공!")
except FileNotFoundError:
    print("❌ 데이터 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    print("💡 Kaggle에서 House Prices Dataset을 다운로드하여 datasets/house_prices/ 폴더에 저장하세요.")

# 데이터 기본 정보 확인
print(f"📊 데이터 크기: {train_data.shape}")
print(f"🏠 총 주택 수: {train_data.shape[0]:,}개")
print(f"📈 총 변수 수: {train_data.shape[1]:,}개")

# 데이터 타입별 변수 현황
print("\n📋 데이터 타입별 변수 현황:")
print(train_data.dtypes.value_counts())
```

**🔍 코드 해설:**
- `pd.read_csv()`: CSV 파일을 pandas DataFrame으로 읽어오는 함수
- `train_data.shape`: 데이터의 (행 수, 열 수)를 반환하는 속성  
- `dtypes.value_counts()`: 각 데이터 타입별 변수 개수를 계산

### 결측치 현황 종합 분석

```python
# 결측치 현황 분석 함수
def analyze_missing_data(df):
    """
    데이터프레임의 결측치 현황을 종합적으로 분석하는 함수
    
    Parameters:
    df (pd.DataFrame): 분석할 데이터프레임
    
    Returns:
    pd.DataFrame: 결측치 분석 결과
    """
    # 결측치 개수와 비율 계산
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100
    
    # 결측치가 있는 컬럼만 필터링
    missing_data = pd.DataFrame({
        'Column': missing_count.index,
        'Missing_Count': missing_count.values,
        'Missing_Percent': missing_percent.values,
        'Data_Type': df.dtypes.values
    })
    
    # 결측치가 있는 컬럼만 선택
    missing_data = missing_data[missing_data['Missing_Count'] > 0]
    
    # 결측치 비율 기준 내림차순 정렬
    missing_data = missing_data.sort_values('Missing_Percent', ascending=False)
    
    return missing_data

# 결측치 분석 실행
missing_analysis = analyze_missing_data(train_data)
print("🔍 결측치 현황 분석:")
print(missing_analysis.head(10))

# 결측치 비율별 분류
high_missing = missing_analysis[missing_analysis['Missing_Percent'] > 50]
medium_missing = missing_analysis[(missing_analysis['Missing_Percent'] > 15) & 
                                 (missing_analysis['Missing_Percent'] <= 50)]
low_missing = missing_analysis[missing_analysis['Missing_Percent'] <= 15]

print(f"\n📊 결측치 수준별 분류:")
print(f"🔴 높은 결측치 (50% 초과): {len(high_missing)}개 컬럼")
print(f"🟡 중간 결측치 (15-50%): {len(medium_missing)}개 컬럼") 
print(f"🟢 낮은 결측치 (15% 이하): {len(low_missing)}개 컬럼")
```

**🔍 코드 해설:**
- `df.isnull().sum()`: 각 컬럼별 결측치 개수를 계산
- `(missing_count / len(df)) * 100`: 결측치 비율을 백분율로 변환
- `sort_values('Missing_Percent', ascending=False)`: 결측치 비율 기준 내림차순 정렬

이 분석을 통해 우리는 어떤 변수들이 높은 결측치를 가지고 있는지, 그리고 이것이 우연한 누락인지 아니면 구조적인 특성인지 파악할 수 있습니다.

### 결측치 시각화와 패턴 분석

숫자로만 보는 결측치 현황을 시각적으로 파악하여 더 깊은 인사이트를 얻어보겠습니다.

```python
# 결측치 시각화 함수
def visualize_missing_data(df, figsize=(15, 10)):
    """
    결측치를 다양한 관점에서 시각화하는 함수
    
    Parameters:
    df (pd.DataFrame): 분석할 데이터프레임
    figsize (tuple): 그래프 크기
    """
    
    # 서브플롯 설정
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('🔍 House Prices Dataset 결측치 종합 분석', fontsize=16, fontweight='bold')
    
    # 1. 결측치 히트맵 (상위 20개 변수)
    missing_data = df.isnull().sum().sort_values(ascending=False)
    top_missing = missing_data[missing_data > 0].head(20)
    
    if len(top_missing) > 0:
        missing_matrix = df[top_missing.index].isnull()
        sns.heatmap(missing_matrix.T, 
                   cbar=True, 
                   cmap='YlOrRd',
                   ax=axes[0,0])
        axes[0,0].set_title('결측치 패턴 히트맵\n(노란색: 값 있음, 빨간색: 결측치)', fontsize=12)
        axes[0,0].set_xlabel('데이터 인덱스')
        axes[0,0].set_ylabel('변수명')
    
    # 2. 결측치 비율 막대그래프
    missing_percent = (missing_data / len(df)) * 100
    top_missing_percent = missing_percent[missing_percent > 0].head(15)
    
    top_missing_percent.plot(kind='barh', ax=axes[0,1], color='coral')
    axes[0,1].set_title('변수별 결측치 비율 (%)', fontsize=12)
    axes[0,1].set_xlabel('결측치 비율 (%)')
    
    # 3. 결측치 수준별 분포
    missing_levels = []
    if len(missing_percent[missing_percent > 50]) > 0:
        missing_levels.append(('50% 초과', len(missing_percent[missing_percent > 50])))
    if len(missing_percent[(missing_percent > 15) & (missing_percent <= 50)]) > 0:
        missing_levels.append(('15-50%', len(missing_percent[(missing_percent > 15) & (missing_percent <= 50)])))
    if len(missing_percent[(missing_percent > 0) & (missing_percent <= 15)]) > 0:
        missing_levels.append(('15% 이하', len(missing_percent[(missing_percent > 0) & (missing_percent <= 15)])))
    if len(missing_percent[missing_percent == 0]) > 0:
        missing_levels.append(('결측치 없음', len(missing_percent[missing_percent == 0])))
    
    if missing_levels:
        levels, counts = zip(*missing_levels)
        colors = ['red', 'orange', 'yellow', 'lightgreen'][:len(levels)]
        axes[1,0].pie(counts, labels=levels, autopct='%1.1f%%', colors=colors)
        axes[1,0].set_title('결측치 수준별 변수 분포', fontsize=12)
    
    # 4. 데이터 타입별 결측치 현황
    type_missing = df.dtypes.to_frame('dtype').join(df.isnull().sum().to_frame('missing'))
    type_summary = type_missing.groupby('dtype')['missing'].agg(['count', 'sum', 'mean'])
    
    type_summary['mean'].plot(kind='bar', ax=axes[1,1], color='skyblue')
    axes[1,1].set_title('데이터 타입별 평균 결측치 수', fontsize=12)
    axes[1,1].set_xlabel('데이터 타입')
    axes[1,1].set_ylabel('평균 결측치 수')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

# 결측치 시각화 실행
visualize_missing_data(train_data)
```

**🔍 코드 해설:**
- `sns.heatmap()`: 결측치 패턴을 색상으로 표현하는 히트맵 생성
- `plot(kind='barh')`: 수평 막대그래프로 결측치 비율 시각화  
- `plt.pie()`: 원형 차트로 결측치 수준별 분포 표현
- `groupby().agg()`: 데이터 타입별로 그룹화하여 집계 통계 계산

> **📊 이미지 생성 프롬프트:**  
> "Create a comprehensive data visualization dashboard showing missing data analysis for a house prices dataset. Include: 1) A heatmap showing missing data patterns across variables and observations with yellow for present values and red for missing values, 2) A horizontal bar chart showing missing data percentages by variable, 3) A pie chart showing distribution of variables by missing data levels (>50%, 15-50%, <15%, no missing), 4) A bar chart showing average missing data count by data type. Use professional color schemes with clear labels and titles in a 2x2 subplot layout."

### 구조적 결측치 vs 임의적 결측치 구분

House Prices 데이터에서는 많은 결측치가 **구조적 특성**을 가지고 있습니다. 예를 들어:

```python
# 구조적 결측치 패턴 분석
def analyze_structural_missing(df):
    """
    구조적 결측치 패턴을 분석하는 함수
    """
    print("🏗️ 구조적 결측치 패턴 분석:\n")
    
    # 1. 지하실 관련 변수들
    basement_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
    basement_missing = df[basement_cols].isnull().sum()
    
    print("🔸 지하실 관련 변수 결측치:")
    for col in basement_cols:
        if col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_percent = (missing_count / len(df)) * 100
            print(f"   {col}: {missing_count}개 ({missing_percent:.1f}%)")
    
    # 2. 차고 관련 변수들  
    garage_cols = ['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond']
    print(f"\n🔸 차고 관련 변수 결측치:")
    for col in garage_cols:
        if col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_percent = (missing_count / len(df)) * 100
            print(f"   {col}: {missing_count}개 ({missing_percent:.1f}%)")
    
    # 3. 수영장 관련 변수들
    if 'PoolQC' in df.columns:
        pool_missing = df['PoolQC'].isnull().sum()
        pool_percent = (pool_missing / len(df)) * 100
        print(f"\n🔸 수영장 품질(PoolQC): {pool_missing}개 ({pool_percent:.1f}%)")
        
        # 수영장 면적과 품질의 관계 확인
        if 'PoolArea' in df.columns:
            pool_area_zero = (df['PoolArea'] == 0).sum()
            print(f"   수영장 면적이 0인 경우: {pool_area_zero}개")
            print(f"   💡 수영장이 없으면 품질 평가도 불가능함을 의미")
    
    # 4. 벽난로 관련 변수들
    if 'FireplaceQu' in df.columns:
        fireplace_missing = df['FireplaceQu'].isnull().sum()
        fireplace_percent = (fireplace_missing / len(df)) * 100
        print(f"\n🔸 벽난로 품질(FireplaceQu): {fireplace_missing}개 ({fireplace_percent:.1f}%)")
        
        if 'Fireplaces' in df.columns:
            no_fireplace = (df['Fireplaces'] == 0).sum()
            print(f"   벽난로 개수가 0인 경우: {no_fireplace}개")
            print(f"   💡 벽난로가 없으면 품질 평가가 불가능함")

# 구조적 결측치 분석 실행
analyze_structural_missing(train_data)
```

**🔍 코드 해설:**
- 구조적 결측치는 특정 시설이 없을 때 해당 시설의 품질이나 특성을 평가할 수 없어서 발생
- 예: 지하실이 없는 주택에서는 지하실 품질(BsmtQual)을 평가할 수 없음
- 이런 경우 결측치는 "정보의 부재"가 아니라 "해당 시설의 부재"를 의미

### 결측치 상관관계 분석

서로 다른 변수들의 결측치가 함께 발생하는 패턴을 분석해보겠습니다.

```python
# 결측치 상관관계 분석
def analyze_missing_correlation(df):
    """
    변수들 간의 결측치 상관관계를 분석하는 함수
    """
    # 결측치가 있는 변수들만 선택
    missing_cols = df.columns[df.isnull().any()].tolist()
    
    if len(missing_cols) > 1:
        # 결측치 패턴을 0(값 있음)과 1(결측치)로 변환
        missing_pattern = df[missing_cols].isnull().astype(int)
        
        # 결측치 간 상관관계 계산
        missing_corr = missing_pattern.corr()
        
        # 상관관계 히트맵 그리기
        plt.figure(figsize=(12, 10))
        mask = np.triu(missing_corr)  # 상삼각 마스크로 중복 제거
        
        sns.heatmap(missing_corr, 
                   mask=mask,
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0,
                   square=True,
                   fmt='.2f')
        
        plt.title('📊 변수 간 결측치 상관관계\n(1에 가까울수록 함께 결측되는 경향)', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # 높은 상관관계를 가진 변수 쌍 찾기
        high_corr_pairs = []
        for i in range(len(missing_corr.columns)):
            for j in range(i+1, len(missing_corr.columns)):
                corr_value = missing_corr.iloc[i, j]
                if abs(corr_value) > 0.7:  # 상관계수 0.7 이상
                    high_corr_pairs.append((
                        missing_corr.columns[i], 
                        missing_corr.columns[j], 
                        corr_value
                    ))
        
        if high_corr_pairs:
            print("\n🔗 높은 결측치 상관관계를 가진 변수 쌍 (|r| > 0.7):")
            for var1, var2, corr in high_corr_pairs:
                print(f"   {var1} ↔ {var2}: {corr:.3f}")
        else:
            print("\n✅ 매우 높은 결측치 상관관계를 가진 변수 쌍은 없습니다.")

# 결측치 상관관계 분석 실행
analyze_missing_correlation(train_data)
```

**🔍 코드 해설:**
- `df.isnull().astype(int)`: True/False를 1/0으로 변환하여 수치 계산 가능하게 함
- `np.triu()`: 상삼각 행렬 마스크로 중복되는 상관계수 제거
- `sns.heatmap(center=0)`: 0을 중심으로 하는 색상 스케일 적용

이 분석을 통해 어떤 변수들이 함께 결측되는 경향이 있는지 파악할 수 있습니다. 예를 들어, 지하실 관련 변수들은 서로 높은 결측치 상관관계를 보일 것입니다.

---

## 📖 4.1.2 결측치 처리 방법론과 실제 구현

결측치 분석을 마쳤으니, 이제 실제로 이를 처리하는 다양한 방법들을 배워보겠습니다. 각 방법의 특징과 적용 시나리오를 이해하고, House Prices 데이터에 실제로 적용해보겠습니다.

### 결측치 처리 방법 분류

결측치 처리 방법은 크게 **삭제(Deletion)**, **대체(Imputation)**, **예측(Prediction)** 방법으로 나눌 수 있습니다.

> **🔍 주요 용어 해설**
> - **대체(Imputation)**: 결측치를 다른 값으로 채워넣는 과정
> - **리스트와이즈 삭제(Listwise Deletion)**: 결측치가 하나라도 있는 행 전체를 삭제
> - **페어와이즈 삭제(Pairwise Deletion)**: 분석에 필요한 변수들에만 결측치가 없는 경우만 사용

### 방법 1: 삭제 기반 처리 (Deletion Methods)

```python
# 삭제 기반 결측치 처리 방법들
def deletion_methods_analysis(df):
    """
    다양한 삭제 방법의 영향을 분석하는 함수
    """
    original_shape = df.shape
    print(f"🏠 원본 데이터: {original_shape[0]:,}행 × {original_shape[1]:,}열")
    
    # 1. 완전 삭제 (Listwise Deletion)
    complete_cases = df.dropna()
    print(f"\n1️⃣ 완전 삭제 후: {complete_cases.shape[0]:,}행 × {complete_cases.shape[1]:,}열")
    print(f"   📉 손실률: {((original_shape[0] - complete_cases.shape[0]) / original_shape[0] * 100):.1f}%")
    
    # 2. 결측치 비율이 높은 변수 삭제
    missing_threshold = 50  # 50% 이상 결측치가 있는 변수 삭제
    missing_percent = (df.isnull().sum() / len(df)) * 100
    high_missing_cols = missing_percent[missing_percent > missing_threshold].index.tolist()
    
    df_reduced_cols = df.drop(columns=high_missing_cols)
    print(f"\n2️⃣ 높은 결측치 변수({missing_threshold}% 이상) 삭제:")
    print(f"   삭제된 변수: {len(high_missing_cols)}개")
    print(f"   남은 변수: {df_reduced_cols.shape[1]}개")
    if high_missing_cols:
        print(f"   삭제된 변수 목록: {', '.join(high_missing_cols[:5])}...")
    
    # 3. 하이브리드 접근: 높은 결측치 변수 삭제 후 행 삭제
    df_hybrid = df_reduced_cols.dropna()
    print(f"\n3️⃣ 하이브리드 접근 (변수 삭제 + 행 삭제):")
    print(f"   최종 데이터: {df_hybrid.shape[0]:,}행 × {df_hybrid.shape[1]:,}열")
    print(f"   📉 전체 손실률: {((original_shape[0] - df_hybrid.shape[0]) / original_shape[0] * 100):.1f}%")
    
    return {
        'original': df,
        'complete_cases': complete_cases,
        'reduced_columns': df_reduced_cols,
        'hybrid': df_hybrid,
        'high_missing_cols': high_missing_cols
    }

# 삭제 방법 분석 실행
deletion_results = deletion_methods_analysis(train_data)
```

**🔍 코드 해설:**
- `df.dropna()`: 결측치가 있는 모든 행을 삭제
- `missing_percent > missing_threshold`: 임계값보다 높은 결측치 비율을 가진 변수 필터링
- `df.drop(columns=...)`: 지정된 컬럼들을 삭제

### 방법 2: 통계적 대체 (Statistical Imputation)

```python
from sklearn.impute import SimpleImputer

# 통계적 대체 방법 구현
def statistical_imputation(df):
    """
    다양한 통계적 대체 방법을 적용하는 함수
    """
    # 원본 데이터 복사 (원본 보존)
    df_imputed = df.copy()
    
    # 수치형과 범주형 변수 분리
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'SalePrice' in numeric_cols:
        numeric_cols.remove('SalePrice')  # 타겟 변수 제외
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    print("📊 통계적 대체 방법 적용:")
    
    # 1. 수치형 변수: 중앙값으로 대체
    if numeric_cols:
        numeric_imputer = SimpleImputer(strategy='median')
        df_imputed[numeric_cols] = numeric_imputer.fit_transform(df_imputed[numeric_cols])
        print(f"✅ 수치형 변수 {len(numeric_cols)}개: 중앙값으로 대체")
    
    # 2. 범주형 변수: 최빈값으로 대체  
    if categorical_cols:
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        df_imputed[categorical_cols] = categorical_imputer.fit_transform(df_imputed[categorical_cols])
        print(f"✅ 범주형 변수 {len(categorical_cols)}개: 최빈값으로 대체")
    
    # 대체 전후 비교
    original_missing = df.isnull().sum().sum()
    final_missing = df_imputed.isnull().sum().sum()
    
    print(f"\n📈 대체 결과:")
    print(f"   대체 전 총 결측치: {original_missing:,}개")
    print(f"   대체 후 총 결측치: {final_missing:,}개")
    print(f"   처리 완료율: {((original_missing - final_missing) / original_missing * 100):.1f}%")
    
    return df_imputed

# 통계적 대체 실행
imputed_data = statistical_imputation(train_data)
```

**🔍 코드 해설:**
- `SimpleImputer(strategy='median')`: scikit-learn의 결측치 대체 도구, 중앙값 사용
- `strategy='most_frequent'`: 범주형 데이터에서 가장 자주 나타나는 값으로 대체
- `fit_transform()`: 대체 규칙을 학습(fit)하고 적용(transform)하는 과정

### 방법 3: 도메인 지식 기반 대체 (Domain-specific Imputation)

House Prices 데이터의 특성을 고려한 전문적인 대체 방법을 구현해보겠습니다.

```python
# 도메인 지식 기반 결측치 처리
def domain_specific_imputation(df):
    """
    부동산 도메인 지식을 활용한 결측치 처리 함수
    """
    df_domain = df.copy()
    
    print("🏡 부동산 도메인 지식 기반 결측치 처리:")
    
    # 1. 지하실 관련 변수들 - "지하실 없음"으로 처리
    basement_quality_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
    for col in basement_quality_cols:
        if col in df_domain.columns:
            before_count = df_domain[col].isnull().sum()
            df_domain[col] = df_domain[col].fillna('None')  # "없음"을 의미하는 값
            after_count = df_domain[col].isnull().sum()
            print(f"   {col}: {before_count}개 → {after_count}개 (None으로 대체)")
    
    # 2. 차고 관련 변수들 - "차고 없음"으로 처리
    garage_quality_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
    for col in garage_quality_cols:
        if col in df_domain.columns:
            before_count = df_domain[col].isnull().sum()
            df_domain[col] = df_domain[col].fillna('None')
            after_count = df_domain[col].isnull().sum()
            print(f"   {col}: {before_count}개 → {after_count}개 (None으로 대체)")
    
    # 3. 차고 건설연도 - 0으로 처리 (차고가 없으면 건설연도도 없음)
    if 'GarageYrBlt' in df_domain.columns:
        before_count = df_domain['GarageYrBlt'].isnull().sum()
        df_domain['GarageYrBlt'] = df_domain['GarageYrBlt'].fillna(0)
        after_count = df_domain['GarageYrBlt'].isnull().sum()
        print(f"   GarageYrBlt: {before_count}개 → {after_count}개 (0으로 대체)")
    
    # 4. 수영장/벽난로 품질 - "없음"으로 처리
    facility_cols = ['PoolQC', 'FireplaceQu', 'Fence', 'MiscFeature']
    for col in facility_cols:
        if col in df_domain.columns:
            before_count = df_domain[col].isnull().sum()
            df_domain[col] = df_domain[col].fillna('None')
            after_count = df_domain[col].isnull().sum()
            print(f"   {col}: {before_count}개 → {after_count}개 (None으로 대체)")
    
    # 5. 골목길 접근 - 대부분 골목길 접근이 없으므로 'No'로 처리
    if 'Alley' in df_domain.columns:
        before_count = df_domain['Alley'].isnull().sum()
        df_domain['Alley'] = df_domain['Alley'].fillna('None')
        after_count = df_domain['Alley'].isnull().sum()
        print(f"   Alley: {before_count}개 → {after_count}개 (None으로 대체)")
    
    # 6. 마감재 관련 - 추가 정보가 없는 경우 처리
    if 'MasVnrType' in df_domain.columns:
        before_count = df_domain['MasVnrType'].isnull().sum()
        df_domain['MasVnrType'] = df_domain['MasVnrType'].fillna('None')
        after_count = df_domain['MasVnrType'].isnull().sum()
        print(f"   MasVnrType: {before_count}개 → {after_count}개 (None으로 대체)")
    
    if 'MasVnrArea' in df_domain.columns:
        before_count = df_domain['MasVnrArea'].isnull().sum()
        # 마감재 타입이 None이면 면적도 0
        mask = df_domain['MasVnrType'] == 'None'
        df_domain.loc[mask, 'MasVnrArea'] = 0
        # 나머지는 중앙값으로 대체
        median_area = df_domain['MasVnrArea'].median()
        df_domain['MasVnrArea'] = df_domain['MasVnrArea'].fillna(median_area)
        after_count = df_domain['MasVnrArea'].isnull().sum()
        print(f"   MasVnrArea: {before_count}개 → {after_count}개 (0 또는 중앙값으로 대체)")
    
    # 7. 전기 시스템 - 표준 시스템으로 가정
    if 'Electrical' in df_domain.columns:
        before_count = df_domain['Electrical'].isnull().sum()
        mode_electrical = df_domain['Electrical'].mode()[0]
        df_domain['Electrical'] = df_domain['Electrical'].fillna(mode_electrical)
        after_count = df_domain['Electrical'].isnull().sum()
        print(f"   Electrical: {before_count}개 → {after_count}개 ({mode_electrical}으로 대체)")
    
    # 처리 결과 요약
    remaining_missing = df_domain.isnull().sum().sum()
    original_missing = df.isnull().sum().sum()
    
    print(f"\n📊 도메인 지식 기반 처리 결과:")
    print(f"   처리 전 총 결측치: {original_missing:,}개")
    print(f"   처리 후 총 결측치: {remaining_missing:,}개")
    print(f"   처리 완료율: {((original_missing - remaining_missing) / original_missing * 100):.1f}%")
    
    return df_domain

# 도메인 지식 기반 처리 실행
domain_imputed_data = domain_specific_imputation(train_data)
```

**🔍 코드 해설:**
- `fillna('None')`: 결측치를 'None' 문자열로 대체 (시설이 없음을 의미)
- `df.mode()[0]`: 최빈값(가장 자주 나타나는 값)의 첫 번째 값
- `df.loc[mask, column]`: 조건(mask)에 맞는 행의 특정 컬럼에 값 할당

도메인 지식을 활용한 처리의 핵심은 **결측치의 의미를 파악**하는 것입니다. "정보가 없다"가 아니라 "해당 시설이 없다"는 의미로 해석하여 적절한 값으로 대체합니다.

### 방법 4: 고급 예측 기반 대체 (Advanced Predictive Imputation)

```python
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import KNNImputer

# 예측 기반 결측치 대체
def predictive_imputation(df, target_col='SalePrice'):
    """
    머신러닝을 활용한 예측 기반 결측치 대체
    """
    df_pred = df.copy()
    
    print("🤖 예측 기반 결측치 대체:")
    
    # 1. KNN 대체 (K-Nearest Neighbors)
    print("\n1️⃣ KNN 대체 방법:")
    
    # 수치형 변수만 선택 (KNN은 수치형 데이터에 적합)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    # KNN 대체기 적용
    knn_imputer = KNNImputer(n_neighbors=5)
    df_pred[numeric_cols] = knn_imputer.fit_transform(df_pred[numeric_cols])
    
    print(f"   ✅ {len(numeric_cols)}개 수치형 변수에 KNN 대체 적용 (k=5)")
    
    # 2. 범주형 변수는 Random Forest로 예측
    print("\n2️⃣ Random Forest 예측 대체:")
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    processed_cols = []
    
    for col in categorical_cols:
        if df_pred[col].isnull().sum() > 0:
            # 결측치가 없는 행으로 학습 데이터 생성
            train_mask = df_pred[col].notnull()
            
            if train_mask.sum() > 10:  # 충분한 학습 데이터가 있는 경우
                # 특성 선택 (수치형 변수 중 결측치가 적은 것들)
                feature_cols = [c for c in numeric_cols 
                               if df_pred[c].isnull().sum() < len(df_pred) * 0.1][:10]
                
                if len(feature_cols) >= 3:  # 최소 3개 특성 필요
                    X_train = df_pred.loc[train_mask, feature_cols]
                    y_train = df_pred.loc[train_mask, col]
                    
                    # Random Forest 분류기 학습
                    rf_classifier = RandomForestClassifier(
                        n_estimators=50, 
                        random_state=42,
                        max_depth=10
                    )
                    rf_classifier.fit(X_train, y_train)
                    
                    # 결측치 예측
                    missing_mask = df_pred[col].isnull()
                    if missing_mask.sum() > 0:
                        X_missing = df_pred.loc[missing_mask, feature_cols]
                        predicted_values = rf_classifier.predict(X_missing)
                        df_pred.loc[missing_mask, col] = predicted_values
                        
                        processed_cols.append(col)
                        print(f"   ✅ {col}: {missing_mask.sum()}개 결측치 예측 대체")
    
    print(f"\n📊 예측 기반 처리 완료:")
    print(f"   KNN 대체: {len(numeric_cols)}개 수치형 변수")
    print(f"   RF 예측 대체: {len(processed_cols)}개 범주형 변수")
    
    final_missing = df_pred.isnull().sum().sum()
    original_missing = df.isnull().sum().sum()
    
    print(f"   처리 전 총 결측치: {original_missing:,}개")
    print(f"   처리 후 총 결측치: {final_missing:,}개")
    print(f"   처리 완료율: {((original_missing - final_missing) / original_missing * 100):.1f}%")
    
    return df_pred

# 예측 기반 대체 실행 (시간이 오래 걸릴 수 있음)
print("⚠️ 예측 기반 대체는 시간이 소요될 수 있습니다...")
predictive_imputed_data = predictive_imputation(train_data)
```

**🔍 코드 해설:**
- `KNNImputer(n_neighbors=5)`: 가장 유사한 5개 데이터의 평균값으로 결측치 대체
- `RandomForestClassifier()`: 다른 변수들을 사용해 결측치가 있는 변수의 값을 예측
- `fit()`: 모델 학습, `predict()`: 예측 수행

이 방법은 가장 정교하지만 계산 시간이 오래 걸리고, 때로는 과적합의 위험이 있습니다.

---

## 📖 4.1.3 이상치 탐지와 처리

결측치 처리를 마쳤으니, 이제 데이터의 또 다른 도전 과제인 **이상치(Outliers)**를 다뤄보겠습니다. 이상치는 다른 데이터와 현저히 다른 값을 가진 관측치로, 모델의 성능에 큰 영향을 미칠 수 있습니다.

### 이상치의 정의와 유형

> **🔍 주요 용어 해설**
> - **이상치(Outlier)**: 다른 관측치들과 현저히 다른 패턴을 보이는 데이터 포인트
> - **IQR (Interquartile Range)**: 제3사분위수에서 제1사분위수를 뺀 값 (중간 50% 데이터의 범위)
> - **Z-Score**: 데이터가 평균으로부터 표준편차의 몇 배 떨어져 있는지를 나타내는 지표

이상치는 발생 원인에 따라 다음과 같이 분류됩니다:

1. **측정 오류 이상치**: 데이터 입력 실수, 센서 오류 등으로 발생
2. **자연 발생 이상치**: 실제로 존재하는 극단적인 케이스 
3. **처리 오류 이상치**: 데이터 처리 과정에서 발생한 오류

### 이상치 탐지 방법 구현

```python
# 이상치 탐지 종합 함수
def detect_outliers(df, columns=None):
    """
    다양한 방법으로 이상치를 탐지하는 종합 함수
    
    Parameters:
    df (pd.DataFrame): 분석할 데이터프레임
    columns (list): 분석할 컬럼 리스트 (None이면 모든 수치형 컬럼)
    
    Returns:
    dict: 각 방법별 이상치 탐지 결과
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        # 타겟 변수 제외
        if 'SalePrice' in columns:
            columns.remove('SalePrice')
    
    outlier_results = {}
    
    print("🔍 이상치 탐지 방법별 결과:\n")
    
    for col in columns[:5]:  # 처음 5개 변수만 분석 (예시)
        if col in df.columns:
            print(f"📊 {col} 변수 이상치 분석:")
            
            # 기본 통계
            col_data = df[col].dropna()
            mean_val = col_data.mean()
            std_val = col_data.std()
            median_val = col_data.median()
            
            # 1. IQR 방법
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            iqr_outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            
            # 2. Z-Score 방법 (절댓값 3 이상)
            z_scores = np.abs((col_data - mean_val) / std_val)
            zscore_outliers = col_data[z_scores > 3]
            
            # 3. Modified Z-Score 방법 (중앙값 기반)
            median_absolute_deviation = np.median(np.abs(col_data - median_val))
            modified_z_scores = 0.6745 * (col_data - median_val) / median_absolute_deviation
            modified_zscore_outliers = col_data[np.abs(modified_z_scores) > 3.5]
            
            # 결과 저장
            outlier_results[col] = {
                'iqr_outliers': iqr_outliers,
                'zscore_outliers': zscore_outliers,
                'modified_zscore_outliers': modified_zscore_outliers,
                'bounds': {'lower': lower_bound, 'upper': upper_bound}
            }
            
            print(f"   IQR 방법: {len(iqr_outliers)}개 이상치 (범위: {lower_bound:.1f} ~ {upper_bound:.1f})")
            print(f"   Z-Score 방법: {len(zscore_outliers)}개 이상치")
            print(f"   Modified Z-Score 방법: {len(modified_zscore_outliers)}개 이상치")
            print()
    
    return outlier_results

# 이상치 탐지 실행
outlier_results = detect_outliers(train_data)
```

**🔍 코드 해설:**
- `quantile(0.25)`, `quantile(0.75)`: 제1사분위수와 제3사분위수 계산
- `1.5 * IQR`: 일반적으로 사용되는 이상치 판별 기준
- `np.abs(z_scores) > 3`: Z-Score의 절댓값이 3보다 큰 경우를 이상치로 판별

### 이상치 시각화

이상치를 시각적으로 확인하여 패턴을 파악해보겠습니다.

```python
# 이상치 시각화 함수
def visualize_outliers(df, columns=None, figsize=(15, 12)):
    """
    이상치를 다양한 방법으로 시각화하는 함수
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'SalePrice' in numeric_cols:
            numeric_cols.remove('SalePrice')
        columns = numeric_cols[:6]  # 상위 6개 변수
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('🔍 이상치 시각화 분석', fontsize=16, fontweight='bold')
    
    for i, col in enumerate(columns):
        if i >= 6:  # 최대 6개만 표시
            break
            
        row = i // 3
        col_idx = i % 3
        
        if col in df.columns:
            # 박스플롯으로 이상치 시각화
            df[col].dropna().plot(kind='box', ax=axes[row, col_idx])
            axes[row, col_idx].set_title(f'{col}\n이상치 탐지', fontsize=10)
            axes[row, col_idx].grid(True, alpha=0.3)
    
    # 빈 서브플롯 제거
    for i in range(len(columns), 6):
        row = i // 3
        col_idx = i % 3
        axes[row, col_idx].remove()
    
    plt.tight_layout()
    plt.show()
    
    # 상세 분석: 대표 변수 하나 선택
    if 'GrLivArea' in df.columns:
        print("📊 상세 분석: GrLivArea (지상층 거주 면적)")
        
        plt.figure(figsize=(15, 5))
        
        # 히스토그램
        plt.subplot(1, 3, 1)
        df['GrLivArea'].hist(bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('거주 면적 분포')
        plt.xlabel('면적 (sq ft)')
        plt.ylabel('빈도')
        plt.grid(True, alpha=0.3)
        
        # 박스플롯
        plt.subplot(1, 3, 2)
        df['GrLivArea'].plot(kind='box')
        plt.title('거주 면적 박스플롯')
        plt.ylabel('면적 (sq ft)')
        plt.grid(True, alpha=0.3)
        
        # 산점도 (SalePrice와의 관계)
        plt.subplot(1, 3, 3)
        if 'SalePrice' in df.columns:
            plt.scatter(df['GrLivArea'], df['SalePrice'], alpha=0.6, color='coral')
            plt.title('거주 면적 vs 판매 가격')
            plt.xlabel('거주 면적 (sq ft)')
            plt.ylabel('판매 가격 ($)')
            plt.grid(True, alpha=0.3)
            
            # 이상치로 보이는 포인트 강조
            outlier_threshold = df['GrLivArea'].quantile(0.99)
            outlier_mask = df['GrLivArea'] > outlier_threshold
            if outlier_mask.sum() > 0:
                plt.scatter(df.loc[outlier_mask, 'GrLivArea'], 
                           df.loc[outlier_mask, 'SalePrice'], 
                           color='red', s=100, alpha=0.8, 
                           label=f'상위 1% 이상치 ({outlier_mask.sum()}개)')
                plt.legend()
        
        plt.tight_layout()
        plt.show()

# 이상치 시각화 실행
visualize_outliers(train_data)
```

> **📊 이미지 생성 프롬프트:**  
> "Create a comprehensive outlier detection visualization dashboard with: 1) A 2x3 grid of box plots showing outliers in different numerical variables (like GrLivArea, LotArea, TotalBsmtSF, etc.) from a house prices dataset, 2) A detailed analysis section with three plots for GrLivArea: histogram showing distribution with outliers highlighted, box plot with outlier points clearly marked, and scatter plot of GrLivArea vs SalePrice with extreme outliers highlighted in red. Use professional styling with clear labels, gridlines, and color coding to distinguish normal data points from outliers."

### 이상치 처리 전략

이상치를 탐지했으니, 이제 이를 어떻게 처리할지 결정해야 합니다.

```python
# 이상치 처리 방법들
def handle_outliers(df, method='iqr', columns=None):
    """
    다양한 방법으로 이상치를 처리하는 함수
    
    Parameters:
    method (str): 'remove', 'cap', 'transform', 'keep'
    """
    df_processed = df.copy()
    
    if columns is None:
        columns = ['GrLivArea', 'LotArea', 'TotalBsmtSF']  # 예시 컬럼들
    
    print(f"🔧 이상치 처리 방법: {method}")
    
    for col in columns:
        if col in df_processed.columns:
            col_data = df_processed[col].dropna()
            
            # IQR 기반 이상치 경계 계산
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 이상치 개수 확인
            outliers_before = ((df_processed[col] < lower_bound) | 
                              (df_processed[col] > upper_bound)).sum()
            
            if method == 'remove':
                # 방법 1: 이상치 제거
                mask = (df_processed[col] >= lower_bound) & (df_processed[col] <= upper_bound)
                df_processed = df_processed[mask | df_processed[col].isnull()]
                outliers_after = 0
                
            elif method == 'cap':
                # 방법 2: 이상치 캡핑 (경계값으로 제한)
                df_processed[col] = df_processed[col].clip(lower=lower_bound, upper=upper_bound)
                outliers_after = 0
                
            elif method == 'transform':
                # 방법 3: 로그 변환 (양의 값만 가능)
                if (df_processed[col] > 0).all():
                    df_processed[col + '_log'] = np.log1p(df_processed[col])
                    print(f"   {col}: 로그 변환 적용 ({col}_log 컬럼 생성)")
                outliers_after = outliers_before  # 변환은 제거가 아님
                
            elif method == 'winsorize':
                # 방법 4: 윈저화 (상하위 5%를 해당 분위수 값으로 대체)
                from scipy.stats import mstats
                df_processed[col] = mstats.winsorize(df_processed[col], limits=[0.05, 0.05])
                outliers_after = 0
                
            else:  # method == 'keep'
                # 방법 5: 이상치 유지 (분석 목적)
                outliers_after = outliers_before
            
            print(f"   {col}: {outliers_before}개 → {outliers_after}개")
    
    print(f"\n📊 처리 결과:")
    print(f"   원본 데이터 크기: {df.shape}")
    print(f"   처리 후 데이터 크기: {df_processed.shape}")
    
    return df_processed

# 각 방법별 이상치 처리 비교
methods = ['keep', 'cap', 'remove', 'transform']
processed_data = {}

for method in methods:
    print(f"\n{'='*50}")
    processed_data[method] = handle_outliers(train_data, method=method)
```

**🔍 코드 해설:**
- `clip(lower, upper)`: 값을 지정된 범위로 제한하는 함수
- `np.log1p()`: log(1+x) 변환으로 0을 포함한 양수에 안전하게 적용
- `mstats.winsorize()`: 극단값을 특정 분위수 값으로 대체

### 비즈니스 맥락에서의 이상치 해석

```python
# 이상치의 비즈니스적 의미 분석
def analyze_outlier_business_impact(df):
    """
    이상치가 비즈니스 관점에서 어떤 의미를 가지는지 분석
    """
    print("🏡 이상치의 비즈니스적 해석:")
    
    if 'GrLivArea' in df.columns and 'SalePrice' in df.columns:
        # 거주 면적 이상치 분석
        area_q99 = df['GrLivArea'].quantile(0.99)
        large_houses = df[df['GrLivArea'] > area_q99]
        
        print(f"\n🏠 대형 주택 분석 (상위 1%, 면적 > {area_q99:.0f} sq ft):")
        print(f"   대형 주택 수: {len(large_houses)}개")
        
        if len(large_houses) > 0:
            avg_price_large = large_houses['SalePrice'].mean()
            avg_price_normal = df[df['GrLivArea'] <= area_q99]['SalePrice'].mean()
            
            print(f"   대형 주택 평균 가격: ${avg_price_large:,.0f}")
            print(f"   일반 주택 평균 가격: ${avg_price_normal:,.0f}")
            print(f"   가격 프리미엄: {((avg_price_large / avg_price_normal - 1) * 100):.1f}%")
            
            # 대형 주택의 특성 분석
            if 'Neighborhood' in df.columns:
                print(f"\n   대형 주택이 많은 지역:")
                neighborhood_dist = large_houses['Neighborhood'].value_counts().head(3)
                for neighborhood, count in neighborhood_dist.items():
                    pct = (count / len(large_houses)) * 100
                    print(f"   - {neighborhood}: {count}개 ({pct:.1f}%)")
    
    # 가격 이상치 분석
    if 'SalePrice' in df.columns:
        price_q99 = df['SalePrice'].quantile(0.99)
        price_q01 = df['SalePrice'].quantile(0.01)
        
        expensive_houses = df[df['SalePrice'] > price_q99]
        cheap_houses = df[df['SalePrice'] < price_q01]
        
        print(f"\n💰 가격 이상치 분석:")
        print(f"   고가 주택 (상위 1%): {len(expensive_houses)}개 (${price_q99:,.0f} 이상)")
        print(f"   저가 주택 (하위 1%): {len(cheap_houses)}개 (${price_q01:,.0f} 이하)")
        
        # 고가 주택의 특징
        if len(expensive_houses) > 0 and 'OverallQual' in df.columns:
            avg_quality_expensive = expensive_houses['OverallQual'].mean()
            avg_quality_normal = df[(df['SalePrice'] >= price_q01) & 
                                   (df['SalePrice'] <= price_q99)]['OverallQual'].mean()
            
            print(f"   고가 주택 평균 품질: {avg_quality_expensive:.1f}/10")
            print(f"   일반 주택 평균 품질: {avg_quality_normal:.1f}/10")

# 비즈니스 영향 분석 실행
analyze_outlier_business_impact(train_data)
```

---

## 🎯 직접 해보기 - 연습 문제

이제 배운 내용을 실제로 적용해보는 연습 문제를 풀어보겠습니다.

### 연습 문제 1: 결측치 패턴 분석 ⭐⭐
다음 코드를 완성하여 결측치 패턴을 분석해보세요.

```python
# 연습 문제 1: 결측치 패턴 분석
def exercise_missing_pattern(df):
    """
    결측치 패턴을 분석하고 처리 전략을 수립하는 함수
    TODO: 빈 부분을 채워보세요
    """
    
    # 1. 결측치가 50% 이상인 변수 찾기
    high_missing_vars = # TODO: 여기를 완성하세요
    
    # 2. 지하실 관련 변수들의 결측치 상관관계 확인
    basement_vars = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1']
    basement_corr = # TODO: 결측치 상관관계 계산
    
    # 3. 처리 전략 출력
    print("📋 결측치 처리 전략:")
    print(f"높은 결측치 변수 ({len(high_missing_vars)}개): 삭제 고려")
    print(f"지하실 변수 상관관계: {basement_corr:.3f}")
    
    return high_missing_vars

# 힌트: isnull(), sum(), corr() 함수들을 활용하세요
```

### 연습 문제 2: 맞춤형 결측치 처리 ⭐⭐⭐
도메인 지식을 활용한 결측치 처리 함수를 작성해보세요.

```python
# 연습 문제 2: 맞춤형 결측치 처리
def exercise_custom_imputation(df):
    """
    부동산 도메인 지식을 활용한 맞춤형 결측치 처리
    """
    df_processed = df.copy()
    
    # TODO: 다음 규칙에 따라 결측치를 처리하세요
    # 1. 수영장 품질(PoolQC)이 결측이면 'No Pool'로 대체
    # 2. 차고 건설연도(GarageYrBlt)가 결측이면 주택 건설연도(YearBuilt)로 대체
    # 3. 지하실 면적(TotalBsmtSF)이 결측이면 0으로 대체
    
    # 여기에 코드를 작성하세요
    
    return df_processed
```

### 연습 문제 3: 이상치 영향 분석 ⭐⭐⭐
이상치가 주택 가격 예측에 미치는 영향을 분석해보세요.

```python
# 연습 문제 3: 이상치 영향 분석
def exercise_outlier_impact(df):
    """
    이상치가 가격 예측에 미치는 영향 분석
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    
    # TODO: 다음 단계를 수행하세요
    # 1. GrLivArea를 사용한 단순 선형 회귀 모델 구축
    # 2. 이상치 제거 전후의 예측 성능(RMSE) 비교
    # 3. 결과 해석 및 권고사항 제시
    
    # 힌트: IQR 방법으로 이상치를 정의하고,
    # 이상치 포함/제외 데이터로 각각 모델을 학습해보세요
    
    pass
```

---

## 📚 핵심 정리

이번 Part에서 배운 핵심 내용을 정리하면 다음과 같습니다:

### ✅ 결측치 처리 핵심 포인트

1. **결측치 유형 이해**: MCAR, MAR, MNAR 구분하여 적절한 처리 방법 선택
2. **구조적 vs 임의적 결측**: 도메인 지식으로 결측치의 의미 파악
3. **처리 방법 선택**: 
   - 단순 삭제 ← 데이터 손실 위험
   - 통계적 대체 ← 빠르고 안전한 기본 방법  
   - 도메인 기반 대체 ← 가장 정확하고 의미있는 방법
   - 예측 기반 대체 ← 정교하지만 복잡하고 과적합 위험

### ✅ 이상치 처리 핵심 포인트

1. **탐지 방법**: IQR, Z-Score, Modified Z-Score 등 다양한 방법 병행
2. **비즈니스 해석**: 이상치가 오류인지 실제 극단값인지 판단
3. **처리 전략**:
   - 제거 ← 명확한 오류인 경우
   - 캡핑/윈저화 ← 극단값 영향 완화
   - 변환 ← 분포 정규화 
   - 유지 ← 중요한 정보인 경우

### 💡 실무 적용 팁

- **항상 원본 데이터 보존**: 처리 과정을 되돌릴 수 있도록 복사본 사용
- **처리 효과 검증**: 처리 전후 데이터 분포와 모델 성능 비교
- **문서화**: 처리 이유와 방법을 명확히 기록
- **단계적 접근**: 한 번에 모든 문제를 해결하려 하지 말고 단계별 처리

---

## 🤔 생각해보기

1. **구조적 결측치의 정보적 가치**: 지하실이나 차고가 없는 주택의 정보가 어떻게 유용할 수 있을까요? 이런 정보를 새로운 특성으로 활용하는 방법은 무엇일까요?

2. **이상치의 양면성**: 매우 큰 주택이나 매우 비싼 주택이 이상치로 탐지되었을 때, 이를 제거하는 것이 항상 옳은 선택일까요? 어떤 경우에 이상치를 보존해야 할까요?

3. **처리 방법의 트레이드오프**: 정교한 예측 기반 대체 방법이 항상 단순한 평균값 대체보다 나은 결과를 보장할까요? 각 방법의 장단점을 실무 관점에서 생각해보세요.

---

## 🔜 다음 Part 예고: 데이터 변환과 정규화

다음 Part에서는 전처리된 데이터를 머신러닝 알고리즘에 적합하도록 변환하는 기법들을 배웁니다:

- **스케일링 기법**: 변수 간 크기 차이 해결 (StandardScaler, MinMaxScaler, RobustScaler)
- **인코딩 기법**: 범주형 변수를 수치형으로 변환 (One-Hot, Label, Target Encoding)  
- **분포 변환**: 왜도 제거와 정규분포 근사 (로그변환, Box-Cox 변환)
- **실전 파이프라인**: scikit-learn Pipeline을 활용한 체계적 전처리 워크플로우

데이터의 품질을 높여 모델의 성능을 극대화하는 전문적인 변환 기법들을 마스터해보겠습니다!

---

*"좋은 데이터 전처리는 좋은 모델의 첫걸음입니다. 결측치와 이상치를 단순한 문제가 아닌 데이터가 전하는 정보로 해석하는 안목을 기르는 것이 중요합니다."*

