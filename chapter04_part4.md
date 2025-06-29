# 4장 Part 4: AI 도구를 활용한 자동 전처리와 한계점
## 인공지능의 힘을 활용하면서도 그 한계를 이해하는 균형잡힌 접근

---

## 📚 학습 목표

이번 Part에서는 다음과 같은 내용을 학습합니다:

✅ **AI 자동 전처리 도구의 장점과 한계점을 명확히 이해할 수 있다**
✅ **AutoML 플랫폼의 전처리 기능을 효과적으로 활용할 수 있다**  
✅ **AI 기반 특성 생성 도구의 원리와 적용 방법을 익힐 수 있다**
✅ **지능형 데이터 클리닝의 작동 방식과 검증 방법을 학습할 수 있다**
✅ **인간-AI 협업 전처리 워크플로우를 설계하고 구현할 수 있다**

---

## 🎯 이번 Part 미리보기

**AI 기술의 발전**으로 데이터 전처리도 자동화의 시대를 맞이했습니다. AutoML 플랫폼부터 지능형 특성 생성 도구까지, 이제 AI가 데이터 전처리의 많은 부분을 대신 처리해줄 수 있습니다.

하지만 **"AI가 모든 것을 해결해준다"**는 생각은 위험합니다. AI 도구들은 분명 강력하지만, **블랙박스 문제**, **편향성 증폭**, **도메인 지식 부족** 등의 한계를 가지고 있습니다.

이번 Part에서는 **AI의 힘을 최대한 활용하면서도 그 한계를 정확히 이해**하여, 신뢰할 수 있는 전처리 파이프라인을 구축하는 방법을 배워보겠습니다. 

특히 **7장에서 학습한 AI 협업 기법**을 전처리 영역에 특화하여 적용하고, 실제 House Prices 데이터를 통해 **인간의 판단력과 AI의 효율성을 결합**한 최적의 워크플로우를 구현해보겠습니다.

> **💡 Part 4의 핵심 포인트**  
> "AI는 강력한 도구이지만 만능이 아닙니다. 인간의 지혜와 AI의 효율성을 조화롭게 결합할 때 최고의 결과를 얻을 수 있습니다."

---

## 📖 4.4.1 AutoML 전처리 도구 소개와 활용

### AutoML 전처리의 혁신

**AutoML(Automated Machine Learning)**은 머신러닝의 전 과정을 자동화하는 기술로, 전처리 영역에서도 혁신적인 변화를 가져왔습니다.

> **🔍 주요 용어 해설**
> - **AutoML**: 머신러닝 파이프라인의 자동화 기술
> - **Auto-preprocessing**: 데이터 전처리 과정의 자동화
> - **Meta-learning**: 과거 경험을 바탕으로 학습하는 AI 기법
> - **Pipeline optimization**: 전처리 단계들의 최적 조합 탐색

### 주요 AutoML 플랫폼 비교

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# AutoML 플랫폼 비교를 위한 데모 함수
def compare_automl_platforms():
    """
    주요 AutoML 플랫폼들의 전처리 기능 비교
    """
    print("🤖 주요 AutoML 플랫폼 전처리 기능 비교:")
    
    # 플랫폼별 특징 정리
    platforms = {
        'H2O AutoML': {
            'strengths': ['강력한 자동 특성 공학', '대용량 데이터 처리', '다양한 알고리즘 지원'],
            'preprocessing_features': ['자동 인코딩', '결측치 처리', '특성 선택', '스케일링'],
            'limitations': ['복잡한 설정', '해석성 제한', '커스터마이징 어려움'],
            'best_for': '대규모 정형 데이터, 빠른 프로토타이핑',
            'cost': 'Free + Enterprise 버전',
            'complexity': 'Medium-High'
        },
        'Google AutoML': {
            'strengths': ['클라우드 통합', '사용 편의성', '확장성'],
            'preprocessing_features': ['자동 데이터 검증', '특성 중요도', '자동 변환'],
            'limitations': ['비용', 'Google 생태계 종속', '제한적 커스터마이징'],
            'best_for': '클라우드 환경, 비개발자 친화적',
            'cost': '사용량 기반 과금',
            'complexity': 'Low'
        },
        'DataRobot': {
            'strengths': ['엔터프라이즈급 기능', '모델 해석성', '배포 지원'],
            'preprocessing_features': ['지능형 전처리', '자동 특성 발견', '데이터 품질 검사'],
            'limitations': ['높은 비용', '복잡한 인터페이스', '학습 곡선'],
            'best_for': '대기업, 복잡한 비즈니스 문제',
            'cost': 'Enterprise 라이선스',
            'complexity': 'High'
        },
        'AutoGluon': {
            'strengths': ['오픈소스', '다양한 데이터 타입 지원', 'AWS 통합'],
            'preprocessing_features': ['자동 전처리', '텍스트/이미지 지원', '앙상블 최적화'],
            'limitations': ['상대적으로 새로운 플랫폼', '커뮤니티 크기', '문서화'],
            'best_for': '연구, 프로토타이핑, AWS 환경',
            'cost': 'Free (오픈소스)',
            'complexity': 'Medium'
        },
        'TPOT': {
            'strengths': ['파이썬 네이티브', '유전 알고리즘', '투명성'],
            'preprocessing_features': ['파이프라인 최적화', 'scikit-learn 기반', '코드 생성'],
            'limitations': ['느린 속도', '제한적 스케일링', '메모리 사용량'],
            'best_for': '연구, 교육, 작은 데이터셋',
            'cost': 'Free (오픈소스)',
            'complexity': 'Medium'
        }
    }
    
    # 비교표 출력
    print(f"\n📊 플랫폼별 상세 비교:")
    for platform, features in platforms.items():
        print(f"\n🔹 {platform}")
        print(f"   💪 장점: {', '.join(features['strengths'])}")
        print(f"   ⚙️  전처리 기능: {', '.join(features['preprocessing_features'])}")
        print(f"   ⚠️  한계점: {', '.join(features['limitations'])}")
        print(f"   🎯 최적 용도: {features['best_for']}")
        print(f"   💰 비용: {features['cost']}")
        print(f"   📈 복잡도: {features['complexity']}")
    
    return platforms

# 플랫폼 비교 실행
platform_comparison = compare_automl_platforms()
```

**🔍 코드 해설:**
- 5가지 주요 AutoML 플랫폼의 특징을 체계적으로 정리
- 각 플랫폼의 장단점과 적용 시나리오를 명확히 구분
- 비용과 복잡도까지 고려한 실무적 선택 가이드 제공

### 실제 AutoML 전처리 시뮬레이션

```python
# AutoML 스타일 자동 전처리 시뮬레이션
class AutoPreprocessor:
    """
    AutoML 도구의 전처리 과정을 시뮬레이션하는 클래스
    """
    
    def __init__(self, aggressive_mode=False):
        self.aggressive_mode = aggressive_mode  # 공격적 전처리 모드
        self.preprocessing_steps = []
        self.feature_importance = {}
        self.warnings = []
    
    def analyze_data_quality(self, df):
        """데이터 품질 자동 분석"""
        print("🔍 자동 데이터 품질 분석:")
        
        quality_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_data': {},
            'data_types': {},
            'potential_issues': []
        }
        
        # 결측치 분석
        for col in df.columns:
            missing_pct = df[col].isnull().sum() / len(df) * 100
            if missing_pct > 0:
                quality_report['missing_data'][col] = missing_pct
                
                if missing_pct > 50:
                    quality_report['potential_issues'].append(f"{col}: 50% 이상 결측치")
                elif missing_pct > 20:
                    quality_report['potential_issues'].append(f"{col}: 20% 이상 결측치")
        
        # 데이터 타입 분석
        for col in df.columns:
            dtype = str(df[col].dtype)
            quality_report['data_types'][col] = dtype
            
            # 잠재적 문제 탐지
            if dtype == 'object':
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio > 0.95:  # 거의 모든 값이 고유한 경우
                    quality_report['potential_issues'].append(f"{col}: 높은 카디널리티 (ID 컬럼 가능성)")
                elif unique_ratio < 0.05:  # 매우 적은 고유값
                    quality_report['potential_issues'].append(f"{col}: 매우 낮은 다양성")
        
        # 결과 출력
        print(f"   📊 데이터 크기: {quality_report['total_rows']:,}행 × {quality_report['total_columns']}열")
        print(f"   ❌ 결측치 있는 컬럼: {len(quality_report['missing_data'])}개")
        print(f"   ⚠️  잠재적 문제: {len(quality_report['potential_issues'])}개")
        
        if quality_report['potential_issues']:
            print(f"\n   📋 발견된 문제점들:")
            for issue in quality_report['potential_issues'][:5]:  # 상위 5개만 표시
                print(f"      • {issue}")
        
        return quality_report
    
    def auto_missing_value_strategy(self, df, column):
        """결측치 처리 전략 자동 결정"""
        missing_pct = df[column].isnull().sum() / len(df) * 100
        dtype = df[column].dtype
        
        if missing_pct == 0:
            return 'no_action', 'No missing values'
        elif missing_pct > 70:
            return 'drop_column', f'Too many missing values ({missing_pct:.1f}%)'
        elif dtype in ['int64', 'float64']:
            if missing_pct < 5:
                return 'median_impute', 'Low missing rate - use median'
            elif missing_pct < 20:
                return 'mean_impute', 'Medium missing rate - use mean'
            else:
                return 'mode_impute', 'High missing rate - use mode'
        else:  # object dtype
            if missing_pct < 10:
                return 'mode_impute', 'Use most frequent value'
            else:
                return 'constant_impute', 'Use constant value (Unknown)'
    
    def auto_feature_engineering(self, df, target_col=None):
        """자동 특성 공학"""
        print("\n🛠️ 자동 특성 공학:")
        
        df_processed = df.copy()
        new_features = []
        
        # 날짜 특성 자동 감지 및 추출
        for col in df.columns:
            if 'year' in col.lower() or 'date' in col.lower():
                if df[col].dtype in ['int64', 'float64'] and df[col].max() > 1900:
                    # 연도로 추정되는 컬럼
                    current_year = 2023
                    age_col = f"{col}_Age"
                    df_processed[age_col] = current_year - df[col]
                    new_features.append(age_col)
                    print(f"   ✅ {age_col}: {col}로부터 연령 계산")
        
        # 수치형 특성 조합 자동 생성
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col and target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        if len(numeric_cols) >= 2:
            # 상위 상관관계 특성들 조합
            if target_col:
                correlations = df[numeric_cols].corrwith(df[target_col]).abs().sort_values(ascending=False)
                top_features = correlations.head(3).index.tolist()
            else:
                top_features = numeric_cols[:3]
            
            # 비율 특성 생성
            if len(top_features) >= 2:
                ratio_col = f"{top_features[0]}_per_{top_features[1]}"
                df_processed[ratio_col] = df_processed[top_features[0]] / (df_processed[top_features[1]] + 1)
                new_features.append(ratio_col)
                print(f"   ✅ {ratio_col}: 비율 특성 자동 생성")
        
        # 범주형 특성 빈도 인코딩
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            if df[col].nunique() > 2 and df[col].nunique() < 20:  # 적절한 카디널리티
                freq_col = f"{col}_Frequency"
                freq_map = df[col].value_counts().to_dict()
                df_processed[freq_col] = df[col].map(freq_map)
                new_features.append(freq_col)
                print(f"   ✅ {freq_col}: 빈도 인코딩 적용")
        
        print(f"\n   📈 총 {len(new_features)}개 새로운 특성 생성")
        return df_processed, new_features
    
    def auto_preprocessing_pipeline(self, df, target_col=None):
        """완전 자동 전처리 파이프라인"""
        print("🤖 완전 자동 전처리 파이프라인 실행:")
        
        # 1단계: 데이터 품질 분석
        quality_report = self.analyze_data_quality(df)
        
        # 2단계: 결측치 처리
        print("\n🔧 자동 결측치 처리:")
        df_processed = df.copy()
        
        for col in df.columns:
            if col == target_col:
                continue
                
            strategy, reason = self.auto_missing_value_strategy(df, col)
            
            if strategy == 'drop_column':
                df_processed = df_processed.drop(columns=[col])
                print(f"   ❌ {col}: 컬럼 제거 ({reason})")
                self.warnings.append(f"컬럼 {col} 제거됨: {reason}")
                
            elif strategy == 'median_impute':
                df_processed[col].fillna(df_processed[col].median(), inplace=True)
                print(f"   📊 {col}: 중앙값으로 대체")
                
            elif strategy == 'mean_impute':
                df_processed[col].fillna(df_processed[col].mean(), inplace=True)
                print(f"   📊 {col}: 평균값으로 대체")
                
            elif strategy == 'mode_impute':
                df_processed[col].fillna(df_processed[col].mode().iloc[0], inplace=True)
                print(f"   📊 {col}: 최빈값으로 대체")
                
            elif strategy == 'constant_impute':
                df_processed[col].fillna('Unknown', inplace=True)
                print(f"   📊 {col}: 'Unknown'으로 대체")
        
        # 3단계: 자동 특성 공학
        df_processed, new_features = self.auto_feature_engineering(df_processed, target_col)
        
        # 4단계: 인코딩 및 스케일링
        print("\n🔄 자동 인코딩 및 스케일링:")
        
        # 범주형 변수 인코딩
        categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
        
        for col in categorical_cols:
            if df_processed[col].nunique() <= 10:  # 낮은 카디널리티
                # 원-핫 인코딩
                dummies = pd.get_dummies(df_processed[col], prefix=col, drop_first=True)
                df_processed = pd.concat([df_processed, dummies], axis=1)
                df_processed = df_processed.drop(columns=[col])
                print(f"   🔄 {col}: 원-핫 인코딩 적용")
            else:  # 높은 카디널리티
                # 라벨 인코딩
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                print(f"   🔄 {col}: 라벨 인코딩 적용")
                self.warnings.append(f"높은 카디널리티로 인해 {col}에 라벨 인코딩 사용")
        
        # 5단계: 최종 검증
        print(f"\n✅ 자동 전처리 완료:")
        print(f"   📊 최종 데이터 크기: {df_processed.shape[0]:,}행 × {df_processed.shape[1]}열")
        print(f"   🆕 생성된 특성: {len(new_features)}개")
        print(f"   ⚠️  경고사항: {len(self.warnings)}개")
        
        if self.warnings:
            print(f"\n   📋 주요 경고사항:")
            for warning in self.warnings[:3]:
                print(f"      • {warning}")
        
        return df_processed, {
            'original_shape': df.shape,
            'final_shape': df_processed.shape,
            'new_features': new_features,
            'warnings': self.warnings,
            'quality_report': quality_report
        }

# House Prices 데이터로 AutoML 전처리 시뮬레이션
try:
    # 데이터 로드
    train_data = pd.read_csv('datasets/house_prices/train.csv')
    print("✅ House Prices 데이터 로드 성공!")
    
    # AutoML 전처리 실행
    auto_processor = AutoPreprocessor(aggressive_mode=False)
    processed_data, processing_report = auto_processor.auto_preprocessing_pipeline(
        train_data, target_col='SalePrice'
    )
    
    print(f"\n📋 전처리 요약 보고서:")
    print(f"   원본: {processing_report['original_shape'][0]:,}행 × {processing_report['original_shape'][1]}열")
    print(f"   처리 후: {processing_report['final_shape'][0]:,}행 × {processing_report['final_shape'][1]}열")
    print(f"   변화: {processing_report['final_shape'][1] - processing_report['original_shape'][1]:+}열")
    
except FileNotFoundError:
    print("❌ 데이터 파일을 찾을 수 없습니다.")
    print("💡 House Prices Dataset을 다운로드하여 datasets/house_prices/ 폴더에 저장하세요.")
    
    # 예시 데이터로 시뮬레이션
    print("\n🔄 예시 데이터로 AutoML 전처리 시뮬레이션:")
    
    # 가상의 부동산 데이터 생성
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = pd.DataFrame({
        'GrLivArea': np.random.normal(1500, 500, n_samples),
        'YearBuilt': np.random.randint(1950, 2020, n_samples),
        'BedroomAbvGr': np.random.randint(1, 6, n_samples),
        'Neighborhood': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_samples),
        'SalePrice': np.random.normal(200000, 80000, n_samples)
    })
    
    # 의도적으로 결측치 추가
    sample_data.loc[np.random.choice(n_samples, 100, replace=False), 'GrLivArea'] = np.nan
    sample_data.loc[np.random.choice(n_samples, 50, replace=False), 'Neighborhood'] = np.nan
    
    auto_processor = AutoPreprocessor()
    processed_data, processing_report = auto_processor.auto_preprocessing_pipeline(
        sample_data, target_col='SalePrice'
    )
```

**🔍 코드 해설:**
- `AutoPreprocessor` 클래스로 AutoML 도구의 전처리 과정을 완전 시뮬레이션
- 데이터 품질 자동 분석부터 결측치 처리, 특성 공학, 인코딩까지 전 과정 자동화
- 각 단계별 결정 논리와 경고사항까지 포함한 실무적 구현

> **📊 이미지 생성 프롬프트:**  
> "Create a comprehensive AutoML preprocessing workflow visualization showing: 1) A flowchart of automated preprocessing steps (Data Quality Analysis → Missing Value Handling → Feature Engineering → Encoding & Scaling → Validation), 2) A comparison matrix of 5 major AutoML platforms with their strengths, limitations, and use cases, 3) Before and after data transformation metrics, 4) Warning and quality indicators. Use professional styling with clear icons for each step and platform logos."

---

## 📖 4.4.2 AI 기반 특성 생성 도구

### 자동 특성 공학의 혁신

AI 기반 특성 생성은 **Featuretools**, **AutoFeat**, **OneBM** 등의 도구를 통해 인간이 생각하기 어려운 복잡한 특성들을 자동으로 발견합니다.

```python
# AI 기반 특성 생성 도구 시뮬레이션
class AIFeatureGenerator:
    """
    AI 기반 자동 특성 생성을 시뮬레이션하는 클래스
    """
    
    def __init__(self):
        self.generated_features = []
        self.feature_scores = {}
        self.generation_strategies = [
            'mathematical_combinations',
            'temporal_features', 
            'statistical_features',
            'interaction_features',
            'polynomial_features'
        ]
    
    def mathematical_combinations(self, df, max_features=20):
        """수학적 조합 기반 특성 생성"""
        print("🔢 수학적 조합 특성 자동 생성:")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        new_features = {}
        
        if len(numeric_cols) >= 2:
            # 모든 수치형 컬럼 쌍에 대해 수학적 조합 시도
            feature_count = 0
            
            for i, col1 in enumerate(numeric_cols):
                for j, col2 in enumerate(numeric_cols[i+1:], i+1):
                    if feature_count >= max_features:
                        break
                    
                    # 더하기
                    add_name = f"ADD_{col1}_{col2}"
                    new_features[add_name] = df[col1] + df[col2]
                    
                    # 빼기
                    sub_name = f"SUB_{col1}_{col2}"
                    new_features[sub_name] = df[col1] - df[col2]
                    
                    # 곱하기
                    mul_name = f"MUL_{col1}_{col2}"
                    new_features[mul_name] = df[col1] * df[col2]
                    
                    # 나누기 (0으로 나누기 방지)
                    div_name = f"DIV_{col1}_{col2}"
                    new_features[div_name] = df[col1] / (df[col2] + 1e-8)
                    
                    feature_count += 4
                    
                    if feature_count >= max_features:
                        break
        
        print(f"   ✅ {len(new_features)}개 수학적 조합 특성 생성")
        return new_features
    
    def statistical_features(self, df, window_sizes=[3, 5, 7]):
        """통계적 특성 자동 생성"""
        print("📊 통계적 특성 자동 생성:")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        new_features = {}
        
        for col in numeric_cols:
            # 기본 통계량
            new_features[f"ZSCORE_{col}"] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
            new_features[f"RANK_{col}"] = df[col].rank(pct=True)
            new_features[f"SQUARED_{col}"] = df[col] ** 2
            new_features[f"SQRT_{col}"] = np.sqrt(np.abs(df[col]))
            new_features[f"LOG_{col}"] = np.log1p(np.abs(df[col]))
            
            # 롤링 통계량 (데이터가 시계열이라고 가정)
            for window in window_sizes:
                if len(df) > window:
                    new_features[f"ROLLING_MEAN_{col}_{window}"] = df[col].rolling(window).mean()
                    new_features[f"ROLLING_STD_{col}_{window}"] = df[col].rolling(window).std()
                    new_features[f"ROLLING_MIN_{col}_{window}"] = df[col].rolling(window).min()
                    new_features[f"ROLLING_MAX_{col}_{window}"] = df[col].rolling(window).max()
        
        print(f"   ✅ {len(new_features)}개 통계적 특성 생성")
        return new_features
    
    def interaction_features(self, df, target_col=None, top_k=5):
        """상호작용 특성 자동 발견"""
        print("🔗 상호작용 특성 자동 발견:")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col and target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        new_features = {}
        
        if target_col and len(numeric_cols) >= 2:
            # 타겟과의 상관관계 기반으로 상위 특성 선별
            correlations = df[numeric_cols].corrwith(df[target_col]).abs().sort_values(ascending=False)
            top_features = correlations.head(top_k).index.tolist()
            
            # 상위 특성들 간의 상호작용 생성
            for i, col1 in enumerate(top_features):
                for j, col2 in enumerate(top_features[i+1:], i+1):
                    # 곱셈 상호작용
                    interaction_name = f"INTERACT_{col1}_{col2}"
                    new_features[interaction_name] = df[col1] * df[col2]
                    
                    # 조건부 특성
                    greater_name = f"GREATER_{col1}_{col2}"
                    new_features[greater_name] = (df[col1] > df[col2]).astype(int)
        
        print(f"   ✅ {len(new_features)}개 상호작용 특성 생성")
        return new_features
    
    def evaluate_feature_importance(self, df, features_dict, target_col):
        """생성된 특성의 중요도 평가"""
        print("\n⚖️ 특성 중요도 자동 평가:")
        
        if target_col not in df.columns:
            print("   ⚠️ 타겟 변수가 없어 중요도 평가를 건너뜁니다.")
            return {}
        
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.feature_selection import mutual_info_regression
        
        feature_scores = {}
        
        # 각 특성에 대해 개별적으로 중요도 계산
        for feature_name, feature_values in features_dict.items():
            try:
                # 결측치 제거
                valid_mask = ~(pd.isna(feature_values) | pd.isna(df[target_col]))
                if valid_mask.sum() < 10:  # 유효한 데이터가 너무 적으면 건너뜀
                    continue
                
                X = feature_values[valid_mask].values.reshape(-1, 1)
                y = df[target_col][valid_mask].values
                
                # 상호정보량 계산
                mi_score = mutual_info_regression(X, y)[0]
                
                # 상관계수 계산
                corr_score = abs(np.corrcoef(X.flatten(), y)[0, 1])
                
                # 종합 점수 (상호정보량과 상관계수의 평균)
                combined_score = (mi_score + corr_score) / 2
                
                feature_scores[feature_name] = {
                    'mutual_info': mi_score,
                    'correlation': corr_score,
                    'combined_score': combined_score
                }
                
            except Exception as e:
                continue  # 오류가 발생한 특성은 건너뜀
        
        # 상위 특성들 출력
        sorted_features = sorted(feature_scores.items(), 
                               key=lambda x: x[1]['combined_score'], 
                               reverse=True)
        
        print(f"   📈 평가된 특성: {len(feature_scores)}개")
        print(f"   🏆 상위 10개 특성:")
        
        for i, (feature_name, scores) in enumerate(sorted_features[:10], 1):
            print(f"      {i:2d}. {feature_name}: {scores['combined_score']:.3f}")
        
        return feature_scores
    
    def generate_ai_features(self, df, target_col=None, max_features_per_type=15):
        """AI 기반 특성 생성 메인 함수"""
        print("🤖 AI 기반 자동 특성 생성 시작:")
        print(f"   📊 입력 데이터: {df.shape[0]:,}행 × {df.shape[1]}열")
        
        all_new_features = {}
        
        # 1. 수학적 조합 특성
        math_features = self.mathematical_combinations(df, max_features_per_type)
        all_new_features.update(math_features)
        
        # 2. 통계적 특성
        stat_features = self.statistical_features(df)
        all_new_features.update(stat_features)
        
        # 3. 상호작용 특성
        if target_col:
            interaction_features = self.interaction_features(df, target_col)
            all_new_features.update(interaction_features)
        
        # 4. 특성 중요도 평가
        if target_col:
            feature_scores = self.evaluate_feature_importance(df, all_new_features, target_col)
            self.feature_scores = feature_scores
        
        # 5. 상위 특성들만 선별
        if target_col and feature_scores:
            # 상위 30개 특성만 선택
            top_features = sorted(feature_scores.items(), 
                                key=lambda x: x[1]['combined_score'], 
                                reverse=True)[:30]
            
            selected_features = {}
            for feature_name, _ in top_features:
                selected_features[feature_name] = all_new_features[feature_name]
            
            all_new_features = selected_features
        
        print(f"\n✅ AI 특성 생성 완료:")
        print(f"   🆕 최종 선택된 특성: {len(all_new_features)}개")
        
        # 새로운 특성들을 원본 데이터프레임에 추가
        df_enhanced = df.copy()
        for feature_name, feature_values in all_new_features.items():
            df_enhanced[feature_name] = feature_values
        
        print(f"   📊 최종 데이터: {df_enhanced.shape[0]:,}행 × {df_enhanced.shape[1]}열")
        
        return df_enhanced, all_new_features, getattr(self, 'feature_scores', {})

# AI 특성 생성 실행
try:
    # House Prices 데이터 로드
    train_data = pd.read_csv('datasets/house_prices/train.csv')
    
    # 주요 수치형 컬럼만 선택 (예시를 위해)
    key_columns = ['GrLivArea', 'YearBuilt', 'TotalBsmtSF', 'GarageArea', 'SalePrice']
    available_columns = [col for col in key_columns if col in train_data.columns]
    
    if len(available_columns) >= 3:
        sample_data = train_data[available_columns].copy()
        
        # AI 특성 생성기 실행
        ai_generator = AIFeatureGenerator()
        enhanced_data, new_features, feature_scores = ai_generator.generate_ai_features(
            sample_data, target_col='SalePrice'
        )
        
        print(f"\n📋 AI 특성 생성 요약:")
        print(f"   원본 특성: {len(available_columns)}개")
        print(f"   생성된 특성: {len(new_features)}개")
        print(f"   증가율: {len(new_features)/len(available_columns)*100:.1f}%")
        
    else:
        print("⚠️ 충분한 수치형 컬럼이 없습니다.")
        
except FileNotFoundError:
    print("❌ 실제 데이터를 찾을 수 없어 예시 데이터로 시연합니다.")
    
    # 예시 데이터 생성
    np.random.seed(42)
    n_samples = 500
    
    sample_data = pd.DataFrame({
        'feature_1': np.random.normal(100, 20, n_samples),
        'feature_2': np.random.normal(50, 10, n_samples),
        'feature_3': np.random.exponential(5, n_samples),
        'target': np.random.normal(200, 50, n_samples)
    })
    
    # 타겟과 특성 간 관계 강화
    sample_data['target'] += sample_data['feature_1'] * 0.5 + sample_data['feature_2'] * 0.3
    
    ai_generator = AIFeatureGenerator()
    enhanced_data, new_features, feature_scores = ai_generator.generate_ai_features(
        sample_data, target_col='target'
    )
```

**🔍 코드 해설:**
- `AIFeatureGenerator`로 실제 AI 특성 생성 도구의 작동 방식 시뮬레이션
- 수학적 조합, 통계적 변환, 상호작용 특성 등 다양한 자동 생성 전략 구현
- 상호정보량과 상관계수를 활용한 객관적 특성 평가 시스템

> **📊 이미지 생성 프롬프트:**  
> "Create a comprehensive AI feature generation visualization showing: 1) A flowchart of mathematical feature combinations (addition, subtraction, multiplication, division), 2) Statistical transformations (z-score, rank, log, sqrt), 3) Interaction detection matrix with correlation coefficients, 4) Feature importance scoring system with mutual information and correlation metrics, 5) Before and after comparison of dataset dimensions (original vs generated features). Use modern data science styling with clear mathematical notation and performance metrics."

---

## 📖 4.4.3 지능형 데이터 클리닝의 작동 방식과 검증 방법

### 지능형 데이터 클리닝의 혁신

**지능형 데이터 클리닝**은 머신러닝과 규칙 엔진을 결합하여 데이터 품질 문제를 자동으로 탐지하고 수정하는 차세대 기술입니다.

> **🔍 주요 용어 해설**
> - **Intelligent Data Cleaning**: AI 기반 자동 데이터 품질 관리 시스템
> - **Pattern-based Detection**: 패턴 학습을 통한 이상 데이터 자동 탐지
> - **Self-learning Cleaning**: 사용자 피드백을 통해 지속 학습하는 클리닝 시스템
> - **Quality Score**: 데이터 품질을 정량화한 종합 점수

### 지능형 클리닝 시스템 구조

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import re
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class IntelligentDataCleaner:
    """
    지능형 데이터 클리닝 시스템
    AI와 규칙 엔진을 결합한 자동 데이터 품질 관리
    """
    
    def __init__(self, learning_mode=True):
        self.learning_mode = learning_mode
        self.quality_rules = {}
        self.learned_patterns = {}
        self.cleaning_history = []
        self.quality_scores = {}
        
    def comprehensive_quality_assessment(self, df):
        """종합적 데이터 품질 평가"""
        print("🔍 지능형 데이터 품질 종합 평가:")
        
        assessment = {
            'completeness': {},    # 완전성
            'consistency': {},     # 일관성  
            'accuracy': {},        # 정확성
            'validity': {},        # 유효성
            'uniqueness': {},      # 고유성
            'overall_score': 0
        }
        
        # 1. 완전성 평가 (결측치 분석)
        print(f"\n📊 1. 완전성 평가:")
        for col in df.columns:
            missing_rate = df[col].isnull().sum() / len(df)
            assessment['completeness'][col] = {
                'missing_rate': missing_rate,
                'score': max(0, 100 - missing_rate * 100),
                'status': 'excellent' if missing_rate < 0.05 else 
                         'good' if missing_rate < 0.15 else 
                         'poor' if missing_rate < 0.50 else 'critical'
            }
            print(f"   {col}: {missing_rate:.1%} 결측 -> {assessment['completeness'][col]['status']}")
        
        # 2. 일관성 평가 (데이터 타입과 형식)
        print(f"\n📊 2. 일관성 평가:")
        for col in df.columns:
            dtype = str(df[col].dtype)
            consistency_score = 100  # 기본 점수
            
            if dtype == 'object':  # 문자열 컬럼 일관성 검사
                values = df[col].dropna()
                if len(values) > 0:
                    # 형식 일관성 체크
                    unique_patterns = set()
                    for val in values.head(100):  # 샘플 100개만 검사
                        pattern = self._extract_pattern(str(val))
                        unique_patterns.add(pattern)
                    
                    pattern_diversity = len(unique_patterns) / min(len(values), 100)
                    consistency_score = max(0, 100 - pattern_diversity * 50)
            
            assessment['consistency'][col] = {
                'data_type': dtype,
                'score': consistency_score,
                'status': 'excellent' if consistency_score >= 90 else
                         'good' if consistency_score >= 70 else
                         'poor' if consistency_score >= 50 else 'critical'
            }
            print(f"   {col}: {dtype} -> {assessment['consistency'][col]['status']}")
        
        # 3. 정확성 평가 (이상치 탐지)
        print(f"\n📊 3. 정확성 평가:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].notna().sum() < 10:  # 데이터가 너무 적으면 건너뜀
                continue
                
            # IQR 방법으로 이상치 탐지
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            outlier_mask = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR))
            outlier_rate = outlier_mask.sum() / df[col].notna().sum()
            
            accuracy_score = max(0, 100 - outlier_rate * 200)  # 이상치 1%당 2점 감점
            
            assessment['accuracy'][col] = {
                'outlier_rate': outlier_rate,
                'score': accuracy_score,
                'status': 'excellent' if outlier_rate < 0.02 else
                         'good' if outlier_rate < 0.05 else
                         'poor' if outlier_rate < 0.10 else 'critical'
            }
            print(f"   {col}: {outlier_rate:.1%} 이상치 -> {assessment['accuracy'][col]['status']}")
        
        # 4. 유효성 평가 (도메인 규칙 검증)
        print(f"\n📊 4. 유효성 평가:")
        validity_issues = self._validate_domain_rules(df)
        
        for col, issues in validity_issues.items():
            invalid_rate = issues['count'] / len(df) if len(df) > 0 else 0
            validity_score = max(0, 100 - invalid_rate * 100)
            
            assessment['validity'][col] = {
                'invalid_rate': invalid_rate,
                'issues': issues['types'],
                'score': validity_score,
                'status': 'excellent' if invalid_rate < 0.01 else
                         'good' if invalid_rate < 0.05 else
                         'poor' if invalid_rate < 0.15 else 'critical'
            }
            print(f"   {col}: {invalid_rate:.1%} 유효성 문제 -> {assessment['validity'][col]['status']}")
        
        # 5. 고유성 평가 (중복 데이터)
        print(f"\n📊 5. 고유성 평가:")
        duplicate_rate = df.duplicated().sum() / len(df)
        uniqueness_score = max(0, 100 - duplicate_rate * 100)
        
        assessment['uniqueness'] = {
            'duplicate_rate': duplicate_rate,
            'score': uniqueness_score,
            'status': 'excellent' if duplicate_rate < 0.01 else
                     'good' if duplicate_rate < 0.05 else
                     'poor' if duplicate_rate < 0.15 else 'critical'
        }
        print(f"   전체 데이터: {duplicate_rate:.1%} 중복 -> {assessment['uniqueness']['status']}")
        
        # 종합 점수 계산
        all_scores = []
        
        # 각 차원별 평균 점수 계산
        if assessment['completeness']:
            completeness_avg = np.mean([v['score'] for v in assessment['completeness'].values()])
            all_scores.append(completeness_avg)
        
        if assessment['consistency']:
            consistency_avg = np.mean([v['score'] for v in assessment['consistency'].values()])
            all_scores.append(consistency_avg)
        
        if assessment['accuracy']:
            accuracy_avg = np.mean([v['score'] for v in assessment['accuracy'].values()])
            all_scores.append(accuracy_avg)
        
        if assessment['validity']:
            validity_avg = np.mean([v['score'] for v in assessment['validity'].values()])
            all_scores.append(validity_avg)
        
        all_scores.append(assessment['uniqueness']['score'])
        
        assessment['overall_score'] = np.mean(all_scores) if all_scores else 0
        
        print(f"\n🏆 종합 데이터 품질 점수: {assessment['overall_score']:.1f}/100")
        if assessment['overall_score'] >= 90:
            print("   등급: A (Excellent) ✨")
        elif assessment['overall_score'] >= 80:
            print("   등급: B (Good) ✅")
        elif assessment['overall_score'] >= 70:
            print("   등급: C (Acceptable) ⚠️")
        else:
            print("   등급: D (Poor) ❌")
        
        return assessment
    
    def _extract_pattern(self, text):
        """문자열 패턴 추출"""
        # 숫자는 N, 문자는 A, 특수문자는 S로 변환
        pattern = ""
        for char in str(text):
            if char.isdigit():
                pattern += "N"
            elif char.isalpha():
                pattern += "A"
            elif char.isspace():
                pattern += " "
            else:
                pattern += "S"
        return pattern
    
    def _validate_domain_rules(self, df):
        """도메인별 유효성 규칙 검증"""
        issues = {}
        
        for col in df.columns:
            col_issues = {'count': 0, 'types': []}
            
            # 일반적인 유효성 규칙들
            if 'price' in col.lower() or 'cost' in col.lower():
                # 가격은 음수가 될 수 없음
                if col in df.select_dtypes(include=[np.number]).columns:
                    negative_count = (df[col] < 0).sum()
                    if negative_count > 0:
                        col_issues['count'] += negative_count
                        col_issues['types'].append('negative_price')
            
            if 'age' in col.lower():
                # 나이는 0-150 범위
                if col in df.select_dtypes(include=[np.number]).columns:
                    invalid_age = ((df[col] < 0) | (df[col] > 150)).sum()
                    if invalid_age > 0:
                        col_issues['count'] += invalid_age
                        col_issues['types'].append('invalid_age_range')
            
            if 'email' in col.lower():
                # 이메일 형식 검증
                if col in df.select_dtypes(include=['object']).columns:
                    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}


                    invalid_emails = df[col].dropna().apply(
                        lambda x: not re.match(email_pattern, str(x))
                    ).sum()
                    if invalid_emails > 0:
                        col_issues['count'] += invalid_emails
                        col_issues['types'].append('invalid_email_format')
            
            if 'year' in col.lower():
                # 연도는 1900-2030 범위
                if col in df.select_dtypes(include=[np.number]).columns:
                    invalid_year = ((df[col] < 1900) | (df[col] > 2030)).sum()
                    if invalid_year > 0:
                        col_issues['count'] += invalid_year
                        col_issues['types'].append('invalid_year_range')
            
            if col_issues['count'] > 0:
                issues[col] = col_issues
        
        return issues
    
    def intelligent_anomaly_detection(self, df):
        """지능형 이상 데이터 탐지"""
        print("\n🤖 지능형 이상 데이터 탐지:")
        
        anomalies = {
            'isolation_forest': {},
            'dbscan_outliers': {},
            'statistical_outliers': {},
            'pattern_anomalies': {}
        }
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            # 1. Isolation Forest 이상치 탐지
            print(f"\n🌲 1. Isolation Forest 이상치 탐지:")
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            
            # 결측값이 없는 수치형 데이터만 사용
            numeric_data = df[numeric_cols].fillna(df[numeric_cols].median())
            
            if len(numeric_data) > 10:  # 최소 데이터 수 확인
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(numeric_data)
                
                anomaly_labels = iso_forest.fit_predict(scaled_data)
                anomaly_indices = np.where(anomaly_labels == -1)[0]
                
                anomalies['isolation_forest'] = {
                    'indices': anomaly_indices.tolist(),
                    'count': len(anomaly_indices),
                    'percentage': len(anomaly_indices) / len(df) * 100
                }
                
                print(f"   탐지된 이상 데이터: {len(anomaly_indices)}개 ({len(anomaly_indices)/len(df)*100:.1f}%)")
            
            # 2. DBSCAN 클러스터링 기반 이상치 탐지
            print(f"\n🔍 2. DBSCAN 클러스터링 이상치 탐지:")
            if len(numeric_data) > 20:  # DBSCAN은 더 많은 데이터 필요
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(numeric_data)
                
                dbscan = DBSCAN(eps=0.5, min_samples=5)
                cluster_labels = dbscan.fit_predict(scaled_data)
                
                outlier_indices = np.where(cluster_labels == -1)[0]
                
                anomalies['dbscan_outliers'] = {
                    'indices': outlier_indices.tolist(),
                    'count': len(outlier_indices),
                    'percentage': len(outlier_indices) / len(df) * 100,
                    'n_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                }
                
                print(f"   클러스터 수: {anomalies['dbscan_outliers']['n_clusters']}개")
                print(f"   이상치: {len(outlier_indices)}개 ({len(outlier_indices)/len(df)*100:.1f}%)")
        
        # 3. 통계적 이상치 탐지 (개별 컬럼)
        print(f"\n📊 3. 통계적 이상치 탐지:")
        for col in numeric_cols:
            if df[col].notna().sum() < 10:
                continue
                
            # Modified Z-Score 방법
            median = df[col].median()
            mad = np.median(np.abs(df[col] - median))
            
            if mad != 0:  # MAD가 0이 아닌 경우만
                modified_z_scores = 0.6745 * (df[col] - median) / mad
                outlier_mask = np.abs(modified_z_scores) > 3.5
                
                outlier_indices = df[outlier_mask].index.tolist()
                
                anomalies['statistical_outliers'][col] = {
                    'indices': outlier_indices,
                    'count': len(outlier_indices),
                    'percentage': len(outlier_indices) / df[col].notna().sum() * 100
                }
                
                print(f"   {col}: {len(outlier_indices)}개 이상치 ({len(outlier_indices)/df[col].notna().sum()*100:.1f}%)")
        
        # 4. 패턴 기반 이상 탐지
        print(f"\n🔎 4. 패턴 기반 이상 탐지:")
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        for col in categorical_cols:
            if df[col].notna().sum() < 10:
                continue
                
            # 빈도 기반 이상 패턴 탐지
            value_counts = df[col].value_counts()
            total_count = value_counts.sum()
            
            # 전체의 1% 미만인 값들을 희귀 패턴으로 간주
            rare_threshold = max(1, total_count * 0.01)
            rare_values = value_counts[value_counts < rare_threshold].index.tolist()
            
            if rare_values:
                rare_indices = df[df[col].isin(rare_values)].index.tolist()
                
                anomalies['pattern_anomalies'][col] = {
                    'rare_values': rare_values,
                    'indices': rare_indices,
                    'count': len(rare_indices),
                    'percentage': len(rare_indices) / df[col].notna().sum() * 100
                }
                
                print(f"   {col}: {len(rare_values)}개 희귀 패턴, {len(rare_indices)}개 데이터 ({len(rare_indices)/df[col].notna().sum()*100:.1f}%)")
        
        return anomalies
    
    def automated_cleaning_recommendations(self, df, quality_assessment, anomalies):
        """자동 클리닝 권고사항 생성"""
        print("\n🛠️ 자동 클리닝 권고사항:")
        
        recommendations = {
            'high_priority': [],    # 높은 우선순위
            'medium_priority': [],  # 중간 우선순위  
            'low_priority': [],     # 낮은 우선순위
            'automated_actions': []  # 자동 실행 가능한 작업
        }
        
        # 1. 결측치 처리 권고
        for col, info in quality_assessment['completeness'].items():
            missing_rate = info['missing_rate']
            
            if missing_rate > 0.5:
                recommendations['high_priority'].append({
                    'action': 'column_removal',
                    'column': col,
                    'reason': f'너무 많은 결측값 ({missing_rate:.1%})',
                    'urgency': 'critical'
                })
            elif missing_rate > 0.2:
                recommendations['medium_priority'].append({
                    'action': 'missing_value_imputation',
                    'column': col,
                    'method': 'advanced' if df[col].dtype in ['int64', 'float64'] else 'mode',
                    'reason': f'상당한 결측값 ({missing_rate:.1%})',
                    'urgency': 'medium'
                })
            elif missing_rate > 0.05:
                recommendations['low_priority'].append({
                    'action': 'missing_value_imputation',
                    'column': col,
                    'method': 'simple',
                    'reason': f'소량의 결측값 ({missing_rate:.1%})',
                    'urgency': 'low'
                })
        
        # 2. 이상치 처리 권고
        for col, info in quality_assessment.get('accuracy', {}).items():
            outlier_rate = info['outlier_rate']
            
            if outlier_rate > 0.1:
                recommendations['high_priority'].append({
                    'action': 'outlier_investigation',
                    'column': col,
                    'reason': f'많은 이상치 ({outlier_rate:.1%})',
                    'urgency': 'high'
                })
            elif outlier_rate > 0.05:
                recommendations['medium_priority'].append({
                    'action': 'outlier_treatment',
                    'column': col,
                    'method': 'capping',
                    'reason': f'보통 수준의 이상치 ({outlier_rate:.1%})',
                    'urgency': 'medium'
                })
        
        # 3. 중복 데이터 처리 권고
        duplicate_info = quality_assessment['uniqueness']
        if duplicate_info['duplicate_rate'] > 0.01:
            priority = 'high_priority' if duplicate_info['duplicate_rate'] > 0.1 else 'medium_priority'
            recommendations[priority].append({
                'action': 'duplicate_removal',
                'reason': f'중복 데이터 ({duplicate_info["duplicate_rate"]:.1%})',
                'urgency': 'high' if duplicate_info['duplicate_rate'] > 0.1 else 'medium'
            })
        
        # 4. 자동 실행 가능한 작업들
        if duplicate_info['duplicate_rate'] > 0:
            recommendations['automated_actions'].append({
                'action': 'remove_exact_duplicates',
                'description': '완전히 동일한 행 제거',
                'safe': True
            })
        
        # 권고사항 출력
        for priority, items in recommendations.items():
            if items:
                priority_name = {
                    'high_priority': '🔴 높은 우선순위',
                    'medium_priority': '🟡 중간 우선순위',
                    'low_priority': '🟢 낮은 우선순위',
                    'automated_actions': '🤖 자동 실행 가능'
                }
                
                print(f"\n{priority_name[priority]}:")
                for i, item in enumerate(items[:5], 1):  # 상위 5개만 표시
                    print(f"   {i}. {item.get('action', 'unknown')}: {item.get('reason', 'N/A')}")
                    if 'column' in item:
                        print(f"      대상: {item['column']}")
                    if 'method' in item:
                        print(f"      방법: {item['method']}")
        
        return recommendations
    
    def generate_cleaning_report(self, df, quality_assessment, anomalies, recommendations):
        """종합 클리닝 보고서 생성"""
        print("\n📋 지능형 데이터 클리닝 종합 보고서:")
        print("=" * 60)
        
        # 데이터 개요
        print(f"\n📊 데이터 개요:")
        print(f"   행 수: {len(df):,}")
        print(f"   열 수: {len(df.columns)}")
        print(f"   메모리 사용량: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # 품질 점수 요약
        print(f"\n🏆 품질 평가 요약:")
        print(f"   종합 점수: {quality_assessment['overall_score']:.1f}/100")
        
        # 차원별 점수
        dimensions = ['completeness', 'consistency', 'accuracy', 'validity']
        for dim in dimensions:
            if quality_assessment[dim]:
                scores = [v['score'] for v in quality_assessment[dim].values()]
                avg_score = np.mean(scores)
                print(f"   {dim.capitalize()}: {avg_score:.1f}/100")
        
        print(f"   Uniqueness: {quality_assessment['uniqueness']['score']:.1f}/100")
        
        # 주요 문제점
        print(f"\n⚠️ 주요 문제점:")
        problem_count = 0
        
        # 심각한 결측치
        for col, info in quality_assessment['completeness'].items():
            if info['status'] in ['poor', 'critical']:
                problem_count += 1
                print(f"   • {col}: {info['missing_rate']:.1%} 결측치")
        
        # 많은 이상치
        for col, info in quality_assessment.get('accuracy', {}).items():
            if info['status'] in ['poor', 'critical']:
                problem_count += 1
                print(f"   • {col}: {info['outlier_rate']:.1%} 이상치")
        
        # 중복 데이터
        if quality_assessment['uniqueness']['status'] in ['poor', 'critical']:
            problem_count += 1
            rate = quality_assessment['uniqueness']['duplicate_rate']
            print(f"   • 전체: {rate:.1%} 중복 데이터")
        
        if problem_count == 0:
            print("   발견된 주요 문제점 없음 ✅")
        
        # 권고사항 요약
        print(f"\n🛠️ 권고사항 요약:")
        total_actions = sum(len(recommendations[key]) for key in recommendations.keys())
        print(f"   총 권고사항: {total_actions}개")
        
        for priority in ['high_priority', 'medium_priority', 'low_priority']:
            count = len(recommendations[priority])
            if count > 0:
                priority_names = {
                    'high_priority': '높은 우선순위',
                    'medium_priority': '중간 우선순위', 
                    'low_priority': '낮은 우선순위'
                }
                print(f"   {priority_names[priority]}: {count}개")
        
        # 예상 개선 효과
        print(f"\n📈 예상 개선 효과:")
        current_score = quality_assessment['overall_score']
        
        if current_score < 70:
            expected_improvement = 20
        elif current_score < 85:
            expected_improvement = 10
        else:
            expected_improvement = 5
        
        expected_score = min(100, current_score + expected_improvement)
        print(f"   권고사항 적용 후 예상 점수: {expected_score:.1f}/100")
        print(f"   예상 개선폭: +{expected_improvement:.1f}점")
        
        print("=" * 60)
        
        return {
            'current_score': current_score,
            'expected_score': expected_score,
            'improvement': expected_improvement,
            'total_recommendations': total_actions,
            'problem_count': problem_count
        }

# House Prices 데이터로 지능형 클리닝 시연
try:
    # 데이터 로드
    train_data = pd.read_csv('datasets/house_prices/train.csv')
    print("✅ House Prices 데이터 로드 성공!")
    
    # 주요 컬럼만 선택 (시연용)
    sample_columns = ['SalePrice', 'GrLivArea', 'YearBuilt', 'TotalBsmtSF', 
                     'GarageArea', 'LotArea', 'Neighborhood', 'BldgType']
    available_columns = [col for col in sample_columns if col in train_data.columns]
    
    if len(available_columns) >= 5:
        sample_data = train_data[available_columns].copy()
        
        # 지능형 데이터 클리너 실행
        cleaner = IntelligentDataCleaner(learning_mode=True)
        
        # 1. 종합 품질 평가
        quality_assessment = cleaner.comprehensive_quality_assessment(sample_data)
        
        # 2. 지능형 이상 데이터 탐지
        anomalies = cleaner.intelligent_anomaly_detection(sample_data)
        
        # 3. 자동 클리닝 권고사항
        recommendations = cleaner.automated_cleaning_recommendations(
            sample_data, quality_assessment, anomalies
        )
        
        # 4. 종합 보고서 생성
        report = cleaner.generate_cleaning_report(
            sample_data, quality_assessment, anomalies, recommendations
        )
        
        print(f"\n📋 지능형 클리닝 시스템 요약:")
        print(f"   현재 품질 점수: {report['current_score']:.1f}/100")
        print(f"   예상 개선 점수: {report['expected_score']:.1f}/100")
        print(f"   총 권고사항: {report['total_recommendations']}개")
        
    else:
        print("⚠️ 충분한 컬럼이 없습니다.")
        
except FileNotFoundError:
    print("❌ 실제 데이터를 찾을 수 없어 가상 데이터로 시연합니다.")
    
    # 가상 데이터 생성 (품질 문제 포함)
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = pd.DataFrame({
        'price': np.random.lognormal(10, 1, n_samples),
        'area': np.random.normal(1500, 400, n_samples),
        'year': np.random.randint(1950, 2023, n_samples),
        'category': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
        'email': ['user' + str(i) + '@email.com' for i in range(n_samples)]
    })
    
    # 의도적으로 품질 문제 추가
    # 결측치 추가
    sample_data.loc[np.random.choice(n_samples, 100, replace=False), 'area'] = np.nan
    sample_data.loc[np.random.choice(n_samples, 50, replace=False), 'category'] = np.nan
    
    # 이상치 추가
    sample_data.loc[np.random.choice(n_samples, 20, replace=False), 'price'] = -1000
    sample_data.loc[np.random.choice(n_samples, 30, replace=False), 'area'] = 10000
    
    # 중복 데이터 추가
    duplicate_indices = np.random.choice(n_samples, 50, replace=False)
    for idx in duplicate_indices:
        sample_data.loc[n_samples + len(duplicate_indices)] = sample_data.loc[idx]
    
    # 잘못된 이메일 추가
    sample_data.loc[np.random.choice(len(sample_data), 30, replace=False), 'email'] = 'invalid_email'
    
    # 지능형 클리너 실행
    cleaner = IntelligentDataCleaner()
    quality_assessment = cleaner.comprehensive_quality_assessment(sample_data)
    anomalies = cleaner.intelligent_anomaly_detection(sample_data)
    recommendations = cleaner.automated_cleaning_recommendations(
        sample_data, quality_assessment, anomalies
    )
    report = cleaner.generate_cleaning_report(
        sample_data, quality_assessment, anomalies, recommendations
    )
```

**🔍 코드 해설:**
- `IntelligentDataCleaner` 클래스로 AI 기반 지능형 데이터 클리닝 시스템 완전 구현
- 5차원 품질 평가 (완전성, 일관성, 정확성, 유효성, 고유성)와 종합 점수 시스템
- Isolation Forest, DBSCAN, 통계적 방법, 패턴 분석 등 다양한 이상 탐지 기법 통합
- 자동 권고사항 생성과 우선순위 기반 클리닝 전략 수립

### AI 클리닝 결과 검증 방법론

```python
class CleaningValidationFramework:
    """
    AI 클리닝 결과의 신뢰성 검증 프레임워크
    """
    
    def __init__(self):
        self.validation_history = []
        self.quality_metrics = {}
        
    def pre_post_comparison(self, original_df, cleaned_df):
        """전후 비교 분석"""
        print("🔍 AI 클리닝 전후 비교 분석:")
        
        comparison = {
            'data_shape': {
                'before': original_df.shape,
                'after': cleaned_df.shape,
                'change': (cleaned_df.shape[0] - original_df.shape[0], 
                          cleaned_df.shape[1] - original_df.shape[1])
            },
            'missing_values': {},
            'data_types': {},
            'statistical_changes': {}
        }
        
        print(f"\n📊 데이터 크기 변화:")
        print(f"   이전: {original_df.shape[0]:,}행 × {original_df.shape[1]}열")
        print(f"   이후: {cleaned_df.shape[0]:,}행 × {cleaned_df.shape[1]}열")
        print(f"   변화: {comparison['data_shape']['change'][0]:+,}행, {comparison['data_shape']['change'][1]:+}열")
        
        # 공통 컬럼에 대해서만 비교
        common_cols = set(original_df.columns) & set(cleaned_df.columns)
        
        # 결측값 비교
        print(f"\n📊 결측값 변화:")
        for col in common_cols:
            before_missing = original_df[col].isnull().sum()
            after_missing = cleaned_df[col].isnull().sum()
            change = after_missing - before_missing
            
            comparison['missing_values'][col] = {
                'before': before_missing,
                'after': after_missing,
                'change': change,
                'improvement': change < 0
            }
            
            if change != 0:
                print(f"   {col}: {before_missing} → {after_missing} ({change:+})")
        
        # 통계적 변화 분석 (수치형 컬럼)
        numeric_cols = original_df.select_dtypes(include=[np.number]).columns
        common_numeric = set(numeric_cols) & common_cols
        
        if common_numeric:
            print(f"\n📊 통계적 특성 변화:")
            for col in common_numeric:
                if original_df[col].notna().sum() > 0 and cleaned_df[col].notna().sum() > 0:
                    before_mean = original_df[col].mean()
                    after_mean = cleaned_df[col].mean()
                    before_std = original_df[col].std()
                    after_std = cleaned_df[col].std()
                    
                    comparison['statistical_changes'][col] = {
                        'mean_change': after_mean - before_mean,
                        'std_change': after_std - before_std,
                        'mean_change_pct': (after_mean - before_mean) / before_mean * 100 if before_mean != 0 else 0,
                        'std_change_pct': (after_std - before_std) / before_std * 100 if before_std != 0 else 0
                    }
                    
                    print(f"   {col}:")
                    print(f"      평균: {before_mean:.2f} → {after_mean:.2f} ({comparison['statistical_changes'][col]['mean_change_pct']:+.1f}%)")
                    print(f"      표준편차: {before_std:.2f} → {after_std:.2f} ({comparison['statistical_changes'][col]['std_change_pct']:+.1f}%)")
        
        return comparison
    
    def validate_cleaning_quality(self, original_df, cleaned_df, cleaning_actions):
        """클리닝 품질 검증"""
        print("\n🎯 AI 클리닝 품질 검증:")
        
        validation_results = {
            'data_integrity': True,
            'logical_consistency': True,
            'business_rules': True,
            'statistical_validity': True,
            'issues': []
        }
        
        # 1. 데이터 무결성 검증
        print(f"\n🔍 1. 데이터 무결성 검증:")
        
        # 예상치 못한 데이터 손실 확인
        expected_loss = sum(1 for action in cleaning_actions if action.get('action') == 'row_removal')
        actual_loss = len(original_df) - len(cleaned_df)
        
        if abs(actual_loss - expected_loss) > len(original_df) * 0.01:  # 1% 이상 차이
            validation_results['data_integrity'] = False
            validation_results['issues'].append("예상보다 많은 데이터 손실")
            print("   ❌ 예상치 못한 데이터 손실 발생")
        else:
            print("   ✅ 데이터 무결성 유지")
        
        # 2. 논리적 일관성 검증
        print(f"\n🔍 2. 논리적 일관성 검증:")
        consistency_issues = 0
        
        for col in cleaned_df.columns:
            if col in original_df.columns:
                # 데이터 타입 변화 확인
                if original_df[col].dtype != cleaned_df[col].dtype:
                    # 의도적인 변화가 아닌 경우 문제
                    type_change_intended = any(
                        action.get('column') == col and action.get('action') == 'type_conversion'
                        for action in cleaning_actions
                    )
                    
                    if not type_change_intended:
                        consistency_issues += 1
                        validation_results['issues'].append(f"{col}: 예상치 못한 데이터 타입 변화")
        
        if consistency_issues == 0:
            print("   ✅ 논리적 일관성 유지")
        else:
            validation_results['logical_consistency'] = False
            print(f"   ❌ {consistency_issues}개 일관성 문제 발견")
        
        # 3. 비즈니스 규칙 검증
        print(f"\n🔍 3. 비즈니스 규칙 검증:")
        business_violations = 0
        
        # 가격 관련 컬럼 검증
        price_cols = [col for col in cleaned_df.columns if 'price' in col.lower() or 'cost' in col.lower()]
        for col in price_cols:
            if col in cleaned_df.select_dtypes(include=[np.number]).columns:
                negative_prices = (cleaned_df[col] < 0).sum()
                if negative_prices > 0:
                    business_violations += 1
                    validation_results['issues'].append(f"{col}: {negative_prices}개 음수 가격")
        
        # 연도 관련 컬럼 검증
        year_cols = [col for col in cleaned_df.columns if 'year' in col.lower()]
        for col in year_cols:
            if col in cleaned_df.select_dtypes(include=[np.number]).columns:
                invalid_years = ((cleaned_df[col] < 1900) | (cleaned_df[col] > 2030)).sum()
                if invalid_years > 0:
                    business_violations += 1
                    validation_results['issues'].append(f"{col}: {invalid_years}개 비현실적 연도")
        
        if business_violations == 0:
            print("   ✅ 비즈니스 규칙 준수")
        else:
            validation_results['business_rules'] = False
            print(f"   ❌ {business_violations}개 비즈니스 규칙 위반")
        
        # 4. 통계적 유효성 검증
        print(f"\n🔍 4. 통계적 유효성 검증:")
        statistical_issues = 0
        
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in original_df.columns and cleaned_df[col].notna().sum() > 10:
                # 극단적인 통계량 변화 확인
                original_mean = original_df[col].mean()
                cleaned_mean = cleaned_df[col].mean()
                
                if original_mean != 0:
                    mean_change_pct = abs(cleaned_mean - original_mean) / abs(original_mean)
                    
                    # 평균이 50% 이상 변했다면 문제 가능성
                    if mean_change_pct > 0.5:
                        statistical_issues += 1
                        validation_results['issues'].append(f"{col}: 평균값 {mean_change_pct*100:.1f}% 변화")
        
        if statistical_issues == 0:
            validation_results['statistical_validity'] = True
            print("   ✅ 통계적 유효성 확인")
        else:
            validation_results['statistical_validity'] = False
            print(f"   ❌ {statistical_issues}개 통계적 이상 현상")
        
        # 종합 판정
        all_passed = all([
            validation_results['data_integrity'],
            validation_results['logical_consistency'], 
            validation_results['business_rules'],
            validation_results['statistical_validity']
        ])
        
        print(f"\n🏆 종합 검증 결과:")
        if all_passed:
            print("   ✅ 모든 검증 통과 - AI 클리닝 결과 신뢰함")
        else:
            print("   ⚠️ 일부 검증 실패 - 추가 검토 필요")
            print("   주요 문제점:")
            for issue in validation_results['issues'][:5]:
                print(f"      • {issue}")
        
        return validation_results
    
    def generate_trust_score(self, validation_results, cleaner_reputation=0.8):
        """AI 클리닝 결과 신뢰도 점수 계산"""
        print(f"\n🎯 AI 클리닝 신뢰도 점수:")
        
        # 기본 점수 (검증 결과 기반)
        validation_score = 0
        weights = {
            'data_integrity': 0.3,
            'logical_consistency': 0.25,
            'business_rules': 0.25,
            'statistical_validity': 0.2
        }
        
        for criterion, weight in weights.items():
            if validation_results[criterion]:
                validation_score += weight
        
        # 클리너 신뢰도 반영 (이전 성능 기반)
        trust_score = (validation_score * 0.7) + (cleaner_reputation * 0.3)
        
        # 이슈 수에 따른 감점
        issue_penalty = min(0.2, len(validation_results['issues']) * 0.05)
        trust_score = max(0, trust_score - issue_penalty)
        
        print(f"   검증 점수: {validation_score:.2f}")
        print(f"   클리너 신뢰도: {cleaner_reputation:.2f}")
        print(f"   이슈 감점: -{issue_penalty:.2f}")
        print(f"   최종 신뢰도: {trust_score:.2f}/1.0")
        
        if trust_score >= 0.9:
            print("   등급: A (매우 신뢰함) ✨")
            recommendation = "결과를 바로 사용해도 안전합니다."
        elif trust_score >= 0.8:
            print("   등급: B (신뢰함) ✅")
            recommendation = "간단한 검토 후 사용을 권장합니다."
        elif trust_score >= 0.7:
            print("   등급: C (보통) ⚠️")
            recommendation = "신중한 검토 후 선별적 사용을 권장합니다."
        else:
            print("   등급: D (낮음) ❌")
            recommendation = "수동 검토 및 재작업을 권장합니다."
        
        print(f"   권고사항: {recommendation}")
        
        return {
            'trust_score': trust_score,
            'validation_score': validation_score,
            'reputation_score': cleaner_reputation,
            'issue_penalty': issue_penalty,
            'recommendation': recommendation
        }

# 검증 프레임워크 시연
print("\n" + "="*60)
print("🎯 AI 클리닝 결과 검증 시연")
print("="*60)

# 가상의 클리닝 전후 데이터 생성
np.random.seed(42)
n_samples = 500

# 원본 데이터 (문제 포함)
original_data = pd.DataFrame({
    'price': np.concatenate([np.random.lognormal(10, 0.5, 450), [-1000] * 10, [0] * 40]),
    'area': np.concatenate([np.random.normal(1500, 300, 480), [np.nan] * 20]),
    'year': np.concatenate([np.random.randint(1950, 2023, 490), [1800] * 10])
})

# 클리닝된 데이터 (문제 해결됨)
cleaned_data = pd.DataFrame({
    'price': np.random.lognormal(10, 0.5, 480),  # 음수 제거
    'area': np.random.normal(1500, 300, 480),     # 결측치 대체
    'year': np.random.randint(1950, 2023, 480)    # 이상한 연도 제거
})

# 가상의 클리닝 액션 로그
cleaning_actions = [
    {'action': 'row_removal', 'reason': 'negative_price', 'count': 10},
    {'action': 'row_removal', 'reason': 'zero_price', 'count': 40}, 
    {'action': 'missing_imputation', 'column': 'area', 'method': 'median', 'count': 20},
    {'action': 'row_removal', 'reason': 'invalid_year', 'count': 10}
]

# 검증 프레임워크 실행
validator = CleaningValidationFramework()

# 1. 전후 비교
comparison = validator.pre_post_comparison(original_data, cleaned_data)

# 2. 품질 검증
validation_results = validator.validate_cleaning_quality(
    original_data, cleaned_data, cleaning_actions
)

# 3. 신뢰도 점수
trust_results = validator.generate_trust_score(validation_results)
```

**🔍 코드 해설:**
- `CleaningValidationFramework`로 AI 클리닝 결과의 신뢰성을 체계적으로 검증
- 4차원 검증 (데이터 무결성, 논리적 일관성, 비즈니스 규칙, 통계적 유효성)
- 정량적 신뢰도 점수와 등급 시스템으로 객관적 평가
- 실무적 권고사항까지 포함한 완전한 검증 프레임워크

> **📊 이미지 생성 프롬프트:**  
> "Create a comprehensive data cleaning validation framework visualization showing: 1) A before/after comparison dashboard with data quality metrics, 2) A 4-dimensional validation radar chart (data integrity, logical consistency, business rules, statistical validity), 3) A trust score calculation formula and grading system (A-D grades), 4) A workflow diagram of validation steps from data input to final recommendation. Use professional data science styling with clear metrics and validation indicators."

---

## 📖 4.4.4 자동화 전처리의 함정과 한계

### 자동화의 그림자: 보이지 않는 위험들

AI 기반 자동 전처리는 강력한 도구이지만, **"마법의 버튼"은 아닙니다**. 무분별한 자동화는 오히려 데이터 품질을 저하시키고 분석 결과를 왜곡할 수 있습니다.

> **🔍 주요 용어 해설**
> - **Automation Bias**: 자동화 결과를 무비판적으로 수용하는 편향
> - **Black Box Problem**: 처리 과정을 이해할 수 없는 블랙박스 문제
> - **Context Ignorance**: 도메인 맥락을 무시하는 일반화의 함정
> - **Statistical Artifacts**: 통계적으로 유의하지만 실제로는 무의미한 결과

### 자동화 전처리의 주요 한계점

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class AutomationPitfallAnalyzer:
    """
    자동화 전처리의 함정과 한계점을 분석하는 클래스
    """
    
    def __init__(self):
        self.pitfall_cases = {}
        self.demonstration_results = {}
        
    def demonstrate_context_ignorance(self):
        """도메인 맥락 무시의 함정 시연"""
        print("⚠️ 함정 1: 도메인 맥락 무시 (Context Ignorance)")
        print("="*50)
        
        # 의료 진단 데이터 시뮬레이션
        np.random.seed(42)
        n_patients = 1000
        
        # 환자 데이터 생성
        medical_data = pd.DataFrame({
            'age': np.random.normal(50, 15, n_patients),
            'blood_pressure': np.random.normal(120, 20, n_patients),
            'cholesterol': np.random.normal(200, 40, n_patients),
            'test_result': np.random.choice([0, 1], n_patients, p=[0.8, 0.2]),  # 20% 양성
            'doctor_notes': ['normal'] * 800 + ['critical'] * 100 + ['urgent'] * 100
        })
        
        # 의료진이 의도적으로 남긴 결측치 (중요한 의미)
        # "혈압이 정상인 경우 콜레스테롤 측정 생략" 정책
        normal_bp_indices = medical_data[medical_data['blood_pressure'] < 130].index
        medical_data.loc[normal_bp_indices[:200], 'cholesterol'] = np.nan
        
        print(f"📊 의료 데이터 현황:")
        print(f"   환자 수: {len(medical_data)}")
        print(f"   콜레스테롤 결측: {medical_data['cholesterol'].isnull().sum()}개")
        print(f"   결측 환자들의 평균 혈압: {medical_data[medical_data['cholesterol'].isnull()]['blood_pressure'].mean():.1f}")
        
        # 잘못된 자동 전처리: 맥락 무시하고 단순 대체
        print(f"\n🤖 자동 전처리 (맥락 무시):")
        auto_imputer = SimpleImputer(strategy='mean')
        auto_processed = medical_data.copy()
        auto_processed['cholesterol'] = auto_imputer.fit_transform(
            auto_processed[['cholesterol']]
        )
        
        print(f"   결측치를 전체 평균({medical_data['cholesterol'].mean():.1f})으로 대체")
        
        # 올바른 도메인 기반 전처리
        print(f"\n👨‍⚕️ 도메인 지식 기반 전처리:")
        domain_processed = medical_data.copy()
        
        # 정상 혈압 환자는 낮은 콜레스테롤로 대체
        normal_bp_mask = (domain_processed['blood_pressure'] < 130) & domain_processed['cholesterol'].isnull()
        domain_processed.loc[normal_bp_mask, 'cholesterol'] = 180  # 정상 범위
        
        # 나머지는 연령대별 평균으로 대체
        remaining_missing = domain_processed['cholesterol'].isnull()
        for age_group in [(0, 40), (40, 60), (60, 100)]:
            age_mask = (domain_processed['age'] >= age_group[0]) & (domain_processed['age'] < age_group[1])
            group_mean = medical_data.loc[age_mask & ~medical_data['cholesterol'].isnull(), 'cholesterol'].mean()
            domain_processed.loc[remaining_missing & age_mask, 'cholesterol'] = group_mean
        
        print(f"   정상 혈압 환자: 정상 콜레스테롤(180)로 대체")
        print(f"   기타: 연령대별 평균으로 대체")
        
        # 두 방법의 차이점 비교
        print(f"\n📊 처리 결과 비교:")
        auto_normal_bp_cholesterol = auto_processed[normal_bp_mask]['cholesterol'].mean()
        domain_normal_bp_cholesterol = domain_processed[normal_bp_mask]['cholesterol'].mean()
        
        print(f"   정상 혈압 환자들의 콜레스테롤:")
        print(f"   - 자동 전처리: {auto_normal_bp_cholesterol:.1f}")
        print(f"   - 도메인 기반: {domain_normal_bp_cholesterol:.1f}")
        print(f"   - 차이: {abs(auto_normal_bp_cholesterol - domain_normal_bp_cholesterol):.1f}")
        
        if abs(auto_normal_bp_cholesterol - domain_normal_bp_cholesterol) > 10:
            print("   ⚠️ 상당한 차이 발생 - 진단 결과에 영향 가능성 높음")
        
        return {
            'original': medical_data,
            'auto_processed': auto_processed,
            'domain_processed': domain_processed,
            'difference': abs(auto_normal_bp_cholesterol - domain_normal_bp_cholesterol)
        }
    
    def demonstrate_overfitting_artifacts(self):
        """과적합과 허위 패턴의 함정 시연"""
        print("\n⚠️ 함정 2: 과적합과 허위 패턴 (Overfitting Artifacts)")
        print("="*50)
        
        # 작은 데이터셋에서 과도한 특성 생성 시뮬레이션
        np.random.seed(42)
        n_samples = 100  # 의도적으로 작은 데이터셋
        
        # 기본 데이터 생성
        small_dataset = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, n_samples),
            'feature_2': np.random.normal(0, 1, n_samples),
            'feature_3': np.random.normal(0, 1, n_samples),
            'target': np.random.normal(0, 1, n_samples)
        })
        
        # 타겟과 약간의 관계 추가 (실제 신호)
        small_dataset['target'] += 0.3 * small_dataset['feature_1'] + 0.2 * small_dataset['feature_2']
        
        print(f"📊 작은 데이터셋 현황:")
        print(f"   샘플 수: {n_samples}")
        print(f"   기본 특성: {len(small_dataset.columns) - 1}개")
        
        # 과도한 자동 특성 생성
        print(f"\n🤖 공격적인 자동 특성 생성:")
        enhanced_dataset = small_dataset.copy()
        
        # 모든 가능한 조합 생성
        feature_cols = ['feature_1', 'feature_2', 'feature_3']
        generated_count = 0
        
        # 2차 특성들
        for i, col1 in enumerate(feature_cols):
            for j, col2 in enumerate(feature_cols[i+1:], i+1):
                enhanced_dataset[f'{col1}_{col2}_multiply'] = enhanced_dataset[col1] * enhanced_dataset[col2]
                enhanced_dataset[f'{col1}_{col2}_add'] = enhanced_dataset[col1] + enhanced_dataset[col2]
                enhanced_dataset[f'{col1}_{col2}_subtract'] = enhanced_dataset[col1] - enhanced_dataset[col2]
                enhanced_dataset[f'{col1}_{col2}_divide'] = enhanced_dataset[col1] / (enhanced_dataset[col2] + 1e-8)
                generated_count += 4
        
        # 고차 특성들
        for col in feature_cols:
            enhanced_dataset[f'{col}_squared'] = enhanced_dataset[col] ** 2
            enhanced_dataset[f'{col}_cubed'] = enhanced_dataset[col] ** 3
            enhanced_dataset[f'{col}_sqrt'] = np.sqrt(np.abs(enhanced_dataset[col]))
            enhanced_dataset[f'{col}_log'] = np.log1p(np.abs(enhanced_dataset[col]))
            generated_count += 4
        
        print(f"   생성된 특성: {generated_count}개")
        print(f"   총 특성: {len(enhanced_dataset.columns) - 1}개")
        print(f"   특성/샘플 비율: {(len(enhanced_dataset.columns) - 1) / n_samples:.2f}")
        
        # 모델 성능 비교
        print(f"\n📊 모델 성능 비교:")
        
        # 기본 데이터셋
        X_basic = small_dataset.drop('target', axis=1)
        y = small_dataset['target']
        X_train_basic, X_test_basic, y_train, y_test = train_test_split(
            X_basic, y, test_size=0.3, random_state=42
        )
        
        basic_model = RandomForestRegressor(n_estimators=100, random_state=42)
        basic_model.fit(X_train_basic, y_train)
        basic_pred = basic_model.predict(X_test_basic)
        basic_r2 = r2_score(y_test, basic_pred)
        
        # 과도한 특성 데이터셋
        X_enhanced = enhanced_dataset.drop('target', axis=1)
        X_train_enhanced, X_test_enhanced, _, _ = train_test_split(
            X_enhanced, y, test_size=0.3, random_state=42
        )
        
        enhanced_model = RandomForestRegressor(n_estimators=100, random_state=42)
        enhanced_model.fit(X_train_enhanced, y_train)
        enhanced_pred = enhanced_model.predict(X_test_enhanced)
        enhanced_r2 = r2_score(y_test, enhanced_pred)
        
        print(f"   기본 특성 모델 R²: {basic_r2:.3f}")
        print(f"   과도한 특성 모델 R²: {enhanced_r2:.3f}")
        
        # 교차 검증으로 일반화 성능 확인
        from sklearn.model_selection import cross_val_score
        
        basic_cv_scores = cross_val_score(basic_model, X_basic, y, cv=5, scoring='r2')
        enhanced_cv_scores = cross_val_score(enhanced_model, X_enhanced, y, cv=5, scoring='r2')
        
        print(f"\n🔄 교차 검증 결과:")
        print(f"   기본 특성 평균 R²: {basic_cv_scores.mean():.3f} (±{basic_cv_scores.std():.3f})")
        print(f"   과도한 특성 평균 R²: {enhanced_cv_scores.mean():.3f} (±{enhanced_cv_scores.std():.3f})")
        
        # 과적합 진단
        if enhanced_cv_scores.mean() < basic_cv_scores.mean():
            print("   ⚠️ 과적합 발생 - 특성 증가가 성능 저하 야기")
        elif enhanced_cv_scores.std() > basic_cv_scores.std() * 1.5:
            print("   ⚠️ 불안정한 성능 - 과도한 특성으로 인한 분산 증가")
        
        return {
            'basic_r2': basic_r2,
            'enhanced_r2': enhanced_r2,
            'basic_cv_mean': basic_cv_scores.mean(),
            'enhanced_cv_mean': enhanced_cv_scores.mean(),
            'overfitting_detected': enhanced_cv_scores.mean() < basic_cv_scores.mean()
        }
    
    def demonstrate_automation_bias(self):
        """자동화 편향의 함정 시연"""
        print("\n⚠️ 함정 3: 자동화 편향 (Automation Bias)")
        print("="*50)
        
        # 편향된 자동 분류 시뮬레이션
        np.random.seed(42)
        n_samples = 1000
        
        # 채용 데이터 시뮬레이션 (편향 포함)
        hiring_data = pd.DataFrame({
            'experience_years': np.random.exponential(3, n_samples),
            'education_score': np.random.normal(75, 15, n_samples),
            'skills_test': np.random.normal(80, 12, n_samples),
            'previous_salary': np.random.lognormal(10, 0.5, n_samples),
            'gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.6, 0.4]),
            'university_tier': np.random.choice(['Top', 'Mid', 'Other'], n_samples, p=[0.2, 0.3, 0.5])
        })
        
        # 편향된 과거 결정 패턴 (남성 선호, 명문대 선호)
        bias_factors = 1.0
        bias_factors += (hiring_data['gender'] == 'Male') * 0.3  # 남성 선호 편향
        bias_factors += (hiring_data['university_tier'] == 'Top') * 0.4  # 명문대 편향
        bias_factors += np.random.normal(0, 0.1, n_samples)  # 무작위 요소
        
        # 실제 성과는 편향과 무관하게 결정
        actual_performance = (
            0.3 * hiring_data['experience_years'] + 
            0.4 * (hiring_data['skills_test'] - 80) / 12 +
            0.3 * (hiring_data['education_score'] - 75) / 15 +
            np.random.normal(0, 0.2, n_samples)
        )
        
        # 편향된 과거 채용 결정
        biased_decisions = (bias_factors + np.random.normal(0, 0.3, n_samples)) > 1.2
        
        hiring_data['hired_historically'] = biased_decisions
        hiring_data['actual_performance'] = actual_performance
        
        print(f"📊 채용 데이터 현황:")
        print(f"   전체 지원자: {n_samples}")
        male_hire_rate = hiring_data[hiring_data['gender'] == 'Male']['hired_historically'].mean()
        female_hire_rate = hiring_data[hiring_data['gender'] == 'Female']['hired_historically'].mean()
        print(f"   남성 채용률: {male_hire_rate:.1%}")
        print(f"   여성 채용률: {female_hire_rate:.1%}")
        
        top_uni_hire_rate = hiring_data[hiring_data['university_tier'] == 'Top']['hired_historically'].mean()
        other_uni_hire_rate = hiring_data[hiring_data['university_tier'] == 'Other']['hired_historically'].mean()
        print(f"   명문대 채용률: {top_uni_hire_rate:.1%}")
        print(f"   기타대 채용률: {other_uni_hire_rate:.1%}")
        
        # 자동화 시스템이 편향을 학습
        print(f"\n🤖 자동화 AI 채용 시스템:")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        
        # 특성 준비
        features = hiring_data[['experience_years', 'education_score', 'skills_test', 'previous_salary']].copy()
        
        # 편향 요소들을 특성에 포함 (실제로는 AI가 패턴을 찾아냄)
        le_gender = LabelEncoder()
        le_uni = LabelEncoder()
        features['gender_encoded'] = le_gender.fit_transform(hiring_data['gender'])
        features['university_encoded'] = le_uni.fit_transform(hiring_data['university_tier'])
        
        # 편향된 과거 데이터로 학습
        ai_model = RandomForestClassifier(n_estimators=100, random_state=42)
        ai_model.fit(features, hiring_data['hired_historically'])
        
        # 새로운 지원자에 대한 예측
        new_candidates = pd.DataFrame({
            'experience_years': [3, 3],
            'education_score': [82, 82],
            'skills_test': [85, 85],
            'previous_salary': [50000, 50000],
            'gender': ['Male', 'Female'],
            'university_tier': ['Mid', 'Mid']
        })
        
        new_features = new_candidates[['experience_years', 'education_score', 'skills_test', 'previous_salary']].copy()
        new_features['gender_encoded'] = le_gender.transform(new_candidates['gender'])
        new_features['university_encoded'] = le_uni.transform(new_candidates['university_tier'])
        
        ai_predictions = ai_model.predict_proba(new_features)[:, 1]
        
        print(f"   동일한 스펙의 지원자 2명:")
        print(f"   - 남성 지원자 채용 확률: {ai_predictions[0]:.1%}")
        print(f"   - 여성 지원자 채용 확률: {ai_predictions[1]:.1%}")
        print(f"   - 차이: {abs(ai_predictions[0] - ai_predictions[1]):.1%}")
        
        # 특성 중요도 분석
        feature_importance = ai_model.feature_importances_
        feature_names = ['경력', '학점', '스킬테스트', '이전연봉', '성별', '대학등급']
        
        print(f"\n📊 AI 모델의 중요도 순위:")
        importance_pairs = list(zip(feature_names, feature_importance))
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        for i, (name, importance) in enumerate(importance_pairs, 1):
            print(f"   {i}. {name}: {importance:.3f}")
            if name in ['성별', '대학등급'] and importance > 0.1:
                print(f"      ⚠️ 편향 요소가 높은 중요도를 가짐")
        
        return {
            'gender_bias': abs(ai_predictions[0] - ai_predictions[1]),
            'male_hire_rate': male_hire_rate,
            'female_hire_rate': female_hire_rate,
            'feature_importance': dict(importance_pairs),
            'bias_detected': abs(ai_predictions[0] - ai_predictions[1]) > 0.1
        }
    
    def demonstrate_black_box_problems(self):
        """블랙박스 문제의 함정 시연"""
        print("\n⚠️ 함정 4: 블랙박스 문제 (Black Box Problem)")
        print("="*50)
        
        # 복잡한 자동 전처리 파이프라인 시뮬레이션
        np.random.seed(42)
        n_samples = 500
        
        # 원본 데이터
        raw_data = pd.DataFrame({
            'feature_A': np.random.normal(100, 20, n_samples),
            'feature_B': np.random.exponential(2, n_samples),
            'feature_C': np.random.choice(['Low', 'Medium', 'High'], n_samples),
            'target': np.random.normal(50, 10, n_samples)
        })
        
        # 블랙박스 자동 전처리 (과정 숨겨짐)
        print(f"🔮 블랙박스 자동 전처리 실행...")
        
        def mysterious_automl_preprocessing(data):
            """복잡하고 설명하기 어려운 자동 전처리"""
            processed = data.copy()
            
            # 1. 비선형 변환 (사용자 모르게)
            processed['feature_A'] = np.log1p(np.abs(processed['feature_A'] - 100))
            
            # 2. 이상한 스케일링 (독자적 알고리즘)
            processed['feature_B'] = (processed['feature_B'] - processed['feature_B'].quantile(0.1)) / (
                processed['feature_B'].quantile(0.9) - processed['feature_B'].quantile(0.1)
            )
            
            # 3. 복잡한 범주형 인코딩
            category_map = {'Low': -0.7, 'Medium': 0.1, 'High': 1.3}  # 비직관적 매핑
            processed['feature_C'] = processed['feature_C'].map(category_map)
            
            # 4. 숨겨진 특성 생성
            processed['hidden_feature_1'] = (
                processed['feature_A'] * processed['feature_B'] + 
                processed['feature_C'] ** 2
            )
            processed['hidden_feature_2'] = np.sin(processed['feature_A']) * np.exp(processed['feature_C'])
            
            # 5. 비밀 가중치 적용
            secret_weights = [0.3, 0.2, 0.15, 0.2, 0.15]
            feature_cols = ['feature_A', 'feature_B', 'feature_C', 'hidden_feature_1', 'hidden_feature_2']
            
            for i, col in enumerate(feature_cols):
                processed[col] *= secret_weights[i]
            
            return processed
        
        processed_data = mysterious_automl_preprocessing(raw_data)
        
        print(f"   ✅ 전처리 완료")
        print(f"   원본 특성: {len(raw_data.columns)-1}개 → 처리된 특성: {len(processed_data.columns)-1}개")
        
        # 사용자가 변화를 파악하려고 시도
        print(f"\n🔍 사용자의 변화 파악 시도:")
        
        original_stats = raw_data.describe()
        processed_stats = processed_data.describe()
        
        print(f"   Feature A 통계 변화:")
        print(f"   - 원본 평균: {original_stats.loc['mean', 'feature_A']:.2f}")
        print(f"   - 처리 후 평균: {processed_stats.loc['mean', 'feature_A']:.2f}")
        print(f"   - 원본 표준편차: {original_stats.loc['std', 'feature_A']:.2f}")
        print(f"   - 처리 후 표준편차: {processed_stats.loc['std', 'feature_A']:.2f}")
        print(f"   ❓ 어떤 변환이 적용되었는지 알 수 없음")
        
        print(f"\n   Feature B 범위 변화:")
        print(f"   - 원본 범위: [{original_stats.loc['min', 'feature_B']:.2f}, {original_stats.loc['max', 'feature_B']:.2f}]")
        print(f"   - 처리 후 범위: [{processed_stats.loc['min', 'feature_B']:.2f}, {processed_stats.loc['max', 'feature_B']:.2f}]")
        print(f"   ❓ 표준화인지 정규화인지 다른 방법인지 불명")
        
        print(f"\n   새로운 특성들:")
        new_features = set(processed_data.columns) - set(raw_data.columns)
        for feature in new_features:
            print(f"   - {feature}: 생성 방법 알 수 없음")
        
        # 문제 상황: 결과 해석 불가능
        print(f"\n❌ 발생하는 문제들:")
        print(f"   1. 전처리 과정 추적 불가능")
        print(f"   2. 새로운 데이터 적용 시 동일한 변환 보장 어려움") 
        print(f"   3. 도메인 전문가와의 소통 불가")
        print(f"   4. 오류 발생 시 원인 파악 불가")
        print(f"   5. 규제 요구사항 충족 어려움")
        
        # 투명한 전처리와 비교
        print(f"\n✅ 투명한 전처리 대안:")
        
        def transparent_preprocessing(data):
            """단계별로 설명 가능한 전처리"""
            processed = data.copy()
            steps = []
            
            # 1. Feature A: 표준화
            scaler = StandardScaler()
            processed['feature_A'] = scaler.fit_transform(processed[['feature_A']])
            steps.append("Feature A: 표준화 적용 (평균=0, 표준편차=1)")
            
            # 2. Feature B: Min-Max 정규화
            min_val = processed['feature_B'].min()
            max_val = processed['feature_B'].max()
            processed['feature_B'] = (processed['feature_B'] - min_val) / (max_val - min_val)
            steps.append(f"Feature B: Min-Max 정규화 적용 (범위: 0-1)")
            
            # 3. Feature C: 순서형 인코딩
            order_map = {'Low': 0, 'Medium': 1, 'High': 2}
            processed['feature_C'] = processed['feature_C'].map(order_map)
            steps.append("Feature C: 순서형 인코딩 (Low=0, Medium=1, High=2)")
            
            return processed, steps
        
        transparent_data, steps = transparent_preprocessing(raw_data)
        
        print(f"   적용된 전처리 단계:")
        for i, step in enumerate(steps, 1):
            print(f"   {i}. {step}")
        
        return {
            'original_shape': raw_data.shape,
            'blackbox_shape': processed_data.shape,
            'transparent_shape': transparent_data.shape,
            'new_features_count': len(new_features),
            'transparency_advantage': len(steps)
        }
    
    def generate_pitfall_prevention_guide(self):
        """함정 방지 가이드라인 생성"""
        print("\n🛡️ 자동화 전처리 함정 방지 가이드라인")
        print("="*50)
        
        guidelines = {
            'context_preservation': [
                "도메인 전문가와 긴밀히 협력하여 비즈니스 맥락 파악",
                "데이터 생성 과정과 의미를 충분히 이해한 후 전처리 적용", 
                "자동 결측치 처리 전 결측 패턴의 의미 분석",
                "업계 표준과 규제 요구사항을 전처리 규칙에 반영"
            ],
            'overfitting_prevention': [
                "데이터 크기 대비 적절한 수준의 특성 생성",
                "교차 검증을 통한 일반화 성능 지속적 모니터링",
                "특성 선택 기법으로 불필요한 특성 제거",
                "조기 중단 메커니즘으로 과도한 복잡성 방지"
            ],
            'bias_mitigation': [
                "훈련 데이터의 편향성을 사전에 분석하고 보정",
                "공정성 지표를 정의하고 지속적으로 모니터링",
                "다양한 그룹에 대한 성능을 개별적으로 평가",
                "편향 완화 기법을 전처리 파이프라인에 통합"
            ],
            'transparency_maintenance': [
                "모든 전처리 단계를 문서화하고 버전 관리",
                "전처리 파이프라인을 재현 가능하게 구성",
                "각 변환의 목적과 효과를 명확히 기록",
                "자동화된 부분과 수동 개입이 필요한 부분 구분"
            ]
        }
        
        print(f"\n📋 핵심 가이드라인:")
        
        for category, items in guidelines.items():
            category_names = {
                'context_preservation': '1. 맥락 보존',
                'overfitting_prevention': '2. 과적합 방지',
                'bias_mitigation': '3. 편향 완화',
                'transparency_maintenance': '4. 투명성 유지'
            }
            
            print(f"\n{category_names[category]}:")
            for i, item in enumerate(items, 1):
                print(f"   {i}) {item}")
        
        # 체크리스트 제공
        print(f"\n✅ 자동화 전처리 체크리스트:")
        checklist = [
            "도메인 전문가 검토를 받았는가?",
            "전처리 과정이 투명하고 설명 가능한가?",
            "편향성 검사를 수행했는가?",
            "과적합 위험을 평가했는가?",
            "비즈니스 규칙을 위반하지 않는가?",
            "재현 가능한 파이프라인인가?",
            "예외 상황에 대한 대응책이 있는가?",
            "성능 모니터링 체계가 구축되어 있는가?"
        ]
        
        for i, item in enumerate(checklist, 1):
            print(f"   □ {item}")
        
        return guidelines

# 함정 분석기 실행
analyzer = AutomationPitfallAnalyzer()

# 각 함정 시연
context_results = analyzer.demonstrate_context_ignorance()
overfitting_results = analyzer.demonstrate_overfitting_artifacts()
bias_results = analyzer.demonstrate_automation_bias()
blackbox_results = analyzer.demonstrate_black_box_problems()

# 방지 가이드라인
guidelines = analyzer.generate_pitfall_prevention_guide()

print(f"\n" + "="*60)
print("📊 자동화 전처리 함정 분석 요약")
print("="*60)
print(f"✅ 도메인 맥락 차이: {context_results['difference']:.1f}")
print(f"✅ 과적합 탐지: {'예' if overfitting_results['overfitting_detected'] else '아니오'}")
print(f"✅ 편향 탐지: {'예' if bias_results['bias_detected'] else '아니오'} ({bias_results['gender_bias']:.1%} 차이)")
print(f"✅ 블랙박스 특성 수: {blackbox_results['new_features_count']}개")
print(f"✅ 투명성 개선: {blackbox_results['transparency_advantage']}단계 명시")
```

**🔍 코드 해설:**
- `AutomationPitfallAnalyzer` 클래스로 자동화 전처리의 4가지 주요 함정 실증적 시연
- 도메인 맥락 무시, 과적합, 자동화 편향, 블랙박스 문제의 구체적 사례와 해결책 제시
- 각 함정별 탐지 방법과 방지 가이드라인, 체크리스트까지 포함한 실무적 접근

> **📊 이미지 생성 프롬프트:**  
> "Create a comprehensive automation pitfalls visualization showing: 1) Four main pitfall categories (context ignorance, overfitting artifacts, automation bias, black box problems) with warning icons, 2) Before/after comparison charts showing the impact of each pitfall, 3) A prevention framework diagram with checkpoints and validation steps, 4) A traffic light system (green/yellow/red) for automation safety assessment. Use professional warning colors and clear visual hierarchy to emphasize the risks."

---

## 📖 4.4.5 프로젝트: AI 보조 전처리 파이프라인 구축

### 종합 프로젝트 개요

이번 프로젝트에서는 지금까지 학습한 내용을 통합하여 **인간의 지혜와 AI의 효율성을 결합한 하이브리드 전처리 파이프라인**을 구축해보겠습니다.

**🎯 프로젝트 목표:**
- AutoML과 수동 전처리의 최적 조합 설계
- AI 결과 검증과 신뢰도 평가 시스템 구축  
- 함정 방지와 품질 관리 메커니즘 통합
- 실제 프로덕션 환경에 배포 가능한 파이프라인 완성

### 하이브리드 전처리 파이프라인 설계

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class HybridPreprocessingPipeline:
    """
    인간-AI 협업 기반 하이브리드 전처리 파이프라인
    """
    
    def __init__(self, domain_expertise=None, safety_level='high'):
        self.domain_expertise = domain_expertise or {}
        self.safety_level = safety_level  # 'low', 'medium', 'high'
        self.processing_history = []
        self.quality_metrics = {}
        self.ai_suggestions = {}
        self.human_overrides = {}
        self.validation_results = {}
        
    def phase1_ai_exploration(self, df, target_col=None):
        """Phase 1: AI 기반 초기 탐색"""
        print("🤖 Phase 1: AI 기반 초기 데이터 탐색")
        print("="*50)
        
        ai_insights = {
            'data_overview': {},
            'quality_issues': {},
            'suggestions': {},
            'risk_flags': []
        }
        
        # 1.1 기본 데이터 개요
        print(f"\n📊 1.1 자동 데이터 개요 생성:")
        ai_insights['data_overview'] = {
            'shape': df.shape,
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
            'data_types': df.dtypes.value_counts().to_dict(),
            'missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        }
        
        print(f"   데이터 크기: {ai_insights['data_overview']['shape'][0]:,}행 × {ai_insights['data_overview']['shape'][1]}열")
        print(f"   메모리 사용량: {ai_insights['data_overview']['memory_usage']:.1f} MB")
        print(f"   전체 결측률: {ai_insights['data_overview']['missing_percentage']:.2f}%")
        
        # 1.2 품질 문제 자동 탐지
        print(f"\n🔍 1.2 자동 품질 문제 탐지:")
        quality_issues = {}
        
        # 결측치 패턴 분석
        missing_by_col = df.isnull().sum()
        high_missing_cols = missing_by_col[missing_by_col > len(df) * 0.5].index.tolist()
        moderate_missing_cols = missing_by_col[(missing_by_col > len(df) * 0.2) & 
                                             (missing_by_col <= len(df) * 0.5)].index.tolist()
        
        quality_issues['high_missing'] = high_missing_cols
        quality_issues['moderate_missing'] = moderate_missing_cols
        
        # 이상치 탐지 (수치형 컬럼)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_cols = []
        
        for col in numeric_cols:
            if df[col].notna().sum() > 10:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
                outlier_rate = outliers / df[col].notna().sum()
                
                if outlier_rate > 0.05:  # 5% 이상 이상치
                    outlier_cols.append((col, outlier_rate))
        
        quality_issues['outlier_columns'] = outlier_cols
        
        # 중복 데이터 확인
        duplicate_count = df.duplicated().sum()
        quality_issues['duplicates'] = duplicate_count
        
        ai_insights['quality_issues'] = quality_issues
        
        if high_missing_cols:
            print(f"   ⚠️ 심각한 결측치: {len(high_missing_cols)}개 컬럼")
            for col in high_missing_cols[:3]:
                print(f"      - {col}: {missing_by_col[col]/len(df)*100:.1f}% 결측")
        
        if outlier_cols:
            print(f"   ⚠️ 이상치 문제: {len(outlier_cols)}개 컬럼")
            for col, rate in outlier_cols[:3]:
                print(f"      - {col}: {rate*100:.1f}% 이상치")
        
        if duplicate_count > 0:
            print(f"   ⚠️ 중복 데이터: {duplicate_count}개 ({duplicate_count/len(df)*100:.1f}%)")
        
        # 1.3 AI 자동 제안 생성
        print(f"\n💡 1.3 AI 자동 제안 생성:")
        suggestions = []
        
        # 결측치 처리 제안
        for col in high_missing_cols:
            suggestions.append({
                'type': 'removal',
                'target': col,
                'reason': f'너무 많은 결측값 ({missing_by_col[col]/len(df)*100:.1f}%)',
                'confidence': 0.9,
                'risk': 'low'
            })
        
        for col in moderate_missing_cols:
            if col in numeric_cols:
                suggestions.append({
                    'type': 'imputation',
                    'target': col,
                    'method': 'median',
                    'reason': f'중간 수준 결측값 ({missing_by_col[col]/len(df)*100:.1f}%)',
                    'confidence': 0.7,
                    'risk': 'medium'
                })
        
        # 이상치 처리 제안
        for col, rate in outlier_cols:
            if rate > 0.15:  # 15% 이상
                suggestions.append({
                    'type': 'outlier_investigation',
                    'target': col,
                    'reason': f'많은 이상치 ({rate*100:.1f}%)',
                    'confidence': 0.8,
                    'risk': 'high'
                })
            else:
                suggestions.append({
                    'type': 'outlier_capping',
                    'target': col,
                    'method': 'iqr',
                    'reason': f'보통 수준 이상치 ({rate*100:.1f}%)',
                    'confidence': 0.6,
                    'risk': 'medium'
                })
        
        # 중복 제거 제안
        if duplicate_count > 0:
            suggestions.append({
                'type': 'duplicate_removal',
                'target': 'all',
                'reason': f'{duplicate_count}개 중복 레코드',
                'confidence': 0.95,
                'risk': 'low'
            })
        
        ai_insights['suggestions'] = suggestions
        
        print(f"   총 {len(suggestions)}개 제안 생성")
        for i, suggestion in enumerate(suggestions[:5], 1):  # 상위 5개만 표시
            confidence_emoji = "🟢" if suggestion['confidence'] > 0.8 else "🟡" if suggestion['confidence'] > 0.6 else "🔴"
            print(f"   {i}. {suggestion['type']}: {suggestion['target']} {confidence_emoji}")
            print(f"      이유: {suggestion['reason']}")
        
        # 1.4 위험 신호 탐지
        print(f"\n🚨 1.4 위험 신호 탐지:")
        risk_flags = []
        
        # 데이터 크기 vs 복잡성
        complexity_ratio = len(df.columns) / len(df)
        if complexity_ratio > 0.1:  # 컬럼이 너무 많음
            risk_flags.append(f"높은 차원: 컬럼/행 비율 {complexity_ratio:.3f}")
        
        # 결측치 패턴
        if ai_insights['data_overview']['missing_percentage'] > 20:
            risk_flags.append(f"높은 결측률: {ai_insights['data_overview']['missing_percentage']:.1f}%")
        
        # 메모리 사용량
        if ai_insights['data_overview']['memory_usage'] > 500:  # 500MB 이상
            risk_flags.append(f"큰 데이터: {ai_insights['data_overview']['memory_usage']:.1f}MB")
        
        ai_insights['risk_flags'] = risk_flags
        
        if risk_flags:
            for flag in risk_flags:
                print(f"   ⚠️ {flag}")
        else:
            print("   ✅ 특별한 위험 신호 없음")
        
        return ai_insights
    
    def phase2_human_review(self, df, ai_insights, domain_rules=None):
        """Phase 2: 인간 전문가 검토 및 개입"""
        print(f"\n👨‍💼 Phase 2: 인간 전문가 검토 및 개입")
        print("="*50)
        
        human_decisions = {
            'approved_suggestions': [],
            'rejected_suggestions': [],
            'custom_rules': [],
            'domain_overrides': {}
        }
        
        # 2.1 도메인 지식 기반 검토
        print(f"\n🧠 2.1 도메인 지식 기반 검토:")
        
        # 예시: 부동산 도메인 규칙들
        default_domain_rules = {
            'house_prices': {
                'SalePrice': {'min': 10000, 'max': 1000000, 'negative_allowed': False},
                'YearBuilt': {'min': 1800, 'max': 2030},
                'GrLivArea': {'min': 300, 'max': 5000, 'unit': 'sqft'},
                'LotArea': {'min': 1000, 'max': 100000, 'unit': 'sqft'},
                'missing_patterns': {
                    'GarageArea': 'if no garage, missing is expected',
                    'PoolArea': 'if no pool, missing is expected'
                }
            }
        }
        
        applied_rules = domain_rules or default_domain_rules.get('house_prices', {})
        
        if applied_rules:
            print(f"   ✅ 도메인 규칙 적용: {len(applied_rules)}개 규칙")
            
            # 도메인 규칙 위반 검사
            violations = []
            for col, rules in applied_rules.items():
                if col in df.columns and 'min' in rules and 'max' in rules:
                    if df[col].dtype in ['int64', 'float64']:
                        min_violations = (df[col] < rules['min']).sum()
                        max_violations = (df[col] > rules['max']).sum()
                        
                        if min_violations > 0 or max_violations > 0:
                            violations.append({
                                'column': col,
                                'min_violations': min_violations,
                                'max_violations': max_violations,
                                'rules': rules
                            })
            
            if violations:
                print(f"   ⚠️ 도메인 규칙 위반 발견:")
                for violation in violations:
                    print(f"      - {violation['column']}: {violation['min_violations']}개 최솟값 위반, {violation['max_violations']}개 최댓값 위반")
        else:
            print(f"   📝 도메인 규칙 없음 - 일반적 규칙 적용")
        
        # 2.2 AI 제안 검토
        print(f"\n🔍 2.2 AI 제안 검토:")
        
        for i, suggestion in enumerate(ai_insights['suggestions']):
            print(f"\n   제안 {i+1}: {suggestion['type']} - {suggestion['target']}")
            print(f"   이유: {suggestion['reason']}")
            print(f"   AI 신뢰도: {suggestion['confidence']:.1%}")
            print(f"   위험도: {suggestion['risk']}")
            
            # 안전 수준에 따른 자동 승인/거부
            if self.safety_level == 'high':
                # 높은 안전: 신뢰도 90% 이상이고 위험도 낮은 것만 자동 승인
                if suggestion['confidence'] >= 0.9 and suggestion['risk'] == 'low':
                    decision = 'approved'
                    print(f"   👍 자동 승인 (높은 신뢰도 + 낮은 위험)")
                else:
                    decision = 'human_review'
                    print(f"   👀 인간 검토 필요")
            elif self.safety_level == 'medium':
                # 중간 안전: 신뢰도 80% 이상이고 위험도 중간 이하
                if suggestion['confidence'] >= 0.8 and suggestion['risk'] in ['low', 'medium']:
                    decision = 'approved'
                    print(f"   👍 자동 승인 (중간 신뢰도)")
                else:
                    decision = 'human_review'
                    print(f"   👀 인간 검토 필요")
            else:  # low safety
                # 낮은 안전: 신뢰도 70% 이상은 모두 승인
                if suggestion['confidence'] >= 0.7:
                    decision = 'approved'
                    print(f"   👍 자동 승인 (기본 신뢰도)")
                else:
                    decision = 'rejected'
                    print(f"   👎 자동 거부 (낮은 신뢰도)")
            
            # 도메인 지식과 충돌하는 경우 인간 검토 요구
            if suggestion['target'] in applied_rules:
                if suggestion['type'] == 'removal':
                    decision = 'human_review'
                    print(f"   🤔 도메인 규칙 충돌 - 인간 검토 필요")
            
            # 결정 기록
            if decision == 'approved':
                human_decisions['approved_suggestions'].append(suggestion)
            elif decision == 'rejected':
                human_decisions['rejected_suggestions'].append(suggestion)
            else:
                # 실제로는 여기서 인간이 개입하여 결정
                # 시뮬레이션에서는 도메인 지식 기반으로 자동 결정
                if suggestion['risk'] == 'high' or suggestion['confidence'] < 0.5:
                    human_decisions['rejected_suggestions'].append(suggestion)
                    print(f"   👎 인간이 거부")
                else:
                    human_decisions['approved_suggestions'].append(suggestion)
                    print(f"   👍 인간이 승인")
        
        # 2.3 커스텀 규칙 추가
        print(f"\n⚙️ 2.3 커스텀 전처리 규칙 추가:")
        custom_rules = [
            {
                'name': 'business_hour_normalization',
                'description': '영업시간 데이터를 24시간 형식으로 정규화',
                'apply_if': 'time_related_columns_exist',
                'priority': 'high'
            },
            {
                'name': 'currency_standardization', 
                'description': '다양한 통화를 USD로 통일',
                'apply_if': 'currency_columns_exist',
                'priority': 'medium'
            },
            {
                'name': 'outlier_business_validation',
                'description': '이상치를 비즈니스 맥락에서 재검토',
                'apply_if': 'outliers_detected',
                'priority': 'high'
            }
        ]
        
        human_decisions['custom_rules'] = custom_rules
        
        for rule in custom_rules:
            print(f"   + {rule['name']}: {rule['description']}")
        
        print(f"\n📊 인간 검토 결과:")
        print(f"   승인된 AI 제안: {len(human_decisions['approved_suggestions'])}개")
        print(f"   거부된 AI 제안: {len(human_decisions['rejected_suggestions'])}개") 
        print(f"   추가 커스텀 규칙: {len(human_decisions['custom_rules'])}개")
        
        return human_decisions
    
    def phase3_hybrid_execution(self, df, ai_insights, human_decisions):
        """Phase 3: 하이브리드 전처리 실행"""
        print(f"\n⚙️ Phase 3: 하이브리드 전처리 실행")
        print("="*50)
        
        processed_df = df.copy()
        execution_log = []
        
        # 3.1 안전한 자동 처리 (승인된 AI 제안)
        print(f"\n🤖 3.1 안전한 자동 처리:")
        
        for suggestion in human_decisions['approved_suggestions']:
            try:
                if suggestion['type'] == 'duplicate_removal':
                    before_count = len(processed_df)
                    processed_df = processed_df.drop_duplicates()
                    after_count = len(processed_df)
                    
                    execution_log.append({
                        'step': 'duplicate_removal',
                        'target': 'all',
                        'before_count': before_count,
                        'after_count': after_count,
                        'removed': before_count - after_count
                    })
                    print(f"   ✅ 중복 제거: {before_count - after_count}개 행 제거")
                
                elif suggestion['type'] == 'removal' and suggestion['target'] != 'all':
                    col = suggestion['target']
                    if col in processed_df.columns:
                        processed_df = processed_df.drop(columns=[col])
                        execution_log.append({
                            'step': 'column_removal',
                            'target': col,
                            'reason': suggestion['reason']
                        })
                        print(f"   ✅ 컬럼 제거: {col}")
                
                elif suggestion['type'] == 'imputation':
                    col = suggestion['target']
                    method = suggestion.get('method', 'median')
                    
                    if col in processed_df.columns:
                        missing_before = processed_df[col].isnull().sum()
                        
                        if method == 'median':
                            fill_value = processed_df[col].median()
                        elif method == 'mean':
                            fill_value = processed_df[col].mean()
                        else:
                            fill_value = processed_df[col].mode().iloc[0] if len(processed_df[col].mode()) > 0 else 0
                        
                        processed_df[col].fillna(fill_value, inplace=True)
                        missing_after = processed_df[col].isnull().sum()
                        
                        execution_log.append({
                            'step': 'imputation',
                            'target': col,
                            'method': method,
                            'fill_value': fill_value,
                            'missing_before': missing_before,
                            'missing_after': missing_after
                        })
                        print(f"   ✅ 결측치 대체: {col} ({method}) - {missing_before - missing_after}개 대체")
                
                elif suggestion['type'] == 'outlier_capping':
                    col = suggestion['target']
                    if col in processed_df.columns and processed_df[col].dtype in ['int64', 'float64']:
                        Q1 = processed_df[col].quantile(0.25)
                        Q3 = processed_df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        outliers_before = ((processed_df[col] < lower_bound) | (processed_df[col] > upper_bound)).sum()
                        
                        processed_df[col] = processed_df[col].clip(lower_bound, upper_bound)
                        
                        outliers_after = ((processed_df[col] < lower_bound) | (processed_df[col] > upper_bound)).sum()
                        
                        execution_log.append({
                            'step': 'outlier_capping',
                            'target': col,
                            'lower_bound': lower_bound,
                            'upper_bound': upper_bound,
                            'outliers_before': outliers_before,
                            'outliers_after': outliers_after
                        })
                        print(f"   ✅ 이상치 제한: {col} - {outliers_before}개 → {outliers_after}개")
                
            except Exception as e:
                print(f"   ❌ 처리 실패: {suggestion['target']} - {str(e)}")
                execution_log.append({
                    'step': 'error',
                    'target': suggestion['target'],
                    'error': str(e)
                })
        
        # 3.2 수동 검증이 필요한 처리
        print(f"\n👨‍💼 3.2 수동 검증 처리:")
        
        # 여기서는 시뮬레이션으로 일부 수동 처리 시연
        manual_steps = [
            '비즈니스 규칙 위반 데이터 개별 검토',
            '도메인 전문가와 이상치 케이스 논의',
            '새로운 특성 생성을 위한 도메인 지식 적용'
        ]
        
        for step in manual_steps:
            print(f"   🔍 {step}")
        
        # 3.3 품질 검증
        print(f"\n🎯 3.3 처리 결과 품질 검증:")
        
        quality_check = {
            'data_integrity': True,
            'missing_reduction': 0,
            'outlier_reduction': 0,
            'issues': []
        }
        
        # 결측치 감소 확인
        original_missing = df.isnull().sum().sum()
        processed_missing = processed_df.isnull().sum().sum()
        quality_check['missing_reduction'] = original_missing - processed_missing
        
        # 이상치 감소 확인 (수치형 컬럼)
        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
        original_outliers = 0
        processed_outliers = 0
        
        for col in numeric_cols:
            if col in df.columns and df[col].notna().sum() > 10:
                # 원본 이상치 수
                Q1_orig = df[col].quantile(0.25)
                Q3_orig = df[col].quantile(0.75)
                IQR_orig = Q3_orig - Q1_orig
                original_outliers += ((df[col] < Q1_orig - 1.5 * IQR_orig) | 
                                    (df[col] > Q3_orig + 1.5 * IQR_orig)).sum()
                
                # 처리된 이상치 수
                if col in processed_df.columns:
                    Q1_proc = processed_df[col].quantile(0.25)
                    Q3_proc = processed_df[col].quantile(0.75)
                    IQR_proc = Q3_proc - Q1_proc
                    processed_outliers += ((processed_df[col] < Q1_proc - 1.5 * IQR_proc) | 
                                         (processed_df[col] > Q3_proc + 1.5 * IQR_proc)).sum()
        
        quality_check['outlier_reduction'] = original_outliers - processed_outliers
        
        # 데이터 무결성 확인
        if len(processed_df) == 0:
            quality_check['data_integrity'] = False
            quality_check['issues'].append("모든 데이터가 제거됨")
        
        if processed_df.shape[1] == 0:
            quality_check['data_integrity'] = False
            quality_check['issues'].append("모든 컬럼이 제거됨")
        
        print(f"   원본: {df.shape[0]:,}행 × {df.shape[1]}열")
        print(f"   처리 후: {processed_df.shape[0]:,}행 × {processed_df.shape[1]}열")
        print(f"   결측치 감소: {quality_check['missing_reduction']:,}개")
        print(f"   이상치 감소: {quality_check['outlier_reduction']:,}개")
        
        if quality_check['data_integrity']:
            print(f"   ✅ 데이터 무결성 유지")
        else:
            print(f"   ❌ 데이터 무결성 문제:")
            for issue in quality_check['issues']:
                print(f"      - {issue}")
        
        return processed_df, execution_log, quality_check
    
    def phase4_performance_validation(self, original_df, processed_df, target_col=None):
        """Phase 4: 성능 검증 및 비교"""
        print(f"\n📊 Phase 4: 성능 검증 및 비교")
        print("="*50)
        
        validation_results = {
            'model_performance': {},
            'statistical_comparison': {},
            'recommendation': ''
        }
        
        if target_col and target_col in original_df.columns and target_col in processed_df.columns:
            print(f"\n🎯 4.1 예측 모델 성능 비교:")
            
            try:
                # 원본 데이터 모델
                orig_features = original_df.drop(columns=[target_col]).select_dtypes(include=[np.number])
                orig_target = original_df[target_col]
                
                # 결측치가 있는 행 제거 (원본)
                orig_complete_mask = orig_features.notna().all(axis=1) & orig_target.notna()
                orig_X = orig_features[orig_complete_mask]
                orig_y = orig_target[orig_complete_mask]
                
                if len(orig_X) > 10:
                    orig_model = RandomForestRegressor(n_estimators=50, random_state=42)
                    orig_scores = cross_val_score(orig_model, orig_X, orig_y, cv=5, scoring='r2')
                    orig_mean_score = orig_scores.mean()
                    orig_std_score = orig_scores.std()
                    
                    print(f"   원본 데이터 성능: R² = {orig_mean_score:.3f} (±{orig_std_score:.3f})")
                    validation_results['model_performance']['original'] = {
                        'r2_mean': orig_mean_score,
                        'r2_std': orig_std_score,
                        'sample_size': len(orig_X)
                    }
                else:
                    print(f"   원본 데이터: 완전한 케이스 부족 ({len(orig_X)}개)")
                    orig_mean_score = 0
                
                # 처리된 데이터 모델
                proc_features = processed_df.drop(columns=[target_col], errors='ignore').select_dtypes(include=[np.number])
                proc_target = processed_df[target_col] if target_col in processed_df.columns else None
                
                if proc_target is not None and len(proc_features) > 10:
                    # 결측치가 있는 행 제거 (처리된 데이터)
                    proc_complete_mask = proc_features.notna().all(axis=1) & proc_target.notna()
                    proc_X = proc_features[proc_complete_mask]
                    proc_y = proc_target[proc_complete_mask]
                    
                    if len(proc_X) > 10:
                        proc_model = RandomForestRegressor(n_estimators=50, random_state=42)
                        proc_scores = cross_val_score(proc_model, proc_X, proc_y, cv=5, scoring='r2')
                        proc_mean_score = proc_scores.mean()
                        proc_std_score = proc_scores.std()
                        
                        print(f"   처리된 데이터 성능: R² = {proc_mean_score:.3f} (±{proc_std_score:.3f})")
                        validation_results['model_performance']['processed'] = {
                            'r2_mean': proc_mean_score,
                            'r2_std': proc_std_score,
                            'sample_size': len(proc_X)
                        }
                        
                        # 성능 비교
                        improvement = proc_mean_score - orig_mean_score
                        print(f"   성능 변화: {improvement:+.3f}")
                        
                        if improvement > 0.05:
                            print(f"   ✅ 상당한 성능 향상")
                        elif improvement > 0.01:
                            print(f"   ✅ 약간의 성능 향상")
                        elif improvement > -0.01:
                            print(f"   ➖ 성능 유지")
                        else:
                            print(f"   ❌ 성능 저하")
                    else:
                        print(f"   처리된 데이터: 완전한 케이스 부족 ({len(proc_X)}개)")
                else:
                    print(f"   처리된 데이터에서 타겟 컬럼 없음")
                
            except Exception as e:
                print(f"   ❌ 모델 성능 비교 실패: {str(e)}")
        
        # 4.2 통계적 특성 비교
        print(f"\n📈 4.2 통계적 특성 비교:")
        
        # 공통 수치형 컬럼들에 대해 통계 비교
        orig_numeric = original_df.select_dtypes(include=[np.number])
        proc_numeric = processed_df.select_dtypes(include=[np.number])
        common_cols = set(orig_numeric.columns) & set(proc_numeric.columns)
        
        stat_comparison = {}
        for col in common_cols:
            if original_df[col].notna().sum() > 0 and processed_df[col].notna().sum() > 0:
                orig_mean = original_df[col].mean()
                proc_mean = processed_df[col].mean()
                orig_std = original_df[col].std()
                proc_std = processed_df[col].std()
                
                mean_change_pct = abs(proc_mean - orig_mean) / abs(orig_mean) * 100 if orig_mean != 0 else 0
                std_change_pct = abs(proc_std - orig_std) / abs(orig_std) * 100 if orig_std != 0 else 0
                
                stat_comparison[col] = {
                    'mean_change_pct': mean_change_pct,
                    'std_change_pct': std_change_pct
                }
                
                if mean_change_pct > 10 or std_change_pct > 10:
                    print(f"   ⚠️ {col}: 평균 {mean_change_pct:.1f}% 변화, 표준편차 {std_change_pct:.1f}% 변화")
        
        validation_results['statistical_comparison'] = stat_comparison
        
        if not stat_comparison:
            print(f"   ✅ 큰 통계적 변화 없음")
        
        # 4.3 최종 권고사항
        print(f"\n💡 4.3 최종 권고사항:")
        
        total_score = 0
        factors = []
        
        # 성능 점수
        if 'processed' in validation_results['model_performance'] and 'original' in validation_results['model_performance']:
            perf_improvement = (validation_results['model_performance']['processed']['r2_mean'] - 
                              validation_results['model_performance']['original']['r2_mean'])
            if perf_improvement > 0.05:
                total_score += 2
                factors.append("상당한 성능 향상")
            elif perf_improvement > 0.01:
                total_score += 1
                factors.append("약간의 성능 향상")
            elif perf_improvement > -0.01:
                total_score += 0
                factors.append("성능 유지")
            else:
                total_score -= 1
                factors.append("성능 저하")
        
        # 통계적 안정성 점수
        major_changes = sum(1 for changes in stat_comparison.values() 
                          if changes['mean_change_pct'] > 10 or changes['std_change_pct'] > 10)
        if major_changes == 0:
            total_score += 1
            factors.append("통계적 안정성 유지")
        elif major_changes <= 2:
            total_score += 0
            factors.append("일부 통계적 변화")
        else:
            total_score -= 1
            factors.append("많은 통계적 변화")
        
        # 데이터 보존 점수
        data_retention = processed_df.shape[0] / original_df.shape[0]
        if data_retention >= 0.95:
            total_score += 1
            factors.append("높은 데이터 보존율")
        elif data_retention >= 0.85:
            total_score += 0
            factors.append("보통 데이터 보존율")
        else:
            total_score -= 1
            factors.append("낮은 데이터 보존율")
        
        # 최종 결정
        if total_score >= 2:
            recommendation = "전처리 결과 적용 권장 ✅"
        elif total_score >= 0:
            recommendation = "조건부 적용 (추가 검토 필요) ⚠️"
        else:
            recommendation = "적용 비권장 (재작업 필요) ❌"
        
        validation_results['recommendation'] = recommendation
        
        print(f"   종합 점수: {total_score}/4")
        print(f"   평가 요소: {', '.join(factors)}")
        print(f"   최종 권고: {recommendation}")
        
        return validation_results
    
    def generate_pipeline_report(self, original_df, processed_df, execution_log, validation_results):
        """종합 파이프라인 보고서 생성"""
        print(f"\n📋 하이브리드 전처리 파이프라인 종합 보고서")
        print("="*60)
        
        # 전체 요약
        print(f"\n📊 처리 요약:")
        print(f"   원본 데이터: {original_df.shape[0]:,}행 × {original_df.shape[1]}열")
        print(f"   처리된 데이터: {processed_df.shape[0]:,}행 × {processed_df.shape[1]}열")
        print(f"   행 변화: {processed_df.shape[0] - original_df.shape[0]:+,}개")
        print(f"   열 변화: {processed_df.shape[1] - original_df.shape[1]:+,}개")
        
        # 실행 단계 요약
        print(f"\n⚙️ 실행된 처리 단계:")
        step_counts = {}
        for log in execution_log:
            step_type = log.get('step', 'unknown')
            step_counts[step_type] = step_counts.get(step_type, 0) + 1
        
        for step_type, count in step_counts.items():
            print(f"   {step_type}: {count}회")
        
        # 성능 결과
        if validation_results['model_performance']:
            print(f"\n📈 성능 검증 결과:")
            if 'original' in validation_results['model_performance']:
                orig_perf = validation_results['model_performance']['original']
                print(f"   원본 모델 R²: {orig_perf['r2_mean']:.3f} (샘플: {orig_perf['sample_size']:,})")
            
            if 'processed' in validation_results['model_performance']:
                proc_perf = validation_results['model_performance']['processed']
                print(f"   처리된 모델 R²: {proc_perf['r2_mean']:.3f} (샘플: {proc_perf['sample_size']:,})")
                
                if 'original' in validation_results['model_performance']:
                    improvement = proc_perf['r2_mean'] - orig_perf['r2_mean']
                    print(f"   성능 변화: {improvement:+.3f}")
        
        # 최종 권고사항
        print(f"\n💡 최종 권고사항:")
        print(f"   {validation_results['recommendation']}")
        
        # 파이프라인 메타데이터
        print(f"\n🔧 파이프라인 설정:")
        print(f"   안전 수준: {self.safety_level}")
        print(f"   도메인 규칙: {'적용됨' if self.domain_expertise else '없음'}")
        print(f"   총 처리 시간: 시뮬레이션")
        
        return {
            'original_shape': original_df.shape,
            'processed_shape': processed_df.shape,
            'execution_summary': step_counts,
            'performance_results': validation_results['model_performance'],
            'final_recommendation': validation_results['recommendation']
        }

# 하이브리드 파이프라인 실행 데모
print("🚀 하이브리드 전처리 파이프라인 실행 데모")
print("="*60)

# 테스트 데이터 생성 (문제가 있는 House Prices 스타일 데이터)
np.random.seed(42)
n_samples = 800

demo_data = pd.DataFrame({
    'SalePrice': np.random.lognormal(12, 0.5, n_samples),
    'GrLivArea': np.random.normal(1500, 400, n_samples),
    'YearBuilt': np.random.randint(1920, 2023, n_samples),
    'TotalBsmtSF': np.random.normal(1000, 300, n_samples),
    'GarageArea': np.random.normal(500, 150, n_samples),
    'LotArea': np.random.exponential(9000, n_samples),
    'Neighborhood': np.random.choice(['A', 'B', 'C'], n_samples, p=[0.3, 0.5, 0.2])
})

# 의도적으로 문제 추가
# 결측치
demo_data.loc[np.random.choice(n_samples, 100, replace=False), 'GarageArea'] = np.nan
demo_data.loc[np.random.choice(n_samples, 80, replace=False), 'TotalBsmtSF'] = np.nan

# 이상치
demo_data.loc[np.random.choice(n_samples, 30, replace=False), 'SalePrice'] = demo_data['SalePrice'] * 5
demo_data.loc[np.random.choice(n_samples, 20, replace=False), 'GrLivArea'] = 10000

# 중복 데이터
for i in range(25):
    demo_data.loc[n_samples + i] = demo_data.loc[i % 100]

# 잘못된 연도
demo_data.loc[np.random.choice(len(demo_data), 15, replace=False), 'YearBuilt'] = 1800

print(f"📊 테스트 데이터 생성 완료: {demo_data.shape[0]:,}행 × {demo_data.shape[1]}열")

# 하이브리드 파이프라인 실행
pipeline = HybridPreprocessingPipeline(safety_level='high')

# Phase 1: AI 탐색
ai_insights = pipeline.phase1_ai_exploration(demo_data, 'SalePrice')

# Phase 2: 인간 검토
human_decisions = pipeline.phase2_human_review(demo_data, ai_insights)

# Phase 3: 하이브리드 실행
processed_data, execution_log, quality_check = pipeline.phase3_hybrid_execution(
    demo_data, ai_insights, human_decisions
)

# Phase 4: 성능 검증
validation_results = pipeline.phase4_performance_validation(
    demo_data, processed_data, 'SalePrice'
)

# 최종 보고서
final_report = pipeline.generate_pipeline_report(
    demo_data, processed_data, execution_log, validation_results
)

print(f"\n🎉 하이브리드 전처리 파이프라인 완료!")
print(f"최종 데이터: {processed_data.shape[0]:,}행 × {processed_data.shape[1]}열")
print(f"권고사항: {validation_results['recommendation']}")
```

**🔍 코드 해설:**
- `HybridPreprocessingPipeline` 클래스로 인간-AI 협업 기반 전처리 시스템 완전 구현
- 4단계 프로세스 (AI 탐색 → 인간 검토 → 하이브리드 실행 → 성능 검증)
- 안전 수준별 자동화 정도 조절과 도메인 지식 통합 메커니즘
- 전체 과정의 투명성과 재현성을 보장하는 종합 파이프라인

> **📊 이미지 생성 프롬프트:**  
> "Create a comprehensive hybrid preprocessing pipeline visualization showing: 1) A 4-phase workflow diagram (AI exploration → Human review → Hybrid execution → Performance validation), 2) A decision tree for automation safety levels, 3) Before/after data quality comparison dashboard, 4) Performance validation metrics with model comparison charts, 5) A final recommendation system with traffic light indicators. Use modern data science styling with clear process flow and validation checkpoints."

---

## 📝 요약 및 핵심 정리

### 🎯 Part 4 핵심 성취

이번 Part에서는 **AI 도구를 활용한 자동 전처리의 양면성**을 깊이 있게 탐구했습니다. 

**✅ 주요 학습 성과:**

1. **AutoML 전처리 도구 완전 마스터**
   - H2O, Google AutoML, DataRobot, AutoGluon, TPOT 5개 주요 플랫폼 비교 분석
   - 각 도구의 장단점과 적용 시나리오를 명확히 구분
   - 실무적 선택 기준과 비용 효율성 평가 방법 습득

2. **AI 기반 특성 생성 시스템 구축**
   - 수학적 조합, 통계적 변환, 상호작용 특성 등 다양한 자동 생성 전략
   - 상호정보량과 상관계수를 활용한 객관적 특성 평가 시스템
   - 60개 이상 특성 생성 후 상위 30개 선별하는 지능형 필터링

3. **지능형 데이터 클리닝 완전 정복**
   - 5차원 품질 평가 (완전성, 일관성, 정확성, 유효성, 고유성)
   - Isolation Forest, DBSCAN, 통계적 방법을 통합한 다층적 이상 탐지
   - 4차원 검증 프레임워크와 A-D 등급 신뢰도 평가 시스템

4. **자동화 함정 방지 전문가 역량**
   - 도메인 맥락 무시, 과적합, 자동화 편향, 블랙박스 문제의 구체적 사례 학습
   - 각 함정별 탐지 방법과 8단계 체크리스트 기반 방지 가이드라인
   - 실증적 시연을 통한 함정의 실제 영향과 해결책 체험

5. **하이브리드 전처리 파이프라인 구축**
   - 인간의 지혜와 AI의 효율성을 최적 조합한 4단계 파이프라인
   - 안전 수준별 자동화 정도 조절과 도메인 지식 통합 메커니즘
   - 투명성과 재현성을 보장하는 완전한 프로덕션 시스템

### 🌟 Part 4의 핵심 메시지

> **"AI는 강력한 조수이지만, 선장은 여전히 인간이어야 합니다."**

AI 자동 전처리는 분명 혁신적인 도구이지만, **맹목적 신뢰는 위험**합니다. 진정한 데이터 분석 전문가는:

- **AI의 힘을 활용하되 그 한계를 정확히 이해**하고
- **도메인 지식과 비즈니스 맥락을 절대 놓치지 않으며**
- **투명성과 검증을 통해 신뢰할 수 있는 결과를 보장**합니다

### 📊 실무 적용 가이드라인

**🔹 AutoML 도구 선택 시:**
- 데이터 크기와 복잡도에 맞는 플랫폼 선택
- 비용과 성능의 균형점 고려
- 해석 가능성 요구사항 사전 확인

**🔹 AI 특성 생성 활용 시:**
- 도메인 지식 기반 사전 검토 필수
- 과적합 방지를 위한 교차 검증 강화
- 특성 선택을 통한 차원 축소 고려

**🔹 지능형 클리닝 적용 시:**
- 5차원 품질 평가로 종합적 진단
- 다층적 이상 탐지로 정확도 향상
- 신뢰도 점수 기반 결과 활용 결정

**🔹 함정 방지를 위해:**
- 8단계 체크리스트 의무적 적용
- 도메인 전문가와의 정기적 검토
- 편향성 모니터링 체계 구축

---

## 🎮 직접 해보기

### ⭐⭐ 연습문제 1: AutoML 도구 비교 분석
**목표:** 다양한 AutoML 플랫폼의 특성을 이해하고 상황별 최적 선택 능력 배양

**과제:**
주어진 3가지 시나리오에 대해 가장 적합한 AutoML 플랫폼을 선택하고 그 이유를 설명하세요.

```python
# 시나리오별 요구사항
scenarios = {
    'startup_mvp': {
        'budget': 'limited',
        'timeline': '2_weeks', 
        'data_size': 'small_10k_rows',
        'team_skill': 'beginner',
        'interpretability': 'medium'
    },
    'enterprise_finance': {
        'budget': 'high',
        'timeline': '3_months',
        'data_size': 'large_1m_rows', 
        'team_skill': 'expert',
        'interpretability': 'critical'
    },
    'research_project': {
        'budget': 'medium',
        'timeline': '6_months',
        'data_size': 'medium_100k_rows',
        'team_skill': 'advanced', 
        'interpretability': 'high'
    }
}

def recommend_automl_platform(scenario_requirements):
    """
    주어진 요구사항에 맞는 AutoML 플랫폼 추천
    
    고려해야 할 플랫폼들:
    - H2O AutoML: 강력하지만 복잡
    - Google AutoML: 사용 쉽지만 비쌈
    - DataRobot: 엔터프라이즈급, 매우 비쌈
    - AutoGluon: 무료, AWS 최적화
    - TPOT: 무료, 연구 친화적
    """
    # 여기에 추천 로직 구현
    pass

# TODO: 각 시나리오별로 함수를 구현하고 선택 이유를 작성하세요
```

**힌트:** 비용, 사용 편의성, 성능, 해석 가능성, 확장성을 종합적으로 고려하세요.

---

### ⭐⭐⭐ 연습문제 2: AI 특성 생성 시스템 구현
**목표:** 도메인 지식을 활용한 지능형 특성 생성 시스템 설계 및 구현

**과제:**
전자상거래 고객 데이터에 대해 AI 기반 특성 생성 시스템을 구현하고, 생성된 특성의 비즈니스 가치를 평가하세요.

```python
import pandas as pd
import numpy as np

# 전자상거래 고객 데이터 (예시)
np.random.seed(42)
n_customers = 1000

ecommerce_data = pd.DataFrame({
    'customer_id': range(1, n_customers + 1),
    'age': np.random.normal(35, 12, n_customers),
    'income': np.random.lognormal(10, 0.8, n_customers),
    'total_purchases': np.random.poisson(15, n_customers),
    'avg_order_value': np.random.lognormal(4, 0.6, n_customers),
    'days_since_last_purchase': np.random.exponential(30, n_customers),
    'website_visits_per_month': np.random.poisson(8, n_customers),
    'mobile_app_usage_hours': np.random.exponential(2, n_customers),
    'customer_service_contacts': np.random.poisson(2, n_customers),
    'return_rate': np.random.beta(2, 8, n_customers),  # 0-1 사이
    'product_categories_purchased': np.random.poisson(3, n_customers),
    'customer_lifetime_value': np.random.lognormal(6, 1, n_customers)  # 타겟 변수
})

class EcommerceFeatureGenerator:
    """전자상거래 특화 AI 특성 생성기"""
    
    def __init__(self):
        self.domain_knowledge = {
            'customer_segments': ['new', 'regular', 'vip', 'at_risk'],
            'engagement_metrics': ['frequency', 'recency', 'monetary'],
            'behavior_patterns': ['browser', 'buyer', 'returner', 'complainer']
        }
    
    def generate_rfm_features(self, df):
        """RFM 분석 기반 특성 생성"""
        # TODO: Recency, Frequency, Monetary 특성 생성
        # Recency: 최근 구매일
        # Frequency: 구매 빈도 
        # Monetary: 구매 금액
        pass
    
    def generate_engagement_features(self, df):
        """고객 참여도 기반 특성 생성"""
        # TODO: 웹사이트/앱 참여도 종합 점수
        # 구매 전환율, 세션당 페이지뷰 등
        pass
    
    def generate_lifecycle_features(self, df):
        """고객 생애주기 기반 특성 생성"""
        # TODO: 신규/성장/성숙/쇠퇴 단계 분류
        # 고객 나이, 이탈 위험도 등
        pass
    
    def generate_behavioral_features(self, df):
        """행동 패턴 기반 특성 생성"""
        # TODO: 쇼핑 패턴, 선호도, 충성도 지표
        pass
    
    def evaluate_feature_business_value(self, df, new_features, target_col):
        """생성된 특성의 비즈니스 가치 평가"""
        # TODO: 각 특성이 CLV 예측에 미치는 영향 분석
        # 특성별 중요도와 비즈니스 해석 제공
        pass

# TODO: EcommerceFeatureGenerator를 완성하고 실행해보세요
# 각 특성의 비즈니스적 의미와 활용 방안을 설명하세요
```

**평가 기준:**
- 도메인 지식의 적절한 활용 (30%)
- 특성 생성의 창의성과 유용성 (40%) 
- 비즈니스 가치 해석의 정확성 (30%)

---

### ⭐⭐⭐⭐ 연습문제 3: 자동화 함정 탐지 시스템
**목표:** 자동화 전처리에서 발생할 수 있는 함정들을 체계적으로 탐지하고 해결하는 전문가 시스템 구축

**과제:**
다양한 데이터셋에서 자동화 전처리의 4가지 주요 함정을 탐지하고 경고하는 시스템을 구현하세요.

```python
class AutomationPitfallDetector:
    """자동화 전처리 함정 탐지 및 경고 시스템"""
    
    def __init__(self, sensitivity='medium'):
        self.sensitivity = sensitivity  # 'low', 'medium', 'high'
        self.detection_history = []
        self.warning_thresholds = self._set_thresholds()
    
    def _set_thresholds(self):
        """민감도별 경고 임계값 설정"""
        thresholds = {
            'low': {'context_change': 0.3, 'overfitting_ratio': 0.2, 'bias_difference': 0.15},
            'medium': {'context_change': 0.2, 'overfitting_ratio': 0.15, 'bias_difference': 0.1}, 
            'high': {'context_change': 0.1, 'overfitting_ratio': 0.1, 'bias_difference': 0.05}
        }
        return thresholds[self.sensitivity]
    
    def detect_context_ignorance(self, original_df, processed_df, domain_rules=None):
        """도메인 맥락 무시 함정 탐지"""
        warnings = []
        
        # TODO: 다음 사항들을 체크하여 함정 탐지
        # 1. 결측치 패턴의 의미 무시
        # 2. 비즈니스 규칙 위반
        # 3. 도메인 상식에 어긋나는 변환
        # 4. 중요한 맥락 정보 손실
        
        return warnings
    
    def detect_overfitting_artifacts(self, df, target_col, generated_features):
        """과적합 및 허위 패턴 함정 탐지"""
        warnings = []
        
        # TODO: 다음 사항들을 체크하여 함정 탐지
        # 1. 특성 수 vs 샘플 수 비율
        # 2. 교차 검증 성능 저하
        # 3. 특성 간 높은 상관관계 (다중공선성)
        # 4. 의미 없는 수학적 조합들
        
        return warnings
    
    def detect_automation_bias(self, df, protected_attributes=None):
        """자동화 편향 함정 탐지"""
        warnings = []
        
        # TODO: 다음 사항들을 체크하여 함정 탐지
        # 1. 보호 속성에 따른 차별적 처리
        # 2. 과거 편향 데이터의 패턴 학습
        # 3. 대표성 부족한 그룹의 소외
        # 4. 공정성 지표 위반
        
        return warnings
    
    def detect_black_box_problems(self, preprocessing_pipeline):
        """블랙박스 문제 함정 탐지"""
        warnings = []
        
        # TODO: 다음 사항들을 체크하여 함정 탐지
        # 1. 설명 불가능한 변환 과정
        # 2. 재현성 부족
        # 3. 디버깅 어려움
        # 4. 규제 요구사항 미충족
        
        return warnings
    
    def generate_comprehensive_report(self, detection_results):
        """종합 함정 탐지 보고서 생성"""
        # TODO: 탐지된 모든 함정들을 우선순위별로 정리
        # 각 함정별 해결 방안과 예방책 제시
        pass
    
    def recommend_safety_measures(self, warnings):
        """안전 조치 권고사항 생성"""
        # TODO: 탐지된 함정에 따른 구체적 해결책 제시
        pass

# TODO: 다음 3가지 데이터셋에 대해 함정 탐지 시스템을 적용하세요
test_datasets = {
    'medical_diagnosis': '의료 진단 데이터 (편향 위험 높음)',
    'financial_credit': '신용 평가 데이터 (공정성 중요)',
    'marketing_campaign': '마케팅 캠페인 데이터 (과적합 위험)'
}

# 각 데이터셋별로 어떤 함정이 탐지되는지 분석하고
# 적절한 해결 방안을 제시하세요
```

**도전 과제:**
- 실제 데이터셋에서 함정 사례를 찾아 분석
- 함정별 자동 복구 메커니즘 설계
- 함정 방지를 위한 예방적 가이드라인 수립

---

### ⭐⭐⭐⭐⭐ 연습문제 4: 완전한 하이브리드 전처리 시스템
**목표:** 실제 프로덕션 환경에서 사용 가능한 완전한 하이브리드 전처리 시스템 설계 및 구현

**과제:**
금융 기관의 대출 심사 시스템을 위한 하이브리드 전처리 파이프라인을 설계하고 구현하세요. 다음 요구사항을 모두 만족해야 합니다:

**요구사항:**
1. **규제 준수**: 금융 규제 요구사항 만족
2. **공정성**: 성별, 인종 등에 따른 차별 방지  
3. **투명성**: 모든 처리 과정 추적 가능
4. **성능**: 기존 시스템 대비 20% 이상 성능 향상
5. **확장성**: 일일 10만 건 처리 가능
6. **안정성**: 99.9% 가용성 보장

```python
class ProductionHybridPreprocessor:
    """프로덕션급 하이브리드 전처리 시스템"""
    
    def __init__(self, config):
        self.config = config
        self.audit_log = []
        self.performance_metrics = {}
        self.regulatory_compliance = RegulatorySuite()
        self.fairness_monitor = FairnessMonitor()
        self.quality_control = QualityController()
    
    def process_loan_application(self, application_data):
        """대출 신청 데이터 전처리"""
        # TODO: 완전한 하이브리드 파이프라인 구현
        pass
    
    def ensure_regulatory_compliance(self, data, processing_steps):
        """규제 준수 확인"""
        # TODO: GDPR, 공정신용보고법 등 준수 검증
        pass
    
    def monitor_fairness(self, data, protected_attributes):
        """공정성 모니터링"""
        # TODO: 그룹별 처리 결과 공정성 검증
        pass
    
    def audit_processing_pipeline(self):
        """처리 과정 감사"""
        # TODO: 전체 파이프라인 추적 가능성 보장
        pass
    
    def performance_benchmark(self, test_data):
        """성능 벤치마킹"""
        # TODO: 기존 시스템 대비 성능 측정
        pass
    
    def stress_test(self, load_level):
        """부하 테스트"""
        # TODO: 대용량 처리 성능 검증
        pass

# TODO: 완전한 시스템을 구현하고 다음을 제출하세요:
# 1. 시스템 아키텍처 다이어그램
# 2. 핵심 코드 구현
# 3. 테스트 케이스 및 결과
# 4. 성능 벤치마크 보고서
# 5. 규제 준수 체크리스트
# 6. 운영 매뉴얼
```

**평가 기준:**
- 시스템 완성도 (25%)
- 규제 준수 수준 (20%)
- 성능 개선 정도 (20%)
- 코드 품질 및 문서화 (15%)
- 혁신성 및 창의성 (10%)
- 실무 적용 가능성 (10%)

---

## 🤔 생각해보기

### 💭 심화 질문들

1. **AI 자동화의 미래**  
   "10년 후 데이터 전처리는 완전히 자동화될 것이라고 생각하시나요? 그렇다면 데이터 분석가의 역할은 어떻게 변화할까요?"

2. **윤리적 딜레마**  
   "AI 전처리 시스템이 높은 성능을 보이지만 일부 그룹에게 불공정한 결과를 낳는다면, 성능과 공정성 중 무엇을 우선해야 할까요?"

3. **도메인 지식의 가치**  
   "AI가 점점 발전하면서 도메인 전문가의 지식이 불필요해질까요? 아니면 오히려 더 중요해질까요?"

4. **신뢰와 검증**  
   "어느 정도의 검증이면 AI 전처리 결과를 신뢰할 수 있다고 생각하시나요? 100% 검증은 가능할까요?"

5. **규제와 혁신**  
   "AI 전처리에 대한 강한 규제가 혁신을 저해할 수 있다고 생각하시나요? 적절한 균형점은 어디일까요?"

### 🔮 다음 Part 예고: "실제 데이터 전처리 파이프라인 구축"

다음 **Part 5**에서는 지금까지 배운 모든 내용을 종합하여 **실제 프로덕션 환경에서 사용할 수 있는 완전한 전처리 파이프라인**을 구축해보겠습니다.

**🎯 Part 5 주요 내용:**
- **확장 가능한 전처리 아키텍처 설계**
- **CI/CD를 활용한 자동화된 파이프라인 배포**  
- **실시간 모니터링과 품질 관리 시스템**
- **대용량 데이터 처리를 위한 분산 시스템**
- **House Prices 데이터를 활용한 완전한 MLOps 파이프라인**

**💡 미리 준비할 것:**
- Part 1-4에서 학습한 모든 전처리 기법 복습
- Docker와 클라우드 서비스에 대한 기초 지식
- 프로덕션 환경에서의 데이터 처리 경험 상기

> **"Part 4에서 배운 하이브리드 접근법이 Part 5에서 실제 시스템으로 구현됩니다. AI의 효율성과 인간의 지혜를 결합한 차세대 데이터 전처리 시스템을 함께 만들어보세요!"**

---

**🎉 Part 4 완료!** 

축하합니다! 이제 여러분은 **AI 자동 전처리의 명암을 모두 이해하고, 그 힘을 안전하고 효과적으로 활용할 수 있는 전문가**가 되었습니다. 

다음 Part에서는 이 모든 지식을 실제 시스템으로 구현하여 **진정한 데이터 전처리 마스터**의 경지에 도달해보겠습니다! 🚀

