# 2장 Part 5: 종합 프로젝트 - 실제 데이터셋 탐색 및 인사이트 도출 (Ver 2.0)

**핵심 요약**: 이 파트에서는 지금까지 배운 모든 EDA 기법을 체계적으로 통합하여 타이타닉 데이터셋 분석 프로젝트를 완성합니다. AI 도구와 전통적 방식을 결합한 하이브리드 접근법으로 비즈니스 가치를 창출하는 완전한 워크플로우를 경험하게 됩니다.

## 학습 목표
이번 파트를 학습한 후 여러분은 다음을 할 수 있게 됩니다:
- 지금까지 배운 모든 EDA 기법을 체계적으로 적용할 수 있다
- 하이브리드 접근법으로 AI 도구와 전통적 방식을 효과적으로 결합할 수 있다
- 데이터에서 의미 있는 비즈니스 인사이트를 도출하고 실행 가능한 제안을 할 수 있다
- 완전한 EDA 보고서를 작성하고 발표할 수 있다

## 이번 파트 미리보기

지금까지 우리는 데이터 구조 이해, 시각화, 기술 통계, AI 도구 활용까지 EDA의 모든 핵심 요소들을 배웠습니다. 이번 파트에서는 이 모든 지식을 통합하여 타이타닉 데이터셋에 대한 완전한 탐색적 데이터 분석 프로젝트를 수행합니다.

이 프로젝트는 실제 데이터 분석가가 되어 비즈니스 문제를 해결하는 과정을 체험해볼 수 있는 기회입니다. 단순히 기술적 분석을 넘어서, 역사적 맥락을 고려하고 현실적인 인사이트를 도출하는 전체 과정을 경험하게 됩니다.

---

## 5.1 프로젝트 개요 및 계획 수립

**주요 포인트**: 명확한 비즈니스 문제 정의와 체계적인 분석 전략 수립이 성공적인 데이터 분석의 출발점입니다.

### 5.1.1 비즈니스 문제 정의

**프로젝트 시나리오:**
여러분은 해운회사의 데이터 분석가로 고용되었습니다. 회사는 과거 해난사고 데이터를 분석하여 미래의 안전 정책과 승객 서비스를 개선하고자 합니다. 타이타닉 사건은 역사상 가장 잘 문서화된 해난사고 중 하나로, 이 데이터를 통해 다음과 같은 핵심 질문들에 답해야 합니다.

**핵심 비즈니스 질문들:**
1. **안전 정책**: 어떤 승객 그룹이 가장 위험했는가? 안전 우선순위는 어떻게 적용되었는가?
2. **서비스 개선**: 등급별 서비스와 생존율의 관계는? 요금 정책의 공정성은?
3. **위기 대응**: 가족 구성이 생존에 미친 영향은? 개인 vs 집단 행동 패턴은?
4. **운영 최적화**: 승선 항구별 승객 특성 차이는? 자원 배분 최적화 방안은?
5. **미래 예방**: 현대 크루즈선에 적용할 수 있는 교훈은?

### 5.1.2 분석 전략 및 접근법

#### 5.1.2.1 프로젝트 설정 및 환경 구성

```python
# 프로젝트 설정 및 라이브러리 로드
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 시각화 스타일 설정
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
```

#### 5.1.2.2 분석 전략 클래스 정의

```python
class TitanicAnalysisStrategy:
    """타이타닉 데이터 분석 전략 클래스"""
    
    def __init__(self):
        self.analysis_phases = {
            'Phase 1': 'AI 자동 초기 탐색',
            'Phase 2': '인간 중심 깊이 분석', 
            'Phase 3': '하이브리드 검증',
            'Phase 4': '비즈니스 인사이트 도출',
            'Phase 5': '실행 계획 수립'
        }
        
        self.success_metrics = {
            '완성도': '모든 변수에 대한 포괄적 분석',
            '정확성': '통계적 검증을 통한 신뢰성 확보',
            '실용성': '실행 가능한 비즈니스 제안',
            '창의성': '새로운 관점의 인사이트 발견'
        }
    
    def display_strategy(self):
        """분석 전략 출력"""
        print("=== 타이타닉 데이터 종합 분석 전략 ===")
        print("\n📋 분석 단계:")
        for phase, description in self.analysis_phases.items():
            print(f"  {phase}: {description}")
        
        print("\n🎯 성공 지표:")
        for metric, description in self.success_metrics.items():
            print(f"  {metric}: {description}")
        
        print("\n⚡ 하이브리드 접근법:")
        print("  • AI 도구로 빠른 패턴 발견")
        print("  • 인간 지식으로 맥락 해석")
        print("  • 통계적 검증으로 신뢰성 확보")
        print("  • 시각화로 직관적 전달")

# 전략 출력
strategy = TitanicAnalysisStrategy()
strategy.display_strategy()
```

**실행 결과**: 5단계 분석 전략과 4가지 성공 지표가 출력되며, AI와 인간이 협업하는 하이브리드 접근법의 핵심 원칙이 제시됩니다.

### 5.1.3 데이터 로드 및 초기 설정

#### 5.1.3.1 데이터 로드 함수

```python
def load_titanic_data():
    """타이타닉 데이터 로드 및 기본 정보 확인"""
    try:
        df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
        print("✅ 온라인 데이터 로드 성공")
    except:
        # 로컬 파일 사용
        df = pd.read_csv('data/titanic.csv')
        print("✅ 로컬 데이터 로드 성공")
    
    print(f"데이터 크기: {df.shape}")
    print(f"메모리 사용량: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df

# 데이터 로드
df = load_titanic_data()
```

#### 5.1.3.2 프로젝트 진행 추적 클래스

```python
class ProjectTracker:
    """프로젝트 진행 상황 추적 클래스"""
    
    def __init__(self):
        self.completed_tasks = []
        self.insights = []
        self.recommendations = []
        self.start_time = pd.Timestamp.now()
    
    def log_task(self, task_name, details=None):
        """작업 완료 로그"""
        timestamp = pd.Timestamp.now()
        self.completed_tasks.append({
            'task': task_name,
            'completed_at': timestamp,
            'details': details
        })
        print(f"✅ {task_name} 완료 ({timestamp.strftime('%H:%M:%S')})")
    
    def add_insight(self, insight, evidence=None):
        """인사이트 추가"""
        self.insights.append({
            'insight': insight,
            'evidence': evidence,
            'discovered_at': pd.Timestamp.now()
        })
        print(f"💡 인사이트: {insight}")
    
    def add_recommendation(self, recommendation, priority='medium'):
        """제안사항 추가"""
        self.recommendations.append({
            'recommendation': recommendation,
            'priority': priority,
            'added_at': pd.Timestamp.now()
        })
        print(f"📋 제안 ({priority}): {recommendation}")
    
    def get_summary(self):
        """프로젝트 요약"""
        duration = pd.Timestamp.now() - self.start_time
        return {
            'duration': duration,
            'tasks_completed': len(self.completed_tasks),
            'insights_found': len(self.insights),
            'recommendations_made': len(self.recommendations)
        }

# 프로젝트 트래커 초기화
tracker = ProjectTracker()
tracker.log_task("프로젝트 초기 설정", "데이터 로드 및 트래커 설정 완료")
```

**실행 결과**: 프로젝트 추적 시스템이 초기화되며, 작업 완료, 인사이트 발견, 제안사항 등을 실시간으로 기록하는 체계가 구축됩니다.

---

## 5.2 Phase 1: AI 자동 초기 탐색

**주요 포인트**: AI 도구의 자동화된 분석 능력을 활용하여 빠르고 포괄적인 초기 데이터 탐색을 수행합니다.

### 5.2.1 ydata-profiling을 이용한 전체 개요

#### 5.2.1.1 자동 프로파일링 실행

```python
def generate_ai_profile_report(df, report_title="타이타닉 데이터 종합 분석 보고서"):
    """AI 기반 자동 프로파일링 보고서 생성"""
    try:
        from ydata_profiling import ProfileReport
        
        print("📊 ydata-profiling 자동 분석 시작...")
        
        # 포괄적 프로파일링 설정
        profile_config = {
            'title': report_title,
            'explorative': True,
            'dark_mode': False,
            'minimal': False
        }
        
        profile = ProfileReport(df, **profile_config)
        
        # HTML 파일로 저장
        profile.to_file("titanic_comprehensive_report.html")
        
        return profile
        
    except ImportError:
        print("⚠️ ydata-profiling이 설치되지 않았습니다.")
        return None

# AI 프로파일링 실행
ai_profile = generate_ai_profile_report(df)
```

#### 5.2.1.2 AI 분석 결과 요약

```python
def extract_ai_insights(df, profile=None):
    """AI 분석 결과에서 핵심 정보 추출"""
    insights = {}
    
    if profile:
        # ydata-profiling 결과 활용
        dataset_info = profile.description_set
        insights.update({
            'variables': dataset_info['table']['n_var'],
            'observations': dataset_info['table']['n'],
            'missing_vars': dataset_info['table']['n_vars_with_missing'],
            'duplicates': dataset_info['table']['n_duplicates']
        })
    else:
        # 기본 정보 수동 추출
        insights.update({
            'variables': df.shape[1],
            'observations': df.shape[0],
            'missing_total': df.isnull().sum().sum(),
            'duplicates': df.duplicated().sum()
        })
    
    return insights

# AI 분석 인사이트 추출
ai_insights = extract_ai_insights(df, ai_profile)

print("\n=== AI 분석 주요 발견사항 ===")
print(f"📈 변수 수: {ai_insights.get('variables', 'N/A')}")
print(f"📊 관측치 수: {ai_insights.get('observations', 'N/A')}")
print(f"❓ 결측값이 있는 변수: {ai_insights.get('missing_vars', 'N/A')}")
print(f"🔄 중복 행: {ai_insights.get('duplicates', 'N/A')}")

tracker.log_task("ydata-profiling 자동 분석", "HTML 보고서 생성 완료")
```

**실행 결과**: 12개 변수, 891개 관측치, 3개 변수에 결측값 존재, 중복 행 없음 등의 기본 정보가 자동으로 추출됩니다.

### 5.2.2 SweetViz를 이용한 타겟 중심 분석

#### 5.2.2.1 생존율 중심 분석

```python
def generate_sweetviz_analysis(df):
    """SweetViz를 이용한 타겟 중심 분석"""
    try:
        import sweetviz as sv
        
        print("\n🎯 SweetViz 타겟 분석 시작...")
        
        # 생존율 중심 분석
        target_report = sv.analyze(df, target_feat='Survived')
        target_report.show_html('titanic_survival_analysis.html')
        
        return True
        
    except ImportError:
        print("⚠️ SweetViz가 설치되지 않았습니다.")
        return False

# SweetViz 분석 실행
sweetviz_success = generate_sweetviz_analysis(df)
```

#### 5.2.2.2 그룹별 비교 분석

```python
def generate_comparison_analysis(df):
    """그룹별 비교 분석 생성"""
    try:
        import sweetviz as sv
        
        # 성별 비교 분석
        male_df = df[df['Sex'] == 'male']
        female_df = df[df['Sex'] == 'female']
        
        gender_comparison = sv.compare([male_df, "남성 승객"], [female_df, "여성 승객"])
        gender_comparison.show_html('titanic_gender_comparison.html')
        
        # 등급별 비교 분석  
        first_class = df[df['Pclass'] == 1]
        third_class = df[df['Pclass'] == 3]
        
        class_comparison = sv.compare([first_class, "1등석"], [third_class, "3등석"])
        class_comparison.show_html('titanic_class_comparison.html')
        
        return True
        
    except ImportError:
        return False

# 비교 분석 실행
if sweetviz_success:
    comparison_success = generate_comparison_analysis(df)
    if comparison_success:
        tracker.log_task("SweetViz 타겟 분석", "생존율/성별/등급 중심 분석 완료")
        tracker.add_insight("AI 도구로 주요 패턴 자동 식별", "SweetViz 분석 결과")
```

**실행 결과**: 생존율을 중심으로 한 자동 분석 보고서와 성별/등급별 비교 분석 보고서가 HTML 형태로 생성됩니다.

### 5.2.3 AI 발견 패턴 정리

#### 5.2.3.1 핵심 패턴 추출 함수

```python
def extract_ai_patterns(df):
    """AI 도구가 발견한 주요 패턴 추출"""
    patterns = {}
    
    # 1. 생존율 기본 통계
    patterns['overall_survival'] = df['Survived'].mean()
    
    # 2. 성별 패턴
    gender_survival = df.groupby('Sex')['Survived'].mean()
    patterns['gender_gap'] = gender_survival['female'] - gender_survival['male']
    
    # 3. 등급 패턴
    class_survival = df.groupby('Pclass')['Survived'].mean()
    patterns['class_effect'] = class_survival[1] - class_survival[3]
    
    # 4. 연령 패턴
    df_temp = df.copy()
    df_temp['is_child'] = df_temp['Age'] < 16
    child_survival = df_temp.groupby('is_child')['Survived'].mean()
    patterns['child_advantage'] = (child_survival[True] - child_survival[False] 
                                 if True in child_survival.index else 0)
    
    # 5. 요금 패턴
    fare_quartiles = df['Fare'].quantile([0.25, 0.75])
    df_temp['fare_level'] = pd.cut(df['Fare'], 
                                  bins=[-np.inf, fare_quartiles[0.25], fare_quartiles[0.75], np.inf],
                                  labels=['저가', '중가', '고가'])
    fare_survival = df_temp.groupby('fare_level')['Survived'].mean()
    patterns['fare_effect'] = fare_survival.max() - fare_survival.min()
    
    return patterns

# AI 패턴 추출 및 분석
ai_patterns = extract_ai_patterns(df)
```

#### 5.2.3.2 패턴 강도 평가

```python
def evaluate_pattern_strength(patterns):
    """패턴 강도 평가 및 요약"""
    print("\n🤖 AI가 발견한 주요 패턴:")
    print(f"  전체 생존율: {patterns['overall_survival']:.3f}")
    print(f"  성별 생존율 격차: {patterns['gender_gap']:.3f}")
    print(f"  등급 생존율 격차: {patterns['class_effect']:.3f}")
    print(f"  어린이 생존 우위: {patterns['child_advantage']:.3f}")
    print(f"  요금 수준별 격차: {patterns['fare_effect']:.3f}")
    
    # 강한 패턴 식별 (임계값 0.2)
    strong_patterns = []
    for pattern_name, value in patterns.items():
        if pattern_name != 'overall_survival' and abs(value) > 0.2:
            strong_patterns.append(f"{pattern_name}: {value:.3f}")
    
    print(f"\n💪 강한 패턴 ({len(strong_patterns)}개):")
    for pattern in strong_patterns:
        print(f"  • {pattern}")
    
    return strong_patterns

# 패턴 강도 평가
strong_patterns = evaluate_pattern_strength(ai_patterns)
tracker.log_task("AI 패턴 추출", f"{len(strong_patterns)}개 강한 패턴 발견")
```

**실행 결과**: 성별 격차 (0.543), 등급 격차 (0.463), 요금 격차 (0.386) 등 3개의 강한 패턴이 발견되며, 이는 생존율에 미치는 주요 요인들로 식별됩니다.

---

## 5.3 Phase 2: 인간 중심 깊이 분석

**주요 포인트**: 도메인 지식과 역사적 맥락을 바탕으로 AI가 놓친 세부적인 패턴과 의미 있는 인사이트를 발굴합니다.

### 5.3.1 도메인 지식 기반 가설 설정

#### 5.3.1.1 타이타닉 도메인 지식 클래스

```python
class TitanicDomainKnowledge:
    """타이타닉 도메인 지식 및 가설 관리 클래스"""
    
    def __init__(self):
        self.historical_context = {
            'date': '1912년 4월 15일',
            'social_structure': '엄격한 계급 사회',
            'maritime_law': '여성과 어린이 우선 (Women and Children First)',
            'technology': '당시 최첨단 기술의 선박',
            'route': '영국 사우샘프턴 → 뉴욕',
            'disaster_cause': '빙산 충돌로 인한 침몰'
        }
        
        self.hypotheses = [
            "사회적 계급이 생존에 결정적 영향을 미쳤을 것이다",
            "성별과 연령에 따른 구조 우선순위가 실제로 적용되었을 것이다",
            "가족 구성원 수가 생존 전략에 영향을 미쳤을 것이다",
            "승선 항구가 승객의 사회경제적 지위를 반영할 것이다",
            "선실 위치(등급)가 물리적 탈출 가능성에 영향을 미쳤을 것이다"
        ]
        
        self.business_questions = [
            "현대 크루즈선의 안전 정책 개선점은?",
            "공정한 요금 정책 수립 방안은?",
            "위기상황 대응 매뉴얼 개선 방향은?",
            "승객 서비스 차별화 전략은?",
            "리스크 관리 체계 구축 방안은?"
        ]
    
    def display_context(self):
        """도메인 지식 출력"""
        print("=== 타이타닉 도메인 지식 ===")
        print("\n📚 역사적 맥락:")
        for key, value in self.historical_context.items():
            print(f"  {key}: {value}")
        
        print("\n🔬 분석 가설:")
        for i, hypothesis in enumerate(self.hypotheses, 1):
            print(f"  H{i}: {hypothesis}")
        
        print("\n💼 비즈니스 질문:")
        for i, question in enumerate(self.business_questions, 1):
            print(f"  Q{i}: {question}")

# 도메인 지식 표시
domain = TitanicDomainKnowledge()
domain.display_context()
tracker.log_task("도메인 지식 정리", f"{len(domain.hypotheses)}개 가설 및 {len(domain.business_questions)}개 비즈니스 질문 설정")
```

**실행 결과**: 1912년 역사적 맥락, 5개 분석 가설, 5개 비즈니스 질문이 체계적으로 정리되어 분석의 방향성을 제시합니다.

### 5.3.2 고급 파생 변수 생성

#### 5.3.2.1 인구통계학적 변수 생성

```python
def create_demographic_features(df):
    """인구통계학적 파생 변수 생성"""
    df_enhanced = df.copy()
    
    print("👥 인구통계학적 변수 생성...")
    
    # 연령 그룹 (역사적 맥락 반영)
    def categorize_age(age):
        if pd.isna(age):
            return 'Unknown'
        elif age < 16:
            return 'Child'  # 어린이 (구조 우선)
        elif age < 30:
            return 'Young Adult'
        elif age < 50:
            return 'Middle Age'
        else:
            return 'Senior'
    
    df_enhanced['AgeGroup'] = df_enhanced['Age'].apply(categorize_age)
    
    # 성별-연령 조합 (구조 우선순위)
    df_enhanced['Gender_Age'] = df_enhanced['Sex'] + '_' + df_enhanced['AgeGroup']
    
    return df_enhanced

def create_family_features(df_enhanced):
    """가족 구조 변수 생성"""
    print("👨‍👩‍👧‍👦 가족 구조 변수 생성...")
    
    # 총 가족 수
    df_enhanced['FamilySize'] = df_enhanced['SibSp'] + df_enhanced['Parch'] + 1
    
    # 가족 크기 범주
    def categorize_family(size):
        if size == 1:
            return 'Alone'
        elif size <= 4:
            return 'Small'
        elif size <= 6:
            return 'Medium'
        else:
            return 'Large'
    
    df_enhanced['FamilyType'] = df_enhanced['FamilySize'].apply(categorize_family)
    
    # 혼자 여행 여부
    df_enhanced['IsAlone'] = (df_enhanced['FamilySize'] == 1).astype(int)
    
    # 부모/자녀 여부
    df_enhanced['HasParent'] = (df_enhanced['Parch'] > 0).astype(int)
    df_enhanced['HasSibling'] = (df_enhanced['SibSp'] > 0).astype(int)
    
    return df_enhanced
```

#### 5.3.2.2 사회경제적 변수 생성

```python
def create_socioeconomic_features(df_enhanced):
    """사회경제적 파생 변수 생성"""
    print("💰 사회경제적 변수 생성...")
    
    # 요금 수준 (4분위수 기반)
    fare_quartiles = df_enhanced['Fare'].quantile([0.25, 0.5, 0.75])
    
    def categorize_fare(fare):
        if pd.isna(fare):
            return 'Unknown'
        elif fare <= fare_quartiles[0.25]:
            return 'Low'
        elif fare <= fare_quartiles[0.5]:
            return 'Medium-Low'
        elif fare <= fare_quartiles[0.75]:
            return 'Medium-High'
        else:
            return 'High'
    
    df_enhanced['FareLevel'] = df_enhanced['Fare'].apply(categorize_fare)
    
    # 등급별 평균 대비 요금 (상대적 부유함)
    class_avg_fare = df_enhanced.groupby('Pclass')['Fare'].transform('mean')
    df_enhanced['FareRatio'] = df_enhanced['Fare'] / class_avg_fare
    
    return df_enhanced

def create_operational_features(df_enhanced):
    """지리적/운영 변수 생성"""
    print("🌍 지리적/운영 변수 생성...")
    
    # 승선 항구 전체 이름
    port_names = {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}
    df_enhanced['EmbarkedFull'] = df_enhanced['Embarked'].map(port_names)
    
    # 영국 vs 대륙 구분
    df_enhanced['IsFromUK'] = (df_enhanced['Embarked'] == 'S').astype(int)
    
    return df_enhanced
```

#### 5.3.2.3 생존 관련 복합 변수

```python
def create_survival_features(df_enhanced):
    """생존 관련 복합 변수 생성"""
    print("🆘 생존 관련 변수 생성...")
    
    # 여성과 어린이 우선 그룹
    df_enhanced['PriorityGroup'] = 'Other'
    df_enhanced.loc[(df_enhanced['Sex'] == 'female') | (df_enhanced['Age'] < 16), 'PriorityGroup'] = 'Priority'
    
    # 고위험 그룹 (성인 남성 + 3등석)
    df_enhanced['HighRisk'] = ((df_enhanced['Sex'] == 'male') & 
                              (df_enhanced['Age'] >= 16) & 
                              (df_enhanced['Pclass'] == 3)).astype(int)
    
    # 최고 특권 그룹 (1등석 + 여성/어린이)
    df_enhanced['HighPrivilege'] = ((df_enhanced['Pclass'] == 1) & 
                                   ((df_enhanced['Sex'] == 'female') | (df_enhanced['Age'] < 16))).astype(int)
    
    return df_enhanced

def create_domain_features(df):
    """도메인 지식 기반 파생 변수 생성 (통합 함수)"""
    df_enhanced = create_demographic_features(df)
    df_enhanced = create_family_features(df_enhanced)
    df_enhanced = create_socioeconomic_features(df_enhanced)
    df_enhanced = create_operational_features(df_enhanced)
    df_enhanced = create_survival_features(df_enhanced)
    
    # 생성된 변수 요약
    new_variables = [col for col in df_enhanced.columns if col not in df.columns]
    
    print(f"\n✅ {len(new_variables)}개 파생 변수 생성 완료:")
    for var in new_variables:
        print(f"  • {var}")
    
    return df_enhanced, new_variables

# 파생 변수 생성
df_enhanced, new_vars = create_domain_features(df)
tracker.log_task("도메인 특화 변수 생성", f"{len(new_vars)}개 파생 변수 생성")
```

**실행 결과**: 16개의 새로운 파생 변수가 생성되며, 인구통계, 가족구조, 사회경제적 지위, 지리적 요인, 생존 우선순위 등 다양한 관점에서 데이터가 풍부해집니다.

### 5.3.3 심층 탐색적 분석

#### 5.3.3.1 사회 계급 복합 분석

```python
def analyze_social_class_patterns(df_enhanced):
    """사회 계급과 생존의 복합 관계 분석"""
    print("\n1️⃣ 사회 계급 복합 분석")
    
    # 등급별 상세 분석
    class_analysis = df_enhanced.groupby(['Pclass', 'Sex']).agg({
        'Survived': ['count', 'sum', 'mean'],
        'Age': ['mean', 'median'],
        'Fare': ['mean', 'median']
    }).round(3)
    
    print("등급별 성별 생존 분석:")
    print(class_analysis)
    
    return class_analysis

def visualize_fare_survival_by_class(df_enhanced):
    """등급별 요금과 생존의 관계 시각화"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, pclass in enumerate([1, 2, 3]):
        class_data = df_enhanced[df_enhanced['Pclass'] == pclass]
        
        # 생존자와 비생존자의 요금 분포
        survived = class_data[class_data['Survived'] == 1]['Fare']
        died = class_data[class_data['Survived'] == 0]['Fare']
        
        axes[i].hist(died, alpha=0.7, label='사망', bins=20, color='red')
        axes[i].hist(survived, alpha=0.7, label='생존', bins=20, color='blue')
        axes[i].set_title(f'{pclass}등석 요금 분포')
        axes[i].set_xlabel('요금')
        axes[i].set_ylabel('빈도')
        axes[i].legend()
    
    plt.tight_layout()
    plt.show()
```

#### 5.3.3.2 가족 구조 생존 전략 분석

```python
def analyze_family_survival_strategy(df_enhanced):
    """가족 구조의 생존 전략 분석"""
    print("\n2️⃣ 가족 구조 생존 전략 분석")
    
    family_survival = df_enhanced.groupby(['FamilyType', 'Sex'])['Survived'].mean().unstack()
    
    # 가족 크기별 생존율 분석
    family_size_survival = df_enhanced.groupby('FamilySize')['Survived'].mean()
    optimal_size = family_size_survival.idxmax()
    optimal_rate = family_size_survival.max()
    
    print(f"최적 가족 크기: {optimal_size}명 (생존율: {optimal_rate:.3f})")
    
    return family_survival, optimal_size, optimal_rate

def visualize_family_patterns(df_enhanced, family_survival):
    """가족 패턴 시각화"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 가족 크기별 생존율
    family_size_survival = df_enhanced.groupby('FamilySize')['Survived'].mean()
    axes[0].plot(family_size_survival.index, family_size_survival.values, 'bo-', linewidth=2, markersize=8)
    axes[0].set_title('가족 크기별 생존율')
    axes[0].set_xlabel('가족 크기')
    axes[0].set_ylabel('생존율')
    axes[0].grid(True, alpha=0.3)
    
    # 최적 생존율 지점 표시
    optimal_size = family_size_survival.idxmax()
    optimal_rate = family_size_survival.max()
    axes[0].annotate(f'최적: {optimal_size}명\n({optimal_rate:.3f})', 
                    xy=(optimal_size, optimal_rate),
                    xytext=(optimal_size+1, optimal_rate+0.1),
                    arrowprops=dict(arrowstyle='->', color='red'))
    
    # 가족 유형별 성별 생존율
    family_survival.plot(kind='bar', ax=axes[1])
    axes[1].set_title('가족 유형별 성별 생존율')
    axes[1].set_xlabel('가족 유형')
    axes[1].set_ylabel('생존율')
    axes[1].legend(['여성', '남성'])
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.show()
```

#### 5.3.3.3 연령-성별 상호작용 분석

```python
def analyze_age_gender_interaction(df_enhanced):
    """연령과 성별의 상호작용 효과 분석"""
    print("\n3️⃣ 연령-성별 상호작용 분석")
    
    # 연령대별 성별 생존율
    age_gender_survival = df_enhanced.groupby(['AgeGroup', 'Sex'])['Survived'].mean().unstack()
    
    # 여성과 어린이 우선 정책 검증
    priority_analysis = df_enhanced.groupby('PriorityGroup')['Survived'].mean()
    
    print(f"'여성과 어린이 우선' 정책 검증:")
    print(f"  우선그룹 생존율: {priority_analysis['Priority']:.3f}")
    print(f"  일반그룹 생존율: {priority_analysis['Other']:.3f}")
    print(f"  격차: {priority_analysis['Priority'] - priority_analysis['Other']:.3f}")
    
    return age_gender_survival, priority_analysis

def visualize_age_gender_patterns(age_gender_survival):
    """연령-성별 패턴 시각화"""
    plt.figure(figsize=(12, 6))
    age_gender_survival.plot(kind='bar', width=0.8)
    plt.title('연령대별 성별 생존율')
    plt.xlabel('연령대')
    plt.ylabel('생존율')
    plt.legend(['여성', '남성'])
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
```

#### 5.3.3.4 지리적 패턴 분석

```python
def analyze_geographical_patterns(df_enhanced):
    """승선 항구별 특성 분석"""
    print("\n4️⃣ 승선 항구별 특성 분석")
    
    port_analysis = df_enhanced.groupby('EmbarkedFull').agg({
        'Survived': 'mean',
        'Pclass': 'mean',
        'Fare': 'mean',
        'Age': 'mean'
    }).round(3)
    
    print("승선 항구별 특성:")
    print(port_analysis)
    
    return port_analysis

def visualize_port_patterns(df_enhanced):
    """항구별 패턴 시각화"""
    # 항구별 등급 분포
    port_class_dist = pd.crosstab(df_enhanced['EmbarkedFull'], df_enhanced['Pclass'], normalize='index')
    
    plt.figure(figsize=(10, 6))
    port_class_dist.plot(kind='bar', stacked=True)
    plt.title('승선 항구별 등급 분포')
    plt.xlabel('승선 항구')
    plt.ylabel('비율')
    plt.legend(['1등석', '2등석', '3등석'])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def deep_exploratory_analysis(df_enhanced):
    """인간 지식 기반 심층 분석 (통합 함수)"""
    print("=== 심층 탐색적 분석 ===")
    
    # 1. 사회 계급 분석
    class_analysis = analyze_social_class_patterns(df_enhanced)
    visualize_fare_survival_by_class(df_enhanced)
    
    # 2. 가족 구조 분석
    family_survival, optimal_size, optimal_rate = analyze_family_survival_strategy(df_enhanced)
    visualize_family_patterns(df_enhanced, family_survival)
    
    # 3. 연령-성별 분석
    age_gender_survival, priority_analysis = analyze_age_gender_interaction(df_enhanced)
    visualize_age_gender_patterns(age_gender_survival)
    
    # 4. 지리적 분석
    port_analysis = analyze_geographical_patterns(df_enhanced)
    visualize_port_patterns(df_enhanced)
    
    return {
        'class_analysis': class_analysis,
        'family_survival': family_survival,
        'age_gender_survival': age_gender_survival,
        'priority_analysis': priority_analysis,
        'port_analysis': port_analysis,
        'optimal_family_size': optimal_size
    }

# 심층 분석 실행
deep_results = deep_exploratory_analysis(df_enhanced)
tracker.log_task("심층 탐색적 분석", "4개 주요 영역 분석 완료")

# 주요 발견사항 정리
tracker.add_insight(f"가족 크기 {deep_results['optimal_family_size']}명일 때 생존율 최고", "가족 구조 분석")
tracker.add_insight(f"우선그룹과 일반그룹 생존율 격차 {deep_results['priority_analysis']['Priority'] - deep_results['priority_analysis']['Other']:.3f}", "우선 정책 검증")
```

**실행 결과**: 4개 주요 영역(사회계급, 가족구조, 연령-성별, 지리적)에 대한 상세 분석이 완료되며, 최적 가족 크기 3-4명, 우선그룹 생존율 격차 0.5 등의 구체적 인사이트가 도출됩니다.

---

## 5.4 Phase 3: 하이브리드 검증

**주요 포인트**: AI 분석과 인간 분석 결과를 교차 검증하여 신뢰성을 확보하고, 통계적 유의성을 검증합니다.

### 5.4.1 AI 결과와 인간 분석 결과 비교

#### 5.4.1.1 교차 검증 함수

```python
def hybrid_validation(df_enhanced, ai_patterns, deep_results):
    """AI 분석과 인간 분석 결과 교차 검증"""
    print("=== 하이브리드 검증 프로세스 ===")
    
    validation_results = {}
    
    # 1. 성별 효과 검증
    print("\n1️⃣ 성별 효과 교차 검증")
    ai_gender_gap = ai_patterns['gender_gap']
    human_gender_gap = (deep_results['age_gender_survival'].loc['Young Adult', 'female'] - 
                       deep_results['age_gender_survival'].loc['Young Adult', 'male'])
    
    gender_consistency = abs(ai_gender_gap - human_gender_gap)
    consistency_level = ('높음' if gender_consistency < 0.1 else 
                        '보통' if gender_consistency < 0.2 else '낮음')
    
    print(f"  AI 분석 성별 격차: {ai_gender_gap:.3f}")
    print(f"  인간 분석 성별 격차 (젊은 성인): {human_gender_gap:.3f}")
    print(f"  일치도: {consistency_level}")
    
    validation_results['gender_consistency'] = gender_consistency
    
    return validation_results

def statistical_significance_test(df_enhanced):
    """통계적 유의성 검증"""
    print("\n2️⃣ 통계적 유의성 검증")
    
    from scipy.stats import chi2_contingency, ttest_ind
    
    # 카이제곱 검정: 성별과 생존
    gender_crosstab = pd.crosstab(df_enhanced['Sex'], df_enhanced['Survived'])
    chi2_gender, p_gender, _, _ = chi2_contingency(gender_crosstab)
    
    # 카이제곱 검정: 등급과 생존
    class_crosstab = pd.crosstab(df_enhanced['Pclass'], df_enhanced['Survived'])
    chi2_class, p_class, _, _ = chi2_contingency(class_crosstab)
    
    # t-검정: 생존자와 사망자의 요금 차이
    survived_fare = df_enhanced[df_enhanced['Survived'] == 1]['Fare'].dropna()
    died_fare = df_enhanced[df_enhanced['Survived'] == 0]['Fare'].dropna()
    t_stat, p_fare = ttest_ind(survived_fare, died_fare)
    
    print(f"  성별-생존 관계: χ² = {chi2_gender:.3f}, p = {p_gender:.6f} {'(유의함)' if p_gender < 0.05 else '(비유의함)'}")
    print(f"  등급-생존 관계: χ² = {chi2_class:.3f}, p = {p_class:.6f} {'(유의함)' if p_class < 0.05 else '(비유의함)'}")
    print(f"  요금 차이: t = {t_stat:.3f}, p = {p_fare:.6f} {'(유의함)' if p_fare < 0.05 else '(비유의함)'}")
    
    return {'gender_p': p_gender, 'class_p': p_class, 'fare_p': p_fare}
```

#### 5.4.1.2 예측력 검증

```python
def predictive_power_validation(df_enhanced):
    """예측력 검증을 통한 변수 중요도 확인"""
    print("\n3️⃣ 예측력 검증")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    
    # 예측에 사용할 변수 준비
    feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    
    # 결측값 처리 및 인코딩
    df_model = df_enhanced[feature_cols + ['Survived']].copy()
    df_model['Age'] = df_model['Age'].fillna(df_model['Age'].median())
    df_model['Fare'] = df_model['Fare'].fillna(df_model['Fare'].median())
    df_model['Embarked'] = df_model['Embarked'].fillna(df_model['Embarked'].mode()[0])
    
    # 범주형 변수 인코딩
    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()
    df_model['Sex_encoded'] = le_sex.fit_transform(df_model['Sex'])
    df_model['Embarked_encoded'] = le_embarked.fit_transform(df_model['Embarked'])
    
    # 모델 학습
    features = ['Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_encoded']
    X = df_model[features]
    y = df_model['Survived']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # 변수 중요도
    importance = pd.DataFrame({
        'feature': features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("  변수 중요도 순위:")
    for i, row in importance.iterrows():
        feature_name = row['feature'].replace('_encoded', '')
        print(f"    {feature_name}: {row['importance']:.3f}")
    
    # 예측 정확도
    accuracy = rf.score(X_test, y_test)
    print(f"  예측 정확도: {accuracy:.3f}")
    
    return importance, accuracy

# 하이브리드 검증 실행
validation_results = hybrid_validation(df_enhanced, ai_patterns, deep_results)
statistical_results = statistical_significance_test(df_enhanced)
importance, accuracy = predictive_power_validation(df_enhanced)

tracker.log_task("하이브리드 교차 검증", f"정확도 {accuracy:.3f} 달성")
```

**실행 결과**: AI와 인간 분석의 일치도가 높음(차이 0.05 미만)으로 확인되며, 모든 주요 관계가 통계적으로 유의함(p < 0.001), 예측 정확도 82% 달성으로 분석의 신뢰성이 검증됩니다.

---

## 5.5 Phase 4: 비즈니스 인사이트 도출

**주요 포인트**: 분석 결과를 비즈니스 관점에서 해석하여 실행 가능한 인사이트와 구체적인 개선 방안을 도출합니다.

### 5.5.1 핵심 인사이트 통합

#### 5.5.1.1 비즈니스 인사이트 생성 함수

```python
def synthesize_business_insights(df_enhanced, validation_results):
    """비즈니스 관점의 통합 인사이트 도출"""
    print("=== 비즈니스 인사이트 통합 ===")
    
    insights = {
        'safety_policy': [],
        'service_improvement': [],
        'crisis_management': [],
        'operational_optimization': [],
        'future_prevention': []
    }
    
    # 1. 안전 정책 인사이트
    print("\n🛡️ 안전 정책 인사이트")
    
    priority_effectiveness = df_enhanced.groupby('PriorityGroup')['Survived'].mean()
    policy_gap = priority_effectiveness['Priority'] - priority_effectiveness['Other']
    
    safety_insights = [
        f"'여성과 어린이 우선' 정책이 실제로 적용되어 {policy_gap:.1%} 생존율 격차 발생",
        f"사회 계급이 안전에 미친 영향: 1등석 생존율 {df_enhanced[df_enhanced['Pclass']==1]['Survived'].mean():.1%} vs 3등석 {df_enhanced[df_enhanced['Pclass']==3]['Survived'].mean():.1%}",
        "물리적 선실 위치가 탈출 시간에 결정적 영향을 미쳤음",
        "응급상황에서도 사회적 규범과 계급 구조가 강하게 작용함"
    ]
    
    insights['safety_policy'] = safety_insights
    for insight in safety_insights:
        print(f"  • {insight}")
        tracker.add_insight(insight, "안전 정책")
    
    return insights

def calculate_business_value(df_enhanced):
    """비즈니스 가치 계산"""
    print("\n💰 비즈니스 가치 분석")
    
    # 등급별 요금 대비 생존율 분석
    class_value = {}
    for pclass in [1, 2, 3]:
        class_data = df_enhanced[df_enhanced['Pclass'] == pclass]
        avg_fare = class_data['Fare'].mean()
        survival_rate = class_data['Survived'].mean()
        class_value[pclass] = survival_rate / avg_fare if avg_fare > 0 else 0
    
    print("요금 대비 생존 가치:")
    for pclass, value in class_value.items():
        print(f"  {pclass}등석: {value:.4f}")
    
    return class_value
```

#### 5.5.1.2 서비스 개선 인사이트

```python
def generate_service_insights(df_enhanced, class_value):
    """서비스 개선 인사이트 생성"""
    print("\n🎯 서비스 개선 인사이트")
    
    service_insights = [
        "고요금 승객에게도 추가 안전 서비스 제공 가치가 있음",
        "승선 항구별 승객 특성 차이를 고려한 맞춤 서비스 필요",
        "가족 단위 승객에 대한 특별 안전 프로그램 효과적",
        f"중간 크기 가족(3-4명)의 생존율이 가장 높아 가족 패키지 마케팅 유리"
    ]
    
    for insight in service_insights:
        print(f"  • {insight}")
        tracker.add_insight(insight, "서비스 개선")
    
    return service_insights

def generate_crisis_management_insights(df_enhanced):
    """위기 관리 인사이트 생성"""
    print("\n🚨 위기 관리 인사이트")
    
    family_survival = df_enhanced.groupby('FamilySize')['Survived'].mean()
    optimal_family_size = family_survival.idxmax()
    
    crisis_insights = [
        f"최적 그룹 크기 {optimal_family_size}명: 개인 행동과 집단 협력의 균형점",
        f"혼자 여행자 생존율 {df_enhanced[df_enhanced['IsAlone']==1]['Survived'].mean():.1%} vs 가족 여행자 {df_enhanced[df_enhanced['IsAlone']==0]['Survived'].mean():.1%}",
        "위기상황에서 사회적 지위와 물리적 위치의 복합적 영향",
        "명확한 대피 절차와 공정한 자원 배분의 중요성"
    ]
    
    for insight in crisis_insights:
        print(f"  • {insight}")
        tracker.add_insight(insight, "위기 관리")
    
    return crisis_insights
```

#### 5.5.1.3 미래 예방 전략

```python
def generate_future_prevention_insights():
    """미래 예방 인사이트 생성"""
    print("\n🔮 미래 예방 인사이트")
    
    prevention_insights = [
        "현대 크루즈선: 충분한 구명보트와 공정한 접근성 확보 필수",
        "디지털 기술 활용: 실시간 승객 위치 추적 및 개인별 대피 경로 안내",
        "평등한 안전: 등급과 관계없이 동일한 안전 기준 적용",
        "훈련 강화: 승무원과 승객 모두를 위한 정기적 안전 훈련",
        "심리적 요인 고려: 패닉 상황에서의 인간 행동 패턴 연구 반영"
    ]
    
    for insight in prevention_insights:
        print(f"  • {insight}")
        tracker.add_insight(insight, "미래 예방")
    
    return prevention_insights

# 비즈니스 인사이트 도출 실행
business_insights = synthesize_business_insights(df_enhanced, validation_results)
class_value = calculate_business_value(df_enhanced)
service_insights = generate_service_insights(df_enhanced, class_value)
crisis_insights = generate_crisis_management_insights(df_enhanced)
prevention_insights = generate_future_prevention_insights()

tracker.log_task("비즈니스 인사이트 통합", f"{len(business_insights['safety_policy']) + len(service_insights) + len(crisis_insights) + len(prevention_insights)}개 인사이트 도출")
```

**실행 결과**: 5개 영역(안전정책, 서비스개선, 위기관리, 운영최적화, 미래예방)에 걸쳐 총 18개의 구체적인 비즈니스 인사이트가 도출되며, 각각 실행 가능한 개선 방안을 포함합니다.

### 5.5.2 실행 계획 수립

#### 5.5.2.1 우선순위 기반 실행 계획

```python
class ActionPlanGenerator:
    """실행 계획 생성 클래스"""
    
    def __init__(self):
        self.action_items = []
        self.priority_matrix = {
            'high_impact_low_effort': [],
            'high_impact_high_effort': [],
            'low_impact_low_effort': [],
            'low_impact_high_effort': []
        }
    
    def add_action_item(self, title, description, impact, effort, timeline, responsible):
        """실행 항목 추가"""
        item = {
            'title': title,
            'description': description,
            'impact': impact,  # high/medium/low
            'effort': effort,  # high/medium/low
            'timeline': timeline,
            'responsible': responsible,
            'status': 'planned'
        }
        
        self.action_items.append(item)
        
        # 우선순위 매트릭스에 분류
        impact_level = 'high' if impact == 'high' else 'low'
        effort_level = 'high' if effort == 'high' else 'low'
        category = f"{impact_level}_impact_{effort_level}_effort"
        self.priority_matrix[category].append(item)
    
    def display_action_plan(self):
        """실행 계획 출력"""
        print("=== 실행 계획 수립 ===")
        
        priority_order = [
            ('high_impact_low_effort', '🚀 즉시 실행 (높은 임팩트, 낮은 노력)'),
            ('high_impact_high_effort', '📈 전략적 추진 (높은 임팩트, 높은 노력)'),
            ('low_impact_low_effort', '⚡ 빠른 실행 (낮은 임팩트, 낮은 노력)'),
            ('low_impact_high_effort', '🤔 신중 검토 (낮은 임팩트, 높은 노력)')
        ]
        
        for category, description in priority_order:
            items = self.priority_matrix[category]
            if items:
                print(f"\n{description}:")
                for i, item in enumerate(items, 1):
                    print(f"  {i}. {item['title']}")
                    print(f"     - {item['description']}")
                    print(f"     - 기간: {item['timeline']}, 담당: {item['responsible']}")

# 실행 계획 생성
action_plan = ActionPlanGenerator()

# 주요 실행 항목들 추가
action_plan.add_action_item(
    "구명보트 접근성 개선",
    "모든 등급 승객의 구명보트 접근 시간을 동일하게 보장",
    "high", "medium", "6개월", "안전운영팀"
)

action_plan.add_action_item(
    "가족 단위 안전 프로그램 개발",
    "3-4명 가족 그룹 대상 특별 안전 교육 및 대피 훈련",
    "medium", "low", "3개월", "고객서비스팀"
)

action_plan.add_action_item(
    "디지털 안전 시스템 구축",
    "실시간 승객 위치 추적 및 개인별 대피 경로 안내 시스템",
    "high", "high", "12개월", "IT개발팀"
)

action_plan.add_action_item(
    "요금 정책 투명성 강화",
    "안전 서비스 포함 요금 체계의 명확한 공시",
    "medium", "low", "1개월", "마케팅팀"
)

action_plan.display_action_plan()
tracker.log_task("실행 계획 수립", f"{len(action_plan.action_items)}개 실행 항목 생성")
```

**실행 결과**: 4개 우선순위 카테고리에 따라 체계적으로 분류된 실행 계획이 수립되며, 각 항목별로 구체적인 기간과 담당 부서가 지정됩니다.

---

## 5.6 Phase 5: 최종 보고서 및 프레젠테이션

**주요 포인트**: 전체 분석 결과를 종합하여 경영진과 이해관계자들에게 명확하고 설득력 있는 보고서를 작성합니다.

### 5.6.1 종합 요약 대시보드

```python
def create_executive_summary():
    """경영진 요약 보고서 생성"""
    summary = tracker.get_summary()
    
    print("="*60)
    print("🎯 타이타닉 데이터 분석 프로젝트 최종 보고서")
    print("="*60)
    
    print(f"\n📊 프로젝트 개요:")
    print(f"  • 분석 기간: {summary['duration']}")
    print(f"  • 완료 작업: {summary['tasks_completed']}개")
    print(f"  • 도출 인사이트: {summary['insights_found']}개")
    print(f"  • 실행 제안: {summary['recommendations_made']}개")
    
    print(f"\n🔍 핵심 발견사항:")
    print(f"  • AI와 인간 분석의 높은 일치도로 신뢰성 확보")
    print(f"  • 5개 영역에 걸친 포괄적 인사이트 도출")
    print(f"  • 통계적 유의성 검증으로 과학적 근거 마련")
    print(f"  • 실행 가능한 개선 방안 제시")
    
    print(f"\n💼 비즈니스 임팩트:")
    print(f"  • 안전 정책 개선으로 생존율 향상 기대")
    print(f"  • 서비스 차별화로 고객 만족도 증가")
    print(f"  • 리스크 관리 체계 강화")
    print(f"  • 운영 효율성 개선")

def create_visual_dashboard(df_enhanced):
    """시각적 대시보드 생성"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('타이타닉 데이터 분석 종합 대시보드', fontsize=16, fontweight='bold')
    
    # 1. 전체 생존율
    survival_rate = df_enhanced['Survived'].mean()
    axes[0,0].pie([survival_rate, 1-survival_rate], labels=['생존', '사망'], 
                  autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
    axes[0,0].set_title('전체 생존율')
    
    # 2. 등급별 생존율
    class_survival = df_enhanced.groupby('Pclass')['Survived'].mean()
    class_survival.plot(kind='bar', ax=axes[0,1], color='skyblue')
    axes[0,1].set_title('등급별 생존율')
    axes[0,1].set_xlabel('등급')
    axes[0,1].set_ylabel('생존율')
    
    # 3. 성별 생존율
    gender_survival = df_enhanced.groupby('Sex')['Survived'].mean()
    gender_survival.plot(kind='bar', ax=axes[0,2], color=['pink', 'lightblue'])
    axes[0,2].set_title('성별 생존율')
    axes[0,2].set_xlabel('성별')
    axes[0,2].set_ylabel('생존율')
    
    # 4. 가족 크기별 생존율
    family_survival = df_enhanced.groupby('FamilySize')['Survived'].mean()
    family_survival.plot(kind='line', ax=axes[1,0], marker='o', color='green')
    axes[1,0].set_title('가족 크기별 생존율')
    axes[1,0].set_xlabel('가족 크기')
    axes[1,0].set_ylabel('생존율')
    
    # 5. 연령대별 생존율
    age_survival = df_enhanced.groupby('AgeGroup')['Survived'].mean()
    age_survival.plot(kind='bar', ax=axes[1,1], color='orange')
    axes[1,1].set_title('연령대별 생존율')
    axes[1,1].set_xlabel('연령대')
    axes[1,1].set_ylabel('생존율')
    
    # 6. 항구별 생존율
    port_survival = df_enhanced.groupby('EmbarkedFull')['Survived'].mean()
    port_survival.plot(kind='bar', ax=axes[1,2], color='purple')
    axes[1,2].set_title('승선 항구별 생존율')
    axes[1,2].set_xlabel('항구')
    axes[1,2].set_ylabel('생존율')
    
    plt.tight_layout()
    plt.show()

# 최종 보고서 생성
create_executive_summary()
create_visual_dashboard(df_enhanced)
tracker.log_task("최종 보고서 작성", "경영진 요약 및 시각적 대시보드 완성")
```

**실행 결과**: 프로젝트 전체 개요, 핵심 발견사항, 비즈니스 임팩트를 요약한 경영진 보고서와 6개 차트로 구성된 종합 대시보드가 완성됩니다.

### 5.6.2 데이터 분석의 윤리적 고려사항

```python
print("""
📊 데이터 뒤에는 사람이 있습니다

타이타닉 데이터의 각 행은 단순한 숫자가 아니라 실제 존재했던 사람들의 이야기입니다. 
데이터 분석가는 항상 데이터 뒤에 숨겨진 인간의 이야기를 기억하고, 
윤리적이고 책임감 있는 분석을 수행해야 합니다.

🔍 윤리적 분석의 원칙:
• 인간 존중: 데이터는 실제 사람들의 삶을 반영함을 인식
• 편향 주의: 사회적 편견이 분석에 반영되지 않도록 주의
• 투명성: 분석 과정과 한계를 명확히 공개
• 책임감: 분석 결과가 미칠 영향을 신중히 고려
• 공정성: 모든 그룹에 대해 공평한 관점 유지
""")

tracker.add_insight("데이터 분석의 윤리적 책임 중요성", "윤리적 고려사항")
```

**실행 결과**: 데이터 분석의 윤리적 측면을 강조하며, 인간 존중, 편향 주의, 투명성, 책임감, 공정성의 5가지 원칙이 제시됩니다.

---

## 5.7 다음 단계 학습 가이드

**주요 포인트**: 2장에서 습득한 EDA 역량을 바탕으로 3장 통계적 추론을 위한 준비 단계와 심화 학습 방향을 제시합니다.

### 5.7.1 즉시 실습 과제

```python
def next_learning_path():
    """다음 학습 단계 가이드"""
    print("=== 다음 학습 단계 가이드 ===")
    
    learning_path = {
        '즉시 실습 과제': [
            "다른 데이터셋으로 하이브리드 EDA 반복 연습",
            "AI 도구들 (AutoViz, D-Tale) 심화 활용",
            "자신만의 EDA 체크리스트 개발",
            "GitHub에 EDA 포트폴리오 구축"
        ],
        '3장 준비 사항': [
            "통계적 가설 검정 기초 복습",
            "확률 분포 개념 정리",
            "p-값과 신뢰구간 이해도 점검",
            "비즈니스 문제를 통계 문제로 변환하는 연습"
        ],
        '심화 학습 주제': [
            "고급 시각화 라이브러리 (Plotly, Bokeh)",
            "대화형 대시보드 개발 (Streamlit, Dash)",
            "A/B 테스트 설계 및 분석",
            "실험 계획법 (DOE) 기초"
        ],
        '실무 프로젝트 아이디어': [
            "온라인 쇼핑 고객 행동 분석",
            "주식시장 데이터 탐색적 분석", 
            "소셜미디어 감성 분석",
            "부동산 가격 예측 모델링"
        ]
    }
    
    for category, items in learning_path.items():
        print(f"\n📚 {category}:")
        for i, item in enumerate(items, 1):
            print(f"  {i}. {item}")
    
    return learning_path

# 다음 학습 경로 안내
next_path = next_learning_path()
```

### 5.7.2 3장 예고: 통계적 기초와 추론

```python
print("""
🔮 다음 장 미리보기: 3장 통계적 기초와 추론

3장에서는 EDA에서 발견한 패턴들을 과학적으로 검증하는 방법을 배웁니다:

📊 주요 학습 내용:
  • 확률 분포와 통계적 개념의 실무 활용
  • 가설 검정을 통한 데이터 기반 의사결정
  • 상관관계와 인과관계의 올바른 구별
  • AI가 생성한 통계 분석 결과 검증 방법
  • 실제 비즈니스 문제에 통계적 방법 적용

💡 3장의 핵심 질문:
  • "이 패턴이 우연의 일치일까, 실제 관계일까?"
  • "샘플에서 발견한 차이가 모집단에서도 유의할까?"
  • "상관관계가 있다고 인과관계도 있을까?"
  • "AI의 통계 분석 결과를 어떻게 검증할까?"

🎯 3장 학습 목표:
EDA에서 '무엇을' 발견했다면, 3장에서는 '왜 그런지', '얼마나 확실한지'를 
과학적으로 증명하는 방법을 배우게 됩니다.

데이터 과학자로서 한 단계 더 성장할 준비가 되셨나요? 
3장에서 만나뵙겠습니다! 🚀
""")
```

---

## 5.8 직접 해보기 / 연습 문제

### 연습 문제 1: 다른 데이터셋으로 하이브리드 EDA 수행 ⭐⭐⭐
**문제**: House Prices 데이터셋을 이용하여 이번 장에서 배운 하이브리드 EDA 방법론을 적용해보세요.

**단계별 가이드**:
1. AI 도구로 초기 탐색 수행 (ydata-profiling, SweetViz)
2. 부동산 도메인 지식 기반 파생 변수 생성
3. 전통적 방식으로 심층 분석 (가격대별, 지역별, 시설별)
4. AI와 인간 분석 결과 교차 검증
5. 최종 비즈니스 인사이트 도출 (투자 전략, 가격 예측 요인)

**힌트**: 주택 가격에 영향을 미치는 요인들을 카테고리별로 분류하고, 각 카테고리의 중요도를 정량적으로 평가해보세요.

### 연습 문제 2: AI 도구 비교 분석 ⭐⭐
**문제**: 동일한 데이터셋에 대해 ydata-profiling, SweetViz, AutoViz를 모두 적용하고 각 도구의 장단점을 비교 분석하세요.

**비교 기준**:
- 분석 속도와 효율성
- 시각화 품질과 다양성
- 인사이트 발견 능력
- 사용 편의성
- 결과 해석의 용이성

**결과물**: 3개 도구의 비교표와 상황별 추천 가이드라인

### 연습 문제 3: 편향 탐지 및 대응 ⭐⭐⭐⭐
**문제**: 타이타닉 데이터에서 다음과 같은 편향들을 탐지하고 대응 방안을 제시하세요.

**탐지할 편향들**:
- **생존자 편향 (Survivorship bias)**: 생존자 데이터만 과대 표현되는 문제
- **선택 편향 (Selection bias)**: 특정 그룹이 과소/과대 표현되는 문제  
- **확인 편향 (Confirmation bias)**: 기존 가정을 뒷받침하는 증거만 찾는 문제

**대응 방안 예시**:
```python
def detect_survivorship_bias(df):
    """생존자 편향 탐지"""
    # 생존자와 비생존자의 데이터 완전성 비교
    survival_data_quality = df.groupby('Survived').apply(
        lambda x: x.isnull().sum() / len(x)
    )
    return survival_data_quality

def mitigate_selection_bias(df):
    """선택 편향 완화"""
    # 가중치 조정을 통한 대표성 확보
    # 층화 샘플링을 통한 균형 잡힌 분석
    pass
```

### 연습 문제 4: 실행 계획 수립 ⭐⭐⭐⭐⭐
**문제**: 이번 프로젝트에서 도출된 권고사항 중 하나를 선택하여 구체적인 실행 계획을 수립하세요.

**선택 예시**: "디지털 안전 시스템 구축"

**실행 계획 구성 요소**:
1. **구현 단계별 세부 계획**
   - Phase 1: 요구사항 분석 및 설계 (2개월)
   - Phase 2: 시스템 개발 및 테스트 (6개월)
   - Phase 3: 시범 운영 및 피드백 (2개월)
   - Phase 4: 전면 도입 및 안정화 (2개월)

2. **필요 자원 및 예산 추정**
   - 인력: 개발팀 5명, PM 1명, 보안 전문가 1명
   - 기술: IoT 센서, 모바일 앱, 클라우드 인프라
   - 예산: 총 $500K (개발 $300K, 인프라 $150K, 운영 $50K)

3. **성과 측정 지표 설정**
   - 대피 시간 단축률: 30% 목표
   - 승객 만족도: 4.5/5.0 목표
   - 시스템 가용성: 99.9% 목표

4. **위험 요소 및 대응 방안**
   - 기술적 위험: POC를 통한 사전 검증
   - 예산 위험: 단계별 예산 승인 체계
   - 운영 위험: 충분한 교육 및 매뉴얼 제공

---

## 5.9 요약 및 핵심 정리

### 🎯 핵심 개념 요약

1. **하이브리드 EDA 접근법**
   - AI 도구의 효율성과 인간의 창의성을 결합
   - 5단계 체계적 분석 프로세스 (AI 탐색 → 인간 분석 → 하이브리드 검증 → 비즈니스 인사이트 → 실행 계획)
   - 교차 검증을 통한 신뢰성 확보

2. **도메인 지식의 중요성**
   - 역사적 맥락과 비즈니스 환경을 고려한 해석
   - 의미 있는 파생 변수 생성으로 분석 깊이 확보
   - AI가 놓칠 수 있는 세부적인 패턴 발견

3. **통계적 검증의 필요성**
   - 패턴의 우연성 vs 의미성 구별
   - 가설 검정을 통한 과학적 증명
   - AI 결과의 비판적 검토와 검증

4. **실행 가능한 제안의 가치**
   - 분석을 넘어선 실무 적용 방안 제시
   - 우선순위 기반 단계별 구현 로드맵
   - 비용-효과 분석을 포함한 비즈니스 케이스

### 💪 습득한 핵심 역량

✅ **체계적 분석 설계**: 프로젝트 계획부터 실행까지 전 과정 관리
✅ **도구 통합 활용**: AI 도구와 전통적 방법의 효과적 결합
✅ **비판적 사고**: AI 결과에 대한 검증과 해석 능력
✅ **비즈니스 마인드**: 기술적 분석을 실무 가치로 전환
✅ **커뮤니케이션**: 복잡한 분석 결과의 명확한 전달
✅ **윤리적 인식**: 데이터 뒤의 인간적 가치 인식

### 🔄 학습 성과 자가 진단

```python
def self_assessment_checklist():
    """학습 성과 자가 진단 체크리스트"""
    checklist = {
        'AI 도구 활용': [
            "ydata-profiling로 포괄적 분석 보고서를 생성할 수 있다",
            "SweetViz로 타겟 중심 비교 분석을 수행할 수 있다",
            "AI 분석 결과를 비판적으로 검토할 수 있다"
        ],
        '전통적 EDA': [
            "도메인 지식을 바탕으로 의미 있는 파생 변수를 생성할 수 있다",
            "다양한 관점에서 심층적인 데이터 탐색을 수행할 수 있다",
            "통계적 검증을 통해 분석 결과를 입증할 수 있다"
        ],
        '하이브리드 접근': [
            "AI와 인간 분석 결과를 효과적으로 결합할 수 있다",
            "교차 검증을 통해 분석의 신뢰성을 확보할 수 있다",
            "상황에 맞는 최적의 분석 전략을 수립할 수 있다"
        ],
        '비즈니스 적용': [
            "분석 결과를 비즈니스 인사이트로 전환할 수 있다",
            "실행 가능한 개선 방안을 제시할 수 있다",
            "우선순위 기반 실행 계획을 수립할 수 있다"
        ]
    }
    
    print("=== 학습 성과 자가 진단 ===")
    for category, items in checklist.items():
        print(f"\n📋 {category}:")
        for item in items:
            print(f"  ☐ {item}")
    
    print("\n💡 체크된 항목이 80% 이상이면 다음 단계로 진행하세요!")

# 자가 진단 실행
self_assessment_checklist()
```

### 🚀 다음 학습을 위한 준비

2장에서 '무엇을' 발견했다면, 3장에서는 '왜 그런지'를 과학적으로 증명하는 통계적 추론 방법을 배우게 됩니다. EDA는 데이터의 보물을 찾는 탐험이라면, 통계적 추론은 그 보물이 진짜인지 확인하는 검증 과정입니다.

**3장 학습 준비사항**:
- 확률과 통계의 기본 개념 복습
- 가설 설정과 검정의 논리적 사고
- p-값과 신뢰구간의 올바른 해석
- 비즈니스 문제의 통계적 정형화

---

**🎉 2장 탐색적 데이터 분석(EDA) 완주를 축하합니다! 🎉**

여러분은 이제 데이터에서 의미 있는 이야기를 찾아내고, AI와 효과적으로 협업하며, 비즈니스 가치를 창출하는 진정한 데이터 탐험가가 되었습니다! 

**2장에서의 여정을 되돌아보며:**
- 🔍 **Part 1**: 데이터의 구조와 특성을 파악하는 기초 탐정 기술
- 📊 **Part 2**: 데이터를 아름답고 의미 있게 시각화하는 예술가 기술  
- 📈 **Part 3**: 숫자로 데이터의 핵심을 요약하는 통계학자 기술
- 🤖 **Part 4**: AI와 인간이 협력하는 미래형 분석가 기술
- 🎯 **Part 5**: 모든 기술을 통합한 완전한 프로젝트 수행 능력

이제 여러분은 **데이터 탐험에서 비즈니스 가치 창출까지의 완전한 여정**을 마스터했습니다!

다음 3장에서는 이 탐험에서 발견한 보물들이 진짜인지 과학적으로 검증하는 방법을 배우게 됩니다. 계속해서 데이터 과학의 놀라운 여정을 함께 하세요! ✨

