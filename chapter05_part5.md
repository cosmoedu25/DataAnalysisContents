# 5장 Part 5: 프로젝트 - 예측 모델 개발 및 평가

## 학습 목표
이번 파트를 완료하면 다음을 할 수 있습니다:
- 실제 비즈니스 문제를 머신러닝 문제로 체계적으로 정의할 수 있다
- 전체 모델링 파이프라인을 처음부터 끝까지 구축할 수 있다
- 모델 성능 개선을 위한 다양한 기법을 체계적으로 적용할 수 있다
- 최종 결과를 해석하고 비즈니스 관점에서 의미 있는 보고서를 작성할 수 있다

## 이번 파트 미리보기

지금까지 Part 1~4에서 머신러닝의 이론과 개별 기법들을 배웠다면, 이제는 그 모든 것을 종합하여 **실제 프로젝트**를 수행할 차례입니다! 

이번 프로젝트에서는 **타이타닉 생존 예측**과 **주택 가격 예측**이라는 두 가지 대표적인 문제를 통해, 실무에서 머신러닝 프로젝트를 진행하는 전체 과정을 체험하게 됩니다.

> 💡 **왜 이 두 프로젝트인가요?**
> - **타이타닉**: 분류 문제의 정석! 역사적 데이터로 흥미롭고, 다양한 특성이 있어 특성 공학을 연습하기 좋습니다.
> - **주택 가격**: 회귀 문제의 대표주자! 실생활과 밀접하고, 복잡한 관계를 모델링하는 연습에 최적입니다.

## 1. 문제 정의와 계획 수립

### 1.1 비즈니스 문제 이해하기

머신러닝 프로젝트의 첫 단계는 **문제를 명확히 정의**하는 것입니다. 

```python
# 프로젝트 목표 정의 프레임워크
class ProjectDefinition:
    def __init__(self, project_name):
        self.project_name = project_name
        self.business_goal = None
        self.ml_problem_type = None
        self.evaluation_metric = None
        self.constraints = []
        self.deliverables = []
    
    def define_project(self):
        """프로젝트를 체계적으로 정의합니다."""
        print(f"=== {self.project_name} 프로젝트 정의 ===")
        print(f"비즈니스 목표: {self.business_goal}")
        print(f"ML 문제 유형: {self.ml_problem_type}")
        print(f"평가 지표: {self.evaluation_metric}")
        print(f"제약 사항: {', '.join(self.constraints)}")
        print(f"산출물: {', '.join(self.deliverables)}")

# 프로젝트 1: 타이타닉 생존 예측
titanic_project = ProjectDefinition("타이타닉 생존 예측")
titanic_project.business_goal = "승객 정보를 기반으로 생존 가능성 예측"
titanic_project.ml_problem_type = "이진 분류 (Binary Classification)"
titanic_project.evaluation_metric = "정확도(Accuracy), 정밀도(Precision), 재현율(Recall)"
titanic_project.constraints = ["1912년 당시 데이터만 사용", "결측치 처리 필요"]
titanic_project.deliverables = ["예측 모델", "특성 중요도 분석", "생존 요인 보고서"]

titanic_project.define_project()
```

**🎯 왜 문제 정의가 중요한가요?**
- **방향성 제시**: 프로젝트의 목표가 명확해야 올바른 방향으로 진행할 수 있습니다.
- **평가 기준 설정**: 어떤 지표로 성공을 측정할지 미리 정해야 합니다.
- **자원 계획**: 필요한 데이터, 시간, 인력을 예측할 수 있습니다.

### 1.2 데이터 이해와 탐색 계획

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 데이터 탐색 체크리스트
class DataExplorationChecklist:
    def __init__(self, df):
        self.df = df
        self.checks = {
            '데이터 크기': self.check_shape,
            '데이터 타입': self.check_dtypes,
            '결측치': self.check_missing,
            '중복값': self.check_duplicates,
            '기본 통계': self.check_statistics
        }
    
    def check_shape(self):
        return f"행: {self.df.shape[0]}, 열: {self.df.shape[1]}"
    
    def check_dtypes(self):
        return self.df.dtypes.value_counts()
    
    def check_missing(self):
        missing = self.df.isnull().sum()
        return missing[missing > 0]
    
    def check_duplicates(self):
        return f"중복 행 수: {self.df.duplicated().sum()}"
    
    def check_statistics(self):
        return self.df.describe()
    
    def run_all_checks(self):
        """모든 체크를 실행합니다."""
        for check_name, check_func in self.checks.items():
            print(f"\n=== {check_name} ===")
            print(check_func())


## 2. 타이타닉 생존 예측 프로젝트

### 2.1 데이터 로드 및 초기 탐색

```python
# 타이타닉 데이터 로드
# 실제로는 pd.read_csv('titanic.csv')로 로드하지만, 여기서는 샘플 데이터를 생성합니다
def load_titanic_data():
    """타이타닉 데이터를 로드합니다."""
    # 샘플 데이터 생성 (실제로는 Kaggle에서 다운로드)
    np.random.seed(42)
    n_samples = 891
    
    data = {
        'PassengerId': range(1, n_samples + 1),
        'Survived': np.random.randint(0, 2, n_samples),
        'Pclass': np.random.choice([1, 2, 3], n_samples, p=[0.24, 0.21, 0.55]),
        'Name': ['Passenger_' + str(i) for i in range(n_samples)],
        'Sex': np.random.choice(['male', 'female'], n_samples, p=[0.65, 0.35]),
        'Age': np.random.normal(30, 15, n_samples),
        'SibSp': np.random.poisson(0.5, n_samples),
        'Parch': np.random.poisson(0.4, n_samples),
        'Ticket': ['Ticket_' + str(i) for i in range(n_samples)],
        'Fare': np.random.exponential(32, n_samples),
        'Cabin': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', np.nan], 
                                 n_samples, p=[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.86]),
        'Embarked': np.random.choice(['S', 'C', 'Q', np.nan], 
                                    n_samples, p=[0.70, 0.20, 0.09, 0.01])
    }
    
    df = pd.DataFrame(data)
    # 일부 Age를 결측치로 만들기
    df.loc[np.random.choice(df.index, 177, replace=False), 'Age'] = np.nan
    
    return df

# 데이터 로드
df_titanic = load_titanic_data()

# 초기 탐색
explorer = DataExplorationChecklist(df_titanic)
explorer.run_all_checks()
```

### 2.2 탐색적 데이터 분석 (EDA)

```python
def perform_titanic_eda(df):
    """타이타닉 데이터에 대한 EDA를 수행합니다."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 생존율 분포
    survival_counts = df['Survived'].value_counts()
    axes[0, 0].pie(survival_counts, labels=['사망', '생존'], autopct='%1.1f%%', 
                   colors=['#ff6b6b', '#4ecdc4'])
    axes[0, 0].set_title('전체 생존율')
    
    # 2. 성별에 따른 생존율
    survival_by_sex = df.groupby('Sex')['Survived'].mean()
    axes[0, 1].bar(survival_by_sex.index, survival_by_sex.values, 
                   color=['#5f9ea0', '#ff69b4'])
    axes[0, 1].set_title('성별 생존율')
    axes[0, 1].set_ylabel('생존율')
    
    # 3. 객실 등급별 생존율
    survival_by_class = df.groupby('Pclass')['Survived'].mean()
    axes[0, 2].bar(survival_by_class.index, survival_by_class.values,
                   color=['#ffd700', '#c0c0c0', '#cd7f32'])
    axes[0, 2].set_title('객실 등급별 생존율')
    axes[0, 2].set_xlabel('객실 등급')
    axes[0, 2].set_ylabel('생존율')
    
    # 4. 나이 분포
    df['Age'].dropna().hist(bins=30, ax=axes[1, 0], color='#6c5ce7')
    axes[1, 0].set_title('승객 나이 분포')
    axes[1, 0].set_xlabel('나이')
    
    # 5. 나이대별 생존율
    age_bins = [0, 18, 35, 50, 65, 100]
    age_labels = ['아동', '청년', '중년', '장년', '노년']
    df['AgeGroup'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)
    
    survival_by_age = df.groupby('AgeGroup')['Survived'].mean()
    axes[1, 1].bar(survival_by_age.index, survival_by_age.values,
                   color='#74b9ff')
    axes[1, 1].set_title('나이대별 생존율')
    axes[1, 1].set_xlabel('나이대')
    axes[1, 1].set_ylabel('생존율')
    
    # 6. 요금 분포
    df['Fare'].hist(bins=30, ax=axes[1, 2], color='#fdcb6e')
    axes[1, 2].set_title('티켓 요금 분포')
    axes[1, 2].set_xlabel('요금')
    
    plt.tight_layout()
    plt.show()
    
    return df

# EDA 수행
df_titanic = perform_titanic_eda(df_titanic)
```

**💡 EDA에서 발견한 인사이트**
1. **성별 차이**: 여성의 생존율이 남성보다 훨씬 높음 (여성과 아이 먼저 원칙)
2. **객실 등급**: 1등급 승객의 생존율이 3등급보다 높음 (경제적 지위의 영향)
3. **나이**: 아동의 생존율이 상대적으로 높음
4. **요금**: 높은 요금을 지불한 승객의 생존율이 높음

### 2.3 특성 공학 (Feature Engineering)

```python
def feature_engineering_titanic(df):
    """타이타닉 데이터에 대한 특성 공학을 수행합니다."""
    df_copy = df.copy()
    
    # 1. 가족 크기 특성 생성
    df_copy['FamilySize'] = df_copy['SibSp'] + df_copy['Parch'] + 1
    
    # 2. 혼자 여행 여부
    df_copy['IsAlone'] = (df_copy['FamilySize'] == 1).astype(int)
    
    # 3. 나이 그룹 (이미 생성됨)
    
    # 4. 요금 구간
    df_copy['FareGroup'] = pd.qcut(df_copy['Fare'], q=4, 
                                   labels=['저가', '중저가', '중고가', '고가'])
    
    # 5. 타이틀 추출 (실제 데이터에서는 이름에서 추출)
    df_copy['Title'] = df_copy['Name'].str.extract(r' ([A-Za-z]+)\.')
    
    # 6. 캐빈 유무
    df_copy['HasCabin'] = df_copy['Cabin'].notna().astype(int)
    
    print("=== 새로 생성된 특성들 ===")
    print(f"가족 크기 분포:\n{df_copy['FamilySize'].value_counts()}")
    print(f"\n혼자 여행하는 승객 비율: {df_copy['IsAlone'].mean():.2%}")
    print(f"\n요금 그룹 분포:\n{df_copy['FareGroup'].value_counts()}")
    print(f"\n캐빈 정보가 있는 승객 비율: {df_copy['HasCabin'].mean():.2%}")
    
    return df_copy

# 특성 공학 적용
df_titanic = feature_engineering_titanic(df_titanic)
```

**🔧 특성 공학이 중요한 이유**
1. **도메인 지식 활용**: 타이타닉 사건의 역사적 맥락을 반영
2. **정보 추출**: 기존 특성에서 숨겨진 패턴 발견
3. **모델 성능 향상**: 더 의미 있는 특성으로 예측력 증가
4. **해석 가능성**: 비즈니스 관점에서 이해하기 쉬운 특성

### 2.4 데이터 전처리

```python
class TitanicPreprocessor:
    """타이타닉 데이터 전처리를 위한 클래스"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def handle_missing_values(self, df):
        """결측치를 처리합니다."""
        df_copy = df.copy()
        
        # 나이: 중앙값으로 대체
        age_median = df_copy['Age'].median()
        df_copy['Age'].fillna(age_median, inplace=True)
        print(f"나이 결측치를 중앙값 {age_median:.1f}로 대체했습니다.")
        
        # Embarked: 최빈값으로 대체
        embarked_mode = df_copy['Embarked'].mode()[0]
        df_copy['Embarked'].fillna(embarked_mode, inplace=True)
        print(f"Embarked 결측치를 최빈값 '{embarked_mode}'로 대체했습니다.")
        
        # Fare: 평균값으로 대체
        fare_mean = df_copy['Fare'].mean()
        df_copy['Fare'].fillna(fare_mean, inplace=True)
        
        return df_copy
    
    def encode_categorical_features(self, df):
        """범주형 변수를 인코딩합니다."""
        df_copy = df.copy()
        
        # 원-핫 인코딩할 변수들
        categorical_cols = ['Sex', 'Embarked', 'Pclass']
        
        for col in categorical_cols:
            # 원-핫 인코딩
            dummies = pd.get_dummies(df_copy[col], prefix=col, drop_first=True)
            df_copy = pd.concat([df_copy, dummies], axis=1)
            df_copy.drop(col, axis=1, inplace=True)
        
        # 순서형 변수 인코딩
        if 'FareGroup' in df_copy.columns:
            fare_group_map = {'저가': 0, '중저가': 1, '중고가': 2, '고가': 3}
            df_copy['FareGroup'] = df_copy['FareGroup'].map(fare_group_map)
        
        return df_copy
    
    def select_features(self, df):
        """모델링에 사용할 특성을 선택합니다."""
        # 불필요한 컬럼 제거
        columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin', 
                          'AgeGroup', 'Title']
        
        df_copy = df.copy()
        for col in columns_to_drop:
            if col in df_copy.columns:
                df_copy.drop(col, axis=1, inplace=True)
        
        return df_copy
    
    def preprocess(self, df):
        """전체 전처리 파이프라인을 실행합니다."""
        print("=== 데이터 전처리 시작 ===")
        
        # 1. 결측치 처리
        df = self.handle_missing_values(df)
        
        # 2. 범주형 변수 인코딩
        df = self.encode_categorical_features(df)
        
        # 3. 특성 선택
        df = self.select_features(df)
        
        print(f"\n최종 특성 수: {df.shape[1]}")
        print(f"최종 특성 목록: {list(df.columns)}")
        
        return df

# 전처리 수행
preprocessor = TitanicPreprocessor()
df_processed = preprocessor.preprocess(df_titanic)
```

### 2.5 모델 학습 및 평가

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score

def train_and_evaluate_models(df):
    """여러 모델을 학습하고 평가합니다."""
    
    # 특성과 타겟 분리
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    # 학습/테스트 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 모델 정의
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    # 결과 저장
    results = {}
    
    print("=== 모델 학습 및 평가 ===")
    
    for name, model in models.items():
        print(f"\n{name} 학습 중...")
        
        # 모델 학습
        if name in ['Logistic Regression', 'SVM']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # 평가 지표 계산
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # 교차 검증
        if name in ['Logistic Regression', 'SVM']:
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        else:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'model': model
        }
        
        print(f"정확도: {accuracy:.4f}")
        print(f"정밀도: {precision:.4f}")
        print(f"재현율: {recall:.4f}")
        print(f"F1 점수: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"교차 검증 평균: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    return results, X_train, X_test, y_train, y_test, scaler

# 모델 학습 및 평가
results, X_train, X_test, y_train, y_test, scaler = train_and_evaluate_models(df_processed)
```

**📊 왜 여러 모델을 비교하나요?**
1. **No Free Lunch 정리**: 모든 문제에 최적인 단일 알고리즘은 없습니다
2. **다양한 관점**: 각 알고리즘은 데이터를 다른 방식으로 학습합니다
3. **앙상블 가능성**: 여러 모델을 조합하면 더 좋은 성능을 낼 수 있습니다
4. **해석 가능성 vs 성능**: 비즈니스 요구사항에 따라 선택할 수 있습니다

### 2.6 모델 최적화

```python
from sklearn.model_selection import GridSearchCV

def optimize_best_model(results, X_train, y_train, scaler):
    """최고 성능 모델을 선택하고 최적화합니다."""
    
    # 최고 성능 모델 선택 (F1 점수 기준)
    best_model_name = max(results.items(), key=lambda x: x[1]['f1'])[0]
    print(f"최고 성능 모델: {best_model_name}")
    
    # Random Forest 하이퍼파라미터 최적화
    if best_model_name == 'Random Forest':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestClassifier(random_state=42)
        
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, 
            scoring='f1', n_jobs=-1, verbose=1
        )
        
        print("\n하이퍼파라미터 최적화 진행 중...")
        grid_search.fit(X_train, y_train)
        
        print(f"\n최적 파라미터: {grid_search.best_params_}")
        print(f"최적 F1 점수: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    return results[best_model_name]['model']

# 모델 최적화
best_model = optimize_best_model(results, X_train, y_train, scaler)
```

### 2.7 특성 중요도 분석

```python
def analyze_feature_importance(model, feature_names):
    """특성 중요도를 분석하고 시각화합니다."""
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        # 특성 중요도 정렬
        indices = np.argsort(importances)[::-1]
        
        # 시각화
        plt.figure(figsize=(10, 6))
        plt.title('특성 중요도 분석')
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), 
                  [feature_names[i] for i in indices], 
                  rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
        # 상위 5개 특성 출력
        print("\n=== 상위 5개 중요 특성 ===")
        for i in range(min(5, len(importances))):
            print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    
    else:
        print("이 모델은 특성 중요도를 제공하지 않습니다.")

# 특성 중요도 분석
feature_names = X_train.columns.tolist()
analyze_feature_importance(best_model, feature_names)
```

**🎯 특성 중요도 분석의 의의**
1. **모델 해석**: 어떤 특성이 예측에 가장 큰 영향을 미치는지 이해
2. **도메인 검증**: 분석 결과가 실제 도메인 지식과 일치하는지 확인
3. **특성 선택**: 불필요한 특성을 제거하여 모델 단순화
4. **비즈니스 인사이트**: 의사결정에 활용할 수 있는 인사이트 도출

## 3. 주택 가격 예측 프로젝트

### 3.1 프로젝트 설정

```python
# 주택 가격 예측 프로젝트 정의
house_project = ProjectDefinition("주택 가격 예측")
house_project.business_goal = "주택 특성을 기반으로 판매 가격 예측"
house_project.ml_problem_type = "회귀 (Regression)"
house_project.evaluation_metric = "RMSE, MAE, R²"
house_project.constraints = ["다중공선성 처리", "이상치 제거"]
house_project.deliverables = ["가격 예측 모델", "주요 가격 결정 요인 분석"]

house_project.define_project()
```

### 3.2 데이터 로드 및 전처리

```python
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_and_prepare_housing_data():
    """주택 데이터를 로드하고 전처리합니다."""
    # California Housing 데이터셋 사용
    housing = fetch_california_housing()
    
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['Price'] = housing.target
    
    print("=== 데이터 정보 ===")
    print(f"데이터 크기: {df.shape}")
    print(f"\n특성 설명:")
    print("- MedInc: 중간 소득")
    print("- HouseAge: 주택 연령")
    print("- AveRooms: 평균 방 수")
    print("- AveBedrms: 평균 침실 수")
    print("- Population: 인구")
    print("- AveOccup: 평균 거주자 수")
    print("- Latitude: 위도")
    print("- Longitude: 경도")
    print("- Price: 주택 가격 (단위: 10만 달러)")
    
    return df

# 데이터 로드
df_housing = load_and_prepare_housing_data()

# 기본 통계 확인
print("\n=== 기본 통계 ===")
print(df_housing.describe())
```

### 3.3 특성 공학 및 EDA

```python
def housing_feature_engineering(df):
    """주택 데이터에 대한 특성 공학을 수행합니다."""
    df_copy = df.copy()
    
    # 1. 방당 침실 비율
    df_copy['BedroomRatio'] = df_copy['AveBedrms'] / df_copy['AveRooms']
    
    # 2. 인구 밀도
    df_copy['PopulationDensity'] = df_copy['Population'] / df_copy['AveOccup']
    
    # 3. 지역 구분 (위도 기반)
    df_copy['Region'] = pd.cut(df_copy['Latitude'], 
                               bins=[32, 34, 36, 38, 40, 42],
                               labels=['남부', '중남부', '중부', '중북부', '북부'])
    
    # 4. 주택 연령 그룹
    df_copy['AgeGroup'] = pd.cut(df_copy['HouseAge'],
                                 bins=[0, 10, 20, 30, 40, 60],
                                 labels=['신축', '10년미만', '20년미만', '30년미만', '노후'])
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 가격 분포
    df_copy['Price'].hist(bins=50, ax=axes[0, 0], color='#3498db')
    axes[0, 0].set_title('주택 가격 분포')
    axes[0, 0].set_xlabel('가격 (10만 달러)')
    
    # 소득 vs 가격
    axes[0, 1].scatter(df_copy['MedInc'], df_copy['Price'], 
                      alpha=0.5, color='#e74c3c')
    axes[0, 1].set_title('중간 소득 vs 주택 가격')
    axes[0, 1].set_xlabel('중간 소득')
    axes[0, 1].set_ylabel('주택 가격')
    
    # 지역별 평균 가격
    region_prices = df_copy.groupby('Region')['Price'].mean()
    axes[1, 0].bar(region_prices.index, region_prices.values, color='#2ecc71')
    axes[1, 0].set_title('지역별 평균 주택 가격')
    axes[1, 0].set_xlabel('지역')
    axes[1, 0].set_ylabel('평균 가격')
    
    # 주택 연령별 평균 가격
    age_prices = df_copy.groupby('AgeGroup')['Price'].mean()
    axes[1, 1].bar(age_prices.index, age_prices.values, color='#f39c12')
    axes[1, 1].set_title('주택 연령별 평균 가격')
    axes[1, 1].set_xlabel('연령 그룹')
    axes[1, 1].set_ylabel('평균 가격')
    
    plt.tight_layout()
    plt.show()
    
    return df_copy

# 특성 공학 적용
df_housing = housing_feature_engineering(df_housing)
```
