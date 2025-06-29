# 2장 Part 2: 데이터 시각화 기법 (개선 버전)

## 학습 목표
이번 Part를 학습한 후 여러분은 다음을 할 수 있게 됩니다:

- Matplotlib과 Seaborn의 차이점을 이해하고 상황에 맞게 선택할 수 있다
- 단변량 시각화 기법을 단계적으로 적용하여 데이터 분포를 완벽하게 파악할 수 있다
- 다변량 시각화를 통해 변수 간의 숨겨진 관계와 패턴을 발견할 수 있다
- 그래프의 색상, 크기, 스타일이 전달하는 메시지를 정확히 해석할 수 있다
- 청중별로 적합한 시각화 방식을 선택하고 설득력 있는 시각적 스토리를 만들 수 있다

## 이번 Part 미리보기

**"한 장의 그림이 천 마디 말보다 낫다"**는 말이 있습니다. 데이터 분석에서도 마찬가지입니다. 

Part 1에서 데이터의 구조와 특성을 숫자로 파악했다면, 이번 Part 2에서는 그 데이터를 **눈으로 볼 수 있는 형태**로 변환하는 방법을 배워보겠습니다. 복잡한 숫자 테이블에서는 보이지 않던 패턴들이 그래프로 그리는 순간 명확하게 드러나게 됩니다.

### 왜 시각화가 중요할까요?

실제 사례를 생각해보세요:
- 📊 **매출 데이터**: 숫자로만 보면 "이번 달 매출 1,200만원"이지만, 그래프로 보면 "3개월 연속 하락 추세"라는 중요한 패턴을 발견할 수 있습니다
- 👥 **고객 분석**: "20대 여성 고객 300명"이라는 숫자보다, 연령대별 분포 그래프가 마케팅 전략에 훨씬 유용한 정보를 제공합니다
- 🔍 **이상치 탐지**: 평균과 표준편차로는 놓치기 쉬운 특이한 값들이 산점도에서는 한눈에 보입니다

### 무엇을 배우나요?

- 🎨 **시각화 도구 마스터**: Matplotlib(기본기)과 Seaborn(고급 스타일링)의 완전 정복
- 📈 **단변량 시각화**: 하나의 변수로 최대한 많은 정보 추출하기
- 🔗 **다변량 시각화**: 여러 변수 간의 관계와 패턴 발견하기
- 🎯 **시각화 선택 전략**: 언제 어떤 그래프를 사용할지 판단하는 노하우
- 🖼️ **스토리텔링**: 데이터를 설득력 있는 시각적 이야기로 만들기

### 실제 사례로 배우기
타이타닉 데이터를 통해 1912년 비극적인 해상사고에서 생존을 가른 요인들을 시각적으로 발견해보겠습니다. 단순한 그래프 그리기를 넘어, 각 시각화가 어떤 인사이트를 제공하는지까지 완전히 이해할 것입니다.

---

## 실습 환경 준비 (시각화 특화)

### 1. 시각화 라이브러리 설치 및 설정

데이터 시각화를 위해서는 기본 분석 라이브러리 외에 추가적인 시각화 도구들이 필요합니다.

**기본 시각화 라이브러리 설치:**
```bash
# 필수 시각화 라이브러리
pip install matplotlib seaborn

# 고급 시각화 (선택사항)
pip install plotly bokeh

# 폰트 관련 (Windows 사용자)
pip install matplotlib-fontconfig
```

**Google Colab 사용자:**
```python
# Colab에는 대부분 설치되어 있지만, 최신 버전으로 업데이트
!pip install --upgrade matplotlib seaborn plotly

# Colab에서 한글 폰트 설치
!apt-get install -y fonts-nanum
!fc-cache -fv
!rm ~/.cache/matplotlib -rf
```

### 2. 한글 폰트 설정 (시각화용 강화 버전)

시각화에서 한글 깨짐 문제는 가장 흔한 문제 중 하나입니다. 완벽한 해결책을 제시하겠습니다.

**Windows 사용자 (완벽 해결 버전):**
```python
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform

def setup_korean_font():
    """
    한글 폰트를 자동으로 설정하는 함수
    """
    if platform.system() == 'Windows':
        # Windows에서 사용 가능한 한글 폰트들 (우선순위별)
        korean_fonts = [
            'Malgun Gothic',      # 맑은 고딕 (Windows 기본)
            'Microsoft YaHei',    # 마이크로소프트 야헤이
            'NanumGothic',       # 나눔고딕
            'AppleGothic',       # 애플고딕
            'DejaVu Sans'        # 최후 수단
        ]
        
        # 설치된 폰트 목록 확인
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        # 사용 가능한 첫 번째 한글 폰트 선택
        selected_font = None
        for font in korean_fonts:
            if font in available_fonts:
                selected_font = font
                break
        
        if selected_font:
            plt.rcParams['font.family'] = selected_font
            plt.rcParams['axes.unicode_minus'] = False
            print(f"✅ 한글 폰트 설정 완료: {selected_font}")
            return selected_font
        else:
            print("⚠️ 한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
            plt.rcParams['axes.unicode_minus'] = False
            return None
    
    elif platform.system() == 'Darwin':  # macOS
        plt.rcParams['font.family'] = 'AppleGothic'
        plt.rcParams['axes.unicode_minus'] = False
        print("✅ macOS 한글 폰트 설정 완료: AppleGothic")
        return 'AppleGothic'
    
    else:  # Linux
        try:
            plt.rcParams['font.family'] = 'NanumGothic'
            plt.rcParams['axes.unicode_minus'] = False
            print("✅ Linux 한글 폰트 설정 완료: NanumGothic")
            return 'NanumGothic'
        except:
            print("⚠️ 나눔고딕을 설치해주세요: sudo apt-get install fonts-nanum")
            plt.rcParams['axes.unicode_minus'] = False
            return None

# 폰트 설정 실행
font_name = setup_korean_font()
```

**Google Colab 사용자:**
```python
# Colab 전용 한글 폰트 설정
import matplotlib.pyplot as plt

# 나눔폰트 설치 (한 번만 실행)
!apt-get install -y fonts-nanum
!fc-cache -fv
!rm ~/.cache/matplotlib -rf

# 폰트 설정
plt.rcParams['font.family'] = 'NanumBarunGothic'
plt.rcParams['axes.unicode_minus'] = False

print("✅ Colab 한글 폰트 설정 완료!")
```

### 3. 시각화 스타일 설정

```python
import seaborn as sns
import numpy as np

# Seaborn 스타일 설정 (더 예쁜 그래프를 위해)
sns.set_style("whitegrid")  # 격자가 있는 깔끔한 스타일
sns.set_palette("husl")     # 화려하지만 구분하기 쉬운 색상 팔레트

# Matplotlib 기본 설정
plt.rcParams['figure.figsize'] = (10, 6)    # 기본 그래프 크기
plt.rcParams['font.size'] = 12               # 기본 글자 크기
plt.rcParams['lines.linewidth'] = 2          # 선 굵기
plt.rcParams['grid.alpha'] = 0.3             # 격자 투명도

print("✅ 시각화 스타일 설정 완료!")
```

### 4. 색상 팔레트 이해하기

```python
# 다양한 색상 팔레트 비교해보기
def show_color_palettes():
    """
    주요 색상 팔레트를 시각적으로 비교하는 함수
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('📊 주요 색상 팔레트 비교', fontsize=16, fontweight='bold')
    
    palettes = [
        ('기본 (default)', 'deep'),
        ('파스텔 (pastel)', 'pastel'),
        ('밝은 (bright)', 'bright'),
        ('어두운 (dark)', 'dark'),
        ('색조 (husl)', 'husl'),
        ('분기별 (Set2)', 'Set2')
    ]
    
    for i, (name, palette) in enumerate(palettes):
        row = i // 3
        col = i % 3
        
        # 샘플 데이터로 막대그래프 그리기
        data = [3, 7, 2, 5, 8, 1, 4]
        categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        
        colors = sns.color_palette(palette, len(data))
        axes[row, col].bar(categories, data, color=colors)
        axes[row, col].set_title(name)
        axes[row, col].set_ylim(0, 10)
    
    plt.tight_layout()
    plt.show()
    
    print("💡 색상 선택 팁:")
    print("- 범주형 데이터: 'Set2', 'pastel' (구분하기 쉬운 색상)")
    print("- 연속형 데이터: 'viridis', 'plasma' (점진적 변화)")
    print("- 발표용: 'deep', 'dark' (선명하고 전문적)")
    print("- 인쇄용: 'colorblind' (색약자도 구분 가능)")

# 색상 팔레트 비교 실행
show_color_palettes()
```

### 5. 데이터 준비 및 확인

```python
import pandas as pd

# 데이터 로드 및 기본 확인
def load_and_prepare_data():
    """
    타이타닉 데이터를 로드하고 시각화를 위해 준비하는 함수
    """
    try:
        # 온라인 데이터 로드
        url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
        df = pd.read_csv(url)
        print("✅ 온라인 데이터 로드 성공!")
        
    except Exception as e:
        print(f"❌ 온라인 로드 실패: {e}")
        print("로컬 파일을 사용해주세요.")
        return None
    
    # 시각화를 위한 기본 전처리
    print(f"📊 데이터 크기: {df.shape[0]}행 {df.shape[1]}열")
    print(f"📋 컬럼: {list(df.columns)}")
    
    # 결측치가 너무 많은 컬럼 확인
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    high_missing = missing_percentage[missing_percentage > 50]
    
    if len(high_missing) > 0:
        print(f"\n⚠️ 결측치가 많은 컬럼 (50% 이상):")
        for col, pct in high_missing.items():
            print(f"  - {col}: {pct:.1f}%")
        print("  → 이 컬럼들은 시각화 시 주의가 필요합니다")
    
    # 시각화에 유용한 파생 변수 생성
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 18, 35, 60, 100], 
                           labels=['어린이/청소년', '젊은 성인', '중년', '고령'])
    
    df['FareGroup'] = pd.cut(df['Fare'], bins=[0, 10, 30, 100, 1000], 
                            labels=['저렴', '보통', '비싼', '최고급'])
    
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    print(f"\n✅ 시각화용 파생 변수 생성 완료!")
    print(f"  - AgeGroup: 연령대 구분")
    print(f"  - FareGroup: 요금대 구분")  
    print(f"  - FamilySize: 가족 규모")
    
    return df

# 데이터 로드 및 준비
df = load_and_prepare_data()

# 폰트 테스트를 위한 간단한 그래프
if df is not None:
    plt.figure(figsize=(8, 5))
    plt.title("🚢 한글 폰트 테스트: 타이타닉 승객 성별 분포")
    df['Sex'].value_counts().plot(kind='bar', color=['skyblue', 'pink'])
    plt.xlabel('성별')
    plt.ylabel('승객 수')
    plt.xticks(rotation=0)
    plt.show()
    
    print("✅ 모든 설정이 완료되었습니다! 시각화를 시작해보세요! 🎨")
```

### 6. 환경 설정 문제 해결 가이드

**자주 발생하는 문제와 해결책:**

**❓ 한글이 깨져 보인다면:**
```python
# 1단계: 현재 설정 확인
print("현재 폰트:", plt.rcParams['font.family'])

# 2단계: 강제 폰트 변경
plt.rcParams['font.family'] = 'DejaVu Sans'  # 영어로 우선 확인

# 3단계: 폰트 캐시 삭제 (Windows)
import matplotlib
print(f"캐시 위치: {matplotlib.get_cachedir()}")
# 해당 폴더를 수동으로 삭제하고 Python 재시작
```

**❓ 그래프가 너무 작게 보인다면:**
```python
# 기본 크기 조정
plt.rcParams['figure.figsize'] = (12, 8)  # 가로 12인치, 세로 8인치

# 또는 개별 그래프마다
plt.figure(figsize=(15, 10))
```

**❓ 색상이 이상하게 보인다면:**
```python
# 기본 색상으로 초기화
plt.style.use('default')
sns.reset_defaults()
```

**❓ Jupyter에서 그래프가 안 보인다면:**
```python
# 매직 명령어 추가
%matplotlib inline

# 또는 인터랙티브 모드
%matplotlib widget
```

---

## 핵심 개념 설명

### 데이터 시각화의 힘: 숫자를 이야기로 바꾸기

데이터 시각화는 단순히 예쁜 그래프를 그리는 것이 아닙니다. **복잡한 데이터에 숨겨진 패턴과 관계를 눈으로 볼 수 있게 만드는 강력한 도구**입니다.

#### 왜 시각화가 필요할까요?

**🧠 인간의 뇌는 시각 정보 처리에 최적화되어 있습니다**
- 텍스트 처리: 초당 약 250단어
- 시각 정보 처리: 초당 약 36,000비트
- **시각 정보가 13배 더 빠르게 처리됩니다!**

**실제 사례로 이해해보기:**

**숫자로만 보는 경우:**
```
매출 데이터: 1월(1200만원), 2월(1150만원), 3월(1180만원), 4월(1100만원)...
→ 어떤 패턴이 있는지 파악하기 어려움
```

**시각화로 보는 경우:**
```python
# 같은 데이터를 시각화하면
months = ['1월', '2월', '3월', '4월', '5월', '6월']
sales = [1200, 1150, 1180, 1100, 1080, 1050]

plt.figure(figsize=(10, 6))
plt.plot(months, sales, marker='o', linewidth=3, markersize=8)
plt.title('📈 월별 매출 추이 (단위: 만원)', fontsize=14, fontweight='bold')
plt.ylabel('매출 (만원)')
plt.grid(True, alpha=0.3)

# 추세선 추가로 패턴 강조
z = np.polyfit(range(len(sales)), sales, 1)
p = np.poly1d(z)
plt.plot(months, p(range(len(sales))), "r--", alpha=0.8, linewidth=2, label='추세선')
plt.legend()
plt.show()

# → 명확한 하락 추세를 한눈에 파악 가능!
```

### 시각화의 3가지 핵심 목적

#### 1. 🔍 **탐색 (Exploration)**: 데이터에서 패턴 발견
- **목적**: 데이터 분석가가 인사이트를 찾기 위해
- **특징**: 빠르고 간단한 그래프, 여러 각도에서 탐색
- **예시**: 히스토그램으로 분포 확인, 산점도로 상관관계 탐색

#### 2. 📊 **설명 (Explanation)**: 발견한 패턴을 증명
- **목적**: 분석 결과를 동료나 상사에게 설명
- **특징**: 정확하고 상세한 정보 포함
- **예시**: 상세한 범례, 통계량 표시, 여러 그래프 조합

#### 3. 🎯 **설득 (Persuasion)**: 행동 변화 유도
- **목적**: 의사결정자의 행동을 이끌어내기 위해
- **특징**: 간결하고 임팩트 있는 메시지
- **예시**: 핵심 메트릭 강조, 감정에 호소하는 디자인

### 시각화 선택 전략: 언제 어떤 그래프를 쓸까?

데이터 타입과 분석 목적에 따라 최적의 시각화 방법이 달라집니다.

#### 📊 데이터 타입별 시각화 전략

| 분석 목적 | 데이터 타입 | 추천 시각화 | 사용 이유 |
|----------|------------|------------|----------|
| **분포 확인** | 연속형 | 히스토그램, 밀도 플롯 | 데이터가 어떻게 퍼져있는지 확인 |
| **분포 확인** | 범주형 | 막대그래프, 파이차트 | 각 카테고리의 비율 확인 |
| **비교 분석** | 연속형 | 박스플롯, 바이올린 플롯 | 그룹 간 분포 차이 비교 |
| **비교 분석** | 범주형 | 묶은 막대그래프 | 카테고리별 수치 비교 |
| **관계 분석** | 연속형 vs 연속형 | 산점도 | 두 변수 간 상관관계 확인 |
| **관계 분석** | 범주형 vs 연속형 | 박스플롯, 바이올린 플롯 | 카테고리별 수치 분포 차이 |
| **시간 변화** | 시계열 | 선 그래프 | 시간에 따른 변화 추이 확인 |
| **구성 비율** | 범주형 | 파이차트, 도넛차트 | 전체에서 각 부분이 차지하는 비율 |

#### 🎨 상황별 시각화 선택 가이드

**📱 프레젠테이션용 (경영진 보고)**
```python
# 간결하고 임팩트 있는 디자인
plt.style.use('seaborn-v0_8-darkgrid')
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']  # 브랜드 컬러 사용
fontsize = 16  # 큰 글씨
```

**📋 분석 보고서용 (동료 공유)**
```python
# 상세하고 정확한 정보 포함
plt.style.use('seaborn-v0_8-whitegrid')
# 범례, 주석, 통계량 모두 포함
# 여러 서브플롯으로 다각도 분석
```

**🔬 탐색용 (개인 분석)**
```python
# 빠르고 간단한 확인
plt.style.use('default')
# 기본 설정으로 빠른 확인
```

### Matplotlib vs Seaborn: 어떤 것을 언제 사용할까?

#### 📐 **Matplotlib**: 기본기가 탄탄한 만능 도구
- **장점**: 
  - 모든 것을 세밀하게 조절 가능
  - 복잡한 커스터마이징 가능
  - 다른 라이브러리의 기반
- **단점**: 
  - 코드가 길고 복잡
  - 기본 스타일이 단조로움
- **언제 사용**: 정밀한 컨트롤이 필요할 때, 특수한 그래프가 필요할 때

```python
# Matplotlib 예시: 세밀한 조절
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
ax.set_xlabel('값', fontsize=12)
ax.set_ylabel('빈도', fontsize=12)
ax.set_title('히스토그램', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
```

#### 🎨 **Seaborn**: 아름답고 똑똑한 고급 도구
- **장점**: 
  - 기본 디자인이 아름다움
  - 통계적 시각화에 특화
  - 간단한 코드로 복잡한 그래프
- **단점**: 
  - Matplotlib보다 커스터마이징 제한적
  - 특수한 그래프 구현 어려움
- **언제 사용**: 빠르고 예쁜 그래프가 필요할 때, 통계 분석 결과 시각화

```python
# Seaborn 예시: 간단하고 아름다운 결과
sns.histplot(data, bins=30, kde=True)  # 한 줄로 히스토그램 + 밀도곡선
```

### 효과적인 시각화의 7가지 원칙

#### 1. 🎯 **명확성 (Clarity)**: 메시지가 분명해야 함
```python
# ❌ 나쁜 예: 메시지가 불분명
plt.scatter(x, y)
plt.title('데이터')

# ✅ 좋은 예: 명확한 메시지
plt.scatter(x, y, alpha=0.6)
plt.title('💰 연령이 높을수록 보험료가 증가하는 추세')
plt.xlabel('나이')
plt.ylabel('월 보험료 (만원)')
```

#### 2. 🎨 **정확성 (Accuracy)**: 데이터를 왜곡하지 않아야 함
```python
# ❌ 나쁜 예: y축 시작점 조작으로 차이 과장
plt.ylim(95, 100)  # 5% 차이가 매우 크게 보임

# ✅ 좋은 예: 0부터 시작하거나 적절한 범위 설정
plt.ylim(0, 100)   # 실제 차이를 정확히 표현
```

#### 3. ⚡ **효율성 (Efficiency)**: 인지적 부담 최소화
```python
# ❌ 나쁜 예: 너무 많은 정보
plt.plot(data1, label='데이터1')
plt.plot(data2, label='데이터2')
plt.plot(data3, label='데이터3')
plt.plot(data4, label='데이터4')
plt.plot(data5, label='데이터5')  # 너무 많은 선

# ✅ 좋은 예: 핵심 정보만 강조
plt.plot(data_main, linewidth=3, label='주요 지표', color='red')
plt.plot(data_others, alpha=0.3, color='gray', label='기타')
```

#### 4. 🌈 **색상 의미 부여**: 색상이 정보를 전달해야 함
```python
# 색상 의미 부여 예시
colors = {
    '수익': 'green',    # 긍정적 의미
    '손실': 'red',      # 부정적 의미
    '중립': 'gray',     # 중립적 의미
    '강조': 'orange'    # 주목할 점
}
```

#### 5. 📏 **일관성 (Consistency)**: 같은 요소는 같은 방식으로
```python
# 여러 그래프에서 일관된 색상과 스타일 사용
company_colors = {
    '우리회사': '#FF6B6B',
    '경쟁사A': '#4ECDC4', 
    '경쟁사B': '#45B7D1'
}
# 모든 그래프에서 동일한 색상 체계 사용
```

#### 6. 📖 **스토리텔링**: 논리적 흐름으로 구성
```python
# 스토리 구조: 문제 제기 → 분석 → 결론
# 1. 현재 상황 (문제)
# 2. 원인 분석 (분석)  
# 3. 해결 방안 (결론)
```

#### 7. 👥 **청중 고려**: 누구를 위한 그래프인가?
- **경영진**: 간결하고 핵심만, 큰 글씨, 임팩트 있는 색상
- **분석팀**: 상세한 정보, 통계량 포함, 여러 관점
- **일반 대중**: 직관적이고 이해하기 쉬운 형태

### 시각화에서 피해야 할 흔한 실수들

#### ❌ **실수 1: 파이차트 남용**
```python
# 문제: 7개 이상 조각, 비슷한 크기들
# 해결: 막대그래프 사용
```

#### ❌ **실수 2: 3D 그래프 과용**
```python
# 문제: 깊이감 때문에 값 읽기 어려움
# 해결: 2D로 명확하게 표현
```

#### ❌ **실수 3: 너무 많은 색상**
```python
# 문제: 무지개색 사용으로 정보 혼란
# 해결: 2-3가지 주요 색상만 사용
```

#### ❌ **실수 4: 의미 없는 애니메이션**
```python
# 문제: 화려하지만 정보 전달에 방해
# 해결: 꼭 필요한 경우에만 사용
```

이제 이론적 기반을 다졌으니, 다음 섹션에서는 실제 코드로 시각화를 구현해보겠습니다!

---

## 핵심 기술 / 코드 구현

### 1단계: 단변량 시각화 - 하나의 변수로 최대한 많은 정보 얻기

단변량 시각화는 **하나의 변수**에 집중하여 그 변수의 분포, 중심값, 퍼진 정도, 이상치 등을 파악하는 것입니다. 마치 새로운 사람을 만났을 때 그 사람의 기본 정보부터 파악하는 것과 같습니다.

#### 1.1 연속형 변수 시각화: 히스토그램으로 분포 파악하기

히스토그램은 **연속형 데이터의 분포**를 보는 가장 기본적인 방법입니다.

```python
# =====================================================
# 1.1 기본 히스토그램: 나이 분포 확인
# =====================================================

# 먼저 데이터 확인
print("🔍 Age 변수 기본 정보:")
print(f"전체 데이터 수: {len(df)}")
print(f"결측치 수: {df['Age'].isnull().sum()}")
print(f"유효 데이터 수: {df['Age'].dropna().shape[0]}")
print(f"최솟값: {df['Age'].min():.1f}세")
print(f"최댓값: {df['Age'].max():.1f}세")
print(f"평균: {df['Age'].mean():.1f}세")
print(f"중앙값: {df['Age'].median():.1f}세")

# 히스토그램 그리기
plt.figure(figsize=(12, 8))

# 서브플롯 1: 기본 히스토그램
plt.subplot(2, 2, 1)
plt.hist(df['Age'].dropna(),     # 결측치 제거하고 그리기
         bins=20,                # 구간을 20개로 나누기
         alpha=0.7,              # 투명도 설정 (0=투명, 1=불투명)
         color='skyblue',        # 막대 색상
         edgecolor='black',      # 막대 테두리 색상
         linewidth=0.8)          # 테두리 굵기
plt.title('기본 히스토그램')
plt.xlabel('나이')
plt.ylabel('빈도')
plt.grid(True, alpha=0.3)       # 격자 표시 (연한 색으로)

# 서브플롯 2: 통계선이 포함된 히스토그램
plt.subplot(2, 2, 2)
age_data = df['Age'].dropna()
plt.hist(age_data, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')

# 평균선 추가
mean_age = age_data.mean()
plt.axvline(mean_age, color='red', linestyle='--', linewidth=2, 
           label=f'평균: {mean_age:.1f}세')

# 중앙값선 추가
median_age = age_data.median()
plt.axvline(median_age, color='green', linestyle='--', linewidth=2,
           label=f'중앙값: {median_age:.1f}세')

plt.title('통계선 포함 히스토그램')
plt.xlabel('나이')
plt.ylabel('빈도')
plt.legend()                    # 범례 표시
plt.grid(True, alpha=0.3)

# 서브플롯 3: 밀도 곡선이 포함된 히스토그램 (Seaborn 사용)
plt.subplot(2, 2, 3)
sns.histplot(data=df, x='Age', kde=True, bins=20)
# kde=True: Kernel Density Estimation (밀도 곡선) 추가
# 밀도 곡선은 히스토그램을 부드럽게 연결한 곡선
plt.title('밀도 곡선 포함 (Seaborn)')
plt.grid(True, alpha=0.3)

# 서브플롯 4: 구간별 정보가 풍부한 히스토그램
plt.subplot(2, 2, 4)
n, bins, patches = plt.hist(age_data, bins=15, alpha=0.7, edgecolor='black')

# 각 막대에 정확한 값 표시
for i in range(len(patches)):
    height = n[i]
    if height > 0:  # 0이 아닌 경우만 표시
        plt.text(patches[i].get_x() + patches[i].get_width()/2., height + 1,
                f'{int(height)}',  # 정수로 표시
                ha='center', va='bottom', fontsize=8)

plt.title('상세 정보 히스토그램')
plt.xlabel('나이')
plt.ylabel('빈도')
plt.grid(True, alpha=0.3)

plt.tight_layout()              # 서브플롯 간격 자동 조정
plt.show()

# 히스토그램 해석 가이드
print("\n📊 히스토그램 해석 가이드:")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("🔍 분포 모양 확인:")
print(f"  - 평균({mean_age:.1f}세) vs 중앙값({median_age:.1f}세)")
if mean_age > median_age:
    print("  → 오른쪽 꼬리 분포: 고령자가 평균을 끌어올림")
elif mean_age < median_age:
    print("  → 왼쪽 꼬리 분포: 저연령층이 평균을 끌어내림")
else:
    print("  → 대칭 분포: 좌우 균형")

print(f"\n📈 주요 특징:")
print(f"  - 가장 많은 연령대: {20 + np.argmax(n)*((age_data.max()-age_data.min())/15):.0f}세 근처")
print(f"  - 데이터 범위: {age_data.max() - age_data.min():.0f}세")
print(f"  - 표준편차: {age_data.std():.1f}세")
```

#### 1.2 박스플롯으로 분포의 5가지 핵심 정보 한눈에 보기

박스플롯은 **최솟값, 1사분위수, 중앙값, 3사분위수, 최댓값**을 한 번에 보여주는 강력한 도구입니다.

```python
# =====================================================
# 1.2 박스플롯: 분포의 핵심 정보 요약
# =====================================================

plt.figure(figsize=(15, 10))

# 서브플롯 1: 기본 박스플롯
plt.subplot(2, 3, 1)
box_plot = plt.boxplot(df['Age'].dropna(), 
                      patch_artist=True,        # 색칠 가능하게
                      boxprops=dict(facecolor='lightblue', alpha=0.7),
                      medianprops=dict(color='red', linewidth=2))
plt.title('기본 박스플롯')
plt.ylabel('나이')

# 박스플롯의 각 요소 설명을 위한 주석 추가
plt.text(1.15, df['Age'].quantile(0.75), '3사분위수\n(Q3)', ha='left', va='center')
plt.text(1.15, df['Age'].median(), '중앙값\n(Q2)', ha='left', va='center', color='red')
plt.text(1.15, df['Age'].quantile(0.25), '1사분위수\n(Q1)', ha='left', va='center')

# 서브플롯 2: 여러 그룹 비교 박스플롯
plt.subplot(2, 3, 2)
sns.boxplot(data=df, x='Pclass', y='Age')
plt.title('등급별 나이 분포 비교')
plt.xlabel('승객 등급')
plt.ylabel('나이')

# 서브플롯 3: 바이올린 플롯 (박스플롯 + 분포 모양)
plt.subplot(2, 3, 3)
sns.violinplot(data=df, x='Pclass', y='Age', inner='box')
# inner='box': 바이올린 안에 박스플롯도 함께 표시
plt.title('등급별 나이 분포 (바이올린 플롯)')
plt.xlabel('승객 등급')
plt.ylabel('나이')

# 서브플롯 4: 성별로 나눈 박스플롯
plt.subplot(2, 3, 4)
sns.boxplot(data=df, x='Sex', y='Age', palette=['lightpink', 'lightblue'])
plt.title('성별 나이 분포 비교')
plt.xlabel('성별')
plt.ylabel('나이')

# 서브플롯 5: 생존 여부별 박스플롯
plt.subplot(2, 3, 5)
sns.boxplot(data=df, x='Survived', y='Age', palette=['lightcoral', 'lightgreen'])
plt.title('생존 여부별 나이 분포')
plt.xlabel('생존 여부 (0=사망, 1=생존)')
plt.ylabel('나이')

# 서브플롯 6: 요금 박스플롯 (로그 스케일)
plt.subplot(2, 3, 6)
# 요금에는 0이 있을 수 있으므로 작은 값을 더해서 로그 변환
fare_log = np.log(df['Fare'].dropna() + 1)
plt.boxplot(fare_log, patch_artist=True, 
           boxprops=dict(facecolor='gold', alpha=0.7))
plt.title('요금 분포 (로그 스케일)')
plt.ylabel('log(요금 + 1)')

plt.tight_layout()
plt.show()

# 박스플롯 통계량 자세히 분석
print("\n📊 박스플롯 핵심 통계량:")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

age_stats = df['Age'].describe()
print("📈 나이 통계량:")
for key, value in age_stats.items():
    print(f"  {key}: {value:.1f}세")

# IQR 계산 및 이상치 탐지
Q1 = df['Age'].quantile(0.25)
Q3 = df['Age'].quantile(0.75)
IQR = Q3 - Q1
outlier_threshold_low = Q1 - 1.5 * IQR
outlier_threshold_high = Q3 + 1.5 * IQR

outliers = df[(df['Age'] < outlier_threshold_low) | (df['Age'] > outlier_threshold_high)]
print(f"\n🚨 이상치 분석:")
print(f"  IQR: {IQR:.1f}")
print(f"  이상치 기준: {outlier_threshold_low:.1f}세 미만 또는 {outlier_threshold_high:.1f}세 초과")
print(f"  이상치 개수: {len(outliers)}명")
if len(outliers) > 0:
    print(f"  이상치 나이: {sorted(outliers['Age'].tolist())}")
```

#### 1.3 범주형 변수 시각화: 막대그래프와 파이차트

범주형 데이터는 **각 카테고리의 빈도와 비율**을 파악하는 것이 중요합니다.

```python
# =====================================================
# 1.3 범주형 변수 시각화
# =====================================================

plt.figure(figsize=(16, 12))

# 서브플롯 1: 기본 막대그래프 (성별 분포)
plt.subplot(3, 4, 1)
sex_counts = df['Sex'].value_counts()
bars = plt.bar(sex_counts.index, sex_counts.values, 
              color=['lightblue', 'lightpink'], alpha=0.8, edgecolor='black')

# 막대 위에 정확한 수치 표시
for bar, count in zip(bars, sex_counts.values):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 5,
            f'{count}명\n({count/len(df)*100:.1f}%)',
            ha='center', va='bottom')

plt.title('성별 분포')
plt.xlabel('성별')
plt.ylabel('승객 수')
plt.grid(True, alpha=0.3, axis='y')  # y축에만 격자

# 서브플롯 2: 수평 막대그래프 (승선 항구별)
plt.subplot(3, 4, 2)
embarked_counts = df['Embarked'].value_counts()
colors = ['coral', 'lightgreen', 'lightyellow']
bars = plt.barh(embarked_counts.index, embarked_counts.values, color=colors)

# 막대 옆에 수치 표시
for i, (bar, count) in enumerate(zip(bars, embarked_counts.values)):
    width = bar.get_width()
    plt.text(width + 5, bar.get_y() + bar.get_height()/2.,
            f'{count}명 ({count/len(df)*100:.1f}%)',
            ha='left', va='center')

plt.title('승선 항구별 분포')
plt.xlabel('승객 수')
plt.ylabel('승선 항구')

# 서브플롯 3: 파이차트 (객실 등급별)
plt.subplot(3, 4, 3)
pclass_counts = df['Pclass'].value_counts().sort_index()
colors = ['gold', 'silver', '#CD7F32']  # 1등급=금색, 2등급=은색, 3등급=동색
explode = (0.05, 0.05, 0.05)  # 각 조각을 약간씩 분리

wedges, texts, autotexts = plt.pie(pclass_counts.values, 
                                  labels=[f'{i}등급' for i in pclass_counts.index],
                                  colors=colors,
                                  explode=explode,
                                  autopct='%1.1f%%',  # 퍼센트 표시
                                  startangle=90,      # 12시 방향부터 시작
                                  shadow=True)        # 그림자 효과

# 텍스트 스타일 조정
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

plt.title('객실 등급별 분포')

# 서브플롯 4: 도넛차트 (생존 여부)
plt.subplot(3, 4, 4)
survived_counts = df['Survived'].value_counts()
colors = ['lightcoral', 'lightgreen']
labels = ['사망', '생존']

# 도넛차트는 내부에 원을 하나 더 그려서 만듦
wedges, texts, autotexts = plt.pie(survived_counts.values,
                                  labels=labels,
                                  colors=colors,
                                  autopct='%1.1f%%',
                                  startangle=90,
                                  pctdistance=0.85)  # 퍼센트 텍스트 위치

# 가운데 원 그리기 (도넛 모양으로 만들기)
centre_circle = plt.Circle((0,0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.title('생존률')

# 서브플롯 5-8: 묶은 막대그래프 (교차 분석)
plt.subplot(3, 4, 5)
# 성별 × 생존 여부 교차표
cross_tab = pd.crosstab(df['Sex'], df['Survived'])
cross_tab.plot(kind='bar', ax=plt.gca(), color=['lightcoral', 'lightgreen'])
plt.title('성별 × 생존 여부')
plt.xlabel('성별')
plt.ylabel('승객 수')
plt.xticks(rotation=0)  # x축 라벨 회전 안 함
plt.legend(['사망', '생존'])

plt.subplot(3, 4, 6)
# 등급별 × 생존 여부 교차표
cross_tab2 = pd.crosstab(df['Pclass'], df['Survived'])
cross_tab2.plot(kind='bar', ax=plt.gca(), color=['lightcoral', 'lightgreen'])
plt.title('등급별 × 생존 여부')
plt.xlabel('객실 등급')
plt.ylabel('승객 수')
plt.xticks(rotation=0)
plt.legend(['사망', '생존'])

# 서브플롯 7: 비율 스택 막대그래프
plt.subplot(3, 4, 7)
# 각 성별 내에서 생존/사망 비율
cross_pct = pd.crosstab(df['Sex'], df['Survived'], normalize='index') * 100
cross_pct.plot(kind='bar', stacked=True, ax=plt.gca(), 
              color=['lightcoral', 'lightgreen'])
plt.title('성별 생존률 비교')
plt.xlabel('성별')
plt.ylabel('비율 (%)')
plt.xticks(rotation=0)
plt.legend(['사망률', '생존률'])

# 서브플롯 8: 히트맵 형태로 교차표 시각화
plt.subplot(3, 4, 8)
cross_tab_pct = pd.crosstab(df['Pclass'], df['Sex'], normalize='columns') * 100
sns.heatmap(cross_tab_pct, annot=True, fmt='.1f', cmap='Blues', 
           cbar_kws={'label': '비율 (%)'})
plt.title('등급별 성별 구성 비율')

plt.tight_layout()
plt.show()

# 범주형 변수 분석 요약
print("\n📊 범주형 변수 분석 요약:")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

print("👥 성별 분포:")
for sex, count in sex_counts.items():
    print(f"  {sex}: {count}명 ({count/len(df)*100:.1f}%)")

print(f"\n🚢 등급별 분포:")
for pclass, count in pclass_counts.items():
    print(f"  {pclass}등급: {count}명 ({count/len(df)*100:.1f}%)")

print(f"\n💀 전체 생존률: {df['Survived'].mean()*100:.1f}%")

# 성별 생존률
print(f"\n👫 성별 생존률:")
for sex in df['Sex'].unique():
    survival_rate = df[df['Sex'] == sex]['Survived'].mean() * 100
    print(f"  {sex}: {survival_rate:.1f}%")
```

### 2단계: 다변량 시각화 - 변수 간의 관계와 패턴 발견하기

다변량 시각화는 **두 개 이상의 변수 간의 관계**를 탐색하여 숨겨진 패턴과 인사이트를 발견하는 것입니다.

#### 2.1 산점도: 두 연속형 변수의 관계 탐색

```python
# =====================================================
# 2.1 산점도: 연속형 변수 간 관계 분석
# =====================================================

plt.figure(figsize=(16, 12))

# 서브플롯 1: 기본 산점도 (나이 vs 요금)
plt.subplot(3, 3, 1)
plt.scatter(df['Age'], df['Fare'], alpha=0.6, s=30)  # s는 점의 크기
plt.xlabel('나이')
plt.ylabel('요금')
plt.title('나이 vs 요금')
plt.grid(True, alpha=0.3)

# 서브플롯 2: 색상으로 그룹 구분 (생존 여부)
plt.subplot(3, 3, 2)
# 생존자와 사망자를 다른 색상으로 표시
survived = df[df['Survived'] == 1]
died = df[df['Survived'] == 0]

plt.scatter(died['Age'], died['Fare'], alpha=0.6, color='red', 
           label='사망', s=30)
plt.scatter(survived['Age'], survived['Fare'], alpha=0.6, color='green', 
           label='생존', s=30)
plt.xlabel('나이')
plt.ylabel('요금')
plt.title('나이 vs 요금 (생존 여부별)')
plt.legend()
plt.grid(True, alpha=0.3)

# 서브플롯 3: Seaborn으로 더 예쁘게
plt.subplot(3, 3, 3)
sns.scatterplot(data=df, x='Age', y='Fare', hue='Survived', 
               palette=['red', 'green'], alpha=0.7)
plt.title('나이 vs 요금 (Seaborn)')
plt.grid(True, alpha=0.3)

# 서브플롯 4: 크기로 정보 추가 (가족 크기)
plt.subplot(3, 3, 4)
# 점의 크기로 가족 크기 표현
sns.scatterplot(data=df, x='Age', y='Fare', 
               size='FamilySize',      # 가족 크기로 점 크기 결정
               hue='Pclass',           # 등급으로 색상 결정
               sizes=(30, 200),        # 최소, 최대 점 크기
               alpha=0.7)
plt.title('나이 vs 요금 (등급별, 가족크기별)')
plt.grid(True, alpha=0.3)

# 서브플롯 5: 회귀선이 있는 산점도
plt.subplot(3, 3, 5)
sns.regplot(data=df, x='Age', y='Fare', 
           scatter_kws={'alpha': 0.5},     # 점의 투명도
           line_kws={'color': 'red'})      # 회귀선 색상
plt.title('나이 vs 요금 (회귀선 포함)')
plt.grid(True, alpha=0.3)

# 서브플롯 6: 등급별로 나눈 산점도
plt.subplot(3, 3, 6)
for pclass in sorted(df['Pclass'].unique()):
    class_data = df[df['Pclass'] == pclass]
    plt.scatter(class_data['Age'], class_data['Fare'], 
               alpha=0.6, label=f'{pclass}등급', s=30)

plt.xlabel('나이')
plt.ylabel('요금')
plt.title('등급별 나이 vs 요금')
plt.legend()
plt.grid(True, alpha=0.3)

# 서브플롯 7: 밀도 플롯이 있는 산점도
plt.subplot(3, 3, 7)
sns.jointplot(data=df, x='Age', y='Fare', kind='scatter', alpha=0.6)
plt.suptitle('나이 vs 요금 (밀도 포함)', y=1.02)

# 서브플롯 8: 육각형 빈 플롯 (점이 너무 많을 때 유용)
plt.subplot(3, 3, 8)
plt.hexbin(df['Age'].dropna(), df['Fare'].dropna(), 
          gridsize=20, cmap='Blues', alpha=0.8)
plt.colorbar(label='밀도')
plt.xlabel('나이')
plt.ylabel('요금')
plt.title('나이 vs 요금 (육각형 밀도)')

# 서브플롯 9: 버블 차트 (3차원 정보 표현)
plt.subplot(3, 3, 9)
# 형제자매 수, 배우자/부모 수, 요금을 모두 한 그래프에
bubble_data = df.dropna(subset=['Age', 'Fare', 'SibSp', 'Parch'])
plt.scatter(bubble_data['Age'], bubble_data['Fare'], 
           s=bubble_data['SibSp'] * 50 + 30,  # 형제자매 수로 크기 결정
           c=bubble_data['Parch'],             # 부모자식 수로 색상 결정
           alpha=0.6, cmap='viridis')
plt.colorbar(label='부모/자식 수')
plt.xlabel('나이')
plt.ylabel('요금')
plt.title('버블 차트 (크기=형제자매수)')

plt.tight_layout()
plt.show()

# 상관관계 분석
print("\n📊 변수 간 상관관계 분석:")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

# 수치형 변수들의 상관관계 계산
numeric_vars = ['Age', 'Fare', 'SibSp', 'Parch', 'Survived']
correlation_matrix = df[numeric_vars].corr()

print("📈 상관계수 (피어슨):")
print(correlation_matrix.round(3))

print(f"\n🔍 주요 상관관계:")
print(f"  나이 ↔ 요금: {correlation_matrix.loc['Age', 'Fare']:.3f}")
print(f"  나이 ↔ 생존: {correlation_matrix.loc['Age', 'Survived']:.3f}")
print(f"  요금 ↔ 생존: {correlation_matrix.loc['Fare', 'Survived']:.3f}")
```

#### 2.2 히트맵: 여러 변수의 상관관계 한눈에 보기

```python
# =====================================================
# 2.2 히트맵: 상관관계 시각화
# =====================================================

plt.figure(figsize=(14, 10))

# 서브플롯 1: 기본 상관관계 히트맵
plt.subplot(2, 2, 1)
# 수치형 변수들의 상관관계
numeric_columns = ['Age', 'Fare', 'SibSp', 'Parch', 'Survived']
correlation = df[numeric_columns].corr()

sns.heatmap(correlation, 
           annot=True,              # 수치 표시
           cmap='coolwarm',         # 색상 팔레트 (차갑다=음의상관, 따뜻하다=양의상관)
           center=0,                # 0을 중심으로 색상 배치
           square=True,             # 정사각형 셀
           fmt='.2f',               # 소수점 2자리까지 표시
           cbar_kws={'label': '상관계수'})
plt.title('변수 간 상관관계')

# 서브플롯 2: 마스크를 사용한 히트맵 (중복 제거)
plt.subplot(2, 2, 2)
# 상삼각형만 표시 (대각선 기준 위쪽만)
mask = np.triu(np.ones_like(correlation, dtype=bool))

sns.heatmap(correlation, 
           annot=True, 
           mask=mask,               # 마스크 적용
           cmap='RdBu_r',          # 빨강-파랑 색상
           center=0,
           square=True,
           fmt='.2f')
plt.title('상관관계 (중복 제거)')

# 서브플롯 3: 더 많은 변수를 포함한 히트맵
plt.subplot(2, 2, 3)
# 범주형 변수도 숫자로 변환해서 포함
df_encoded = df.copy()
df_encoded['Sex_encoded'] = df['Sex'].map({'male': 1, 'female': 0})
df_encoded['Embarked_encoded'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

extended_columns = ['Age', 'Fare', 'SibSp', 'Parch', 'Survived', 
                   'Pclass', 'Sex_encoded', 'Embarked_encoded']
extended_correlation = df_encoded[extended_columns].corr()

sns.heatmap(extended_correlation, 
           annot=True, 
           cmap='viridis',
           center=0,
           square=True,
           fmt='.2f',
           xticklabels=['나이', '요금', '형제자매', '부모자식', '생존', 
                       '등급', '성별', '승선항구'],
           yticklabels=['나이', '요금', '형제자매', '부모자식', '생존', 
                       '등급', '성별', '승선항구'])
plt.title('확장 상관관계 분석')
plt.xticks(rotation=45)
plt.yticks(rotation=0)

# 서브플롯 4: 교차표 히트맵 (범주형 × 범주형)
plt.subplot(2, 2, 4)
# 성별과 등급의 교차표
cross_tab = pd.crosstab(df['Sex'], df['Pclass'], normalize='columns') * 100
sns.heatmap(cross_tab, 
           annot=True, 
           fmt='.1f',
           cmap='Oranges',
           cbar_kws={'label': '비율 (%)'},
           xticklabels=['1등급', '2등급', '3등급'],
           yticklabels=['여성', '남성'])
plt.title('등급별 성별 구성 비율')
plt.xlabel('객실 등급')
plt.ylabel('성별')

plt.tight_layout()
plt.show()

# 상관관계 해석 가이드
print("\n📊 상관관계 해석 가이드:")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("📈 상관계수 범위와 의미:")
print("  +0.7 ~ +1.0: 강한 양의 상관관계")
print("  +0.3 ~ +0.7: 보통 양의 상관관계") 
print("  -0.3 ~ +0.3: 약한 상관관계 (거의 무관계)")
print("  -0.7 ~ -0.3: 보통 음의 상관관계")
print("  -1.0 ~ -0.7: 강한 음의 상관관계")

print(f"\n🔍 주요 발견사항:")
strong_correlations = []
for i in range(len(correlation.columns)):
    for j in range(i+1, len(correlation.columns)):
        corr_value = correlation.iloc[i, j]
        if abs(corr_value) > 0.3:  # 절댓값 0.3 이상인 상관관계
            var1, var2 = correlation.columns[i], correlation.columns[j]
            strong_correlations.append((var1, var2, corr_value))

for var1, var2, corr in strong_correlations:
    direction = "양의" if corr > 0 else "음의"
    strength = "강한" if abs(corr) > 0.7 else "보통"
    print(f"  {var1} ↔ {var2}: {corr:.3f} ({strength} {direction} 상관관계)")

---

## 상세 예제 / 미니 프로젝트

### 타이타닉 데이터 시각화 종합 대시보드 구축

이제 배운 시각화 기법들을 종합하여 **완전한 분석 대시보드**를 만들어보겠습니다. 실제 데이터 분석가가 경영진에게 보고할 수 있는 수준의 전문적인 시각화를 구현해보겠습니다.

#### 프로젝트 목표
- 타이타닉 승객 데이터의 핵심 인사이트를 한눈에 파악할 수 있는 대시보드 제작
- 다양한 시각화 기법을 조합하여 스토리텔링 구현
- 비즈니스 관점에서 의미 있는 발견사항 도출

#### Step 1: 전문적인 대시보드 레이아웃 설계
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = ['Malgun Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("📊 타이타닉 데이터 분석 및 시각화")
print("=" * 50)

# 1. 데이터 가져오기
print("1. 데이터 로딩 중...")
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print(f"✅ 데이터 로딩 완료! 총 {len(df)}개 행, {len(df.columns)}개 열")
print("\n기본 정보:")
print(df.info())
print("\n처음 5행:")
print(df.head())

# 2. 데이터 전처리
print("\n2. 데이터 전처리...")
# 결측치 처리
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop('Cabin', axis=1, inplace=True)

# 새로운 특성 생성
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# 연령대 그룹 생성
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], 
                       labels=['어린이', '청소년', '청년', '중년', '노년'])

print("✅ 전처리 완료!")

# 3. 다변량 시각화
print("\n3. 다변량 시각화 생성 중...")

# 3-1. 상관관계 히트맵
plt.figure(figsize=(12, 8))
numeric_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone']
correlation_matrix = df[numeric_cols].corr()

plt.subplot(2, 2, 1)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5)
plt.title('변수간 상관관계 히트맵', fontsize=14, fontweight='bold')

# 3-2. Pairplot (산점도 매트릭스)
plt.subplot(2, 2, 2)
key_vars = ['Age', 'Fare', 'FamilySize', 'Survived']
df_sample = df[key_vars].sample(n=min(500, len(df)))  # 샘플링으로 성능 향상

for i, var1 in enumerate(['Age', 'Fare']):
    for j, var2 in enumerate(['FamilySize']):
        plt.scatter(df_sample[var1], df_sample[var2], 
                   c=df_sample['Survived'], cmap='viridis', alpha=0.6, s=30)
        
plt.xlabel('나이', fontsize=12)
plt.ylabel('가족 크기', fontsize=12)
plt.title('나이 vs 가족크기 (생존여부별)', fontsize=14, fontweight='bold')
plt.colorbar(label='생존 여부')

# 3-3. 다차원 분석 (성별, 등급, 생존)
plt.subplot(2, 2, 3)
survival_by_class_gender = df.groupby(['Pclass', 'Sex'])['Survived'].mean().unstack()
survival_by_class_gender.plot(kind='bar', ax=plt.gca(), width=0.8)
plt.title('등급별, 성별 생존율', fontsize=14, fontweight='bold')
plt.xlabel('등급', fontsize=12)
plt.ylabel('생존율', fontsize=12)
plt.legend(['여성', '남성'], loc='upper right')
plt.xticks(rotation=0)

# 3-4. 연령대별 분석
plt.subplot(2, 2, 4)
age_survival = df.groupby('AgeGroup')['Survived'].agg(['mean', 'count'])
age_survival['mean'].plot(kind='bar', ax=plt.gca(), color='skyblue', alpha=0.7)
plt.title('연령대별 생존율', fontsize=14, fontweight='bold')
plt.xlabel('연령대', fontsize=12)
plt.ylabel('생존율', fontsize=12)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# 4. 고급 다변량 시각화
print("\n4. 고급 다변량 시각화...")

plt.figure(figsize=(15, 10))

# 4-1. 3D 산점도 스타일 시각화
plt.subplot(2, 3, 1)
scatter = plt.scatter(df['Age'], df['Fare'], 
                     c=df['Survived'], s=df['FamilySize']*20, 
                     alpha=0.6, cmap='RdYlBu')
plt.xlabel('나이', fontsize=12)
plt.ylabel('운임료', fontsize=12)
plt.title('나이-운임료-가족크기-생존 관계', fontsize=12, fontweight='bold')
plt.colorbar(scatter, label='생존 여부')

# 4-2. 박스플롯 - 다중 비교
plt.subplot(2, 3, 2)
df_melted = df.melt(id_vars=['Survived'], 
                   value_vars=['Age', 'Fare'], 
                   var_name='변수', value_name='값')
sns.boxplot(data=df_melted, x='변수', y='값', hue='Survived', ax=plt.gca())
plt.title('생존 여부별 나이/운임료 분포', fontsize=12, fontweight='bold')
plt.yscale('log')

# 4-3. 바이올린 플롯
plt.subplot(2, 3, 3)
sns.violinplot(data=df, x='Pclass', y='Age', hue='Survived', ax=plt.gca())
plt.title('등급별 나이 분포 (생존여부)', fontsize=12, fontweight='bold')

# 4-4. 복합 막대 그래프
plt.subplot(2, 3, 4)
crosstab = pd.crosstab(df['Embarked'], df['Survived'], normalize='index')
crosstab.plot(kind='bar', ax=plt.gca(), stacked=True, 
              color=['lightcoral', 'lightgreen'])
plt.title('출발항구별 생존율', fontsize=12, fontweight='bold')
plt.xlabel('출발 항구', fontsize=12)
plt.ylabel('비율', fontsize=12)
plt.legend(['사망', '생존'])
plt.xticks(rotation=0)

# 4-5. 히트맵 - 교차표
plt.subplot(2, 3, 5)
pivot_table = df.pivot_table(values='Survived', 
                            index='AgeGroup', 
                            columns='Pclass', 
                            aggfunc='mean')
sns.heatmap(pivot_table, annot=True, cmap='RdYlGn', 
            fmt='.2f', ax=plt.gca())
plt.title('연령대-등급별 생존율', fontsize=12, fontweight='bold')

# 4-6. 분포 비교
plt.subplot(2, 3, 6)
for cls in [1, 2, 3]:
    subset = df[df['Pclass'] == cls]
    plt.hist(subset['Age'], alpha=0.7, label=f'{cls}등급', bins=20)
plt.xlabel('나이', fontsize=12)
plt.ylabel('빈도', fontsize=12)
plt.title('등급별 나이 분포', fontsize=12, fontweight='bold')
plt.legend()

plt.tight_layout()
plt.show()

# 5. 인터렉티브 시각화
print("\n5. 인터렉티브 시각화 생성 중...")

# 5-1. 인터렉티브 3D 산점도
fig1 = px.scatter_3d(df, x='Age', y='Fare', z='FamilySize', 
                     color='Survived', size='Pclass',
                     hover_data=['Name', 'Sex', 'Embarked'],
                     title='3D 산점도: 나이-운임료-가족크기 (생존여부별)',
                     labels={'Age': '나이', 'Fare': '운임료', 
                            'FamilySize': '가족 크기', 'Survived': '생존 여부'})
fig1.show()

# 5-2. 인터렉티브 히트맵
survival_rate = df.groupby(['Pclass', 'Sex', 'AgeGroup'])['Survived'].mean().reset_index()
fig2 = px.density_heatmap(survival_rate, x='Pclass', y='AgeGroup', 
                         z='Survived', facet_col='Sex',
                         title='등급-연령대-성별 생존율 히트맵',
                         labels={'Pclass': '등급', 'AgeGroup': '연령대', 
                                'Survived': '생존율', 'Sex': '성별'})
fig2.show()

# 5-3. 인터렉티브 병렬 좌표 플롯
fig3 = px.parallel_coordinates(df.sample(200), 
                              dimensions=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare'],
                              color='Survived',
                              title='병렬 좌표 플롯 - 다변수 관계 분석',
                              labels={'Pclass': '등급', 'Age': '나이', 
                                     'SibSp': '형제자매', 'Parch': '부모자녀',
                                     'Fare': '운임료', 'Survived': '생존'})
fig3.show()

# 5-4. 인터렉티브 선버스트 차트
fig4 = px.sunburst(df, path=['Sex', 'Pclass', 'AgeGroup'], 
                   values='Fare', color='Survived',
                   title='계층적 데이터 분석 - 성별→등급→연령대',
                   color_continuous_scale='RdYlBu')
fig4.show()

# 5-5. 인터렉티브 박스플롯
fig5 = px.box(df, x='Pclass', y='Fare', color='Survived',
              facet_col='Sex', hover_data=['Age', 'FamilySize'],
              title='등급별 운임료 분포 (성별, 생존여부)',
              labels={'Pclass': '등급', 'Fare': '운임료', 
                     'Survived': '생존 여부', 'Sex': '성별'})
fig5.show()

# 5-6. 대시보드 스타일 복합 시각화
fig6 = make_subplots(
    rows=2, cols=2,
    subplot_titles=('연령대별 생존율', '등급별 성별 분포', '운임료 분포', '가족크기별 생존'),
    specs=[[{'type': 'bar'}, {'type': 'pie'}],
           [{'type': 'histogram'}, {'type': 'scatter'}]]
)

# 서브플롯 1: 연령대별 생존율
age_survival = df.groupby('AgeGroup')['Survived'].mean()
fig6.add_trace(go.Bar(x=age_survival.index, y=age_survival.values, 
                     name='생존율', marker_color='lightblue'), row=1, col=1)

# 서브플롯 2: 성별 분포
gender_counts = df['Sex'].value_counts()
fig6.add_trace(go.Pie(labels=gender_counts.index, values=gender_counts.values,
                     name='성별 분포'), row=1, col=2)

# 서브플롯 3: 운임료 분포
fig6.add_trace(go.Histogram(x=df['Fare'], nbinsx=30, name='운임료',
                           marker_color='lightgreen'), row=2, col=1)

# 서브플롯 4: 가족크기별 생존
fig6.add_trace(go.Scatter(x=df['FamilySize'], y=df['Survived'], 
                         mode='markers', name='가족크기-생존',
                         marker=dict(color=df['Age'], colorscale='viridis', size=8)),
              row=2, col=2)

fig6.update_layout(height=700, title_text="타이타닉 데이터 종합 대시보드", 
                  showlegend=False)
fig6.show()

print("\n" + "="*50)
print("📊 시각화 완료!")
print("생성된 시각화:")
print("- 상관관계 히트맵")
print("- 다변량 산점도")
print("- 등급별 성별 생존율")
print("- 연령대별 분석")
print("- 3D 인터렉티브 산점도")
print("- 인터렉티브 히트맵") 
print("- 병렬 좌표 플롯")
print("- 선버스트 차트")
print("- 인터렉티브 박스플롯")
print("- 종합 대시보드")
print("="*50)

# 6. 주요 인사이트 요약
print("\n📈 주요 분석 결과:")
print("1. 여성의 생존율이 남성보다 현저히 높음")
print("2. 1등급 승객의 생존율이 가장 높음") 
print("3. 나이가 어릴수록 생존율이 높은 경향")
print("4. 가족 규모와 생존율 간의 복잡한 관계")
print("5. 운임료가 높을수록 생존율이 높은 경향")

```

---

## 직접 해보기 / 연습 문제

### 연습 문제 1: 기본 시각화 (난이도: ⭐)

**목표**: 기본적인 히스토그램과 막대그래프를 그려보세요.

**단계별 가이드**:
1. 타이타닉 데이터의 Age 컬럼으로 히스토그램을 그리세요
2. Sex 컬럼으로 막대그래프를 그리세요
3. 각 그래프에 제목과 축 레이블을 추가하세요

```python
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 로드
df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

# TODO: 1. Age 히스토그램 그리기
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
# 여기에 히스토그램 코드 작성
plt.hist(df['Age'].dropna(), bins=20, alpha=0.7, color='skyblue')
plt.title('승객 나이 분포')
plt.xlabel('나이')
plt.ylabel('빈도')

# TODO: 2. Sex 막대그래프 그리기
plt.subplot(1, 2, 2)
# 여기에 막대그래프 코드 작성
sex_counts = df['Sex'].value_counts()
plt.bar(sex_counts.index, sex_counts.values, color=['lightblue', 'lightpink'])
plt.title('성별 분포')
plt.xlabel('성별')
plt.ylabel('승객 수')

plt.tight_layout()
plt.show()
```

**💡 힌트**: 
- `plt.hist()`를 사용하여 히스토그램을 그릴 수 있습니다
- `value_counts()`로 범주별 개수를 계산한 후 `plt.bar()`로 막대그래프를 그릴 수 있습니다
- 결측치는 `dropna()`로 제거해야 합니다

**✅ 예상 결과**:
- 나이 분포는 20-40세 구간에 가장 많은 승객이 분포
- 남성 승객이 여성 승객보다 약 1.8배 많음

---

### 연습 문제 2: 그룹별 비교 시각화 (난이도: ⭐⭐)

**목표**: 그룹별 차이를 시각화하여 패턴을 발견해보세요.

**단계별 가이드**:
1. 등급별(Pclass) 요금(Fare) 분포를 박스플롯으로 그리세요
2. 성별과 생존 여부의 교차표를 시각화하세요
3. 각 그래프에서 발견한 패턴을 해석해보세요

```python
# TODO: 1. 등급별 요금 박스플롯
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
# Seaborn을 사용한 박스플롯
import seaborn as sns
sns.boxplot(data=df, x='Pclass', y='Fare')
plt.title('등급별 요금 분포')
plt.xlabel('객실 등급')
plt.ylabel('요금')

# TODO: 2. 성별 × 생존 여부 교차표 시각화
plt.subplot(1, 3, 2)
cross_tab = pd.crosstab(df['Sex'], df['Survived'])
cross_tab.plot(kind='bar', ax=plt.gca(), color=['lightcoral', 'lightgreen'])
plt.title('성별 × 생존 여부')
plt.xlabel('성별')
plt.ylabel('승객 수')
plt.legend(['사망', '생존'])
plt.xticks(rotation=0)

# TODO: 3. 생존률을 퍼센트로 표시
plt.subplot(1, 3, 3)
survival_by_sex = df.groupby('Sex')['Survived'].mean() * 100
bars = plt.bar(survival_by_sex.index, survival_by_sex.values, 
              color=['lightblue', 'lightpink'])

# 막대 위에 퍼센트 표시
for bar, rate in zip(bars, survival_by_sex.values):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.title('성별 생존률')
plt.xlabel('성별')
plt.ylabel('생존률 (%)')
plt.ylim(0, 100)

plt.tight_layout()
plt.show()

# TODO: 발견한 패턴 해석해보기
print("🔍 발견한 패턴:")
print("1. 등급별 요금 차이:")
for pclass in sorted(df['Pclass'].unique()):
    median_fare = df[df['Pclass'] == pclass]['Fare'].median()
    print(f"   {pclass}등급 중앙값: £{median_fare:.1f}")

print("\n2. 성별 생존률:")
for sex in df['Sex'].unique():
    survival_rate = df[df['Sex'] == sex]['Survived'].mean() * 100
    print(f"   {sex}: {survival_rate:.1f}%")
```

**💡 힌트**: 
- `seaborn.boxplot()`을 사용하면 그룹별 박스플롯을 쉽게 그릴 수 있습니다
- `pd.crosstab()`으로 교차표를 만들고 `.plot(kind='bar')`로 시각화할 수 있습니다
- `groupby()`와 `mean()`을 조합해서 그룹별 평균을 계산할 수 있습니다

**✅ 정답 해설**:
```python
"""
예상 패턴:
1. 등급별 요금 차이가 매우 큼 (1등급 >> 2등급 > 3등급)
2. 여성의 생존률이 남성보다 훨씬 높음 (약 74% vs 19%)
3. 1등급 승객의 요금 편차가 가장 큼 (스위트룸 vs 일반실)
"""
```

---

### 연습 문제 3: 상관관계 분석 (난이도: ⭐⭐⭐)

**목표**: 여러 변수 간의 관계를 히트맵과 산점도로 분석해보세요.

**단계별 가이드**:
1. 수치형 변수들의 상관관계 히트맵을 그리세요
2. Age와 Fare의 산점도를 그리고 생존 여부로 색상을 구분하세요
3. 가족 크기가 생존에 미치는 영향을 분석해보세요

```python
# TODO: 1. 상관관계 히트맵
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
# 수치형 변수 선택
numeric_vars = ['Age', 'Fare', 'SibSp', 'Parch', 'Survived']
correlation = df[numeric_vars].corr()

# 히트맵 그리기
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0,
           square=True, fmt='.2f')
plt.title('변수 간 상관관계')

# TODO: 2. Age vs Fare 산점도 (생존 여부별 색상)
plt.subplot(1, 3, 2)
survived_passengers = df[df['Survived'] == 1]
died_passengers = df[df['Survived'] == 0]

plt.scatter(died_passengers['Age'], died_passengers['Fare'], 
           alpha=0.6, color='red', label='사망', s=30)
plt.scatter(survived_passengers['Age'], survived_passengers['Fare'], 
           alpha=0.6, color='green', label='생존', s=30)

plt.xlabel('나이')
plt.ylabel('요금')
plt.title('나이 vs 요금 (생존 여부별)')
plt.legend()
plt.grid(True, alpha=0.3)

# TODO: 3. 가족 크기별 생존률
plt.subplot(1, 3, 3)
# 가족 크기 계산 (본인 + 형제자매 + 부모자식)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# 가족 크기별 생존률 계산
family_survival = df.groupby('FamilySize')['Survived'].agg(['count', 'mean'])
family_survival['survival_rate'] = family_survival['mean'] * 100

# 충분한 샘플이 있는 경우만 표시 (5명 이상)
family_survival_filtered = family_survival[family_survival['count'] >= 5]

bars = plt.bar(family_survival_filtered.index, family_survival_filtered['survival_rate'],
              color='orange', alpha=0.7)

# 막대 위에 정보 표시
for idx, bar in zip(family_survival_filtered.index, bars):
    height = bar.get_height()
    count = family_survival.loc[idx, 'count']
    plt.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{height:.1f}%\n({count}명)', ha='center', va='bottom', fontsize=9)

plt.xlabel('가족 크기')
plt.ylabel('생존률 (%)')
plt.title('가족 크기별 생존률')
plt.ylim(0, 100)

plt.tight_layout()
plt.show()

# TODO: 상관관계 해석
print("📊 상관관계 분석 결과:")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

strong_correlations = []
for i in range(len(correlation.columns)):
    for j in range(i+1, len(correlation.columns)):
        corr_value = correlation.iloc[i, j]
        if abs(corr_value) > 0.1:  # 절댓값 0.1 이상
            var1, var2 = correlation.columns[i], correlation.columns[j]
            strong_correlations.append((var1, var2, corr_value))

print("🔍 주요 상관관계 (절댓값 0.1 이상):")
for var1, var2, corr in sorted(strong_correlations, key=lambda x: abs(x[2]), reverse=True):
    direction = "양의" if corr > 0 else "음의"
    strength = "강한" if abs(corr) > 0.5 else "보통" if abs(corr) > 0.3 else "약한"
    print(f"  {var1} ↔ {var2}: {corr:.3f} ({strength} {direction} 상관관계)")

print(f"\n💡 가족 크기 인사이트:")
optimal_family_size = family_survival_filtered['survival_rate'].idxmax()
optimal_survival_rate = family_survival_filtered.loc[optimal_family_size, 'survival_rate']
print(f"  최적 가족 크기: {optimal_family_size}명 (생존률 {optimal_survival_rate:.1f}%)")
```

**💡 힌트**: 
- `sns.heatmap()`에서 `annot=True`로 수치를 표시할 수 있습니다
- 산점도에서 두 그룹을 다른 색상으로 표시하려면 각각 따로 그려야 합니다
- `groupby()`와 `agg(['count', 'mean'])`으로 그룹별 통계를 한 번에 계산할 수 있습니다

**✅ 정답 해설**:
```python
"""
예상 발견사항:
1. 상관관계:
   - Fare ↔ Survived: 양의 상관관계 (요금이 높을수록 생존률 높음)
   - Age ↔ Survived: 약한 음의 상관관계 (나이가 많을수록 생존률 낮음)

2. 산점도:
   - 고요금 구간에서 생존자 비율이 높음
   - 어린이(낮은 나이)에서 생존자가 많이 보임

3. 가족 크기:
   - 혼자 여행한 승객보다 2-4명 가족의 생존률이 높음
   - 너무 큰 가족(5명+)은 오히려 생존률이 낮아짐
"""
```

---

### 연습 문제 4: 종합 시각화 프로젝트 (난이도: ⭐⭐⭐⭐)

**목표**: 지금까지 배운 모든 기법을 사용하여 완전한 분석 보고서를 만들어보세요.

**미션**: 타이타닉 데이터 분석가가 되어 "생존에 영향을 미친 주요 요인" 보고서를 작성하세요.

**요구사항**:
1. 4×3 서브플롯으로 구성된 종합 대시보드 생성
2. 최소 6가지 이상의 다른 시각화 기법 사용
3. 각 그래프마다 명확한 제목과 인사이트 포함
4. 전체적인 스토리라인이 있어야 함

```python
def create_survival_analysis_report(df):
    """
    생존 분석 종합 보고서 생성 함수
    """
    
    # TODO: 여러분이 직접 구현해보세요!
    
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle('🚢 타이타닉 생존 분석 종합 보고서', 
                fontsize=20, fontweight='bold')
    
    # 1행: 기본 현황
    # TODO: 1-1. 전체 생존률 파이차트
    plt.subplot(4, 3, 1)
    # 여기에 코드 작성
    
    # TODO: 1-2. 등급별 승객 분포
    plt.subplot(4, 3, 2)
    # 여기에 코드 작성
    
    # TODO: 1-3. 성별 분포
    plt.subplot(4, 3, 3)
    # 여기에 코드 작성
    
    # 2행: 생존 요인 분석
    # TODO: 2-1. 등급별 생존률
    plt.subplot(4, 3, 4)
    # 여기에 코드 작성
    
    # TODO: 2-2. 성별 생존률
    plt.subplot(4, 3, 5)
    # 여기에 코드 작성
    
    # TODO: 2-3. 나이대별 생존률
    plt.subplot(4, 3, 6)
    # 여기에 코드 작성
    
    # 3행: 분포 분석
    # TODO: 3-1. 나이 분포 히스토그램
    plt.subplot(4, 3, 7)
    # 여기에 코드 작성
    
    # TODO: 3-2. 요금 박스플롯
    plt.subplot(4, 3, 8)
    # 여기에 코드 작성
    
    # TODO: 3-3. 가족 크기별 생존률
    plt.subplot(4, 3, 9)
    # 여기에 코드 작성
    
    # 4행: 고급 분석
    # TODO: 4-1. 상관관계 히트맵
    plt.subplot(4, 3, 10)
    # 여기에 코드 작성
    
    # TODO: 4-2. 나이 vs 요금 산점도
    plt.subplot(4, 3, 11)
    # 여기에 코드 작성
    
    # TODO: 4-3. 핵심 인사이트 요약
    plt.subplot(4, 3, 12)
    # 텍스트로 주요 발견사항 정리
    
    plt.tight_layout()
    plt.show()
    
    # TODO: 결론 및 제언 작성
    print("\n📋 생존 분석 결론:")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("🎯 주요 발견사항:")
    print("  1. 성별이 생존에 가장 큰 영향 (여성 우선 구조)")
    print("  2. 객실 등급이 높을수록 생존률 증가 (사회경제적 지위)")
    print("  3. 어린이의 생존률이 상대적으로 높음 (어린이 우선)")
    print("  4. 중간 크기 가족의 생존률이 가장 높음")
    print("\n💡 시사점:")
    print("  - 재난 상황에서 사회적 규범과 경제적 지위가 생존에 영향")
    print("  - 효과적인 대피 계획 수립 시 취약계층 고려 필요")

# 종합 보고서 생성
create_survival_analysis_report(df)
```

**💡 평가 기준**:
- **완성도**: 모든 서브플롯이 의미 있는 내용으로 채워져 있는가?
- **다양성**: 다양한 시각화 기법을 적절히 활용했는가?
- **인사이트**: 각 그래프에서 의미 있는 패턴을 발견했는가?
- **스토리텔링**: 전체적인 흐름이 논리적인가?
- **시각적 완성도**: 그래프가 보기 좋고 이해하기 쉬운가?

**🏆 도전 과제**:
- 색상 테마를 일관성 있게 적용해보세요
- 각 그래프에 주석(annotation)을 추가해보세요
- 인터랙티브 요소를 추가해보세요 (Plotly 사용)

---

## 요약 / 핵심 정리

이번 Part에서 배운 핵심 내용을 정리하면 다음과 같습니다:

### 🎯 주요 학습 성과

**1. 시각화 도구 완전 마스터**
- ✅ **Matplotlib**: 정밀한 컨트롤과 세부 커스터마이징
- ✅ **Seaborn**: 아름답고 통계적인 시각화
- ✅ **환경 설정**: 한글 폰트, 색상 팔레트, 스타일 설정 완벽 정복
- ✅ **최적화**: 메모리 효율적 시각화와 고품질 저장 방법

**2. 체계적인 시각화 선택 전략**
- ✅ **목적별 선택**: 탐색 vs 설명 vs 설득에 따른 시각화 전략
- ✅ **데이터 타입별**: 연속형, 이산형, 범주형에 최적화된 시각화
- ✅ **청중별 맞춤**: 경영진, 분석팀, 일반 대중을 위한 차별화된 접근

**3. 단변량 시각화 완전 정복**
- ✅ **히스토그램**: 분포 모양, 중심값, 퍼진 정도 한눈에 파악
- ✅ **박스플롯**: 5가지 핵심 통계량과 이상치 탐지
- ✅ **막대그래프/파이차트**: 범주형 데이터의 효과적 시각화

**4. 다변량 시각화로 관계 탐색**
- ✅ **산점도**: 두 연속형 변수의 상관관계와 패턴 발견
- ✅ **히트맵**: 다중 변수 상관관계 한눈에 비교
- ✅ **그룹 비교**: 범주별 차이를 명확히 드러내는 시각화

**5. 전문적인 대시보드 구축**
- ✅ **스토리텔링**: 데이터를 논리적 흐름의 시각적 이야기로 구성
- ✅ **인터랙티브**: Plotly를 활용한 동적 시각화
- ✅ **최적화**: 성능과 품질을 모두 고려한 프로덕션급 시각화

### 💡 핵심 Python 기술

**Matplotlib 핵심 패턴:**
```python
# 기본 구조
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y, label='데이터')
ax.set_title('제목', fontweight='bold')
ax.set_xlabel('x축 라벨')
ax.set_ylabel('y축 라벨')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

**Seaborn 핵심 패턴:**
```python
# 통계적 시각화
sns.scatterplot(data=df, x='var1', y='var2', hue='category')
sns.boxplot(data=df, x='category', y='value')
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
```

**색상과 스타일:**
```python
# 색상 팔레트
sns.set_palette("husl")
colors = {'category1': '#FF6B6B', 'category2': '#4ECDC4'}

# 스타일 설정
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12
```

### 🔍 실무 적용 가이드라인

**1. 시각화 프로젝트 시작할 때 체크리스트**
- [ ] 분석 목적 명확화 (탐색/설명/설득)
- [ ] 타겟 청중 파악 (경영진/분석팀/일반)
- [ ] 데이터 타입별 최적 시각화 선택
- [ ] 색상과 스타일 가이드 설정
- [ ] 스토리라인 구성

**2. 효과적인 시각화 7원칙 적용**
- **명확성**: 한 번에 하나의 메시지만 전달
- **정확성**: 데이터 왜곡 없이 정직한 표현
- **효율성**: 인지적 부담 최소화
- **색상 의미**: 색상이 정보를 전달하도록
- **일관성**: 같은 요소는 같은 방식으로
- **스토리텔링**: 논리적 흐름 구성
- **청중 고려**: 누구를 위한 그래프인지 고려

**3. 흔한 실수 방지하기**
- 파이차트 남용 (7개 이상 조각 지양)
- 3D 그래프 과용 (깊이감으로 인한 오해)
- 무지개색 남용 (정보 혼란 야기)
- 의미 없는 애니메이션 (정보 전달 방해)

### 🚀 다음 Part 예고

**Part 3: 기술 통계와 데이터 요약**에서는 시각화로 발견한 패턴들을 **정확한 수치**로 뒷받침하는 방법을 배우겠습니다:

- **📊 기술 통계**: 평균, 중앙값, 분산을 넘어선 고급 통계량
- **📈 분포 분석**: 정규성 검정, 왜도, 첨도 등 분포의 특성 파악  
- **🔗 관계 분석**: 상관계수, 공분산 등 변수 간 관계 정량화
- **📋 요약 테이블**: 그룹별 통계량을 체계적으로 정리하는 방법
- **🎯 실무 활용**: 비즈니스 리포트에 적합한 통계 요약 기법

시각화로 "보이는" 패턴을 이제 "증명할 수 있는" 수치로 만들어보겠습니다!

---

**🎓 학습 완료 체크리스트**
- [ ] Matplotlib과 Seaborn의 차이점을 이해하고 상황에 맞게 선택할 수 있다
- [ ] 데이터 타입별로 적절한 시각화 방법을 선택할 수 있다
- [ ] 히스토그램, 박스플롯, 산점도, 히트맵을 능숙하게 그릴 수 있다
- [ ] 그래프의 색상, 크기, 스타일을 의미 있게 설정할 수 있다
- [ ] 여러 시각화를 조합하여 스토리가 있는 대시보드를 만들 수 있다
- [ ] 연습 문제 4개를 모두 완료했다

모든 항목이 체크되었다면 Part 3으로 진행하세요! 🎉

---

**참고 자료**
- Matplotlib 공식 문서: https://matplotlib.org/
- Seaborn 공식 문서: https://seaborn.pydata.org/
- Plotly 공식 문서: https://plotly.com/python/
- 색상 이론: ColorBrewer (https://colorbrewer2.org/)
- 시각화 원칙: "The Visual Display of Quantitative Information" by Edward Tufte
```

---
