# 2장 Part 1: 데이터 구조와 특성 이해하기 (개선 버전)

## 학습 목표
이번 Part를 학습한 후 여러분은 다음을 할 수 있게 됩니다:

- 데이터의 기본 구조와 타입을 체계적으로 분석할 수 있다
- 결측치와 이상치를 식별하고 그 원인을 추론할 수 있다
- 데이터 분포의 기본 개념을 이해하고 적절한 시각화로 표현할 수 있다
- Python과 Pandas를 사용하여 데이터의 기본 정보를 단계별로 추출할 수 있다
- 데이터 품질 문제를 조기에 발견하고 해결 방향을 제시할 수 있다

## 이번 Part 미리보기

데이터 분석의 첫 걸음은 **"이 데이터가 무엇인가?"**라는 질문에서 시작됩니다. 

실제 세계의 데이터는 깔끔한 교과서와 달리 복잡하고 불완전합니다. 누락된 정보, 잘못 입력된 값, 예상치 못한 패턴들이 숨어있죠. 마치 새로운 도시에 도착했을 때 지도를 먼저 살펴보듯이, 데이터를 분석하기 전에 전체적인 구조와 특성을 파악하는 것이 중요합니다.

이번 Part에서는 **탐색적 데이터 분석(Exploratory Data Analysis, EDA)**의 첫 단계로, 데이터와 친해지는 방법을 배우겠습니다. 

### 무엇을 배우나요?
- 📊 **데이터 타입별 특성**: 숫자인지, 범주인지, 시간인지에 따른 분석 전략
- 🔍 **결측치와 이상치**: 빠진 데이터와 튀는 값들이 주는 메시지 읽기
- 📈 **기초 통계와 시각화**: 숫자 너머의 스토리 발견하기
- ⚡ **Python 도구 활용**: Pandas로 효율적인 데이터 탐색하기

### 실제 사례로 배우기
타이타닉 데이터셋을 통해 실제 데이터 분석가가 되어보겠습니다. 100년 전 비극적인 해상사고 데이터에서 생존 패턴을 찾아보고, 데이터의 품질 문제들을 함께 해결해보겠습니다.

---

## 실습 환경 준비

### 1. 필요한 라이브러리 설치

데이터 분석을 시작하기 전에 필요한 Python 라이브러리들을 설치해야 합니다.

**Windows 명령 프롬프트나 아나콘다 프롬프트에서:**
```bash
# 기본 데이터 분석 라이브러리 설치
pip install pandas numpy matplotlib seaborn

# 선택사항: 추가 시각화 도구
pip install plotly jupyter
```

**Google Colab을 사용하는 경우:**
```python
# Colab에는 대부분의 라이브러리가 이미 설치되어 있습니다
# 최신 버전으로 업데이트하려면:
!pip install --upgrade pandas matplotlib seaborn

# 나눔 폰트 설치 (예시)
!sudo apt-get install -y fonts-nanum
!sudo fc-cache -fv

# 런타임을 다시 시작해야 폰트가 적용될 수 있습니다.
print("나눔 폰트 설치 완료. 런타임을 다시 시작해주세요.")
```

### 2. 한글 폰트 설정 (중요!)

Python의 Matplotlib는 기본적으로 한글을 지원하지 않습니다. 한글이 깨져 보이는 것을 방지하기 위해 다음 설정을 해주세요.

**시스템에 설치된 한글 폰트 확인(Google Colab)**
Colab에서 사용 가능한 한글 폰트를 확인하고 설정하는 방법은 다음과 같습니다.
먼저, 시스템에 설치된 폰트 목록을 확인합니다.

```python
import matplotlib.font_manager as fm

# 시스템에 설치된 폰트 목록 가져오기
font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')

# 한글 폰트가 포함된 폰트 이름 출력
fonts = [f.split('/')[-1] for f in font_list if 'Nanum' in f or 'Malgun' in f] # Nanum 또는 Malgun 폰트 필터링
print("사용 가능한 한글 폰트:")
for font in fonts:
    print(font)
```
만약 원하는 한글 폰트가 설치되어 있지 않다면, Colab 환경에 직접 설치할 수 있습니다.
나눔 폰트를 설치하는 방법은 다음과 같습니다.
```python
# 나눔 폰트 설치 (예시)
!sudo apt-get install -y fonts-nanum
!sudo fc-cache -fv

# 런타임을 다시 시작해야 폰트가 적용될 수 있습니다.
print("나눔 폰트 설치 완료. 런타임을 다시 시작해주세요.")
```

**시스템에 설치된 한글 폰트 확인(로컬 환경)**
```python
from matplotlib import font_manager

# 시스템에 설치된 모든 TTF 폰트 경로 확인
fonts = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')

# 폰트 이름과 경로 출력
for font in fonts:
    try:
        font_name = font_manager.FontProperties(fname=font).get_name()
        if 'Malgun' in font_name: # 또는 'NanumGothic', 'AppleGothic' 등
            print(font_name, '->', font)
    except Exception as e:
        print(f"Error reading font: {font}, {e}")
```

**Windows 사용자:**
```python
import matplotlib.pyplot as plt
import platform

# Windows에서 한글 폰트 설정
if platform.system() == 'Windows':
    # Malgun Gothic이 없는 경우 대체 폰트들
    font_list = ['Malgun Gothic', 'DejaVu Sans', 'Arial Unicode MS']
    
    for font in font_list:
        try:
            plt.rcParams['font.family'] = font
            plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
            print(f"폰트 설정 완료: {font}")
            break
        except:
            continue
```

**macOS 사용자:**
```python
# macOS에서 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
```


**운영체제별 추천 한글 폰트**
| 운영체제 | 추천 한글 폰트            | 설치 여부    |
| ------- | ----------------------  | --------    |
| Windows | Malgun Gothic (맑은 고딕)| 기본 설치됨   |
| macOS   | AppleGothic, Sandoll 등 | 기본 설치됨   |



plt.rc와 plt.rcParams는 Matplotlib에서 그래프 설정을 변경하는 데 사용되지만, 사용 방법과 목적에 차이가 있습니다. plt.rc는 함수 형태로, 한 번에 여러 설정을 변경할 때 사용됩니다. 반면, plt.rcParams는 딕셔너리 형태로, 특정 설정을 더 세부적으로 변경하거나, 설정 파일에 저장하여 모든 노트북에 공통 적용할 때 사용됩니다. 

**plt.rc (함수 형태):**
- 한 번에 여러 설정 (예: 폰트, 색상, 크기 등)을 변경할 때 사용합니다.
- plt.rc('font', family='Malgun Gothic', size=10) 와 같이 사용합니다. 
- 새로운 설정을 적용하려면 plt.rc를 다시 호출해야 합니다.

**plt.rcParams (딕셔너리 형태):**
- plt.rcParams는 딕셔너리 형태로, Matplotlib의 전역 설정 (Runtime Configuration Parameters)을 저장합니다. 
- 특정 설정 (예: 폰트 크기, 축 색상 등)을 개별적으로 변경할 때 사용합니다.
plt.rcParams['axes.facecolor'] = 'lightgray' 와 같이 사용합니다. 
- 변경 사항은 해당 그래프나 세션에만 적용됩니다. 
- 설정 파일에 저장하여 모든 노트북에 공통 적용할 수도 있습니다. 

**추가 설명:**
- plt.rc는 주로 그래프를 그리기 전에 스타일을 설정할 때 사용됩니다. 
plt.rcParams는 그래프를 그리는 도중에 설정을 변경하거나, 특정 설정 값을 확인하는 데 사용될 수 있습니다. 
- Matplotlib에서 한글 폰트를 사용하려면, 먼저 폰트를 시스템에 설치하고, plt.rc나 plt.rcParams를 사용하여 폰트 설정을 변경해야 합니다. 
- plt.show() 함수는 그래프를 화면에 표시하는 기능을 하지만, Jupyter나 IPython 환경에서는 자동 표시되므로 호출할 필요가 없습니다. 
### 3. 개발 환경 선택 가이드

**초보자 추천 순서:**

1. **Google Colab** (웹 브라우저)
   - 장점: 설치 불필요, 어디서나 접근 가능
   - 단점: 인터넷 연결 필요
   - 접속: colab.research.google.com

2. **Jupyter Notebook** (로컬 설치)
   - 장점: 오프라인 사용, 빠른 실행
   - 설치: `pip install jupyter` 후 `jupyter notebook` 실행

3. **VS Code** (코드 에디터)
   - 장점: 다양한 확장 기능, 전문적인 개발 환경
   - Python 확장팩 설치 필요


### 4. 데이터 파일 준비
데이터분석을 위한 Titanic 생존자 데이터셋을 온라인 혹은 로컬 파일에서 불러오는 두 가지 방법.
1. 인터넷에서 직접 불러오기 (권장):

- URL을 통해 .csv 파일을 직접 읽어와 즉시 분석 시작 가능

2. 로컬에 저장된 파일 불러오기:

- 사용자의 PC에 다운로드된 데이터를 경로를 지정하여 읽음

**온라인에서 직접 불러오기 (추천):**

```python
import pandas as pd

# 온라인으로 직접 데이터 불러오기
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)
print("데이터 로드 성공!")
```

**로컬 파일 사용하기:**
```python
# 파일을 다운로드한 경우
df = pd.read_csv('경로/파일명.csv')  # 실제 파일 경로로 변경

# 예시: Windows
df = pd.read_csv(r'C:\Users\사용자명\Documents\titanic.csv')

# 예시: macOS/Linux  
df = pd.read_csv('/Users/사용자명/Documents/titanic.csv')
```
- 경로와 파일명이 정확하지 않으면 FileNotFoundError 발생
- 한글 경로 또는 공백 포함 시에는 이스케이프 처리(\\ 또는 r'') 필요

**pd.read_csv() 함수 상세 설명**
| 파라미터              | 설명                    | 예시                                      |
| -------------------- | ----------------------- | --------------------------------------- |
| `filepath_or_buffer` | 읽어올 파일 경로 또는 URL  | `'file.csv'`, `'https://...'`           |
| `sep`                | 구분자 지정 (기본: 쉼표)   | `sep=';'`                               |
| `header`             | 헤더 행 지정 (기본: 첫 줄) | `header=0`                              |
| `encoding`           | 문자 인코딩               | `encoding='utf-8'`, `encoding='euc-kr'` |
| `usecols`            | 사용할 열만 선택          | `usecols=['Name', 'Survived']`          |

### 5. 환경 설정 확인

다음 코드로 모든 설정이 올바르게 되었는지 확인해보세요:

```python
# 환경 설정 확인 코드
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("✅ 라이브러리 import 성공!")

# 한글 폰트 테스트
plt.figure(figsize=(6, 4))
plt.title("한글 폰트 테스트")
plt.text(0.5, 0.5, "안녕하세요! 한글이 잘 보이나요?", 
         ha='center', va='center', fontsize=14)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()

print("✅ 환경 설정 완료!")
```

**💡 문제가 발생했다면?**
- 한글이 깨져 보인다면: 위의 폰트 설정 코드를 다시 실행해보세요
- 라이브러리 import 오류: `pip install 라이브러리명`으로 설치를 확인해보세요
- 파일을 찾을 수 없다면: 파일 경로가 올바른지 확인하거나 온라인 URL을 사용해보세요

---

## 핵심 개념 설명

### 데이터 타입의 이해: 숫자 너머의 의미 파악하기

데이터 분석을 시작하기 전에 가장 먼저 해야 할 일은 **"이 데이터가 무엇을 나타내는가?"**를 파악하는 것입니다. 같은 숫자 '1'이라도 맥락에 따라 전혀 다른 의미를 가질 수 있거든요.

예를 들어보겠습니다:
- 키: 175 (cm) → 계산이 가능한 연속적인 값
- 자녀 수: 2 (명) → 셀 수 있는 개별적인 값  
- 혈액형: A → 순서가 없는 범주
- 학년: 3학년 → 순서가 있는 범주

이처럼 데이터의 **타입**을 정확히 파악해야 적절한 분석 방법을 선택할 수 있습니다.

#### 1. 수치형 데이터 (Numerical Data): 계산이 가능한 데이터

수치형 데이터는 숫자로 표현되며, 수학적 연산(더하기, 빼기, 평균 계산 등)이 의미가 있는 데이터입니다.

**📊 연속형 데이터 (Continuous Data)**
- **특징**: 이론적으로 무한한 값을 가질 수 있음
- **예시**: 
  - 키: 175.3cm, 175.31cm, 175.315cm... (더 정밀하게 측정 가능)
  - 몸무게: 68.5kg, 68.52kg...
  - 온도: 23.7°C, 기온은 연속적으로 변함
  - 시간: 2.5시간, 운동 시간은 연속적

```python
# 연속형 데이터 예시
import numpy as np

heights = [165.2, 170.8, 175.3, 168.9, 172.1]
weights = [55.3, 68.7, 72.1, 61.2, 69.8]

# 연속형 데이터로 할 수 있는 연산들
print(f"평균 키: {np.mean(heights):.1f}cm")  # 170.5cm
print(f"키의 표준편차: {np.std(heights):.1f}cm")  # 3.8cm
```

**🔢 이산형 데이터 (Discrete Data)**
- **특징**: 셀 수 있는 개별적인 값들
- **예시**:
  - 자녀 수: 0명, 1명, 2명 (2.5명은 불가능)
  - 방문 횟수: 1회, 2회, 3회...
  - 점수: 80점, 85점, 90점 (정수로 제한된 경우)

```python
# 이산형 데이터 예시
children_count = [0, 1, 2, 1, 3, 0, 2]
visit_count = [5, 12, 8, 15, 3]

# 이산형 데이터 분석
print(f"평균 자녀 수: {np.mean(children_count):.1f}명")  # 1.3명
print(f"최빈 자녀 수: {max(set(children_count), key=children_count.count)}명")  # 가장 많은 경우
```

#### 2. 범주형 데이터 (Categorical Data): 그룹을 나타내는 데이터

범주형 데이터는 그룹이나 카테고리를 나타내며, 수학적 연산보다는 **빈도 분석**이나 **비율 계산**이 적합합니다.

**🏷️ 명목형 데이터 (Nominal Data)**
- **특징**: 순서나 크기 비교가 불가능한 카테고리
- **예시**:
  - 성별: 남성, 여성
  - 혈액형: A, B, AB, O (A가 B보다 크다고 할 수 없음)
  - 거주 지역: 서울, 부산, 대구...
  - 취미: 독서, 영화감상, 운동...

```python
# 명목형 데이터 예시
blood_types = ['A', 'B', 'O', 'A', 'AB', 'O', 'A']
regions = ['서울', '부산', '대구', '서울', '인천', '서울']

# 명목형 데이터 분석
from collections import Counter
print("혈액형 분포:", Counter(blood_types))
# 출력: Counter({'A': 3, 'O': 2, 'B': 1, 'AB': 1})
```

**📊 순서형 데이터 (Ordinal Data)**
- **특징**: 순서나 등급이 있는 카테고리 (하지만 간격이 일정하지 않음)
- **예시**:
  - 학력: 초졸 < 중졸 < 고졸 < 대졸 < 대학원졸
  - 만족도: 매우 불만족 < 불만족 < 보통 < 만족 < 매우 만족
  - 성적 등급: F < D < C < B < A
  - 티셔츠 사이즈: XS < S < M < L < XL

```python
# 순서형 데이터 예시
satisfaction = ['만족', '보통', '매우 만족', '불만족', '만족']
grades = ['B', 'A', 'C', 'B', 'A', 'C']

# 순서형 데이터는 순서를 고려한 분석이 필요
satisfaction_order = ['매우 불만족', '불만족', '보통', '만족', '매우 만족']
grade_order = ['F', 'D', 'C', 'B', 'A']
```

#### 3. 기타 중요한 데이터 타입

**📅 시간 데이터 (Temporal Data)**
- **특징**: 시간의 흐름을 나타내는 데이터
- **예시**: 생년월일, 거래 시간, 접속 로그
- **분석 포인트**: 트렌드, 계절성, 주기성 분석 가능

```python
# 시간 데이터 예시
import pandas as pd
dates = ['2024-01-15', '2024-02-20', '2024-03-10']
times = pd.to_datetime(dates)
print("요일 분석:", times.day_name())  # Monday, Tuesday, Saturday
```

**📝 텍스트 데이터 (Text Data)**
- **특징**: 자유로운 형태의 문자열 데이터
- **예시**: 리뷰, 댓글, 뉴스 기사, 소셜미디어 게시글
- **분석 포인트**: 감성 분석, 키워드 추출, 토픽 모델링

```python
# 텍스트 데이터 예시
reviews = [
    "이 제품 정말 좋아요! 추천합니다.",
    "배송이 너무 늦어서 아쉬워요.",
    "가성비 최고! 만족스럽습니다."
]

# 텍스트 길이 분석
text_lengths = [len(review) for review in reviews]
print(f"평균 리뷰 길이: {np.mean(text_lengths):.1f}자")
```

### 🎯 데이터 타입별 분석 전략

| 데이터 타입 | 주요 분석 방법 | 적합한 시각화 | 대표 통계량 |
|------------|---------------|--------------|------------|
| **연속형** | 분포 분석, 상관관계 | 히스토그램, 산점도 | 평균, 표준편차 |
| **이산형** | 빈도 분석, 패턴 탐지 | 막대그래프, 박스플롯 | 최빈값, 중앙값 |
| **명목형** | 빈도 분석, 교차분석 | 파이차트, 막대그래프 | 최빈값, 비율 |
| **순서형** | 순위 분석, 분포 분석 | 막대그래프, 박스플롯 | 중앙값, 분위수 |
| **시간** | 트렌드, 계절성 분석 | 시계열 그래프 | 주기, 증감률 |
| **텍스트** | 감성, 키워드 분석 | 워드클라우드 | 빈도, 감성점수 |

### ⚠️ 흔한 실수와 주의사항

**1. 숫자라고 모두 수치형이 아닙니다!**
```python
# 잘못된 예시: 학번을 연속형으로 취급
student_ids = [20240001, 20240002, 20240003]
# 학번의 평균을 구하는 것은 의미가 없습니다!

# 올바른 예시: 학번은 명목형으로 취급
student_ids = ['20240001', '20240002', '20240003']
```

**2. 데이터 타입 확인의 중요성**
```python
# 데이터를 잘못 해석하면 엉뚱한 결과가 나올 수 있습니다
survey_data = [1, 2, 3, 1, 2]  # 1=불만족, 2=보통, 3=만족

# 잘못된 분석: 평균 만족도 1.8? 무슨 의미인가요?
wrong_analysis = np.mean(survey_data)  

# 올바른 분석: 각 응답의 빈도 확인
correct_analysis = Counter(survey_data)
# {1: 2명, 2: 2명, 3: 1명} → 불만족 40%, 보통 40%, 만족 20%
```

이제 데이터 타입의 기본 개념을 이해했으니, 실제 Python 코드로 데이터를 탐색하는 방법을 배워보겠습니다!

---

## 핵심 기술 / 코드 구현

### 단계별 데이터 탐색: 처음 만나는 데이터와 친해지기

실제 데이터 분석을 시작할 때는 단계적으로 접근하는 것이 중요합니다. 마치 새로운 사람을 만났을 때 처음에는 기본적인 정보부터 파악하듯이 말이죠.

#### 1단계: 라이브러리 import와 기본 설정

먼저 필요한 도구들을 준비해보겠습니다.

```python
# =====================================================
# 1단계: 필요한 라이브러리 불러오기
# =====================================================

import pandas as pd          # 데이터 조작과 분석을 위한 라이브러리
import numpy as np           # 수치 계산을 위한 라이브러리
import matplotlib.pyplot as plt  # 기본 시각화 도구
import seaborn as sns        # 고급 시각화 도구 (matplotlib 기반)

# 한글 폰트 설정 (그래프에서 한글 깨짐 방지)
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
# plt.rcParams['font.family'] = 'AppleGothic'  # macOS 사용자
plt.rcParams['axes.unicode_minus'] = False    # 마이너스 기호 깨짐 방지

# 출력 옵션 설정
pd.set_option('display.max_columns', 20)      # 최대 20개 열까지 표시
pd.set_option('display.width', 1000)         # 출력 너비 설정

print("✅ 라이브러리 준비 완료!")
```

**📝 각 라이브러리가 하는 일:**
- **pandas**: 엑셀 같은 표 형태 데이터를 다루는 핵심 도구
- **numpy**: 수학 계산을 빠르게 처리하는 도구
- **matplotlib**: 그래프를 그리는 기본 도구
- **seaborn**: 더 예쁘고 통계적인 그래프를 쉽게 그리는 도구

#### 2단계: 데이터 불러오기

```python
# =====================================================
# 2단계: 데이터 불러오기
# =====================================================

# 방법 1: 온라인에서 직접 불러오기 (추천)
data_url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'

try:
    df = pd.read_csv(data_url)  # CSV 파일을 DataFrame으로 읽어옴
    print("✅ 온라인 데이터 로드 성공!")
except:
    print("❌ 온라인 연결 실패. 로컬 파일을 사용해주세요.")
    # 방법 2: 로컬 파일 사용
    # df = pd.read_csv('titanic.csv')

# DataFrame이란? 
# - Excel의 worksheet 같은 2차원 표 구조
# - 행(row)과 열(column)로 구성
# - 각 열은 서로 다른 데이터 타입을 가질 수 있음

print(f"데이터 불러오기 완료! {df.shape[0]}행 {df.shape[1]}열의 데이터입니다.")
```

**🔍 DataFrame 구조 이해하기:**
```
    PassengerId  Survived  Pclass     Name      Sex   Age  ...
0             1         0       3   Braund   male  22.0  ...  ← 첫 번째 행 (index=0)
1             2         1       1  Cumings female  38.0  ...  ← 두 번째 행 (index=1)
2             3         1       3  Heikkinen female  26.0  ...
...
    ↑            ↑         ↑        ↑        ↑     ↑
  컬럼1        컬럼2      컬럼3     컬럼4     컬럼5   컬럼6
```

#### 3단계: 데이터 구조 파악하기

```python
# =====================================================
# 3단계: 데이터의 기본 구조 확인
# =====================================================

print("=" * 60)
print("🔍 데이터 기본 정보")
print("=" * 60)

# 3-1. 데이터 크기 확인
print(f"📊 데이터 크기: {df.shape[0]:,}행 × {df.shape[1]}열")
# shape는 데이터프레임의 (행 수, 열 수) 튜플을 반환
# df.shape[0]: 행 수 (관측값의 개수)
# df.shape[1]: 열 수 (변수의 개수)

# 3-2. DataFrame에서 사용하는 메모리 사용량 확인
memory_usage = df.memory_usage(deep=True).sum() / 1024  # KB 단위로 변환
print(f"💾 메모리 사용량: {memory_usage:.1f} KB")

# 3-3. 컬럼 이름 확인
print(f"📋 컬럼 목록: {list(df.columns)}")

print("\n" + "=" * 60)
print("👀 데이터 미리보기")
print("=" * 60)

# 3-4. 처음 몇 행 확인
print("🔝 처음 5개 행:")
print(df.head())  # 기본값은 5개 행
# head(n)으로 n개 행을 볼 수 있음

print("\n🔚 마지막 3개 행:")
print(df.tail(3))  # 마지막 3개 행 확인
```

**💡 왜 head()와 tail()을 확인하나요?**
- **head()**: 데이터가 제대로 불러와졌는지, 컬럼 이름이 올바른지 확인
- **tail()**: 데이터 끝 부분에 이상한 값이나 요약 행이 있는지 확인

#### 4단계: 데이터 타입과 정보 분석

```python
# =====================================================
# 4단계: 데이터 타입 및 상세 정보 확인
# =====================================================

print("\n" + "=" * 60)
print("🧮 데이터 타입 분석")
print("=" * 60)

# 4-1. 각 컬럼의 데이터 타입 확인
print("📊 각 컬럼의 데이터 타입:")
for i, (column, dtype) in enumerate(df.dtypes.items(), 1):
    print(f"{i:2d}. {column:12s} → {str(dtype):10s}")

# 4-2. 상세 정보 한눈에 보기
print(f"\n📋 데이터 상세 정보:")
df.info()
# info()는 다음 정보를 보여줍니다:
# - 각 컬럼의 이름
# - Non-Null Count (결측값이 아닌 데이터 개수)
# - 데이터 타입
# - 메모리 사용량

# 4-3. 각 컬럼의 고유값 개수 확인
print(f"\n🔢 각 컬럼의 고유값 개수:")
unique_counts = df.nunique().sort_values(ascending=False)
for column, count in unique_counts.items():
    percentage = (count / len(df)) * 100
    print(f"{column:12s}: {count:3d}개 ({percentage:5.1f}%)")
```

**🎯 데이터 타입 해석 가이드:**
- **int64**: 정수 (나이, 개수 등)
- **float64**: 실수 (가격, 비율 등)
- **object**: 문자열 또는 혼합 타입 (이름, 카테고리 등)
- **datetime64**: 날짜/시간
- **bool**: True/False

#### 5단계: 결측치 분석하기
데이터프레임(df)에서 결측치(Missing Value)를 분석하고 Pandas 라이브러리를 사용하여 데이터프레임의 결측치 상태를 다양한 관점에서 확인합니다.

- 컬럼별 결측치 개수 확인: 각 컬럼의 결측치 개수와 비율을 출력.
- 결측치 요약 테이블 생성: 결측치가 있는 컬럼에 대해 개수, 비율, 데이터가 있는 행 수를 정리한 테이블 생성.
- 전체 결측치 비율 계산: 데이터프레임 전체의 결측치 비율을 계산하여 출력.
데이터 전처리 과정에서 결측치 문제를 파악하고, 어떤 컬럼에 결측치가 집중되어 있는지, 전체 데이터에서 결측치가 차지하는 비중을 이해하는 데 유용합니다.

**5-1. 각 컬럼별 결측치 개수 확인**
목적: 데이터프레임의 각 컬럼에 대해 결측치 개수를 확인하고, 결측치가 있는 컬럼만 출력.

```python
print("\n" + "=" * 60)
print("결측치 분석")
print("=" * 60)

# 컬럼별 결측치 개수 확인
missing_counts = df.isnull().sum()
print("📊 컬럼별 결측치 개수:")
for column, count in missing_counts.items():
    if count > 0:  # 결측치가 있는 컬럼만 표시
        percentage = (count / len(df)) * 100
        print(f"{column:12s}: {count:3d}개 ({percentage:5.1f}%)")

```
- df.isnull(): 데이터프레임에서 결측치(NaN, None 등)를 True로, 비결측치를 False로 표시하는 불리언 데이터프레임 반환.
- .sum(): 각 컬럼의 True 값(결측치)을 합산하여 결측치 개수를 계산.
- missing_counts.items(): 컬럼 이름과 결측치 개수를 쌍으로 순회.
- if count > 0: 결측치가 0보다 큰 컬럼만 출력.
- percentage = (count / len(df)) * 100: 결측치 비율을 계산 (컬럼의 결측치 개수 / 총 행 수 * 100).
- print(f"{column:12s}: {count:3d}개 ({percentage:5.1f}%)"): 포맷팅을 사용해 컬럼 이름(12자리 왼쪽 정렬), 결측치 개수(3자리 정수), 비율(소수점 1자리) 출력.

**주요 함수:**
1. df.isnull():
- 기능: 데이터프레임에서 결측치(NaN, None)를 True, 비결측치를 False로 표시.
- 출력: 동일한 크기의 불리언 데이터프레임.
- 사용 예시: df = pd.DataFrame({'A': [1, None, 3], 'B': [None, 5, 6]}) → df.isnull()는 A: [False, True, False], B: [True, False, False] 반환.
2. pd.Series.sum():
- 기능: 시리즈의 모든 값을 합산.
- 코드 내 역할: 각 컬럼의 결측치(True) 개수를 계산.
- 사용 예시: pd.Series([True, False, True]).sum() → 2 (True는 1로 계산).
3. Series.items():
- 기능: 시리즈의 인덱스(여기서는 컬럼 이름)와 값을 쌍으로 반환.
- 사용 예시: pd.Series({'A': 2, 'B': 0}).items() → ('A', 2), ('B', 0).

**5-2. 결측치 요약 테이블 만들기**
목적: 결측치 관련 정보를 요약한 데이터프레임을 생성하고, 결측치가 있는 컬럼만 비율 기준 내림차순으로 정렬.
```python
# 결측치 요약 테이블 생성
missing_summary = pd.DataFrame({
    '결측치_개수': missing_counts,
    '결측치_비율(%)': (missing_counts / len(df)) * 100,
    '데이터_있는_개수': len(df) - missing_counts
}).round(2)

# 결측치가 있는 컬럼만 표시
missing_summary = missing_summary[missing_summary['결측치_개수'] > 0]
missing_summary = missing_summary.sort_values('결측치_비율(%)', ascending=False)

print(f"\n📋 결측치 요약:")
print(missing_summary)

```

- pd.DataFrame({...}): 세 개의 컬럼(결측치_개수, 결측치_비율(%), 데이터_있는_개수)으로 구성된 데이터프레임 생성.
    - '결측치_개수': missing_counts (컬럼별 결측치 개수).
    - '결측치_비율(%)': (missing_counts / len(df)) * 100 (결측치 비율).
    - '데이터_있는_개수': len(df) - missing_counts (비결측치 행 수).
- .round(2): 모든 숫자를 소수점 둘째 자리로 반올림.
- missing_summary[missing_summary['결측치_개수'] > 0]: 결측치가 0보다 큰 행(컬럼)만 필터링.
- .sort_values('결측치_비율(%)', ascending=False): 결측치 비율 기준 내림차순 정렬.
- print(missing_summary): 요약 테이블 출력.

**주요 함수:**
1. pd.DataFrame(data):
- 기능: 딕셔너리나 배열을 기반으로 데이터프레임 생성.
- 코드 내 역할: 결측치 관련 통계를 데이터프레임으로 구성.
- 사용 예시: pd.DataFrame({'A': [1, 2], 'B': [3, 4]}) → niên
2. DataFrame.round(n):
- 기능: 데이터프레임의 숫자 값을 소수점 n자리로 반올림.
- 코드 내 역할: 비율을 소수점 둘째 자리로 반올림.
- 사용 예시: pd.Series([1.234, 5.678]).round(2) → [1.23, 5.68].
3. DataFrame.sort_values(by, ascending):
- 기능: 지정된 컬럼(by)을 기준으로 데이터프레임 정렬.
- 코드 내 역할: 결측치 비율 기준 내림차순 정렬.
- 사용 예시: df.sort_values('A', ascending=False) → 컬럼 'A'를 내림차순 정렬.

**5-3. 전체 데이터에서 결측치 비율**
목적: 데이터프레임 전체의 결측치 비율을 계산하고, 전체 셀 수와 결측치 수를 함께 출력

```python
# 전체 결측치 비율 계산
total_missing = df.isnull().sum().sum()
total_cells = df.shape[0] * df.shape[1]
overall_missing_rate = (total_missing / total_cells) * 100

print(f"\n📈 전체 결측치 현황:")
print(f"전체 셀 개수: {total_cells:,}개")
print(f"결측치 총 개수: {total_missing:,}개")
print(f"전체 결측치 비율: {overall_missing_rate:.2f}%")
```
- df.isnull().sum().sum(): 모든 컬럼의 결측치 개수를 합산하여 총 결측치 수 계산.
- df.shape[0] * df.shape[1]: 데이터프레임의 행 수(shape[0])와 열 수(shape[1])를 곱해 총 셀 수 계산.
- overall_missing_rate: 결측치 비율을 백분율로 계산.
- print 문으로 전체 셀 수, 결측치 수, 비율 출력.

**🔍 결측치 분석 해석:**
- **0-5%**: 일반적으로 문제없음
- **5-15%**: 주의깊게 처리 필요
- **15% 이상**: 심각한 문제, 해당 컬럼 사용 여부 재검토

#### 6단계: 기초 통계량 확인하기
**6-1. 수치형 변수 기초 통계량**
목적: 데이터프레임의 수치형 컬럼에 대해 기초 통계량(개수, 평균, 표준편차, 최소값, 사분위수, 최대값)을 계산하고 출력.
```python
print("\n" + "=" * 60)
print("📊 기초 통계량 분석")
print("=" * 60)

# 6-1. 수치형 변수 기초 통계량
print("🔢 수치형 변수 기초 통계량:")
numeric_summary = df.describe()
print(numeric_summary.round(2))
```
- df.describe(): 수치형 컬럼에 대해 기본 통계량을 계산하여 데이터프레임으로 반환.
- 반환되는 통계량: count, mean, std, min, 25%, 50%, 75%, max.
    - count: 결측값이 아닌 데이터 개수
    - mean: 평균값
    - std: 표준편차 (데이터의 퍼진 정도)
    - min: 최솟값
    - 25%: 1사분위수 (하위 25% 지점)
    - 50%: 중앙값 (median)
    - 75%: 3사분위수 (하위 75% 지점)
    - max: 최댓값
- .round(2): 통계량을 소수점 둘째 자리로 반올림하여 가독성 향상.
- print(numeric_summary.round(2)): 결과를 출력.

1. df.describe():
- 기능: 데이터프레임의 수치형 컬럼에 대해 기초 통계량을 계산하여 데이터프레임으로 반환.
- 코드 내 역할: 수치형 변수의 통계량을 요약.
- 사용 예시: df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}) → df.describe()는 count, mean, std 등을 포함한 데이터프레임 반환.
- 매개변수:
    - include: 분석할 데이터 타입 지정 (기본: 수치형만).
    - percentiles: 표시할 백분위수 지정 (기본: [0.25, 0.5, 0.75]).
2. DataFrame.round(n):
- 기능: 데이터프레임의 숫자 값을 소수점 n자리로 반올림.
- 코드 내 역할: 통계량을 소수점 둘째 자리로 반올림.
- 사용 예시: pd.DataFrame({'A': [1.234, 5.678]}).round(2) → A: [1.23, 5.68].

**6-2. 범주형 변수 분석**
목적: 범주형 컬럼의 고유값 분포를 확인하고, 각 값의 개수와 비율을 출력. 고유값이 많을 경우 상위 5개만 표시.
```python
print(f"\n🏷️ 범주형 변수 분석:")
categorical_columns = df.select_dtypes(include=['object']).columns

for column in categorical_columns:
    print(f"\n📋 {column} 분포:")
    value_counts = df[column].value_counts()
    
    # 상위 5개만 표시 (너무 많으면 요약)
    if len(value_counts) > 5:
        print(value_counts.head())
        print(f"... (총 {len(value_counts)}개 고유값)")
    else:
        for value, count in value_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {str(value):15s}: {count:3d}개 ({percentage:5.1f}%)")
```
- df.select_dtypes(include=['object']): 데이터프레임에서 object 타입(문자열, 범주형 데이터)의 컬럼을 선택.
- .columns: 선택된 컬럼의 이름 리스트 반환.
- for column in categorical_columns: 각 범주형 컬럼을 순회.
- df[column].value_counts(): 해당 컬럼의 고유값별 빈도수를 계산.
- if len(value_counts) > 5: 고유값이 5개 초과면 상위 5개만 출력 (value_counts.head()).
- else: 고유값이 5개 이하면 모든 값의 개수와 비율(count / len(df) * 100) 출력.
- print(f" {str(value):15s}: {count:3d}개 ({percentage:5.1f}%)"): 값(15자리 왼쪽 정렬), 개수(3자리 정수), 비율(소수점 1자리) 출력.

1. df.select_dtypes(include):
기능: 지정된 데이터 타입의 컬럼만 선택.
코드 내 역할: object 타입(문자열, 범주형) 컬럼 선택.
사용 예시: df.select_dtypes(include=['object']) → 문자열 타입 컬럼만 반환.
매개변수:
include: 선택할 데이터 타입 (예: ['object'], ['int64', 'float64']).
2. Series.value_counts():
기능: 시리즈의 고유값별 빈도수를 계산하여 시리즈로 반환.
코드 내 역할: 범주형 컬럼의 각 값의 등장 횟수 계산.
사용 예시: pd.Series(['A', 'B', 'A']).value_counts() → A: 2, B: 1.
3. Series.head(n):
기능: 시리즈 또는 데이터프레임의 상위 n개 행 반환 (기본값: 5).
코드 내 역할: 고유값이 많을 때 상위 5개만 출력.
사용 예시: pd.Series([1, 2, 3, 4, 5]).head(3) → [1, 2, 3].
4. Series.items():
기능: 시리즈의 인덱스(고유값)와 값을 쌍으로 반환.
코드 내 역할: 고유값과 빈도수를 순회하며 출력.
사용 예시: pd.Series({'A': 2, 'B': 1}).items() → ('A', 2), ('B', 1).

**📈 통계량 해석 팁:**
- **평균 vs 중앙값**: 차이가 크면 데이터가 한쪽으로 치우쳤음 (이상치 존재 가능)
- **표준편차**: 평균의 절반 이상이면 데이터가 많이 퍼져있음
- **min/max**: 상식적으로 말이 안 되는 값이 있는지 확인

---

## 상세 예제 / 미니 프로젝트

### 타이타닉 데이터 완전 정복: 체계적 데이터 탐색 프로젝트

이제 배운 내용을 종합하여 타이타닉 데이터셋을 완전히 분석해보겠습니다. 실제 데이터 분석가가 되어 단계별로 진행해보세요!

#### 프로젝트 목표
- 타이타닉 호에 탑승한 승객들의 데이터 구조 완전 파악
- 데이터 품질 문제점 발견 및 해결 방향 제시
- 각 변수의 특성과 분포 심층 분석
- 개선된 시각화로 데이터 패턴 발견

#### Step 1: 종합 데이터 개요 함수 만들기

```python
# =====================================================
# 타이타닉 데이터 종합 분석 함수
# =====================================================

def comprehensive_data_overview(dataframe, dataset_name="데이터셋"):
    """
    데이터셋의 종합적인 개요를 제공하는 함수
    
    Parameters:
    dataframe: pandas DataFrame
    dataset_name: 데이터셋 이름 (출력용)
    """
    
    print("🚢" + "="*60)
    print(f"📊 {dataset_name} 종합 분석 보고서")
    print("="*60)
    
    # 1. 기본 정보
    print(f"\n1️⃣ 기본 정보")
    print(f"   📏 데이터 크기: {dataframe.shape[0]:,}행 × {dataframe.shape[1]}열")
    print(f"   💾 메모리 사용량: {dataframe.memory_usage(deep=True).sum()/1024:.1f} KB")
    print(f"   📅 분석 일시: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 2. 컬럼 분류
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = dataframe.select_dtypes(include=['object']).columns.tolist()
    
    print(f"\n2️⃣ 변수 분류")
    print(f"   🔢 수치형 변수: {len(numeric_cols)}개")
    for col in numeric_cols:
        unique_count = dataframe[col].nunique()
        col_type = "연속형" if unique_count > 20 else "이산형"
        print(f"      - {col} ({col_type}, {unique_count}개 고유값)")
    
    print(f"   🏷️ 범주형 변수: {len(categorical_cols)}개")
    for col in categorical_cols:
        unique_count = dataframe[col].nunique()
        print(f"      - {col} ({unique_count}개 고유값)")
    
    # 3. 결측치 현황
    print(f"\n3️⃣ 데이터 품질")
    missing_data = dataframe.isnull().sum()
    missing_cols = missing_data[missing_data > 0]
    
    if len(missing_cols) > 0:
        print(f"   ⚠️ 결측치 발견: {len(missing_cols)}개 컬럼")
        for col, count in missing_cols.items():
            percentage = (count / len(dataframe)) * 100
            status = "🔴 심각" if percentage > 20 else "🟡 주의" if percentage > 5 else "🟢 양호"
            print(f"      - {col}: {count}개 ({percentage:.1f}%) {status}")
    else:
        print(f"   ✅ 결측치 없음 - 완벽한 데이터!")
    
    # 4. 이상치 빠른 체크 (수치형 변수)
    print(f"\n4️⃣ 이상치 빠른 체크")
    outlier_summary = []
    
    for col in numeric_cols:
        if dataframe[col].dtype in ['int64', 'float64']:
            Q1 = dataframe[col].quantile(0.25)
            Q3 = dataframe[col].quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR > 0:  # IQR이 0이 아닌 경우만
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = dataframe[(dataframe[col] < lower_bound) | 
                                   (dataframe[col] > upper_bound)]
                outlier_count = len(outliers)
                outlier_percentage = (outlier_count / len(dataframe)) * 100
                
                if outlier_count > 0:
                    outlier_summary.append({
                        '컬럼': col,
                        '이상치수': outlier_count,
                        '비율(%)': round(outlier_percentage, 1)
                    })
    
    if outlier_summary:
        outlier_df = pd.DataFrame(outlier_summary)
        print(outlier_df.to_string(index=False))
    else:
        print("   ✅ 주요 이상치 없음")
    
    return {
        'numeric_columns': numeric_cols,
        'categorical_columns': categorical_cols,
        'missing_columns': missing_cols.index.tolist() if len(missing_cols) > 0 else [],
        'outlier_columns': [item['컬럼'] for item in outlier_summary]
    }

# 함수 실행
analysis_result = comprehensive_data_overview(df, "타이타닉 승객 데이터")
```

#### Step 2: 개선된 결측치 시각화

```python
# =====================================================
# 결측치 패턴 고급 시각화
# =====================================================

def enhanced_missing_data_visualization(dataframe):
    """
    결측치 패턴을 다양한 방법으로 시각화하는 함수
    """
    
    missing_data = dataframe.isnull()
    
    # 그래프 설정
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('🔍 결측치 패턴 종합 분석', fontsize=16, fontweight='bold')
    
    # 1. 히트맵으로 결측치 패턴 시각화
    sns.heatmap(missing_data.iloc[:100],  # 처음 100행만 (전체가 너무 크면)
                cmap='viridis', 
                cbar=True,
                ax=axes[0,0],
                yticklabels=False)
    axes[0,0].set_title('📊 결측치 패턴 히트맵\n(노란색: 결측치, 보라색: 정상값)')
    axes[0,0].set_xlabel('컬럼')
    axes[0,0].set_ylabel('행 인덱스 (샘플)')
    
    # 2. 컬럼별 결측치 비율 막대그래프
    missing_percentage = (missing_data.sum() / len(dataframe)) * 100
    missing_percentage = missing_percentage[missing_percentage > 0]  # 결측치가 있는 컬럼만
    
    if len(missing_percentage) > 0:
        missing_percentage.plot(kind='bar', ax=axes[0,1], color='coral')
        axes[0,1].set_title('📈 컬럼별 결측치 비율')
        axes[0,1].set_ylabel('결측치 비율 (%)')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 심각도별 색상 구분
        for i, (col, pct) in enumerate(missing_percentage.items()):
            color = 'red' if pct > 20 else 'orange' if pct > 5 else 'green'
            axes[0,1].bar(i, pct, color=color)
    else:
        axes[0,1].text(0.5, 0.5, '✅ 결측치 없음!', 
                      ha='center', va='center', transform=axes[0,1].transAxes,
                      fontsize=14, fontweight='bold')
        axes[0,1].set_title('📈 컬럼별 결측치 비율')
    
    # 3. 행별 결측치 개수 분포
    missing_per_row = missing_data.sum(axis=1)
    missing_per_row.value_counts().sort_index().plot(kind='bar', ax=axes[1,0], color='skyblue')
    axes[1,0].set_title('📊 행별 결측치 개수 분포')
    axes[1,0].set_xlabel('행당 결측치 개수')
    axes[1,0].set_ylabel('해당 행의 개수')
    
    # 4. 결측치 상관관계 (어떤 컬럼들이 함께 결측되는가?)
    if len(missing_percentage) > 1:
        # 결측치가 있는 컬럼들만 선택
        missing_cols = missing_percentage.index
        missing_corr = missing_data[missing_cols].corr()
        
        sns.heatmap(missing_corr, 
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0,
                   ax=axes[1,1],
                   square=True)
        axes[1,1].set_title('🔗 결측치 상관관계\n(높을수록 함께 결측됨)')
    else:
        axes[1,1].text(0.5, 0.5, '결측치 상관관계\n분석 불가\n(결측 컬럼 < 2개)', 
                      ha='center', va='center', transform=axes[1,1].transAxes,
                      fontsize=12)
        axes[1,1].set_title('🔗 결측치 상관관계')
    
    plt.tight_layout()
    plt.show()
    
    # 결측치 패턴 해석 가이드
    print("\n🔍 결측치 패턴 해석 가이드:")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("📊 히트맵 해석:")
    print("  • 세로로 긴 노란색 줄: 해당 컬럼에 결측치가 많음")
    print("  • 가로로 긴 노란색 줄: 해당 행(관측값)에 여러 컬럼의 결측치 존재")
    print("  • 대각선 패턴: 특정 조건에서 여러 변수가 동시에 결측될 가능성")
    print("\n📈 비율 막대그래프:")
    print("  • 🟢 녹색 (0-5%): 일반적으로 문제없음")
    print("  • 🟡 주황 (5-20%): 주의깊게 처리 필요")
    print("  • 🔴 빨강 (20%+): 심각한 문제, 해당 컬럼 사용 여부 재검토")
    print("\n🔗 상관관계:")
    print("  • 높은 양의 상관관계: 컬럼들이 함께 결측되는 경향")
    print("  • 이는 데이터 수집 과정의 체계적 문제를 시사할 수 있음")

# 함수 실행
enhanced_missing_data_visualization(df)
```

#### Step 3: 수치형 변수 심층 분석

```python
# =====================================================
# 수치형 변수 분포 분석 및 시각화
# =====================================================

def analyze_numeric_variables(dataframe, numeric_cols):
    """
    수치형 변수들의 분포를 심층 분석하는 함수
    """
    
    # 수치형 컬럼 중 실제로 연속형인 것들 선별
    continuous_cols = []
    discrete_cols = []
    
    for col in numeric_cols:
        if dataframe[col].nunique() > 20:
            continuous_cols.append(col)
        else:
            discrete_cols.append(col)
    
    print(f"🔢 수치형 변수 상세 분석")
    print(f"연속형: {continuous_cols}")
    print(f"이산형: {discrete_cols}")
    
    # 연속형 변수 시각화
    if continuous_cols:
        n_cols = len(continuous_cols)
        fig, axes = plt.subplots(2, n_cols, figsize=(5*n_cols, 10))
        
        if n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle('📊 연속형 변수 분포 분석', fontsize=16, fontweight='bold')
        
        for i, col in enumerate(continuous_cols):
            # 결측치 제거
            data = dataframe[col].dropna()
            
            # 히스토그램
            axes[0, i].hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, i].axvline(data.mean(), color='red', linestyle='--', 
                              label=f'평균: {data.mean():.1f}')
            axes[0, i].axvline(data.median(), color='green', linestyle='--', 
                              label=f'중앙값: {data.median():.1f}')
            axes[0, i].set_title(f'{col} 분포')
            axes[0, i].set_xlabel(col)
            axes[0, i].set_ylabel('빈도')
            axes[0, i].legend()
            
            # 박스플롯
            axes[1, i].boxplot(data)
            axes[1, i].set_title(f'{col} 박스플롯')
            axes[1, i].set_ylabel(col)
            
            # 이상치 정보 표시
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = data[(data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)]
            
            if len(outliers) > 0:
                axes[1, i].text(0.5, 0.95, f'이상치: {len(outliers)}개', 
                               transform=axes[1, i].transAxes, 
                               ha='center', va='top',
                               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        plt.tight_layout()
        plt.show()
        
        # 통계 요약
        print(f"\n📈 연속형 변수 통계 요약:")
        print(dataframe[continuous_cols].describe().round(2))
    
    # 이산형 변수 시각화
    if discrete_cols:
        n_cols = len(discrete_cols)
        fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 5))
        
        if n_cols == 1:
            axes = [axes]
        
        fig.suptitle('📊 이산형 변수 분포 분석', fontsize=16, fontweight='bold')
        
        for i, col in enumerate(discrete_cols):
            value_counts = dataframe[col].value_counts().sort_index()
            
            axes[i].bar(value_counts.index, value_counts.values, 
                       alpha=0.7, color='lightcoral', edgecolor='black')
            axes[i].set_title(f'{col} 분포')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('빈도')
            
            # 막대 위에 수치 표시
            for j, v in enumerate(value_counts.values):
                axes[i].text(value_counts.index[j], v + max(value_counts.values)*0.01, 
                           str(v), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        # 빈도 요약
        print(f"\n📊 이산형 변수 빈도 요약:")
        for col in discrete_cols:
            print(f"\n{col}:")
            value_counts = dataframe[col].value_counts()
            for value, count in value_counts.items():
                percentage = (count / len(dataframe)) * 100
                print(f"  {value}: {count}개 ({percentage:.1f}%)")

# 실행
numeric_columns = analysis_result['numeric_columns']
analyze_numeric_variables(df, numeric_columns)
```

#### Step 4: 범주형 변수 탐색적 분석

```python
# =====================================================
# 범주형 변수 심층 분석
# =====================================================

def analyze_categorical_variables(dataframe, categorical_cols):
    """
    범주형 변수들을 심층 분석하는 함수
    """
    
    print(f"🏷️ 범주형 변수 심층 분석")
    print(f"분석 대상: {categorical_cols}")
    
    for col in categorical_cols:
        print(f"\n" + "="*50)
        print(f"📋 {col} 분석")
        print(f"="*50)
        
        # 기본 정보
        unique_count = dataframe[col].nunique()
        missing_count = dataframe[col].isnull().sum()
        
        print(f"고유값 개수: {unique_count}")
        print(f"결측치 개수: {missing_count}")
        
        # 빈도 분석
        value_counts = dataframe[col].value_counts()
        
        print(f"\n빈도 분포:")
        for i, (value, count) in enumerate(value_counts.items()):
            percentage = (count / len(dataframe)) * 100
            print(f"{i+1:2d}. {str(value):20s}: {count:4d}개 ({percentage:5.1f}%)")
            
            if i >= 9:  # 상위 10개만 표시
                remaining = len(value_counts) - 10
                if remaining > 0:
                    print(f"    ... 기타 {remaining}개 카테고리")
                break
    
    # 시각화 (고유값이 적은 변수들만)
    viz_cols = [col for col in categorical_cols 
                if dataframe[col].nunique() <= 10]
    
    if viz_cols:
        n_cols = len(viz_cols)
        fig, axes = plt.subplots(1, n_cols, figsize=(6*n_cols, 5))
        
        if n_cols == 1:
            axes = [axes]
        
        fig.suptitle('📊 범주형 변수 분포 시각화', fontsize=16, fontweight='bold')
        
        for i, col in enumerate(viz_cols):
            value_counts = dataframe[col].value_counts()
            
            # 파이차트 또는 막대그래프 선택
            if len(value_counts) <= 5:
                # 파이차트
                axes[i].pie(value_counts.values, labels=value_counts.index, 
                           autopct='%1.1f%%', startangle=90)
                axes[i].set_title(f'{col} 분포 (파이차트)')
            else:
                # 막대그래프
                axes[i].bar(range(len(value_counts)), value_counts.values,
                           tick_label=value_counts.index, color='lightgreen', alpha=0.7)
                axes[i].set_title(f'{col} 분포 (막대그래프)')
                axes[i].set_ylabel('빈도')
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

# 실행
categorical_columns = analysis_result['categorical_columns']
analyze_categorical_variables(df, categorical_columns)

---

## 직접 해보기 / 연습 문제

### 연습 문제 1: 기본 데이터 탐색 (난이도: ⭐)

**목표**: 데이터의 기본 정보를 파악해보세요.

**단계별 가이드**:
1. 먼저 `df.shape`를 사용하여 데이터 크기를 확인하세요
2. `df.columns`로 컬럼 이름을 확인하세요
3. `df.dtypes`로 각 컬럼의 데이터 타입을 확인하세요

```python
# 연습용 데이터 로드
import pandas as pd

# 온라인에서 데이터 로드 (또는 로컬 파일 경로 사용)
df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

# TODO: 다음 정보들을 출력해보세요
# 1. 데이터셋의 크기 (행, 열)
print("데이터셋 크기:", df.shape)

# 2. 컬럼 이름 목록
print("컬럼 목록:", df.columns.tolist())

# 3. 각 컬럼의 데이터 타입
print("데이터 타입:")
print(df.dtypes)

# 4. 결측치가 있는 컬럼과 개수
print("결측치 정보:")
print(df.isnull().sum()[df.isnull().sum() > 0])
```

**💡 힌트**: 
- `df.shape[0]`은 행의 개수, `df.shape[1]`은 열의 개수입니다
- object 타입은 주로 문자열 데이터를 의미합니다

**✅ 예상 결과**:
```python
데이터셋 크기: (891, 12)  # 891행 12열
컬럼 목록: ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
```

---

### 연습 문제 2: 데이터 타입 분석 (난이도: ⭐⭐)

**목표**: 각 컬럼을 적절한 데이터 타입으로 분류해보세요.

**단계별 가이드**:
1. 수치형과 범주형을 구분하세요
2. 수치형 중에서도 연속형과 이산형을 구분하세요
3. 각 카테고리별로 분석 전략을 생각해보세요

```python
def analyze_data_types(dataframe):
    """
    데이터프레임을 분석하여 각 컬럼의 특성을 파악하는 함수
    """
    result = {
        'numeric_continuous': [],
        'numeric_discrete': [],
        'categorical': [],
        'text': []
    }
    
    for column in dataframe.columns:
        # TODO: 각 컬럼을 적절한 카테고리로 분류하는 로직을 작성하세요
        
        if dataframe[column].dtype in ['int64', 'float64']:
            # 수치형 데이터 처리
            unique_count = dataframe[column].nunique()
            if unique_count > 20:  # 임계값은 조정 가능
                result['numeric_continuous'].append(column)
            else:
                result['numeric_discrete'].append(column)
        else:
            # 문자형 데이터 처리
            unique_count = dataframe[column].nunique()
            if unique_count <= 10:  # 임계값은 조정 가능
                result['categorical'].append(column)
            else:
                result['text'].append(column)
    
    return result

# 함수 테스트
data_classification = analyze_data_types(df)
for data_type, columns in data_classification.items():
    print(f"{data_type}: {columns}")
```

**💡 힌트**: 
- 고유값이 많으면 연속형, 적으면 이산형/범주형일 가능성이 높습니다
- PassengerId는 숫자이지만 실제로는 식별자(범주형)입니다

**✅ 정답 해설**:
```python
# 올바른 분류 예시
correct_classification = {
    'numeric_continuous': ['Age', 'Fare'],           # 나이, 요금은 연속적 값
    'numeric_discrete': ['Survived', 'Pclass', 'SibSp', 'Parch'],  # 생존여부, 등급, 형제자매수 등
    'categorical': ['Sex', 'Embarked'],              # 성별, 승선항구
    'text': ['Name', 'Ticket', 'Cabin']            # 이름, 티켓번호, 객실번호
}

# PassengerId는 실제로는 식별자이므로 분석에서 제외하거나 text로 분류하는 것이 좋습니다
```

---

### 연습 문제 3: 결측치 패턴 분석 (난이도: ⭐⭐⭐)

**목표**: 결측치의 패턴을 분석하고 비즈니스적 의미를 해석해보세요.

**단계별 가이드**:
1. 각 컬럼의 결측치 비율을 계산하세요
2. 결측치가 가장 많은 컬럼을 찾고 이유를 생각해보세요
3. 결측치 간의 관계를 분석해보세요

```python
def analyze_missing_patterns(dataframe):
    """
    결측치 패턴을 분석하는 함수
    """
    missing_info = []
    
    for column in dataframe.columns:
        missing_count = dataframe[column].isnull().sum()
        if missing_count > 0:
            missing_info.append({
                '컬럼': column,
                '결측치_개수': missing_count,
                '결측치_비율(%)': round((missing_count / len(dataframe)) * 100, 2),
                '데이터타입': str(dataframe[column].dtype)
            })
    
    return pd.DataFrame(missing_info).sort_values('결측치_비율(%)', ascending=False)

# TODO: 함수를 실행하고 결과를 해석해보세요
missing_analysis = analyze_missing_patterns(df)
print("결측치 분석 결과:")
print(missing_analysis.to_string(index=False))

# TODO: 다음 질문에 답해보세요
print("\n🤔 결측치 해석 연습:")
print("1. 결측치가 가장 많은 컬럼은?")
print("2. 왜 Cabin 정보가 많이 없을까요?")
print("3. Age 정보가 없는 이유는?")
print("4. 이러한 결측치들을 어떻게 처리해야 할까요?")
```

**💡 힌트**: 
- 1912년 타이타닉 시대를 생각해보세요
- 모든 승객이 개인 객실을 가졌을까요?
- 나이 정보를 정확히 기록하기 어려운 이유가 있을까요?

**✅ 정답 해설**:
```python
# 예상 결과 및 해석
"""
결측치 분석 결과:
    컬럼  결측치_개수  결측치_비율(%)  데이터타입
   Cabin        687        77.10      object
     Age        177        19.87     float64
Embarked          2         0.22      object
    Fare          1         0.11     float64

비즈니스적 해석:
1. Cabin (77.1% 결측): 
   - 1등석 승객만 개인 객실 정보가 기록됨
   - 2, 3등석 승객들은 공동 숙소나 갑판에서 잠
   - 당시 사회적 계급 차이를 반영

2. Age (19.9% 결측):
   - 1912년에는 출생증명서가 보편적이지 않음
   - 이민자들의 정확한 나이 파악 어려움
   - 특히 어린이나 노인층에서 더 많은 결측

3. Embarked (0.2% 결측):
   - 거의 완전한 데이터, 소수의 기록 누락

4. Fare (0.1% 결측):
   - 거의 완전한 데이터, 아주 소수의 기록 누락
"""
```

---

### 연습 문제 4: 기초 통계량 해석 (난이도: ⭐⭐⭐)

**목표**: 통계량을 계산하고 비즈니스 인사이트를 도출해보세요.

```python
# 수치형 변수들의 기초 통계량
numeric_stats = df[['Age', 'Fare', 'SibSp', 'Parch']].describe()
print("기초 통계량:")
print(numeric_stats.round(2))

# TODO: 다음 질문에 답해보세요
print("\n=== 해석 연습 ===")

# 1. Age 컬럼에서 평균과 중앙값을 비교했을 때 어떤 특징을 발견할 수 있나요?
age_mean = df['Age'].mean()
age_median = df['Age'].median()
print(f"Age - 평균: {age_mean:.2f}, 중앙값: {age_median:.2f}")

# 2. Fare 컬럼의 표준편차가 평균보다 큽니다. 이것이 의미하는 바는?
fare_mean = df['Fare'].mean()
fare_std = df['Fare'].std()
print(f"Fare - 평균: {fare_std:.2f}, 표준편차: {fare_std:.2f}")

# 3. SibSp와 Parch의 75% 값이 같습니다. 이것이 의미하는 바는?
print(f"SibSp 75%: {df['SibSp'].quantile(0.75)}")
print(f"Parch 75%: {df['Parch'].quantile(0.75)}")

# TODO: 이상치를 찾아보세요
def find_outliers_detailed(data, column):
    """이상치를 찾고 상세 정보를 제공하는 함수"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    
    print(f"\n{column} 이상치 분석:")
    print(f"정상 범위: {lower_bound:.2f} ~ {upper_bound:.2f}")
    print(f"이상치 개수: {len(outliers)}개")
    
    if len(outliers) > 0:
        print(f"이상치 예시 (상위 5개):")
        print(outliers[column].nlargest(5).values)
    
    return outliers

# 각 수치형 컬럼에 대해 이상치 분석 실행
for col in ['Age', 'Fare', 'SibSp', 'Parch']:
    outliers = find_outliers_detailed(df, col)
```

**💡 힌트**: 
- 평균 > 중앙값이면 오른쪽 꼬리 분포 (높은 값들이 평균을 끌어올림)
- 표준편차가 평균보다 크면 데이터가 매우 넓게 퍼져있음
- 75% 값이 낮다는 것은 대부분의 사람들이 낮은 값을 가진다는 의미

**✅ 정답 해설**:
```python
"""
통계량 해석:

1. Age 분석:
   - 평균(29.7) > 중앙값(28.0): 오른쪽 꼬리 분포
   - 일부 고령 승객들이 평균을 끌어올림
   - 젊은 층이 더 많이 탑승했음을 시사

2. Fare 분석:
   - 표준편차(49.7) > 평균(32.2): 매우 큰 편차
   - 1등석과 3등석 간의 극심한 요금 차이
   - 사회적 계층 차이가 매우 컸음을 보여줌

3. SibSp, Parch:
   - 75% 값이 모두 1 이하
   - 대부분의 승객이 혼자 또는 소규모 가족 단위로 탑승
   - 대가족은 매우 드물었음

4. 이상치 의미:
   - Fare 이상치: 초고가 스위트룸 승객들
   - Age 이상치: 영유아나 고령 승객들
   - 이는 데이터 오류가 아닌 실제 극값일 가능성 높음
"""
```

---

### 연습 문제 5: 종합 프로젝트 (난이도: ⭐⭐⭐⭐)

**목표**: 배운 모든 내용을 종합하여 완전한 데이터 탐색 보고서를 작성해보세요.

**미션**: 타이타닉 데이터를 새로 받은 데이터 분석가가 되어 첫 분석 보고서를 작성하세요.

**요구사항**:
1. 데이터 개요 (크기, 구조, 주요 변수)
2. 데이터 품질 평가 (결측치, 이상치)
3. 각 변수별 특성 분석
4. 초기 인사이트와 추가 분석 방향 제시

```python
def create_comprehensive_report(dataframe):
    """
    종합 데이터 탐색 보고서를 생성하는 함수
    """
    
    print("🚢" + "="*80)
    print("📊 타이타닉 데이터셋 종합 분석 보고서")
    print("="*80)
    print(f"📅 분석 일시: {pd.Timestamp.now().strftime('%Y년 %m월 %d일 %H시 %M분')}")
    print(f"👨‍💼 분석가: [여러분의 이름]")
    
    # TODO: 여러분이 작성해보세요!
    
    # 1. 데이터 개요
    print(f"\n1️⃣ 데이터 개요")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    # TODO: 데이터 크기, 변수 개수, 관측 기간 등을 기술하세요
    
    # 2. 데이터 품질 평가
    print(f"\n2️⃣ 데이터 품질 평가")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    # TODO: 결측치 현황, 이상치 현황, 데이터 일관성을 평가하세요
    
    # 3. 변수별 특성 분석
    print(f"\n3️⃣ 변수별 특성 분석")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    # TODO: 각 변수의 분포, 특이사항, 비즈니스 의미를 분석하세요
    
    # 4. 주요 발견사항
    print(f"\n4️⃣ 주요 발견사항")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    # TODO: 3가지 이상의 주요 인사이트를 제시하세요
    
    # 5. 추천사항 및 다음 분석 방향
    print(f"\n5️⃣ 추천사항 및 다음 분석 방향")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    # TODO: 데이터 전처리 방향, 추가 분석 아이디어를 제시하세요

# 여러분만의 분석 보고서를 작성해보세요!
create_comprehensive_report(df)
```

**💡 평가 기준**:
- **완성도**: 모든 섹션이 충실히 작성되었는가?
- **정확성**: 통계량과 해석이 올바른가?
- **통찰력**: 데이터에서 의미있는 패턴을 발견했는가?
- **실무성**: 비즈니스 관점에서 유용한 정보를 제공하는가?

---

## 요약 / 핵심 정리

이번 Part에서 배운 핵심 내용을 정리하면 다음과 같습니다:

### 🎯 주요 학습 성과

**1. 체계적인 데이터 탐색 프로세스 마스터**
- ✅ 환경 설정부터 종합 분석까지 6단계 워크플로우 완성
- ✅ 실무에서 바로 적용 가능한 단계별 접근법 습득
- ✅ 데이터 타입별 맞춤형 분석 전략 수립 능력

**2. 고급 시각화와 해석 능력**
- ✅ 결측치 패턴을 4가지 관점으로 분석하는 종합 시각화
- ✅ 그래프의 색상, 크기, 패턴이 전달하는 메시지 완전 이해
- ✅ 수치 통계를 비즈니스 스토리로 변환하는 해석 능력

**3. 실무 중심의 문제 해결 역량**
- ✅ 타이타닉 데이터를 통한 실제 역사적 맥락과 데이터 연결
- ✅ 결측치와 이상치의 비즈니스적 의미 파악
- ✅ 데이터 품질 문제에 대한 체계적 대응 방안 수립

### 💡 핵심 Python 기술

**Pandas 마스터 기법:**
```python
# 데이터 구조 파악
df.shape, df.info(), df.describe()
df.head(), df.tail(), df.dtypes

# 결측치 분석
df.isnull().sum()
(df.isnull().sum() / len(df)) * 100

# 데이터 타입별 선택
df.select_dtypes(include=[np.number])
df.select_dtypes(include=['object'])

# 고유값과 빈도 분석
df[column].nunique()
df[column].value_counts()
```

**시각화 Best Practice:**
```python
# 결측치 히트맵
sns.heatmap(df.isnull(), cmap='viridis', cbar=True)

# 분포 분석 (히스토그램 + 통계선)
plt.hist(data, bins=30, alpha=0.7)
plt.axvline(data.mean(), color='red', linestyle='--')
plt.axvline(data.median(), color='green', linestyle='--')

# 이상치 탐지 (박스플롯)
plt.boxplot(data)
```

### 🔍 실무 적용 가이드라인

**1. 새로운 데이터셋을 받았을 때 체크리스트**
- [ ] 데이터 크기와 메모리 사용량 확인
- [ ] 각 컬럼의 의미와 데이터 타입 파악
- [ ] 결측치와 이상치 현황 점검
- [ ] 기초 통계량을 통한 분포 특성 이해
- [ ] 도메인 지식과 통계 결과의 일치성 검증

**2. 데이터 타입별 분석 전략**
- **연속형 수치**: 분포 분석 → 이상치 탐지 → 변환 필요성 검토
- **이산형 수치**: 빈도 분석 → 카테고리 통합 검토 → 순서성 확인
- **명목형 범주**: 빈도 분석 → 희소 카테고리 처리 → 인코딩 전략
- **순서형 범주**: 순서 확인 → 간격 일정성 검토 → 수치 변환 고려

**3. 흔한 함정과 대응 방법**
- **숫자 = 수치형 오해**: 학번, ID는 범주형으로 처리
- **결측치 무시**: 결측 패턴에서 의미있는 정보 추출
- **이상치 무조건 제거**: 도메인 지식으로 오류 vs 극값 구분
- **통계량만 보기**: 항상 시각화로 분포 형태 확인

### 🚀 다음 Part 예고

**Part 2: 데이터 시각화 기법**에서는 오늘 배운 탐색 기법을 한 단계 발전시켜보겠습니다:

- **📊 고급 시각화**: Matplotlib과 Seaborn의 모든 기능 활용
- **🎨 스토리텔링**: 데이터를 설득력 있는 시각적 스토리로 변환
- **🔄 인터랙티브**: Plotly로 동적 그래프 제작
- **📈 대시보드**: 종합적인 시각적 분석 대시보드 구축

데이터의 구조를 파악했다면, 이제 그 안에 숨겨진 패턴과 관계를 시각적으로 발견할 차례입니다!

---

**🎓 학습 완료 체크리스트**
- [ ] 6단계 데이터 탐색 프로세스를 혼자서 실행할 수 있다
- [ ] 결측치와 이상치의 원인을 비즈니스 관점에서 해석할 수 있다  
- [ ] 데이터 타입에 따른 적절한 분석 방법을 선택할 수 있다
- [ ] Python 코드로 기초 통계량을 계산하고 시각화할 수 있다
- [ ] 연습 문제 5개를 모두 완료했다

모든 항목이 체크되었다면 Part 2로 진행하세요! 🎉

---

**참고 자료**
- Pandas 공식 문서: https://pandas.pydata.org/docs/
- Seaborn 공식 문서: https://seaborn.pydata.org/
- Matplotlib 공식 문서: https://matplotlib.org/
- 실습 데이터: Titanic Dataset (Kaggle)
```

---
