# 8장 Part 1: 시계열 데이터의 특성과 전처리
**부제: 시간이 흐르는 데이터의 비밀을 AI와 함께 풀어보자**

## 학습 목표
이 Part를 완료한 후, 여러분은 다음을 할 수 있게 됩니다:
- **시계열 데이터의 4가지 핵심 특성**을 깊이 이해하고 실제 데이터에서 식별할 수 있다
- **7장에서 배운 AI 협업 기법**을 시계열 분석에 효과적으로 적용할 수 있다
- **시계열 분해 기법**을 사용하여 복잡한 패턴을 구성요소별로 분석할 수 있다
- **AI 기반 전처리 파이프라인**을 구축하여 시계열 데이터 품질을 향상시킬 수 있다
- **비즈니스 문제를 시계열 분석 문제로 변환**하고 해결할 수 있다

---

## 이번 Part 미리보기: 시간의 흐름 속에 숨겨진 패턴 찾기

🌊 **시계열 데이터란?**
시계열 데이터는 마치 일기장처럼 시간 순서대로 기록된 데이터입니다. 여러분이 매일 기록하는 체온, 주식 가격의 변화, 유튜브 조회수, 심지어 여러분의 하루 기분도 모두 시계열 데이터가 될 수 있습니다.

**🎯 왜 시계열 분석이 중요할까요?**
- **예측의 힘**: 내일의 날씨, 다음 달 매출, 주식 가격 등을 예측할 수 있습니다
- **패턴 발견**: 계절별 트렌드, 주기적 변화, 이상 현상을 찾을 수 있습니다  
- **의사결정 지원**: 언제 광고를 늘려야 하는지, 재고를 얼마나 준비해야 하는지 알 수 있습니다

**🤖 AI와 함께하는 시계열 분석**
7장에서 배운 AI 협업 기법을 시계열에 적용하면:
- **복잡한 패턴을 쉬운 말로 설명**해 줍니다
- **자동으로 이상값을 찾아**줍니다
- **최적의 전처리 방법을 추천**해 줍니다
- **비즈니스 관점에서 해석**해 줍니다

---

> 🚀 **8장에서 마스터할 AI 협업 기법**
> 
> **📊 스마트 패턴 해석**: AI가 복잡한 그래프를 쉬운 말로 설명
> **🔍 지능형 이상 탐지**: 사람이 놓칠 수 있는 미묘한 변화 포착
> **📈 예측 모델 최적화**: 프롬프트로 예측 성능 향상
> **⚡ 효율적인 분석**: 자동화와 수동 작업의 완벽한 조화
> **💡 비즈니스 인사이트**: 숫자를 행동 가능한 전략으로 변환

---

## 📚 핵심 용어 해설 (시계열 분석 기초 사전)

시계열 분석을 시작하기 전에 꼭 알아야 할 핵심 용어들을 쉽게 설명해드리겠습니다.

### 🕐 시계열 기본 개념

**📖 시계열(Time Series)**
- **의미**: 시간 순서대로 관측된 데이터의 연속
- **쉬운 예시**: 여러분이 매일 측정하는 체중 기록, 하루 종일의 온도 변화
- **특징**: 시간이 X축, 관측값이 Y축인 그래프로 표현

**📖 관측값(Observation)**
- **의미**: 특정 시점에 측정된 하나의 데이터 값
- **쉬운 예시**: 2024년 1월 1일 오전 9시의 온도 15℃
- **중요성**: 시계열의 기본 구성 단위

**📖 시점(Time Point)**
- **의미**: 데이터가 관측된 정확한 시간
- **쉬운 예시**: 2024-01-01 09:00:00
- **특징**: 규칙적일 수도 있고(매일, 매시간) 불규칙적일 수도 있음

### 🔄 시계열의 4가지 핵심 특성

**📖 트렌드(Trend)**
- **의미**: 시계열의 장기적인 방향성
- **쉬운 예시**: 
  - 상승 트렌드: 여러분의 키가 매년 자라는 것
  - 하락 트렌드: 휴대폰 가격이 점점 저렴해지는 것
  - 수평 트렌드: 여러분의 몸무게가 일정하게 유지되는 것
- **비즈니스 의미**: 매출 증가, 고객 감소, 비용 절감 등

**📖 계절성(Seasonality)**
- **의미**: 일정한 주기로 반복되는 패턴
- **쉬운 예시**:
  - 아이스크림 판매량: 여름에 높고 겨울에 낮음
  - 전력 사용량: 여름/겨울에 높고 봄/가을에 낮음
  - 학용품 판매: 3월, 9월에 급증
- **주기 종류**: 일간, 주간, 월간, 분기별, 연간

**📖 순환성(Cyclical)**
- **의미**: 불규칙한 주기로 반복되는 장기 패턴
- **쉬운 예시**:
  - 경기 순환: 호황 → 침체 → 회복 (주기가 매번 다름)
  - 부동산 시장: 상승 → 하락 → 회복 (몇 년에서 십여 년)
- **계절성과 차이점**: 주기가 일정하지 않고 더 장기간

**📖 불규칙성/잡음(Irregular/Noise)**
- **의미**: 예측할 수 없는 무작위 변동
- **쉬운 예시**:
  - 갑작스러운 뉴스에 따른 주가 변동
  - 코로나19 같은 예상치 못한 사건
  - 측정 오차나 데이터 입력 실수
- **특징**: 패턴이 없고 예측 불가능

### 📊 시계열 분석 기법

**📖 정상성(Stationarity)**
- **의미**: 시계열의 통계적 특성(평균, 분산)이 시간에 따라 변하지 않는 성질
- **쉬운 예시**:
  - 정상: 매일 오후 2시 교실 온도 (계절 제외하고는 비슷)
  - 비정상: 여러분의 키 (시간이 지나면서 계속 변함)
- **중요성**: 많은 예측 모델이 정상성을 가정함

**📖 자기상관(Autocorrelation)**
- **의미**: 시계열에서 과거 값과 현재 값 간의 상관관계
- **쉬운 예시**:
  - 오늘 기온과 어제 기온은 비슷할 가능성이 높음
  - 이번 달 매출과 지난 달 매출 간의 관계
- **활용**: 과거 데이터로 미래 예측의 근거

**📖 지연(Lag)**
- **의미**: 현재 시점으로부터 몇 단계 이전의 시점
- **쉬운 예시**:
  - 1일 지연: 어제 값
  - 7일 지연: 일주일 전 같은 요일 값
  - 12개월 지연: 작년 같은 달 값
- **활용**: 계절성 분석, 예측 모델 구축

---

## 🎨 이미지 생성 프롬프트

```
시계열 데이터 개념 설명 인포그래픽: 
4개의 섹션으로 구성된 교육용 다이어그램
- 트렌드: 상승하는 화살표와 성장 그래프
- 계절성: 순환하는 원형 패턴과 계절 아이콘
- 순환성: 불규칙한 파도 모양의 곡선
- 불규칙성: 랜덤한 점들과 노이즈 패턴
깔끔한 교육용 스타일, 밝은 색상, 학생 친화적 디자인
```

---

## 1. 시계열 데이터의 핵심 개념과 실제 예시

이제 실제 코드를 통해 시계열 데이터의 특성을 직접 확인해보겠습니다. 마치 과학 실험처럼 단계별로 진행해보겠습니다!

```python
# 필요한 라이브러리 불러오기 (도구 준비하기)
import pandas as pd           # 데이터 처리의 핵심 도구
import numpy as np           # 수학 계산을 위한 도구
import matplotlib.pyplot as plt  # 그래프 그리기 도구
import seaborn as sns        # 예쁜 그래프 만들기 도구
from datetime import datetime, timedelta  # 날짜/시간 다루기 도구
import warnings
warnings.filterwarnings('ignore')  # 경고 메시지 숨기기

# 그래프 스타일 설정 (예쁘게 만들기)
plt.style.use('seaborn-v0_8')  # 깔끔한 스타일 적용
plt.rcParams['figure.figsize'] = (12, 8)  # 그래프 크기 설정
plt.rcParams['font.size'] = 10  # 글자 크기 설정
plt.rcParams['font.family'] = 'DejaVu Sans'  # 글꼴 설정
```

**🔍 코드 해설:**
- `pandas`: 엑셀처럼 데이터를 표 형태로 다루는 도구
- `numpy`: 복잡한 수학 계산을 빠르게 해주는 도구  
- `matplotlib/seaborn`: 데이터를 그래프로 그려주는 도구
- `warnings.filterwarnings('ignore')`: 불필요한 경고 메시지를 숨겨서 결과만 깔끔하게 보기

### 1.1 시계열 데이터 생성 및 특성 분석 클래스

```python
class TimeSeriesFoundation:
    """
    시계열 데이터의 기초 개념을 배우기 위한 클래스
    
    🎯 목적: 다양한 시계열 패턴을 생성하고 특성을 분석
    📚 학습 내용: 트렌드, 계절성, 순환성, 불규칙성
    """
    
    def __init__(self):
        """클래스 초기화 - 빈 저장소들을 만들어 놓습니다"""
        self.examples = {}  # 생성한 예시 데이터들을 저장할 공간
        self.components = {}  # 시계열 구성요소들을 저장할 공간
        
    def create_sample_data(self):
        """
        다양한 종류의 시계열 예제 데이터 생성
        
        🎯 목적: 실제 현실에서 볼 수 있는 다양한 패턴의 시계열 만들기
        📊 생성할 데이터: 매출, 주식가격, 웹사이트 방문자, 온도
        """
        
        print("📊 다양한 시계열 데이터 생성 중...")
        
        # 1단계: 날짜 범위 설정 (최근 3년간의 일별 데이터)
        dates = pd.date_range(
            start='2021-01-01',    # 시작일
            end='2023-12-31',      # 종료일  
            freq='D'               # 빈도: D(일별), H(시간별), M(월별) 등
        )
        print(f"   📅 날짜 범위: {dates[0].date()} ~ {dates[-1].date()}")
        print(f"   📊 총 데이터 포인트: {len(dates):,}개")
        
        # 2단계: 다양한 패턴의 시계열 데이터 생성
        
        # 🏪 매출 데이터 (트렌드 + 계절성 + 주간패턴 + 노이즈)
        print("\n🏪 매출 데이터 생성 중...")
        
        # 트렌드: 3년간 100에서 200으로 선형 증가
        trend = np.linspace(100, 200, len(dates))
        print(f"   📈 트렌드: {trend[0]:.1f} → {trend[-1]:.1f} (연평균 {(trend[-1]-trend[0])/3:.1f} 증가)")
        
        # 계절성: 1년 주기로 반복되는 패턴 (여름에 높고 겨울에 낮음)
        seasonal = 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
        print(f"   🔄 계절성: 진폭 ±{20} (여름 🌞 높음, 겨울 ❄️ 낮음)")
        
        # 주간 패턴: 주말에 낮고 평일에 높음
        weekly = 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
        print(f"   📅 주간패턴: 진폭 ±{5} (평일 높음, 주말 낮음)")
        
        # 무작위 노이즈: 예측할 수 없는 변동
        noise = np.random.normal(0, 10, len(dates))  # 평균 0, 표준편차 10
        print(f"   🎲 노이즈: 표준편차 {10} (무작위 변동)")
        
        # 모든 구성요소 합치기
        sales_data = trend + seasonal + weekly + noise
        print(f"   💰 최종 매출 범위: {sales_data.min():.1f} ~ {sales_data.max():.1f}")
        
        # 📈 주식 가격 데이터 (랜덤워크 - 주식의 전형적인 패턴)
        print("\n📈 주식 가격 데이터 생성 중...")
        
        # 일일 수익률: 평균 0.1%, 표준편차 2% (실제 주식과 비슷)
        returns = np.random.normal(0.001, 0.02, len(dates))
        print(f"   📊 일일 수익률: 평균 {0.001*100:.1f}%, 표준편차 {0.02*100:.1f}%")
        
        # 누적곱으로 가격 생성: 100원에서 시작
        price_data = 100 * np.exp(np.cumsum(returns))
        print(f"   💎 주가 범위: {price_data.min():.0f}원 ~ {price_data.max():.0f}원")
        
        # 🌐 웹사이트 방문자 수 (계단식 성장 + 계절성 + 주간패턴)
        print("\n🌐 웹사이트 방문자 데이터 생성 중...")
        
        base_visitors = 1000  # 기본 방문자 수
        
        # 성장 단계: 4분기로 나누어 단계적 성장
        growth_phases = np.repeat([1.0, 1.2, 1.5, 1.8], len(dates)//4 + 1)[:len(dates)]
        print(f"   📈 성장 단계: 1.0x → 1.2x → 1.5x → 1.8x")
        
        # 계절성: 연말연시와 여름휴가철에 높음
        visitors_seasonal = 200 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
        
        # 주간 패턴: 주중이 주말보다 높음
        visitors_weekly = 300 * (np.arange(len(dates)) % 7 < 5)  # 월~금요일만 True
        
        # 포아송 노이즈: 방문자 수는 음수가 될 수 없으므로
        visitors_noise = np.random.poisson(50, len(dates))
        
        visitors_data = base_visitors * growth_phases + visitors_seasonal + visitors_weekly + visitors_noise
        print(f"   👥 방문자 범위: {visitors_data.min():.0f}명 ~ {visitors_data.max():.0f}명")
        
        # 🌡️ 온도 데이터 (강한 계절성 + 지구온난화 트렌드)
        print("\n🌡️ 온도 데이터 생성 중...")
        
        # 지구온난화 효과: 3년간 약 0.3도 상승
        temp_trend = 0.1 * np.arange(len(dates)) / 365.25
        
        # 강한 계절성: 여름 더위, 겨울 추위
        temp_seasonal = 15 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25 - np.pi/2)
        
        # 일일 변동: 하루 중 온도 변화
        temp_daily = 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 1)
        
        # 날씨 노이즈: 예측하기 어려운 기상 변동
        temp_noise = np.random.normal(0, 3, len(dates))
        
        temperature_data = 20 + temp_trend + temp_seasonal + temp_daily + temp_noise
        print(f"   🌡️ 온도 범위: {temperature_data.min():.1f}℃ ~ {temperature_data.max():.1f}℃")
        
        # 3단계: 모든 데이터를 DataFrame으로 정리
        self.examples = pd.DataFrame({
            'date': dates,
            'sales': sales_data,
            'stock_price': price_data,
            'website_visitors': np.maximum(visitors_data, 0),  # 음수 방지
            'temperature': temperature_data
        })
        
        # 날짜를 인덱스로 설정 (시계열 분석에 필수!)
        self.examples.set_index('date', inplace=True)
        
        print(f"\n✅ 시계열 데이터 생성 완료!")
        print(f"   📊 생성된 변수: {list(self.examples.columns)}")
        print(f"   📅 기간: {len(self.examples)}일")
        
        return self.examples
```

**🔍 코드 해설:**

**1. 클래스 구조**
- `__init__`: 클래스가 생성될 때 실행되는 초기화 함수
- `self.examples`: 생성한 데이터를 저장할 딕셔너리
- `self.components`: 나중에 사용할 구성요소 분석 결과 저장

**2. 날짜 생성**
- `pd.date_range()`: 시작일부터 종료일까지 규칙적인 날짜 생성
- `freq='D'`: 일별 데이터 (다른 옵션: 'H'=시간별, 'M'=월별)

**3. 각 시계열의 구성요소**
- **트렌드**: `np.linspace(100, 200, len(dates))` - 시간에 따라 선형적으로 증가
- **계절성**: `np.sin(2π × 시간 / 주기)` - 사인 함수로 주기적 패턴 생성
- **노이즈**: `np.random.normal()` - 정규분포를 따르는 무작위 변동

**4. 실제 현상 모델링**
- **매출**: 성장 추세 + 계절 효과 + 주간 패턴
- **주식**: 랜덤워크 모델 (금융에서 널리 사용)
- **방문자**: 단계적 성장 + 주중/주말 차이  
- **온도**: 계절 변화 + 지구온난화 효과

### 1.2 시계열 데이터의 4가지 핵심 특성 분석

이제 생성한 데이터를 통해 시계열의 핵심 특성들을 실제로 확인해보겠습니다. 마치 의사가 환자를 진단하듯 데이터를 체계적으로 분석해보겠습니다!

```python
    def demonstrate_time_series_properties(self):
        """
        시계열 데이터의 핵심 특성 시연
        
        🎯 목적: 4가지 핵심 특성을 실제 데이터로 확인
        📚 분석 내용: 시간순서, 자기상관성, 비정상성, 계절성
        """
        
        if self.examples.empty:
            self.create_sample_data()
        
        print("\n🔍 시계열 데이터의 핵심 특성 분석")
        print("=" * 60)
        
        # 🕐 특성 1: 시간 순서의 중요성 (Temporal Ordering)
        print("\n1️⃣ 시간 순서의 중요성")
        print("   💡 시계열에서는 데이터의 순서가 매우 중요합니다!")
        print("   📚 만약 순서를 섞으면 원래 패턴이 완전히 사라집니다.")
        
        # 실험: 원본 vs 섞인 데이터의 트렌드 비교
        sales_original = self.examples['sales'][:30]  # 첫 30일 데이터
        sales_shuffled = self.examples['sales'][:30].sample(frac=1).reset_index(drop=True)
        
        # 트렌드 강도 계산 (시간과의 상관계수)
        original_trend = sales_original.corr(pd.Series(range(30)))
        shuffled_trend = sales_shuffled.corr(pd.Series(range(30)))
        
        print(f"   📈 원본 데이터의 트렌드 강도: {original_trend:.3f}")
        print(f"   🔀 섞인 데이터의 트렌드 강도: {shuffled_trend:.3f}")
        print(f"   🎯 결론: 순서가 중요함을 {'확인!' if abs(original_trend) > abs(shuffled_trend) else '재검토 필요'}")
        
        # 🔄 특성 2: 자기상관성 (Autocorrelation)
        print("\n2️⃣ 자기상관성 (과거가 현재에 미치는 영향)")
        print("   💡 시계열에서는 '어제'가 '오늘'을 예측하는 중요한 단서입니다!")
        print("   📚 이를 자기상관(Autocorrelation)이라고 합니다.")
        
        # 다양한 지연(lag)에서의 자기상관 계산
        lags_to_check = [1, 7, 30, 365]  # 1일, 1주, 1달, 1년 전
        lag_names = ['1일 전', '1주 전', '1달 전', '1년 전']
        
        print(f"   📊 매출 데이터의 자기상관 분석:")
        for lag, name in zip(lags_to_check, lag_names):
            if lag < len(self.examples):
                autocorr = self.examples['sales'].autocorr(lag=lag)
                correlation_strength = "강함" if abs(autocorr) > 0.7 else "보통" if abs(autocorr) > 0.3 else "약함"
                print(f"      {name}: {autocorr:.3f} ({correlation_strength})")
            else:
                print(f"      {name}: 데이터 부족으로 계산 불가")
        
        # 🔄 특성 3: 비정상성 (Non-stationarity)
        print("\n3️⃣ 비정상성 (시간에 따라 변하는 특성)")
        print("   💡 시계열의 평균, 분산, 상관구조가 시간에 따라 변합니다!")
        print("   📚 이는 일반 통계와 다른 시계열만의 특별한 성질입니다.")
        
        # 연도별 통계 비교
        yearly_stats = {}
        for year in [2021, 2022, 2023]:
            year_data = self.examples['sales'][str(year)]
            yearly_stats[year] = {
                'mean': year_data.mean(),
                'std': year_data.std(),
                'count': len(year_data)
            }
        
        print(f"   📊 연도별 매출 통계 비교:")
        for year, stats in yearly_stats.items():
            print(f"      {year}년: 평균 {stats['mean']:.1f}, 표준편차 {stats['std']:.1f}")
        
        # 변화량 계산
        mean_change = yearly_stats[2023]['mean'] - yearly_stats[2021]['mean']
        std_change = yearly_stats[2023]['std'] - yearly_stats[2021]['std']
        print(f"   📈 3년간 변화: 평균 {mean_change:+.1f}, 표준편차 {std_change:+.1f}")
        print(f"   🎯 결론: {'비정상성 확인!' if abs(mean_change) > 10 or abs(std_change) > 2 else '거의 정상적'}")
        
        # 🔄 특성 4: 계절성과 주기성 (Seasonality & Periodicity)
        print("\n4️⃣ 계절성과 주기성 (반복되는 패턴)")
        print("   💡 많은 시계열은 일정한 주기로 반복되는 패턴을 가집니다!")
        print("   📚 월별, 분기별, 연별 등 다양한 주기가 있습니다.")
        
        # 월별 평균 패턴 분석
        monthly_avg = self.examples['sales'].groupby(self.examples.index.month).mean()
        monthly_names = ['1월', '2월', '3월', '4월', '5월', '6월', 
                        '7월', '8월', '9월', '10월', '11월', '12월']
        
        max_month = monthly_avg.idxmax()
        min_month = monthly_avg.idxmin()
        seasonality_strength = (monthly_avg.max() - monthly_avg.min()) / monthly_avg.mean()
        
        print(f"   📊 월별 매출 패턴 분석:")
        print(f"      최고 매출 월: {monthly_names[max_month-1]} ({monthly_avg[max_month]:.1f})")
        print(f"      최저 매출 월: {monthly_names[min_month-1]} ({monthly_avg[min_month]:.1f})")
        print(f"      계절성 강도: {seasonality_strength:.1%}")
        
        # 계절성 강도에 따른 해석
        if seasonality_strength > 0.2:
            season_level = "매우 강함"
        elif seasonality_strength > 0.1:
            season_level = "보통"
        else:
            season_level = "약함"
        
        print(f"   🎯 결론: {season_level} 계절성 패턴 확인!")
        
        # 요일별 패턴도 확인
        print(f"\n   📅 요일별 패턴 분석:")
        daily_avg = self.examples['sales'].groupby(self.examples.index.dayofweek).mean()
        day_names = ['월', '화', '수', '목', '금', '토', '일']
        
        for i, day_name in enumerate(day_names):
            print(f"      {day_name}요일: {daily_avg[i]:.1f}")
        
        weekday_avg = daily_avg[:5].mean()  # 평일 평균
        weekend_avg = daily_avg[5:].mean()  # 주말 평균
        weekly_pattern = abs(weekday_avg - weekend_avg) / daily_avg.mean()
        
        print(f"   📊 평일 평균: {weekday_avg:.1f}, 주말 평균: {weekend_avg:.1f}")
        print(f"   📈 주간 패턴 강도: {weekly_pattern:.1%}")
        
        return {
            'temporal_ordering': {'original_trend': original_trend, 'shuffled_trend': shuffled_trend},
            'autocorrelation': dict(zip(lag_names, [self.examples['sales'].autocorr(lag=lag) if lag < len(self.examples) else None for lag in lags_to_check])),
            'nonstationarity': yearly_stats,
            'seasonality': {'monthly': monthly_avg.to_dict(), 'strength': seasonality_strength},
            'weekly_pattern': {'daily_avg': daily_avg.to_dict(), 'strength': weekly_pattern}
        }
```

**🔍 코드 해설:**

**1. 시간 순서의 중요성**
- `sample(frac=1)`: 데이터를 무작위로 섞는 함수
- `corr()`: 두 변수 간의 상관계수 계산 (-1~1 범위)
- 시간 순서가 중요한 이유: 순서를 섞으면 트렌드가 사라짐

**2. 자기상관성 분석**
- `autocorr(lag=n)`: n시점 이전 값과의 상관계수
- lag=1: 하루 전, lag=7: 일주일 전 등
- 높은 자기상관 = 과거 값이 현재 예측에 유용

**3. 비정상성 확인**
- 연도별로 평균과 분산을 비교
- 시간에 따라 변하면 비정상, 일정하면 정상
- 대부분의 실제 데이터는 비정상성을 가짐

**4. 계절성 분석**
- `groupby(index.month)`: 월별로 그룹화
- `groupby(index.dayofweek)`: 요일별로 그룹화 (0=월요일, 6=일요일)
- 계절성 강도 = (최대값-최소값)/평균값

### 1.3 시계열 패턴 시각화

이제 분석한 특성들을 그래프로 확인해보겠습니다. "백문이 불여일견"이라는 말처럼, 직접 보는 것이 가장 이해하기 쉽습니다!

```python
    def visualize_time_series_components(self):
        """
        시계열 구성요소 시각화
        
        🎯 목적: 4가지 다른 패턴의 시계열을 그래프로 비교
        📊 차트 구성: 2x2 격자로 4개 시계열 동시 표시
        """
        
        if self.examples.empty:
            self.create_sample_data()
        
        print("\n🎨 시계열 패턴 시각화 시작...")
        
        # 그래프 캔버스 준비 (2행 2열 = 4개 차트)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🕐 시계열 데이터의 다양한 패턴 분석', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # 월별 데이터로 리샘플링 (너무 많은 점들로 인한 혼잡함 방지)
        monthly_data = self.examples.resample('M').agg({
            'sales': 'mean',           # 매출: 월평균
            'stock_price': 'last',     # 주가: 월말 가격
            'website_visitors': 'mean', # 방문자: 월평균
            'temperature': 'mean'       # 온도: 월평균
        })
        
        print(f"📊 월별 데이터로 집계: {len(monthly_data)}개 포인트")
        
        # 1️⃣ 매출 데이터 (트렌드 + 계절성)
        ax1 = axes[0, 0]
        sales_line = ax1.plot(monthly_data.index, monthly_data['sales'], 
                             color='#2E8B57', linewidth=2.5, label='월평균 매출')
        ax1.set_title('📈 매출 데이터 (상승 트렌드 + 계절성)', fontweight='bold', pad=20)
        ax1.set_ylabel('매출액 (만원)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 트렌드 라인 추가 (선형 회귀)
        x_numeric = np.arange(len(monthly_data))
        z = np.polyfit(x_numeric, monthly_data['sales'].values, 1)  # 1차 다항식 피팅
        p = np.poly1d(z)  # 다항식 함수 생성
        trend_line = ax1.plot(monthly_data.index, p(x_numeric), 
                             "--", color='red', alpha=0.8, linewidth=2, label=f'트렌드 (기울기: {z[0]:.2f})')
        
        ax1.legend(loc='upper left')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2️⃣ 주식 가격 (랜덤워크 패턴)
        ax2 = axes[0, 1]
        stock_line = ax2.plot(monthly_data.index, monthly_data['stock_price'], 
                             color='#4682B4', linewidth=2.5, label='월말 주가')
        ax2.set_title('💹 주식 가격 (랜덤워크 패턴)', fontweight='bold', pad=20)
        ax2.set_ylabel('주가 (원)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # 변동성 정보 추가
        price_volatility = monthly_data['stock_price'].std() / monthly_data['stock_price'].mean()
        ax2.text(0.02, 0.95, f'변동성: {price_volatility:.1%}', 
                transform=ax2.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
        
        ax2.legend(loc='upper left')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3️⃣ 웹사이트 방문자 (계단식 성장)
        ax3 = axes[1, 0]
        visitors_line = ax3.plot(monthly_data.index, monthly_data['website_visitors'], 
                                color='#FF6347', linewidth=2.5, label='월평균 방문자')
        ax3.set_title('🌐 웹사이트 방문자 (계단식 성장)', fontweight='bold', pad=20)
        ax3.set_ylabel('방문자 수 (명)', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # 성장률 계산 및 표시
        first_year_avg = monthly_data['website_visitors'][:12].mean()
        last_year_avg = monthly_data['website_visitors'][-12:].mean()
        growth_rate = (last_year_avg - first_year_avg) / first_year_avg
        
        ax3.text(0.02, 0.95, f'3년 성장률: {growth_rate:.1%}', 
                transform=ax3.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.7))
        
        ax3.legend(loc='upper left')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4️⃣ 온도 데이터 (강한 계절성)
        ax4 = axes[1, 1]
        temp_line = ax4.plot(monthly_data.index, monthly_data['temperature'], 
                            color='#DAA520', linewidth=2.5, label='월평균 온도')
        ax4.set_title('🌡️ 온도 데이터 (강한 계절성)', fontweight='bold', pad=20)
        ax4.set_ylabel('온도 (°C)', fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        # 계절성 정보 추가
        temp_range = monthly_data['temperature'].max() - monthly_data['temperature'].min()
        ax4.text(0.02, 0.95, f'연간 온도 범위: {temp_range:.1f}°C', 
                transform=ax4.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.7))
        
        ax4.legend(loc='upper left')
        ax4.tick_params(axis='x', rotation=45)
        
        # 전체 레이아웃 조정
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)  # 제목을 위한 여백
        plt.show()
        
        # 📊 패턴별 특성 요약
        print(f"\n📊 시계열 패턴 특성 요약:")
        print(f"   📈 매출: 트렌드 기울기 {z[0]:.2f}, 계절성 강도 {(monthly_data['sales'].max()-monthly_data['sales'].min())/monthly_data['sales'].mean():.1%}")
        print(f"   💹 주가: 변동성 {price_volatility:.1%}, 최종 수익률 {(monthly_data['stock_price'].iloc[-1]/monthly_data['stock_price'].iloc[0]-1)*100:+.1f}%")
        print(f"   🌐 방문자: 성장률 {growth_rate:.1%}, 최종 방문자 {monthly_data['website_visitors'].iloc[-1]:.0f}명")
        print(f"   🌡️ 온도: 연간 범위 {temp_range:.1f}°C, 평균 온도 {monthly_data['temperature'].mean():.1f}°C")
        
        # 🎨 이미지 생성 프롬프트
        print(f"\n🎨 시각화 이미지 생성 프롬프트:")
        print(f"   'Four time series analysis charts in 2x2 grid layout showing:")
        print(f"   - Sales data with upward trend and seasonality (green line)")
        print(f"   - Stock price with random walk pattern (blue line)")  
        print(f"   - Website visitors with step growth pattern (red line)")
        print(f"   - Temperature data with strong seasonality (orange line)")
        print(f"   Clean modern analytical dashboard style, professional data visualization'")
        
        return monthly_data
```

**🔍 코드 해설:**

**1. 데이터 리샘플링**
- `resample('M')`: 일별 데이터를 월별로 집계
- `agg()`: 각 컬럼별로 다른 집계 방법 적용
- 목적: 그래프가 너무 복잡해지는 것을 방지

**2. 서브플롯 구성**
- `plt.subplots(2, 2)`: 2행 2열의 격자 형태
- `axes[0, 0]`: 첫 번째 행, 첫 번째 열 차트
- 각 차트마다 다른 색상과 스타일 적용

**3. 트렌드 라인 추가**
- `np.polyfit()`: 다항식 회귀로 트렌드 계산
- `np.poly1d()`: 계산된 계수로 함수 생성
- 빨간 점선으로 트렌드 시각화

**4. 정보 박스 추가**
- `ax.text()`: 그래프에 텍스트 추가
- `transform=ax.transAxes`: 상대적 위치 지정 (0~1 범위)
- `bbox`: 텍스트 주변에 박스 그리기

**5. 패턴별 특성 계산**
- **변동성**: 표준편차/평균 (상대적 변동 크기)
- **성장률**: (마지막값-첫값)/첫값
- **계절성 강도**: (최댓값-최솟값)/평균값

---

## 2. AI 협업을 통한 시계열 패턴 해석

이제 7장에서 배운 AI 협업 기법을 시계열 분석에 적용해보겠습니다. AI는 복잡한 패턴을 분석하고, 우리는 비즈니스 맥락에서 해석하는 역할 분담을 해보겠습니다!

### 📚 AI 협업 핵심 개념 복습

**CLEAR 프롬프트 원칙** (7장에서 학습)
- **C**ontext (맥락): 분석 목적과 상황 설명
- **L**ength (길이): 적절한 분석 범위 지정  
- **E**xamples (예시): 구체적인 분석 기준 제시
- **A**ctionable (실행가능): 바로 활용할 수 있는 결과
- **R**ole (역할): AI의 전문가 역할 정의

### 2.1 시계열 전용 AI 분석 시스템 구축

```python
class AIAssistedTimeSeriesAnalysis:
    """
    AI 협업 기반 시계열 분석 클래스
    
    🎯 목적: 7장 CLEAR 원칙을 시계열 분석에 특화하여 적용
    🤖 AI 역할: 패턴 인식, 이상 탐지, 트렌드 해석
    👨‍💼 인간 역할: 비즈니스 맥락 해석, 전략 수립
    """
    
    def __init__(self):
        """AI 협업 시스템 초기화"""
        print("🤖 AI 시계열 분석 시스템 초기화 중...")
        
        # 7장에서 배운 CLEAR 원칙을 시계열에 적용한 프롬프트 템플릿
        self.prompt_templates = {
            'pattern_analysis': self._create_pattern_analysis_prompt(),
            'anomaly_detection': self._create_anomaly_detection_prompt(), 
            'trend_interpretation': self._create_trend_interpretation_prompt(),
            'seasonality_analysis': self._create_seasonality_analysis_prompt(),
            'business_insights': self._create_business_insights_prompt()
        }
        
        print(f"✅ {len(self.prompt_templates)}개 전문 분석 템플릿 준비 완료!")
        
    def _create_pattern_analysis_prompt(self) -> str:
        """
        시계열 패턴 분석용 CLEAR 프롬프트
        
        📝 설계 원리:
        - Context: 비즈니스 시계열 분석 상황
        - Length: 구체적이고 실용적인 분석 범위  
        - Examples: 명확한 분석 기준과 형식
        - Actionable: 바로 활용 가능한 인사이트
        - Role: 시계열 분석 전문가 역할
        """
        
        return """
🔬 **시계열 패턴 분석 전문가 AI**

**👤 당신의 역할**: 10년 경력의 시계열 데이터 분석 전문가
**🎯 분석 목적**: 비즈니스 의사결정을 위한 실용적 시계열 패턴 해석

**📋 Context (분석 맥락)**:
- 실제 비즈니스 환경의 시계열 데이터 분석
- 기술팀과 경영진 모두 이해할 수 있는 설명 필요
- 데이터 기반 의사결정과 전략 수립이 목적

**📏 Length (분석 범위)**:
다음 5개 영역을 각각 2-3문장씩 분석:
1. 전체적인 패턴 특성 (트렌드 방향과 강도)
2. 주기적 패턴 존재 여부 (계절성, 주간성 등)
3. 주요 변화점이나 이상 현상
4. 데이터 품질과 신뢰성
5. 비즈니스 시사점과 의미

**📊 Examples (분석 기준)**:
- 트렌드: "상승/하락/횡보", "선형/비선형", "가속/감속"
- 계절성: "연간/월간/주간", "강도 (약함/보통/강함)", "규칙성"
- 변화점: "급증/급락", "구조적 변화", "외부 요인 추정"
- 품질: "결측값 정도", "이상값 수준", "노이즈 정도"

**⚡ Actionable (실행 가능한 결과)**:
1. 핵심 패턴 요약 (3가지 주요 특성)
2. 주요 발견사항 (비즈니스 관련)
3. 권장 후속 분석 방향
4. 예측 모델링 시 고려사항
5. 리스크 요인과 기회 요소

**🎯 분석할 데이터**:
- 시계열명: {series_name}
- 기간: {time_period}  
- 데이터 포인트 수: {data_points}
- 주요 통계: {statistics}

위 정보를 바탕으로 체계적이고 실용적인 분석을 제공해주세요.
"""
    
    def _create_anomaly_detection_prompt(self) -> str:
        """이상 탐지 전문 프롬프트"""
        
        return """
🚨 **시계열 이상값 탐지 전문가 AI**

**👤 역할**: 시계열 이상 현상 탐지 및 원인 분석 전문가
**🎯 목적**: 비즈니스에 영향을 줄 수 있는 이상 현상 조기 발견

**🔍 분석 요청사항**:
1. **통계적 이상값**: Z-score, IQR 기준으로 수치적 이상값 식별
2. **맥락적 이상값**: 계절성과 트렌드를 고려한 상대적 이상값 발견  
3. **패턴 변화**: 기존 패턴에서 벗어나는 구조적 변화점 탐지
4. **비즈니스 영향도**: 각 이상값이 비즈니스에 미치는 잠재적 영향

**📊 제공할 정보**:
- 시계열 데이터: {series_data}
- 기간: {time_range}
- 탐지된 이상값: {anomaly_candidates}
- 임계값 설정: {thresholds}

**📋 결과 형식**:
각 이상값에 대해 다음을 제시:
- 발생 시기와 수치
- 이상 정도 (심각도: 낮음/보통/높음)
- 추정 원인 (계절적/구조적/외부적)
- 권장 대응 방안
- 지속 모니터링 필요성
"""
    
    def _create_trend_interpretation_prompt(self) -> str:
        """트렌드 해석 전문 프롬프트"""
        
        return """
📈 **트렌드 분석 전문가 AI**

**👤 역할**: 시계열 트렌드 패턴 해석 및 미래 전망 전문가
**🎯 목적**: 장기 트렌드의 비즈니스적 의미와 전략적 시사점 도출

**📊 트렌드 분석 관점**:
1. **장기 트렌드**: 전체 기간에 걸친 일반적 방향성
2. **단기 변화**: 최근 3-6개월의 트렌드 변화
3. **변화점 분석**: 트렌드가 바뀐 주요 시점과 원인
4. **지속성 평가**: 현재 트렌드의 지속 가능성
5. **미래 전망**: 단기/중기 트렌드 예상 방향

**📈 분석 데이터**:
- 트렌드 데이터: {trend_data}
- 분석 기간: {analysis_period}
- 트렌드 지표: {trend_metrics}
- 변화점 정보: {change_points}

**💼 기대 결과**:
- 트렌드의 비즈니스적 의미
- 전략적 시사점과 대응 방안
- 리스크와 기회 요인 분석
- 의사결정을 위한 권고사항
"""
    
    def _create_seasonality_analysis_prompt(self) -> str:
        """계절성 분석 전문 프롬프트"""
        
        return """
🔄 **계절성 패턴 분석 전문가 AI**

**👤 역할**: 주기적 패턴 분석 및 계절성 활용 전략 전문가
**🎯 목적**: 반복 패턴을 활용한 비즈니스 최적화 방안 도출

**📊 계절성 분석 요소**:
1. **주기 길이**: 연간/분기별/월간/주간/일간 패턴 식별
2. **계절성 강도**: 전체 변동 대비 계절적 변동 비율
3. **패턴 안정성**: 연도별/주기별 패턴 일관성 평가
4. **피크/저점 시기**: 최고점과 최저점의 시기와 지속성
5. **변화 트렌드**: 계절성 패턴 자체의 변화 여부

**📈 분석 데이터**:
- 계절성 데이터: {seasonal_data}
- 분해 결과: {decomposition_results}
- 주기별 평균: {period_averages}
- 변동 지표: {variation_metrics}

**🎯 기대 결과**:
- 계절성을 활용한 비즈니스 전략
- 재고 관리 및 마케팅 계획 최적화
- 예측 모델링에서의 계절성 활용 방안
- 계절별 리스크 관리 전략
"""
    
    def _create_business_insights_prompt(self) -> str:
        """비즈니스 인사이트 도출 프롬프트"""
        
        return """
💡 **비즈니스 인사이트 도출 전문가 AI**

**👤 역할**: 데이터 분석을 비즈니스 가치로 전환하는 전략 컨설턴트
**🎯 목적**: 시계열 분석 결과를 실행 가능한 비즈니스 전략으로 변환

**🔍 인사이트 도출 영역**:
1. **현재 상황 진단**: 데이터가 보여주는 비즈니스 현실
2. **기회 요인 발견**: 성장 가능성과 최적화 지점
3. **리스크 요인 식별**: 잠재적 위험과 대응 필요 영역
4. **전략적 제안**: 구체적이고 실행 가능한 액션 플랜
5. **성과 측정**: 개선 효과를 측정할 KPI와 방법

**📊 활용할 분석 결과**:
- 패턴 분석: {pattern_analysis}
- 트렌드 해석: {trend_analysis}
- 계절성 정보: {seasonality_info}
- 이상값 분석: {anomaly_analysis}

**💼 결과 형식**:
- 핵심 인사이트 3가지 (우선순위 순)
- 단기 실행 방안 (1-3개월)
- 중장기 전략 방향 (6-12개월)
- 예상 효과와 성과 지표
- 실행 시 주의사항과 모니터링 포인트
"""

    def analyze_time_series_with_ai(self, data: pd.Series, analysis_type: str = 'pattern_analysis') -> str:
        """
        AI 협업 시계열 분석 실행
        
        🎯 목적: 실제 AI API 호출을 시뮬레이션하여 분석 수행
        📊 분석 과정:
        1. 데이터 기본 통계 계산
        2. 적절한 프롬프트 템플릿 선택
        3. AI 분석 시뮬레이션 실행
        4. 비즈니스 관점의 해석 제공
        """
        
        print(f"🤖 AI {analysis_type} 분석 시작...")
        
        # 1단계: 기본 통계 계산 (AI에게 제공할 정보)
        stats = self._calculate_comprehensive_stats(data)
        
        # 2단계: 프롬프트 템플릿 확인
        if analysis_type not in self.prompt_templates:
            print(f"⚠️ '{analysis_type}'는 지원하지 않는 분석 유형입니다.")
            analysis_type = 'pattern_analysis'
            print(f"🔄 기본 패턴 분석으로 변경합니다.")
        
        # 3단계: AI 분석 시뮬레이션 (실제로는 OpenAI API 등 호출)
        ai_analysis = self._simulate_ai_analysis(data, stats, analysis_type)
        
        # 4단계: 분석 결과 검증 및 보완
        verified_analysis = self._verify_analysis_quality(ai_analysis, data, stats)
        
        print(f"✅ AI 분석 완료! ({analysis_type})")
        return verified_analysis
    
    def _calculate_comprehensive_stats(self, data: pd.Series) -> dict:
        """시계열 데이터의 종합 통계 계산"""
        
        clean_data = data.dropna()
        
        if len(clean_data) == 0:
            return {'error': 'No valid data points'}
        
        # 기본 통계
        basic_stats = {
            'count': len(clean_data),
            'mean': clean_data.mean(),
            'std': clean_data.std(),
            'min': clean_data.min(),
            'max': clean_data.max(),
            'range': clean_data.max() - clean_data.min()
        }
        
        # 시계열 특성 통계
        ts_stats = {
            'trend_slope': self._calculate_trend_slope(clean_data),
            'seasonality_strength': self._calculate_seasonality_strength(clean_data),
            'volatility': clean_data.std() / clean_data.mean() if clean_data.mean() != 0 else 0,
            'autocorr_lag1': clean_data.autocorr(lag=1) if len(clean_data) > 1 else 0
        }
        
        # 데이터 품질 지표
        quality_stats = {
            'missing_ratio': data.isnull().sum() / len(data),
            'outlier_count': self._count_outliers(clean_data),
            'data_completeness': len(clean_data) / len(data)
        }
        
        return {**basic_stats, **ts_stats, **quality_stats}
    
    def _calculate_trend_slope(self, data: pd.Series) -> float:
        """트렌드 기울기 계산 (선형 회귀)"""
        if len(data) < 2:
            return 0.0
        
        x = np.arange(len(data))
        slope = np.polyfit(x, data.values, 1)[0]
        return slope
    
    def _calculate_seasonality_strength(self, data: pd.Series) -> float:
        """계절성 강도 계산 (월별 변동 기준)"""
        if len(data) < 365:  # 1년 미만 데이터
            return 0.0
        
        try:
            # 월별 평균 계산
            monthly_means = data.groupby(data.index.month).mean()
            seasonality = (monthly_means.max() - monthly_means.min()) / data.mean()
            return min(seasonality, 2.0)  # 최대 200%로 제한
        except:
            return 0.0
    
    def _count_outliers(self, data: pd.Series) -> int:
        """Z-score 기준 이상값 개수 계산"""
        if len(data) < 3:
            return 0
        
        z_scores = np.abs((data - data.mean()) / data.std())
        return (z_scores > 2.5).sum()  # 2.5 시그마 이상
    
    def _simulate_ai_analysis(self, data: pd.Series, stats: dict, analysis_type: str) -> str:
        """
        AI 분석 시뮬레이션 (실제 환경에서는 LLM API 호출)
        
        📝 설계 철학:
        - 실제 AI의 분석 패턴을 모방
        - 통계적 근거에 기반한 해석
        - 비즈니스 관점의 실용적 조언 제공
        """
        
        series_name = data.name if data.name else "분석 대상 시계열"
        time_range = f"{data.index.min().date()} ~ {data.index.max().date()}"
        
        if analysis_type == 'pattern_analysis':
            return self._generate_pattern_analysis(series_name, time_range, stats)
        elif analysis_type == 'anomaly_detection':
            return self._generate_anomaly_analysis(series_name, time_range, stats)
        elif analysis_type == 'trend_interpretation':
            return self._generate_trend_analysis(series_name, time_range, stats)
        elif analysis_type == 'seasonality_analysis':
            return self._generate_seasonality_analysis(series_name, time_range, stats)
        else:
            return f"🤖 {analysis_type} 분석 결과가 준비 중입니다."
    
    def _generate_pattern_analysis(self, series_name: str, time_range: str, stats: dict) -> str:
        """종합 패턴 분석 생성"""
        
        # 트렌드 방향 결정
        if stats['trend_slope'] > stats['std'] * 0.01:
            trend_direction = "뚜렷한 상승"
            trend_strength = "강함"
        elif stats['trend_slope'] < -stats['std'] * 0.01:
            trend_direction = "뚜렷한 하락"  
            trend_strength = "강함"
        elif abs(stats['trend_slope']) > stats['std'] * 0.005:
            trend_direction = "완만한 " + ("상승" if stats['trend_slope'] > 0 else "하락")
            trend_strength = "보통"
        else:
            trend_direction = "횡보"
            trend_strength = "약함"
        
        # 계절성 수준 결정
        if stats['seasonality_strength'] > 0.3:
            seasonality_level = "매우 강한"
        elif stats['seasonality_strength'] > 0.15:
            seasonality_level = "뚜렷한"
        elif stats['seasonality_strength'] > 0.05:
            seasonality_level = "약간의"
        else:
            seasonality_level = "거의 없는"
        
        # 변동성 수준 결정
        if stats['volatility'] > 0.3:
            volatility_level = "높은"
        elif stats['volatility'] > 0.15:
            volatility_level = "보통의"
        else:
            volatility_level = "낮은"
        
        # 데이터 품질 평가
        if stats['missing_ratio'] < 0.05 and stats['outlier_count'] < stats['count'] * 0.05:
            quality_assessment = "우수한"
        elif stats['missing_ratio'] < 0.15 and stats['outlier_count'] < stats['count'] * 0.1:
            quality_assessment = "양호한"
        else:
            quality_assessment = "개선이 필요한"
        
        analysis = f"""
🔬 **{series_name} 종합 패턴 분석 결과**

📊 **1. 전체적인 패턴 특성**
이 시계열은 {trend_direction} 트렌드를 보이며, 트렌드 강도는 {trend_strength}입니다.
전체 변동 중 약 {stats['seasonality_strength']:.1%}가 계절적 요인에 기인하며,
{volatility_level} 변동성({stats['volatility']:.1%})을 나타냅니다.

🔄 **2. 주기적 패턴 분석**
{seasonality_level} 계절성 패턴이 관찰됩니다.
{'예측 모델링 시 계절 조정이 필수적' if stats['seasonality_strength'] > 0.2 else '계절성 고려는 선택적'}이며,
자기상관계수 {stats['autocorr_lag1']:.3f}로 {'강한' if abs(stats['autocorr_lag1']) > 0.7 else '보통의' if abs(stats['autocorr_lag1']) > 0.3 else '약한'} 
시계열 의존성을 보입니다.

🚨 **3. 이상 현상 및 데이터 품질**
총 {stats['outlier_count']}개의 이상값이 탐지되어 전체 데이터의 {stats['outlier_count']/stats['count']:.1%}를 차지합니다.
결측값 비율은 {stats['missing_ratio']:.1%}로 {quality_assessment} 데이터 품질을 보입니다.
{'추가적인 데이터 정제가 권장됩니다.' if stats['missing_ratio'] > 0.1 or stats['outlier_count']/stats['count'] > 0.05 else '현재 품질로 분석 진행이 적절합니다.'}

💼 **4. 비즈니스 시사점**
- {'지속적인 성장 모멘텀 유지 전략 필요' if trend_direction.startswith('상승') else '하락 원인 분석 및 반전 전략 검토' if trend_direction.startswith('하락') else '현 상태 유지 및 변곡점 모니터링'}
- {'계절별 차별화된 마케팅 및 운영 전략 수립 기회' if stats['seasonality_strength'] > 0.1 else '안정적인 연중 운영 계획 수립 가능'}
- {'변동성 관리 및 리스크 헤징 전략 강화 필요' if stats['volatility'] > 0.2 else '예측 가능한 범위 내에서 안정적 운영'}

🎯 **5. 핵심 특성 요약**
1. {trend_direction} 트렌드 (강도: {trend_strength})
2. {seasonality_level} 계절성 (강도: {stats['seasonality_strength']:.1%})
3. {volatility_level} 변동성 (수준: {stats['volatility']:.1%})

🔮 **6. 권장 후속 분석**
1. {'변화점 탐지로 트렌드 전환점 식별' if abs(stats['trend_slope']) > stats['std'] * 0.01 else '정상성 검정 및 차분 분석'}
2. {'계절성 분해를 통한 세부 패턴 분석' if stats['seasonality_strength'] > 0.1 else '노이즈 성분 분석 및 스무딩 검토'}
3. {'외부 요인과의 상관관계 분석으로 변동성 원인 규명' if stats['volatility'] > 0.2 else '예측 모델 구축을 위한 특성 공학'}

⚙️ **7. 예측 모델링 고려사항**
- {'SARIMA나 계절성 지수평활법 등 계절성 모델 적합' if stats['seasonality_strength'] > 0.15 else 'ARIMA나 단순 지수평활법으로 충분'}
- {'변동성 모델링(GARCH 등) 추가 고려' if stats['volatility'] > 0.3 else '기본적인 시계열 모델로 적절'}
- 교차검증 시 시간 순서 유지 필수
- {'추가 특성 공학으로 외부 변수 고려' if stats['autocorr_lag1'] < 0.5 else '자기회귀 특성 충분히 활용'}
        """.strip()
        
        return analysis
```

**🔍 코드 해설:**

**1. 클래스 설계 철학**
- **역할 분담**: AI는 패턴 인식, 인간은 비즈니스 해석
- **템플릿 기반**: 일관성 있는 분석을 위한 표준화된 프롬프트
- **확장성**: 새로운 분석 유형을 쉽게 추가할 수 있는 구조

**2. CLEAR 원칙 적용**
- **Context**: 비즈니스 시계열 분석 맥락 명시
- **Length**: 5개 영역으로 구체적 범위 지정
- **Examples**: 명확한 분석 기준과 형식 제시
- **Actionable**: 바로 활용할 수 있는 인사이트 도출
- **Role**: 전문가 역할 부여로 품질 향상

**3. 통계 계산의 의미**
- **트렌드 기울기**: 시간당 평균 증가/감소량
- **계절성 강도**: 전체 변동 중 계절적 변동 비율
- **변동성**: 평균 대비 표준편차 (리스크 지표)
- **자기상관**: 과거 의존성 정도 (예측 가능성 지표)

**4. AI 분석 시뮬레이션**
- 실제 LLM의 사고 과정을 모방
- 통계적 근거에 기반한 객관적 판단
- 비즈니스 언어로 번역된 해석
- 구체적이고 실행 가능한 제안

### 2.2 실제 데이터로 AI 협업 분석 실행

```python
# 시계열 기초 클래스와 AI 분석 시스템 초기화
ts_foundation = TimeSeriesFoundation()
ai_ts_analyzer = AIAssistedTimeSeriesAnalysis()

print("🚀 시계열 데이터와 AI 협업 분석 시작!")
print("=" * 70)

# 1단계: 샘플 데이터 생성
print("📊 1단계: 다양한 시계열 데이터 생성")
sample_data = ts_foundation.create_sample_data()

# 2단계: 시계열 특성 분석
print("\n🔍 2단계: 시계열 핵심 특성 분석")
properties_analysis = ts_foundation.demonstrate_time_series_properties()

# 3단계: 시각화
print("\n🎨 3단계: 시계열 패턴 시각화")
monthly_data = ts_foundation.visualize_time_series_components()

# 4단계: AI 협업 분석 실행
print("\n🤖 4단계: AI 협업 기반 심화 분석")

# 매출 데이터 종합 패턴 분석
print("\n💰 매출 데이터 AI 종합 분석:")
sales_analysis = ai_ts_analyzer.analyze_time_series_with_ai(
    sample_data['sales'], 'pattern_analysis'
)
print(sales_analysis)

# 주식 데이터 이상값 탐지
print("\n📈 주식 가격 AI 이상값 분석:")
stock_anomaly_analysis = ai_ts_analyzer.analyze_time_series_with_ai(
    sample_data['stock_price'], 'anomaly_detection'
)
print(stock_anomaly_analysis)

# 온도 데이터 계절성 분석  
print("\n🌡️ 온도 데이터 AI 계절성 분석:")
temp_seasonality_analysis = ai_ts_analyzer.analyze_time_series_with_ai(
    sample_data['temperature'], 'seasonality_analysis'
)
print(temp_seasonality_analysis)

print(f"\n✅ AI 협업 시계열 분석 완료!")
print(f"   🧠 CLEAR 프롬프트로 정확한 패턴 해석")
print(f"   🤖 자동화된 이상값 및 계절성 분석") 
print(f"   💡 비즈니스 중심의 실행 가능한 인사이트")
print(f"   📊 데이터 품질 진단 및 모델링 가이드")
```

**🔍 코드 해설:**

**실행 순서의 의미**
1. **데이터 생성**: 실제와 유사한 다양한 패턴의 시계열 생성
2. **특성 분석**: 4가지 핵심 특성을 정량적으로 확인
3. **시각화**: 패턴을 직관적으로 이해할 수 있도록 그래프 제작
4. **AI 분석**: 복잡한 패턴을 AI가 해석하고 인사이트 도출

**AI 협업의 장점**
- **속도**: 복잡한 분석을 몇 초 만에 완료
- **일관성**: 표준화된 분석 프레임워크로 일관된 품질
- **포괄성**: 인간이 놓칠 수 있는 미묘한 패턴까지 포착
- **실용성**: 기술적 분석을 비즈니스 언어로 번역

---

## 🎨 이미지 생성 프롬프트

```
AI와 인간 협업 시계열 분석 워크플로우:
왼쪽에는 AI 로봇이 데이터 차트를 분석하고,
오른쪽에는 비즈니스맨이 결과를 해석하는 모습
중앙에는 다양한 시계열 그래프들이 떠다니며
화살표로 분석 과정을 연결
현대적이고 전문적인 데이터 사이언스 스타일
파란색과 주황색 색상 조합
```

---

## 3. 시계열 구성요소 분해 (Time Series Decomposition)

시계열 분해는 마치 음악에서 멜로디, 리듬, 하모니를 분리하는 것과 같습니다. 복잡해 보이는 시계열을 트렌드, 계절성, 잔차로 나누어 각각을 따로 분석할 수 있게 해줍니다!

### 📚 시계열 분해 핵심 개념

**📖 시계열 분해(Time Series Decomposition)**
- **의미**: 복잡한 시계열을 의미 있는 구성요소들로 분리하는 기법
- **목적**: 각 구성요소를 개별적으로 분석하여 전체 패턴을 더 잘 이해
- **활용**: 예측 모델링, 이상값 탐지, 트렌드 분석의 기초

**📖 구성요소별 의미**
- **트렌드(Trend)**: 장기적인 증가/감소 패턴
- **계절성(Seasonal)**: 규칙적으로 반복되는 주기적 패턴  
- **순환성(Cyclical)**: 불규칙한 주기의 장기 파동
- **잔차(Residual/Error)**: 설명되지 않는 무작위 변동

**📖 분해 방법의 종류**
- **가법적 분해**: Y(t) = Trend(t) + Seasonal(t) + Error(t)
- **승법적 분해**: Y(t) = Trend(t) × Seasonal(t) × Error(t)
- **STL 분해**: 계절성과 트렌드 분해를 위한 로버스트 방법

### 3.1 시계열 분해 시스템 구축

```python
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesDecomposition:
    """
    시계열 분해 및 구성요소 분석 클래스
    
    🎯 목적: 복잡한 시계열을 이해하기 쉬운 구성요소로 분해
    📚 분해 방법: 가법적, 승법적, STL 분해 지원
    🔍 분석 기능: 각 구성요소의 강도와 특성 정량 분석
    """
    
    def __init__(self):
        """분해 시스템 초기화"""
        print("🔬 시계열 분해 시스템 초기화 중...")
        
        # 다양한 분해 방법 등록
        self.decomposition_methods = {
            'additive': self._additive_decomposition,      # 가법적 분해
            'multiplicative': self._multiplicative_decomposition,  # 승법적 분해
            'stl': self._stl_decomposition                # STL 분해
        }
        
        print(f"✅ {len(self.decomposition_methods)}가지 분해 방법 준비 완료!")
        
    def _additive_decomposition(self, series: pd.Series, period: int = None) -> dict:
        """
        가법적 분해 (Additive Decomposition)
        
        📚 수학적 모델: Y(t) = Trend(t) + Seasonal(t) + Residual(t)
        🎯 적용 상황: 계절 변동의 크기가 일정한 경우
        📊 예시: 온도, 전력 사용량 등
        """
        
        print("➕ 가법적 분해 수행 중...")
        
        if period is None:
            period = self._estimate_period(series)
            print(f"   🔍 자동 추정된 주기: {period}")
        
        try:
            decomposition = seasonal_decompose(
                series, 
                model='additive',  # 가법적 모델
                period=period,     # 계절성 주기
                extrapolate_trend='freq'  # 트렌드 외삽
            )
            
            print(f"   ✅ 가법적 분해 완료 (주기: {period})")
            
            return {
                'original': series,
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid,
                'period': period,
                'model': 'additive'
            }
            
        except Exception as e:
            print(f"   ❌ 가법적 분해 실패: {e}")
            return self._fallback_decomposition(series, period)
    
    def _multiplicative_decomposition(self, series: pd.Series, period: int = None) -> dict:
        """
        승법적 분해 (Multiplicative Decomposition)
        
        📚 수학적 모델: Y(t) = Trend(t) × Seasonal(t) × Residual(t)
        🎯 적용 상황: 계절 변동의 크기가 트렌드에 비례하는 경우
        📊 예시: 매출, 주가, 방문자 수 등
        """
        
        print("✖️ 승법적 분해 수행 중...")
        
        if period is None:
            period = self._estimate_period(series)
            print(f"   🔍 자동 추정된 주기: {period}")
        
        # 음수값 또는 0값 확인
        if (series <= 0).any():
            print("   ⚠️ 음수 또는 0값 발견: 가법적 분해로 전환합니다.")
            return self._additive_decomposition(series, period)
        
        try:
            decomposition = seasonal_decompose(
                series,
                model='multiplicative',  # 승법적 모델  
                period=period,
                extrapolate_trend='freq'
            )
            
            print(f"   ✅ 승법적 분해 완료 (주기: {period})")
            
            return {
                'original': series,
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid,
                'period': period,
                'model': 'multiplicative'
            }
            
        except Exception as e:
            print(f"   ❌ 승법적 분해 실패: {e}")
            print("   🔄 가법적 분해로 전환합니다.")
            return self._additive_decomposition(series, period)
    
    def _stl_decomposition(self, series: pd.Series, period: int = None) -> dict:
        """
        STL 분해 (Seasonal and Trend decomposition using Loess)
        
        📚 특징: 계절성과 트렌드를 로버스트하게 분해
        🎯 장점: 이상값에 강하고 유연한 분해
        📊 적용: 복잡하거나 불규칙한 패턴의 시계열
        """
        
        print("🔄 STL 분해 수행 중...")
        
        try:
            from statsmodels.tsa.seasonal import STL
            
            if period is None:
                period = self._estimate_period(series)
                print(f"   🔍 자동 추정된 주기: {period}")
            
            stl = STL(series, seasonal=period, robust=True)
            decomposition = stl.fit()
            
            print(f"   ✅ STL 분해 완료 (주기: {period})")
            
            return {
                'original': series,
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid,
                'period': period,
                'model': 'stl'
            }
            
        except ImportError:
            print("   ⚠️ STL을 사용할 수 없습니다. 가법적 분해를 사용합니다.")
            return self._additive_decomposition(series, period)
        except Exception as e:
            print(f"   ❌ STL 분해 실패: {e}")
            return self._additive_decomposition(series, period)
    
    def _estimate_period(self, series: pd.Series) -> int:
        """
        주기 자동 추정
        
        🎯 목적: 데이터 특성에 맞는 최적 주기 선택
        📊 방법: 데이터 길이와 주기성을 고려한 휴리스틱
        """
        
        data_length = len(series)
        
        # 데이터 길이에 따른 주기 결정
        if data_length >= 365 * 2:  # 2년 이상 일별 데이터
            period = 365  # 연간 주기
            period_name = "연간"
        elif data_length >= 52 * 2:  # 2년치 주간 데이터  
            period = 52   # 연간 주기 (주단위)
            period_name = "연간(주별)"
        elif data_length >= 24:     # 24개 이상 데이터
            period = 12   # 월간 주기
            period_name = "월간"
        else:
            period = min(data_length // 2, 7)  # 최소 주기
            period_name = "기본"
        
        print(f"   📊 데이터 길이: {data_length}, 추정 주기: {period} ({period_name})")
        return period
    
    def _fallback_decomposition(self, series: pd.Series, period: int) -> dict:
        """분해 실패 시 대체 방법"""
        
        print("   🔧 간단한 분해 방법으로 대체합니다.")
        
        # 단순 이동평균으로 트렌드 추출
        window = max(period, 7)
        trend = series.rolling(window=window, center=True).mean()
        
        # 트렌드 제거 후 계절성 추출
        detrended = series - trend
        seasonal = detrended.groupby(detrended.index % period).transform('mean')
        
        # 잔차 계산
        residual = series - trend - seasonal
        
        return {
            'original': series,
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual,
            'period': period,
            'model': 'simple'
        }
    
    def decompose_time_series(self, series: pd.Series, method: str = 'auto', period: int = None) -> dict:
        """
        시계열 분해 실행
        
        🎯 목적: 사용자 요청에 맞는 분해 방법 적용
        🤖 자동 선택: 데이터 특성에 따른 최적 방법 추천
        """
        
        print(f"\n🔬 시계열 분해 시작 (방법: {method})")
        print(f"   📊 시계열명: {series.name or '무제'}")
        print(f"   📅 기간: {series.index.min().date()} ~ {series.index.max().date()}")
        print(f"   📈 데이터 포인트: {len(series):,}개")
        
        # 자동 방법 선택
        if method == 'auto':
            method = self._select_optimal_method(series)
            print(f"   🤖 자동 선택된 방법: {method}")
        
        # 지원하는 방법인지 확인
        if method not in self.decomposition_methods:
            print(f"   ⚠️ '{method}'는 지원하지 않는 방법입니다.")
            method = 'additive'
            print(f"   🔄 기본 가법적 분해로 변경합니다.")
        
        # 분해 실행
        result = self.decomposition_methods[method](series, period)
        
        print(f"   ✅ {result['model']} 분해 완료!")
        return result
    
    def _select_optimal_method(self, series: pd.Series) -> str:
        """
        데이터 특성에 따른 최적 분해 방법 자동 선택
        
        🧠 선택 로직:
        - 음수값 있음 → 가법적 분해
        - 변동성이 트렌드에 비례 → 승법적 분해
        - 복잡한 패턴 → STL 분해
        - 기본 → 가법적 분해
        """
        
        clean_data = series.dropna()
        
        if len(clean_data) == 0:
            return 'additive'
        
        # 음수값 확인
        if (clean_data <= 0).any():
            print("   🔍 음수값 발견 → 가법적 분해 선택")
            return 'additive'
        
        # 변동성 패턴 분석
        first_half = clean_data[:len(clean_data)//2]
        second_half = clean_data[len(clean_data)//2:]
        
        if len(first_half) > 0 and len(second_half) > 0:
            cv_first = first_half.std() / first_half.mean() if first_half.mean() > 0 else 0
            cv_second = second_half.std() / second_half.mean() if second_half.mean() > 0 else 0
            
            # 변동계수가 크게 변했으면 승법적 분해 고려
            if abs(cv_second - cv_first) > 0.1 and cv_second > cv_first:
                print("   🔍 변동성 증가 패턴 → 승법적 분해 선택")
                return 'multiplicative'
        
        # 기본적으로 가법적 분해
        print("   🔍 표준 패턴 → 가법적 분해 선택")
        return 'additive'
```

**🔍 코드 해설:**

**1. 분해 방법별 특징**
- **가법적**: 계절 변동이 일정한 크기 (온도, 전력사용량)
- **승법적**: 계절 변동이 트렌드에 비례 (매출, 주가)
- **STL**: 이상값에 강하고 유연한 분해 (복잡한 패턴)

**2. 주기 추정 로직**
- 데이터 길이에 따라 적절한 주기 자동 선택
- 2년 이상 → 365일 (연간), 2년 이하 → 12개월 등
- 너무 짧은 데이터는 기본 주기 적용

**3. 오류 처리 및 대체**
- 분해 실패 시 간단한 이동평균 기반 분해로 대체
- 음수값 있을 때 자동으로 가법적 분해로 전환
- 로버스트한 시스템 구현

**4. 자동 방법 선택**
- 데이터의 특성을 분석하여 최적 방법 추천
- 변동성 패턴, 음수값 존재 여부 등을 고려
- 사용자가 직접 선택할 수도 있음

### 3.2 구성요소별 상세 분석

```python
    def analyze_components(self, decomposition: dict) -> dict:
        """
        분해된 구성요소 심층 분석
        
        🎯 목적: 각 구성요소의 특성을 정량적으로 분석
        📊 분석 내용: 트렌드, 계절성, 잔차의 개별 특성
        """
        
        print("\n🔍 구성요소별 심층 분석 시작...")
        
        analysis = {
            'trend_analysis': self._analyze_trend(decomposition['trend']),
            'seasonal_analysis': self._analyze_seasonality(decomposition['seasonal']),
            'residual_analysis': self._analyze_residuals(decomposition['residual']),
            'component_strength': self._calculate_component_strength(decomposition)
        }
        
        print("✅ 구성요소 분석 완료!")
        return analysis
    
    def _analyze_trend(self, trend: pd.Series) -> dict:
        """
        트렌드 구성요소 분석
        
        📈 분석 내용:
        - 트렌드 방향과 기울기
        - 트렌드 변화율과 가속도
        - 주요 변화점 탐지
        """
        
        print("   📈 트렌드 분석 중...")
        
        # 결측값 제거
        trend_clean = trend.dropna()
        
        if len(trend_clean) < 2:
            return {'analysis': '트렌드 분석에 충분한 데이터가 없습니다.'}
        
        # 1. 트렌드 기울기 계산 (선형 회귀)
        x = np.arange(len(trend_clean))
        slope, intercept = np.polyfit(x, trend_clean.values, 1)
        
        # 2. 트렌드 방향 결정
        if abs(slope) < trend_clean.std() * 0.001:
            direction = 'stable'
            direction_kr = '안정'
        elif slope > 0:
            direction = 'increasing'
            direction_kr = '상승'
        else:
            direction = 'decreasing'
            direction_kr = '하락'
        
        # 3. 트렌드 변화율 계산
        first_third = trend_clean[:len(trend_clean)//3].mean()
        last_third = trend_clean[-len(trend_clean)//3:].mean()
        change_rate = (last_third - first_third) / first_third if first_third != 0 else 0
        
        # 4. 트렌드 가속도 분석 (2차 다항식 피팅)
        try:
            poly_coeffs = np.polyfit(x, trend_clean.values, 2)
            acceleration = poly_coeffs[0] * 2  # 2차 계수의 2배가 가속도
            
            if abs(acceleration) < trend_clean.std() * 0.0001:
                accel_type = 'linear'
                accel_kr = '선형'
            elif acceleration > 0:
                accel_type = 'accelerating'
                accel_kr = '가속'
            else:
                accel_type = 'decelerating'
                accel_kr = '감속'
        except:
            acceleration = 0
            accel_type = 'linear'
            accel_kr = '선형'
        
        # 5. 변화점 탐지 (단순한 방법)
        change_points = []
        window_size = max(len(trend_clean) // 20, 5)  # 전체의 5% 또는 최소 5개
        
        for i in range(window_size, len(trend_clean) - window_size):
            before = trend_clean.iloc[i-window_size:i].mean()
            after = trend_clean.iloc[i:i+window_size].mean()
            
            # 표준편차의 2배 이상 변화하면 변화점으로 간주
            if abs(after - before) > 2 * trend_clean.std():
                change_points.append({
                    'date': trend_clean.index[i],
                    'change': after - before,
                    'type': 'increase' if after > before else 'decrease'
                })
        
        # 6. 트렌드 강도 계산 (기울기 대비 변동성)
        trend_strength = abs(slope) / trend_clean.std() if trend_clean.std() > 0 else 0
        
        print(f"      방향: {direction_kr}, 기울기: {slope:.6f}")
        print(f"      변화율: {change_rate:.1%}, 가속도: {accel_kr}")
        print(f"      변화점: {len(change_points)}개, 강도: {trend_strength:.3f}")
        
        return {
            'slope': slope,
            'intercept': intercept,
            'direction': direction,
            'direction_korean': direction_kr,
            'change_rate': change_rate,
            'acceleration': acceleration,
            'acceleration_type': accel_type,
            'acceleration_korean': accel_kr,
            'change_points': change_points[:5],  # 최대 5개까지
            'strength': trend_strength,
            'r_squared': np.corrcoef(x, trend_clean.values)[0,1]**2 if len(x) > 1 else 0
        }
    
    def _analyze_seasonality(self, seasonal: pd.Series) -> dict:
        """
        계절성 구성요소 분석
        
        🔄 분석 내용:
        - 계절성 강도와 진폭
        - 주기별 패턴 특성
        - 계절성 안정성
        """
        
        print("   🔄 계절성 분석 중...")
        
        seasonal_clean = seasonal.dropna()
        
        if len(seasonal_clean) == 0:
            return {'analysis': '계절성 분석에 충분한 데이터가 없습니다.'}
        
        # 1. 계절성 강도 계산
        seasonal_range = seasonal_clean.max() - seasonal_clean.min()
        seasonal_mean = abs(seasonal_clean.mean())
        seasonal_std = seasonal_clean.std()
        
        # 강도 지표: 범위 대비 표준편차
        if seasonal_mean != 0:
            strength = seasonal_std / seasonal_mean
        else:
            strength = seasonal_std / (seasonal_range / 4) if seasonal_range > 0 else 0
        
        # 2. 주기별 패턴 분석
        # 간단한 주기 추정 (FFT 사용 시도)
        try:
            # 푸리에 변환으로 주요 주기 찾기
            from scipy.fft import fft, fftfreq
            
            n = len(seasonal_clean)
            fft_values = fft(seasonal_clean.values)
            frequencies = fftfreq(n)
            
            # DC 성분 제외하고 가장 강한 주파수 찾기
            power_spectrum = np.abs(fft_values[1:n//2])
            dominant_freq_idx = np.argmax(power_spectrum) + 1
            dominant_period = int(1 / abs(frequencies[dominant_freq_idx])) if frequencies[dominant_freq_idx] != 0 else 12
            
            print(f"      주요 주기: {dominant_period}")
            
        except:
            dominant_period = 12  # 기본값
        
        # 3. 주기별 평균 패턴
        if len(seasonal_clean) >= dominant_period:
            period_patterns = []
            for i in range(dominant_period):
                pattern_values = seasonal_clean.iloc[i::dominant_period]
                if len(pattern_values) > 0:
                    period_patterns.append(pattern_values.mean())
                else:
                    period_patterns.append(0)
            
            peak_period = np.argmax(period_patterns)
            trough_period = np.argmin(period_patterns)
        else:
            period_patterns = []
            peak_period = 0
            trough_period = 0
        
        # 4. 계절성 규칙성 평가
        if len(period_patterns) > 0:
            pattern_std = np.std(period_patterns)
            pattern_range = max(period_patterns) - min(period_patterns)
            regularity = 1 - (pattern_std / pattern_range) if pattern_range > 0 else 1
        else:
            regularity = 0
        
        # 5. 계절성 분류
        if strength > 0.5:
            intensity = 'very_strong'
            intensity_kr = '매우 강함'
        elif strength > 0.3:
            intensity = 'strong'
            intensity_kr = '강함'
        elif strength > 0.1:
            intensity = 'moderate'
            intensity_kr = '보통'
        else:
            intensity = 'weak'
            intensity_kr = '약함'
        
        print(f"      강도: {intensity_kr} ({strength:.3f})")
        print(f"      진폭: {seasonal_range:.3f}, 규칙성: {regularity:.3f}")
        
        return {
            'strength': strength,
            'amplitude': seasonal_range,
            'intensity': intensity,
            'intensity_korean': intensity_kr,
            'dominant_period': dominant_period,
            'pattern': period_patterns,
            'peak_period': peak_period,
            'trough_period': trough_period,
            'regularity': regularity
        }
    
    def _analyze_residuals(self, residuals: pd.Series) -> dict:
        """
        잔차 구성요소 분석
        
        🎲 분석 내용:
        - 잔차의 통계적 특성
        - 이상값과 패턴 존재 여부
        - 모델 적합도 평가
        """
        
        print("   🎲 잔차 분석 중...")
        
        residuals_clean = residuals.dropna()
        
        if len(residuals_clean) == 0:
            return {'analysis': '잔차 분석에 충분한 데이터가 없습니다.'}
        
        # 1. 기본 통계
        mean_residual = residuals_clean.mean()
        std_residual = residuals_clean.std()
        
        # 2. 이상값 탐지 (Z-score 기준)
        if std_residual > 0:
            z_scores = np.abs((residuals_clean - mean_residual) / std_residual)
            outliers = residuals_clean[z_scores > 2.5]  # 2.5 시그마 기준
        else:
            outliers = pd.Series([], dtype=float)
        
        # 3. 자기상관 검사 (잔차에 패턴이 남아있는지)
        autocorr_lag1 = residuals_clean.autocorr(lag=1) if len(residuals_clean) > 1 else 0
        
        # 4. 정규성 검사 (간단한 방법)
        from scipy import stats
        
        # 왜도(skewness)와 첨도(kurtosis)로 정규성 평가
        skewness = residuals_clean.skew()
        kurtosis = residuals_clean.kurtosis()
        
        # 정규성 지표 (왜도가 -1~1, 첨도가 -1~1 범위면 근사적으로 정규분포)
        normality_score = 1 - (abs(skewness) + abs(kurtosis)) / 4
        normality_score = max(0, min(1, normality_score))  # 0~1 범위로 제한
        
        # 5. 백색잡음 여부 판단
        is_white_noise = (
            abs(autocorr_lag1) < 0.1 and  # 자기상관 낮음
            abs(mean_residual) < 0.1 * std_residual and  # 평균이 0에 가까움
            normality_score > 0.7  # 정규분포에 가까움
        )
        
        # 6. 잔차 품질 평가
        outlier_ratio = len(outliers) / len(residuals_clean)
        
        if outlier_ratio < 0.05 and is_white_noise:
            quality = 'excellent'
            quality_kr = '우수'
        elif outlier_ratio < 0.1 and abs(autocorr_lag1) < 0.3:
            quality = 'good'
            quality_kr = '양호'
        elif outlier_ratio < 0.2:
            quality = 'fair'
            quality_kr = '보통'
        else:
            quality = 'poor'
            quality_kr = '불량'
        
        print(f"      평균: {mean_residual:.6f}, 표준편차: {std_residual:.3f}")
        print(f"      이상값: {len(outliers)}개 ({outlier_ratio:.1%})")
        print(f"      품질: {quality_kr}, 백색잡음: {'예' if is_white_noise else '아니오'}")
        
        return {
            'mean': mean_residual,
            'std': std_residual,
            'outlier_count': len(outliers),
            'outlier_percentage': outlier_ratio * 100,
            'outlier_indices': outliers.index.tolist()[:10],  # 최대 10개
            'autocorr_lag1': autocorr_lag1,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'normality_score': normality_score,
            'is_white_noise': is_white_noise,
            'quality': quality,
            'quality_korean': quality_kr
        }
    
    def _calculate_component_strength(self, decomposition: dict) -> dict:
        """
        구성요소 강도 계산 및 모델 품질 평가
        
        📊 계산 방법: 각 구성요소의 분산 비율
        🎯 목적: 어떤 구성요소가 가장 중요한지 파악
        """
        
        print("   ⚖️ 구성요소 강도 계산 중...")
        
        original = decomposition['original']
        trend = decomposition['trend'].dropna()
        seasonal = decomposition['seasonal'].dropna()
        residual = decomposition['residual'].dropna()
        
        # 전체 분산 계산
        total_var = original.var()
        
        if total_var > 0 and len(trend) > 0 and len(seasonal) > 0 and len(residual) > 0:
            # 각 구성요소의 분산 비율 계산
            trend_strength = trend.var() / total_var
            seasonal_strength = seasonal.var() / total_var
            residual_strength = residual.var() / total_var
            
            # 설명력 계산 (트렌드 + 계절성)
            explained_variance = trend_strength + seasonal_strength
            
            # 모델 품질 (잔차가 작을수록 좋음)
            model_quality = 1 - min(residual_strength, 1)
            
        else:
            trend_strength = seasonal_strength = residual_strength = 0
            explained_variance = 0
            model_quality = 0
        
        # 주요 구성요소 식별
        strengths = {
            'trend': trend_strength,
            'seasonal': seasonal_strength,
            'residual': residual_strength
        }
        
        dominant_component = max(strengths, key=strengths.get)
        dominant_component_kr = {
            'trend': '트렌드',
            'seasonal': '계절성',
            'residual': '잔차'
        }[dominant_component]
        
        print(f"      트렌드: {trend_strength:.1%}, 계절성: {seasonal_strength:.1%}")
        print(f"      잔차: {residual_strength:.1%}, 설명력: {explained_variance:.1%}")
        print(f"      주요 구성요소: {dominant_component_kr}")
        
        return {
            'trend_strength': trend_strength,
            'seasonal_strength': seasonal_strength,
            'residual_strength': residual_strength,
            'explained_variance': explained_variance,
            'model_quality': model_quality,
            'dominant_component': dominant_component,
            'dominant_component_korean': dominant_component_kr
        }
```

**🔍 코드 해설:**

**1. 트렌드 분석**
- **기울기**: 선형 회귀로 전체적인 증가/감소 방향 측정
- **가속도**: 2차 다항식으로 트렌드의 변화 속도 분석
- **변화점**: 일정 구간에서 급격한 변화가 있는 시점 탐지
- **강도**: 기울기 대비 변동성으로 트렌드의 명확성 측정

**2. 계절성 분석**
- **FFT 분석**: 푸리에 변환으로 주요 주기 자동 탐지
- **주기별 패턴**: 각 주기별 평균값으로 계절 패턴 파악
- **규칙성**: 패턴의 일관성 정도 측정
- **강도 분류**: 계절성의 정도를 5단계로 구분

**3. 잔차 분석**
- **이상값 탐지**: Z-score 2.5 기준으로 극값 식별
- **자기상관**: 잔차에 아직 패턴이 남아있는지 확인
- **정규성**: 왜도와 첨도로 정규분포 근접도 측정
- **백색잡음**: 이상적인 잔차 조건 만족 여부

**4. 구성요소 강도**
- **분산 분해**: 전체 변동을 구성요소별로 분해
- **설명력**: 트렌드+계절성이 설명하는 비율
- **모델 품질**: 잔차가 작을수록 좋은 분해
- **주요 구성요소**: 가장 큰 영향을 미치는 요소 식별

### 3.3 분해 결과 시각화

```python
    def visualize_decomposition(self, decomposition: dict, analysis: dict = None):
        """
        분해 결과 시각화
        
        🎨 목적: 분해된 구성요소들을 직관적으로 이해
        📊 구성: 원본, 트렌드, 계절성, 잔차 4개 차트
        """
        
        print("\n🎨 시계열 분해 결과 시각화...")
        
        # 메인 분해 차트 (4개 서브플롯)
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        fig.suptitle(f'🔬 시계열 분해 결과 ({decomposition["model"].upper()})', 
                    fontsize=16, fontweight='bold')
        
        components = ['original', 'trend', 'seasonal', 'residual']
        titles = ['📊 원본 시계열', '📈 트렌드 구성요소', '🔄 계절성 구성요소', '🎲 잔차 구성요소']
        colors = ['#2E8B57', '#4682B4', '#FF6347', '#DAA520']
        
        for i, (component, title, color) in enumerate(zip(components, titles, colors)):
            ax = axes[i]
            data = decomposition[component]
            
            # 데이터 플롯
            ax.plot(data.index, data.values, color=color, linewidth=1.5, alpha=0.8)
            ax.set_title(title, fontweight='bold', pad=10)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            
            # 각 구성요소별 추가 정보 표시
            if analysis and component in ['trend', 'seasonal', 'residual']:
                info_text = self._get_component_info_text(component, analysis)
                if info_text:
                    ax.text(0.02, 0.95, info_text, transform=ax.transAxes, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                           fontsize=9, verticalalignment='top')
            
            # Y축 범위 조정 (극값 제거)
            data_clean = data.dropna()
            if len(data_clean) > 0:
                q1, q99 = data_clean.quantile([0.01, 0.99])
                if q99 > q1:
                    ax.set_ylim(q1 - (q99-q1)*0.1, q99 + (q99-q1)*0.1)
        
        plt.tight_layout()
        plt.show()
        
        # 구성요소 강도 비교 차트
        if analysis and 'component_strength' in analysis:
            self._visualize_component_strength(analysis['component_strength'])
        
        # 이미지 생성 프롬프트
        print(f"\n🎨 분해 시각화 이미지 생성 프롬프트:")
        print(f"   'Time series decomposition visualization with 4 stacked charts:")
        print(f"   - Original time series (green)")
        print(f"   - Trend component (blue) showing long-term direction")
        print(f"   - Seasonal component (red) showing periodic patterns")
        print(f"   - Residual component (orange) showing remaining noise")
        print(f"   Clean analytical style, statistical visualization, data science theme'")
    
    def _get_component_info_text(self, component: str, analysis: dict) -> str:
        """구성요소별 정보 텍스트 생성"""
        
        if component == 'trend' and 'trend_analysis' in analysis:
            trend_info = analysis['trend_analysis']
            direction = trend_info.get('direction_korean', '알 수 없음')
            strength = trend_info.get('strength', 0)
            return f"방향: {direction}\n강도: {strength:.3f}"
        
        elif component == 'seasonal' and 'seasonal_analysis' in analysis:
            seasonal_info = analysis['seasonal_analysis']
            intensity = seasonal_info.get('intensity_korean', '알 수 없음')
            regularity = seasonal_info.get('regularity', 0)
            return f"강도: {intensity}\n규칙성: {regularity:.3f}"
        
        elif component == 'residual' and 'residual_analysis' in analysis:
            residual_info = analysis['residual_analysis']
            quality = residual_info.get('quality_korean', '알 수 없음')
            outliers = residual_info.get('outlier_percentage', 0)
            return f"품질: {quality}\n이상값: {outliers:.1f}%"
        
        return ""
    
    def _visualize_component_strength(self, strength: dict):
        """구성요소 강도 시각화"""
        
        print("📊 구성요소 강도 비교 차트 생성...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('📊 시계열 구성요소 강도 분석', fontsize=14, fontweight='bold')
        
        # 1. 구성요소별 강도 바 차트
        components = ['트렌드', '계절성', '잔차']
        strengths = [
            strength['trend_strength'],
            strength['seasonal_strength'], 
            strength['residual_strength']
        ]
        colors = ['#4682B4', '#FF6347', '#DAA520']
        
        bars = ax1.bar(components, strengths, color=colors, alpha=0.7)
        ax1.set_title('구성요소별 분산 기여도', fontweight='bold')
        ax1.set_ylabel('분산 비율')
        ax1.grid(True, alpha=0.3)
        
        # 값 표시
        for bar, strength_val in zip(bars, strengths):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{strength_val:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # 2. 설명력 파이 차트
        explained = strength['explained_variance']
        unexplained = 1 - explained
        
        pie_data = [explained, unexplained]
        pie_labels = [f'설명됨\n({explained:.1%})', f'설명 안됨\n({unexplained:.1%})']
        pie_colors = ['#90EE90', '#FFB6C1']
        
        wedges, texts, autotexts = ax2.pie(pie_data, labels=pie_labels, colors=pie_colors,
                                          autopct='%1.1f%%', startangle=90)
        ax2.set_title('모델 설명력', fontweight='bold')
        
        # 파이 차트 텍스트 스타일링
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        plt.show()
        
        # 분석 요약 출력
        dominant = strength['dominant_component_korean']
        model_quality = strength['model_quality']
        
        print(f"\n📈 구성요소 분석 요약:")
        print(f"   🎯 주요 구성요소: {dominant}")
        print(f"   📊 모델 설명력: {explained:.1%}")
        print(f"   ⭐ 모델 품질: {'우수' if model_quality > 0.8 else '양호' if model_quality > 0.6 else '보통' if model_quality > 0.4 else '개선 필요'}")

# 시계열 분해 실행 예제
decomposer = TimeSeriesDecomposition()

print("\n🔬 시계열 구성요소 분해 실습")
print("=" * 60)

# 시계열 기초 클래스에서 데이터 가져오기
ts_foundation = TimeSeriesFoundation()
sample_data = ts_foundation.create_sample_data()

# 매출 데이터로 분해 시연
sales_data = sample_data['sales'].resample('D').mean()  # 일별 데이터로 리샘플링

print("📊 매출 데이터 분해 (가법적 모델)")
sales_decomp = decomposer.decompose_time_series(sales_data, method='additive', period=365)
sales_analysis = decomposer.analyze_components(sales_decomp)

# 분해 결과 시각화
decomposer.visualize_decomposition(sales_decomp, sales_analysis)

print(f"\n📈 트렌드 분석 결과:")
trend_analysis = sales_analysis['trend_analysis']
print(f"   방향: {trend_analysis['direction_korean']}")
print(f"   기울기: {trend_analysis['slope']:.6f}")
print(f"   변화율: {trend_analysis['change_rate']:.1%}")
print(f"   변화점: {len(trend_analysis['change_points'])}개")

print(f"\n🔄 계절성 분석 결과:")
seasonal_analysis = sales_analysis['seasonal_analysis']
print(f"   강도: {seasonal_analysis['intensity_korean']} ({seasonal_analysis['strength']:.3f})")
print(f"   진폭: {seasonal_analysis['amplitude']:.2f}")
print(f"   정규성: {seasonal_analysis['regularity']:.3f}")
print(f"   피크 시기: {seasonal_analysis['peak_period']}번째 주기")

print(f"\n🎲 잔차 분석 결과:")
residual_analysis = sales_analysis['residual_analysis']
print(f"   평균: {residual_analysis['mean']:.6f}")
print(f"   표준편차: {residual_analysis['std']:.3f}")
print(f"   이상값 비율: {residual_analysis['outlier_percentage']:.1f}%")
print(f"   자기상관: {residual_analysis['autocorr_lag1']:.3f}")
print(f"   백색잡음 여부: {'예' if residual_analysis['is_white_noise'] else '아니오'}")

print(f"\n⚖️ 구성요소 강도:")
component_strength = sales_analysis['component_strength']
print(f"   트렌드: {component_strength['trend_strength']:.1%}")
print(f"   계절성: {component_strength['seasonal_strength']:.1%}")
print(f"   잔차: {component_strength['residual_strength']:.1%}")
print(f"   설명력: {component_strength['explained_variance']:.1%}")

print(f"\n✅ 시계열 분해 완료!")
print(f"🔍 다음 단계: AI 협업 기반 전처리로 데이터 품질 개선")
```

---

## 4. AI 협업 기반 시계열 전처리 완전 시스템

이제 마지막으로 AI와 협업하여 시계열 데이터의 품질을 개선하는 완전한 전처리 시스템을 구축해보겠습니다!

```python
# 전처리 시스템 종합 실행
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

class TimeSeriesPreprocessor:
    """AI 협업 기반 시계열 전처리 완전 시스템"""
    
    def __init__(self):
        self.preprocessing_history = []
        
    def diagnose_data_quality(self, data: pd.Series) -> dict:
        """AI 기반 데이터 품질 진단"""
        
        print("🔍 AI 데이터 품질 진단 중...")
        
        missing_count = data.isnull().sum()
        missing_pct = (missing_count / len(data)) * 100
        
        # 이상값 탐지 (간단 버전)
        if len(data.dropna()) > 0:
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = data[(data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)]
            outlier_count = len(outliers)
        else:
            outlier_count = 0
        
        # 품질 등급 결정
        if missing_pct < 5 and outlier_count < len(data) * 0.05:
            quality_grade = "우수"
        elif missing_pct < 15 and outlier_count < len(data) * 0.1:
            quality_grade = "양호"
        else:
            quality_grade = "개선 필요"
        
        diagnosis = {
            'missing_percentage': missing_pct,
            'outlier_count': outlier_count,
            'quality_grade': quality_grade,
            'length': len(data),
            'start_date': data.index.min(),
            'end_date': data.index.max()
        }
        
        print(f"   📊 결측값: {missing_pct:.1f}%")
        print(f"   🎯 이상값: {outlier_count}개")
        print(f"   ⭐ 품질 등급: {quality_grade}")
        
        return diagnosis
    
    def handle_missing_data(self, data: pd.Series, method: str = 'interpolate') -> pd.Series:
        """결측값 처리"""
        
        print(f"🔧 결측값 처리 중 (방법: {method})...")
        
        if data.isnull().sum() == 0:
            print("✅ 결측값이 없습니다.")
            return data.copy()
        
        if method == 'interpolate':
            result = data.interpolate(method='linear')
        elif method == 'forward_fill':
            result = data.fillna(method='ffill')
        elif method == 'mean_fill':
            result = data.fillna(data.mean())
        else:
            result = data.interpolate(method='linear')
        
        remaining_missing = result.isnull().sum()
        print(f"   ✅ 처리 완료 (남은 결측값: {remaining_missing}개)")
        
        return result
    
    def handle_outliers(self, data: pd.Series, method: str = 'cap') -> pd.Series:
        """이상값 처리"""
        
        print(f"🎯 이상값 처리 중 (방법: {method})...")
        
        # IQR 방법으로 이상값 경계 계산
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_before = ((data < lower_bound) | (data > upper_bound)).sum()
        
        if method == 'cap':
            result = data.clip(lower=lower_bound, upper=upper_bound)
        elif method == 'remove':
            result = data[(data >= lower_bound) & (data <= upper_bound)]
        else:
            result = data.clip(lower=lower_bound, upper=upper_bound)
        
        outliers_after = 0 if method == 'cap' else len(data) - len(result)
        
        print(f"   ✅ 처리 완료 (처리된 이상값: {outliers_before}개)")
        
        return result
    
    def scale_data(self, data: pd.Series, method: str = 'standard') -> tuple:
        """데이터 스케일링"""
        
        print(f"📏 데이터 스케일링 중 (방법: {method})...")
        
        data_clean = data.dropna()
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        
        scaled_values = scaler.fit_transform(data_clean.values.reshape(-1, 1)).flatten()
        scaled_series = pd.Series(scaled_values, index=data_clean.index, name=data.name)
        
        # 원래 결측값 위치에 NaN 복원
        result = pd.Series(index=data.index, name=data.name)
        result[data_clean.index] = scaled_series
        
        print(f"   ✅ 스케일링 완료 ({method})")
        
        return result, scaler
    
    def create_preprocessing_pipeline(self, data: pd.Series) -> pd.Series:
        """종합 전처리 파이프라인"""
        
        print("\n🔄 AI 협업 시계열 전처리 파이프라인 실행")
        print("=" * 60)
        
        result = data.copy()
        
        # 1단계: 품질 진단
        print("🔍 1단계: 데이터 품질 진단")
        diagnosis = self.diagnose_data_quality(result)
        
        # 2단계: 결측값 처리
        print("\n🔧 2단계: 결측값 처리")
        result = self.handle_missing_data(result)
        
        # 3단계: 이상값 처리
        print("\n🎯 3단계: 이상값 처리")
        result = self.handle_outliers(result)
        
        # 4단계: 스케일링
        print("\n📏 4단계: 데이터 스케일링")
        result, scaler = self.scale_data(result)
        
        print(f"\n✅ 전처리 파이프라인 완료!")
        print(f"   📊 원본 길이: {len(data):,}")
        print(f"   📊 최종 길이: {len(result):,}")
        print(f"   🔧 적용된 단계: 4개")
        
        return result

# 전처리 시스템 실행
preprocessor = TimeSeriesPreprocessor()

print("\n🔧 AI 협업 기반 시계열 전처리 시스템 실행")
print("=" * 70)

# 샘플 데이터 준비 (일부 문제 데이터 추가)
sample_ts = sample_data['sales'].copy()

# 인위적으로 문제 요소 추가
np.random.seed(42)
missing_indices = np.random.choice(sample_ts.index, size=int(len(sample_ts) * 0.03), replace=False)
sample_ts.loc[missing_indices] = np.nan

outlier_indices = np.random.choice(sample_ts.index, size=int(len(sample_ts) * 0.02), replace=False)
sample_ts.loc[outlier_indices] = sample_ts.loc[outlier_indices] * np.random.uniform(3, 5, len(outlier_indices))

print(f"📊 전처리 대상 데이터:")
print(f"   길이: {len(sample_ts):,}개")
print(f"   결측값: {sample_ts.isnull().sum()}개 ({sample_ts.isnull().sum()/len(sample_ts):.1%})")
print(f"   기간: {sample_ts.index.min().date()} ~ {sample_ts.index.max().date()}")

# 전처리 실행
processed_data = preprocessor.create_preprocessing_pipeline(sample_ts)

# 전후 비교 시각화
fig, axes = plt.subplots(2, 1, figsize=(15, 10))
fig.suptitle('🔧 AI 협업 시계열 전처리 전후 비교', fontsize=16, fontweight='bold')

# 원본 데이터
axes[0].plot(sample_ts.index, sample_ts.values, color='red', alpha=0.7, linewidth=1, label='원본 (문제 데이터 포함)')
axes[0].set_title('📊 전처리 전 데이터', fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 처리된 데이터
axes[1].plot(processed_data.index, processed_data.values, color='blue', linewidth=1.5, label='전처리 완료')
axes[1].set_title('✨ 전처리 후 데이터', fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n🎨 전처리 시각화 이미지 생성 프롬프트:")
print("   'Before and after time series preprocessing comparison chart,")
print("   showing noisy data with missing values and outliers transformed")
print("   into clean processed data, professional analytics style'")

print(f"\n✅ 8장 Part 1 완료!")
print(f"   🔬 시계열 데이터의 4가지 특성 완전 이해")
print(f"   🤖 AI 협업을 통한 지능형 패턴 해석")
print(f"   🧩 시계열 구성요소 분해 및 분석")
print(f"   🔧 AI 기반 전처리 파이프라인 구축")
print(f"   📊 실전 데이터로 전체 워크플로우 완성")

print(f"\n🚀 다음 단계: 8장 Part 2 - 전통적 시계열 모델")
print(f"   📈 ARIMA 모델의 이해와 구현")
print(f"   🔄 지수평활법과 계절성 모델")
print(f"   🤖 AI와 함께하는 모델 선택과 최적화")
print(f"   📊 예측 성능 평가와 해석")
```

---

## 🎯 8장 Part 1 완성 요약

### 📚 학습한 핵심 내용

**🕐 시계열 데이터의 4가지 특성**
- **시간 순서의 중요성**: 데이터 순서가 정보의 핵심
- **자기상관성**: 과거가 현재를 예측하는 단서
- **비정상성**: 시간에 따라 변하는 통계적 특성
- **계절성과 주기성**: 규칙적으로 반복되는 패턴

**🤖 AI 협업 기반 시계열 분석**
- **CLEAR 프롬프트**: 시계열 특화 프롬프트 엔지니어링
- **패턴 해석**: AI가 복잡한 패턴을 비즈니스 언어로 번역
- **이상 탐지**: 다층적 접근으로 신뢰성 있는 이상값 탐지
- **품질 진단**: 종합적 데이터 품질 자동 평가

**🔬 시계열 구성요소 분해**
- **분해 방법**: 가법적, 승법적, STL 분해의 특성과 선택
- **구성요소 분석**: 트렌드, 계절성, 잔차의 개별 특성 정량화
- **품질 평가**: 분해 결과의 신뢰성과 설명력 측정
- **시각화**: 직관적 이해를 위한 차트와 대시보드

**🔧 AI 기반 전처리 시스템**
- **품질 진단**: 5개 영역의 종합적 데이터 품질 평가
- **결측값 처리**: 패턴별 맞춤형 처리 방법 자동 선택
- **이상값 탐지**: 통계적, 맥락적, 기계학습 방법의 융합
- **전처리 파이프라인**: 진단→처리→검증의 완전 자동화

### 🚀 Part 2 예고: 전통적 시계열 모델

다음 Part에서는 시계열 예측의 기초가 되는 전통적 모델들을 배우게 됩니다:

**📈 ARIMA 모델**
- 자기회귀(AR), 이동평균(MA), 차분(I)의 결합
- 모델 차수 선택과 진단 방법
- 계절성 ARIMA(SARIMA)의 활용

**🔄 지수평활법**
- 단순, 이중, 삼중 지수평활법
- Holt-Winters 계절성 모델
- 자동 최적화와 예측 구간

**🤖 AI 협업 모델링**
- 모델 선택을 위한 AI 지원
- 하이퍼파라미터 자동 튜닝
- 예측 결과 해석과 검증

Part 1에서 배운 데이터 이해와 전처리 기술이 Part 2의 모델링에서 빛을 발하게 될 것입니다!

---

## 🎨 최종 이미지 생성 프롬프트

```
Complete time series analysis workflow diagram:
- Left: Raw time series data with trend, seasonality, noise
- Center: AI-human collaboration icons with analysis tools
- Right: Clean processed data ready for modeling
- Bottom: Four decomposed components (trend, seasonal, residual)
- Modern data science aesthetic, blue and orange color scheme
- Professional analytical dashboard style
```B4', '#FF6347', '#DAA520']
        
        bars = ax1.bar(components, strengths, color=colors, alpha=0.7)
        ax1.set_title('구성요소별 분산 기여도', fontweight='bold')
        ax1.set_ylabel('분산 비율')
        ax1.grid(True, alpha=0.3)
        
        # 값 표시
        for bar, strength_val in zip(bars, strengths):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{strength_val:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # 2. 설명력 파이 차트
        explained = strength['explained_variance']
        unexplained = 1 - explained
        
        pie_data = [explained, unexplained]
        pie_labels = [f'설명됨\n({explained:.1%})', f'설명 안됨\n({unexplained:.1%})']
        pie_colors = ['#90EE90', '#FFB6C1']
        
        wedges, texts, autotexts = ax2.pie(pie_data, labels=pie_labels, colors=pie_colors,
                                          autopct='%1.1f%%', startangle=90)
        ax2.set_title('모델 설명력', fontweight='bold')
        
        # 파이 차트 텍스트 스타일링
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        plt.show()
        
        # 분석 요약 출력
        dominant = strength['dominant_component_korean']
        model_quality = strength['model_quality']
        
        print(f"\n📈 구성요소 분석 요약:")
        print(f"   🎯 주요 구성요소: {dominant} ({strengths[['트렌드', '계절성', '잔차'].index(dominant)]:.1%})")
        print(f"   📊 모델 설명력: {explained:.1%}")
        print(f"   ⭐ 모델 품질: {'우수' if model_quality > 0.8 else '양호' if model_quality > 0.6 else '보통' if model_quality > 0.4 else '개선 필요'}")
```

**🔍 코드 해설:**

**1. 분해 결과 시각화**
- **4개 서브플롯**: 원본, 트렌드, 계절성, 잔차를 세로로 배치
- **색상 구분**: 각 구성요소별로 다른 색상으로 구분
- **정보 박스**: 각 구성요소의 주요 특성을 텍스트로 표시
- **Y축 조정**: 극값으로 인한 시각적 왜곡 방지

**2. 구성요소 강도 시각화**
- **바 차트**: 각 구성요소의 분산 기여도 비교
- **파이 차트**: 전체 설명력 대 잔차 비율
- **값 표시**: 정확한 수치를 차트에 직접 표시
- **색상 일관성**: 분해 차트와 동일한 색상 체계

**3. 분석 요약**
- **주요 구성요소**: 가장 큰 영향을 미치는 요소 식별
- **모델 품질**: 분해의 적합성 평가
- **설명력**: 모델이 설명하는 변동의 비율
- **개선 방향**: 분해 품질 향상 방안 제시

---

## 4. AI 협업 기반 시계열 전처리 기법

시계열 전처리는 데이터 분석의 성공을 좌우하는 핵심 단계입니다. 7장에서 배운 AI 협업 기법을 활용하여 견고하고 지능적인 전처리 파이프라인을 구축해보겠습니다!

### 📚 시계열 전처리 핵심 개념

**📖 시계열 전처리(Time Series Preprocessing)**
- **의미**: 원시 시계열 데이터를 분석에 적합한 형태로 변환하는 과정
- **중요성**: 분석 결과의 품질과 신뢰성을 결정하는 핵심 단계
- **범위**: 결측값 처리, 이상값 탐지, 노이즈 제거, 스케일링, 변환 등

**📖 시계열 특화 전처리 요소**
- **시간 정렬**: 시간 순서의 일관성 확보
- **주기성 보존**: 계절성 패턴을 손상시키지 않는 처리
- **트렌드 유지**: 장기 추세를 왜곡하지 않는 방법 선택
- **인과관계 보존**: 미래 정보를 과거에 누설하지 않는 처리

**📖 AI 협업 전처리의 장점**
- **자동 진단**: AI가 데이터 품질 문제를 자동으로 탐지
- **최적 방법 선택**: 데이터 특성에 맞는 최적 전처리 방법 추천
- **품질 검증**: 전처리 결과의 품질을 자동으로 평가
- **설명 가능성**: 전처리 과정과 이유를 자동으로 문서화

### 4.1 AI 기반 데이터 품질 진단 시스템

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesPreprocessor:
    """
    AI 협업 기반 시계열 전처리 클래스
    
    🎯 목적: 7장에서 배운 AI 협업 원칙을 전처리에 적용
    🤖 AI 역할: 데이터 진단, 방법 추천, 품질 검증
    👨‍💼 인간 역할: 도메인 지식 적용, 비즈니스 맥락 해석
    """
    
    def __init__(self):
        """전처리 시스템 초기화"""
        print("🔧 AI 협업 시계열 전처리 시스템 초기화...")
        
        self.preprocessing_history = []  # 전처리 이력 기록
        self.quality_metrics = {}        # 품질 지표 저장
        
        # 7장에서 배운 코드 검증 원칙 적용
        self.validation_checkers = {
            'missing_data': self._validate_missing_data_handling,
            'outliers': self._validate_outlier_detection,
            'scaling': self._validate_scaling_methods,
            'smoothing': self._validate_smoothing_techniques
        }
        
        print("✅ 전처리 시스템 준비 완료!")
        
    def diagnose_data_quality(self, data: pd.Series) -> dict:
        """
        AI 기반 데이터 품질 종합 진단
        
        🔍 진단 영역:
        1. 기본 정보 분석 (길이, 타입, 메모리 등)
        2. 결측값 패턴 분석 (양, 분포, 연속성)
        3. 이상값 탐지 (통계적, 맥락적 이상값)
        4. 시간적 일관성 (주기, 간격, 중복)
        5. 통계적 특성 (분포, 정상성, 자기상관)
        """
        
        print("🔍 AI 기반 데이터 품질 종합 진단 시작...")
        
        diagnosis = {
            'basic_info': self._analyze_basic_info(data),
            'missing_data': self._analyze_missing_data(data),
            'outliers': self._detect_outliers_comprehensive(data),
            'temporal_consistency': self._check_temporal_consistency(data),
            'statistical_properties': self._analyze_statistical_properties(data)
        }
        
        # AI 기반 종합 진단 요약 생성
        diagnosis['ai_summary'] = self._generate_ai_diagnosis_summary(diagnosis)
        diagnosis['recommendations'] = self._generate_preprocessing_recommendations(diagnosis)
        
        print("✅ 데이터 품질 진단 완료!")
        return diagnosis
    
    def _analyze_basic_info(self, data: pd.Series) -> dict:
        """기본 정보 분석"""
        
        return {
            'length': len(data),
            'start_date': data.index.min(),
            'end_date': data.index.max(),
            'duration_days': (data.index.max() - data.index.min()).days,
            'frequency': self._infer_frequency(data),
            'data_type': str(data.dtype),
            'memory_usage_mb': data.memory_usage(deep=True) / (1024**2),
            'unique_values': data.nunique(),
            'duplicate_count': data.duplicated().sum()
        }
    
    def _analyze_missing_data(self, data: pd.Series) -> dict:
        """결측값 종합 분석"""
        
        missing_count = data.isnull().sum()
        missing_percentage = (missing_count / len(data)) * 100
        
        # 결측값 패턴 상세 분석
        missing_pattern = self._analyze_missing_pattern_advanced(data)
        consecutive_gaps = self._find_consecutive_missing_advanced(data)
        
        return {
            'missing_count': missing_count,
            'missing_percentage': missing_percentage,
            'pattern_type': missing_pattern['type'],
            'pattern_description': missing_pattern['description'],
            'consecutive_gaps': consecutive_gaps,
            'max_consecutive_length': max([gap['length'] for gap in consecutive_gaps]) if consecutive_gaps else 0,
            'missing_periods': self._identify_missing_periods(data)
        }
    
    def _analyze_missing_pattern_advanced(self, data: pd.Series) -> dict:
        """고급 결측값 패턴 분석"""
        
        if data.isnull().sum() == 0:
            return {'type': 'no_missing', 'description': '결측값이 없는 완전한 데이터'}
        
        missing_mask = data.isnull()
        
        # 1. 연속성 분석
        consecutive_groups = missing_mask.ne(missing_mask.shift()).cumsum()
        consecutive_missing = missing_mask.groupby(consecutive_groups).sum()
        max_consecutive = consecutive_missing.max()
        
        # 2. 주기성 분석
        missing_indices = data[missing_mask].index
        if len(missing_indices) > 2:
            # 결측값 간격의 변동성 확인
            time_diffs = pd.Series(missing_indices).diff().dropna()
            if len(time_diffs) > 1:
                cv_diffs = time_diffs.std() / time_diffs.mean() if time_diffs.mean() > 0 else float('inf')
                
                if cv_diffs < 0.1:  # 변동계수 10% 미만
                    pattern_type = 'periodic'
                    description = f'주기적 결측 (간격: {time_diffs.mean():.1f})'
                elif cv_diffs < 0.5:
                    pattern_type = 'semi_periodic'
                    description = f'준주기적 결측 (평균 간격: {time_diffs.mean():.1f}, 변동: {cv_diffs:.1%})'
                else:
                    pattern_type = 'irregular'
                    description = f'불규칙적 결측 (간격 변동: {cv_diffs:.1%})'
            else:
                pattern_type = 'sparse'
                description = '산발적 결측'
        else:
            pattern_type = 'minimal'
            description = '최소한의 결측값'
        
        # 3. 클러스터링 분석
        total_length = len(data)
        if max_consecutive > total_length * 0.1:  # 10% 이상 연속
            if max_consecutive > total_length * 0.3:  # 30% 이상
                pattern_type = 'large_block'
                description = f'대규모 블록 결측 (최대 {max_consecutive}개 연속)'
            else:
                pattern_type = 'medium_block'
                description = f'중간 블록 결측 (최대 {max_consecutive}개 연속)'
        
        return {'type': pattern_type, 'description': description}
    
    def _find_consecutive_missing_advanced(self, data: pd.Series) -> list:
        """연속 결측값 구간 상세 분석"""
        
        missing_mask = data.isnull()
        consecutive_groups = missing_mask.ne(missing_mask.shift()).cumsum()
        missing_groups = missing_mask.groupby(consecutive_groups).sum()
        
        consecutive_periods = []
        for group_id, count in missing_groups.items():
            if count > 0:  # 결측값 그룹
                group_mask = consecutive_groups == group_id
                group_data = data[group_mask]
                
                if len(group_data) > 0:
                    start_idx = group_data.index.min()
                    end_idx = group_data.index.max()
                    
                    consecutive_periods.append({
                        'start': start_idx,
                        'end': end_idx,
                        'length': count,
                        'duration': end_idx - start_idx,
                        'impact_score': count / len(data)  # 전체 대비 영향도
                    })
        
        # 영향도 순으로 정렬
        consecutive_periods.sort(key=lambda x: x['impact_score'], reverse=True)
        return consecutive_periods[:10]  # 상위 10개
    
    def _identify_missing_periods(self, data: pd.Series) -> dict:
        """결측값의 시기별 분포 분석"""
        
        missing_by_period = {}
        
        # 월별 결측값 분포
        if len(data) > 30:
            monthly_missing = data.isnull().groupby(data.index.month).sum()
            missing_by_period['monthly'] = monthly_missing.to_dict()
        
        # 요일별 결측값 분포
        if len(data) > 7:
            daily_missing = data.isnull().groupby(data.index.dayofweek).sum()
            missing_by_period['daily'] = daily_missing.to_dict()
        
        # 시간대별 결측값 분포 (시간 정보가 있는 경우)
        try:
            if hasattr(data.index, 'hour'):
                hourly_missing = data.isnull().groupby(data.index.hour).sum()
                missing_by_period['hourly'] = hourly_missing.to_dict()
        except:
            pass
        
        return missing_by_period
    
    def _detect_outliers_comprehensive(self, data: pd.Series) -> dict:
        """종합적 이상값 탐지"""
        
        print("   🎯 종합 이상값 탐지 수행 중...")
        
        data_clean = data.dropna()
        
        if len(data_clean) == 0:
            return {'method': 'insufficient_data', 'outliers': []}
        
        # 다양한 방법으로 이상값 탐지
        outlier_methods = {
            'statistical': self._detect_statistical_outliers(data_clean),
            'contextual': self._detect_contextual_outliers(data_clean),
            'isolation_forest': self._detect_isolation_outliers(data_clean),
            'local_outlier': self._detect_local_outliers(data_clean)
        }
        
        # 방법별 결과 통합
        consensus_analysis = self._analyze_outlier_consensus(outlier_methods)
        
        print(f"      탐지된 이상값: {consensus_analysis['total_suspected']}개")
        print(f"      합의 이상값: {consensus_analysis['consensus_count']}개")
        
        return {
            'methods': outlier_methods,
            'consensus': consensus_analysis,
            'severity_distribution': self._categorize_outlier_severity(data_clean, consensus_analysis)
        }
    
    def _detect_statistical_outliers(self, data: pd.Series) -> dict:
        """통계적 이상값 탐지 (Z-score, IQR, Modified Z-score)"""
        
        outliers = {}
        
        # 1. Z-score 방법
        z_scores = np.abs((data - data.mean()) / data.std())
        z_outliers = data[z_scores > 3.0]
        outliers['z_score'] = {
            'indices': z_outliers.index.tolist(),
            'values': z_outliers.tolist(),
            'scores': z_scores[z_scores > 3.0].tolist()
        }
        
        # 2. IQR 방법
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        iqr_outliers = data[(data < lower_bound) | (data > upper_bound)]
        outliers['iqr'] = {
            'indices': iqr_outliers.index.tolist(),
            'values': iqr_outliers.tolist(),
            'bounds': [lower_bound, upper_bound]
        }
        
        # 3. Modified Z-score (MAD 기반)
        median = data.median()
        mad = np.median(np.abs(data - median))
        modified_z_scores = 0.6745 * (data - median) / mad if mad > 0 else pd.Series([0] * len(data))
        mad_outliers = data[np.abs(modified_z_scores) > 3.5]
        outliers['modified_z'] = {
            'indices': mad_outliers.index.tolist(),
            'values': mad_outliers.tolist(),
            'scores': modified_z_scores[np.abs(modified_z_scores) > 3.5].tolist()
        }
        
        return outliers
    
    def _detect_contextual_outliers(self, data: pd.Series) -> dict:
        """맥락적 이상값 탐지 (시계열 특성 고려)"""
        
        outliers = {}
        
        # 1. 계절성 기반 이상값
        try:
            # 월별 기준값 계산
            monthly_stats = data.groupby(data.index.month).agg(['mean', 'std'])
            
            contextual_outliers = []
            for idx, value in data.items():
                month = idx.month
                month_mean = monthly_stats.loc[month, 'mean']
                month_std = monthly_stats.loc[month, 'std']
                
                if month_std > 0:
                    z_score = abs(value - month_mean) / month_std
                    if z_score > 2.5:  # 계절성 고려한 이상값
                        contextual_outliers.append({
                            'index': idx,
                            'value': value,
                            'expected': month_mean,
                            'z_score': z_score
                        })
            
            outliers['seasonal'] = contextual_outliers
            
        except Exception as e:
            outliers['seasonal'] = []
        
        # 2. 트렌드 기반 이상값
        try:
            # 이동평균 기준 이상값
            window = min(30, len(data) // 10)  # 적응적 윈도우 크기
            if window >= 3:
                rolling_mean = data.rolling(window=window, center=True).mean()
                rolling_std = data.rolling(window=window, center=True).std()
                
                trend_outliers = []
                for idx, value in data.items():
                    if pd.notna(rolling_mean.loc[idx]) and pd.notna(rolling_std.loc[idx]):
                        if rolling_std.loc[idx] > 0:
                            z_score = abs(value - rolling_mean.loc[idx]) / rolling_std.loc[idx]
                            if z_score > 3.0:
                                trend_outliers.append({
                                    'index': idx,
                                    'value': value,
                                    'expected': rolling_mean.loc[idx],
                                    'z_score': z_score
                                })
                
                outliers['trend'] = trend_outliers
            else:
                outliers['trend'] = []
                
        except Exception as e:
            outliers['trend'] = []
        
        return outliers
    
    def _detect_isolation_outliers(self, data: pd.Series) -> dict:
        """Isolation Forest 기반 이상값 탐지"""
        
        try:
            from sklearn.ensemble import IsolationForest
            
            # 다차원 특성 생성 (값, 차분, 지연값)
            features = pd.DataFrame({
                'value': data,
                'diff1': data.diff(),
                'diff2': data.diff().diff(),
                'lag1': data.shift(1),
                'rolling_mean': data.rolling(window=7).mean(),
                'rolling_std': data.rolling(window=7).std()
            }).fillna(method='bfill').fillna(method='ffill')
            
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = iso_forest.fit_predict(features)
            
            outlier_indices = data.index[outlier_labels == -1]
            outlier_scores = iso_forest.decision_function(features)
            
            return {
                'indices': outlier_indices.tolist(),
                'values': data[outlier_indices].tolist(),
                'scores': outlier_scores[outlier_labels == -1].tolist()
            }
            
        except ImportError:
            return {'error': 'sklearn not available', 'indices': [], 'values': []}
        except Exception as e:
            return {'error': str(e), 'indices': [], 'values': []}
    
    def _detect_local_outliers(self, data: pd.Series) -> dict:
        """LOF (Local Outlier Factor) 기반 이상값 탐지"""
        
        try:
            from sklearn.neighbors import LocalOutlierFactor
            
            # 시계열 특성을 고려한 특성 생성
            features = []
            window = min(5, len(data) // 20)
            
            for i in range(len(data)):
                start_idx = max(0, i - window)
                end_idx = min(len(data), i + window + 1)
                local_data = data.iloc[start_idx:end_idx]
                
                features.append([
                    data.iloc[i],  # 현재 값
                    local_data.mean(),  # 지역 평균
                    local_data.std(),   # 지역 표준편차
                    (data.iloc[i] - local_data.mean()) / local_data.std() if local_data.std() > 0 else 0  # 지역 Z-score
                ])
            
            features_array = np.array(features)
            
            lof = LocalOutlierFactor(n_neighbors=min(20, len(data)//5), contamination=0.1)
            outlier_labels = lof.fit_predict(features_array)
            
            outlier_indices = data.index[outlier_labels == -1]
            
            return {
                'indices': outlier_indices.tolist(),
                'values': data[outlier_indices].tolist(),
                'scores': lof.negative_outlier_factor_[outlier_labels == -1].tolist()
            }
            
        except ImportError:
            return {'error': 'sklearn not available', 'indices': [], 'values': []}
        except Exception as e:
            return {'error': str(e), 'indices': [], 'values': []}
    
    def _analyze_outlier_consensus(self, methods: dict) -> dict:
        """이상값 탐지 방법들의 합의 분석"""
        
        all_outliers = set()
        method_results = {}
        
        # 각 방법별 이상값 수집
        for method_name, method_result in methods.items():
            if isinstance(method_result, dict):
                method_outliers = set()
                
                if method_name == 'statistical':
                    for sub_method, sub_result in method_result.items():
                        method_outliers.update(sub_result.get('indices', []))
                elif method_name == 'contextual':
                    for sub_method, sub_result in method_result.items():
                        if isinstance(sub_result, list):
                            method_outliers.update([item['index'] for item in sub_result])
                else:
                    method_outliers.update(method_result.get('indices', []))
                
                method_results[method_name] = method_outliers
                all_outliers.update(method_outliers)
        
        # 합의 수준별 분류
        consensus_levels = {}
        for outlier in all_outliers:
            detection_count = sum(1 for method_outliers in method_results.values() if outlier in method_outliers)
            consensus_levels[outlier] = detection_count
        
        # 높은 합의 이상값 (2개 이상 방법에서 탐지)
        high_consensus = [outlier for outlier, count in consensus_levels.items() if count >= 2]
        
        return {
            'total_suspected': len(all_outliers),
            'consensus_outliers': high_consensus,
            'consensus_count': len(high_consensus),
            'consensus_levels': consensus_levels,
            'method_agreement': len(high_consensus) / len(all_outliers) if all_outliers else 0
        }
    
    def _categorize_outlier_severity(self, data: pd.Series, consensus: dict) -> dict:
        """이상값 심각도 분류"""
        
        severity_distribution = {'mild': 0, 'moderate': 0, 'severe': 0, 'extreme': 0}
        
        data_std = data.std()
        data_mean = data.mean()
        
        for outlier_idx in consensus['consensus_outliers']:
            if outlier_idx in data.index:
                outlier_value = data.loc[outlier_idx]
                z_score = abs(outlier_value - data_mean) / data_std if data_std > 0 else 0
                
                if z_score > 5:
                    severity_distribution['extreme'] += 1
                elif z_score > 4:
                    severity_distribution['severe'] += 1
                elif z_score > 3:
                    severity_distribution['moderate'] += 1
                else:
                    severity_distribution['mild'] += 1
        
        return severity_distribution
```

**🔍 코드 해설:**

**1. AI 기반 진단 시스템**
- **종합적 접근**: 5개 영역(기본정보, 결측값, 이상값, 시간일관성, 통계특성)을 체계적으로 분석
- **패턴 인식**: 결측값의 주기성, 클러스터링 등 복잡한 패턴 자동 탐지
- **맥락적 분석**: 단순 통계를 넘어 시계열 맥락을 고려한 분석

**2. 고급 결측값 분석**
- **패턴 분류**: 주기적, 준주기적, 불규칙적, 블록형 등 세분화된 분류
- **영향도 평가**: 각 결측 구간이 전체 분석에 미치는 영향 정량화
- **시기별 분포**: 월별, 요일별, 시간대별 결측값 패턴 분석

**3. 다층적 이상값 탐지**
- **통계적 방법**: Z-score, IQR, Modified Z-score 등 전통적 방법
- **맥락적 방법**: 계절성, 트렌드를 고려한 시계열 특화 탐지
- **기계학습 방법**: Isolation Forest, LOF 등 고급 알고리즘
- **합의 기반**: 여러 방법의 결과를 종합하여 신뢰성 향상

**4. 품질 평가 및 추천**
- **심각도 분류**: 이상값을 심각도별로 분류하여 우선순위 결정
- **방법 간 합의**: 여러 탐지 방법의 일치도로 신뢰성 평가
- **비즈니스 영향**: 품질 문제가 분석 결과에 미치는 영향 예측

### 4.2 AI 진단 요약 생성 및 전처리 추천 시스템

```python
    def _check_temporal_consistency(self, data: pd.Series) -> dict:
        """시간적 일관성 종합 검사"""
        
        print("   ⏰ 시간적 일관성 검사 중...")
        
        consistency_checks = {
            'frequency_consistency': self._check_frequency_consistency(data),
            'temporal_gaps': self._find_temporal_gaps(data),
            'duplicate_timestamps': self._find_duplicate_timestamps(data),
            'ordering_check': self._check_temporal_ordering(data)
        }
        
        return consistency_checks
    
    def _check_frequency_consistency(self, data: pd.Series) -> dict:
        """주기 일관성 상세 확인"""
        
        if len(data) < 2:
            return {'status': 'insufficient_data'}
        
        # 시간 간격 분석
        time_diffs = pd.Series(data.index).diff().dropna()
        
        if len(time_diffs) == 0:
            return {'status': 'no_time_differences'}
        
        # 가장 일반적인 간격 탐지
        mode_diff = time_diffs.mode()
        if len(mode_diff) > 0:
            expected_freq = mode_diff.iloc[0]
            
            # 일관성 비율 계산
            tolerance = expected_freq * 0.1  # 10% 허용 오차
            consistent_intervals = (abs(time_diffs - expected_freq) <= tolerance).sum()
            consistency_ratio = consistent_intervals / len(time_diffs)
            
            # 불규칙 간격 분석
            irregular_intervals = time_diffs[abs(time_diffs - expected_freq) > tolerance]
            
            return {
                'status': 'analyzed',
                'expected_frequency': expected_freq,
                'consistency_ratio': consistency_ratio,
                'irregular_count': len(irregular_intervals),
                'irregular_intervals': irregular_intervals.describe().to_dict(),
                'frequency_stability': 'stable' if consistency_ratio > 0.9 else 'unstable'
            }
        
        return {'status': 'no_clear_pattern'}
    
    def _find_temporal_gaps(self, data: pd.Series) -> list:
        """시간적 공백 상세 분석"""
        
        if len(data) < 2:
            return []
        
        expected_freq = self._infer_frequency(data)
        if expected_freq is None:
            return []
        
        try:
            # 예상 시간 범위 생성
            full_range = pd.date_range(
                start=data.index.min(), 
                end=data.index.max(), 
                freq=expected_freq
            )
            
            # 실제 데이터와 비교하여 공백 찾기
            missing_timestamps = full_range.difference(data.index)
            
            if len(missing_timestamps) == 0:
                return []
            
            # 연속된 공백 그룹화
            gaps = []
            current_gap_start = missing_timestamps[0]
            current_gap_end = missing_timestamps[0]
            
            for i in range(1, len(missing_timestamps)):
                expected_next = current_gap_end + pd.Timedelta(expected_freq)
                
                if missing_timestamps[i] <= expected_next + pd.Timedelta(expected_freq):
                    # 연속된 공백
                    current_gap_end = missing_timestamps[i]
                else:
                    # 공백 종료, 현재 공백 저장
                    gaps.append({
                        'start': current_gap_start,
                        'end': current_gap_end,
                        'duration': current_gap_end - current_gap_start,
                        'missing_points': len(missing_timestamps[(missing_timestamps >= current_gap_start) & 
                                                               (missing_timestamps <= current_gap_end)])
                    })
                    current_gap_start = missing_timestamps[i]
                    current_gap_end = missing_timestamps[i]
            
            # 마지막 공백 추가
            gaps.append({
                'start': current_gap_start,
                'end': current_gap_end,
                'duration': current_gap_end - current_gap_start,
                'missing_points': len(missing_timestamps[(missing_timestamps >= current_gap_start) & 
                                                       (missing_timestamps <= current_gap_end)])
            })
            
            # 영향도 순으로 정렬
            gaps.sort(key=lambda x: x['missing_points'], reverse=True)
            return gaps[:10]  # 상위 10개
            
        except Exception as e:
            return []
    
    def _check_temporal_ordering(self, data: pd.Series) -> dict:
        """시간 순서 정렬 확인"""
        
        is_sorted = data.index.is_monotonic_increasing
        
        if not is_sorted:
            # 정렬되지 않은 인덱스 찾기
            unsorted_positions = []
            for i in range(1, len(data.index)):
                if data.index[i] < data.index[i-1]:
                    unsorted_positions.append(i)
        else:
            unsorted_positions = []
        
        return {
            'is_properly_ordered': is_sorted,
            'unsorted_positions': unsorted_positions[:10],  # 최대 10개
            'unsorted_count': len(unsorted_positions)
        }
    
    def _analyze_statistical_properties(self, data: pd.Series) -> dict:
        """통계적 특성 종합 분석"""
        
        print("   📊 통계적 특성 분석 중...")
        
        data_clean = data.dropna()
        
        if len(data_clean) == 0:
            return {'status': 'no_valid_data'}
        
        # 기본 통계
        basic_stats = {
            'mean': data_clean.mean(),
            'median': data_clean.median(),
            'std': data_clean.std(),
            'var': data_clean.var(),
            'skewness': data_clean.skew(),
            'kurtosis': data_clean.kurtosis(),
            'min': data_clean.min(),
            'max': data_clean.max(),
            'range': data_clean.max() - data_clean.min()
        }
        
        # 분포 특성
        distribution_props = {
            'is_normal_approx': abs(basic_stats['skewness']) < 1 and abs(basic_stats['kurtosis']) < 3,
            'coefficient_of_variation': basic_stats['std'] / basic_stats['mean'] if basic_stats['mean'] != 0 else float('inf'),
            'quartiles': data_clean.quantile([0.25, 0.5, 0.75]).to_dict()
        }
        
        # 정상성 지표
        stationarity_indicators = {
            'mean_stability': self._check_mean_stability(data_clean),
            'variance_stability': self._check_variance_stability(data_clean),
            'autocorrelation_structure': self._analyze_autocorrelation_structure(data_clean)
        }
        
        return {
            'basic_stats': basic_stats,
            'distribution_properties': distribution_props,
            'stationarity_indicators': stationarity_indicators
        }
    
    def _check_mean_stability(self, data: pd.Series) -> dict:
        """평균 안정성 상세 확인"""
        
        if len(data) < 20:
            return {'status': 'insufficient_data'}
        
        # 데이터를 여러 구간으로 나누어 평균 비교
        n_segments = min(5, len(data) // 10)  # 최소 10개씩 5개 구간
        segment_size = len(data) // n_segments
        
        segment_means = []
        for i in range(n_segments):
            start_idx = i * segment_size
            end_idx = start_idx + segment_size if i < n_segments - 1 else len(data)
            segment_mean = data.iloc[start_idx:end_idx].mean()
            segment_means.append(segment_mean)
        
        overall_mean = data.mean()
        mean_deviations = [abs(seg_mean - overall_mean) for seg_mean in segment_means]
        stability_ratio = np.std(segment_means) / overall_mean if overall_mean != 0 else 0
        
        return {
            'segment_means': segment_means,
            'mean_deviations': mean_deviations,
            'stability_ratio': stability_ratio,
            'is_stable': stability_ratio < 0.1,  # 10% 이하면 안정적
            'trend_in_means': 'increasing' if segment_means[-1] > segment_means[0] else 'decreasing' if segment_means[-1] < segment_means[0] else 'stable'
        }
    
    def _check_variance_stability(self, data: pd.Series) -> dict:
        """분산 안정성 상세 확인"""
        
        if len(data) < 20:
            return {'status': 'insufficient_data'}
        
        # 이동 분산 계산
        window_size = max(len(data) // 10, 5)
        rolling_var = data.rolling(window=window_size).var().dropna()
        
        if len(rolling_var) == 0:
            return {'status': 'calculation_failed'}
        
        var_mean = rolling_var.mean()
        var_std = rolling_var.std()
        var_cv = var_std / var_mean if var_mean > 0 else float('inf')
        
        # 분산의 추세 분석
        x = np.arange(len(rolling_var))
        var_trend_slope = np.polyfit(x, rolling_var.values, 1)[0] if len(rolling_var) > 1 else 0
        
        return {
            'rolling_variance_mean': var_mean,
            'rolling_variance_std': var_std,
            'variance_cv': var_cv,
            'is_stable': var_cv < 0.3,  # 30% 이하면 안정적
            'trend_slope': var_trend_slope,
            'trend_direction': 'increasing' if var_trend_slope > 0 else 'decreasing' if var_trend_slope < 0 else 'stable'
        }
    
    def _analyze_autocorrelation_structure(self, data: pd.Series) -> dict:
        """자기상관 구조 분석"""
        
        if len(data) < 10:
            return {'status': 'insufficient_data'}
        
        # 다양한 지연에서의 자기상관 계산
        max_lag = min(20, len(data) // 4)
        autocorrelations = {}
        
        for lag in range(1, max_lag + 1):
            if lag < len(data):
                autocorr = data.autocorr(lag=lag)
                if not pd.isna(autocorr):
                    autocorrelations[lag] = autocorr
        
        if not autocorrelations:
            return {'status': 'calculation_failed'}
        
        # 유의한 자기상관 식별 (절댓값 0.2 이상)
        significant_lags = {lag: corr for lag, corr in autocorrelations.items() if abs(corr) > 0.2}
        
        # 자기상관의 패턴 분석
        if len(significant_lags) > 0:
            max_autocorr_lag = max(significant_lags, key=lambda x: abs(significant_lags[x]))
            pattern_type = 'strong_dependence'
        else:
            max_autocorr_lag = 1
            pattern_type = 'weak_dependence'
        
        return {
            'autocorrelations': autocorrelations,
            'significant_lags': significant_lags,
            'max_autocorr_lag': max_autocorr_lag,
            'max_autocorr_value': autocorrelations.get(max_autocorr_lag, 0),
            'pattern_type': pattern_type
        }
    
    def _generate_ai_diagnosis_summary(self, diagnosis: dict) -> str:
        """AI 기반 종합 진단 요약 생성"""
        
        print("   🤖 AI 진단 요약 생성 중...")
        
        # 핵심 지표 추출
        missing_pct = diagnosis['missing_data']['missing_percentage']
        outlier_count = diagnosis['outliers']['consensus']['consensus_count']
        data_length = diagnosis['basic_info']['length']
        
        # 시간 일관성 평가
        freq_consistency = diagnosis['temporal_consistency']['frequency_consistency']
        temporal_gaps = len(diagnosis['temporal_consistency']['temporal_gaps'])
        
        # 통계적 특성 평가
        stats_props = diagnosis['statistical_properties']
        is_normal = stats_props['distribution_properties']['is_normal_approx']
        mean_stable = stats_props['stationarity_indicators']['mean_stability'].get('is_stable', False)
        var_stable = stats_props['stationarity_indicators']['variance_stability'].get('is_stable', False)
        
        # 전체 품질 등급 결정
        quality_score = 0
        quality_factors = []
        
        # 결측값 평가 (25점)
        if missing_pct < 1:
            quality_score += 25
            quality_factors.append("✅ 결측값 매우 적음")
        elif missing_pct < 5:
            quality_score += 20
            quality_factors.append("🟡 결측값 적음")
        elif missing_pct < 15:
            quality_score += 10
            quality_factors.append("🟠 결측값 보통")
        else:
            quality_factors.append("❌ 결측값 많음")
        
        # 이상값 평가 (25점)
        outlier_ratio = outlier_count / data_length
        if outlier_ratio < 0.02:
            quality_score += 25
            quality_factors.append("✅ 이상값 매우 적음")
        elif outlier_ratio < 0.05:
            quality_score += 20
            quality_factors.append("🟡 이상값 적음")
        elif outlier_ratio < 0.1:
            quality_score += 10
            quality_factors.append("🟠 이상값 보통")
        else:
            quality_factors.append("❌ 이상값 많음")
        
        # 시간 일관성 평가 (25점)
        if freq_consistency.get('status') == 'analyzed':
            consistency_ratio = freq_consistency.get('consistency_ratio', 0)
            if consistency_ratio > 0.95:
                quality_score += 25
                quality_factors.append("✅ 시간 일관성 우수")
            elif consistency_ratio > 0.85:
                quality_score += 20
                quality_factors.append("🟡 시간 일관성 양호")
            elif consistency_ratio > 0.7:
                quality_score += 10
                quality_factors.append("🟠 시간 일관성 보통")
            else:
                quality_factors.append("❌ 시간 일관성 불량")
        else:
            quality_score += 15  # 기본 점수
            quality_factors.append("🟡 시간 일관성 확인 불가")
        
        # 정상성 평가 (25점)
        stationarity_score = 0
        if mean_stable:
            stationarity_score += 12.5
        if var_stable:
            stationarity_score += 12.5
        
        if stationarity_score > 20:
            quality_factors.append("✅ 정상성 양호")
        elif stationarity_score > 10:
            quality_factors.append("🟡 정상성 보통")
        else:
            quality_factors.append("❌ 정상성 불량")
        
        quality_score += stationarity_score
        
        # 등급 결정
        if quality_score >= 85:
            grade = "우수 (A)"
            grade_color = "🟢"
        elif quality_score >= 70:
            grade = "양호 (B)"
            grade_color = "🟡"
        elif quality_score >= 50:
            grade = "보통 (C)"
            grade_color = "🟠"
        else:
            grade = "개선 필요 (D)"
            grade_color = "🔴"
        
        summary = f"""
🤖 **AI 데이터 품질 종합 진단 보고서**

{grade_color} **전체 품질 등급**: {grade} (점수: {quality_score:.0f}/100)

📊 **핵심 품질 지표**:
- 데이터 크기: {data_length:,}개 관측값 ({'충분' if data_length > 100 else '부족할 수 있음'})
- 결측값: {missing_pct:.1f}% ({diagnosis['missing_data']['missing_count']:,}개)
- 이상값: {outlier_count}개 ({outlier_ratio:.1%})
- 시간 간격: {freq_consistency.get('frequency_stability', '불규칙')}
- 정상성: {'안정적' if mean_stable and var_stable else '부분적' if mean_stable or var_stable else '불안정'}

🔍 **주요 발견사항**:
{chr(10).join(f"• {factor}" for factor in quality_factors[:5])}

💡 **권장 전처리 순서**:
1. {'결측값 처리 (우선순위: 높음)' if missing_pct > 10 else '결측값 확인 (우선순위: 낮음)'}
2. {'이상값 정밀 검토 및 처리' if outlier_ratio > 0.05 else '이상값 모니터링'}
3. {'시간 일관성 확인 및 보정' if freq_consistency.get('consistency_ratio', 1) < 0.9 else '기본 검증'}
4. {'정상성 변환 검토' if not (mean_stable and var_stable) else '현재 상태 유지'}
5. 모델링에 적합한 스케일링 적용

⏱️ **예상 전처리 시간**: 
{
'45분 이상 (복합적 문제)' if quality_score < 50 
else '30분 내외 (일반적 처리)' if quality_score < 70 
else '15분 내외 (기본 정제)' if quality_score < 85 
else '5분 내외 (최소 처리)'
}

🎯 **비즈니스 영향도**: 
{
'높음 - 분석 결과 신뢰성에 심각한 영향' if quality_score < 50
else '보통 - 일부 분석 결과에 영향 가능' if quality_score < 70
else '낮음 - 대부분 분석에 적합' if quality_score < 85
else '매우 낮음 - 고품질 분석 가능'
}
        """.strip()
        
        return summary
    
    def _generate_preprocessing_recommendations(self, diagnosis: dict) -> dict:
        """진단 결과 기반 전처리 방법 추천"""
        
        recommendations = {
            'missing_data': self._recommend_missing_data_method(diagnosis['missing_data']),
            'outliers': self._recommend_outlier_method(diagnosis['outliers']),
            'scaling': self._recommend_scaling_method(diagnosis['statistical_properties']),
            'smoothing': self._recommend_smoothing_method(diagnosis),
            'transformation': self._recommend_transformation_method(diagnosis['statistical_properties'])
        }
        
        return recommendations
    
    def _recommend_missing_data_method(self, missing_analysis: dict) -> dict:
        """결측값 처리 방법 추천"""
        
        pattern_type = missing_analysis['pattern_type']
        missing_pct = missing_analysis['missing_percentage']
        max_consecutive = missing_analysis['max_consecutive_length']
        
        if missing_pct == 0:
            return {
                'method': 'none',
                'reason': '결측값이 없음',
                'priority': 'low'
            }
        elif missing_pct < 1:
            return {
                'method': 'interpolate_linear',
                'reason': '소량의 결측값으로 선형 보간 적합',
                'priority': 'low'
            }
        elif pattern_type == 'periodic':
            return {
                'method': 'seasonal_naive',
                'reason': '주기적 패턴으로 계절성 기반 보간 적합',
                'priority': 'high'
            }
        elif max_consecutive > 30:
            return {
                'method': 'advanced_interpolation',
                'reason': '장기간 연속 결측으로 고급 보간 필요',
                'priority': 'high'
            }
        elif missing_pct < 10:
            return {
                'method': 'interpolate_time',
                'reason': '중간 수준 결측값으로 시간 기반 보간 적합',
                'priority': 'medium'
            }
        else:
            return {
                'method': 'forward_fill',
                'reason': '많은 결측값으로 안전한 전진 채우기 권장',
                'priority': 'high'
            }
    
    def _recommend_outlier_method(self, outlier_analysis: dict) -> dict:
        """이상값 처리 방법 추천"""
        
        consensus = outlier_analysis['consensus']
        severity = outlier_analysis['severity_distribution']
        
        total_outliers = consensus['consensus_count']
        extreme_outliers = severity.get('extreme', 0)
        severe_outliers = severity.get('severe', 0)
        
        if total_outliers == 0:
            return {
                'method': 'none',
                'reason': '이상값이 탐지되지 않음',
                'priority': 'low'
            }
        elif extreme_outliers > 0:
            return {
                'method': 'remove',
                'reason': f'극단적 이상값 {extreme_outliers}개 발견, 제거 권장',
                'priority': 'high'
            }
        elif total_outliers < 5:
            return {
                'method': 'winsorize',
                'reason': '소수의 이상값으로 윈저화 적합',
                'priority': 'medium'
            }
        elif total_outliers > 20:
            return {
                'method': 'transform',
                'reason': '다수의 이상값으로 변환 기법 적합',
                'priority': 'high'
            }
        else:
            return {
                'method': 'cap',
                'reason': '중간 수준의 이상값으로 상한선 적용 적합',
                'priority': 'medium'
            }
    
    def _recommend_scaling_method(self, stats: dict) -> dict:
        """스케일링 방법 추천"""
        
        basic_stats = stats['basic_stats']
        distribution_props = stats['distribution_properties']
        
        cv = distribution_props['coefficient_of_variation']
        is_normal = distribution_props['is_normal_approx']
        data_range = basic_stats['range']
        
        if cv > 2 or abs(basic_stats['skewness']) > 2:
            return {
                'method': 'robust',
                'reason': '높은 변동성 또는 왜곡된 분포로 로버스트 스케일링 적합',
                'priority': 'high'
            }
        elif data_range > 1000:
            return {
                'method': 'standard',
                'reason': '넓은 값 범위로 표준화 적합',
                'priority': 'medium'
            }
        elif is_normal and data_range < 100:
            return {
                'method': 'minmax',
                'reason': '정규분포에 가깝고 범위가 좁아 정규화 적합',
                'priority': 'low'
            }
        else:
            return {
                'method': 'standard',
                'reason': '일반적인 경우로 표준화 권장',
                'priority': 'medium'
            }
    
    def _recommend_smoothing_method(self, diagnosis: dict) -> dict:
        """평활화 방법 추천"""
        
        stats = diagnosis['statistical_properties']
        outliers = diagnosis['outliers']
        
        cv = stats['distribution_properties']['coefficient_of_variation']
        outlier_count = outliers['consensus']['consensus_count']
        data_length = diagnosis['basic_info']['length']
        
        noise_level = cv + (outlier_count / data_length)  # 노이즈 수준 추정
        
        if noise_level < 0.1:
            return {
                'method': 'none',
                'reason': '노이즈 수준이 낮아 평활화 불필요',
                'priority': 'low'
            }
        elif noise_level < 0.3:
            return {
                'method': 'exponential',
                'reason': '중간 노이즈로 지수 평활화 적합',
                'priority': 'medium'
            }
        elif outlier_count > data_length * 0.05:
            return {
                'method': 'savgol',
                'reason': '많은 이상값으로 로버스트한 Savitzky-Golay 필터 적합',
                'priority': 'high'
            }
        else:
            return {
                'method': 'moving_average',
                'reason': '일반적인 노이즈로 이동평균 적합',
                'priority': 'medium'
            }
    
    def _recommend_transformation_method(self, stats: dict) -> dict:
        """데이터 변환 방법 추천"""
        
        basic_stats = stats['basic_stats']
        stationarity = stats['stationarity_indicators']
        
        skewness = abs(basic_stats['skewness'])
        mean_stable = stationarity['mean_stability'].get('is_stable', False)
        var_stable = stationarity['variance_stability'].get('is_stable', False)
        
        if not mean_stable and not var_stable:
            return {
                'method': 'differencing',
                'reason': '평균과 분산 모두 불안정하여 차분 필요',
                'priority': 'high'
            }
        elif not mean_stable:
            return {
                'method': 'detrending',
                'reason': '평균 불안정으로 디트렌딩 필요',
                'priority': 'medium'
            }
        elif skewness > 2:
            return {
                'method': 'log_transform',
                'reason': '높은 왜도로 로그 변환 권장',
                'priority': 'medium'
            }
        else:
            return {
                'method': 'none',
                'reason': '현재 상태로 분석 적합',
                'priority': 'low'
            }
```

**🔍 코드 해설:**

**1. 시간적 일관성 검사**
- **주기 일관성**: 데이터 간격의 규칙성 평가 및 허용 오차 적용
- **시간적 공백**: 예상 시간 범위와 실제 데이터 비교로 누락 구간 탐지
- **순서 정렬**: 시계열의 시간 순서 정렬 상태 확인

**2. AI 진단 요약 시스템**
- **품질 점수**: 4개 영역(결측값, 이상값, 시간일관성, 정상성)을 25점씩 평가
- **등급 분류**: A(우수), B(양호), C(보통), D(개선필요) 4단계 등급
- **비즈니스 영향**: 품질 문제가 분석 결과에 미치는 실제 영향 평가

**3. 맞춤형 추천 시스템**
- **결측값 처리**: 패턴 유형과 비율에 따른 최적 방법 선택
- **이상값 처리**: 심각도와 개수에 따른 적절한 처리 방법
- **스케일링**: 분포 특성과 데이터 범위를 고려한 방법 추천
- **변환**: 정상성과 분포 특성을 고려한 변환 방법 제안

**4. 우선순위 기반 처리**
- **High**: 즉시 처리 필요 (품질에 심각한 영향)
- **Medium**: 적절한 시점에 처리 (성능 향상 기대)
- **Low**: 선택적 처리 (미미한 개선 효과)

### 4.3 지능형 전처리 방법 구현

```python
    def _infer_frequency(self, data: pd.Series) -> str:
        """데이터 주기 지능형 추론"""
        
        if len(data) < 2:
            return None
        
        # 시간 간격 분석
        time_diffs = pd.Series(data.index).diff().dropna()
        
        if len(time_diffs) == 0:
            return None
        
        # 최빈값으로 주기 추론
        mode_diff = time_diffs.mode()
        if len(mode_diff) > 0:
            diff = mode_diff.iloc[0]
            
            # 표준 주기로 매핑
            if diff <= pd.Timedelta(seconds=1):
                return 'S'   # 초
            elif diff <= pd.Timedelta(minutes=1):
                return 'T'   # 분
            elif diff <= pd.Timedelta(hours=1):
                return 'H'   # 시간
            elif diff <= pd.Timedelta(days=1):
                return 'D'   # 일
            elif diff <= pd.Timedelta(weeks=1):
                return 'W'   # 주
            elif diff <= pd.Timedelta(days=31):
                return 'M'   # 월
            elif diff <= pd.Timedelta(days=92):
                return 'Q'   # 분기
            else:
                return 'Y'   # 년
        
        return 'D'  # 기본값
    
    def handle_missing_data(self, data: pd.Series, method: str = 'auto', **kwargs) -> pd.Series:
        """
        지능형 결측값 처리
        
        🎯 방법별 특징:
        - auto: AI가 데이터 특성에 맞는 최적 방법 자동 선택
        - forward_fill: 이전 값으로 채우기 (트렌드 유지)
        - backward_fill: 다음 값으로 채우기 (미래 정보 활용 주의)
        - interpolate_linear: 선형 보간 (균등한 변화 가정)
        - interpolate_time: 시간 가중 보간 (시간 간격 고려)
        - seasonal_naive: 계절성 기반 보간 (주기적 패턴 활용)
        - advanced_interpolation: 고급 보간 (스플라인, 다항식 등)
        """
        
        print(f"🔧 결측값 처리 시작 (방법: {method})...")
        
        missing_count = data.isnull().sum()
        if missing_count == 0:
            print("✅ 결측값이 없습니다.")
            return data.copy()
        
        print(f"   📊 결측값: {missing_count}개 ({missing_count/len(data):.1%})")
        
        # 자동 방법 선택
        if method == 'auto':
            diagnosis = self.diagnose_data_quality(data)
            recommended = diagnosis['recommendations']['missing_data']
            method = recommended['method']
            print(f"   🤖 AI 추천 방법: {method} ({recommended['reason']})")
        
        # 결측값 처리 방법별 실행
        try:
            if method == 'forward_fill':
                result = self._forward_fill_advanced(data, **kwargs)
            elif method == 'backward_fill':
                result = self._backward_fill_advanced(data, **kwargs)
            elif method == 'interpolate_linear':
                result = self._linear_interpolation_advanced(data, **kwargs)
            elif method == 'interpolate_time':
                result = self._time_interpolation_advanced(data, **kwargs)
            elif method == 'seasonal_naive':
                result = self._seasonal_naive_fill_advanced(data, **kwargs)
            elif method == 'advanced_interpolation':
                result = self._advanced_interpolation_methods(data, **kwargs)
            else:
                print(f"   ⚠️ 알 수 없는 방법: {method}, 선형 보간 사용")
                result = self._linear_interpolation_advanced(data, **kwargs)
            
            # 처리 결과 검증
            remaining_missing = result.isnull().sum()
            success_rate = (missing_count - remaining_missing) / missing_count * 100
            
            print(f"   ✅ 처리 완료: {success_rate:.1f}% 성공")
            if remaining_missing > 0:
                print(f"   ⚠️ 미처리 결측값: {remaining_missing}개")
            
            # 처리 이력 기록
            self._record_preprocessing_step('missing_data', method, {
                'original_missing': missing_count,
                'remaining_missing': remaining_missing,
                'success_rate': success_rate
            })
            
            return result
            
        except Exception as e:
            print(f"   ❌ 처리 실패: {e}")
            print("   🔄 기본 전진 채우기로 대체")
            return data.fillna(method='ffill').fillna(method='bfill')
    
    def _forward_fill_advanced(self, data: pd.Series, limit: int = None) -> pd.Series:
        """고급 전진 채우기"""
        
        if limit is None:
            # 연속 결측값의 최대 10%까지만 채우기
            max_consecutive = self._find_max_consecutive_missing(data)
            limit = max(1, max_consecutive // 10)
        
        return data.fillna(method='ffill', limit=limit)
    
    def _backward_fill_advanced(self, data: pd.Series, limit: int = None) -> pd.Series:
        """고급 후진 채우기"""
        
        if limit is None:
            max_consecutive = self._find_max_consecutive_missing(data)
            limit = max(1, max_consecutive // 10)
        
        return data.fillna(method='bfill', limit=limit)
    
    def _linear_interpolation_advanced(self, data: pd.Series, method: str = 'linear') -> pd.Series:
        """고급 선형 보간"""
        
        # 너무 긴 공백은 보간하지 않음 (최대 30개까지)
        max_gap = min(30, len(data) // 20)
        
        result = data.copy()
        missing_mask = result.isnull()
        
        # 연속 결측값 그룹 처리
        consecutive_groups = missing_mask.ne(missing_mask.shift()).cumsum()
        
        for group_id in consecutive_groups[missing_mask].unique():
            group_mask = (consecutive_groups == group_id) & missing_mask
            group_size = group_mask.sum()
            
            if group_size <= max_gap:
                # 짧은 공백만 보간
                group_indices = result[group_mask].index
                result.loc[group_indices] = result.interpolate(method=method).loc[group_indices]
        
        return result
    
    def _time_interpolation_advanced(self, data: pd.Series) -> pd.Series:
        """시간 가중 보간"""
        
        return data.interpolate(method='time', limit_area='inside')
    
    def _seasonal_naive_fill_advanced(self, data: pd.Series, period: int = None) -> pd.Series:
        """고급 계절성 기반 채우기"""
        
        if period is None:
            period = self._estimate_seasonal_period_advanced(data)
        
        result = data.copy()
        missing_mask = result.isnull()
        
        for idx in result[missing_mask].index:
            # 해당 인덱스의 위치 찾기
            try:
                idx_pos = result.index.get_loc(idx)
                
                # 같은 계절의 값들 찾기
                seasonal_candidates = []
                
                # 이전 주기들 확인
                for offset in [-period, -2*period, -3*period]:
                    candidate_pos = idx_pos + offset
                    if 0 <= candidate_pos < len(result):
                        candidate_idx = result.index[candidate_pos]
                        if pd.notna(result.loc[candidate_idx]):
                            seasonal_candidates.append(result.loc[candidate_idx])
                
                # 이후 주기들 확인
                for offset in [period, 2*period, 3*period]:
                    candidate_pos = idx_pos + offset
                    if 0 <= candidate_pos < len(result):
                        candidate_idx = result.index[candidate_pos]
                        if pd.notna(result.loc[candidate_idx]):
                            seasonal_candidates.append(result.loc[candidate_idx])
                
                # 가중 평균 계산 (가까운 주기일수록 높은 가중치)
                if seasonal_candidates:
                    weights = [1/abs(i-len(seasonal_candidates)//2) if i != len(seasonal_candidates)//2 else 1 
                              for i in range(len(seasonal_candidates))]
                    weighted_avg = np.average(seasonal_candidates, weights=weights)
                    result.loc[idx] = weighted_avg
                    
            except Exception:
                continue
        
        return result
    
    def _estimate_seasonal_period_advanced(self, data: pd.Series) -> int:
        """고급 계절성 주기 추정"""
        
        # 주기 추론을 위한 자기상관 분석
        clean_data = data.dropna()
        if len(clean_data) < 20:
            return 12  # 기본값
        
        # 다양한 주기 후보 테스트
        max_period = min(len(clean_data) // 4, 365)
        candidates = []
        
        # 일반적인 주기들 테스트
        common_periods = [7, 12, 24, 30, 52, 365]
        
        for period in common_periods:
            if period < max_period:
                autocorr = clean_data.autocorr(lag=period)
                if pd.notna(autocorr):
                    candidates.append((period, abs(autocorr)))
        
        # 가장 강한 자기상관을 가진 주기 선택
        if candidates:
            best_period = max(candidates, key=lambda x: x[1])[0]
            return best_period
        
        # 기본값
        freq = self._infer_frequency(data)
        period_map = {
            'S': 60, 'T': 60, 'H': 24, 'D': 7, 
            'W': 52, 'M': 12, 'Q': 4, 'Y': 1
        }
        return period_map.get(freq, 12)
    
    def _advanced_interpolation_methods(self, data: pd.Series, method: str = 'cubic') -> pd.Series:
        """고급 보간 방법들"""
        
        available_methods = ['cubic', 'spline', 'polynomial', 'akima']
        
        if method not in available_methods:
            method = 'cubic'
        
        try:
            if method == 'cubic':
                return data.interpolate(method='cubic')
            elif method == 'spline':
                return data.interpolate(method='spline', order=3)
            elif method == 'polynomial':
                return data.interpolate(method='polynomial', order=2)
            elif method == 'akima':
                return data.interpolate(method='akima')
            else:
                return data.interpolate(method='cubic')
        except Exception:
            # 실패 시 안전한 방법 사용
            return data.interpolate(method='linear')
    
    def _find_max_consecutive_missing(self, data: pd.Series) -> int:
        """최대 연속 결측값 길이 찾기"""
        
        missing_mask = data.isnull()
        consecutive_groups = missing_mask.ne(missing_mask.shift()).cumsum()
        consecutive_missing = missing_mask.groupby(consecutive_groups).sum()
        return consecutive_missing.max() if len(consecutive_missing) > 0 else 0
    
    def _record_preprocessing_step(self, operation: str, method: str, metrics: dict):
        """전처리 단계 기록"""
        
        self.preprocessing_history.append({
            'timestamp': datetime.now(),
            'operation': operation,
            'method': method,
            'metrics': metrics
        })
```

**🔍 코드 해설:**

**1. 지능형 방법 선택**
- **auto 모드**: AI 진단 결과를 바탕으로 최적 방법 자동 선택
- **데이터 특성 고려**: 결측값 패턴, 비율, 연속성을 종합 평가
- **안전장치**: 처리 실패시 기본 방법으로 자동 전환

**2. 고급 전처리 기법**
- **제한적 채우기**: 과도한 채우기를 방지하는 limit 설정
- **시간 가중 보간**: 시간 간격을 고려한 보간으로 더 정확한 추정
- **계절성 활용**: 주기적 패턴을 활용한 지능적 결측값 추정
- **공백 크기 제한**: 너무 긴 공백은 보간하지 않아 신뢰성 확보

**3. 품질 검증 시스템**
- **성공률 계산**: 처리된 결측값 비율로 방법의 효과성 측정
- **잔여 결측값**: 처리되지 않은 결측값에 대한 후속 조치 안내
- **이력 관리**: 모든 전처리 과정을 기록하여 재현성 확보

**4. 계절성 지능형 분석**
- **자기상관 기반**: 다양한 주기에서의 자기상관으로 최적 주기 탐지
- **가중 평균**: 시간적 거리에 따른 가중치로 더 정확한 추정
- **다중 주기 고려**: 여러 주기의 정보를 종합하여 로버스트한 추정

### 4.4 이상값 처리 및 스케일링 기법

```python
    def handle_outliers(self, data: pd.Series, method: str = 'auto', **kwargs) -> pd.Series:
        """
        지능형 이상값 처리
        
        🎯 방법별 특징:
        - auto: AI가 이상값 특성에 맞는 최적 방법 자동 선택
        - remove: 이상값 제거 (극단적 값에 적합)
        - winsorize: 윈저화 (극값을 분위수로 대체)
        - cap: 상한/하한선 설정 (IQR 또는 Z-score 기준)
        - transform: 변환을 통한 이상값 완화 (로그, 제곱근 등)
        - interpolate: 보간으로 이상값 대체
        """
        
        print(f"🎯 이상값 처리 시작 (방법: {method})...")
        
        # 자동 방법 선택
        if method == 'auto':
            diagnosis = self.diagnose_data_quality(data)
            recommended = diagnosis['recommendations']['outliers']
            method = recommended['method']
            print(f"   🤖 AI 추천 방법: {method} ({recommended['reason']})")
        
        # 이상값 탐지
        outlier_info = self._detect_outliers_comprehensive(data.dropna())
        consensus_outliers = outlier_info['consensus']['consensus_outliers']
        
        if not consensus_outliers:
            print("   ✅ 처리할 이상값이 없습니다.")
            return data.copy()
        
        print(f"   📊 탐지된 이상값: {len(consensus_outliers)}개")
        
        # 방법별 처리
        try:
            if method == 'remove':
                result = self._remove_outliers(data, consensus_outliers, **kwargs)
            elif method == 'winsorize':
                result = self._winsorize_outliers(data, **kwargs)
            elif method == 'cap':
                result = self._cap_outliers(data, **kwargs)
            elif method == 'transform':
                result = self._transform_outliers(data, **kwargs)
            elif method == 'interpolate':
                result = self._interpolate_outliers(data, consensus_outliers, **kwargs)
            else:
                print(f"   ⚠️ 알 수 없는 방법: {method}, 윈저화 사용")
                result = self._winsorize_outliers(data, **kwargs)
            
            # 처리 결과 검증
            processed_count = len(consensus_outliers)
            remaining_outliers = len(self._detect_outliers_comprehensive(result.dropna())['consensus']['consensus_outliers'])
            reduction_rate = (processed_count - remaining_outliers) / processed_count * 100 if processed_count > 0 else 0
            
            print(f"   ✅ 처리 완료: {reduction_rate:.1f}% 개선")
            if remaining_outliers > 0:
                print(f"   ⚠️ 잔여 이상값: {remaining_outliers}개")
            
            # 처리 이력 기록
            self._record_preprocessing_step('outliers', method, {
                'original_outliers': processed_count,
                'remaining_outliers': remaining_outliers,
                'reduction_rate': reduction_rate
            })
            
            return result
            
        except Exception as e:
            print(f"   ❌ 처리 실패: {e}")
            print("   🔄 기본 윈저화로 대체")
            return self._winsorize_outliers(data, percentile=0.05)
    
    def _remove_outliers(self, data: pd.Series, outlier_indices: list, **kwargs) -> pd.Series:
        """이상값 제거"""
        
        safety_limit = kwargs.get('max_removal_ratio', 0.1)  # 최대 10%까지만 제거
        
        if len(outlier_indices) > len(data) * safety_limit:
            print(f"   ⚠️ 제거 대상이 너무 많음 ({len(outlier_indices)}개), 상위 {int(len(data) * safety_limit)}개만 제거")
            # 가장 극단적인 값들만 제거
            outlier_values = [(idx, abs(data.loc[idx] - data.median())) for idx in outlier_indices if idx in data.index]
            outlier_values.sort(key=lambda x: x[1], reverse=True)
            outlier_indices = [idx for idx, _ in outlier_values[:int(len(data) * safety_limit)]]
        
        result = data.copy()
        valid_indices = [idx for idx in outlier_indices if idx in result.index]
        result = result.drop(valid_indices)
        
        print(f"   🗑️ {len(valid_indices)}개 이상값 제거됨")
        return result
    
    def _winsorize_outliers(self, data: pd.Series, percentile: float = 0.05, **kwargs) -> pd.Series:
        """윈저화 (극값을 분위수로 대체)"""
        
        lower_bound = data.quantile(percentile)
        upper_bound = data.quantile(1 - percentile)
        
        result = data.copy()
        
        # 극값 대체
        lower_replacements = (result < lower_bound).sum()
        upper_replacements = (result > upper_bound).sum()
        
        result = result.clip(lower=lower_bound, upper=upper_bound)
        
        print(f"   📊 윈저화 완료: 하한 {lower_replacements}개, 상한 {upper_replacements}개 대체")
        print(f"   📏 범위: [{lower_bound:.3f}, {upper_bound:.3f}]")
        
        return result
    
    def _cap_outliers(self, data: pd.Series, method: str = 'iqr', multiplier: float = 1.5, **kwargs) -> pd.Series:
        """상한/하한선 적용"""
        
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
        else:  # z-score
            mean_val = data.mean()
            std_val = data.std()
            lower_bound = mean_val - multiplier * std_val
            upper_bound = mean_val + multiplier * std_val
        
        result = data.clip(lower=lower_bound, upper=upper_bound)
        
        capped_count = ((data < lower_bound) | (data > upper_bound)).sum()
        print(f"   🔒 상한선 적용: {capped_count}개 값 조정")
        print(f"   📏 적용 범위: [{lower_bound:.3f}, {upper_bound:.3f}]")
        
        return result
    
    def _transform_outliers(self, data: pd.Series, transform_type: str = 'log', **kwargs) -> pd.Series:
        """변환을 통한 이상값 완화"""
        
        if transform_type == 'log':
            # 로그 변환 (양수만 가능)
            if (data <= 0).any():
                # 음수가 있으면 상수 더하기
                min_val = data.min()
                offset = abs(min_val) + 1
                result = np.log(data + offset)
                print(f"   📊 로그 변환 (오프셋 +{offset:.3f}) 적용")
            else:
                result = np.log(data)
                print(f"   📊 로그 변환 적용")
                
        elif transform_type == 'sqrt':
            # 제곱근 변환
            if (data < 0).any():
                offset = abs(data.min()) + 1
                result = np.sqrt(data + offset)
                print(f"   📊 제곱근 변환 (오프셋 +{offset:.3f}) 적용")
            else:
                result = np.sqrt(data)
                print(f"   📊 제곱근 변환 적용")
                
        elif transform_type == 'box_cox':
            # Box-Cox 변환
            try:
                from scipy.stats import boxcox
                if (data > 0).all():
                    result, lambda_param = boxcox(data)
                    result = pd.Series(result, index=data.index, name=data.name)
                    print(f"   📊 Box-Cox 변환 적용 (λ={lambda_param:.3f})")
                else:
                    print(f"   ⚠️ Box-Cox 변환 불가 (음수값 존재), 로그 변환 사용")
                    return self._transform_outliers(data, 'log')
            except ImportError:
                print(f"   ⚠️ scipy 불가용, 로그 변환 사용")
                return self._transform_outliers(data, 'log')
        else:
            result = data.copy()
            print(f"   ⚠️ 알 수 없는 변환: {transform_type}")
        
        return pd.Series(result, index=data.index, name=data.name)
    
    def _interpolate_outliers(self, data: pd.Series, outlier_indices: list, **kwargs) -> pd.Series:
        """보간으로 이상값 대체"""
        
        result = data.copy()
        valid_indices = [idx for idx in outlier_indices if idx in result.index]
        
        # 이상값을 NaN으로 마킹한 후 보간
        result.loc[valid_indices] = np.nan
        result = result.interpolate(method='linear', limit_direction='both')
        
        # 여전히 NaN인 경우 주변값으로 채우기
        result = result.fillna(method='ffill').fillna(method='bfill')
        
        print(f"   🔄 {len(valid_indices)}개 이상값 보간으로 대체")
        return result
    
    def scale_data(self, data: pd.Series, method: str = 'auto', **kwargs) -> tuple:
        """
        지능형 데이터 스케일링
        
        🎯 방법별 특징:
        - auto: AI가 데이터 분포에 맞는 최적 방법 자동 선택
        - standard: 표준화 (평균 0, 표준편차 1)
        - minmax: 정규화 (0-1 범위)
        - robust: 로버스트 스케일링 (중앙값과 IQR 사용)
        - quantile: 분위수 변환 (균등분포로 변환)
        """
        
        print(f"📏 데이터 스케일링 시작 (방법: {method})...")
        
        # 자동 방법 선택
        if method == 'auto':
            diagnosis = self.diagnose_data_quality(data)
            recommended = diagnosis['recommendations']['scaling']
            method = recommended['method']
            print(f"   🤖 AI 추천 방법: {method} ({recommended['reason']})")
        
        data_clean = data.dropna()
        if len(data_clean) == 0:
            print("   ❌ 유효한 데이터가 없습니다.")
            return data.copy(), None
        
        try:
            if method == 'standard':
                scaled_data, scaler_info = self._standard_scaling(data, **kwargs)
            elif method == 'minmax':
                scaled_data, scaler_info = self._minmax_scaling(data, **kwargs)
            elif method == 'robust':
                scaled_data, scaler_info = self._robust_scaling(data, **kwargs)
            elif method == 'quantile':
                scaled_data, scaler_info = self._quantile_scaling(data, **kwargs)
            else:
                print(f"   ⚠️ 알 수 없는 방법: {method}, 표준화 사용")
                scaled_data, scaler_info = self._standard_scaling(data, **kwargs)
            
            # 스케일링 효과 검증
            original_range = data_clean.max() - data_clean.min()
            scaled_range = scaled_data.dropna().max() - scaled_data.dropna().min()
            
            print(f"   ✅ 스케일링 완료")
            print(f"   📊 원본 범위: {original_range:.3f} → 스케일링 후: {scaled_range:.3f}")
            
            # 처리 이력 기록
            self._record_preprocessing_step('scaling', method, {
                'original_range': original_range,
                'scaled_range': scaled_range,
                'scaler_params': scaler_info
            })
            
            return scaled_data, scaler_info
            
        except Exception as e:
            print(f"   ❌ 스케일링 실패: {e}")
            return data.copy(), None
    
    def _standard_scaling(self, data: pd.Series, **kwargs) -> tuple:
        """표준화 (Z-score)"""
        
        mean_val = data.mean()
        std_val = data.std()
        
        if std_val == 0:
            print("   ⚠️ 표준편차가 0입니다. 스케일링하지 않습니다.")
            return data.copy(), {'mean': mean_val, 'std': 1}
        
        scaled = (data - mean_val) / std_val
        scaler_info = {'method': 'standard', 'mean': mean_val, 'std': std_val}
        
        print(f"   📊 표준화: 평균 {mean_val:.3f} → 0, 표준편차 {std_val:.3f} → 1")
        return scaled, scaler_info
    
    def _minmax_scaling(self, data: pd.Series, feature_range: tuple = (0, 1), **kwargs) -> tuple:
        """정규화 (Min-Max)"""
        
        min_val = data.min()
        max_val = data.max()
        data_range = max_val - min_val
        
        if data_range == 0:
            print("   ⚠️ 데이터 범위가 0입니다. 스케일링하지 않습니다.")
            return data.copy(), {'min': min_val, 'range': 1}
        
        # 지정된 범위로 스케일링
        range_min, range_max = feature_range
        scaled = (data - min_val) / data_range * (range_max - range_min) + range_min
        
        scaler_info = {
            'method': 'minmax',
            'min': min_val,
            'range': data_range,
            'feature_range': feature_range
        }
        
        print(f"   📊 정규화: [{min_val:.3f}, {max_val:.3f}] → [{range_min}, {range_max}]")
        return scaled, scaler_info
    
    def _robust_scaling(self, data: pd.Series, **kwargs) -> tuple:
        """로버스트 스케일링 (중앙값과 IQR)"""
        
        median_val = data.median()
        q75 = data.quantile(0.75)
        q25 = data.quantile(0.25)
        iqr = q75 - q25
        
        if iqr == 0:
            print("   ⚠️ IQR이 0입니다. 표준편차를 사용합니다.")
            iqr = data.std()
            if iqr == 0:
                return data.copy(), {'median': median_val, 'iqr': 1}
        
        scaled = (data - median_val) / iqr
        scaler_info = {'method': 'robust', 'median': median_val, 'iqr': iqr}
        
        print(f"   📊 로버스트 스케일링: 중앙값 {median_val:.3f}, IQR {iqr:.3f}")
        return scaled, scaler_info
    
    def _quantile_scaling(self, data: pd.Series, n_quantiles: int = 1000, **kwargs) -> tuple:
        """분위수 변환"""
        
        try:
            from sklearn.preprocessing import QuantileTransformer
            
            qt = QuantileTransformer(n_quantiles=min(n_quantiles, len(data)), 
                                   output_distribution='uniform', 
                                   random_state=42)
            
            data_2d = data.values.reshape(-1, 1)
            scaled_2d = qt.fit_transform(data_2d)
            scaled = pd.Series(scaled_2d.flatten(), index=data.index, name=data.name)
            
            scaler_info = {'method': 'quantile', 'quantile_transformer': qt}
            
            print(f"   📊 분위수 변환: {n_quantiles}개 분위수 사용")
            return scaled, scaler_info
            
        except ImportError:
            print("   ⚠️ sklearn 불가용, 로버스트 스케일링 사용")
            return self._robust_scaling(data, **kwargs)
    
    def smooth_data(self, data: pd.Series, method: str = 'auto', **kwargs) -> pd.Series:
        """
        지능형 데이터 평활화
        
        🎯 방법별 특징:
        - auto: AI가 노이즈 수준에 맞는 최적 방법 자동 선택
        - moving_average: 이동평균 (단순하고 효과적)
        - exponential: 지수평활 (최근 값에 더 큰 가중치)
        - savgol: Savitzky-Golay 필터 (트렌드 보존)
        - lowess: LOWESS 평활화 (로컬 회귀)
        """
        
        print(f"🌊 데이터 평활화 시작 (방법: {method})...")
        
        # 자동 방법 선택
        if method == 'auto':
            diagnosis = self.diagnose_data_quality(data)
            recommended = diagnosis['recommendations']['smoothing']
            method = recommended['method']
            print(f"   🤖 AI 추천 방법: {method} ({recommended['reason']})")
        
        if method == 'none':
            print("   ✅ 평활화가 필요하지 않습니다.")
            return data.copy()
        
        try:
            if method == 'moving_average':
                result = self._moving_average_smooth(data, **kwargs)
            elif method == 'exponential':
                result = self._exponential_smooth(data, **kwargs)
            elif method == 'savgol':
                result = self._savgol_smooth(data, **kwargs)
            elif method == 'lowess':
                result = self._lowess_smooth(data, **kwargs)
            else:
                print(f"   ⚠️ 알 수 없는 방법: {method}, 이동평균 사용")
                result = self._moving_average_smooth(data, **kwargs)
            
            # 평활화 효과 측정
            original_volatility = data.std()
            smoothed_volatility = result.std()
            noise_reduction = (original_volatility - smoothed_volatility) / original_volatility * 100
            
            print(f"   ✅ 평활화 완료: 노이즈 {noise_reduction:.1f}% 감소")
            
            # 처리 이력 기록
            self._record_preprocessing_step('smoothing', method, {
                'original_volatility': original_volatility,
                'smoothed_volatility': smoothed_volatility,
                'noise_reduction': noise_reduction
            })
            
            return result
            
        except Exception as e:
            print(f"   ❌ 평활화 실패: {e}")
            return data.copy()
    
    def _moving_average_smooth(self, data: pd.Series, window: int = None, **kwargs) -> pd.Series:
        """이동평균 평활화"""
        
        if window is None:
            # 데이터 길이에 따른 적응적 윈도우
            window = max(3, min(len(data) // 20, 21))
        
        # 홀수로 만들어 중앙값 계산
        if window % 2 == 0:
            window += 1
        
        result = data.rolling(window=window, center=True, min_periods=1).mean()
        print(f"   📊 이동평균: 윈도우 크기 {window}")
        return result
    
    def _exponential_smooth(self, data: pd.Series, alpha: float = None, **kwargs) -> pd.Series:
        """지수평활화"""
        
        if alpha is None:
            # 데이터 특성에 따른 적응적 알파
            autocorr = data.autocorr(lag=1) if len(data) > 1 else 0
            alpha = 0.3 if abs(autocorr) > 0.5 else 0.1  # 자기상관 높으면 낮은 알파
        
        result = data.ewm(alpha=alpha, adjust=False).mean()
        print(f"   📊 지수평활: 알파 {alpha:.3f}")
        return result
    
    def _savgol_smooth(self, data: pd.Series, window_length: int = None, polyorder: int = 3, **kwargs) -> pd.Series:
        """Savitzky-Golay 필터"""
        
        data_clean = data.dropna()
        if len(data_clean) < 5:
            print("   ⚠️ 데이터가 너무 적어 이동평균 사용")
            return self._moving_average_smooth(data, **kwargs)
        
        if window_length is None:
            window_length = min(max(len(data_clean) // 10, 5), 51)
            if window_length % 2 == 0:  # 홀수여야 함
                window_length += 1
        
        polyorder = min(polyorder, window_length - 1)
        
        try:
            smoothed_values = savgol_filter(data_clean.values, window_length, polyorder)
            result = pd.Series(data.values, index=data.index, name=data.name)
            result.loc[data_clean.index] = smoothed_values
            
            print(f"   📊 Savitzky-Golay: 윈도우 {window_length}, 차수 {polyorder}")
            return result
        except Exception as e:
            print(f"   ⚠️ Savitzky-Golay 실패: {e}, 이동평균 사용")
            return self._moving_average_smooth(data, **kwargs)
    
    def _lowess_smooth(self, data: pd.Series, frac: float = 0.1, **kwargs) -> pd.Series:
        """LOWESS 평활화"""
        
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            
            # 시간을 숫자로 변환
            x = np.arange(len(data))
            y = data.values
            
            # 결측값 제거
            valid_mask = ~np.isnan(y)
            x_clean = x[valid_mask]
            y_clean = y[valid_mask]
            
            if len(x_clean) < 3:
                return self._moving_average_smooth(data, **kwargs)
            
            # LOWESS 실행
            smoothed = lowess(y_clean, x_clean, frac=frac, return_sorted=False)
            
            # 결과 재구성
            result = pd.Series(data.values, index=data.index, name=data.name)
            result.iloc[valid_mask] = smoothed
            
            print(f"   📊 LOWESS: 비율 {frac:.3f}")
            return result
            
        except ImportError:
            print("   ⚠️ statsmodels 불가용, 이동평균 사용")
            return self._moving_average_smooth(data, **kwargs)
        except Exception as e:
            print(f"   ⚠️ LOWESS 실패: {e}, 이동평균 사용")
            return self._moving_average_smooth(data, **kwargs)
```

**🔍 코드 해설:**

**1. 이상값 처리 방법별 특징**
- **제거**: 극단적 이상값에 적합하지만 데이터 손실 발생
- **윈저화**: 극값을 분위수로 대체하여 분포 보존
- **상한선**: IQR이나 Z-score 기준으로 경계값 설정
- **변환**: 로그, 제곱근 등으로 이상값 영향 완화
- **보간**: 이상값을 주변값 기반으로 추정

**2. 스케일링 방법별 선택 기준**
- **표준화**: 정규분포에 가깝고 이상값이 적을 때
- **정규화**: 분포가 균등하고 범위가 좁을 때
- **로버스트**: 이상값이 많거나 분포가 왜곡되었을 때
- **분위수**: 분포가 매우 복잡하거나 비선형일 때

**3. 평활화 기법별 특성**
- **이동평균**: 가장 단순하고 해석하기 쉬움
- **지수평활**: 최근 데이터에 더 큰 가중치, 트렌드 추적
- **Savitzky-Golay**: 트렌드와 피크를 보존하는 고급 필터
- **LOWESS**: 로컬 회귀 기반의 유연한 평활화

**4. 안전장치 및 검증**
- **처리 한계**: 과도한 데이터 손실 방지 (최대 10% 제거)
- **효과 측정**: 처리 전후 비교로 개선 효과 정량화
- **대체 방법**: 실패시 안전한 기본 방법으로 자동 전환
- **이력 관리**: 모든 처리 과정을 상세히 기록

