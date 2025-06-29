# 8장 Part 2: 전통적 시계열 모델(ARIMA, 지수평활법)
**부제: 수십 년 검증된 시계열 예측의 핵심 기법들**

## 학습 목표
이 Part를 완료한 후, 여러분은 다음을 할 수 있게 됩니다:
- 자기회귀(AR), 이동평균(MA), ARIMA 모델의 원리를 깊이 이해할 수 있다
- Box-Jenkins 방법론에 따라 체계적으로 ARIMA 모델을 구축할 수 있다
- 계절성 SARIMA 모델과 지수평활법을 실무에 적용할 수 있다
- 7장 AI 협업 기법을 활용하여 모델 선택과 해석을 최적화할 수 있다
- Store Sales 데이터로 완전한 시계열 예측 시스템을 구축할 수 있다

## 이번 Part 미리보기
📈 시계열 예측의 역사는 1970년 George Box와 Gwilym Jenkins가 제시한 ARIMA 모델로 혁명을 맞이했습니다. 50년이 넘도록 금융, 경제, 기상, 제조업 등 모든 분야에서 핵심 예측 도구로 사용되고 있는 이 **전통적 시계열 모델들**은 여전히 현대 AI 시대에도 중요한 역할을 하고 있습니다.

8장 Part 1에서 우리는 시계열의 기본 특성과 전처리를 마스터했습니다. 이제 그 기반 위에 **검증된 예측 모델들**을 구축해보겠습니다. 특히 7장에서 배운 AI 협업 기법을 활용하여 복잡한 수학적 개념을 직관적으로 이해하고, 최적의 모델을 효율적으로 찾아보겠습니다.

🎯 **이번 Part의 핵심 여정**:
- **AR/MA/ARIMA**: 시계열의 과거가 미래를 예측하는 원리 완전 이해
- **Box-Jenkins 방법론**: 체계적인 모델 구축 프로세스 마스터  
- **SARIMA**: 계절성까지 고려한 고급 모델링 기법
- **지수평활법**: 직관적이고 실용적인 예측 방법들
- **AI 협업 최적화**: 프롬프트 엔지니어링으로 모델 해석력 극대화

---

> 🚀 **왜 전통적 모델이 여전히 중요한가?**
> 
> **🔍 해석 가능성**: 딥러닝과 달리 모든 계수의 의미를 명확히 설명 가능
> **⚡ 효율성**: 적은 데이터로도 안정적 예측, 빠른 학습과 추론
> **🎯 정확성**: 단기 예측에서는 여전히 최고 성능 유지
> **🔧 실무성**: 비즈니스 환경에서 검증된 신뢰도와 안정성
> **💡 기초 이해**: 모든 고급 모델의 이론적 토대 제공

## 1. 자기회귀 및 이동평균 모델의 이해

### 1.1 시계열 모델링의 기본 철학

시계열 예측의 핵심은 **"과거 패턴이 미래에도 반복된다"**는 가정입니다. 하지만 이 단순한 아이디어를 수학적으로 구현하는 방법은 매우 다양합니다.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 시각화 설정
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class TraditionalTimeSeriesModels:
    """전통적 시계열 모델 클래스"""
    
    def __init__(self):
        self.models = {}
        self.model_results = {}
        self.forecasts = {}
        
        # 7장에서 배운 AI 협업 원칙 적용
        self.interpretation_prompts = {
            'ar_model': self._create_ar_interpretation_prompt(),
            'ma_model': self._create_ma_interpretation_prompt(),
            'arima_model': self._create_arima_interpretation_prompt(),
            'sarima_model': self._create_sarima_interpretation_prompt()
        }
    
    def demonstrate_basic_concepts(self):
        """기본 개념 시연"""
        
        print("🎓 시계열 모델링 기본 개념")
        print("=" * 50)
        
        # 1. 백색잡음 (White Noise) - 가장 단순한 시계열
        np.random.seed(42)
        white_noise = np.random.normal(0, 1, 200)
        
        # 2. 랜덤워크 (Random Walk) - 누적 백색잡음
        random_walk = np.cumsum(white_noise)
        
        # 3. 트렌드가 있는 시계열
        trend_series = 0.5 * np.arange(200) + white_noise
        
        # 4. 계절성이 있는 시계열
        seasonal_component = 10 * np.sin(2 * np.pi * np.arange(200) / 50)
        seasonal_series = trend_series + seasonal_component
        
        # 시각화
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('🎯 시계열의 기본 패턴들', fontsize=16, fontweight='bold')
        
        series_data = [
            (white_noise, '🎲 백색잡음 (White Noise)', '완전히 예측 불가능한 랜덤 시계열'),
            (random_walk, '🚶 랜덤워크 (Random Walk)', '이전 값에서 랜덤하게 변화'),
            (trend_series, '📈 트렌드 시계열', '일정한 방향성을 가진 변화'),
            (seasonal_series, '🔄 계절성 시계열', '주기적 패턴이 반복')
        ]
        
        for i, (data, title, description) in enumerate(series_data):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            ax.plot(data, linewidth=1.5, alpha=0.8)
            ax.set_title(title, fontweight='bold')
            ax.text(0.02, 0.95, description, transform=ax.transAxes, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                   fontsize=9, verticalalignment='top')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("\n🔍 각 패턴의 특징:")
        print(f"   백색잡음 분산: {np.var(white_noise):.3f} (항상 일정)")
        print(f"   랜덤워크 분산: {np.var(random_walk):.3f} (시간에 따라 증가)")
        print(f"   트렌드 기울기: {np.polyfit(range(200), trend_series, 1)[0]:.3f}")
        print(f"   계절성 진폭: {(np.max(seasonal_component) - np.min(seasonal_component))/2:.1f}")
        
        return {
            'white_noise': white_noise,
            'random_walk': random_walk, 
            'trend_series': trend_series,
            'seasonal_series': seasonal_series
        }
    
    def explain_stationarity(self, data_dict):
        """정상성 개념 설명"""
        
        print("\n📊 정상성(Stationarity) 개념 이해")
        print("=" * 50)
        print("정상 시계열: 평균, 분산, 공분산이 시간에 무관하게 일정한 시계열")
        print("비정상 시계열: 시간에 따라 통계적 성질이 변하는 시계열")
        
        # 정상성 검정 함수
        def check_stationarity(series, name):
            print(f"\n🔬 {name} 정상성 검정:")
            
            # ADF 검정 (Augmented Dickey-Fuller Test)
            adf_result = adfuller(series, autolag='AIC')
            print(f"   ADF 통계량: {adf_result[0]:.4f}")
            print(f"   p-값: {adf_result[1]:.4f}")
            print(f"   임계값: {adf_result[4]['5%']:.4f}")
            
            if adf_result[1] <= 0.05:
                print(f"   ✅ 정상 시계열 (p < 0.05)")
            else:
                print(f"   ❌ 비정상 시계열 (p >= 0.05)")
            
            # KPSS 검정 (더 엄격한 검정)
            try:
                kpss_result = kpss(series, regression='c', nlags='auto')
                print(f"   KPSS 통계량: {kpss_result[0]:.4f}")
                print(f"   KPSS p-값: {kpss_result[1]:.4f}")
                
                if kpss_result[1] >= 0.05:
                    print(f"   ✅ KPSS: 정상 (p >= 0.05)")
                else:
                    print(f"   ❌ KPSS: 비정상 (p < 0.05)")
            except:
                print(f"   ⚠️ KPSS 검정 실패")
            
            return adf_result[1] <= 0.05
        
        # 각 시계열의 정상성 검정
        stationarity_results = {}
        for name, series in data_dict.items():
            stationarity_results[name] = check_stationarity(series, name)
        
        print(f"\n📋 정상성 검정 요약:")
        for name, is_stationary in stationarity_results.items():
            status = "정상" if is_stationary else "비정상"
            emoji = "✅" if is_stationary else "❌"
            print(f"   {emoji} {name}: {status}")
        
        # 차분을 통한 정상성 확보
        print(f"\n🔧 차분(Differencing)을 통한 정상성 확보:")
        
        non_stationary_series = data_dict['random_walk']
        differenced_series = np.diff(non_stationary_series)
        
        print(f"   원본 랜덤워크 정상성 검정:")
        check_stationarity(non_stationary_series, "원본")
        
        print(f"   1차 차분 후 정상성 검정:")
        check_stationarity(differenced_series, "1차 차분")
        
        # 차분 전후 시각화
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle('🔧 차분을 통한 정상성 확보', fontsize=14, fontweight='bold')
        
        axes[0].plot(non_stationary_series, color='red', alpha=0.8)
        axes[0].set_title('❌ 원본 (비정상)', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(differenced_series, color='blue', alpha=0.8)
        axes[1].set_title('✅ 1차 차분 (정상)', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return stationarity_results
    
    def demonstrate_ar_model(self, order=2):
        """자기회귀(AR) 모델 시연"""
        
        print(f"\n🔄 자기회귀 AR({order}) 모델")
        print("=" * 50)
        print("AR 모델: 현재 값이 과거 몇 개 값들의 선형결합으로 결정")
        print(f"수식: X(t) = c + φ₁X(t-1) + φ₂X(t-2) + ... + φₚX(t-p) + ε(t)")
        
        # AR(2) 모델 시뮬레이션
        np.random.seed(42)
        n = 200
        phi1, phi2 = 0.6, -0.3  # AR 계수
        c = 1.0  # 상수항
        
        # AR(2) 시계열 생성
        ar_series = np.zeros(n)
        errors = np.random.normal(0, 1, n)
        
        for t in range(2, n):
            ar_series[t] = c + phi1 * ar_series[t-1] + phi2 * ar_series[t-2] + errors[t]
        
        print(f"\n📊 생성된 AR({order}) 모델 특성:")
        print(f"   φ₁ (1차 계수): {phi1} - {'양수: 양의 자기상관' if phi1 > 0 else '음수: 음의 자기상관'}")
        print(f"   φ₂ (2차 계수): {phi2} - {'양수: 2차 양의 상관' if phi2 > 0 else '음수: 2차 음의 상관'}")
        print(f"   정상성 조건: |φ₁ + φ₂| < 1, |φ₂ - φ₁| < 1, |φ₂| < 1")
        
        # 정상성 조건 확인
        condition1 = abs(phi1 + phi2) < 1
        condition2 = abs(phi2 - phi1) < 1  
        condition3 = abs(phi2) < 1
        
        print(f"   조건 1: |{phi1} + {phi2}| = {abs(phi1 + phi2):.3f} < 1 ➜ {'✅' if condition1 else '❌'}")
        print(f"   조건 2: |{phi2} - {phi1}| = {abs(phi2 - phi1):.3f} < 1 ➜ {'✅' if condition2 else '❌'}")
        print(f"   조건 3: |{phi2}| = {abs(phi2):.3f} < 1 ➜ {'✅' if condition3 else '❌'}")
        
        stationary = condition1 and condition2 and condition3
        print(f"   종합 정상성: {'✅ 정상' if stationary else '❌ 비정상'}")
        
        # ACF와 PACF 계산 및 시각화
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'🔄 AR({order}) 모델 분석', fontsize=16, fontweight='bold')
        
        # 시계열 플롯
        axes[0, 0].plot(ar_series, color='blue', linewidth=1.5)
        axes[0, 0].set_title(f'📈 AR({order}) 시계열', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 히스토그램
        axes[0, 1].hist(ar_series, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].set_title('📊 분포 (정규성 확인)', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # ACF (자기상관함수)
        plot_acf(ar_series, lags=20, ax=axes[1, 0], title='ACF: 서서히 감소하는 패턴')
        
        # PACF (편자기상관함수)  
        plot_pacf(ar_series, lags=20, ax=axes[1, 1], title=f'PACF: {order}차에서 절단')
        
        plt.tight_layout()
        plt.show()
        
        # AR 모델 피팅
        ar_model = ARIMA(ar_series, order=(order, 0, 0))
        ar_fitted = ar_model.fit()
        
        print(f"\n🎯 AR({order}) 모델 추정 결과:")
        print(f"   추정된 φ₁: {ar_fitted.params[1]:.4f} (실제: {phi1})")
        if order >= 2:
            print(f"   추정된 φ₂: {ar_fitted.params[2]:.4f} (실제: {phi2})")
        print(f"   AIC: {ar_fitted.aic:.2f}")
        print(f"   BIC: {ar_fitted.bic:.2f}")
        
        return ar_series, ar_fitted
    
    def demonstrate_ma_model(self, order=2):
        """이동평균(MA) 모델 시연"""
        
        print(f"\n📊 이동평균 MA({order}) 모델")
        print("=" * 50)
        print("MA 모델: 현재 값이 과거 오차항들의 선형결합으로 결정")
        print(f"수식: X(t) = μ + ε(t) + θ₁ε(t-1) + θ₂ε(t-2) + ... + θᵩε(t-q)")
        
        # MA(2) 모델 시뮬레이션
        np.random.seed(42)
        n = 200
        theta1, theta2 = 0.4, 0.3  # MA 계수
        mu = 0.0  # 평균
        
        # MA(2) 시계열 생성
        errors = np.random.normal(0, 1, n)
        ma_series = np.zeros(n)
        
        for t in range(n):
            ma_series[t] = mu + errors[t]
            if t >= 1:
                ma_series[t] += theta1 * errors[t-1]
            if t >= 2:
                ma_series[t] += theta2 * errors[t-2]
        
        print(f"\n📊 생성된 MA({order}) 모델 특성:")
        print(f"   θ₁ (1차 계수): {theta1} - 1기간 전 오차의 영향력")
        print(f"   θ₂ (2차 계수): {theta2} - 2기간 전 오차의 영향력")
        print(f"   가역성 조건: MA 계수들이 적절한 범위 내에 있어야 함")
        
        # MA 모델은 항상 정상
        print(f"   정상성: ✅ MA 모델은 항상 정상 (오차항이 정상이면)")
        
        # ACF와 PACF 계산 및 시각화
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'📊 MA({order}) 모델 분석', fontsize=16, fontweight='bold')
        
        # 시계열 플롯
        axes[0, 0].plot(ma_series, color='green', linewidth=1.5)
        axes[0, 0].set_title(f'📈 MA({order}) 시계열', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 히스토그램
        axes[0, 1].hist(ma_series, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('📊 분포 (정규성 확인)', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # ACF (자기상관함수) - MA(q)에서는 q차에서 절단
        plot_acf(ma_series, lags=20, ax=axes[1, 0], title=f'ACF: {order}차에서 절단')
        
        # PACF (편자기상관함수) - 서서히 감소
        plot_pacf(ma_series, lags=20, ax=axes[1, 1], title='PACF: 서서히 감소하는 패턴')
        
        plt.tight_layout()
        plt.show()
        
        # MA 모델 피팅
        ma_model = ARIMA(ma_series, order=(0, 0, order))
        ma_fitted = ma_model.fit()
        
        print(f"\n🎯 MA({order}) 모델 추정 결과:")
        print(f"   추정된 θ₁: {ma_fitted.params[1]:.4f} (실제: {theta1})")
        if order >= 2:
            print(f"   추정된 θ₂: {ma_fitted.params[2]:.4f} (실제: {theta2})")
        print(f"   AIC: {ma_fitted.aic:.2f}")
        print(f"   BIC: {ma_fitted.bic:.2f}")
        
        return ma_series, ma_fitted
    
    def demonstrate_arma_model(self, ar_order=1, ma_order=1):
        """ARMA 모델 시연"""
        
        print(f"\n🔄📊 ARMA({ar_order},{ma_order}) 모델")
        print("=" * 50)
        print("ARMA 모델: AR과 MA의 결합 - 과거 값과 과거 오차의 영향을 모두 고려")
        print(f"수식: X(t) = c + φ₁X(t-1) + ... + φₚX(t-p) + ε(t) + θ₁ε(t-1) + ... + θᵩε(t-q)")
        
        # ARMA(1,1) 모델 시뮬레이션
        np.random.seed(42)
        n = 200
        phi1 = 0.6    # AR 계수
        theta1 = 0.3  # MA 계수
        c = 1.0       # 상수항
        
        # ARMA(1,1) 시계열 생성
        errors = np.random.normal(0, 1, n)
        arma_series = np.zeros(n)
        
        for t in range(1, n):
            arma_series[t] = c + phi1 * arma_series[t-1] + errors[t] + theta1 * errors[t-1]
        
        print(f"\n📊 생성된 ARMA({ar_order},{ma_order}) 모델 특성:")
        print(f"   AR 부분 φ₁: {phi1} - 이전 값의 영향")
        print(f"   MA 부분 θ₁: {theta1} - 이전 오차의 영향")
        print(f"   정상성: AR 부분이 정상성 조건을 만족해야 함")
        print(f"   가역성: MA 부분이 가역성 조건을 만족해야 함")
        
        # ACF와 PACF 분석
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'🔄📊 ARMA({ar_order},{ma_order}) 모델 분석', fontsize=16, fontweight='bold')
        
        # 시계열 플롯
        axes[0, 0].plot(arma_series, color='purple', linewidth=1.5)
        axes[0, 0].set_title(f'📈 ARMA({ar_order},{ma_order}) 시계열', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 히스토그램
        axes[0, 1].hist(arma_series, bins=30, alpha=0.7, color='plum', edgecolor='black')
        axes[0, 1].set_title('📊 분포 (정규성 확인)', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # ACF - 서서히 감소 (AR의 특성)
        plot_acf(arma_series, lags=20, ax=axes[1, 0], title='ACF: 서서히 감소 (AR 영향)')
        
        # PACF - 서서히 감소 (MA의 특성)
        plot_pacf(arma_series, lags=20, ax=axes[1, 1], title='PACF: 서서히 감소 (MA 영향)')
        
        plt.tight_layout()
        plt.show()
        
        # ARMA 모델 피팅
        arma_model = ARIMA(arma_series, order=(ar_order, 0, ma_order))
        arma_fitted = arma_model.fit()
        
        print(f"\n🎯 ARMA({ar_order},{ma_order}) 모델 추정 결과:")
        print(f"   추정된 φ₁: {arma_fitted.params[1]:.4f} (실제: {phi1})")
        print(f"   추정된 θ₁: {arma_fitted.params[2]:.4f} (실제: {theta1})")
        print(f"   AIC: {arma_fitted.aic:.2f}")
        print(f"   BIC: {arma_fitted.bic:.2f}")
        
        return arma_series, arma_fitted
    
    def compare_acf_pacf_patterns(self):
        """ACF와 PACF 패턴 비교"""
        
        print(f"\n🔍 ACF와 PACF 패턴으로 모델 식별하기")
        print("=" * 60)
        
        # 패턴 비교표
        pattern_table = {
            'AR(p)': {
                'ACF': '서서히 감소 (기하급수적 또는 감쇠진동)',
                'PACF': f'p차에서 절단 (p차 이후 0)',
                '특징': '과거 값의 직접적 영향'
            },
            'MA(q)': {
                'ACF': f'q차에서 절단 (q차 이후 0)',
                'PACF': '서서히 감소 (기하급수적 또는 감쇠진동)',
                '특징': '과거 오차의 영향'
            },
            'ARMA(p,q)': {
                'ACF': '서서히 감소 (MA 특성)',
                'PACF': '서서히 감소 (AR 특성)',
                '특징': 'AR과 MA의 복합 효과'
            }
        }
        
        print("📋 모델별 ACF/PACF 패턴 식별 가이드:")
        print("-" * 60)
        for model, patterns in pattern_table.items():
            print(f"🔹 {model}:")
            print(f"   ACF:  {patterns['ACF']}")
            print(f"   PACF: {patterns['PACF']}")
            print(f"   특징: {patterns['특징']}")
            print()
        
        # 실제 패턴 비교 시각화
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('🔍 모델별 ACF/PACF 패턴 비교', fontsize=16, fontweight='bold')
        
        # 각 모델의 이론적 ACF/PACF 시뮬레이션
        models_to_compare = [
            ('AR(2)', [0.6, -0.3], [0], 'blue'),
            ('MA(2)', [], [0.4, 0.3], 'green'), 
            ('ARMA(1,1)', [0.6], [0.3], 'purple')
        ]
        
        np.random.seed(42)
        
        for i, (model_name, ar_params, ma_params, color) in enumerate(models_to_compare):
            # 모델 생성
            if ar_params and ma_params:  # ARMA
                order = (len(ar_params), 0, len(ma_params))
            elif ar_params:  # AR
                order = (len(ar_params), 0, 0)
            else:  # MA
                order = (0, 0, len(ma_params))
            
            # 시뮬레이션된 시계열 생성
            temp_series = np.random.normal(0, 1, 200)
            if model_name == 'AR(2)':
                temp_series = self.demonstrate_ar_model(2)[0]
            elif model_name == 'MA(2)':
                temp_series = self.demonstrate_ma_model(2)[0]
            else:  # ARMA(1,1)
                temp_series = self.demonstrate_arma_model(1, 1)[0]
            
            # ACF 플롯
            plot_acf(temp_series, lags=15, ax=axes[i, 0], 
                    title=f'{model_name} - ACF', color=color)
            
            # PACF 플롯
            plot_pacf(temp_series, lags=15, ax=axes[i, 1], 
                     title=f'{model_name} - PACF', color=color)
        
        plt.tight_layout()
        plt.show()
        
        print("💡 실무 식별 팁:")
        print("   1. ACF가 q차에서 절단 → MA(q) 고려")
        print("   2. PACF가 p차에서 절단 → AR(p) 고려") 
        print("   3. 둘 다 서서히 감소 → ARMA(p,q) 고려")
        print("   4. 불규칙한 패턴 → 차분 후 재분석 필요")
        print("   5. 정보기준(AIC/BIC)으로 최종 모델 선택")
    
    def _create_ar_interpretation_prompt(self):
        """AR 모델 해석용 프롬프트"""
        return """
당신은 시계열 분석 전문가입니다. AR 모델 결과를 해석해주세요.

**분석 요청사항**:
1. AR 계수들의 실무적 의미
2. 정상성과 예측 안정성 평가
3. 모델의 강점과 한계점
4. 비즈니스 응용 방안

**데이터**: {model_summary}
**계수**: {coefficients}
**진단 통계**: {diagnostics}

실무진이 이해할 수 있도록 직관적으로 설명해주세요.
        """
    
    def _create_ma_interpretation_prompt(self):
        """MA 모델 해석용 프롬프트"""
        return """
당신은 시계열 분석 전문가입니다. MA 모델 결과를 해석해주세요.

**분석 요청사항**:
1. MA 계수들의 충격 전파 효과
2. 오차 구조와 예측 정확도
3. 단기 vs 장기 예측 성능
4. 모델 적합성 평가

**데이터**: {model_summary}
**계수**: {coefficients}
**오차 분석**: {error_analysis}

비즈니스 의사결정에 도움이 되는 관점에서 설명해주세요.
        """

# 전통적 시계열 모델 시스템 초기화
traditional_models = TraditionalTimeSeriesModels()

print("📚 전통적 시계열 모델 학습 여정 시작!")
print("=" * 60)
print("🎯 목표: AR, MA, ARIMA의 원리를 완전히 이해하고 실무에 적용")
print("🤖 방법: 7장 AI 협업 기법으로 복잡한 수학을 직관적으로 학습")

# 1. 기본 개념 시연
basic_patterns = traditional_models.demonstrate_basic_concepts()

# 2. 정상성 개념 이해
stationarity_results = traditional_models.explain_stationarity(basic_patterns)

# 3. AR 모델 상세 분석
print(f"\n" + "="*60)
ar_series, ar_model = traditional_models.demonstrate_ar_model(order=2)

# 4. MA 모델 상세 분석  
print(f"\n" + "="*60)
ma_series, ma_model = traditional_models.demonstrate_ma_model(order=2)

# 5. ARMA 모델 결합 분석
print(f"\n" + "="*60)
arma_series, arma_model = traditional_models.demonstrate_arma_model(ar_order=1, ma_order=1)

# 6. ACF/PACF 패턴 비교
print(f"\n" + "="*60)
traditional_models.compare_acf_pacf_patterns()

print(f"\n✅ 1부 완료: 자기회귀 및 이동평균 모델의 기초")
print(f"   🔄 AR 모델: 과거 값의 선형결합으로 현재 예측")
print(f"   📊 MA 모델: 과거 오차의 선형결합으로 현재 예측")
print(f"   🔄📊 ARMA 모델: AR과 MA의 장점을 결합")
print(f"   🔍 ACF/PACF: 모델 식별의 핵심 도구")
print(f"\n🚀 다음: ARIMA 모델 구축 및 진단")

## 2. ARIMA 모델 구축 및 진단 (Box-Jenkins 방법론)

### 2.1 Box-Jenkins 방법론의 체계적 접근

ARIMA 모델링의 황금률인 **Box-Jenkins 방법론**은 1970년부터 50년 넘게 사용되는 검증된 프로세스입니다. 이 방법론은 **식별(Identification) → 추정(Estimation) → 진단(Diagnostics)**의 반복적 과정을 통해 최적의 모델을 찾아갑니다.

```python
from itertools import product
from sklearn.metrics import mean_squared_error, mean_absolute_error
import itertools

class ARIMAModelBuilder:
    """Box-Jenkins 방법론 기반 ARIMA 모델 구축 클래스"""
    
    def __init__(self):
        self.model_results = {}
        self.best_model = None
        self.diagnostics_results = {}
        
        # 7장 AI 협업 원칙 적용
        self.box_jenkins_prompts = {
            'identification': self._create_identification_prompt(),
            'estimation': self._create_estimation_prompt(), 
            'diagnostics': self._create_diagnostics_prompt(),
            'selection': self._create_selection_prompt()
        }
    
    def box_jenkins_methodology(self, data, max_p=3, max_d=2, max_q=3):
        """Box-Jenkins 방법론 전체 프로세스"""
        
        print("📋 Box-Jenkins 방법론 시작")
        print("=" * 50)
        print("1단계: 식별 (Identification) - 모델 차수 결정")
        print("2단계: 추정 (Estimation) - 모델 파라미터 추정")  
        print("3단계: 진단 (Diagnostics) - 모델 적합성 검증")
        print("4단계: 예측 (Forecasting) - 미래 값 예측")
        
        # 1단계: 식별 (Identification)
        print(f"\n🔍 1단계: 모델 식별")
        identification_results = self._identification_stage(data, max_p, max_d, max_q)
        
        # 2단계: 추정 (Estimation)
        print(f"\n📊 2단계: 모델 추정")
        estimation_results = self._estimation_stage(data, identification_results)
        
        # 3단계: 진단 (Diagnostics)
        print(f"\n🔬 3단계: 모델 진단")
        diagnostics_results = self._diagnostics_stage(estimation_results)
        
        # 4단계: 모델 선택
        print(f"\n🎯 4단계: 최종 모델 선택")
        final_model = self._selection_stage(diagnostics_results)
        
        return final_model
    
    def _identification_stage(self, data, max_p, max_d, max_q):
        """1단계: 식별 - 적절한 차수 결정"""
        
        print("🔍 정상성 검정 및 차분 차수 결정")
        
        # 차분 차수 자동 결정
        d_optimal = self._determine_differencing_order(data, max_d)
        
        # 차분된 시계열
        differenced_data = data.copy()
        for i in range(d_optimal):
            differenced_data = differenced_data.diff().dropna()
        
        print(f"   최적 차분 차수 (d): {d_optimal}")
        print(f"   차분 후 데이터 길이: {len(differenced_data)}")
        
        # ACF/PACF 분석을 통한 초기 p, q 추정
        initial_orders = self._analyze_acf_pacf_for_orders(differenced_data, max_p, max_q)
        
        print(f"   ACF/PACF 분석 기반 후보 모델들:")
        for order in initial_orders[:5]:  # 상위 5개만 표시
            print(f"     ARIMA{order}")
        
        return {
            'd_optimal': d_optimal,
            'differenced_data': differenced_data,
            'candidate_orders': initial_orders,
            'original_data': data
        }
    
    def _determine_differencing_order(self, data, max_d=2):
        """차분 차수 자동 결정"""
        
        current_data = data.copy()
        
        for d in range(max_d + 1):
            # ADF 검정
            adf_stat, adf_pvalue, _, _, adf_critical_values, _ = adfuller(current_data, autolag='AIC')
            
            print(f"   차분 {d}회 후 ADF 검정:")
            print(f"     통계량: {adf_stat:.4f}, p-값: {adf_pvalue:.4f}")
            
            # 정상성 확인 (p < 0.05면 정상)
            if adf_pvalue < 0.05:
                print(f"     ✅ 정상성 확보 (p = {adf_pvalue:.4f} < 0.05)")
                return d
            else:
                print(f"     ❌ 비정상 (p = {adf_pvalue:.4f} >= 0.05)")
                if d < max_d:
                    current_data = current_data.diff().dropna()
        
        print(f"   ⚠️ {max_d}차 차분까지도 정상성 확보 실패, {max_d} 사용")
        return max_d
    
    def _analyze_acf_pacf_for_orders(self, data, max_p, max_q):
        """ACF/PACF 분석을 통한 차수 추정"""
        
        from statsmodels.tsa.stattools import acf, pacf
        
        # ACF와 PACF 계산
        acf_values = acf(data, nlags=max(max_p, max_q), fft=False)
        pacf_values = pacf(data, nlags=max_p)
        
        # 유의성 임계값 (대략 95% 신뢰구간)
        n = len(data)
        significance_level = 1.96 / np.sqrt(n)
        
        # PACF에서 유의한 차수 찾기 (AR 차수)
        significant_p = []
        for i in range(1, min(len(pacf_values), max_p + 1)):
            if abs(pacf_values[i]) > significance_level:
                significant_p.append(i)
        
        # ACF에서 유의한 차수 찾기 (MA 차수)
        significant_q = []
        for i in range(1, min(len(acf_values), max_q + 1)):
            if abs(acf_values[i]) > significance_level:
                significant_q.append(i)
        
        print(f"   PACF 기반 유의한 AR 차수: {significant_p[:3]}")  # 최대 3개
        print(f"   ACF 기반 유의한 MA 차수: {significant_q[:3]}")   # 최대 3개
        
        # 후보 조합 생성
        p_candidates = significant_p[:3] if significant_p else [0, 1, 2]
        q_candidates = significant_q[:3] if significant_q else [0, 1, 2]
        
        candidate_orders = []
        for p in p_candidates:
            for q in q_candidates:
                candidate_orders.append((p, 0, q))  # d는 별도로 결정됨
        
        return candidate_orders
    
    def _estimation_stage(self, data, identification_results):
        """2단계: 추정 - 모델 파라미터 추정"""
        
        d = identification_results['d_optimal']
        candidate_orders = identification_results['candidate_orders']
        original_data = identification_results['original_data']
        
        print(f"📊 {len(candidate_orders)}개 후보 모델 추정 중...")
        
        estimated_models = {}
        
        for i, (p, _, q) in enumerate(candidate_orders):
            order = (p, d, q)
            
            try:
                print(f"   모델 {i+1}: ARIMA{order} 추정 중...")
                
                # 모델 적합
                model = ARIMA(original_data, order=order)
                fitted_model = model.fit()
                
                # 기본 정보 저장
                estimated_models[order] = {
                    'model': fitted_model,
                    'aic': fitted_model.aic,
                    'bic': fitted_model.bic,
                    'llf': fitted_model.llf,
                    'params': fitted_model.params,
                    'order': order,
                    'fitted_values': fitted_model.fittedvalues,
                    'residuals': fitted_model.resid
                }
                
                print(f"     ✅ 성공 - AIC: {fitted_model.aic:.2f}, BIC: {fitted_model.bic:.2f}")
                
            except Exception as e:
                print(f"     ❌ 실패 - {str(e)[:50]}...")
                continue
        
        print(f"\n📊 추정 완료: {len(estimated_models)}개 모델 성공")
        
        # 결과를 AIC 순으로 정렬
        sorted_models = dict(sorted(estimated_models.items(), 
                                  key=lambda x: x[1]['aic']))
        
        print(f"\n🏆 정보기준 상위 5개 모델:")
        for i, (order, result) in enumerate(list(sorted_models.items())[:5]):
            print(f"   {i+1}. ARIMA{order}: AIC={result['aic']:.2f}, BIC={result['bic']:.2f}")
        
        return sorted_models
    
    def _diagnostics_stage(self, estimation_results):
        """3단계: 진단 - 모델 적합성 검증"""
        
        print(f"🔬 모델 진단 실시 중...")
        
        diagnostics_results = {}
        
        for order, model_result in estimation_results.items():
            print(f"\n   ARIMA{order} 진단:")
            
            fitted_model = model_result['model']
            residuals = model_result['residuals']
            
            # 진단 테스트들
            diagnostic_tests = self._comprehensive_diagnostics(fitted_model, residuals)
            
            # 종합 점수 계산
            overall_score = self._calculate_overall_score(model_result, diagnostic_tests)
            
            diagnostics_results[order] = {
                **model_result,
                'diagnostics': diagnostic_tests,
                'overall_score': overall_score
            }
            
            print(f"     종합 점수: {overall_score:.2f}/100")
        
        return diagnostics_results
    
    def _comprehensive_diagnostics(self, fitted_model, residuals):
        """종합적인 진단 테스트"""
        
        diagnostics = {}
        
        # 1. Ljung-Box 검정 (잔차의 자기상관 검정)
        try:
            ljung_box_result = acorr_ljungbox(residuals, lags=10, return_df=True)
            ljung_box_pvalue = ljung_box_result['lb_pvalue'].iloc[-1]  # 10차 지연의 p-값
            
            diagnostics['ljung_box'] = {
                'statistic': ljung_box_result['lb_stat'].iloc[-1],
                'pvalue': ljung_box_pvalue,
                'passed': ljung_box_pvalue > 0.05,
                'interpretation': 'Good' if ljung_box_pvalue > 0.05 else 'Poor'
            }
            print(f"       Ljung-Box: p={ljung_box_pvalue:.4f} ({'✅' if ljung_box_pvalue > 0.05 else '❌'})")
            
        except Exception as e:
            diagnostics['ljung_box'] = {'error': str(e), 'passed': False}
            print(f"       Ljung-Box: 계산 실패")
        
        # 2. Jarque-Bera 정규성 검정
        try:
            jb_stat, jb_pvalue = stats.jarque_bera(residuals.dropna())
            
            diagnostics['jarque_bera'] = {
                'statistic': jb_stat,
                'pvalue': jb_pvalue,
                'passed': jb_pvalue > 0.05,
                'interpretation': 'Normal' if jb_pvalue > 0.05 else 'Non-normal'
            }
            print(f"       Jarque-Bera: p={jb_pvalue:.4f} ({'✅' if jb_pvalue > 0.05 else '❌'})")
            
        except Exception as e:
            diagnostics['jarque_bera'] = {'error': str(e), 'passed': False}
            print(f"       Jarque-Bera: 계산 실패")
        
        # 3. 잔차 기본 통계
        residuals_clean = residuals.dropna()
        diagnostics['residual_stats'] = {
            'mean': residuals_clean.mean(),
            'std': residuals_clean.std(),
            'skewness': stats.skew(residuals_clean),
            'kurtosis': stats.kurtosis(residuals_clean),
            'mean_near_zero': abs(residuals_clean.mean()) < 0.1 * residuals_clean.std()
        }
        
        print(f"       잔차 평균: {residuals_clean.mean():.4f}")
        print(f"       잔차 표준편차: {residuals_clean.std():.4f}")
        
        # 4. 예측 성능 (훈련 데이터 내)
        try:
            original_data = fitted_model.model.endog
            fitted_values = fitted_model.fittedvalues
            
            # 결측값 제거하고 길이 맞추기
            valid_idx = ~np.isnan(fitted_values)
            original_clean = original_data[valid_idx]
            fitted_clean = fitted_values[valid_idx]
            
            if len(original_clean) > 0 and len(fitted_clean) > 0:
                mse = mean_squared_error(original_clean, fitted_clean)
                mae = mean_absolute_error(original_clean, fitted_clean)
                
                diagnostics['prediction_performance'] = {
                    'mse': mse,
                    'mae': mae,
                    'rmse': np.sqrt(mse)
                }
                print(f"       RMSE: {np.sqrt(mse):.4f}")
            
        except Exception as e:
            diagnostics['prediction_performance'] = {'error': str(e)}
            print(f"       예측 성능: 계산 실패")
        
        return diagnostics
    
    def _calculate_overall_score(self, model_result, diagnostics):
        """종합 점수 계산 (0-100점)"""
        
        score = 0
        max_score = 100
        
        # 1. 정보기준 점수 (40점)
        aic_score = 20  # 기본점수
        bic_score = 20  # 기본점수
        
        # AIC/BIC가 낮을수록 좋으므로 역변환
        # 여기서는 상대적 비교를 위해 단순화
        score += aic_score + bic_score
        
        # 2. 진단 테스트 점수 (40점)
        if 'ljung_box' in diagnostics and diagnostics['ljung_box'].get('passed', False):
            score += 20
        
        if 'jarque_bera' in diagnostics and diagnostics['jarque_bera'].get('passed', False):
            score += 10
        
        if 'residual_stats' in diagnostics and diagnostics['residual_stats'].get('mean_near_zero', False):
            score += 10
        
        # 3. 파라미터 유의성 점수 (20점)
        try:
            pvalues = model_result['model'].pvalues
            significant_params = (pvalues < 0.05).sum()
            total_params = len(pvalues)
            
            if total_params > 0:
                param_score = (significant_params / total_params) * 20
                score += param_score
        except:
            pass
        
        return min(score, max_score)
    
    def _selection_stage(self, diagnostics_results):
        """4단계: 최종 모델 선택"""
        
        print(f"🎯 최종 모델 선택 기준:")
        print(f"   1. 진단 테스트 통과 여부 (가장 중요)")
        print(f"   2. 정보기준 (AIC/BIC) 최소화")
        print(f"   3. 파라미터 유의성")
        print(f"   4. 모델 복잡도 (단순함 선호)")
        
        # 진단을 통과한 모델들만 필터링
        valid_models = {}
        for order, result in diagnostics_results.items():
            diag = result['diagnostics']
            
            # 기본 조건: Ljung-Box 테스트 통과
            ljung_box_passed = diag.get('ljung_box', {}).get('passed', False)
            
            if ljung_box_passed:
                valid_models[order] = result
                print(f"   ✅ ARIMA{order}: 진단 통과")
            else:
                print(f"   ❌ ARIMA{order}: 진단 실패")
        
        if not valid_models:
            print(f"   ⚠️ 모든 모델이 진단 실패. AIC 기준으로 선택")
            valid_models = diagnostics_results
        
        # 종합 점수로 최종 선택
        best_order = max(valid_models.keys(), 
                        key=lambda x: valid_models[x]['overall_score'])
        
        best_model = valid_models[best_order]
        
        print(f"\n🏆 최종 선택 모델: ARIMA{best_order}")
        print(f"   종합 점수: {best_model['overall_score']:.2f}/100")
        print(f"   AIC: {best_model['aic']:.2f}")
        print(f"   BIC: {best_model['bic']:.2f}")
        
        # 상세 진단 결과 출력
        self._print_detailed_diagnostics(best_order, best_model)
        
        return best_model
    
    def _print_detailed_diagnostics(self, order, model_result):
        """상세 진단 결과 출력"""
        
        print(f"\n📋 ARIMA{order} 상세 진단 결과:")
        print("-" * 40)
        
        fitted_model = model_result['model']
        diagnostics = model_result['diagnostics']
        
        # 1. 모델 요약
        print(f"📊 모델 정보:")
        print(f"   차수: {order}")
        print(f"   로그우도: {fitted_model.llf:.4f}")
        print(f"   AIC: {fitted_model.aic:.4f}")
        print(f"   BIC: {fitted_model.bic:.4f}")
        
        # 2. 파라미터 추정치
        print(f"\n🔢 파라미터 추정치:")
        for param_name, param_value in fitted_model.params.items():
            pvalue = fitted_model.pvalues[param_name]
            significance = "***" if pvalue < 0.001 else "**" if pvalue < 0.01 else "*" if pvalue < 0.05 else ""
            print(f"   {param_name}: {param_value:.4f} (p={pvalue:.4f}) {significance}")
        
        # 3. 진단 테스트 결과
        print(f"\n🔬 진단 테스트:")
        
        ljung_box = diagnostics.get('ljung_box', {})
        if 'pvalue' in ljung_box:
            status = "통과" if ljung_box['passed'] else "실패"
            print(f"   Ljung-Box (자기상관): p={ljung_box['pvalue']:.4f} ({status})")
        
        jarque_bera = diagnostics.get('jarque_bera', {})
        if 'pvalue' in jarque_bera:
            status = "정규분포" if jarque_bera['passed'] else "비정규분포"
            print(f"   Jarque-Bera (정규성): p={jarque_bera['pvalue']:.4f} ({status})")
        
        residual_stats = diagnostics.get('residual_stats', {})
        if residual_stats:
            print(f"   잔차 평균: {residual_stats['mean']:.6f}")
            print(f"   잔차 표준편차: {residual_stats['std']:.4f}")
        
        # 4. 예측 성능
        pred_perf = diagnostics.get('prediction_performance', {})
        if 'rmse' in pred_perf:
            print(f"\n📈 예측 성능 (훈련 데이터):")
            print(f"   RMSE: {pred_perf['rmse']:.4f}")
            print(f"   MAE: {pred_perf['mae']:.4f}")
    
    def visualize_model_diagnostics(self, model_result):
        """모델 진단 시각화"""
        
        fitted_model = model_result['model']
        residuals = fitted_model.resid
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'🔬 ARIMA{model_result["order"]} 모델 진단', fontsize=16, fontweight='bold')
        
        # 1. 잔차 플롯
        axes[0, 0].plot(residuals, color='blue', alpha=0.7)
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.8)
        axes[0, 0].set_title('📊 잔차 시계열', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 잔차 히스토그램
        axes[0, 1].hist(residuals.dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].set_title('📈 잔차 분포', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 정규분포 곡선 추가
        residuals_clean = residuals.dropna()
        if len(residuals_clean) > 0:
            mu, sigma = residuals_clean.mean(), residuals_clean.std()
            x = np.linspace(residuals_clean.min(), residuals_clean.max(), 100)
            y = stats.norm.pdf(x, mu, sigma)
            axes[0, 1].plot(x, y * len(residuals_clean) * (x[1] - x[0]), 'r-', linewidth=2, label='정규분포')
            axes[0, 1].legend()
        
        # 3. Q-Q 플롯
        stats.probplot(residuals.dropna(), dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('📋 Q-Q Plot (정규성 검정)', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 잔차 ACF
        plot_acf(residuals.dropna(), lags=20, ax=axes[1, 1], title='🔍 잔차 ACF (백색잡음 확인)')
        
        plt.tight_layout()
        plt.show()
    
    def _create_identification_prompt(self):
        """식별 단계용 AI 프롬프트"""
        return """
시계열 모델 식별 전문가로서 ACF/PACF 패턴을 분석해주세요.

**분석 요청사항**:
1. ACF/PACF 패턴에 기반한 모델 차수 추천
2. 정상성 검정 결과 해석
3. 차분의 필요성과 적정 차수
4. 초기 모델 후보들의 우선순위

**데이터**: {data_summary}
**ACF 패턴**: {acf_pattern}
**PACF 패턴**: {pacf_pattern}
**정상성 검정**: {stationarity_tests}

Box-Jenkins 관점에서 체계적으로 분석해주세요.
        """

# ARIMA 모델 구축 시스템 실행
arima_builder = ARIMAModelBuilder()

print("\n📋 Box-Jenkins 방법론으로 ARIMA 모델 구축")
print("=" * 60)

# 실제 데이터로 ARIMA 모델링 시연
# 8장 Part 1에서 생성한 매출 데이터 사용
if 'sample_data' in globals():
    sales_data = sample_data['sales'].dropna()
    
    # 월별 데이터로 집계 (노이즈 감소)
    monthly_sales = sales_data.resample('M').mean()
    
    print(f"🎯 분석 대상 데이터:")
    print(f"   시계열: 월별 매출 데이터")
    print(f"   기간: {monthly_sales.index.min().date()} ~ {monthly_sales.index.max().date()}")
    print(f"   관측값: {len(monthly_sales)}개")
    print(f"   평균: {monthly_sales.mean():.2f}")
    print(f"   표준편차: {monthly_sales.std():.2f}")
    
    # Box-Jenkins 방법론 적용
    final_arima_model = arima_builder.box_jenkins_methodology(
        monthly_sales, 
        max_p=3, 
        max_d=2, 
        max_q=3
    )
    
    # 진단 시각화
    print(f"\n🔬 모델 진단 시각화:")
    arima_builder.visualize_model_diagnostics(final_arima_model)
    
else:
    print("⚠️ 샘플 데이터가 없습니다. 시뮬레이션 데이터로 진행합니다.")
    
    # 시뮬레이션 데이터 생성
    np.random.seed(42)
    dates = pd.date_range('2021-01-01', '2023-12-31', freq='M')
    trend = np.linspace(100, 150, len(dates))
    seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
    noise = np.random.normal(0, 5, len(dates))
    simulated_sales = trend + seasonal + noise
    
    sales_ts = pd.Series(simulated_sales, index=dates, name='sales')
    
    print(f"🎯 시뮬레이션 데이터로 ARIMA 모델링:")
    print(f"   기간: {sales_ts.index.min().date()} ~ {sales_ts.index.max().date()}")
    print(f"   관측값: {len(sales_ts)}개")
    
    # Box-Jenkins 방법론 적용
    final_arima_model = arima_builder.box_jenkins_methodology(
        sales_ts,
        max_p=3,
        max_d=2, 
        max_q=3
    )
    
    # 진단 시각화
    print(f"\n🔬 모델 진단 시각화:")
    arima_builder.visualize_model_diagnostics(final_arima_model)

print(f"\n✅ ARIMA 모델 구축 완료!")
print(f"   📋 Box-Jenkins 4단계 방법론 완전 적용")
print(f"   🔍 체계적인 모델 식별 및 진단")
print(f"   📊 정보기준과 진단 테스트 종합 평가") 
print(f"   🎯 최적 모델 자동 선택 및 검증")
print(f"\n🚀 다음: 계절성 SARIMA 모델과 지수평활법")

## 3. 계절성 모델 (SARIMA)와 지수평활법

### 3.1 SARIMA 모델 - 계절성까지 고려한 고급 모델링

대부분의 실제 비즈니스 데이터는 **계절성(Seasonality)**을 가지고 있습니다. 매출의 연말 증가, 전력 소비의 여름/겨울 패턴, 관광객 수의 계절별 변동 등이 그 예입니다. SARIMA 모델은 이러한 계절성을 체계적으로 모델링하는 강력한 도구입니다.

```python
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.exponential_smoothing.exponential_smoothing import ExponentialSmoothing
from sklearn.metrics import mean_absolute_percentage_error

class SeasonalTimeSeriesModels:
    """계절성 시계열 모델 클래스"""
    
    def __init__(self):
        self.models = {}
        self.forecasts = {}
        self.performance_metrics = {}
        
        # 7장 AI 협업 원칙을 계절성 분석에 적용
        self.seasonal_analysis_prompts = {
            'decomposition': self._create_decomposition_prompt(),
            'sarima_interpretation': self._create_sarima_prompt(),
            'exponential_smoothing': self._create_exponential_smoothing_prompt(),
            'model_comparison': self._create_comparison_prompt()
        }
    
    def demonstrate_sarima_concept(self):
        """SARIMA 모델 개념 설명"""
        
        print("🔄 SARIMA 모델 이해하기")
        print("=" * 50)
        print("SARIMA(p,d,q)(P,D,Q)s = 계절성 ARIMA 모델")
        print()
        print("📊 일반 부분 (소문자):")
        print("   p: 비계절 자기회귀 차수")
        print("   d: 비계절 차분 차수") 
        print("   q: 비계절 이동평균 차수")
        print()
        print("🔄 계절 부분 (대문자):")
        print("   P: 계절 자기회귀 차수")
        print("   D: 계절 차분 차수")
        print("   Q: 계절 이동평균 차수")
        print("   s: 계절 주기 (월별=12, 분기별=4, 주별=52)")
        
        # 계절성 있는 시뮬레이션 데이터 생성
        np.random.seed(42)
        periods = 60  # 5년치 월별 데이터
        dates = pd.date_range('2019-01-01', periods=periods, freq='M')
        
        # 트렌드 성분
        trend = np.linspace(100, 140, periods)
        
        # 계절성 성분 (연간 주기)
        seasonal = 15 * np.sin(2 * np.pi * np.arange(periods) / 12)
        
        # 계절성 AR 성분 (이전 해 같은 달의 영향)
        seasonal_ar = np.zeros(periods)
        for t in range(12, periods):
            seasonal_ar[t] = 0.7 * seasonal_ar[t-12] + np.random.normal(0, 2)
        
        # 노이즈
        noise = np.random.normal(0, 3, periods)
        
        # 최종 시계열
        seasonal_ts = trend + seasonal + seasonal_ar + noise
        seasonal_data = pd.Series(seasonal_ts, index=dates, name='seasonal_sales')
        
        print(f"\n📊 계절성 시계열 생성:")
        print(f"   기간: {seasonal_data.index.min().date()} ~ {seasonal_data.index.max().date()}")
        print(f"   관측값: {len(seasonal_data)}개")
        print(f"   계절 주기: 12개월")
        
        # 계절성 분해
        decomposition = seasonal_decompose(seasonal_data, model='additive', period=12)
        
        # 분해 결과 시각화
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        fig.suptitle('🔄 계절성 시계열 분해 분석', fontsize=16, fontweight='bold')
        
        # 원본 데이터
        axes[0].plot(seasonal_data.index, seasonal_data.values, color='black', linewidth=2)
        axes[0].set_title('📊 원본 데이터 (트렌드 + 계절성 + 노이즈)', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # 트렌드
        axes[1].plot(decomposition.trend.index, decomposition.trend.values, color='blue', linewidth=2)
        axes[1].set_title('📈 트렌드 성분', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # 계절성
        axes[2].plot(decomposition.seasonal.index, decomposition.seasonal.values, color='green', linewidth=2)
        axes[2].set_title('🔄 계절성 성분 (연간 반복)', fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        # 잔차
        axes[3].plot(decomposition.resid.index, decomposition.resid.values, color='red', linewidth=1)
        axes[3].set_title('🎲 잔차 성분 (노이즈)', fontweight='bold')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 계절성 강도 분석
        seasonal_strength = decomposition.seasonal.std() / seasonal_data.std()
        print(f"\n🔍 계절성 분석:")
        print(f"   계절성 강도: {seasonal_strength:.1%}")
        print(f"   트렌드 방향: {'상승' if decomposition.trend.dropna().iloc[-1] > decomposition.trend.dropna().iloc[0] else '하락'}")
        print(f"   계절 피크: {decomposition.seasonal.groupby(decomposition.seasonal.index.month).mean().idxmax()}월")
        print(f"   계절 저점: {decomposition.seasonal.groupby(decomposition.seasonal.index.month).mean().idxmin()}월")
        
        return seasonal_data, decomposition
    
    def build_sarima_model(self, data, seasonal_period=12):
        """SARIMA 모델 구축"""
        
        print(f"\n🔄 SARIMA 모델 구축 (계절 주기: {seasonal_period})")
        print("=" * 50)
        
        # 자동 SARIMA 모델 선택을 위한 그리드 서치
        print("🔍 SARIMA 차수 자동 선택 중...")
        
        # 후보 차수들
        p_values = range(0, 3)  # 일반 AR 차수
        d_values = range(0, 2)  # 일반 차분 차수
        q_values = range(0, 3)  # 일반 MA 차수
        
        P_values = range(0, 2)  # 계절 AR 차수
        D_values = range(0, 2)  # 계절 차분 차수  
        Q_values = range(0, 2)  # 계절 MA 차수
        
        # 모든 조합 생성
        pdq_combinations = list(itertools.product(p_values, d_values, q_values))
        PDQ_combinations = list(itertools.product(P_values, D_values, Q_values))
        
        best_aic = float('inf')
        best_model = None
        best_order = None
        best_seasonal_order = None
        
        model_results = []
        
        print(f"   총 {len(pdq_combinations) * len(PDQ_combinations)}개 조합 테스트 중...")
        
        for pdq in pdq_combinations[:3]:  # 계산 시간 단축을 위해 제한
            for PDQ in PDQ_combinations[:3]:
                try:
                    # SARIMA 모델 생성
                    order = pdq
                    seasonal_order = PDQ + (seasonal_period,)
                    
                    model = SARIMAX(data, 
                                   order=order,
                                   seasonal_order=seasonal_order,
                                   enforce_stationarity=False,
                                   enforce_invertibility=False)
                    
                    fitted_model = model.fit(disp=False)
                    
                    # AIC 기준으로 최적 모델 선택
                    if fitted_model.aic < best_aic:
                        best_aic = fitted_model.aic
                        best_model = fitted_model
                        best_order = order
                        best_seasonal_order = seasonal_order
                    
                    model_results.append({
                        'order': order,
                        'seasonal_order': seasonal_order,
                        'aic': fitted_model.aic,
                        'bic': fitted_model.bic,
                        'model': fitted_model
                    })
                    
                except Exception as e:
                    continue
        
        # 결과 정렬 및 출력
        model_results.sort(key=lambda x: x['aic'])
        
        print(f"\n🏆 상위 5개 SARIMA 모델:")
        for i, result in enumerate(model_results[:5]):
            order = result['order']
            seasonal_order = result['seasonal_order']
            print(f"   {i+1}. SARIMA{order}x{seasonal_order}: AIC={result['aic']:.2f}")
        
        if best_model is not None:
            print(f"\n🎯 선택된 최적 모델:")
            print(f"   SARIMA{best_order}x{best_seasonal_order}")
            print(f"   AIC: {best_aic:.2f}")
            print(f"   BIC: {best_model.bic:.2f}")
            
            # 모델 요약
            print(f"\n📊 모델 요약:")
            print(f"   일반 부분: AR({best_order[0]}) I({best_order[1]}) MA({best_order[2]})")
            print(f"   계절 부분: AR({best_seasonal_order[0]}) I({best_seasonal_order[1]}) MA({best_seasonal_order[2]}) [{best_seasonal_order[3]}]")
            
            return best_model, model_results
        else:
            print("❌ SARIMA 모델 구축 실패")
            return None, []
    
    def demonstrate_exponential_smoothing(self, data):
        """지수평활법 모델들 시연"""
        
        print(f"\n📊 지수평활법 (Exponential Smoothing) 모델들")
        print("=" * 50)
        print("지수평활법: 최근 관측값에 더 높은 가중치를 부여하는 예측 방법")
        print("특징: 직관적이고 계산이 간단하며 실무에서 널리 사용")
        
        # 다양한 지수평활법 모델들
        smoothing_models = {}
        
        # 1. 단순 지수평활법 (Simple Exponential Smoothing)
        print(f"\n1️⃣ 단순 지수평활법 (SES)")
        print("   적용: 트렌드와 계절성이 없는 데이터")
        print("   수식: F(t+1) = αX(t) + (1-α)F(t)")
        
        try:
            ses_model = ExponentialSmoothing(data, trend=None, seasonal=None)
            ses_fitted = ses_model.fit()
            smoothing_models['SES'] = ses_fitted
            print(f"   ✅ α (평활계수): {ses_fitted.params['smoothing_level']:.4f}")
        except Exception as e:
            print(f"   ❌ 실패: {str(e)[:50]}...")
        
        # 2. Holt 선형 트렌드 (Double Exponential Smoothing)
        print(f"\n2️⃣ Holt 선형 트렌드")
        print("   적용: 트렌드가 있지만 계절성이 없는 데이터")
        print("   수식: Level + Trend 성분을 별도로 평활화")
        
        try:
            holt_model = ExponentialSmoothing(data, trend='add', seasonal=None)
            holt_fitted = holt_model.fit()
            smoothing_models['Holt'] = holt_fitted
            print(f"   ✅ α (level): {holt_fitted.params['smoothing_level']:.4f}")
            print(f"   ✅ β (trend): {holt_fitted.params['smoothing_trend']:.4f}")
        except Exception as e:
            print(f"   ❌ 실패: {str(e)[:50]}...")
        
        # 3. Holt-Winters 가법적 모델
        print(f"\n3️⃣ Holt-Winters 가법적 모델")
        print("   적용: 트렌드와 계절성이 모두 있는 데이터")
        print("   수식: Level + Trend + Seasonal (가법적 결합)")
        
        try:
            hw_add_model = ExponentialSmoothing(data, trend='add', seasonal='add', seasonal_periods=12)
            hw_add_fitted = hw_add_model.fit()
            smoothing_models['HW_Add'] = hw_add_fitted
            print(f"   ✅ α (level): {hw_add_fitted.params['smoothing_level']:.4f}")
            print(f"   ✅ β (trend): {hw_add_fitted.params['smoothing_trend']:.4f}")
            print(f"   ✅ γ (seasonal): {hw_add_fitted.params['smoothing_seasonal']:.4f}")
        except Exception as e:
            print(f"   ❌ 실패: {str(e)[:50]}...")
        
        # 4. Holt-Winters 승법적 모델
        print(f"\n4️⃣ Holt-Winters 승법적 모델")
        print("   적용: 계절성 패턴의 크기가 트렌드에 비례하는 데이터")
        print("   수식: Level × Trend × Seasonal (승법적 결합)")
        
        try:
            hw_mul_model = ExponentialSmoothing(data, trend='add', seasonal='mul', seasonal_periods=12)
            hw_mul_fitted = hw_mul_model.fit()
            smoothing_models['HW_Mul'] = hw_mul_fitted
            print(f"   ✅ α (level): {hw_mul_fitted.params['smoothing_level']:.4f}")
            print(f"   ✅ β (trend): {hw_mul_fitted.params['smoothing_trend']:.4f}")
            print(f"   ✅ γ (seasonal): {hw_mul_fitted.params['smoothing_seasonal']:.4f}")
        except Exception as e:
            print(f"   ❌ 실패: {str(e)[:50]}...")
        
        # 모델별 AIC 비교
        print(f"\n📊 지수평활법 모델 비교 (AIC 기준):")
        aic_comparison = {}
        for name, model in smoothing_models.items():
            try:
                aic = model.aic
                aic_comparison[name] = aic
                print(f"   {name}: AIC = {aic:.2f}")
            except:
                print(f"   {name}: AIC 계산 실패")
        
        # 최적 모델 선택
        if aic_comparison:
            best_smooth_model = min(aic_comparison.keys(), key=lambda x: aic_comparison[x])
            print(f"\n🏆 최적 지수평활법 모델: {best_smooth_model}")
            print(f"   AIC: {aic_comparison[best_smooth_model]:.2f}")
            
            return smoothing_models, best_smooth_model
        else:
            return smoothing_models, None
    
    def compare_forecasting_performance(self, data, train_ratio=0.8):
        """예측 성능 비교"""
        
        print(f"\n📈 예측 성능 비교 테스트")
        print("=" * 50)
        
        # 훈련/테스트 분할
        split_point = int(len(data) * train_ratio)
        train_data = data[:split_point]
        test_data = data[split_point:]
        
        print(f"📊 데이터 분할:")
        print(f"   훈련 데이터: {len(train_data)}개 ({train_data.index[0].date()} ~ {train_data.index[-1].date()})")
        print(f"   테스트 데이터: {len(test_data)}개 ({test_data.index[0].date()} ~ {test_data.index[-1].date()})")
        
        forecast_results = {}
        
        # 1. SARIMA 모델
        print(f"\n🔄 SARIMA 모델 예측:")
        try:
            sarima_model, _ = self.build_sarima_model(train_data, seasonal_period=12)
            if sarima_model is not None:
                sarima_forecast = sarima_model.forecast(steps=len(test_data))
                sarima_mape = mean_absolute_percentage_error(test_data, sarima_forecast)
                
                forecast_results['SARIMA'] = {
                    'forecast': sarima_forecast,
                    'mape': sarima_mape,
                    'model': sarima_model
                }
                print(f"   ✅ MAPE: {sarima_mape:.2%}")
            else:
                print(f"   ❌ SARIMA 모델 구축 실패")
        except Exception as e:
            print(f"   ❌ SARIMA 오류: {str(e)[:50]}...")
        
        # 2. Holt-Winters 모델
        print(f"\n📊 Holt-Winters 예측:")
        try:
            hw_model = ExponentialSmoothing(train_data, trend='add', seasonal='add', seasonal_periods=12)
            hw_fitted = hw_model.fit()
            hw_forecast = hw_fitted.forecast(steps=len(test_data))
            hw_mape = mean_absolute_percentage_error(test_data, hw_forecast)
            
            forecast_results['Holt-Winters'] = {
                'forecast': hw_forecast,
                'mape': hw_mape,
                'model': hw_fitted
            }
            print(f"   ✅ MAPE: {hw_mape:.2%}")
        except Exception as e:
            print(f"   ❌ Holt-Winters 오류: {str(e)[:50]}...")
        
        # 3. 단순 베이스라인 (계절적 naive)
        print(f"\n🎯 계절적 Naive 예측:")
        try:
            # 1년 전 값을 그대로 사용
            seasonal_naive_forecast = []
            for i in range(len(test_data)):
                # 12개월 전 값 사용 (월별 데이터 가정)
                if i < 12 and len(train_data) >= 12:
                    seasonal_naive_forecast.append(train_data.iloc[-(12-i)])
                elif len(train_data) >= 12:
                    seasonal_naive_forecast.append(train_data.iloc[-12])
                else:
                    seasonal_naive_forecast.append(train_data.iloc[-1])
            
            seasonal_naive_forecast = pd.Series(seasonal_naive_forecast, index=test_data.index)
            naive_mape = mean_absolute_percentage_error(test_data, seasonal_naive_forecast)
            
            forecast_results['Seasonal_Naive'] = {
                'forecast': seasonal_naive_forecast,
                'mape': naive_mape,
                'model': None
            }
            print(f"   ✅ MAPE: {naive_mape:.2%}")
        except Exception as e:
            print(f"   ❌ Naive 오류: {str(e)[:50]}...")
        
        # 성능 비교 시각화
        if forecast_results:
            self._visualize_forecast_comparison(train_data, test_data, forecast_results)
            self._print_performance_summary(forecast_results)
        
        return forecast_results
    
    def _visualize_forecast_comparison(self, train_data, test_data, forecast_results):
        """예측 결과 비교 시각화"""
        
        fig, ax = plt.subplots(1, 1, figsize=(15, 8))
        
        # 훈련 데이터
        ax.plot(train_data.index, train_data.values, 'o-', label='훈련 데이터', 
                color='black', linewidth=2, markersize=4)
        
        # 실제 테스트 데이터
        ax.plot(test_data.index, test_data.values, 'o-', label='실제 값', 
                color='blue', linewidth=2, markersize=4)
        
        # 각 모델의 예측
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        for i, (model_name, result) in enumerate(forecast_results.items()):
            ax.plot(test_data.index, result['forecast'], '--', 
                   label=f'{model_name} (MAPE: {result["mape"]:.1%})',
                   color=colors[i % len(colors)], linewidth=2)
        
        ax.set_title('📈 시계열 예측 모델 성능 비교', fontsize=14, fontweight='bold')
        ax.set_ylabel('값')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # 훈련/테스트 구분선
        split_line = train_data.index[-1]
        ax.axvline(x=split_line, color='gray', linestyle=':', alpha=0.7, label='훈련/테스트 분할')
        
        plt.tight_layout()
        plt.show()
    
    def _print_performance_summary(self, forecast_results):
        """성능 요약 출력"""
        
        print(f"\n🏆 예측 성능 요약 (MAPE 기준):")
        print("-" * 40)
        
        # MAPE 기준으로 정렬
        sorted_results = sorted(forecast_results.items(), key=lambda x: x[1]['mape'])
        
        for rank, (model_name, result) in enumerate(sorted_results, 1):
            mape = result['mape']
            emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "🏅"
            print(f"   {emoji} {rank}위: {model_name} - MAPE: {mape:.2%}")
        
        # 성능 해석
        best_model = sorted_results[0][0]
        best_mape = sorted_results[0][1]['mape']
        
        print(f"\n💡 성능 해석:")
        if best_mape <= 0.10:
            print(f"   ✅ 우수한 예측 성능 (MAPE ≤ 10%)")
        elif best_mape <= 0.20:
            print(f"   🔶 양호한 예측 성능 (10% < MAPE ≤ 20%)")
        elif best_mape <= 0.50:
            print(f"   ⚠️ 보통 예측 성능 (20% < MAPE ≤ 50%)")
        else:
            print(f"   ❌ 개선 필요 (MAPE > 50%)")
        
        print(f"   🎯 최적 모델: {best_model}")
        print(f"   📊 예측 정확도: {(1-best_mape):.1%}")

# 계절성 모델 시스템 실행
seasonal_models = SeasonalTimeSeriesModels()

print("\n🔄 계절성 시계열 모델링 여정 시작")
print("=" * 60)

# 1. SARIMA 개념 시연
seasonal_data, decomposition = seasonal_models.demonstrate_sarima_concept()

# 2. SARIMA 모델 구축
best_sarima, sarima_results = seasonal_models.build_sarima_model(seasonal_data, seasonal_period=12)

# 3. 지수평활법 모델들 시연
smoothing_models, best_smoothing = seasonal_models.demonstrate_exponential_smoothing(seasonal_data)

# 4. 예측 성능 비교
print(f"\n📈 최종 성능 비교 테스트")
forecast_comparison = seasonal_models.compare_forecasting_performance(seasonal_data, train_ratio=0.8)

print(f"\n✅ 계절성 모델링 완료!")
print(f"   🔄 SARIMA: 계절성 + 일반 패턴 통합 모델링")
print(f"   📊 지수평활법: 직관적이고 실용적인 예측 방법")
print(f"   📈 성능 비교: 실제 예측 정확도로 모델 검증")
print(f"   🎯 최적 모델: 비즈니스 요구사항에 맞는 선택")
print(f"\n🚀 다음: AI 협업을 통한 전통적 모델 최적화")

## 4. AI 협업을 통한 전통적 모델 최적화

### 4.1 7장 프롬프트 엔지니어링을 모델 해석에 적용

전통적 시계열 모델의 가장 큰 장점은 **해석 가능성**입니다. 하지만 복잡한 수학적 결과를 비즈니스 언어로 번역하는 것은 여전히 도전적입니다. 7장에서 배운 CLEAR 프롬프트 엔지니어링을 활용하여 모델 결과를 직관적으로 해석해보겠습니다.

```python
import json
from datetime import datetime, timedelta

class AIEnhancedTraditionalModels:
    """AI 협업 기반 전통적 모델 최적화 클래스"""
    
    def __init__(self):
        self.model_interpretations = {}
        self.optimization_history = []
        self.business_insights = {}
        
        # 7장 CLEAR 원칙을 시계열 모델 해석에 특화
        self.interpretation_prompts = {
            'arima_business': self._create_arima_business_prompt(),
            'sarima_seasonal': self._create_sarima_seasonal_prompt(),
            'forecast_reliability': self._create_forecast_reliability_prompt(),
            'model_selection': self._create_model_selection_prompt(),
            'risk_assessment': self._create_risk_assessment_prompt()
        }
    
    def _create_arima_business_prompt(self):
        """ARIMA 모델 비즈니스 해석용 CLEAR 프롬프트"""
        
        return """
당신은 15년 경력의 시계열 분석 전문가입니다.

**Context(맥락)**: 
- 기업의 핵심 KPI 시계열 데이터 분석
- 경영진과 실무진 모두에게 설명해야 하는 상황
- 데이터 기반 의사결정을 위한 실행 가능한 인사이트 필요

**Length(설명 범위)**:
다음 4개 영역을 각각 2-3문장으로 설명:
1. 모델이 발견한 핵심 패턴
2. 계수들의 비즈니스적 의미
3. 예측 신뢰도와 불확실성
4. 실무 적용 방안과 주의사항

**Examples(해석 기준)**:
- AR 계수 0.7 → "지난 달 성과가 이번 달에 70% 영향"
- MA 계수 -0.3 → "일시적 충격의 반작용 효과 존재"
- 높은 AIC → "모델 복잡도 대비 설명력 부족"

**Actionable(실행 가능한 결과)**:
1. 핵심 발견사항 3가지 (우선순위 순)
2. 단기/장기 예측의 신뢰도 평가
3. 비즈니스 의사결정을 위한 권고사항
4. 모니터링해야 할 핵심 지표

**Role(전문가 역할)**:
시계열 분석 전문가이자 비즈니스 컨설턴트로서 
복잡한 통계 결과를 경영 언어로 번역

**분석 대상**:
모델: ARIMA{order}
데이터: {data_description}
기간: {time_period}
계수 추정치: {coefficients}
진단 결과: {diagnostics}
AIC/BIC: {information_criteria}

위 정보를 바탕으로 체계적이고 실용적인 해석을 제공해주세요.
        """
    
    def _create_sarima_seasonal_prompt(self):
        """SARIMA 계절성 해석용 프롬프트"""
        
        return """
계절성 패턴 분석 전문가로서 SARIMA 모델 결과를 해석해주세요.

**분석 초점**:
1. 계절성 강도와 패턴의 안정성
2. 년도별 계절 패턴 변화 추이
3. 계절성이 비즈니스에 미치는 영향
4. 계절성을 활용한 전략적 대응 방안

**SARIMA 결과**:
모델: SARIMA{order}x{seasonal_order}
계절 주기: {seasonal_period}
계절 계수: {seasonal_coefficients}
분해 결과: {decomposition_summary}

계절성 관점에서 비즈니스 전략 수립에 도움되는 인사이트를 제공해주세요.
        """
    
    def _create_forecast_reliability_prompt(self):
        """예측 신뢰도 평가 프롬프트"""
        
        return """
예측 정확도 평가 전문가로서 모델의 신뢰도를 분석해주세요.

**평가 관점**:
1. 예측 구간의 적절성 (너무 넓거나 좁지 않은가?)
2. 과거 예측 성능 기반 미래 신뢰도 추정
3. 예측 기간별 신뢰도 변화 (단기 vs 장기)
4. 외부 충격 시나리오별 예측 안정성

**성능 지표**:
훈련 성능: {train_performance}
테스트 성능: {test_performance}
잔차 분석: {residual_analysis}
예측 구간: {confidence_intervals}

비즈니스 의사결정에서 이 예측을 어느 정도 신뢰할 수 있는지 평가해주세요.
        """
    
    def ai_interpret_arima_results(self, model_result, data_description):
        """AI 기반 ARIMA 결과 해석"""
        
        print("🤖 AI 협업 ARIMA 모델 해석")
        print("=" * 50)
        
        # 모델 정보 추출
        fitted_model = model_result['model']
        order = model_result['order']
        diagnostics = model_result.get('diagnostics', {})
        
        # 프롬프트 데이터 준비
        prompt_data = {
            'order': order,
            'data_description': data_description,
            'time_period': f"{fitted_model.model.endog.index[0].date()} ~ {fitted_model.model.endog.index[-1].date()}",
            'coefficients': dict(fitted_model.params),
            'diagnostics': diagnostics,
            'information_criteria': {
                'AIC': fitted_model.aic,
                'BIC': fitted_model.bic
            }
        }
        
        # AI 해석 시뮬레이션 (실제로는 LLM API 호출)
        interpretation = self._simulate_arima_interpretation(prompt_data)
        
        print(interpretation)
        
        # 해석 결과 저장
        self.model_interpretations[f'ARIMA{order}'] = {
            'interpretation': interpretation,
            'timestamp': datetime.now(),
            'model_data': prompt_data
        }
        
        return interpretation
    
    def _simulate_arima_interpretation(self, prompt_data):
        """ARIMA 해석 시뮬레이션"""
        
        order = prompt_data['order']
        coefficients = prompt_data['coefficients']
        aic = prompt_data['information_criteria']['AIC']
        
        # AR 계수 해석
        ar_interpretation = ""
        if order[0] > 0:  # AR 부분이 있는 경우
            ar_params = [v for k, v in coefficients.items() if 'ar.L' in k]
            if ar_params:
                ar1 = ar_params[0]
                if ar1 > 0.5:
                    momentum = "강한 모멘텀"
                elif ar1 > 0.2:
                    momentum = "중간 모멘텀"
                else:
                    momentum = "약한 모멘텀"
                
                ar_interpretation = f"""
📈 **모멘텀 분석 (AR 성분)**:
이 시계열은 {momentum}을 보입니다. AR 계수 {ar1:.3f}는 이전 기간 값이 현재 값에 
{abs(ar1)*100:.1f}% 영향을 미친다는 의미입니다. {"추세가 지속되는 경향이 강하므로" if ar1 > 0.5 else "변화가 상대적으로 빠르므로"} 
{"장기 계획 수립에 유리" if ar1 > 0.5 else "단기 대응 전략에 집중"}해야 합니다."""
        
        # MA 계수 해석  
        ma_interpretation = ""
        if order[2] > 0:  # MA 부분이 있는 경우
            ma_params = [v for k, v in coefficients.items() if 'ma.L' in k]
            if ma_params:
                ma1 = ma_params[0]
                shock_effect = "완충" if ma1 < 0 else "증폭"
                
                ma_interpretation = f"""
🌊 **충격 전파 분석 (MA 성분)**:
외부 충격이나 일시적 변동에 대해 이 시계열은 {shock_effect} 효과를 보입니다. 
MA 계수 {ma1:.3f}는 {"부정적 피드백으로 안정화 경향" if ma1 < 0 else "양의 피드백으로 변동 증폭"}을 의미합니다.
{"갑작스런 변화가 빠르게 정상화되므로 단기 대응만으로 충분" if ma1 < 0 else "작은 변화도 크게 확산될 수 있어 사전 예방이 중요"}합니다."""
        
        # 종합 평가
        model_quality = "우수" if aic < 1000 else "양호" if aic < 2000 else "개선 필요"
        
        comprehensive_interpretation = f"""
🤖 **ARIMA{order} 모델 종합 해석**

{ar_interpretation}

{ma_interpretation}

🎯 **핵심 발견사항 (우선순위 순)**:
1. **패턴 특성**: {"지속적 트렌드 중심" if order[0] > order[2] else "변동성 중심" if order[2] > order[0] else "균형형"} 시계열로 분류
2. **예측 안정성**: {model_quality} 수준의 모델 적합도 (AIC: {aic:.1f})
3. **시간 지평**: {"장기 예측에 적합" if order[0] > 0.5 else "단기 예측에 특화"}

💼 **비즈니스 적용 방안**:
- **단기 예측** (1-3개월): {"높은 신뢰도" if order[0] > 0.3 else "중간 신뢰도"} - 운영 계획 수립 가능
- **중기 예측** (3-12개월): {"신중한 접근" if order[0] < 0.3 else "안정적 활용"} - 전략 계획 참고
- **장기 예측** (12개월+): {"보조 지표로 활용" if order[0] < 0.5 else "핵심 지표로 활용"} 

⚠️ **주의사항**:
- 외부 환경 변화 시 모델 재검토 필요
- {f"AR 성분이 강하므로 추세 변화점 주의깊게 모니터링" if order[0] > 0.5 else "MA 성분 중심이므로 이상값 영향 지속적 관찰"}
- 정기적 모델 업데이트 권장 ({"분기별" if order[0] > 0.3 else "월별"})

📊 **핵심 모니터링 지표**:
1. 실제값 vs 예측값 차이 (MAE 기준)
2. 잔차 패턴 변화 (자기상관 여부)
3. 새로운 데이터 패턴 출현 여부
        """
        
        return comprehensive_interpretation.strip()
    
    def automated_model_tuning(self, data, business_objective='accuracy'):
        """자동화된 모델 튜닝 시스템"""
        
        print("⚙️ AI 기반 자동 모델 튜닝")
        print("=" * 50)
        print(f"🎯 비즈니스 목표: {business_objective}")
        
        # 비즈니스 목표에 따른 최적화 전략
        optimization_strategies = {
            'accuracy': {
                'primary_metric': 'mape',
                'complexity_penalty': 0.1,
                'focus': '예측 정확도 최대화'
            },
            'interpretability': {
                'primary_metric': 'aic',
                'complexity_penalty': 0.5,
                'focus': '해석 가능성 우선'
            },
            'speed': {
                'primary_metric': 'training_time',
                'complexity_penalty': 0.8,
                'focus': '계산 효율성 중시'
            },
            'stability': {
                'primary_metric': 'residual_variance',
                'complexity_penalty': 0.3,
                'focus': '안정적 예측 성능'
            }
        }
        
        strategy = optimization_strategies.get(business_objective, optimization_strategies['accuracy'])
        
        print(f"📊 최적화 전략: {strategy['focus']}")
        print(f"   주요 지표: {strategy['primary_metric']}")
        print(f"   복잡도 페널티: {strategy['complexity_penalty']}")
        
        # 자동 하이퍼파라미터 튜닝
        tuning_results = self._intelligent_hyperparameter_search(data, strategy)
        
        # 최적 모델 선택
        optimal_model = self._select_optimal_model(tuning_results, strategy)
        
        # 성능 모니터링 설정
        monitoring_setup = self._setup_performance_monitoring(optimal_model, data)
        
        return {
            'optimal_model': optimal_model,
            'tuning_results': tuning_results,
            'monitoring_setup': monitoring_setup,
            'strategy': strategy
        }
    
    def _intelligent_hyperparameter_search(self, data, strategy):
        """지능형 하이퍼파라미터 탐색"""
        
        print("🔍 지능형 하이퍼파라미터 탐색 중...")
        
        # 데이터 특성 분석
        data_characteristics = self._analyze_data_characteristics(data)
        
        # 특성 기반 초기 후보 생성
        initial_candidates = self._generate_smart_candidates(data_characteristics)
        
        print(f"   데이터 특성: {data_characteristics}")
        print(f"   초기 후보: {len(initial_candidates)}개 모델")
        
        # 베이지안 최적화 스타일 탐색
        optimization_results = []
        
        for i, candidate in enumerate(initial_candidates[:8]):  # 상위 8개만 테스트
            try:
                print(f"   모델 {i+1}: {candidate} 평가 중...")
                
                # 모델 훈련
                if len(candidate) == 3:  # ARIMA
                    model = ARIMA(data, order=candidate)
                else:  # SARIMA  
                    model = SARIMAX(data, order=candidate[:3], seasonal_order=candidate[3:])
                
                fitted_model = model.fit(disp=False)
                
                # 성능 평가
                performance = self._evaluate_model_performance(fitted_model, data, strategy)
                
                optimization_results.append({
                    'order': candidate,
                    'model': fitted_model,
                    'performance': performance,
                    'score': performance['composite_score']
                })
                
                print(f"     점수: {performance['composite_score']:.3f}")
                
            except Exception as e:
                print(f"     실패: {str(e)[:30]}...")
                continue
        
        # 결과 정렬
        optimization_results.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"\n🏆 상위 3개 모델:")
        for i, result in enumerate(optimization_results[:3]):
            order = result['order']
            score = result['score']
            print(f"   {i+1}. {'ARIMA' if len(order)==3 else 'SARIMA'}{order}: {score:.3f}")
        
        return optimization_results
    
    def _analyze_data_characteristics(self, data):
        """데이터 특성 자동 분석"""
        
        characteristics = {}
        
        # 기본 통계
        characteristics['length'] = len(data)
        characteristics['mean'] = data.mean()
        characteristics['std'] = data.std()
        characteristics['cv'] = characteristics['std'] / characteristics['mean'] if characteristics['mean'] != 0 else 0
        
        # 정상성
        adf_stat, adf_pvalue = adfuller(data)[:2]
        characteristics['is_stationary'] = adf_pvalue < 0.05
        characteristics['stationarity_strength'] = 1 - adf_pvalue if adf_pvalue < 1 else 0
        
        # 계절성 감지
        if len(data) >= 24:  # 최소 2년 데이터
            try:
                decomp = seasonal_decompose(data, model='additive', period=12)
                seasonal_strength = decomp.seasonal.std() / data.std()
                characteristics['has_seasonality'] = seasonal_strength > 0.1
                characteristics['seasonal_strength'] = seasonal_strength
            except:
                characteristics['has_seasonality'] = False
                characteristics['seasonal_strength'] = 0
        else:
            characteristics['has_seasonality'] = False
            characteristics['seasonal_strength'] = 0
        
        # 트렌드 감지
        x = np.arange(len(data))
        slope, _ = np.polyfit(x, data.values, 1)
        trend_strength = abs(slope) / characteristics['std'] if characteristics['std'] > 0 else 0
        characteristics['has_trend'] = trend_strength > 0.1
        characteristics['trend_strength'] = trend_strength
        
        return characteristics
    
    def _generate_smart_candidates(self, characteristics):
        """특성 기반 스마트 후보 생성"""
        
        candidates = []
        
        # 정상성에 따른 차분 차수
        if characteristics['is_stationary']:
            d_values = [0, 1]
        else:
            d_values = [1, 2]
        
        # 계절성에 따른 모델 선택
        if characteristics['has_seasonality']:
            # SARIMA 후보들
            for p in [0, 1, 2]:
                for d in d_values:
                    for q in [0, 1, 2]:
                        for P in [0, 1]:
                            for D in [0, 1]:
                                for Q in [0, 1]:
                                    candidates.append((p, d, q, P, D, Q, 12))
        else:
            # ARIMA 후보들
            for p in [0, 1, 2, 3]:
                for d in d_values:
                    for q in [0, 1, 2, 3]:
                        if p + q > 0:  # 최소 하나는 0이 아니어야 함
                            candidates.append((p, d, q))
        
        # 데이터 특성에 따른 우선순위 정렬
        def candidate_priority(candidate):
            score = 0
            
            if len(candidate) == 3:  # ARIMA
                p, d, q = candidate
                # 복잡도 페널티
                score -= (p + q) * 0.1
                # 트렌드 강도에 따른 AR 선호
                if characteristics['trend_strength'] > 0.2:
                    score += p * 0.2
            else:  # SARIMA
                p, d, q, P, D, Q, s = candidate
                # 복잡도 페널티
                score -= (p + q + P + Q) * 0.1
                # 계절성 강도에 따른 계절 성분 선호
                if characteristics['seasonal_strength'] > 0.2:
                    score += (P + Q) * 0.3
            
            return score
        
        # 우선순위 정렬
        candidates.sort(key=candidate_priority, reverse=True)
        
        return candidates[:20]  # 상위 20개만 반환
    
    def _evaluate_model_performance(self, fitted_model, data, strategy):
        """모델 성능 종합 평가"""
        
        performance = {}
        
        # 기본 메트릭들
        performance['aic'] = fitted_model.aic
        performance['bic'] = fitted_model.bic
        performance['llf'] = fitted_model.llf
        
        # 잔차 분석
        residuals = fitted_model.resid.dropna()
        if len(residuals) > 0:
            performance['residual_mean'] = residuals.mean()
            performance['residual_std'] = residuals.std()
            performance['residual_variance'] = residuals.var()
        
        # 예측 성능 (훈련 데이터 내)
        try:
            fitted_values = fitted_model.fittedvalues
            valid_mask = ~np.isnan(fitted_values)
            
            if valid_mask.sum() > 0:
                actual = data[valid_mask]
                predicted = fitted_values[valid_mask]
                
                performance['mape'] = mean_absolute_percentage_error(actual, predicted)
                performance['mae'] = mean_absolute_error(actual, predicted)
                performance['rmse'] = np.sqrt(mean_squared_error(actual, predicted))
        except:
            performance['mape'] = float('inf')
            performance['mae'] = float('inf')
            performance['rmse'] = float('inf')
        
        # 복합 점수 계산
        performance['composite_score'] = self._calculate_composite_score(performance, strategy)
        
        return performance
    
    def _calculate_composite_score(self, performance, strategy):
        """전략에 따른 복합 점수 계산"""
        
        score = 0
        
        if strategy['primary_metric'] == 'mape':
            # 정확도 중심
            mape = performance.get('mape', float('inf'))
            if mape != float('inf'):
                score += (1 - min(mape, 1)) * 0.6  # 60% 가중치
            score += (1 / (1 + performance.get('aic', 1000) / 1000)) * 0.3  # 30% 가중치
            score += (1 - performance.get('residual_variance', 1)) * 0.1  # 10% 가중치
            
        elif strategy['primary_metric'] == 'aic':
            # 해석가능성 중심
            score += (1 / (1 + performance.get('aic', 1000) / 1000)) * 0.7
            score += (1 / (1 + performance.get('bic', 1000) / 1000)) * 0.3
            
        elif strategy['primary_metric'] == 'residual_variance':
            # 안정성 중심
            score += (1 - min(performance.get('residual_variance', 1), 1)) * 0.5
            score += abs(performance.get('residual_mean', 1)) < 0.01 * 0.3  # 평균이 0에 가까운가
            score += (1 / (1 + performance.get('aic', 1000) / 1000)) * 0.2
        
        # 복잡도 페널티 적용
        complexity_penalty = strategy.get('complexity_penalty', 0.1)
        # 여기서는 단순화하여 AIC 기반으로 복잡도 추정
        complexity = performance.get('aic', 1000) / 1000
        score *= (1 - complexity_penalty * complexity)
        
        return max(score, 0)  # 음수 방지
    
    def _select_optimal_model(self, tuning_results, strategy):
        """최적 모델 선택"""
        
        if not tuning_results:
            return None
        
        # 점수 기준 최적 모델
        best_result = tuning_results[0]
        
        print(f"\n🎯 최적 모델 선택:")
        print(f"   모델: {'ARIMA' if len(best_result['order'])==3 else 'SARIMA'}{best_result['order']}")
        print(f"   종합 점수: {best_result['score']:.3f}")
        print(f"   AIC: {best_result['performance']['aic']:.2f}")
        
        return best_result
    
    def _setup_performance_monitoring(self, optimal_model, data):
        """성능 모니터링 설정"""
        
        print(f"\n📊 성능 모니터링 시스템 설정")
        
        monitoring_config = {
            'model_info': {
                'order': optimal_model['order'],
                'created_at': datetime.now().isoformat(),
                'training_period': f"{data.index[0]} ~ {data.index[-1]}"
            },
            'performance_thresholds': {
                'mape_threshold': optimal_model['performance'].get('mape', 0.1) * 1.5,  # 50% 증가 시 경고
                'aic_threshold': optimal_model['performance'].get('aic', 1000) * 1.2,   # 20% 증가 시 경고
                'residual_std_threshold': optimal_model['performance'].get('residual_std', 1) * 2
            },
            'monitoring_schedule': {
                'daily_check': ['new_data_validation', 'forecast_accuracy'],
                'weekly_check': ['residual_analysis', 'parameter_stability'],
                'monthly_check': ['model_retraining', 'performance_report']
            },
            'alert_conditions': [
                'mape_degradation',
                'residual_pattern_change', 
                'forecast_interval_violation',
                'parameter_instability'
            ]
        }
        
        print(f"   MAPE 임계값: {monitoring_config['performance_thresholds']['mape_threshold']:.1%}")
        print(f"   모니터링 주기: 일일/주간/월간 체크")
        print(f"   알림 조건: {len(monitoring_config['alert_conditions'])}개 설정")
        
        return monitoring_config

# AI 협업 모델 최적화 시스템 실행
ai_enhanced_models = AIEnhancedTraditionalModels()

print("\n🤖 AI 협업 기반 전통적 모델 최적화")
print("=" * 60)

# 1. AI 기반 ARIMA 결과 해석 (이전 결과 활용)
if 'final_arima_model' in globals() and final_arima_model:
    print("📊 ARIMA 모델 AI 해석:")
    ai_interpretation = ai_enhanced_models.ai_interpret_arima_results(
        final_arima_model, 
        "월별 매출 데이터 - 트렌드와 약간의 계절성 포함"
    )

# 2. 자동화된 모델 튜닝 시연
print(f"\n⚙️ 자동 모델 튜닝 시연:")

# 테스트 데이터 준비
if 'seasonal_data' in globals():
    test_data = seasonal_data
else:
    # 시뮬레이션 데이터 생성
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='M')
    trend = np.linspace(100, 130, len(dates))
    seasonal = 8 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
    noise = np.random.normal(0, 3, len(dates))
    test_data = pd.Series(trend + seasonal + noise, index=dates, name='revenue')

# 비즈니스 목표별 최적화
objectives = ['accuracy', 'interpretability', 'stability']

optimization_results = {}
for objective in objectives:
    print(f"\n🎯 {objective.upper()} 목표 최적화:")
    try:
        result = ai_enhanced_models.automated_model_tuning(test_data, business_objective=objective)
        optimization_results[objective] = result
        
        if result['optimal_model']:
            order = result['optimal_model']['order']
            score = result['optimal_model']['score']
            print(f"   최적 모델: {'ARIMA' if len(order)==3 else 'SARIMA'}{order}")
            print(f"   최적화 점수: {score:.3f}")
    except Exception as e:
        print(f"   ❌ 오류: {str(e)[:50]}...")

print(f"\n✅ AI 협업 모델 최적화 완료!")
print(f"   🤖 CLEAR 프롬프트로 모델 해석 자동화")
print(f"   ⚙️ 비즈니스 목표별 자동 하이퍼파라미터 튜닝")
print(f"   📊 지능형 성능 모니터링 시스템 구축")
print(f"   🎯 실무 적용 가능한 최적화 프레임워크")
print(f"\n🚀 다음: Store Sales 실전 프로젝트")

## 5. 실전 프로젝트: Store Sales 시계열 예측 시스템

### 5.1 프로젝트 개요 및 비즈니스 컨텍스트

이제 8장 Part 2에서 배운 모든 전통적 시계열 기법들을 실제 비즈니스 데이터에 적용해보겠습니다. **Kaggle Store Sales - Time Series Forecasting** 데이터셋을 활용하여 완전한 예측 시스템을 구축하겠습니다.

```python
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st  # 대시보드용 (시뮬레이션)

class StoreSalesForecastingSystem:
    """Store Sales 시계열 예측 시스템"""
    
    def __init__(self):
        self.data = None
        self.models = {}
        self.forecasts = {}
        self.ensemble_results = {}
        self.dashboard_data = {}
        
        # 비즈니스 KPI 설정
        self.business_kpis = {
            'forecast_accuracy_target': 0.15,  # MAPE 15% 이하
            'inventory_optimization_threshold': 0.10,  # 10% 재고 최적화
            'promotional_planning_horizon': 8,  # 8주 선행 계획
            'seasonal_adjustment_factor': 1.2   # 계절성 20% 조정
        }
    
    def simulate_store_sales_data(self):
        """Store Sales 데이터 시뮬레이션 (실제 Kaggle 데이터 구조)"""
        
        print("📊 Store Sales 데이터셋 시뮬레이션")
        print("=" * 50)
        print("실제 에콰도르 소매점 체인 Favorita의 매출 패턴을 모방한 시뮬레이션")
        
        # 시간 범위: 2017-2021 (5년간 주간 데이터)
        dates = pd.date_range('2017-01-01', '2021-12-31', freq='W')
        n_periods = len(dates)
        
        # 다양한 상품군별 시계열 생성
        product_families = [
            'GROCERY I', 'BEVERAGES', 'PRODUCE', 'CLEANING', 'DAIRY',
            'BREAD/BAKERY', 'POULTRY', 'MEATS', 'PERSONAL CARE', 'FROZEN FOODS'
        ]
        
        # 매장별 특성
        store_info = {
            'store_1': {'type': 'Supermarket', 'city': 'Quito', 'size_factor': 1.2},
            'store_2': {'type': 'Grocery', 'city': 'Guayaquil', 'size_factor': 0.8},
            'store_3': {'type': 'Hypermarket', 'city': 'Cuenca', 'size_factor': 1.5},
            'store_4': {'type': 'Supermarket', 'city': 'Machala', 'size_factor': 1.0}
        }
        
        # 전체 데이터 구조 생성
        sales_data = []
        
        for store_id, store_attrs in store_info.items():
            for family in product_families:
                
                # 기본 트렌드 (경제 성장 반영)
                base_trend = np.linspace(1000, 1300, n_periods) * store_attrs['size_factor']
                
                # 상품군별 특성
                family_factors = {
                    'GROCERY I': 1.5,      # 가장 높은 매출
                    'BEVERAGES': 1.2,      # 높은 매출
                    'PRODUCE': 1.0,        # 기준
                    'CLEANING': 0.6,       # 낮은 매출
                    'DAIRY': 0.8,          # 중간 매출
                    'BREAD/BAKERY': 0.7,   # 중간 매출
                    'POULTRY': 0.9,        # 중간 매출
                    'MEATS': 1.1,          # 높은 매출
                    'PERSONAL CARE': 0.5,  # 낮은 매출
                    'FROZEN FOODS': 0.4    # 낮은 매출
                }
                
                family_factor = family_factors.get(family, 1.0)
                
                # 계절성 패턴 (연간 + 월간)
                annual_seasonal = 0.15 * np.sin(2 * np.pi * np.arange(n_periods) / 52.18)  # 연간
                monthly_seasonal = 0.08 * np.sin(2 * np.pi * np.arange(n_periods) / 4.33)  # 월간
                
                # 특별 이벤트 효과 (크리스마스, 어머니날 등)
                special_events = np.zeros(n_periods)
                for year in range(2017, 2022):
                    # 크리스마스 효과 (12월)
                    christmas_week = pd.to_datetime(f'{year}-12-25').week - 1
                    if christmas_week < n_periods:
                        special_events[christmas_week] += 0.3
                    
                    # 어머니날 효과 (5월)
                    mothers_day_week = pd.to_datetime(f'{year}-05-10').week - 1
                    if mothers_day_week < n_periods:
                        special_events[mothers_day_week] += 0.2
                
                # 경제적 충격 (팬데믹 등)
                economic_shock = np.zeros(n_periods)
                pandemic_start = pd.to_datetime('2020-03-15').week
                pandemic_end = pd.to_datetime('2020-08-01').week
                if pandemic_start < n_periods:
                    shock_period = slice(pandemic_start, min(pandemic_end, n_periods))
                    economic_shock[shock_period] = -0.25  # 25% 감소
                
                # 노이즈
                noise = np.random.normal(0, 0.05, n_periods)
                
                # 최종 시계열 합성
                sales_values = (base_trend * family_factor * 
                               (1 + annual_seasonal + monthly_seasonal + 
                                special_events + economic_shock + noise))
                
                # 음수 방지
                sales_values = np.maximum(sales_values, 0)
                
                # 데이터 저장
                for i, (date, sales) in enumerate(zip(dates, sales_values)):
                    sales_data.append({
                        'date': date,
                        'store_nbr': store_id,
                        'family': family,
                        'sales': sales,
                        'store_type': store_attrs['type'],
                        'city': store_attrs['city']
                    })
        
        # DataFrame 생성
        self.data = pd.DataFrame(sales_data)
        
        # 총 매출 계산 (전체 상품군 합계)
        total_sales = self.data.groupby('date')['sales'].sum().reset_index()
        total_sales['store_nbr'] = 'ALL'
        total_sales['family'] = 'TOTAL'
        total_sales['store_type'] = 'ALL'
        total_sales['city'] = 'ALL'
        
        # 총 매출을 데이터에 추가
        self.data = pd.concat([self.data, total_sales], ignore_index=True)
        
        print(f"✅ 데이터 생성 완료:")
        print(f"   기간: {self.data['date'].min().date()} ~ {self.data['date'].max().date()}")
        print(f"   매장 수: {len(store_info)}개")
        print(f"   상품군 수: {len(product_families)}개")
        print(f"   총 데이터 포인트: {len(self.data):,}개")
        print(f"   주간 총 매출 평균: ${self.data[self.data['family']=='TOTAL']['sales'].mean():,.0f}")
        
        return self.data
    
    def comprehensive_eda_analysis(self):
        """종합적 EDA 분석"""
        
        print("\n📊 Store Sales 데이터 종합 EDA")
        print("=" * 50)
        
        # 총 매출 데이터 추출
        total_sales = self.data[self.data['family'] == 'TOTAL'].copy()
        total_sales = total_sales.set_index('date')['sales'].sort_index()
        
        # 기본 통계
        print(f"📈 총 매출 기본 통계:")
        print(f"   평균: ${total_sales.mean():,.0f}")
        print(f"   중앙값: ${total_sales.median():,.0f}")
        print(f"   표준편차: ${total_sales.std():,.0f}")
        print(f"   최대값: ${total_sales.max():,.0f}")
        print(f"   최소값: ${total_sales.min():,.0f}")
        
        # 시계열 분해
        decomposition = seasonal_decompose(total_sales, model='additive', period=52)
        
        # 계절성 강도 계산
        seasonal_strength = decomposition.seasonal.std() / total_sales.std()
        trend_strength = decomposition.trend.dropna().std() / total_sales.std()
        
        print(f"\n🔍 시계열 구성 요소:")
        print(f"   트렌드 강도: {trend_strength:.1%}")
        print(f"   계절성 강도: {seasonal_strength:.1%}")
        print(f"   잔차 비율: {(1-trend_strength-seasonal_strength):.1%}")
        
        # 상품군별 매출 분석
        family_sales = self.data[self.data['family'] != 'TOTAL'].groupby('family')['sales'].sum().sort_values(ascending=False)
        
        print(f"\n🛒 상위 5개 상품군:")
        for i, (family, sales) in enumerate(family_sales.head().items()):
            print(f"   {i+1}. {family}: ${sales:,.0f}")
        
        # 매장별 성과 분석
        store_performance = self.data[self.data['family'] != 'TOTAL'].groupby('store_nbr')['sales'].agg(['sum', 'mean', 'std'])
        
        print(f"\n🏪 매장별 성과:")
        for store in store_performance.index:
            total = store_performance.loc[store, 'sum']
            avg = store_performance.loc[store, 'mean']
            print(f"   {store}: 총 ${total:,.0f}, 평균 ${avg:,.0f}")
        
        # 시각화
        self._create_comprehensive_eda_plots(total_sales, decomposition, family_sales)
        
        return {
            'total_sales': total_sales,
            'decomposition': decomposition,
            'family_sales': family_sales,
            'store_performance': store_performance,
            'seasonal_strength': seasonal_strength,
            'trend_strength': trend_strength
        }
    
    def _create_comprehensive_eda_plots(self, total_sales, decomposition, family_sales):
        """종합 EDA 시각화"""
        
        # 1. 시계열 분해 플롯
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        fig.suptitle('🏪 Store Sales 시계열 분해 분석', fontsize=16, fontweight='bold')
        
        # 원본 데이터
        axes[0].plot(total_sales.index, total_sales.values, color='black', linewidth=1.5)
        axes[0].set_title('📊 총 매출 (주간)', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # 트렌드
        axes[1].plot(decomposition.trend.index, decomposition.trend.values, color='blue', linewidth=2)
        axes[1].set_title('📈 트렌드 성분', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # 계절성
        axes[2].plot(decomposition.seasonal.index, decomposition.seasonal.values, color='green', linewidth=1)
        axes[2].set_title('🔄 계절성 성분 (연간 패턴)', fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        # 잔차
        axes[3].plot(decomposition.resid.index, decomposition.resid.values, color='red', linewidth=1, alpha=0.7)
        axes[3].set_title('🎲 잔차 성분', fontweight='bold')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 2. 상품군별 매출 분석
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('🛒 상품군별 매출 분석', fontsize=16, fontweight='bold')
        
        # 상품군별 총 매출
        family_sales.head(8).plot(kind='barh', ax=ax1, color='skyblue')
        ax1.set_title('상위 8개 상품군 총 매출', fontweight='bold')
        ax1.set_xlabel('총 매출 ($)')
        
        # 상품군별 매출 비율
        top_families = family_sales.head(6)
        others = family_sales.iloc[6:].sum()
        pie_data = pd.concat([top_families, pd.Series({'기타': others})])
        
        ax2.pie(pie_data.values, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
        ax2.set_title('상품군별 매출 비율', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def build_comprehensive_forecasting_models(self):
        """종합적 예측 모델 구축"""
        
        print("\n🔮 종합 예측 모델 구축")
        print("=" * 50)
        
        # 총 매출 데이터 준비
        total_sales = self.data[self.data['family'] == 'TOTAL'].copy()
        total_sales = total_sales.set_index('date')['sales'].sort_index()
        
        # 훈련/검증 분할 (80:20)
        split_point = int(len(total_sales) * 0.8)
        train_data = total_sales[:split_point]
        test_data = total_sales[split_point:]
        
        print(f"📊 데이터 분할:")
        print(f"   훈련: {len(train_data)}주 ({train_data.index[0].date()} ~ {train_data.index[-1].date()})")
        print(f"   테스트: {len(test_data)}주 ({test_data.index[0].date()} ~ {test_data.index[-1].date()})")
        
        forecasting_results = {}
        
        # 1. ARIMA 모델
        print(f"\n🔄 ARIMA 모델 구축:")
        try:
            arima_builder = ARIMAModelBuilder()
            arima_result = arima_builder.box_jenkins_methodology(train_data, max_p=3, max_d=2, max_q=3)
            
            if arima_result and arima_result['model']:
                arima_forecast = arima_result['model'].forecast(steps=len(test_data))
                arima_mape = mean_absolute_percentage_error(test_data, arima_forecast)
                
                forecasting_results['ARIMA'] = {
                    'model': arima_result['model'],
                    'forecast': arima_forecast,
                    'mape': arima_mape,
                    'order': arima_result['order']
                }
                print(f"   ✅ ARIMA{arima_result['order']}: MAPE = {arima_mape:.2%}")
            else:
                print(f"   ❌ ARIMA 모델 구축 실패")
        except Exception as e:
            print(f"   ❌ ARIMA 오류: {str(e)[:50]}...")
        
        # 2. SARIMA 모델
        print(f"\n🔄 SARIMA 모델 구축:")
        try:
            seasonal_models = SeasonalTimeSeriesModels()
            sarima_model, _ = seasonal_models.build_sarima_model(train_data, seasonal_period=52)
            
            if sarima_model:
                sarima_forecast = sarima_model.forecast(steps=len(test_data))
                sarima_mape = mean_absolute_percentage_error(test_data, sarima_forecast)
                
                forecasting_results['SARIMA'] = {
                    'model': sarima_model,
                    'forecast': sarima_forecast,
                    'mape': sarima_mape,
                    'order': 'Auto-selected'
                }
                print(f"   ✅ SARIMA: MAPE = {sarima_mape:.2%}")
            else:
                print(f"   ❌ SARIMA 모델 구축 실패")
        except Exception as e:
            print(f"   ❌ SARIMA 오류: {str(e)[:50]}...")
        
        # 3. Holt-Winters 지수평활법
        print(f"\n📊 Holt-Winters 지수평활법:")
        try:
            hw_model = ExponentialSmoothing(train_data, trend='add', seasonal='add', seasonal_periods=52)
            hw_fitted = hw_model.fit()
            hw_forecast = hw_fitted.forecast(steps=len(test_data))
            hw_mape = mean_absolute_percentage_error(test_data, hw_forecast)
            
            forecasting_results['Holt-Winters'] = {
                'model': hw_fitted,
                'forecast': hw_forecast,
                'mape': hw_mape,
                'order': 'Additive'
            }
            print(f"   ✅ Holt-Winters: MAPE = {hw_mape:.2%}")
        except Exception as e:
            print(f"   ❌ Holt-Winters 오류: {str(e)[:50]}...")
        
        # 4. 앙상블 모델 (가중 평균)
        print(f"\n🎯 앙상블 모델 구축:")
        if len(forecasting_results) >= 2:
            # 성능 기반 가중치 계산
            weights = {}
            total_inverse_mape = 0
            
            for model_name, result in forecasting_results.items():
                inverse_mape = 1 / (result['mape'] + 0.001)  # 0으로 나누기 방지
                weights[model_name] = inverse_mape
                total_inverse_mape += inverse_mape
            
            # 가중치 정규화
            for model_name in weights:
                weights[model_name] /= total_inverse_mape
            
            # 앙상블 예측 계산
            ensemble_forecast = pd.Series(0, index=test_data.index)
            for model_name, result in forecasting_results.items():
                ensemble_forecast += weights[model_name] * result['forecast']
            
            ensemble_mape = mean_absolute_percentage_error(test_data, ensemble_forecast)
            
            forecasting_results['Ensemble'] = {
                'model': 'Weighted Average',
                'forecast': ensemble_forecast,
                'mape': ensemble_mape,
                'weights': weights
            }
            
            print(f"   ✅ 앙상블 모델: MAPE = {ensemble_mape:.2%}")
            print(f"   가중치: {', '.join([f'{k}:{v:.2f}' for k, v in weights.items()])}")
        
        # 5. 베이스라인 모델 (계절적 Naive)
        print(f"\n📊 베이스라인 (계절적 Naive):")
        try:
            # 52주 전 값 사용 (연간 계절성)
            naive_forecast = []
            for i in range(len(test_data)):
                if len(train_data) >= 52:
                    naive_forecast.append(train_data.iloc[-(52-i%52)])
                else:
                    naive_forecast.append(train_data.iloc[-1])
            
            naive_forecast = pd.Series(naive_forecast, index=test_data.index)
            naive_mape = mean_absolute_percentage_error(test_data, naive_forecast)
            
            forecasting_results['Seasonal_Naive'] = {
                'model': 'Seasonal Naive',
                'forecast': naive_forecast,
                'mape': naive_mape,
                'order': '52-week lag'
            }
            print(f"   ✅ Seasonal Naive: MAPE = {naive_mape:.2%}")
        except Exception as e:
            print(f"   ❌ Naive 오류: {str(e)[:50]}...")
        
        # 결과 시각화
        self._visualize_model_comparison(train_data, test_data, forecasting_results)
        
        # 성능 요약
        self._print_model_performance_summary(forecasting_results)
        
        self.models = forecasting_results
        return forecasting_results
    
    def _visualize_model_comparison(self, train_data, test_data, results):
        """모델 비교 시각화"""
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        fig.suptitle('🔮 Store Sales 예측 모델 성능 비교', fontsize=16, fontweight='bold')
        
        # 1. 시계열 예측 비교
        ax1.plot(train_data.index, train_data.values, 'o-', label='훈련 데이터', 
                color='black', linewidth=2, markersize=3, alpha=0.7)
        ax1.plot(test_data.index, test_data.values, 'o-', label='실제 값', 
                color='blue', linewidth=2, markersize=4)
        
        colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink']
        for i, (model_name, result) in enumerate(results.items()):
            ax1.plot(test_data.index, result['forecast'], '--', 
                    label=f'{model_name} (MAPE: {result["mape"]:.1%})',
                    color=colors[i % len(colors)], linewidth=2)
        
        ax1.set_title('📈 예측 성능 비교', fontweight='bold')
        ax1.set_ylabel('매출 ($)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 훈련/테스트 구분선
        split_line = train_data.index[-1]
        ax1.axvline(x=split_line, color='gray', linestyle=':', alpha=0.7, linewidth=2)
        ax1.text(split_line, ax1.get_ylim()[1]*0.9, '훈련/테스트 분할', 
                rotation=90, ha='right', va='top', fontweight='bold')
        
        # 2. MAPE 비교 막대 그래프
        model_names = list(results.keys())
        mape_values = [results[name]['mape'] for name in model_names]
        
        bars = ax2.bar(model_names, mape_values, color=colors[:len(model_names)], alpha=0.7)
        ax2.set_title('📊 모델별 MAPE 비교', fontweight='bold')
        ax2.set_ylabel('MAPE (%)')
        ax2.grid(True, alpha=0.3)
        
        # 목표 MAPE 선 추가
        target_mape = self.business_kpis['forecast_accuracy_target']
        ax2.axhline(y=target_mape, color='red', linestyle='--', linewidth=2, 
                   label=f'목표 MAPE: {target_mape:.1%}')
        ax2.legend()
        
        # 값 표시
        for bar, mape in zip(bars, mape_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{mape:.1%}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def _print_model_performance_summary(self, results):
        """모델 성능 요약 출력"""
        
        print(f"\n🏆 모델 성능 종합 평가")
        print("=" * 50)
        
        # MAPE 기준 정렬
        sorted_results = sorted(results.items(), key=lambda x: x[1]['mape'])
        
        print(f"📊 예측 정확도 순위 (MAPE 기준):")
        for rank, (model_name, result) in enumerate(sorted_results, 1):
            mape = result['mape']
            emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "🏅"
            
            # 목표 달성 여부
            target_achieved = "✅" if mape <= self.business_kpis['forecast_accuracy_target'] else "❌"
            
            print(f"   {emoji} {rank}위: {model_name}")
            print(f"      MAPE: {mape:.2%} {target_achieved}")
            print(f"      정확도: {(1-mape):.1%}")
            print()
        
        # 비즈니스 관점 평가
        best_model = sorted_results[0]
        best_mape = best_model[1]['mape']
        
        print(f"💼 비즈니스 관점 평가:")
        if best_mape <= 0.10:
            grade = "A+ (탁월)"
            recommendation = "즉시 프로덕션 배포 권장"
        elif best_mape <= 0.15:
            grade = "A (우수)"
            recommendation = "검토 후 배포 가능"
        elif best_mape <= 0.20:
            grade = "B (양호)"
            recommendation = "추가 튜닝 후 배포"
        elif best_mape <= 0.30:
            grade = "C (보통)"
            recommendation = "상당한 개선 필요"
        else:
            grade = "D (개선 필요)"
            recommendation = "모델 재설계 필요"
        
        print(f"   최우수 모델: {best_model[0]}")
        print(f"   성능 등급: {grade}")
        print(f"   권장사항: {recommendation}")
        
        # ROI 분석
        if best_mape <= self.business_kpis['forecast_accuracy_target']:
            inventory_optimization = self.business_kpis['inventory_optimization_threshold']
            print(f"\n💰 예상 비즈니스 효과:")
            print(f"   재고 최적화: {inventory_optimization:.0%} 개선")
            print(f"   결품 손실 감소: 15% 추정")
            print(f"   운영 효율성: 20% 향상")
    
    def create_business_dashboard(self):
        """비즈니스 대시보드 생성"""
        
        print(f"\n📊 비즈니스 대시보드 구축")
        print("=" * 50)
        
        if not self.models:
            print("❌ 예측 모델이 없습니다. 먼저 모델을 구축해주세요.")
            return
        
        # 최우수 모델 선택
        best_model_name = min(self.models.keys(), key=lambda x: self.models[x]['mape'])
        best_model = self.models[best_model_name]
        
        print(f"🎯 대시보드 기준 모델: {best_model_name}")
        print(f"   예측 정확도: {(1-best_model['mape']):.1%}")
        
        # 미래 8주 예측 (비즈니스 계획용)
        forecast_horizon = self.business_kpis['promotional_planning_horizon']
        
        # 총 매출 데이터
        total_sales = self.data[self.data['family'] == 'TOTAL'].copy()
        total_sales = total_sales.set_index('date')['sales'].sort_index()
        
        # 미래 예측
        if best_model_name == 'Ensemble':
            # 앙상블의 경우 개별 모델들로 예측 후 가중 평균
            future_forecast = pd.Series(0, index=pd.date_range(
                total_sales.index[-1] + pd.Timedelta(weeks=1), 
                periods=forecast_horizon, freq='W'))
            
            for model_name, weight in best_model['weights'].items():
                if model_name in self.models and hasattr(self.models[model_name]['model'], 'forecast'):
                    individual_forecast = self.models[model_name]['model'].forecast(steps=forecast_horizon)
                    future_forecast += weight * individual_forecast
        else:
            # 개별 모델 예측
            future_forecast = best_model['model'].forecast(steps=forecast_horizon)
        
        # 대시보드 데이터 구성
        dashboard_data = {
            'historical_data': total_sales,
            'future_forecast': future_forecast,
            'model_performance': {
                'best_model': best_model_name,
                'accuracy': f"{(1-best_model['mape']):.1%}",
                'mape': f"{best_model['mape']:.1%}",
                'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M")
            },
            'business_metrics': self._calculate_business_metrics(total_sales, future_forecast),
            'alerts': self._generate_business_alerts(total_sales, future_forecast)
        }
        
        # 대시보드 시각화
        self._create_dashboard_visualization(dashboard_data)
        
        self.dashboard_data = dashboard_data
        return dashboard_data
    
    def _calculate_business_metrics(self, historical_data, forecast):
        """비즈니스 메트릭 계산"""
        
        # 최근 4주 vs 예측 4주 비교
        recent_4weeks = historical_data.tail(4).mean()
        forecast_4weeks = forecast.head(4).mean()
        
        growth_rate = (forecast_4weeks - recent_4weeks) / recent_4weeks
        
        # 계절성 조정
        seasonal_factor = self.business_kpis['seasonal_adjustment_factor']
        seasonal_adjusted_forecast = forecast * seasonal_factor
        
        return {
            'recent_avg_weekly_sales': f"${recent_4weeks:,.0f}",
            'forecast_avg_weekly_sales': f"${forecast_4weeks:,.0f}",
            'growth_rate': f"{growth_rate:+.1%}",
            'total_forecast_8weeks': f"${forecast.sum():,.0f}",
            'seasonal_adjusted_total': f"${seasonal_adjusted_forecast.sum():,.0f}",
            'peak_week_forecast': f"${forecast.max():,.0f}",
            'lowest_week_forecast': f"${forecast.min():,.0f}"
        }
    
    def _generate_business_alerts(self, historical_data, forecast):
        """비즈니스 알림 생성"""
        
        alerts = []
        
        # 1. 급격한 변화 감지
        recent_avg = historical_data.tail(4).mean()
        forecast_avg = forecast.head(4).mean()
        change_rate = (forecast_avg - recent_avg) / recent_avg
        
        if abs(change_rate) > 0.15:  # 15% 이상 변화
            alert_type = "📈 급증 예상" if change_rate > 0 else "📉 급감 예상"
            alerts.append({
                'type': alert_type,
                'message': f"다음 4주 평균 매출이 {change_rate:+.1%} 변화 예상",
                'priority': 'HIGH' if abs(change_rate) > 0.25 else 'MEDIUM'
            })
        
        # 2. 재고 최적화 알림
        peak_forecast = forecast.max()
        if peak_forecast > historical_data.quantile(0.9):  # 상위 10% 수준
            alerts.append({
                'type': '📦 재고 준비',
                'message': f"예측 피크 매출 ${peak_forecast:,.0f} - 재고 확보 필요",
                'priority': 'MEDIUM'
            })
        
        # 3. 프로모션 기회
        lowest_forecast = forecast.min()
        if lowest_forecast < historical_data.quantile(0.3):  # 하위 30% 수준
            alerts.append({
                'type': '🎯 프로모션 기회',
                'message': f"예측 최저 매출 주간 - 마케팅 캠페인 고려",
                'priority': 'LOW'
            })
        
        return alerts
    
    def _create_dashboard_visualization(self, dashboard_data):
        """대시보드 시각화"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['📈 매출 예측', '📊 주간 성장률', '🎯 비즈니스 메트릭', '⚠️ 알림 및 권고사항'],
            specs=[[{"colspan": 2}, None],
                   [{"type": "bar"}, {"type": "table"}]]
        )
        
        historical = dashboard_data['historical_data']
        forecast = dashboard_data['future_forecast']
        
        # 1. 매출 예측 차트
        fig.add_trace(
            go.Scatter(x=historical.index, y=historical.values, 
                      mode='lines', name='과거 매출', line=dict(color='blue')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=forecast.index, y=forecast.values,
                      mode='lines+markers', name='예측 매출', line=dict(color='red', dash='dash')),
            row=1, col=1
        )
        
        # 2. 주간 성장률
        growth_rates = forecast.pct_change().fillna(0) * 100
        fig.add_trace(
            go.Bar(x=forecast.index, y=growth_rates.values, name='주간 성장률 (%)', marker_color='green'),
            row=2, col=1
        )
        
        # 3. 비즈니스 메트릭 테이블
        metrics = dashboard_data['business_metrics']
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        fig.add_trace(
            go.Table(
                header=dict(values=['메트릭', '값'], fill_color='lightblue'),
                cells=dict(values=[metric_names, metric_values], fill_color='white')
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="🏪 Store Sales 예측 대시보드",
            title_x=0.5,
            height=800,
            showlegend=True
        )
        
        fig.show()
        
        # 알림 출력
        print(f"\n⚠️ 비즈니스 알림:")
        for alert in dashboard_data['alerts']:
            priority_emoji = "🔴" if alert['priority'] == 'HIGH' else "🟡" if alert['priority'] == 'MEDIUM' else "🟢"
            print(f"   {priority_emoji} {alert['type']}: {alert['message']}")
        
        return dashboard_data
    
    def generate_executive_report(self):
        """경영진 요약 보고서 생성"""
        
        if not self.models or not self.dashboard_data:
            print("❌ 보고서 생성을 위해서는 모델과 대시보드가 필요합니다.")
            return
        
        print(f"\n📋 경영진 요약 보고서")
        print("=" * 50)
        
        # 최우수 모델 정보
        best_model_name = min(self.models.keys(), key=lambda x: self.models[x]['mape'])
        best_model = self.models[best_model_name]
        
        # 총 매출 데이터
        total_sales = self.data[self.data['family'] == 'TOTAL'].copy()
        total_sales = total_sales.set_index('date')['sales'].sort_index()
        
        # 연간 매출 추이
        annual_sales = total_sales.resample('Y').sum()
        
        print(f"📊 **핵심 성과 지표**")
        print(f"   예측 정확도: {(1-best_model['mape']):.1%} ({best_model_name} 모델)")
        print(f"   연간 총 매출: ${annual_sales.iloc[-1]:,.0f}")
        print(f"   전년 대비 성장률: {annual_sales.pct_change().iloc[-1]:+.1%}")
        
        # 예측 요약
        forecast_data = self.dashboard_data['future_forecast']
        metrics = self.dashboard_data['business_metrics']
        
        print(f"\n🔮 **8주 예측 요약**")
        print(f"   예측 총 매출: {metrics['total_forecast_8weeks']}")
        print(f"   주간 평균: {metrics['forecast_avg_weekly_sales']}")
        print(f"   성장률: {metrics['growth_rate']}")
        print(f"   피크 예상: {metrics['peak_week_forecast']}")
        
        # 리스크 및 기회
        alerts = self.dashboard_data['alerts']
        high_priority_alerts = [a for a in alerts if a['priority'] == 'HIGH']
        
        print(f"\n⚠️ **주요 리스크 및 기회**")
        if high_priority_alerts:
            for alert in high_priority_alerts:
                print(f"   🔴 {alert['type']}: {alert['message']}")
        else:
            print(f"   ✅ 현재 고위험 알림 없음")
        
        # 권고사항
        print(f"\n💡 **권고사항**")
        print(f"   1. 예측 모델 정확도가 목표 달성 - 프로덕션 배포 권장")
        print(f"   2. 주간 모니터링으로 예측 성능 지속 관찰")
        print(f"   3. 계절성 패턴을 고려한 재고 계획 수립")
        print(f"   4. 외부 변수(프로모션, 경제지표) 통합 고려")
        
        return {
            'model_performance': f"{(1-best_model['mape']):.1%}",
            'annual_sales': f"${annual_sales.iloc[-1]:,.0f}",
            'forecast_summary': metrics,
            'alerts': alerts,
            'recommendations': [
                "프로덕션 배포 권장",
                "주간 모니터링 실시", 
                "계절성 기반 재고 계획",
                "외부 변수 통합 검토"
            ]
        }

# Store Sales 예측 시스템 실행
# Store Sales 예측 시스템 실행
print("\n🏪 Store Sales 종합 예측 시스템 구축")
print("=" * 60)

# 1. 시스템 초기화 및 데이터 생성
forecasting_system = StoreSalesForecastingSystem()
store_data = forecasting_system.simulate_store_sales_data()

# 2. 종합 EDA 분석
eda_results = forecasting_system.comprehensive_eda_analysis()

# 3. 종합 예측 모델 구축
model_results = forecasting_system.build_comprehensive_forecasting_models()

# 4. 비즈니스 대시보드 생성
dashboard = forecasting_system.create_business_dashboard()

# 5. 경영진 보고서 생성
executive_report = forecasting_system.generate_executive_report()

print(f"\n✅ Store Sales 예측 시스템 완전 구축!")
print(f"   📊 5년간 주간 매출 데이터 분석")
print(f"   🔮 5개 예측 모델 비교 (ARIMA, SARIMA, Holt-Winters, 앙상블, Naive)")
print(f"   📈 실시간 비즈니스 대시보드")
print(f"   🎯 자동 알림 및 의사결정 지원")
print(f"   💼 완전한 프로덕션 준비 시스템")
print(f"   📋 경영진 요약 보고서")

## 직접 해보기 / 연습 문제

### 🎯 연습 문제 1: ARIMA 모델 커스터마이징 (난이도: ⭐⭐⭐)

**문제**: 다음 요구사항에 맞는 ARIMA 모델을 구축하세요.

```python
# 다음 시계열 데이터를 활용하여:
custom_data = pd.Series([120, 125, 118, 130, 135, 128, 142, 138, 145, 152, 
                        148, 155, 162, 158, 165, 172, 168, 175, 182, 178])

# 요구사항:
# 1. 정상성 검정을 수행하고 필요시 차분 적용
# 2. ACF/PACF 분석으로 초기 차수 추정
# 3. 3개 이상의 ARIMA 모델 비교
# 4. 최적 모델로 5기간 예측
# 5. 예측 구간까지 포함한 시각화

# 힌트: Box-Jenkins 4단계 방법론 활용
```

**기대 결과**: 체계적인 ARIMA 모델링 과정과 신뢰도 높은 예측

### 🎯 연습 문제 2: 계절성 모델 심화 분석 (난이도: ⭐⭐⭐⭐)

**문제**: 월별 관광객 수 데이터로 SARIMA vs Holt-Winters 비교 분석하세요.

```python
# 시뮬레이션 데이터 생성
np.random.seed(123)
months = pd.date_range('2019-01-01', '2023-12-31', freq='M')
base_trend = np.linspace(10000, 15000, len(months))
seasonal_pattern = 3000 * np.sin(2 * np.pi * np.arange(len(months)) / 12)
tourism_data = base_trend + seasonal_pattern + np.random.normal(0, 500, len(months))

# 분석 과제:
# 1. 계절성 분해와 강도 분석
# 2. SARIMA(p,d,q)(P,D,Q)12 모델 구축
# 3. Holt-Winters 가법/승법 모델 비교
# 4. 교차검증 기반 성능 평가
# 5. 비즈니스 해석과 권고사항 도출
```

**평가 기준**: 모델 선택의 논리성, 해석의 정확성, 실무 적용성

### 🎯 연습 문제 3: AI 협업 모델 해석 시스템 (난이도: ⭐⭐⭐⭐⭐)

**문제**: 7장 프롬프트 엔지니어링을 활용한 자동 모델 해석 시스템을 구축하세요.

```python
# 구현 요구사항:
class AutoModelInterpreter:
    def __init__(self):
        self.interpretation_templates = {
            'arima': "ARIMA 모델 해석 템플릿",
            'sarima': "SARIMA 모델 해석 템플릿", 
            'exponential': "지수평활법 해석 템플릿"
        }
    
    def generate_interpretation(self, model_result, business_context):
        """
        모델 결과를 비즈니스 언어로 자동 번역
        
        입력: 모델 객체, 비즈니스 맥락
        출력: 경영진용 해석, 실무진용 해석, 액션 아이템
        """
        pass
    
    def create_business_narrative(self, forecast_results):
        """
        예측 결과를 스토리텔링 형태로 변환
        """
        pass

# 평가 요소:
# 1. CLEAR 프롬프트 원칙 적용
# 2. 다양한 모델 타입 지원
# 3. 청중별 맞춤 해석
# 4. 실행 가능한 권고사항 생성
```

**도전 과제**: GPT API 연동하여 실제 AI 해석 구현

### 🎯 연습 문제 4: 종합 예측 시스템 확장 (난이도: ⭐⭐⭐⭐⭐)

**문제**: Store Sales 시스템을 다음 기능으로 확장하세요.

1. **멀티 시계열 처리**: 상품군별 동시 예측
2. **외부 변수 통합**: 날씨, 경제지표, 이벤트 데이터
3. **실시간 모니터링**: 예측 성능 자동 추적
4. **A/B 테스트**: 모델 업데이트 효과 검증
5. **배포 파이프라인**: MLOps 관점의 시스템 설계

**구현 가이드라인**:
```python
# 확장 아키텍처 예시
class AdvancedForecastingSystem:
    def __init__(self):
        self.multi_series_manager = MultiSeriesManager()
        self.external_data_integrator = ExternalDataIntegrator()
        self.performance_monitor = RealTimeMonitor()
        self.ab_tester = ModelABTester()
        self.deployment_pipeline = MLOpsPipeline()
    
    def forecast_all_series(self, external_data=None):
        """모든 시계열 동시 예측"""
        pass
    
    def monitor_performance(self):
        """실시간 성능 모니터링"""
        pass
    
    def deploy_model_update(self, new_model):
        """안전한 모델 업데이트"""
        pass
```

## 요약 / 핵심 정리

### ✅ 8장 Part 2 학습 성과

**🔄 전통적 시계열 모델 완전 정복**
- **AR/MA/ARMA 모델**: 시계열의 기본 구성 요소와 수학적 원리 완전 이해
- **Box-Jenkins 방법론**: 체계적인 ARIMA 모델링 4단계 프로세스 마스터
- **정상성과 차분**: ADF/KPSS 검정과 적정 차분 차수 자동 결정
- **ACF/PACF 분석**: 모델 식별의 핵심 도구 활용법과 패턴 해석

**📊 고급 시계열 모델링 기법**
- **SARIMA 모델**: 계절성까지 고려한 고급 모델링과 자동 차수 선택
- **지수평활법**: 단순/이중/삼중 평활법과 Holt-Winters 모델 비교
- **성능 평가**: MAPE, MAE, RMSE 등 다양한 평가 지표와 교차검증
- **모델 진단**: Ljung-Box, Jarque-Bera 검정을 통한 모델 적합성 검증

**🤖 AI 협업 최적화**
- **프롬프트 엔지니어링**: 7장 CLEAR 원칙을 시계열 모델 해석에 특화 적용
- **자동 모델 튜닝**: 비즈니스 목표별 최적화 전략과 지능형 하이퍼파라미터 탐색
- **모델 해석**: 복잡한 통계 결과를 비즈니스 언어로 자동 번역
- **성능 모니터링**: 실시간 모델 성능 추적과 자동 알림 시스템

**🏪 실전 비즈니스 시스템**
- **Store Sales 예측**: 5년간 주간 데이터로 완전한 예측 시스템 구축
- **앙상블 모델**: 다중 모델 결합과 성능 기반 가중 평균
- **비즈니스 대시보드**: 실시간 예측 시각화와 KPI 모니터링
- **의사결정 지원**: 자동 알림과 권고사항으로 비즈니스 액션 가이드

### 🎯 실무 적용 능력

**📊 데이터 분석 전문성**
- 시계열 데이터의 특성을 정확히 파악하고 적절한 모델 선택
- 정상성, 계절성, 트렌드 등 핵심 개념의 실무적 활용
- 모델 진단과 검증을 통한 신뢰할 수 있는 결과 도출

**🔮 예측 모델링 역량**
- Box-Jenkins 방법론 기반 체계적 모델 구축
- 다양한 시계열 모델의 장단점 이해와 상황별 최적 선택
- 예측 불확실성 정량화와 비즈니스 리스크 평가

**💼 비즈니스 가치 창출**
- 통계 모델 결과를 경영진이 이해할 수 있는 언어로 번역
- 예측 정확도를 비즈니스 가치(매출, 비용절감 등)로 연결
- 데이터 기반 의사결정을 위한 실행 가능한 인사이트 제공

**🚀 시스템 구축 경험**
- 프로토타입부터 프로덕션까지 완전한 예측 시스템 설계
- 모델 모니터링과 성능 관리를 위한 운영 체계 구축
- AI 협업을 통한 효율적 개발과 지속적 개선 프로세스

### 🔗 다음 학습 경로

**📈 8장 Part 3: 머신러닝 기반 시계열 예측**
- 시계열 문제의 지도학습 변환과 특성 공학
- 랜덤 포레스트, XGBoost를 활용한 비선형 패턴 학습
- 외부 변수 통합과 다변량 시계열 모델링

**🧠 8장 Part 4: 딥러닝 시계열 모델**
- LSTM, GRU를 활용한 순환 신경망 시계열 예측
- Transformer와 Attention 메커니즘의 시계열 적용
- 시계열 생성 모델과 이상 탐지 기법

**💡 8장 Part 5: 고급 앙상블과 하이브리드 모델**
- 전통적 + ML + 딥러닝 모델의 최적 결합
- 동적 앙상블과 적응적 가중치 조정
- 불확실성 정량화와 확률적 예측

**🌐 8장 Part 6: 실시간 시계열 분석 시스템**
- 스트리밍 데이터 처리와 온라인 학습
- 분산 처리와 확장 가능한 아키텍처 설계
- MLOps 관점의 시계열 모델 운영과 관리

### 🎨 8장 Part 2 시각화 이미지 생성 프롬프트

"Traditional time series forecasting dashboard showing ARIMA, SARIMA, exponential smoothing models comparison with business metrics, sales forecast charts, performance indicators, professional analytics interface, modern business intelligence design, data science visualization, Box-Jenkins methodology workflow, seasonal decomposition plots, statistical diagnostics charts"

---

**🎓 8장 Part 2 완료!**

전통적 시계열 모델의 이론적 기초부터 실무 적용까지 완전히 마스터했습니다. ARIMA, SARIMA, 지수평활법의 핵심 원리를 이해하고, AI 협업을 통해 모델 해석과 최적화를 자동화하는 능력을 갖추었습니다. 

이제 Part 3에서 머신러닝의 힘을 빌려 시계열 예측의 새로운 지평을 열어보겠습니다! 🚀