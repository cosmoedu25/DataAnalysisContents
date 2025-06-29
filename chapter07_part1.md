# 7장 Part 1: 프롬프트 엔지니어링 기법
**부제: AI와 대화하는 기술 - 정확한 질문으로 정확한 답을 얻어내기**

## 학습 목표
이 Part를 완료한 후, 여러분은 다음을 할 수 있게 됩니다:
- 효과적인 프롬프트의 핵심 구성 요소를 이해하고 적용할 수 있다
- 데이터 분석 작업에 특화된 프롬프트 패턴을 활용할 수 있다
- 프롬프트를 체계적으로 개선하여 AI 도구의 성능을 극대화할 수 있다
- AI와의 협업 시 명확한 의도 전달과 품질 높은 결과 도출이 가능하다

## 이번 Part 미리보기
AI 시대의 데이터 분석가에게 가장 중요한 스킬 중 하나는 바로 **프롬프트 엔지니어링**입니다. 마치 숙련된 요리사가 재료의 특성을 알고 적절한 조리법을 선택하듯이, 효과적인 프롬프트는 AI 도구가 가진 잠재력을 최대한 끌어내는 열쇠입니다.

이번 Part에서는 AI와 소통하는 기술을 체계적으로 학습합니다. 단순히 질문을 던지는 것이 아니라, AI가 우리의 의도를 정확히 파악하고 유용한 결과를 생성할 수 있도록 돕는 전략적 소통 방법을 익히게 됩니다.

실제 데이터 분석 프로젝트에서 자주 마주치는 상황들을 바탕으로, 프롬프트 설계의 원칙부터 고급 기법까지 단계별로 학습합니다. SMS 스팸 탐지 프로젝트를 통해 실전에서 바로 활용할 수 있는 프롬프트 패턴들을 익혀보겠습니다.

---

> 📝 **중요 용어**: **프롬프트 엔지니어링(Prompt Engineering)**
> 
> AI 모델(특히 대규모 언어 모델)로부터 원하는 결과를 얻기 위해 입력 문장(프롬프트)을 체계적으로 설계하고 최적화하는 기법입니다. 마치 프로그래밍에서 함수를 호출할 때 올바른 매개변수를 전달하는 것처럼, AI에게 적절한 맥락과 지시사항을 제공하여 품질 높은 출력을 얻는 것이 목표입니다.

## 1. 효과적인 프롬프트 설계의 핵심 원칙

프롬프트 엔지니어링은 예술이면서 동시에 과학입니다. 창의적 사고와 체계적 접근이 모두 필요합니다. 효과적인 프롬프트를 작성하기 위한 **CLEAR** 원칙을 소개합니다.

### 1.1 CLEAR 원칙: 좋은 프롬프트의 5가지 핵심 요소

#### **C - Context (맥락 제공)**
AI에게 상황과 배경을 명확히 설명합니다. 마치 새로운 동료에게 업무를 설명할 때처럼, 필요한 배경 정보를 충분히 제공해야 합니다.

**❌ 나쁜 예시:**
```
데이터를 분석해줘
```

**✅ 좋은 예시:**
```
당신은 마케팅 팀의 데이터 분석가입니다. SMS 마케팅 캠페인의 효과를 평가하기 위해 
스팸 메시지와 정상 메시지를 구분하는 분류 모델을 개발하려고 합니다. 
5,574개의 SMS 메시지가 포함된 데이터셋이 있고, 각 메시지는 'spam' 또는 'ham'으로 
라벨링되어 있습니다.
```

#### **L - Length (적절한 길이)**
너무 짧으면 맥락이 부족하고, 너무 길면 핵심이 흐려집니다. 필요한 정보만 간결하게 포함시킵니다.

> 💡 **프롬프트 길이 가이드라인**
> - **간단한 작업**: 1-2문장 (예: 코드 설명 요청)
> - **일반적 분석**: 3-5문장 (예: 데이터 전처리 도움)
> - **복잡한 프로젝트**: 6-10문장 (예: 전체 워크플로우 설계)

#### **E - Examples (예시 제공)**
구체적인 예시는 AI가 우리의 의도를 정확히 파악하도록 돕는 가장 효과적인 방법입니다.

**Few-shot 프롬프팅 예시:**
```
다음과 같이 SMS 메시지를 분류해주세요:

예시 1:
입력: "Congratulations! You've won $1000. Call now!"
출력: spam (이유: 과장된 보상 제안 + 즉시 행동 요구)

예시 2:
입력: "Hey, are you free for lunch tomorrow?"
출력: ham (이유: 자연스러운 개인적 대화)

이제 다음 메시지를 분류해주세요:
입력: "FREE entry in 2 a wkly comp to win FA Cup final tkts"
```

#### **A - Actionable (실행 가능한 지시사항)**
AI가 구체적으로 무엇을 해야 하는지 명확하게 지시합니다. 추상적인 요청보다는 구체적인 행동을 요구합니다.

**❌ 모호한 지시:**
```
데이터를 더 좋게 만들어줘
```

**✅ 구체적인 지시:**
```
다음 작업을 순서대로 수행해주세요:
1. 결측치가 있는 행의 개수를 확인하고 보고하세요
2. 텍스트 데이터를 소문자로 변환하세요  
3. 구두점을 제거하세요
4. 처리 결과를 pandas DataFrame으로 반환하세요
```

#### **R - Role (역할 부여)**
AI에게 특정 전문가의 역할을 부여하면 더 전문적이고 맥락에 맞는 답변을 얻을 수 있습니다.

**역할 부여 예시들:**
- "당신은 10년 경력의 데이터 사이언티스트입니다"
- "파이썬 전문가의 관점에서 코드를 검토해주세요"
- "마케팅 분석가로서 비즈니스 인사이트를 도출해주세요"

### 1.2 프롬프트 구조화 템플릿

효과적인 프롬프트는 일관된 구조를 가져야 합니다. 다음 템플릿을 참고하세요:

```
[역할 설정] + [맥락/배경] + [구체적 작업] + [출력 형식] + [제약사항/조건]
```

**실제 적용 예시:**
```
당신은 경험 많은 데이터 분석가입니다. [역할]

SMS 스팸 탐지 프로젝트를 진행 중이며, 텍스트 전처리가 필요합니다. [맥락]

다음 SMS 메시지들에 대해 토큰화, 불용어 제거, 어간 추출을 수행해주세요. [작업]

결과는 Python 코드와 처리된 텍스트 리스트로 제공해주세요. [출력 형식]

NLTK 라이브러리를 사용하고, 영어 불용어를 기준으로 하세요. [제약사항]
```

> 🖼️ **이미지 생성 프롬프트**: 
> "CLEAR 원칙을 시각화한 인포그래픽. 가운데에 'CLEAR'가 큰 글자로 쓰여있고, 각 알파벳 주변에 Context(맥락), Length(길이), Examples(예시), Actionable(실행가능), Role(역할)이 아이콘과 함께 배치된 모던한 디자인. 색상은 파란색과 초록색 계열로 구성"

## 2. 데이터 분석 특화 프롬프트 패턴

데이터 분석 작업에는 고유한 특성이 있습니다. 이 섹션에서는 실제 분석 업무에서 자주 사용되는 프롬프트 패턴들을 소개합니다.

### 2.1 데이터 탐색 패턴 (EDA Pattern)

탐색적 데이터 분석을 위한 체계적 접근 방법입니다.

**기본 탐색 패턴:**
```
[데이터셋 정보] + [분석 목적] + [구체적 질문] + [출력 요구사항]
```

**실전 예시:**
```
SMS 스팸 데이터셋(5,574개 메시지, spam/ham 라벨)을 분석 중입니다.

목적: 스팸과 정상 메시지의 특성 차이를 파악하여 분류 모델 개발에 활용

다음을 분석해주세요:
1. 스팸/정상 메시지의 길이 분포 비교
2. 가장 빈번하게 사용되는 단어 Top 10 (각 클래스별)
3. 특수문자 사용 패턴의 차이점

출력: Python 코드 + 시각화 + 핵심 인사이트 3가지
```

**고급 탐색 패턴 (문제 상황 포함):**
```
상황: SMS 데이터에서 클래스 불균형 문제가 의심됩니다.

배경 지식: 일반적으로 스팸은 전체 메시지의 10-15% 정도입니다.

분석 요청:
1. 실제 클래스 분포 확인
2. 불균형이 심각하다면 해결 방안 3가지 제시
3. 각 방법의 장단점 비교 표 작성

주의사항: 비즈니스 관점에서 False Positive(정상을 스팸으로 분류)의 
위험성을 고려해주세요.
```

### 2.2 모델링 지원 패턴 (Modeling Assistant Pattern)

머신러닝 모델 개발 과정에서 AI의 도움을 받는 패턴입니다.

**모델 선택 패턴:**
```
문제 유형: [분류/회귀/군집화]
데이터 특성: [크기, 차원, 데이터 타입]
비즈니스 제약: [해석가능성, 속도, 정확도 우선순위]
현재 상황: [베이스라인 모델, 성능 기준]

요청: 적합한 알고리즘 3가지 추천 + 선택 이유 + 구현 코드 스케치
```

**예시:**
```
문제 유형: 텍스트 이진 분류 (SMS 스팸 탐지)
데이터 특성: 5,574개 샘플, 텍스트 데이터, 약간의 클래스 불균형
비즈니스 제약: 
- 해석가능성 중요 (왜 스팸으로 분류했는지 설명 필요)
- 실시간 처리 가능해야 함
- False Positive 최소화 우선

현재 상황: 단순 키워드 기반 규칙으로 70% 정확도

요청: 적합한 머신러닝 알고리즘 3가지 추천 + 
각각의 장단점 + 간단한 구현 가이드
```

### 2.3 코드 검토 및 개선 패턴 (Code Review Pattern)

AI에게 코드를 검토받고 개선 방안을 얻는 패턴입니다.

**기본 검토 패턴:**
```
역할: Python/데이터 분석 전문가
코드 목적: [구체적 기능 설명]
현재 문제: [성능, 가독성, 버그 등]

검토 요청:
1. 코드 품질 평가 (1-10점)
2. 개선 포인트 3가지
3. 최적화된 버전 제시

[코드 블록]
```

**실전 예시:**
```python
# 현재 코드 (검토 요청)
def preprocess_sms(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    return ' '.join(words)

# 프롬프트
역할: 파이썬 전문가 겸 NLP 엔지니어
코드 목적: SMS 텍스트 전처리 (스팸 분류 모델용)
현재 문제: 처리 속도가 느리고, 일부 유용한 정보가 손실됨

검토 요청:
1. 코드 효율성 및 기능성 평가
2. NLP 관점에서 개선점 제시
3. 성능 최적화 버전 작성

추가 고려사항: 
- 숫자와 특수문자도 스팸 탐지에 중요할 수 있음
- 5000개 이상 메시지 일괄 처리 필요
```

### 2.4 결과 해석 패턴 (Interpretation Pattern)

분석 결과나 모델 출력을 해석하고 비즈니스 인사이트를 도출하는 패턴입니다.

**기본 해석 패턴:**
```
역할: [도메인 전문가 + 데이터 분석가]
분석 결과: [수치, 표, 그래프 등]
비즈니스 맥락: [회사/프로젝트 배경]

해석 요청:
1. 핵심 발견사항 3가지
2. 비즈니스 임플리케이션
3. 다음 단계 액션 아이템
```

**실전 예시:**
```
역할: 마케팅 분석가 겸 데이터 사이언티스트

분석 결과: 
- 스팸 메시지 평균 길이: 138자
- 정상 메시지 평균 길이: 71자  
- 스팸에서 '!', '

, 'FREE' 등 특수문자/특정단어 빈도 3-5배 높음
- 숫자 포함 비율: 스팸 78%, 정상 23%

비즈니스 맥락: 
통신사의 SMS 필터링 시스템 개발
사용자 만족도가 핵심 KPI (잘못된 차단 시 고객 불만 급증)

해석 요청:
1. 이 결과가 시사하는 바 (스팸의 특징)
2. 필터링 규칙 설계시 고려사항
3. 성능 측정 지표 추천
4. 잠재적 위험요소와 대응방안
```

### 2.5 문제 해결 패턴 (Problem-Solving Pattern)

특정 문제 상황에 대한 솔루션을 요청하는 패턴입니다.

**구조화된 문제 해결 패턴:**
```
문제 상황: [구체적 오류/이슈 설명]
시도한 방법: [이미 해본 것들]
제약 조건: [시간, 자원, 기술적 제약]
목표: [달성하고자 하는 결과]

요청: 단계별 해결 방안 + 예상 시간 + 대안책
```

**예시:**
```
문제 상황: 
SMS 텍스트 분류 모델의 정확도가 85%에서 더 이상 개선되지 않음

시도한 방법:
- 다양한 알고리즘 테스트 (SVM, Random Forest, Naive Bayes)
- 하이퍼파라미터 튜닝
- 기본적인 텍스트 전처리

제약 조건:
- 추가 데이터 수집 불가
- 실시간 처리 필요 (응답시간 < 100ms)
- 기존 인프라 활용 (Python 기반)

목표: 90% 이상 정확도 달성

요청: 
1. 성능 개선 전략 5가지
2. 각 전략의 예상 효과와 구현 난이도
3. 우선순위 기준 추천
```

> 💡 **프롬프트 패턴 선택 가이드**
> 
> - **탐색 단계**: EDA 패턴 사용
> - **모델 개발**: 모델링 지원 패턴
> - **코드 작성 후**: 코드 검토 패턴  
> - **결과 분석**: 해석 패턴
> - **문제 발생**: 문제 해결 패턴

> 🖼️ **이미지 생성 프롬프트**: 
> "데이터 분석 워크플로우와 각 단계별 프롬프트 패턴을 보여주는 플로우차트. 시작부터 EDA → 모델링 → 검토 → 해석 → 문제해결 단계로 이어지며, 각 단계마다 해당하는 프롬프트 패턴 아이콘이 표시된 모던한 디자인"

## 3. 프롬프트 반복 개선 과정 (Iterative Refinement)

좋은 프롬프트는 한 번에 만들어지지 않습니다. 마치 레시피를 완성하기 위해 여러 번 시행착오를 거치는 것처럼, 프롬프트도 점진적으로 개선해야 합니다.

### 3.1 프롬프트 개선 사이클 (PDCA Approach)

**Plan (계획)** → **Do (실행)** → **Check (평가)** → **Act (개선)** 사이클을 적용합니다.

#### **1단계: Plan (초기 프롬프트 설계)**
목표를 명확히 하고 첫 번째 버전을 작성합니다.

```python
# 목표: SMS 텍스트 전처리 코드 생성 요청

# 초기 프롬프트 (버전 1.0)
initial_prompt = """
SMS 텍스트를 전처리하는 파이썬 코드를 작성해주세요.
"""
```

#### **2단계: Do (AI 실행 및 결과 확인)**
AI의 응답을 받고 결과를 분석합니다.

```
AI 응답 (버전 1.0 결과):
- 매우 기본적인 코드만 제공
- 구체적 요구사항 반영 부족
- SMS 도메인 특성 미고려
```

#### **3단계: Check (품질 평가)**
다음 기준으로 결과를 평가합니다:

**평가 기준 체크리스트:**
- [ ] **완전성**: 요청한 모든 기능이 포함되었는가?
- [ ] **정확성**: 코드가 올바르게 작동하는가?
- [ ] **효율성**: 성능이 적절한가?
- [ ] **맥락 적합성**: 도메인 특성이 반영되었는가?
- [ ] **실용성**: 실제로 사용 가능한가?

#### **4단계: Act (프롬프트 개선)**
평가 결과를 바탕으로 프롬프트를 개선합니다.

```python
# 개선된 프롬프트 (버전 2.0)
improved_prompt = """
당신은 NLP 전문가입니다.

SMS 스팸 분류 프로젝트용 텍스트 전처리 함수를 작성해주세요.

요구사항:
1. 소문자 변환, 특수문자 처리, 공백 정리
2. 5000개 메시지 일괄 처리 가능한 효율성
3. SMS 특성 고려 (짧은 텍스트, 축약어, 이모티콘)

출력: 
- 함수 코드
- 사용 예시
- 처리 시간 예상치

제약: pandas, nltk 라이브러리 사용
"""
```

### 3.2 실전 프롬프트 개선 사례: SMS 데이터 분석

실제 SMS 스팸 데이터를 분석하면서 프롬프트를 점진적으로 개선해보겠습니다.

#### **사례 1: 데이터 탐색 프롬프트 개선**

**버전 1.0 (너무 모호함):**
```
SMS 데이터를 분석해주세요.
```
*결과: 일반적인 분석만 제공, 구체성 부족*

**버전 2.0 (맥락 추가):**
```
SMS 스팸 탐지를 위해 5,574개 메시지 데이터를 분석해주세요. 
스팸과 정상 메시지의 차이점을 찾아주세요.
```
*결과: 더 나아졌지만 여전히 표면적*

**버전 3.0 (구체적 요구사항 추가):**
```
SMS 스팸 탐지 모델 개발을 위한 EDA를 수행해주세요.

데이터셋: 5,574개 SMS 메시지 (라벨: spam/ham)
목적: 효과적인 특성 추출을 위한 패턴 발견

분석 요청:
1. 클래스별 메시지 길이 분포 (통계량 + 시각화)
2. 스팸 특징적 단어/패턴 탐지 (TF-IDF 기반)
3. 특수문자, 숫자, 대문자 사용 패턴 비교
4. 시간 관련 표현 빈도 분석

출력 형식:
- Python 코드 (pandas, matplotlib 사용)
- 핵심 인사이트 3가지
- 모델링 제안사항

제약사항: 메모리 효율적 처리, 재현 가능한 결과
```
*결과: 매우 구체적이고 실용적인 분석 제공*

#### **사례 2: 코드 최적화 프롬프트 개선**

**문제 상황**: 텍스트 전처리 코드가 5000개 메시지 처리에 30초 소요

**버전 1.0:**
```python
# 현재 코드
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# 프롬프트
이 코드를 빠르게 만들어주세요.
```

**버전 2.0 (개선된 프롬프트):**
```python
역할: 파이썬 성능 최적화 전문가

현재 상황:
- SMS 텍스트 전처리 함수가 5000개 메시지 처리에 30초 소요
- 실시간 시스템에 적용 예정 (목표: < 5초)
- 메모리 사용량도 중요 (서버 RAM 제한)

현재 코드:
[코드 블록]

최적화 요청:
1. 처리 속도 5배 이상 향상
2. 메모리 효율성 개선
3. 코드 가독성 유지

제공해주세요:
1. 최적화된 코드 (주석 포함)
2. 성능 개선 원리 설명
3. 벤치마크 테스트 코드
4. 추가 개선 가능성

제약: numpy, pandas 외 라이브러리 추가 설치 불가
```

### 3.3 프롬프트 품질 측정 지표

프롬프트의 효과를 객관적으로 측정하기 위한 지표들입니다.

#### **정량적 지표**
```python
class PromptQualityMetrics:
    def __init__(self):
        self.metrics = {}
    
    def measure_response_quality(self, prompt, ai_response, expected_outcome):
        """프롬프트 품질을 다차원으로 측정"""
        
        # 1. 완전성 점수 (0-1)
        completeness = self.calculate_completeness(ai_response, expected_outcome)
        
        # 2. 정확성 점수 (0-1)  
        accuracy = self.calculate_accuracy(ai_response, expected_outcome)
        
        # 3. 실용성 점수 (0-1)
        usability = self.calculate_usability(ai_response)
        
        # 4. 응답 시간 (초)
        response_time = self.measure_response_time(prompt)
        
        # 5. 종합 점수
        overall_score = (completeness * 0.3 + 
                        accuracy * 0.4 + 
                        usability * 0.3)
        
        return {
            'completeness': completeness,
            'accuracy': accuracy, 
            'usability': usability,
            'response_time': response_time,
            'overall_score': overall_score
        }
```

#### **정성적 평가 체크리스트**

**프롬프트 자체 평가:**
- [ ] 명확한 목표 설정
- [ ] 충분한 맥락 제공  
- [ ] 구체적인 요구사항
- [ ] 적절한 제약조건
- [ ] 예상 출력 형식 명시

**AI 응답 평가:**
- [ ] 요청사항 완전 이행
- [ ] 기술적 정확성
- [ ] 실무 적용 가능성
- [ ] 추가 질문의 필요성
- [ ] 창의적 통찰 포함

### 3.4 고급 프롬프트 기법

#### **체인 오브 씽킹 (Chain of Thought)**
AI가 단계별로 사고하도록 유도하는 기법입니다.

```
SMS 스팸 분류 모델을 설계할 때, 다음과 같이 단계별로 사고해주세요:

1단계: 문제 정의
- 이진 분류 문제인가? 다중 분류 문제인가?
- 클래스 불균형 정도는?
- 비즈니스 제약사항은?

2단계: 데이터 특성 분석  
- 텍스트 길이 분포
- 어휘 다양성
- 도메인 특성

3단계: 알고리즘 선택
- 텍스트 분류에 적합한 알고리즘들
- 해석가능성 vs 성능 트레이드오프
- 실시간 처리 요구사항 고려

4단계: 평가 지표 선정
- 정확도만으로 충분한가?
- Precision vs Recall 중 무엇이 더 중요한가?

각 단계별로 근거와 함께 결론을 제시해주세요.
```

#### **역할 플레이 프롬프팅**
AI에게 특정 전문가 역할을 부여하여 더 전문적인 답변을 얻는 기법입니다.

```
상황: 당신은 10년 경력의 시니어 데이터 사이언티스트입니다. 
신입 개발자가 다음과 같이 질문했습니다.

신입 개발자: "SMS 스팸 분류에서 TF-IDF와 Word2Vec 중 어떤 것이 
더 좋을까요?"

시니어로서 다음을 포함하여 답변해주세요:
1. 각 방법의 장단점 설명
2. SMS 도메인 특성을 고려한 선택 기준
3. 실제 프로젝트 경험 기반 조언
4. 하이브리드 접근법 제안
5. 성능 측정 방법

신입이 이해하기 쉽게, 구체적인 예시와 함께 설명해주세요.
```

> 💡 **프롬프트 개선 팁**
> 
> 1. **점진적 개선**: 한 번에 모든 것을 바꾸지 말고 하나씩 개선
> 2. **A/B 테스트**: 두 가지 버전을 비교하여 더 나은 결과 선택
> 3. **피드백 루프**: AI 응답을 바탕으로 지속적으로 개선
> 4. **도메인 특화**: 일반적인 프롬프트보다 도메인에 특화된 프롬프트가 효과적
> 5. **문서화**: 효과적인 프롬프트는 팀과 공유하고 재사용

> 🖼️ **이미지 생성 프롬프트**: 
> "프롬프트 개선 사이클을 보여주는 순환 다이어그램. Plan → Do → Check → Act 4단계가 원형으로 배치되고, 각 단계마다 구체적인 작업 내용이 아이콘과 함께 표시된 모던한 인포그래픽 스타일"

## 4. 실전 미니 프로젝트: SMS 데이터 프롬프트 엔지니어링

이제 배운 내용을 SMS Spam Collection 데이터셋을 활용한 실전 프로젝트로 적용해보겠습니다.

### 4.1 프로젝트 준비

먼저 프로젝트 환경을 설정하고 데이터를 준비해보겠습니다.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (시각화용)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (10, 6)

# SMS 스팸 데이터 로드 (Kaggle에서 다운로드)
# URL: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
sms_data = pd.read_csv('spam.csv', encoding='latin-1')

# 필요한 컬럼만 선택하고 이름 변경
sms_data = sms_data[['v1', 'v2']].copy()
sms_data.columns = ['label', 'message']

print(f"데이터셋 크기: {sms_data.shape}")
print(f"클래스 분포:\n{sms_data['label'].value_counts()}")
print(f"\n샘플 데이터:")
print(sms_data.head())
```

**코드 해설:**
- `encoding='latin-1'`: SMS 데이터는 특수문자가 많아 인코딩 지정이 중요합니다
- 필요한 컬럼만 선택하여 메모리 효율성을 높입니다
- 데이터 분포를 확인하여 클래스 불균형 여부를 파악합니다

### 4.2 단계별 프롬프트 적용 실습

#### **실습 1: 효과적인 EDA 프롬프트 작성**

다음은 CLEAR 원칙을 적용한 EDA 프롬프트 예시입니다:

```python
# 프롬프트 설계 실습
eda_prompt = """
당신은 마케팅 데이터 분석 전문가입니다. [Role]

SMS 마케팅 캠페인의 스팸 필터링 시스템 개발을 위해 
5,574개의 SMS 메시지 데이터를 분석 중입니다. 
각 메시지는 'spam' 또는 'ham'(정상)으로 라벨링되어 있습니다. [Context]

다음 분석을 Python 코드로 수행해주세요: [Actionable]

1. 클래스별 메시지 길이 분포 비교 (통계량 + 박스플롯)
2. 스팸에서 자주 사용되는 단어 Top 10 추출  
3. 특수문자(!,$,%) 사용 빈도 클래스별 비교
4. 대문자 비율 분석 (스팸 vs 정상)

예상 출력: 각 분석마다 코드 + 시각화 + 핵심 인사이트 [Examples]

사용 라이브러리: pandas, matplotlib, seaborn, collections
출력 형식: 실행 가능한 코드 블록 + 해석 [Length - 적절한 상세도]
"""

print("EDA 프롬프트:")
print(eda_prompt)
```

**실제 분석 코드 (프롬프트 응답 예시):**

```python
# 1. 메시지 길이 분포 분석
sms_data['message_length'] = sms_data['message'].str.len()

# 클래스별 통계량
length_stats = sms_data.groupby('label')['message_length'].describe()
print("클래스별 메시지 길이 통계:")
print(length_stats)

# 시각화
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sms_data.boxplot(column='message_length', by='label', ax=plt.gca())
plt.title('클래스별 메시지 길이 분포')
plt.suptitle('')

plt.subplot(1, 2, 2)
for label in ['ham', 'spam']:
    data = sms_data[sms_data['label'] == label]['message_length']
    plt.hist(data, alpha=0.7, label=label, bins=30)
plt.xlabel('메시지 길이')
plt.ylabel('빈도')
plt.title('메시지 길이 히스토그램')
plt.legend()

plt.tight_layout()
plt.show()

# 핵심 인사이트
print("\n📊 핵심 인사이트:")
spam_avg = sms_data[sms_data['label'] == 'spam']['message_length'].mean()
ham_avg = sms_data[sms_data['label'] == 'ham']['message_length'].mean()
print(f"• 스팸 평균 길이: {spam_avg:.1f}자")
print(f"• 정상 평균 길이: {ham_avg:.1f}자") 
print(f"• 스팸이 정상 메시지보다 {spam_avg/ham_avg:.1f}배 김")
```

**코드 해설:**
- `str.len()`: 문자열 길이를 효율적으로 계산하는 pandas 메서드
- `groupby().describe()`: 그룹별 기술통계를 한 번에 계산
- `boxplot()`: 분포의 중심값, 사분위수, 이상치를 한눈에 파악
- 시각화를 통해 수치 데이터를 직관적으로 이해할 수 있게 함

#### **실습 2: 특성 추출 프롬프트 개선**

초기 프롬프트와 개선된 프롬프트를 비교해보겠습니다:

```python
# 초기 프롬프트 (너무 간단함)
basic_prompt = "SMS 텍스트에서 특성을 추출하는 코드를 작성해주세요."

# 개선된 프롬프트
advanced_prompt = """
역할: NLP 및 피처 엔지니어링 전문가

배경: SMS 스팸 분류 모델을 위한 특성 추출 함수 개발
목표: 스팸과 정상 메시지를 구분하는 효과적인 특성 생성

요구사항:
1. 기본 텍스트 특성 (길이, 단어 수, 문장 수)
2. 특수문자 관련 특성 (!, $, % 개수 및 비율)  
3. 대소문자 관련 특성 (대문자 비율)
4. 숫자 관련 특성 (숫자 포함 여부, 전화번호 패턴)
5. 키워드 기반 특성 (FREE, WIN, CALL 등 스팸 단어)

출력:
- feature_extraction(text) 함수
- 각 특성의 의미와 중요성 설명
- 5개 샘플 메시지에 대한 특성 추출 결과 예시

제약사항:
- pandas, numpy, re 라이브러리만 사용
- 5000개 메시지 일괄 처리 가능한 효율성
- 특성별 정규화/스케일링 고려
"""

print("개선된 특성 추출 프롬프트:")
print(advanced_prompt)
```

**프롬프트 응답 예시:**

```python
import re
import numpy as np

def extract_sms_features(text):
    """
    SMS 메시지에서 스팸 분류를 위한 특성을 추출합니다.
    
    Args:
        text (str): SMS 메시지 텍스트
        
    Returns:
        dict: 추출된 특성들의 딕셔너리
    """
    
    # 기본 텍스트 특성
    char_count = len(text)
    word_count = len(text.split())
    sentence_count = len(re.findall(r'[.!?]+', text))
    avg_word_length = np.mean([len(word) for word in text.split()]) if word_count > 0 else 0
    
    # 특수문자 특성 (스팸에서 자주 사용)
    exclamation_count = text.count('!')
    dollar_count = text.count('

)
    percent_count = text.count('%')
    special_char_ratio = (exclamation_count + dollar_count + percent_count) / char_count if char_count > 0 else 0
    
    # 대소문자 특성
    upper_count = sum(1 for c in text if c.isupper())
    upper_ratio = upper_count / char_count if char_count > 0 else 0
    
    # 숫자 관련 특성
    digit_count = sum(1 for c in text if c.isdigit())
    digit_ratio = digit_count / char_count if char_count > 0 else 0
    has_phone = bool(re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text))
    
    # 스팸 키워드 특성
    spam_keywords = ['free', 'win', 'winner', 'cash', 'prize', 'urgent', 'call', 'claim']
    text_lower = text.lower()
    spam_keyword_count = sum(1 for keyword in spam_keywords if keyword in text_lower)
    
    # URL/링크 특성  
    has_url = bool(re.search(r'http[s]?://|www\.', text.lower()))
    
    return {
        'char_count': char_count,
        'word_count': word_count, 
        'sentence_count': sentence_count,
        'avg_word_length': avg_word_length,
        'exclamation_count': exclamation_count,
        'special_char_ratio': special_char_ratio,
        'upper_ratio': upper_ratio,
        'digit_ratio': digit_ratio,
        'has_phone': int(has_phone),
        'spam_keyword_count': spam_keyword_count,
        'has_url': int(has_url)
    }

# 테스트 및 예시
sample_messages = [
    "FREE entry in 2 a wkly comp to win FA Cup final tkts",
    "Hey, are you free for lunch tomorrow?", 
    "URGENT! Call 123-456-7890 to claim your $1000 prize NOW!",
    "Thanks for your message. I'll get back to you soon.",
    "Win £100 cash! Text WIN to 88888"
]

print("특성 추출 결과 예시:")
for i, msg in enumerate(sample_messages, 1):
    features = extract_sms_features(msg)
    print(f"\n메시지 {i}: {msg[:50]}...")
    print(f"특성: {features}")
```

**코드 해설:**
- **효율성**: 벡터화된 연산 대신 문자열 메서드를 사용하여 가독성과 성능 균형
- **도메인 특성**: SMS/스팸 도메인의 특성을 반영한 특성 설계 (전화번호, 스팸 키워드 등)
- **정규화**: 길이에 독립적인 비율 특성 사용으로 다양한 메시지 길이에 대응
- **확장성**: 새로운 특성을 쉽게 추가할 수 있는 구조 설계

### 4.3 프롬프트 품질 평가 실습

작성한 프롬프트의 품질을 평가해보겠습니다:

```python
class PromptEvaluator:
    """프롬프트 품질을 평가하는 클래스"""
    
    def __init__(self):
        self.evaluation_criteria = {
            'clarity': '명확성 (이해하기 쉬운가?)',
            'completeness': '완전성 (필요한 정보가 모두 포함되었는가?)', 
            'specificity': '구체성 (구체적인 요구사항이 있는가?)',
            'actionability': '실행가능성 (AI가 실행하기 쉬운가?)',
            'context': '맥락성 (충분한 배경 정보가 있는가?)'
        }
    
    def evaluate_prompt(self, prompt, max_score=5):
        """프롬프트를 다차원으로 평가합니다."""
        
        print("📋 프롬프트 품질 평가")
        print("=" * 50)
        print(f"평가 대상 프롬프트:\n{prompt[:100]}...\n")
        
        scores = {}
        total_score = 0
        
        for criterion, description in self.evaluation_criteria.items():
            print(f"\n🎯 {description}")
            
            # 간단한 휴리스틱 기반 평가 (실제로는 더 정교한 평가 필요)
            if criterion == 'clarity':
                # 문장 길이와 복잡성 확인
                avg_sentence_length = len(prompt.split()) / len(prompt.split('.'))
                score = min(5, max(1, 6 - avg_sentence_length/10))
                
            elif criterion == 'completeness':
                # 핵심 요소 포함 여부 확인
                key_elements = ['역할', '맥락', '요구사항', '출력', '제약']
                found_elements = sum(1 for element in key_elements 
                                   if any(keyword in prompt.lower() 
                                         for keyword in [element, element.lower()]))
                score = (found_elements / len(key_elements)) * 5
                
            elif criterion == 'specificity':
                # 구체적 숫자나 예시 포함 여부
                specific_indicators = re.findall(r'\d+|예시|example|구체적', prompt.lower())
                score = min(5, len(specific_indicators) * 0.5 + 2)
                
            elif criterion == 'actionability':
                # 동작 동사 포함 여부
                action_verbs = ['분석', '작성', '생성', '계산', '시각화', '비교']
                found_verbs = sum(1 for verb in action_verbs if verb in prompt)
                score = min(5, found_verbs * 0.8 + 1)
                
            elif criterion == 'context':
                # 배경 정보 풍부함
                context_words = ['데이터셋', '프로젝트', '목표', '상황', '배경']
                found_context = sum(1 for word in context_words if word in prompt)
                score = min(5, found_context * 1.0 + 1)
            
            scores[criterion] = round(score, 1)
            total_score += score
            
            print(f"점수: {score:.1f}/5.0")
            
        average_score = total_score / len(self.evaluation_criteria)
        
        print(f"\n📊 종합 평가")
        print("=" * 30)
        print(f"평균 점수: {average_score:.1f}/5.0")
        
        if average_score >= 4.0:
            print("🎉 우수한 프롬프트입니다!")
        elif average_score >= 3.0:
            print("👍 양호한 프롬프트입니다. 일부 개선 여지가 있습니다.")
        else:
            print("⚠️ 개선이 필요한 프롬프트입니다.")
            
        return scores, average_score

# 평가 실습
evaluator = PromptEvaluator()

# 앞서 작성한 고급 프롬프트 평가
scores, avg_score = evaluator.evaluate_prompt(advanced_prompt)
```

**코드 해설:**
- **다차원 평가**: 단일 지표가 아닌 여러 관점에서 종합적 평가
- **객관적 지표**: 휴리스틱을 사용하여 주관적 평가를 최대한 객관화
- **개선 방향 제시**: 평가 결과를 바탕으로 구체적인 개선 방향 제시

## 5. 직접 해보기: 연습 문제

다음 연습 문제들을 통해 프롬프트 엔지니어링 기술을 연마해보세요.

### **연습 문제 1: 기본 프롬프트 개선하기**

다음 기본 프롬프트를 CLEAR 원칙에 따라 개선해보세요:

```
기본 프롬프트: "머신러닝 모델을 만들어주세요."

개선 과제:
1. Role, Context, Examples, Actionable, Length를 모두 포함하여 개선
2. SMS 스팸 분류 문제에 특화
3. 구체적인 성능 목표와 제약사항 포함
4. 예상 출력 형식 명시

작성 공간:
[여기에 개선된 프롬프트를 작성하세요]
```

### **연습 문제 2: 문제 해결 프롬프트 작성**

다음 상황에 대한 효과적인 문제 해결 프롬프트를 작성하세요:

```
상황: SMS 분류 모델의 정밀도(Precision)는 높지만 재현율(Recall)이 낮습니다.
현재 성능: Precision 0.95, Recall 0.65, F1-Score 0.77
목표: F1-Score 0.85 이상 달성

과제: 이 문제를 해결하기 위한 체계적인 프롬프트를 작성하세요.
포함 요소: 문제 진단, 해결 방안, 구현 코드, 성능 검증 방법

작성 공간:
[여기에 문제 해결 프롬프트를 작성하세요]
```

### **연습 문제 3: 프롬프트 A/B 테스트**

동일한 작업에 대해 두 가지 다른 접근법의 프롬프트를 작성하고 비교하세요:

```
작업: SMS 텍스트에서 감정 분석 수행

프롬프트 A (규칙 기반 접근):
[여기에 규칙 기반 감정 분석 프롬프트 작성]

프롬프트 B (머신러닝 기반 접근):
[여기에 ML 기반 감정 분석 프롬프트 작성]

비교 기준:
- 구현 복잡도
- 정확도 예상치
- 해석 가능성
- 확장성

예상 결과 비교표:
| 기준 | 프롬프트 A | 프롬프트 B |
|------|------------|------------|
| 복잡도 | | |
| 정확도 | | |
| 해석성 | | |
| 확장성 | | |
```

### **연습 문제 4: 창의적 프롬프트 설계**

다음 도전적인 과제를 위한 창의적 프롬프트를 설계하세요:

```
도전 과제: SMS 메시지의 "긴급도"를 자동으로 측정하는 시스템 개발

요구사항:
- 1(낮음) ~ 5(매우 높음) 척도로 긴급도 점수 부여
- 단순 키워드 기반이 아닌 맥락 고려 필요
- 개인적 메시지와 비즈니스 메시지 구분 필요

창의적 요소:
- 언어학적 특성 활용 (어조, 문체 등)
- 시간/날짜 정보 고려
- 발신자 정보 활용 방안

작성 공간:
[여기에 창의적인 긴급도 측정 프롬프트를 작성하세요]
```

## 6. 핵심 정리 및 요약

### 🎯 **이번 Part에서 배운 핵심 내용**

1. **CLEAR 원칙**: 효과적인 프롬프트의 5가지 핵심 요소
   - **Context**: 충분한 맥락과 배경 정보 제공
   - **Length**: 적절한 길이로 핵심만 간결하게
   - **Examples**: 구체적인 예시로 의도 명확화
   - **Actionable**: 실행 가능한 구체적 지시사항
   - **Role**: 적절한 전문가 역할 부여

2. **데이터 분석 특화 패턴**: 실무에 바로 적용 가능한 5가지 패턴
   - EDA 패턴, 모델링 지원 패턴, 코드 검토 패턴
   - 결과 해석 패턴, 문제 해결 패턴

3. **반복 개선 프로세스**: PDCA 사이클을 통한 체계적 개선
   - Plan → Do → Check → Act의 순환 구조
   - 정량적/정성적 평가 기준 활용

4. **고급 기법**: Chain of Thought, 역할 플레이 등 전문 기법

### 💡 **실무 적용 가이드라인**

- 프로젝트 초기에는 **EDA 패턴**으로 데이터 이해
- 모델 개발 시에는 **모델링 지원 패턴** 활용
- 코드 작성 후에는 **코드 검토 패턴**으로 품질 확보
- 결과 분석 시에는 **해석 패턴**으로 비즈니스 가치 도출
- 문제 발생 시에는 **문제 해결 패턴**으로 체계적 접근

### 🔄 **지속적 개선 전략**

1. **팀 프롬프트 라이브러리 구축**: 효과적인 프롬프트 공유
2. **A/B 테스트**: 다양한 접근법 비교 실험
3. **피드백 루프**: AI 응답 품질 지속 모니터링
4. **도메인 특화**: 프로젝트별 맞춤형 프롬프트 개발

## 7. 다음 Part 예고: AI 생성 코드 검증 및 최적화

다음 Part에서는 AI가 생성한 코드를 어떻게 검증하고 최적화할지 배워보겠습니다.

**다음 Part 주요 내용:**
- AI 생성 코드의 일반적 오류 패턴과 탐지 방법
- 코드 품질 평가 기준과 자동화 도구
- 성능 최적화 전략과 리팩토링 기법
- 보안과 안정성을 고려한 코드 검증 프로세스

**미리 생각해볼 질문:**
- AI가 생성한 코드를 그대로 사용해도 안전할까요?
- 어떤 부분을 중점적으로 검토해야 할까요?
- 코드의 품질을 객관적으로 측정하는 방법은 무엇일까요?

> 🖼️ **이미지 생성 프롬프트**: 
> "프롬프트 엔지니어링의 학습 여정을 보여주는 로드맵 이미지. CLEAR 원칙부터 시작하여 패턴 적용, 반복 개선, 실전 프로젝트까지의 단계가 연결된 길로 표현되고, 각 단계마다 성취 뱃지가 있는 게이미피케이션 스타일"

---

**🎉 축하합니다! 7장 Part 1을 완료했습니다!**

프롬프트 엔지니어링은 AI 시대 데이터 분석가의 핵심 역량입니다. 이번 Part에서 배운 CLEAR 원칙과 패턴들을 실제 프로젝트에 적용하며 지속적으로 연습해보세요. 좋은 프롬프트는 하루아침에 만들어지지 않습니다. 꾸준한 실습과 개선을 통해 AI와의 협업 능력을 키워나가시길 바랍니다!

