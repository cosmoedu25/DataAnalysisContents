# 7장 Part 2: AI 생성 코드 검증 및 최적화
**부제: AI 코드를 믿어도 될까? - 체계적인 검증과 개선 전략**

## 학습 목표
이 Part를 완료한 후, 여러분은 다음을 할 수 있게 됩니다:
- AI가 생성한 코드의 일반적인 오류 패턴을 식별하고 탐지할 수 있다
- 코드 품질을 객관적으로 평가하는 기준과 도구를 활용할 수 있다
- 성능, 가독성, 안정성 관점에서 코드를 체계적으로 개선할 수 있다
- 보안과 안정성을 고려한 코드 검증 프로세스를 수립할 수 있다

## 이번 Part 미리보기
AI가 생성한 코드를 그대로 사용하는 것은 마치 자동차의 자율주행 기능을 맹신하는 것과 같습니다. 대부분의 경우 잘 작동하지만, 예상치 못한 상황에서는 문제가 발생할 수 있습니다. 

똑똑한 운전자가 자율주행 중에도 항상 주의를 기울이듯이, 데이터 분석가도 AI가 생성한 코드를 비판적으로 검토하고 개선할 수 있어야 합니다.

이번 Part에서는 AI 생성 코드의 숨겨진 함정들을 발견하고, 체계적인 검증 프로세스를 통해 안정적이고 효율적인 코드로 개선하는 방법을 학습합니다. SMS 스팸 탐지 프로젝트를 계속 활용하여 실제 발생할 수 있는 다양한 코드 품질 이슈들을 직접 경험해보겠습니다.

---

> 📝 **중요 용어**: **코드 검증(Code Verification)**
> 
> 작성된 코드가 의도한 대로 동작하는지, 효율적이고 안전한지를 체계적으로 확인하는 과정입니다. 기능성, 성능, 가독성, 보안, 유지보수성 등 다양한 관점에서 코드를 평가하고 개선점을 찾아내는 것이 목표입니다.

## 1. AI 생성 코드의 일반적 오류 패턴

AI는 매우 똑똑하지만 완벽하지는 않습니다. 마치 재능 있는 신입 개발자처럼, 기본적인 코딩은 잘하지만 세부적인 부분에서 실수를 할 수 있습니다. 이 섹션에서는 AI가 자주 범하는 오류 패턴들을 살펴보겠습니다.

### 1.1 기능적 오류 (Functional Errors)

#### **오류 유형 1: 엣지 케이스 처리 부족**

AI는 일반적인 상황에 대해서는 잘 작동하는 코드를 생성하지만, 극단적이거나 예외적인 상황(엣지 케이스)을 놓치는 경우가 많습니다.

**❌ AI 생성 코드 예시 (문제가 있는 버전):**
```python
def calculate_sms_statistics(messages):
    """SMS 메시지 통계를 계산하는 함수"""
    total_chars = sum(len(msg) for msg in messages)
    avg_length = total_chars / len(messages)
    
    word_counts = [len(msg.split()) for msg in messages]
    avg_words = sum(word_counts) / len(word_counts)
    
    return {
        'total_messages': len(messages),
        'avg_char_length': avg_length,
        'avg_word_count': avg_words
    }

# 테스트
messages = ["Hello world", "How are you?", "FREE money now!!!"]
stats = calculate_sms_statistics(messages)
print(stats)
```

**🚨 문제점 분석:**
1. **빈 리스트 처리 안됨**: `messages`가 빈 리스트면 `ZeroDivisionError` 발생
2. **None 값 처리 안됨**: 메시지 중 `None`이 있으면 `TypeError` 발생
3. **빈 문자열 처리 미흡**: 빈 메시지가 있어도 에러는 없지만 부정확한 통계

**✅ 개선된 버전:**
```python
def calculate_sms_statistics(messages):
    """
    SMS 메시지 통계를 계산하는 함수 (엣지 케이스 처리 포함)
    
    Args:
        messages (list): SMS 메시지 리스트
        
    Returns:
        dict: 통계 정보 딕셔너리
        
    Raises:
        ValueError: 입력이 유효하지 않은 경우
    """
    
    # 입력 검증
    if not isinstance(messages, list):
        raise ValueError("messages는 리스트여야 합니다")
    
    if len(messages) == 0:
        return {
            'total_messages': 0,
            'avg_char_length': 0,
            'avg_word_count': 0,
            'warning': 'Empty message list'
        }
    
    # None 값과 비문자열 제거
    valid_messages = []
    for msg in messages:
        if msg is not None and isinstance(msg, str):
            valid_messages.append(msg)
    
    if len(valid_messages) == 0:
        return {
            'total_messages': len(messages),
            'valid_messages': 0,
            'avg_char_length': 0,
            'avg_word_count': 0,
            'warning': 'No valid string messages found'
        }
    
    # 통계 계산
    total_chars = sum(len(msg) for msg in valid_messages)
    avg_length = total_chars / len(valid_messages)
    
    word_counts = [len(msg.split()) for msg in valid_messages if len(msg.strip()) > 0]
    avg_words = sum(word_counts) / len(word_counts) if word_counts else 0
    
    return {
        'total_messages': len(messages),
        'valid_messages': len(valid_messages),
        'avg_char_length': round(avg_length, 2),
        'avg_word_count': round(avg_words, 2)
    }

# 다양한 엣지 케이스 테스트
test_cases = [
    [],  # 빈 리스트
    [None, "", "Hello"],  # None과 빈 문자열 포함
    ["", "   ", "\n"],  # 공백만 있는 메시지들
    ["Normal message", "Another one"]  # 정상 케이스
]

for i, test_case in enumerate(test_cases, 1):
    try:
        result = calculate_sms_statistics(test_case)
        print(f"테스트 {i}: {result}")
    except Exception as e:
        print(f"테스트 {i} 에러: {e}")
```

**코드 해설:**
- `isinstance()` 체크로 타입 안전성 확보
- 단계별 검증으로 각 엣지 케이스 대응
- 의미 있는 에러 메시지와 경고 정보 제공
- `round()` 함수로 가독성 있는 소수점 표시

#### **오류 유형 2: 데이터 타입 혼동**

AI는 때때로 pandas DataFrame과 리스트, 또는 numpy 배열과 일반 리스트를 혼동하여 잘못된 메서드를 사용하는 경우가 있습니다.

**❌ 문제가 있는 AI 코드:**
```python
def preprocess_sms_data(data):
    """SMS 데이터 전처리 (잘못된 버전)"""
    
    # 데이터 타입을 고려하지 않은 처리
    cleaned_messages = []
    for message in data:
        # DataFrame인 경우와 리스트인 경우를 구분하지 않음
        clean_msg = message.lower().strip()  # AttributeError 위험!
        cleaned_messages.append(clean_msg)
    
    return cleaned_messages
```

**✅ 개선된 버전:**
```python
import pandas as pd
import numpy as np

def preprocess_sms_data(data, text_column='message'):
    """
    SMS 데이터 전처리 (타입 안전 버전)
    
    Args:
        data: pandas DataFrame, list, numpy array, 또는 단일 문자열
        text_column: DataFrame인 경우 텍스트가 있는 컬럼명
        
    Returns:
        list: 전처리된 메시지 리스트
    """
    
    cleaned_messages = []
    
    # 데이터 타입별 처리
    if isinstance(data, pd.DataFrame):
        if text_column not in data.columns:
            raise ValueError(f"컬럼 '{text_column}'가 DataFrame에 없습니다")
        messages = data[text_column].fillna('').astype(str)
        
    elif isinstance(data, (list, np.ndarray)):
        messages = [str(msg) if msg is not None else '' for msg in data]
        
    elif isinstance(data, str):
        messages = [data]
        
    else:
        raise ValueError(f"지원하지 않는 데이터 타입: {type(data)}")
    
    # 안전한 전처리
    for message in messages:
        try:
            clean_msg = str(message).lower().strip()
            # 빈 문자열이 아닌 경우만 추가
            if clean_msg:
                cleaned_messages.append(clean_msg)
        except Exception as e:
            print(f"메시지 처리 중 오류: {message} -> {e}")
            continue
    
    return cleaned_messages

# 다양한 타입 테스트
import pandas as pd

# DataFrame 테스트
df_data = pd.DataFrame({'message': ['Hello', 'WORLD', '  spaces  '], 'label': [0, 1, 0]})
result1 = preprocess_sms_data(df_data, 'message')
print(f"DataFrame 결과: {result1}")

# 리스트 테스트  
list_data = ['Hello', 'WORLD', None, '  spaces  ']
result2 = preprocess_sms_data(list_data)
print(f"리스트 결과: {result2}")

# 단일 문자열 테스트
single_data = "Single Message"
result3 = preprocess_sms_data(single_data)
print(f"단일 문자열 결과: {result3}")
```

**코드 해설:**
- `isinstance()` 체크로 입력 데이터 타입 확인
- 타입별 적절한 처리 로직 분리
- 예외 상황 처리와 의미 있는 에러 메시지
- `fillna()`, `astype()` 등 pandas 메서드의 안전한 사용

### 1.2 성능 관련 오류

#### **오류 유형 3: 비효율적인 반복문**

AI는 종종 간단하고 직관적인 코드를 생성하지만, 성능 최적화는 놓치는 경우가 많습니다.

**❌ 비효율적인 AI 코드:**
```python
def extract_spam_keywords(messages, spam_keywords):
    """스팸 키워드 추출 (비효율적 버전)"""
    
    results = []
    for message in messages:
        keyword_count = 0
        found_keywords = []
        
        # 이중 반복문 - O(n*m) 복잡도
        for keyword in spam_keywords:
            if keyword.lower() in message.lower():
                keyword_count += 1
                found_keywords.append(keyword)
        
        results.append({
            'message': message,
            'keyword_count': keyword_count,
            'found_keywords': found_keywords
        })
    
    return results

# 테스트 (성능 측정)
import time

spam_keywords = ['free', 'win', 'cash', 'prize', 'urgent', 'limited', 'offer', 'click', 'call', 'now']
sample_messages = ["FREE cash prize! Call NOW!", "Hello, how are you?"] * 1000  # 큰 데이터 시뮬레이션

start_time = time.time()
result = extract_spam_keywords(sample_messages, spam_keywords)
end_time = time.time()

print(f"비효율적 버전 실행 시간: {end_time - start_time:.4f}초")
```

**✅ 최적화된 버전:**
```python
import re
from collections import defaultdict

def extract_spam_keywords_optimized(messages, spam_keywords):
    """
    스팸 키워드 추출 (최적화 버전)
    
    정규표현식과 집합 연산을 활용한 성능 개선
    """
    
    # 키워드를 정규표현식 패턴으로 컴파일 (한 번만 수행)
    keyword_pattern = re.compile(
        r'\b(' + '|'.join(re.escape(keyword.lower()) for keyword in spam_keywords) + r')\b',
        re.IGNORECASE
    )
    
    results = []
    
    for message in messages:
        # 정규표현식으로 한 번에 모든 키워드 찾기
        found_matches = keyword_pattern.findall(message.lower())
        
        # 중복 제거 및 카운트
        unique_keywords = list(set(found_matches))
        keyword_count = len(found_matches)
        
        results.append({
            'message': message,
            'keyword_count': keyword_count,
            'found_keywords': unique_keywords
        })
    
    return results

# 더 나은 최적화 버전 (벡터화)
import pandas as pd

def extract_spam_keywords_vectorized(messages, spam_keywords):
    """
    벡터화된 스팸 키워드 추출 (pandas 활용)
    
    대용량 데이터에 최적화된 버전
    """
    
    df = pd.DataFrame({'message': messages})
    df['message_lower'] = df['message'].str.lower()
    
    # 각 키워드별 포함 여부 체크 (벡터화)
    keyword_columns = {}
    for keyword in spam_keywords:
        col_name = f'has_{keyword}'
        keyword_columns[col_name] = df['message_lower'].str.contains(
            r'\b' + re.escape(keyword.lower()) + r'\b', 
            regex=True, 
            na=False
        )
    
    keyword_df = pd.DataFrame(keyword_columns)
    
    # 통계 계산
    df['keyword_count'] = keyword_df.sum(axis=1)
    
    # 찾은 키워드 리스트 생성
    def get_found_keywords(row_idx):
        found = []
        for i, keyword in enumerate(spam_keywords):
            col_name = f'has_{keyword}'
            if keyword_columns[col_name].iloc[row_idx]:
                found.append(keyword)
        return found
    
    df['found_keywords'] = [get_found_keywords(i) for i in range(len(df))]
    
    return df[['message', 'keyword_count', 'found_keywords']].to_dict('records')

# 성능 비교
start_time = time.time()
result_optimized = extract_spam_keywords_optimized(sample_messages, spam_keywords)
end_time = time.time()
print(f"최적화 버전 실행 시간: {end_time - start_time:.4f}초")

start_time = time.time()
result_vectorized = extract_spam_keywords_vectorized(sample_messages, spam_keywords)
end_time = time.time()
print(f"벡터화 버전 실행 시간: {end_time - start_time:.4f}초")

# 결과 확인 (첫 3개만)
print("\n결과 예시:")
for i in range(3):
    print(f"메시지 {i+1}: {result_optimized[i]}")
```

**코드 해설:**
- **정규표현식 컴파일**: 패턴을 한 번만 컴파일하여 반복 사용
- **벡터화 연산**: pandas의 벡터화된 문자열 연산 활용
- **메모리 효율성**: 불필요한 중간 결과 저장 최소화
- **시간 복잡도 개선**: O(n*m)에서 O(n)으로 개선

### 1.3 로직 오류 (Logic Errors)

#### **오류 유형 4: 잘못된 가정**

AI는 문제를 단순화하여 해결하려는 경향이 있어, 실제 데이터의 복잡성을 간과하는 경우가 있습니다.

**❌ 잘못된 가정을 포함한 코드:**
```python
def classify_sms_simple(message):
    """
    간단한 SMS 분류기 (잘못된 가정 포함)
    
    가정: 스팸 키워드가 있으면 무조건 스팸, 없으면 정상
    """
    
    spam_keywords = ['free', 'win', 'money', 'prize', 'call']
    
    message_lower = message.lower()
    
    # 너무 단순한 규칙
    for keyword in spam_keywords:
        if keyword in message_lower:
            return 'spam'
    
    return 'ham'

# 문제가 되는 테스트 케이스들
test_messages = [
    "I won the lottery!",  # 'win' 때문에 스팸으로 잘못 분류
    "Call me when you're free",  # 'call', 'free' 때문에 스팸으로 잘못 분류  
    "The money is in the bank",  # 'money' 때문에 스팸으로 잘못 분류
    "Win the game tonight!",  # 'win' 때문에 스팸으로 잘못 분류
    "FREE unlimited data plan"  # 실제 스팸이지만 너무 단순한 규칙
]

print("간단한 분류기 결과:")
for msg in test_messages:
    result = classify_sms_simple(msg)
    print(f"'{msg}' -> {result}")
```

**✅ 개선된 버전 (맥락 고려):**
```python
import re
from collections import defaultdict

def classify_sms_advanced(message):
    """
    고급 SMS 분류기 (맥락과 패턴 고려)
    
    여러 신호를 종합적으로 고려한 분류
    """
    
    # 가중치가 있는 스팸 신호들
    spam_indicators = {
        'urgent_words': {
            'words': ['urgent', 'immediate', 'expires', 'limited time', 'act now'],
            'weight': 3
        },
        'money_related': {
            'words': ['free', 'cash', 'money', 'prize', '

, '£', '€'],
            'weight': 2
        },
        'action_words': {
            'words': ['call now', 'click here', 'text back', 'reply stop'],
            'weight': 2
        },
        'suspicious_patterns': {
            'patterns': [
                r'\b(win|won)\s+[$£€]\d+',  # "win $100" 패턴
                r'\d{4,}',  # 긴 숫자 (전화번호 등)
                r'[A-Z]{3,}',  # 연속 대문자
                r'[!]{2,}',  # 연속 느낌표
            ],
            'weight': 2
        }
    }
    
    message_lower = message.lower()
    spam_score = 0
    found_indicators = []
    
    # 단어 기반 신호 확인
    for category, info in spam_indicators.items():
        if 'words' in info:
            for word in info['words']:
                # 단어 경계 고려 정규표현식
                pattern = r'\b' + re.escape(word.lower()) + r'\b'
                if re.search(pattern, message_lower):
                    spam_score += info['weight']
                    found_indicators.append(f"{category}: {word}")
        
        # 패턴 기반 신호 확인
        elif 'patterns' in info:
            for pattern in info['patterns']:
                if re.search(pattern, message, re.IGNORECASE):
                    spam_score += info['weight']
                    found_indicators.append(f"{category}: pattern match")
    
    # 추가 휴리스틱
    # 메시지 길이 고려
    if len(message) > 100:
        spam_score += 1
        found_indicators.append("length: long message")
    
    # 대문자 비율 고려  
    upper_ratio = sum(1 for c in message if c.isupper()) / len(message)
    if upper_ratio > 0.3:
        spam_score += 2
        found_indicators.append("style: high uppercase ratio")
    
    # 숫자 밀도 고려
    digit_ratio = sum(1 for c in message if c.isdigit()) / len(message)
    if digit_ratio > 0.1:
        spam_score += 1
        found_indicators.append("content: high digit ratio")
    
    # 분류 결정 (임계값 기반)
    threshold = 4
    classification = 'spam' if spam_score >= threshold else 'ham'
    
    return {
        'classification': classification,
        'spam_score': spam_score,
        'indicators': found_indicators,
        'confidence': min(1.0, spam_score / 10)  # 0-1 신뢰도
    }

# 테스트
print("고급 분류기 결과:")
for msg in test_messages:
    result = classify_sms_advanced(msg)
    print(f"'{msg}'")
    print(f"  -> {result['classification']} (점수: {result['spam_score']}, 신뢰도: {result['confidence']:.2f})")
    print(f"  -> 지표: {result['indicators']}")
    print()
```

**코드 해설:**
- **가중치 시스템**: 단순한 키워드 매칭 대신 신호의 중요도 차별화
- **패턴 인식**: 정규표현식으로 복잡한 스팸 패턴 탐지
- **맥락 고려**: 단어 경계, 메시지 구조 등 종합적 판단
- **신뢰도 제공**: 분류 결과의 확실성 정도 표시

> 💡 **AI 코드 오류 탐지 체크리스트**
> 
> **기능적 측면:**
> - [ ] 빈 입력, None 값 처리
> - [ ] 데이터 타입 검증
> - [ ] 예외 상황 핸들링
> 
> **성능적 측면:**
> - [ ] 반복문 효율성
> - [ ] 메모리 사용량
> - [ ] 벡터화 가능성
> 
> **논리적 측면:**
> - [ ] 가정의 타당성
> - [ ] 엣지 케이스 고려
> - [ ] 결과의 현실성

> 🖼️ **이미지 생성 프롬프트**: 
> "AI 코드 오류 패턴을 보여주는 인포그래픽. 기능적 오류(빨간색), 성능 오류(주황색), 논리 오류(노란색)로 구분되어 있고, 각 오류 유형별로 대표적인 예시 아이콘과 해결책이 표시된 모던한 다이어그램"

## 2. 코드 품질 평가 기준과 자동화 도구

코드 품질을 평가하는 것은 마치 음식의 맛을 평가하는 것과 같습니다. 맛, 영양, 모양, 가격 등 여러 기준을 종합적으로 고려해야 합니다. 코드도 마찬가지로 기능성, 성능, 가독성, 유지보수성 등 다양한 관점에서 평가해야 합니다.

### 2.1 코드 품질 평가의 5가지 차원

#### **차원 1: 기능적 정확성 (Functional Correctness)**
코드가 의도한 대로 동작하는지 확인합니다.

```python
class CodeFunctionalityTester:
    """코드의 기능적 정확성을 테스트하는 클래스"""
    
    def __init__(self):
        self.test_results = []
    
    def test_function(self, func, test_cases, description=""):
        """
        함수의 기능을 다양한 테스트 케이스로 검증
        
        Args:
            func: 테스트할 함수
            test_cases: [(입력, 예상출력), ...] 형태의 테스트 케이스
            description: 테스트 설명
        """
        print(f"\n🧪 {description} 테스트 시작")
        print("=" * 50)
        
        passed = 0
        failed = 0
        
        for i, (inputs, expected) in enumerate(test_cases, 1):
            try:
                # 함수 실행
                if isinstance(inputs, tuple):
                    result = func(*inputs)
                else:
                    result = func(inputs)
                
                # 결과 비교
                if self._compare_results(result, expected):
                    print(f"✅ 테스트 {i}: PASS")
                    passed += 1
                else:
                    print(f"❌ 테스트 {i}: FAIL")
                    print(f"   입력: {inputs}")
                    print(f"   예상: {expected}")
                    print(f"   실제: {result}")
                    failed += 1
                    
            except Exception as e:
                print(f"💥 테스트 {i}: ERROR - {e}")
                print(f"   입력: {inputs}")
                failed += 1
        
        # 결과 요약
        total = passed + failed
        success_rate = passed / total if total > 0 else 0
        print(f"\n📊 테스트 결과: {passed}/{total} 통과 ({success_rate:.1%})")
        
        self.test_results.append({
            'description': description,
            'passed': passed,
            'failed': failed,
            'success_rate': success_rate
        })
        
        return success_rate
    
    def _compare_results(self, actual, expected):
        """결과 비교 (타입과 값 모두 고려)"""
        if type(actual) != type(expected):
            return False
        
        if isinstance(expected, dict):
            return all(actual.get(k) == v for k, v in expected.items())
        elif isinstance(expected, (list, tuple)):
            return len(actual) == len(expected) and all(a == e for a, e in zip(actual, expected))
        else:
            return actual == expected

# SMS 길이 계산 함수 테스트 예시
def calculate_message_length(message):
    """SMS 메시지 길이 계산"""
    if not isinstance(message, str):
        raise ValueError("메시지는 문자열이어야 합니다")
    return len(message.strip())

# 테스트 실행
tester = CodeFunctionalityTester()

# 정상 케이스부터 엣지 케이스까지 다양한 테스트
test_cases = [
    ("Hello", 5),                    # 정상 케이스
    ("  Hello  ", 5),                # 공백 제거
    ("", 0),                         # 빈 문자열
    ("   ", 0),                      # 공백만
    ("Hi👋", 3),                     # 이모티콘 포함
    ("안녕하세요", 5),                # 한글
]

# 에러 케이스 (예외 발생 예상)
error_cases = [
    (None, ValueError),              # None 입력
    (123, ValueError),               # 숫자 입력
    ([], ValueError),                # 리스트 입력
]

# 정상 테스트
success_rate = tester.test_function(
    calculate_message_length, 
    test_cases, 
    "SMS 길이 계산 함수"
)

# 에러 케이스 테스트
print(f"\n🚨 에러 케이스 테스트")
for inputs, expected_error in error_cases:
    try:
        result = calculate_message_length(inputs)
        print(f"❌ 예상 에러 발생 안함: {inputs} -> {result}")
    except expected_error:
        print(f"✅ 예상 에러 발생: {inputs} -> {expected_error.__name__}")
    except Exception as e:
        print(f"⚠️ 다른 에러 발생: {inputs} -> {type(e).__name__}: {e}")
```

**코드 해설:**
- **포괄적 테스트**: 정상 케이스부터 엣지 케이스, 에러 케이스까지 체계적 검증
- **자동화된 비교**: 다양한 데이터 타입에 대한 결과 비교 로직
- **명확한 피드백**: 실패한 테스트에 대한 상세한 정보 제공

#### **차원 2: 성능 효율성 (Performance Efficiency)**
코드의 실행 속도와 메모리 사용량을 측정합니다.

```python
import time
import psutil
import memory_profiler
from functools import wraps
import matplotlib.pyplot as plt

class PerformanceProfiler:
    """코드 성능을 프로파일링하는 클래스"""
    
    def __init__(self):
        self.profile_results = []
    
    def profile_function(self, func, test_data, description=""):
        """
        함수의 성능을 측정합니다
        
        Args:
            func: 측정할 함수
            test_data: 테스트에 사용할 데이터
            description: 함수 설명
        """
        print(f"\n⚡ {description} 성능 측정")
        print("=" * 50)
        
        # 메모리 사용량 측정 (시작)
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # 실행 시간 측정
        start_time = time.perf_counter()
        
        try:
            result = func(test_data)
            execution_successful = True
        except Exception as e:
            print(f"❌ 실행 오류: {e}")
            execution_successful = False
            result = None
        
        end_time = time.perf_counter()
        
        # 메모리 사용량 측정 (종료)
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        # 결과 계산
        execution_time = end_time - start_time
        memory_used = memory_after - memory_before
        
        # 결과 출력
        print(f"📊 실행 시간: {execution_time:.4f}초")
        print(f"💾 메모리 사용: {memory_used:.2f}MB")
        
        if execution_successful:
            print(f"📋 결과 크기: {len(result) if hasattr(result, '__len__') else 'N/A'}")
            print(f"✅ 실행 성공")
        
        # 성능 등급 계산
        performance_grade = self._calculate_performance_grade(execution_time, memory_used)
        print(f"🏆 성능 등급: {performance_grade}")
        
        # 결과 저장
        profile_data = {
            'description': description,
            'execution_time': execution_time,
            'memory_used': memory_used,
            'success': execution_successful,
            'grade': performance_grade
        }
        self.profile_results.append(profile_data)
        
        return profile_data
    
    def _calculate_performance_grade(self, time_sec, memory_mb):
        """성능 등급 계산 (A-F)"""
        # 시간 점수 (5초 기준)
        time_score = max(0, 100 - (time_sec / 5.0) * 100)
        
        # 메모리 점수 (100MB 기준)  
        memory_score = max(0, 100 - (memory_mb / 100.0) * 100)
        
        # 종합 점수
        total_score = (time_score + memory_score) / 2
        
        if total_score >= 90: return "A"
        elif total_score >= 80: return "B"
        elif total_score >= 70: return "C"
        elif total_score >= 60: return "D"
        else: return "F"
    
    def compare_functions(self, functions_and_data, test_size=1000):
        """여러 함수의 성능을 비교합니다"""
        
        print(f"\n🆚 함수 성능 비교 (테스트 크기: {test_size})")
        print("=" * 60)
        
        results = []
        
        for func, data, description in functions_and_data:
            # 테스트 데이터 크기 조정
            if hasattr(data, '__len__') and len(data) > test_size:
                test_data = data[:test_size]
            else:
                test_data = data
            
            result = self.profile_function(func, test_data, description)
            results.append(result)
        
        # 비교 차트 생성
        self._create_comparison_chart(results)
        
        return results
    
    def _create_comparison_chart(self, results):
        """성능 비교 차트 생성"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        descriptions = [r['description'] for r in results]
        times = [r['execution_time'] for r in results]
        memories = [r['memory_used'] for r in results]
        
        # 실행 시간 비교
        bars1 = ax1.bar(descriptions, times, color='skyblue', alpha=0.7)
        ax1.set_ylabel('실행 시간 (초)')
        ax1.set_title('실행 시간 비교')
        ax1.tick_params(axis='x', rotation=45)
        
        # 값 표시
        for bar, time_val in zip(bars1, times):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{time_val:.3f}s', ha='center', va='bottom')
        
        # 메모리 사용량 비교
        bars2 = ax2.bar(descriptions, memories, color='lightcoral', alpha=0.7)
        ax2.set_ylabel('메모리 사용량 (MB)')
        ax2.set_title('메모리 사용량 비교')
        ax2.tick_params(axis='x', rotation=45)
        
        # 값 표시
        for bar, mem_val in zip(bars2, memories):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{mem_val:.1f}MB', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

# 성능 테스트 예시: 다양한 SMS 처리 함수 비교
def process_sms_basic(messages):
    """기본적인 SMS 처리"""
    results = []
    for msg in messages:
        if isinstance(msg, str):
            clean = msg.lower().strip()
            word_count = len(clean.split())
            char_count = len(clean)
            results.append({
                'message': clean,
                'word_count': word_count,
                'char_count': char_count
            })
    return results

def process_sms_optimized(messages):
    """최적화된 SMS 처리 (벡터화)"""
    import pandas as pd
    
    # pandas로 벡터화된 처리
    df = pd.DataFrame({'message': messages})
    df = df[df['message'].notna() & (df['message'] != '')]
    
    df['clean_message'] = df['message'].str.lower().str.strip()
    df['word_count'] = df['clean_message'].str.split().str.len()
    df['char_count'] = df['clean_message'].str.len()
    
    return df[['clean_message', 'word_count', 'char_count']].rename(
        columns={'clean_message': 'message'}
    ).to_dict('records')

# 테스트 데이터 생성
sample_messages = [
    "Hello world",
    "FREE money NOW!!!",
    "How are you today?",
    "Win $1000 cash prize",
    "Call me when you're free"
] * 200  # 1000개 메시지 생성

# 성능 비교 실행
profiler = PerformanceProfiler()

comparison_data = [
    (process_sms_basic, sample_messages, "기본 처리"),
    (process_sms_optimized, sample_messages, "최적화 처리"),
]

comparison_results = profiler.compare_functions(comparison_data, test_size=1000)
```

**코드 해설:**
- **다차원 성능 측정**: 실행 시간과 메모리 사용량 동시 측정
- **성능 등급 시스템**: A-F 등급으로 성능을 직관적으로 표현
- **시각적 비교**: matplotlib를 사용한 성능 비교 차트 생성
- **실제적 테스트**: 동일한 작업을 수행하는 다른 구현 방식 비교

#### **차원 3: 코드 품질 (Code Quality)**
코드의 가독성, 유지보수성, 스타일 일관성을 평가합니다.

```python
import ast
import re
from collections import defaultdict

class CodeQualityAnalyzer:
    """코드 품질을 분석하는 클래스"""
    
    def __init__(self):
        self.quality_metrics = defaultdict(int)
        self.warnings = []
        self.suggestions = []
    
    def analyze_code_quality(self, code_string, filename="<string>"):
        """
        파이썬 코드의 품질을 종합적으로 분석
        
        Args:
            code_string: 분석할 파이썬 코드 문자열
            filename: 파일명 (선택사항)
        """
        print(f"\n📋 코드 품질 분석: {filename}")
        print("=" * 50)
        
        # 코드 파싱
        try:
            tree = ast.parse(code_string)
        except SyntaxError as e:
            print(f"❌ 구문 오류: {e}")
            return {'overall_grade': 'F', 'error': str(e)}
        
        # 각 품질 지표 분석
        self._analyze_complexity(tree)
        self._analyze_naming(tree)
        self._analyze_structure(tree)
        self._analyze_documentation(code_string)
        self._analyze_style(code_string)
        
        # 종합 평가
        overall_grade = self._calculate_overall_grade()
        
        # 결과 출력
        self._print_analysis_results(overall_grade)
        
        return {
            'overall_grade': overall_grade,
            'metrics': dict(self.quality_metrics),
            'warnings': self.warnings,
            'suggestions': self.suggestions
        }
    
    def _analyze_complexity(self, tree):
        """복잡도 분석 (순환 복잡도, 중첩 깊이 등)"""
        
        class ComplexityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.max_depth = 0
                self.current_depth = 0
                self.decision_points = 0
                self.function_count = 0
                self.long_functions = 0
            
            def visit_FunctionDef(self, node):
                self.function_count += 1
                
                # 함수 길이 체크 (20줄 이상이면 경고)
                if len(node.body) > 20:
                    self.long_functions += 1
                
                self.generic_visit(node)
            
            def visit_If(self, node):
                self.decision_points += 1
                self._visit_nested(node)
            
            def visit_For(self, node):
                self.decision_points += 1
                self._visit_nested(node)
            
            def visit_While(self, node):
                self.decision_points += 1
                self._visit_nested(node)
            
            def _visit_nested(self, node):
                self.current_depth += 1
                self.max_depth = max(self.max_depth, self.current_depth)
                self.generic_visit(node)
                self.current_depth -= 1
        
        visitor = ComplexityVisitor()
        visitor.visit(tree)
        
        # 복잡도 지표 저장
        self.quality_metrics['max_nesting_depth'] = visitor.max_depth
        self.quality_metrics['cyclomatic_complexity'] = visitor.decision_points + 1
        self.quality_metrics['function_count'] = visitor.function_count
        self.quality_metrics['long_functions'] = visitor.long_functions
        
        # 경고 생성
        if visitor.max_depth > 4:
            self.warnings.append(f"중첩 깊이가 깊음 ({visitor.max_depth}레벨)")
        
        if visitor.decision_points > 10:
            self.warnings.append(f"순환 복잡도가 높음 ({visitor.decision_points + 1})")
        
        if visitor.long_functions > 0:
            self.warnings.append(f"긴 함수 발견 ({visitor.long_functions}개)")
    
    def _analyze_naming(self, tree):
        """네이밍 컨벤션 분석"""
        
        class NamingVisitor(ast.NodeVisitor):
            def __init__(self):
                self.function_names = []
                self.variable_names = []
                self.class_names = []
            
            def visit_FunctionDef(self, node):
                self.function_names.append(node.name)
                self.generic_visit(node)
            
            def visit_ClassDef(self, node):
                self.class_names.append(node.name)
                self.generic_visit(node)
            
            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Store):
                    self.variable_names.append(node.id)
        
        visitor = NamingVisitor()
        visitor.visit(tree)
        
        # 네이밍 규칙 체크
        naming_score = 0
        total_names = 0
        
        # 함수명 체크 (snake_case)
        for name in visitor.function_names:
            total_names += 1
            if re.match(r'^[a-z][a-z0-9_]*

, name):
                naming_score += 1
            else:
                self.warnings.append(f"함수명 '{name}'이 snake_case 규칙에 맞지 않음")
        
        # 클래스명 체크 (PascalCase)
        for name in visitor.class_names:
            total_names += 1
            if re.match(r'^[A-Z][a-zA-Z0-9]*

, name):
                naming_score += 1
            else:
                self.warnings.append(f"클래스명 '{name}'이 PascalCase 규칙에 맞지 않음")
        
        # 변수명 체크 (의미있는 이름)
        short_names = [name for name in visitor.variable_names if len(name) < 3 and name not in ['i', 'j', 'k', 'x', 'y', 'z']]
        if short_names:
            self.warnings.append(f"의미없는 짧은 변수명: {short_names}")
        
        # 네이밍 점수 저장
        self.quality_metrics['naming_score'] = (naming_score / total_names * 100) if total_names > 0 else 100
    
    def _analyze_structure(self, tree):
        """코드 구조 분석"""
        
        # 함수당 평균 라인 수
        function_lines = []
        
        class StructureVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                # 함수 내부 라인 수 계산
                if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                    lines = node.end_lineno - node.lineno + 1
                    function_lines.append(lines)
                self.generic_visit(node)
        
        visitor = StructureVisitor()
        visitor.visit(tree)
        
        if function_lines:
            avg_function_length = sum(function_lines) / len(function_lines)
            self.quality_metrics['avg_function_length'] = avg_function_length
            
            if avg_function_length > 30:
                self.suggestions.append("함수를 더 작은 단위로 분리하는 것을 고려하세요")
    
    def _analyze_documentation(self, code_string):
        """문서화 분석"""
        
        lines = code_string.split('\n')
        
        # 주석 라인 수 계산
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        docstring_lines = code_string.count('"""') // 2 * 3  # 간단한 추정
        
        total_lines = len([line for line in lines if line.strip()])
        
        if total_lines > 0:
            comment_ratio = (comment_lines + docstring_lines) / total_lines * 100
            self.quality_metrics['documentation_ratio'] = comment_ratio
            
            if comment_ratio < 10:
                self.suggestions.append("코드에 더 많은 주석과 문서화를 추가하세요")
            elif comment_ratio > 50:
                self.warnings.append("주석이 너무 많을 수 있습니다. 코드의 명확성을 확인하세요")
    
    def _analyze_style(self, code_string):
        """코딩 스타일 분석"""
        
        lines = code_string.split('\n')
        
        # 긴 라인 체크 (80자 초과)
        long_lines = [i+1 for i, line in enumerate(lines) if len(line) > 80]
        if long_lines:
            self.warnings.append(f"긴 라인 발견 (80자 초과): 라인 {long_lines}")
        
        # 빈 라인 체크
        consecutive_empty = 0
        max_consecutive_empty = 0
        
        for line in lines:
            if line.strip() == '':
                consecutive_empty += 1
                max_consecutive_empty = max(max_consecutive_empty, consecutive_empty)
            else:
                consecutive_empty = 0
        
        if max_consecutive_empty > 2:
            self.suggestions.append("연속된 빈 라인이 너무 많습니다")
        
        # 스타일 점수 계산
        style_score = 100
        style_score -= len(long_lines) * 5  # 긴 라인마다 5점 감점
        style_score -= max(0, max_consecutive_empty - 2) * 10  # 과도한 빈 라인 감점
        
        self.quality_metrics['style_score'] = max(0, style_score)
    
    def _calculate_overall_grade(self):
        """종합 등급 계산"""
        
        # 가중치 적용 점수 계산
        weights = {
            'complexity': 0.3,
            'naming': 0.2,
            'documentation': 0.2,
            'style': 0.3
        }
        
        # 복잡도 점수 (낮을수록 좋음)
        complexity_score = max(0, 100 - self.quality_metrics.get('cyclomatic_complexity', 1) * 5)
        complexity_score -= self.quality_metrics.get('max_nesting_depth', 0) * 10
        complexity_score = max(0, complexity_score)
        
        # 각 영역별 점수
        scores = {
            'complexity': complexity_score,
            'naming': self.quality_metrics.get('naming_score', 80),
            'documentation': min(100, self.quality_metrics.get('documentation_ratio', 10) * 5),
            'style': self.quality_metrics.get('style_score', 80)
        }
        
        # 가중 평균 계산
        weighted_score = sum(scores[area] * weights[area] for area in weights)
        
        # 경고에 따른 감점
        weighted_score -= len(self.warnings) * 5
        weighted_score = max(0, weighted_score)
        
        # 등급 결정
        if weighted_score >= 90: return 'A'
        elif weighted_score >= 80: return 'B'
        elif weighted_score >= 70: return 'C'
        elif weighted_score >= 60: return 'D'
        else: return 'F'
    
    def _print_analysis_results(self, overall_grade):
        """분석 결과 출력"""
        
        print(f"📊 복잡도:")
        print(f"  - 순환 복잡도: {self.quality_metrics.get('cyclomatic_complexity', 'N/A')}")
        print(f"  - 최대 중첩 깊이: {self.quality_metrics.get('max_nesting_depth', 'N/A')}")
        print(f"  - 함수 개수: {self.quality_metrics.get('function_count', 'N/A')}")
        
        print(f"\n📝 네이밍:")
        print(f"  - 네이밍 점수: {self.quality_metrics.get('naming_score', 'N/A'):.1f}%")
        
        print(f"\n📚 문서화:")
        print(f"  - 문서화 비율: {self.quality_metrics.get('documentation_ratio', 'N/A'):.1f}%")
        
        print(f"\n🎨 스타일:")
        print(f"  - 스타일 점수: {self.quality_metrics.get('style_score', 'N/A'):.1f}%")
        
        print(f"\n🏆 종합 등급: {overall_grade}")
        
        if self.warnings:
            print(f"\n⚠️ 경고사항:")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        if self.suggestions:
            print(f"\n💡 개선 제안:")
            for suggestion in self.suggestions:
                print(f"  - {suggestion}")

# 코드 품질 분석 예시
sample_code = '''
def processMessages(msgs):
    result=[]
    for m in msgs:
        if m!=None:
            if len(m)>0:
                if isinstance(m,str):
                    clean=m.lower().strip()
                    if len(clean)>0:
                        wc=len(clean.split())
                        cc=len(clean)
                        result.append({"msg":clean,"wc":wc,"cc":cc})
    return result

class dataProcessor:
    def __init__(self):
        self.data=[]
    
    def add(self,item):
        self.data.append(item)
'''

# 분석 실행
analyzer = CodeQualityAnalyzer()
quality_result = analyzer.analyze_code_quality(sample_code, "sample_sms_processor.py")
```

**코드 해설:**
- **AST 파싱**: Python의 Abstract Syntax Tree를 활용한 정적 코드 분석
- **다차원 평가**: 복잡도, 네이밍, 문서화, 스타일 등 여러 관점에서 종합 평가
- **실용적 피드백**: 구체적인 문제점과 개선 방안 제시
- **점수화 시스템**: 객관적인 품질 지표를 통한 등급 산정

> 💡 **코드 품질 자동 평가 도구 추천**
> 
> **정적 분석 도구:**
> - **pylint**: 종합적인 코드 품질 분석
> - **flake8**: PEP 8 스타일 가이드 준수 확인
> - **mypy**: 타입 힌트 검증
> 
> **복잡도 분석:**
> - **radon**: 순환 복잡도, 유지보수성 지수 계산
> - **xenon**: 복잡도 기반 품질 모니터링
> 
> **보안 분석:**
> - **bandit**: 보안 취약점 스캔
> - **safety**: 의존성 패키지 보안 검사

> 🖼️ **이미지 생성 프롬프트**: 
> "코드 품질 평가의 5차원을 보여주는 레이더 차트. 기능적 정확성, 성능 효율성, 코드 품질, 보안성, 유지보수성이 5각형으로 배치되고, 각 차원마다 점수가 표시된 모던한 대시보드 스타일"

## 3. 성능 최적화 전략과 리팩토링 기법

코드 최적화는 마치 자동차 튜닝과 같습니다. 기본적으로 잘 달리는 차도 전문가의 손길을 거치면 더욱 빠르고 효율적으로 달릴 수 있습니다. AI가 생성한 코드도 마찬가지로 체계적인 최적화를 통해 성능을 크게 향상시킬 수 있습니다.

### 3.1 데이터 구조 최적화

#### **최적화 전략 1: 적합한 자료구조 선택**

AI는 종종 가장 직관적인 자료구조(리스트)를 선택하지만, 작업의 특성에 따라 더 효율적인 자료구조가 있을 수 있습니다.

**❌ 비효율적인 데이터 구조 사용:**
```python
def find_spam_keywords_inefficient(messages, keywords):
    """
    스팸 키워드 검색 (비효율적 - 리스트 사용)
    시간 복잡도: O(n * m * k) where n=메시지 수, m=키워드 수, k=평균 메시지 길이
    """
    spam_messages = []
    
    for message in messages:
        found_keywords = []
        
        # 리스트에서 키워드 검색 - O(m) 시간 소요
        for keyword in keywords:
            if keyword.lower() in message.lower():
                found_keywords.append(keyword)
        
        if found_keywords:
            spam_messages.append({
                'message': message,
                'keywords': found_keywords,
                'spam_score': len(found_keywords)
            })
    
    return spam_messages

# 성능 테스트
import time
import random

# 테스트 데이터 생성
keywords = ['free', 'win', 'cash', 'prize', 'urgent', 'limited', 'offer', 'call', 'click', 'now'] * 10  # 100개 키워드
messages = [
    f"This is message {i} with some free offers and win prizes" 
    for i in range(1000)
]

start_time = time.time()
result_inefficient = find_spam_keywords_inefficient(messages, keywords)
time_inefficient = time.time() - start_time
print(f"비효율적 버전: {time_inefficient:.4f}초")
```

**✅ 최적화된 데이터 구조:**
```python
def find_spam_keywords_optimized(messages, keywords):
    """
    스팸 키워드 검색 (최적화 - 집합과 트라이 사용)
    시간 복잡도: O(n * k) where n=메시지 수, k=평균 메시지 길이
    """
    
    # 키워드를 집합으로 변환 - O(1) 검색 시간
    keyword_set = {keyword.lower() for keyword in keywords}
    
    spam_messages = []
    
    for message in messages:
        # 메시지를 단어로 분할하고 집합으로 변환
        message_words = set(word.lower().strip('.,!?;:') for word in message.split())
        
        # 교집합으로 한 번에 찾기 - O(min(len(keyword_set), len(message_words)))
        found_keywords = list(message_words.intersection(keyword_set))
        
        if found_keywords:
            spam_messages.append({
                'message': message,
                'keywords': found_keywords,
                'spam_score': len(found_keywords)
            })
    
    return spam_messages

# 더 고급 최적화: 트라이(Trie) 자료구조 활용
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.word = ""

class SpamKeywordTrie:
    """트라이를 활용한 고성능 키워드 검색"""
    
    def __init__(self, keywords):
        self.root = TrieNode()
        self._build_trie(keywords)
    
    def _build_trie(self, keywords):
        """키워드들로 트라이 구축"""
        for keyword in keywords:
            self._insert(keyword.lower())
    
    def _insert(self, word):
        """트라이에 단어 삽입"""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_word = True
        node.word = word
    
    def find_keywords_in_text(self, text):
        """텍스트에서 모든 키워드 찾기"""
        text = text.lower()
        found_keywords = []
        
        # 모든 시작 위치에서 키워드 검색
        for i in range(len(text)):
            keywords = self._search_from_position(text, i)
            found_keywords.extend(keywords)
        
        return list(set(found_keywords))  # 중복 제거
    
    def _search_from_position(self, text, start_pos):
        """특정 위치에서 시작하는 키워드들 찾기"""
        node = self.root
        found_keywords = []
        
        for i in range(start_pos, len(text)):
            char = text[i]
            if char not in node.children:
                break
            
            node = node.children[char]
            if node.is_word:
                # 단어 경계 확인 (간단한 버전)
                if (start_pos == 0 or not text[start_pos-1].isalnum()) and \
                   (i == len(text)-1 or not text[i+1].isalnum()):
                    found_keywords.append(node.word)
        
        return found_keywords

def find_spam_keywords_trie(messages, keywords):
    """트라이를 활용한 스팸 키워드 검색"""
    
    # 트라이 구축 (한 번만 수행)
    trie = SpamKeywordTrie(keywords)
    
    spam_messages = []
    
    for message in messages:
        found_keywords = trie.find_keywords_in_text(message)
        
        if found_keywords:
            spam_messages.append({
                'message': message,
                'keywords': found_keywords,
                'spam_score': len(found_keywords)
            })
    
    return spam_messages

# 성능 비교
start_time = time.time()
result_optimized = find_spam_keywords_optimized(messages, keywords)
time_optimized = time.time() - start_time
print(f"집합 최적화 버전: {time_optimized:.4f}초")

start_time = time.time()
result_trie = find_spam_keywords_trie(messages, keywords)
time_trie = time.time() - start_time
print(f"트라이 최적화 버전: {time_trie:.4f}초")

# 성능 개선 비율 계산
improvement_set = (time_inefficient - time_optimized) / time_inefficient * 100
improvement_trie = (time_inefficient - time_trie) / time_inefficient * 100

print(f"\n📈 성능 개선:")
print(f"집합 사용: {improvement_set:.1f}% 개선")
print(f"트라이 사용: {improvement_trie:.1f}% 개선")
```

**코드 해설:**
- **집합(Set) 활용**: O(n) 검색을 O(1)으로 개선
- **교집합 연산**: 벡터화된 집합 연산으로 효율성 증대
- **트라이 자료구조**: 문자열 검색에 특화된 트리 구조로 접두사 매칭 최적화
- **메모리 vs 시간 트레이드오프**: 초기 구축 비용은 있지만 반복 검색에서 큰 이득

#### **최적화 전략 2: 메모리 효율적 처리**

대용량 데이터 처리 시 메모리 사용량을 최소화하는 것이 중요합니다.

**❌ 메모리 비효율적인 코드:**
```python
def process_large_sms_dataset_inefficient(file_path):
    """
    대용량 SMS 데이터셋 처리 (메모리 비효율적)
    모든 데이터를 메모리에 로드
    """
    
    # 전체 데이터를 한 번에 메모리에 로드
    with open(file_path, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()  # 전체 파일을 메모리에 로드
    
    processed_data = []
    
    for line in all_lines:
        # 각 줄 처리
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            label, message = parts[0], parts[1]
            
            # 복잡한 처리 (메모리 사용량 증가)
            processed_message = {
                'original': message,
                'clean': message.lower().strip(),
                'words': message.split(),
                'char_count': len(message),
                'word_count': len(message.split()),
                'label': label
            }
            processed_data.append(processed_message)
    
    return processed_data
```

**✅ 메모리 효율적인 코드:**
```python
def process_large_sms_dataset_efficient(file_path, batch_size=1000):
    """
    대용량 SMS 데이터셋 처리 (메모리 효율적)
    스트리밍 처리와 배치 처리 결합
    """
    
    def process_batch(batch):
        """배치 단위 처리"""
        processed_batch = []
        
        for line in batch:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                label, message = parts[0], parts[1]
                
                # 필요한 정보만 저장 (메모리 절약)
                processed_message = {
                    'message': message.lower().strip(),
                    'length': len(message),
                    'word_count': len(message.split()),
                    'label': label
                }
                processed_batch.append(processed_message)
        
        return processed_batch
    
    # 제너레이터를 사용한 스트리밍 처리
    def data_generator():
        with open(file_path, 'r', encoding='utf-8') as f:
            batch = []
            
            for line in f:  # 한 줄씩 읽기
                batch.append(line)
                
                if len(batch) >= batch_size:
                    yield process_batch(batch)
                    batch = []  # 배치 초기화로 메모리 해제
            
            # 마지막 배치 처리
            if batch:
                yield process_batch(batch)
    
    # 결과 수집 (필요시 파일로 직접 저장 가능)
    all_processed = []
    for batch_result in data_generator():
        all_processed.extend(batch_result)
        
        # 메모리 사용량 모니터링 (선택사항)
        if len(all_processed) % 10000 == 0:
            print(f"처리 완료: {len(all_processed)}개 메시지")
    
    return all_processed

# 더 고급: pandas를 활용한 청크 단위 처리
import pandas as pd

def process_with_pandas_chunks(file_path, chunk_size=5000):
    """
    pandas 청크를 활용한 효율적 처리
    """
    
    results = []
    
    # 청크 단위로 파일 읽기
    for chunk in pd.read_csv(file_path, sep='\t', header=None, 
                            names=['label', 'message'], 
                            chunksize=chunk_size):
        
        # 벡터화된 처리
        chunk['clean_message'] = chunk['message'].str.lower().str.strip()
        chunk['length'] = chunk['message'].str.len()
        chunk['word_count'] = chunk['message'].str.split().str.len()
        
        # 필요한 컬럼만 선택하여 메모리 절약
        processed_chunk = chunk[['label', 'clean_message', 'length', 'word_count']]
        
        results.append(processed_chunk)
        
        # 메모리 사용량 출력
        print(f"청크 처리 완료: {len(processed_chunk)}개 행")
    
    # 모든 청크 결합
    final_result = pd.concat(results, ignore_index=True)
    return final_result

# 메모리 사용량 비교 테스트
import psutil
import os

def monitor_memory_usage(func, *args, **kwargs):
    """함수 실행 중 메모리 사용량 모니터링"""
    
    process = psutil.Process(os.getpid())
    
    # 시작 메모리
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # 함수 실행
    result = func(*args, **kwargs)
    
    # 종료 메모리
    memory_after = process.memory_info().rss / 1024 / 1024  # MB
    
    memory_used = memory_after - memory_before
    
    print(f"메모리 사용량: {memory_used:.2f}MB")
    print(f"처리된 데이터 크기: {len(result)}개")
    
    return result, memory_used

# 가상의 대용량 SMS 파일 생성 (테스트용)
def create_large_sms_file(file_path, num_records=50000):
    """테스트용 대용량 SMS 파일 생성"""
    
    import random
    
    spam_templates = [
        "FREE money! Call {} NOW!",
        "Win ${} cash prize! Text back!",
        "URGENT: Claim your {} reward!",
        "Limited time offer: {} discount!"
    ]
    
    ham_templates = [
        "Hey, how are you doing today?",
        "Meeting at {} tomorrow, see you there",
        "Thanks for the {} yesterday",
        "Can you pick up {} from the store?"
    ]
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for i in range(num_records):
            if random.random() < 0.2:  # 20% 스팸
                label = "spam"
                template = random.choice(spam_templates)
                message = template.format(random.randint(100, 9999))
            else:  # 80% 정상
                label = "ham" 
                template = random.choice(ham_templates)
                message = template.format(random.choice(["lunch", "coffee", "milk", "3pm"]))
            
            f.write(f"{label}\t{message}\n")

# 테스트 파일 생성
test_file = "large_sms_test.txt"
create_large_sms_file(test_file, 20000)

print("🧪 메모리 효율성 테스트")
print("=" * 40)

# 비효율적 방법 (주의: 메모리 부족 가능)
try:
    result1, memory1 = monitor_memory_usage(
        process_large_sms_dataset_inefficient, test_file
    )
    print(f"비효율적 방법: {memory1:.2f}MB")
except Exception as e:
    print(f"비효율적 방법 실패: {e}")

# 효율적 방법
result2, memory2 = monitor_memory_usage(
    process_large_sms_dataset_efficient, test_file, 1000
)
print(f"효율적 방법: {memory2:.2f}MB")

# pandas 청크 방법
result3, memory3 = monitor_memory_usage(
    process_with_pandas_chunks, test_file, 2000
)
print(f"pandas 청크 방법: {memory3:.2f}MB")

# 파일 정리
os.remove(test_file)
```

**코드 해설:**
- **스트리밍 처리**: 전체 데이터를 메모리에 올리지 않고 순차적 처리
- **배치 처리**: 적당한 크기의 배치로 나누어 메모리 사용량 제어
- **제너레이터 활용**: `yield`를 사용한 지연 평가로 메모리 효율성 증대
- **pandas 청킹**: 대용량 데이터 처리에 최적화된 pandas 기능 활용

### 3.2 알고리즘 최적화

#### **최적화 전략 3: 시간 복잡도 개선**

**❌ 비효율적인 알고리즘:**
```python
def find_similar_messages_naive(messages, threshold=0.8):
    """
    유사한 메시지 찾기 (순진한 방법)
    시간 복잡도: O(n²) - 모든 쌍을 비교
    """
    
    def jaccard_similarity(text1, text2):
        """자카드 유사도 계산"""
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0
    
    similar_pairs = []
    
    # 모든 쌍을 비교 - O(n²)
    for i in range(len(messages)):
        for j in range(i + 1, len(messages)):
            similarity = jaccard_similarity(messages[i], messages[j])
            
            if similarity >= threshold:
                similar_pairs.append({
                    'message1': messages[i],
                    'message2': messages[j],
                    'similarity': similarity
                })
    
    return similar_pairs
```

**✅ 최적화된 알고리즘:**
```python
from collections import defaultdict
import hashlib

def find_similar_messages_optimized(messages, threshold=0.8):
    """
    유사한 메시지 찾기 (최적화된 방법)
    LSH(Locality Sensitive Hashing) 사용
    """
    
    def get_shingles(text, k=3):
        """k-shingle 생성"""
        words = text.lower().split()
        if len(words) < k:
            return {' '.join(words)}
        
        shingles = set()
        for i in range(len(words) - k + 1):
            shingle = ' '.join(words[i:i+k])
            shingles.add(shingle)
        return shingles
    
    def minhash_signature(shingles, num_hashes=100):
        """MinHash 시그니처 생성"""
        signature = []
        
        for i in range(num_hashes):
            min_hash = float('inf')
            
            for shingle in shingles:
                # 해시 함수 (i를 시드로 사용)
                hash_val = int(hashlib.md5((shingle + str(i)).encode()).hexdigest(), 16)
                min_hash = min(min_hash, hash_val)
            
            signature.append(min_hash)
        
        return signature
    
    def estimate_jaccard_similarity(sig1, sig2):
        """MinHash 시그니처로 자카드 유사도 추정"""
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)
    
    # 1단계: 모든 메시지의 MinHash 시그니처 계산
    signatures = []
    message_data = []
    
    for i, message in enumerate(messages):
        shingles = get_shingles(message)
        if shingles:  # 빈 shingle 집합 제외
            signature = minhash_signature(shingles)
            signatures.append(signature)
            message_data.append((i, message))
    
    # 2단계: LSH 버킷팅 (선택사항 - 더 큰 최적화)
    # 여기서는 간단한 버전으로 모든 시그니처 비교
    
    similar_pairs = []
    
    # 시그니처 비교 (여전히 O(n²)이지만 실제 텍스트 비교보다 훨씬 빠름)
    for i in range(len(signatures)):
        for j in range(i + 1, len(signatures)):
            estimated_similarity = estimate_jaccard_similarity(signatures[i], signatures[j])
            
            if estimated_similarity >= threshold:
                # 정확한 유사도 계산 (필요시)
                actual_similarity = jaccard_similarity(
                    message_data[i][1], message_data[j][1]
                )
                
                if actual_similarity >= threshold:
                    similar_pairs.append({
                        'message1': message_data[i][1],
                        'message2': message_data[j][1],
                        'similarity': actual_similarity,
                        'estimated_similarity': estimated_similarity
                    })
    
    return similar_pairs

def jaccard_similarity(text1, text2):
    """자카드 유사도 계산 (공통 함수)"""
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0

# 더 고급: sklearn을 활용한 벡터화 접근
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def find_similar_messages_vectorized(messages, threshold=0.8):
    """
    벡터화를 활용한 유사 메시지 찾기
    TF-IDF + 코사인 유사도 사용
    """
    
    # TF-IDF 벡터화
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),  # 1-gram to 3-gram
        min_df=1,
        max_features=10000,
        stop_words='english'
    )
    
    # 벡터 행렬 생성
    tfidf_matrix = vectorizer.fit_transform(messages)
    
    # 코사인 유사도 계산 (한 번에 모든 쌍)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # 임계값 이상인 쌍 찾기
    similar_pairs = []
    
    # 상삼각 행렬만 확인 (대칭이므로)
    rows, cols = np.where((similarity_matrix >= threshold) & 
                         (similarity_matrix < 1.0))  # 자기 자신 제외
    
    for row, col in zip(rows, cols):
        if row < col:  # 중복 제거
            similar_pairs.append({
                'message1': messages[row],
                'message2': messages[col],
                'similarity': similarity_matrix[row, col]
            })
    
    return similar_pairs

# 성능 비교 테스트
test_messages = [
    "Free money call now",
    "Call now for free money",
    "Win cash prize today",
    "Hello how are you",
    "Hi how are you doing",
    "Get free cash now",
    "Today win cash prize",
    "Are you doing well",
    "Free money available call",
    "Cash prize win today"
] * 50  # 500개 메시지로 확장

print("🚀 알고리즘 성능 비교")
print("=" * 40)

# 순진한 방법
import time
start = time.time()
result_naive = find_similar_messages_naive(test_messages[:100], 0.6)  # 작은 데이터로 테스트
time_naive = time.time() - start
print(f"순진한 방법 (100개): {time_naive:.4f}초, {len(result_naive)}개 쌍")

# 최적화 방법
start = time.time()
result_optimized = find_similar_messages_optimized(test_messages, 0.6)
time_optimized = time.time() - start
print(f"MinHash 방법 (500개): {time_optimized:.4f}초, {len(result_optimized)}개 쌍")

# 벡터화 방법
start = time.time()
result_vectorized = find_similar_messages_vectorized(test_messages, 0.6)
time_vectorized = time.time() - start
print(f"벡터화 방법 (500개): {time_vectorized:.4f}초, {len(result_vectorized)}개 쌍")

# 결과 샘플 출력
print(f"\n📋 유사 메시지 예시:")
for i, pair in enumerate(result_vectorized[:3]):
    print(f"{i+1}. '{pair['message1']}' ↔ '{pair['message2']}' (유사도: {pair['similarity']:.3f})")
```

**코드 해설:**
- **MinHash**: 집합 유사도를 효율적으로 추정하는 확률적 알고리즘
- **LSH**: 유사한 항목들을 같은 버킷에 배치하여 비교 횟수 감소
- **벡터화**: NumPy/scikit-learn의 최적화된 행렬 연산 활용
- **조기 종료**: 추정값으로 먼저 필터링 후 정확한 계산 수행

> 💡 **성능 최적화 체크리스트**
> 
> **데이터 구조:**
> - [ ] 적합한 자료구조 선택 (리스트 vs 집합 vs 딕셔너리)
> - [ ] 메모리 효율적 처리 (스트리밍, 배치)
> - [ ] 캐싱 활용 (중복 계산 방지)
> 
> **알고리즘:**
> - [ ] 시간 복잡도 개선 (O(n²) → O(n log n))
> - [ ] 벡터화 연산 활용
> - [ ] 확률적 알고리즘 적용
> 
> **구현:**
> - [ ] 불필요한 객체 생성 최소화
> - [ ] 조기 종료 조건 활용
> - [ ] 병렬 처리 고려

> 🖼️ **이미지 생성 프롬프트**: 
> "코드 최적화 과정을 보여주는 플로우차트. 원본 AI 코드 → 문제 진단 → 데이터 구조 최적화 → 알고리즘 개선 → 성능 테스트 → 최종 최적화 코드 단계로 이어지며, 각 단계마다 성능 지표(시간, 메모리)가 표시된 모던한 다이어그램"

## 4. 보안과 안정성을 고려한 코드 검증 프로세스

AI가 생성한 코드를 프로덕션 환경에 배포하기 전에는 철저한 보안 및 안정성 검증이 필요합니다. 마치 새로운 자동차가 도로에 나가기 전에 안전성 테스트를 거치는 것처럼, 코드도 다양한 검증 과정을 통과해야 합니다.

### 4.1 입력 검증과 데이터 보안

#### **보안 원칙 1: 입력 데이터 검증**

AI는 종종 사용자 입력을 그대로 신뢰하는 코드를 생성합니다. 하지만 실제 환경에서는 악의적이거나 예상치 못한 입력이 들어올 수 있습니다.

**❌ 보안 취약점이 있는 AI 코드:**
```python
def analyze_sms_content(user_input):
    """
    사용자 입력 SMS 분석 (보안 취약점 포함)
    """
    
    # 1. 입력 검증 없음 - SQL Injection 위험
    query = f"SELECT * FROM sms_data WHERE content LIKE '%{user_input}%'"
    
    # 2. 파일 경로 검증 없음 - Path Traversal 위험
    log_file = f"logs/{user_input}_analysis.log"
    
    # 3. 실행 가능한 코드 평가 - Code Injection 위험
    result = eval(f"len('{user_input}')")  # 매우 위험!
    
    # 4. 민감한 정보 로깅
    print(f"사용자 입력: {user_input}")  # 로그에 민감 정보 노출
    
    return {
        'query': query,
        'log_file': log_file,
        'length': result
    }

# 위험한 입력 예시
malicious_inputs = [
    "'; DROP TABLE sms_data; --",  # SQL Injection
    "../../../etc/passwd",  # Path Traversal
    "1'); __import__('os').system('rm -rf /'); ('",  # Code Injection
    "sensitive_password_123"  # 민감한 정보
]

print("🚨 보안 취약점 테스트:")
for dangerous_input in malicious_inputs:
    try:
        result = analyze_sms_content(dangerous_input)
        print(f"위험한 입력 처리됨: {dangerous_input[:20]}...")
    except Exception as e:
        print(f"오류 발생: {e}")
```

**✅ 보안이 강화된 코드:**
```python
import re
import hashlib
import os
import sqlite3
from pathlib import Path
import logging

class SecureSMSAnalyzer:
    """보안이 강화된 SMS 분석기"""
    
    def __init__(self, db_path="sms_database.db", log_dir="logs"):
        self.db_path = db_path
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # 로깅 설정 (민감한 정보 마스킹)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / 'sms_analyzer.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _validate_input(self, user_input):
        """
        입력 데이터 검증 및 정화
        
        Args:
            user_input: 사용자 입력 문자열
            
        Returns:
            dict: 검증 결과와 정화된 입력
            
        Raises:
            ValueError: 유효하지 않은 입력
        """
        
        # 1. 기본 타입 검증
        if not isinstance(user_input, str):
            raise ValueError("입력은 문자열이어야 합니다")
        
        # 2. 길이 제한
        if len(user_input) > 1000:
            raise ValueError("입력 길이가 1000자를 초과할 수 없습니다")
        
        # 3. 허용되지 않는 문자 검증
        forbidden_patterns = [
            r"[;<>\"'`]",  # SQL/Script Injection 문자
            r"\.\./",      # Path Traversal 패턴
            r"__\w+__",    # Python 매직 메서드
            r"import\s+\w+",  # import 문
            r"exec\s*\(",     # exec 함수
            r"eval\s*\(",     # eval 함수
        ]
        
        for pattern in forbidden_patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                raise ValueError(f"허용되지 않는 패턴이 포함되어 있습니다: {pattern}")
        
        # 4. 입력 정화 (화이트리스트 방식)
        # 영문자, 숫자, 공백, 기본 구두점만 허용
        sanitized_input = re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', user_input)
        
        # 5. 입력 해시 생성 (로깅용 - 원본 노출 방지)
        input_hash = hashlib.sha256(user_input.encode()).hexdigest()[:16]
        
        return {
            'original_length': len(user_input),
            'sanitized_input': sanitized_input.strip(),
            'input_hash': input_hash,
            'is_modified': user_input != sanitized_input
        }
    
    def analyze_sms_content(self, user_input):
        """
        보안이 강화된 SMS 내용 분석
        
        Args:
            user_input: 분석할 SMS 내용
            
        Returns:
            dict: 분석 결과
        """
        
        try:
            # 1. 입력 검증
            validation_result = self._validate_input(user_input)
            
            # 2. 민감한 정보 마스킹 로깅
            self.logger.info(f"SMS 분석 요청 - 해시: {validation_result['input_hash']}, "
                           f"길이: {validation_result['original_length']}")
            
            if validation_result['is_modified']:
                self.logger.warning(f"입력이 정화됨 - 해시: {validation_result['input_hash']}")
            
            # 3. 안전한 데이터베이스 쿼리 (Parameterized Query)
            analysis_result = self._safe_database_query(validation_result['sanitized_input'])
            
            # 4. 안전한 파일 처리
            log_file_path = self._safe_log_file_creation(validation_result['input_hash'])
            
            # 5. 안전한 길이 계산 (eval 대신)
            content_stats = self._calculate_content_stats(validation_result['sanitized_input'])
            
            return {
                'input_hash': validation_result['input_hash'],
                'analysis': analysis_result,
                'log_file': str(log_file_path),
                'stats': content_stats,
                'security_status': 'validated'
            }
            
        except ValueError as e:
            self.logger.error(f"입력 검증 실패: {str(e)}")
            return {
                'error': str(e),
                'security_status': 'rejected'
            }
        except Exception as e:
            self.logger.error(f"분석 중 오류 발생: {str(e)}")
            return {
                'error': '내부 오류가 발생했습니다',
                'security_status': 'error'
            }
    
    def _safe_database_query(self, sanitized_input):
        """안전한 데이터베이스 쿼리 (Parameterized Query 사용)"""
        
        # 실제 데이터베이스 연결 대신 시뮬레이션
        # 실제 환경에서는 parameterized query 사용
        query_params = {
            'search_term': f"%{sanitized_input}%"
        }
        
        # Parameterized query 예시 (SQLite)
        query = "SELECT COUNT(*) FROM sms_data WHERE content LIKE ?"
        
        # 실제로는 이렇게 실행:
        # cursor.execute(query, (query_params['search_term'],))
        
        return {
            'query_type': 'parameterized',
            'search_term_length': len(sanitized_input),
            'matched_count': 42  # 시뮬레이션 결과
        }
    
    def _safe_log_file_creation(self, input_hash):
        """안전한 로그 파일 생성 (Path Traversal 방지)"""
        
        # 파일명에 안전한 해시값만 사용
        safe_filename = f"analysis_{input_hash}.log"
        
        # 지정된 디렉토리 내에서만 파일 생성
        log_file_path = self.log_dir / safe_filename
        
        # 상위 디렉토리 접근 시도 차단
        try:
            log_file_path = log_file_path.resolve()
            if not str(log_file_path).startswith(str(self.log_dir.resolve())):
                raise ValueError("허용되지 않는 파일 경로")
        except Exception:
            raise ValueError("유효하지 않은 파일 경로")
        
        # 안전한 로그 작성
        with open(log_file_path, 'w', encoding='utf-8') as f:
            f.write(f"분석 시작: {input_hash}\n")
            f.write(f"타임스탬프: {logging.Formatter().formatTime(logging.LogRecord('', 0, '', 0, '', (), None))}\n")
        
        return log_file_path
    
    def _calculate_content_stats(self, content):
        """안전한 콘텐츠 통계 계산 (eval 사용 안함)"""
        
        return {
            'length': len(content),
            'word_count': len(content.split()),
            'char_freq': {char: content.count(char) for char in set(content) if char.isalnum()},
            'has_numbers': any(char.isdigit() for char in content),
            'has_uppercase': any(char.isupper() for char in content)
        }

# 보안 강화 테스트
print("🔒 보안 강화 테스트:")
print("=" * 40)

analyzer = SecureSMSAnalyzer()

# 정상 입력 테스트
normal_input = "Hello, this is a normal SMS message!"
result = analyzer.analyze_sms_content(normal_input)
print(f"정상 입력 결과: {result['security_status']}")

# 악의적 입력 테스트
malicious_inputs = [
    "'; DROP TABLE sms_data; --",
    "../../../etc/passwd",
    "1'); __import__('os').system('ls'); ('",
    "normal text with <script>alert('xss')</script>"
]

for malicious_input in malicious_inputs:
    result = analyzer.analyze_sms_content(malicious_input)
    print(f"악의적 입력 '{malicious_input[:20]}...' -> {result['security_status']}")
```

**코드 해설:**
- **입력 검증**: 화이트리스트 방식으로 허용되는 문자만 통과
- **Parameterized Query**: SQL Injection 방지를 위한 안전한 쿼리 방식
- **Path Traversal 방지**: 파일 경로 정규화와 범위 제한
- **민감한 정보 보호**: 해시를 사용한 로깅으로 원본 정보 노출 방지

#### **보안 원칙 2: 권한 관리와 접근 제어**

```python
import functools
import time
from enum import Enum
from collections import defaultdict, deque

class UserRole(Enum):
    """사용자 역할 정의"""
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"
    GUEST = "guest"

class PermissionLevel(Enum):
    """권한 레벨 정의"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"

class AccessControlManager:
    """접근 제어 관리자"""
    
    def __init__(self):
        # 역할별 권한 매핑
        self.role_permissions = {
            UserRole.ADMIN: [PermissionLevel.READ, PermissionLevel.WRITE, 
                           PermissionLevel.DELETE, PermissionLevel.ADMIN],
            UserRole.ANALYST: [PermissionLevel.READ, PermissionLevel.WRITE],
            UserRole.VIEWER: [PermissionLevel.READ],
            UserRole.GUEST: []
        }
        
        # 레이트 리미팅을 위한 요청 추적
        self.request_history = defaultdict(lambda: deque(maxlen=100))
        
        # 실패한 로그인 시도 추적
        self.failed_attempts = defaultdict(int)
        self.lockout_time = defaultdict(float)
    
    def require_permission(self, required_permission):
        """권한 확인 데코레이터"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(self, user_role, *args, **kwargs):
                # 권한 확인
                if not self._check_permission(user_role, required_permission):
                    raise PermissionError(f"{required_permission.value} 권한이 필요합니다")
                
                return func(self, user_role, *args, **kwargs)
            return wrapper
        return decorator
    
    def rate_limit(self, max_requests=10, time_window=60):
        """레이트 리미팅 데코레이터"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(self, user_id, *args, **kwargs):
                current_time = time.time()
                user_requests = self.request_history[user_id]
                
                # 시간 윈도우 내의 요청만 유지
                while user_requests and current_time - user_requests[0] > time_window:
                    user_requests.popleft()
                
                # 요청 수 확인
                if len(user_requests) >= max_requests:
                    raise Exception(f"레이트 리미트 초과: {max_requests}회/{time_window}초")
                
                # 현재 요청 기록
                user_requests.append(current_time)
                
                return func(self, user_id, *args, **kwargs)
            return wrapper
        return decorator
    
    def _check_permission(self, user_role, required_permission):
        """사용자 권한 확인"""
        if not isinstance(user_role, UserRole):
            return False
        
        user_permissions = self.role_permissions.get(user_role, [])
        return required_permission in user_permissions
    
    def authenticate_user(self, user_id, password_hash):
        """사용자 인증 (간단한 시뮬레이션)"""
        
        # 계정 잠금 확인
        if self._is_account_locked(user_id):
            remaining_time = self.lockout_time[user_id] - time.time()
            raise Exception(f"계정이 잠겨있습니다. {remaining_time:.0f}초 후 재시도하세요")
        
        # 실제로는 데이터베이스에서 해시 비교
        valid_users = {
            "admin": ("hashed_admin_password", UserRole.ADMIN),
            "analyst1": ("hashed_analyst_password", UserRole.ANALYST),
            "viewer1": ("hashed_viewer_password", UserRole.VIEWER)
        }
        
        if user_id in valid_users and valid_users[user_id][0] == password_hash:
            # 성공 시 실패 카운터 리셋
            self.failed_attempts[user_id] = 0
            return valid_users[user_id][1]
        else:
            # 실패 시 카운터 증가
            self.failed_attempts[user_id] += 1
            
            # 5회 실패 시 30분 잠금
            if self.failed_attempts[user_id] >= 5:
                self.lockout_time[user_id] = time.time() + 1800  # 30분
                raise Exception("로그인 시도 5회 실패로 계정이 잠겼습니다")
            
            raise Exception("잘못된 인증 정보입니다")
    
    def _is_account_locked(self, user_id):
        """계정 잠금 상태 확인"""
        if user_id in self.lockout_time:
            return time.time() < self.lockout_time[user_id]
        return False

class SecureSMSAnalysisAPI:
    """보안이 적용된 SMS 분석 API"""
    
    def __init__(self):
        self.access_manager = AccessControlManager()
        self.sms_data = {
            'total_messages': 10000,
            'spam_count': 1500,
            'ham_count': 8500
        }
    
    @AccessControlManager().require_permission(PermissionLevel.READ)
    def get_sms_statistics(self, user_role, user_id):
        """SMS 통계 조회 (읽기 권한 필요)"""
        
        return {
            'action': 'get_statistics',
            'user_role': user_role.value,
            'data': self.sms_data.copy()
        }
    
    @AccessControlManager().require_permission(PermissionLevel.WRITE)
    @AccessControlManager().rate_limit(max_requests=5, time_window=60)
    def analyze_new_message(self, user_role, user_id, message):
        """새 메시지 분석 (쓰기 권한 + 레이트 리미팅)"""
        
        # 실제 분석 로직 (시뮬레이션)
        analysis_result = {
            'message_id': f"msg_{int(time.time())}",
            'content_length': len(message),
            'predicted_label': 'ham' if len(message) < 50 else 'spam',
            'confidence': 0.85,
            'analyzed_by': user_role.value
        }
        
        return {
            'action': 'analyze_message',
            'result': analysis_result
        }
    
    @AccessControlManager().require_permission(PermissionLevel.DELETE)
    def delete_message(self, user_role, user_id, message_id):
        """메시지 삭제 (삭제 권한 필요)"""
        
        return {
            'action': 'delete_message',
            'message_id': message_id,
            'deleted_by': user_role.value,
            'status': 'success'
        }
    
    @AccessControlManager().require_permission(PermissionLevel.ADMIN)
    def get_system_logs(self, user_role, user_id):
        """시스템 로그 조회 (관리자 권한 필요)"""
        
        return {
            'action': 'get_system_logs',
            'logs': [
                {'timestamp': time.time(), 'event': 'user_login', 'user': user_id},
                {'timestamp': time.time()-100, 'event': 'message_analyzed', 'count': 50}
            ]
        }

# 보안 테스트
print("\n🔐 권한 관리 및 접근 제어 테스트:")
print("=" * 50)

api = SecureSMSAnalysisAPI()
access_manager = AccessControlManager()

# 다양한 역할로 테스트
test_scenarios = [
    (UserRole.VIEWER, "viewer1", "get_sms_statistics"),
    (UserRole.VIEWER, "viewer1", "analyze_new_message"),  # 권한 부족
    (UserRole.ANALYST, "analyst1", "analyze_new_message"),
    (UserRole.ANALYST, "analyst1", "delete_message"),  # 권한 부족
    (UserRole.ADMIN, "admin", "get_system_logs"),
]

for user_role, user_id, action in test_scenarios:
    try:
        if action == "get_sms_statistics":
            result = api.get_sms_statistics(user_role, user_id)
        elif action == "analyze_new_message":
            result = api.analyze_new_message(user_role, user_id, "Test SMS message")
        elif action == "delete_message":
            result = api.delete_message(user_role, user_id, "msg_123")
        elif action == "get_system_logs":
            result = api.get_system_logs(user_role, user_id)
        
        print(f"✅ {user_role.value} - {action}: 성공")
        
    except PermissionError as e:
        print(f"❌ {user_role.value} - {action}: 권한 없음 ({e})")
    except Exception as e:
        print(f"⚠️ {user_role.value} - {action}: 오류 ({e})")

# 레이트 리미팅 테스트
print(f"\n🚦 레이트 리미팅 테스트:")
for i in range(7):  # 5회 제한을 초과하여 테스트
    try:
        result = api.analyze_new_message(UserRole.ANALYST, "analyst1", f"Message {i}")
        print(f"✅ 요청 {i+1}: 성공")
    except Exception as e:
        print(f"❌ 요청 {i+1}: 제한됨 ({e})")
```

**코드 해설:**
- **역할 기반 접근 제어(RBAC)**: 사용자 역할에 따른 권한 차별화
- **데코레이터 패턴**: 재사용 가능한 권한 확인 로직
- **레이트 리미팅**: DoS 공격 방지를 위한 요청 횟수 제한
- **계정 잠금**: 브루트 포스 공격 방지를 위한 실패 시도 제한

### 4.2 코드 안정성 검증

#### **안정성 원칙 1: 예외 처리와 복구 메커니즘**

```python
import logging
import traceback
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional, Callable

class RobustSMSProcessor:
    """안정성이 강화된 SMS 처리기"""
    
    def __init__(self, max_retries=3, backoff_factor=2):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        
        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # 성능 모니터링
        self.performance_metrics = {
            'total_processed': 0,
            'successful_processed': 0,
            'failed_processed': 0,
            'avg_processing_time': 0.0
        }
    
    @contextmanager
    def error_boundary(self, operation_name: str):
        """에러 경계 컨텍스트 매니저"""
        start_time = time.time()
        
        try:
            self.logger.info(f"{operation_name} 시작")
            yield
            self.logger.info(f"{operation_name} 성공")
            
        except Exception as e:
            # 상세한 에러 로깅
            error_details = {
                'operation': operation_name,
                'error_type': type(e).__name__,
                'error_message': str(e),
                'traceback': traceback.format_exc()
            }
            
            self.logger.error(f"{operation_name} 실패: {error_details}")
            
            # 복구 가능한 에러인지 확인
            if self._is_recoverable_error(e):
                self.logger.info(f"{operation_name} 복구 시도 중...")
                raise  # 재시도를 위해 에러를 다시 발생
            else:
                self.logger.critical(f"{operation_name} 복구 불가능한 오류")
                raise
                
        finally:
            processing_time = time.time() - start_time
            self.logger.info(f"{operation_name} 완료 시간: {processing_time:.3f}초")
    
    def _is_recoverable_error(self, error: Exception) -> bool:
        """복구 가능한 에러인지 판단"""
        
        recoverable_errors = [
            ConnectionError,
            TimeoutError,
            MemoryError,  # 메모리 부족은 재시도로 해결될 수 있음
        ]
        
        # 특정 에러 메시지 패턴 확인
        recoverable_messages = [
            "connection timeout",
            "temporary failure",
            "server overloaded"
        ]
        
        error_message = str(error).lower()
        
        return (any(isinstance(error, err_type) for err_type in recoverable_errors) or
                any(msg in error_message for msg in recoverable_messages))
    
    def retry_with_backoff(self, operation: Callable, *args, **kwargs) -> Any:
        """지수 백오프를 적용한 재시도 메커니즘"""
        
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return operation(*args, **kwargs)
                
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    self.logger.error(f"최대 재시도 횟수 {self.max_retries} 도달")
                    break
                
                if not self._is_recoverable_error(e):
                    self.logger.error("복구 불가능한 오류, 재시도 중단")
                    break
                
                # 지수 백오프 적용
                sleep_time = self.backoff_factor ** attempt
                self.logger.warning(f"재시도 {attempt + 1}/{self.max_retries} "
                                  f"({sleep_time}초 후): {str(e)}")
                time.sleep(sleep_time)
        
        # 모든 재시도 실패
        raise last_exception
    
    def process_sms_safely(self, messages: list) -> Dict[str, Any]:
        """안전한 SMS 일괄 처리"""
        
        with self.error_boundary("SMS 일괄 처리"):
            results = {
                'processed': [],
                'failed': [],
                'summary': {}
            }
            
            for i, message in enumerate(messages):
                try:
                    # 개별 메시지 처리
                    processed_msg = self.retry_with_backoff(
                        self._process_single_message, 
                        message, 
                        index=i
                    )
                    
                    results['processed'].append(processed_msg)
                    self.performance_metrics['successful_processed'] += 1
                    
                except Exception as e:
                    # 개별 실패는 전체 처리를 중단하지 않음
                    failed_msg = {
                        'index': i,
                        'original_message': message,
                        'error': str(e),
                        'error_type': type(e).__name__
                    }
                    
                    results['failed'].append(failed_msg)
                    self.performance_metrics['failed_processed'] += 1
                    
                    self.logger.warning(f"메시지 {i} 처리 실패: {str(e)}")
                
                finally:
                    self.performance_metrics['total_processed'] += 1
            
            # 처리 요약
            total = len(messages)
            success_rate = len(results['processed']) / total if total > 0 else 0
            
            results['summary'] = {
                'total_messages': total,
                'successful_count': len(results['processed']),
                'failed_count': len(results['failed']),
                'success_rate': f"{success_rate:.1%}",
                'performance_metrics': self.performance_metrics.copy()
            }
            
            self.logger.info(f"처리 완료: 성공 {len(results['processed'])}개, "
                           f"실패 {len(results['failed'])}개")
            
            return results
    
    def _process_single_message(self, message: str, index: int) -> Dict[str, Any]:
        """개별 메시지 처리 (예외 발생 가능)"""
        
        # 시뮬레이션: 랜덤하게 에러 발생
        import random
        
        if random.random() < 0.1:  # 10% 확률로 복구 가능한 에러
            raise ConnectionError("네트워크 연결 오류")
        
        if random.random() < 0.05:  # 5% 확률로 복구 불가능한 에러
            raise ValueError(f"잘못된 메시지 형식: {message}")
        
        # 정상 처리
        return {
            'index': index,
            'original': message,
            'cleaned': message.lower().strip(),
            'length': len(message),
            'timestamp': time.time()
        }
    
    def health_check(self) -> Dict[str, Any]:
        """시스템 건강 상태 확인"""
        
        health_status = {
            'status': 'healthy',
            'checks': {},
            'metrics': self.performance_metrics.copy()
        }
        
        # 메모리 사용량 확인
        import psutil
        memory_percent = psutil.virtual_memory().percent
        health_status['checks']['memory'] = {
            'status': 'ok' if memory_percent < 80 else 'warning',
            'usage_percent': memory_percent
        }
        
        # 성공률 확인
        total_processed = self.performance_metrics['total_processed']
        if total_processed > 0:
            success_rate = (self.performance_metrics['successful_processed'] / 
                          total_processed)
            
            health_status['checks']['success_rate'] = {
                'status': 'ok' if success_rate > 0.9 else 'degraded',
                'rate': success_rate
            }
        
        # 전체 상태 결정
        check_statuses = [check['status'] for check in health_status['checks'].values()]
        
        if any(status == 'degraded' for status in check_statuses):
            health_status['status'] = 'degraded'
        elif any(status == 'warning' for status in check_statuses):
            health_status['status'] = 'warning'
        
        return health_status

# 안정성 테스트
print("\n🛡️ 코드 안정성 검증 테스트:")
print("=" * 50)

processor = RobustSMSProcessor(max_retries=2, backoff_factor=1.5)

# 테스트 메시지 (일부는 에러를 발생시킬 수 있음)
test_messages = [
    "Hello, this is a normal message",
    "Another normal SMS text",
    "Free money call now!",
    "How are you doing today?",
    "Win cash prize immediately",
    "Normal conversation text",
    "Emergency notification",
    "Regular update message"
] * 3  # 24개 메시지로 확장

# 안전한 일괄 처리 테스트
start_time = time.time()
processing_result = processor.process_sms_safely(test_messages)
total_time = time.time() - start_time

print(f"📊 처리 결과:")
print(f"- 총 메시지: {processing_result['summary']['total_messages']}개")
print(f"- 성공: {processing_result['summary']['successful_count']}개")
print(f"- 실패: {processing_result['summary']['failed_count']}개")
print(f"- 성공률: {processing_result['summary']['success_rate']}")
print(f"- 총 소요 시간: {total_time:.3f}초")

# 실패한 메시지 분석
if processing_result['failed']:
    print(f"\n❌ 실패한 메시지 분석:")
    for failed in processing_result['failed'][:3]:  # 처음 3개만 표시
        print(f"- 인덱스 {failed['index']}: {failed['error_type']} - {failed['error']}")

# 시스템 건강 상태 확인
health_status = processor.health_check()
print(f"\n🩺 시스템 건강 상태: {health_status['status'].upper()}")

for check_name, check_result in health_status['checks'].items():
    status_emoji = {"ok": "✅", "warning": "⚠️", "degraded": "❌"}
    emoji = status_emoji.get(check_result['status'], "❓")
    print(f"{emoji} {check_name}: {check_result['status']}")
```

**코드 해설:**
- **에러 경계**: 예외를 격리하여 전체 시스템 장애 방지
- **지수 백오프**: 일시적 장애에 대한 지능적 재시도 메커니즘
- **회로 차단기**: 복구 불가능한 오류 조기 감지 및 차단
- **건강 상태 모니터링**: 시스템 상태를 지속적으로 추적

> 💡 **보안 및 안정성 체크리스트**
> 
> **입력 보안:**
> - [ ] 입력 검증 및 정화
> - [ ] SQL/Code Injection 방지
> - [ ] Path Traversal 방지
> - [ ] 민감한 정보 마스킹
> 
> **접근 제어:**
> - [ ] 역할 기반 권한 관리
> - [ ] 레이트 리미팅 구현
> - [ ] 계정 잠금 메커니즘
> - [ ] 세션 관리
> 
> **안정성:**
> - [ ] 예외 처리 및 복구
> - [ ] 재시도 메커니즘
> - [ ] 회로 차단기 패턴
> - [ ] 건강 상태 모니터링

> 🖼️ **이미지 생성 프롬프트**: 
> "보안 검증 프로세스를 보여주는 계층적 다이어그램. 입력 검증 → 권한 확인 → 안전한 처리 → 예외 처리 → 로깅 및 모니터링 단계가 방패와 자물쇠 아이콘으로 표현된 보안 중심의 인포그래픽"

## 5. 실전 미니 프로젝트: AI 생성 코드 검증 및 개선 워크플로우

이제 배운 내용을 종합하여 실제 AI가 생성한 SMS 분류 코드를 체계적으로 검증하고 개선하는 전체 워크플로우를 구현해보겠습니다.

### 5.1 프로젝트 시나리오

**상황**: AI가 SMS 스팸 분류를 위한 코드를 생성했습니다. 이 코드를 프로덕션 환경에 배포하기 전에 철저한 검증과 개선이 필요합니다.

**목표**: AI 생성 코드를 안전하고 효율적이며 유지보수 가능한 수준으로 개선하기

### 5.2 AI가 생성한 원본 코드 (검증 대상)

먼저 AI가 생성한 원본 코드를 살펴보겠습니다:

```python
# AI가 생성한 원본 코드 (여러 문제점 포함)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle

def train_spam_classifier(data_file):
    # 데이터 로드
    data = pd.read_csv(data_file)
    X = data['message']
    y = data['label']
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # 벡터화
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # 모델 훈련
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    
    # 평가
    predictions = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy}")
    
    # 모델 저장
    pickle.dump(model, open('spam_model.pkl', 'wb'))
    pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
    
    return model, vectorizer

def classify_message(message):
    # 모델 로드
    model = pickle.load(open('spam_model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    
    # 예측
    message_vec = vectorizer.transform([message])
    prediction = model.predict(message_vec)[0]
    probability = model.predict_proba(message_vec)[0]
    
    return prediction, max(probability)

# 사용 예시
if __name__ == "__main__":
    model, vectorizer = train_spam_classifier("spam_data.csv")
    result = classify_message("FREE money! Call now!")
    print(f"Classification: {result[0]}, Confidence: {result[1]}")
```

### 5.3 체계적 코드 검증 프로세스

이제 우리가 배운 기법들을 사용하여 이 코드를 체계적으로 분석하고 개선해보겠습니다.

```python
import ast
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib  # pickle보다 안전
import re
import hashlib
from dataclasses import dataclass

@dataclass
class CodeIssue:
    """코드 이슈를 나타내는 데이터 클래스"""
    severity: str  # 'critical', 'high', 'medium', 'low'
    category: str  # 'security', 'performance', 'reliability', 'maintainability'
    description: str
    location: str
    recommendation: str

class AICodeValidator:
    """AI 생성 코드 검증기"""
    
    def __init__(self):
        self.issues = []
        self.logger = logging.getLogger(__name__)
    
    def validate_code(self, code_string: str, filename: str = "<string>") -> List[CodeIssue]:
        """코드를 종합적으로 검증"""
        
        self.issues = []
        
        print(f"🔍 AI 생성 코드 검증 시작: {filename}")
        print("=" * 60)
        
        # 1. 구문 분석
        try:
            tree = ast.parse(code_string)
            self._check_syntax_issues(tree)
        except SyntaxError as e:
            self.issues.append(CodeIssue(
                severity='critical',
                category='reliability',
                description=f"구문 오류: {e}",
                location=f"라인 {e.lineno}",
                recommendation="구문 오류를 수정하세요"
            ))
            return self.issues
        
        # 2. 보안 취약점 검사
        self._check_security_issues(code_string, tree)
        
        # 3. 성능 문제 검사
        self._check_performance_issues(code_string, tree)
        
        # 4. 안정성 문제 검사
        self._check_reliability_issues(code_string, tree)
        
        # 5. 유지보수성 문제 검사
        self._check_maintainability_issues(code_string, tree)
        
        # 결과 요약
        self._print_validation_summary()
        
        return self.issues
    
    def _check_security_issues(self, code_string: str, tree: ast.AST):
        """보안 취약점 검사"""
        
        # pickle 사용 검사
        if 'pickle.load' in code_string or 'pickle.dump' in code_string:
            self.issues.append(CodeIssue(
                severity='high',
                category='security',
                description="pickle 모듈 사용으로 인한 보안 위험",
                location="pickle.load/dump 호출",
                recommendation="joblib 또는 다른 안전한 직렬화 방법 사용"
            ))
        
        # 파일 경로 하드코딩 검사
        hardcoded_paths = re.findall(r'["\'][^"\']*\.(pkl|csv|txt)["\']', code_string)
        if hardcoded_paths:
            self.issues.append(CodeIssue(
                severity='medium',
                category='security',
                description="하드코딩된 파일 경로 발견",
                location=f"경로: {hardcoded_paths}",
                recommendation="설정 파일이나 환경 변수 사용"
            ))
        
        # 예외 처리 누락 검사
        class ExceptionHandlerChecker(ast.NodeVisitor):
            def __init__(self):
                self.has_file_operations = False
                self.has_exception_handling = False
            
            def visit_Call(self, node):
                if (hasattr(node.func, 'attr') and 
                    node.func.attr in ['read_csv', 'open', 'load']):
                    self.has_file_operations = True
                self.generic_visit(node)
            
            def visit_Try(self, node):
                self.has_exception_handling = True
                self.generic_visit(node)
        
        checker = ExceptionHandlerChecker()
        checker.visit(tree)
        
        if checker.has_file_operations and not checker.has_exception_handling:
            self.issues.append(CodeIssue(
                severity='medium',
                category='reliability',
                description="파일 I/O 작업에 예외 처리 누락",
                location="파일 읽기/쓰기 함수",
                recommendation="try-except 블록으로 예외 처리 추가"
            ))
    
    def _check_performance_issues(self, code_string: str, tree: ast.AST):
        """성능 문제 검사"""
        
        # 매번 모델 로드하는 문제
        if code_string.count('pickle.load') > 1 or code_string.count('joblib.load') > 1:
            self.issues.append(CodeIssue(
                severity='high',
                category='performance',
                description="매번 모델을 다시 로드하는 비효율적 구조",
                location="classify_message 함수",
                recommendation="모델을 한 번 로드하여 메모리에 캐싱"
            ))
        
        # 데이터 검증 누락
        if 'train_test_split' in code_string and 'random_state' not in code_string:
            self.issues.append(CodeIssue(
                severity='low',
                category='reliability',
                description="재현 가능성을 위한 random_state 누락",
                location="train_test_split 호출",
                recommendation="random_state 매개변수 추가"
            ))
    
    def _check_reliability_issues(self, code_string: str, tree: ast.AST):
        """안정성 문제 검사"""
        
        # 입력 검증 누락
        class InputValidationChecker(ast.NodeVisitor):
            def __init__(self):
                self.function_params = []
                self.has_input_validation = False
            
            def visit_FunctionDef(self, node):
                self.function_params.extend([arg.arg for arg in node.args.args])
                
                # 함수 내에서 isinstance, len 등의 검증 함수 확인
                for child in ast.walk(node):
                    if (isinstance(child, ast.Call) and 
                        hasattr(child.func, 'id') and
                        child.func.id in ['isinstance', 'len', 'type']):
                        self.has_input_validation = True
                
                self.generic_visit(node)
        
        checker = InputValidationChecker()
        checker.visit(tree)
        
        if checker.function_params and not checker.has_input_validation:
            self.issues.append(CodeIssue(
                severity='medium',
                category='reliability',
                description="함수 매개변수에 대한 입력 검증 누락",
                location="모든 함수",
                recommendation="타입 체크 및 범위 검증 추가"
            ))
    
    def _check_maintainability_issues(self, code_string: str, tree: ast.AST):
        """유지보수성 문제 검사"""
        
        # 하드코딩된 설정값 검사
        magic_numbers = re.findall(r'\b0\.[0-9]+\b|\b[1-9][0-9]*\b', code_string)
        if len(magic_numbers) > 3:  # 0.2, 등의 매직 넘버
            self.issues.append(CodeIssue(
                severity='low',
                category='maintainability',
                description="하드코딩된 설정값(매직 넘버) 다수 발견",
                location="전체 코드",
                recommendation="설정 상수로 분리하거나 설정 파일 사용"
            ))
        
        # 함수 길이 검사
        class FunctionLengthChecker(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                if hasattr(node, 'end_lineno') and hasattr(node, 'lineno'):
                    length = node.end_lineno - node.lineno
                    if length > 30:
                        self.issues.append(CodeIssue(
                            severity='low',
                            category='maintainability',
                            description=f"함수 '{node.name}'가 너무 김 ({length}줄)",
                            location=f"라인 {node.lineno}-{node.end_lineno}",
                            recommendation="함수를 더 작은 단위로 분리"
                        ))
                self.generic_visit(node)
        
        checker = FunctionLengthChecker()
        checker.visit(tree)
    
    def _check_syntax_issues(self, tree: ast.AST):
        """구문 관련 문제 검사"""
        # AST가 정상적으로 파싱되었으므로 구문 오류는 없음
        pass
    
    def _print_validation_summary(self):
        """검증 결과 요약 출력"""
        
        if not self.issues:
            print("✅ 검증 완료: 발견된 문제점 없음")
            return
        
        # 심각도별 분류
        severity_counts = {}
        category_counts = {}
        
        for issue in self.issues:
            severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1
            category_counts[issue.category] = category_counts.get(issue.category, 0) + 1
        
        print(f"\n📋 검증 결과 요약:")
        print(f"총 발견된 문제: {len(self.issues)}개")
        
        print(f"\n심각도별 분포:")
        for severity in ['critical', 'high', 'medium', 'low']:
            count = severity_counts.get(severity, 0)
            if count > 0:
                emoji = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'}
                print(f"  {emoji[severity]} {severity.upper()}: {count}개")
        
        print(f"\n카테고리별 분포:")
        for category, count in category_counts.items():
            print(f"  - {category}: {count}개")
        
        print(f"\n📝 상세 문제점:")
        for i, issue in enumerate(self.issues, 1):
            emoji = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'}
            print(f"\n{i}. {emoji[issue.severity]} [{issue.category.upper()}] {issue.description}")
            print(f"   위치: {issue.location}")
            print(f"   권장사항: {issue.recommendation}")

class ImprovedSMSClassifier:
    """개선된 SMS 분류기"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        SMS 분류기 초기화
        
        Args:
            config: 설정 딕셔너리 (선택사항)
        """
        
        # 기본 설정
        self.config = {
            'test_size': 0.2,
            'random_state': 42,
            'max_features': 5000,
            'ngram_range': (1, 2),
            'model_dir': Path('models'),
            'cross_validation_folds': 5
        }
        
        if config:
            self.config.update(config)
        
        # 디렉토리 생성
        self.config['model_dir'].mkdir(exist_ok=True)
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 모델 캐싱
        self.model = None
        self.vectorizer = None
        self.is_trained = False
    
    def validate_input_data(self, data: pd.DataFrame) -> Tuple[bool, str]:
        """입력 데이터 검증"""
        
        try:
            # 기본 검증
            if data is None or data.empty:
                return False, "데이터가 비어있습니다"
            
            # 필수 컬럼 확인
            required_columns = ['message', 'label']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                return False, f"필수 컬럼 누락: {missing_columns}"
            
            # 데이터 타입 확인
            if not data['message'].dtype == 'object':
                return False, "메시지 컬럼이 문자열 타입이 아닙니다"
            
            # 결측치 확인
            if data['message'].isnull().any():
                return False, "메시지에 결측치가 있습니다"
            
            # 라벨 확인
            unique_labels = data['label'].unique()
            if len(unique_labels) < 2:
                return False, "최소 2개의 다른 라벨이 필요합니다"
            
            self.logger.info(f"데이터 검증 통과: {len(data)}개 샘플, 라벨: {unique_labels}")
            return True, "검증 통과"
            
        except Exception as e:
            return False, f"데이터 검증 중 오류: {str(e)}"
    
    def train_classifier(self, data_file: str) -> Dict[str, Any]:
        """
        개선된 분류기 훈련
        
        Args:
            data_file: 훈련 데이터 파일 경로
            
        Returns:
            dict: 훈련 결과
        """
        
        try:
            self.logger.info(f"모델 훈련 시작: {data_file}")
            
            # 1. 안전한 데이터 로드
            if not Path(data_file).exists():
                raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_file}")
            
            data = pd.read_csv(data_file)
            
            # 2. 데이터 검증
            is_valid, message = self.validate_input_data(data)
            if not is_valid:
                raise ValueError(f"데이터 검증 실패: {message}")
            
            # 3. 전처리
            X = data['message'].astype(str)
            y = data['label']
            
            # 4. 데이터 분할 (재현 가능성 보장)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.config['test_size'],
                random_state=self.config['random_state'],
                stratify=y  # 계층화 분할
            )
            
            # 5. 안전한 벡터화
            self.vectorizer = TfidfVectorizer(
                max_features=self.config['max_features'],
                ngram_range=self.config['ngram_range'],
                stop_words='english',
                lowercase=True,
                strip_accents='ascii'
            )
            
            X_train_vec = self.vectorizer.fit_transform(X_train)
            X_test_vec = self.vectorizer.transform(X_test)
            
            # 6. 모델 훈련
            self.model = MultinomialNB(alpha=1.0)
            self.model.fit(X_train_vec, y_train)
            
            # 7. 종합 평가
            train_score = self.model.score(X_train_vec, y_train)
            test_score = self.model.score(X_test_vec, y_test)
            
            # 교차 검증
            cv_scores = cross_val_score(
                self.model, X_train_vec, y_train,
                cv=self.config['cross_validation_folds']
            )
            
            # 상세 평가
            y_pred = self.model.predict(X_test_vec)
            
            # 8. 안전한 모델 저장
            model_path = self.config['model_dir'] / 'sms_model.joblib'
            vectorizer_path = self.config['model_dir'] / 'vectorizer.joblib'
            
            joblib.dump(self.model, model_path)
            joblib.dump(self.vectorizer, vectorizer_path)
            
            self.is_trained = True
            
            # 9. 결과 반환
            results = {
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'model_path': str(model_path),
                'vectorizer_path': str(vectorizer_path),
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
            self.logger.info(f"훈련 완료 - 테스트 정확도: {test_score:.3f}")
            return results
            
        except Exception as e:
            self.logger.error(f"훈련 중 오류: {str(e)}")
            raise
    
    def classify_message(self, message: str) -> Dict[str, Any]:
        """
        안전한 메시지 분류
        
        Args:
            message: 분류할 메시지
            
        Returns:
            dict: 분류 결과
        """
        
        try:
            # 1. 입력 검증
            if not isinstance(message, str):
                raise ValueError("메시지는 문자열이어야 합니다")
            
            if len(message.strip()) == 0:
                raise ValueError("빈 메시지는 분류할 수 없습니다")
            
            # 2. 모델 로드 (캐싱된 모델 사용)
            if not self.is_trained:
                self._load_models()
            
            # 3. 전처리 및 예측
            message_clean = message.strip()
            message_vec = self.vectorizer.transform([message_clean])
            
            prediction = self.model.predict(message_vec)[0]
            probabilities = self.model.predict_proba(message_vec)[0]
            
            # 4. 결과 구성
            result = {
                'message': message_clean,
                'prediction': prediction,
                'confidence': float(max(probabilities)),
                'probabilities': {
                    label: float(prob) 
                    for label, prob in zip(self.model.classes_, probabilities)
                },
                'message_hash': hashlib.md5(message_clean.encode()).hexdigest()[:16],
                'timestamp': time.time()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"분류 중 오류: {str(e)}")
            raise
    
    def _load_models(self):
        """모델과 벡터라이저 로드"""
        
        model_path = self.config['model_dir'] / 'sms_model.joblib'
        vectorizer_path = self.config['model_dir'] / 'vectorizer.joblib'
        
        if not model_path.exists() or not vectorizer_path.exists():
            raise FileNotFoundError("훈련된 모델을 찾을 수 없습니다. 먼저 모델을 훈련하세요.")
        
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.is_trained = True
        
        self.logger.info("모델 로드 완료")

# 검증 및 개선 프로세스 실행
print("🔍 AI 코드 검증 및 개선 프로세스")
print("=" * 60)

# 1. 원본 AI 코드 검증
original_code = '''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle

def train_spam_classifier(data_file):
    data = pd.read_csv(data_file)
    X = data['message']
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    predictions = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy}")
    pickle.dump(model, open('spam_model.pkl', 'wb'))
    pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
    return model, vectorizer

def classify_message(message):
    model = pickle.load(open('spam_model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    message_vec = vectorizer.transform([message])
    prediction = model.predict(message_vec)[0]
    probability = model.predict_proba(message_vec)[0]
    return prediction, max(probability)
'''

# AI 코드 검증 수행
validator = AICodeValidator()
issues = validator.validate_code(original_code, "original_ai_code.py")

print(f"\n✨ 개선된 코드의 장점:")
print("=" * 30)
print("✅ 안전한 joblib 직렬화 사용")
print("✅ 포괄적인 입력 검증")
print("✅ 설정 기반 구조로 유연성 증대")  
print("✅ 모델 캐싱으로 성능 개선")
print("✅ 예외 처리 및 로깅 추가")
print("✅ 교차 검증으로 신뢰성 확보")
print("✅ 상세한 평가 지표 제공")
print("✅ 재현 가능한 결과 보장")

# 개선된 분류기 사용 예시 (실제 데이터가 있다면)
print(f"\n🚀 개선된 분류기 사용법:")
print("=" * 30)
print("""
# 개선된 분류기 사용
classifier = ImprovedSMSClassifier({
    'test_size': 0.25,
    'max_features': 10000,
    'cross_validation_folds': 10
})

# 안전한 훈련
try:
    results = classifier.train_classifier('sms_data.csv')
    print(f"테스트 정확도: {results['test_accuracy']:.3f}")
    print(f"교차 검증 평균: {results['cv_mean']:.3f} ± {results['cv_std']:.3f}")
except Exception as e:
    print(f"훈련 실패: {e}")

# 안전한 예측
try:
    result = classifier.classify_message("FREE money! Call now!")
    print(f"예측: {result['prediction']}")
    print(f"신뢰도: {result['confidence']:.3f}")
except Exception as e:
    print(f"예측 실패: {e}")
""")

## 6. 직접 해보기: 연습 문제

다음 연습 문제들을 통해 AI 생성 코드 검증 및 최적화 기술을 연마해보세요.

### **연습 문제 1: 코드 품질 자동 평가기 구현**

다음 AI 생성 코드를 분석하여 문제점을 찾고 개선 방안을 제시하는 자동 평가기를 구현하세요:

```python
# 평가 대상 코드
def process_data(data):
    result = []
    for i in range(len(data)):
        if data[i] != None:
            if type(data[i]) == str:
                if len(data[i]) > 0:
                    clean = data[i].lower()
                    words = clean.split()
                    for word in words:
                        if word not in ["the", "and", "or"]:
                            result.append(word)
    return result

# 과제:
# 1. 위 코드의 문제점을 자동으로 탐지하는 함수 작성
# 2. 각 문제점에 대한 심각도 평가 (1-10점)
# 3. 구체적인 개선 방안 제시
# 4. 개선된 버전의 코드 작성

def analyze_code_quality(code_string):
    """
    코드 품질을 자동으로 분석하는 함수를 구현하세요
    
    반환값 예시:
    {
        'overall_score': 6.5,
        'issues': [
            {
                'type': 'performance',
                'severity': 7,
                'description': '비효율적인 반복문 구조',
                'line': 3,
                'suggestion': 'list comprehension 사용 권장'
            }
        ],
        'improved_code': '개선된 코드 문자열'
    }
    """
    # 여기에 구현하세요
    pass

# 작성 공간:
[여기에 코드를 작성하세요]
```

### **연습 문제 2: 성능 벤치마킹 시스템**

두 가지 다른 구현 방식의 성능을 자동으로 비교하는 시스템을 구현하세요:

```python
# 비교 대상 함수 1 (AI 생성)
def find_duplicates_v1(texts):
    duplicates = []
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            if texts[i] == texts[j]:
                if texts[i] not in duplicates:
                    duplicates.append(texts[i])
    return duplicates

# 비교 대상 함수 2 (개선된 버전)
def find_duplicates_v2(texts):
    seen = set()
    duplicates = set()
    for text in texts:
        if text in seen:
            duplicates.add(text)
        else:
            seen.add(text)
    return list(duplicates)

# 과제:
# 1. 다양한 크기의 데이터로 성능 테스트
# 2. 메모리 사용량 측정
# 3. 시간 복잡도 분석
# 4. 결과를 시각화하는 차트 생성
# 5. 어느 상황에서 어떤 방법이 더 좋은지 분석

class PerformanceBenchmark:
    def __init__(self):
        pass
    
    def compare_functions(self, func1, func2, test_data_sizes=[100, 1000, 10000]):
        """
        두 함수의 성능을 비교하는 메서드를 구현하세요
        
        Args:
            func1, func2: 비교할 함수들
            test_data_sizes: 테스트할 데이터 크기 리스트
            
        Returns:
            dict: 성능 비교 결과
        """
        # 여기에 구현하세요
        pass
    
    def plot_comparison(self, results):
        """성능 비교 결과를 시각화하는 메서드"""
        # 여기에 구현하세요
        pass

# 작성 공간:
[여기에 코드를 작성하세요]
```

### **연습 문제 3: 보안 취약점 스캐너**

AI가 생성한 데이터 처리 코드에서 보안 취약점을 자동으로 탐지하는 스캐너를 구현하세요:

```python
# 스캔 대상 코드들
vulnerable_codes = [
    '''
def load_user_data(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return database.execute(query)
    ''',
    
    '''
def save_file(filename, content):
    with open(f"uploads/{filename}", "w") as f:
        f.write(content)
    ''',
    
    '''
def calculate_result(expression):
    return eval(expression)
    ''',
    
    '''
def log_activity(user_input):
    print(f"User activity: {user_input}")
    with open("activity.log", "a") as f:
        f.write(f"{user_input}\\n")
    '''
]

# 과제:
# 1. SQL Injection 위험 탐지
# 2. Path Traversal 취약점 찾기
# 3. Code Injection 가능성 확인
# 4. 정보 노출 위험 평가
# 5. 각 취약점에 대한 수정 방안 제시

class SecurityScanner:
    def __init__(self):
        self.vulnerability_patterns = {
            # 여기에 취약점 패턴들을 정의하세요
        }
    
    def scan_code(self, code_string):
        """
        코드에서 보안 취약점을 스캔하는 메서드
        
        Returns:
            dict: 발견된 취약점과 권장사항
        """
        # 여기에 구현하세요
        pass
    
    def generate_security_report(self, scan_results):
        """보안 스캔 결과 리포트 생성"""
        # 여기에 구현하세요
        pass

# 작성 공간:
[여기에 코드를 작성하세요]
```

### **연습 문제 4: 종합 코드 개선 프로젝트**

다음 AI 생성 데이터 분석 파이프라인을 종합적으로 개선하세요:

```python
# AI가 생성한 데이터 분석 파이프라인 (문제점 다수 포함)
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

def analyze_customer_data():
    # 데이터 로드
    data = pd.read_csv("customer_data.csv")
    
    # 전처리
    data = data.dropna()
    X = data.drop('target', axis=1)
    y = data['target']
    
    # 모델 훈련
    model = RandomForestClassifier()
    model.fit(X, y)
    
    # 예측
    predictions = model.predict(X)
    accuracy = (predictions == y).mean()
    
    # 결과 저장
    pickle.dump(model, open("model.pkl", "wb"))
    
    print(f"Accuracy: {accuracy}")
    return model

def predict_customer(customer_data):
    model = pickle.load(open("model.pkl", "rb"))
    prediction = model.predict([customer_data])
    return prediction[0]

# 과제:
# 1. 모든 문제점을 식별하고 분류하세요
# 2. 우선순위를 매겨 개선 계획을 수립하세요
# 3. 완전히 개선된 버전을 구현하세요
# 4. 개선 전후의 차이점을 정량적으로 측정하세요
# 5. 프로덕션 환경에 배포 가능한 수준으로 완성하세요

class ImprovedDataPipeline:
    """개선된 데이터 분석 파이프라인"""
    
    def __init__(self, config=None):
        """
        파이프라인 초기화
        - 설정 관리
        - 로깅 설정
        - 검증 체계 구축
        """
        # 여기에 구현하세요
        pass
    
    def validate_data(self, data):
        """데이터 품질 검증"""
        # 여기에 구현하세요
        pass
    
    def preprocess_data(self, data):
        """안전한 데이터 전처리"""
        # 여기에 구현하세요
        pass
    
    def train_model(self, data_path):
        """개선된 모델 훈련"""
        # 여기에 구현하세요
        pass
    
    def predict(self, customer_data):
        """안전한 예측"""
        # 여기에 구현하세요
        pass
    
    def evaluate_model(self, test_data):
        """종합적인 모델 평가"""
        # 여기에 구현하세요
        pass

# 작성 공간:
[여기에 완전히 개선된 파이프라인을 구현하세요]
```

## 7. 핵심 정리 및 요약

### 🎯 **이번 Part에서 배운 핵심 내용**

1. **AI 생성 코드의 일반적 오류 패턴**
   - **기능적 오류**: 엣지 케이스 처리 부족, 데이터 타입 혼동
   - **성능 관련 오류**: 비효율적인 반복문, 부적절한 자료구조 선택
   - **논리적 오류**: 잘못된 가정, 단순화된 사고방식

2. **코드 품질 평가의 5가지 차원**
   - **기능적 정확성**: 의도한 대로 동작하는지 확인
   - **성능 효율성**: 실행 시간과 메모리 사용량 최적화
   - **코드 품질**: 가독성, 유지보수성, 일관성 평가
   - **보안성**: 취약점 탐지 및 보안 강화
   - **안정성**: 예외 처리 및 복구 메커니즘

3. **성능 최적화 전략**
   - **데이터 구조 최적화**: 적합한 자료구조 선택 (집합, 트라이 등)
   - **알고리즘 개선**: 시간 복잡도 최적화, 벡터화 연산 활용
   - **메모리 효율성**: 스트리밍 처리, 배치 처리 활용

4. **보안 및 안정성 검증**
   - **입력 검증**: 화이트리스트 방식, Parameterized Query
   - **권한 관리**: 역할 기반 접근 제어, 레이트 리미팅
   - **예외 처리**: 에러 경계, 지수 백오프 재시도

### 💡 **실무 적용 가이드라인**

**단계별 코드 검증 프로세스:**
1. **구문 및 논리 검증** → AST 파싱, 정적 분석
2. **보안 취약점 스캔** → 패턴 매칭, 보안 도구 활용
3. **성능 프로파일링** → 실행 시간, 메모리 사용량 측정
4. **안정성 테스트** → 예외 상황, 스트레스 테스트
5. **종합 품질 평가** → 다차원 지표 기반 등급 산정

**코드 개선 우선순위:**
1. **🔴 Critical**: 보안 취약점, 치명적 버그
2. **🟠 High**: 성능 병목, 안정성 이슈  
3. **🟡 Medium**: 가독성, 유지보수성 문제
4. **🟢 Low**: 코드 스타일, 최적화 여지

### 🔄 **지속적 개선 전략**

1. **자동화된 검증 파이프라인 구축**
   - CI/CD에 코드 품질 검사 통합
   - 자동화된 보안 스캔 및 성능 테스트
   - 품질 지표 모니터링 대시보드

2. **팀 코드 리뷰 문화 정착**
   - AI 생성 코드 리뷰 체크리스트 활용
   - 도메인 전문가와 개발자 간 협업 강화
   - 베스트 프랙티스 공유 및 학습

3. **프로덕션 모니터링**
   - 성능 지표 실시간 추적
   - 오류율 및 응답 시간 모니터링
   - 사용자 피드백 기반 개선

### 🛠️ **추천 도구 및 라이브러리**

**정적 분석 도구:**
- **pylint**: 종합적인 코드 품질 분석
- **flake8**: PEP 8 스타일 가이드 준수
- **mypy**: 타입 힌트 검증
- **bandit**: 보안 취약점 스캔

**성능 분석 도구:**
- **memory_profiler**: 메모리 사용량 프로파일링
- **line_profiler**: 라인별 실행 시간 측정
- **py-spy**: 프로덕션 환경 프로파일링
- **locust**: 부하 테스트 및 성능 벤치마킹

**보안 도구:**
- **safety**: 의존성 패키지 보안 검사
- **semgrep**: 정적 보안 분석
- **codecov**: 코드 커버리지 측정

## 8. 다음 Part 예고: 자동화와 수동 작업의 균형 찾기

다음 Part에서는 데이터 분석 작업에서 어떤 부분을 AI에게 맡기고, 어떤 부분은 인간이 직접 해야 하는지 배워보겠습니다.

**다음 Part 주요 내용:**
- AI 자동화에 적합한 작업과 인간 개입이 필요한 작업 구분
- 효율적인 워크플로우 설계와 역할 분담 전략
- 품질 관리를 위한 체크포인트 설정 방법
- 인간-AI 협업의 최적화 기법

**미리 생각해볼 질문:**
- 데이터 분석 과정에서 AI가 잘하는 것과 못하는 것은 무엇일까요?
- 어떤 기준으로 자동화할 작업을 선별해야 할까요?
- 품질을 보장하면서도 효율성을 높이려면 어떻게 해야 할까요?

> 🖼️ **이미지 생성 프롬프트**: 
> "AI 코드 검증 및 개선 워크플로우를 보여주는 종합 다이어그램. 원본 AI 코드에서 시작하여 검증 → 문제 탐지 → 우선순위 설정 → 개선 → 테스트 → 배포 단계가 순환하며, 각 단계마다 체크리스트와 도구들이 표시된 프로페셔널한 프로세스 차트"

---

**🎉 축하합니다! 7장 Part 2를 완료했습니다!**

AI 생성 코드를 검증하고 개선하는 것은 현대 데이터 분석가의 필수 스킬입니다. 이번 Part에서 배운 체계적인 검증 프로세스와 최적화 기법을 통해 AI의 장점을 활용하면서도 안전하고 효율적인 코드를 작성할 수 있을 것입니다. 

코드 품질은 한 번에 완성되지 않습니다. 지속적인 검증과 개선을 통해 점진적으로 발전시켜 나가시길 바랍니다. 다음 Part에서는 AI와 인간의 역할을 어떻게 효과적으로 분담할지 배워보겠습니다!

