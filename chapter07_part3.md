# 7장 Part 3: 자동화와 수동 작업의 균형 찾기
**부제: 효율성과 품질의 최적 균형점을 찾아서**

## 학습 목표
이 Part를 완료한 후, 여러분은 다음을 할 수 있게 됩니다:
- 데이터 분석 작업에서 자동화에 적합한 영역과 인간 개입이 필요한 영역을 구분할 수 있다
- 효율성과 품질을 동시에 보장하는 인간-AI 협업 워크플로우를 설계할 수 있다
- 품질 관리를 위한 체크포인트와 검증 체계를 수립할 수 있다
- 프로젝트 특성에 맞는 최적의 자동화 전략을 선택하고 구현할 수 있다

## 이번 Part 미리보기
AI 시대의 데이터 분석은 마치 오케스트라와 같습니다. 각 악기(AI와 인간)가 자신의 장점을 살려 연주할 때 아름다운 하모니가 만들어집니다. 모든 것을 AI에게 맡기거나, 반대로 모든 것을 수동으로 하는 것은 비효율적입니다.

핵심은 **언제 AI를 활용하고, 언제 인간이 개입해야 하는지**를 정확히 판단하는 것입니다. 단순 반복 작업은 AI가, 창의적 해석과 비즈니스 판단은 인간이 맡는 것이 기본 원칙이지만, 실제로는 더 복잡하고 미묘한 균형이 필요합니다.

이번 Part에서는 데이터 분석 프로젝트의 각 단계별로 최적의 자동화 전략을 수립하고, 품질을 보장하면서도 효율성을 극대화하는 방법을 학습합니다. SMS 스팸 탐지 프로젝트를 통해 실제적인 워크플로우 설계 경험을 쌓아보겠습니다.

---

> 📝 **중요 용어**: **하이브리드 워크플로우(Hybrid Workflow)**
> 
> 인간의 창의성과 판단력, AI의 처리 능력과 일관성을 조합하여 설계된 작업 흐름입니다. 각 단계에서 가장 적합한 주체(인간 또는 AI)가 작업을 수행하되, 상호 검증과 피드백을 통해 품질을 보장하는 협업 방식입니다.

## 1. 자동화 적합성 평가 기준

데이터 분석 작업을 자동화할지 결정하는 것은 마치 요리에서 어떤 재료를 기계로 자르고 어떤 것을 손으로 다듬을지 결정하는 것과 같습니다. 각 작업의 특성을 정확히 파악해야 최적의 선택을 할 수 있습니다.

### 1.1 STAR 프레임워크: 자동화 적합성 평가

**STAR 프레임워크**는 작업의 자동화 적합성을 체계적으로 평가하는 도구입니다.

#### **S - Standardization (표준화 가능성)**
작업이 얼마나 표준화되어 있고 일관된 규칙을 따르는가?

```python
class TaskStandardizationAnalyzer:
    """작업 표준화 정도 분석기"""
    
    def __init__(self):
        self.standardization_criteria = {
            'input_consistency': '입력 데이터의 일관성',
            'process_repeatability': '프로세스 반복 가능성', 
            'output_predictability': '출력 결과의 예측 가능성',
            'rule_clarity': '작업 규칙의 명확성'
        }
    
    def evaluate_standardization(self, task_description):
        """
        작업의 표준화 정도를 평가
        
        Args:
            task_description: 작업 설명 딕셔너리
            
        Returns:
            dict: 표준화 평가 결과
        """
        
        scores = {}
        
        # 입력 일관성 평가 (1-5점)
        input_types = task_description.get('input_types', [])
        if len(set(input_types)) == 1:  # 동일한 타입
            scores['input_consistency'] = 5
        elif len(set(input_types)) <= 3:  # 제한적 타입
            scores['input_consistency'] = 3
        else:  # 다양한 타입
            scores['input_consistency'] = 1
        
        # 프로세스 반복성 평가
        steps_variability = task_description.get('steps_variability', 'high')
        process_scores = {'low': 5, 'medium': 3, 'high': 1}
        scores['process_repeatability'] = process_scores.get(steps_variability, 1)
        
        # 출력 예측성 평가
        output_variance = task_description.get('output_variance', 'high')
        output_scores = {'low': 5, 'medium': 3, 'high': 1}
        scores['output_predictability'] = output_scores.get(output_variance, 1)
        
        # 규칙 명확성 평가
        rule_complexity = task_description.get('rule_complexity', 'complex')
        rule_scores = {'simple': 5, 'moderate': 3, 'complex': 1}
        scores['rule_clarity'] = rule_scores.get(rule_complexity, 1)
        
        # 종합 점수 계산
        total_score = sum(scores.values()) / len(scores)
        
        # 표준화 등급 결정
        if total_score >= 4:
            standardization_level = 'high'
            automation_recommendation = 'strongly_recommended'
        elif total_score >= 3:
            standardization_level = 'medium' 
            automation_recommendation = 'recommended'
        elif total_score >= 2:
            standardization_level = 'low'
            automation_recommendation = 'partial_automation'
        else:
            standardization_level = 'very_low'
            automation_recommendation = 'manual_preferred'
        
        return {
            'scores': scores,
            'total_score': total_score,
            'standardization_level': standardization_level,
            'automation_recommendation': automation_recommendation,
            'detailed_analysis': self._generate_analysis(scores)
        }
    
    def _generate_analysis(self, scores):
        """상세 분석 생성"""
        analysis = []
        
        for criterion, score in scores.items():
            criterion_name = self.standardization_criteria[criterion]
            
            if score >= 4:
                analysis.append(f"✅ {criterion_name}: 자동화에 매우 적합")
            elif score >= 3:
                analysis.append(f"🟡 {criterion_name}: 자동화 가능")
            else:
                analysis.append(f"❌ {criterion_name}: 인간 개입 필요")
        
        return analysis

# SMS 스팸 탐지 프로젝트의 작업들 평가 예시
analyzer = TaskStandardizationAnalyzer()

# 다양한 작업들의 표준화 정도 평가
tasks_to_evaluate = {
    'data_cleaning': {
        'description': '결측치 처리 및 데이터 정제',
        'input_types': ['csv', 'json'],
        'steps_variability': 'low',
        'output_variance': 'low', 
        'rule_complexity': 'simple'
    },
    'feature_engineering': {
        'description': '텍스트 특성 추출',
        'input_types': ['text'],
        'steps_variability': 'medium',
        'output_variance': 'medium',
        'rule_complexity': 'moderate'
    },
    'model_interpretation': {
        'description': '모델 결과 해석 및 비즈니스 인사이트 도출',
        'input_types': ['model_output', 'metrics', 'domain_knowledge'],
        'steps_variability': 'high',
        'output_variance': 'high',
        'rule_complexity': 'complex'
    },
    'data_validation': {
        'description': '데이터 품질 검증',
        'input_types': ['dataframe'],
        'steps_variability': 'low',
        'output_variance': 'low',
        'rule_complexity': 'simple'
    }
}

print("📊 작업별 표준화 정도 평가")
print("=" * 50)

for task_name, task_info in tasks_to_evaluate.items():
    result = analyzer.evaluate_standardization(task_info)
    
    print(f"\n🔍 {task_name.upper()}: {task_info['description']}")
    print(f"표준화 등급: {result['standardization_level'].upper()}")
    print(f"자동화 권고: {result['automation_recommendation']}")
    print(f"종합 점수: {result['total_score']:.1f}/5.0")
    
    print("상세 분석:")
    for analysis in result['detailed_analysis']:
        print(f"  {analysis}")
```

**코드 해설:**
- **다차원 평가**: 단순히 하나의 기준이 아닌 4가지 관점에서 종합 평가
- **점수화 시스템**: 주관적 판단을 객관적 점수로 변환하여 일관성 확보
- **실행 가능한 권고**: 평가 결과를 바탕으로 구체적인 자동화 전략 제시

#### **T - Time Sensitivity (시간 민감성)**
작업의 시간적 제약과 즉시성 요구사항을 평가합니다.

```python
import time
from datetime import datetime, timedelta
from enum import Enum

class TimeSensitivity(Enum):
    """시간 민감성 수준"""
    REAL_TIME = "real_time"      # 실시간 (< 1초)
    NEAR_REAL_TIME = "near_real_time"  # 준실시간 (< 1분)
    BATCH_HOURLY = "batch_hourly"      # 시간별 배치 (< 1시간)
    BATCH_DAILY = "batch_daily"        # 일별 배치 (< 1일)
    PERIODIC = "periodic"              # 주기적 (> 1일)

class TimeSensitivityAnalyzer:
    """시간 민감성 분석기"""
    
    def __init__(self):
        self.automation_recommendations = {
            TimeSensitivity.REAL_TIME: {
                'automation_priority': 'critical',
                'human_involvement': 'minimal',
                'strategy': 'full_automation_with_monitoring'
            },
            TimeSensitivity.NEAR_REAL_TIME: {
                'automation_priority': 'high', 
                'human_involvement': 'oversight',
                'strategy': 'automated_with_human_oversight'
            },
            TimeSensitivity.BATCH_HOURLY: {
                'automation_priority': 'medium',
                'human_involvement': 'quality_check',
                'strategy': 'automated_with_quality_gates'
            },
            TimeSensitivity.BATCH_DAILY: {
                'automation_priority': 'medium',
                'human_involvement': 'review_and_validate',
                'strategy': 'hybrid_approach'
            },
            TimeSensitivity.PERIODIC: {
                'automation_priority': 'low',
                'human_involvement': 'full_involvement',
                'strategy': 'human_led_with_ai_assistance'
            }
        }
    
    def evaluate_time_sensitivity(self, task_requirements):
        """
        작업의 시간 민감성 평가
        
        Args:
            task_requirements: 작업 요구사항 딕셔너리
            
        Returns:
            dict: 시간 민감성 평가 결과
        """
        
        # 요구 응답 시간 분석
        max_response_time = task_requirements.get('max_response_time_seconds', 86400)
        
        if max_response_time < 1:
            sensitivity = TimeSensitivity.REAL_TIME
        elif max_response_time < 60:
            sensitivity = TimeSensitivity.NEAR_REAL_TIME
        elif max_response_time < 3600:
            sensitivity = TimeSensitivity.BATCH_HOURLY
        elif max_response_time < 86400:
            sensitivity = TimeSensitivity.BATCH_DAILY
        else:
            sensitivity = TimeSensitivity.PERIODIC
        
        # 비즈니스 임팩트 분석
        business_impact = task_requirements.get('delay_impact', 'low')
        impact_multiplier = {'low': 1.0, 'medium': 1.5, 'high': 2.0, 'critical': 3.0}
        
        # 자동화 우선순위 계산
        base_priority = self.automation_recommendations[sensitivity]['automation_priority']
        
        priority_scores = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        adjusted_score = priority_scores[base_priority] * impact_multiplier[business_impact]
        
        if adjusted_score >= 6:
            final_priority = 'critical'
        elif adjusted_score >= 4:
            final_priority = 'high'
        elif adjusted_score >= 2:
            final_priority = 'medium'
        else:
            final_priority = 'low'
        
        recommendation = self.automation_recommendations[sensitivity].copy()
        recommendation['automation_priority'] = final_priority
        
        return {
            'time_sensitivity': sensitivity.value,
            'max_response_time': max_response_time,
            'business_impact': business_impact,
            'automation_priority': final_priority,
            'strategy': recommendation['strategy'],
            'human_involvement': recommendation['human_involvement'],
            'implementation_guidelines': self._generate_guidelines(sensitivity, business_impact)
        }
    
    def _generate_guidelines(self, sensitivity, impact):
        """구현 가이드라인 생성"""
        
        guidelines = []
        
        if sensitivity == TimeSensitivity.REAL_TIME:
            guidelines.extend([
                "완전 자동화된 파이프라인 구축 필수",
                "실시간 모니터링 및 알람 시스템 구축",
                "장애 복구 자동화 메커니즘 준비",
                "인간 개입 최소화, 사후 검토만 수행"
            ])
        elif sensitivity == TimeSensitivity.NEAR_REAL_TIME:
            guidelines.extend([
                "자동화 + 실시간 품질 모니터링",
                "임계값 기반 자동 알람 설정",
                "빠른 인간 개입 프로세스 구축"
            ])
        else:
            guidelines.extend([
                "단계별 품질 검증 포인트 설정",
                "정기적인 인간 검토 및 승인 과정",
                "점진적 자동화 확대 전략"
            ])
        
        if impact in ['high', 'critical']:
            guidelines.append("다중 검증 체계 및 롤백 계획 필수")
        
        return guidelines

# SMS 스팸 탐지 프로젝트의 시간 민감성 평가
time_analyzer = TimeSensitivityAnalyzer()

sms_tasks_time_requirements = {
    'real_time_classification': {
        'description': '실시간 SMS 스팸 분류',
        'max_response_time_seconds': 0.5,  # 0.5초
        'delay_impact': 'high'
    },
    'model_retraining': {
        'description': '모델 재학습',
        'max_response_time_seconds': 86400,  # 1일
        'delay_impact': 'medium'
    },
    'performance_monitoring': {
        'description': '성능 모니터링 및 리포팅',
        'max_response_time_seconds': 3600,  # 1시간
        'delay_impact': 'low'
    },
    'business_analysis': {
        'description': '비즈니스 인사이트 분석',
        'max_response_time_seconds': 604800,  # 1주일
        'delay_impact': 'medium'
    }
}

print("\n⏰ 작업별 시간 민감성 평가")
print("=" * 50)

for task_name, requirements in sms_tasks_time_requirements.items():
    result = time_analyzer.evaluate_time_sensitivity(requirements)
    
    print(f"\n📋 {task_name.upper()}: {requirements['description']}")
    print(f"시간 민감성: {result['time_sensitivity']}")
    print(f"자동화 우선순위: {result['automation_priority'].upper()}")
    print(f"권장 전략: {result['strategy']}")
    print(f"인간 개입 수준: {result['human_involvement']}")
    
    print("구현 가이드라인:")
    for guideline in result['implementation_guidelines']:
        print(f"  • {guideline}")
```

**코드 해설:**
- **시간 기반 분류**: 응답 시간 요구사항에 따른 자동화 전략 차별화
- **비즈니스 임팩트 반영**: 지연 시 비즈니스 영향도를 고려한 우선순위 조정
- **실용적 가이드라인**: 각 시간 민감성 수준별 구체적 구현 방안 제시

#### **A - Accuracy Requirements (정확도 요구사항)**
작업에서 요구되는 정확도 수준과 오류의 비즈니스 임팩트를 평가합니다.

```python
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

@dataclass
class AccuracyRequirement:
    """정확도 요구사항 정의"""
    min_accuracy: float
    error_cost: str  # 'low', 'medium', 'high', 'critical'
    false_positive_tolerance: float
    false_negative_tolerance: float
    
class AccuracyAnalyzer:
    """정확도 요구사항 분석기"""
    
    def __init__(self):
        self.error_cost_weights = {
            'low': 1.0,
            'medium': 2.0, 
            'high': 5.0,
            'critical': 10.0
        }
        
        self.automation_thresholds = {
            'accuracy': 0.95,      # 95% 이상 정확도
            'reliability': 0.99,   # 99% 이상 안정성
            'consistency': 0.98    # 98% 이상 일관성
        }
    
    def evaluate_accuracy_requirements(self, task_spec, historical_performance=None):
        """
        정확도 요구사항 평가
        
        Args:
            task_spec: 작업 명세 (AccuracyRequirement 객체)
            historical_performance: 과거 성능 데이터 (선택사항)
            
        Returns:
            dict: 정확도 분석 결과
        """
        
        # 1. 기본 자동화 적합성 평가
        automation_score = self._calculate_automation_score(task_spec)
        
        # 2. AI vs 인간 성능 비교
        performance_comparison = self._compare_ai_human_performance(task_spec, historical_performance)
        
        # 3. 하이브리드 접근법 평가
        hybrid_strategy = self._design_hybrid_strategy(task_spec, performance_comparison)
        
        # 4. 최종 권고사항 생성
        recommendation = self._generate_recommendation(
            automation_score, 
            performance_comparison, 
            hybrid_strategy
        )
        
        return {
            'accuracy_requirements': {
                'min_accuracy': task_spec.min_accuracy,
                'error_cost': task_spec.error_cost,
                'fp_tolerance': task_spec.false_positive_tolerance,
                'fn_tolerance': task_spec.false_negative_tolerance
            },
            'automation_score': automation_score,
            'performance_comparison': performance_comparison,
            'hybrid_strategy': hybrid_strategy,
            'recommendation': recommendation
        }
    
    def _calculate_automation_score(self, task_spec):
        """자동화 점수 계산"""
        
        # 정확도 요구사항이 높을수록 자동화 어려움
        accuracy_penalty = max(0, (task_spec.min_accuracy - 0.8) * 2)
        
        # 오류 비용이 높을수록 자동화 신중
        error_cost_penalty = self.error_cost_weights[task_spec.error_cost] * 0.1
        
        # False Positive/Negative 허용도가 낮을수록 자동화 어려움
        fp_penalty = max(0, (0.1 - task_spec.false_positive_tolerance) * 5)
        fp_penalty = max(0, (0.1 - task_spec.false_negative_tolerance) * 5)
        
        # 기본 점수에서 페널티 차감
        base_score = 1.0
        total_penalty = accuracy_penalty + error_cost_penalty + fp_penalty + fp_penalty
        
        automation_score = max(0, base_score - total_penalty)
        
        return {
            'score': automation_score,
            'penalties': {
                'accuracy': accuracy_penalty,
                'error_cost': error_cost_penalty,
                'false_positive': fp_penalty,
                'false_negative': fp_penalty
            },
            'suitability': self._classify_automation_suitability(automation_score)
        }
    
    def _classify_automation_suitability(self, score):
        """자동화 적합성 분류"""
        if score >= 0.8:
            return 'highly_suitable'
        elif score >= 0.6:
            return 'suitable_with_monitoring'
        elif score >= 0.4:
            return 'partial_automation_recommended'
        else:
            return 'human_led_preferred'
    
    def _compare_ai_human_performance(self, task_spec, historical_data):
        """AI vs 인간 성능 비교"""
        
        # 시뮬레이션된 성능 데이터 (실제로는 실험 결과 사용)
        if historical_data is None:
            # SMS 스팸 분류 작업 기준 시뮬레이션
            ai_performance = {
                'accuracy': 0.92,
                'precision': 0.89,
                'recall': 0.94,
                'consistency': 0.98,  # AI는 일관성이 높음
                'speed': 0.001,       # 초당 처리 시간
                'cost_per_task': 0.0001
            }
            
            human_performance = {
                'accuracy': 0.85,
                'precision': 0.88,
                'recall': 0.82,
                'consistency': 0.75,  # 인간은 일관성이 상대적으로 낮음
                'speed': 30,          # 초당 처리 시간
                'cost_per_task': 0.50
            }
        else:
            ai_performance = historical_data['ai']
            human_performance = historical_data['human']
        
        # 성능 비교 분석
        comparison = {}
        for metric in ai_performance:
            ai_val = ai_performance[metric]
            human_val = human_performance[metric]
            
            if metric in ['speed', 'cost_per_task']:
                # 낮을수록 좋은 지표 (시간, 비용)
                advantage = 'ai' if ai_val < human_val else 'human'
                ratio = human_val / ai_val if ai_val != 0 else float('inf')
            else:
                # 높을수록 좋은 지표 (정확도, 정밀도 등)
                advantage = 'ai' if ai_val > human_val else 'human'
                ratio = ai_val / human_val if human_val != 0 else float('inf')
            
            comparison[metric] = {
                'ai_value': ai_val,
                'human_value': human_val,
                'advantage': advantage,
                'ratio': ratio
            }
        
        return comparison
    
    def _design_hybrid_strategy(self, task_spec, performance_comparison):
        """하이브리드 전략 설계"""
        
        # AI의 강점과 약점 분석
        ai_strengths = []
        ai_weaknesses = []
        
        for metric, comp in performance_comparison.items():
            if comp['advantage'] == 'ai':
                ai_strengths.append(metric)
            else:
                ai_weaknesses.append(metric)
        
        # 하이브리드 전략 설계
        if task_spec.error_cost in ['critical', 'high']:
            strategy = {
                'primary_approach': 'human_review_required',
                'ai_role': 'preprocessing_and_flagging',
                'human_role': 'final_decision_and_quality_assurance',
                'workflow': [
                    'AI가 1차 분류 수행',
                    '신뢰도 낮은 케이스 인간에게 전달',
                    '인간이 최종 검토 및 승인',
                    'AI 결과와 인간 판단 비교 학습'
                ]
            }
        elif 'accuracy' in ai_strengths and 'consistency' in ai_strengths:
            strategy = {
                'primary_approach': 'ai_with_human_oversight',
                'ai_role': 'primary_classification',
                'human_role': 'exception_handling_and_monitoring',
                'workflow': [
                    'AI가 대부분 작업 자동 처리',
                    '임계값 벗어난 케이스만 인간 검토',
                    '정기적인 성능 모니터링',
                    '점진적인 자동화 확대'
                ]
            }
        else:
            strategy = {
                'primary_approach': 'collaborative',
                'ai_role': 'data_processing_and_feature_extraction',
                'human_role': 'analysis_and_decision_making',
                'workflow': [
                    'AI가 데이터 전처리 및 특성 추출',
                    '인간이 패턴 분석 및 해석',
                    'AI가 반복 작업 자동화',
                    '인간이 창의적 분석 및 전략 수립'
                ]
            }
        
        return strategy
    
    def _generate_recommendation(self, automation_score, performance_comparison, hybrid_strategy):
        """최종 권고사항 생성"""
        
        score_val = automation_score['score']
        suitability = automation_score['suitability']
        
        if suitability == 'highly_suitable':
            recommendation = {
                'approach': 'full_automation',
                'confidence': 'high',
                'implementation_priority': 'immediate',
                'risk_level': 'low'
            }
        elif suitability == 'suitable_with_monitoring':
            recommendation = {
                'approach': 'automated_with_monitoring',
                'confidence': 'medium-high',
                'implementation_priority': 'near_term',
                'risk_level': 'medium'
            }
        elif suitability == 'partial_automation_recommended':
            recommendation = {
                'approach': 'hybrid',
                'confidence': 'medium',
                'implementation_priority': 'gradual',
                'risk_level': 'medium-high'
            }
        else:
            recommendation = {
                'approach': 'human_led',
                'confidence': 'high',
                'implementation_priority': 'ai_assistance_only',
                'risk_level': 'low'
            }
        
        recommendation['strategy_details'] = hybrid_strategy
        
        return recommendation

# SMS 스팸 분류 작업의 정확도 요구사항 평가
accuracy_analyzer = AccuracyAnalyzer()

# 다양한 시나리오별 정확도 요구사항
accuracy_scenarios = {
    'customer_facing_filter': AccuracyRequirement(
        min_accuracy=0.95,
        error_cost='high',  # 고객 불만으로 이어질 수 있음
        false_positive_tolerance=0.01,  # 정상 메시지 차단 최소화
        false_negative_tolerance=0.05   # 일부 스팸 통과는 허용
    ),
    'internal_preprocessing': AccuracyRequirement(
        min_accuracy=0.90,
        error_cost='medium',
        false_positive_tolerance=0.05,
        false_negative_tolerance=0.05
    ),
    'research_analysis': AccuracyRequirement(
        min_accuracy=0.85,
        error_cost='low',
        false_positive_tolerance=0.10,
        false_negative_tolerance=0.10
    )
}

print("\n🎯 작업별 정확도 요구사항 분석")
print("=" * 50)

for scenario_name, requirements in accuracy_scenarios.items():
    result = accuracy_analyzer.evaluate_accuracy_requirements(requirements)
    
    print(f"\n📋 {scenario_name.upper()}")
    print(f"최소 정확도 요구: {requirements.min_accuracy:.1%}")
    print(f"오류 비용: {requirements.error_cost}")
    print(f"자동화 적합성: {result['automation_score']['suitability']}")
    print(f"권장 접근법: {result['recommendation']['approach']}")
    print(f"구현 우선순위: {result['recommendation']['implementation_priority']}")
    
    print("\n권장 워크플로우:")
    for step in result['recommendation']['strategy_details']['workflow']:
        print(f"  {step}")
```

**코드 해설:**
- **다차원 정확도 평가**: 단순한 정확도뿐만 아니라 FP/FN 비율, 오류 비용 고려
- **성능 비교 분석**: AI와 인간의 상대적 강점/약점을 정량적으로 비교
- **맞춤형 전략 설계**: 요구사항에 따른 최적 하이브리드 전략 자동 생성

#### **R - Resource Requirements (자원 요구사항)**
작업에 필요한 인적/기술적 자원과 비용을 평가합니다.

```python
from datetime import datetime
import json

class ResourceAnalyzer:
    """자원 요구사항 분석기"""
    
    def __init__(self):
        # 비용 기준표 (USD 기준)
        self.cost_benchmarks = {
            'human_analyst_hourly': 50,
            'data_scientist_hourly': 75,
            'cloud_compute_hourly': 0.10,
            'ai_api_calls_per_1k': 0.002,
            'storage_gb_monthly': 0.023
        }
        
        # 자원 유형별 특성
        self.resource_characteristics = {
            'human': {
                'setup_time': 'low',
                'scalability': 'limited',
                'consistency': 'variable',
                'expertise_required': 'high'
            },
            'ai_automation': {
                'setup_time': 'high',
                'scalability': 'excellent', 
                'consistency': 'high',
                'expertise_required': 'medium'
            },
            'hybrid': {
                'setup_time': 'medium',
                'scalability': 'good',
                'consistency': 'good',
                'expertise_required': 'high'
            }
        }
    
    def analyze_resource_requirements(self, task_profile):
        """
        작업의 자원 요구사항 분석
        
        Args:
            task_profile: 작업 프로필 딕셔너리
            
        Returns:
            dict: 자원 분석 결과
        """
        
        # 1. 작업량 분석
        workload_analysis = self._analyze_workload(task_profile)
        
        # 2. 접근법별 비용 계산
        cost_analysis = self._calculate_costs(task_profile, workload_analysis)
        
        # 3. ROI 분석
        roi_analysis = self._calculate_roi(cost_analysis, task_profile)
        
        # 4. 자원 최적화 권고
        optimization_recommendations = self._generate_optimization_recommendations(
            workload_analysis, cost_analysis, roi_analysis
        )
        
        return {
            'workload_analysis': workload_analysis,
            'cost_analysis': cost_analysis,
            'roi_analysis': roi_analysis,
            'optimization_recommendations': optimization_recommendations,
            'summary': self._generate_summary(cost_analysis, roi_analysis)
        }
    
    def _analyze_workload(self, task_profile):
        """작업량 분석"""
        
        # 기본 작업량 파라미터
        volume = task_profile.get('monthly_volume', 1000)
        complexity = task_profile.get('complexity_score', 5)  # 1-10
        variability = task_profile.get('variability', 'medium')  # low, medium, high
        
        # 처리 시간 추정
        base_time_per_task = {
            'human': complexity * 2,      # 분 단위
            'ai': complexity * 0.1,       # 분 단위
            'hybrid': complexity * 0.5    # 분 단위
        }
        
        # 변동성에 따른 조정
        variability_multipliers = {
            'low': 1.0,
            'medium': 1.2,
            'high': 1.5
        }
        
        multiplier = variability_multipliers[variability]
        
        monthly_hours = {}
        for approach in base_time_per_task:
            time_per_task = base_time_per_task[approach] * multiplier
            total_hours = (volume * time_per_task) / 60
            monthly_hours[approach] = total_hours
        
        return {
            'volume': volume,
            'complexity_score': complexity,
            'variability': variability,
            'time_per_task_minutes': {k: v * multiplier for k, v in base_time_per_task.items()},
            'monthly_hours': monthly_hours,
            'peak_capacity_needed': monthly_hours['human'] * 1.5  # 피크 시간 대비
        }
    
    def _calculate_costs(self, task_profile, workload_analysis):
        """접근법별 비용 계산"""
        
        monthly_volume = workload_analysis['volume']
        monthly_hours = workload_analysis['monthly_hours']
        
        # 인간 전용 접근법 비용
        human_costs = {
            'labor': monthly_hours['human'] * self.cost_benchmarks['human_analyst_hourly'],
            'training': 500,  # 월간 교육 비용
            'management': 200,  # 관리 비용
            'quality_control': 100
        }
        human_total = sum(human_costs.values())
        
        # AI 자동화 접근법 비용
        ai_costs = {
            'api_calls': (monthly_volume / 1000) * self.cost_benchmarks['ai_api_calls_per_1k'],
            'compute': monthly_hours['ai'] * self.cost_benchmarks['cloud_compute_hourly'],
            'storage': 10 * self.cost_benchmarks['storage_gb_monthly'],  # 10GB 저장소
            'development': 2000,  # 초기 개발 비용 (월간 분할)
            'maintenance': 300   # 월간 유지보수
        }
        ai_total = sum(ai_costs.values())
        
        # 하이브리드 접근법 비용
        hybrid_costs = {
            'labor': monthly_hours['hybrid'] * self.cost_benchmarks['data_scientist_hourly'],
            'ai_infrastructure': ai_costs['api_calls'] + ai_costs['compute'] + ai_costs['storage'],
            'integration': 500,  # 통합 유지보수
            'quality_assurance': 300
        }
        hybrid_total = sum(hybrid_costs.values())
        
        return {
            'human': {
                'breakdown': human_costs,
                'total_monthly': human_total,
                'cost_per_task': human_total / monthly_volume
            },
            'ai': {
                'breakdown': ai_costs,
                'total_monthly': ai_total,
                'cost_per_task': ai_total / monthly_volume
            },
            'hybrid': {
                'breakdown': hybrid_costs,
                'total_monthly': hybrid_total,
                'cost_per_task': hybrid_total / monthly_volume
            }
        }
    
    def _calculate_roi(self, cost_analysis, task_profile):
        """ROI 분석"""
        
        # 기준선 (현재 비용) - 보통 인간 전용 방식
        baseline_cost = cost_analysis['human']['total_monthly']
        
        # 비즈니스 가치 추정
        monthly_value = task_profile.get('monthly_business_value', baseline_cost * 2)
        
        roi_analysis = {}
        
        for approach in ['human', 'ai', 'hybrid']:
            monthly_cost = cost_analysis[approach]['total_monthly']
            monthly_profit = monthly_value - monthly_cost
            
            if monthly_cost > 0:
                roi_percent = (monthly_profit / monthly_cost) * 100
            else:
                roi_percent = float('inf')
            
            # 투자 회수 기간 (개월)
            initial_investment = task_profile.get('initial_investment', {}).get(approach, 0)
            if monthly_profit > 0:
                payback_months = initial_investment / monthly_profit
            else:
                payback_months = float('inf')
            
            roi_analysis[approach] = {
                'monthly_cost': monthly_cost,
                'monthly_profit': monthly_profit,
                'roi_percent': roi_percent,
                'payback_months': payback_months,
                'annual_savings_vs_baseline': (baseline_cost - monthly_cost) * 12
            }
        
        return roi_analysis
    
    def _generate_optimization_recommendations(self, workload, costs, roi):
        """최적화 권고사항 생성"""
        
        recommendations = []
        
        # 비용 효율성 분석
        cost_ranking = sorted(costs.items(), key=lambda x: x[1]['total_monthly'])
        most_cost_effective = cost_ranking[0][0]
        
        # ROI 분석
        roi_ranking = sorted(roi.items(), key=lambda x: x[1]['roi_percent'], reverse=True)
        best_roi = roi_ranking[0][0]
        
        recommendations.append({
            'type': 'cost_optimization',
            'priority': 'high',
            'description': f"가장 비용 효율적인 접근법: {most_cost_effective}",
            'monthly_savings': costs['human']['total_monthly'] - costs[most_cost_effective]['total_monthly']
        })
        
        recommendations.append({
            'type': 'roi_optimization', 
            'priority': 'high',
            'description': f"가장 높은 ROI 접근법: {best_roi}",
            'roi_percentage': roi[best_roi]['roi_percent']
        })
        
        # 확장성 고려사항
        if workload['volume'] > 5000:  # 대용량 작업
            recommendations.append({
                'type': 'scalability',
                'priority': 'medium',
                'description': "대용량 작업으로 인해 AI 자동화 또는 하이브리드 접근법 권장",
                'rationale': "인간 전용 접근법은 확장성 제한"
            })
        
        # 품질 고려사항
        if workload['complexity_score'] >= 8:  # 높은 복잡도
            recommendations.append({
                'type': 'quality_assurance',
                'priority': 'high',
                'description': "높은 복잡도로 인해 인간 전문가 개입 필수",
                'rationale': "AI 단독으로는 복잡한 판단 어려움"
            })
        
        return recommendations
    
    def _generate_summary(self, costs, roi):
        """요약 정보 생성"""
        
        # 최적 선택지 결정
        approaches = ['human', 'ai', 'hybrid']
        
        # 비용 대비 효과 종합 점수 계산
        scores = {}
        for approach in approaches:
            cost_score = 1 / (costs[approach]['total_monthly'] / 1000)  # 비용이 낮을수록 높은 점수
            roi_score = max(0, roi[approach]['roi_percent'] / 100)      # ROI가 높을수록 높은 점수
            
            # 가중 평균 (비용 60%, ROI 40%)
            scores[approach] = (cost_score * 0.6) + (roi_score * 0.4)
        
        best_approach = max(scores, key=scores.get)
        
        return {
            'recommended_approach': best_approach,
            'confidence_score': scores[best_approach],
            'key_factors': {
                'most_cost_effective': min(costs, key=lambda x: costs[x]['total_monthly']),
                'highest_roi': max(roi, key=lambda x: roi[x]['roi_percent']),
                'fastest_payback': min(roi, key=lambda x: roi[x]['payback_months'])
            },
            'decision_rationale': self._generate_decision_rationale(best_approach, costs, roi)
        }
    
    def _generate_decision_rationale(self, best_approach, costs, roi):
        """의사결정 근거 생성"""
        
        cost = costs[best_approach]['total_monthly']
        roi_pct = roi[best_approach]['roi_percent']
        
        rationale = f"{best_approach} 접근법을 권장하는 이유:\n"
        rationale += f"- 월간 비용: ${cost:,.0f}\n"
        rationale += f"- ROI: {roi_pct:.1f}%\n"
        
        if best_approach == 'ai':
            rationale += "- 높은 확장성과 일관성\n- 장기적 비용 절감 효과"
        elif best_approach == 'hybrid':
            rationale += "- 균형잡힌 비용과 품질\n- 점진적 자동화 확대 가능"
        else:
            rationale += "- 높은 품질과 유연성\n- 복잡한 판단 상황에 적합"
        
        return rationale

# SMS 스팸 탐지 프로젝트의 자원 요구사항 분석
resource_analyzer = ResourceAnalyzer()

sms_project_profile = {
    'monthly_volume': 50000,        # 월간 5만개 SMS 처리
    'complexity_score': 6,          # 중간 복잡도
    'variability': 'medium',        # 중간 변동성
    'monthly_business_value': 10000, # 월간 비즈니스 가치 $10,000
    'initial_investment': {         # 초기 투자비용
        'human': 0,
        'ai': 15000,
        'hybrid': 8000
    }
}

print("\n💰 자원 요구사항 및 비용 분석")
print("=" * 50)

resource_result = resource_analyzer.analyze_resource_requirements(sms_project_profile)

# 작업량 분석 출력
workload = resource_result['workload_analysis']
print(f"\n📊 작업량 분석:")
print(f"월간 처리량: {workload['volume']:,}개")
print(f"복잡도 점수: {workload['complexity_score']}/10")
print(f"변동성: {workload['variability']}")

# 접근법별 월간 소요 시간
print(f"\n⏱️ 접근법별 월간 소요 시간:")
for approach, hours in workload['monthly_hours'].items():
    print(f"  {approach}: {hours:.1f}시간")

# 비용 분석 출력
costs = resource_result['cost_analysis']
print(f"\n💸 접근법별 월간 비용:")
for approach, cost_data in costs.items():
    print(f"\n{approach.upper()} 접근법:")
    print(f"  총 월간 비용: ${cost_data['total_monthly']:,.2f}")
    print(f"  작업당 비용: ${cost_data['cost_per_task']:.4f}")
    
    print(f"  비용 구성:")
    for category, amount in cost_data['breakdown'].items():
        print(f"    - {category}: ${amount:,.2f}")

# ROI 분석 출력
roi_data = resource_result['roi_analysis']
print(f"\n📈 ROI 분석:")
for approach, roi_info in roi_data.items():
    print(f"\n{approach.upper()}:")
    print(f"  ROI: {roi_info['roi_percent']:.1f}%")
    print(f"  투자회수기간: {roi_info['payback_months']:.1f}개월")
    print(f"  연간 절약액: ${roi_info['annual_savings_vs_baseline']:,.0f}")

# 최적화 권고사항
recommendations = resource_result['optimization_recommendations']
print(f"\n🎯 최적화 권고사항:")
for i, rec in enumerate(recommendations, 1):
    print(f"\n{i}. [{rec['priority'].upper()}] {rec['description']}")
    if 'monthly_savings' in rec:
        print(f"   월간 절약액: ${rec['monthly_savings']:,.2f}")
    if 'roi_percentage' in rec:
        print(f"   예상 ROI: {rec['roi_percentage']:.1f}%")

# 종합 결론
summary = resource_result['summary']
print(f"\n🏆 종합 결론:")
print(f"권장 접근법: {summary['recommended_approach'].upper()}")
print(f"신뢰도 점수: {summary['confidence_score']:.2f}")
print(f"\n결정 근거:")
print(summary['decision_rationale'])
```

**코드 해설:**
- **종합적 비용 분석**: 초기 투자, 운영 비용, ROI를 모두 고려한 의사결정 지원
- **확장성 고려**: 작업량 증가에 따른 각 접근법의 대응 능력 평가
- **비즈니스 가치 연계**: 단순 비용 절감을 넘어서 비즈니스 가치 창출 관점에서 분석

> 💡 **STAR 프레임워크 활용 팁**
> 
> **S (표준화)**: 높을수록 자동화 유리
> **T (시간 민감성)**: 빠를수록 자동화 필수
> **A (정확도)**: 높을수록 신중한 접근 필요
> **R (자원)**: 제약이 클수록 효율성 중시
> 
> 4가지 관점을 종합하여 최적의 자동화 전략을 수립하세요!

> 🖼️ **이미지 생성 프롬프트**: 
> "STAR 프레임워크를 보여주는 4분면 매트릭스. Standardization(표준화), Time sensitivity(시간민감성), Accuracy requirements(정확도 요구사항), Resource requirements(자원 요구사항)가 각각 다른 색상으로 표현되고, 중앙에는 자동화 적합성 점수가 표시된 분석적 다이어그램"

## 2. 인간-AI 협업 모델 설계

효과적인 인간-AI 협업은 마치 잘 조율된 듀엣과 같습니다. 각자의 장점을 살리면서 서로의 약점을 보완하는 조화로운 워크플로우를 설계해야 합니다.

### 2.1 협업 패턴의 3가지 유형

#### **패턴 1: 순차적 협업 (Sequential Collaboration)**
AI와 인간이 단계별로 순차적으로 작업을 수행하는 방식입니다.

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import time
import logging

class TaskStatus(Enum):
    """작업 상태"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress" 
    COMPLETED = "completed"
    FAILED = "failed"
    REQUIRES_REVIEW = "requires_review"

@dataclass
class WorkflowTask:
    """워크플로우 작업 정의"""
    task_id: str
    task_type: str
    assigned_to: str  # 'ai' or 'human'
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]] = None
    status: TaskStatus = TaskStatus.PENDING
    confidence_score: Optional[float] = None
    review_required: bool = False
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

class WorkflowAgent(ABC):
    """워크플로우 에이전트 기본 클래스"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{agent_id}")
    
    @abstractmethod
    def can_handle(self, task: WorkflowTask) -> bool:
        """작업 처리 가능 여부 확인"""
        pass
    
    @abstractmethod
    def process_task(self, task: WorkflowTask) -> WorkflowTask:
        """작업 처리"""
        pass
    
    def validate_input(self, task: WorkflowTask) -> bool:
        """입력 데이터 검증"""
        return task.input_data is not None

class AIAgent(WorkflowAgent):
    """AI 에이전트"""
    
    def __init__(self, agent_id: str, capabilities: List[str]):
        super().__init__(agent_id)
        self.capabilities = capabilities
        self.processing_time_per_task = 0.1  # 초
    
    def can_handle(self, task: WorkflowTask) -> bool:
        """AI가 처리할 수 있는 작업인지 확인"""
        return task.task_type in self.capabilities and task.assigned_to == 'ai'
    
    def process_task(self, task: WorkflowTask) -> WorkflowTask:
        """AI 작업 처리"""
        
        if not self.validate_input(task):
            task.status = TaskStatus.FAILED
            return task
        
        task.status = TaskStatus.IN_PROGRESS
        self.logger.info(f"AI 처리 시작: {task.task_id}")
        
        # 작업 유형별 처리
        try:
            if task.task_type == 'text_preprocessing':
                result = self._preprocess_text(task.input_data)
            elif task.task_type == 'spam_classification':
                result = self._classify_spam(task.input_data)
            elif task.task_type == 'feature_extraction':
                result = self._extract_features(task.input_data)
            else:
                raise ValueError(f"지원하지 않는 작업 유형: {task.task_type}")
            
            task.output_data = result
            task.status = TaskStatus.COMPLETED
            
            # 신뢰도 기반 검토 필요성 판단
            if task.confidence_score and task.confidence_score < 0.8:
                task.review_required = True
                task.status = TaskStatus.REQUIRES_REVIEW
            
            self.logger.info(f"AI 처리 완료: {task.task_id}, 신뢰도: {task.confidence_score}")
            
        except Exception as e:
            self.logger.error(f"AI 처리 실패: {task.task_id}, 오류: {str(e)}")
            task.status = TaskStatus.FAILED
            task.output_data = {'error': str(e)}
        
        return task
    
    def _preprocess_text(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """텍스트 전처리"""
        text = input_data.get('text', '')
        
        # 시뮬레이션된 전처리
        cleaned_text = text.lower().strip()
        word_count = len(cleaned_text.split())
        
        return {
            'cleaned_text': cleaned_text,
            'word_count': word_count,
            'character_count': len(cleaned_text)
        }
    
    def _classify_spam(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """스팸 분류 (시뮬레이션)"""
        text = input_data.get('text', '')
        
        # 간단한 규칙 기반 시뮬레이션
        spam_indicators = ['free', 'win', 'money', 'prize', 'urgent']
        spam_count = sum(1 for indicator in spam_indicators if indicator in text.lower())
        
        if spam_count >= 2:
            prediction = 'spam'
            confidence = 0.8 + (spam_count * 0.1)
        elif spam_count == 1:
            prediction = 'ham'
            confidence = 0.6
        else:
            prediction = 'ham'
            confidence = 0.9
        
        confidence = min(confidence, 1.0)
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'spam_indicators_found': spam_count
        }
    
    def _extract_features(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """특성 추출"""
        text = input_data.get('text', '')
        
        features = {
            'length': len(text),
            'word_count': len(text.split()),
            'exclamation_count': text.count('!'),
            'capital_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'digit_ratio': sum(1 for c in text if c.isdigit()) / len(text) if text else 0
        }
        
        return {'features': features}

class HumanAgent(WorkflowAgent):
    """인간 에이전트 (시뮬레이션)"""
    
    def __init__(self, agent_id: str, expertise_areas: List[str]):
        super().__init__(agent_id)
        self.expertise_areas = expertise_areas
        self.processing_time_per_task = 30  # 초 (시뮬레이션)
    
    def can_handle(self, task: WorkflowTask) -> bool:
        """인간이 처리할 수 있는 작업인지 확인"""
        return (task.task_type in self.expertise_areas and 
                task.assigned_to == 'human') or task.review_required
    
    def process_task(self, task: WorkflowTask) -> WorkflowTask:
        """인간 작업 처리 (시뮬레이션)"""
        
        task.status = TaskStatus.IN_PROGRESS
        self.logger.info(f"인간 처리 시작: {task.task_id}")
        
        try:
            if task.task_type == 'quality_review':
                result = self._review_quality(task.input_data)
            elif task.task_type == 'business_interpretation':
                result = self._interpret_business_impact(task.input_data)
            elif task.review_required:
                result = self._review_ai_output(task)
            else:
                raise ValueError(f"지원하지 않는 작업 유형: {task.task_type}")
            
            task.output_data = result
            task.status = TaskStatus.COMPLETED
            task.review_required = False
            
            self.logger.info(f"인간 처리 완료: {task.task_id}")
            
        except Exception as e:
            self.logger.error(f"인간 처리 실패: {task.task_id}, 오류: {str(e)}")
            task.status = TaskStatus.FAILED
            task.output_data = {'error': str(e)}
        
        return task
    
    def _review_quality(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """품질 검토"""
        # 시뮬레이션된 품질 검토
        return {
            'quality_score': 0.85,
            'issues_found': ['minor_inconsistency'],
            'recommendations': ['추가 데이터 정제 필요']
        }
    
    def _interpret_business_impact(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """비즈니스 임팩트 해석"""
        results = input_data.get('analysis_results', {})
        
        # 시뮬레이션된 비즈니스 해석
        interpretation = {
            'key_insights': [
                '스팸 탐지율 95% 달성으로 고객 만족도 향상 예상',
                '월간 약 500건의 스팸 차단으로 시스템 효율성 개선'
            ],
            'business_value': 'high',
            'recommended_actions': [
                '현재 모델을 프로덕션에 배포',
                '월간 성능 모니터링 체계 구축'
            ]
        }
        
        return interpretation
    
    def _review_ai_output(self, task: WorkflowTask) -> Dict[str, Any]:
        """AI 출력 검토"""
        ai_output = task.output_data
        
        # 시뮬레이션된 AI 출력 검토
        if task.task_type == 'spam_classification':
            prediction = ai_output.get('prediction', '')
            confidence = ai_output.get('confidence', 0)
            
            if confidence < 0.7:
                # 인간이 재분류
                reviewed_prediction = 'ham'  # 시뮬레이션
                return {
                    'original_prediction': prediction,
                    'reviewed_prediction': reviewed_prediction,
                    'review_reason': '낮은 신뢰도로 인한 인간 재검토',
                    'final_confidence': 0.9
                }
            else:
                return {
                    'prediction': prediction,
                    'confidence': confidence,
                    'human_approval': True
                }
        
        return ai_output

class SequentialWorkflow:
    """순차적 워크플로우 관리자"""
    
    def __init__(self):
        self.agents: Dict[str, WorkflowAgent] = {}
        self.workflow_definition: List[Dict[str, Any]] = []
        self.task_queue: List[WorkflowTask] = []
        self.completed_tasks: List[WorkflowTask] = []
        self.logger = logging.getLogger(__name__)
    
    def register_agent(self, agent: WorkflowAgent):
        """에이전트 등록"""
        self.agents[agent.agent_id] = agent
        self.logger.info(f"에이전트 등록: {agent.agent_id}")
    
    def define_workflow(self, workflow_steps: List[Dict[str, Any]]):
        """워크플로우 정의"""
        self.workflow_definition = workflow_steps
        self.logger.info(f"워크플로우 정의: {len(workflow_steps)}단계")
    
    def execute_workflow(self, initial_data: Dict[str, Any]) -> Dict[str, Any]:
        """워크플로우 실행"""
        
        self.logger.info("순차적 워크플로우 실행 시작")
        
        current_data = initial_data
        execution_results = []
        
        for step_index, step_config in enumerate(self.workflow_definition):
            step_name = step_config['name']
            task_type = step_config['task_type']
            assigned_to = step_config['assigned_to']
            
            # 작업 생성
            task = WorkflowTask(
                task_id=f"{step_name}_{int(time.time())}",
                task_type=task_type,
                assigned_to=assigned_to,
                input_data=current_data
            )
            
            # 적절한 에이전트 찾기
            assigned_agent = self._find_agent_for_task(task)
            
            if not assigned_agent:
                self.logger.error(f"작업 처리 가능한 에이전트 없음: {task.task_id}")
                break
            
            # 작업 처리
            processed_task = assigned_agent.process_task(task)
            execution_results.append(processed_task)
            
            if processed_task.status == TaskStatus.FAILED:
                self.logger.error(f"워크플로우 중단: {step_name} 실패")
                break
            
            # 검토가 필요한 경우 인간 에이전트에게 전달
            if processed_task.review_required:
                review_agent = self._find_human_agent()
                if review_agent:
                    reviewed_task = review_agent.process_task(processed_task)
                    execution_results[-1] = reviewed_task
                    current_data = reviewed_task.output_data
                else:
                    self.logger.warning(f"검토 가능한 인간 에이전트 없음: {task.task_id}")
                    current_data = processed_task.output_data
            else:
                current_data = processed_task.output_data
        
        self.logger.info("순차적 워크플로우 실행 완료")
        
        return {
            'final_result': current_data,
            'execution_steps': execution_results,
            'total_steps': len(execution_results),
            'success_rate': sum(1 for task in execution_results 
                               if task.status == TaskStatus.COMPLETED) / len(execution_results)
        }
    
    def _find_agent_for_task(self, task: WorkflowTask) -> Optional[WorkflowAgent]:
        """작업에 적합한 에이전트 찾기"""
        for agent in self.agents.values():
            if agent.can_handle(task):
                return agent
        return None
    
    def _find_human_agent(self) -> Optional[HumanAgent]:
        """인간 에이전트 찾기"""
        for agent in self.agents.values():
            if isinstance(agent, HumanAgent):
                return agent
        return None

# SMS 스팸 탐지 순차적 워크플로우 구현 예시
print("🔄 순차적 협업 워크플로우 시연")
print("=" * 50)

# 에이전트 생성 및 등록
ai_agent = AIAgent(
    agent_id="sms_ai_processor",
    capabilities=['text_preprocessing', 'spam_classification', 'feature_extraction']
)

human_agent = HumanAgent(
    agent_id="sms_human_analyst", 
    expertise_areas=['quality_review', 'business_interpretation']
)

# 워크플로우 생성
workflow = SequentialWorkflow()
workflow.register_agent(ai_agent)
workflow.register_agent(human_agent)

# 워크플로우 단계 정의
workflow_steps = [
    {
        'name': 'text_preprocessing',
        'task_type': 'text_preprocessing',
        'assigned_to': 'ai'
    },
    {
        'name': 'spam_classification', 
        'task_type': 'spam_classification',
        'assigned_to': 'ai'
    },
    {
        'name': 'business_interpretation',
        'task_type': 'business_interpretation', 
        'assigned_to': 'human'
    }
]

workflow.define_workflow(workflow_steps)

# 테스트 데이터로 워크플로우 실행
test_messages = [
    "FREE money! Call now to claim your prize!",
    "Hey, how are you doing today?",
    "URGENT: Your account will be suspended unless you call 123-456-7890"
]

print("\n📨 테스트 메시지 처리:")
for i, message in enumerate(test_messages, 1):
    print(f"\n--- 메시지 {i}: {message[:30]}... ---")
    
    initial_data = {'text': message}
    result = workflow.execute_workflow(initial_data)
    
    print(f"처리 단계 수: {result['total_steps']}")
    print(f"성공률: {result['success_rate']:.1%}")
    
    # 최종 결과 출력
    final_result = result['final_result']
    if 'key_insights' in final_result:
        print(f"주요 인사이트: {final_result['key_insights'][0]}")
```

**코드 해설:**
- **에이전트 기반 설계**: AI와 인간을 동일한 인터페이스로 추상화하여 유연한 협업 가능
- **신뢰도 기반 검토**: AI의 신뢰도가 낮을 때 자동으로 인간 검토 요청
- **상태 추적**: 각 작업의 진행 상태를 체계적으로 관리
- **확장 가능한 구조**: 새로운 에이전트나 작업 유형을 쉽게 추가 가능

#### **패턴 2: 병렬적 협업 (Parallel Collaboration)**
AI와 인간이 동시에 작업하여 결과를 결합하는 방식입니다.

```python
import asyncio
import concurrent.futures
from typing import List, Callable, Tuple
import json

class ParallelWorkflowResult:
    """병렬 워크플로우 결과"""
    
    def __init__(self):
        self.ai_results: List[Dict[str, Any]] = []
        self.human_results: List[Dict[str, Any]] = []
        self.combined_results: Dict[str, Any] = {}
        self.consensus_score: float = 0.0
        self.disagreement_areas: List[str] = []

class ParallelCollaborationEngine:
    """병렬 협업 엔진"""
    
    def __init__(self):
        self.ai_agents: List[AIAgent] = []
        self.human_agents: List[HumanAgent] = []
        self.combination_strategies = {
            'majority_vote': self._majority_vote_combination,
            'weighted_average': self._weighted_average_combination,
            'confidence_based': self._confidence_based_combination,
            'expert_arbitration': self._expert_arbitration_combination
        }
    
    def register_ai_agent(self, agent: AIAgent):
        """AI 에이전트 등록"""
        self.ai_agents.append(agent)
    
    def register_human_agent(self, agent: HumanAgent):
        """인간 에이전트 등록"""
        self.human_agents.append(agent)
    
    async def execute_parallel_workflow(self, 
                                      task_data: Dict[str, Any],
                                      task_type: str,
                                      combination_strategy: str = 'confidence_based') -> ParallelWorkflowResult:
        """병렬 워크플로우 실행"""
        
        result = ParallelWorkflowResult()
        
        # AI와 인간 작업을 병렬로 실행
        ai_tasks = []
        human_tasks = []
        
        # AI 에이전트 작업 생성
        for ai_agent in self.ai_agents:
            if task_type in ai_agent.capabilities:
                task = WorkflowTask(
                    task_id=f"ai_{ai_agent.agent_id}_{int(time.time())}",
                    task_type=task_type,
                    assigned_to='ai',
                    input_data=task_data
                )
                ai_tasks.append(self._execute_ai_task_async(ai_agent, task))
        
        # 인간 에이전트 작업 생성
        for human_agent in self.human_agents:
            if task_type in human_agent.expertise_areas:
                task = WorkflowTask(
                    task_id=f"human_{human_agent.agent_id}_{int(time.time())}",
                    task_type=task_type,
                    assigned_to='human',
                    input_data=task_data
                )
                human_tasks.append(self._execute_human_task_async(human_agent, task))
        
        # 병렬 실행
        if ai_tasks:
            ai_results = await asyncio.gather(*ai_tasks, return_exceptions=True)
            result.ai_results = [r for r in ai_results if not isinstance(r, Exception)]
        
        if human_tasks:
            human_results = await asyncio.gather(*human_tasks, return_exceptions=True)
            result.human_results = [r for r in human_results if not isinstance(r, Exception)]
        
        # 결과 결합
        if combination_strategy in self.combination_strategies:
            combination_func = self.combination_strategies[combination_strategy]
            result.combined_results = combination_func(result.ai_results, result.human_results)
            result.consensus_score = self._calculate_consensus_score(result)
            result.disagreement_areas = self._identify_disagreement_areas(result)
        
        return result
    
    async def _execute_ai_task_async(self, agent: AIAgent, task: WorkflowTask) -> WorkflowTask:
        """AI 작업 비동기 실행"""
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = loop.run_in_executor(executor, agent.process_task, task)
            return await future
    
    async def _execute_human_task_async(self, agent: HumanAgent, task: WorkflowTask) -> WorkflowTask:
        """인간 작업 비동기 실행 (시뮬레이션)"""
        # 실제로는 인간의 비동기 작업을 위한 큐 시스템 필요
        await asyncio.sleep(0.1)  # 시뮬레이션을 위한 짧은 대기
        return agent.process_task(task)
    
    def _majority_vote_combination(self, ai_results: List[WorkflowTask], 
                                 human_results: List[WorkflowTask]) -> Dict[str, Any]:
        """다수결 투표 방식 결합"""
        
        all_predictions = []
        
        # AI 결과에서 예측 수집
        for task in ai_results:
            if task.output_data and 'prediction' in task.output_data:
                all_predictions.append(task.output_data['prediction'])
        
        # 인간 결과에서 예측 수집
        for task in human_results:
            if task.output_data and 'reviewed_prediction' in task.output_data:
                all_predictions.append(task.output_data['reviewed_prediction'])
            elif task.output_data and 'prediction' in task.output_data:
                all_predictions.append(task.output_data['prediction'])
        
        if not all_predictions:
            return {'error': 'No valid predictions found'}
        
        # 다수결 계산
        from collections import Counter
        vote_counts = Counter(all_predictions)
        final_prediction = vote_counts.most_common(1)[0][0]
        confidence = vote_counts[final_prediction] / len(all_predictions)
        
        return {
            'final_prediction': final_prediction,
            'confidence': confidence,
            'vote_breakdown': dict(vote_counts),
            'combination_method': 'majority_vote'
        }
    
    def _confidence_based_combination(self, ai_results: List[WorkflowTask], 
                                    human_results: List[WorkflowTask]) -> Dict[str, Any]:
        """신뢰도 기반 결합"""
        
        weighted_predictions = []
        
        # AI 결과 처리
        for task in ai_results:
            if task.output_data and 'prediction' in task.output_data:
                prediction = task.output_data['prediction']
                confidence = task.output_data.get('confidence', 0.5)
                weighted_predictions.append((prediction, confidence, 'ai'))
        
        # 인간 결과 처리 (인간의 경우 높은 신뢰도 부여)
        for task in human_results:
            if task.output_data and 'reviewed_prediction' in task.output_data:
                prediction = task.output_data['reviewed_prediction']
                confidence = task.output_data.get('final_confidence', 0.9)
                weighted_predictions.append((prediction, confidence, 'human'))
            elif task.output_data and 'prediction' in task.output_data:
                prediction = task.output_data['prediction']
                confidence = 0.85  # 인간 판단에 대한 기본 신뢰도
                weighted_predictions.append((prediction, confidence, 'human'))
        
        if not weighted_predictions:
            return {'error': 'No valid predictions found'}
        
        # 신뢰도 가중 평균
        spam_weight = sum(conf for pred, conf, source in weighted_predictions if pred == 'spam')
        ham_weight = sum(conf for pred, conf, source in weighted_predictions if pred == 'ham')
        
        total_weight = spam_weight + ham_weight
        
        if total_weight == 0:
            final_prediction = 'unknown'
            final_confidence = 0.0
        elif spam_weight > ham_weight:
            final_prediction = 'spam'
            final_confidence = spam_weight / total_weight
        else:
            final_prediction = 'ham'
            final_confidence = ham_weight / total_weight
        
        return {
            'final_prediction': final_prediction,
            'confidence': final_confidence,
            'spam_weight': spam_weight,
            'ham_weight': ham_weight,
            'combination_method': 'confidence_based',
            'contributor_breakdown': [
                {'source': source, 'prediction': pred, 'confidence': conf}
                for pred, conf, source in weighted_predictions
            ]
        }
    
    def _weighted_average_combination(self, ai_results: List[WorkflowTask], 
                                    human_results: List[WorkflowTask]) -> Dict[str, Any]:
        """가중 평균 결합 (숫자 결과용)"""
        # 특성 값들의 가중 평균 계산 예시
        return {'combination_method': 'weighted_average', 'note': 'Numeric results only'}
    
    def _expert_arbitration_combination(self, ai_results: List[WorkflowTask], 
                                      human_results: List[WorkflowTask]) -> Dict[str, Any]:
        """전문가 중재 결합"""
        # 복잡한 경우 전문가가 최종 판단
        return {'combination_method': 'expert_arbitration', 'note': 'Requires human expert decision'}
    
    def _calculate_consensus_score(self, result: ParallelWorkflowResult) -> float:
        """합의 점수 계산"""
        
        all_predictions = []
        
        # AI 결과에서 예측 수집
        for task in result.ai_results:
            if task.output_data and 'prediction' in task.output_data:
                all_predictions.append(task.output_data['prediction'])
        
        # 인간 결과에서 예측 수집
        for task in result.human_results:
            if task.output_data and 'prediction' in task.output_data:
                all_predictions.append(task.output_data['prediction'])
        
        if len(all_predictions) < 2:
            return 1.0  # 단일 결과인 경우 완전 합의
        
        # 가장 많은 예측과 일치하는 비율 계산
        from collections import Counter
        most_common_prediction = Counter(all_predictions).most_common(1)[0][0]
        consensus_count = all_predictions.count(most_common_prediction)
        
        return consensus_count / len(all_predictions)
    
    def _identify_disagreement_areas(self, result: ParallelWorkflowResult) -> List[str]:
        """의견 불일치 영역 식별"""
        
        disagreements = []
        
        # AI vs 인간 예측 비교
        ai_predictions = [task.output_data.get('prediction') for task in result.ai_results 
                         if task.output_data and 'prediction' in task.output_data]
        human_predictions = [task.output_data.get('prediction') for task in result.human_results
                           if task.output_data and 'prediction' in task.output_data]
        
        if ai_predictions and human_predictions:
            ai_majority = Counter(ai_predictions).most_common(1)[0][0]
            human_majority = Counter(human_predictions).most_common(1)[0][0]
            
            if ai_majority != human_majority:
                disagreements.append(f"AI 다수의견({ai_majority}) vs 인간 다수의견({human_majority})")
        
        return disagreements

# 병렬 협업 시연
print("\n⚡ 병렬 협업 워크플로우 시연")
print("=" * 50)

async def demo_parallel_collaboration():
    """병렬 협업 데모"""
    
    # 병렬 협업 엔진 생성
    parallel_engine = ParallelCollaborationEngine()
    
    # 여러 AI 에이전트 등록 (다양한 접근법)
    ai_agent1 = AIAgent("rule_based_ai", ['spam_classification'])
    ai_agent2 = AIAgent("ml_based_ai", ['spam_classification'])
    
    # 인간 전문가 등록
    human_expert = HumanAgent("sms_expert", ['spam_classification'])
    
    parallel_engine.register_ai_agent(ai_agent1)
    parallel_engine.register_ai_agent(ai_agent2)
    parallel_engine.register_human_agent(human_expert)
    
    # 테스트 메시지
    test_message = {
        'text': "Congratulations! You've won $1000! Call 555-0123 now to claim your prize!"
    }
    
    print(f"📨 분석 메시지: {test_message['text']}")
    
    # 다양한 결합 전략으로 병렬 처리
    strategies = ['majority_vote', 'confidence_based']
    
    for strategy in strategies:
        print(f"\n🔄 {strategy} 전략 실행:")
        
        result = await parallel_engine.execute_parallel_workflow(
            task_data=test_message,
            task_type='spam_classification',
            combination_strategy=strategy
        )
        
        print(f"AI 결과 수: {len(result.ai_results)}")
        print(f"인간 결과 수: {len(result.human_results)}")
        print(f"합의 점수: {result.consensus_score:.2f}")
        
        if result.combined_results:
            combined = result.combined_results
            print(f"최종 예측: {combined.get('final_prediction', 'N/A')}")
            print(f"최종 신뢰도: {combined.get('confidence', 0):.3f}")
        
        if result.disagreement_areas:
            print(f"의견 불일치: {result.disagreement_areas}")

# 비동기 데모 실행
import asyncio
asyncio.run(demo_parallel_collaboration())
```

**코드 해설:**
- **비동기 처리**: AI와 인간의 작업을 동시에 실행하여 전체 처리 시간 단축
- **다양한 결합 전략**: 다수결, 신뢰도 기반, 가중 평균 등 상황에 맞는 결합 방식 선택
- **합의 점수**: 참여자들 간의 의견 일치 정도를 정량적으로 측정
- **의견 불일치 식별**: 추가 논의나 조정이 필요한 영역을 자동 탐지

#### **패턴 3: 계층적 협업 (Hierarchical Collaboration)**
AI가 1차 처리를 하고 인간이 검토/승인하는 계층 구조입니다.

```python
from typing import Protocol, Dict, Any, List
from enum import Enum

class ReviewDecision(Enum):
    """검토 결정"""
    APPROVE = "approve"
    REJECT = "reject" 
    MODIFY = "modify"
    ESCALATE = "escalate"

class QualityGate(Protocol):
    """품질 게이트 인터페이스"""
    
    def evaluate(self, task_result: WorkflowTask) -> Dict[str, Any]:
        """품질 평가"""
        ...
    
    def should_escalate(self, task_result: WorkflowTask) -> bool:
        """상위 레벨로 에스컬레이션 필요 여부"""
        ...

class AIQualityGate:
    """AI 품질 게이트"""
    
    def __init__(self, thresholds: Dict[str, float]):
        self.thresholds = thresholds
    
    def evaluate(self, task_result: WorkflowTask) -> Dict[str, Any]:
        """AI 결과 품질 평가"""
        
        if not task_result.output_data:
            return {
                'pass': False,
                'reason': 'No output data',
                'confidence_score': 0.0
            }
        
        confidence = task_result.output_data.get('confidence', 0.0)
        
        # 신뢰도 기반 평가
        if confidence >= self.thresholds.get('high_confidence', 0.9):
            return {
                'pass': True,
                'level': 'high_confidence',
                'confidence_score': confidence,
                'auto_approve': True
            }
        elif confidence >= self.thresholds.get('medium_confidence', 0.7):
            return {
                'pass': True, 
                'level': 'medium_confidence',
                'confidence_score': confidence,
                'requires_review': True
            }
        else:
            return {
                'pass': False,
                'level': 'low_confidence',
                'confidence_score': confidence,
                'requires_human_review': True
            }
    
    def should_escalate(self, task_result: WorkflowTask) -> bool:
        """에스컬레이션 필요 여부"""
        evaluation = self.evaluate(task_result)
        return evaluation.get('requires_human_review', False)

class HumanReviewGate:
    """인간 검토 게이트"""
    
    def __init__(self, reviewer_expertise: List[str]):
        self.reviewer_expertise = reviewer_expertise
        self.review_criteria = {
            'accuracy': 0.8,
            'relevance': 0.8,
            'business_impact': 0.7
        }
    
    def evaluate(self, task_result: WorkflowTask) -> Dict[str, Any]:
        """인간 검토 평가"""
        
        # 시뮬레이션된 인간 검토
        accuracy_score = 0.85  # 실제로는 인간 검토자가 평가
        relevance_score = 0.90
        business_impact_score = 0.75
        
        overall_score = (accuracy_score + relevance_score + business_impact_score) / 3
        
        decision = ReviewDecision.APPROVE
        if overall_score < 0.6:
            decision = ReviewDecision.REJECT
        elif overall_score < 0.75:
            decision = ReviewDecision.MODIFY
        
        return {
            'decision': decision,
            'overall_score': overall_score,
            'detailed_scores': {
                'accuracy': accuracy_score,
                'relevance': relevance_score,
                'business_impact': business_impact_score
            },
            'comments': self._generate_review_comments(decision, overall_score)
        }
    
    def should_escalate(self, task_result: WorkflowTask) -> bool:
        """상위 관리자 에스컬레이션 필요 여부"""
        evaluation = self.evaluate(task_result)
        return evaluation['decision'] == ReviewDecision.ESCALATE
    
    def _generate_review_comments(self, decision: ReviewDecision, score: float) -> str:
        """검토 의견 생성"""
        if decision == ReviewDecision.APPROVE:
            return f"품질 기준을 충족합니다 (점수: {score:.2f})"
        elif decision == ReviewDecision.MODIFY:
            return f"일부 수정이 필요합니다 (점수: {score:.2f})"
        else:
            return f"기준에 미달하여 재작업이 필요합니다 (점수: {score:.2f})"

class HierarchicalWorkflow:
    """계층적 워크플로우"""
    
    def __init__(self):
        self.ai_layer: List[AIAgent] = []
        self.quality_gates: List[QualityGate] = []
        self.human_reviewers: List[HumanAgent] = []
        self.escalation_levels = ['junior', 'senior', 'manager']
        self.processing_stats = {
            'total_processed': 0,
            'auto_approved': 0,
            'human_reviewed': 0,
            'escalated': 0,
            'rejected': 0
        }
    
    def add_ai_agent(self, agent: AIAgent):
        """AI 에이전트 추가"""
        self.ai_layer.append(agent)
    
    def add_quality_gate(self, gate: QualityGate):
        """품질 게이트 추가"""
        self.quality_gates.append(gate)
    
    def add_human_reviewer(self, reviewer: HumanAgent, level: str = 'junior'):
        """인간 검토자 추가"""
        reviewer.review_level = level
        self.human_reviewers.append(reviewer)
    
    def process_hierarchical_workflow(self, task_data: Dict[str, Any], 
                                    task_type: str) -> Dict[str, Any]:
        """계층적 워크플로우 처리"""
        
        self.processing_stats['total_processed'] += 1
        
        # 1단계: AI 처리
        ai_result = self._process_ai_layer(task_data, task_type)
        if not ai_result:
            return {'error': 'AI 처리 실패', 'stage': 'ai_processing'}
        
        # 2단계: 품질 게이트 통과
        quality_result = self._evaluate_quality_gates(ai_result)
        
        # 3단계: 자동 승인 or 인간 검토
        if quality_result.get('auto_approve', False):
            self.processing_stats['auto_approved'] += 1
            return {
                'final_result': ai_result.output_data,
                'processing_path': 'auto_approved',
                'quality_score': quality_result.get('confidence_score', 0),
                'stage': 'completed'
            }
        
        # 4단계: 인간 검토 계층 처리
        if quality_result.get('requires_review', False):
            review_result = self._process_human_review_hierarchy(ai_result)
            return review_result
        
        # 5단계: 거부된 경우
        self.processing_stats['rejected'] += 1
        return {
            'final_result': None,
            'processing_path': 'rejected',
            'quality_score': quality_result.get('confidence_score', 0),
            'stage': 'rejected'
        }
    
    def _process_ai_layer(self, task_data: Dict[str, Any], 
                         task_type: str) -> Optional[WorkflowTask]:
        """AI 계층 처리"""
        
        for ai_agent in self.ai_layer:
            if ai_agent.can_handle(WorkflowTask('', task_type, 'ai', task_data)):
                task = WorkflowTask(
                    task_id=f"hierarchical_{ai_agent.agent_id}_{int(time.time())}",
                    task_type=task_type,
                    assigned_to='ai',
                    input_data=task_data
                )
                
                result = ai_agent.process_task(task)
                if result.status == TaskStatus.COMPLETED:
                    return result
        
        return None
    
    def _evaluate_quality_gates(self, task_result: WorkflowTask) -> Dict[str, Any]:
        """품질 게이트 평가"""
        
        for gate in self.quality_gates:
            evaluation = gate.evaluate(task_result)
            
            # 첫 번째 게이트가 결정적
            if not evaluation.get('pass', False):
                return evaluation
            
            # 에스컬레이션 필요한 경우
            if gate.should_escalate(task_result):
                evaluation['requires_review'] = True
            
            return evaluation
        
        # 품질 게이트가 없는 경우 기본 승인
        return {'pass': True, 'auto_approve': True}
    
    def _process_human_review_hierarchy(self, task_result: WorkflowTask) -> Dict[str, Any]:
        """인간 검토 계층 처리"""
        
        self.processing_stats['human_reviewed'] += 1
        
        # 검토 레벨 순서대로 처리
        for level in self.escalation_levels:
            reviewers = [r for r in self.human_reviewers if r.review_level == level]
            
            if not reviewers:
                continue
            
            # 첫 번째 적합한 검토자 선택
            reviewer = reviewers[0]
            
            # 검토 수행
            review_gate = HumanReviewGate(reviewer.expertise_areas)
            review_result = review_gate.evaluate(task_result)
            
            decision = review_result['decision']
            
            if decision == ReviewDecision.APPROVE:
                return {
                    'final_result': task_result.output_data,
                    'processing_path': f'human_approved_{level}',
                    'review_score': review_result['overall_score'],
                    'reviewer_level': level,
                    'stage': 'completed'
                }
            
            elif decision == ReviewDecision.REJECT:
                self.processing_stats['rejected'] += 1
                return {
                    'final_result': None,
                    'processing_path': f'human_rejected_{level}',
                    'review_score': review_result['overall_score'],
                    'rejection_reason': review_result['comments'],
                    'stage': 'rejected'
                }
            
            elif decision == ReviewDecision.MODIFY:
                # 수정 후 재처리 (시뮬레이션)
                modified_result = self._apply_modifications(task_result, review_result)
                if modified_result:
                    return {
                        'final_result': modified_result.output_data,
                        'processing_path': f'modified_and_approved_{level}',
                        'review_score': review_result['overall_score'],
                        'stage': 'completed'
                    }
            
            elif decision == ReviewDecision.ESCALATE:
                # 다음 레벨로 에스컬레이션
                self.processing_stats['escalated'] += 1
                continue
        
        # 모든 레벨을 거쳐도 해결되지 않은 경우
        return {
            'final_result': None,
            'processing_path': 'escalation_failed',
            'stage': 'requires_management_decision'
        }
    
    def _apply_modifications(self, task_result: WorkflowTask, 
                           review_result: Dict[str, Any]) -> Optional[WorkflowTask]:
        """수정사항 적용 (시뮬레이션)"""
        
        # 실제로는 수정 지침에 따라 재처리
        modified_output = task_result.output_data.copy()
        modified_output['human_modified'] = True
        modified_output['modification_reason'] = review_result['comments']
        
        task_result.output_data = modified_output
        return task_result
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """처리 통계 반환"""
        
        total = self.processing_stats['total_processed']
        if total == 0:
            return self.processing_stats
        
        return {
            **self.processing_stats,
            'auto_approval_rate': self.processing_stats['auto_approved'] / total,
            'human_review_rate': self.processing_stats['human_reviewed'] / total,
            'rejection_rate': self.processing_stats['rejected'] / total,
            'escalation_rate': self.processing_stats['escalated'] / total
        }

# 계층적 협업 시연
print("\n🏢 계층적 협업 워크플로우 시연")
print("=" * 50)

# 계층적 워크플로우 설정
hierarchical_workflow = HierarchicalWorkflow()

# AI 에이전트 추가
ai_classifier = AIAgent("hierarchical_ai", ['spam_classification'])
hierarchical_workflow.add_ai_agent(ai_classifier)

# 품질 게이트 추가
quality_gate = AIQualityGate({
    'high_confidence': 0.9,
    'medium_confidence': 0.7
})
hierarchical_workflow.add_quality_gate(quality_gate)

# 인간 검토자 추가 (다단계)
junior_reviewer = HumanAgent("junior_analyst", ['spam_classification'])
senior_reviewer = HumanAgent("senior_analyst", ['spam_classification'])
manager_reviewer = HumanAgent("manager", ['spam_classification'])

hierarchical_workflow.add_human_reviewer(junior_reviewer, 'junior')
hierarchical_workflow.add_human_reviewer(senior_reviewer, 'senior')
hierarchical_workflow.add_human_reviewer(manager_reviewer, 'manager')

# 다양한 신뢰도 수준의 테스트 메시지
test_cases = [
    {
        'text': 'Hello, how are you today?',  # 높은 신뢰도 예상 (정상)
        'expected_confidence': 'high'
    },
    {
        'text': 'Free money! Call now!',  # 높은 신뢰도 예상 (스팸)
        'expected_confidence': 'high'
    },
    {
        'text': 'Limited time offer for students',  # 중간 신뢰도 예상
        'expected_confidence': 'medium'
    },
    {
        'text': 'Complex message with ambiguous content',  # 낮은 신뢰도 예상
        'expected_confidence': 'low'
    }
]

print(f"\n📋 {len(test_cases)}개 메시지 계층적 처리:")

for i, test_case in enumerate(test_cases, 1):
    print(f"\n--- 테스트 케이스 {i} ---")
    print(f"메시지: {test_case['text']}")
    print(f"예상 신뢰도: {test_case['expected_confidence']}")
    
    result = hierarchical_workflow.process_hierarchical_workflow(
        task_data={'text': test_case['text']},
        task_type='spam_classification'
    )
    
    print(f"처리 경로: {result.get('processing_path', 'unknown')}")
    print(f"처리 단계: {result.get('stage', 'unknown')}")
    
    if result.get('final_result'):
        final_result = result['final_result']
        print(f"최종 결과: {final_result.get('prediction', 'N/A')}")
        if 'confidence' in final_result:
            print(f"신뢰도: {final_result['confidence']:.3f}")

# 처리 통계 출력
print(f"\n📊 처리 통계:")
stats = hierarchical_workflow.get_processing_statistics()
for key, value in stats.items():
    if key.endswith('_rate'):
        print(f"{key}: {value:.1%}")
    else:
        print(f"{key}: {value}")
```

**코드 해설:**
- **다단계 품질 관리**: AI → 품질 게이트 → 인간 검토의 계층적 품질 보장
- **에스컬레이션 체계**: 복잡도에 따라 적절한 레벨의 전문가에게 자동 배정
- **처리 통계**: 자동화 효율성과 인간 개입 필요성을 정량적으로 추적
- **유연한 승인 프로세스**: 상황에 따른 승인, 거부, 수정, 에스컬레이션 결정

> 💡 **협업 패턴 선택 가이드**
> 
> **순차적 협업**: 
> - 단계별 정확성이 중요한 경우
> - 각 단계의 결과가 다음 단계의 입력이 되는 경우
> 
> **병렬적 협업**:
> - 빠른 처리 시간이 필요한 경우
> - 다양한 관점의 검증이 필요한 경우
> 
> **계층적 협업**:
> - 품질 보장이 최우선인 경우
> - 복잡도에 따른 차별적 처리가 필요한 경우

> 🖼️ **이미지 생성 프롬프트**: 
> "인간-AI 협업의 3가지 패턴을 보여주는 플로우 다이어그램. 왼쪽에는 순차적(Sequential), 가운데에는 병렬적(Parallel), 오른쪽에는 계층적(Hierarchical) 협업 패턴이 각각 다른 색상과 화살표 방향으로 표현된 모던한 워크플로우 차트"

## 3. 품질 관리를 위한 체크포인트 설정

품질 관리는 마치 제조업의 품질 검사와 같습니다. 제품이 완성되기 전 여러 단계에서 품질을 확인하여 최종 제품의 품질을 보장합니다. 데이터 분석에서도 각 단계마다 적절한 체크포인트를 설정하여 품질을 관리해야 합니다.

### 3.1 체크포인트 설계 원칙

#### **원칙 1: Critical Control Points (CCP) 식별**
데이터 분석 워크플로우에서 품질에 가장 큰 영향을 미치는 핵심 통제점을 식별합니다.

```python
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import json
import hashlib

class CheckpointType(Enum):
    """체크포인트 유형"""
    DATA_QUALITY = "data_quality"
    ALGORITHM_PERFORMANCE = "algorithm_performance"
    BUSINESS_LOGIC = "business_logic"
    ETHICAL_COMPLIANCE = "ethical_compliance"
    SECURITY_VALIDATION = "security_validation"

class CheckpointSeverity(Enum):
    """체크포인트 심각도"""
    BLOCKER = "blocker"      # 진행 불가
    CRITICAL = "critical"    # 즉시 수정 필요
    MAJOR = "major"         # 수정 권장
    MINOR = "minor"         # 개선 제안

@dataclass
class CheckpointResult:
    """체크포인트 결과"""
    checkpoint_id: str
    checkpoint_type: CheckpointType
    passed: bool
    score: float  # 0.0 ~ 1.0
    severity: CheckpointSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

class QualityCheckpoint:
    """품질 체크포인트 기본 클래스"""
    
    def __init__(self, 
                 checkpoint_id: str,
                 checkpoint_type: CheckpointType,
                 acceptance_threshold: float = 0.8):
        self.checkpoint_id = checkpoint_id
        self.checkpoint_type = checkpoint_type
        self.acceptance_threshold = acceptance_threshold
        self.validation_rules = []
    
    def add_validation_rule(self, rule_func: Callable, weight: float = 1.0):
        """검증 규칙 추가"""
        self.validation_rules.append({
            'function': rule_func,
            'weight': weight
        })
    
    def evaluate(self, data: Dict[str, Any]) -> CheckpointResult:
        """체크포인트 평가 실행"""
        
        if not self.validation_rules:
            return CheckpointResult(
                checkpoint_id=self.checkpoint_id,
                checkpoint_type=self.checkpoint_type,
                passed=True,
                score=1.0,
                severity=CheckpointSeverity.MINOR,
                message="검증 규칙이 정의되지 않음"
            )
        
        # 각 규칙별 점수 계산
        rule_scores = []
        rule_details = {}
        
        for i, rule in enumerate(self.validation_rules):
            try:
                rule_result = rule['function'](data)
                if isinstance(rule_result, dict):
                    score = rule_result.get('score', 0.0)
                    details = rule_result.get('details', {})
                else:
                    score = float(rule_result)
                    details = {}
                
                weighted_score = score * rule['weight']
                rule_scores.append(weighted_score)
                rule_details[f'rule_{i}'] = {
                    'score': score,
                    'weight': rule['weight'],
                    'weighted_score': weighted_score,
                    'details': details
                }
                
            except Exception as e:
                rule_scores.append(0.0)
                rule_details[f'rule_{i}'] = {
                    'error': str(e),
                    'score': 0.0
                }
        
        # 가중 평균 계산
        total_weight = sum(rule['weight'] for rule in self.validation_rules)
        overall_score = sum(rule_scores) / total_weight if total_weight > 0 else 0.0
        
        # 합격/불합격 판정
        passed = overall_score >= self.acceptance_threshold
        
        # 심각도 결정
        if overall_score < 0.3:
            severity = CheckpointSeverity.BLOCKER
        elif overall_score < 0.6:
            severity = CheckpointSeverity.CRITICAL
        elif overall_score < 0.8:
            severity = CheckpointSeverity.MAJOR
        else:
            severity = CheckpointSeverity.MINOR
        
        # 결과 메시지 생성
        message = self._generate_message(passed, overall_score, severity)
        
        # 권장사항 생성
        recommendations = self._generate_recommendations(overall_score, rule_details)
        
        return CheckpointResult(
            checkpoint_id=self.checkpoint_id,
            checkpoint_type=self.checkpoint_type,
            passed=passed,
            score=overall_score,
            severity=severity,
            message=message,
            details=rule_details,
            recommendations=recommendations
        )
    
    def _generate_message(self, passed: bool, score: float, severity: CheckpointSeverity) -> str:
        """결과 메시지 생성"""
        status = "통과" if passed else "실패"
        return f"{self.checkpoint_id} {status} (점수: {score:.2f}, 심각도: {severity.value})"
    
    def _generate_recommendations(self, score: float, rule_details: Dict) -> List[str]:
        """권장사항 생성"""
        recommendations = []
        
        if score < 0.5:
            recommendations.append("전체적인 품질 개선이 시급합니다")
        
        # 실패한 규칙들에 대한 구체적 권장사항
        for rule_name, details in rule_details.items():
            if details.get('score', 0) < 0.6:
                recommendations.append(f"{rule_name} 개선 필요")
        
        if score >= 0.8:
            recommendations.append("현재 품질 수준을 유지하세요")
        
        return recommendations

class DataQualityCheckpoint(QualityCheckpoint):
    """데이터 품질 체크포인트"""
    
    def __init__(self):
        super().__init__("data_quality_check", CheckpointType.DATA_QUALITY, 0.85)
        self._setup_validation_rules()
    
    def _setup_validation_rules(self):
        """데이터 품질 검증 규칙 설정"""
        
        # 완전성 검사
        self.add_validation_rule(self._check_completeness, weight=2.0)
        
        # 일관성 검사
        self.add_validation_rule(self._check_consistency, weight=1.5)
        
        # 정확성 검사
        self.add_validation_rule(self._check_accuracy, weight=2.0)
        
        # 유효성 검사
        self.add_validation_rule(self._check_validity, weight=1.0)
    
    def _check_completeness(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """완전성 검사 - 결측치 비율"""
        
        dataset = data.get('dataset')
        if dataset is None:
            return {'score': 0.0, 'details': {'error': 'No dataset provided'}}
        
        try:
            import pandas as pd
            if isinstance(dataset, pd.DataFrame):
                total_cells = dataset.size
                missing_cells = dataset.isnull().sum().sum()
                completeness_ratio = 1 - (missing_cells / total_cells)
                
                return {
                    'score': completeness_ratio,
                    'details': {
                        'total_cells': total_cells,
                        'missing_cells': missing_cells,
                        'completeness_ratio': completeness_ratio
                    }
                }
            else:
                return {'score': 1.0, 'details': {'note': 'Non-dataframe data assumed complete'}}
                
        except Exception as e:
            return {'score': 0.0, 'details': {'error': str(e)}}
    
    def _check_consistency(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """일관성 검사 - 데이터 타입 및 형식 일관성"""
        
        dataset = data.get('dataset')
        if dataset is None:
            return {'score': 0.0, 'details': {'error': 'No dataset provided'}}
        
        try:
            import pandas as pd
            if isinstance(dataset, pd.DataFrame):
                consistency_scores = []
                
                for column in dataset.columns:
                    if dataset[column].dtype == 'object':
                        # 문자열 컬럼의 형식 일관성 확인
                        unique_patterns = len(set(
                            str(val).strip().lower() for val in dataset[column].dropna()
                        ))
                        total_values = len(dataset[column].dropna())
                        
                        if total_values > 0:
                            pattern_ratio = unique_patterns / total_values
                            # 패턴이 너무 다양하면 일관성이 낮음
                            consistency_score = max(0, 1 - pattern_ratio * 2)
                            consistency_scores.append(consistency_score)
                
                overall_consistency = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 1.0
                
                return {
                    'score': overall_consistency,
                    'details': {
                        'column_consistency_scores': dict(zip(dataset.columns, consistency_scores)),
                        'overall_consistency': overall_consistency
                    }
                }
            else:
                return {'score': 0.8, 'details': {'note': 'Basic consistency assumed for non-dataframe'}}
                
        except Exception as e:
            return {'score': 0.0, 'details': {'error': str(e)}}
    
    def _check_accuracy(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """정확성 검사 - 비즈니스 규칙 준수"""
        
        dataset = data.get('dataset')
        business_rules = data.get('business_rules', {})
        
        if dataset is None:
            return {'score': 0.0, 'details': {'error': 'No dataset provided'}}
        
        try:
            accuracy_scores = []
            
            # SMS 메시지 길이 규칙 (예시)
            if 'message' in dataset.columns:
                message_lengths = dataset['message'].str.len()
                valid_length_ratio = ((message_lengths >= 1) & (message_lengths <= 1600)).mean()
                accuracy_scores.append(valid_length_ratio)
            
            # 라벨 값 유효성 (예시)
            if 'label' in dataset.columns:
                valid_labels = business_rules.get('valid_labels', ['spam', 'ham'])
                valid_label_ratio = dataset['label'].isin(valid_labels).mean()
                accuracy_scores.append(valid_label_ratio)
            
            overall_accuracy = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0.8
            
            return {
                'score': overall_accuracy,
                'details': {
                    'accuracy_checks': len(accuracy_scores),
                    'overall_accuracy': overall_accuracy
                }
            }
            
        except Exception as e:
            return {'score': 0.0, 'details': {'error': str(e)}}
    
    def _check_validity(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """유효성 검사 - 데이터 스키마 준수"""
        
        dataset = data.get('dataset')
        expected_schema = data.get('expected_schema', {})
        
        if dataset is None:
            return {'score': 0.0, 'details': {'error': 'No dataset provided'}}
        
        try:
            validity_scores = []
            
            # 필수 컬럼 존재 확인
            required_columns = expected_schema.get('required_columns', [])
            if required_columns:
                present_columns = sum(1 for col in required_columns if col in dataset.columns)
                column_validity = present_columns / len(required_columns)
                validity_scores.append(column_validity)
            
            # 데이터 타입 확인
            expected_dtypes = expected_schema.get('dtypes', {})
            if expected_dtypes:
                type_matches = 0
                for col, expected_dtype in expected_dtypes.items():
                    if col in dataset.columns:
                        actual_dtype = str(dataset[col].dtype)
                        if expected_dtype in actual_dtype or actual_dtype in expected_dtype:
                            type_matches += 1
                
                dtype_validity = type_matches / len(expected_dtypes)
                validity_scores.append(dtype_validity)
            
            overall_validity = sum(validity_scores) / len(validity_scores) if validity_scores else 0.9
            
            return {
                'score': overall_validity,
                'details': {
                    'validity_checks': len(validity_scores),
                    'overall_validity': overall_validity
                }
            }
            
        except Exception as e:
            return {'score': 0.0, 'details': {'error': str(e)}}

class AlgorithmPerformanceCheckpoint(QualityCheckpoint):
    """알고리즘 성능 체크포인트"""
    
    def __init__(self):
        super().__init__("algorithm_performance_check", CheckpointType.ALGORITHM_PERFORMANCE, 0.80)
        self._setup_validation_rules()
    
    def _setup_validation_rules(self):
        """알고리즘 성능 검증 규칙 설정"""
        
        # 정확도 검사
        self.add_validation_rule(self._check_accuracy_metrics, weight=2.0)
        
        # 일반화 성능 검사
        self.add_validation_rule(self._check_generalization, weight=1.5)
        
        # 편향성 검사
        self.add_validation_rule(self._check_bias, weight=1.0)
        
        # 안정성 검사
        self.add_validation_rule(self._check_stability, weight=1.0)
    
    def _check_accuracy_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """정확도 지표 검사"""
        
        metrics = data.get('performance_metrics', {})
        
        if not metrics:
            return {'score': 0.0, 'details': {'error': 'No performance metrics provided'}}
        
        # 주요 지표들의 임계값
        thresholds = {
            'accuracy': 0.85,
            'precision': 0.80,
            'recall': 0.80,
            'f1_score': 0.80
        }
        
        scores = []
        details = {}
        
        for metric_name, threshold in thresholds.items():
            if metric_name in metrics:
                metric_value = metrics[metric_name]
                metric_score = min(1.0, metric_value / threshold)
                scores.append(metric_score)
                details[metric_name] = {
                    'value': metric_value,
                    'threshold': threshold,
                    'score': metric_score
                }
        
        overall_score = sum(scores) / len(scores) if scores else 0.0
        
        return {
            'score': overall_score,
            'details': details
        }
    
    def _check_generalization(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """일반화 성능 검사 - 과적합 탐지"""
        
        train_metrics = data.get('train_metrics', {})
        test_metrics = data.get('test_metrics', {})
        
        if not train_metrics or not test_metrics:
            return {'score': 0.5, 'details': {'warning': 'Insufficient data for generalization check'}}
        
        # 훈련/테스트 성능 차이 분석
        performance_gaps = {}
        gap_scores = []
        
        common_metrics = set(train_metrics.keys()) & set(test_metrics.keys())
        
        for metric in common_metrics:
            train_val = train_metrics[metric]
            test_val = test_metrics[metric]
            
            # 성능 차이 (과적합 정도)
            gap = abs(train_val - test_val)
            gap_score = max(0, 1 - gap * 2)  # 차이가 클수록 점수 낮음
            
            performance_gaps[metric] = {
                'train': train_val,
                'test': test_val,
                'gap': gap,
                'score': gap_score
            }
            gap_scores.append(gap_score)
        
        overall_generalization = sum(gap_scores) / len(gap_scores) if gap_scores else 0.5
        
        return {
            'score': overall_generalization,
            'details': {
                'performance_gaps': performance_gaps,
                'generalization_score': overall_generalization
            }
        }
    
    def _check_bias(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """편향성 검사"""
        
        predictions = data.get('predictions', [])
        true_labels = data.get('true_labels', [])
        
        if not predictions or not true_labels:
            return {'score': 0.8, 'details': {'note': 'Bias check skipped due to insufficient data'}}
        
        try:
            # 클래스별 성능 균형 확인
            from collections import Counter
            
            # 예측 분포
            pred_dist = Counter(predictions)
            true_dist = Counter(true_labels)
            
            # 분포 균형 점수 계산
            total_predictions = len(predictions)
            balance_scores = []
            
            for label in true_dist.keys():
                true_ratio = true_dist[label] / len(true_labels)
                pred_ratio = pred_dist.get(label, 0) / total_predictions
                
                # 분포 차이가 클수록 편향성 높음
                ratio_diff = abs(true_ratio - pred_ratio)
                balance_score = max(0, 1 - ratio_diff * 2)
                balance_scores.append(balance_score)
            
            overall_balance = sum(balance_scores) / len(balance_scores) if balance_scores else 0.8
            
            return {
                'score': overall_balance,
                'details': {
                    'true_distribution': dict(true_dist),
                    'predicted_distribution': dict(pred_dist),
                    'balance_score': overall_balance
                }
            }
            
        except Exception as e:
            return {'score': 0.5, 'details': {'error': str(e)}}
    
    def _check_stability(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """안정성 검사 - 예측 일관성"""
        
        # 여러 실행 결과 또는 교차 검증 결과 확인
        cv_scores = data.get('cross_validation_scores', [])
        
        if not cv_scores:
            return {'score': 0.7, 'details': {'note': 'Stability check skipped due to insufficient data'}}
        
        try:
            import numpy as np
            
            mean_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)
            
            # 표준편차가 작을수록 안정성 높음
            stability_score = max(0, 1 - std_score * 5)
            
            return {
                'score': stability_score,
                'details': {
                    'cv_scores': cv_scores,
                    'mean_score': mean_score,
                    'std_score': std_score,
                    'stability_score': stability_score
                }
            }
            
        except Exception as e:
            return {'score': 0.5, 'details': {'error': str(e)}}

class QualityGateSystem:
    """품질 게이트 시스템"""
    
    def __init__(self):
        self.checkpoints: List[QualityCheckpoint] = []
        self.execution_history: List[Dict[str, Any]] = []
        self.global_thresholds = {
            CheckpointSeverity.BLOCKER: 0.0,     # 차단 레벨은 진행 불가
            CheckpointSeverity.CRITICAL: 0.6,    # 중요 이슈 임계값
            CheckpointSeverity.MAJOR: 0.8,       # 주요 이슈 임계값
            CheckpointSeverity.MINOR: 0.9        # 경미한 이슈 임계값
        }
    
    def add_checkpoint(self, checkpoint: QualityCheckpoint):
        """체크포인트 추가"""
        self.checkpoints.append(checkpoint)
    
    def execute_all_checkpoints(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """모든 체크포인트 실행"""
        
        execution_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        checkpoint_results = []
        overall_passed = True
        blocker_count = 0
        critical_count = 0
        
        for checkpoint in self.checkpoints:
            result = checkpoint.evaluate(data)
            checkpoint_results.append(result)
            
            # 심각도별 카운트
            if result.severity == CheckpointSeverity.BLOCKER:
                blocker_count += 1
                overall_passed = False
            elif result.severity == CheckpointSeverity.CRITICAL:
                critical_count += 1
                if not result.passed:
                    overall_passed = False
        
        # 전체 점수 계산 (가중 평균)
        total_score = sum(result.score for result in checkpoint_results)
        average_score = total_score / len(checkpoint_results) if checkpoint_results else 0.0
        
        # 진행 가능 여부 결정
        can_proceed = overall_passed and blocker_count == 0
        
        execution_summary = {
            'execution_id': execution_id,
            'timestamp': time.time(),
            'overall_passed': overall_passed,
            'can_proceed': can_proceed,
            'average_score': average_score,
            'checkpoint_results': checkpoint_results,
            'summary_stats': {
                'total_checkpoints': len(checkpoint_results),
                'passed_checkpoints': sum(1 for r in checkpoint_results if r.passed),
                'blocker_count': blocker_count,
                'critical_count': critical_count,
                'major_count': sum(1 for r in checkpoint_results if r.severity == CheckpointSeverity.MAJOR),
                'minor_count': sum(1 for r in checkpoint_results if r.severity == CheckpointSeverity.MINOR)
            },
            'recommendations': self._generate_global_recommendations(checkpoint_results)
        }
        
        # 실행 기록 저장
        self.execution_history.append(execution_summary)
        
        return execution_summary
    
    def _generate_global_recommendations(self, results: List[CheckpointResult]) -> List[str]:
        """전체 권장사항 생성"""
        
        recommendations = []
        
        # 차단 이슈가 있는 경우
        blockers = [r for r in results if r.severity == CheckpointSeverity.BLOCKER]
        if blockers:
            recommendations.append(f"🚨 {len(blockers)}개의 차단 이슈를 즉시 해결해야 합니다")
        
        # 중요 이슈가 많은 경우
        criticals = [r for r in results if r.severity == CheckpointSeverity.CRITICAL]
        if len(criticals) > 2:
            recommendations.append(f"⚠️ {len(criticals)}개의 중요 이슈가 있습니다. 우선순위를 정해 해결하세요")
        
        # 전체적인 품질 수준 평가
        avg_score = sum(r.score for r in results) / len(results) if results else 0
        if avg_score >= 0.9:
            recommendations.append("✅ 전체적으로 우수한 품질 수준입니다")
        elif avg_score >= 0.7:
            recommendations.append("👍 양호한 품질 수준이지만 일부 개선 여지가 있습니다")
        else:
            recommendations.append("📋 전체적인 품질 개선이 필요합니다")
        
        return recommendations
    
    def get_quality_trends(self) -> Dict[str, Any]:
        """품질 트렌드 분석"""
        
        if len(self.execution_history) < 2:
            return {'message': 'Insufficient data for trend analysis'}
        
        # 최근 실행들의 점수 추이
        recent_scores = [exec['average_score'] for exec in self.execution_history[-10:]]
        
        # 트렌드 계산
        if len(recent_scores) >= 2:
            trend = recent_scores[-1] - recent_scores[-2]
            if trend > 0.05:
                trend_status = 'improving'
            elif trend < -0.05:
                trend_status = 'declining'
            else:
                trend_status = 'stable'
        else:
            trend_status = 'insufficient_data'
        
        return {
            'recent_scores': recent_scores,
            'trend_status': trend_status,
            'average_score': sum(recent_scores) / len(recent_scores),
            'score_variance': sum((x - sum(recent_scores)/len(recent_scores))**2 for x in recent_scores) / len(recent_scores) if len(recent_scores) > 1 else 0
        }

# SMS 스팸 탐지 프로젝트 품질 게이트 시연
print("\n🎯 품질 관리 체크포인트 시연")
print("=" * 50)

# 품질 게이트 시스템 설정
quality_system = QualityGateSystem()

# 데이터 품질 체크포인트 추가
data_quality_checkpoint = DataQualityCheckpoint()
quality_system.add_checkpoint(data_quality_checkpoint)

# 알고리즘 성능 체크포인트 추가
algorithm_performance_checkpoint = AlgorithmPerformanceCheckpoint()
quality_system.add_checkpoint(algorithm_performance_checkpoint)

# 시뮬레이션된 테스트 데이터
import pandas as pd
import numpy as np

# SMS 데이터셋 시뮬레이션
sms_data = pd.DataFrame({
    'message': [
        'Hello, how are you today?',
        'FREE money! Call now!',
        'Meeting at 3pm tomorrow',
        'URGENT: Click this link immediately!',
        'Thanks for the great dinner last night'
    ],
    'label': ['ham', 'spam', 'ham', 'spam', 'ham']
})

# 성능 지표 시뮬레이션
performance_metrics = {
    'accuracy': 0.92,
    'precision': 0.88,
    'recall': 0.95,
    'f1_score': 0.91
}

# 교차 검증 점수 시뮬레이션
cv_scores = [0.89, 0.91, 0.90, 0.93, 0.88]

# 테스트 데이터 구성
test_data = {
    'dataset': sms_data,
    'business_rules': {
        'valid_labels': ['spam', 'ham']
    },
    'expected_schema': {
        'required_columns': ['message', 'label'],
        'dtypes': {'message': 'object', 'label': 'object'}
    },
    'performance_metrics': performance_metrics,
    'train_metrics': performance_metrics,
    'test_metrics': {k: v * 0.95 for k, v in performance_metrics.items()},  # 약간 낮은 테스트 성능
    'predictions': ['spam', 'ham', 'ham', 'spam', 'ham'],
    'true_labels': ['spam', 'ham', 'ham', 'spam', 'ham'],
    'cross_validation_scores': cv_scores
}

print(f"\n📊 테스트 데이터:")
print(f"데이터셋 크기: {len(sms_data)} 샘플")
print(f"컬럼: {list(sms_data.columns)}")
print(f"성능 지표: 정확도 {performance_metrics['accuracy']:.1%}")

# 품질 게이트 실행
print(f"\n🔍 품질 게이트 실행:")
result = quality_system.execute_all_checkpoints(test_data)

print(f"실행 ID: {result['execution_id']}")
print(f"전체 통과 여부: {'✅ 통과' if result['overall_passed'] else '❌ 실패'}")
print(f"진행 가능 여부: {'✅ 가능' if result['can_proceed'] else '🚫 불가'}")
print(f"평균 점수: {result['average_score']:.3f}")

# 체크포인트별 결과
print(f"\n📋 체크포인트별 결과:")
for checkpoint_result in result['checkpoint_results']:
    status_icon = "✅" if checkpoint_result.passed else "❌"
    print(f"{status_icon} {checkpoint_result.checkpoint_id}")
    print(f"   점수: {checkpoint_result.score:.3f}")
    print(f"   심각도: {checkpoint_result.severity.value}")
    print(f"   메시지: {checkpoint_result.message}")
    
    if checkpoint_result.recommendations:
        print(f"   권장사항:")
        for rec in checkpoint_result.recommendations[:2]:  # 최대 2개만 표시
            print(f"     • {rec}")

# 전체 권장사항
print(f"\n🎯 전체 권장사항:")
for rec in result['recommendations']:
    print(f"• {rec}")

# 통계 요약
stats = result['summary_stats']
print(f"\n📈 요약 통계:")
print(f"총 체크포인트: {stats['total_checkpoints']}")
print(f"통과한 체크포인트: {stats['passed_checkpoints']}")
print(f"차단 이슈: {stats['blocker_count']}")
print(f"중요 이슈: {stats['critical_count']}")
print(f"주요 이슈: {stats['major_count']}")
print(f"경미한 이슈: {stats['minor_count']}")
```

**코드 해설:**
- **체계적 품질 관리**: 데이터 품질부터 알고리즘 성능까지 포괄적 품질 검증
- **다차원 평가**: 완전성, 일관성, 정확성, 유효성 등 다양한 관점에서 품질 측정
- **자동화된 의사결정**: 점수와 심각도에 따른 자동 통과/실패 판정
- **실행 가능한 권고**: 구체적이고 실행 가능한 개선 방안 자동 생성

### 3.2 실시간 모니터링과 알람 시스템

품질 관리는 일회성이 아닌 지속적인 프로세스입니다. 실시간 모니터링을 통해 품질 이상 상황을 조기에 감지하고 대응해야 합니다.

```python
from datetime import datetime, timedelta
from typing import Optional, List
import threading
import queue
import time

class QualityAlert:
    """품질 알람"""
    
    def __init__(self, alert_id: str, severity: str, message: str, 
                 metric_name: str, current_value: float, threshold: float):
        self.alert_id = alert_id
        self.severity = severity
        self.message = message
        self.metric_name = metric_name
        self.current_value = current_value
        self.threshold = threshold
        self.timestamp = datetime.now()
        self.acknowledged = False
        self.resolved = False

class QualityMonitor:
    """실시간 품질 모니터링 시스템"""
    
    def __init__(self):
        self.monitoring_active = False
        self.alert_queue = queue.Queue()
        self.monitoring_thread = None
        self.quality_metrics = {}
        self.alert_thresholds = {
            'data_completeness': 0.95,
            'model_accuracy': 0.85,
            'prediction_confidence': 0.80,
            'processing_speed': 2.0  # 초당 처리 건수
        }
        self.alert_handlers = []
    
    def add_alert_handler(self, handler_func):
        """알람 핸들러 추가"""
        self.alert_handlers.append(handler_func)
    
    def start_monitoring(self):
        """모니터링 시작"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitor_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            print("🔍 품질 모니터링 시작")
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        print("⏹️ 품질 모니터링 중지")
    
    def update_metric(self, metric_name: str, value: float):
        """품질 지표 업데이트"""
        self.quality_metrics[metric_name] = {
            'value': value,
            'timestamp': datetime.now()
        }
    
    def _monitor_loop(self):
        """모니터링 루프"""
        while self.monitoring_active:
            self._check_thresholds()
            time.sleep(1)  # 1초마다 체크
    
    def _check_thresholds(self):
        """임계값 확인"""
        current_time = datetime.now()
        
        for metric_name, threshold in self.alert_thresholds.items():
            if metric_name in self.quality_metrics:
                metric_data = self.quality_metrics[metric_name]
                current_value = metric_data['value']
                metric_time = metric_data['timestamp']
                
                # 최근 데이터만 확인 (5분 이내)
                if current_time - metric_time < timedelta(minutes=5):
                    if self._is_threshold_violated(metric_name, current_value, threshold):
                        alert = self._create_alert(metric_name, current_value, threshold)
                        self.alert_queue.put(alert)
                        self._notify_handlers(alert)
    
    def _is_threshold_violated(self, metric_name: str, current_value: float, threshold: float) -> bool:
        """임계값 위반 확인"""
        
        if metric_name in ['data_completeness', 'model_accuracy', 'prediction_confidence']:
            # 높을수록 좋은 지표
            return current_value < threshold
        elif metric_name == 'processing_speed':
            # 낮을수록 좋은 지표 (처리 시간)
            return current_value > threshold
        
        return False
    
    def _create_alert(self, metric_name: str, current_value: float, threshold: float) -> QualityAlert:
        """알람 생성"""
        
        alert_id = f"{metric_name}_{int(time.time())}"
        
        # 심각도 결정
        if metric_name in ['model_accuracy', 'data_completeness']:
            if current_value < threshold * 0.8:
                severity = 'critical'
            elif current_value < threshold * 0.9:
                severity = 'major'
            else:
                severity = 'minor'
        else:
            severity = 'minor'
        
        # 메시지 생성
        if metric_name == 'data_completeness':
            message = f"데이터 완전성이 임계값 이하입니다 ({current_value:.1%} < {threshold:.1%})"
        elif metric_name == 'model_accuracy':
            message = f"모델 정확도가 임계값 이하입니다 ({current_value:.1%} < {threshold:.1%})"
        elif metric_name == 'prediction_confidence':
            message = f"예측 신뢰도가 임계값 이하입니다 ({current_value:.1%} < {threshold:.1%})"
        elif metric_name == 'processing_speed':
            message = f"처리 속도가 임계값을 초과했습니다 ({current_value:.1f}초 > {threshold:.1f}초)"
        else:
            message = f"{metric_name} 지표가 임계값을 벗어났습니다"
        
        return QualityAlert(alert_id, severity, message, metric_name, current_value, threshold)
    
    def _notify_handlers(self, alert: QualityAlert):
        """알람 핸들러에 통지"""
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                print(f"알람 핸들러 오류: {e}")
    
    def get_active_alerts(self) -> List[QualityAlert]:
        """활성 알람 목록 반환"""
        alerts = []
        while not self.alert_queue.empty():
            try:
                alert = self.alert_queue.get_nowait()
                if not alert.resolved:
                    alerts.append(alert)
            except queue.Empty:
                break
        return alerts

# 알람 핸들러 함수들
def console_alert_handler(alert: QualityAlert):
    """콘솔 알람 핸들러"""
    severity_icons = {
        'critical': '🚨',
        'major': '⚠️',
        'minor': '💡'
    }
    
    icon = severity_icons.get(alert.severity, '📢')
    print(f"{icon} [{alert.severity.upper()}] {alert.message}")
    print(f"   시간: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   지표: {alert.metric_name}")
    print(f"   현재값: {alert.current_value:.3f}")
    print(f"   임계값: {alert.threshold:.3f}")

def email_alert_handler(alert: QualityAlert):
    """이메일 알람 핸들러 (시뮬레이션)"""
    if alert.severity in ['critical', 'major']:
        print(f"📧 이메일 발송: {alert.message}")

def slack_alert_handler(alert: QualityAlert):
    """Slack 알람 핸들러 (시뮬레이션)"""
    print(f"💬 Slack 메시지: {alert.message}")

# 실시간 모니터링 시연
print("\n📊 실시간 품질 모니터링 시연")
print("=" * 50)

# 품질 모니터 설정
monitor = QualityMonitor()
monitor.add_alert_handler(console_alert_handler)
monitor.add_alert_handler(email_alert_handler)
monitor.add_alert_handler(slack_alert_handler)

# 모니터링 시작
monitor.start_monitoring()

# 정상적인 지표들 업데이트
print("✅ 정상 상태 지표 업데이트:")
monitor.update_metric('data_completeness', 0.98)
monitor.update_metric('model_accuracy', 0.92)
monitor.update_metric('prediction_confidence', 0.87)
monitor.update_metric('processing_speed', 1.2)

time.sleep(2)  # 모니터링이 실행될 시간 제공

# 문제 상황 시뮬레이션
print("\n🚨 문제 상황 시뮬레이션:")
print("데이터 완전성 저하...")
monitor.update_metric('data_completeness', 0.85)  # 임계값 이하

time.sleep(1)

print("모델 정확도 급락...")
monitor.update_metric('model_accuracy', 0.75)  # 임계값 이하

time.sleep(1)

print("처리 속도 저하...")
monitor.update_metric('processing_speed', 3.5)  # 임계값 초과

time.sleep(3)  # 알람이 발생할 시간 제공

# 활성 알람 확인
active_alerts = monitor.get_active_alerts()
print(f"\n📋 활성 알람 수: {len(active_alerts)}")

# 모니터링 중지
monitor.stop_monitoring()
```

**코드 해설:**
- **실시간 모니터링**: 별도 스레드에서 지속적으로 품질 지표 감시
- **다단계 알람**: 심각도에 따른 차별적 알람 처리
- **다중 채널 통지**: 콘솔, 이메일, Slack 등 다양한 채널로 알람 전송
- **임계값 기반 자동 감지**: 설정된 임계값 기준으로 자동 이상 상황 탐지

> 🖼️ **이미지 생성 프롬프트**: 
> "품질 관리 대시보드를 보여주는 모니터링 화면. 여러 개의 게이지 차트로 데이터 완전성, 모델 정확도, 처리 속도 등의 지표가 표시되고, 우측에는 실시간 알람 패널이 있는 현대적인 인터페이스"

## 4. 미니 프로젝트: SMS 스팸 탐지 하이브리드 워크플로우 구축

이제 지금까지 학습한 내용을 종합하여 SMS 스팸 탐지 프로젝트에 완전한 하이브리드 워크플로우를 구축해보겠습니다.

### 4.1 프로젝트 설계

```python
from typing import Dict, List, Any, Optional
import time
from dataclasses import dataclass

@dataclass
class WorkflowConfig:
    """워크플로우 설정"""
    automation_strategy: str = 'hybrid'  # 'full_auto', 'human_led', 'hybrid'
    quality_threshold: float = 0.85
    confidence_threshold: float = 0.80
    max_processing_time: int = 300  # 초
    enable_monitoring: bool = True
    
class SMSSpamDetectionWorkflow:
    """SMS 스팸 탐지 하이브리드 워크플로우"""
    
    def __init__(self, config: WorkflowConfig):
        self.config = config
        self.workflow_components = {}
        self.quality_monitor = QualityMonitor() if config.enable_monitoring else None
        self.processing_stats = {
            'total_processed': 0,
            'auto_processed': 0,
            'human_reviewed': 0,
            'quality_issues': 0
        }
    
    def setup_workflow(self):
        """워크플로우 구성 요소 설정"""
        
        # 1. STAR 평가기
        self.workflow_components['star_analyzer'] = TaskStandardizationAnalyzer()
        
        # 2. AI 에이전트들
        self.workflow_components['ai_agents'] = [
            AIAgent('text_processor', ['text_preprocessing']),
            AIAgent('spam_classifier', ['spam_classification']),
            AIAgent('feature_extractor', ['feature_extraction'])
        ]
        
        # 3. 인간 검토자들
        self.workflow_components['human_reviewers'] = [
            HumanAgent('junior_analyst', ['quality_review']),
            HumanAgent('senior_analyst', ['business_interpretation'])
        ]
        
        # 4. 품질 게이트
        self.workflow_components['quality_gates'] = [
            DataQualityCheckpoint(),
            AlgorithmPerformanceCheckpoint()
        ]
        
        # 5. 협업 엔진
        self.workflow_components['collaboration_engine'] = ParallelCollaborationEngine()
        for agent in self.workflow_components['ai_agents']:
            self.workflow_components['collaboration_engine'].register_ai_agent(agent)
        for reviewer in self.workflow_components['human_reviewers']:
            self.workflow_components['collaboration_engine'].register_human_reviewer(reviewer)
        
        # 6. 모니터링 설정
        if self.quality_monitor:
            self.quality_monitor.add_alert_handler(console_alert_handler)
            self.quality_monitor.start_monitoring()
        
        print("🔧 하이브리드 워크플로우 설정 완료")
    
    async def process_sms_batch(self, sms_data: List[Dict[str, str]]) -> Dict[str, Any]:
        """SMS 배치 처리"""
        
        start_time = time.time()
        results = []
        
        print(f"📨 {len(sms_data)}개 SMS 메시지 처리 시작")
        
        for i, sms in enumerate(sms_data):
            print(f"\n--- 메시지 {i+1}/{len(sms_data)} ---")
            print(f"내용: {sms['text'][:50]}{'...' if len(sms['text']) > 50 else ''}")
            
            # 1단계: 자동화 적합성 평가
            task_profile = {
                'input_types': ['text'],
                'steps_variability': 'low',
                'output_variance': 'medium',
                'rule_complexity': 'moderate'
            }
            
            star_result = self.workflow_components['star_analyzer'].evaluate_standardization(task_profile)
            automation_recommendation = star_result['automation_recommendation']
            
            print(f"자동화 권고: {automation_recommendation}")
            
            # 2단계: 워크플로우 분기
            if automation_recommendation in ['strongly_recommended', 'recommended']:
                # AI 우선 처리
                message_result = await self._process_with_ai_first(sms)
            elif automation_recommendation == 'partial_automation':
                # 병렬 협업 처리
                message_result = await self._process_with_collaboration(sms)
            else:
                # 인간 우선 처리
                message_result = await self._process_with_human_first(sms)
            
            # 3단계: 품질 검증
            quality_result = self._validate_quality(message_result)
            
            # 4단계: 모니터링 업데이트
            if self.quality_monitor:
                self._update_monitoring_metrics(message_result, quality_result)
            
            # 결과 저장
            results.append({
                'message_id': i + 1,
                'input': sms,
                'automation_strategy': automation_recommendation,
                'processing_result': message_result,
                'quality_result': quality_result
            })
            
            self.processing_stats['total_processed'] += 1
        
        processing_time = time.time() - start_time
        
        # 배치 처리 결과 요약
        summary = self._generate_batch_summary(results, processing_time)
        
        return {
            'results': results,
            'summary': summary,
            'processing_stats': self.processing_stats
        }
    
    async def _process_with_ai_first(self, sms: Dict[str, str]) -> Dict[str, Any]:
        """AI 우선 처리"""
        
        print("🤖 AI 우선 처리 모드")
        
        # AI 처리
        ai_agent = self.workflow_components['ai_agents'][1]  # spam_classifier
        task = WorkflowTask(
            task_id=f"ai_first_{int(time.time())}",
            task_type='spam_classification',
            assigned_to='ai',
            input_data=sms
        )
        
        ai_result = ai_agent.process_task(task)
        
        # 신뢰도 확인
        confidence = ai_result.output_data.get('confidence', 0) if ai_result.output_data else 0
        
        if confidence >= self.config.confidence_threshold:
            # 자동 승인
            self.processing_stats['auto_processed'] += 1
            return {
                'final_decision': ai_result.output_data.get('prediction'),
                'confidence': confidence,
                'processing_path': 'ai_auto_approved',
                'details': ai_result.output_data
            }
        else:
            # 인간 검토 필요
            human_reviewer = self.workflow_components['human_reviewers'][0]
            review_result = human_reviewer.process_task(ai_result)
            
            self.processing_stats['human_reviewed'] += 1
            return {
                'final_decision': review_result.output_data.get('reviewed_prediction', 
                                                               ai_result.output_data.get('prediction')),
                'confidence': review_result.output_data.get('final_confidence', confidence),
                'processing_path': 'ai_with_human_review',
                'details': {
                    'ai_result': ai_result.output_data,
                    'human_review': review_result.output_data
                }
            }
    
    async def _process_with_collaboration(self, sms: Dict[str, str]) -> Dict[str, Any]:
        """병렬 협업 처리"""
        
        print("🤝 병렬 협업 처리 모드")
        
        collaboration_result = await self.workflow_components['collaboration_engine'].execute_parallel_workflow(
            task_data=sms,
            task_type='spam_classification',
            combination_strategy='confidence_based'
        )
        
        combined = collaboration_result.combined_results
        
        return {
            'final_decision': combined.get('final_prediction'),
            'confidence': combined.get('confidence', 0),
            'processing_path': 'parallel_collaboration',
            'details': {
                'ai_results': len(collaboration_result.ai_results),
                'human_results': len(collaboration_result.human_results),
                'consensus_score': collaboration_result.consensus_score,
                'combination_details': combined
            }
        }
    
    async def _process_with_human_first(self, sms: Dict[str, str]) -> Dict[str, Any]:
        """인간 우선 처리"""
        
        print("👨‍💼 인간 우선 처리 모드")
        
        # 인간 분석가가 주도적으로 처리
        human_analyst = self.workflow_components['human_reviewers'][-1]  # senior_analyst
        
        task = WorkflowTask(
            task_id=f"human_first_{int(time.time())}",
            task_type='spam_classification',
            assigned_to='human',
            input_data=sms
        )
        
        human_result = human_analyst.process_task(task)
        self.processing_stats['human_reviewed'] += 1
        
        return {
            'final_decision': human_result.output_data.get('prediction', 'unknown'),
            'confidence': 0.9,  # 인간 판단에 높은 신뢰도 부여
            'processing_path': 'human_led',
            'details': human_result.output_data
        }
    
    def _validate_quality(self, processing_result: Dict[str, Any]) -> Dict[str, Any]:
        """품질 검증"""
        
        # 간단한 품질 검증
        quality_score = 1.0
        issues = []
        
        # 신뢰도 검사
        confidence = processing_result.get('confidence', 0)
        if confidence < 0.7:
            quality_score -= 0.3
            issues.append("낮은 신뢰도")
        
        # 처리 경로 검사
        processing_path = processing_result.get('processing_path', '')
        if 'auto' in processing_path and confidence < 0.9:
            quality_score -= 0.2
            issues.append("자동 처리에 비해 낮은 신뢰도")
        
        if quality_score < self.config.quality_threshold:
            self.processing_stats['quality_issues'] += 1
        
        return {
            'quality_score': quality_score,
            'passed': quality_score >= self.config.quality_threshold,
            'issues': issues
        }
    
    def _update_monitoring_metrics(self, processing_result: Dict[str, Any], quality_result: Dict[str, Any]):
        """모니터링 지표 업데이트"""
        
        if self.quality_monitor:
            confidence = processing_result.get('confidence', 0)
            quality_score = quality_result.get('quality_score', 0)
            
            self.quality_monitor.update_metric('prediction_confidence', confidence)
            self.quality_monitor.update_metric('overall_quality', quality_score)
    
    def _generate_batch_summary(self, results: List[Dict[str, Any]], processing_time: float) -> Dict[str, Any]:
        """배치 처리 요약 생성"""
        
        total_messages = len(results)
        
        # 처리 경로별 통계
        path_counts = {}
        for result in results:
            path = result['processing_result']['processing_path']
            path_counts[path] = path_counts.get(path, 0) + 1
        
        # 품질 통계
        quality_passed = sum(1 for r in results if r['quality_result']['passed'])
        avg_confidence = sum(r['processing_result'].get('confidence', 0) for r in results) / total_messages
        
        # 예측 결과 통계
        predictions = [r['processing_result']['final_decision'] for r in results]
        from collections import Counter
        prediction_counts = Counter(predictions)
        
        return {
            'total_messages': total_messages,
            'processing_time': processing_time,
            'avg_time_per_message': processing_time / total_messages,
            'processing_paths': path_counts,
            'quality_stats': {
                'passed_count': quality_passed,
                'pass_rate': quality_passed / total_messages,
                'avg_confidence': avg_confidence
            },
            'prediction_stats': dict(prediction_counts),
            'efficiency_metrics': {
                'auto_processing_rate': self.processing_stats['auto_processed'] / total_messages,
                'human_review_rate': self.processing_stats['human_reviewed'] / total_messages,
                'quality_issue_rate': self.processing_stats['quality_issues'] / total_messages
            }
        }
    
    def cleanup(self):
        """워크플로우 정리"""
        if self.quality_monitor:
            self.quality_monitor.stop_monitoring()
        print("🧹 워크플로우 정리 완료")

# SMS 스팸 탐지 하이브리드 워크플로우 시연
print("\n🚀 SMS 스팸 탐지 하이브리드 워크플로우 시연")
print("=" * 60)

# 워크플로우 설정
config = WorkflowConfig(
    automation_strategy='hybrid',
    quality_threshold=0.8,
    confidence_threshold=0.85,
    enable_monitoring=True
)

# 워크플로우 생성 및 설정
workflow = SMSSpamDetectionWorkflow(config)
workflow.setup_workflow()

# 테스트 SMS 데이터
test_sms_data = [
    {'text': 'Hello, how are you doing today?'},  # 명확한 정상 메시지
    {'text': 'FREE money! Call 123-456-7890 NOW!'},  # 명확한 스팸
    {'text': 'Limited time offer for students only'},  # 애매한 경우
    {'text': 'URGENT: Your account needs verification'},  # 애매한 경우
    {'text': 'Thanks for the dinner yesterday!'}  # 명확한 정상 메시지
]

# 비동기 워크플로우 실행
import asyncio

async def run_workflow_demo():
    result = await workflow.process_sms_batch(test_sms_data)
    
    print(f"\n📊 배치 처리 결과 요약")
    print("=" * 40)
    
    summary = result['summary']
    print(f"총 메시지 수: {summary['total_messages']}")
    print(f"처리 시간: {summary['processing_time']:.2f}초")
    print(f"메시지당 평균 시간: {summary['avg_time_per_message']:.2f}초")
    
    print(f"\n📈 처리 경로별 통계:")
    for path, count in summary['processing_paths'].items():
        print(f"  {path}: {count}개 ({count/summary['total_messages']:.1%})")
    
    print(f"\n✅ 품질 통계:")
    quality_stats = summary['quality_stats']
    print(f"  품질 통과: {quality_stats['passed_count']}/{summary['total_messages']} ({quality_stats['pass_rate']:.1%})")
    print(f"  평균 신뢰도: {quality_stats['avg_confidence']:.3f}")
    
    print(f"\n🎯 예측 결과:")
    for prediction, count in summary['prediction_stats'].items():
        print(f"  {prediction}: {count}개")
    
    print(f"\n⚡ 효율성 지표:")
    efficiency = summary['efficiency_metrics']
    print(f"  자동 처리율: {efficiency['auto_processing_rate']:.1%}")
    print(f"  인간 검토율: {efficiency['human_review_rate']:.1%}")
    print(f"  품질 이슈율: {efficiency['quality_issue_rate']:.1%}")
    
    # 개별 결과 상세 보기
    print(f"\n📋 개별 메시지 결과:")
    for result_item in result['results'][:3]:  # 처음 3개만 표시
        msg_id = result_item['message_id']
        processing = result_item['processing_result']
        quality = result_item['quality_result']
        
        print(f"\n  메시지 {msg_id}:")
        print(f"    예측: {processing['final_decision']}")
        print(f"    신뢰도: {processing['confidence']:.3f}")
        print(f"    처리 경로: {processing['processing_path']}")
        print(f"    품질 점수: {quality['quality_score']:.3f}")
        if quality['issues']:
            print(f"    품질 이슈: {', '.join(quality['issues'])}")

# 워크플로우 실행
asyncio.run(run_workflow_demo())

# 정리
workflow.cleanup()
```

**코드 해설:**
- **통합적 워크플로우**: STAR 평가부터 품질 관리까지 전체 과정 통합
- **적응적 처리**: 메시지 특성에 따른 최적 처리 경로 자동 선택
- **실시간 모니터링**: 처리 과정 중 품질 지표 실시간 추적
- **종합적 평가**: 효율성, 품질, 사용자 만족도 등 다차원 성과 측정

> 🖼️ **이미지 생성 프롬프트**: 
> "SMS 스팸 탐지 하이브리드 워크플로우의 전체 아키텍처를 보여주는 플로우차트. 입력 SMS에서 시작하여 STAR 평가, AI/인간 협업, 품질 게이트, 모니터링을 거쳐 최종 결과에 이르는 완전한 프로세스가 화살표와 분기점으로 표현된 시스템 다이어그램"

## 직접 해보기 / 연습 문제

### **연습 문제 1: 자동화 전략 설계 (초급)**

다음 데이터 분석 작업들에 대해 STAR 프레임워크를 적용하여 최적의 자동화 전략을 수립하세요.

**작업 시나리오:**
1. **고객 만족도 설문 분석**: 매월 1,000개의 설문 응답 분석
2. **금융 이상 거래 탐지**: 실시간으로 초당 100건의 거래 모니터링
3. **마케팅 캠페인 효과 분석**: 분기별 다양한 채널의 캠페인 성과 종합 분석

**요구사항:**
- 각 작업에 대해 STAR(Standardization, Time sensitivity, Accuracy requirements, Resource requirements) 점수 산정
- 권장 자동화 수준과 근거 제시
- 인간-AI 협업 방식 제안

```python
# 여러분의 답안을 작성하세요
def analyze_automation_strategy():
    # STAR 프레임워크를 활용한 분석 코드
    pass
```

### **연습 문제 2: 협업 패턴 구현 (중급)**

다음 요구사항에 맞는 인간-AI 협업 패턴을 구현하세요.

**시나리오:** 의료 데이터 분석 시스템
- AI가 의료 이미지에서 이상 징후를 1차 탐지
- 신뢰도가 높은 경우(>90%) 자동 분류
- 신뢰도가 중간인 경우(70-90%) 의사가 검토
- 신뢰도가 낮은 경우(<70%) 전문의에게 에스컬레이션

**구현 요구사항:**
- 계층적 협업 패턴 사용
- 각 단계별 품질 게이트 설정
- 처리 통계 및 성과 지표 추적

```python
# 여러분의 솔루션을 구현하세요
class MedicalImageAnalysisWorkflow:
    def __init__(self):
        # 초기화 코드
        pass
    
    def process_medical_image(self, image_data):
        # 의료 이미지 처리 워크플로우 구현
        pass
```

### **연습 문제 3: 품질 체크포인트 설계 (고급)**

전자상거래 추천 시스템을 위한 종합적인 품질 관리 시스템을 설계하세요.

**요구사항:**
1. **데이터 품질 체크포인트**: 
   - 사용자 행동 데이터의 완전성 및 일관성 검증
   - 상품 정보의 정확성 확인

2. **알고리즘 성능 체크포인트**:
   - 추천 정확도 및 다양성 측정
   - 실시간 성능 모니터링

3. **비즈니스 로직 체크포인트**:
   - 추천 결과의 비즈니스 규칙 준수 확인
   - 사용자 개인정보 보호 정책 준수

**구현 목표:**
- 실시간 모니터링 대시보드
- 자동 알람 및 에스컬레이션 시스템
- 품질 트렌드 분석 및 예측

```python
# 여러분의 품질 관리 시스템을 설계하세요
class EcommerceQualityManagementSystem:
    def __init__(self):
        # 시스템 초기화
        pass
    
    def setup_quality_gates(self):
        # 품질 게이트 설정
        pass
    
    def monitor_recommendation_quality(self):
        # 추천 품질 모니터링
        pass
```

## 요약 / 핵심 정리

이번 Part에서는 **자동화와 수동 작업의 균형**을 찾는 체계적인 방법론을 학습했습니다. 핵심 내용을 정리하면:

### 🎯 **핵심 개념**

1. **STAR 프레임워크**: 자동화 적합성을 4가지 관점(표준화, 시간민감성, 정확도 요구사항, 자원 요구사항)에서 체계적으로 평가

2. **인간-AI 협업 모델**: 
   - **순차적 협업**: 단계별 정확성이 중요한 경우
   - **병렬적 협업**: 빠른 처리와 다양한 관점이 필요한 경우  
   - **계층적 협업**: 품질 보장이 최우선인 경우

3. **품질 관리 체크포인트**: Critical Control Points(CCP)를 식별하고 다차원 품질 검증 시스템 구축

### 💡 **실무 적용 포인트**

- **자동화 vs 인간 개입의 기준을 명확히 설정**하여 일관된 의사결정 가능
- **신뢰도 기반 에스컬레이션**으로 품질과 효율성의 균형 달성
- **실시간 모니터링**을 통한 품질 이상 상황 조기 감지 및 대응
- **처리 통계 추적**으로 워크플로우 최적화를 위한 데이터 기반 의사결정

### 🚀 **주요 성과**

- AI의 **계산 능력**과 인간의 **창의성 및 판단력**을 조화롭게 결합하는 방법 습득
- 프로젝트 특성에 맞는 **최적 자동화 전략** 수립 능력 배양  
- **품질을 보장하면서도 효율성을 극대화**하는 하이브리드 워크플로우 설계 역량
- 실무에서 바로 적용 가능한 **템플릿과 가이드라인** 확보

### 🎯 **다음 단계 예고**

다음 Part에서는 **대규모 언어 모델(LLM)을 활용한 데이터 분석**을 다룹니다. ChatGPT, Claude와 같은 최신 LLM을 데이터 분석 워크플로우에 효과적으로 통합하는 방법을 학습하게 됩니다.

---

> 💬 **학습 성찰**
> 
> "자동화와 인간 작업의 균형을 찾는 것은 단순히 효율성만의 문제가 아니라, **품질과 신뢰성을 보장하면서도 혁신과 창의성을 잃지 않는** 지혜로운 선택의 문제입니다. 
> 
> 중요한 것은 기술이 인간을 대체하는 것이 아니라, **인간의 능력을 증강하고 더 가치 있는 일에 집중할 수 있도록 돕는** 도구로 활용하는 것입니다."

> 🖼️ **이미지 생성 프롬프트**: 
> "인간과 AI가 협력하는 모습을 상징하는 추상적 일러스트레이션. 왼쪽에는 인간의 창의적 사고를 나타내는 다채로운 아이디어 구름이, 오른쪽에는 AI의 체계적 처리를 나타내는 기하학적 패턴이 있고, 중앙에서 두 요소가 조화롭게 결합되어 새로운 가치를 창출하는 모습"s else 0.8
            
            return {
                'score': overall_balance,
                'details': {
                    'true_distribution': dict(true_dist),
                    'predicted_distribution': dict(pred_dist),
                    'balance_score': overall_balance
                }
            }
            
        except Exception as e:
            return {'score': 0.5, 'details': {'error': str(e)}}
    
    def _check_stability(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """안정성 검사 - 예측 일관성"""
        
        # 여러 실행 결과 또는 교차 검증 결과 확인
        cv_scores = data.get('cross_validation_scores', [])
        
        if not cv_scores:
            return {'score': 0.7, 'details': {'note': 'Stability check skipped due to insufficient data'}}
        
        try:
            import numpy as np
            
            mean_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)
            
            # 표준편차가 작을수록 안정성 높음
            stability_score = max(0, 1 - std_score * 5)
            
            return {
                'score': stability_score,
                'details': {
                    'cv_scores': cv_scores,
                    'mean_score': mean_score,
                    'std_score': std_score,
                    'stability_score': stability_score
                }
            }
            
        except Exception as e:
            return {'score': 0.5, 'details': {'error': str(e)}}

class QualityGateSystem:
    """품질 게이트 시스템"""
    
    def __init__(self):
        self.checkpoints: List[QualityCheckpoint] = []
        self.execution_history: List[Dict[str, Any]] = []
        self.global_thresholds = {
            CheckpointSeverity.BLOCKER: 0.0,     # 차단 레벨은 진행 불가
            CheckpointSeverity.CRITICAL: 0.6,    # 중요 이슈 임계값
            CheckpointSeverity.MAJOR: 0.8,       # 주요 이슈 임계값
            CheckpointSeverity.MINOR: 0.9        # 경미한 이슈 임계값
        }
    
    def add_checkpoint(self, checkpoint: QualityCheckpoint):
        """체크포인트 추가"""
        self.checkpoints.append(checkpoint)
    
    def execute_all_checkpoints(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """모든 체크포인트 실행"""
        
        execution_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        checkpoint_results = []
        overall_passed = True
        blocker_count = 0
        critical_count = 0
        
        for checkpoint in self.checkpoints:
            result = checkpoint.evaluate(data)
            checkpoint_results.append(result)
            
            # 심각도별 카운트
            if result.severity == CheckpointSeverity.BLOCKER:
                blocker_count += 1
                overall_passed = False
            elif result.severity == CheckpointSeverity.CRITICAL:
                critical_count += 1
                if not result.passed:
                    overall_passed = False
        
        # 전체 점수 계산 (가중 평균)
        total_score = sum(result.score for result in checkpoint_results)
        average_score = total_score / len(checkpoint_results) if checkpoint_results else 0.0
        
        # 진행 가능 여부 결정
        can_proceed = overall_passed and blocker_count == 0
        
        execution_summary = {
            'execution_id': execution_id,
            'timestamp': time.time(),
            'overall_passed': overall_passed,
            'can_proceed': can_proceed,
            'average_score': average_score,
            'checkpoint_results': checkpoint_results,
            'summary_stats': {
                'total_checkpoints': len(checkpoint_results),
                'passed_checkpoints': sum(1 for r in checkpoint_results if r.passed),
                'blocker_count': blocker_count,
                'critical_count': critical_count,
                'major_count': sum(1 for r in checkpoint_results if r.severity == CheckpointSeverity.MAJOR),
                'minor_count': sum(1 for r in checkpoint_results if r.severity == CheckpointSeverity.MINOR)
            },
            'recommendations': self._generate_global_recommendations(checkpoint_results)
        }
        
        # 실행 기록 저장
        self.execution_history.append(execution_summary)
        
        return execution_summary
    
    def _generate_global_recommendations(self, results: List[CheckpointResult]) -> List[str]:
        """전체 권장사항 생성"""
        
        recommendations = []
        
        # 차단 이슈가 있는 경우
        blockers = [r for r in results if r.severity == CheckpointSeverity.BLOCKER]
        if blockers:
            recommendations.append(f"🚨 {len(blockers)}개의 차단 이슈를 즉시 해결해야 합니다")
        
        # 중요 이슈가 많은 경우
        criticals = [r for r in results if r.severity == CheckpointSeverity.CRITICAL]
        if len(criticals) > 2:
            recommendations.append(f"⚠️ {len(criticals)}개의 중요 이슈가 있습니다. 우선순위를 정해 해결하세요")
        
        # 전체적인 품질 수준 평가
        avg_score = sum(r.score for r in results) / len(results) if results else 0
        if avg_score >= 0.9:
            recommendations.append("✅ 전체적으로 우수한 품질 수준입니다")
        elif avg_score >= 0.7:
            recommendations.append("👍 양호한 품질 수준이지만 일부 개선 여지가 있습니다")
        else:
            recommendations.append("📋 전체적인 품질 개선이 필요합니다")
        
        return recommendations
    
    def get_quality_trends(self) -> Dict[str, Any]:
        """품질 트렌드 분석"""
        
        if len(self.execution_history) < 2:
            return {'message': 'Insufficient data for trend analysis'}
        
        # 최근 실행들의 점수 추이
        recent_scores = [exec['average_score'] for exec in self.execution_history[-10:]]
        
        # 트렌드 계산
        if len(recent_scores) >= 2:
            trend = recent_scores[-1] - recent_scores[-2]
            if trend > 0.05:
                trend_status = 'improving'
            elif trend < -0.05:
                trend_status = 'declining'
            else:
                trend_status = 'stable'
        else:
            trend_status = 'insufficient_data'
        
        return {
            'recent_scores': recent_scores,
            'trend_status': trend_status,
            'average_score': sum(recent_scores) / len(recent_scores),
            'score_variance': np.var(recent_scores) if len(recent_scores) > 1 else 0
        }

# SMS 스팸 탐지 프로젝트 품질 게이트 시연
print("\n🎯 품질 관리 체크포인트 시연")
print("=" * 50)

# 품질 게이트 시스템 설정
quality_system = QualityGateSystem()

# 데이터 품질 체크포인트 추가
data_quality_checkpoint = DataQualityCheckpoint()
algorithm_performance_checkpoint = AlgorithmPerformanceCheckpoint()

quality_system.add_checkpoint(data_quality_checkpoint)
quality_system.add_checkpoint(algorithm_performance_checkpoint)

# 시뮬레이션된 테스트 데이터
import pandas as pd
import numpy as np

# 샘플 데이터셋 생성
sample_data = pd.DataFrame({
    'message': [
        'Hello how are you',
        'FREE money call now',
        'Meeting at 3pm today',
        None,  # 결측치
        'Win $1000 prize!'
    ],
    'label': ['ham', 'spam', 'ham', 'ham', 'spam']
})

# 성능 지표 시뮬레이션
performance_metrics = {
    'accuracy': 0.87,
    'precision': 0.83,
    'recall': 0.89,
    'f1_score': 0.86
}

train_metrics = {
    'accuracy': 0.92,
    'precision': 0.88,
    'recall': 0.91,
    'f1_score': 0.89
}

test_metrics = {
    'accuracy': 0.87,
    'precision': 0.83,
    'recall': 0.89,
    'f1_score': 0.86
}

# 테스트 데이터 구성
test_data = {
    'dataset': sample_data,
    'expected_schema': {
        'required_columns': ['message', 'label'],
        'dtypes': {'message': 'object', 'label': 'object'}
    },
    'business_rules': {
        'valid_labels': ['spam', 'ham']
    },
    'performance_metrics': performance_metrics,
    'train_metrics': train_metrics,
    'test_metrics': test_metrics,
    'cross_validation_scores': [0.85, 0.87, 0.86, 0.88, 0.84],
    'predictions': ['ham', 'spam', 'ham', 'ham', 'spam'],
    'true_labels': ['ham', 'spam', 'ham', 'ham', 'spam']
}

# 품질 게이트 실행
print("📋 품질 체크포인트 실행 중...")
execution_result = quality_system.execute_all_checkpoints(test_data)

# 결과 출력
print(f"\n🎯 품질 게이트 실행 결과:")
print(f"실행 ID: {execution_result['execution_id']}")
print(f"전체 통과: {'✅ 예' if execution_result['overall_passed'] else '❌ 아니오'}")
print(f"진행 가능: {'✅ 예' if execution_result['can_proceed'] else '❌ 아니오'}")
print(f"평균 점수: {execution_result['average_score']:.3f}")

# 체크포인트별 상세 결과
print(f"\n📊 체크포인트별 결과:")
for result in execution_result['checkpoint_results']:
    status_emoji = "✅" if result.passed else "❌"
    severity_emoji = {
        CheckpointSeverity.BLOCKER: "🚨",
        CheckpointSeverity.CRITICAL: "⚠️", 
        CheckpointSeverity.MAJOR: "🟡",
        CheckpointSeverity.MINOR: "🟢"
    }
    
    print(f"{status_emoji} {result.checkpoint_id}")
    print(f"   점수: {result.score:.3f} | 심각도: {severity_emoji[result.severity]} {result.severity.value}")
    print(f"   메시지: {result.message}")
    
    if result.recommendations:
        print(f"   권장사항:")
        for rec in result.recommendations[:2]:  # 상위 2개만 표시
            print(f"     • {rec}")

# 전체 권장사항
print(f"\n💡 전체 권장사항:")
for rec in execution_result['recommendations']:
    print(f"  {rec}")

# 통계 요약
stats = execution_result['summary_stats']
print(f"\n📈 통계 요약:")
print(f"  총 체크포인트: {stats['total_checkpoints']}")
print(f"  통과한 체크포인트: {stats['passed_checkpoints']}")
print(f"  차단 이슈: {stats['blocker_count']}")
print(f"  중요 이슈: {stats['critical_count']}")
print(f"  주요 이슈: {stats['major_count']}")
print(f"  경미한 이슈: {stats['minor_count']}")
```

**코드 해설:**
- **계층적 품질 관리**: 여러 체크포인트를 통한 다층 품질 검증 체계
- **가중치 기반 평가**: 각 검증 규칙의 중요도에 따른 차별적 평가
- **심각도 분류**: 문제의 심각성에 따른 대응 우선순위 설정
- **실행 가능한 권장사항**: 구체적이고 실천 가능한 개선 방안 제시

#### **원칙 2: 실시간 모니터링과 알람**
품질 문제를 조기에 발견하고 대응할 수 있는 모니터링 시스템을 구축합니다.

```python
import threading
import queue
import time
from typing import Callable, Dict, Any
from collections import deque
import json

class QualityAlert:
    """품질 알람"""
    
    def __init__(self, 
                 alert_id: str,
                 severity: CheckpointSeverity,
                 message: str,
                 checkpoint_id: str,
                 data: Dict[str, Any] = None):
        self.alert_id = alert_id
        self.severity = severity
        self.message = message
        self.checkpoint_id = checkpoint_id
        self.data = data or {}
        self.timestamp = time.time()
        self.acknowledged = False

class RealTimeQualityMonitor:
    """실시간 품질 모니터"""
    
    def __init__(self, max_history_size: int = 1000):
        self.checkpoints: Dict[str, QualityCheckpoint] = {}
        self.alert_handlers: Dict[CheckpointSeverity, List[Callable]] = {
            CheckpointSeverity.BLOCKER: [],
            CheckpointSeverity.CRITICAL: [],
            CheckpointSeverity.MAJOR: [],
            CheckpointSeverity.MINOR: []
        }
        
        # 모니터링 상태
        self.is_monitoring = False
        self.monitoring_thread = None
        self.data_queue = queue.Queue()
        
        # 품질 기록
        self.quality_history = deque(maxlen=max_history_size)
        self.active_alerts: List[QualityAlert] = []
        
        # 모니터링 설정
        self.monitoring_interval = 1.0  # 초
        self.alert_cooldown = {}  # 알람 쿨다운 관리
    
    def register_checkpoint(self, checkpoint: QualityCheckpoint):
        """체크포인트 등록"""
        self.checkpoints[checkpoint.checkpoint_id] = checkpoint
    
    def add_alert_handler(self, severity: CheckpointSeverity, handler: Callable):
        """알람 핸들러 추가"""
        self.alert_handlers[severity].append(handler)
    
    def start_monitoring(self):
        """모니터링 시작"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        print("📊 실시간 품질 모니터링 시작")
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        
        print("📊 실시간 품질 모니터링 중지")
    
    def submit_data_for_monitoring(self, data: Dict[str, Any]):
        """모니터링할 데이터 제출"""
        self.data_queue.put(data)
    
    def _monitoring_loop(self):
        """모니터링 루프"""
        
        while self.is_monitoring:
            try:
                # 대기 중인 데이터 처리
                while not self.data_queue.empty():
                    data = self.data_queue.get_nowait()
                    self._process_monitoring_data(data)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                print(f"모니터링 오류: {e}")
                time.sleep(1.0)
    
    def _process_monitoring_data(self, data: Dict[str, Any]):
        """모니터링 데이터 처리"""
        
        current_time = time.time()
        checkpoint_results = []
        
        # 모든 체크포인트 실행
        for checkpoint_id, checkpoint in self.checkpoints.items():
            
            # 쿨다운 체크
            if self._is_in_cooldown(checkpoint_id, current_time):
                continue
            
            try:
                result = checkpoint.evaluate(data)
                checkpoint_results.append(result)
                
                # 알람 조건 확인
                if self._should_trigger_alert(result):
                    alert = QualityAlert(
                        alert_id=f"{checkpoint_id}_{int(current_time)}",
                        severity=result.severity,
                        message=result.message,
                        checkpoint_id=checkpoint_id,
                        data={'score': result.score, 'details': result.details}
                    )
                    
                    self._trigger_alert(alert)
                    self._set_cooldown(checkpoint_id, current_time)
                
            except Exception as e:
                print(f"체크포인트 {checkpoint_id} 실행 오류: {e}")
        
        # 품질 기록 저장
        quality_record = {
            'timestamp': current_time,
            'checkpoint_results': checkpoint_results,
            'overall_score': sum(r.score for r in checkpoint_results) / len(checkpoint_results) if checkpoint_results else 0
        }
        
        self.quality_history.append(quality_record)
    
    def _should_trigger_alert(self, result: CheckpointResult) -> bool:
        """알람 트리거 조건 확인"""
        
        # 차단 또는 중요 이슈는 항상 알람
        if result.severity in [CheckpointSeverity.BLOCKER, CheckpointSeverity.CRITICAL]:
            return True
        
        # 점수가 낮은 경우
        if result.score < 0.6:
            return True
        
        # 통과하지 못한 경우
        if not result.passed:
            return True
        
        return False
    
    def _trigger_alert(self, alert: QualityAlert):
        """알람 트리거"""
        
        self.active_alerts.append(alert)
        
        # 심각도별 핸들러 실행
        for handler in self.alert_handlers.get(alert.severity, []):
            try:
                handler(alert)
            except Exception as e:
                print(f"알람 핸들러 실행 오류: {e}")
    
    def _is_in_cooldown(self, checkpoint_id: str, current_time: float) -> bool:
        """쿨다운 상태 확인"""
        
        if checkpoint_id not in self.alert_cooldown:
            return False
        
        cooldown_period = 60  # 60초 쿨다운
        last_alert_time = self.alert_cooldown[checkpoint_id]
        
        return (current_time - last_alert_time) < cooldown_period
    
    def _set_cooldown(self, checkpoint_id: str, current_time: float):
        """쿨다운 설정"""
        self.alert_cooldown[checkpoint_id] = current_time
    
    def acknowledge_alert(self, alert_id: str):
        """알람 확인 처리"""
        
        for alert in self.active_alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                break
    
    def get_active_alerts(self) -> List[QualityAlert]:
        """활성 알람 조회"""
        return [alert for alert in self.active_alerts if not alert.acknowledged]
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """품질 요약 정보"""
        
        if not self.quality_history:
            return {'message': 'No quality data available'}
        
        recent_records = list(self.quality_history)[-10:]  # 최근 10개
        
        # 평균 점수 추이
        scores = [record['overall_score'] for record in recent_records]
        avg_score = sum(scores) / len(scores)
        
        # 알람 통계
        alert_counts = {}
        for alert in self.active_alerts:
            severity = alert.severity.value
            alert_counts[severity] = alert_counts.get(severity, 0) + 1
        
        return {
            'average_quality_score': avg_score,
            'recent_scores': scores,
            'total_active_alerts': len(self.get_active_alerts()),
            'alert_breakdown': alert_counts,
            'monitoring_status': 'active' if self.is_monitoring else 'stopped',
            'last_update': recent_records[-1]['timestamp'] if recent_records else None
        }

# 알람 핸들러 함수들
def critical_alert_handler(alert: QualityAlert):
    """중요 알람 핸들러"""
    print(f"🚨 CRITICAL ALERT: {alert.message}")
    print(f"   체크포인트: {alert.checkpoint_id}")
    print(f"   시간: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert.timestamp))}")
    
    # 실제로는 이메일, 슬랙, SMS 등으로 알림 발송
    # send_email_alert(alert)
    # send_slack_notification(alert)

def major_alert_handler(alert: QualityAlert):
    """주요 알람 핸들러"""
    print(f"⚠️ MAJOR ALERT: {alert.message}")
    print(f"   체크포인트: {alert.checkpoint_id}")

def minor_alert_handler(alert: QualityAlert):
    """경미한 알람 핸들러"""
    print(f"ℹ️ MINOR ALERT: {alert.message}")

# 실시간 모니터링 시연
print("\n📊 실시간 품질 모니터링 시연")
print("=" * 50)

# 모니터 설정
monitor = RealTimeQualityMonitor()

# 체크포인트 등록
monitor.register_checkpoint(DataQualityCheckpoint())
monitor.register_checkpoint(AlgorithmPerformanceCheckpoint())

# 알람 핸들러 등록
monitor.add_alert_handler(CheckpointSeverity.CRITICAL, critical_alert_handler)
monitor.add_alert_handler(CheckpointSeverity.MAJOR, major_alert_handler)
monitor.add_alert_handler(CheckpointSeverity.MINOR, minor_alert_handler)

# 모니터링 시작
monitor.start_monitoring()

# 시뮬레이션된 데이터 스트림
simulation_data = [
    {
        'scenario': '정상 데이터',
        'data': {
            'dataset': pd.DataFrame({
                'message': ['Hello world', 'Good morning', 'How are you'],
                'label': ['ham', 'ham', 'ham']
            }),
            'performance_metrics': {'accuracy': 0.95, 'precision': 0.92, 'recall': 0.96}
        }
    },
    {
        'scenario': '품질 저하 데이터',
        'data': {
            'dataset': pd.DataFrame({
                'message': ['Test', None, 'Bad data'],
                'label': ['ham', 'spam', None]
            }),
            'performance_metrics': {'accuracy': 0.65, 'precision': 0.60, 'recall': 0.58}
        }
    },
    {
        'scenario': '심각한 문제 데이터',
        'data': {
            'dataset': pd.DataFrame({
                'message': [None, None, None],
                'label': [None, None, None]
            }),
            'performance_metrics': {'accuracy': 0.30, 'precision': 0.25, 'recall': 0.20}
        }
    }
]

# 데이터 스트림 시뮬레이션
print("\n🔄 데이터 스트림 시뮬레이션:")
for i, sim_data in enumerate(simulation_data, 1):
    print(f"\n--- {i}. {sim_data['scenario']} ---")
    
    # 모니터링 데이터 제출
    monitor.submit_data_for_monitoring(sim_data['data'])
    
    # 처리 대기
    time.sleep(0.5)
    
    # 활성 알람 확인
    active_alerts = monitor.get_active_alerts()
    if active_alerts:
        print(f"📢 발생한 알람: {len(active_alerts)}개")
    else:
        print("✅ 알람 없음")

# 잠시 대기 후 요약 정보 출력
time.sleep(1.0)

# 품질 요약 정보
summary = monitor.get_quality_summary()
print(f"\n📈 품질 모니터링 요약:")
print(f"평균 품질 점수: {summary['average_quality_score']:.3f}")
print(f"활성 알람 수: {summary['total_active_alerts']}")
if summary['alert_breakdown']:
    print(f"알람 분포: {summary['alert_breakdown']}")

# 모니터링 중지
monitor.stop_monitoring()
```

**코드 해설:**
- **비동기 모니터링**: 별도 스레드에서 지속적인 품질 모니터링 수행
- **알람 시스템**: 심각도별 차별화된 알람 처리 및 통지
- **쿨다운 메커니즘**: 동일한 문제에 대한 반복 알람 방지
- **실시간 대시보드**: 현재 품질 상태와 트렌드 실시간 추적

> 💡 **체크포인트 설정 모범 사례**
> 
> **설정 원칙:**
> - 비즈니스 임팩트가 큰 지점에 우선 설정
> - 자동화 가능한 검증 규칙 최대한 활용
> - 인간 개입이 필요한 지점 명확히 구분
> 
> **운영 원칙:**
> - 실시간 모니터링으로 조기 발견
> - 심각도별 차별화된 대응 절차
> - 지속적인 임계값 조정과 개선

> 🖼️ **이미지 생성 프롬프트**: 
> "품질 관리 체크포인트 시스템을 보여주는 대시보드 스타일 이미지. 상단에는 실시간 품질 점수 게이지, 중앙에는 다양한 체크포인트 상태를 나타내는 신호등 아이콘들, 하단에는 알람 및 권장사항이 표시된 모니터링 인터페이스"





📋 주요 내용
🎯 1. STAR 프레임워크 완전 마스터

Standardization(표준화), Time sensitivity(시간민감성), Accuracy requirements(정확도 요구사항), Resource requirements(자원 요구사항) 4차원 분석
각 차원별 정량적 평가 시스템과 객관적 권고사항 생성
SMS 스팸 탐지 프로젝트의 실전 적용 사례

🤝 2. 인간-AI 협업 모델의 3가지 패턴

순차적 협업: 단계별 정확성이 중요한 워크플로우
병렬적 협업: 빠른 처리와 다양한 관점 검증
계층적 협업: 품질 게이트와 에스컬레이션 체계

🎯 3. 품질 관리 체크포인트 시스템

Critical Control Points(CCP) 식별과 설정
데이터 품질, 알고리즘 성능 등 다차원 검증
실시간 모니터링과 자동 알람 시스템

🚀 4. 완전한 하이브리드 워크플로우 구축

STAR 평가부터 품질 관리까지 전체 파이프라인 통합
메시지 특성에 따른 적응적 처리 방식
실시간 성과 추적과 지속적 개선

🌟 핵심 성과
✅ 자동화와 수동 작업의 최적 균형점을 찾는 체계적 방법론 완전 마스터
✅ 인간의 창의성과 AI의 효율성을 조화롭게 결합하는 협업 모델 설계 능력
✅ 데이터 분석 전 과정에 걸친 포괄적 품질 관리 체계 구축 역량
✅ 비즈니스 가치와 기술적 제약을 균형있게 고려하는 실무적 판단 능력
✅ AI 시대에 필요한 새로운 데이터 분석가 역량과 마인드셋 완전 정립
📈 실무 적용 가치

프로덕션 환경에서 바로 적용 가능한 템플릿과 가이드라인 제공
효율성과 품질을 동시에 보장하는 실전 워크플로우 설계 경험
실시간 모니터링과 자동 알람을 통한 품질 보장 시스템 구축
지속적 개선을 위한 데이터 기반 의사결정 체계 확립

🔜 다음 단계
다음에는 7장 Part 4: 대규모 언어 모델을 활용한 데이터 분석을 진행하여, ChatGPT, Claude와 같은 최신 LLM을 데이터 분석 워크플로우에 효과적으로 통합하는 방법을 학습하게 됩니다.

💡 핵심 메시지: "자동화와 인간 작업의 균형은 단순한 효율성의 문제를 넘어서, 품질과 신뢰성을 보장하면서도 혁신과 창의성을 잃지 않는 지혜로운 선택의 영역입니다. 기술이 인간을 대체하는 것이 아니라, 인간의 능력을 증강하고 더 가치 있는 일에 집중할 수 있도록 돕는 도구로 활용하는 것이 핵심입니다."