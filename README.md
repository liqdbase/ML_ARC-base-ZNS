ZNS-ARC: ML-Enhanced Cache Management System
개요 (Overview)
ZNS-ARC는 ZNS(Zoned Namespace) SSD 환경의 성능 병목을 해결하기 위해 설계된 차세대 캐시 시스템입니다. 전통적인 ARC(Adaptive Replacement Cache) 알고리즘에 XGBoost 기반 지능형 예측 엔진을 결합하여 스토리지 레이어의 효율성을 극대화합니다.

ZNS-ARC is a next-generation cache system designed to address performance bottlenecks in Zoned Namespace (ZNS) SSD environments. It maximizes storage layer efficiency by integrating an XGBoost-based intelligent prediction engine with the traditional ARC (Adaptive Replacement Cache) algorithm.

핵심 기술 요약 (Technical Highlights)
예측형 캐싱 (Predictive Caching)

150개의 결정 트리(Decision Tree)로 구성된 XGBoost 모델이 실시간으로 데이터의 재사용 확률을 계산합니다.

An XGBoost model consisting of 150 decision trees calculates data reuse probability in real-time.

ZNS 특화 설계 (ZNS-Aware Design)

Zone별 독립적인 락(Lock) 체계와 시퀀셜 락(Seqlock)을 도입하여 ZNS 하드웨어의 병렬성을 최적화하고 데이터 일관성을 보장합니다.

Optimizes parallelism and ensures data consistency by implementing per-zone independent locks and Seqlocks tailored for ZNS hardware.

라이브러리 독립형 ML (Zero-Library ML)

외부 라이브러리 없이 C 언어로만 구현된 경량 추론 엔진을 탑재하여 임베디드 및 커널 환경에서도 제약 없이 동작합니다.

Features a lightweight inference engine implemented purely in C without external dependencies, allowing seamless operation in embedded and kernel environments.

동작 원리 (How It Works)
데이터 수집 (Input)

참조 빈도, 시간 간격 등 페이지의 주요 특징량(Features)을 실시간 수집합니다.

Collects key page features such as reference frequency and recency in real-time.

추론 엔진 (Inference)

내장된 150개의 결정 트리를 순회하며 노드별 임계값을 비교 연산합니다.

Traverses 150 embedded decision trees and performs threshold comparisons at each node.

결과 도출 (Output)

트리의 합산 결과에 Sigmoid 연산을 적용하여 최종 재사용 확률을 산출합니다.

Calculates the final reuse probability by applying a Sigmoid operation to the aggregated tree scores.

정책 반영 (Action)

예측된 확률에 따라 ARC 리스트(T1, T2) 간 이동 및 데이터 제거 우선순위를 동적으로 조정합니다.

Dynamically adjusts migration between ARC lists (T1, T2) and eviction priorities based on the predicted probability.

성능 모니터링 지표 (Key Performance Monitors)
캐시 히트율 (Cache Hit Ratio): ML 예측을 통한 캐시 효율성 향상 지표.

Measures cache efficiency improvements driven by ML predictions.

경합 분석 (Contention Analysis): Zone별 락 경합 상태를 모니터링하여 병목 구간 식별.

Identifies bottlenecks by monitoring lock contention status across individual zones.

제거 지연 시간 (Eviction Latency): 지연 제거 큐(Deferred Eviction Queue)를 통한 응답 속도 최적화 수치.

Tracks response time optimization achieved through the Deferred Eviction Queue.

기술 스택 (Tech Stack)
Language: C (Atomics, Seqlock, Pthread)

Algorithm: Adaptive Replacement Cache (ARC)

ML Model: XGBoost (Flattened Array Inference)

Environment: Zoned Namespace (ZNS) SSD / QEMU (FEMU)
