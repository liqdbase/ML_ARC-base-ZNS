/* arc_ml_predictor_seqlock.h - Thread-Safe Version with Seqlock */

#ifndef ARC_ML_PREDICTOR_SEQLOCK_H
#define ARC_ML_PREDICTOR_SEQLOCK_H

#ifdef __cplusplus
extern "C" {
#endif

	/**
	 * XGBoost Inference for ZNS-ARC
	 * Flat Array Implementation with Thread-Safe Feature Extraction
	 */

	 // 기존 예측 함수 (피처 배열 직접 전달)


	// 원자적 피처 복사 + 예측 함수 (Seqlock 사용)
	// PageFeatures 구조체를 직접 받아 안전하게 복사 후 예측



	// ML 모델 초기화/해제
	void init_ml_predictor(void* unused);
	void free_ml_predictor(void);

#ifdef __cplusplus
}
#endif

#endif /* ARC_ML_PREDICTOR_SEQLOCK_H */
