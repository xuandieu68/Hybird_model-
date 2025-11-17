

### **Hybrid Approach: Machine Learning–Enhanced Traditional Regression in Explaining Firm Value**

---

## **1. 연구 배경 및 동기 (Introduction & Motivation)**

### 1.1 기업가치(Firm Value) 연구의 중요성

* 기업 재무 의사결정, 투자 전략, 기업정책 평가에서 핵심 지표
* 대표 측정치: **Tobin’s Q, Market Value (MV)**

### 1.2 기존 연구의 한계

* 전통적 회귀분석은

  * 선형 가정(linearity assumption)
  * 다중공선성 문제
  * 비선형 관계 및 변수 간 상호작용을 포착하기 어려움
* 최근 ML 연구는 높은 예측력을 보이나

  * “블랙박스” 문제
  * 경제학적 해석 부족

### 1.3 본 연구의 기여

* **경제계량(econometrics)**의 해석력 + **ML**의 예측 성능 결합
* **Hybrid Econometric–Machine Learning Framework** 구축
* ML로 비선형 구조와 상호작용 탐지 → 회귀모형 해석 강화

---

## **2. 데이터 및 변수 구성 (Data & Variables)**

### 2.1 표본

* 한국 비금융 기업
* **2006–2024년**, 연도별 균형패널(balanced panel) 구축
* 데이터 출처: FnGuide
* Train 2006-2020
*  Test 2021-2024
  
### 2.2 종속변수 (Firm Value)

* **Tobin’s Q**
* **Market Value  (MV)**

### 2.3 주요 설명변수

* **수익성(EBITDA/TA, ROA, ROE, OPM, NPM)**
* **재무정책(레버리지, 배당정책)**
* **기업 규모(Size)**
* **성장성(Growth)**
* **유동성(Liquidity)**

---

## **3. 연구 방법론 (Methodology)**

### 3.1 1단계: 선형 기반 모델 (Linear Models)

* OLS, Ridge, LASSO, Elastic Net
* 목적:

  * 선형 관계 파악
  * 변수 선택(variable selection)
  * 해석 가능성 확보

### 3.2 2단계: 비선형 ML 모델 (Non-linear ML Models)

* XGBoost
* Random Forest
* LightGBM
* 목적:

  * 비선형성 포착
  * 변수 간 상호작용 탐지

### 3.3 3단계: Hybrid Model 설계

* 선형 + ML 결합
* **모델별 Out-of-sample MAE** 기반 가중치(weight) 최적화
  
* 목적:
  * ML의 예측력 + Econometrics의 해석력 결합

---

## **4. 실증결과 (Empirical Results)**

### 4.1 모델별 성능 비교


  * **LASSO가 가장 높은 R²/MAE 성능**
  * 수익성·규모·레버리지 계수 안정적

* **Hybrid 모델이 단일 모델보다 일관적으로 우수한 성능**

---

## **5. 변수 중요도 분석 (Feature Importance & SHAP)**

### 5.1 ML 기반 변수 중요도

* XGBoost Feature Importance
* LightGBM Gain-based importance

가장 중요한 변수:

1. **EBITDA/TA**
2. **배당정책(Dividend payout or dividend dummy)**
3. **ROA**
4. **기업 규모(Size)**

### 5.2 SHAP 분석 결과

* EBITDA 상승 → 기업가치 증가
* 레버리지 효과는 비선형이며 구간별로 상이
* 배당정책은 높은 양(+)의 기여
* 규모(Size)의 영향은 안정적이고 일관적으로 양(+)

---

## **6. 추가 검증 (Robustness Checks)**

### 6.1 고정효과 패널 회귀 (Fixed Effects Regression)

* SHAP으로 추출한 주요 변수 중심
* 결과:

  * EBITDA, ROA, 배당정책, 규모의 유의성 유지
  * ML에서 도출된 중요 변수가 경제계량에서도 robust

### 6.2 산업/연도 FE 통제

* 산업별 충격과 경기 변동성 조정 완료
* 결과의 안정성 추가 확인

---

## **7. 논의 및 시사점 (Discussion & Implications)**

### 7.1 학술적 시사점

* Hybrid 접근법은

  * 기존 자본구조 및 기업가치 문헌의 확장
  * ML 기반 탐색 + 회귀 기반 검증의 새로운 방법론 제시

### 7.2 실무적 시사점

* 기업 재무전략 의사결정 시

  * 수익성·배당정책·규모 변수가 가장 중요
* 레버리지 정책은 비선형적 영향
* Hybrid 모델은 기업가치 예측모형으로 활용 가능

---

## **8. 결론 (Conclusion)**

* ML과 Econometrics 결합으로 성능 및 해석력 모두 개선
* LASSO(해석력) + XGBoost(비선형·예측력) 결합이 최적
* SHAP 기반 인사이트는 전통적 분석과도 일관
* 미래 연구

  * DML(Double Machine Learning) 적용
  * 임계값·비선형성에 대한 구조적 모형 확장
  * 산업별·경기 국면별 차이 분석

---

원하시면,

* **슬라이드 버전(PPT 구조)**
* **한국어/영어 비교 버전**
* **교수님 질문 예상 리스트 + 답변 템플릿**
  도 추가로 만들어드릴게요.
