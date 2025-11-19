

### **Hybrid Approach: Machine Learning–Enhanced Traditional Regression in Explaining Firm Value**

---

## **1. 연구 배경 및 동기 **

### 1.1 기업가치 연구의 중요성

* 기업 재무 의사결정, 투자 전략, 기업정책 평가에서 핵심 지표
* 대표 측정치: **Tobin’s Q, Market Value (MV)**

### 1.2 기존 연구의 한계

* 전통적 회귀분석은

  * 선형 가정
  * 다중공선성 문제
  * 비선형 관계 및 변수 간 상호작용을 포착하기 어려움
* 최근 ML 연구는 높은 예측력을 보이나

  * “블랙박스” 문제
  * 경제학적 해석 부족

### 연구 질문 
* 선형 및 비선형 모델을 결합하는 것이 기업 가치를 예측하고 해석하는 데 더 효과적일까요?
* ML + SHAP은 기존 모델의 설명 가능성과 안정성을 향상시킵니까?
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

* 수익성(EBITDA/TA, ROA, ROE, OPM, NPM)
* 재무정책(레버리지, 배당정책)
* 기업 규모(Size)
* 성장성(Growth)
* 유동성(Liquidity)
* TANG,CH,FCF_TA,Z_SCORE
---

## **3. 연구 방법론 (Methodology)**

### 3.1 1단계: 선형 기반 모델 (Linear Models)

*  Ridge, LASSO, Elastic Net
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
                                                                    |

### 3.3 3단계: Hybrid Model 설계

* 선형 + ML 결합 (
* **모델별 결정계수** 기반 가중치(weight) 최적화
 
$$
\hat{y}_{\text{hybrid}} = w_1 \hat{y}_{\text{lasso}} + w_2 \hat{y}_{\text{xgb}}, \quad \text{with} \quad w_1 + w_2 = 1
$$
 

* 목적:
  * ML의 예측력 + Econometrics의 해석력 결합
  
| **Mô hình**                  | **Tham số**                                                                                                                                                                                               |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Random Forest (RF)**       | - **n_estimators**: [100, 300] <br> - **max_depth**: [None, 10, 20] <br> - **min_samples_split**: [10] <br> - **min_samples_leaf**: [4]                                                                   |
| **XGBoost (XGB)**            | - **max_depth**: [6] <br> - **gamma**: [0] <br> - **reg_alpha**: [1] <br> - **reg_lambda**: [10] <br> - **colsample_bytree**: [1.0] <br> - **min_child_weight**: [20] <br> - **n_estimators**: [200, 300] |
| **LightGBM (LGBM)**          | - **n_estimators**: [200, 300] <br> - **max_depth**: [6, 10] <br> - **learning_rate**: [0.05, 0.1] <br> - **num_leaves**: [31, 50]                                                                        |
| **Ridge Regression (Ridge)** | - **alpha**: [0.1, 1.0, 10.0]                                                                                                                                                                             |
| **Lasso Regression (Lasso)** | - **alpha**: [0.001, 0.01, 0.1, 1.0]                                                                                                                                                                      |
| **ElasticNet**               | - **alpha**: [0.001, 0.01, 0.1] <br> - **l1_ratio**: [0.2, 0.5, 0.8]                                                                                                                                      |


### 3.4 4단계: Feature important, SHAP 분석 
* 각 모형의 feature important 통해 모델이 학습 및 출력 예측에 사용하는 중요한 변수들을 밝히다
  *  Feature Importance(랜덤 포레스트, XGBoost 또는 LightGBM에서와 같이)는 각 변수가 전체 모델 트리에서 예측 오류를 줄이거나 불순도(impurity)를 감소시키는 데 기여하는 평균 정도를 측정합니다.	이는 데이터 분리에 대한 변수의 전반적인 힘을 반영하지만, 영향의 방향(양 또는 음)은 고려하지 않으며, 해당 변수가 여러 값을 가지거나 다른 변수와 높은 상관관계를 가질 경우 편향될 수 있습니다.

* SHAP과 결합하여 각 변수가 기업 가치에 미치는 영향과 그 영향력의 정도를 시각화·정량화
   * 	SHAP 값은 Shapley 게임 이론(Shapley Additive Explanations)에 기반하여 각 관측값에서 각 변수의 한계 기여도를 추정합니다. SHAP은 예측값의 증가/감소 방향과 변수 간의 비선형성 또는 상호작용 정도를 모두 나타냅니다.



---
# Result



#### ** Performance of Linear and Non-Linear Models**
**Table 2 Linear model results** 
|            | Train_R2 | Train_RMSE | Train_MAE | Test_R2 | Test_RMSE | Test_MAE |
| :--------- | -------: | ---------: | --------: | ------: | --------: | -------: |
|            |          |  Panel A : |         Q |         |           |          |
| Ridge      |   0.2440 |     1.2274 |    0.6192 |  0.1354 |    1.5396 |   0.7934 |
| LASSO      |   0.2437 |     1.2276 |    0.6184 |  **0.1421** |    1.5337 |   0.7916 |
| ElasticNet |   0.2438 |     1.2276 |    0.6182 |  0.1409 |    1.5347 |   0.7915 |
|            |          |  Panel B : |        MV |         |           |          |
| Ridge      |   0.2453 |     1.2179 |    0.6087 |  0.1378 |    1.5339 |   0.7886 |
| LASSO      |   0.2450 |     1.2181 |    0.6080 | **0.1451** |    1.5275 |   0.7864 |
| ElasticNet |   0.2451 |     1.2181 |    0.6079 |  0.1437 |    1.5286 |   0.7865 |

- 제한된 설명력


**Table 3 Non-Linear model results**
|              | Train_R2 | Train_RMSE | Train_MAE | Test_R2 | Test_RMSE | Test_MAE |
| :----------- | -------: | ---------: | --------: | ------: | --------: | -------: |
|              |          |  Panel A : |         Q |         |           |          |
| RandomForest |   0.8158 |     0.6058 |    0.2572 |  0.5245 |    1.1418 |   0.6298 |
| XGBoost      |   0.9059 |     0.4331 |    0.2720 |  0.6060 |    1.0394 |   0.5888 |
| LightGBM     |   0.8484 |     0.5496 |    0.3324 |  0.5949 |    1.0539 |   0.5851 |
|              |          |  Panel B : |        MV |         |           |          |
| RandomForest |   0.8149 |     0.6031 |    0.2539 |  0.5194 |    1.1452 |   0.6299 |
| XGBoost      |   0.9101 |     0.4203 |    0.2652 |  0.6086 |    1.0335 |   0.5802 |
| LightGBM     |   0.8312 |     0.5760 |    0.3394 |  0.5920 |    1.0552 |   0.5846 |

- 모델들은 선형 그룹보다 훨씬 높은 설명력
-  XGBoost는 높은 예측 정확도와 강력한 일반화 기능을 결합하여 전반적인 성능이 가장 우수
  
#### **Hybrid Model **
두 모델 그룹의 성능 결과를 바탕으로 선형 그룹을 대표할 LASSO와 Hybrid 모델에 결합할 비선형 그룹을 대표할 XGBoost를 선택했습니다.

$$
\hat{y}_{\text{hybrid}} = w_1 \hat{y}_{\text{lasso}} + w_2 \hat{y}_{\text{xgb}}, \quad \text{with} \quad w_1 + w_2 = 1
$$

Table 4: Hybird model metric performance
|     | Train_R2 | Test_R2 | Train_RMSE | Test_RMSE | Train_MAE | Test_MAE | Weight_Linear | Weight_Nonlinear |
| --- | -------- | ------- | ---------- | --------- | --------- | -------- | ------------- | ---------------- |
| Q   | 0.8678   | 0.6205  | 0.5132     | 1.0200      | 0.3065    | 0.5866   | 0.19          | 0.81             |
| MV  | 0.8721   | 0.6216  | 0.5014     | 1.0162    | 0.2998    | 0.5794   | 0.19       | 0.81           |

표 4의 결과는 결정계수 가중치 기여도(0.19 및 0.81)를 기준으로 LASSO와 XGBoost를 결합하여 구축한 하이브리드 모델이 두 모델 단독보다 더 높은 예측 성능을 달성한다는 것을 보여줍니다. 하이브리드 접근 방식은 두 가지 기업 가치 측정치(Q 및 MV)에 대해 약 0.62의 테스트 R²를 산출하며, 개별 모델에 비해 RMSE 및 MAE가 감소합니다. R²의 개선은 크지 않지만, 이 모델은 XGBoost에 비해 과적합이 감소하여 선형 가중치 구성 요소가 변수 간의 관계가 주로 선형인 영역에서 예측을 효과적으로 정규화한다는 것을 시사합니다. 동시에, XGBoost에서 물려받은 비선형 학습 능력은 그대로 유지되어 하이브리드 모델이 기업 특성 간의 복잡한 상호작용을 포착할 수 있습니다. 전반적으로 이러한 결과는 계량경제학적 해석 가능성과 머신러닝 유연성 간의 상호보완성을 확인하여 하이브리드 모델이 기업 가치 예측을 위한 보다 균형 잡히고 강력한 프레임워크임을 뒷받침합니다.


 Fig 1. Sensitivity analysis
<img width="961" height="313" alt="image" src="https://github.com/user-attachments/assets/b245f2c8-8ead-4cb9-95d7-ebd0ed74160b" />

이 연구에서는 가중치 변화에 대한 결과의 민감도도 조사했습니다. 그림 1의 결과를 보면 두 패널 모두 선형 가중치 w_linear = 0.15일 때 Hybrid 모델의 R²가 최대(0.6214 및 0.6228)로 나타나 이 조합이 설명 부분 변수의 경제적 유의성을 유지하면서 최적의 예측 성능을 제공한다는 것을 확인할 수 있습니다.

---
3. Feature Importance Analysis

<figure>
  <img width="1042" height="405" alt="image" src="https://github.com/user-attachments/assets/f1d6f22a-5458-4a8f-a156-a9feb9e3a999" />
  <figcaption style="text-align:center;font-style:italic;">Figure 1. Initial Variable Importance across Machine Learning Models</figcaption>
</figure>

그림 2.1과 2.2의 세 가지 비선형 모델 모두에서 유의미한 변수는 일관성이 높으며, 대부분 각 모델에서 상위 5개에 속하는 기업 규모(SIZE), 수익성(EBITDA_TA, ROA), 재무 안정성(Z_SCORE)과 함께 DIV_DUMMY가 기업 가치의 주요 예측 변수로 나타났습니다.  모델 간의 높은 일관성은 이러한 결과의 견고함을 강조하며, 특징 순위의 약간의 차이는 각 알고리즘이 기업 특성 간의 비선형 관계를 포착하는 뚜렷한 방식을 보여줍니다.


<img width="900" height="393" alt="image" src="https://github.com/user-attachments/assets/9242d16b-bbed-4b12-9e95-aeed99c339e1" />

Figure 3. Top 10 Hybrid Feature Importance combining LASSO and XGBoost
<img width="913" height="398" alt="image" src="https://github.com/user-attachments/assets/c8eb545a-3186-40d0-801d-439b5e68ea49" />

---


Figure 4. SHAP value plots
### For Q
<img width="1000" height="400" alt="image" src="https://github.com/user-attachments/assets/14507732-9be6-4a19-aaec-adecfad15877" />

### For MV
<table>
  <tr>
    <img width="330" height="400" alt="image" src="https://github.com/user-attachments/assets/f0fffe83-dcd6-420a-aa98-3d1c73542697" />
    <img width="330" height="400" alt="image" src="https://github.com/user-attachments/assets/ee1d3fd8-abad-4d57-a824-e5374c174d36" />
    <img width="330" height="400" alt="image" src="https://github.com/user-attachments/assets/62da4fae-fe2c-4f3a-b419-351f2e598686" />
  </tr>
</table>

모델에서 변수들의 SHAP 값 분포의 유사성을 살펴보면, Size의 증가가 SHAP 값과 긍정적인 상관관계를 가지며, 마찬가지로 Z_SCORE도 모델 출력과 긍정적인 상관관계를 보입니다. 반면, EBITDA_TA, ROA, OPM, NPM, ROE와 같은 수익성 변수들은 SHAP 값의 0.0을 기준으로 왼쪽과 오른쪽에 빨간 점들이 나타납니다. 하지만 왼쪽에서의 더 큰 겹침은 많은 경우 수익성이 높은 기업이 모델의 예측 값에 음의 기여를 한다는 것을 나타냅니다. 또한, 오른쪽 축에서 변수 값이 강하고 약한 값이 넓게 분포하며 섞여 있는 것은 수익성 변수들이 비선형적 관계를 가질 수 있거나 다른 변수들과 상호작용할 수 있음을 나타냅니다. LEV는 세 모델 모두에서 명확한 방향성을 보이지 않으며, 대신 상황에 따라 다른 변수들의 영향을 증폭시키거나 변경시킵니다.

--

### FE model 

 <img width="1127" height="686" alt="image" src="https://github.com/user-attachments/assets/854f9eac-0e24-4268-86e6-bac2a06bb8cf" />








---

**SHAP Dependence Plots**

<img width="653" height="453" alt="image" src="https://github.com/user-attachments/assets/aa8cb198-9ed2-45cf-8f75-2f11ee3d31c8" />
<img width="662" height="453" alt="image" src="https://github.com/user-attachments/assets/60dceed2-debe-47ca-8d2d-72fcfe3cddaf" />
<img width="645" height="453" alt="image" src="https://github.com/user-attachments/assets/78325c82-2a75-420c-b1fe-4688fbad7121" />
<img width="666" height="453" alt="image" src="https://github.com/user-attachments/assets/f0c91d45-99ed-43fa-b249-f54592570ee4" />
<img width="662" height="453" alt="image" src="https://github.com/user-attachments/assets/6a66732a-f7a3-4207-90d6-9004645a9737" />
<img width="666" height="453" alt="image" src="https://github.com/user-attachments/assets/c3d64710-b66c-4784-b900-7a057100ace9" />
<img width="654" height="453" alt="image" src="https://github.com/user-attachments/assets/27bec632-a63d-46ad-9037-35154afa31c7" />
<img width="683" height="453" alt="image" src="https://github.com/user-attachments/assets/4b48e225-9e0f-4076-a2a4-04e153c7f5e0" />
<img width="666" height="453" alt="image" src="https://github.com/user-attachments/assets/8ecceef2-e8b1-4fcc-9ed3-6131dc30a2c1" />
<img width="666" height="453" alt="image" src="https://github.com/user-attachments/assets/004649a2-f049-4b89-9abf-5b12893a9282" />
<img width="653" height="453" alt="image" src="https://github.com/user-attachments/assets/a915c903-5093-457f-a867-35e05b85a3ec" />

SHAP 요약 플롯은 각 특징에 대한 일반적인 개요를 제공하고, SHAP 의존성 그래프는 모델 출력이 기능 값에 따라 어떻게 변하는지를 보여줍니다

---

## VIF 
<img width="215" height="503" alt="image" src="https://github.com/user-attachments/assets/4444d057-3a29-4fe7-9371-cd5ce2278a40" />
SIZE는 다른 변수들과 높은 다중공선성을 보임 -> 전통적 회귀 분석 시 주의가 필요함

## VIF with SIZE_RESID

| LEV          | 5.010903 |
| ------------ | -------- |
| Z_SCORE      | 4.365959 |
| EBITDA_TA    | 3.972848 |
| TANG         | 2.937023 |
| ROA          | 2.472748 |
| GP           | 2.321513 |
| DIV_DUMMY    | 2.252178 |
| FCF_TA       | 2.166699 |
| CH           | 2.101364 |
| ROE          | 2.019562 |
| OPM          | 1.453009 |
| LIQ          | 1.358045 |
| NPM          | 1.300048 |
| GROWTH_SALES | 1.005229 |
| SIZE_RESID   | 1.000872 |






---

## **6. 추가 검증 (Robustness Checks)**

 고정효과 패널 회귀 (Fixed Effects Regression)

* SHAP으로 추출한 주요 변수 중심



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


