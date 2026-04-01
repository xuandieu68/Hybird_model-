
# new project




---

## 1. 방법론 및 다중공선성 처리 (Methodology and Multicollinearity Handling)

### 표 1: VIF 결과 (Table 1: VIF)

VIF 결과에 따르면, **SIZE** 변수는 **12.0922**의 값을 보여 다중공선성(multicollinearity) 수준이 유의미함을 반영하며, 변수들이 많은 정보를 공유할 때 전통적인 선형 회귀 모형이 **왜곡된 계수**를 산출할 수 있음을 경고한다. 이러한 한계로 인해, 비선형 관계 및 변수 간 상호작용을 잘 처리할 수 있으며 비다중공선성 가정을 따르지 않는 기계 학습 모형으로 분석을 확장할 필요성이 제기된다. 따라서 ML로의 전환은 예측 정확도 향상뿐만 아니라, 변수 간 높은 상관관계가 있는 데이터 맥락에서 결과의 견고성(robustness)을 테스트하기 위함이다.

또한, 본 연구는 계량경제학적 모형에서 잔차 변환(residualized transformation)을 적용하여 **SIZE\_RESID** 변수를 생성하였는데, 이는 다중공선성을 최소화하면서 변수의 경제적 유의성을 보존하기 위함이다. SIZE\_RESID 변수를 사용한 VIF 값은 1.0009로 현저히 감소하여, 다중공선성 문제가 성공적으로 해결되었음을 나타낸다.


---

## 2. 모형 성과 및 비교 (Model Performance and Comparison)


## Table 2.1 Predictive Performance with GridSearch Timeseries Cross-Validation 
|            |   Train_R2 |   Train_RMSE |   Train_MAE |   Test_R2 |   Test_RMSE |   Test_MAE |
|:-----------|-----------:|-------------:|------------:|----------:|------------:|-----------:|
| Linear | | | | | 
| OLS        |     0.1680 |       1.3067 |      0.6496 |    0.0413 |      1.4989 |     0.8012 |
| Ridge      |     0.1679 |       1.3067 |      0.6494 |    0.0433 |      1.4972 |     0.8004 |
| LASSO      |     0.1676 |       1.3070 |      0.6496 |    0.0509 |      1.4914 |     0.7990 |
| ElasticNet |     0.1678 |       1.3069 |      0.6492 |    0.0465 |      1.4948 |     0.7993 |
|Nonlinear|||||||
| RandomForest |     0.7119 |       0.7690 |      0.3236 |    0.2989 |      1.2817 |     0.7220 |
| XGBoost      |     0.8089 |       0.6263 |      0.3713 |    0.2519 |      1.3240 |     0.7324 |
| LightGBM     |     0.6876 |       0.8007 |      0.4409 |    0.2998 |      1.2809 |     0.7003 |



## Table 2.2  Predictive Performance with Rolling expanding window
|              |   Train_R2 |   Train_RMSE |   Train_MAE |   Test_R2 |   Test_RMSE |   Test_MAE |   CV_MSE |
|:-------------|-----------:|-------------:|------------:|----------:|------------:|-----------:|---------:|
| OLS          |     0.1680 |       1.3067 |      0.6496 |    0.0413 |      1.4989 |     0.8012 |   2.4769 |
| Ridge        |     0.1679 |       1.3067 |      0.6494 |    0.0433 |      1.4972 |     0.8004 |   2.4723 |
| LASSO        |     0.1676 |       1.3070 |      0.6496 |    0.0509 |      1.4914 |     0.7990 |   2.4711 |
| ElasticNet   |     0.1632 |       1.3104 |      0.6488 |    0.0624 |      1.4822 |     0.7919 |   2.4711 |
| RandomForest |     0.6686 |       0.8246 |      0.3833 |    0.3002 |      1.2805 |     0.7183 |   2.9785 |
| XGBoost      |     0.6356 |       0.8648 |      0.4828 |    0.3105 |      1.2711 |     0.7087 |   2.1955 |
| LightGBM     |     0.6140 |       0.8900 |      0.4550 |    0.3033 |      1.2777 |     0.6974 |   2.1530 |

Tables 2.1 and 2.2 report the predictive performance of linear and nonlinear models under Time Series Cross-Validation and Rolling Expanding Window, respectively. Across both validation strategies, nonlinear models consistently outperform their linear counterparts, confirming that firm value is governed by complex, nonlinear relationships poorly captured by conventional specifications. Linear models yield test R² of only 0.04–0.06, whereas tree-based models achieve substantially higher out-of-sample accuracy. Under Time Series Cross-Validation, LightGBM leads with a test R² of 0.2998; under the Rolling Expanding Window, XGBoost peaks at 0.3105 while LightGBM records the lowest MAE of 0.6974. Overall, LightGBM demonstrates the greatest stability across both strategies and is therefore selected for subsequent SHAP-based interpretation.

#### Timeseries hybird

<img width="847" height="540" alt="image" src="https://github.com/user-attachments/assets/c414ea6c-8dbf-4f8d-9d2c-d1c2a49fc22b" />

### rolling 

<img width="855" height="547" alt="image" src="https://github.com/user-attachments/assets/6792c3ae-6092-4085-ac6f-7c78b25260a3" />

## feature important 
### cv  ( Appendix.)

<img width="1040" height="403" alt="image" src="https://github.com/user-attachments/assets/c892b64e-4591-4f79-b11d-8e51e14c9192" />

### rolling
<img width="1040" height="403" alt="image" src="https://github.com/user-attachments/assets/b0716987-785d-4925-b160-45f2030ca4d2" />

## SHAP 
### cv

<img width="1991" height="591" alt="image" src="https://github.com/user-attachments/assets/72fc2aaf-c88d-47ca-8fff-82bdd86264e2" />

<img width="762" height="755" alt="image" src="https://github.com/user-attachments/assets/8b3ed216-eb35-4f55-bd01-f40e7083e772" />

### rolling 

<img width="1991" height="591" alt="image" src="https://github.com/user-attachments/assets/016d5d72-faff-47b8-ada3-0ec5a2d9e1aa" />

Figture 1.  LightGBM SHAP value 

<img width="762" height="741" alt="image" src="https://github.com/user-attachments/assets/19d649ec-df8e-4dff-8c6d-a7d26394234b" />

Figure 1 presents the SHAP beeswarm plot for LightGBM, revealing both the ranking and directional effect of each predictor on Tobin's Q. SIZE emerges as the dominant driver, with high values generating large positive SHAP contributions, consistent with the notion that larger Korean firms command a market premium. OPM and DIV_DUMMY rank second and third, where high operating profitability and dividend-paying status both associate positively with firm value. GP and Z_SCORE follow, suggesting that gross profitability and financial soundness are meaningful signals to the market. Notably, LEV exhibits a predominantly negative directional effect, indicating that higher leverage reduces Tobin's Q — consistent with agency cost theory. Variables in the lower tier — including ROA, EBITDA_TA, FCF_TA, ROE, and TANG — contribute marginally, implying limited incremental predictive power once upper-tier features are accounted for. Overall, the SHAP results highlight that market-based size and profitability quality dominate firm value formation in the Korean market.

# Dependent plot 

| ![Image 1](https://github.com/user-attachments/assets/430f7545-c61b-4281-876c-ad889e69de1d) | ![Image 2](https://github.com/user-attachments/assets/77d28c8c-72b6-46ee-a784-da7eafe3830e) |
|:----------------------------------------------------------:|:-----------------------------------------------------------:|
| ![Image 7](https://github.com/user-attachments/assets/2a75aa6a-cfab-4b94-8452-2afe3ee40263)  | ![Image 4](https://github.com/user-attachments/assets/65eae4c2-642a-4861-88f7-4f6ba874efac) |
| ![Image 5](https://github.com/user-attachments/assets/02e74099-1fcd-4aca-925e-9d4a8d096854) | ![Image 6](https://github.com/user-attachments/assets/ddac38a5-8c10-440e-a755-3b7577aea57f) | 

The SHAP dependence plots reveal that the relationships between financial indicators and firm value are strongly nonlinear and heterogeneous. CH, SIZE, Z_SCORE, and LIQ generally show positive effects on predicted firm value, although these effects are not constant across their distributions. In particular, SIZE and Z_SCORE exhibit threshold-type patterns, while LIQ and CH display positive effects with diminishing or heterogeneous marginal contributions at higher levels. LEV shows an inverted U-shaped relationship, suggesting that moderate leverage may enhance firm value whereas excessive leverage is penalized. By contrast, OPM presents a more complex and interaction-dependent pattern, indicating that the effect of operating profitability cannot be interpreted in isolation. Overall, the color dispersion across the plots confirms the presence of interaction effects, particularly involving OPM and SIZE. These findings suggest that firm value is driven by conditional and nonlinear mechanisms, which are better captured by machine learning models than by a purely linear framework.

## Average SHAP Waterfall Plot for Market:
###  KSE

<img width="837" height="622" alt="image" src="https://github.com/user-attachments/assets/dd9df907-25dd-4e70-a97c-badf7d27d2f4" />

### KOSDAQ

<img width="837" height="622" alt="image" src="https://github.com/user-attachments/assets/ef2c22a4-1956-46b5-a11c-e68a1ce791cd" />

## DML with LGBM

Table 2 reports the Double ML estimates of the causal effect of each financial characteristic on Tobin's Q, after partialling out high-dimensional confounders via cross-fitted LightGBM. The univariate DML results suggest that several variables exhibit statistically significant partial effects on firm value under the current specification, including SIZE, Z_SCORE, LEV, GP, LIQ, CH, and DIV_DUMMY. Among them, SIZE, Z_SCORE, LEV, GP, LIQ, and CH are positively associated with the outcome, whereas DIV_DUMMY shows a negative coefficient. In contrast, OPM, NPM, ROA, EBITDA_TA, FCF_TA, ROE, and TANG fail to achieve statistical significance, indicating that their associations with Tobin's Q observed in the predictive stage are largely attributable to confounding rather than genuine causal pathways. 


Table 2. Double ML estimates the causal effect

| Treatment   |   Theta |   Std.Error |   P-value |
|:------------|--------:|------------:|----------:|
| SIZE        |  0.2167 |      0.0083 |    0.0000 |
| Z_SCORE     |  0.9279 |      0.1201 |    0.0000 |
| LEV         |  0.7017 |      0.1119 |    0.0000 |
| GP          |  0.7128 |      0.0658 |    0.0000 |
| OPM         |  0.0845 |      0.0614 |    0.1691 |
| LIQ         |  0.0104 |      0.0035 |    0.0027 |
| CH          |  0.7959 |      0.1283 |    0.0000 |
| DIV_DUMMY   | -0.3586 |      0.0142 |    0.0000 |
| ROE          | -0.1303 |      0.1687 |    0.4400 |
| EBITDA_TA    |  0.3967 |      0.3357 |    0.2373 |
| NPM          |  0.0017 |      0.0035 |    0.6319 |
| TANG         | -0.0340 |      0.0460 |    0.4608 |
| ROA          |  0.4228 |      0.2900 |    0.1449 |
| GROWTH_SALES |  0.0013 |      0.0027 |    0.6196 |
| FCF_TA       |  0.0442 |      0.0491 |    0.3674 |

## Rolling DML 

<img width="1187" height="690" alt="image" src="https://github.com/user-attachments/assets/5311ffef-6e77-4eb2-99ad-081500aa041f" />

Figure X plots the time-varying causal effects (θ) of significant treatment variables estimated via rolling Double ML from 2012 to 2023, offering three key insights.

**Strengthening effects over time.** GP and CH exhibit the most pronounced upward trajectories, with GP rising from approximately 0.37 to 0.78 and CH surging sharply after 2019 to peak near 0.90 in 2021 before stabilizing. This pattern suggests that gross profitability and cash holdings have become increasingly rewarded by the Korean market over the sample period, potentially reflecting post-COVID investor preference for financially resilient firms.

**Stable and persistent effects.** SIZE maintains a remarkably flat and consistent causal effect around 0.20–0.22 throughout the entire period, confirming that the size premium in Korea is structural rather than cyclical. Z_SCORE similarly trends upward gradually, indicating growing market sensitivity to financial distress risk over time.

**Sign-consistent negative effects.** DIV_DUMMY sustains a persistently negative causal effect across all years, ranging from −0.25 to −0.38, reinforcing the conclusion that dividend payment systematically destroys firm value in the Korean context — a finding robust to time variation. OPM fluctuates around zero with no clear directional trend, consistent with its statistical insignificance in the pooled DML estimates.

**LEV** shows a moderate but gradually increasing positive effect post-2019, potentially linked to low interest rate environments incentivizing leverage as a value-enhancing signal during that period.

## 
| Term       |    coef | std err |       t |     P> |   2.5 % | 97.5 % | Treatment_Base |
| :--------- | ------: | ------: | ------: | -----: | ------: | -----: | :------------- |
| OPM        |  0.0619 |  0.0569 |  1.0876 | 0.2768 | -0.0555 | 0.1735 | OPM            |
| OPM_sq     | -0.0039 |  0.0024 | -1.6444 | 0.1001 | -0.0086 | 0.0007 | OPM            |
| LEV        |  0.4553 |  0.0958 |  4.7512 | 0.0000 |  0.2675 | 0.6432 | LEV            |
| LEV_sq     |  0.1632 |  0.0990 |  1.6476 | 0.0994 | -0.0303 | 0.3573 | LEV            |
| Z_SCORE    |  0.6909 |  0.0870 |  7.9372 | 0.0000 |  0.5252 | 0.8615 | Z_SCORE        |
| Z_SCORE_sq |  0.0601 |  0.0121 |  4.9746 | 0.0000 |  0.0355 | 0.0837 | Z_SCORE        |


# thêm biến 
|              |   Train_R2 |   Train_RMSE |   Train_MAE |   Test_R2 |   Test_RMSE |   Test_MAE |   CV_MSE |
|:-------------|-----------:|-------------:|------------:|----------:|------------:|-----------:|---------:|
| OLS          |     0.1942 |       1.2863 |      0.6323 |    0.0749 |      1.4729 |     0.7725 |   2.7187 |
| Ridge        |     0.1901 |       1.2896 |      0.6324 |    0.0768 |      1.4715 |     0.7727 |   2.4575 |
| LASSO        |     0.1754 |       1.3012 |      0.6370 |    0.0820 |      1.4673 |     0.7683 |   2.5850 |
| ElasticNet   |     0.1842 |       1.2943 |      0.6345 |    0.0824 |      1.4670 |     0.7715 |   2.4641 |
| RandomForest |     0.7761 |       0.6781 |      0.3549 |    0.3092 |      1.2728 |     0.6900 |   2.8684 |
| XGBoost      |     0.6539 |       0.8430 |      0.4738 |    0.3318 |      1.2518 |     0.6745 |   2.1467 |
| LightGBM     |     0.5941 |       0.9129 |      0.4537 |    0.3438 |      1.2406 |     0.6712 |   2.1249 |
