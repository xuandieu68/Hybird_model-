#  **“Hybrid Approach: Machine Learning–Enhanced Traditional Regression in Explaining Firm Value”**

## ABSTRACT 

This study develops a Hybrid Econometric–Machine Learning Framework to analyze and predict firm value (FV) by combining the interpretability of traditional regression models with the predictive power of nonlinear machine learning algorithms.
Using a balanced panel of non-financial firms from 2006–2024, the study first applies linear models — Linear, Ridge, LASSO, and Elastic Net — to identify key determinants of firm value such as profitability, leverage, dividend policy, and firm size. Subsequently, XGBoost, Random Forest, and LightGBM are employed to capture nonlinear patterns and variable interactions.
Empirical results indicate that LASSO and XGBoost outperform other models, and a weighted hybrid model combining the two (weights: 0.19 and 0.81) achieves the best predictive performance (R² = 0.62).
Feature importance and SHAP analysis highlight EBITDA, dividend policy, ROA, and firm size as the most influential determinants of firm value.
The findings demonstrate that integrating econometric and machine learning approaches enhances both predictive accuracy and economic interpretability, offering a promising methodological direction for contemporary corporate finance research.

---
 **Table 1: Correlation table**

|              | Q        | MV       | GP       | FCF_TA   | TANG     | CH       | ROA     | ROE      | NPM      | LIQ      | Z_SCORE   | LEV      | SIZE    | EBITDA_TA   | OPM     | GROWTH_SALES   | DIV_DUMMY   |
|:-------------|:---------|:---------|:---------|:---------|:---------|:---------|:--------|:---------|:---------|:---------|:----------|:---------|:--------|:------------|:--------|:---------------|:------------|
| Q            | 1.0***   |          |          |          |          |          |         |          |          |          |           |          |         |             |         |                |             |
| MV           | 1.0***   | 1.0***   |          |          |          |          |         |          |          |          |           |          |         |             |         |                |             |
| GP           | 0.09***  | 0.1***   | 1.0***   |          |          |          |         |          |          |          |           |          |         |             |         |                |             |
| FCF_TA       | -0.16*** | -0.15*** | -0.01    | 1.0***   |          |          |         |          |          |          |           |          |         |             |         |                |             |
| TANG         | -0.1***  | -0.1***  | -0.08*** | 0.3***   | 1.0***   |          |         |          |          |          |           |          |         |             |         |                |             |
| CH           | 0.19***  | 0.19***  | 0.13***  | -0.37*** | -0.24*** | 1.0***   |         |          |          |          |           |          |         |             |         |                |             |
| ROA          | -0.12*** | -0.13*** | 0.21***  | 0.42***  | 0.05***  | 0.03***  | 1.0***  |          |          |          |           |          |         |             |         |                |             |
| ROE          | -0.08*** | -0.08*** | 0.24***  | 0.22***  | 0.02***  | 0.02***  | 0.54*** | 1.0***   |          |          |           |          |         |             |         |                |             |
| NPM          | -0.1***  | -0.1***  | 0.05***  | 0.2***   | 0.01     | -0.03*** | 0.38*** | 0.15***  | 1.0***   |          |           |          |         |             |         |                |             |
| LIQ          | 0.12***  | 0.1***   | -0.03*** | -0.27*** | -0.19*** | 0.14***  | 0.05*** | 0.01*    | 0.01**   | 1.0***   |           |          |         |             |         |                |             |
| Z_SCORE      | -0.02**  | 0.02**   | 0.35***  | 0.01*    | 0.03***  | 0.01     | 0.16*** | 0.17***  | 0.05***  | -0.08*** | 1.0***    |          |         |             |         |                |             |
| LEV          | -0.1***  | -0.06*** | -0.07*** | 0.37***  | 0.25***  | -0.21*** | -0.2*** | -0.18*** | -0.06*** | -0.41*** | 0.18***   | 1.0***   |         |             |         |                |             |
| SIZE         | 0.27***  | 0.27***  | 0.07***  | 0.18***  | -0.04*** | -0.01*   | 0.22*** | 0.18***  | 0.06***  | 0.03***  | -0.04***  | -0.09*** | 1.0***  |             |         |                |             |
| EBITDA_TA    | -0.11*** | -0.11*** | 0.41***  | 0.34***  | 0.12***  | 0.02**   | 0.64*** | 0.69***  | 0.2***   | -0.03*** | 0.29***   | -0.12*** | 0.23*** | 1.0***      |         |                |             |
| OPM          | -0.23*** | -0.23*** | 0.14***  | 0.23***  | 0.03***  | -0.09*** | 0.33*** | 0.33***  | 0.35***  | -0.05*** | 0.13***   | -0.03*** | 0.04*** | 0.47***     | 1.0***  |                |             |
| GROWTH_SALES | 0.01*    | 0.01*    | 0.02**   | 0.02***  | -0.02**  | 0.0      | 0.02*** | 0.03***  | 0.01*    | 0.0      | 0.01      | -0.0     | 0.02*** | 0.04***     | 0.02*** | 1.0***         |             |
| DIV_DUMMY    | -0.18*** | -0.18*** | 0.13***  | 0.11***  | 0.07***  | -0.07*** | 0.31*** | 0.24***  | 0.11***  | 0.04***  | 0.15***   | -0.2***  | 0.27*** | 0.35***     | 0.19*** | -0.02***       | 1.0***      |

---

**Linear model results** 
|            |   Train_R2 |   Train_RMSE |   Train_MAE |   Test_R2 |   Test_RMSE |   Test_MAE |
|:-----------|-----------:|-------------:|------------:|----------:|------------:|-----------:|
| Linear     |     0.2440 |       1.2274 |      0.6192 |    0.1354 |      1.5397 |     0.7934 |
| Ridge      |     0.2440 |       1.2274 |      0.6192 |    0.1354 |      1.5396 |     0.7934 |
| LASSO      |     0.2437 |       1.2276 |      0.6184 |    0.1421 |      1.5337 |     0.7916 |
| ElasticNet |     0.2438 |       1.2276 |      0.6182 |    0.1409 |      1.5347 |     0.7915 |

---

**Non-Linear model results**
|              |   Train_R2 |   Train_RMSE |   Train_MAE |   Test_R2 |   Test_RMSE |   Test_MAE |
|:-------------|-----------:|-------------:|------------:|----------:|------------:|-----------:|
| RandomForest |     0.8158 |       0.6058 |      0.2572 |    0.5245 |      1.1418 |     0.6298 |
| XGBoost      |     0.9059 |       0.4331 |      0.2720 |    0.6060 |      1.0394 |     0.5888 |
| LightGBM     |     0.8484 |       0.5496 |      0.3324 |    0.5949 |      1.0539 |     0.5851 |


---

<img width="551" height="379" alt="Figure 2025-10-30 202703" src="https://github.com/user-attachments/assets/8bd3f73a-65f5-4ff2-9247-d8eae68a0746" />

<img width="551" height="379" alt="Figure 2025-10-30 202810" src="https://github.com/user-attachments/assets/5e12971e-5393-4618-adaa-a392f716df4a" />

<img width="551" height="379" alt="Figure 2025-10-30 202824" src="https://github.com/user-attachments/assets/01c1a845-f9f4-4103-b731-ace9572415a1" />

---

<img width="552" height="533" alt="Figure 2025-10-30 202853" src="https://github.com/user-attachments/assets/3f8ab649-0b2d-4b5b-89d9-2dfd4d5f5630" />

<img width="552" height="533" alt="Figure 2025-10-30 202841" src="https://github.com/user-attachments/assets/4bd83167-3785-4079-81ba-584f11b2c2e7" />

<img width="550" height="533" alt="Figure 2025-10-30 203059" src="https://github.com/user-attachments/assets/4e98b304-1e97-4f46-8349-dba7e6573b9b" />


---



Hybird model metric 
| Train_R2 | Test_R2 | Train_RMSE | Test_RMSE | Train_MAE | Test_MAE | Weight_Linear | Weight_Nonlinear |
| -------: | ------: | ---------: | --------: | --------: | -------: | ------------: | ---------------: |
| 0.867841 | 0.62049 |    0.51319 |   1.02006 |  0.306461 | 0.586578 |          0.19 |             0.81 |





<img width="550" height="400" alt="image" src="https://github.com/user-attachments/assets/af1daf59-230d-4951-93b7-128da46ff8ae" />



---
PanelOLS F.E 

| Variable         | (1) Q                                  | (2) Q                                  | (3)   MV                               | (4) MV                                 |
| ---------------- | -------------------------------------- | -------------------------------------- | -------------------------------------- | -------------------------------------- |
| EBITDA_TA        | -6.1555***<br><small>(-14.646)</small> |                                        | -6.1772***<br><small>(-14.758)</small> |                                        |
| ROA              |                                        | -3.0644***<br><small>(-11.641)</small> |                                        | -3.0515***<br><small>(-11.599)</small> |
| SIZE             | 0.4687***<br><small>(20.444)</small>   | 0.4335***<br><small>(19.995)</small>   | 0.4725***<br><small>(20.779)</small>   | 0.4367***<br><small>(20.293)</small>   |
| CH               | 2.7901***<br><small>(12.546)</small>   | 2.8913***<br><small>(12.699)</small>   | 2.8775***<br><small>(13.011)</small>   | 2.9818***<br><small>(13.166)</small>   |
| LEV              | -0.3453***<br><small>(-2.8911)</small> | -0.3757***<br><small>(-2.9666)</small> | -0.0050<br><small>(-0.0423)</small>    | -0.0304<br><small>(-0.2420)</small>    |
| Z_SCORE          | 0.4389***<br><small>(4.2093)</small>   | 0.3249***<br><small>(3.0264)</small>   | 0.4972***<br><small>(4.8858)</small>   | 0.3814***<br><small>(3.6349)</small>   |
| GP               | 1.3265***<br><small>(8.0544)</small>   | 0.6373***<br><small>(3.8814)</small>   | 1.3522***<br><small>(8.2945)</small>   | 0.6568***<br><small>(4.0598)</small>   |
| LIQ              | 0.0182***<br><small>(2.9983)</small>   | 0.0250***<br><small>(3.6269)</small>   | 0.0174***<br><small>(2.9347)</small>   | 0.0242***<br><small>(3.5933)</small>   |
| No. Observations | 9,230                                  | 9,230                                  | 9,230                                  | 9,230                                  |
| R-squared        | 0.2352                                 | 0.2121                                 | 0.2407                                 | 0.2162                                 |
| Effects          | IND, Time                           | IND, Time                           | IND, Time                           | IND, Time                           |

T-stats in parentheses. Stars: *** p<0.01.



