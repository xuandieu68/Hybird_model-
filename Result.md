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

**Table 2 Linear model results** 
|            | Train_R2 | Train_RMSE | Train_MAE | Test_R2 | Test_RMSE | Test_MAE |
| :--------- | -------: | ---------: | --------: | ------: | --------: | -------: |
|            |          |  Panel A : |         Q |         |           |          |
| Ridge      |   0.2440 |     1.2274 |    0.6192 |  0.1354 |    1.5396 |   0.7934 |
| LASSO      |   0.2437 |     1.2276 |    0.6184 |  0.1421 |    1.5337 |   0.7916 |
| ElasticNet |   0.2438 |     1.2276 |    0.6182 |  0.1409 |    1.5347 |   0.7915 |
|            |          |  Panel B : |        MV |         |           |          |
| Ridge      |   0.2453 |     1.2179 |    0.6087 |  0.1378 |    1.5339 |   0.7886 |
| LASSO      |   0.2450 |     1.2181 |    0.6080 |  0.1451 |    1.5275 |   0.7864 |
| ElasticNet |   0.2451 |     1.2181 |    0.6079 |  0.1437 |    1.5286 |   0.7865 |

In table 2, the estimation results show that the linear econometric model group (Ridge, LASSO, Elastic Net) has an out-of-sample coefficient of determination (R²) ranging from 0.135–0.145 and with MV as the dependent variable, it shows a slightly better performance than Tobins’ Q, but still reflects a relatively limited explanatory power. This is partly explained by the complex relationship between financial variables and firm value – especially the threshold, nonlinear, or interaction effects that are not captured by the linear model.
In contrast, the results in both panels of table 3 reveal a consistent pattern across the three nonlinear machine learning models (Random Forest, XGBoost, and LightGBM) in predicting firm value (Q and MV). These models achieve substantially higher explanatory power than the linear group, with R² values on the test set ranging from 0.52 to 0.61—more than four times higher than the linear benchmarks—while RMSE and MAE are considerably lower. Among them, XGBoost delivers the best overall performance, combining high predictive accuracy with strong generalization capability. Its results (R² = 0.606 and 0.609 for Q and MV, respectively) highlight its ability to effectively capture the complex, nonlinear interactions among key firm-level determinants such as leverage, profitability, size, and other fundamentals. Although there are slight signs of overfitting due to the very high training R², the nonlinear models—particularly XGBoost—still successfully reflect the nonlinear and multidimensional nature of the relationship between financial structure and firm value.

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


Based on the performance results from the 2 model groups, LASSO is selected to represent the linear group and XGBoost to represent the nonlinear group to combine in the hybrid model

2.  Hybird model
Two models representing linear and nonlinear were combined with optimal weights based on the explanatory power of each model (R2):

$$
\hat{y}_{\text{hybrid}} = w_1 \hat{y}_{\text{lasso}} + w_2 \hat{y}_{\text{xgb}}, \quad \text{with} \quad w_1 + w_2 = 1
$$


Table 4: Hybird model metric performance

|     | Train_R2 | Test_R2 | Train_RMSE | Test_RMSE | Train_MAE | Test_MAE | Weight_Linear | Weight_Nonlinear |
| --- | -------- | ------- | ---------- | --------- | --------- | -------- | ------------- | ---------------- |
| Q   | 0.8678   | 0.6205  | 0.5132     | 1.0200      | 0.3065    | 0.5866   | 0.19          | 0.81             |
| MV  | 0.8721   | 0.6216  | 0.5014     | 1.0162    | 0.2998    | 0.5794   | 0.19       | 0.81           |

The results in Table 4 indicate that the hybrid model, constructed by combining LASSO and XGBoost based on their R²-weighted contributions (0.19 and 0.81), achieves higher predictive performance than either model alone. The hybrid approach yields a test R² of approximately 0.62 for both firm value measures (Q and MV), with reductions in RMSE and MAE relative to the individual models. Although the improvement in R² is modest, the model exhibits reduced overfitting compared to XGBoost, suggesting that the linear-weighted component effectively regularizes predictions in regions where the relationship between variables is predominantly linear. At the same time, the nonlinear learning capacity inherited from XGBoost is retained, allowing the hybrid model to capture complex interactions among firm characteristics. Overall, these findings confirm the complementarity between econometric interpretability and machine-learning flexibility, supporting the hybrid model as a more balanced and robust framework for predicting firm value.

 Fig 1. Sensitivity analysis
<img width="961" height="313" alt="image" src="https://github.com/user-attachments/assets/b245f2c8-8ead-4cb9-95d7-ebd0ed74160b" />

The study also examined the sensitivity of the results to changes in the weights. The results in Figure 1 show that the R² of the hybrid model is maximized(0.6214 and 0.6228) when the linear weight w_linear = 0.15 for both panels, confirming that this combination provides optimal forecasting performance while maintaining the economic significance of the variables in the explanatory part.

3. Feature Importance Analysis
   
$$
S_{\text{hybrid}} = w_1 \times |\beta_{\text{lasso}}| + w_2 \times I_{\text{xgb}}
$$

 
<figure>
  <img width="1042" height="405" alt="image" src="https://github.com/user-attachments/assets/f1d6f22a-5458-4a8f-a156-a9feb9e3a999" />
  <figcaption style="text-align:center;font-style:italic;">Figure 1. Initial Variable Importance across Machine Learning Models</figcaption>
</figure>

In all three non-linear models in fig 2.1 and 2.2, the significant variables are highly consistent, with most appearing in the top 5 of each model,  firm size (SIZE), profitability (EBITDA_TA, ROA), and financial stability (Z_SCORE) along with DIV_DUMMY being the main predictors of firm value. 
The high degree of consistency across models underscores the robustness of these findings, while slight differences in feature rankings reveal the distinct ways each algorithm captures nonlinear relationships among firm characteristics

<img width="900" height="393" alt="image" src="https://github.com/user-attachments/assets/9242d16b-bbed-4b12-9e95-aeed99c339e1" />


Figure 3. Top 10 Hybrid Feature Importance combining LASSO and XGBoost
<img width="913" height="398" alt="image" src="https://github.com/user-attachments/assets/c8eb545a-3186-40d0-801d-439b5e68ea49" />


Figure 3 illustrates the feature importance of the hybrid model that combines LASSO and XGBoost using R² -based weights. The results show that EBITDA/TA, DIV_DUMMY, ROA, and SIZE remain the most influential variables, consistent with the nonlinear models, indicating that firm fundamentals continue to drive firm value. The appearance of CH and Z_SCORE among the moderately important features reflects the linear model’s contribution in emphasizing liquidity and financial stability. A noticeable difference between the two panels is that LEV is relatively important in the model with Q as the dependent variable but becomes less significant in the MV model, suggesting that leverage affects market-based firm value to a lesser extent.

Figure 4. SHAP value plots
### For Q
<img width="913" height="396" alt="image" src="https://github.com/user-attachments/assets/14507732-9be6-4a19-aaec-adecfad15877" />

### For MV
<table>
  <tr>
    <img width="330" height="400" alt="image" src="https://github.com/user-attachments/assets/f0fffe83-dcd6-420a-aa98-3d1c73542697" />
    <img width="330" height="400" alt="image" src="https://github.com/user-attachments/assets/ee1d3fd8-abad-4d57-a824-e5374c174d36" />
    <img width="330" height="400" alt="image" src="https://github.com/user-attachments/assets/62da4fae-fe2c-4f3a-b419-351f2e598686" />
  </tr>
</table>

 
Figure 4 visualizes the SHAP value distributions for Random Forest, XGBoost, and LightGBM, providing insights into both the direction and magnitude of each variable’s nonlinear contribution to firm value predictions
Looking at the similarity in the distribution of SHAP values of the variables in the models, we observe that the increase in Size is positively correlated with the SHAP value, and similarly, Z_SCORE also shows a positive correlation with the model output. In contrast, profitability variables such as EBITDA_TA, ROA, OPM, NPM, and ROE show red dots both on the left and right of the 0.0 SHAP value. However, the larger overlap on the left indicates that in many cases, firms with high profitability contribute negatively to the model's predicted value. Additionally, the wide spread and mixing of both strong and weak variable values on the right side of the axis suggest that profitability variables might have a non-linear relationship or interact with other variables. LEV shows no clear directional pattern across all three models—rather, it amplifies or alters the impact of other variables depending on the context

---
PanelOLS F.E 

|             | 1         | 2         | 3         | 4         |
| :---------- | :-------- | :-------- | :-------- | :-------- |
| EBITDA_TA   | -3.893*** | -2.467*** | -3.971*** | -2.533*** |
|             | (-7.82)   | (-5.44)   | (-8.03)   | (-5.63)   |
| ROA         | -1.270*** | -0.852*** | -1.232*** | -0.810*** |
|             | (-5.65)   | (-4.00)   | (-5.49)   | (-3.82)   |
| OPM         | -0.100*** | -0.177*** | -0.098*** | -0.176*** |
|             | (-3.34)   | (-6.19)   | (-3.29)   | (-6.18)   |
| NPM         | -0.004    | -0.004    | -0.004    | -0.004    |
|             | (-0.81)   | (-0.89)   | (-0.85)   | (-0.93)   |
| SIZE        | 0.460***  |           | 0.464***  |           |
|             | (19.88)   |           | (20.22)   |           |
| SIZE_RESID  |           | 0.528***  |           | 0.533***  |
|             |           | (21.11)   |           | (21.47)   |
| CH          | 2.620***  | 2.393***  | 2.710***  | 2.481***  |
|             | (11.54)   | (10.82)   | (12.01)   | (11.28)   |
| LEV         | -0.465*** | -0.527*** | -0.122    | -0.184    |
|             | (-3.96)   | (-4.52)   | (-1.04)   | (-1.59)   |
| Z_SCORE     | 0.438***  | 0.294***  | 0.496***  | 0.351***  |
|             | (4.14)    | (2.91)    | (4.81)    | (3.57)    |
| GP          | 1.202***  | 1.164***  | 1.231***  | 1.193***  |
|             | (7.46)    | (7.45)    | (7.72)    | (7.72)    |
| LIQ         | 0.019***  | 0.020***  | 0.018***  | 0.019***  |
|             | (3.12)    | (3.44)    | (3.05)    | (3.39)    |
| Industry FE | Yes       | Yes       | Yes       | Yes       |
| Year FE     | Yes       | Yes       | Yes       | Yes       |
| N           | 9230      | 9230      | 9230      | 9230      |
| Adj. R²     | 0.248     | 0.274     | 0.253     | 0.280     |

T-stats in parentheses. Stars: *** p<0.01.

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















