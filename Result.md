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

In table 2, the estimation results show that the linear econometric model group (Ridge, LASSO, Elastic Net) has an out-of-sample coefficient of determination (R²) ranging from 0.135–0.145 and with MV as the dependent variable, it shows a slightly better performance than Tobins’ Q, but still reflects a relatively limited explanatory power. This is partly explained by the complex relationship between financial variables and firm value – especially the threshold, nonlinear, or interaction effects that are not captured by the linear model.
In contrast, the results in both panels of table 3 reveal a consistent pattern across the three nonlinear machine learning models (Random Forest, XGBoost, and LightGBM) in predicting firm value (Q and MV). These models achieve substantially higher explanatory power than the linear group, with R² values on the test set ranging from 0.52 to 0.61—more than four times higher than the linear benchmarks—while RMSE and MAE are considerably lower. Among them, XGBoost delivers the best overall performance, combining high predictive accuracy with strong generalization capability. Its results (R² = 0.606 and 0.609 for Q and MV, respectively) highlight its ability to effectively capture the complex, nonlinear interactions among key firm-level determinants such as leverage, profitability, size, and other fundamentals. Although there are slight signs of overfitting due to the very high training R², the nonlinear models—particularly XGBoost—still successfully reflect the nonlinear and multidimensional nature of the relationship between financial structure and firm value.

**Non-Linear model results**
|              |   Train_R2 |   Train_RMSE |   Train_MAE |   Test_R2 |   Test_RMSE |   Test_MAE |
|:-------------|-----------:|-------------:|------------:|----------:|------------:|-----------:|
| RandomForest |     0.8158 |       0.6058 |      0.2572 |    0.5245 |      1.1418 |     0.6298 |
| XGBoost      |     0.9059 |       0.4331 |      0.2720 |    0.6060 |      1.0394 |     0.5888 |
| LightGBM     |     0.8484 |       0.5496 |      0.3324 |    0.5949 |      1.0539 |     0.5851 |


---
In contrast, the results in both panels of table 3 reveal a consistent pattern across the three nonlinear machine learning models (Random Forest, XGBoost, and LightGBM) in predicting firm value (Q and MV). These models achieve substantially higher explanatory power than the linear group, with R² values on the test set ranging from 0.52 to 0.61—more than four times higher than the linear benchmarks—while RMSE and MAE are considerably lower. Among them, XGBoost delivers the best overall performance, combining high predictive accuracy with strong generalization capability. Its results (R² = 0.606 and 0.609 for Q and MV, respectively) highlight its ability to effectively capture the complex, nonlinear interactions among key firm-level determinants such as leverage, profitability, size, and other fundamentals. Although there are slight signs of overfitting due to the very high training R², the nonlinear models—particularly XGBoost—still successfully reflect the nonlinear and multidimensional nature of the relationship between financial structure and firm value.

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

The results in Table 4 indicate that the hybrid model, constructed by combining LASSO and XGBoost based on their (R2)-weighted contributions (0.19 and 0.81), achieves higher predictive performance than either model alone. The hybrid approach yields a test (R2) of approximately 0.62 for both firm value measures (Q and MV), with reductions in RMSE and MAE relative to the individual models. Although the improvement in (R2) is modest, the model exhibits reduced overfitting compared to XGBoost, suggesting that the linear-weighted component effectively regularizes predictions in regions where the relationship between variables is predominantly linear. At the same time, the nonlinear learning capacity inherited from XGBoost is retained, allowing the hybrid model to capture complex interactions among firm characteristics. Overall, these findings confirm the complementarity between econometric interpretability and machine-learning flexibility, supporting the hybrid model as a more balanced and robust framework for predicting firm value.

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

In all three non-linear models in fig 2.1 and 2.2, the significant variables are highly consistent, with most appearing in the top 5 of each model,  firm size (SIZE), profitability (EBITDA_TA, ROA), and financial stability (Z_SCORE) along with DIV_DUMMY being the main predictors of firm value. These results imply that larger, more profitable, and financially stable firms tend to achieve higher valuations, reflecting both operational efficiency and lower risk exposure. Meanwhile, financial policy variables such as leverage (LEV) and dividend policy show secondary but notable effects, suggesting that firms’ financing and payout decisions still convey information about firm quality. The high degree of consistency across models underscores the robustness of these findings, while slight differences in feature rankings reveal the distinct ways each algorithm captures nonlinear relationships among firm characteristics

<img width="900" height="393" alt="image" src="https://github.com/user-attachments/assets/9242d16b-bbed-4b12-9e95-aeed99c339e1" />


Figure 3. Top 10 Hybrid Feature Importance combining LASSO and XGBoost
<img width="913" height="398" alt="image" src="https://github.com/user-attachments/assets/c8eb545a-3186-40d0-801d-439b5e68ea49" />


Figure 3 illustrates the feature importance of the hybrid model that combines LASSO and XGBoost using R² -based weights. The results show that EBITDA/TA, DIV_DUMMY, ROA, and SIZE remain the most influential variables, consistent with the nonlinear models, indicating that firm fundamentals continue to drive firm value. The appearance of CH and Z_SCORE among the moderately important features reflects the linear model’s contribution in emphasizing liquidity and financial stability. A noticeable difference between the two panels is that LEV is relatively important in the model with Q as the dependent variable but becomes less significant in the MV model, suggesting that leverage affects market-based firm value to a lesser extent.

Figure 4. SHAP value plots
<img width="913" height="396" alt="image" src="https://github.com/user-attachments/assets/14507732-9be6-4a19-aaec-adecfad15877" />

Figure 4 visualizes the SHAP value distributions for Random Forest, XGBoost, and LightGBM, providing insights into both the direction and magnitude of each variable’s nonlinear contribution to firm value predictions

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




