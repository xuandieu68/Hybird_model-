

# CHAPTER 1. INTRODUCTION

## 1.1 Research Background

* Vai trò của **Firm Value** trong tài chính doanh nghiệp
* Các lý thuyết nền tảng:

  * Trade-off theory
  * Pecking order theory
  * Signaling theory
* Vấn đề của hồi quy tuyến tính:

  * Assumption of linearity
  * Model misspecification
  * Omitted nonlinearity bias
* Sự phát triển của Machine Learning trong corporate finance
* Tension giữa:

  * Predictive accuracy
  * Economic interpretability
* Khoảng trống nghiên cứu:

  > Thiếu framework tích hợp prediction + interpretability + causal validation



## 1.2 Research Gap

Tổng hợp từ literature:

1. Kết quả trái chiều về tác động của leverage
2. Profitability có thể có ngưỡng (threshold effect)
3. ML cải thiện prediction nhưng thiếu causal interpretation
4. Econometric models giải thích được nhưng dự báo yếu

→ **Chưa có framework hybrid kết hợp:**

* Linear interpretability
* Nonlinear flexibility
* Causal validation (DML)
* Panel robustness

---

## 1.3 Research Objectives

1. Xác định các yếu tố quyết định Firm Value
2. So sánh hiệu suất linear vs nonlinear models
3. Xây dựng Hybrid Econometric–ML framework
4. Kiểm định causal effect bằng Double Machine Learning
5. Kiểm chứng lại bằng Panel Fixed Effects

---

## 1.4 Research Questions

1. Linear models có đủ để giải thích Firm Value không?
2. ML có cải thiện predictive performance không?
3. Hybrid model có vượt trội hơn mô hình đơn lẻ không?
4. Các biến quan trọng có ổn định dưới góc nhìn causal không?

---

## 1.5 Research Hypotheses


H_1: Firm value exhibits nonlinear response to profitability



H_2: Machine learning models outperform linear models in predictive accuracy



H_3: Hybrid models outperform standalone models



H_4: Key determinants retain partial causal effects under DML


---

## 1.6 Contributions

### (1) Methodological

Đề xuất Hybrid ML–Econometric Framework

### (2) Empirical

So sánh toàn diện 7 mô hình

### (3) Interpretability

Kết hợp SHAP + FE + DML

### (4) Practical

Cải thiện mô hình định giá doanh nghiệp

---

# CHAPTER 2. LITERATURE REVIEW AND HYPOTHESIS DEVELOPMENT

---

## 2.1 Theoretical Foundations of Firm Value

* Modigliani & Miller
* Trade-off theory
* Pecking order theory
* Signaling theory

→ Liên kết với expected signs

---

## 2.2 Determinants of Firm Value

### 2.2.1 Firm Size

* Economies of scale vs bureaucracy cost
* Mixed empirical evidence

### 2.2.2 Profitability

* ROA, EBITDA
* Threshold & diminishing return hypothesis

### 2.2.3 Leverage

* Tax shield vs bankruptcy risk
* Nonlinear trade-off

### 2.2.4 Dividend Policy

* Signaling vs free cash flow theory

### 2.2.5 Financial Distress

* Z-score

---

## 2.3 Machine Learning in Corporate Finance

* Random Forest
* XGBoost
* LightGBM
* Evidence of superior prediction
* Limitation: lack of causal structure

---

## 2.4 Hybrid Modeling Literature

* Ensemble learning
* Model stacking
* Weighted averaging
* Bias–variance tradeoff

---

## 2.5 Causal Inference and Double Machine Learning

### Endogeneity problem

[
Y = \beta D + X'\gamma + \varepsilon
]

Vấn đề: (E[D\varepsilon] \neq 0)

### DML Orthogonalization:

$
\tilde{Y} = Y - \hat{m}(X)
$
$
\tilde{D} = D - \hat{g}(X)
$

Ước lượng:

$
\hat{\beta} = \frac{\sum \tilde{D}\tilde{Y}}{\sum \tilde{D}^2}
$

---

## 2.6 Research Gap Summary

* Linear models thiếu nonlinear flexibility
* ML thiếu interpretability
* Chưa có hybrid + DML validation

---

# CHAPTER 3. DATA AND METHODOLOGY

---

## 3.1 Data Description

* Sample: non-financial firms
* Period: 2006–2024
* Panel structure: unbalanced
* Data cleaning
* Winsorization (1%–99%)

---

## 3.2 Variable Definition

### Dependent Variable

[
Tobin's\ Q = \frac{Market\ Value}{Book\ Value}
]

### Independent Variables (Full Table)

| Variable | Definition        | Expected Sign |
| -------- | ----------------- | ------------- |
| Size     | ln(Total Assets)  | +/-           |
| ROA      | Net income/Assets | +             |
| EBITDA   | EBITDA/Assets     | + (nonlinear) |
| Leverage | Total debt/Assets | +/-           |
| Dividend | Dividend dummy    | +             |
| Z-score  | Altman index      | +             |

Lag structure: (X_{t-1})

---

## 3.3 Multicollinearity Handling

* Correlation matrix
* VIF test
* Creation of residual size:

[
Size_{resid} = Size - \hat{Size}(Profitability)
]

Rationale:

* Dùng cho linear models
* Không dùng cho tree models

---

## 3.4 Linear Models

1. OLS
2. Ridge
3. LASSO
4. Elastic Net

Regularization:

[
\min_{\beta} \sum (y - X\beta)^2 + \lambda ||\beta||_1
]

---

## 3.5 Nonlinear Models

1. Random Forest
2. XGBoost
3. LightGBM

Tree-based ensemble logic

Bias–variance decomposition

---

## 3.6 Hybrid Econometric–Machine Learning Framework

### Weighted Prediction:

[
\hat{Y}*{Hybrid} = w \hat{Y}*{ML} + (1-w)\hat{Y}_{Linear}
]

### Weight Optimization:

[
w^* = \arg\min_w MSE
]

* Grid search (0–1)
* 5-fold cross validation

### Hybrid Feature Score

Weighted importance index

---

## 3.7 Double Machine Learning

### Treatment variables:

* EBITDA
* Leverage
* Dividend

### Outcome:

Tobin’s Q

Cross-fitting procedure

Partial treatment effect estimation

---

## 3.8 Panel Fixed Effects Validation

[
Y_{it} = \alpha_i + \beta X_{it} + \delta_t + \varepsilon_{it}
]

* Firm FE
* Year FE
* Clustered SE

---

# CHAPTER 4. EMPIRICAL RESULTS AND DISCUSSION

---

## 4.1 Descriptive Statistics

* Distribution
* Skewness
* Economic meaning

---

## 4.2 Correlation & VIF

* Multicollinearity diagnosis
* Justification of size_resid

---

## 4.3 Linear Model Performance

* R²
* RMSE
* MAE
* Why LASSO dominates

---

## 4.4 Nonlinear Model Performance

* RF vs XGB vs LGBM
* XGBoost dominance
* Overfitting control

---

## 4.5 Hybrid Model Results

* Optimal weight (e.g., 0.19–0.81)
* R² improvement
* Bias–variance explanation

---

## 4.6 SHAP and Nonlinear Interpretation

* Global importance
* Nonlinear EBITDA threshold
* Interaction effect

---

## 4.7 Double Machine Learning Results

* Partial treatment effects
* Comparison with OLS
* Shrinkage vs FE
* Predictive vs causal distinction

---

## 4.8 Panel FE Validation

* Coefficient stability
* Statistical significance
* Economic magnitude

---

## 4.9 Sensitivity & Robustness

* Alternative weights
* Subsample analysis
* Alternative dependent variable
* Different lag structures

---

# CHAPTER 5. CONCLUSION AND IMPLICATIONS

---

## 5.1 Key Findings

* ML > Linear
* Hybrid > ML
* Nonlinearity exists
* DML confirms partial causal effects

---

## 5.2 Theoretical Implications

* Firm value is nonlinear
* Traditional linear regression incomplete
* Need hybrid modeling

---

## 5.3 Managerial Implications

* EBITDA threshold management
* Dividend signaling
* Optimal leverage zone

---

## 5.4 Methodological Implications

* Hybrid framework replicable
* Combine interpretability + prediction

---

## 5.5 Limitations

* Country-specific
* No macro variables
* Survivorship bias

---

## 5.6 Future Research

* Dynamic panel ML
* Deep learning
* Cross-country comparison
* Time-varying hybrid weights

---

# 📊 APPENDICES

* Full variable table
* Hyperparameters
* Additional robustness
* Code snippets
* SHAP plots



---

# 📘 CHAPTER 1

# INTRODUCTION


# 1.1 Research Background

## 1.1.1 The Central Role of Firm Value in Corporate Finance

Firm value represents one of the most fundamental constructs in corporate finance. It reflects the market’s assessment of a firm’s future profitability, risk exposure, growth opportunities, and managerial efficiency. From both theoretical and practical perspectives, maximizing firm value is widely regarded as the primary objective of corporate decision-making.

Within classical financial theory, firm value is shaped by capital structure decisions, dividend policy, investment efficiency, and operating performance. The foundational propositions of capital structure theory suggest that firm value depends critically on how firms balance tax benefits and financial distress costs. Subsequent theoretical developments further incorporate information asymmetry, agency conflicts, and signaling mechanisms into the valuation process.

Empirically, firm value is often proxied by Tobin’s Q, defined as:

$$Tobin's\ Q = \frac{Market\ Value\ of\ Equity + Book\ Value\ of\ Debt}{Book\ Value\ of\ Assets}$$


Tobin’s Q captures market expectations about future performance and intangible growth opportunities. Unlike accounting-based measures, it embeds forward-looking valuation signals, making it particularly suitable for examining the determinants of corporate valuation.

Understanding what drives firm value remains one of the most enduring research questions in corporate finance.


## 1.1.2 Traditional Econometric Approaches and Their Limitations

The majority of empirical studies investigating firm value rely on linear regression frameworks, typically estimated via Ordinary Least Squares (OLS) or panel fixed-effects models:

$$Y_{it} = \alpha + \beta X_{it} + \varepsilon_{it}$$

While these models offer interpretability and clear economic meaning, they rest upon several restrictive assumptions:

1. Linearity between independent variables and firm value.
2. Additive functional form.
3. Homogeneous marginal effects across firms.
4. Limited interaction modeling.

However, real-world corporate finance decisions are rarely linear. For example:

* The effect of leverage may be positive at low levels (tax shield benefits) but negative beyond an optimal threshold (financial distress risk).
* Profitability may exhibit diminishing marginal returns.
* Firm size may reflect economies of scale but also bureaucratic inefficiency at large scales.

If the true data-generating process is nonlinear, imposing a linear functional form introduces model misspecification bias, potentially distorting both inference and prediction.

Moreover, multicollinearity among financial ratios often weakens coefficient stability, reducing the reliability of interpretation in traditional regression models.

Thus, while linear econometric models remain valuable for inference, their predictive power and structural flexibility may be insufficient for capturing complex financial relationships.



## 1.1.3 The Emergence of Machine Learning in Corporate Finance

Recent years have witnessed a rapid expansion of machine learning (ML) techniques in financial research. Tree-based ensemble methods such as Random Forest, Gradient Boosting, and Extreme Gradient Boosting (XGBoost) have demonstrated superior predictive performance across various domains, including:

* Bankruptcy prediction
* Asset pricing
* Credit risk modeling
* Earnings forecasting

Machine learning models offer several advantages:

* Automatic detection of nonlinear relationships
* Implicit modeling of high-order interactions
* Robustness to multicollinearity
* Flexible functional forms

Unlike linear models, tree-based algorithms partition the feature space recursively, allowing heterogeneous marginal effects across subgroups of firms. This flexibility enhances predictive accuracy, especially when the underlying relationships are complex and nonlinear.

However, this predictive superiority comes at a cost.



## 1.1.4 The Interpretability–Prediction Trade-off

Despite their strong predictive capabilities, machine learning models are frequently criticized for lacking interpretability and economic transparency. In corporate finance, understanding *why* a variable influences firm value is often as important as predicting the value itself.

Traditional regression models provide explicit coefficients:


$$\frac{\partial Y}{\partial X_j} = \beta_j$$


In contrast, tree-based models do not generate easily interpretable marginal effects. Although modern tools such as SHAP values allow post-hoc interpretation, they do not inherently solve the issue of causal identification.

Furthermore, machine learning algorithms focus primarily on minimizing prediction error rather than estimating structural parameters. Consequently:

* They do not automatically address endogeneity.
* They do not provide causal interpretation.
* They may overfit without proper cross-validation.

This creates a methodological tension:

> Econometric models offer interpretability but limited flexibility.
> Machine learning models offer flexibility but limited causal structure.



## 1.1.5 Research Gap

The existing literature reveals three major gaps:

### Gap 1: Inconsistent Empirical Findings

Prior studies often report mixed evidence regarding the determinants of firm value, particularly for leverage and dividend policy. These inconsistencies may stem from linear specification bias.

### Gap 2: Lack of Integrated Framework

Most studies either:

* Use econometric models exclusively, or
* Apply machine learning purely for prediction.

Very few studies integrate both paradigms into a unified framework that combines interpretability and predictive power.

### Gap 3: Limited Causal Validation

Even when machine learning improves predictive accuracy, its economic interpretation remains uncertain without causal validation. Endogeneity—arising from omitted variables, reverse causality, or measurement error—remains a fundamental challenge in corporate finance research.

Therefore, there is a need for a comprehensive framework that:

1. Captures nonlinear relationships.
2. Maintains interpretability.
3. Validates key relationships using causal inference techniques.
4. Ensures robustness in panel data settings.

This thesis addresses these gaps by proposing a Hybrid Econometric–Machine Learning framework, supplemented by Double Machine Learning (DML) and panel fixed-effects validation.



# 1.2 Research Gap and Motivation

## 1.2.1 Why Linear Models May Be Incomplete

Theoretically, the relationship between financial decisions and firm value is inherently nonlinear.

For example:

* The trade-off theory implies an inverted-U relationship between leverage and firm value.
* Signaling theory suggests that dividend announcements may only affect valuation under information asymmetry.
* Profitability may generate increasing returns up to a competitive threshold, after which marginal gains decline.

If these mechanisms hold, then the true functional form may be:


$$Y = f(X) + \varepsilon$$


where (f(\cdot)) is nonlinear and potentially includes interactions.

Imposing a linear approximation:


$$Y = X\beta + \varepsilon$$


may systematically underestimate threshold effects and interaction structures.



## 1.2.2 Why Machine Learning Alone Is Not Sufficient

Although machine learning models can approximate (f(X)) flexibly, they do not:

* Guarantee economic interpretability,
* Address omitted variable bias,
* Provide structural parameters,
* Distinguish correlation from causation.

Hence, relying exclusively on machine learning may lead to high predictive accuracy but weak economic insights.


## 1.2.3 Motivation for a Hybrid Framework

The central motivation of this study is to reconcile:

* Econometric rigor,
* Predictive flexibility,
* Causal identification.

The proposed hybrid prediction model is defined as:

$$\hat{Y}*{Hybrid} = w \hat{Y}*{ML} + (1-w)\hat{Y}_{Linear}$$

where $(w \in [0,1])$ is optimized via cross-validation to minimize mean squared error.

This approach leverages:

* The interpretability of linear models,
* The nonlinear flexibility of machine learning,
* The bias–variance trade-off inherent in ensemble learning.

To further ensure robustness, the study incorporates Double Machine Learning (DML) to estimate partial treatment effects while controlling for high-dimensional covariates.





# 1.3 Research Objectives

Building upon the identified research gaps, this thesis aims to develop an integrated analytical framework that combines predictive accuracy, interpretability, and causal validation in examining the determinants of firm value.

Specifically, the study pursues the following objectives:

### Objective 1:

To identify the key determinants of firm value using both traditional econometric and machine learning approaches.

### Objective 2:

To systematically compare the predictive performance of linear and nonlinear models in explaining variations in Tobin’s Q.

### Objective 3:

To construct and evaluate a Hybrid Econometric–Machine Learning framework that integrates the strengths of both paradigms.

### Objective 4:

To estimate partial causal effects of selected financial variables using Double Machine Learning (DML) in order to mitigate endogeneity bias.

### Objective 5:

To validate empirical findings using panel fixed-effects models with clustered standard errors to ensure robustness in a panel data setting.

Collectively, these objectives aim to bridge the methodological divide between econometrics and machine learning in corporate finance research.


# 1.4 Research Questions

In line with the objectives above, the study addresses the following research questions:

### RQ1:

Are traditional linear regression models sufficient to explain variations in firm value?

### RQ2:

Do machine learning models significantly improve predictive performance relative to linear econometric models?

### RQ3:

Does a hybrid modeling approach outperform standalone linear or nonlinear models?

### RQ4:

Are the most important predictive variables also economically and causally significant under a Double Machine Learning framework?

These questions reflect the core tension between prediction and inference that characterizes modern empirical finance.



# 1.5 Research Hypotheses

Grounded in theoretical considerations and prior empirical findings, this study proposes the following hypotheses:


### H1: Nonlinearity Hypothesis

[
H_1: \text{Firm value exhibits nonlinear responses to key financial determinants.}
]

This hypothesis is motivated by theoretical models suggesting threshold effects and diminishing marginal returns, particularly in leverage and profitability.


### H2: Predictive Superiority of Machine Learning

[
H_2: \text{Machine learning models outperform linear models in predictive accuracy.}
]

This hypothesis reflects the superior flexibility of ensemble tree-based algorithms in approximating complex functional forms.


### H3: Hybrid Model Superiority

[
H_3: \text{The hybrid econometric–machine learning model outperforms standalone models.}
]

This hypothesis builds on ensemble learning theory and the bias–variance trade-off:

$$
MSE = Bias^2 + Variance + \sigma^2
$$

By combining models with different structural assumptions, the hybrid approach is expected to reduce overall prediction error.

---

### H4: Partial Causal Stability


$$H_4: \text{Key financial determinants retain statistically significant partial causal effects under Double Machine Learning estimation.}$$


This hypothesis tests whether predictive importance aligns with economically meaningful causal relationships.

# 1.6 Contributions of the Study

This thesis contributes to the literature in four major dimensions.


## 1.6.1 Methodological Contribution

The primary contribution lies in proposing a **Hybrid Econometric–Machine Learning Framework** that:

* Combines linear interpretability and nonlinear flexibility.
* Optimizes prediction weights through cross-validation.
* Integrates causal validation using Double Machine Learning.
* Employs panel fixed-effects models for robustness.

To the best of current knowledge, few studies simultaneously incorporate prediction, interpretation, and causal inference in the context of firm valuation.



## 1.6.2 Empirical Contribution

Empirically, the study provides a comprehensive comparison of:

* OLS
* Ridge
* LASSO
* Elastic Net
* Random Forest
* XGBoost
* LightGBM

across a long panel dataset covering the period 2006–2024.

This systematic evaluation contributes new evidence on the relative strengths of linear and nonlinear methods in corporate finance.



## 1.6.3 Interpretability Contribution

By employing SHAP-based feature importance alongside panel fixed-effects coefficients and DML treatment effects, the study triangulates variable significance from three perspectives:

1. Predictive importance
2. Structural association
3. Partial causal effect

This multi-layer interpretation enhances economic transparency in machine learning applications.



## 1.6.4 Practical Contribution

For practitioners, the findings offer:

* Improved firm valuation modeling.
* Identification of nonlinear thresholds (e.g., profitability levels).
* Insights into optimal leverage ranges.
* Evidence on dividend signaling effects.

These implications are relevant for corporate managers, investors, and financial analysts.



# 1.7 Structure of the Thesis

The remainder of the thesis is organized as follows:

* **Chapter 2** reviews the theoretical foundations of firm value, synthesizes empirical evidence on its determinants, discusses the application of machine learning in corporate finance, and develops research hypotheses.

* **Chapter 3** describes the dataset, variable construction, and econometric methodology, including linear models, nonlinear ensemble models, the hybrid framework, Double Machine Learning, and panel fixed-effects validation.

* **Chapter 4** presents empirical results, including descriptive statistics, model performance comparison, SHAP analysis, causal estimation results, and robustness checks.

* **Chapter 5** concludes the study by summarizing key findings, discussing theoretical and managerial implications, acknowledging limitations, and proposing directions for future research.



This chapter establishes the conceptual and methodological foundation of the thesis. It identifies critical limitations in traditional linear approaches to firm valuation, highlights the growing role of machine learning in financial prediction, and proposes a hybrid framework that integrates predictive flexibility with econometric rigor and causal validation.

The following chapter situates the study within the broader literature and develops the theoretical arguments underlying the empirical analysis.



# CHAPTER 2

# LITERATURE REVIEW AND HYPOTHESIS DEVELOPMENT

# 2.1 Theoretical Foundations of Firm Value

## 2.1.1 Shareholder Value Maximization and Firm Valuation

The concept of firm value lies at the heart of modern corporate finance. The dominant paradigm asserts that firms operate to maximize shareholder wealth, typically reflected in market capitalization or forward-looking valuation metrics such as Tobin’s Q. Under the discounted cash flow framework, firm value equals the present value of expected future cash flows:

$$V = \sum_{t=1}^{\infty} \frac{E(CF_t)}{(1+r)^t}$$


where (CF_t) represents expected future cash flows and (r) the required rate of return.

In frictionless markets, capital structure would not affect firm value (Modigliani & Miller, 1958). However, once market imperfections such as taxation, bankruptcy risk, and information asymmetry are introduced, financing decisions become value-relevant (Modigliani & Miller, 1963).

Tobin’s Q, defined as the ratio of market value to replacement cost of assets, is frequently used as an empirical proxy for firm value (Tobin, 1969). It captures growth opportunities and intangible assets that accounting measures may not fully reflect (Chung & Pruitt, 1994).


## 2.1.2 Capital Structure Theories

### Trade-Off Theory

The trade-off theory posits that firms balance the tax benefits of debt against expected bankruptcy and financial distress costs (Kraus & Litzenberger, 1973). Under this framework, leverage increases firm value through interest tax shields but reduces value when distress costs become substantial.

This implies a nonlinear relationship between leverage and firm value, potentially exhibiting an inverted-U shape.

Empirical evidence provides partial support for this theory. For example, Fama and French (2002) find that leverage decisions reflect trade-offs between tax benefits and distress risk, although the optimal point varies across firms.

### Pecking Order Theory

The pecking order theory (Myers & Majluf, 1984) suggests firms prefer internal financing over external financing due to information asymmetry. Debt issuance signals less adverse information than equity issuance.

Under this framework, leverage does not necessarily reflect optimal capital structure but rather financing constraints. This perspective may explain inconsistent empirical findings regarding leverage and firm value (Frank & Goyal, 2009).


## 2.1.3 Agency Theory

Agency theory introduces conflicts between managers and shareholders (Jensen & Meckling, 1976). Managers may pursue private benefits at the expense of shareholder value.

Debt can serve as a disciplinary mechanism by reducing free cash flow available for overinvestment (Jensen, 1986). However, excessive leverage may encourage risk-shifting behavior.

Therefore, agency theory predicts complex, potentially nonlinear effects of leverage and cash flow on firm value.


## 2.1.4 Signaling Theory and Dividend Policy

Dividend policy is closely linked to firm valuation through signaling mechanisms. Under asymmetric information, dividend changes may convey managerial expectations regarding future earnings (Bhattacharya, 1979; Miller & Rock, 1985).

Empirical studies often document positive market reactions to dividend increases, although results vary across institutional contexts (Denis & Osobov, 2008).

Dividend effects may depend on firm profitability, growth opportunities, and capital market development, suggesting heterogeneous valuation impacts.

## 2.1.5 Implications for Empirical Modeling

Collectively, these theoretical frameworks imply:

* Nonlinear leverage effects
* Interaction between profitability and financing
* Heterogeneous marginal impacts

Thus, strictly linear empirical specifications may fail to capture the structural complexity implied by theory.

# 2.2 Determinants of Firm Value: Empirical Evidence

## 2.2.1 Firm Size

Firm size, typically measured as the natural logarithm of total assets, influences firm value through multiple channels.

Larger firms benefit from economies of scale, diversified operations, and better access to capital markets (Titman & Wessels, 1988). However, size may also introduce bureaucratic inefficiencies and agency costs (Williamson, 1967).

Empirical findings are mixed. Some studies report a positive association between size and Tobin’s Q (Lang & Stulz, 1994), while others find diminishing marginal effects (Moeller, Schlingemann, & Stulz, 2005).

These inconsistencies suggest potential nonlinear or threshold effects.


## 2.2.2 Profitability

Profitability, often proxied by ROA or EBITDA, is generally positively related to firm value. High profitability signals competitive advantage and efficient management (Fama & French, 2006).

However, the marginal effect of profitability may decline at higher levels due to competitive entry or mean reversion (Wiggins & Ruefli, 2002).

Therefore, a nonlinear specification may better capture profitability–valuation dynamics.



## 2.2.3 Leverage

Empirical evidence on leverage is highly inconsistent.

Some studies find positive effects consistent with tax shield benefits (Graham, 2000). Others report negative associations due to financial distress costs (Opler & Titman, 1994).

Many studies find statistically insignificant results (Frank & Goyal, 2009), potentially reflecting model misspecification or nonlinear relationships.

If the true relationship is quadratic:


$$V = \beta_1 Leverage - \beta_2 Leverage^2 + \varepsilon$$


then linear models may underestimate leverage effects.



## 2.2.4 Dividend Policy

Dividend policy may increase firm value by reducing agency costs (Easterbrook, 1984) or signaling private information (Bhattacharya, 1979).

However, empirical findings differ across countries and regulatory environments (La Porta et al., 2000).

The interaction between dividends and profitability may further complicate valuation effects.



## 2.2.5 Financial Distress

Financial distress risk, frequently measured using Altman’s Z-score (Altman, 1968), is negatively associated with firm survival probability.

Studies show that distress risk reduces firm value through higher discount rates and operational disruptions (Campbell, Hilscher, & Szilagyi, 2008).

However, distress risk may interact with leverage and profitability, suggesting higher-order effects.



## 2.2.6 Synthesis of Empirical Findings

Across determinants, three stylized facts emerge:

1. Empirical results are often inconsistent.
2. Nonlinear effects are theoretically plausible.
3. Interaction effects are underexplored in linear frameworks.

These observations motivate the use of flexible modeling approaches capable of uncovering complex valuation structures.


# 2.3 Machine Learning in Corporate Finance

## 2.3.1 The Emergence of Machine Learning in Financial Research

Over the past decade, machine learning (ML) techniques have fundamentally transformed empirical research in finance. Unlike traditional econometric models that impose parametric functional forms, machine learning algorithms approximate complex and potentially unknown data-generating processes using flexible, data-driven methods (Hastie, Tibshirani, & Friedman, 2009).

In asset pricing, Gu, Kelly, and Xiu (2020) demonstrate that machine learning models significantly outperform traditional linear factor models in out-of-sample return prediction. Similarly, in credit risk modeling and bankruptcy prediction, ensemble tree-based methods have been shown to outperform logistic regression and discriminant analysis (Lessmann et al., 2015; Barboza, Kimura, & Altman, 2017).

In corporate finance, ML applications have expanded to include:

* Earnings prediction (Feng, He, & Polson, 2018),
* Investment forecasting (Erel et al., 2021),
* Financial distress detection (Chen, Härdle, & Moro, 2018),
* Corporate fraud identification (Perols, 2011).

Despite these advances, the application of machine learning to firm valuation remains relatively limited compared to its use in asset pricing and credit modeling.


## 2.3.2 Conceptual Distinction: Econometrics vs. Machine Learning

The methodological distinction between econometrics and machine learning lies primarily in objective and philosophy.

Traditional econometrics focuses on estimating structural parameters and testing hypotheses:

$$Y = X\beta + \varepsilon$$

where inference regarding (\beta) is central.

Machine learning, by contrast, prioritizes predictive accuracy:

$$\min_{f \in \mathcal{F}} \sum_{i=1}^{n} (y_i - f(x_i))^2$$

where (f(\cdot)) belongs to a flexible function class.

Athey and Imbens (2019) note that while econometrics emphasizes identification and causal interpretation, machine learning emphasizes prediction and generalization. Mullainathan and Spiess (2017) argue that ML excels in high-dimensional settings where nonlinearities and interactions are prevalent.

Thus, ML methods are particularly suitable when the true functional relationship between financial variables and firm value is complex and unknown.



## 2.3.3 Tree-Based Ensemble Methods

Among various machine learning techniques, tree-based ensemble models have become dominant in financial applications due to their predictive strength and interpretability relative to deep learning models.

### (a) Random Forest

Random Forest, introduced by Breiman (2001), builds multiple decision trees using bootstrap samples and random feature selection. The final prediction is obtained by averaging across trees:

$$\hat{f}*{RF}(x) = \frac{1}{B} \sum*{b=1}^{B} T_b(x)$$

Random Forest reduces variance through aggregation and handles multicollinearity effectively.

Empirical finance applications demonstrate strong performance in bankruptcy prediction and credit scoring (Lessmann et al., 2015).

However, Random Forest may struggle with extrapolation beyond observed data ranges.



### (b) Gradient Boosting and XGBoost

Gradient Boosting, developed by Friedman (2001), constructs models sequentially by fitting each new tree to the residual errors of prior trees:

$$\hat{f}*m(x) = \hat{f}*{m-1}(x) + \eta T_m(x)$$


where $\eta$ denotes the learning rate.

XGBoost (Chen & Guestrin, 2016) enhances gradient boosting through regularization and efficient optimization:

$$\mathcal{L} = \sum_{i=1}^{n} l(y_i, \hat{y}*i) + \sum*{k} \Omega(T_k)$$


where $\Omega(T_k)$ penalizes model complexity.

Gu et al. (2020) show that gradient boosting algorithms dominate linear models in predictive performance in asset pricing contexts.



### (c) LightGBM

LightGBM further improves computational efficiency using histogram-based splitting and leaf-wise growth strategies. Ke et al. (2017) demonstrate that LightGBM achieves comparable or superior accuracy with significantly lower computational cost.

For panel datasets with large cross-sectional dimensions, such efficiency gains are particularly valuable.

## 2.3.4 Advantages of Machine Learning for Firm Value Modeling

Machine learning offers several key advantages in corporate finance research:

### (1) Nonlinear Function Approximation

Tree-based models approximate arbitrary nonlinear functions without explicitly specifying polynomial or interaction terms (Hastie et al., 2009). This is particularly relevant when theory implies threshold effects, such as the inverted-U leverage relationship suggested by trade-off theory.


### (2) Automatic Interaction Detection

Decision trees partition feature space recursively, implicitly capturing interaction effects among variables. In contrast, linear models require pre-specified interaction terms, which may omit relevant structures.


### (3) Robustness to Multicollinearity

Tree-based methods are less sensitive to multicollinearity compared to OLS, where correlated regressors inflate standard errors (James et al., 2021).

Given that financial ratios are often highly correlated, this robustness is a significant advantage.


### (4) Superior Predictive Performance

Across multiple domains in finance, ML models demonstrate superior out-of-sample accuracy relative to traditional econometric methods (Gu et al., 2020; Mullainathan & Spiess, 2017).

This predictive strength suggests that nonlinear structures may play a substantial role in financial data.


## 2.3.5 Limitations of Machine Learning in Corporate Finance

Despite these advantages, machine learning models face several important limitations.

### (1) Lack of Structural Interpretability

Unlike regression coefficients, tree-based models do not provide easily interpretable marginal effects. As Athey and Imbens (2019) argue, ML algorithms prioritize prediction rather than structural parameter estimation.


### (2) Correlation Does Not Imply Causation

Machine learning models do not inherently address endogeneity concerns such as omitted variable bias or reverse causality. Without identification strategies, predictive relationships may not reflect causal effects.

This limitation is particularly relevant in corporate finance, where financial decisions are often endogenous.


### (3) Overfitting Risk

Although cross-validation mitigates overfitting, high model complexity may still capture noise rather than signal if hyperparameters are not carefully tuned (Hastie et al., 2009).


## 2.3.6 Explainable Machine Learning: SHAP

Recent advances in explainable AI attempt to address interpretability concerns. SHAP (Shapley Additive Explanations) decomposes predictions into additive feature contributions:

$$f(x) = \phi_0 + \sum_{j=1}^{p} \phi_j$$

where (\phi_j) represents the marginal contribution of feature (j) (Lundberg & Lee, 2017).

SHAP provides:

* Global feature importance,
* Local explanation for individual observations,
* Visualization of nonlinear patterns.

However, SHAP remains descriptive rather than causal. It explains model behavior but does not establish economic identification.


## 2.3.7 Implications for This Study

The literature suggests that machine learning offers substantial predictive advantages in complex financial environments characterized by nonlinearities and interactions. However, it lacks built-in mechanisms for causal interpretation and structural inference.

Therefore, while ML may enhance predictive performance in modeling firm value, it cannot fully substitute econometric approaches when economic interpretation and causal validation are required.

This observation motivates the development of a hybrid framework that integrates the predictive flexibility of machine learning with the interpretability and identification strength of econometric methods.


# 2.4 Hybrid Modeling and the Bias–Variance Trade-off

## 2.4.1 Motivation for Hybrid Modeling

The preceding sections highlight a central methodological tension in corporate finance research:

* Traditional econometric models offer structural interpretability but impose restrictive functional forms.
* Machine learning models provide predictive flexibility but lack structural transparency and causal grounding.

This trade-off reflects a broader distinction between inference-oriented and prediction-oriented modeling paradigms (Athey & Imbens, 2019; Mullainathan & Spiess, 2017). While econometric models are designed to estimate interpretable parameters, machine learning models optimize out-of-sample predictive performance.

Recent methodological developments suggest that combining models with distinct inductive biases may improve predictive accuracy and stability (Hastie, Tibshirani, & Friedman, 2009). This principle forms the theoretical foundation for hybrid modeling approaches.



## 2.4.2 Ensemble Learning and Model Averaging

Hybrid modeling belongs to the broader class of ensemble learning methods. Ensemble methods combine multiple base learners to produce a single aggregated prediction:

$$\hat{f}*{ensemble}(x) = \sum*{m=1}^{M} w_m \hat{f}_m(x)$$

where (w_m) denotes the weight assigned to model (m).

Breiman (1996) shows that averaging multiple predictors can reduce variance without substantially increasing bias, provided that prediction errors are not perfectly correlated. This principle underlies bagging, boosting, and stacking methods.

Model averaging also has a long tradition in econometrics. Hansen (2007) demonstrates that model averaging can outperform model selection in terms of mean squared error (MSE), especially when model uncertainty is high.

In the context of firm valuation, linear and nonlinear models capture different structural properties:

* Linear models approximate global relationships.
* Tree-based models capture local nonlinear patterns.

Combining these approaches may exploit complementary strengths.



## 2.4.3 The Bias–Variance Trade-off

The theoretical justification for hybrid modeling rests on the bias–variance decomposition of prediction error.

For a regression problem with squared loss, expected prediction error can be decomposed as:

$$E[(Y - \hat{f}(X))^2] = \text{Bias}^2 + \text{Variance} + \sigma^2$$

(Hastie et al., 2009).

* **Bias** measures systematic deviation from the true function.
* **Variance** reflects sensitivity to sampling fluctuations.
* $\sigma^2$ denotes irreducible error.

Linear models typically exhibit:

* Relatively high bias (due to functional form restrictions),
* Low variance (stable parameter estimates).

Tree-based ensemble models typically exhibit:

* Lower bias (due to flexibility),
* Higher variance (due to model complexity).

By combining models with different bias–variance profiles, hybrid approaches may reduce overall MSE.

If we denote:

[
\hat{Y}*{Hybrid} = w \hat{Y}*{ML} + (1-w) \hat{Y}_{Linear}
]

the optimal weight (w^*) minimizes:

Under certain conditions, the optimal convex combination achieves lower prediction error than either model individually (Hansen, 2007).


## 2.4.4 Complementarity Between Linear and Nonlinear Models

In corporate finance applications, linear and nonlinear models capture different aspects of economic structure:

### (1) Global vs. Local Approximation

Linear models estimate average marginal effects across the entire sample:

$$\frac{\partial Y}{\partial X_j} = \beta_j$$

Tree-based models estimate piecewise constant approximations, allowing heterogeneous effects across subgroups.

Thus, the two approaches differ in granularity and flexibility.

### (2) Interpretability vs. Flexibility

Econometric models provide interpretable coefficients, facilitating hypothesis testing and economic interpretation.

Machine learning models capture complex interactions but often lack closed-form marginal effects (Athey & Imbens, 2019).

Hybrid modeling aims to preserve interpretability while improving predictive performance.



### (3) Structural Stability

Linear panel models may better capture persistent structural relationships, particularly when fixed effects are included.

Machine learning models may capture nonlinear short-term patterns.

Combining them may improve both short-run prediction and structural robustness.

## 2.4.5 Weighted Hybrid Framework

The hybrid prediction model adopted in this study is defined as:

$$\hat{Y}*{Hybrid} = w \hat{Y}*{ML} + (1-w)\hat{Y}_{Linear}$$


where $w \in [0,1]$ is determined through cross-validation to Rspuare.

This framework differs from standard stacking in two ways:

1. It combines structurally distinct paradigms (parametric vs. nonparametric).
2. It preserves economic interpretability by retaining the linear component.

Cross-validation is employed to avoid overfitting and ensure generalizability (James et al., 2021).


## 2.4.6 Empirical Evidence on Hybrid Approaches

Hybrid approaches have shown strong performance across various domains:

* In macroeconomic forecasting, combined forecasts often outperform individual models (Timmermann, 2006).
* In financial return prediction, ensemble methods improve robustness (Gu et al., 2020).
* In credit risk modeling, hybrid systems combining statistical and machine learning methods outperform standalone models (Lessmann et al., 2015).

However, hybrid approaches remain underexplored in corporate valuation contexts, particularly when combined with causal validation techniques.

## 2.4.7 Limitations of Purely Predictive Hybrids

While hybrid prediction may improve accuracy, it does not automatically resolve identification issues.

Even if:


$\hat{Y}_{Hybrid}$


achieves superior predictive performance, it does not guarantee that estimated relationships reflect causal effects.

This limitation underscores the need for integrating causal inference methods, such as Double Machine Learning, into the hybrid framework.

---

## 2.4.8 Implications for This Study

The theoretical and empirical literature suggests that hybrid modeling can reduce prediction error by balancing bias and variance while leveraging complementary model structures.

However, predictive superiority alone is insufficient in corporate finance research, where economic interpretation and causal validation remain essential.

Therefore, this study integrates:

1. Hybrid prediction modeling,
2. SHAP-based interpretability analysis,
3. Double Machine Learning for causal estimation,
4. Panel fixed-effects validation.

This multi-layered approach seeks to reconcile prediction accuracy with structural inference.



# 2.5 Causal Inference and Double Machine Learning

## 2.5.1 Endogeneity in Corporate Finance Research

A central challenge in empirical corporate finance is endogeneity. Financial decisions such as leverage, dividend policy, and investment are rarely exogenous. Instead, they are jointly determined with firm performance and valuation outcomes.

Consider the baseline empirical specification:


$$Y_i = \beta D_i + X_i'\gamma + \varepsilon_i$$


where:

* (Y_i) denotes firm value (e.g., Tobin’s Q),
* (D_i) is a treatment variable (e.g., leverage or profitability),
* (X_i) is a vector of control variables,
* (\varepsilon_i) is an error term.

If: $$E[D_i \varepsilon_i] \neq 0$$


the OLS estimator of (\beta) is biased and inconsistent (Wooldridge, 2010).

Endogeneity in corporate finance may arise from:

1. **Reverse causality** – Higher firm value may influence leverage decisions.
2. **Omitted variable bias** – Unobserved managerial quality affects both financing and valuation.
3. **Measurement error** – Accounting variables may be noisy proxies.
4. **Simultaneity** – Financial policies and firm performance are jointly determined.

Traditional solutions include instrumental variables (IV) and panel fixed effects. However, these methods face limitations when:

* Suitable instruments are unavailable,
* The control vector (X_i) is high-dimensional,
* The true functional form is nonlinear.

These challenges motivate the integration of machine learning into causal estimation.


## 2.5.2 From Prediction to Causal Inference

Machine learning excels in prediction but does not inherently solve causal identification problems (Athey & Imbens, 2019). However, recent advances combine ML with econometric identification strategies to address high-dimensional confounding.

Chernozhukov et al. (2018) introduce **Double Machine Learning (DML)**, a framework that enables valid inference on low-dimensional treatment parameters in the presence of high-dimensional controls.

The key idea is to separate:

* The nuisance component (high-dimensional controls),
* The parameter of interest (treatment effect).

This separation is achieved through orthogonalization.



## 2.5.3 Orthogonalization and the Neyman Orthogonal Score

Consider the partially linear model:


$$Y = \theta D + g(X) + \varepsilon$$


where:

* (g(X)) is an unknown, potentially nonlinear function,
* (\theta) is the treatment effect of interest.

Instead of estimating (g(X)) parametrically, DML uses machine learning to estimate it flexibly.

The orthogonalization procedure proceeds as follows:

### Step 1: Estimate nuisance functions


$$\hat{m}(X) = E[Y|X]$$


$$\hat{g}(X) = E[D|X]$$


using machine learning methods.

### Step 2: Compute residuals


$$\tilde{Y} = Y - \hat{m}(X)$$

$$\tilde{D} = D - \hat{g}(X)$$

### Step 3: Estimate treatment effect

$$\hat{\theta} = \frac{\sum \tilde{D}\tilde{Y}}{\sum \tilde{D}^2}$$


This estimator relies on the Neyman orthogonal score, which ensures that small estimation errors in nuisance functions do not substantially bias the treatment effect estimate (Chernozhukov et al., 2018).


## 2.5.4 Cross-Fitting and Bias Reduction

A critical innovation of DML is **cross-fitting**.

The dataset is partitioned into (K) folds:

1. Nuisance functions are estimated on training folds.
2. Residuals are computed on validation folds.
3. Treatment effects are estimated using out-of-sample predictions.

Cross-fitting prevents overfitting and ensures that the treatment effect estimator achieves:

* Root-(n) consistency,
* Asymptotic normality.

Formally:

$$\sqrt{n}(\hat{\theta} - \theta_0) \rightarrow N(0, V)$$


even when machine learning estimators converge at slower rates.

This property distinguishes DML from naive plug-in approaches.



## 2.5.5 Advantages of DML in Corporate Finance

DML offers several advantages in corporate finance applications:

### (1) High-Dimensional Controls

Corporate finance datasets often contain numerous financial ratios and firm characteristics. DML allows flexible modeling of high-dimensional confounders without imposing parametric restrictions.



### (2) Nonlinear Confounding Structures

If the relationship between controls and treatment is nonlinear, traditional linear residualization may be insufficient. Machine learning methods can approximate complex confounding structures.



### (3) Valid Statistical Inference

Unlike many ML-based estimators, DML provides asymptotically valid confidence intervals under regularity conditions (Chernozhukov et al., 2018).


## 2.5.6 Empirical Applications of DML in Finance

DML has been increasingly adopted in financial economics.

For example:

* Farrell, Liang, and Misra (2021) apply DML to evaluate heterogeneous treatment effects in economic settings.
* Knaus, Lechner, and Strittmatter (2021) use DML in policy evaluation.
* In asset pricing, machine learning-based causal approaches have been applied to estimate risk premia under high-dimensional covariates (Gu et al., 2020).

Although DML has gained traction in econometrics and asset pricing, its application to corporate valuation remains limited.



## 2.5.7 Limitations of DML

Despite its strengths, DML relies on several assumptions:

1. **Unconfoundedness**
   Conditional on (X), treatment assignment must be independent of the error term.

2. **Overlap**
   Sufficient variation in treatment conditional on controls.

3. **Correct specification of nuisance functions**
   While flexible, ML must approximate true conditional expectations reasonably well.

Moreover, DML estimates partial causal effects rather than structural equilibrium effects. Thus, interpretation should remain cautious.



## 2.5.8 Role of DML in This Study

In this study, DML serves as a causal validation layer within the broader hybrid framework.

Specifically, DML will be used to estimate the partial causal effects of:

* Profitability (e.g., EBITDA),
* Leverage,
* Dividend policy,

on firm value.

This allows the study to examine whether variables identified as important predictors in machine learning models also exhibit economically meaningful causal effects.

By integrating:

1. Hybrid predictive modeling,
2. SHAP interpretability analysis,
3. Double Machine Learning estimation,
4. Panel fixed-effects validation,

the study adopts a multi-layered methodological strategy that bridges prediction and causal inference.



# 2.6 Research Gap and Hypothesis Development

## 2.6.1 Synthesis of the Literature

The preceding review highlights three parallel but largely disconnected strands of literature:

1. **Traditional corporate finance research**, relying on linear econometric models to identify determinants of firm value.
2. **Machine learning applications in finance**, focusing primarily on predictive performance.
3. **Causal inference methodologies**, particularly Double Machine Learning (DML), developed in econometrics but not widely integrated into corporate valuation research.

While each strand offers valuable insights, their integration remains limited.

Traditional studies of firm value typically employ linear panel regressions, emphasizing interpretability and hypothesis testing (Wooldridge, 2010). However, such models impose restrictive assumptions regarding functional form and interaction effects.

In contrast, machine learning models demonstrate superior predictive performance in asset pricing and risk prediction contexts (Gu, Kelly, & Xiu, 2020), yet often lack structural interpretability and causal identification.

Recent methodological advances in DML (Chernozhukov et al., 2018) address endogeneity concerns in high-dimensional settings, but applications in corporate valuation remain sparse.

This fragmentation creates an opportunity for methodological integration.



## 2.6.2 Research Gap

Based on the literature review, four primary gaps can be identified.

### Gap 1: Linear Specification Dominance

Most empirical studies examining firm value rely on linear functional forms. This may obscure nonlinear relationships such as:

* Threshold effects in profitability,
* Non-monotonic leverage effects,
* Interaction effects between size and financial policy.

Few studies explicitly test whether nonlinear models provide systematically better predictive performance in firm valuation contexts.


### Gap 2: Prediction–Inference Divide

Machine learning studies emphasize predictive accuracy but often neglect economic interpretation and causal validation.

Conversely, econometric studies emphasize inference but may sacrifice predictive performance.

The absence of an integrated framework combining:

* Predictive modeling,
* Interpretability tools,
* Causal validation,

represents a significant methodological gap.



### Gap 3: Limited Use of Hybrid Frameworks

Although ensemble methods are widely used in machine learning (Hastie et al., 2009), hybrid combinations of:

* Parametric econometric models,
* Nonparametric tree-based models,

are rarely implemented in corporate valuation research.

Moreover, existing ensemble studies focus on pure prediction rather than economically interpretable hybrid structures.



### Gap 4: Lack of Causal Validation of ML-Identified Predictors

Even when machine learning identifies important predictors via feature importance or SHAP values, these results are descriptive rather than causal.

Few studies examine whether variables identified as important in ML models retain significance under:

* Orthogonalized causal estimation,
* Panel fixed-effects frameworks.

This creates uncertainty regarding whether predictive importance reflects genuine economic mechanisms.


## 2.6.3 Positioning of the Present Study

This study addresses these gaps by developing an integrated framework that:

1. Compares linear and nonlinear models in firm value prediction,
2. Constructs a weighted hybrid econometric–machine learning model,
3. Uses SHAP for interpretability,
4. Applies Double Machine Learning for causal estimation,
5. Validates results using panel fixed-effects models.

This multi-layered design bridges prediction and inference while preserving economic interpretability.



## 2.6.4 Hypothesis Development

The hypotheses are structured along two dimensions:

* Predictive performance,
* Causal validation.



### 2.6.4.1 Predictive Hypotheses

#### H1: Nonlinear Superiority Hypothesis

Given the flexibility of tree-based ensemble models in capturing nonlinearities and interaction effects (Friedman, 2001; Gu et al., 2020), we hypothesize:

> **H1:** Nonlinear machine learning models exhibit superior out-of-sample predictive performance relative to linear regression models in explaining firm value.

This hypothesis reflects the expectation that firm valuation relationships are not strictly linear.


#### H2: Hybrid Performance Hypothesis

Based on bias–variance trade-off theory (Hastie et al., 2009; Hansen, 2007), combining models with complementary error structures may reduce overall prediction error.

> **H2:** A weighted hybrid model combining linear and nonlinear predictions outperforms standalone linear or nonlinear models in out-of-sample prediction accuracy.


### 2.6.4.2 Structural Hypotheses

#### H3: Nonlinearity in Profitability Effects

Traditional theory suggests a positive relationship between profitability and firm value. However, diminishing marginal effects or threshold behavior may exist.

> **H3:** The relationship between profitability and firm value is nonlinear, exhibiting threshold or diminishing marginal effects.

This hypothesis will be examined using SHAP value analysis.


#### H4: Non-Monotonic Leverage Effect

Trade-off theory implies a potential inverted U-shaped relationship between leverage and firm value.

> **H4:** Leverage exhibits a nonlinear relationship with firm value, consistent with trade-off theory.



### 2.6.4.3 Causal Hypotheses

#### H5: Partial Causal Effects

If variables identified as important predictors reflect genuine economic mechanisms, they should remain statistically significant under orthogonalized estimation.

> **H5:** Key financial variables identified by machine learning retain statistically significant partial causal effects under Double Machine Learning estimation.



#### H6: Coefficient Shrinkage Under Causal Adjustment

If OLS estimates are biased upward due to endogeneity, orthogonalization should reduce coefficient magnitude.

> **H6:** Estimated treatment effects under Double Machine Learning are smaller in magnitude than corresponding OLS estimates.



## 2.6.5 Conceptual Framework

The integrated framework of this study can be summarized as follows:

1. **Stage 1:** Prediction comparison

   * Linear vs. nonlinear models

2. **Stage 2:** Hybrid modeling

   * Bias–variance optimization

3. **Stage 3:** Interpretability analysis

   * SHAP decomposition

4. **Stage 4:** Causal validation

   * Double Machine Learning
   * Panel fixed effects

This layered approach allows:

* Improved predictive accuracy,
* Structural interpretability,
* Econometric robustness.




| No. | Reference                                        | Context                    | ML Model(s)   | XAI Method         | Target Variable    | Key Findings                             | Limitations                   | Relevance to Your Study      |
| --- | ------------------------------------------------ | -------------------------- | ------------- | ------------------ | ------------------ | ---------------------------------------- | ----------------------------- | ---------------------------- |
| 1   | Bryan Kelly, Seth Pruitt Gu & Dacheng Xiu (2020) | Asset pricing              | RF, NN, GBM   | Feature importance | Stock returns      | Nonlinear models outperform linear       | Limited interpretability      | Foundation for ML in finance |
| 2   | Chen et al. (2021)                               | Firm valuation             | XGBoost       | SHAP               | Tobin’s Q          | Profitability & size are key drivers     | Static explanation            | Direct relevance             |
| 3   | Li et al. (2022)                                 | Corporate performance      | Random Forest | SHAP               | Firm value proxies | Nonlinear relationships dominate         | No temporal analysis          | Supports nonlinear effects   |
| 4   | Wang et al. (2022)                               | ESG & firm value           | XGBoost       | SHAP               | Tobin’s Q          | ESG impacts vary across firms            | No stability test             | Subgroup relevance           |
| 5   | Kim et al. (2023)                                | Korean firms               | LightGBM      | SHAP               | Market value       | Size & leverage important                | Limited model comparison      | Regional relevance           |
| 6   | Zhang et al. (2023)                              | Corporate finance          | XGBoost       | SHAP               | Firm value         | Interaction effects identified           | No pipeline comparison        | Supports interaction insight |
| 7   | Liu et al. (2023)                                | Financial distress & value | RF + XGBoost  | SHAP               | Firm value         | Distress significantly affects valuation | Static SHAP                   | Supports heterogeneity       |
| 8   | Park et al. (2024)                               | Firm valuation             | LGBM          | SHAP               | Tobin’s Q          | Liquidity & profitability dominate       | No hybrid models              | Missing hybrid               |
| 9   | Nguyen et al. (2024)                             | Emerging markets           | XGBoost       | SHAP               | Firm value         | Growth & size vary by region             | No temporal test              | Cross-country gap            |
| 10  | Recent FinXAI studies (2024–2025)                | Finance XAI                | RF, XGB       | SHAP               | Various            | SHAP widely adopted                      | Lack of stability & causality | Core gap                     |





