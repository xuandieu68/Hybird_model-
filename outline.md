# Slide 1. 기업 투자구성의 가치평가

## Valuation of Corporate Investment Components

### 발표 스크립트

안녕하십니까 교수님. 오늘은 제가 진행하고 있는 **기업 투자구성의 가치평가, Valuation of Corporate Investment Components** 연구를 보고드리겠습니다.

이 연구의 출발점은 비교적 단순합니다. 기업은 한 시점에 하나의 투자만 선택하는 것이 아니라, 설비투자 **CAPEX**, 유형자산 **TANG**, 연구개발 **R&D**, 인식된 무형자산 **INTANG**, 판매관리비 **SG&A**, 그리고 금융자산 **FIN** 등 여러 형태의 자원배분을 동시에 결정합니다.

그런데 기존 연구에서는 이러한 투자항목을 개별적으로 분석하는 경우가 많습니다. 예를 들어 R&D와 기업가치의 관계, CAPEX와 기업가치의 관계, 또는 현금 및 금융자산과 기업가치의 관계를 각각 따로 분석합니다.

제 연구에서는 이 문제를 **investment composition**, 즉 기업 전체 투자구성의 관점에서 접근합니다. 구체적으로는 각 투자항목이 단독으로 어느 정도의 가치관련성을 가지는지, 그리고 다른 투자항목을 동시에 고려한 후에도 추가적인 가치정보를 유지하는지를 구분합니다.

실증분석은 2012년부터 2025년까지 한국 상장 비금융기업 2,493개, 총 23,726개의 firm-year observations를 사용하며, 고정효과 회귀와 **Double Machine Learning, DML**을 결합합니다.

오늘 발표에서는 연구문제와 이론적 배경을 먼저 말씀드리고, 이후 데이터와 DML 설계, 핵심 결과, 강건성 검정, 기업환경별 이질성, 그리고 은행 불확실성의 조절효과 순서로 설명드리겠습니다.

### 질문 대비

**Q. 이 연구의 한 문장 contribution은 무엇입니까?**

> 이 연구의 핵심 contribution은 기업투자를 개별 항목으로 보는 대신 하나의 **investment composition**으로 보고, 각 항목의 **standalone valuation relevance**와 **incremental valuation relevance**를 구분했다는 점입니다.

**Q. 이 연구는 투자수익률이나 investment efficiency 연구입니까?**

> 아닙니다. 본 연구의 종속변수는 Tobin's Q이며, 투자항목이 시장가치에 어떤 정보를 제공하는지를 분석하는 **valuation study**입니다.

**Q. DML을 사용했으니 causal study라고 볼 수 있습니까?**

> 그렇지 않습니다. DML은 observed confounders와의 복잡한 비선형 관계를 유연하게 조정하기 위해 사용하며, 본 연구의 기본 해석은 **valuation relevance after flexible adjustment for observed confounders**입니다.

---

# Slide 2. 발표 흐름

## Presentation Roadmap

### 발표 스크립트

발표는 여섯 부분으로 구성하겠습니다.

첫째, 왜 투자항목을 개별적으로 분석하는 것이 충분하지 않을 수 있는지 연구문제를 설명하겠습니다.

둘째, **multi-capital q-theory**, **real options**, **financing frictions**, 그리고 **agency theory**를 중심으로 이론적 배경과 가설을 말씀드리겠습니다.

셋째, 데이터와 변수구성, 그리고 separate-treatment DML과 simultaneous-treatment DML의 차이를 설명하겠습니다.

넷째, 핵심 실증결과를 보고드리겠습니다. 특히 Table 4와 Table 5의 비교가 이 연구에서 가장 중요한 부분입니다.

다섯째, alternative learners, leave-one-component-out, DML-IV, subperiod 및 subgroup 분석을 통해 결과의 강건성과 이질성을 확인합니다.

마지막으로 은행자산과 은행자금조달의 불확실성이 개별 투자항목의 valuation relevance를 어떻게 변화시키는지 살펴보겠습니다.

### 질문 대비

이 slide에서는 별도의 기술적 질문보다는 발표의 **narrative**를 명확히 전달하는 것이 중요합니다.

교수님이 "무엇이 main analysis이고 무엇이 extension인가?"라고 물으면 다음처럼 답하는 것이 좋습니다.

> Main analysis는 standalone versus incremental valuation relevance입니다. Heterogeneity와 bank uncertainty는 이 기본 valuation pattern이 어떤 환경에서 달라지는지를 보는 extension입니다.

---

# Slide 3. 연구 동기: 기업은 하나의 투자만 선택하지 않는다

## Motivation: Firms Do Not Choose One Investment Margin

### 발표 스크립트

이 연구의 첫 번째 동기는 기업의 실제 자원배분이 **multi-dimensional**이라는 점입니다.

기업은 생산능력을 확대하기 위해 CAPEX를 지출할 수 있고, 기존 유형자산을 축적할 수 있으며, R&D를 통해 새로운 기술을 개발할 수도 있습니다. 또한 소프트웨어나 특허와 같이 회계적으로 인식되는 무형자산을 보유하고, SG&A를 통해 브랜드, 고객관계, 조직능력이나 상업화 역량을 구축할 수도 있습니다. 마지막으로 금융자산을 보유해 미래 투자에 필요한 financial flexibility를 확보할 수도 있습니다.

이 항목들은 경제적 의미가 서로 다릅니다.

CAPEX는 현재의 물적 투자 flow이고, TANG는 이미 축적된 physical asset stock입니다. R&D는 불확실하지만 확장성이 큰 growth option에 가깝고, INTANG는 일정 부분 회계적으로 인식된 무형자산 stock입니다.

SG&A는 문제가 조금 더 복잡합니다. 일부는 routine overhead이지만, 일부는 브랜드, 고객관계, 조직자본, 판매 네트워크 등 미래에 지속되는 capability를 형성할 수 있습니다.

FIN 역시 operating asset은 아니지만 외부금융이 어려운 상황에서 future investment를 지속할 수 있는 financial flexibility를 제공할 수 있습니다.

따라서 이 여섯 항목은 불확실성, 회계가시성, 담보가능성, 재배치 가능성, 성장옵션, financing role 측면에서 서로 다르기 때문에 시장이 동일한 valuation weight를 부여할 이유가 없습니다.

이것이 제가 investment composition을 연구의 기본 단위로 설정한 이유입니다.

### 반드시 이해할 개념

Multi-capital q-theory는 여러 종류의 자본이 서로 다른 **shadow value**를 가질 수 있다는 논리를 제공합니다. 다만 본 연구는 structural marginal q를 추정하는 것은 아닙니다. Draft에서도 empirical coefficients를 structural marginal q로 해석하지 않고, broader investment composition을 고려한 valuation information으로 한정합니다.

### 질문 대비

**Q. CAPEX와 TANG는 둘 다 physical investment인데 왜 따로 봅니까?**

> CAPEX는 현재 투자 flow이고, TANG는 누적된 stock입니다. CAPEX는 expansion signal을 포함할 수 있지만 TANG는 assets-in-place, collateral, maturity를 더 많이 반영합니다. 따라서 시장에 전달하는 정보가 동일하지 않습니다.

**Q. R&D와 INTANG는 중복 아닌가요?**

> R&D는 현재 knowledge creation expenditure이고 대부분 즉시 비용처리됩니다. INTANG는 회계적으로 recognition 기준을 통과한 stock입니다. 따라서 innovation input과 recognised intangible accumulation이라는 서로 다른 정보를 포착합니다.

**Q. FIN을 왜 investment component라고 합니까?**

> FIN은 productive capital이라고 주장하지 않습니다. 본 연구에서는 accounting-based resource-allocation component로 정의하고, 경제적 역할을 financial flexibility로 해석합니다.

---

# Slide 4. 기존 단일항목 접근의 한계

## Why Single-Component Valuation Tests Can Be Misleading

### 발표 스크립트

기존 접근의 가장 큰 문제는 기업의 투자선택이 서로 상관되어 있다는 점입니다.

예를 들어 R&D를 많이 하는 기업은 그 기술을 시장에 상업화하기 위해 SG&A도 많이 지출할 수 있습니다. 유형자산이 적은 intangible-intensive firm은 담보능력이 낮기 때문에 더 많은 내부유동성이나 금융자산을 보유할 수도 있습니다.

따라서 단순히 R&D만 넣은 valuation regression에서 R&D coefficient는 순수한 R&D 정보뿐 아니라 SG&A나 INTANG 등 함께 움직이는 다른 투자선택의 정보를 일부 반영할 수 있습니다.

이 경우 제가 말하는 **standalone valuation relevance**는 "이 항목이 단독으로 informative한가?"를 의미합니다.

반면 **incremental valuation relevance**는 "다른 투자항목들의 정보까지 모두 관찰한 후에도 이 항목이 독립적인 valuation information을 남기는가?"라는 질문입니다.

따라서 simultaneous model은 단순한 robustness specification이 아닙니다. 실제로 **estimand 자체가 달라집니다.**

Separate model의 질문은:

> Is this component informative?

이고,

Simultaneous model의 질문은:

> Does this component retain information once the rest of the investment mix is observed?

입니다.

이 구분이 본 연구에서 가장 중요합니다.

### 질문 대비

**Q. 그냥 multivariate OLS에 여섯 변수를 같이 넣으면 되는 것 아닌가요?**

> Joint OLS가 중요한 benchmark입니다. 실제로 본 연구도 그것을 먼저 수행합니다. 다만 valuation과 investment choices는 size, growth, leverage, liquidity, ownership 등과 비선형적으로 연결될 수 있습니다. DML은 이러한 nuisance relationships에 더 flexible한 adjustment를 허용한다는 추가적인 장점이 있습니다.

**Q. Simultaneous coefficient를 direct effect라고 불러도 됩니까?**

> 엄밀하게는 direct causal effect라고 부르지 않는 것이 좋습니다. 본 연구에서는 **incremental valuation relevance conditional on the other observed components**라고 표현합니다.

**Q. 왜 component interaction을 넣어서 complementarity를 직접 보지 않습니까?**

> 현재 연구질문은 incremental information에 초점을 둡니다. Simultaneous model은 complementarity를 공식적으로 식별하지 않습니다. Pairwise interactions 또는 heterogeneous treatment effect design은 future extension입니다.

---

# Slide 5. 연구질문과 가설

## Research Questions and Hypotheses

### 발표 스크립트

이 논리를 바탕으로 세 가지 기본 가설을 설정했습니다.

첫 번째 H1a는 개별 투자항목의 standalone valuation coefficient가 동일하지 않을 것이라는 가설입니다.

두 번째 H1b는 다른 투자항목을 동시에 포함했을 때 적어도 일부 항목의 coefficient magnitude 또는 statistical precision이 변할 것이라는 가설입니다. 이것은 investment components가 서로 공유하는 정보가 있다는 논리와 연결됩니다.

세 번째 H1c는 knowledge-oriented components, 특히 R&D와 recognised intangible assets가 simultaneous specification에서도 전통적인 physical components보다 상대적으로 큰 positive point estimates를 유지할 것이라는 가설입니다.

다만 여기서 중요한 것은 제가 universal ranking을 가정하지 않는다는 점입니다.

Multi-capital theory 자체는 R&D가 항상 가장 높은 valuation coefficient를 가져야 한다고 예측하지 않습니다. 각 component의 valuation은 기업환경과 financing environment에 따라 달라질 수 있습니다.

따라서 이후 분석에서는 KOSPI와 KOSDAQ, Chaebol과 non-Chaebol, high-tech와 low-tech, 그리고 banking uncertainty에 따른 heterogeneity를 research questions 형태로 분석합니다.

### 질문 대비

**Q. H1c는 ex-post result를 보고 만든 가설처럼 보일 위험은 없습니까?**

> 그 우려가 있습니다. 따라서 H1c를 강한 coefficient ranking hypothesis로 표현하지 않고 "knowledge-oriented components retain larger positive point estimates" 정도의 directional expectation으로 제한했습니다.

**Q. 왜 heterogeneity는 hypothesis가 아니라 RQ입니까?**

> 각 subgroup에서 이론적으로 opposing mechanisms가 존재해 sign을 명확히 예측하기 어렵기 때문입니다. 예를 들어 Chaebol은 financing constraints를 완화하지만 동시에 agency concerns도 가질 수 있습니다.

**Q. H1c 검정에서 R&D와 INTANG가 CAPEX보다 statistically significantly larger한지 검정했습니까?**

> 모든 pairwise coefficient differences를 formal test한 것은 아닙니다. 따라서 H1c는 coefficient pattern과 point estimates에 대한 evidence로 해석하며 universal statistical ranking이라고 주장하지 않습니다.

---

# Slide 6. 이론적 연결고리

## Theoretical Channels

### 발표 스크립트

이 연구의 이론은 네 가지 channel을 결합합니다.

첫째는 **multi-capital q-theory**입니다. 전통적인 q-model은 자본을 하나의 homogeneous stock으로 단순화하지만, multi-capital framework에서는 서로 다른 capital goods가 서로 다른 shadow values와 adjustment processes를 가질 수 있습니다.

둘째는 **real-options theory**입니다. 특히 R&D는 현재 cash flow를 바로 생산하기보다는 미래의 제품, 기술, 시장 진입 가능성을 열어주는 growth option의 성격을 가집니다. Financial assets 역시 외부금융이 비쌀 때 미래 투자기회를 보존할 수 있다는 측면에서 option value를 가질 수 있습니다.

셋째는 **financing frictions**입니다. 유형자산은 collateral value를 제공할 수 있지만 R&D나 organisational capital은 담보로 사용하기 어렵습니다. 따라서 intangible-intensive firm은 external finance에 더 취약하거나 내부유동성을 더 많이 보유할 동기를 가질 수 있습니다.

넷째는 **agency theory**입니다. 예를 들어 금융자산은 valuable flexibility일 수도 있지만, 경영자가 value-destroying investment를 할 수 있는 slack으로 해석될 수도 있습니다.

따라서 같은 component라도 기업의 governance와 external financing condition에 따라 시장의 valuation이 달라질 수 있습니다.

### 질문 대비

**Q. 이론이 너무 많아 보이는데 main theory가 무엇입니까?**

> Main organizing framework는 **multi-capital valuation**입니다. Real options, financing frictions, agency는 component별 valuation weight가 왜 달라질 수 있는지를 설명하는 supporting mechanisms입니다.

**Q. q-theory인데 왜 investment equation이 아니라 Tobin's Q regression입니까?**

> 본 연구는 structural q-investment equation을 estimate하는 연구가 아닙니다. q-theory를 component-specific valuation weights가 가능하다는 conceptual foundation으로 사용합니다.

**Q. Average q와 marginal q를 동일하게 보는 것 아닙니까?**

> 아닙니다. Hayashi equivalence에는 restrictive assumptions가 필요합니다. 따라서 empirical TobinQ coefficient를 structural marginal q라고 해석하지 않습니다. Draft도 이를 명시적으로 제한합니다.

---

# Slide 7. 데이터와 표본구성

## Data and Sample Construction

### 발표 스크립트

실증분석은 2012년부터 2025년까지 한국 KOSPI와 KOSDAQ에 상장된 비금융기업을 대상으로 합니다.

최종 표본은 2,493개 기업과 23,726개의 firm-year observations로 구성된 unbalanced panel입니다.

기업의 재무정보, governance, ownership, market segment 등의 자료는 FNGuide에서 수집했습니다.

모든 focal investment variables는 기본적으로 한 시점 lag를 사용합니다. 즉, t−1의 투자구성과 t 시점의 Tobin's Q를 연결합니다.

이 lag structure는 완전한 endogeneity solution은 아닙니다. 다만 contemporaneous investment decision과 current valuation 사이의 가장 직접적인 simultaneity를 완화하기 위한 timing convention입니다.

Continuous variables는 극단치 영향을 줄이기 위해 1st와 99th percentile에서 winsorise했으며, Chaebol dummy, Big 4 dummy와 같은 binary variables는 winsorisation하지 않았습니다.

Inference에서 가장 중요한 dependence unit는 firm입니다. Panel observations가 동일 기업 내에서 독립적이지 않기 때문에 cross-fitting과 bootstrap 모두 firm dependence를 고려하도록 설계했습니다.

### 질문 대비

**Q. 왜 firm fixed effects가 아니라 industry FE를 사용했습니까?**

> 현재 baseline design은 cross-sectional valuation differences와 component variation을 유지하면서 industry/year heterogeneity를 통제하는 방식입니다. Firm FE를 사용하면 persistent investment composition variation이 크게 줄어들 수 있습니다. 다만 교수님이 강하게 질문할 가능성이 있으므로 firm FE robustness 여부는 추가 검토할 가치가 있습니다.

**Q. Lag를 사용하면 endogeneity가 해결됩니까?**

> 아닙니다. Lagging은 temporal ordering을 개선할 뿐 omitted variables나 expectation-driven investment를 제거하지 못합니다.

**Q. 2012–2025를 선택한 이유는 무엇입니까?**

> Final usable sample은 focal variables, controls, governance data 및 lag construction의 공통 availability를 기준으로 형성되었습니다.

**Q. Unbalanced panel이 문제되지 않습니까?**

> 자체적으로 bias를 의미하지는 않지만 attrition이나 data availability가 non-random할 수 있습니다. 따라서 결과는 observed sample conditional interpretation으로 보는 것이 적절합니다.

---

# Slide 8. 핵심 투자변수와 경제적 의미

## Investment Components and Economic Interpretation

### 발표 스크립트

이 표는 여섯 투자항목을 flow와 stock, 그리고 경제적 역할에 따라 정리한 것입니다.

CAPEX는 physical expansion의 현재 flow입니다. 성장기회를 반영할 수 있지만 동시에 overinvestment 가능성도 있습니다.

TANG는 accumulated physical asset stock으로, collateral과 redeployability의 장점이 있지만 동시에 mature, asset-heavy business profile을 반영할 수 있습니다.

R&D는 knowledge creation을 위한 flow이며 미래 growth option을 나타낼 수 있지만 프로젝트 실패 가능성이 높고 quality를 관찰하기 어렵습니다.

INTANG는 recognised intangible asset stock입니다. R&D와 달리 balance sheet에 recognition된 assets라는 점에서 accounting visibility가 더 높지만, accounting rules 때문에 internally generated intangibles의 상당 부분은 여전히 포함되지 않습니다.

SG&A는 organisational capital proxy로 사용합니다. 브랜드, 고객관계, distribution system, managerial routines를 포함할 수 있지만 routine expense와 분리되지 않는다는 측정상의 한계가 있습니다.

FIN은 productive asset이라기보다 financial flexibility component입니다. Future investment를 지원할 수 있는 liquidity buffer일 수도 있고, agency slack 또는 financialisation을 나타낼 수도 있습니다.

이러한 ambiguity 때문에 coefficient sign이나 ranking을 단순하게 사전에 결정하기 어렵습니다.

### 질문 대비

**Q. SG&A를 organisational capital로 쓰는 것은 너무 noisy하지 않습니까?**

> 맞습니다. 그래서 SG&A 전체를 organisational capital이라고 주장하지 않습니다. SG&A는 organisational/customer/commercialisation investment의 noisy proxy이며 routine overhead도 포함합니다.

**Q. FIN에 구체적으로 어떤 계정을 포함합니까?**

> 이 부분은 실제 변수정의표에서 정확한 account mapping을 설명할 수 있어야 합니다. 발표 전 반드시 dataset code와 variable appendix에서 FIN definition을 다시 확인해 두는 것이 좋습니다.

**Q. Flow와 stock을 coefficient magnitude로 비교해도 됩니까?**

> 원단위 비교는 어렵습니다. 그래서 focal components를 z-standardise합니다. 다만 standardisation이 경제적 개념을 동일하게 만드는 것은 아니므로 "same productive unit"로 해석하면 안 됩니다.

---

# Slide 9. 은행 불확실성 변수의 구성

## Construction of Banking Uncertainty Measures

### 발표 스크립트

은행 불확실성 extension에서는 두 개의 annual banking-sector uncertainty measure를 사용합니다.

첫 번째는 **AUNC**, bank asset-growth uncertainty입니다.

두 번째는 **FUNC**, bank funding-growth uncertainty입니다.

구성방식은 Buch et al. 계열의 disaggregate uncertainty approach를 따릅니다.

먼저 각 은행의 asset growth 또는 funding growth를 bank fixed effects와 year fixed effects에 대해 residualise합니다.

그 다음 각 연도에서 bank-specific residual의 cross-sectional standard deviation을 계산합니다.

따라서 AUNC나 FUNC는 은행산업 전체의 단순 성장률이나 volatility가 아니라, 해당 연도에 은행별 unexpected outcomes가 얼마나 서로 분산되어 있는지를 나타냅니다.

FUNC에서 broad funding은 deposit liabilities와 borrowed funding을 합한 개념을 사용합니다.

그 후 연도별 uncertainty index를 firm-year panel에 merge하고 standardise하여 각 투자항목과 interaction을 구성합니다.

여기서 중요한 제한이 있습니다.

AUNC와 FUNC는 같은 연도의 모든 기업이 공유하는 annual series입니다. 따라서 year fixed effects를 넣으면 uncertainty의 main effect 자체는 흡수됩니다.

실제로 관심있는 것은 uncertainty level에 따라 firm-level investment component의 valuation slope가 달라지는가입니다.

따라서 이것은 external banking shock의 causal effect가 아니라 **conditional valuation pattern**으로 해석합니다.

### 질문 대비

**Q. Uncertainty와 volatility의 차이가 무엇입니까?**

> 여기서는 aggregate time-series volatility가 아니라 bank-specific unexpected outcome의 cross-sectional dispersion입니다.

**Q. Year FE가 있으면 AUNC/FUNC를 어떻게 식별합니까?**

> Main effect는 식별되지 않습니다. Component × uncertainty interaction은 같은 연도 내 firm별 component exposure 차이를 통해 식별됩니다.

**Q. Annual observation 수가 14개 정도밖에 안 되는데 문제가 아닙니까?**

> 중요한 limitation입니다. Common year-specific shocks에 민감할 수 있기 때문에 causal claim을 피하고 extension으로 제한적으로 해석합니다.

**Q. 왜 policy uncertainty가 아니라 banking uncertainty입니까?**

> 본 연구의 mechanism은 external financing environment, collateral, liquidity, continuation financing과 더 직접적으로 연결되기 때문입니다.

Draft 역시 banking uncertainty 분석을 total investment response가 아니라 component-specific valuation coefficient가 달라지는지를 보는 extension으로 정의합니다.

---

# Slide 10. 실증전략: Standalone vs Incremental

## Empirical Strategy: Separate vs Simultaneous Estimands

### 발표 스크립트

이 slide가 방법론적으로 가장 중요합니다.

Separate-treatment DML에서는 여섯 component를 한 번에 하나씩 treatment로 사용합니다.

예를 들어 R&D model에서는 R&D가 focal treatment이고, firm characteristics와 fixed effects를 nuisance controls로 사용합니다.

따라서 여기서 얻는 coefficient는 **standalone valuation relevance**입니다.

반면 simultaneous-treatment DML에서는 CAPEX, TANG, INTANG, R&D, SG&A, FIN의 전체 vector를 동시에 treatment set으로 사용합니다.

따라서 R&D coefficient는 다른 다섯 component를 조건부로 고정했을 때 남는 R&D의 incremental valuation information을 나타냅니다.

중요한 것은 simultaneous model을 단순히 "더 많은 controls를 넣은 regression"으로 설명하지 않는 것입니다.

제가 강조하고 싶은 것은 **research question 자체가 달라진다**는 것입니다.

Separate model은 "이 component가 informative한가?"

Simultaneous model은 "나머지 investment mix를 관찰한 이후에도 이 component가 informative한가?"

입니다.

### 질문 대비

**Q. Other components는 control입니까 treatment입니까?**

> Simultaneous DML에서는 여섯 component 모두 target treatment vector에 포함됩니다. 일반 controls와 구분해야 합니다.

**Q. 왜 component 간 multicollinearity 문제는 없습니까?**

> Correlation은 존재하지만 VIF가 conventional danger threshold보다 낮습니다. 특히 maximum VIF가 약 2 정도이므로 severe multicollinearity라고 보기는 어렵습니다.

**Q. Joint model coefficient를 ceteris paribus effect라고 불러도 됩니까?**

> Statistical conditional coefficient라는 의미에서는 가능하지만 causal ceteris paribus effect라는 표현은 피합니다.

---

# Slide 11. DML 구현 세부사항

## Double Machine Learning Implementation

### 발표 스크립트

DML은 Chernozhukov et al.의 partially linear framework를 사용합니다.

핵심은 outcome과 treatment를 observed controls에 대해 각각 flexible machine learning model로 residualise한 후, orthogonalised residual variation을 이용하여 target coefficient를 추정하는 것입니다.

Nuisance learner는 baseline에서 LightGBM을 사용합니다.

Cross-fitting은 5-fold이며 5회 반복합니다.

중요한 점은 fold assignment를 observation 단위가 아니라 **firm 단위로 block**했다는 것입니다.

즉, 동일 기업의 2018년 observation이 training set에 있고 2019년 observation이 test set에 들어가는 식의 information leakage를 방지합니다.

Inference는 firm-level multiplier bootstrap을 1,000회 수행합니다.

또한 simultaneous model에서 여러 개의 investment coefficients를 동시에 검정하므로 Romano-Wolf stepdown, Benjamini-Yekutieli, Bonferroni adjusted p-values도 보고합니다.

### 질문 대비

**Q. 왜 LightGBM을 선택했습니까?**

> Nonlinearities와 interactions를 flexible하게 포착하면서 tabular data에서 prediction performance가 안정적이기 때문입니다. 다만 learner dependence 우려를 줄이기 위해 XGBoost와 Lasso를 robustness에서 사용합니다.

**Q. DML에서 nuisance prediction accuracy가 좋으면 coefficient도 무조건 unbiased합니까?**

> 아닙니다. Orthogonal score는 nuisance estimation error에 대한 first-order sensitivity를 줄이지만 identification assumptions가 필요하며 unobserved confounding을 제거하지는 않습니다.

**Q. 왜 random K-fold가 아니라 firm-blocking입니까?**

> Panel dependence와 leakage 때문입니다. 동일 firm의 다른 연도 정보를 train/test에 동시에 배치하면 nuisance prediction이 비현실적으로 쉬워질 수 있습니다.

**Q. Clustered standard errors 대신 bootstrap을 쓰는 이유는?**

> DML influence scores를 이용해 firm-level dependence를 반영한 multiplier-bootstrap variance를 사용하고 있습니다.

Draft의 DML design도 firm-blocked cross-fitting과 firm-level multiplier bootstrap을 명시합니다.

---

# Slide 12. 표준화와 계수 해석

## Standardisation and Coefficient Interpretation

### 발표 스크립트

여섯 focal investment components는 각 estimation sample 내에서 z-standardise합니다.

따라서 coefficient magnitude를 서로 비교할 수 있습니다.

예를 들어 simultaneous DML에서 R&D coefficient가 약 0.145라면, R&D intensity가 해당 estimation sample에서 1 standard deviation 증가할 때 Tobin's Q가 약 0.145 높게 나타나는 conditional association으로 해석합니다.

Outcome인 Tobin's Q는 standardise하지 않습니다.

따라서 coefficient는 Tobin's Q의 actual unit으로 표현됩니다.

Subgroup analysis에서는 각 subgroup 내 standard deviation을 사용합니다.

이 점은 매우 중요합니다.

예를 들어 KOSPI의 R&D coefficient와 KOSDAQ의 R&D coefficient는 각각 해당 group에서의 1-SD change에 대한 coefficient입니다. 따라서 원단위로 정확히 동일한 R&D 증가량에 대한 효과를 비교하는 것은 아닙니다.

Bank uncertainty와 interaction도 standardised scale을 사용합니다.

따라서 main component coefficient는 mean uncertainty 수준에서의 component valuation relevance로 해석되고, interaction coefficient는 uncertainty가 1-SD 증가할 때 component slope가 얼마나 변하는지를 나타냅니다.

### 질문 대비

**Q. 왜 standardise했습니까?**

> CAPEX, TANG, R&D 등 원래 분포와 scale이 크게 다르기 때문에 coefficient magnitude의 descriptive comparison을 가능하게 하기 위해서입니다.

**Q. Standardised coefficient가 economic significance를 의미합니까?**

> 어느 정도 비교에는 유용하지만 경제적 비용이나 실제 dollar investment를 직접 반영하지 않습니다. 따라서 economic significance를 논할 때는 sample mean과 원분포를 함께 봐야 합니다.

**Q. 0.145가 큰 효과인가요?**

> Sample mean TobinQ가 약 1.363이므로 약 10.6%에 해당하는 association입니다. 다만 causal economic effect로 해석하지 않습니다. Draft도 이 magnitude comparison을 제시합니다.

---

# Slide 13. 기준모형: Industry + Year FE

## Baseline Fixed-Effects Benchmark

### 발표 스크립트

먼저 conventional benchmark로 industry와 year fixed effects를 포함한 linear regression을 추정했습니다.

Separate FE 결과를 보면 CAPEX, INTANG, R&D, SG&A, FIN은 Tobin's Q와 positive association을 보이며, TANG는 통계적으로 유의하지 않습니다.

표준화된 magnitude를 보면 R&D가 약 0.238로 가장 크고, 그 다음 SG&A, INTANG, CAPEX, FIN 순입니다.

Joint FE에서도 broad pattern은 유지됩니다.

R&D가 약 0.203으로 가장 크고, INTANG과 SG&A가 그 뒤를 따릅니다.

TANG는 0.028 정도로 매우 작고 10% 수준에서만 유의합니다.

이 결과는 joint investment composition의 중요성이 단순히 machine learning에서만 나타나는 것이 아니라 linear benchmark에서도 어느 정도 보인다는 점을 보여줍니다.

다만 linear regression은 controls와 outcome 및 treatment 사이의 relationship을 선형적으로 제한합니다.

따라서 이후 DML에서는 observed confounding relationship을 더 flexible하게 조정한 후 동일한 component pattern이 유지되는지 확인합니다.

### 질문 대비

**Q. OLS와 DML 결과가 비슷하면 DML이 필요한가요?**

> 결과가 완전히 뒤집히지 않는다는 것이 오히려 reassuring합니다. DML의 목적은 새로운 sign을 만들기보다는 linear specification에 대한 dependence를 줄이는 것입니다.

**Q. 왜 control coefficient는 발표하지 않습니까?**

> 연구의 focal estimand는 six investment components입니다. Control coefficients는 structural interpretation 대상이 아니므로 main presentation에서는 생략했습니다.

---

# Slide 14. 개별 DML 결과

## Separate-Treatment DML Estimates

### 발표 스크립트

Table 4는 각 investment component를 개별 treatment로 추정한 separate-treatment DML 결과입니다.

가장 큰 coefficient는 R&D입니다.

Tobin's Q 기준 약 0.168이고, industry-adjusted Tobin's Q에서도 거의 동일합니다.

INTANG는 약 0.130으로 두 번째로 큰 coefficient입니다.

SG&A도 약 0.111로 positive and statistically significant합니다.

CAPEX는 약 0.049로 positive하지만 상대적으로 작습니다.

FIN은 약 0.025로 positive하지만 Tobin's Q에서는 statistically insignificant하고 industry-adjusted Q에서만 약한 significance를 보입니다.

TANG는 약 −0.025이며 statistically insignificant합니다.

따라서 standalone perspective에서는 knowledge-oriented components, 특히 R&D와 recognised intangible assets가 가장 강한 valuation signals를 보입니다.

그러나 이 결과만으로 각 component의 independent information을 판단할 수는 없습니다.

다음 slide의 simultaneous model이 핵심입니다.

### 질문 대비

**Q. R&D가 큰 이유를 causal productivity로 설명해도 됩니까?**

> 아니요. R&D coefficient는 innovation, growth opportunities, investor expectations 등 다양한 정보를 반영할 수 있습니다. 현재 결과는 valuation relevance입니다.

**Q. TANG가 negative인 이유는 무엇입니까?**

> Asset-heavy mature profile 또는 낮은 growth options와 연관될 수 있지만 coefficient가 insignificant하므로 negative effect라고 주장하지 않습니다.

**Q. FIN이 왜 insignificant입니까?**

> Standalone model에서 FIN은 tangibility와 같은 correlated investment composition을 함께 반영하기 때문일 수 있습니다. 이 부분이 simultaneous model에서 흥미롭게 바뀝니다.

---

# Slide 15. 핵심표: 동시처리 DML

## Core Result: Simultaneous-Treatment DML

### 발표 스크립트

이 Table 5가 연구의 가장 중요한 결과입니다.

여섯 investment components를 하나의 DML system에 동시에 포함했습니다.

먼저 R&D는 0.1445이고 highly significant합니다.

INTANG는 0.1326으로 역시 highly significant합니다.

SG&A는 0.0718, FIN은 0.0509, CAPEX는 0.0497이며 모두 positive and significant합니다.

반면 TANG는 0.0190으로 작고 statistically insignificant합니다.

Industry-adjusted Tobin's Q를 사용해도 결과는 거의 동일합니다.

Multiple-testing correction에서도 R&D, INTANG, SG&A, CAPEX는 매우 안정적이고 FIN도 Romano-Wolf p-value 0.006으로 유의합니다.

따라서 main result는 다음과 같습니다.

첫째, R&D와 INTANG는 다른 investment margins를 동시에 고려한 후에도 가장 큰 positive valuation coefficients를 유지합니다.

둘째, FIN은 standalone에서는 weak했지만 simultaneous model에서는 positive and significant해집니다.

셋째, TANG는 다른 components를 관찰한 후에는 independent valuation relevance를 거의 보여주지 않습니다.

이 결과는 H1a와 H1b를 명확하게 지지하며, H1c와도 일관된 coefficient pattern을 보여줍니다.

다만 모든 pairwise coefficient differences를 formal test한 것은 아니므로 R&D가 statistically 모든 component보다 크다고 주장하지는 않습니다.

실제 simultaneous estimates와 multiple-testing 결과는 draft Table 5와 일치합니다.

### 질문 대비

**Q. 왜 FIN coefficient가 오히려 커집니까?**

> FIN이 standalone model에서는 low tangibility firms의 valuation profile까지 함께 반영할 수 있기 때문입니다. 다른 components를 control하면 financial flexibility의 incremental signal이 더 명확해질 가능성이 있습니다.

**Q. TANG가 insignificant하다는 것은 tangible assets가 가치가 없다는 의미입니까?**

> 아닙니다. Collateral이나 operating capacity로서 경제적 가치가 없다는 뜻이 아니라, 다른 observed investment components를 조건부로 했을 때 별도의 market valuation information이 크지 않다는 뜻입니다.

**Q. R&D 0.1445와 INTANG 0.1326은 statistically different합니까?**

> 해당 pairwise difference를 별도로 검정하지 않았습니다. 따라서 ranking은 point-estimate pattern으로만 설명합니다.

---

# Slide 16. Standalone → Incremental: 계수 변화

## What the Coefficient Changes Mean

### 발표 스크립트

Separate와 simultaneous results를 직접 비교하면 investment composition이 왜 중요한지 더 명확하게 보입니다.

R&D는 0.168에서 0.145로 감소합니다.

즉, standalone R&D coefficient의 일부는 다른 knowledge-related investments와 공유된 정보를 반영하지만, 상당한 independent valuation relevance가 남습니다.

INTANG는 0.130에서 0.133으로 거의 변화하지 않습니다.

따라서 recognised intangible assets의 valuation signal은 investment mix를 통제해도 매우 안정적입니다.

SG&A는 0.111에서 0.072로 감소합니다.

이는 standalone SG&A premium의 일부가 R&D 또는 INTANG와 공유되는 information을 포함할 가능성을 보여줍니다.

CAPEX는 0.049에서 0.050으로 거의 변하지 않습니다.

FIN은 가장 흥미로운 변화입니다.

0.025에서 0.051로 약 두 배가 되고 statistically significant해집니다.

반면 TANG는 −0.025에서 0.019로 이동하면서 negative signal이 사라집니다.

따라서 이 연구의 핵심은 어느 coefficient가 가장 큰가만이 아니라, **investment composition을 관찰했을 때 coefficient가 어떻게 재구성되는가**입니다.

Draft에서도 SG&A attenuation, FIN increase, TANG negative coefficient disappearance를 main interpretation으로 설명합니다.

### 질문 대비

**Q. Coefficient increase가 suppression effect입니까?**

> 통계적으로 suppression-like pattern으로 볼 수는 있지만, 특정 suppressor variable 하나를 formal identification한 것은 아닙니다. 따라서 correlated investment composition을 통제하면서 FIN signal이 clearer해졌다고 표현하는 것이 안전합니다.

**Q. 왜 INTANG는 거의 안 변합니까?**

> INTANG가 다른 components와 공유되지 않는 독립적인 accounting visibility 또는 accumulated intangible information을 상당히 보유할 가능성이 있습니다.

---

# Slide 17. 경제적 해석

## Why FIN Rises and SG&A Falls

### 발표 스크립트

FIN의 coefficient movement를 조금 더 자세히 보면 Table 2에서 FIN과 TANG의 correlation이 약 −0.453입니다.

즉, tangible assets가 적은 기업이 상대적으로 더 많은 financial assets를 보유하는 경향이 있습니다.

Standalone FIN regression에서는 FIN coefficient가 financial flexibility뿐 아니라 low-tangibility firm의 valuation profile까지 일부 반영할 수 있습니다.

TANG와 다른 investment components를 동시에 포함하면 이 shared composition effect가 분리되고, FIN의 financial flexibility signal이 더 뚜렷해질 수 있습니다.

SG&A는 반대 방향입니다.

SG&A는 R&D와 commercialisation 과정에서 함께 움직일 수 있고, INTANG와도 organisational/intangible capital information을 공유할 수 있습니다.

따라서 standalone coefficient가 joint model에서 감소하는 것은 이러한 shared information이 일부 제거된 결과로 해석할 수 있습니다.

TANG의 경우 standalone에서는 negative coefficient이지만 joint model에서는 0에 가까워집니다.

이는 apparent tangibility discount의 일부가 실제로 firm's intangible orientation이나 다른 investment mix와의 상관을 반영했을 가능성을 보여줍니다.

다만 이 세 가지 설명은 direct mediation test가 아니라 economic interpretation입니다.

### 질문 대비

**Q. FIN을 financial flexibility로 해석하기에는 agency slack 가능성도 있지 않습니까?**

> 맞습니다. Aggregate FIN은 pure financial-flexibility measure가 아닙니다. 그렇기 때문에 governance heterogeneity와 Chaebol split이 중요합니다.

**Q. FIN-TANG correlation −0.453이면 substitution을 의미한다고 단정할 수 있습니까?**

> Descriptive balance-sheet substitution과 consistent하다고 표현합니다. Causal substitution을 식별한 것은 아닙니다.

Draft discussion도 이 distinction을 명시합니다.

---

# Slide 18. 강건성 1: Alternative Nuisance Learners

## Robustness I

### 발표 스크립트

첫 번째 robustness는 nuisance learner와 tuning에 대한 sensitivity입니다.

Baseline에서는 LightGBM을 사용했지만 depth와 learning rate를 바꾼 세 가지 specification, XGBoost, 그리고 Lasso를 추가로 사용했습니다.

결과를 보면 INTANG, R&D, SG&A, CAPEX는 모든 주요 specification에서 positive and significant합니다.

R&D는 learner에 따라 대략 0.148에서 0.204 사이이며 항상 가장 큰 coefficient 중 하나입니다.

INTANG는 약 0.131에서 0.133으로 매우 안정적입니다.

FIN 역시 모든 specification에서 positive하며 대부분 statistically significant합니다.

반면 TANG는 대부분 insignificant하고 Lasso에서만 약한 significance를 보입니다.

따라서 central component pattern이 특정 LightGBM parameter choice나 tree-based learner에만 의존하는 것은 아닙니다.

### 질문 대비

**Q. Lasso에서 coefficient가 더 큰 이유는 무엇입니까?**

> Linear nuisance model이 nonlinear learner보다 residualisation을 다르게 수행하기 때문일 수 있습니다. 중요한 것은 exact magnitude가 아니라 sign과 relative pattern의 robustness입니다.

**Q. 왜 Random Forest는 없습니까?**

> 추가할 수 있습니다. 현재 robustness는 boosting family와 regularised linear learner를 포함해 model-class variation을 제공하는 것이 목적입니다.

---

# Slide 19. 강건성 2: Leave-One-Component-Out

## Robustness II

### 발표 스크립트

두 번째 robustness는 investment composition의 sensitivity를 직접 확인하기 위한 leave-one-component-out test입니다.

Simultaneous model에서 한 번에 하나의 component를 제외하고 나머지 coefficients를 다시 추정했습니다.

R&D, INTANG, SG&A는 어떤 component를 제외해도 sign과 significance가 매우 안정적입니다.

반면 FIN과 TANG는 composition-sensitive합니다.

특히 INTANG를 제외하면 FIN coefficient가 0.0184로 감소하고 insignificant해집니다.

TANG 역시 INTANG를 제외할 때 −0.0396으로 negative and significant해집니다.

이 결과는 INTANG가 valuation system 내에서 TANG와 FIN interpretation에 중요한 역할을 한다는 것을 보여줍니다.

다만 이를 "INTANG가 FIN을 causally suppresses한다"라고 표현해서는 안 됩니다.

더 적절한 해석은 TANG와 FIN의 estimated valuation relevance가 firm's intangible orientation을 어떻게 model에 포함하는지에 민감하다는 것입니다.

### 질문 대비

**Q. 이 결과는 model instability 아닙니까?**

> 일부 instability가 존재한다는 점은 인정해야 합니다. 하지만 바로 그 sensitivity 자체가 paper의 composition argument와 연결됩니다. 동시에 identification robustness 관점에서는 limitation으로도 인정해야 합니다.

**Q. 왜 pairwise interaction을 직접 추정하지 않습니까?**

> Leave-one-out은 component dependence를 진단하는 sensitivity test입니다. Formal complementarity는 별도의 research design이 필요합니다.

---

# Slide 20. 강건성 3: DML-IV

## DML-IV and Identification Diagnostics

### 발표 스크립트

Main DML은 observed covariates에 대한 conditional exogeneity를 전제로 합니다.

Reverse causality concern을 추가로 확인하기 위해 DML-IV를 실시했습니다.

각 component에 대해 두 가지 instrument를 사용합니다.

첫 번째는 해당 component의 own second lag이고,

두 번째는 동일 industry-year 내 leave-one-out peer average입니다.

Lag instrument는 investment policy persistence를 활용합니다.

Peer instrument는 같은 industry-year 환경에서 공통적으로 나타나는 allocation condition을 활용하지만 focal firm 자체는 평균에서 제외합니다.

Anderson underidentification test의 p-value는 0.000으로 instrument relevance에 대한 evidence를 제공합니다.

Sargan p-value는 0.102이므로 conventional level에서 overidentifying restrictions를 reject하지 않습니다.

DML-IV coefficient도 main simultaneous DML과 방향적으로 유사합니다.

CAPEX, INTANG, R&D, SG&A는 positive and significant하며, FIN도 positive하지만 precision이 다소 낮습니다.

TANG는 multiple-testing correction 이후 강한 evidence를 제공하지 못합니다.

하지만 이 결과를 causal proof로 해석하지 않습니다.

특히 peer instrument는 industry-year common shocks와 reflection 문제에서 완전히 자유롭지 않습니다.

따라서 DML-IV는 reverse-causality concern에 대한 **supportive sensitivity evidence**입니다.

Draft도 exclusion restriction을 definitive하게 주장하지 않고 같은 방식으로 제한합니다.

### 질문 대비

**Q. Second lag가 valid instrument라는 근거는 무엇입니까?**

> Relevance는 investment persistence에서 나오지만 exclusion restriction은 강한 가정입니다. Firm value에 lagged investment가 직접 또는 persistent channel을 통해 영향을 줄 수 있으므로 definitive causal IV로 해석하지 않습니다.

**Q. Peer average는 reflection problem이 있지 않습니까?**

> 맞습니다. Leave-one-out은 mechanical own-observation correlation만 제거하며 common shocks와 endogenous peer effects 문제를 완전히 해결하지 못합니다.

**Q. 그러면 DML-IV를 빼는 것이 낫지 않습니까?**

> Main identification claim을 causal로 하지 않는다면 robustness appendix 또는 supportive evidence로 유지할 수 있습니다. 교수님 의견을 받아 본문/appendix 위치를 결정하는 것이 좋습니다.

---

# Slide 21. 시간에 따른 변화

## Intertemporal Dynamics

### 발표 스크립트

다음으로 sample을 네 개의 subperiod로 나누어 valuation coefficient가 시간에 따라 어떻게 달라지는지 살펴봤습니다.

가장 뚜렷한 변화는 R&D입니다.

2012–2015에는 coefficient가 거의 0에 가깝고 insignificant합니다.

2016–2018에는 0.142로 증가하고,

2019–2021에는 0.232로 가장 높습니다.

2022–2025에도 0.195로 높은 수준을 유지합니다.

FIN 역시 2019–2021에 0.086으로 가장 크게 나타납니다.

이는 disruption period에 financial flexibility가 더 중요했을 가능성과 consistent하지만, 이 분석이 COVID effect를 causal하게 식별하는 것은 아닙니다.

INTANG는 초기에 더 큰 coefficient를 보이고 시간이 지나면서 감소하는 pattern을 보입니다.

CAPEX는 비교적 안정적이고, TANG는 모든 subperiod에서 weak합니다.

따라서 temporal heterogeneity가 존재하지만 broad component pattern은 유지됩니다.

### 질문 대비

**Q. 왜 period를 이렇게 나눴습니까?**

> 경제환경 변화를 반영한 descriptive windows입니다. Structural break test로 정한 cutoffs는 아니므로 임의성 우려가 있습니다.

**Q. 2019–2021 R&D 증가를 COVID라고 해석해도 됩니까?**

> 직접적으로는 안 됩니다. 해당 기간에는 COVID 외에도 market valuation, technology cycle, sample composition 등이 변했습니다.

**Q. Formal coefficient difference test를 했습니까?**

> 현재 table은 period-specific estimates 중심의 descriptive heterogeneity입니다. Formal difference tests를 추가하면 stronger evidence가 될 수 있습니다.

---

# Slide 22. 기업환경별 이질성: 전체 표

## Heterogeneity Across Market, Governance, and Technology

### 발표 스크립트

Table 10은 KOSPI와 KOSDAQ, Chaebol과 non-Chaebol, 그리고 high-tech와 low-tech의 simultaneous DML estimates를 비교합니다.

이 table을 해석할 때 가장 중요한 원칙은 두 가지를 구분하는 것입니다.

첫 번째는 각 subgroup 내 coefficient가 statistically significant한가이고,

두 번째는 두 subgroup coefficient의 **difference 자체가 significant한가**입니다.

예를 들어 KOSPI와 KOSDAQ에서 각각 coefficient가 significant하다고 해서 두 coefficient가 statistically different하다는 뜻은 아닙니다.

따라서 이 table에서는 반드시 Diff column을 중심으로 해석해야 합니다.

다음 slide에서 statistically more informative한 세 가지 pattern만 정리하겠습니다.

### 질문 대비

**Q. Subgroup estimates는 서로 independent하다고 가정합니까?**

> Between-group difference SE는 disjoint subgroup estimators의 independence를 전제로 계산합니다.

**Q. Group-specific standardisation이 difference test에 영향을 주지 않습니까?**

> 맞습니다. 각 group의 1-SD change에 대한 coefficient difference이므로 identical raw-unit treatment effect difference는 아닙니다. Interpretation에 이 제한을 명시해야 합니다.

---

# Slide 23. 이질성의 핵심 해석

## Key Heterogeneity Findings

### 발표 스크립트

첫 번째로 KOSPI와 KOSDAQ에서는 INTANG의 차이가 가장 명확합니다.

KOSPI의 INTANG coefficient는 약 0.040인 반면 KOSDAQ에서는 약 0.175이며 difference는 −0.135로 statistically significant합니다.

이 결과는 상대적으로 information opacity가 높은 KOSDAQ 환경에서 accounting-recognised intangible assets가 더 informative할 가능성과 consistent합니다.

하지만 disclosure mechanism을 직접 test한 것은 아닙니다.

두 번째는 Chaebol과 non-Chaebol의 FIN 차이입니다.

Chaebol firm에서는 FIN coefficient가 약 −0.036이고 insignificant한 반면, non-Chaebol에서는 0.064로 positive and significant합니다.

Between-group difference도 −0.100으로 significant합니다.

이는 Chaebol internal capital market가 independent financial asset holdings의 marginal flexibility value를 낮출 가능성과 consistent합니다.

세 번째는 technology split입니다.

TANG는 high-tech에서 더 negative하고 difference가 −0.087로 significant합니다.

SG&A difference는 positive이지만 10% 수준의 marginal evidence입니다.

R&D는 low-tech에서 point estimate가 더 크지만 between-group difference는 statistically significant하지 않습니다.

따라서 "R&D is more valuable in low-tech"라고 강하게 주장해서는 안 됩니다.

Draft 역시 이 subgroup mechanisms를 interpretations rather than mediation tests로 제한합니다.

### 질문 대비

**Q. 왜 KOSDAQ에서 INTANG가 더 높습니까?**

> Accounting visibility, growth orientation, investor attention 차이 등이 possible mechanism입니다. 하지만 direct mechanism test는 아닙니다.

**Q. Chaebol FIN 결과가 internal capital market 때문이라고 어떻게 압니까?**

> 알 수 없습니다. Pattern이 해당 theory와 consistent할 뿐입니다. Group-level transfers를 직접 측정하지 않았습니다.

**Q. High-tech R&D coefficient가 오히려 작으면 theory와 모순 아닙니까?**

> 꼭 그렇지 않습니다. High-tech에서 R&D는 survival requirement 또는 baseline expenditure일 수 있어 marginal signalling content가 작을 수 있습니다. 그러나 difference가 insignificant하므로 strong interpretation은 피합니다.

---

# Slide 24. 은행 불확실성: 전체 결과

## Banking Uncertainty and Component-Specific Valuation

### 발표 스크립트

Table 11은 AUNC와 FUNC를 사용한 simultaneous DML moderation results입니다.

각 model에는 여섯 component main effects와 여섯 component × uncertainty interactions, 총 12개의 target coefficients가 포함됩니다.

Main component coefficients는 uncertainty가 standardised mean, 즉 0일 때의 valuation relevance입니다.

Interaction coefficient는 uncertainty가 1 standard deviation 증가할 때 해당 component slope가 얼마나 변하는지를 나타냅니다.

전체 결과에서 대부분의 interaction은 statistically precise하지 않습니다.

CAPEX, TANG, INTANG, FIN의 interactions는 AUNC와 FUNC 모두에서 강한 evidence를 보이지 않습니다.

반면 R&D와 SG&A에서 일관된 interaction pattern이 나타납니다.

따라서 banking uncertainty가 모든 corporate investment components를 일괄적으로 repricing한다고 해석할 수는 없습니다.

다음 slide에서는 이 두 핵심 결과만 강조하겠습니다.

### 질문 대비

**Q. 12개의 interaction을 동시에 보는데 multiple testing correction은 했습니까?**

> Main table significance는 raw p-values 기준입니다. 이 점은 limitation이며, joint correction을 추가하면 robustness가 강화됩니다.

**Q. Main coefficient가 Table 5와 다른 이유는 무엇입니까?**

> Moderation model에서 component main effect는 uncertainty=0, 즉 mean uncertainty에서의 conditional coefficient이고 model specification도 interactions를 포함하기 때문입니다.

---

# Slide 25. 은행 불확실성: 핵심 상호작용

## The Two Main Interaction Patterns

### 발표 스크립트

첫 번째 핵심 결과는 R&D interaction입니다.

R&D × AUNC는 약 −0.065이고,

R&D × FUNC는 약 −0.058입니다.

Industry-adjusted Q에서도 동일하게 negative coefficient가 나타납니다.

즉, banking uncertainty가 높은 연도에는 current R&D expenditure의 conditional valuation relevance가 약해지는 pattern이 관찰됩니다.

경제적으로는 R&D가 collateral value가 낮고 payoff uncertainty가 높기 때문에, external financing environment가 불안정할 때 시장이 future continuation value를 더 보수적으로 평가할 가능성이 있습니다.

두 번째는 SG&A입니다.

SG&A × AUNC는 약 +0.068,

SG&A × FUNC는 약 +0.073으로 positive합니다.

이는 uncertainty가 높은 환경에서 broad organisational 또는 commercialisation expenditure가 더 높은 conditional valuation relevance를 보이는 pattern입니다.

다만 여기서 "SG&A creates resilience"라고 주장하면 너무 강합니다.

SG&A는 broad expenditure measure이고 organisational resilience를 직접 측정하지 않습니다.

따라서 현재 가장 안전한 conclusion은:

> Banking uncertainty is associated with weaker conditional valuation relevance for R&D and stronger relevance for SG&A.

입니다.

CAPEX, TANG, INTANG, FIN에는 precise interaction이 없기 때문에 general repricing story는 지지되지 않습니다. Draft도 결과가 R&D와 SGA에 집중된다고 명시합니다.

### 질문 대비

**Q. 왜 uncertainty가 높을 때 R&D valuation이 낮아집니까?**

> Financing continuation risk, opacity, weak collateralizability가 possible channels입니다. 그러나 mechanism은 직접 식별하지 않았습니다.

**Q. 왜 SG&A는 오히려 높아집니까?**

> Commercialisation capability, customer relationships, organisational flexibility 등이 possible explanation이지만 현재 SG&A measure로는 구체적인 mechanism을 구분할 수 없습니다.

**Q. Year FE가 있으면 macro shocks는 모두 통제되는 것 아닌가요?**

> Additive common year shocks는 통제하지만 firm component exposure와 year condition이 상호작용하는 다른 omitted mechanisms까지 제거하는 것은 아닙니다.

---

# Slide 26. 연구의 기여와 해석범위

## Contribution and Scope of Interpretation

### 발표 스크립트

이 연구의 contribution은 네 가지로 정리할 수 있습니다.

첫째, 여섯 accounting-based investment components를 하나의 valuation framework에서 동시에 분석합니다.

둘째, standalone valuation relevance와 incremental valuation relevance를 명확하게 구분합니다.

이 distinction을 통해 single-component regression coefficient가 broader investment composition을 얼마나 반영할 수 있는지 보여줍니다.

셋째, DML을 사용하여 valuation과 investment choices가 firm characteristics와 nonlinear하게 연결될 가능성을 flexible하게 조정합니다.

넷째, 한국 시장이라는 institutional setting에서 KOSPI/KOSDAQ, Chaebol affiliation, technology orientation, subperiod, banking conditions에 따른 heterogeneity를 문서화합니다.

하지만 해석범위를 명확히 해야 합니다.

이 연구는 structural marginal q를 추정하지 않습니다.

또한 DML coefficient는 exogenous policy intervention의 treatment effect가 아닙니다.

가장 정확한 표현은:

> component-specific valuation relevance after flexible adjustment for observed confounders and the broader investment mix

입니다.

### 질문 대비

**Q. 가장 중요한 contribution 하나만 남긴다면?**

> Standalone versus incremental valuation distinction입니다.

**Q. DML 자체가 contribution입니까?**

> Methodological novelty만을 main contribution으로 두는 것은 약합니다. DML은 investment-composition question을 더 flexible하게 estimate하기 위한 enabling method로 보는 것이 좋습니다.

**Q. Korea contribution은 무엇입니까?**

> Market segment, Chaebol internal capital market, technology orientation이라는 distinct institutional heterogeneity를 하나의 valuation framework에서 비교할 수 있다는 점입니다.

Draft introduction도 네 가지 contribution을 이와 유사하게 정리합니다.

---

# Slide 27. 한계와 지도교수님께 논의드릴 쟁점

## Limitations and Questions for Supervisor Discussion

### 발표 스크립트

마지막으로 현재 연구의 한계와 교수님께 조언을 구하고 싶은 부분을 정리했습니다.

첫 번째 limitation은 DML이 observed confounders를 flexible하게 조정하지만 unobserved confounding을 제거하는 것은 아니라는 점입니다.

두 번째로 DML-IV를 추가했지만 lag와 peer instruments의 exclusion restriction은 contestable하기 때문에 causal identification으로 강하게 주장할 수 없습니다.

세 번째로 simultaneous DML은 component complementarity를 formal하게 검정하는 것이 아닙니다. 두 투자항목을 동시에 증가시킬 때 super-additive value가 발생하는지는 현재 design에서 알 수 없습니다.

네 번째로 banking uncertainty는 annual common series이기 때문에 time-series dimension이 제한적이고 common shocks에 민감할 수 있습니다.

다섯 번째로 flow와 stock components를 함께 비교한다는 conceptual asymmetry가 존재합니다.

마지막으로 Korean setting의 결과가 다른 capital-market institutions에 그대로 generalise된다고 볼 수 없습니다.

이와 관련해서 교수님께 특히 여섯 가지 부분에 대해 의견을 구하고 싶습니다.

첫째, banking uncertainty extension을 현재처럼 본문 contribution으로 유지할지 아니면 secondary extension으로 더 약하게 둘지입니다.

둘째, H1c를 directional hypothesis로 유지할지, 아니면 coefficient ordering을 사전에 강하게 예측하기 어렵다는 점을 고려해 RQ로 전환할지입니다.

셋째, DML-IV를 본문 robustness로 유지할지 appendix로 이동할지입니다.

넷째, flow와 stock 비교 framing을 더 보수적으로 조정할 필요가 있는지입니다.

다섯째, TANG와 INTANG의 composition sensitivity를 별도의 theoretical mechanism으로 확장할 가치가 있는지입니다.

여섯째, subperiod analysis를 어느 정도까지 main narrative에 포함할지입니다.

현재 제가 생각하는 이 연구의 가장 간결한 conclusion은 다음과 같습니다.

> 시장은 하나의 'investment'에 하나의 가격을 매기는 것이 아니라, 기업의 전체 **investment composition** 안에서 각 component를 조건부로 평가한다.

R&D와 recognised intangible assets는 가장 강하고 안정적인 incremental valuation relevance를 보이고, FIN과 TANG의 해석은 broader composition에 더 민감합니다.

은행 불확실성은 모든 component를 재평가시키는 것이 아니라 R&D와 SG&A에 집중된 conditional repricing pattern을 보여줍니다.

### 최종 질문 대비 — 교수님이 가장 물을 가능성이 높은 10개

**1. "그래서 이 논문의 핵심 dependent variable은 왜 Tobin's Q입니까?"**

> 기업의 investment composition에 대해 equity market이 부여하는 valuation information을 분석하는 것이 목적이기 때문입니다. Industry-adjusted Q도 parallel outcome으로 사용해 industry-year valuation level의 영향을 줄였습니다.

**2. "왜 firm fixed effects가 없습니까?"**

> 현재 design은 cross-sectional and temporal investment composition variation을 활용하면서 industry/year effects를 통제합니다. 다만 persistent omitted firm traits 우려가 있으므로 firm-FE robustness를 추가하는 것은 중요한 extension입니다.

**3. "DML이면 왜 causal이 아닙니까?"**

> DML은 nuisance estimation과 regularization bias 문제를 완화하지만 selection on observables assumption을 대체하지 않습니다. Unobserved confounding은 여전히 가능합니다.

**4. "여섯 treatment를 동시에 넣는 DML이 안정적입니까?"**

> VIF가 높지 않고 learner sensitivity에서도 broad pattern이 유지됩니다. 다만 leave-one-out에서 FIN과 TANG의 composition sensitivity가 확인되므로 이를 substantive result이자 limitation으로 함께 설명합니다.

**5. "R&D가 가장 크다는 것이 statistically 검정됐습니까?"**

> 모든 pairwise difference는 검정하지 않았으므로 universal ranking이라고 주장하지 않습니다. Reported point estimates에서 가장 크고 여러 specification에서 robust하다고만 설명합니다.

**6. "SG&A를 왜 investment라고 합니까?"**

> SG&A 전체를 investment라고 보지는 않습니다. Literature에 따라 organisational/customer/commercialisation capital을 포함할 수 있는 broad proxy로 사용하며 routine overhead contamination을 명시적으로 limitation으로 인정합니다.

**7. "FIN coefficient 증가가 정말 flexibility 때문입니까?"**

> 직접 식별된 mechanism은 아닙니다. FIN-TANG negative correlation과 joint coefficient movement가 financial-flexibility interpretation과 consistent하다는 수준입니다.

**8. "Banking uncertainty interaction은 annual observation이 너무 적지 않습니까?"**

> 맞습니다. 그래서 causal effect가 아니라 conditional pattern으로 제한하고 main contribution보다 extension으로 해석합니다.

**9. "왜 bank uncertainty가 R&D에는 negative, SG&A에는 positive입니까?"**

> R&D의 funding continuation risk와 weak collateralizability가 가능한 설명이고, SG&A는 commercialisation/organisational capacity를 반영할 가능성이 있습니다. 그러나 mechanism test가 아니므로 확정적으로 표현하지 않습니다.

**10. "이 연구에서 다음 단계 하나만 추가한다면 무엇입니까?"**

> 가장 우선순위가 높은 것은 main composition result의 identification을 강화하는 것입니다. 예를 들어 stronger firm-level fixed-effects robustness, alternative timing, treatment-specific instruments 또는 formal coefficient-difference tests를 추가하는 것이 bank-uncertainty extension을 더 확장하는 것보다 우선이라고 생각합니다.

---

# 발표 전체에서 피해야 할 표현

발표 중 다음과 같은 표현은 사용하지 않는 것이 좋습니다.

**피해야 함:**
“R&D causes firm value to increase by 0.145.”

**권장:**
“R&D is associated with a 0.145 higher Tobin's Q for a one-standard-deviation increase, conditional on the observed covariates and the other investment components.”

---

**피해야 함:**
“DML solves endogeneity.”

**권장:**
“DML flexibly adjusts for observed confounding and reduces sensitivity to nuisance-model functional form.”

---

**피해야 함:**
“The IV proves causality.”

**권장:**
“The DML-IV estimates provide supportive sensitivity evidence regarding reverse causality, subject to contestable exclusion restrictions.”

---

**피해야 함:**
“SG&A becomes more valuable because it creates resilience.”

**권장:**
“The positive SG&A interaction is consistent with stronger conditional valuation relevance in uncertain banking years, but the underlying mechanism is not directly identified.”

---

**피해야 함:**
“KOSDAQ values intangible assets more because of information opacity.”

**권장:**
“The stronger INTANG coefficient in KOSDAQ is consistent with an accounting-visibility interpretation, although the mechanism is not directly tested.”

---

# 발표 전 반드시 외워둘 숫자

1. **Sample:** 2,493 firms / 23,726 firm-years / 2012–2025
2. **Separate DML:** R&D 0.168 / INTANG 0.130 / SG&A 0.111
3. **Simultaneous DML:** R&D 0.145 / INTANG 0.133 / SG&A 0.072 / FIN 0.051 / CAPEX 0.050 / TANG 0.019
4. **FIN movement:** 0.025 → 0.051
5. **SG&A movement:** 0.111 → 0.072
6. **FIN–TANG correlation:** −0.453
7. **DML-IV:** Anderson p=0.000 / Sargan p=0.102
8. **KOSDAQ INTANG difference:** −0.1351*** when defined KOSPI − KOSDAQ
9. **Chaebol FIN difference:** −0.1000*** when defined Chaebol − Non-Chaebol
10. **Bank uncertainty:** R&D × AUNC −0.065**, R&D × FUNC −0.058**; SG&A × AUNC +0.068**, SG&A × FUNC +0.073**
