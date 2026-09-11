

# Abstract

Research on investment and firm value has largely developed around separate physical, knowledge-based, organisational, and liquidity-related margins, although firms choose among these uses jointly. This study asks whether the valuation information attributed to an investment margin in isolation remains when the broader investment composition is considered. The empirical design represents four theoretically distinct uses of corporate resources with six observable components in a panel of Korean listed non-financial firms. The analysis combines fixed-effects regressions with Double Machine Learning, allowing valuation and investment choices to vary flexibly with firm characteristics, governance, ownership, industry, and time. R&D and recognised intangible assets have larger estimated valuation coefficients than the physical components. Financial assets also contain positive valuation information once the components are considered jointly, whereas tangible assets have weaker and less stable valuation relevance. The main pattern is broadly stable across alternative nuisance learners, and DML-IV estimates using lagged and peer instruments point in the same direction; subgroup analyses identify more targeted differences across firm environments. Raw banking-uncertainty interactions suggest weaker conditional valuation relevance for R&D and stronger relevance for SG&A, but exact year-cluster sensitivity analysis does not retain the R&D interaction after multiplicity adjustment. The evidence indicates that firm value is related to how investment is composed, not only to its aggregate amount.

**Keywords:** firm valuation; investment components; corporate investment; intangible capital; financial flexibility; double machine learning; Korea.

# 1. Introduction

The relation between investment and firm value has traditionally been studied through aggregate physical capital and Tobin's q (Tobin, 1969; Hayashi, 1982). That aggregation becomes restrictive when firms employ several forms of capital with different adjustment costs and shadow values (Hayashi and Inoue, 1991; Chirinko, 1993). Subsequent research has shown that knowledge capital, organisational capability, and financial flexibility also affect growth opportunities and financing capacity (Lev and Sougiannis, 1996; Lev and Radhakrishnan, 2005; Peters and Taylor, 2017; Falato et al., 2022). The literature therefore gives a clear reason to move beyond total investment, but it has not produced a unified empirical account of how these investment margins are valued when firms choose them together.

Most empirical studies retain the boundaries of their respective literatures. Physical-investment studies focus on capital expenditure or tangible assets; innovation studies examine R&D, patents, or recognised intangibles; organisational-capital studies extract an investment signal from SG&A; and corporate-liquidity studies analyse cash or financial assets. These constructs are economically connected. R&D may require commercialisation expenditure, weak collateral associated with intangible investment may increase liquidity demand, and recognised intangibles may reflect earlier innovation or acquisitions (Lev and Sougiannis, 1996; Lev and Radhakrishnan, 2005; Opler et al., 1999; Hall et al., 2005; Faulkender and Wang, 2006). A coefficient estimated within one literature may therefore capture both the focal margin and information shared with other corporate uses of funds. The unresolved question is whether the valuation relevance assigned to each margin survives when the broader investment composition is observed.

The empirical component set is derived from this question. It represents four distinctions established in prior research: investment flows versus accumulated stocks, physical versus knowledge-based capital, organisational expenditure versus recognised assets, and productive investment versus financial flexibility. Six consistently observable accounting measures operationalise these distinctions: capital expenditure (CAPEX), tangible assets (TANG), recognised intangible assets (INTANG), R&D expenditure (RND), SG&A expenditure (SGA), and financial assets (FIN). This set is not intended as an exhaustive taxonomy of corporate investment. It is a theory-based empirical representation that permits the main margins studied separately in earlier work to be evaluated in one system. Firm value is measured by TobinQ and industry-adjusted TobinQ. The analysis compares standalone with incremental valuation relevance, then examines whether the estimates vary across firm environments and external banking conditions.

Korea provides a useful setting. Its listed market includes mature, asset-heavy firms and growth-oriented technology firms, allowing physical and intangible components to be compared within one market. Chaebol internal capital markets can affect financing constraints and the allocation of investment across affiliated firms (Shin and Park, 1999; Almeida et al., 2015). Concentrated control creates a separate governance concern because controlling shareholders may influence how corporate resources are deployed and valued (Claessens et al., 2000; Baek et al., 2004). Evidence on the Korea discount further shows that chaebol-related valuation effects vary across firms and over time (Lee et al., 2010; Ducret and Isakov, 2020, 2024). The final sample contains 2,493 listed firms and 23,726 firm-year observations from 2012 to 2025.

The condition of the banking system adds an external dimension to this valuation problem. Banking uncertainty, measured as the cross-sectional dispersion of unexpected bank-level outcomes, is associated with more cautious lending and higher external-financing costs (Buch et al., 2015; Huynh, 2025). These conditions need not affect all investment components equally. Tangible assets may preserve borrowing capacity through collateral (Almeida and Campello, 2007; Benmelech and Bergman, 2009), whereas R&D and other intangible expenditures can be harder for lenders and investors to assess (Myers and Majluf, 1984; Falato et al., 2022). Financial assets may instead provide liquidity when external funding becomes less dependable (Acharya et al., 2007; Gamba and Triantis, 2008). The moderation analysis therefore tests whether the conditional valuation relevance of the six components changes with uncertainty in bank asset growth and funding growth. Because these indices vary annually, the interactions are interpreted as conditional valuation patterns rather than causal effects of exogenous banking shocks.

The empirical strategy follows this distinction. I first estimate industry and year fixed-effects regressions. I then use Double Machine Learning to partial out observed confounders flexibly before estimating the investment-component coefficients, following the orthogonal-score and cross-fitting framework of Chernozhukov et al. (2018) and its DoubleML implementation by Bach et al. (2022). The separate-treatment estimates ask whether one component is informative on its own. The simultaneous estimates ask whether it remains informative after the other five components are included. Cross-fitting is blocked by firm and inference uses a firm-level multiplier bootstrap. Recent corporate-finance applications illustrate how DML can retain an interpretable target coefficient while allowing nonlinear nuisance relations (Movaghari, 2024; Movaghari et al., 2025). The estimates in this study are interpreted as valuation relevance after flexible adjustment for observed determinants of firm value and investment choices.

The fixed-effects estimates link CAPEX, INTANG, RND, SGA, and FIN positively to TobinQ, whereas TANG is statistically weak. In the separate DML estimates, R&D and recognised intangible assets have the largest coefficients. R&D remains the largest component in the simultaneous specification, and INTANG, CAPEX, SGA, and FIN retain positive incremental relevance. TANG contributes little once the other components are included. The robustness and DML-IV exercises give similar results. The estimates also differ by firm environment: recognised intangible assets are stronger among KOSDAQ firms, financial assets are stronger among non-chaebol firms, and R&D and FIN are larger in 2019–2021. In the banking-uncertainty extension, raw interaction estimates are negative for R&D and positive for SG&A, but the R&D result is not retained by a year-clustered sensitivity analysis.

The study makes four contributions. First, it brings investment margins that are usually studied in separate literatures into a common valuation framework and distinguishes standalone from incremental valuation relevance. The contribution is the joint comparison, not the claim that the six measures form a complete taxonomy of investment. Second, it connects multi-capital valuation theory with evidence on intangible capital, organisational expenditure, and financial flexibility, and examines whether banking uncertainty changes the conditional valuation of these margins. Third, it uses DML to accommodate nonlinear observed relationships between valuation, investment choices, and firm characteristics while retaining interpretable component coefficients. Finally, it documents how the estimated valuation pattern varies across Korean market segments, chaebol affiliation, technology orientation, subperiods, and banking conditions.

# 2. Literature review and theoretical background

## 2.1. Multi-capital valuation and the investment-composition problem

Valuation theory links firm value to the expected productivity of capital relative to its replacement cost (Tobin, 1969; Hayashi, 1982). In the canonical model, this relationship is expressed through a single, homogeneous capital stock, which yields a clean theoretical equivalence between average and marginal q. The aggregation is restrictive for empirical work, however, because firms commit resources to margins that differ in adjustment costs, uncertainty, redeployability, accounting visibility, and financing role. The market need not assign a common valuation weight to assets-in-place, growth options, organisational capability, and financial flexibility, and whether it does so is precisely the question this study addresses.

Multi-capital q-theory provides the formal basis for heterogeneous weights. Hayashi and Inoue (1991) show that with multiple capital goods carrying distinct shadow values, the relation between firm growth and q no longer collapses to a single capital measure. Chirinko (1993) demonstrates that heterogeneous capital inputs introduce valuation-relevant terms omitted from conventional single-capital specifications. Later research identifies specific margins hidden by aggregate capital: incorporating intangible capital improves the empirical performance of q-based valuation and investment equations (Peters and Taylor, 2017; Crouzet and Eberly, 2019), and the shift toward intangible capital alters collateral capacity and financing behaviour (Falato et al., 2022). Taken as a body of research, these studies motivate a joint test of distinct investment margins; they do not prescribe a single exhaustive list of components.

Three classes of mechanism connect the components and shape their information content. Real-options logic implies that investment components are valued for the opportunities they create or preserve rather than for current assets-in-place: R&D can open technological trajectories, recognised intangibles can represent partly realised options, SG&A can support commercialisation, and financial assets can preserve the option to fund future projects when external finance is costly (Myers, 1977; Dixit and Pindyck, 1994). Information asymmetry makes these components noisy rather than transparent signals of productive capital, particularly for R&D, organisational expenditure, and financial holdings (Akerlof, 1970; Myers and Majluf, 1984; Chan et al., 2001). Financing frictions and agency problems imply, finally, that the value of any component depends on the cost of external finance and on how investors expect managers or controlling shareholders to deploy capital (Fazzari et al., 1988; Jensen, 1986). Because the same mechanisms make the components jointly chosen as well as individually informative, a coefficient estimated for one margin in isolation blends its own valuation content with information shared across the investment mix.

This theory also disciplines the interpretation of the estimates. Hayashi's equivalence between average and marginal q rests on restrictive assumptions, and empirical Tobin's Q combines expected productivity, accounting measurement, market expectations, and omitted opportunities (Hayashi, 1982; Chirinko, 1993). The coefficients estimated in this study are therefore not structural marginal-q parameters from a fully specified dynamic investment model. The empirical objective is narrower, and the construct is stated once here: **valuation relevance** denotes the association between an observed investment component and firm value, conditional on observed firm characteristics, market conditions, and the other modelled investment components. The question is whether each component carries distinct valuation information once the broader investment composition is observed. Sections 2.2 and 2.3 develop the component mapping and the estimation problem that follows from it.

## 2.2. Heterogeneous investment components

Multi-capital q-theory explains why component-specific valuation weights can exist, but it neither selects accounting measures nor implies a universal ranking. The empirical mapping follows a sequence of four distinctions in the prior literature: current investment versus accumulated capital, physical versus knowledge-based capital, recognised assets versus internally developed capabilities, and productive assets versus liquidity retained for future opportunities. These distinctions identify four domains—physical capital, knowledge-based capital, organisational capability, and financial flexibility—from which the six observable proxies are selected. A measure enters the empirical set only when prior research gives it a distinct valuation interpretation, it can be constructed consistently across the full panel, and it does not merely duplicate another component. The resulting measures are therefore theory-based proxies within a common allocation system, rather than six independently selected candidates. Domains where theory yields an ambiguous prediction are precisely those the joint estimation is designed to resolve.

| Component | Type  | Main valuation role                                        | Main ambiguity                                       | Theoretical sign prediction                                                |
| --------- | ----- | ---------------------------------------------------------- | ---------------------------------------------------- | -------------------------------------------------------------------------- |
| CAPEX     | Flow  | Physical expansion and replacement investment              | Growth signal versus overinvestment                 | Positive-leaning ambiguous: expansion signal (+) versus overinvestment (−) |
| TANG      | Stock | Assets-in-place, collateral, and redeployability           | Collateral value versus mature low-growth profile   | Ambiguous: pledgeability (+) versus fewer growth options (−)              |
| RND       | Flow  | Innovation effort and growth options                       | Productive experimentation versus noisy failed projects | Positive, attenuated by signal noise                                    |
| INTANG    | Stock | Accounting-recognised intangible capital                   | Visibility versus selective recognition             | Positive, attenuated by recognition selectivity                            |
| SGA       | Flow  | Organisational, customer, and commercialisation investment | Durable capability versus routine overhead          | Ambiguous: capability (+) versus expensed overhead (−)                    |
| FIN       | Stock | Financial flexibility and investment capacity              | Liquidity buffer versus agency slack                | Ambiguous: option value of liquidity (+) versus discount for slack (−)    |

CAPEX and TANG both relate to physical capital, but they carry different signals. CAPEX is a current investment flow and can reveal management's view of expansion opportunities; market reactions to capital-expenditure announcements are often consistent with this interpretation (McConnell and Muscarella, 1985). TANG is an accumulated stock. Asset tangibility can relax financing constraints by increasing pledgeability (Almeida and Campello, 2007), while asset redeployability affects collateral value in debt markets (Benmelech and Bergman, 2009). Tangible assets also shape firms' observed capital structures (Titman and Wessels, 1988; Campello and Giambona, 2013). Collateral value is not the same as growth-option value. A high tangible-asset share can describe a mature, asset-heavy business with fewer scalable opportunities as value creation shifts toward intangible capital (Peters and Taylor, 2017; Crouzet and Eberly, 2019; Falato et al., 2022). CAPEX and TANG are therefore estimated separately before testing whether either remains relevant after the intangible, organisational, and financial components are included.

R&D is the clearest flow measure of knowledge investment. As a growth option, current R&D can give a firm access to future products, technologies, or processes whose value depends on subsequent states of the world (Myers, 1977). Capitalised R&D and patent citations contain information about future earnings and market value (Lev and Sougiannis, 1996; Hall et al., 2005), although the valuation of R&D differs across institutional settings (Hall and Oriani, 2006). It is also a noisy signal because investors observe spending but not project quality, appropriability, or the probability of failure (Chan et al., 2001). Innovation further requires tolerance for costly experimentation before payoffs become visible (Manso, 2011). Korean studies report R&D associations that differ with internal finance and firm size (Lee, 2012; Kwon, 2014), as well as with ownership, governance, and business-group context (Kang et al., 2019; Hong et al., 2023).

Recognised intangible assets (INTANG) occupy a different position in the investment process. Unlike RND, which records current innovation effort, INTANG records a stock that has cleared formal accounting recognition. Software capitalisation illustrates how recognition can make a class of intangible assets visible to investors (Aboody and Lev, 1998). Recognition remains selective because many internally generated intangibles fall outside conventional balance sheets (Lev and Zarowin, 1999; Lev, 2001). Peters and Taylor (2017) combine accounting data to construct a broader intangible-capital stock, while Ewens et al. (2024) use market prices to measure intangible capital. Evidence also distinguishes the valuation implications of intangible intensity (Intara and Suwansin, 2024) and different recognised intangible-asset categories (Dong and Doukas, 2025). These distinctions support estimating recognised intangible assets separately from current R&D expenditure.

SG&A is used as a proxy for organisational investment, but it must be interpreted cautiously. Some SG&A builds brand equity, customer relationships, distribution systems, managerial routines, and organisational infrastructure (Lev and Radhakrishnan, 2005; Eisfeldt and Papanikolaou, 2013; Banker et al., 2019). Some is ordinary overhead. Financial statements do not cleanly separate durable organisational investment from routine operating costs (Enache and Srivastava, 2018). Korean evidence also links non-R&D intangible investment, including advertising and human capital, to firm performance among smaller firms (Seo and Kim, 2020). SG&A is included because it lies between knowledge investment and operating expense.

FIN differs from the other components because it does not directly create operating output. Its valuation role concerns financial flexibility. Internal liquidity can fund future opportunities, absorb shocks, and reduce reliance on costly external finance (Opler et al., 1999; Almeida et al., 2004; Gamba and Triantis, 2008). Investors may also discount such holdings if they indicate agency slack, weak payout discipline, or a substitute for productive investment (Faulkender and Wang, 2006; Pinkowitz et al., 2006; Dittmar and Mahrt-Smith, 2007). The distinction is particularly relevant for intangible-intensive firms, where weak collateral capacity can increase demand for financial buffers (Falato et al., 2022). Financial assets may support precautionary holdings or crowd out innovation activity (Liu et al., 2024). FIN is therefore treated as a financial-flexibility component rather than a conventional productive-capital stock.

Prior studies therefore do not support reducing these components to one investment variable: physical investment can signal expansion or maturity, R&D can represent scalable growth options or noisy experimentation, recognised intangibles improve visibility only for assets that accounting rules recognise, SG&A mixes organisational investment with routine expense, and financial assets can represent flexibility or agency slack. The central empirical question is whether each component remains valuation-relevant when these distinct domains are modelled jointly.

## 2.3. Component interdependence, omitted confounding, and the empirical gap

The need for joint estimation follows from the way firms allocate capital. If each component has its own valuation content, estimating one component in isolation can be misleading whenever components are correlated and jointly chosen. A single-component regression may load not only on the focal component, but also on omitted signals from related components. This does not make earlier single-component studies uninformative. It means their coefficients are best interpreted as composite associations that combine direct valuation relevance with information shared across the investment mix.

Several mechanisms generate this interdependence. Growth options are distributed unevenly across the investment mix: R&D and related intangible expenditures can preserve future opportunities, whereas tangible assets are more closely tied to assets-in-place and collateral value (Myers, 1977; Almeida and Campello, 2007). Intangible intensity can also reduce debt capacity and increase demand for internal financial buffers (Falato et al., 2022). Firms must therefore allocate finite internal funds across competing uses when external finance is costly because of information asymmetry or financing constraints (Myers and Majluf, 1984; Fazzari et al., 1988). Agency conflict creates a separate allocation problem because managers may retain or invest resources even when their private incentives diverge from shareholder value (Jensen, 1986). Components may also be complements or substitutes: R&D may require SG&A to convert innovation into sales, recognised intangibles may reflect cumulative prior innovation, and financial assets may substitute for collateral when tangible assets are limited. Accounting visibility further distinguishes components because some expenditures are recognised as assets, some are expensed, and others are only imperfectly observed.

These mechanisms motivate the paper's central distinction. Standalone valuation relevance asks whether one component is associated with firm value when evaluated on its own. Incremental valuation relevance asks whether the same component remains informative after the other five observed components are included. The simultaneous model therefore changes the estimand from "is this component informative?" to "does this component retain valuation information conditional on the other modelled investment margins?"

This interdependence also creates an estimation challenge. Firm size, growth opportunities, financing constraints, governance, ownership, foreign monitoring, liquidity, leverage, payout policy, and firm age may predict both firm value and several investment choices in nonlinear ways. Fixed-effects regressions provide a transparent benchmark, but their linear adjustment structure may not fully capture these observed confounder relationships. DML addresses this specification concern by estimating the nuisance relations flexibly and recovering target coefficients from orthogonalised residual variation (Chernozhukov et al., 2018; Bach et al., 2022). Recent corporate-finance applications use the same logic to retain interpretable parameters while allowing machine learning to handle complex nuisance functions (Movaghari, 2024; Movaghari et al., 2025; Shi et al., 2025). In this study, DML is used for flexible observed-confounder adjustment, not because prediction is the final objective.

## 2.4. External banking uncertainty and the conditional valuation of investment components

Whether banking uncertainty changes the market's valuation of individual investment components remains largely undocumented. Existing studies link banking-sector uncertainty to credit supply and to the level of corporate investment (Buch et al., 2015; Huynh, 2025; Huynh and Phan, 2024, 2026), but they stop short of the valuation question: does uncertainty in bank asset or funding growth alter the valuation coefficient attached to a given investment margin once the broader investment composition is held constant? This section reviews the mechanisms that make such an interaction plausible and establishes the interpretive discipline under which the corresponding tests are conducted.

External banking uncertainty is treated here as a banking-sector condition rather than as policy uncertainty or firm-specific risk. Following Buch et al. (2015), it is measured by the cross-sectional dispersion of unexpected bank-level outcomes: wider dispersion indicates that bank asset or funding growth has become less predictable. Such dispersion does not mechanically imply an aggregate credit contraction, but it is associated with more cautious lending and higher external-financing costs (Soto, 2021; Huynh, 2025). Evidence from Vietnam further links banking uncertainty to lower corporate debt, identifying debt costs, trade-credit substitution, and reduced investment as transmission channels (Huynh and Phan, 2024). The construct therefore captures an external financing environment that can change both how firms fund investment and how investors assess its continuation.

Transmission is unlikely to be uniform across components. Intangible and physical capital differ in irreversibility, adjustment costs, and redeployability (Bontempi, 2016), and tangible investment is more sensitive to external finance whereas intangible investment relies more heavily on internal funds (Thum-Thysen et al., 2019). Consistent with this distinction, Huynh and Phan (2026) report that banking uncertainty contracts tangible investment substantially more than intangible investment, with cost of debt, financial constraints, and bank debt accounting for the tangible response but not the intangible response. Financial frictions more generally amplify investment reductions following uncertainty shocks, particularly among constrained firms (Alfaro et al., 2024). This evidence motivates analysing investment components separately rather than treating aggregate investment as homogeneous.

The valuation sign, however, is theoretically ambiguous, and this ambiguity is precisely what makes the interaction an empirical question. Pledgeable tangible assets preserve borrowing capacity when funding conditions deteriorate (Almeida and Campello, 2007), yet a high tangible-asset share can simultaneously signal fewer growth options as value creation shifts toward intangible capital (Peters and Taylor, 2017; Falato et al., 2022). R&D embodies valuable growth options, but its opaque and weakly collateralizable payoffs make continued external funding less credible when external finance deteriorates (Myers, 1977; Myers and Majluf, 1984; Falato et al., 2022). Financial assets insure future investment capacity (Gamba and Triantis, 2008) yet may be discounted by investors as agency slack (Dittmar and Mahrt-Smith, 2007). Related evidence shows that uncertainty widens corporate credit spreads and weakens the relation between investment and the cost of capital, but these studies examine policy uncertainty rather than uncertainty originating within banks (Drobetz et al., 2018; Kaviani et al., 2020). They support a pricing mechanism without determining the sign of the banking-uncertainty interactions examined here.

The objective of this extension is therefore not to re-estimate whether banking uncertainty reduces total investment, but to test whether it changes the conditional valuation relevance of six distinct components (CAPEX, TANG, INTANG, RND, SGA, and FIN) within the joint model. Section 4.8 operationalises this objective using uncertainty in bank asset growth (AUNC) and bank funding growth (FUNC). Because these indices vary only annually, the interactions are identified from year-level variation in the moderator; they are interpreted as conditional valuation patterns over the sample period rather than as causal effects of an exogenous banking shock. The inference design in Section 4.8 reflects this annual structure.

## 2.5. Korean market context and hypotheses

Korea is a useful setting because investment valuation is likely to vary with information conditions, ownership structure, and financing environment. The sample contains both KOSPI and KOSDAQ firms, allowing the analysis to compare listed firms that differ in size, maturity, and technology orientation. This distinction matters because intangible investment is harder to verify in real time and may be priced differently when firms differ in maturity, disclosure, and investor attention. The same R&D expenditure or recognised intangible-asset stock can therefore convey different information across listing segments.

Business-group affiliation introduces another source of heterogeneity. Korean chaebols combine internal capital markets with concentrated ownership and complex control structures. Internal capital markets can relax financing constraints and support affiliated-firm investment during adverse conditions (Almeida et al., 2015). Concentrated ownership and control, however, can expose outside shareholders to agency costs (Claessens et al., 2000; Joh, 2003), and Korean crisis evidence links governance quality to changes in firm value (Baek et al., 2004). Studies of the Korea discount show that chaebol valuation effects are not uniform across firms or periods (Lee et al., 2010; Ducret and Isakov, 2020, 2024). This motivates subgroup tests rather than a single Korea-wide valuation coefficient.

Technological orientation is another source of variation. Patents and intangible capital carry valuation information when innovation shapes firms' growth opportunities (Hall et al., 2005; Peters and Taylor, 2017). R&D need not, however, be priced identically across sectors. In high-technology industries, it may be a baseline requirement for survival; in low-technology industries, the same spending may signal strategic upgrading. Within Korea, the valuation of R&D varies with the control–ownership wedge and chaebol status (Kang et al., 2019). Other Korean evidence links R&D valuation to governance and ownership structure (Sul, 2021) and relates internal capital markets to R&D investment within chaebols (Hong et al., 2023). These differences motivate tests of heterogeneity rather than the assumption of a universal valuation weight.

The hypotheses follow from the preceding literature rather than from the observed ranking of the six variables. Multi-capital theory predicts different valuation weights, joint capital allocation predicts that isolated and simultaneous estimates need not coincide, and research on growth options and intangible capital provides the directional expectation for knowledge-based investment. These arguments lead to three hypotheses:

H1a: Individual investment components exhibit heterogeneous standalone valuation coefficients.

H1b: At least some component coefficients change in magnitude or precision when the other investment components are included simultaneously.

H1c: Knowledge-oriented components, especially research and development expenditure and recognised intangible assets, retain larger positive point estimates than traditional physical components in the simultaneous specification.

The heterogeneity and moderation analyses are organised around two research questions:

RQ2: Does the valuation relevance of investment components vary across firm environments, including market segment, business-group affiliation, and technological orientation?

RQ3: Does external banking uncertainty alter the conditional valuation relevance of investment components?

Together, these hypotheses and research questions position firm valuation as an investment-composition problem. The paper does not assume a universal ranking for all firms. Instead, it asks whether capital markets assign different valuation weights to heterogeneous investment components after the broader investment mix and institutional context are taken into account.


# 3. Data and methodology

## 3.1. Data and sample construction

The empirical analysis uses Korean listed firms from 2012 to 2025, with firm-level financial, governance, and market-segment data obtained from FNGuide, a commercial Korean financial database. After applying the variable-availability and cleaning requirements described below, the final unbalanced panel contains 23,726 firm-year observations for 2,493 firms. The sample includes both KOSPI and KOSDAQ firms, and it retains industry and market identifiers so that the analysis can examine heterogeneity across listing segments and technological environments.

The unit of observation is the firm-year. The panel is sorted by firm identifier and fiscal year, and the focal explanatory variables enter the main models in lagged form. This timing convention does not remove all endogeneity concerns, but it reduces the most direct simultaneity between current valuation and current investment choices. When lagged variables are not directly available in the raw file, I construct them from the corresponding current-period series using within-firm one-year shifts. Because finance panel data often exhibit within-firm dependence and unbalanced observation histories, the inference design treats the firm as the relevant dependence unit, consistent with the concerns discussed by Petersen (2009). Continuous regression variables are winsorized at the 1st and 99th percentiles to reduce the influence of extreme observations; binary indicators, including Big 4 auditor status, chaebol affiliation, and the loss indicator, are left unchanged.

## 3.2. Variables

I use two market-based valuation measures. Tobin's Q (TobinQ) is the primary outcome and is defined as the market value of equity plus the book value of debt, scaled by total assets, following the valuation literature (Hayashi, 1982; Chung and Pruitt, 1994; Peters and Taylor, 2017). Industry-adjusted Tobin's Q (industry-adjusted TobinQ), defined as firm-level TobinQ minus the industry-year median TobinQ, is used in parallel to remove common valuation differences across industries and years while preserving firm-level dispersion around the sector benchmark, following the logic of industry-adjusted valuation comparisons in Berger and Ofek (1995).

The six focal variables are lagged investment components. Capital expenditure intensity (CAPEX) measures current investment in physical capacity. Tangible-asset intensity (TANG) represents the accumulated physical asset stock. Recognised intangible-asset intensity (INTANG) measures the accounting-recognised stock of intangible assets. Research and development intensity (RND) is the proxy for innovation investment (Lev and Sougiannis, 1996; Peters and Taylor, 2017). Selling, general, and administrative expenditure intensity (SGA) proxies for the development of organisational and customer-related capabilities (Lev and Radhakrishnan, 2005; Eisfeldt and Papanikolaou, 2013; Enache and Srivastava, 2018). Financial-asset intensity (FIN) captures financial holdings and liquidity-like flexibility (Opler et al., 1999; Faulkender and Wang, 2006; Gamba and Triantis, 2008). All six variables enter in lagged form so that the timing of the valuation tests is consistent across components.

All investment components enter the empirical models with a one-year lag to establish temporal ordering between corporate investment decisions and subsequent valuation. Flow variables are scaled by total assets to improve comparability across firms. Stock variables are also measured relative to total assets where appropriate. This design does not make flows and stocks conceptually identical; instead, it permits their valuation relevance to be compared after standardisation in the empirical tables. Firm age (AGE) is measured as the natural logarithm of firm age. Foreign ownership (FOREIGN) is expressed in percentage points, while ownership concentration (OWNCONC) is measured as a proportion. Analyst monitoring (ANALYST\_MONITORING) is a transformed analyst-coverage measure, with a value of zero indicating the absence of analyst coverage. The negative minimum value observed for capital expenditure intensity reflects the net-investment measure available in the source data, which can become negative when asset disposals or accounting adjustments exceed gross capital additions.

The moderator series comes from a separate Korean bank panel and follows the disaggregate-uncertainty procedure of Buch et al. (2015), as implemented by Huynh (2025). For each bank \(i\) and year \(t\), the relevant bank outcome \(X_{it}\) is residualised on bank and year fixed effects, \(X_{it}=\alpha_i+\lambda_t+\varepsilon_{it}\). The annual uncertainty index is the cross-sectional standard deviation of these residuals, \(UNC_t=SD_i(\varepsilon_{it})\). It measures dispersion in unexpected bank-level outcomes after common year conditions and time-invariant bank differences are removed. It is not the aggregate growth rate or volatility of the banking sector. AUNC uses log growth in total bank assets. FUNC uses log growth in broad funding, defined as deposit liabilities plus borrowed funding. Growth inputs are winsorized at the 1st and 99th percentiles before residualisation, and an annual index is reported only when at least five banks contribute observations. Median absolute deviation, interquartile-range, and leave-one-bank-out versions provide construction checks. Each annual series is merged to the firm-year panel by year, standardised, and interacted with standardised lagged investment components.

Control variables capture firm characteristics that prior research links to investment choices and market valuation. Size and leverage represent scale, information conditions, capital structure, and financing exposure (Rajan and Zingales, 1995). Cash holdings and operating cash flow capture access to internal finance (Almeida et al., 2004), while liquidity and asset tangibility relate to financing constraints and borrowing capacity (Almeida and Campello, 2007). Debt maturity, short-term debt pressure, and interest burden describe the structure and cost of external finance; these conditions are particularly relevant to R&D investment (Lee, 2012; He and Wintoki, 2016; Giebel and Kraft, 2024). Sales growth, firm age, profitability, and loss status capture demand conditions, lifecycle maturity, operating performance, and distress risk. Foreign ownership provides one measure of external monitoring in emerging-market capital structures (Do et al., 2020). Chaebol affiliation captures access to internal capital markets (Shin and Park, 1999; Hong et al., 2023), whereas ownership concentration captures control incentives relevant to R&D and valuation (Baek et al., 2004; Kang et al., 2019). Evidence that chaebol outcomes vary across business groups provides an additional reason to retain group affiliation as a control and subgroup variable (Ducret and Isakov, 2024). Big 4 auditor status and analyst monitoring provide controls for information quality and external scrutiny. ROA volatility is not included in the baseline because it reduces the usable sample without materially changing the empirical design.

## 3.3. Empirical framework

The empirical design estimates whether heterogeneous investment components contain valuation information after observed firm characteristics and market conditions are taken into account. I begin with a conventional fixed-effects benchmark. For each component, the separate OLS specification is:

\[
Y_{it} = \alpha + \theta_k D_{k,it-1} + \beta'X_{it-1} + \gamma_j + \lambda_t + \varepsilon_{it}.
\]

The joint OLS specification is:

\[
Y_{it} = \alpha + \sum_k \theta_k D_{k,it-1} + \beta'X_{it-1} + \gamma_j + \lambda_t + \varepsilon_{it}.
\]

Here, \(Y_{it}\) denotes TobinQ or industry-adjusted TobinQ, \(D_{k,it-1}\) is one lagged investment component, \(X_{it-1}\) contains firm-level controls, \(\gamma_j\) denotes industry fixed effects, and \(\lambda_t\) denotes year fixed effects. The separate model estimates the association between firm value and one component at a time. The joint model asks whether each component remains informative when the other investment components enter the same valuation equation.

To make the focal coefficients comparable across variables measured on different scales, the six lagged investment components are z-standardised using the mean and standard deviation of the full estimation sample before any subgroup or subperiod split. Accordingly, the investment-component coefficients reported in Tables 3–11 measure the estimated change in firm value associated with the same full-sample one-standard-deviation increase in the corresponding component. This common reference scale permits comparisons across firm environments and periods; the outcome variables are not standardised, and controls remain in the units described in Section 3.2 unless otherwise indicated. As a secondary robustness check, the subgroup and subperiod models are also estimated using within-group standard deviations, but those estimates are not used for the principal cross-group or cross-period comparisons. In the moderation models, AUNC and FUNC are also z-standardised before the component-by-uncertainty products are constructed; a component main-effect coefficient is therefore evaluated at the sample mean of the relevant uncertainty index, and an interaction coefficient measures the change in that component effect for a one-standard-deviation increase in uncertainty.

The DML specification follows the partially linear model of Robinson (1988) and Chernozhukov et al. (2018):

\[
Y_{it} = D'_{it-1}\theta + g(W_{it}) + \varepsilon_{it},
\]

\[
D_{it-1} = m(W_{it}) + v_{it}.
\]

In the separate-treatment design, \(D_{it-1}\) contains one investment component. In the simultaneous-treatment design, it contains the full vector of six components. \(W_{it}\) contains the observed controls and fixed-effect controls. The nuisance functions \(g(\cdot)\) and \(m(\cdot)\) are learned with machine-learning methods rather than imposed as linear functions. The parameter vector \(\theta\) is then estimated from residual variation after both the outcome and treatment variables have been partialled out with respect to \(W_{it}\).

## 3.4. Double Machine Learning implementation

The main estimation strategy is double/debiased machine learning in the sense of Chernozhukov et al. (2018), implemented through the DoubleML Python framework (Bach et al., 2022). I use the partially linear regression score with the "partialling out" formulation. For each specification, machine learning estimates the nuisance components, and the target coefficient is recovered from orthogonalized residual variation.

All baseline DML estimates use firm-blocked five-fold cross-fitting with five repetitions. All observations belonging to the same firm are assigned to the same fold, so the nuisance functions are not trained on one year of a firm and evaluated on another year of the same firm. This design reduces leakage in the unbalanced firm-year panel and is consistent with the panel-DML concern that nuisance learning should respect within-unit dependence (Clarke and Polselli, 2025). The main nuisance learner is LightGBM, a gradient-boosting decision-tree algorithm designed for efficient high-dimensional prediction (Ke et al., 2017), with tuning conducted over a pre-specified grid before final estimation. The tuning grid, selected hyperparameters, auxiliary learner settings, and nuisance-model RMSE statistics are reported in Appendix A.

For inference, standard errors are obtained from a firm-level multiplier-bootstrap variance estimator based on DML influence scores with 1,000 replications, and t-statistics are reported in parentheses. Because the simultaneous specifications test several investment components in the same table, adjusted p-values are reported using the Romano-Wolf stepdown procedure, the Benjamini-Yekutieli false-discovery-rate correction, and the Bonferroni correction (Romano and Wolf, 2005; Benjamini and Yekutieli, 2001). The main specifications include industry and year fixed effects in the control vector. The banking-uncertainty moderation models require an additional few-cluster sensitivity analysis because their moderators vary only annually. For each annual index, I therefore re-evaluate the six pre-specified interaction terms using an exact Rademacher wild bootstrap over calendar-year clusters; all 16,384 sign patterns are enumerated for the 14 sample years.

To provide a conventional benchmark, the paper estimates industry and year fixed-effects regressions with the same investment measures and controls. These regressions provide a familiar reference point, while differences between fixed-effects estimates and DML estimates help show the role of nonlinear observed confounding and joint dependence among investment components.

The final robustness exercise is a DML-IV specification estimated with the partially linear IV module of DoubleML. Each investment component is instrumented with its own second lag and a leave-one-out peer average within the same industry-year cell. The lag instruments exploit persistence in firm investment policy. The peer instruments capture common allocation conditions among firms operating in similar investment environments, consistent with evidence that peers affect corporate financial policy (Leary and Roberts, 2014). Excluding the focal firm from the peer average prevents a mechanical own-observation correlation, but it does not by itself resolve the reflection problem described by Manski (1993). More generally, exposure-based instruments require careful examination of which underlying shocks generate identifying variation (Goldsmith-Pinkham et al., 2020; Borusyak et al., 2022). The DML-IV estimates are therefore used as supportive evidence on reverse-causality concerns and interpreted alongside the overidentification diagnostics, rather than as definitive causal estimates.

# 4. Empirical results

## 4.1. Descriptive statistics and correlation analysis

The final estimation sample contains 23,726 firm-year observations for 2,493 Korean listed firms over 2012–2025. Table 1 shows substantial cross-sectional variation in both valuation outcomes and investment components. Mean TobinQ is 1.363 and the distribution is right-skewed, with a median of 1.032, indicating that a subset of high-growth firms pulls the average upward. The industry-adjusted TobinQ, which removes the industry-year median, has a mean of 0.307 and is centred near zero by construction. It therefore captures valuation differences across investment choices within an industry and year.

Among the six focal investment components, tangible assets represent the largest balance-sheet component on average (mean TANG = 0.302), followed by SG&A intensity (0.160) and financial assets (0.139). CAPEX is modest on average (0.038) but highly variable relative to its mean, consistent with the investment lumpiness documented in plant-level capital adjustment research (Doms and Dunne, 1998). R&D is small on average (0.014) but extremely right-skewed, with observations in the upper tail reaching 0.152, confirming that innovation investment is concentrated in a subset of technology-oriented firms, as in prior evidence on financing innovation and R&D-intensive firms (Brown et al., 2009; He and Wintoki, 2016). Recognised intangible assets show a similar pattern, suggesting that intangible accumulation is highly uneven across the Korean listed-firm universe.

**Table 1. Descriptive statistics**

| **Variable**                                           | **Obs.** | **Min.** | **Mean** | **Median** | **SD**  | **Max.** |
| ------------------------------------------------------ | -------- | -------- | -------- | ---------- | ------- | -------- |
| **Panel A: Dependent variables**                       |          |          |          |            |         |          |
| **TobinQ**                                             | 23,726   | 0\.387   | 1\.363   | 1\.032     | 1\.074  | 7\.804   |
| **Industry-adjusted TobinQ**                           | 23,726   | −1.179   | 0\.307   | 0\.000     | 1\.055  | 6\.966   |
| **Panel B: Investment components**                     |          |          |          |            |         |          |
| **CAPEX**                                              | 23,726   | -0.060   | 0\.038   | 0\.022     | 0\.049  | 0\.251   |
| **TANG**                                               | 23,726   | 0\.004   | 0\.302   | 0\.292     | 0\.187  | 0\.768   |
| **INTANG**                                             | 23,726   | 0\.000   | 0\.043   | 0\.016     | 0\.068  | 0\.389   |
| **RND**                                                | 23,726   | 0\.000   | 0\.014   | 0\.003     | 0\.026  | 0\.153   |
| **SGA**                                                | 23,726   | 0\.017   | 0\.160   | 0\.111     | 0\.153  | 0\.917   |
| **FIN**                                                | 23,726   | 0\.001   | 0\.139   | 0\.085     | 0\.148  | 0\.698   |
| **Panel C: Firm-level controls**                       |          |          |          |            |         |          |
| **SIZE**                                               | 23,726   | 16\.641  | 19\.322  | 19\.040    | 1\.489  | 24\.151  |
| **LEV**                                                | 23,726   | 0\.051   | 0\.437   | 0\.438     | 0\.199  | 0\.884   |
| **ROA**                                                | 23,726   | -0.549   | -0.003   | 0\.020     | 0\.115  | 0\.223   |
| **CASH**                                               | 23,726   | 0\.002   | 0\.104   | 0\.077     | 0\.095  | 0\.531   |
| **GROWTH**                                             | 23,726   | -0.654   | 0\.099   | 0\.045     | 0\.384  | 2\.299   |
| **OCF**                                                | 23,726   | -0.290   | 0\.035   | 0\.041     | 0\.089  | 0\.270   |
| **AGE**                                                | 23,726   | 0\.000   | 2\.597   | 2\.708     | 0\.791  | 3\.912   |
| **LIQ**                                                | 23,726   | 0\.335   | 2\.339   | 1\.515     | 2\.575  | 18\.912  |
| **Panel D: Governance, monitoring, and risk controls** |          |          |          |            |         |          |
| **FOREIGN**                                            | 23,726   | 0\.000   | 6\.341   | 2\.230     | 10\.035 | 55\.636  |
| **OWNCONC**                                            | 23,726   | 0\.071   | 0\.395   | 0\.388     | 0\.167  | 0\.790   |
| **BIG4**                                               | 23,726   | 0\.000   | 0\.437   | 0\.000     | 0\.496  | 1\.000   |
| **KM\_RATIO**                                          | 23,726   | 0\.000   | 0\.173   | 0\.153     | 0\.141  | 0\.544   |
| **CHAEBOL**                                            | 23,726   | 0\.000   | 0\.138   | 0\.000     | 0\.345  | 1\.000   |
| **ANALYST\_MONITORING**                                | 23,726   | 0\.000   | 0\.047   | 0\.000     | 0\.277  | 2\.000   |
| **LT\_DEBT\_RATIO**                                    | 23,726   | 0\.000   | 0\.321   | 0\.263     | 0\.282  | 1\.000   |
| **STD\_PRESSURE**                                      | 23,726   | 0\.000   | 0\.560   | 0\.584     | 0\.323  | 1\.000   |
| **INTEREST\_BURDEN**                                   | 23,726   | 0\.000   | 0\.021   | 0\.017     | 0\.021  | 0\.118   |
| **LOSS\_DUMMY**                                        | 23,726   | 0\.000   | 0\.335   | 0\.000     | 0\.472  | 1\.000   |

Notes: *The table reports descriptive statistics for the final estimation sample after applying the DML pipeline.* *All continuous variables are winsorized at the 1st and 99th percentiles; binary indicators are not winsorized. Industry-adjusted TobinQ is firm-level TobinQ minus the industry-year median.*

The correlation structure in Table 2 shows that the six investment components are related but not mutually redundant. CAPEX is positively correlated with tangibility (0.406), while FIN is negatively correlated with tangibility (-0.453) and positively correlated with liquidity (0.470), indicating that real and financial investment choices tend to substitute at the balance-sheet level. At the outcome level, R&D is the most strongly correlated component with TobinQ (0.340) and industry-adjusted TobinQ (0.342), followed by SG&A (0.235, 0.209) and recognised intangibles (0.160, 0.143). Tangibility carries a negative bivariate correlation with both valuation measures, which motivates the subsequent analysis that conditions on the other modelled components. The VIF statistics for the six focal components are all well below conventional danger thresholds (maximum 2.040 for TANG), suggesting that multicollinearity among the investment components is not severe enough to undermine the joint estimates.


**Table 2. Selected correlations and variance inflation factors**

| **Variable**       | **TobinQ** | **Ind-adj TobinQ** |     | **CAPEX** | **TANG** | **INTANG** | **RND** | **SGA** | **FIN** | **VIF** |
| ------------------ | ---------- | :----------------: | --- | --------- | -------- | ---------- | ------- | ------- | ------- | ------- |
| **TobinQ**         | 1\.000     |                    |     |           |          |            |         |         |         |         |
| **Ind-adj TobinQ** | 0\.985     |       1\.000       |     |           |          |            |         |         |         |         |
| **CAPEX**          | 0\.037     |       0\.040       |     | 1\.000    |          |            |         |         |         | 1\.328  |
| **TANG**           | -0.134     |       −0.119       |     | 0\.406    | 1\.000   |            |         |         |         | 2\.040  |
| **INTANG**         | 0\.160     |       0\.143       |     | -0.079    | -0.228   | 1\.000     |         |         |         | 1\.208  |
| **RND**            | 0\.340     |       0\.342       |     | 0\.036    | -0.103   | 0\.065     | 1\.000  |         |         | 1\.245  |
| **SGA**            | 0\.235     |       0\.209       |     | -0.053    | -0.211   | 0\.182     | 0\.266  | 1\.000  |         | 1\.225  |
| **FIN**            | 0\.136     |       0\.125       |     | -0.170    | -0.453   | -0.027     | 0\.143  | 0\.085  | 1\.000  | 1\.727  |

*Notes: Lower-triangular pairwise Pearson correlations and variance inflation factors (VIF) computed on the final estimation sample. VIFs are from an OLS regression of each investment component on the remaining five components and the full control set.*

## 4.2. Baseline industry and year fixed-effects regressions

Table 3 reports the conventional industry and year fixed-effects regressions. Columns (1)-(6) include one focal investment component at a time, together with the full control set, industry fixed effects, and year fixed effects. Column (7) includes all six investment components jointly and therefore provides a linear fixed-effects benchmark for the simultaneous DML estimates.

**Table 3. Industry and year fixed-effects regression baseline: TobinQ**


| **Variable**           | **(1) CAPEX** | **(2) TANG** | **(3) INTANG** | **(4) RND** | **(5) SGA** | **(6) FIN** | **(7) Joint** |
| :----------------- | :--------- | :--------- | :--------- | :--------- | :--------- | :--------- | :--------- |
| **CAPEX**               | 0.0743\*\*\*  |            |            |            |            |            | 0.0797\*\*\*  |
|                    | (7.485)    |            |            |            |            |            | (8.195)    |
| **TANG**              |            | -0.0160    |            |            |            |            | 0.0281\*    |
|                    |            | (-1.063)   |            |            |            |            | (1.864)    |
| **INTANG**             |            |            | 0.1317\*\*\*  |            |            |            | 0.1320\*\*\*  |
|                    |            |            | (5.902)    |            |            |            | (5.880)    |
| **RND**                |            |            |            | 0.2384\*\*\*  |            |            | 0.2029\*\*\*  |
|                    |            |            |            | (10.977)   |            |            | (9.301)    |
| **SGA**                |            |            |            |            | 0.1574\*\*\*  |            | 0.1024\*\*\*  |
|                    |            |            |            |            | (8.383)    |            | (5.767)    |
| **FIN**                |            |            |            |            |            | 0.0547\*\*\*  | 0.0740\*\*\*  |
|                    |            |            |            |            |            | (3.145)    | (4.240)    |
| **SIZE**               | -0.2065\*\*\* | -0.2057\*\*\* | -0.2162\*\*\* | -0.1869\*\*\* | -0.1847\*\*\* | -0.2068\*\*\* | -0.1887\*\*\* |
|                    | (-14.008)  | (-13.910)  | (-14.679)  | (-13.552)  | (-12.731)  | (-14.024)  | (-13.768)  |
| **LEV**                | 0.4272\*\*\*  | 0.3678\*\*\*  | 0.4557\*\*\*  | 0.4431\*\*\*  | 0.3139\*\*\*  | 0.4025\*\*\*  | 0.5601\*\*\*  |
|                    | (4.293)    | (3.592)    | (4.769)    | (4.709)    | (3.201)    | (4.079)    | (6.082)    |
| **ROA**                | -0.5725\*\*\* | -0.5998\*\*\* | -0.4670\*\*\* | -0.3035\*\*  | -0.3516\*\*  | -0.5680\*\*\* | -0.0003    |
|                    | (-3.538)   | (-3.691)   | (-2.875)   | (-1.984)   | (-2.244)   | (-3.524)   | (-0.002)   |
| **CASH**               | 1.1317\*\*\*  | 1.0700\*\*\*  | 1.1161\*\*\*  | 0.9157\*\*\*  | 1.0372\*\*\*  | 1.2241\*\*\*  | 1.1797\*\*\*  |
|                    | (6.127)    | (5.720)    | (6.067)    | (5.230)    | (5.747)    | (6.346)    | (6.424)    |
| **GROWTH**             | 0.2570\*\*\*  | 0.2604\*\*\*  | 0.2252\*\*\*  | 0.2351\*\*\*  | 0.2559\*\*\*  | 0.2614\*\*\*  | 0.1951\*\*\*  |
|                    | (9.884)    | (10.032)   | (8.543)    | (9.763)    | (10.003)   | (10.110)   | (8.102)    |
| **OCF**                | -0.9153\*\*\* | -0.7677\*\*\* | -0.8518\*\*\* | -0.5785\*\*\* | -0.8142\*\*\* | -0.7922\*\*\* | -0.8536\*\*\* |
|                    | (-5.950)   | (-5.006)   | (-5.520)   | (-4.041)   | (-5.378)   | (-5.177)   | (-5.968)   |
| **AGE**                | -0.1405\*\*\* | -0.1575\*\*\* | -0.1433\*\*\* | -0.1040\*\*\* | -0.1461\*\*\* | -0.1519\*\*\* | -0.0645\*\*\* |
|                    | (-7.563)   | (-8.543)   | (-7.697)   | (-5.630)   | (-8.023)   | (-8.339)   | (-3.495)   |
| **LIQ**                | 0.0424\*\*\*  | 0.0369\*\*\*  | 0.0435\*\*\*  | 0.0330\*\*\*  | 0.0419\*\*\*  | 0.0298\*\*\*  | 0.0368\*\*\*  |
|                    | (4.584)    | (3.927)    | (4.802)    | (3.673)    | (4.612)    | (2.986)    | (3.872)    |
| **FOREIGN**            | 0.0130\*\*\*  | 0.0134\*\*\*  | 0.0132\*\*\*  | 0.0117\*\*\*  | 0.0117\*\*\*  | 0.0133\*\*\*  | 0.0102\*\*\*  |
|                    | (6.385)    | (6.524)    | (6.506)    | (6.090)    | (6.045)    | (6.535)    | (5.665)    |
| **OWNCONC**            | -0.5326\*\*\* | -0.5477\*\*\* | -0.5149\*\*\* | -0.3448\*\*\* | -0.5430\*\*\* | -0.5309\*\*\* | -0.2877\*\*\* |
|                    | (-5.771)   | (-5.915)   | (-5.772)   | (-3.900)   | (-5.958)   | (-5.738)   | (-3.463)   |
| **BIG4**               | 0.0300     | 0.0263     | 0.0293     | -0.0143    | -0.0009    | 0.0292     | -0.0160    |
|                    | (1.118)    | (0.975)    | (1.103)    | (-0.549)   | (-0.034)   | (1.081)    | (-0.634)   |
| **KM_RATIO**           | -0.4293\*\*\* | -0.2893\*\*  | -0.3153\*\*\* | -0.3450\*\*\* | -0.1784    | -0.2479\*\*  | -0.2944\*\*\* |
|                    | (-3.689)   | (-2.472)   | (-2.757)   | (-3.122)   | (-1.560)   | (-2.125)   | (-2.707)   |
| **STD_PRESSURE**       | -0.0605    | -0.0667    | -0.0568    | -0.0446    | -0.0629    | -0.0722    | -0.0564    |
|                    | (-1.034)   | (-1.145)   | (-0.985)   | (-0.790)   | (-1.109)   | (-1.243)   | (-1.044)   |
| **INTEREST_BURDEN**    | 2.6898\*\*\*  | 2.4164\*\*\*  | 1.9433\*\*\*  | 3.0535\*\*\*  | 2.7170\*\*\*  | 2.3283\*\*\*  | 2.7991\*\*\*  |
|                    | (3.723)    | (3.365)    | (2.819)    | (4.365)    | (3.844)    | (3.206)    | (4.231)    |
| **CHAEBOL**            | 0.2479\*\*\*  | 0.2380\*\*\*  | 0.2236\*\*\*  | 0.2482\*\*\*  | 0.2387\*\*\*  | 0.2422\*\*\*  | 0.2453\*\*\*  |
|                    | (5.009)    | (4.781)    | (4.552)    | (5.358)    | (4.874)    | (4.876)    | (5.390)    |
| **LT_DEBT_RATIO**      | 0.0834     | 0.1287\*\*   | 0.0950     | 0.1340\*\*   | 0.0953     | 0.1264\*\*   | 0.0205     |
|                    | (1.311)    | (2.050)    | (1.534)    | (2.207)    | (1.541)    | (2.015)    | (0.348)    |
| **ANALYST_MONITORING** | 0.4356\*\*\*  | 0.4374\*\*\*  | 0.4244\*\*\*  | 0.4224\*\*\*  | 0.4340\*\*\*  | 0.4402\*\*\*  | 0.4112\*\*\*  |
|                    | (9.037)    | (9.055)    | (8.774)    | (9.458)    | (9.052)    | (9.111)    | (9.228)    |
| **LOSS_DUMMY**         | 0.0720\*\*\*  | 0.0675\*\*\*  | 0.0544\*\*   | 0.0718\*\*\*  | 0.0816\*\*\*  | 0.0635\*\*   | 0.0683\*\*\*  |
|                    | (2.796)    | (2.606)    | (2.147)    | (2.882)    | (3.201)    | (2.473)    | (2.889)    |
| **Industry FE**        | Yes        | Yes        | Yes        | Yes        | Yes        | Yes        | Yes        |
| **Year FE**            | Yes        | Yes        | Yes        | Yes        | Yes        | Yes        | Yes        |
| **No. observations**   | 23,726     | 23,726     | 23,726     | 23,726     | 23,726     | 23,726     | 23,726     |
| **Adjusted R-squared** | 0.226      | 0.222      | 0.235      | 0.262      | 0.239      | 0.224      | 0.286      |


*Notes: The table reports OLS regressions with industry and year fixed effects. Columns (1)-(6) introduce one focal investment component at a time, while column (7) includes all six investment components jointly. Investment-component scaling follows Section 3.3; control coefficients remain in their stated units. Standard errors are clustered at the firm level. t-statistics are reported in parentheses. \*\*\*, \*\*, and \* denote significance at the 1%, 5%, and 10% levels, respectively.*
The fixed-effects results show the basic empirical pattern. In the separate OLS specifications, CAPEX, INTANG, R&D, SG&A, and FIN are positively associated with TobinQ, while TANG is statistically insignificant. Because the focal variables now share a one-standard-deviation scale, their magnitudes can be compared directly. R&D has the largest point estimate (0.238), followed by SG&A (0.157), recognised intangible assets (0.132), CAPEX (0.074), and FIN (0.055). This ordering is consistent with evidence that R&D expenditure contains information about future growth and market value (Chan et al., 2001; Hall et al., 2005), while the positive SG&A coefficient suggests that organisational and commercial expenditure may contain valuation-relevant intangible investment rather than representing only routine overhead (Banker et al., 2019).

Column (7) shows that the main valuation components remain visible when all six investment-component variables enter the same linear fixed-effects regression. R&D retains the largest coefficient (0.203), followed by INTANG (0.132), SGA (0.102), CAPEX (0.080), and FIN (0.074); TANG is smaller and significant only at the 10% level (0.028). The joint OLS model therefore provides a useful linear benchmark for the simultaneous DML estimates. It shows that the broad incremental valuation pattern is already visible in the linear specification, while leaving open whether flexible adjustment for nonlinear observed confounding changes magnitudes and inference.

These results provide a useful benchmark, but they also show the limits of the linear fixed-effects design. The OLS model imposes a restrictive functional form on the relation between controls, treatment assignment, and firm value. The DML analysis asks a more demanding question: whether the same components remain valuation-relevant after flexible adjustment for observed confounders and conditioning on the other investment components.

## 4.3. Standalone valuation relevance: separate-treatment DML estimates

I first estimate the valuation association of each investment component separately. The models use the DML framework of Chernozhukov et al. (2018), orthogonalised partially linear regression scores, five firm-blocked folds, and five repetitions. Inference uses a firm-level multiplier bootstrap with 1,000 replications.

Table 4 reports the separate-treatment estimates. Each model includes one component, the same controls, and industry and year fixed effects. Tuned LightGBM learners estimate the nuisance functions. The specification asks whether the focal component is associated with valuation when considered on its own.


**Table 4. Separate-treatment DML estimates**

| **Investment component** |       **TobinQ**        |  **Ind.-adj. TobinQ**   |
| :----------------------- | :---------------------: | :---------------------: |
| **CAPEX**                | 0.0490\*\*\*<br>(5.043) | 0.0470\*\*\*<br>(4.799) |
| **TANG**                 |   -0.0247<br>(-1.510)   |   -0.0258<br>(-1.570)   |
| **INTANG**               | 0.1296\*\*\*<br>(5.802) | 0.1290\*\*\*<br>(5.795) |
| **RND**                  | 0.1675\*\*\*<br>(7.612) | 0.1681\*\*\*<br>(7.629) |
| **SGA**                  | 0.1106\*\*\*<br>(6.363) | 0.1061\*\*\*<br>(6.029) |
| **FIN**                  |    0.0249<br>(1.563)    |   0.0278\*<br>(1.736)   |
| **Controls**             |           Yes           |           Yes           |
| **Industry + Year FE**   |           Yes           |           Yes           |
| **Observations**         |         23,726          |         23,726          |

*Note: Separate-treatment DML estimates (Chernozhukov et al., 2018) using the partially linear regression score with partialling-out, implemented via DoubleML (Bach et al., 2022). Each row is an independent estimation with one focal treatment; variable scaling follows Section 3.3. Nuisance functions are estimated by LightGBM tuned via 5-fold cross-validation; the selected configuration is depth 5 and learning rate 0.05 for both outcomes. Cross-fitting uses firm-blocked 5 folds and 5 repetitions. Industry and year fixed effects enter the nuisance stage as binary dummies. Standard errors are obtained from a firm-level multiplier bootstrap variance estimator based on DML influence scores with 1,000 replications. t-statistics in parentheses. \*\*\*, \*\*, \* denote significance at the 1%, 5%, 10% levels.*

The estimates differ markedly across components. R&D has the largest standardised coefficient, 0.168 for both TobinQ measures, which is consistent with a valuation premium for innovation investment (Hall, 1993; Lev and Sougiannis, 1996; Peters and Taylor, 2017). Recognised intangible assets also have positive coefficients of about 0.129. SG&A is positive, consistent with the view that part of this expenditure represents organisational capital (Eisfeldt and Papanikolaou, 2013; Banker et al., 2019). CAPEX is positive but smaller. FIN is positive but statistically weak, in line with evidence that the value of liquid financial holdings depends on governance and firm context (Faulkender and Wang, 2006; Dittmar and Mahrt-Smith, 2007). Tangible assets have negative but insignificant coefficients. The separate models therefore place R&D and recognised intangibles ahead of the financial and tangible components.

The relative magnitudes also support separating innovation flows from recognised intangible stocks. RND measures current knowledge-creation expenditure, much of which accounting statements do not capitalise (Lev, 2001; Lev and Sougiannis, 1996). INTANG measures the recognised stock of intangible assets. Recent market-based measurement work shows that accounting and constructed measures capture different portions of intangible capital (Peters and Taylor, 2017; Ewens et al., 2024). The positive estimates in this study therefore suggest that current innovation input and recognised intangible accumulation provide related but distinct valuation signals.

## 4.4. Incremental valuation relevance: simultaneous DML results

The simultaneous DML model includes all six components in the same orthogonalised system. Its coefficients measure the valuation association of one component conditional on the other investment margins. This matters because physical investment, intangible accumulation, innovation spending, organisational capability, and financial flexibility are chosen jointly within the firm. Table 5 tests whether the associations found in the separate models remain after these choices are considered together.


**Table 5. Simultaneous DML estimates**

| Investment component |      TobinQ effect      | Romano-Wolf p | BY p  | Bonferroni p | Ind.-adj. TobinQ effect | Romano-Wolf p | BY p  | Bonferroni p |
| -------------------- | :---------------------: | :-----------: | :---: | :----------: | :---------------------: | :-----------: | :---: | :----------: |
| CAPEX                | 0.0497\*\*\*<br>(5.319) |     0.000     | 0.000 |    0.000     | 0.0441\*\*\*<br>(4.628) |     0.000     | 0.000 |    0.000     |
| TANG                 |    0.0190<br>(1.125)    |     0.272     | 0.666 |    1.000     |    0.0211<br>(1.241)    |     0.228     | 0.559 |    1.000     |
| INTANG               | 0.1326\*\*\*<br>(5.707) |     0.000     | 0.000 |    0.000     | 0.1304\*\*\*<br>(5.600) |     0.000     | 0.000 |    0.000     |
| RND                  | 0.1445\*\*\*<br>(6.251) |     0.000     | 0.000 |    0.000     | 0.1446\*\*\*<br>(6.233) |     0.000     | 0.000 |    0.000     |
| SGA                  | 0.0718\*\*\*<br>(4.101) |     0.000     | 0.000 |    0.000     | 0.0677\*\*\*<br>(3.816) |     0.000     | 0.000 |    0.000     |
| FIN                  | 0.0509\*\*\*<br>(2.994) |     0.006     | 0.015 |    0.030     | 0.0519\*\*\*<br>(3.050) |     0.003     | 0.009 |    0.018     |
| Controls             |           Yes           |               |       |              |           Yes           |               |       |              |
| Industry + Year FE   |           Yes           |               |       |              |           Yes           |               |       |              |
| Observations         |         23,726          |               |       |              |         23,726          |               |       |              |

*Notes: The table reports simultaneous-treatment DML-PLR estimates; variable scaling follows Section 3.3. Adjusted p-values are reported using the Romano-Wolf stepdown procedure, the Benjamini-Yekutieli (BY) false-discovery-rate correction, and the Bonferroni correction. Standard errors are obtained from a firm-level multiplier bootstrap variance estimator based on DML influence scores with 1,000 replications. t-statistics are reported in parentheses. \*, \*\*, and \*\*\* denote significance based on bootstrap p-values.*

Most coefficients remain positive and statistically significant in the simultaneous model. CAPEX changes little, from 0.049 to 0.050 on the standardised scale. INTANG also remains stable, rising from 0.130 to 0.133. R&D falls from 0.168 to 0.144, but remains the largest reported component.

Not all components display the same degree of stability. SG&A loses part of its standalone magnitude, falling from 0.111 to 0.072, whereas FIN moves in the opposite direction. Its standardised coefficient rises from 0.025, where it is statistically insignificant in Table 4, to 0.051 with a Romano-Wolf p-value of 0.006. At the same time, the negative coefficient on TANG disappears entirely. Once the remaining investment components are considered, tangibility no longer contains independent valuation information.

The changes show that the separate estimates contain information shared with correlated investment choices. R&D and INTANG have the largest reported standardised coefficients, followed by SG&A; CAPEX and FIN are positive but smaller, and TANG remains weak. This ordering describes point estimates and is not a test of differences between coefficients.

Tables 4 and 5 are consistent with H1a–H1c. The standalone coefficients differ across components; the magnitude or precision of SG&A, FIN, and TANG changes when the components enter jointly; and R&D and INTANG retain larger positive point estimates than CAPEX and TANG in the simultaneous model. Because the analysis does not formally test every pairwise coefficient difference, the evidence supports the reported pattern of estimates, not a universal statistical ranking.

The coefficient changes carry immediate economic information. The increase in FIN suggests that the valuation signal of financial flexibility becomes more visible once real investment, intangible accumulation, innovation spending, and organisational expenditure are controlled for jointly, consistent with the idea that intangible-intensive firms may hold financial assets partly because intangible capital weakens collateral capacity and increases precautionary liquidity demand (Falato et al., 2022). Conversely, the disappearance of the negative TANG coefficient indicates that the apparent discount on tangibility in sparse specifications partly reflects its correlation with other investment choices rather than an independent valuation penalty. Tangible assets can support borrowing capacity through collateral channels, but this financing role does not mechanically imply an incremental valuation premium once the rest of the investment-component vector is observed (Almeida and Campello, 2007). The attenuation of SG&A suggests that part of its standalone premium overlaps with innovation and intangible-capital components, but the remaining positive coefficient indicates that organisational expenditure still contains incremental valuation information.

The leave-one-component-out results reported below provide additional information about the interpretation of tangible assets. TANG is generally weak in the baseline simultaneous model, but it becomes negative and statistically significant when INTANG is omitted from the investment-component vector. This pattern suggests that tangible-asset intensity is partly evaluated relative to the firm's intangible orientation (Peters and Taylor, 2017; Falato et al., 2022).


![Figure 1A. Separate-treatment DML magnitudes on TobinQ](../../outputs/fig1a_fc_magnitude_tobinq_separate.png)

![Figure 1B. Simultaneous-treatment DML magnitudes on TobinQ](../../outputs/fig1b_fc_magnitude_tobinq_simultaneous.png)

*Figure 1. R&D has the largest DML magnitude in both standalone and simultaneous specifications. The figure compares the absolute estimated effects of the investment components in the separate-treatment specification (Figure 1A) and the simultaneous-treatment specification (Figure 1B). The estimates correspond to the DML results reported in Tables 4 and 5 using the tuned LightGBM nuisance learner.*








## 4.5. Robustness, sensitivity, and identification diagnostics

### 4.5.1. Alternative nuisance learners and tuning

Table 6 examines whether the simultaneous DML results depend on a particular LightGBM configuration or nuisance-learner family. The three LightGBM sensitivity specifications vary tree depth and shrinkage around the selected model, while XGBoost and Lasso provide nonlinear boosting and regularised-linear alternatives. Across these specifications, INTANG, RND, SGA, and CAPEX remain positive and statistically significant; FIN remains positive, although its precision varies. TANG is generally weak. The central component-level pattern is therefore not specific to one LightGBM tuning choice.

**Table 6. Alternative nuisance learners and tuning sensitivity of simultaneous DML estimates**

| Investment component |       d=4; s=0.05       |       d=4; s=0.03       |       d=5; s=0.03       |         XGBoost         |          Lasso          |
| -------------------- | :---------------------: | :---------------------: | :---------------------: | :---------------------: | :---------------------: |
| CAPEX                | 0.0501\*\*\*<br>(5.260) | 0.0531\*\*\*<br>(5.537) | 0.0514\*\*\*<br>(5.453) | 0.0525\*\*\*<br>(5.470) | 0.0779\*\*\*<br>(7.943) |
| TANG                 |    0.0175<br>(1.052)    |    0.0087<br>(0.528)    |    0.0125<br>(0.744)    |    0.0134<br>(0.819)    |   0.0287\*<br>(1.876)   |
| INTANG               | 0.1330\*\*\*<br>(5.707) | 0.1321\*\*\*<br>(5.636) | 0.1317\*\*\*<br>(5.632) | 0.1331\*\*\*<br>(5.698) | 0.1312\*\*\*<br>(5.797) |
| RND                  | 0.1476\*\*\*<br>(6.416) | 0.1545\*\*\*<br>(6.544) | 0.1481\*\*\*<br>(6.329) | 0.1503\*\*\*<br>(6.486) | 0.2036\*\*\*<br>(9.434) |
| SGA                  | 0.0749\*\*\*<br>(4.296) | 0.0766\*\*\*<br>(4.379) | 0.0748\*\*\*<br>(4.320) | 0.0771\*\*\*<br>(4.396) | 0.1020\*\*\*<br>(5.691) |
| FIN                  | 0.0503\*\*\*<br>(2.926) |  0.0465\*\*<br>(2.653)  | 0.0494\*\*\*<br>(2.844) |  0.0475\*\*<br>(2.732)  | 0.0744\*\*\*<br>(4.232) |
| Controls             |           Yes           |           Yes           |           Yes           |           Yes           |           Yes           |
| Industry and year FE |           Yes           |           Yes           |           Yes           |           Yes           |           Yes           |
| Observations         |         23,726          |         23,726          |         23,726          |         23,726          |         23,726          |

*Notes: The table reports simultaneous-treatment DML estimates for TobinQ under alternative nuisance-learning specifications; variable scaling follows Section 3.3. The first three columns vary the LightGBM interaction depth \(d\) and shrinkage \(s\); the final two columns replace LightGBM with XGBoost and Lasso. All specifications retain the controls, firm-blocked cross-fitting, and firm-level multiplier-bootstrap inference used in the main model. t-statistics are reported in parentheses. \*, \*\*, and \*\*\* denote significance at the 10%, 5%, and 1% levels.*


### 4.5.2. Leave-one-component-out sensitivity

Table 7 assesses whether the simultaneous estimates are driven by the inclusion of any single investment component. INTANG, RND, and SGA retain their signs and significance across all leave-one-out specifications. FIN becomes insignificant when INTANG is excluded, indicating that its incremental signal is sensitive to recognised intangible accumulation. TANG is the most composition-sensitive component and becomes significantly negative when INTANG is omitted.

**Table 7. Leave-one-component-out sensitivity for TobinQ**

| Investment component |           (1)           |           (2)           |           (3)           |           (4)           |           (5)           |           (6)           |
| :------------------- | :---------------------: | :---------------------: | :---------------------: | :---------------------: | :---------------------: | :---------------------: |
| CAPEX                |                         | 0.0556\*\*\*<br>(5.896) | 0.0517\*\*\*<br>(5.395) | 0.0580\*\*\*<br>(6.128) | 0.0514\*\*\*<br>(5.538) | 0.0498\*\*\*<br>(5.404) |
| TANG                 |  0.0383\*\*<br>(2.267)  |                         | -0.0396\*\*<br>(-2.223) |    0.0186<br>(1.076)    |    0.0120<br>(0.706)    |    0.0065<br>(0.427)    |
| INTANG               | 0.1303\*\*\*<br>(5.628) | 0.1303\*\*\*<br>(5.722) |                         | 0.1365\*\*\*<br>(5.708) | 0.1349\*\*\*<br>(6.055) | 0.1232\*\*\*<br>(5.358) |
| RND                  | 0.1468\*\*\*<br>(6.371) | 0.1424\*\*\*<br>(6.150) | 0.1433\*\*\*<br>(6.211) |                         | 0.1540\*\*\*<br>(7.055) | 0.1512\*\*\*<br>(6.425) |
| SGA                  | 0.0726\*\*\*<br>(4.199) | 0.0727\*\*\*<br>(4.185) | 0.0758\*\*\*<br>(4.362) | 0.1004\*\*\*<br>(5.728) |                         | 0.0696\*\*\*<br>(3.965) |
| FIN                  | 0.0509\*\*\*<br>(2.985) | 0.0447\*\*\*<br>(2.900) |    0.0184<br>(0.999)    | 0.0626\*\*\*<br>(3.630) |  0.0479\*\*<br>(2.817)  |                         |
| Controls             |           Yes           |           Yes           |           Yes           |           Yes           |           Yes           |           Yes           |
| Industry and year FE |           Yes           |           Yes           |           Yes           |           Yes           |           Yes           |           Yes           |
| Dropped component    |          CAPEX          |          TANG           |         INTANG          |           RND           |           SGA           |           FIN           |
| Observations         |         23,726          |         23,726          |         23,726          |         23,726          |         23,726          |         23,726          |

*Notes: Each column re-estimates the simultaneous DML specification while omitting the component identified in the “Dropped component” row. The baseline estimates containing all six components are reported in Table 5, and variable scaling follows Section 3.3. All specifications use tuned LightGBM nuisance learners with firm-blocked five-fold cross-fitting and firm-level multiplier-bootstrap inference. t-statistics are reported in parentheses. \*, \*\*, and \*\*\* denote significance at the 10%, 5%, and 1% levels.*

### 4.5.3. Instrumental-variable diagnostics

The main analysis assumes conditional exogeneity after adjustment for the observed covariates. I therefore complement it with a double machine learning instrumental-variable model (DML-IV). Each component is instrumented with its own second lag and a leave-one-out industry-year peer average. The Anderson statistic assesses relevance and the Sargan test assesses the overidentifying restrictions.

Table 8 reports the IV results. The lag instruments use persistence in investment choices, while the peer instruments measure leave-one-out peer investment in the same industry-year setting. The Anderson underidentification test rejects underidentification (p = 0.000). The Sargan test gives p = 0.102, so the overidentifying restrictions are not rejected at conventional levels. This result does not prove the exclusion restriction, particularly because industry-year shocks may directly affect valuation. The IV estimates are supporting evidence on reverse causality, not a standalone quasi-experimental design.

The IV estimates are directionally consistent with the simultaneous DML results. CAPEX, INTANG, RND, and SGA remain positive after multiple-testing adjustment, while FIN is positive but less precisely estimated. TANG does not survive the Romano-Wolf correction and therefore does not overturn its weak result in Tables 4 and 5. The somewhat larger IV estimates for several components may reflect attenuation from noisy regressors, but this remains an interpretation rather than a direct measurement-error test (Griliches and Hausman, 1986).


**Table 8. DML-IV estimates with lagged and peer instruments**

| Investment component     |      TobinQ effect      | Romano-Wolf p | BY p  | Bonferroni p | Ind.-adj. TobinQ effect | Romano-Wolf p | BY p  | Bonferroni p |
| ------------------------ | :---------------------: | :-----------: | :---: | :----------: | :---------------------: | :-----------: | :---: | :----------: |
| **CAPEX**                | 0.0662\*\*\*<br>(3.537) |     0.000     | 0.000 |    0.000     | 0.0653\*\*\*<br>(3.488) |     0.001     | 0.000 |    0.000     |
| **TANG**                 |   0.0348\*<br>(1.814)   |     0.056     | 0.137 |    0.336     |   0.0337\*<br>(1.744)   |     0.066     | 0.162 |    0.396     |
| **INTANG**               | 0.1536\*\*\*<br>(5.661) |     0.000     | 0.000 |    0.000     | 0.1516\*\*\*<br>(5.605) |     0.000     | 0.000 |    0.000     |
| **RND**                  | 0.1455\*\*\*<br>(5.689) |     0.000     | 0.000 |    0.000     | 0.1456\*\*\*<br>(5.698) |     0.000     | 0.000 |    0.000     |
| **SGA**                  | 0.0895\*\*\*<br>(4.909) |     0.000     | 0.000 |    0.000     | 0.0912\*\*\*<br>(4.967) |     0.000     | 0.000 |    0.000     |
| **FIN**                  |  0.0511\*\*<br>(2.166)  |     0.046     | 0.088 |    0.180     |  0.0544\*\*<br>(2.296)  |     0.030     | 0.062 |    0.126     |
| **Controls**             |           Yes           |               |       |              |           Yes           |               |       |              |
| **Industry and year FE** |           Yes           |               |       |              |           Yes           |               |       |              |
| **Anderson statistic**   |        3812.553         |               |       |              |        3812.553         |               |       |              |
| **Anderson p-value**     |          0.000          |               |       |              |          0.000          |               |       |              |
| **Sargan statistic**     |         10.596          |               |       |              |         10.596          |               |       |              |
| **Sargan p-value**       |          0.102          |               |       |              |          0.102          |               |       |              |
| **Observations**         |         23,635          |               |       |              |         23,635          |               |       |              |
*Notes: The table reports simultaneous-treatment DML-IV estimates using the partially linear IV module of DoubleML; variable scaling follows Section 3.3. Each investment component is instrumented with its own second lag and a leave-one-out industry-year peer average. The Anderson test p-value tests for underidentification. The Sargan statistic tests overidentification under homoskedasticity. Romano-Wolf, Benjamini-Yekutieli (BY), and Bonferroni adjusted p-values are reported. Standard errors are obtained from a firm-level multiplier bootstrap variance estimator with 1,000 replications. t-statistics are reported in parentheses. \*\*\*, \*\*, and \* denote significance at the 1%, 5%, and 10% levels.*


## 4.6. Intertemporal dynamics

I next examine whether the component estimates change over time. The sample is divided into 2012–2015, 2016–2018, 2019–2021, and 2022–2025. These estimates are descriptive because the periods differ in macroeconomic conditions, accounting practice, and COVID-related shocks.

**Table 9. Evolution of TobinQ effects across subperiods**

| **Investment component** |      **2012–2015**      |      **2016–2018**      |      **2019–2021**      |      **2022–2025**      |
| ------------------------ | :---------------------: | :---------------------: | :---------------------: | :---------------------: |
| **CAPEX**                | 0.0343\*\*<br>(2.617)  | 0.0617\*\*\*<br>(3.020) | 0.0759\*\*\*<br>(3.252) | 0.0461\*\*\*<br>(2.925) |
| **TANG**                 |    0.0067<br>(0.324)    |   -0.0049<br>(-0.183)   |    0.0252<br>(0.987)    |    0.0000<br>(0.002)    |
| **INTANG**               | 0.1511\*\*\*<br>(4.445) | 0.1787\*\*\*<br>(4.594) | 0.1077\*\*\*<br>(3.442) | 0.0750\*\*\*<br>(2.834) |
| **RND**                  |    0.0159<br>(0.639)    | 0.1552\*\*\*<br>(4.046) | 0.2267\*\*\*<br>(6.656) | 0.1703\*\*\*<br>(5.284) |
| **SGA**                  | 0.1334\*\*\*<br>(4.981) | 0.0843\*\*\*<br>(2.993) |  0.0644\*\*<br>(2.462)  |   0.0329\*<br>(1.830)   |
| **FIN**                  |    0.0454<br>(1.623)    |    0.0474<br>(1.534)    | 0.0829\*\*\*<br>(2.985) |   0.0391\*<br>(1.760)   |
| **Controls**             |           Yes           |           Yes           |           Yes           |           Yes           |
| **Industry and year FE** |           Yes           |           Yes           |           Yes           |           Yes           |
| **Observations**         |          5,565          |          4,546          |          5,486          |          8,129          |

*Notes: The table reports DML estimates with firm-blocked five-fold cross-fitting, five repeated sample splits, and median aggregation. Each component is standardised using the full estimation-sample mean and standard deviation before the subperiod split, so every column represents the same one-full-sample-SD increase. Standard errors are obtained from a firm-level multiplier bootstrap variance estimator based on DML influence scores with 1,000 replications. Hyperparameters were selected using 5-fold cross-validation. The final LightGBM configuration used a maximum depth of 5 and a learning rate of 0.05. t-statistics are reported in parentheses.*

Table 9 shows that the relative magnitudes change over time, although the broad component pattern remains. The clearest change concerns R&D: its estimate is negligible in 2012–2015, becomes positive from 2016 onward, and is largest during 2019–2021. FIN is also most visible during 2019–2021, consistent with the greater value of financial flexibility during disruption but not identifying that mechanism (Fahlenbrach et al., 2021). CAPEX is comparatively stable, while TANG remains statistically weak in every subperiod. Because the windows also differ in firm composition and macroeconomic conditions, these estimates describe temporal heterogeneity rather than a structural time trend.



## 4.7. Heterogeneity across market segments, governance, and technology

This subsection examines whether the component estimates differ across firm environments. The subgroup results are descriptive rather than evidence of structural moderation. The analysis compares KOSPI and KOSDAQ firms, chaebol and non-chaebol firms, and high-tech and low-tech industries. Table 10 reports the simultaneous-treatment estimates.

### 4.7.1. Market segment

The clearest market-segment difference concerns INTANG, whose coefficient is significantly larger among KOSDAQ firms than among KOSPI firms after within-contrast Holm adjustment. Balance-sheet recognition may be more informative in the relatively opaque KOSDAQ environment, although the analysis does not test this mechanism directly (Aboody and Lev, 1998). The R&D and FIN point estimates differ across segments, but their between-group differences are not statistically established after adjustment.

### 4.7.2. Business-group affiliation

The principal business-group result concerns FIN: its coefficient is positive for non-chaebol firms and significantly lower for chaebol affiliates after within-contrast Holm adjustment. This pattern is consistent with internal capital markets reducing the marginal value of independently held financial assets, but it is not a direct test of intragroup transfers (Shin and Park, 1999; Almeida et al., 2015). The remaining component differences across the chaebol split are not statistically precise.

### 4.7.3. Technology orientation

Technology-based differences are less uniform. TANG is more negative and R&D is lower in point-estimate terms among high-tech firms, but neither difference survives within-contrast Holm adjustment. SGA is numerically larger among high-tech firms, but the difference is imprecise. The technology split therefore provides limited evidence of systematic component repricing and should not be used to claim a universal high-tech premium.


**Table 10. Heterogeneous TobinQ effects across firm groups**

| Investment component   |      KOSPI<br>(1)       |      KOSDAQ<br>(2)      |      Diff<br>(1)-(2)      |     Chaebol<br>(3)      |   Non-chaebol<br>(4)    |      Diff<br>(3)-(4)      |    High-tech<br>(5)     |     Low-tech<br>(6)     |     Diff<br>(5)-(6)     |
| ---------------------- | :---------------------: | :---------------------: | :-----------------------: | :---------------------: | :---------------------: | :-----------------------: | :---------------------: | :---------------------: | :---------------------: |
| **CAPEX**              | 0.0531\*\*\*<br>(3.542) | 0.0521\*\*\*<br>(4.305) |     0.0009<br>(0.049)     |   0.0571\*<br>(1.851)   | 0.0548\*\*\*<br>(5.497) |     0.0023<br>(0.071)     |    0.0395<br>(1.590)    | 0.0590\*\*\*<br>(5.713) |   -0.0195<br>(-0.724)   |
| **TANG**               |   -0.0347<br>(-1.592)   |    0.0312<br>(1.334)    |   -0.0659\*\*<br>(-2.061) |   -0.0510<br>(-1.410)   |    0.0102<br>(0.545)    |    -0.0612<br>(-1.503)    |   -0.0709\*<br>(-1.744) |    0.0213<br>(1.204)    | -0.0922\*\*<br>(-2.079) |
| **INTANG**             |    0.0422<br>(1.579)    | 0.1703\*\*\*<br>(5.561) | -0.1281\*\*\*<br>(-3.149) |  0.0851\*\*<br>(2.035)  | 0.1484\*\*\*<br>(5.537) |    -0.0634<br>(-1.276)    |    0.0654<br>(1.556)    | 0.1595\*\*\*<br>(5.420) |   -0.0941\*<br>(-1.835)   |
| **RND**                | 0.2795\*\*\*<br>(4.241) | 0.1293\*\*\*<br>(4.901) |  0.1501\*\*<br>(2.115)  | 0.3241\*\*\*<br>(2.731) | 0.1444\*\*\*<br>(6.072) |     0.1797<br>(1.484)     |    0.0584<br>(1.121)    | 0.1855\*\*\*<br>(6.857) | -0.1271\*\*<br>(-2.165) |
| **SGA**                |  0.0844\*\*<br>(2.606)  | 0.0755\*\*\*<br>(4.147) |     0.0089<br>(0.240)     |   0.0851\*<br>(1.886)   | 0.0852\*\*\*<br>(4.465) |    -0.0001<br>(-0.002)    | 0.1054\*\*\*<br>(2.887) | 0.0583\*\*\*<br>(3.362) |    0.0471<br>(1.165)     |
| **FIN**                |    0.0204<br>(0.885)    |  0.0511\*\*<br>(2.270)  |    -0.0307<br>(-0.954)    |   -0.0444<br>(-1.252)   | 0.0641\*\*\*<br>(3.332) | -0.1085\*\*\*<br>(-2.690) |    0.0627<br>(1.360)    |  0.0415\*\*<br>(2.283)  |    0.0212<br>(0.427)    |
| **Controls**           |           Yes           |           Yes           |                           |           Yes           |           Yes           |                           |           Yes           |           Yes           |                         |
| **Industry & Year FE** |           Yes           |           Yes           |                           |           Yes           |           Yes           |                           |           Yes           |           Yes           |                         |
| **Observations**       |          9,029          |         14,697          |                           |          3,267          |         20,459          |                           |          4,097          |         19,629          |                         |


*Notes: The table reports simultaneous-treatment DML estimates of TobinQ for each firm subgroup, together with the between-group difference Δ = θ\_A − θ\_B. Each component is standardised using the full estimation-sample mean and standard deviation before the subgroup split, so every column represents the same one-full-sample-SD increase. Standard errors for Δ are computed under the independence of the two subgroup estimators. Hyperparameters were selected using 5-fold cross-validation. Bootstrap SE and t-statistics are from 1,000 firm-level multiplier bootstrap replications with cross-fitting blocked at the firm level so that all observations for a given firm fall in the same fold. t-statistics are reported in parentheses. Stars beside difference estimates use raw p-values; within-contrast Holm-adjusted p-values are used in the text. Significance levels: \*\*\* p<0.01, \*\* p<0.05, \* p<0.10.*




Together, these subgroup results address RQ2 and show that component estimates are not uniform across firm environments. The market-segment, business-group, and technology splits are descriptive implementations of this common question. Under within-contrast Holm adjustment, the clearest differences are stronger INTANG valuation relevance among KOSDAQ firms and stronger FIN valuation relevance among non-chaebol firms. The remaining patterns are not sufficiently precise after adjustment to support a component-specific mechanism. None of the proposed mechanisms is estimated directly, so they remain interpretations rather than mediation tests.

## 4.8. External bank uncertainty as a moderator

The pooled estimates show that investment components do not have the same valuation relevance. I next examine whether their conditional association with valuation changes with banking uncertainty. Bank uncertainty may alter the value of internal funds, collateral, and liquid financial flexibility. It may also alter how investors interpret R&D, recognised intangible assets, and SG&A.

Table 11 reports simultaneous DML moderation estimates using two annual, disaggregate banking-sector uncertainty measures. For each component, the table presents both its valuation coefficient at the mean level of uncertainty and its interaction with uncertainty. AUNC captures dispersion in unexpected bank asset-growth shocks, while FUNC captures dispersion in unexpected funding-growth shocks (Buch et al., 2015; Huynh, 2025). Because both indices vary only by year, the table's firm-clustered statistics are interpreted as descriptive conditional valuation patterns rather than firm-level causal responses to an exogenous banking shock.

Under the table's firm-clustered raw p-values, R&D interacts negatively with both uncertainty measures and SGA interacts positively. The exact calendar-year wild-bootstrap sensitivity, which evaluates the six pre-specified interactions as a family for each index, qualifies this pattern. R&D does not retain Romano-Wolf-adjusted significance for either AUNC (p = 0.137) or FUNC (p = 0.308). The negative CAPEX x AUNC interaction retains adjusted significance (p = 0.005), while the positive SGA x FUNC interaction does so for funding uncertainty (p = 0.022); SGA x AUNC is marginal after adjustment (p = 0.071). The remaining interactions are imprecise under the year-clustered test. These two clustering approaches do not produce identical rankings, so the results are best read as a limited set of conditional patterns rather than as evidence of a general repricing mechanism.

These results address RQ3 cautiously. Banking uncertainty may be associated with component-specific changes in conditional valuation relevance, but the evidence is sensitive to the level at which inference is conducted and does not support a uniform result across the investment mix.

**Table 11. Bank uncertainty and investment-component valuation**
| Investment component and interaction | TobinQ: AUNC           | TobinQ: FUNC           | Ind.-adj. TobinQ: AUNC | Ind.-adj. TobinQ: FUNC |
| ------------------------------------ | ---------------------- | ---------------------- | ---------------------- | ---------------------- |
| CAPEX                                | 0.049\*\*<br>(2.292)   | 0.052\*\*<br>(2.224)   | 0.045\*\*<br>(2.121)   | 0.055\*\*<br>(2.323)   |
| CAPEX x uncertainty                  | -0.030<br>(-1.056)     | -0.029<br>(-0.989)     | -0.031<br>(-1.064)     | -0.040<br>(-1.324)     |
| TANG                                 | 0.031<br>(1.286)       | 0.036<br>(1.406)       | 0.036<br>(1.491)       | 0.039<br>(1.514)       |
| TANG x uncertainty                   | 0.022<br>(0.689)       | 0.004<br>(0.090)       | 0.031<br>(0.981)       | 0.022<br>(0.563)       |
| INTANG                               | 0.214\*\*\*<br>(3.906) | 0.196\*\*\*<br>(3.351) | 0.218\*\*\*<br>(4.040) | 0.213\*\*\*<br>(3.595) |
| INTANG x uncertainty                 | 0.045<br>(1.150)       | 0.032<br>(0.803)       | 0.044<br>(1.052)       | 0.035<br>(0.855)       |
| RND                                  | 0.174\*\*\*<br>(3.973) | 0.269\*\*\*<br>(4.825) | 0.202\*\*\*<br>(4.587) | 0.291\*\*\*<br>(5.300) |
| RND x uncertainty                    | -0.065\*\*<br>(-2.438) | -0.058\*\*<br>(-1.896) | -0.064\*\*<br>(-2.348) | -0.075\*\*<br>(-2.446) |
| SGA                                  | 0.059\*<br>(1.684)     | 0.054\*<br>(1.638)     | 0.061\*<br>(1.749)     | 0.056\*<br>(1.769)     |
| SGA x uncertainty                    | 0.068\*\*<br>(2.465)   | 0.073\*\*<br>(2.447)   | 0.070\*\*\*<br>(2.546) | 0.089\*\*\*<br>(2.895) |
| FIN                                  | 0.068\*<br>(2.074)     | 0.054<br>(1.519)       | 0.063\*<br>(1.924)     | 0.052<br>(1.499)       |
| FIN x uncertainty                    | 0.032<br>(0.652)       | -0.015<br>(-0.260)     | 0.030<br>(0.624)       | -0.027<br>(-0.474)     |
| Controls                             | Yes                    | Yes                    | Yes                    | Yes                    |
| Industry + Year FE                   | Yes                    | Yes                    | Yes                    | Yes                    |
| Observations                         | 23,726                 | 23,726                 | 23,726                 | 23,726                 |

*Notes: The table reports the 12 target coefficients from each simultaneous-treatment DML moderation model: six investment-component coefficients and six component-by-uncertainty interactions; variable and interaction scaling follows Section 3.3. AUNC is the annual cross-sectional standard deviation of bank-specific residuals from total-asset log-growth regressions with bank and year fixed effects. FUNC is constructed identically from broad-funding log growth, where broad funding equals deposit liabilities plus borrowed funding. The displayed t-statistics use firm-level multiplier-bootstrap inference. A focused sensitivity analysis re-evaluates the six interaction terms for each annual index using an exact Rademacher wild bootstrap over 14 calendar-year clusters and all 16,384 sign patterns; its adjusted p-values are reported in the text. All specifications use tuned LightGBM nuisance learners, industry and year fixed effects, and firm-blocked five-fold cross-fitting. Significance levels in this table are based on raw firm-clustered p-values: \*\*\* p<0.01, \*\* p<0.05, \* p<0.10.*

# 5. Discussion and economic implications

## 5.1. Overall interpretation

The results depend on the investment mix rather than on one investment margin. R&D is positive and statistically robust in the main models. Recognised intangibles and CAPEX retain positive incremental relevance, and SG&A remains positive after the innovation-related components are included. FIN becomes more visible in the simultaneous model, whereas TANG has limited incremental relevance. The bank-uncertainty results add one qualification: raw-p-value interactions are compatible with lower conditional valuation relevance for current R&D and higher conditional relevance for broad SG&A in more uncertain banking years. They do not identify organisational resilience or a specific SG&A mechanism.

Standalone and incremental valuation relevance are different empirical objects. The separate models ask whether a component is informative in isolation. The simultaneous models ask whether it remains informative after correlated investment choices are included. A coefficient on R&D, SG&A, FIN, or TANG in a sparse model may therefore reflect omitted information from other components.

## 5.2. Contribution to valuation research

The study distinguishes standalone from incremental investment-component valuation. Existing studies often focus on one category, such as capital expenditure, R&D, organisational capital, or cash holdings. That approach can obscure the fact that firms allocate capital across physical capacity, knowledge creation, organisational capability, and financial flexibility at the same time.

Estimating the six components separately and jointly shows that single-component estimates can move in either direction once the other observed investment margins are included. FIN is weak in the separate model but positive in the simultaneous model, and Table 2 reports a sizeable negative FIN-TANG correlation (-0.453). This is consistent with balance-sheet substitution: firms with fewer tangible assets tend to hold more financial assets, while tangibility has null-to-negative valuation relevance in the main analysis. In a standalone model, FIN mixes financial flexibility with the valuation profile of firms that hold fewer tangible assets. Once TANG and the remaining components are included, FIN is more readily interpreted as financial flexibility. SG&A moves in the opposite direction. It remains positive, but falls when R&D and recognised intangibles are included, which suggests shared information with other knowledge-related components.

The evidence also speaks to multi-capital q-theory and real-options reasoning. The findings do not imply that one component is universally dominant. Instead, they show that valuation depends on whether an investment component provides incremental information within the firm's broader investment mix, and on whether that component embeds uncertainty, irreversibility, redeployability, or financial flexibility. This is the central theoretical implication of the empirical analysis.

## 5.3. Practical implications

For managers, the estimates indicate that an isolated increase in one component is difficult to interpret without the surrounding investment mix. In Table 5, a one-standard-deviation increase in R&D is associated with a 0.145 increase in TobinQ, about 10.6% of the sample mean and almost three times the CAPEX effect of 0.050. Recognised intangible assets are close to R&D in standardised magnitude (0.133); SG&A (0.072) and FIN (0.051) are smaller. Innovation and recognised intangible accumulation therefore carry the largest reported valuation associations, while organisational expenditure and financial flexibility are smaller components of the broader system.

For investors and analysts, the results imply that valuation models should treat the components as a connected system. A high level of R&D, SG&A, or financial assets is difficult to interpret without the other components. FIN is an example: it has a larger conditional valuation coefficient after controlling for tangibility, consistent with liquid flexibility rather than simply the absence of physical assets. The simultaneous DML estimates separate the incremental part of each component from the part that proxies for correlated investment choices.

For Korean corporate governance, the chaebol and non-chaebol split is especially informative. The lower FIN premium among chaebol firms is consistent with internal capital markets reducing the marginal value of standalone financial asset holdings. For non-chaebol firms, financial assets carry a stronger flexibility signal in the estimates, possibly because they cannot be substituted as easily by group-level support.

## 5.4. Limitations and future research

Several limitations qualify the interpretation of these findings. First, the DML design identifies valuation relevance after flexible adjustment for observable confounders. It should not be interpreted as showing how firm value would respond to an exogenous policy-induced change in an investment component. The DML-IV analysis provides sensitivity evidence regarding reverse causality, but the exclusion restrictions remain contestable.

Second, the simultaneous DML estimates do not identify formal complementarity among investment components. They show whether each component has incremental valuation relevance conditional on the others, but they do not test whether jointly increasing two components creates more value than increasing each component separately. Future work could use heterogeneous treatment effects or interaction-based designs to study such complementarities more directly.

Third, the banking-uncertainty analysis has an additional time-series limitation. AUNC and FUNC are annual series shared by all firms, so year fixed effects absorb their common main effects and the interaction coefficients are identified from differences in firm-level investment exposure within each year. The limited number of annual periods nevertheless makes the moderation estimates potentially sensitive to common year-specific shocks. They are therefore treated as conditional valuation patterns rather than causal effects of externally generated banking shocks.

Fourth, the six investment components do not exhaust all relevant investment margins. Human capital, environmental investment, digital transformation expenditure, and supply-chain resilience are likely to matter for valuation, but consistent measurement over the full 2012–2025 Korean listed-firm panel is limited. Finally, the Korean sample provides variation in market segment and business-group affiliation, but the results may not generalise to markets with different ownership structures, accounting rules, and financing institutions.


# 6. Conclusion

This study examines how equity markets value heterogeneous investment components when firms allocate across physical capacity, tangible assets, recognised intangibles, R&D, organisational expenditure, and financial assets. Standalone and incremental valuation relevance are different empirical objects. A component can appear important in isolation because it proxies for correlated choices, or become more informative once those choices are modelled jointly.

Using Korean listed firms from 2012 to 2025, I combine industry and year fixed-effects regressions with double/debiased machine learning. The fixed-effects regressions provide a conventional benchmark. The DML design then flexibly partials out observed confounding relationships and estimates separate-treatment and simultaneous-treatment specifications under firm-blocked cross-fitting and firm-level bootstrap inference.

The main evidence shows that R&D has a positive and statistically robust valuation association in the principal separate, simultaneous, alternative-learner, and DML-IV specifications. Recognised intangible assets, capital expenditures, SG&A, and financial assets also retain positive incremental relevance in the simultaneous specification. Tangible asset intensity has limited independent valuation relevance once the broader investment-component system is considered, although leave-one-component-out estimates suggest that tangibility can carry a negative signal when intangible accumulation is omitted. These estimates do not establish that aggregate FIN is a pure financial-flexibility measure.

The heterogeneity analysis shows that valuation weights differ across firm environments. Recognised intangible assets have a significantly larger coefficient among KOSDAQ firms, and FIN has a significantly larger coefficient among non-chaebol firms. In the technology split, TANG is significantly more negative among high-tech firms and the SGA difference is only marginal, while the R&D difference is not statistically established. These patterns suggest that the market's interpretation of an investment component depends on information conditions, governance structure, and the firm's broader investment context.

The banking-uncertainty extension further qualifies the pooled estimates. At the mean level of each uncertainty index, CAPEX, INTANG, and R&D retain positive valuation coefficients, while the interaction estimates based on raw p-values are consistent with weaker conditional valuation relevance for R&D and stronger relevance for SG&A in more uncertain banking years. The remaining interactions are imprecise, so the evidence does not support a general repricing of all investment components.

The estimates should be interpreted with caution. DML identifies valuation relevance after adjustment for observed covariates, not externally identified responses to exogenous investment shocks. The DML-IV results support the direction of the main estimates, and the overidentification test does not reject the overidentifying restrictions at conventional levels, although this does not establish instrument validity. The study shows that firm valuation depends on the investment-component system as a whole and offers a flexible framework for separating standalone from incremental valuation relevance.


# Appendix

## Appendix A. Empirical tuning details

**Table A1. LightGBM tuning grid and selected parameters**

| **Parameter**               | **Candidate values**     |
| --------------------------- | ------------------------ |
| **Maximum depth**           | 2, 3, 4, 5, 6            |
| **Learning rate/shrinkage** | 0\.05, 0.035, 0.03, 0.01 |
| **Number of leaves**        | 4, 8, 16, 24, 32         |
| **Minimum child samples**   | 30, 50                   |
| **Tuning folds**            | 5-fold cross-validation  |
| **Maximum tuning sample**   | 8,000 observations       |


**Table A2. Nuisance-model RMSE for the main DML estimates**

| Investment component | Separate: outcome RMSE | Separate: treatment RMSE | Simultaneous: outcome RMSE | Simultaneous: treatment RMSE |
| -------------------- | :--------------------: | :----------------------: | :------------------------: | :--------------------------: |
| CAPEX                |         0.896          |          0.862           |           0.867            |            0.803             |
| TANG                 |         0.896          |          0.687           |           0.865            |            0.586             |
| INTANG               |         0.896          |          0.911           |           0.879            |            0.843             |
| RND                  |         0.896          |          0.849           |           0.875            |            0.764             |
| SGA                  |         0.896          |          0.858           |           0.867            |            0.824             |
| FIN                  |         0.896          |          0.759           |           0.865            |            0.692             |

*Notes: The table reports cross-fitted RMSE statistics for the outcome nuisance model and treatment nuisance model in the TobinQ DML specifications. Treatment variables are standardised before DML estimation, so treatment RMSE values refer to the standardised treatment scale.*


























## Appendix B. Variable definitions

**Table B1. Variable definitions**

| Abbreviation       | Description                                                                                                                                                                                       | Citation                                                                                                                 |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| TobinQ             | Tobin's Q (TobinQ) is the market value of equity plus the book value of debt, scaled by total assets.                                                                                             | Hayashi (1982); Chung and Pruitt (1994); Peters and Taylor (2017)                                                        |
| Ind-adj TobinQ     | Industry-adjusted Tobin's Q (Ind-adj TobinQ) is firm-level TobinQ minus the industry-year median TobinQ.                                                                                          | Berger and Ofek (1995); this study                                                                                       |
| CAPEX              | Capital expenditure intensity (CAPEX) is capital expenditure scaled by total assets and captures investment in physical productive capacity.                                                      | Hayashi (1982); McConnell and Muscarella (1985)                                                                          |
| TANG               | Tangibility (TANG) is tangible assets scaled by total assets and captures the stock of physical, collateralizable assets.                                                                         | Titman and Wessels (1988); Almeida and Campello (2007)                                                                   |
| INTANG             | Recognised intangible assets (INTANG) are accounting-recognised intangible assets scaled by total assets.                                                                                         | Lev (2001); Peters and Taylor (2017); Ewens et al. (2024)                                                                |
| RND                | Research and development intensity (RND) is R&D expenditure scaled by total assets and captures current investment in knowledge creation.                                                         | Lev and Sougiannis (1996); Chan et al. (2001); Hall et al. (2005); Peters and Taylor (2017)                              |
| SGA                | Selling, general, and administrative expenditure intensity (SGA) is SG&A expenditure scaled by total assets and is used as a proxy for organisational and customer-related capability investment. | Lev and Radhakrishnan (2003, 2005); Eisfeldt and Papanikolaou (2013); Enache and Srivastava (2018); Banker et al. (2019) |
| FIN                | Financial asset intensity (FIN) is financial assets scaled by total assets and captures the firm's financial holdings and liquidity-like flexibility.                                             | Opler et al. (1999); Faulkender and Wang (2006); Gamba and Triantis (2008); Pinkowitz et al. (2006)                      |
| SHORTFIN           | Short-term financial asset intensity (SHORTFIN) is short-term financial assets scaled by total assets and captures liquid financial flexibility.                                                  | Faulkender and Wang (2006); Gamba and Triantis (2008); this study                                                        |
| LONGFIN            | Long-term financial asset intensity (LONGFIN) is long-term financial assets scaled by total assets and captures longer-maturity financial holdings.                                               | Davis (2018); Tori and Onaran (2018); this study                                                                         |
| SIZE               | Firm size (SIZE) is the natural logarithm of total assets.                                                                                                                                        | Rajan and Zingales (1995); Fama and French (1998)                                                                        |
| LEV                | Leverage (LEV) is total liabilities scaled by total assets.                                                                                                                                       | Rajan and Zingales (1995); Titman and Wessels (1988)                                                                     |
| ROA                | Return on assets (ROA) is net income scaled by total assets and captures accounting profitability.                                                                                                | Fama and French (1998); this study                                                                                       |
| CASH               | Cash holdings (CASH) are cash and cash equivalents scaled by total assets.                                                                                                                        | Opler et al. (1999); Almeida et al. (2004)                                                                               |
| GROWTH             | Sales growth (GROWTH) is the annual growth rate of sales and captures operating expansion.                                                                                                        | Fama and French (1998); this study                                                                                       |
| OCF                | Operating cash flow (OCF) is operating cash flow scaled by total assets.                                                                                                                          | Almeida et al. (2004); Brown et al. (2009)                                                                               |
| AGE                | Firm age (AGE) is the natural logarithm of firm age.                                                                                                                                              | Brown et al. (2009); this study                                                                                          |
| LIQ                | Liquidity (LIQ) captures short-term liquidity capacity, measured using the firm's liquidity ratio from the financial statements.                                                                  | Opler et al. (1999); Almeida et al. (2004); this study                                                                   |
| FOREIGN            | Foreign ownership (FOREIGN) is the percentage of shares held by foreign investors.                                                                                                                | Baek et al. (2004); this study                                                                                           |
| OWNCONC            | Ownership concentration (OWNCONC) is the ownership share held by large or controlling shareholders.                                                                                               | Claessens et al. (2000); Joh (2003); Baek et al. (2004)                                                                  |
| BIG4               | Big 4 auditor indicator (BIG4) equals one if the firm is audited by a Big 4 audit firm and zero otherwise.                                                                                        | FNGuide; this study                                                                                                      |
| KM_RATIO           | KM financing proxy (KM_RATIO) captures the firm's financing-condition proxy used in the empirical controls.                                                                                       | FNGuide; this study                                                                                                      |
| STD_PRESSURE       | Short-term debt pressure (STD_PRESSURE) measures the share of short-term debt pressure in the firm's financing structure.                                                                         | FNGuide; this study                                                                                                      |
| INTEREST_BURDEN    | Interest burden (INTEREST_BURDEN) captures the firm's interest expense burden relative to its financial scale.                                                                                    | FNGuide; this study                                                                                                      |
| CHAEBOL            | Chaebol affiliation (CHAEBOL) equals one if the firm belongs to a Korean chaebol business group and zero otherwise.                                                                               | Claessens et al. (2000); Joh (2003); Baek et al. (2004); Ducret and Isakov (2020, 2024)                                  |
| LT_DEBT_RATIO      | Long-term debt ratio (LT_DEBT_RATIO) measures the long-term debt component of the firm's debt structure.                                                                                          | Titman and Wessels (1988); Rajan and Zingales (1995); this study                                                         |
| ANALYST_MONITORING | Analyst monitoring (ANALYST_MONITORING) is a transformed analyst-coverage measure, with zero indicating no analyst coverage.                                                                      | FNGuide; this study                                                                                                      |
| LOSS_DUMMY         | Loss indicator (LOSS_DUMMY) equals one if the firm reports a loss and zero otherwise.                                                                                                             | FNGuide; this study                                                                                                      |
| KOSPI              | KOSPI listing indicator (KOSPI) identifies firms listed on the Korea Composite Stock Price Index market segment.                                                                                  | FNGuide; this study                                                                                                      |
| KOSDAQ             | KOSDAQ listing indicator (KOSDAQ) identifies firms listed on the Korean growth-oriented KOSDAQ market segment.                                                                                    | FNGuide; this study                                                                                                      |
| HIGH_TECH          | High-technology industry indicator (HIGH_TECH) identifies firms operating in high-technology industries according to the industry classification used in the heterogeneity analysis.              | Hall et al. (2005); Peters and Taylor (2017); this study                                                                 |

*Notes: The table defines the main dependent variables, investment components, controls, and subgroup indicators used in the empirical analysis. Lagged versions of the investment-component variables are used in the main DML and fixed-effects specifications unless otherwise stated.*


# References

Aboody, D., and Lev, B. (1998). The value relevance of intangibles: The case of software capitalization. *Journal of Accounting Research*, 36, 161-191. <https://doi.org/10.2307/2491312>

Abel, A. B., and Eberly, J. C. (1994). A unified model of investment under uncertainty. *American Economic Review*, 84(5), 1369-1384.

Acharya, V. V., Almeida, H., and Campello, M. (2007). Is cash negative debt? A hedging perspective on corporate financial policies. *Journal of Financial Intermediation*, 16(4), 515-554. <https://doi.org/10.1016/j.jfi.2007.04.001>

Akerlof, G. A. (1970). The market for "lemons": Quality uncertainty and the market mechanism. *Quarterly Journal of Economics*, 84(3), 488-500.

Almeida, H., Campello, M., and Weisbach, M. S. (2004). The cash flow sensitivity of cash. *Journal of Finance*, 59(4), 1777-1804. <https://doi.org/10.1111/j.1540-6261.2004.00679.x>

Almeida, H., and Campello, M. (2007). Financial constraints, asset tangibility, and corporate investment. *Review of Financial Studies*, 20(5), 1429-1460. <https://doi.org/10.1093/rfs/hhm019>

Almeida, H., Kim, C.-S., and Kim, H. B. (2015). Internal capital markets in business groups: Evidence from the Asian financial crisis. *Journal of Finance*, 70(6), 2539-2586.

Alfaro, I., Bloom, N., and Lin, X. (2024). The finance uncertainty multiplier. *Journal of Political Economy*, 132(2), 577-615. <https://doi.org/10.1086/726230>

Bach, P., Chernozhukov, V., Kurz, M. S., and Spindler, M. (2022). DoubleML: An object-oriented implementation of double machine learning in Python. *Journal of Machine Learning Research*, 23(53), 1-6.

Baek, J.-S., Kang, J.-K., and Park, K. S. (2004). Corporate governance and firm value: Evidence from the Korean financial crisis. *Journal of Financial Economics*, 71(2), 265-313. <https://doi.org/10.1016/S0304-405X(03)00167-3>

Banker, R. D., Huang, R., Natarajan, R., and Zhao, S. (2019). Market valuation of intangible asset: Evidence on SG&A expenditure. *The Accounting Review*, 94(6), 61-90.

Belloni, A., Chernozhukov, V., and Hansen, C. (2014). Inference on treatment effects after selection among high-dimensional controls. *Review of Economic Studies*, 81(2), 608-650. <https://doi.org/10.1093/restud/rdt044>

Benmelech, E., and Bergman, N. K. (2009). Collateral pricing. *Journal of Financial Economics*, 91(3), 339-360.

Benjamini, Y., and Yekutieli, D. (2001). The control of the false discovery rate in multiple testing under dependency. *Annals of Statistics*, 29(4), 1165-1188.

Berger, P. G., and Ofek, E. (1995). Diversification's effect on firm value. *Journal of Financial Economics*, 37(1), 39-65. <https://doi.org/10.1016/0304-405X(94)00798-6>

Bloom, N. (2007). Uncertainty and the dynamics of R&D. *American Economic Review*, 97(2), 250-255. <https://doi.org/10.1257/aer.97.2.250>

Bontempi, M. E. (2016). Investment-uncertainty relationship: Differences between intangible and physical capital. *Economics of Innovation and New Technology*, 25(3), 240-268. <https://doi.org/10.1080/10438599.2015.1076197>

Buch, C. M., Buchholz, M., and Tonzer, L. (2015). Uncertainty, bank lending, and bank-level heterogeneity. *IMF Economic Review*, 63(4), 919-954. <https://doi.org/10.1057/imfer.2015.35>

Borusyak, K., Hull, P., and Jaravel, X. (2022). Quasi-experimental shift-share research designs. *Review of Economic Studies*, 89(1), 181-213. <https://doi.org/10.1093/restud/rdab030>

Brown, J. R., Fazzari, S. M., and Petersen, B. C. (2009). Financing innovation and growth: Cash flow, external equity, and the 1990s R&D boom. *Journal of Finance*, 64(1), 151-185.

Campello, M., and Giambona, E. (2013). Real assets and capital structure. *Journal of Financial and Quantitative Analysis*, 48(5), 1333-1370. <https://doi.org/10.1017/S0022109013000413>

Chan, L. K. C., Lakonishok, J., and Sougiannis, T. (2001). The stock market valuation of research and development expenditures. *Journal of Finance*, 56(6), 2431-2456.

Chernozhukov, V., Chetverikov, D., and Kato, K. (2013). Gaussian approximations and multiplier bootstrap for maxima of sums of high-dimensional random vectors. *Annals of Statistics*, 41(6), 2786-2819. <https://doi.org/10.1214/13-AOS1161>

Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., and Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. *Econometrics Journal*, 21(1), C1-C68.

Chirinko, R. S. (1993). Multiple capital inputs, Q, and investment spending. *Journal of Economic Dynamics and Control*, 17(5-6), 907-928. <https://doi.org/10.1016/0165-1889(93)90022-K>

Chirinko, R. S., and Schaller, H. (2009). The irreversibility premium. *Journal of Monetary Economics*, 56(3), 390-408.

Chung, K. H., and Pruitt, S. W. (1994). A simple approximation of Tobin's q. *Financial Management*, 23(3), 70-74. <https://doi.org/10.2307/3665623>

Claessens, S., Djankov, S., and Lang, L. H. P. (2000). The separation of ownership and control in East Asian corporations. *Journal of Financial Economics*, 58(1-2), 81-112.

Clarke, P. S., and Polselli, A. (2025). Double machine learning for static panel models with fixed effects. *Econometrics Journal*. <https://doi.org/10.1093/ectj/utaf011>

Corrado, C., and Hulten, C. (2010). How do you measure a "technological revolution"? *American Economic Review*, 100(2), 99-104. <https://doi.org/10.1257/aer.100.2.99>

Crouzet, N., and Eberly, J. (2019). Understanding weak capital investment: The role of market concentration and intangibles. NBER Working Paper No. 25869. <https://doi.org/10.3386/w25869>

Davis, L. E. (2018). Financialization and the non-financial corporation: An investigation of firm-level investment behavior in the United States. *Metroeconomica*, 69(1), 270-307. <https://doi.org/10.1111/meca.12179>

Dittmar, A., and Mahrt-Smith, J. (2007). Corporate governance and the value of cash holdings. *Journal of Financial Economics*, 83(3), 599-634. <https://doi.org/10.1016/j.jfineco.2005.12.006>

Do, T. K., Lai, T. N., and Tran, T. T. C. (2020). Foreign ownership and capital structure dynamics. *Finance Research Letters*, 36, 101337. <https://doi.org/10.1016/j.frl.2019.101337>

Dixit, A. K., and Pindyck, R. S. (1994). *Investment under Uncertainty*. Princeton University Press.

Döttling, R., and Ratnovski, L. (2023). Monetary policy and intangible investment. *Journal of Monetary Economics*, 134, 53-72. <https://doi.org/10.1016/j.jmoneco.2022.11.001>

Drobetz, W., El Ghoul, S., Guedhami, O., and Janzen, M. (2018). Policy uncertainty, investment, and the cost of capital. *Journal of Financial Stability*, 39, 28-45. <https://doi.org/10.1016/j.jfs.2018.08.005>

Doms, M., and Dunne, T. (1998). Capital adjustment patterns in manufacturing plants. *Review of Economic Dynamics*, 1(2), 409-429.

Dong, F., and Doukas, J. (2025). The role of intangible assets in shaping firm value. *European Financial Management*, 31(4), 1325-1353. <https://doi.org/10.1111/eufm.12547>

Ducret, R., and Isakov, D. (2020). The Korea discount and chaebols. *Pacific-Basin Finance Journal*, 63, 101396. <https://doi.org/10.1016/j.pacfin.2020.101396>

Ducret, R., and Isakov, D. (2024). Business group heterogeneity and firm outcomes: Evidence from Korean chaebols. *Global Finance Journal*, 63, 101056. <https://doi.org/10.1016/j.gfj.2024.101056>

Eisfeldt, A. L., and Papanikolaou, D. (2013). Organization capital and the cross-section of expected returns. *Journal of Finance*, 68(4), 1365-1406. <https://doi.org/10.1111/jofi.12034>

Enache, L., and Srivastava, A. (2018). Should intangible investments be reported separately or commingled with operating expenses? New evidence. *Management Science*, 64, 3446-3468. <https://doi.org/10.1287/mnsc.2017.2769>

Ewens, M., Peters, R. H., and Wang, S. (2024). Measuring intangible capital with market prices. *Management Science*, 71(1), 407-427.

Fahlenbrach, R., Rageth, K., and Stulz, R. M. (2021). How valuable is financial flexibility when revenue stops? Evidence from the COVID-19 crisis. *Review of Financial Studies*, 34(11), 5474-5521.

Falato, A., Kadyrzhanova, D., Sim, J., and Steri, R. (2022). Rising intangible capital, shrinking debt capacity, and the U.S. corporate savings glut. *Journal of Finance*, 77(5), 2799-2852. <https://doi.org/10.1111/jofi.13174>

Fama, E. F., and French, K. R. (1998). Taxes, financing decisions, and firm value. *Journal of Finance*, 53(3), 819-843. <https://doi.org/10.1111/0022-1082.00036>

Faulkender, M., and Wang, R. (2006). Corporate financial policy and the value of cash. *Journal of Finance*, 61(4), 1957-1990.

Fazzari, S. M., Hubbard, R. G., and Petersen, B. C. (1988). Financing constraints and corporate investment. *Brookings Papers on Economic Activity*, 1988(1), 141-206.

Gamba, A., and Triantis, A. (2008). The value of financial flexibility. *Journal of Finance*, 63(5), 2263-2296. <https://doi.org/10.1111/j.1540-6261.2008.01397.x>

Giebel, M., and Kraft, K. (2024). R&D investments under financing constraints. *Industry and Innovation*, 31(9), 1141-1168. <https://doi.org/10.1080/13662716.2024.2328008>

Goldsmith-Pinkham, P., Sorkin, I., and Swift, H. (2020). Bartik instruments: What, when, why, and how. *American Economic Review*, 110(8), 2586-2624. <https://doi.org/10.1257/aer.20181047>

Gormley, T. A., and Matsa, D. A. (2014). Common errors: How to (and not to) control for unobserved heterogeneity. *Review of Financial Studies*, 27(2), 617-661. <https://doi.org/10.1093/rfs/hht047>

Griliches, Z., and Hausman, J. A. (1986). Errors in variables in panel data. *Journal of Econometrics*, 31(1), 93-118. <https://doi.org/10.1016/0304-4076(86)90058-8>

Hall, B. H. (1993). The stock market's valuation of R&D investment during the 1980's. *American Economic Review*, 83(2), 259-264.

Hall, B. H., Jaffe, A., and Trajtenberg, M. (2005). Market value and patent citations. *RAND Journal of Economics*, 36(1), 16-38.

Hall, B. H., and Oriani, R. (2006). Does the market value R&D investment by European firms? Evidence from a panel of manufacturing firms in France, Germany and Italy. *International Journal of Industrial Organization*, 24(5), 971-993. <https://doi.org/10.1016/j.ijindorg.2005.12.001>

Hayashi, F. (1982). Tobin's marginal q and average q: A neoclassical interpretation. *Econometrica*, 50(1), 213-224.

Hayashi, F., and Inoue, T. (1991). The relation between firm growth and Q with multiple capital goods: Theory and evidence from panel data on Japanese firms. *Econometrica*, 59(3), 731-753. <https://doi.org/10.2307/2938228>

He, Z., and Wintoki, M. B. (2016). The cost of innovation: R&D and high cash holdings in U.S. firms. *Journal of Corporate Finance*, 41, 280-303. <https://doi.org/10.1016/j.jcorpfin.2016.10.006>

Hong, H., Lim, T., and Stein, J. C. (2000). Bad news travels slowly: Size, analyst coverage, and the profitability of momentum strategies. *Journal of Finance*, 55(1), 265-295. <https://doi.org/10.1111/0022-1082.00206>

Hong, S., Oh, F. D., and Shin, D. (2023). Internal capital markets and R&D investment: Evidence from Korean chaebols. *Emerging Markets Finance and Trade*, 59(8), 2493-2506. <https://doi.org/10.1080/1540496X.2023.2185095>

Huynh, J. (2025). Banking uncertainty and corporate financial constraints. *International Journal of Finance & Economics*, 30(1), 626-651. <https://doi.org/10.1002/ijfe.2938>

Huynh, J., and Phan, T. M. H. (2024). Uncertainty in banking and debt financing of firms in Vietnam. *PLOS ONE*, 19(7), e0305724. <https://doi.org/10.1371/journal.pone.0305724>

Huynh, J., and Phan, T. M. H. (2026). The impact of banking uncertainty on firm investment: A look into intangible assets. *PLOS ONE*, 21(1), e0340913. <https://doi.org/10.1371/journal.pone.0340913>

Intara, P., and Suwansin, N. (2024). Intangible assets, firm value, and performance: Does intangible-intensive matter? *Cogent Economics & Finance*, 12(1), 2375341. <https://doi.org/10.1080/23322039.2024.2375341>

Jensen, M. C. (1986). Agency costs of free cash flow, corporate finance, and takeovers. *American Economic Review*, 76(2), 323-329.

Joh, S. W. (2003). Corporate governance and firm profitability: Evidence from Korea before the economic crisis. *Journal of Financial Economics*, 68(2), 287-322.

Kang, M., Kim, S., and Cho, M. K. (2019). The effect of R&D and the control-ownership wedge on firm value: Evidence from Korean chaebol firms. *Sustainability*, 11(10), 2986. <https://doi.org/10.3390/su11102986>

Kaviani, M. S., Kryzanowski, L., Maleki, H., and Savor, P. (2020). Policy uncertainty and corporate credit spreads. *Journal of Financial Economics*, 138(3), 838-865. <https://doi.org/10.1016/j.jfineco.2020.07.001>

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., and Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30.

Kwon, G.-J. (2014). The role of R&D investment in firm valuation for small and medium Korean companies. *Asian Social Science*, 10(15), 169. <https://doi.org/10.5539/ass.v10n15p169>

Leary, M. T., and Roberts, M. R. (2014). Do peer firms affect corporate financial policy? *Journal of Finance*, 69(1), 139-178. <https://doi.org/10.1111/jofi.12094>

Lev, B. (2001). *Intangibles: Management, Measurement, and Reporting*. Brookings Institution Press.

Lev, B., and Radhakrishnan, S. (2003). The measurement of firm-specific organization capital. NBER Working Paper No. 9581.

Lev, B., and Radhakrishnan, S. (2005). The valuation of organization capital. In C. Corrado, J. Haltiwanger, and D. Sichel (Eds.), *Measuring Capital in the New Economy* (pp. 73-110). University of Chicago Press.

Lev, B., and Sougiannis, T. (1996). The capitalization, amortization, and value-relevance of R&D. *Journal of Accounting and Economics*, 21(1), 107-138. <https://doi.org/10.1016/0165-4101(95)00410-6>

Lev, B., and Zarowin, P. (1999). The boundaries of financial reporting and how to extend them. *Journal of Accounting Research*, 37(2), 353-385.

Lee, K., Kim, J. Y., and Lee, O. (2010). Long-term evolution of the firm value and behavior of business groups: Korean chaebols between weak premium, strong discount, and strong premium. *Journal of the Japanese and International Economies*, 24(3), 412-440. <https://doi.org/10.1016/j.jjie.2010.01.004>

Lee, S. (2012). Financial determinants of corporate R&D investment in Korea. *Asian Economic Journal*. <https://doi.org/10.1111/j.1467-8381.2012.02080.x>

La Porta, R., Lopez-de-Silanes, F., Shleifer, A., and Vishny, R. (2002). Investor protection and corporate valuation. *Journal of Finance*, 57(3), 1147-1170.

Li, L. (2026). The role of intangible investment in predicting stock returns: Six decades of evidence. *Financial Management*, 55(1), 99-119. <https://doi.org/10.1111/fima.12505>

Liu, Y., Wen, Y., Xiao, Y., Zhang, L., and Huang, S. (2024). Identification of the enterprise financialization motivation on crowding out R&D innovation: Evidence from listed companies in China. *AIMS Mathematics*, 9(3), 5951-5970. <https://doi.org/10.3934/math.2024291>

Manski, C. F. (1993). Identification of endogenous social effects: The reflection problem. *Review of Economic Studies*, 60(3), 531-542. <https://doi.org/10.2307/2298123>

Manso, G. (2011). Motivating innovation. *Journal of Finance*, 66(5), 1823-1860. <https://doi.org/10.1111/j.1540-6261.2011.01688.x>

McConnell, J. J., and Muscarella, C. J. (1985). Corporate capital expenditure decisions and the market value of the firm. *Journal of Financial Economics*, 14(3), 399-422.

Megna, P., and Klock, M. (1993). The impact of intangible capital on Tobin's q in the semiconductor industry. *American Economic Review*, 83(2), 265-269.

Milgrom, P., and Roberts, J. (1990). The economics of modern manufacturing: Technology, strategy, and organization. *American Economic Review*, 80(3), 511-528.

Mundlak, Y. (1978). On the pooling of time series and cross section data. *Econometrica*, 46(1), 69-85.

Movaghari, H. (2024). *Three applications of machine learning methods in corporate finance* [PhD thesis, University of Glasgow]. University of Glasgow Enlighten: Theses. <https://theses.gla.ac.uk/84298/>

Movaghari, H., Tsoukas, S., and Vagenas-Nanos, E. (2025). Corporate cash policy and double machine learning. *International Journal of Finance & Economics*, 30, 3261-3279. <https://doi.org/10.1002/ijfe.3039>

Myers, S. C. (1977). Determinants of corporate borrowing. *Journal of Financial Economics*, 5(2), 147-175.

Myers, S. C., and Majluf, N. S. (1984). Corporate financing and investment decisions when firms have information that investors do not have. *Journal of Financial Economics*, 13(2), 187-221.

Opler, T., Pinkowitz, L., Stulz, R., and Williamson, R. (1999). The determinants and implications of corporate cash holdings. *Journal of Financial Economics*, 52(1), 3-46.

Petersen, M. A. (2009). Estimating standard errors in finance panel data sets: Comparing approaches. *Review of Financial Studies*, 22(1), 435-480. <https://doi.org/10.1093/rfs/hhn053>

Peters, R. H., and Taylor, L. A. (2017). Intangible capital and the investment-q relation. *Journal of Financial Economics*, 123(2), 251-272. <https://doi.org/10.1016/j.jfineco.2016.03.011>

Pinkowitz, L., Stulz, R., and Williamson, R. (2006). Does the contribution of corporate cash holdings and dividends to firm value depend on governance? A cross-country analysis. *Journal of Finance*, 61(6), 2725-2751. <https://doi.org/10.1111/j.1540-6261.2006.01003.x>

Rajan, R. G., and Zingales, L. (1995). What do we know about capital structure? Some evidence from international data. *Journal of Finance*, 50(5), 1421-1460. <https://doi.org/10.1111/j.1540-6261.1995.tb05184.x>

Robinson, P. M. (1988). Root-N-consistent semiparametric regression. *Econometrica*, 56(4), 931-954. <https://doi.org/10.2307/1912705>

Romano, J. P., and Wolf, M. (2005). Stepwise multiple testing as formalized data snooping. *Econometrica*, 73(4), 1237-1282. <https://doi.org/10.1111/j.1468-0262.2005.00615.x>

Seo, H., and Kim, Y. (2020). Intangible assets investment and firms' performance: Evidence from small and medium-sized enterprises in Korea. *Journal of Business Economics and Management*, 21, 421-445. <https://doi.org/10.3846/jbem.2020.12022>

Shi, H., Xia, Y., Cheng, Z., Zhang, X., and Liu, S. (2025). Unleashing the effect of data asset information disclosure on corporate investment efficiency: Fresh evidence from double-debiased machine learning. *International Review of Economics & Finance*, 104, 104698. <https://doi.org/10.1016/j.iref.2025.104698>

Shin, H.-H., and Park, Y. S. (1999). Financing constraints and internal capital markets: Evidence from Korean chaebols. *Journal of Corporate Finance*, 5(2), 169-191. <https://doi.org/10.1016/S0929-1199(99)00002-4>

Soto, P. E. (2021). Breaking the Word Bank: Measurement and effects of bank level uncertainty. *Journal of Financial Services Research*, 59(1-2), 1-45. <https://doi.org/10.1007/s10693-020-00338-5>

Sul, W.-S. (2021). R&D investment and firm value: Focusing on the moderating effect of corporate governance and ownership structure. *Journal of Industrial Convergence*, 19(5), 13-19. <https://doi.org/10.22678/JIC.2021.19.5.013>

Teece, D. J., Pisano, G., and Shuen, A. (1997). Dynamic capabilities and strategic management. *Strategic Management Journal*, 18(7), 509-533.

Thum-Thysen, A., Voigt, P., Bilbao-Osorio, B., Maier, C., and Ognyanova, D. (2019). Investment dynamics in Europe: Distinct drivers and barriers for investing in intangible versus tangible assets? *Structural Change and Economic Dynamics*, 51, 77-88. <https://doi.org/10.1016/j.strueco.2019.06.010>

Titman, S., and Wessels, R. (1988). The determinants of capital structure choice. *Journal of Finance*, 43(1), 1-19.

Tobin, J. (1969). A general equilibrium approach to monetary theory. *Journal of Money, Credit and Banking*, 1(1), 15-29.

Tori, D., and Onaran, Ö. (2018). The effects of financialization on investment: Evidence from firm-level data for the UK. *Cambridge Journal of Economics*, 42(5), 1393-1416. <https://doi.org/10.1093/cje/bex085>

Williamson, O. E. (1988). Corporate finance and corporate governance. *Journal of Finance*, 43(3), 567-591.

Wooldridge, J. M. (2019). Correlated random effects models with unbalanced panels. *Journal of Econometrics*, 211(1), 137-150.

Xing, L., Han, D., and Hui, X. (2023). The impact of carbon policy on corporate risk-taking with a double/debiased machine learning based difference-in-differences approach. *Finance Research Letters*, 58, 104502. <https://doi.org/10.1016/j.frl.2023.104502>
