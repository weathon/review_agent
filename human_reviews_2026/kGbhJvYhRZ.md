# Learning Utility‑Calibrated Routing for Hierarchical Multi-Agents in Portfolio Decision‑Making

- Decision: Reject
- Scores: 6, 0, 4, 4

## Abstract
We study how tool‑using agents can make high‑stakes decisions under uncertainty and costs, with a focus on portfolio allocation. We introduce a hierarchical agent with a learned router that dispatches market contexts to specialized tools (e.g., event extractors, forecasters, options pricers) and an allocator that turns probabilistic predictions into trades under explicit risk and transaction constraints. Our training objective couples proper scoring rules for probabilistic calibration with risk‑sensitive portfolio utility and cost regularization, yielding utility‑calibrated predictions that are natively decision‑aware. To enable reliable offline assessment, we derive a doubly‑robust off‑policy evaluation procedure tailored to backtesting with market frictions, reducing bias and providing uncertainty estimates. Across two challenging settings—options‑only allocation over large‑cap technology names and multi‑asset allocation in the U.S. SP500 sector—our approach delivers consistent gains in expected utility and Sharpe, markedly improved probability calibration, and lower turnover while satisfying risk and exposure constraints. The architecture is modular and data‑agnostic, enabling seamless integration of new tools and experts while preserving end‑to‑end differentiability through the router and allocator. We release code and reproducible benchmarks to support rigorous evaluation of risk‑aware, tool‑using agents for financial decision‑making and beyond.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents a hierarchical multi-agent framework for portfolio decision-making that jointly learns utility-calibrated routing and allocation under realistic financial constraints. A learned router dispatches market contexts to specialized experts, while a differentiable allocator converts probabilistic forecasts into portfolio weights by solving a convex optimization problem respecting exposure, turnover, and transaction cost limits. The system is trained end-to-end using a composite objective that couples proper scoring rules (for calibrated probability forecasts) with risk-sensitive utilities, yielding decisions that are both statistically consistent and utility-optimal.

To evaluate policies offline, the authors develop a doubly-robust off-policy estimator that accounts for frictions and policy mismatch, enabling unbiased backtesting. Experiments on BigTech options and SP500 multi-asset portfolios show improved Sharpe ratios, calibration, and stability over baselines, validating the method’s ability to align probabilistic reasoning with practical portfolio utility.

### Strengths
The work’s key originality lies in coupling probabilistic calibration with downstream, friction-aware portfolio utility. This unifies traditionally separate areas, calibration, optimization, and risk control, under a single differentiable training objective.


The introduction of a learned router that dispatches to non-neural financial tools is conceptually novel compared to standard mixture-of-experts frameworks that focus on predictive accuracy rather than decision utility.


The adaptation of doubly-robust off-policy evaluation to financial markets with transaction costs and constraints is an important methodological contribution for reliable offline evaluation, often overlooked in trading-oriented ML works.

### Weaknesses
The experiments are convincing but confined to two datasets (BigTech and SP500) and relatively short test periods (2024–2025). It would be stronger to include multi-year or multi-market validations to assess regime generalization. The evaluation focuses on Sharpe/CAGR but could incorporate drawdown-adjusted metrics to assess risk asymmetry.


The paper compares mainly against heuristics and standard RL policies. However, methods such as decision-focused learning (Donti et al., 2017) or differentiable convex layers (Agrawal et al., 2019) could serve as more rigorous baselines. Without this, it is difficult to isolate how much gain stems from the hierarchical routing versus the allocator or the calibration coupling.


The OPE formulation assumes positivity and correct specification of either propensities or value models. In financial data, such assumptions are often violated e.g., due to censored liquidity or non-stationarity. The authors could discuss how robust these estimates remain under misspecified propensities.

### Questions
What happens if two experts cover similar market regimes? Does the load-balancing penalty (Eq. 22) ensure redundancy avoidance, or can experts collapse?

The OPE estimator assumes accurate propensities or value models. Have the authors tested robustness when propensities are estimated with noise or model misspecification?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
2

### Summary
The paper introduces a hierarchical multi-agent framework for portfolio decision-making, where a learned router selects among expert tools (e.g., forecasters, option pricers) and an allocator determines trades using a utility-calibrated objective.  
The authors claim contributions in: combining probabilistic calibration with utility-based training,  designing a differentiable allocator that handles portfolio constraints and market frictions, and developing a doubly-robust off-policy evaluation method for financial backtesting. Experiments on BigTech and S&P500 datasets reportedly show gains in Sharpe ratio and calibration metrics compared to baseline methods.

### Strengths
The topic decision-focused learning and utility calibration for portfolio optimization sounds relevant.

### Weaknesses
1. **Clarity and structure:** The exposition is very difficult to follow. Many definitions, assumptions, and equations are presented without sufficient context or explanation. Several symbols and parameters are never defined, making the methodology unclear. For example, what are $L,u_i, \tau_{\max}$ in equation (1)? And $\boldsymbol{\alpha}, \Gamma_t$ in eq (2)? Or $\lambda_{bal},\Omega$ in eq (8)? etc.. The paper reads as a collection of disjointed technical components rather than a coherent framework.  
3. **Conceptual confusion:** It is not clear what the true novelty is. The connections between the sections are not convincingly motivated.  
4. **Experiments:** The empirical section lacks clear baselines and justification. Key comparisons (e.g., CVaR optimization) are missing, and reported gains are not well supported by rigorous statistical testing.  
5. **Writing and logic:** Sentences are grammatically correct but often incoherent in logical flow. Many claims (e.g., “the objective is Fisher-consistent for the target utility under mild conditions”) are asserted without justification or proof.  
6. **Theoretical contribution:** The claimed theoretical insights are vague or standard results restated without derivation.

### Questions
- Several symbols and parameters are never defined e.g. $L,u_i, \tau_{\max}$ in equation (1), $\boldsymbol{\alpha}, \Gamma_t$ in eq (2) and $\lambda_{bal},\Omega$ in eq (8). Can you please explain and properly define all the parameters in equations 1- 16?

- Where are the formal theorems stated?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a three-tier hierarchical system for portfolio allocation that optimizes a risk-sensitive utility function considering market frictions. The system uses a learned “router” to select specialized expert toolchains and a differentiable convex “allocator” to translate expert predictions into optimal portfolios. It’s trained end-to-end with a novel “utility-calibrated” objective that combines a proper scoring rule for accuracy with portfolio utility. The authors validate the approach with a doubly-robust off-policy evaluation (DR-OPE) procedure adapted for financial backtesting with frictions, presenting results on two equity allocation benchmarks.

Soundness
The paper’s core mathematical framework is sound; the formulation of the allocator as a differentiable convex program is valid, and the utility-calibrated objective is a well-motivated idea. However, the central claims lack adequate supporting evidence. The core methodology, the "Hierarchical Multi-Agents" system, is described in metaphorical terms (e.g., "auction/consensus layer") rather than in reproducible technical terms. Additionally, the empirical validation is insufficient to demonstrate robustness, as it relies on a single, short test period (Jan 2024 to Jan 2025) for non-stationary financial data. More rigorous validation such as walk-forward analysis or multiple train-test splits would be required to prove robustness and generalizability.

Presentation
The paper is well-motivated and clearly frames an important research problem. However, the presentation quality is undermined by two main issues. First, as noted in Soundness, Section 4 fails to provide a clear, reproducible description of the central architecture. Second, the presentation of results is potentially misleading; the main results (Table 1) use a weak "Buy & Hold" baseline, while a more critical ablation (Table 3) that reveals a key trade-off is presented later without adequate discussion, obscuring one of the most important empirical findings.

Contribution
The paper addresses an important question and proposes an interesting research direction, as aligning predictive models with downstream, friction-aware utility is a significant problem. However, the contribution as submitted is limited. The primary methodological contribution (the hierarchical routing system) is not clearly explained and, based on the paper's own results, its benefits are not convincingly demonstrated. The differentiable allocator is a more incremental contribution. Thus, the paper identifies a valuable research gap but does not deliver a fully validated or clearly described solution in its current form.

### Strengths
1.	The paper presents a strong problem formulation by motivating the need to move beyond standard predictive metrics, such as NLL, and directly optimize for downstream, risk-sensitive utility under real-world constraints and transaction costs. This is a critical and highly relevant problem. 
2.	The core ideas of using a differentiable convex optimization layer for the allocator and coupling it with a proper scoring rule for end-to-end training are interesting and well-founded.
3.	Focusing on realistic frictions is a valuable practical contribution. Explicitly modelling transaction costs, turnover limits, and other constraints within a differentiable framework makes it possible to create more deployable machine learning systems for finance.

### Weaknesses
1.	Ablation Results Appear to Contradict the Paper's Narrative: The paper's most important result, presented in Table 3, shows that a simpler "Optimizer-only" baseline achieves a significantly higher Sharpe Ratio (2.58 vs. 2.06) and lower Turnover (0.049 vs. 0.081) than the full "Consensus router+Bayes" system. While the full system does reduce Max Drawdown, the paper fails to frame this result as an intentional risk-return trade-off, making the complex routing architecture appear detrimental to the stated goals.
2.	Vague Core Methodology: The paper's central architectural contributions are not described with sufficient clarity for reproduction. Tier 1's transformation into an "enriched state" is presented as a black box without discussion of the feature engineering or analysis of its properties. Tier 2's "cooperative multi-agent layer" is described using metaphorical terms ("auction," "consensus," "vetting") without formal definitions, making it unclear what the "agents" are (e.g., neural networks, heuristics) or how they function.
3.	Insufficient Empirical Validation: The empirical validation is insufficient to demonstrate robustness. Relying on a single, short test period (~13 months) and a weak primary baseline ("Buy & Hold") prevents meaningful conclusions about the method's superiority in a non-stationary domain like finance. The paper also lacks crucial comparisons to other recent multi-agent systems for portfolio management baselines, which are necessary to validate the contribution of the architecture positioned in this paper.

### Questions
1.	Your ablation in Table 3 presents a clear trade-off: the "Optimizer-only" model yields a higher Sharpe Ratio, while the full "Consensus router" model yields a lower Max Drawdown. Could you elaborate on this? Is the primary goal of your system to be defensive and minimize drawdown, even at the cost of Sharpe Ratio? A more detailed characterization of this risk-return profile (e.g., with Sortino or Calmar ratios) would be helpful.
2.	Could you please provide precise technical definitions and detailed descriptions of the components in Tier 1 and Tier 2? Specifically, what constitutes the ‘enriched state’ in Tier 1? How do you ensure that learned states are “enriched”? Do you perform disentanglement? Also, what are the model architectures for the “experts” and the “router” (gating network) in Tier 2? Finally, what is the exact mathematical formulation of the “consensus layer” that aggregates expert predictions?
3.	Given that your "Optimizer-only" model is a very strong baseline, why was it not used for comparison in the main results table (Table 1)? Also, since the paper’s main contribution is presented as a “hierarchical multi-agent system,” could the authors compare their approach with other recent multi-agent systems (existing relevant state-of-the-art approaches) for portfolio management to validate the claimed benefits?
4.	Could you please explain how you can assure the reader of the robustness of your findings given the use of a single ~13-month test period, particularly for the financial data prone to distribution shift?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a novel framework for evolutionary alpha factor discovery using large language models (LLMs) to tackle sparse portfolio optimization under ℓ₀ constraints. Instead of relying on static or manually designed factors, the authors employ an LLM-driven evolutionary loop that continually generates, mutates, and refines interpretable alpha formulas based on back-testing performance.

### Strengths
The paper is clearly written and logically structured. Figures 2–6 effectively convey the pipeline, though a simplified flow chart summarizing the evolutionary cycle would aid readers.

### Weaknesses
Theory: lacks formal analysis of convergence or generalization for the evolutionary loop.

Compute cost: no clear reporting of LLM query volume or wall-time overhead.

Diversity control: mutation and crossover rules could be described more rigorously (e.g., probability distributions).

### Questions
1. The paper claims to introduce the first autonomous LLM + EA framework for continuous alpha generation. I am just wondering whether authors have seen Kirtac and Germono (2024)? Why is a clear alpha evidence not cited in your paper?

2. I am also having a hard time without contextualization. How do LLMs identify factors? What do those factors mean in a traditional financial context?

3. If the paper is contributing as an alpha generator with LLMs? Many papers show alpha with LLMs. If not, what is the main contribution?

### Soundness
3

### Presentation
3

### Contribution
2
