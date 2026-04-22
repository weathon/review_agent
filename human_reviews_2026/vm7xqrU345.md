# FinThink: An LLM-based Multi-agent System for Financial Reasoning

- Avg Score: 4.67
- Decision: Reject
- Scores: 8, 2, 4

## Abstract
Multi-Agent Systems (MASs) in finance are often constrained by static workflows and simplistic learning, limiting their adaptability in real-world markets where signals are frequently conflicting and complex. This market dynamism is well captured by the Adaptive Markets Hypothesis (AMH), which states that the core factors behind profitability shift as markets evolve over time. To bridge the gap between existing agent architectures and this market reality, we introduce FinThink, an AMH-grounded MAS designed for dynamic adaptation. FinThink's novelty lies in three core components: (1) A Context-aware Workflow for Reasoning (CWRM), which enables architectural adaptivity by dynamically adjusting reasoning depth based on signal complexity. (2) A Reasoning-Driven Hierarchical Memory (R-Mem), which facilitates evolutionary adaptivity by teaching the system to navigate signal conflicts in varying market regimes. (3) A Sentiment-To-Logic (STL) Prompt Protocol, which ensures reasoning stability by preventing the multi-agent process from degenerating into simplistic voting. In backtests on five major tech stocks (AAPL, GOOG, MSFT, TSLA, and AMZN), FinThink demonstrates significant improvements over contemporary LLM-based agents, achieving a median advantage of 9.2% higher Sharpe Ratio, 113.1% higher Calmar Ratio, and a 46.3% reduction in Maximum Drawdown.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces FinThink, a LLM-based multi-agent system designed for financial reasoning and trading. The authors posit that existing financial MAS fail because their static workflows are misaligned with the dynamic, evolving nature of real-world markets, a concept described by the Adaptive Markets Hypothesis.
To solve this, FinThink introduces three core innovations:
(1) Context-aware Workflow for Reasoning. An adaptive architecture where an LLM planner proposes reasoning steps and depth based on market signal complexity, which is then validated by a deterministic Finite State Machine (FSM). This allows the system to "think deeper" during periods of uncertainty.
(2) Reasoning-Driven Hierarchical Memory. An "evolutionary" memory system that moves beyond simple PnL triggers. It reflects on entire trade lifecycles to distill "corrective heuristics." Critically, it organizes these memories into "Narrative Driver Clusters" to enable cross-asset learning.
(3) Sentiment-To-Logic Prompt Protocol. A structured prompting strategy that decouples analysis from decision-making. Analytical agents are constrained to produce objective, logical assessments, preventing the system from collapsing into a simplistic "vote" between "BUY" or "SELL".

### Strengths
The paper's foundation in the Adaptive Markets Hypothesis is a significant strength. It provides a clear theoretical justification for why existing static models fail and what a successful system needs. This moves the work beyond simple engineering and into sound scientific reasoning.
The Reasoning-Driven Hierarchical Memory is the standout contribution. The idea of learning from entire trade lifecycles rather than reactive, trigger-based reflections, like a simple PnL drop, is far more robust. The concept of "Narrative Driver Clusters" to facilitate cross-asset generalization is particularly insightful and well-argued.
The ablation study in Table 3 is excellent. It clearly demonstrates the critical importance of both the R-Mem (performance is catastrophic without it) and the STL protocol (performance severely degrades without it). This provides strong quantitative support for the authors' design choices.

### Weaknesses
The backtest period (2022-2023) is well-chosen for its volatility but represents only one type of market (high-inflation, rate-hike-driven). The architecture's performance is not validated in other, equally common regimes, such as a sustained, low-volatility bull market (e.g., 2017) or a sudden crisis-driven crash (e.g., 2008 or Mar 2020). It is possible the system's "adaptive" nature is overfit to the specific type of volatility seen in the test window. Even though for that period the author may not be able to compare with the SOTA, you can compare with the market and simple strategy.

### Questions
On CWRM Hyperparameters: Figure 4 shows the system's reasoning depth, sometimes hitting a "max round hit" (which appears to be 15 rounds from the graph). How was this maximum depth chosen? Is it a fixed hyperparameter, and how sensitive is the system's performance (and, just as importantly, its computational cost/latency) to this value?

How is a "trade lifecycle" practically defined for the R-Mem reflection? Is it simply from the OPEN to CLOSE of a given position? How does the system handle more complex scenarios like partially closing a position or scaling into a position over time?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces FinThink, an LLM-based multi-agent system designed for adaptive financial reasoning grounded in the Adaptive Markets Hypothesis (AMH). The framework integrates three key components: a Context-Aware Workflow for Reasoning (CWRM) that dynamically adjusts reasoning depth based on market uncertainty, a Reasoning-Driven Hierarchical Memory (R-Mem) that captures cross-asset learning through Narrative Driver Clusters, and a Sentiment-to-Logic (STL) prompt protocol that prevents shallow “voting-style” reasoning. Backtests on five major technology stocks are conducted based on key financial metrics. The paper argues that FinThink’s modular adaptivity and reflective memory design enable stable, explainable decision-making under dynamic market conditions.

### Strengths
1) The paper is clearly written and well-structured, making a technically dense topic accessible. Each core component of the proposed framework is accompanied by informative visualizations that effectively illustrate its architecture.

2) The related work (Section 2) is comprehensive and well contextualized, providing succinct but informative introductions to prior studies across both multi-agent systems and financial LLM agents. 

3) A good point is the accurate and thoughtful application of classical financial theories to the system’s design rationale. The authors are able to ground FinThink in the Adaptive Markets Hypothesis (AMH), effectively using it to motivate dynamic adaptivity rather than invoking it superficially.

### Weaknesses
1) While FinThink presents an elaborate architecture combining a Context-aware Workflow (CWRM), a Reasoning-driven Hierarchical Memory (R-Mem), and a Sentiment-to-Logic (STL) prompt protocol, the degree of novelty appears limited compared to existing financial multi-agent framework such as FinCon[1] and HedgeAgents[2]. The conceptual framing around adaptivity and reflective reasoning is largely incremental over these works, which already explored hierarchical memory, risk control, and multi-agent reasoning frameworks for trading decision-making.

[1] Yu, Y., Yao, Z., Li, H., Deng, Z., Jiang, Y., Cao, Y., ... & Xie, Q. (2024). Fincon: A synthesized llm multi-agent system with conceptual verbal reinforcement for enhanced financial decision making. Advances in Neural Information Processing Systems, 37, 137010-137045.

[2] Li, X., Zeng, Y., Xing, X., Xu, J., & Xu, X. (2025, May). Hedgeagents: A balanced-aware multi-agent financial trading system. In Companion Proceedings of the ACM on Web Conference 2025 (pp. 296-305).

2) The experimental outcomes summarized in the Table 1 and 2 reveal that the empirical improvements are not consistently significant across all evaluated assets and metrics. As seen in Tables 1–2 (pp. 9–10), FinThink outperforms baselines in Sharpe and Calmar Ratios on average, but its Total Return (TR) often falls below FinCon for certain stocks (e.g., AAPL, GOOG, MSFT), suggesting that the method’s advantage could lie primarily in conservative risk management rather than genuine predictive superiority. This raises concerns about whether the reported gains reflect robust reasoning enhancement or simply stronger regularization.

3) Although Appendix A.1 includes an ablation study for R-Mem and STL modules, it omits a systematic examination of the modified risk module introduced in the main text (via STL-II integration and cross-asset memory), as the authors emphasize that this is a major systematic change of their framework versus prior works. The paper does not isolate how these design elements contribute to the final performance, leaving their effectiveness short for evidences. A deeper ablation or sensitivity analysis on risk-module parameters, together with statistical significance testing, would be necessary to substantiate the claimed benefits and demonstrate the distinct contribution of each innovation.

### Questions
Refer to what I mentioned in the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes FinThink, a Multi-Agent System based on the AMH theory. While its motivation (addressing static workflows) is reasonable, the proposed methods, CWRM and R-Mem, appear highly heuristic and over-engineered, lacking clear formal definitions and data-driven validation. Furthermore, the paper's primary performance claims rely heavily on risk-adjusted metrics, while the cumulative return plots in the appendix (Fig 8) show FinThink often underperforms multiple baselines in terms of Total Return. This, combined with several clarity issues and inconsistencies in key metric definitions (Sec A.4), makes the paper's contribution difficult to verify and its effectiveness questionable.

### Strengths
1. The paper grounds its framework in the Adaptive Markets Hypothesis (AMH), which is a strong conceptual starting point. This provides a sound justification for why the system needs "architectural adaptivity" and "evolutionary adaptivity".


2. Despite issues with total return, the system is highly successful in achieving its risk management objective.

### Weaknesses
1. The paper claims "significant improvements" and "superior risk-adjusted returns" in the abstract and experiments section. These claims rest entirely on risk-adjusted metrics (SR, CR) and MDD. However, the cumulative return curves in Appendix A.2 reveal a very different story: FinThink is mediocre in terms of total return. FinThink's total return is visibly lower than the B&H benchmark and other models like FinCon.

2. The core of CWRM is the "FSM Controller," which acts as a "rule enforcer" for "logical coherence". Yet, the specific state transition rules of this FSM are never defined in the paper. Algorithm 1 is just a black-box validateTransition call.  The core innovation of R-Mem, the "Narrative Driver Clusters", is admitted to be a "fundamentally a heuristic design".

3. The experiment relies on Gemini-2.0-flash only. It is important to see how FinThink works for different LLMs, at least a stronger one.

4. The experimental setup for the cross-asset memory is questionable. The paper mixes TSLA and AMZN data for training and then evaluates the model on these same assets. This methodology is poorly explained and raises serious concerns about potential data leakage, as it fails to test the model's ability to generalize to truly unseen assets.

5. The memory methodology is vague and insufficiently evaluated. The paper states the memory is trained for only 3 epochs, and the process of how "corrective heuristics" are precisely extracted from trade cycles is not clearly articulated; it reads more like a narrative than a defined algorithm. Furthermore, the memory component is not directly compared to baseline mechanisms (like FinCon's) and lacks dedicated ablation studies or case studies to validate its specific contribution.

### Questions
Please see the weaknesses above.

### Soundness
2

### Presentation
2

### Contribution
1
