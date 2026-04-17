# Trade in Minutes! Rationality-Driven Agentic System for Quantitative Financial Trading

- Decision: Accept (Poster)
- Scores: 6, 4, 4

## Abstract
Recent advancements in large language models (LLMs) and agentic systems have shown exceptional decision-making capabilities, revealing significant potential for autonomic finance. Current financial trading agents predominantly simulate anthropomorphic roles that inadvertently introduce emotional biases and rely on peripheral information, while being constrained by the necessity for continuous inference during deployment. In this paper, we pioneer the harmonization of strategic depth in agents with the mechanical rationality essential for quantitative trading. Consequently, we present TiMi (Trade in Minutes), a rationality-driven multi-agent system that architecturally decouples strategy development from minute-level deployment. TiMi leverages specialized LLM capabilities of semantic analysis, code programming, and mathematical reasoning within a comprehensive policy-optimization-deployment chain. Specifically, we propose a two-tier analytical paradigm from macro patterns to micro customization, layered programming design for trading bot implementation, and closed-loop optimization driven by mathematical reflection. Extensive evaluations across 200+ trading pairs in stock and cryptocurrency markets empirically validate the efficacy of TiMi in stable profitability, action efficiency, and risk control under volatile market dynamics.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces TiMi (Trade in Minutes), a multi-agent system designed for quantitative financial trading. This system includes (1) Policy Stage (Offline), a Macro Analysis Agent identifies market patterns from technical data to form "general strategies," which a Strategy Adaptation Agent then customizes for specific trading pairs. (2) Optimization Stage (Offline): a Bot Evolution Agent translates these strategies into programmatic trading bots. These bots are then simulated, and a Feedback Reflection Agent analyzes their performance, using mathematical reasoning to solve for optimal parameters and refine the bots. (3) Deployment Stage (Live): the final, optimized programmatic bots are deployed. These bots are low-latency and execute trades at a minute-level frequency using technical indicators, without requiring any live LLM inference.

### Strengths
The core contribution—decoupling offline, high-cognition strategy development (Policy & Optimization) from online, low-latency execution (Deployment)—is a significant and practical innovation. It directly addresses the primary bottleneck (latency and cost) of applying agentic LLM systems to real-time, high-frequency domains.

The "Optimization Stage" is a major strength. The use of a Feedback Reflection Agent to translate simulation feedback (e.g., high-risk scenarios) into formal mathematical constraints (as shown in Appendix A) is a novel and powerful mechanism for feedback, far superior to simple parameter-tuning.

The authors validate their system comprehensively. The experiments span multiple markets (stocks, crypto), a large number of trading pairs (200+), and include strong baselines from three distinct categories (Quant, ML/RL, and LLM-Agents). The results, particularly the low latency (Fig. 3), stable returns (Fig. 4), and effective ablation study (Fig. 5), compellingly support the architectural design.

### Weaknesses
The paper describes the agents' functions at a high, conceptual level but provides few implementation details. To be convincing, the authors can show their work. For example, what prompts are used? What does the "general strategy" output by Ama actually look like? Is it text, JSON, or code?

### Questions
Is the strategy space limited to grid-trading archetypes, or can your agentic system invent and implement entirely different strategies (e.g., a momentum-crossover strategy or a statistical arbitrage strategy) from scratch?

What is the total offline computational cost to fully train and optimize one bot for a single trading pair, from the Policy stage to the Deployment stage?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes TiMi, a rationality-driven multi-agent trading system that leverages LLM specialization across three dimensions: semantic analysis (ϕ), code programming (ψ), and mathematical reasoning (γ). The core idea is to decouple strategy analysis from deployment, building trading bots offline through multi-agent reasoning (macro → micro → bot evolution) and deploying lightweight, deterministic bots for real-time trading. The architecture includes:

Strategic decoupling of reasoning and execution.

A two-tier analytical paradigm (macro pattern discovery → pair-specific customization).

A layered programming design for bot implementation.

A closed-loop optimization mechanism based on mathematical reflection (Linear Programming-style constraints).

Empirical results across 200+ trading pairs show better profitability and latency compared to rule-based, ML, and prior LLM-agent systems.

### Strengths
1） The paper identifies a central challenge in LLM-based trading: continuous multi-agent reasoning causes latency and instability that limit minute-level decision-making. Framing this as a policy–deployment decoupling problem establishes a clear and theoretically relevant research direction.

2）The four-agent design (macro analysis, strategy adaptation, bot evolution, feedback reflection) aligns with the three capability dimensions of semantic analysis, code generation, and mathematical reasoning, yielding a systematic and interpretable framework rather than an ad hoc workflow. It is a relevantly reasonable design.

3) The experiments encompass both stock and cryptocurrency trading markets, which makes the evaluation relatively more comprehensive.

### Weaknesses
1) Although the paper focuses on higher-frequency trading (minute-level), the experimental time span of only 3–4 months may be too short to adequately evaluate the system’s performance under diverse market conditions. Extending the time horizon to at least half or a full trading year would provide a more robust assessment.

2) More necessary ablation studies can be needed. The experimental section does not isolate the contribution of the main components. In particular, there is no ablation for LLM specialization (ϕ vs ψ vs γ), for the layered programming design, or for the LP-style reflection; the reported ablation is mainly on optimization depth.

### Questions
Same as what I mentioned in the weaknesses.

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
2

### Summary
TiMi is a rationality-driven, multi-agent LLM-based trading system designed for minute-level quantitative trading. TiMi decouples complex strategic analysis (policy and optimization stages) from real-time execution (deployment stage) via a modular framework of specialized agents (semantic analysis, code programming, mathematical reasoning). The system uses macro-to-micro strategy formulation, layered bot programming, and hierarchical optimization.

### Strengths
- Clear decoupling between inference and execution phases enables simultaneous low-latency deployment and high-frequency efficiency.
- The closed-loop optimization mechanism based on mathematical reflection (e.g., parameter solving via linear programming) is innovative and holds significant practical value.
- Supports large amounts of assets (NP=213) while demonstrating stable operational efficiency and capital utilization—achievements beyond the reach of most reinforcement learning and large language model approaches.

### Weaknesses
- Roles of individual agents (e.g., Afr vs. Asa) lack clear ablation to quantify their individual importance to final performance.
- Unlike some LLM-trading papers that provide human-readable rationales, TiMi’s decisions seem more black-boxed in deployment.

### Questions
Can the authors provide ablation studies on agent components (e.g., remove Afr or skip hierarchical optimization) to isolate their impact?

Besides, could TiMi be extended to generate interpretable rationales for its trades?

### Soundness
2

### Presentation
3

### Contribution
2
