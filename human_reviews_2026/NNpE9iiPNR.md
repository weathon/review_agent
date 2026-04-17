# MASS:Multi-Agent Simulation Scaling  for Portfolio Construction

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
The application of LLM-based agents in financial investment has shown significant promise, yet existing approaches often require intermediate steps like predicting individual stock movements or rely on predefined, static workflows. These limitations restrict their adaptability and effectiveness in constructing optimal portfolios. In this paper, we introduce the Multi-Agent Scaling Simulation (MASS), a novel framework that leverages multi-agent simulation for direct, end-to-end portfolio construction. At its core, MASS employs a backward optimization process to dynamically learn the optimal distribution of heterogeneous agents, enabling the system to adapt to evolving market regimes. A key finding enabled by our framework is the exploration of the scaling effect for portfolio construction: we demonstrate that as the number of agents increases exponentially (up to 512), the aggregated decisions yield progressively higher excess returns. Extensive experiments are conducted on a challenging, proprietary cross-market dataset from 2023, showing that MASS gains enhanced performance over nine state-of-the-art baselines. Further backtesting, stability analyses, the experiment on data leakage concerns, and case studies validate its enhanced profitability and robustness. We have open-sourced our code, dataset, and training snapshots at https://anonymous.4open.science/r/MASS-AC96 to foster further research.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The author introduces a framework, MASS, to construct optimal stock trading strategies using LLM based models in a multi-agent manner. A voting mechanism is leveraged to aggregate the strategies from multiple trading agents. Extensive evaluation is conducted with comprehensive selection of baselines. The empirical results, with varied performance metrics, shows that the proposed MASS framework consistently outperformed all the solid baselines.

### Strengths
- It is the first work that explores the influence of scales when leveraging multi-agent settings for stock trading. 
- The evaluation is solid, with varied SoTA baseline methods, and all the claims are justified through solid testing results. 
- Ablation study is conducted to sufficiently justify the choice of each component of the pipeline. 
- The design of the entire framework is vivid, where varied agents are designed to represent individuals with different trading styles.

### Weaknesses
- The testing performance of the baseline methods at the beginning of 2025 is missing. Having such a comparison is important to justify the consistency of leading performance of MASS under the data-leakage-concern experiment. 
- Though the overall experimental design is soundable and vivid, the contribution on the methodology aspect is slightly lacking, where the entire framework is heavily dominated by prompt engineering, and the proposed aggregation module is essentially a voting mechanism where the voting weight is adjusted based on the past performance. 
- The complexity of the entire system is a concern for the feasibility of real-world deployment.

### Questions
- 1) Since the presentation is mostly emphasized on the performance, I'm wondering if there are any analysis or observation from the trading strategy perspective. For example, if running the multi-agents inference with multiple trails, will the similar strategy always retained (both for the aggregated strategy or each agent with varied trading style individually)? 
- 2) Continued from the 1st question, if, hypothetically, the strategy will varied due to stochastic process in the LLM generation, to what extent it might influence the pattern of the optimal agent distribution identified by the backward optimization? 
- 3) A final thought: since in the real market, if a strategy is known by more people, the expected return of such a strategy will reduce. Though this is slightly paradox with the disagreement hypothesis, is there any potential observation in the multi-agent environment, where there might be more diversity in the trading strategy when the overall return is relatively higher? Having these analysis will provide more insights on the behavior of the proposed framework.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes MASS, a multi-agent simulation framework that directly constructs investment portfolios without predicting stock prices or relying on static decision workflows. It adaptively adjusts the distribution of LLM-based agents via backward optimization, enabling responsiveness to changing market regimes. Experiments on the 2023 China A-share market show that MASS scales effectively with more agents and outperforms multiple SOTA baselines in profitability and robustness.

### Strengths
1. The idea of modeling investors as heterogeneous LLM-based agents with diverse preferences and strategies is conceptually appealing.
2. The release of code and dataset enhances openness and encourages further research in this emerging area.

### Weaknesses
1. The use of simulated annealing as an optimizer is conceptually inconsistent with a dynamic market learning framework. SA requires hundreds of perturbations for convergence, whereas financial markets demand fast, step-wise adaptation. This slow, global optimization process fundamentally conflicts with the model’s claim of online learning and timely market responsiveness. Moreover, in MASS’s experiments, the dataset only covers stock market daily data from 2023, providing merely a few hundred data steps. Given such a limited time horizon, can SA truly converge in practice?
2. Market disagreement hypothesis plays a critical underlying role in the MASS portfolio management method. However, the definition of the signal in Eq4 implies that higher disagreement reduces the resulting signal, leading to a negative portfolio allocation for the stock when applying the top-K strategy. This is somehow inconsistent with the market disagreement hypothesis, which states that greater disagreement may actually lead to higher prices due to optimistic investors dominating valuation under short-selling constraints.

3. MASS evaluates performance using IC, RankIC, and related correlation metrics to measure alignment between its signals and realized returns. This experimental design is essentially identical to that used in classic factor-based models [1, 2], where factors are constructed to predict future returns and assessed through the same correlation metrics.

However, the paper explicitly states that the framework is not intended for price prediction (lines 90–92). Given this positioning, it would be beneficial to clarify the relationship between MASS and traditional factor-mining approaches. In particular, including representative factor model methods [1, 2] as baselines could help demonstrate whether the proposed multi-agent simulation captures information beyond conventional return-predictive factors. Otherwise, it remains somewhat unclear how much of the reported performance improvement should be attributed to the agent-based design.

​	
[1] [22'AAAI]FactorVAE: A Probabilistic Dynamic Factor Model Based on Variational Autoencoder for Predicting Cross-Sectional Stock Returns
[2] [23'IJCAI]HireVAE: An Online and Adaptive Factor Model Based on Hierarchical and Regime-Switch VAE
some concerns addressed in weakness part

### Questions
Some concerns addressed in weakness part

### Soundness
2

### Presentation
2

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
The paper introduces the Multi-Agent Scaling Simulation , a framework that employs multi-agent simulations for direct, end-to-end portfolio construction. Unlike traditional approaches that rely on intermediate steps like stock movement predictions or predefined workflows, MASS optimizes the distribution of heterogeneous agents in a backward optimization process, enabling adaptation to changing market conditions.

### Strengths
Originality: The introduction of a multi-agent simulation framework for direct portfolio construction is highly original. It moves beyond the typical stock prediction models by creating a dynamic system that adapts to market changes, offering a new approach to portfolio optimization.

### Weaknesses
Lack of Deep Contextualization: The paper does not fully contextualize how MASS compares to other multi-agent simulation approaches in finance, nor does it explore the potential limitations of the approach in different financial contexts (e.g., different asset classes or regimes).

Complexity of Agent Interactions: While the paper introduces a sophisticated framework, the interaction between agents and their impact on portfolio performance could be explained more clearly.

### Questions
Are there any limitations to the scaling effect, such as diminishing returns at very high numbers of agents? A more detailed analysis of potential trade-offs would be valuable.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces MASS (Multi-Agent Simulation Scaling), an LLM-based framework for direct, end-to-end portfolio construction. Unlike previous models that rely on individual stock prediction or fixed workflows, MASS simulates a heterogeneous population of investor agents whose decisions are optimized through a novel backward optimization process. The system aggregates agent opinions based on the market disagreement hypothesis, balancing consensus and diversity, and demonstrates that increasing the number of agents (up to 512) leads to a scaling effect—higher returns and improved stability. Experiments on a self-collected 2023 Chinese A-share dataset show MASS outperforming seven state-of-the-art baselines across indices (SSE 50, CSI 300, ChiNext 100) and remaining robust under unseen 2025 data, suggesting strong adaptability and resistance to data leakage.

### Strengths
The paper makes a strong conceptual and empirical contribution to multi-agent financial modeling by reframing portfolio construction as a dynamic learning process rather than a prediction task. The backward optimization mechanism is novel and elegantly designed to simulate market adaptation, with clear empirical validation. The experiments are extensive—covering multiple indices, backtesting, ablation, and sensitivity studies. The scaling effect analysis is particularly impressive, revealing a near-linear improvement as agent count increases. The work’s open-sourced dataset and transparency on cost/time complexity further enhance its reproducibility and practical relevance.

### Weaknesses
Despite its innovation, the paper’s realism and generalizability could be questioned. The experiments rely exclusively on Chinese A-share data from 2023–2025, which may limit transferability to other markets. The backward optimization’s reliance on simulated annealing and hand-tuned hyperparameters introduces concerns about scalability and computational feasibility in live systems. Moreover, the LLM-driven agent logic depends heavily on textual prompts and lacks rigorous analysis of interpretability or reasoning validity. While the results are strong, there’s minimal theoretical justification for why the scaling effect should remain linear beyond the tested range, or how overfitting to historical macro features is prevented.

### Questions
How does MASS perform under longer backtesting horizons (e.g., multi-year or cross-market datasets)?

Can the backward optimization process be extended to reinforcement learning or differentiable optimization for efficiency?

How sensitive is MASS to the choice of LLM backbone (e.g., Qwen vs. GPT) given potential domain pretraining differences?

Is there a risk that the observed scaling benefit plateaus or reverses beyond 512 agents due to noise amplification?

### Soundness
3

### Presentation
3

### Contribution
2
