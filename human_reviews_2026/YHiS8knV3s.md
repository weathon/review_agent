# Discovery of Diverse and Realistic Financial Tail-Risk Using Generative Flow Networks

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 2, 6

## Abstract
Accurate modeling of rare high-impact financial events is essential for robust risk management, but existing methods face notable challenges. Sampling-based techniques like Sequential Markov Chain Monte Carlo generate diverse scenarios but often lack focus on specific high-risk outcomes, leading to inefficient exploration. In contrast, Deep Reinforcement Learning methods optimize sequential decision-making effectively but tend to converge on a narrow set of high-reward scenarios, limiting diversity. To address these shortcomings, we propose using Generative Flow Networks (GFlowNets), which naturally learn to sample according to a predefined reward distribution. Our approach systematically generates diverse and realistic tail-risk scenarios by explicitly defining rewards that prioritize high-risk, impactful financial scenarios. Experiments on real-world financial data show that our GFlowNet-based method significantly enhances scenario diversity and realism, effectively capturing the complex nonlinear dependencies typical of financial markets. These findings demonstrate the potential of GFlowNets as a robust framework for financial risk modeling, offering actionable insights into rare but critical market events.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces GRID, a method based on Generative Flow Networks (GFlowNets), for discovering diverse and realistic financial tail-risk scenarios. The authors present this as a sequential decision-making problem where a GFlowNet policy is trained to sample trajectories of macroeconomic variables. The sampling process is guided by a reward signal provided by a pre-trained “Oracle” model, which aims to sample trajectories in proportion to their predicted risk, specifically large negative S&P 500 returns. The paper claims that this approach offers a superior trade-off between scenario diversity and risk-severity discovery compared to standard Reinforcement Learning (RL) or MCMC baselines.


Soundness
The central claims of the paper are not adequately supported by the evidence, and the research methodology appears to be unsound. The paper's effectiveness depends substantially on the quality of the Oracle-provided reward signal. However, the paper's own evidence (Table 2) reports negative R² values for the Oracle, indicating its predictions are worse than a trivial mean-baseline. This makes it difficult to validate the value of the reward signal, making subsequent experimental results uninterpretable. Furthermore, using two different Oracles for training and testing is a non-standard approach that complicates the comparison.

Presentation
The paper is generally well-written, but the presentation of the core methodology is confusing and omits important information. The experimental setup involving two different Oracles is not convincingly justified, and the performance metrics for the main (test-time) Oracle are not presented. This makes it difficult for the reader to validate the experiment and assess the reliability of the results.

Contribution
While the idea of applying GFlowNets to risk discovery is novel and well-motivated, the paper's execution does not provide a sound or convincing validation. Due to fundamental issues in the experimental setup, particularly the unvalidated reward signal and the non-standard train/test mismatch, the evidence presented is insufficient to substantiate the claimed contributions. The results are difficult to interpret and as seen in the Q90 metric, do not show a clear or consistent advantage over the baselines.

### Strengths
•	Significant Problem: The paper addresses a highly significant and challenging problem: the discovery of diverse, high-impact tail risks, which is a known limitation of standard financial modeling.
•	Novel Problem Formulation: The conceptual framing of risk discovery as a reward-proportional sampling problem (via GFlowNets) rather than a reward-maximization problem (via standard RL) is a strong theoretical fit for the domain and represents a promising research direction.

### Weaknesses
1.	Problematic Setup: Questionable Reward Signal. The paper's entire methodology depends on the Oracle's quality, as the authors state: "its effectiveness depends on oracle quality. (Line 481)" However, the performance reported for the training Oracle in Table 2 includes R² values of -0.4762 and -0.8638. A negative R² value, by definition means the model's predictions are worse than a trivial baseline like naive mean. This suggests that the reward signal could be uncorrelated with the real-world outcome it claims to represent and potentially training the agents to optimize noise.
2.	Confusing Two-Oracle Methodology. The authors state they use a simple MLP Oracle for training and a different ensemble Oracle for testing (evaluation). This is not standard practice and introduces problems:
o	Train/Test Skew: It is likely to introduce a distributional shift in the objective function itself. The agent is optimized to find scenarios that exploit one reward model (the MLP) and then evaluated on its ability to find scenarios that score well on a different reward model (the ensemble).
o	Uninterpretable Results: This makes the results in Table 3 uninterpretable. We do not know if GRID's performance is due to its superior risk-finding ability or its incidental ability to generalize from the training MLP Oracle to the (unknown) ensemble, better than the RL baselines.
o	Notable Omission: The paper provides no performance metrics (R², RMSE, MAE) for the test-time ensemble Oracle. Without validating the actual reward function used for evaluation, the results are unverifiable.
3.	Empirical Results Do Not Demonstrate Clear Superiority.
o	Even if (charitably) ignoring the confusing Oracle, the data in Table 3 does not show a clear win for GRID. The paper's core claim is to find tail-risk, but on the Q90 (extreme tail) metric, RL baselines outperform GRID in 2 of 3 scenarios (REINFORCE in 2002, SAC in 2008). While GRID does show strong performance on median reward and diversity (in 2008 & 2021), it fails to consistently win on the primary metric of extreme tail-risk (Q90) and this significantly weakens its central claims.
4.	Undersupported Methodological Claims. The paper proposes novel technical components (e.g., the dynamic mixture model) with claims that they "improve trajectory diversity" but lacks ablation studies or empirical evidence to further support these design choices.

### Questions
1.	Could the authors please elaborate on the negative R² values in Table 2? Given that this metric indicates the model is worse than a mean predictor, how can this Oracle be considered a valid source of "realistic" reward signals for training?
2.	Could the authors provide a stronger justification for the train/test Oracle mismatch? Could they also please provide the full performance metrics (R², RMSE, MAE) for the ensemble Oracle that was used for the main evaluation in Tables 3 & 4?
3.	How do the authors justify their narrative of superiority in tail-risk discovery with the empirical results in Table 3, which show that RL baselines (REINFORCE and SAC) found higher-reward tail-risk scenarios (Q90) in two of the three evaluated periods?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes GRID, a generative flow networks (GFlowNets)-based method to discover diverse macro-financial tail-risk scenarios. GRID samples trajectories by predicting specific parametric distributions for each macro-variable. Training utilizes trajectory balance so that scenarios are sampled proportional to an oracle-defined terminal reward. The experiments on three crisis periods report good diversity-quality tradeoffs vs SMCMC and RL baselines.

### Strengths
1. Clear problem framing: Tail-risk discovery with explicit reward shaping toward adverse outcomes, not just one worst case.

2. GFlowNet fit: proportional-to-reward sampling naturally encourages multi-modal coverage.

3. Continuous-state design: Nice engineering to let different variables use specific distributions.

4. Evaluation and analysis: Basic risk management metrics, trajectory and terminalstate distance metrics are reported, and the comparison result achieves a good balance between generation quality and diversity.

### Weaknesses
1. Oracle dependence: Heavy dependence on an oracle model whose out-of-sample 𝑅2 is negative on two of the three crisis windows; this weakens the claim that discovered scenarios are financially risky rather than artefacts of the predictor.

2. Realism checks are light: The paper does impose per-step bound to retain plausibility, but this is still weaker than evaluating or checking stylized facts and cross-variable co-movement that “realistic” in the title would suggest.

3. Reward inconsistency: Table 3 states rewards are clipped at 100, but later figures show > 100. It is unclear at which stage clipping is applied and whether all methods use the same rule. The oracle’s predicted negative return exceeding 100 also raised some concerns.

4. Evaluation metrics are narrow: Since the paper is explicitly positioned for stress/scenario analysis, reporting more tail coverage metric like CVaR and tail frequency of generated paths would strengthen the results and aligned with standard risk practice.

5. Editorial and notation issues/confusions: Some typos remain, e.g. “measure he financial losses” in line 355; The figure 3 in the appendix has a zero score for normalized diversity score for the third subplot.

### Questions
1. With negative 𝑅2 on two crises, why trust the oracle’s sign/magnitude for tail rewards?

2. How to evaluate the “realistic” of the generated path? Can you show that GRID meet the basic variable dynamics (e.g. VIX↑ S&P↓) in generated trajectories?

3. In 4.1, the paper claims the initial state “begins at a specified macroeconomic state”. Please clarify the determination of the specified state.

4. Please clarify Eq. (5). If using a unimodal proxy for mixtures, why this specific variance form?

5. Why rank-select top-K for SMCMC but sample for GRID/RL?

6. How sensitive are results to the per-step change bounds? Can you provide ablations like looser/tighter bounds or bounds derived from crisis windows?

7. Claiming: Given small variable set and monthly granularity, what exactly is meant by “first to apply GFlowNets to large continuous state spaces”.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper describes a method to generate diverse samples for risk management in finance applications. The method is based on well-studied GFlowNet.

### Strengths
1. The paper is written reasonably well, it's easy to follow. 
2. The proposed method may have other potential applications to other broad areas.

### Weaknesses
1. The proposed method is based on a GFlowNet: the level of novelty is limited.
2. Better exposition of the financial market application would benefit broader audience

### Questions
1. Perhaps it's obvious to the authors, but many readers of ICLR may not fully understand the key issues in the financial market applications. A brief description of the variables involved would be a big plus.
2. The authors didn't explain how important the performance of the *oracle* model is.  Based on Tab 2, if the oracle model fails to capture the three extreme events, how would the GRID perform? 
3. Again related to the oracle model, what is the rationale of having MLP and model ensembles in the training and testing phase? Should an ablation study be conducted?
4. Tab 3, it's hard to comprehend GFlowNet does well, when it underperforms SMCMC by 100% for 2021 event! 
5. Same argument can be made for Table 4
6. Figure 2, what is the desirable behavior? 
7. Since the goal is to sample extremely rare events, should $Q99$ be used instead of $Q90$ (line 422)? I would argue that even $Q99$ may not reflect the extremely rare events.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces GRID, a GFlowNet-based method for discovering diverse and realistic financial tail-risk scenarios. Unlike sampling methods (SMCMC) that explore broadly but inefficiently, or deep RL methods (REINFORCE, SAC) that converge to narrow high-reward modes, GFlowNets learn to sample trajectories proportionally to their risk rewards, naturally balancing diversity and impact. The key innovation is successfully applying GFlowNets to large continuous state spaces by using flexible, feature-specific distribution families (Gaussian mixtures, Beta distributions) for modeling macroeconomic variables. Experiments on three major financial crises (2002, 2008, 2021) show GRID generates more diverse high-risk scenarios than baselines, achieving top median rewards in 2/3 cases and superior 90th-percentile performance. The method enables effective stress testing by systematically discovering multiple plausible pathways to market crashes, rather than just finding the single worst-case scenario.

### Strengths
1. The paper redefines financial tail-risk discovery as a proportional sampling problem rather than an optimization problem, which is a conceptual innovation. Technically, it is the first to successfully apply GFlowNets to large-scale continuous state spaces, breaking through the previous limitation to discrete domains only. Particularly clever is the design of using heterogeneous distribution families for different environmental variables (Beta for VIX, Gaussian mixtures for interest rates). This feature-aware modeling is unprecedented in the GFlowNet literature and provides new insights for multi-variable modeling in scientific domains.

2. The experimental design is excellent: testing on three real financial crises with strict temporal splits ensuring no data leakage (Oracle trained only on pre-crisis data). Baseline selection covers classical probabilistic methods (SMCMC) and modern deep RL (REINFORCE, SAC). The dual evaluation framework—reward metrics (median, Q90) and diversity metrics (DTW trajectory distance, terminal-state Euclidean distance)—provides a comprehensive perspective. The normalized diversity score (Equation 9) reveals that SAC's apparent diversity actually stems from instability rather than true exploration. Using an ensemble Oracle (4 models) during testing while training with a single MLP effectively prevents overfitting. Each method undergoes 30 independent runs with 6,000 trajectories total, demonstrating strong statistical robustness.

3. Addresses a critical practical problem: tail-risk events cause enormous losses (2008 GFC exceeded $10 trillion), yet traditional VaR models systematically underestimate their probability. The multimodal distribution shown in Figure 2c perfectly illustrates the value of diversity: multiple distinct pathways can lead to market crashes (liquidity crisis vs. inflation shock), and risk management requires stress testing against all pathways. The method's output of 200 diverse crisis trajectories can be directly used for portfolio stress testing and hedging strategy development.

### Weaknesses
The most serious weakness of this paper is its strong dependence on Oracle model quality, while Oracle itself performs poorly. Table 2 shows alarming Oracle performance across three scenarios: 2002 ($R^2 = -0.4762$, a negative value means worse than predicting the mean), 2008 ($R^2 = 0.0942$, almost no predictive power), and 2021 ($R^2 = -0.8638$, severe failure). This creates a fatal circular reasoning problem: if the Oracle cannot accurately predict market crashes, how can a GFlowNet trained on Oracle rewards possibly discover realistic tail-risk scenarios? The reward function $R(s_T) = \max(0, -\mathcal{O}(s_T))$ becomes meaningless when $\mathcal{O}(s_T)$ is unreliable, and the entire sampling distribution $P_F(x) \propto R(x)$ is essentially learning from incorrect targets. The paper claims to generate "realistic" scenarios but provides no validation against actual historical crisis trajectories or ground truth. The high-reward distributions may simply be overfitting to the Oracle's erroneous predictions rather than capturing genuine financial risk patterns.

### Questions
This is very good work. I believe your experimental results are credible, and I think this paper is recommended to be accepted. However, if I still cannot obtain the experimental code to reproduce the results by myself during the rebuttal phase, I may have to consider lowering the score.

### Soundness
4

### Presentation
3

### Contribution
3
