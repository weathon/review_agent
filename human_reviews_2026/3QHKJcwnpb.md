# Bradley-Terry and Multi-Objective Reward Modeling Are Complementary

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
Reward models trained on human preference data have demonstrated strong effectiveness in aligning Large Language Models (LLMs) with human intent under the framework of Reinforcement Learning from Human Feedback (RLHF). However, RLHF remains vulnerable to reward hacking, where the policy exploits imperfections in the reward function rather than genuinely learning the intended behavior. 
Although significant efforts have been made to mitigate reward hacking, they predominantly focus on and evaluate in-distribution scenarios, where the training and testing data for the reward model share the same distribution.
In this paper, we empirically show that state-of-the-art methods struggle in more challenging out-of-distribution (OOD) settings. We further demonstrate that incorporating fine-grained multi-attribute scores helps address this challenge. However, the limited availability of high-quality data often leads to weak performance of multi-objective reward functions, which can negatively impact overall performance and become the bottleneck. To address this issue, we propose a unified reward modeling framework that jointly trains Bradley-Terry (BT) single-objective and multi-objective regression-based reward functions using a shared embedding space. We theoretically establish a connection between the BT loss and the regression objective and highlight their complementary benefits. Specifically, the regression task enhances the single-objective reward function’s ability to mitigate reward hacking in challenging OOD settings, while BT-based training improves the scoring capability of the multi-objective reward function, enabling a 7B model to outperform a 70B baseline.
Extensive experimental results demonstrate that our framework significantly improves both the robustness and the scoring performance of reward models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates the problem of reward hacking in reinforcement learning from human feedback, particularly under out-of-distribution (OOD) conditions. The authors propose SMORM, a unified framework that jointly trains a Bradley–Terry (BT) single-objective reward model and a multi-objective regression-based reward model using a shared embedding space and provided theoretical support on the connection between the BT loss and the regression objective and highlight their complementary benefits.

### Strengths
1. The paper is well-structured and clearly presented

2. The idea of co-training BT and multi-objective regression heads under a shared embedding space is new and well-motivated.

3. The proposed methods is supported by empirical evaluations, under both in-distribution (ID) and out-of-distribution (OOD) settings.

### Weaknesses
1. While the joint-training idea is new, the method itself (two linear heads sharing embeddings) is architecturally simple and appears as a simple combination of existing method. Therefore, in my view, the main novelty lies in the theoretical analysis of complementary benefits of BT and multi-objective reward modeling.

2. The theoretical results rely on several strong and unverifiable assumptions, such as positive-definite covariances and global positive correlation between single-objective and multi-objective reward signals. In practice, it is unclear whether these assumptions are satisfied in noisy environments.

3. No code provided currently.

### Questions
1. Under the BT model, rewards are identifiable only up to an additive constant. In Theorem 1, the two sides may implicitly live on different scales. How do you resolve the shift when coupling BT and regression heads?

2. The shared embedding f typically lives in high dimensions; in practice its covariance matrices can be singular. Many proofs assume positive definiteness. If this fails (e.g., rank deficiency due to collinearity), what precisely degrades?

3. With massive over-parameterization, the Fisher Information Matrix  for LLM parameters is typically singular. Page 24’s asymptotic variance claims appear to rely on an invertible FIM is unlikely to hold. Thus, it requires further clarification.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
## Summary
The paper introduces **SMORM**, a joint reward-modeling framework that shares one backbone and trains (i) a Bradley–Terry (BT) single-objective head from pairwise preferences and (ii) a multi-objective regression head from fine-grained attribute scores. The method offers single-pass inference via three variants: **SMORM-F** (use the BT head only), **SMORM-L** (mean of attribute heads), and **SMORM-M** (average of BT and attributes).


Empirically, SMORM improves out-of-distribution (OOD) robustness for PPO and Best-of-N and boosts scoring on RewardBench and RM-Bench. The authors also report that, using the same multi-objective dataset, **a 7B SMORM model surpasses a 70B baseline** on RewardBench.

### Strengths
- **Clear OOD focus with concrete evidence.** The paper explicitly studies PPO/BoN under prompt-distribution shifts and shows baselines (e.g., GRM, ODIN) misspecify or overfit signals that lead to reward hacking, whereas SMORM variants maintain rising gold scores.  
- **Principled link between BT and regression.** Lemma 1 upper-bounds expected BT loss by the regression MSE; Theorem 2 uses Fisher information to argue strictly better asymptotic MSE for both heads under joint training.  
- **Practical single-pass design with strong results.** One shared forward pass and simple inference variants; results include RewardBench/RM-Bench gains and the 7B > 70B headline.

### Weaknesses
- **Assumptions (1–3) may be stringent in practice.** Theorem 1 requires (i) **bounded features**, (ii) **positive-definite** covariance matrices for BT and multi-objective features, and (iii) a **positive-correlation** condition via a coupling vector. In realistic learned embeddings, covariances can be **rank-deficient** (e.g., collinearity, limited per-attribute data, heavy regularization), and verifying the correlation condition empirically is non-trivial. Discussion or diagnostics on how often these hold in real RM pipelines would strengthen the claims.  
- **Limited optimization-stability analysis for the joint objective.** The paper notes that “their joint training is non-trivial due to their fundamentally different forms,” but beyond asymptotic analysis and aggregate curves, there’s little on **training stability/convergence** (e.g., loss weighting sensitivity, gradient interference). An appendix gives some hyperparameter analysis, yet a focused study on training stability (or convergence) is still missing.
- Missing a related work  "On the Robustness of Reward Models for Language Model Alignment", which focuses also on the OOD setting.

### Questions
1. **If Assumption (2) in Theorem 1 (positive-definite covariances) is relaxed to positive-semi-definite, do the results still hold?**  
2. For the **7B > 70B** claim, a consolidated table with **confidence intervals** and multiple base models would better characterize robustness of this result across settings.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces SMORM (Single and Multi-Objective Reward Model), which jointly trains a Bradley–Terry (BT) single-objective head and a multi-objective regression head on a shared embedding space. The authors claim that this co-training improves robustness against reward hacking under out-of-distribution (OOD) conditions and enhances fine-grained multi-attribute reward estimation. The paper provides theoretical analyses connecting BT loss and regression loss through Fisher Information and MSE bounds and reports empirical results showing modest improvements over GRM, ODIN, and baseline classifiers on RewardBench, RM-Bench, and PPO/BoN experiments.

### Strengths
1. The paper offers a theoretically well-grounded contribution. The link between BT loss and regression loss via Fisher Information analysis is original and potentially influential.
2. The joint single/multi-objective framework is conceptually neat and aligns with the broader goal of multi-dimensional reward alignment.
3. The idea of embedding-space complementarity between preference learning and multi-attribute regression is interesting and could inspire follow-up research.
4. Writing and structure are clear, and the motivation is intuitive.

### Weaknesses
1. Lack of statistical rigor.
The paper reports only mean scores without standard deviations or multiple-seed averages. Given the small performance margins (≈1–2 points on RewardBench or RM-Bench), these gains could easily fall within noise. This omission is especially problematic because the theoretical claim centers on variance reduction, yet no variance statistics are provided.

2. Weak OOD validation and limited generalization coverage.
The paper claims robustness “in OOD settings” but evaluates only one prompt–response distribution shift. Unlike Hong et al. (ICML, 2025) (“On the Robustness of Reward Models”), which explicitly distinguishes in-domain (ID), prompt-disjoint, response-disjoint, and mutual-disjoint scenarios, this work tests only a single OOD split. No experiments examine robustness to unseen model families, novel prompt distributions, or response-generation biases, which would be expected to stress the model’s generalization. Consequently, the empirical evidence does not convincingly demonstrate the claimed OOD robustness.

3. Fine-grained performance not well supported.
Although the framework is motivated by multi-attribute supervision (helpfulness, coherence, verbosity, etc.), evaluation results remain at the aggregate level. No attribute-wise performance or correlation analysis is presented to verify the “fine-grained” improvement.

4. Small effect sizes and lack of reproducibility details.
Reported improvements over GRM or ODIN are minor and not statistically supported. Also, RLHF results (PPO/BoN) appear based on single runs with no mention of seeds or error bars.

### Questions
1. Could the authors report standard deviations or confidence intervals across multiple seeds to verify statistical significance?
2. Can the OOD analysis be explicitly decomposed into settings to substantiate the robustness claims (e.g., prompt-disjoint, response-disjoint, and mutual-disjoint)?
3. Could the authors report per attribute ranking accuracy or correlation for helpfulness, correctness, coherence, complexity, and verbosity? It would be better, if authors include ablations that remove the multi objective head to isolate its contribution.
4. Have the authors compared SMORM with simpler stabilization methods such as Batch Sum-to-Zero Regularization (BSR) under identical OOD conditions?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces SMORM, a single-backbone with one Bradley–Terry (BT) reward head and one multi-objective regression head. The authors also provide theory proofs linking the two to explain the robustness gains. The architecture is novel while synthesizing insights from prior work (e.g., Armo, GRM). Empirically, SMORM improves scoring and out-of-distribution robustness and also alleviates reward hacking in RLHF.

### Strengths
* The paper is well written and clearly positioned relative to prior work, while thoughtfully incorporating insights from it.


* The method is simple and well-motivated, and the theoretical analysis makes the core idea more compelling.


* The evaluation is thorough, covering both reward modeling and RLHF.

### Weaknesses
* The design of SMORL-L uses the mean of the multi-objective scores. However, using the mean does not make sense, which means you give all rewards equal weights given all prompts. Actually, this should be utilized in a contextual way and dynamically adapt to the user prompts [1][2].

* The authors claim SMORM is flexible because the two heads are trained on different prompt–response pairs. In practice, this complicates using and coordinating two distinct datasets. I therefore disagree with the flexibility claim and suggest revising it.

* Because the method uses two datasets, add a control that trains a BT reward model on the union of the multi-objective preference data (transferred to BT data using rules) and BT data. This would make the comparison fairer by isolating architectural gains from “more data” effects.

* Based on question 2, I'm interested if not including new data if the conclusion still holds. This can be validated using a multi-objective preference dataset and transfer it to BT data using certain rules. Then you do the same ablations study comparing to baselines (multi-objective only, BT-only, combining them using two models,  and SMORM) trained on the same dataset. 

* Table 3 mixes base models: the paper evaluates with Llama-3.1-8B-Instruct but compares to ArmoRM on Llama-3-8B. Against contemporaneous baselines built on Llama-3.1-8B-Instruct (e.g., QRM-Llama-3.1-8B-v2, Skywork-Reward-Llama-3.1-8B-v0.2), the method shows no clear gains; notably, Skywork-Reward-Llama-3.1-8B-v0.2, trained only on Skywork80K (without multi-objective data), scores 93.1 on RewardBench. This weakens the claim.

* Since the paper leverages multi-objective preference learning, Appendix A should include a more complete related-work section to cover this line of research [1–4].





[1] Improving context-aware preference modeling for language models, 2024.

[2] MiCRo: Mixture Modeling and Context-aware Routing for Personalized Preference Learning, 2025.

[3] Rethinking Diverse Human Preference Learning through Principal Component Analysis, 2025.

[4] Personalizing reinforcement learning from human feedback with variational preference learning.

### Questions
Please refer to the weakness section.

### Soundness
3

### Presentation
3

### Contribution
3
