# KPF: DOMINATING MULTI-AGENT ADVERSARIAL COMPETITION VIA KALMAN-INSPIRED POLICY FU- SION MECHANISM

- Decision: Reject
- Scores: 4, 2, 2, 6

## Abstract
Despite rapid advancements in Multi-Agent Reinforcement Learning (MARL), its application to complex, highly stochastic, and dynamic environments has been hindered by limitations in generalization capabilities and robustness, often resulting in degraded performance. To address this challenge, this paper proposes Kalman Policy Fusion (KPF), a novel decision fusion mechanism inspired by the Kalman filter. The core of the KPF mechanism lies in the learning-oriented adaptive weighting and iterative optimization of policy distributions during testing. It dynamically fuses multi-agent policies while minimizing their differences, thus effectively improving generalization and robustness in highly stochastic environments and achieving exceptional performance in dynamic adversarial tasks. Furthermore, this work is the first to empirically and systematically demonstrate that the efficacy of the base models, their distributional characteristics, and their mutual complementarity are the key prerequisites that determine the upper bound of fusion performance. Comprehensive evaluations demonstrate that our mechanism establishes new SOTA benchmarks across four diverse environments, including the StarCraft Multi-Agent Challenge (SMAC) and its successor SMACv2, Google Research Football (GRF), and the Multi-Agent Particle Environment (MPE). Notably, within the complex domain of StarCraft II, KPF achieves a perfect 100% win rate on numerous challenging maps. By optimizing policy weights to approximate an unknown optimal policy, our results validate the efficacy of the Kalman-based approach in MARL decision optimization, offering valuable insights for building more robust and efficient multi-agent systems.Our code will be released on GitHub upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper propose to enhance the generalization capability of MARL in adversarial competition via Kalman policy fusion. Technically, the policy is fused by adaptive weighting and iterative optimization of policy distributions. Experiments achieves SOTA performance on SMAC, SMAC v2, GRF and MPE.

### Strengths
1. Generalization is an important topic in MARL. The paper propose to fuse multiple policies by dynamic reweighting, which is a promising  direction to explore.

2. The empirical performance is strong over multiple benchmark, against multiple strong baselines.

### Weaknesses
I hold a mixed perspective on this work. On one hand, the problem is important and the result is promising. On the other hand, the motivation is not immediately clear to me and the comoparison with related baselines are required.

1. My biggest concern is why Kalman filtering is proposed as a particularly important way of policy fusion. While Bayesian estimation provides a nice framework of making decisions under uncertainties, I wonder why Kalman filtering is selected over learning-based methods, and since in Eqn. 2 the method is using a heuristic metric unrelated to reward to measure the policy weighting, I wonder what is the underlying reason of it to work empirically.

2. This is a theoretical discussion. Technically different policies master different type of tasks. But why a re-weighting of different policies can guarantee performance improvement? For example, in SMAC, if policy A tends to attack upper enemies,  while policy B tends to attack bottom  enemies, then mixing them harms focus fire and is no longer optimal.

3. Since the method highlight generalization performance, the method should also discuss and compare with generalization/transferability in MARL, such as [1]  and [2]. Also, existing literature includes methods like task/role decomposition, which is not well discussed and compared in the paper. Finally, since authors highlight the generalization capability, tasks apart from standard evaluation schemes should be designed to verify the generalization capability.

[1] Cooperative Multi-Agent Transfer Learning with Level-Adaptive Credit Assignment

[2] Decompose a Task into Generalizable Subtasks in Multi-Agent Reinforcement Learning

4. Problem formulation is missing which harms completeness. Additional theoretical analysis is appreciated.

### Questions
See weakness.

### Soundness
2

### Presentation
4

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
Inspired by the Kalman filter, this work proposes a Kalman-based policy fusion method that enhances the generalization and robustness of MARL by dynamically fusing multiple agent policies while regularizing their divergence. The approach is evaluated on SMAC, SMACv2, MPE, and GRF, demonstrating strong performance on challenging tasks.

### Strengths
1. The introduction of Kalman filter into MARL is interesting.

### Weaknesses
1. Methodological clarity and completeness: The description of the proposed Kalman Policy Fusion (KPF) method omits critical details.

1）Rationale for policy pairs: Why are only two policy distributions, P1 and P2, considered? Please justify this design choice and discuss its generality (e.g., extension to more than two policies).

2）Action space assumptions: The formulation in Eq. (2) appears limited to discrete action spaces. How does KPF handle continuous actions? Provide a principled extension or clarify the scope.

3）Potential typo in Eq. (6): The two terms inside the brackets seem identical. If not a typo, explain the difference; if so, correct it.

4）Implementation: The algorithmic details are essential and should be moved from the appendix to the main text. Omitting them makes it difficult for readers to understand how KPF is executed in practice.

2. Writing and presentation: The manuscript would benefit from substantial improvements in exposition.

1） Clarify contributions: As written, the contributions read primarily as empirical findings. The authors need to briefly articulate the core technical challenges you address and the key methodological innovations that resolve them when introducing the contributions.

2） Add a methodology overview: Include a high-level pipeline or schematic early in the Introduction to convey the operational logic of KPF.

3） Justify the use of Kalman filtering: Explain why the Kalman filter is suitable for policy fusion, given its traditional role in state estimation.

4） Terminology and consistency: Avoid repeatedly redefining abbreviations (e.g., KPF, MARL) and ensure consistent use of full names and acronyms. A thorough proofreading pass is recommended prior to submission.

3. Empirical validation of claims: Since the paper claims improved generalization and robustness for MARL, the experiments should explicitly substantiate these claims.

### Questions
Please refer to the Weaknesses section.

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Kalman Policy Fusion (KPF), a dynamic decision-fusion mechanism inspired by the Kalman filter, intended to improve robustness and generalization in multi-agent reinforcement learning (MARL). The authors argue that existing static fusion methods (e.g., naive averaging, voting) fail in dynamic, stochastic environments due to fixed weighting schemes. KPF instead introduces an uncertainty-aware, adaptive, covariance-based fusion process that iteratively updates action distributions using Kalman gain dynamics. Empirical evaluations across multiple benchmarks, including SMAC, SMACv2, Google Research Football (GRF), and MPE, show KPF outperforming strong baselines such as MAPPO, HAPPO, HATRPO, and QMIX. The paper further presents ablation studies on key hyperparameters and examines the effects of policy complementarity and distribution smoothness on fusion success.

### Strengths
**Originality**: The paper offers an interesting idea of transposing Kalman filtering techniques—traditionally used in state estimation—into the policy fusion domain. The proposal of dynamically adjusting policy weights online using covariance information provides a novel angle on MARL.

**Quality & Significance**: Experiments are extensive, covering four major MARL environments (SMAC, SMACv2, GRF, MPE) and including reasonable baselines and ablation studies.The reported results are impressive, particularly the 100% win rates on hard SMAC maps and the significant gains in the non-stationary SMACv2 environment.

### Weaknesses
1. While the Kalman analogy is creative, the core of KPF essentially performs a weighted averaging of policy distributions based on covariance heuristics. The “Kalman” aspect functions largely as a metaphorical framing rather than a mathematically rigorous adaptation.
2. Although the paper claims KPF is optimal in MMSE sense and ensures convergence, this is not formally proven. There are no theorems or convergence proofs beyond inheriting properties from the classical Kalman filter (which assumes linear Gaussian systems — not the case here). Without bounding analyses or stability proofs under nonlinear policy distributions, the theoretical assurances remain speculative.
3. The analysis focuses on a specific pair of fused models (HAPPO + HATRPO). It remains unclear whether KPF generalizes across heterogeneous architecture types (e.g., combining value-based and policy-based MARL). 
4. The notion of “policy complementarity” is intuitive but not quantitatively defined, e.g., no clear metric is provided to measure the complementarity before or after fusion.
5. This paper does not analyze the computational cost of dynamic fusion vs. training a single stronger baseline. For real-time MARL, the additional covariance computation can be substantial.

### Questions
1. Can the authors provide a formal convergence guarantee or an analysis showing that the policy-fusion process minimizes an explicit upper bound on expected regret or KL divergence from the optimal policy?
2. How does KPF differ mathematically from Bayesian Policy Merging or ensemble RL methods using uncertainty weighting (e.g., Bootstrapped DQN, Bayesian Actor-Critic)? 
3. You argue that “policy complementarity” is crucial, yet it remains qualitative. Could you propose or evaluate a metric (e.g., average pairwise KL divergence weighted by performance similarity) to quantify it?
4. What is the practical impact of KPF on inference speed? Please provide a quantitative analysis (e.g., average time per step or episode for KPF vs. a single policy baseline) on a representative benchmark like some complex SMAC maps. For a system with $n$ agents and action space $|A|$, what is the per-step computational complexity of KPF?
5. Can KPF tolerate significantly weaker auxiliary models, or does their noise dominate the fusion? The paper currently assumes both base models are “high-quality.”
6. Would KPF work on continuous-action MARL tasks (e.g., MuJoCo multi-agent extensions)? The current formulation assumes discrete probability vectors.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes **Kalman Policy Fusion (KPF)**: at test time, two (or more) trained policies’ **action distributions** are treated as “prior/observation” and are **dynamically fused step-by-step** using a Kalman-style gain while maintaining a covariance update, yielding robust decision fusion in multi-agent adversarial tasks. The authors provide a flow diagram and pseudo-code, and claim convergence and (MMSE-style) minimum mean-squared error properties. Experiments span **GRF, MPE, SMAC, and SMACv2**. The method reportedly achieves **100% win rate** on several hard SMAC maps and strong gains on SMACv2 against dynamic opponents built from strong experts (HAPPO/HATRPO). The paper further observes that when base policies in MPE are overly “peaked,” KPF’s gains are modest, motivating the thesis that **policy complementarity is a prerequisite** for successful fusion.

### Strengths
* **Novel methodological angle**: Transplants “covariance-driven online updating” from state estimation to **policy-level decision fusion**, distinct from static averaging/voting and simple ensembling.
* **Broad and strong empirical coverage**: Four benchmark families including non-stationary/adversarial settings; the 100% win rates on hard SMAC maps and clear SMACv2 improvements are compelling.
* **Reusability and interpretability**: Clear procedure and covariance update make the mechanism “plug-and-play” across expert policies; the (k)-step update analysis clarifies the adapt-vs-converge trade-off.
* **Practical insight**: Systematic emphasis that **complementarity** (performance gap + distributional diversity) governs the attainable fusion gains, giving actionable guidance for choosing experts to fuse.

### Weaknesses
* **Theoretical rigor needs bolstering**: Action distributions live on the probability simplex. The paper applies clipping/renormalization plus covariance updates and then invokes Kalman convergence/MMSE optimality. Formal conditions on the simplex geometry (or a suitable reparameterization) are missing, making the “Kalman” claim look heuristic rather than guaranteed.
* **Insufficient treatment of cross-agent correlations**: The implementation appears close to **per-agent independent fusion**. There is no explicit modeling of **cross-agent covariance blocks/low-rank structure**, limiting both theory and performance explanations in strongly coupled tasks.
* **Baselines could be stronger**: Absent comparisons to **dynamic gating/consistency-regularized ensembles** (e.g., temperature-weighted logit fusion, confidence gating, MoE-style gating, KL-aligned consistency, uncertainty-aware Q/Advantage ensembles). Without these, KPF’s relative ceiling is unclear.

### Questions
1. **KPF on the probability simplex**: After clipping and renormalization, under what **sufficient conditions** do your convergence and MMSE-optimality claims hold? Can you provide an equivalent derivation or consistency result in an appropriate parameter space (e.g., Dirichlet/natural parameters or logit space)?
2. **Cross-agent correlation**: Does the implementation assume independent per-agent updates? If so, can you evaluate **block-diagonal/low-rank** cross-agent covariance approximations and discuss accuracy–cost trade-offs?
3. **Stronger dynamic-fusion baselines**: Please add comparisons to temperature-weighted logit fusion, confidence gating, MoE gating, and KL-consistency regularization, with significance testing on SMACv2/GRF.

### Soundness
3

### Presentation
3

### Contribution
3
