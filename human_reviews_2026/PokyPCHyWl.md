# KL-Regularization Is Sufficient in Contextual Bandits and RLHF

- Avg Score: 6.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 8, 4, 6, 6

## Abstract
Recently, reinforcement learning from human feedback (RLHF) has demonstrated remarkable efficiency in fine-tuning large language models (LLMs), fueling a surge of interest in KL regularization. Yet, the theoretical foundations of KL regularization remain underexplored. Many prior works employ either explicit online exploration strategies—such as UCB, Thompson sampling, and forced sampling—or optimism-embedded optimization techniques (e.g.,  Xie et al. 2024) *in addition to KL regularization* to achieve sublinear regret in online RLHF. In this paper, we show, for the first time to our best knowledge, that such additional exploration strategies are unnecessary if KL regularization is already included. That is, KL regularization alone suffices to guarantee sublinear regret. To handle general function classes, we assume access to an online regression oracle and propose **KL-EXP** (and its RLHF variant, **OEPO**), which achieves logarithmic KL-regularized regret—the standard objective in KL-regularized contextual bandits and RLHF—while also attaining an *unregularized* regret of  $\tilde{\mathcal{O}}(\sqrt{\smash[b]{\log N  \cdot T \text{Reg}\_{\text{Sq}}(T) } })$,  where $N$ is the number of actions, $T$ is the total number of rounds, and $\text{Reg}\_{\text{Sq}}(T)$ is the online regression oracle bound. To the best of our knowledge, this is the first result to achieve regret with only logarithmic dependence on $N$ in oracle-based contextual bandits.  As a special case, in linear contextual bandits, we establish a $\tilde{\mathcal{O}}(\sqrt{dT \log N})$ bound on the unregularized regret, where $d$ is the feature dimension. To our best knowledge, this is the first $\tilde{\mathcal{O}}(\sqrt{dT \log N})$-type regret bound achieved without resorting to supLin-type algorithms, making it substantially more practical.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper considers the problem of contextual bandits and RLHF, in which they consider the class of algorithms that choose an action based on a policy that maximizes KL-regularized cumulative reward. This has become highly popular in recent times particuarly for RLHF. While there are several such algorithms proposed in literature that fall under this class, all of them use some sort of forced exploration to address the exploration-exploitation trade-off. In this work, the authors state and prove that no additional exploration term is required and the policies that are obtained by maximizing KL-regularized cumulative reward achieve optimal regret. The authors prove their claims for both contextual bandits and RLHF. They also corroborate their theoretical findings via extensive empirical studies on both synthetic and real-world tasks.

### Strengths
I think the paper is solid and delivers a simple yet neat result, that can be useful for several applications. The key result in this work is where the authors show that the instantaneous (KL-regularized) regret is is bounded by the squared error on the reward estimation process. This provides a nice bridge between regret minimization via KL-regularized policies and reward estimation, which is typical in online learning.

### Weaknesses
I think the reader would strongly benefit from a discussion about **why** this approach works? There is obvious a mathematical description as to why it works, but an intuitive description would definitely help improve the paper and make it more accessible to readers who do not want to go into the technical details.

### Questions
In addition to the above question regarding an intuitive explanation, I have some follow-up questions:

- In the "exact second-order" Taylor expansion, how and why does the factor $(1 - \alpha)$ appear in the integral in Eqn. 5? It might just be a different way of writing things but I do not recall seeing such an expression.

- What happens when the context and actions spaces are infinite? While they are indeed finite for RLHF, it may not be so for other contextual bandit problems. How does your approach scale in such situations?

- Do you think this methodology can be combined with lazy update schemes to reduce computational costs?

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
3

### Summary
The paper studies KL-regularized contextual bandits and an RLHF variant. The authors propose an algorithm that plays policy via a single step of exponential update from the reference policy $\pi_t(\cdot | x) \propto \pi_{\mathrm{ref}}(\cdot | x) \exp(\eta \hat R_t(x, \cdot))$, where $\hat R_t(x, \cdot)$ is computed from an online regression oracle. The paper claims this is the first algorithm that removes additional exploration in this setting. They show that the algorithm achieves regret of $O(\eta Reg_{Sq}(T))$ against the KL-regularized policy, and an unregularized regret $O(\eta Reg_{Sq}(T) + DT / \eta)$, where $D = (1/T) \sum_t \mathrm{KL}(\pi^* \| \pi_{\mathrm{ref}})$. The key lemma uses a second-order expansion around the ground-truth $R^*$ to avoid UCB/optimism. Finally, experimental results are presented for both linear and neural bandits, as well as an online DPO pipeline.

### Strengths
The paper studies KL-regularized bandits, which is an important problem in reinforcement learning from human feedback. The algorithm is the first that removes the pure exploration phase in this setting, matching the desiderata that many researchers hope for in RLHF. Experiments are provided to demonstrate the practical effectiveness of the algorithm.

### Weaknesses
- The final bound of the algorithm is exactly a two-term trade-off between the stability $Reg_{Sq}(T)$ and the bias $\sum_t \mathrm{KL}(\pi^* \| \pi_{\mathrm{ref}})$, which mirrors the standard mirror-descent/FTRL analyses for entropic regularization. The central "no optimism" claim hinges on the second-order expansion, but the resulting decomposition and the KL-to-unregularized conversion are conceptually standard once written. From my perspective, the regularized step is itself another standard way of introducing exploration. As a result, I do not see enough conceptual distance from known adversarial-bandit templates to justify the novelty of the "KL is sufficient" claim.

- Even though the paper claims that the algorithm does not require prior knowledge of the eluder dimension as in previous works, it still requires prior knowledge of the oracle’s regret bound $Reg_{Sq}(T)$, which is in fact a similar complexity quantity that measures the hardness of online decision making. There is no parameter-free schedule to handle this issue in the theory.

### Questions
- Do you think it is possible to provide a parameter-free $\eta$ schedule that does not require any prior knowledge about the online regression oracle?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This is a clever and elegant idea: adding a KL-to-reference term with finite inverse temperature (η>0) yields a simple Gibbs policy and an exact second‑order expansion that shows KL itself induces exploration, turning regularized regret into a squared estimation‑error term without UCB/TS. However, weaknesses remain: the theory mainly covers single‑step bandits (not general MDPs), it relies on realizability and a nonzero temperature during training, and the experiments look underpowered—baseline methods like LinUCB/LinTS are sensitive to exploration parameters that should be thoroughly swept, η sensitivity/annealing should be reported, and the average accuracy gaps are small, warranting multi‑seed runs and confidence intervals.

### Strengths
Strong, elegant theory for single-step settings: The “exact second-order expansion” shows that KL-to-reference (finite inverse temperature η) alone yields sufficient exploration and tight regularized-regret bounds, without UCB/TS. It works with general function approximation via regression oracles and gives a closed-form Gibbs update.

Practical simplicity and scalability: Only one main knob (η, the inverse temperature); no confidence sets or posterior sampling machinery. The Gibbs update is cheap and plugs into diverse regressors, which makes the method easy to scale (important for RLHF/LLMs).

Stability via anchoring to a reference policy: KL(π || π_ref) both induces soft exploration and constrains distribution shift, a desirable property in RLHF (style/safety preservation). The role of temperature is conceptually clear: finite η>0 is needed; η→∞ (greedy) breaks exploration.

### Weaknesses
Experimental evaluation is underpowered and lacks key sweeps: Baselines likely under-tuned: regret of LinUCB/LinTS is sensitive to the exploration parameter (e.g., α in LinUCB, prior/variance in LinTS). These should be systematically swept (and reported) because regret curves can change materially with α. Small differences in average accuracy on RLHF benchmarks: Given the narrow gaps, results should include confidence intervals, multiple seeds, and statistical tests. Sensitivity to decoding/training randomness should be reported. η sensitivity/ablations are insufficient: Accuracy and regret typically depend on η (and any annealing schedule). It’s important to plot performance vs η and vs target KL levels to π_ref. I cannot confirm from the current context whether the authors ran a comprehensive η sweep; if not, this is a clear gap to address.

Limited scope beyond bandits (no MDP guarantees): The results hinge on single-step structure. Extending the second-order technique to MDPs faces fixed-point coupling, occupancy/concentrability, and error propagation issues; without strong coverage/mixing assumptions, KL alone may not ensure state-space exploration.

Dependence on assumptions and constants: Requires realizability and strong regression oracles; bounds include constants that can be large (e.g., κ from preference curvature, D = E[KL(π* || π_ref)]). If π_ref is far from optimal, guarantees degrade. Necessitates nonzero temperature during training (finite η). Deterministic training (η→∞) can yield poor exploration and invalidate the proof technique.

### Questions
Please refer to the weakness part.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper's central argument is that KL regularization alone is sufficient to guarantee efficient exploration in contextual bandits and Reinforcement Learning from Human Feedback (RLHF). The authors show that the additional, complex exploration strategies commonly used in prior work—such as UCB, Thompson sampling, or optimism—are "unnecessary".

### Strengths
1. A New, Simpler Algorithm: It proposes KL-EXP for contextual bandits and its variant OEPO for RLHF. Unlike previous methods, these algorithms do not use any explicit exploration bonuses (like UCB) and rely solely on the inherent exploration provided by the KL-regularized objective.

2. Logarithmic KL-Regularized Regret: The paper proves that KL-EXP/OEPO achieves logarithmic KL-regularized regret (e.g., $\mathcal{O}(\eta d \log T)$ for linear classes). This is the first theoretical result to show that a KL-regularization-only approach can achieve this standard, strong regret bound without extra exploration mechanisms6666

### Weaknesses
1. It is interesting that they do not need exploration, but in the proof sketch, it is not clear why their analysis do not need optimism. Could the authors provide more ideas about how to derive equation (5) in the proof sketch.

2. Their proof technique resembles this work [1]. They are suggested to cite this paper.

[1] Qingyue Zhao, et al, Towards a Sharp Analysis of Offline Policy Learning for f-Divergence-Regularized Contextual Bandits

3. For the experiment, it is better to ablate on the effect of different values of eta to study its effect on the performance.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
