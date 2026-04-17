# Learning What to Do and What Not To Do: Offline Imitation from Expert and Undesirable Demonstrations

- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
Offline imitation learning typically learns from expert and unlabeled demonstrations, yet often overlooks the valuable signal in explicitly undesirable behaviors. In this work, we study offline imitation learning from contrasting behaviors, where the dataset contains both expert and undesirable demonstrations. We propose a novel formulation that optimizes a difference of KL divergences over the state-action visitation distributions of expert and undesirable (or bad) data. Although the resulting objective is a DC (Difference-of-Convex) program, we prove that it becomes *convex* when expert demonstrations outweigh undesirable demonstrations, enabling a practical and stable non-adversarial training objective. Our method avoids adversarial training and handles both positive and negative demonstrations in a unified framework. Extensive experiments on standard offline imitation learning benchmarks demonstrate that our approach consistently outperforms state-of-the-art baselines.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes ContraDICE, a novel offline imitation learning framework that learns both what to do from expert demonstrations and what not to do from undesirable ones. The authors formulate the learning objective as the difference between two KL divergences, enabling contrastive learning over state–action visitation distributions of good and bad behaviors.They prove that when the weight of expert data dominates (i.e., $\alpha\leq1$), the objective remains convex, ensuring a stable and non-adversarial optimization process. Building on this, the paper derives a tractable Q-learning formulation with a lower-bound surrogate objective and introduces a Q-weighted behavior cloning method for efficient policy extraction. Extensive experiments on MuJoCo, Adroit, and FrankaKitchen benchmarks demonstrate that ContraDICE consistently outperforms existing offline imitation learning baselines, effectively leveraging both positive and negative demonstrations. Overall, the work provides a theoretically sound and practically effective approach for learning from contrasting behaviors.

### Strengths
1. The paper proposes a contrastive dual-KL objective that simultaneously encourages the policy distribution to match expert demonstrations while diverging from undesirable ones. This formulation is both conceptually elegant and theoretically grounded, providing a unified convex optimization view of “learning what to do and what not to do.”
2. Although the original exponential-form objective can cause numerical instability during optimization, the authors carefully derive a tractable lower-bound surrogate. This design maintains theoretical consistency with the main objective while significantly improving training stability and efficiency.
3. The empirical evaluation is extensive, covering multiple D4RL benchmarks (MuJoCo, Adroit, and FrankaKitchen) and diverse data quality settings. The paper provides clear ablations on the influence of $\alpha$, robustness to bad demonstrations, and comparisons to strong baselines, which convincingly support the claimed advantages of the proposed method.

### Weaknesses
1. Although the lower-bound approximation improves training stability, it inevitably introduces a gap between the surrogate and the true objective. Ideally, this gap could be avoided if an unbiased estimator of the true objective were available, as it would allow direct optimization without sacrificing stability.
2. While the experiments on D4RL benchmarks are extensive, all tasks belong to continuous control domains with relatively clean and well-structured data. The paper would be stronger if it included experiments in more challenging or realistic environments (e.g., high-dimensional visual inputs, noisy demonstrations, or partially observable settings) to further test the robustness and generality of the method.

### Questions
Where is the derivation of Q-learning formulation below Eq. 2?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper addresses the challenge of offline IL when the dataset contains both expert and explicitly undesirable demonstrations. The authors propose ContraDICE, simultaneously matching the learned policy's state-action distribution to the expert distribution while actively diverging from the undesirable distribution. The core contribution is a "contrasting" objective defined as the difference of two KL divergences: $\min f(\pi) = D_{KL}(d^{\pi}||d^{G}) - \alpha D_{KL}(d^{\pi}||d^{B})$, which is proved to remain convex under the condition $\alpha < 1$. Analogously to the DICE family and IQ-Learn, they leverage Lagrangian duality and transform the constrained convex problem into an unconstrained, non-adversarial Q-learning objective. Furthermore, they introduce Q-Weighted BC for final policy extraction to avoid the reliance on "state values". The method shows good empirical performance in the Mujoco, Adroit and Kitchen environments.

### Strengths
1. The “learn what to do and what not to do” framing is intuitively appealing and practically relevant to safety-critical applications.
1. The paper provides a new formulation that combines attraction toward expert data and repulsion from undesired data, and proves its convexity under the condition $\alpha<1$.
1. The paper avoids unstable training by deriving a dual Q-learning formulation.
1. The proposed approach achieves good empirical performance improvements across diverse tasks and datasets.

### Weaknesses
1. The paper never clearly defines what qualifies as undesired. Are they failed trajectories, suboptimal but safe data, or catastrophic behaviors? The difference matters: discouraging mild inefficiency vs. avoiding dangerous actions are conceptually distinct. Without a formal definition or metric for “undesired,” it is unclear how $\alpha$ or the classifier boundaries correspond to actual safety or task performance. This ambiguity also limits the interpretability of results, since different tasks may treat “bad” differently (e.g., low-reward vs. unsafe). 

1. If the undesired behaviors are defined as catastrophic behaviors, the claim that the method "avoids" bad demonstrations is theoretically overstated. The objective $\alpha D_{KL}(d^{\pi}||d^{B})$ is a discouragement term, not a strict constraint. It pushes the learning policy away from $\pi^B$ but does not guarantee that the support of $d^\pi$ will be disjoint from that of $d^B$. 

1. The trade-off weight $\alpha$ determines how strongly bad behaviors are discouraged. However, no theoretical guidence for choosing its values, which is importance because the values near 1 lead to numerical instability (due to $1−\alpha$ in denominators).

1. The paper provides guarantees on the optimization problem (convexity) but lacks guarantees on the performance of the learned policy, i.e., there is no formal optimality gap with $J(π^E)$. Since the optimal solution is not necessarily the expert policy $π^E$ itself, it is unclear what the ultimate performance target of the algorithm is.

### Questions
1. Could the authors clarify whether their method assumes access to catastrophically bad trajectories (e.g., crashes), or whether it also works with mildly suboptimal ones? How sensitive is performance to label noise in $\mathcal{B}_B$? 

1. How should $\alpha$ be set in principle? Could it be learned adaptively?

1. Since $d_G$ is not always the minimizer, how should one interpret the target of ContraDICE?

1. Could the authors show quantitative metrics on how much bad behavior remains in learned trajectories? Do you evaluate in truly safety-critical tasks (e.g., with irreversible failures)?

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
4

### Summary
The paper introduces ContraDICE, a unified offline imitation learning method for datasets that mix expert demonstrations with clearly undesirable trajectories. The key idea is to learn a policy by minimizing a contrastive divergence over state-action visitation distributions,
$f(d_\pi) = D_{\mathrm{KL}}(d_\pi \parallel d_G) - \alpha D_{\mathrm{KL}}(d_\pi \parallel d_B)$,  with a provable convexity guarantee in $d_\pi$​ where $\alpha \le 1$. This condition keeps the objective convex in the occupancy measure and allows tractable optimization via Lagrangian duality. 
To make the method practical, the authors adopt the Extreme-V update from XQL to compute soft value functions efficiently, avoiding costly log-sum-exp operations. For policy extraction, they introduce Q-weighted behavior cloning and show that it achieves the same optimum as advantage-weighted behavior cloning while being more numerically stable in practice, as formalized in Proposition 5.1.

### Strengths
- The authors present a clear, principled goal: optmize $f(d_\pi) = D_{\mathrm{KL}}(d_\pi \parallel d_G) - \alpha D_{\mathrm{KL}}(d_\pi \parallel d_B)$ and show that it is convex in the occupancy measure when $\alpha \le 1$ (Proposition 4.1). This convexity enables a stable, non-adversarial optimization via duality (Section 4 and Appendix A). In short, framing the problem as a difference of KLs and proving convexity is clean and actionable.


- The authors recognize that the exponential terms in their dual formulation could cause numerical instability and address this through a tractable lower bound approximation. The resulting algorithm avoids adversarial training while maintaining theoretical guarantees.


- The evaluation spans a solid mix of settings, MuJoCo locomotion, Adroit Manipulation, and FrankaKitchen, and shows consistent gains over baselines (Table 1), with especially large improvements on the harder manipulation tasks where several prior methods fail outright. The paper supports reproducibility through comprehensive documentation, including full pseudo-code in Algorithm 2, detailed dataset construction procedures, complete hyperparameter specifications, and computational resource descriptions with concrete throughput metrics in Appendices C.1 and C.5.

### Weaknesses
- The convexity only holds for KL divergence, as stated in the paper. Mininizing KL divergence has its own advantages and disadvantages due to its characteristics. It only holds for $\alpha$ < 1 with Appendix D.11 showing performance degradation when $\alpha$ >= 1. This is a limitation when one wishes to emphasize avoidance of bad demonstration. 

- A significant limitation of this work is the lack of comparison with other DICE-based approaches that could utilize the same contrastive reward signal. The authors represent $\Psi(s, a) = \log\left(\frac{d_G(s, a)}{d_U(s, a)}\right) - \alpha \log\left(\frac{d_B(s, a)}{d_U(s, a)}\right)$ via good vs. bad classifiers, yet they do not evaluate whether existing DICE frameworks like DemoDICE [1] could achieve similar performance when equipped with its corresponding reward term (a variant or DemoDICE using $\Psi(s,a)$ instead of $r(s,a) = \log d^E(s,a)/ d^U(s,a)$). This omission makes it difficult to determine whether the performance gains stem from the novel dual KL formulation or simply from incorporating classifier-derived rewards. Even the remark in L247 acknowledges this could be formulated as a standard offline RL problem with reward $\Psi(s, a)$, but no empirical comparison follows.

- Performance is quite sensitive to the choice of  $\alpha$ and $\beta$ as shown in Figure 2 and Table 9. While the authors provide task-specific values in Table 3, there's no principled method for selecting these parameters without online evaluation. In offline settings where validation through environment interaction is impossible, this sensitivity presents a practical challenge for deployment.

### Questions
- Could you provide empirical comparisons where existing DICE-based methods (e.g., DemoDICE or standard offline RL algorithms) are given the same contrastive reward function $\Psi(s, a) = \log\left(\frac{d_G(s, a)}{d_U(s, a)}\right) - \alpha \log\left(\frac{d_B(s, a)}{d_U(s, a)}\right)$? This would help isolate the contribution of your dual KL formulation from the benefit of classifier-based rewards.


- How do you choose $\alpha$ in practice? Figure 2 shows sensitivity and Table 3 lists per‑task values, but a principled selection rule or validation criterion would help with deployment. See page 9 and page 23.


- Although the authors have not claimed computational efficiency, it would be great to include wall‑clock time as a fixed evaluation metric under a shared hardware profile for each method, and a micro‑benchmark quantifying the overhead of the two discriminators relative to a strong baseline. See Appendix C.5.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper considers a practical offline imitation learning (IL) problem where the dataset contains both expert and explicitly undesirable demonstrations. The authors propose ContraDICE, an offline IL algorithm designed to imitate the good while avoiding the bad. The method builds on the DICE framework by introducing a difference of KL divergences between the learned policy’s occupancy distribution and those of the good ($d_G$​) and bad ($d_B$​) behaviors.

To derive a practical algorithm, ContraDICE reformulates this convex objective in an inverse  Q-learning framework, extending the IQ-Learn formulation [1] to handle contrasting expert and undesirable data. Since the resulting objective involves exponential terms that cause numerical instability, the authors further (1) introduce a tractable lower-bounded surrogate objective that linearizes the exponential term and (2) adopt the Extreme Q-Learning (XQL) [2] to compute the soft value function efficiently.

Empirical results on an extensive set of benchmark environments demonstrate that ContraDICE consistently outperforms baselines, showing both robustness to undesirable data and high data efficiency under limited expert supervision.

### Strengths
### **Strength 1. Practically useful problem setting**

The paper tackles a practically important and underexplored problem in offline imitation learning, which uses expert and explicitly undesirable demonstrations simultaneously. The authors provide a mathematically grounded formulation based on a difference of KL divergences, offering a clear and principled way to encode both imitation and avoidance objectives within a unified convex optimization framework.  

### **Strength 2. Clear presentation and reproducibility**

Despite minor typographical errors, the paper is generally well written and logically organized. The mathematical exposition is clear enough for readers to follow the motivation, derivation, and algorithmic flow, leading to an overall strong presentation quality. In addition, the authors include all source code and implementation details as supplementary materials, ensuring full reproducibility and facilitating future research based on this work.

### **Strength 3. Strong technical treatment and handling of numerical issues**

The derivations are technically solid, and the authors carefully address potential numerical instability arising from the exponential terms in the dual formulation. Their introduction of a lower-bounded surrogate objective $\tilde{L}(Q,\pi)$ demonstrates strong awareness of practical optimization challenges and results in a stable, non-adversarial learning procedure.

### Weaknesses
### **Weakness 1. Insufficient justification for adopting the IQ-Learn framework**

From the main objective (Eq. 2) to the IQ-Learn-based objective introduced around line 223, the concrete derivation steps are missing even in Appendix. Moreover, the motivation for explicitly adopting the IQ-Learn framework remains underdeveloped. In Appendix B.1, the authors discuss the limitation of applying Lagrangian duality to general $f$-divergences but do not explain why IQ-Learn is preferable to KL-specific dual formulations or other DICE-style approaches (e.g., DemoDICE [3]). This omission leaves both a logical and empirical gap.

The paper should clarify which concrete property of IQ-Learn motivates its use for the offline IL setting, for example, improved convexity or conditioning of the optimization landscape, enhanced stability or variance reduction in the Q-function space, or algorithmic advantages for offline data such as avoiding adversarial training. Currently, none of these aspects are theoretically analyzed or empirically validated against KL-based DICE baselines.

Without such evidence, the adoption of IQ-Learn appears to be a subjective design choice rather than a principled methodological decision. The authors are encouraged to:

**1. Provide explicit derivation steps** connecting Eq. (2) to the IQ-Learn objective.  
**2. Add formal or empirical comparisons** isolating the effect of IQ-Learn versus a KL-based dual formulation (e.g., in terms of convergence, stability, or performance).  
**3. Present a theoretical justification** demonstrating the advantage of IQ-Learn under assumptions aligned with their setting, relative to DICE-based formulations.

Without these clarifications, the conceptual contribution remains ambiguous, and the necessity of adopting IQ-Learn over simpler dual methods is not convincingly justified.

### **Weakness 2. Missing comparison with natural extensions of DICE-based approaches**

Although the method defines a reward term, $\Psi(s,a) = \log\frac{d_G(s,a)}{d_U(s,a)} - \alpha \log\frac{d_B(s,a)}{d_U(s,a)}$, derived from good/bad classifiers, the paper does not compare ContraDICE against a straightforward baseline that employs this similar term within existing frameworks such as SMODICE, DemoDICE [3] (with the different choice of discriminator-based rewards) or offline RL approaches (such as XQL[2], OptiDICE [4], IQL[5], …) that directly uses $\Psi$ as a reward, as discussed on Remark in line 247), neither in theoretical nor empirical perspectives.
Such comparisons would isolate the benefit of the proposed dual-KL formulation from the effect of the classifier-based reward itself. 
This omission weakens the empirical validation, as it remains unclear whether the observed performance gains stem from the new optimization objective or simply from incorporating classifier-derived contrastive rewards.


### **Minor Comments**

- The name ContraDICE may cause conceptual misunderstanding, as the proposed method primarily builds upon the IQ-Learn formulation rather than the stationary distribution correction principle central to the DICE family. Hence, it might mislead readers into expecting explicit occupancy ratio estimation or dual f-divergence optimization. It would be clearer to adopt a name that more directly conveys its IQ-Learn-based nature or contrastive learning perspective—similar in spirit to UNIQ [6], which operates in a special case of the problem.

- In Table 1, standard deviations are reported as error bars; however, standard errors or confidence intervals would provide a more statistically meaningful measure of uncertainty.

- Typos:
  - Duplicated words: Equation equation (e.g. `Equation equation 3` in line 247, 255. `Equation equation 4` in line 270, `Equation equation 7` in line 939, …)
   - In the objective (2), the constraint should be considered for all $s\in S, a \in A$.
   - In line 766, $\log p^\pi$ seems to be corrected as $\log d^\pi$

### **References**

[1] Garg et al., “IQ-Learn: Inverse Soft-Q Learning for Imitation”, NeurIPS 2021.

[2] Garg & Hejna et al., “Extreme Q-Learning: MaxEnt RL without Entropy”, ICLR 2023.

[3] Kim et al., “DemoDICE: Offline Imitation Learning with Supplementary Imperfect Demonstrations”, ICLR 2022.

[4] Lee et al., “OptiDICE: Offline Policy Optimization via Stationary Distribution Correction Estimation”, ICLR 2021.

[5] Kostrikov et al., “Offline Reinforcement Learning with Implicit Q-learning”, ICLR 2022.

[6] Hoang et al., “UNIQ: Offline Inverse Q-Learning for Avoiding Undesirable Demonstrations”, 2024.

### Questions
**Q1.**  Among the technical components of ContraDICE, (1) the surrogate objective (Eq. 4, discussed in Appendix D.12) and (2) the Extreme-V objective (Eq. 6), which contributes most to the observed performance improvement or training stability? Is the commonly used log-sum-exp trick (i.e., `max + logsumexp(x − max)`) insufficient to address numerical instability? It seems that an additional ablation study would remain to clarify this point.

**Q2.**  What is the underlying reason for ContraDICE’s strong performance when trained solely with the good + mixed dataset configuration?

**Q3.**  In the undesirable demonstration-only setting discussed in Appendix D.10, what accounts for ContraDICE’s superior performance compared to UNIQ? Specifically, it is unclear whether (1) ContraDICE still leverages expert demonstrations in this setting (I’m confused because this section discusses the expert support), and (2) what the exact differences are between ContraDICE and UNIQ in terms of objective formulation and training setup.

### Soundness
2

### Presentation
3

### Contribution
3
