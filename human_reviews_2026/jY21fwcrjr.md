# GUIDE: Gated Uncertainty-Informed Disentangled Experts for Long-tailed Recognition

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 6

## Abstract
Long-Tailed Recognition (LTR) remains a significant challenge in deep learning. While multi-expert architectures are a prominent paradigm, we argue that their efficacy is fundamentally limited by a series of deeply entangled problems at the levels of representation, policy, and optimization. These entanglements induce homogeneity collapse among experts, suboptimal dynamic adjustments, and unstable meta-learning. In this paper, we introduce GUIDE, a novel framework conceived from the philosophy of Hierarchical Disentanglement. We systematically address these issues at three distinct levels. First, we disentangle expert representations and decisions through competitive specialization objectives to foster genuine diversity. Second, we disentangle policy-making from ambiguous signals by using online uncertainty decomposition to guide a dynamic expert refinement module, enabling a differentiated response to model ignorance versus data ambiguity. Third, we disentangle the optimization of the main task and the meta-policy via a two-timescale update mechanism, ensuring stable convergence. Extensive experiments on five challenging LTR benchmarks, including  ImageNet-LT, iNaturalist 2018, CIFAR-100-LT, CIFAR-10-LT and Places-LT, demonstrate that GUIDE establishes a new state of the art, validating the efficacy of our disentanglement approach. Code is available at Supplement.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
GUIDE aims to improve Long-Tailed Recognition problem by solving homogeneity collapse in multi-expert architectures. Three levels of disentanglement in the learning process is identified as the bottleneck and handled sequentially. **Representation-decision entanglement** is handled by two regularization losses which encourages diversity at feature level and predictive logits level separately. **Cause-symptom entanglement** is handled by decomposing epistemic and aleatoric uncertainty to guide the adaptation policy. **Learning-meta-learning entanglement** in meta learning is handled by using different scales of learning rate for two sets of parameters. Through experiments, GUIDE shows efficacy at medium and few shot classes. The ablations show the necessity and synergy between disentanglements.

### Strengths
1. The paper is well written with clarity and well-illustrated in general.
2. The insights that different levels of entanglements are novel. Especially the disentanglement for representation-decision establishes a new paradigm beyond traditional logit adjustment.
3. The necessity of most components in the contributions is examined in the ablation study. Hyperparameter sensitivity also looks stable. 
4. Proof in the theorems are valid and clear.

### Weaknesses
1. Theorem 1 only suggests maximizing JSD, not mentioning minimizing the cosine similarity between the feature vectors. Even though the synergy between two competitive regularizers are shown in the ablation, the motivation of discouraging cosine similarity is heuristic.
2. The applied approach decomposition of AU and EU is classical and recently shown as entangled [1]. Using separate modules for AU and EU may even boost performance. 
3. adaptive residual mixture is not ablated and there is no motivation
4. It is bit hard to understand the modle architecture from figure 2 as it contains both models, losses and optimization process. 

[1] Mucsányi, B., Kirchhof, M., & Oh, S. J. (2024). Benchmarking uncertainty disentanglement: Specialized uncertainties for specialized tasks. Advances in neural information processing systems, 37, 50972-51038.

### Questions
1. What is motivation of designing the refinement strength a monotonically decreasing function with respect to its aleatoric uncertainty?
2. In sec 3.2, what are pathways? Why are x as the intput to the pathways?
3. How to get classwise EU and AU using class-level Exponential Moving Averages of the diagnosed uncertainties? as the entropy in (4) and (5) works on the whole predictive distribution.
4. In table 5, right, is the total uncertainty the entropy of average predictive distribution? What is the performance of using ambiguous signals like high training loss?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes GUIDE, a novel framework for solving the long-tail identification problem. The authors argue that existing multi-expert architectures suffer from three levels of entanglement: representation-decision entanglement, cause-symptom entanglement, and learning-meta-learning entanglement. GUIDE systematically addresses these problems through hierarchical entanglement resolution methods: (1) competitive specialization objectives achieve representation deentanglement, (2) adaptive policies based on uncertainty decomposition achieve policy deentanglement, and (3) dual-timescale optimization achieves optimization deentanglement. Experiments on five long-tail benchmark datasets demonstrate that GUIDE achieves a new SOTA.

### Strengths
1. The paper provides a novel and insightful diagnosis of long-tailed recognition challenges.
2. The module design is well-motivated. The uncertainty decomposition, though lacking rigorous justification, provides a practical heuristic for adaptive refinement that works in practice.
3. The experimental evaluation is thorough and convincing.

### Weaknesses
### 1. **Overstated Connection Between JSD Maximization and Performance Improvement**

The paper's Theorem 1(b) claims that "maximizing JSD serves to tighten this performance bound," but this relationship is **not rigorously proven**. The mathematical derivation shows:

-  **Problematic**: The gap in Jensen's bound ≠ JSD
-  **Unproven**: Maximizing JSD → Tightening the bound

The authors acknowledge in the proof that *"the gap in the Jensen bound is not strictly equal to JSD"* and that maximizing JSD *"empirically encourages"* diversity. This is an **empirical heuristic**, not a theoretical guarantee. The authors can provide rigorous proof, or explicitly redefine it as an empirical observation rather than a theoretical contribution.

### 2. **Lack of Theoretical Foundation for Uncertainty Decomposition**

The epistemic/aleatoric uncertainty decomposition (Equations 4-5) is presented as a principled approach but lacks theoretical justification:
```
Aleatoric: AleT(x) = (1/E)ΣH(pe,T(·|x))
Epistemic: EpiT(x) = H(p̄T(·|x)) - AleT(x)
```

**Issues**:
- No proof that this decomposition correctly identifies "model ignorance" vs "data ambiguity"
- No theoretical guarantee that high epistemic uncertainty → need more refinement
- No validation that high aleatoric uncertainty → avoid over-refinement

While the appendix mentions this is a *"standard and widely-accepted definition in the Bayesian deep learning community,"* this is **community convention**, not mathematical fact. The entire Level ❷ policy is built on this unproven assumption. The authors need to clarify that this is an empirical design choice and provide empirical validation (e.g., showing the correlation between these quantities and actual model/data properties), or provide theoretical justification for why this decomposition is suitable for this task.

### 3. **Lack of related work**

MDCS: More diverse experts with consistency self-distillation for long-tailed recognition (ICCV2023) is also a method that uses diversity experts and ensemble learning in long-tail recognition, and it needs to be compared and discussed.

### Questions
How do the inference efficiency and training memory consumption of this method compare to other methods?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces GUIDE, a novel framework conceived from the philosophy of Hierarchical Disentanglement.

### Strengths
A novel framework conceived from the philosophy of Hierarchical Disentanglement is proposed. In addition, this paper systematically addresses these issues at distinct levels.

### Weaknesses
The authors only verify the method on computer vision datasets. If possible, the proposed method is suggested on more long-tailed datasets (e.g., MEET) from more domains.

### Questions
Please consider to improve the experimental analysis on more datasets.

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
4

### Summary
This paper addresses the problem of long tailed recognition (LTR) in multi-expert architectures via a so-called hierarchical disentanglement (GUIDE). This is done by introducing specific mechanisms that address problems such as entaglement causing homogeneity collapse, entanglement of policy-making from ambiguous signals by decomposing epistemic/aleatoric uncertainty with a dynamic refinement module, and introduces an approach for stabilizing convergence via different timescale updates. Experiments on several dataset demonstrate improvements. Please see below for discussion.

### Strengths
Below I list some of the main strengths of the paper:
- the three leveled framework is theoretically motivated 
- results appear to improve over state of the art
- particularly strong improvement on few-shot classes
- comprehensive evaluation (e.g. distribution shift)
- ablation studies validate the contribution of each component separately, and the impact of combining two components leading to further improvements.

### Weaknesses
- Not sufficient discussion on complexity and convergence rates
- Discussion on how hyperparameters (inc learning rates and various λ) affect guarantees
- hierarchical disentanglement can mostly be considered as a smart combination of existing approaches
- would be interesting to include more analysis of failure cases
- improve for clarity - be clear on problems that are existing challenges and not new, and on what is proven theoretically vs empirically validated

### Questions
- some more ellaboration on design choices; e.g. why in fig 3B only marginal gains are achieved with E=4? Why residual gating? (eq. 6)
- B.4 - robustness to mis-guided refinement: 21.7% of samples were correctly classified after the refinement step, even when guided by incorrect proxy label. What about the rest? what is the net effect?
- analysis on computational cost, memory footprint
- please see also above

### Soundness
3

### Presentation
3

### Contribution
3
