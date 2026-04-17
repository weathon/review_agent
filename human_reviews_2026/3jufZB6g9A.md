# Utility Boundary of Dataset Distillation: Scaling and Configuration-Coverage Laws

- Decision: Reject
- Scores: 4, 6, 4, 2

## Abstract
Dataset distillation (DD) aims to construct compact synthetic datasets that allow models to achieve comparable performance to full-data training while substantially reducing storage and computation. Despite rapid empirical progress, its theoretical foundations remain limited: existing methods (gradient, distribution, trajectory matching) are built on heterogeneous surrogate objectives and optimization assumptions, which makes it difficult to analyze their common principles or provide general guarantees. Moreover, it is still unclear under what conditions distilled data can retain the effectiveness of full datasets when the training configuration, such as optimizer, architecture, or augmentation, changes. To answer these questions, we propose a unified theoretical framework, termed configuration–dynamics–error analysis, which reformulates major DD approaches under a common generalization-error perspective and provides two main results: (i) a scaling law that provides a single-configuration upper bound, characterizing how the error decreases as the distilled sample size increases and explaining the commonly observed performance saturation effect; and (ii) a coverage law showing that the required distilled sample size scales linearly with configuration diversity, with provably matching upper and lower bounds. In addition, our unified analysis reveals that various matching methods are interchangeable surrogates, reducing the same generalization error, clarifying why they can all achieve dataset distillation and providing guidance on how surrogate choices affect sample efficiency and robustness. Experiments across diverse methods and configurations empirically confirm the derived laws, advancing a theoretical foundation for DD and enabling theory-driven design of compact, configuration-robust dataset distillation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper presents a unified theoretical framework for dataset distillation (DD), a field focused on synthesizing compact datasets that preserve the performance of the original, large-scale data. The authors make significant contributions by proposing a "configuration-dynamics-error" framework that unifies the three dominant DD paradigms—Gradient Matching (GM), Distribution Matching (DM), and Trajectory Matching (TM)—and derives two fundamental laws governing the process.

### Strengths
1. The paper fills a critical gap in the DD literature by providing the first unified theoretical foundation. The coverage law, in particular, is a novel and powerful concept that moves beyond the typical single-configuration analysis and directly addresses the practical requirement of robustness in distilled datasets.

2. The experimental section is well-designed, validating both the scaling and coverage laws across multiple datasets (MNIST, CIFAR-10/100, ImageNette) and several canonical DD methods.

### Weaknesses
1. The theoretical analysis relies on several strong assumptions, most notably the Polyak-Łojasiewicz (PL) condition for the inner-loop optimization dynamics.

2. The definition of "configuration" is focused on optimization hyperparameters and architecture. It does not explicitly model more fundamental distribution shifts, such as semantic or domain shifts, which are also critical for robustness.

3. While the experiments cover standard benchmarks, validating the coverage law on larger, more complex datasets like full ImageNet or in cross-domain transfer settings would further strengthen the paper's impact.

### Questions
See weaknesses.

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
4

### Summary
This paper proposes a unified configuration–dynamics–error framework that integrates gradient, distribution, and trajectory matching within a generalization-error analysis. It establishes the scaling law and coverage law linking distilled sample size to performance and configuration diversity, theoretically and empirically unifying major dataset distillation methods.

### Strengths
1.	The paper presents a unified bi-level generalization error framework that connects gradient, trajectory, and distribution-based DD, providing an important step toward a unified theoretical foundation for DD.
2.	The proposed scaling and coverage laws formalize intuitive empirical observations into mathematically grounded relationships.

### Weaknesses
1.	The framework relies on PL conditions and Lipschitz continuity. While these assumptions are standard in convergence analysis, they may not strictly hold for modern deep networks with non-smooth activations, normalization layers, and stochastic training components. The practical relevance of the theoretical results could be further clarified by discussing their validity under relaxed or empirically realistic assumptions.
2.	The validation of the proposed laws relies mainly on curve-fitting without statistical significance tests, variance analysis, or sensitivity checks. While the observed trends are consistent with the theory, the slopes vary across datasets and architectures, and further analysis could help clarify these differences and strengthen the empirical support for the proposed laws.
3.	While the paper introduces the configuration–dynamics–error decomposition, its underlying structure closely follows standard generalization-error decompositions and stability analyses. The framework builds on established theoretical tools and primarily reinterprets them within the context of dataset distillation, rather than introducing fundamentally new analytical techniques or tighter bounds. As such, the contribution may be viewed more as a conceptual consolidation than a substantial theoretical advance.
4.	The coverage diversity $H(A,r)$ is defined abstractly through a covering number on configuration space. While this formulation is theoretically elegant, it may be challenging to compute or approximate in practical training scenarios.
5.	The theoretical insights could be further translated into clear, actionable guidelines (e.g., for selecting IPC or surrogate objectives), which would help enhance the practical relevance of the work.

### Questions
1.	Could the authors clarify the practicality of the PL and Lipschitz assumptions for modern non-smooth architectures, and whether the results hold under weaker conditions?
2. It would be helpful if the authors could clarify whether any statistical tests or variance analyses were conducted to validate the fitted slopes, and how sensitive the results are to random initialization or dataset variations.
3. Please clarify the main theoretical novelty of the configuration–dynamics–error decomposition beyond its conceptual unification of existing generalization and stability analyses.
4. The coverage diversity term $H(A,r)$ is defined via an abstract covering number in configuration space. Is there a practical way to estimate or approximate this quantity in real-world training scenarios?
5. Could the theoretical insights, such as the scaling and coverage laws, provide clearer practical implications for choosing IPC values, surrogate objectives, or data selection strategies in dataset distillation?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a unified theoretical framework to study dataset distillation (DD) from a generalization-error perspective. The authors derive two key theoretical results: (i) a *scaling law* that characterizes how test error decreases with distilled dataset size under a fixed training configuration, and (ii) a *coverage law* that quantifies how the required distilled sample size must scale with the diversity of training configurations (e.g., architectures, optimizers, augmentations) to maintain performance. The framework unifies three major DD paradigms: gradient matching (GM), distribution matching (DM), and trajectory matching (TM), as different surrogates minimizing the same underlying alignment discrepancy. Empirical validation is provided across standard benchmarks (MNIST, CIFAR-10/100, ImageNette) and representative DD methods.

### Strengths
Strengths:
1. **Theoretical novelty and unification**: The paper offers the first generalization-error-based–based framework that unifies disparate DD approaches under a common lens. This is a significant conceptual advance over prior paradigm-specific analyses.  
2. **Clear and impactful theoretical results**: The scaling law explains the widely observed IPC saturation phenomenon, while the coverage law formally defines the “utility boundary” of distilled data across configuration shifts, a practically relevant and previously unaddressed question.  
3. **Tight bounds with matching upper/lower results**: The coverage law includes both upper and lower bounds that match up to constants, establishing near-optimality of the √H/k rate.  
4. **Empirical validation aligns with theory**: Experiments across multiple DD methods and datasets consistently show the predicted 1/√k scaling and √H/k coverage behavior, lending strong support to the theoretical claims.

### Weaknesses
---

**Weaknesses:**  
1. **Limited experimental scale**: All experiments are conducted on relatively small-scale vision datasets (MNIST, CIFAR, ImageNette). The absence of evaluation on larger, more realistic benchmarks (e.g., ImageNet-1K or language datasets) raises concerns about the practical relevance and scalability of the derived laws in modern settings.  
2. **Proxy for configuration diversity**: The paper approximates coverage complexity Hcov(A, r) by log M (M = number of configurations), which may oversimplify the true geometric structure of the configuration space. A more refined empirical estimation of dA (e.g., via gradient-based distances) would strengthen the experimental validation.  
3. **Assumption dependence**: The theoretical analysis relies on Polyak–Łojasiewicz (PL)-type contraction and Lipschitz assumptions, which may not hold in highly non-convex or adaptive optimization settings (e.g., large-batch AdamW). The robustness of the laws under such conditions remains unverified.  
4. **Lack of comparison to recent large-scale DD methods**: While the paper includes MGD3 (a diffusion-based method), it omits comparison to state-of-the-art scalable DD approaches like SRe2L or TESLA, which are designed specifically for large datasets and may challenge or refine the proposed utility boundary.
5. "Missing some SOTA baselines": I have listed them below. Please show the experiments about them. 



Towards Lossless Dataset Distillation via Difficulty-Aligned Trajectory Matching. ICLR 2024

Dataset Distillation with Neural Characteristic Function: A Minmax Perspective. CVPR 2025.

### Questions
See Weakness.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper proposes a unified framework, the configuration-dynamics-error framework, to better understand the scaling and generalizability of existing dataset distillation algorithm. This frameworks places existing dataset distillation algorithm such as gradient matching, distribution matching, and trajectory matching into a single framework for analysis. The framework provides a scaling law, which captures the relationship between distilled dataset size and test error. The framework also provides coverage law, which show how distilled sample size should scale with configuration diversity.

### Strengths
1. How to better analyze existing dataset distillation algorithms into a unified theoretical framework is a very important problem to solve as many of the existing dataset distillation algorithms lack strong theoretical foundations.
2. The paper offers a large amount of theoretical derivations (44 out of the 51 pages of the paper)

### Weaknesses
1. The key formulation, the unified form of stability summarized by equation 6, is not well justified. Why does the absolute difference in the expected risk is approximately upper bounded by optimization residual + statistical fluctuations + matching term. The paper to motivate them from stability and information-theoretical approaches to generalization but none of the three cited work are relevant. The notion of mutual information is not mentioned anywhere in the prior text.
2. The quality and quantity of the experimental justification to the proposed theoretical framework is very weak. Only 1 of the 51 pages are spent on experiments. The paper attempts to justify its scaling laws by fitting linear regression curves on data that is clearly not linear.
3. The presentation of the paper can considerably be improved. Figure 3, for instance, contains a total 15 plots. Some of the plots are very text that is completely unreadable. The captions also needs to be more descriptive on what the figure is displaying. The x axis labels are also covering the x-ticks which needs to be fixed.

### Questions
1. Table 1 suggest robustness of trajectory matching is low compared to distribution matching. What are the implications of this since trajectory matching is among the most popular dataset distillation algorithms that can work reliably across many different contexts. 
2. How does dataset distillation with Backpropogation through time (BPTT) fit in this unified framework?

### Soundness
2

### Presentation
1

### Contribution
2
