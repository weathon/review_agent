# Estimating structural shifts in graph domain adaptation via pairwise likelihood maximization

- Avg Score: 5.00
- Decision: Reject
- Scores: 2, 6, 6, 6

## Abstract
Graph domain adaptation (GDA) emerges as an important problem in graph machine learning when the distribution of the source graph data used for training is different from that of the target graph data used for testing. While much of the prior work on GDA has focused on the idea of aligning node representations across source and target domains, recent studies show that such approaches can be suboptimal in the presence of conditional structure shift (CSS), where the distribution of graph edges conditioned on labels changes across domains. In this work, we develop a unified framework to solve CSS and show that existing GDA methods for CSS arise as special cases of our framework. This framework further allows us to develop a new method, Pairwise-Likelihood maximization for graph Structure Alignment (PLSA), which uses rich information from pairwise nodes and edges to improve the estimation of target connection probabilities. We establish conditions under which our method is identifiable and introduce a simple edge reweighting scheme based on importance weights to align the source and target graphs. Theoretically, under the contextual stochastic block model (CSBM), we derive finite-sample guarantees using recent results in matrix concentration inequalities for U-statistics. We complement our theoretical results with empirical studies that demonstrate the effectiveness of our method.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper studies graph domain adaptation (GDA) under conditional structure shift (CSS)  a scenario where the conditional edge distributions given node labels differ across domains. The authors propose a unified theoretical framework and introduce Pairwise-Likelihood maximization for Graph Structure Alignment (PLSA), which estimates target connection probabilities via pairwise likelihood matching with a calibrated source predictor. Theoretical guarantees are derived under the Contextual Stochastic Block Model (CSBM), showing finite-sample error bounds based on matrix concentration inequalities for U-statistics

### Strengths
The paper provides a mathematically principled framework for conditional structure shift estimation. The identifiability analysis and finite-sample guarantees are technically sound and build upon nontrivial extensions of label shift theory to the graph domain.
Using pairwise likelihood maximization for structural alignment is an elegant generalization of label-shift maximum-likelihood estimation to GDA. The unified view encompassing existing methods such as Structural Reweighting and Pair-Align adds theoretical coherence to an emerging subfield.

### Weaknesses
1. **Incorrect or overly strong assumption (Line 063):**
   The statement *“assuming that the joint distribution of features and labels are invariant across source and target domains”* is conceptually inconsistent with the GDA setting.

   * In graph domain adaptation, the core challenge arises because **the joint distribution ( p(x, y) )** is *not* invariant across domains; otherwise, the task degenerates to a standard supervised setting.
   * The authors likely intend to isolate *conditional structure shift* by assuming ( p(y) ) and ( p(x|y) ) are invariant, but phrasing it as joint invariance misrepresents the GDA assumptions and should be corrected.

2. **Incomplete related work discussion:**
   The related work section omits recent state-of-the-art methods that are directly relevant for GDA under structural or spectral shift. 

[1] Pang, Jinhui, et al. "Sa-gda: Spectral augmentation for graph domain adaptation." Proceedings of the 31st ACM international conference on multimedia. 2023.

[2] Fang R, Li B, Zeng Q, et al. On the Benefits of Attribute-Driven Graph Domain Adaptation[C]//The Thirteenth International Conference on Learning Representations.

[3] Yang L, Chen X, Zhuo J, et al. Disentangled Graph Spectral Domain Adaptation[C]//Forty-second International Conference on Machine Learning.

3. **Limited experimental validation:**
   The experiments are restricted to synthetic CSBM data and the small-scale Airport dataset, which do not represent standard GDA benchmarks.

   * Commonly adopted datasets such as **Citation networks**, **MAG dataset**, and **BlogCatalog** are missing.(Liu M, Zhang Z, Tang J, et al. Revisiting, benchmarking and understanding unsupervised graph domain adaptation[J]. Advances in Neural Information Processing Systems, 2024, 37: 89408-89436.)

   * Without evaluations on these real-world benchmarks, it isn't easy to assess whether PLSA generalizes beyond the stylized CSBM scenario.
   * Additionally, ablation studies on calibration quality and sparsity sensitivity would strengthen empirical claims.

4. **Broader applicability and assumptions:**
   The CSS-only assumption (Assumption 3.1) is quite restrictive. In practice, label shift and structure shift often coexist. Although Appendix B sketches a potential extension, the main text does not empirically demonstrate PLSA’s robustness under mixed shifts.

### Questions
See Weaknesses

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents a unified framework to solve conditional structure shift (CSS) problem and show that existing GDA methods for CSS arise as special cases by theoretical analysis. Then, the authors proposed a new method called Pairwise-Likelihood maximization for graph Structure Alignment (PLSA）by estimating the target connection probability by matching the distribution of features and edges through nodes in the latent space.

### Strengths
1. Sufficient and Solid Theoretical Analysis. This paper provides non-asymptotic error upper bounds under CSBM, explicitly pointing out the relationship between sample complexity, the number of classes, and calibration errors, which enhances the credibility and soundness of the method.
2.  Good Performance on Sparse Graph Scenarios. PLSA uses unconditioned pairs (including non-edge pairs) rather than restricting to “edge” samples. From a statistical efficiency standpoint, this retains more information in sparse graphs. Experiments (Figures 1 and 2) show that in sparse settings, PLSA significantly outperforms Pair-Align, which uses only edge information.

### Weaknesses
1.  The comparative experiments are not sufficiently comprehensive. Although the paper reviews some existing methods in the GDA field, it omits some important approaches (e.g., meta-learning–based or adversarial-training–based approaches), making it difficult to fully demonstrate the superiority of the model.
2. The article’s theoretical and methodological designs largely based on the assumption that class priors and class-conditional feature distributions are domain-invariant (Assumption 3.1). This assumption weakens the method’s applicability to more complex real-world scenarios (e.g., when both label shift and feature distribution drift are present).
3. While the experimental results show that PLSA outperforms the baselines, no experiment is provided to analyze parameter sensitivity.

### Questions
1. The re-sampling and re-weighting scheme proposed in Section 4.3 (using Bernoulli insertion/deletion for each class pair) introduces additional random noise. The paper does not analyze how this randomization affects the variance of downstream GNN training, nor does it compare the pros and cons of using expected weights (soft-weight) versus sampling.
2. It is recommended to further discuss the model’s time-complexity such as analyzing the efficiency of the proposed method on large-scale graph datasets.
3. For experiments on Airport dataset , the graph structure is real but the node features are synthesized (the feature-label association is manually designed). Under real-world conditions with genuine node features, the model’s performance may be affected. It is suggested to include some datasets with authentic features for experimental analysis.

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
This paper handles the graph domain adaptation problem, specifically focusing on dealing with the conditional structure shift (CSS) by assuming no shift in conditional feature and label distribution. Particularly, it improves upon previous work like pair-align by considering a more general edge distribution including existing edges and non-edges with pairwise likelihood maximization. They also include the analysis bounding the estimation of importance weights considering sample gap and classifier miscalibration gap.

### Strengths
- This paper focus on a crucial point ignored by previous literature in solving CSS using edge reweighting, specifically point out the importance of consider the full distribution of potential edges by removing the condition that given an existing edge, this essentially consider currently non-existed edges which work well even under sparse graph case.
- They form a unified and clear comparison with previous works by correctly position their contribution and distinctions from previous weight estimation methods and the shift cases considered.
- The estimation is also supported by error bound and they additionally consider the impact from calibrated classifier beyong simply assuming the invariant conditional feature distribution.

### Weaknesses
- Although the focus might be on the theoretical part and the paper verifies them via synthetic datasets, but it could be better if we can add more real datasets, especially the ones that have more sparse structure to showcase the benefit of this new weight estimation
- Also, it could be better if you can motivate and highlight in the introduction or before talking about exact method regarding why previous methods are insufficient using some dataset statistics, like how sparse they are, how biased they can be under this case.
- I believe it could be better if you put appendix B to the main text including both CSS and label shift, or this work is more like comparing to StruRW without label shift. Then, you might want to clarify how we need to ensure the ratio is not biased with label shift.

### Questions
- Based on my understanding, one additional benefit is that we consider a calibrated classifier in this case in addition to pair align method. Then to what extend you think the benefit might come from this calibrated classifier besides the benefit we consider full edge distribution using PLSA. Can there by some ablation study on this?
- After getting the ratio that we need to adjust the source graph, the way the paper did is actually resampling instead of reweighting the source graph right? Can you evaluate the strength and weakness of resampling compared to reweighting in this case?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper identified and solevd the problem of conditional structure shift in Graph domain adaptation. They proposed Pairwise-Likelihood maximization for graph Structure Alignment for estimating and correcting conditional structure shift in node classification tasks.

### Strengths
1. The probelm of CSS is important and interesting, the problem setup is clear.
2. The alignment of the structure with divergent $p(y,y^{\prime})$ is novel.
3. The method is reasonable, and the theoretical results seem correct.

### Weaknesses
1. Line 24-25 seems like a LLM-style polish, em dash is not usually used in academic paper.


Frankly speaking, I'm not the expert in GNN, so I strongly encourage the AC to add another expert or ignore this review.

### Questions
See weakness

### Soundness
2

### Presentation
2

### Contribution
2
