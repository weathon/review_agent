# Fingerprinting Deep Neural Networks for Ownership Protection: An Analytical Approach

- Decision: Accept (Poster)
- Scores: 8, 4, 6, 6

## Abstract
Adversarial-example-based fingerprinting approaches, which leverage the decision boundary characteristics of deep neural networks (DNNs) to craft fingerprints, has proven effective for protecting model ownership. However, a fundamental challenge remains unresolved: how far a fingerprint should be placed from the decision boundary to simultaneously satisfy two essential properties—robustness and uniqueness—required for effective and reliable ownership protection. Despite the importance of the fingerprint-to-boundary distance, existing works offer no theoretical solution and instead rely on empirical heuristics to determine it, which may lead to violations of either robustness or uniqueness properties.

We propose AnaFP, an analytical fingerprinting scheme that constructs fingerprints under theoretical guidance. Specifically, we formulate the fingerprint generation task as the problem of controlling the fingerprint-to-boundary distance through a tunable stretch factor. To ensure both robustness and uniqueness, we mathematically formalize these properties that determine the lower and upper bounds of the stretch factor. These bounds jointly define an admissible interval within which the stretch factor must lie, thereby establishing a theoretical connection between the two constraints and the fingerprint-to-boundary distance. To enable practical fingerprint generation, we approximate the original (infinite) sets of pirated and independently trained models using two finite surrogate model pools and employ a quantile-based relaxation strategy to relax the derived bounds. Particularly, due to the circular dependency between the lower bound and the stretch factor, we apply a grid search strategy over the admissible interval to determine the most feasible stretch factor. Extensive experimental results demonstrate that AnaFP consistently outperforms prior methods, achieving effective and reliable ownership verification across diverse model architectures and model modification attacks.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes an adversarial-example-based fingerprinting approach for Deep Neural Networks (DNNs). It investigates the fingerprint-to-boundary distance, which is a fundamental challenge in current DNN fingerprinting methods. By introducing a tunable stretch factor and determine the lower and upper bounds of this factor, the authors mathematically define an admissible interval within which the fingerprint-to-boundary distance must lie.

### Strengths
The paper is well-written and clearly introduces the challenges associated with current DNN fingerprinting methods as well as the proposed solution. 


The paper offers an effective solution for quantitatively characterizing model boundaries for models that function similarly or identically but originate from different sources.

### Weaknesses
The effectiveness of the proposed method heavily depends on the prior knowledge provided by the 120 pirated-independent model pairs in the model pool. To protect a model, it is necessary to construct a model pool, and techniques such as Knowledge Distillation (KD) and Adversarial Training (AT) introduce additional overheads that may limit practical applications.

### Questions
The authors claim that selecting high-confidence samples ensures strong uniqueness. However, logically, the closer the fingerprint is to the decision boundary, the more likely it is to ensure uniqueness. In generating adversarial fingerprint samples, a smaller update step size should, in theory, bring the fingerprint closer to the decision boundary. Selecting high-confidence samples as anchors could potentially increase the number of iterations, but this may not necessarily reduce the distance of the generated adversarial fingerprint sample from the decision boundary. If time permits, I would appreciate it if the authors could provide experimental results using mid-confidence or low-confidence examples as anchors.

### Soundness
3

### Presentation
4

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
This paper introduces AnaFP, an analytical framework for generating adversarial-example-based fingerprints for deep neural network model ownership verification. The key contribution is the mathematical formalization and unified treatment of both robustness and uniqueness constraints, enabling derivation of a theoretically principled interval for positioning fingerprints relative to the decision boundary. AnaFP employs surrogate model pools and a quantile relaxation strategy to make implementation practical and robust to variations in model modification and diversity.

### Strengths
- The paper rigorously formalizes the requirements for fingerprint robustness and uniqueness, deriving both lower and upper bounds on the stretch factor that controls fingerprint placement. 

- AnaFP is demonstrated on a diverse set of DNN models, indicating broad applicability. Unlike some baselines (UAP, MarginFinger), AnaFP naturally extends to non-Euclidean domains.

- The paper provides well-designed ablations on surrogate pool size/diversity, quantile relaxation parameters, and the selection of the stretch factor τ. 

- The methodology is well structured and enhanced with explanatory figures.

### Weaknesses
- While the authors thoroughly test robustness to model modifications and evaluate discriminability between pirated and independent models, there is no assessment of scenarios with "unknown" models or more ambiguous conditions (e.g., surrogate attacks specifically designed to evade fingerprints; adaptive adversaries targeting decision geometry). Similarly, detailed analysis of the potential for false positives in more open-world settings is missing.

- AnaFP's practicality depends on the construction of surrogate pirated and independent pools. While sensitivity to pool size/diversity is empirically analyzed, the method still assumes access to representative pools, which may not always be feasible—especially in dynamic, heterogeneous model deployment ecosystems. There is little discussion of how errors in pool selection could impact the upper/lower bound estimation and thus fingerprint viability.
 
- Although quantile-based relaxation is well motivated, the chosen quantiles are somewhat arbitrary and task dependent. The method remains sensitive to settings that are not fully justified theoretically.

- The fingerprint creation protocol may incur significant computational costs for large-scale or production systems. There is no empirical profiling of runtime or scalability as the number of fingerprints, surrogates, or model size increases.

### Questions
- How would AnaFP fare against adaptive attackers who are aware of the fingerprint generation protocol and attempt to evade by specifically distorting or randomizing localized decision boundaries? Are there (perhaps adversarially optimized) model modifications that could reduce the efficacy of the derived interval without violating standard performance constraints?

- Can the authors clarify or provide more guidance on choosing quantile thresholds for the relaxation strategy (see Table 3), beyond the trial-and-error or “reasonable number of fingerprints” heuristic? Any analytical or automated procedure to further reduce parameter sensitivity?

- The use of grid search with potentially many anchors and surrogate models may be computationally expensive, especially for large-scale or production systems. Could you provide more details on the runtime complexity and memory requirements for AnaFP in realistic settings?  

- The evaluation focuses on known model modification attacks (pruning, fine-tuning, KD, AT). Have you considered or tested robustness against adaptive attackers who might specifically attempt to evade fingerprinting by perturbing near-anchor regions or strategically altering decision boundaries?  

- While AnaFP shows strong results on CNNs, MLPs, and GNNs, are there any specific architectural choices, dataset characteristics, or domains where the approach might face limitations? For example, very large-scale models or models trained with self-supervised or unsupervised objectives?

### Soundness
2

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
This paper presents AnaFP, an analytical framework for generating adversarial fingerprint samples to verify the ownership of deep neural networks. Unlike prior heuristic approaches that place fingerprints near decision boundaries, AnaFP derives theoretical upper and lower bounds on the fingerprint-to-boundary distance, governed by a scaling factor. The bounds are motivated by two conflicting objectives: robustness against model modifications and uniqueness across independently trained models. The authors further propose a practical approximation using surrogate model pools and quantile relaxation, followed by a grid search to select feasible fingerprints.

### Strengths
1. The paper provides a clean analytical formulation for the long-standing heuristic choice of how far fingerprints should lie from the decision boundary. By deriving explicit upper and lower bounds on the scaling factor, the authors turn this into a principled optimization problem rather than an empirical guess.
2. Experiments cover multiple model families (CNN/MLP/GNN), datasets (image and graph), a wide range of model modification attacks, several baselines, and repeated runs to report mean ± std. Tables and figures indicate consistently strong AUCs for AnaFP.

### Weaknesses
1. The related-work section lacks a systematic overview of existing fingerprinting or watermarking techniques for deep models.
2. While the appendix mentions some hyperparameters, key details such as exact model architectures used and number of models in each surrogate pool are missing or only briefly listed. These are central to understanding the experiment setup. Important configurations should appear in the main text, not just in appendices.
3. While AnaFP is analytically grounded, its deployment requires maintaining surrogate model pools, performing adversarial optimization (e.g., C&W attacks) for fingerprint generation, and running grid search for each anchor. These steps incur substantial training and inference cost, which may limit practicality in large-scale or resource-constrained settings. The paper would benefit from reporting quantitative cost and discussing potential efficiency improvements.

### Questions
See Weaknesses

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper focuses on the fingerprint generation for model ownership verification based on two essential properties—robustness and uniqueness from a theoretical perspective, which links the two constraints introduced by robustness and uniqueness to the fingerprint-to-boundary distance. The authors formalize two key properties—robustness (against model modifications) and uniqueness (against independently trained models)—and derive τ bounds that satisfy both. They further relax these bounds via surrogate model pools and quantile estimation, and determine τ via grid search. Extensive experiments show superior AUC over prior work across CNN, MLP, and GNN architectures.

### Strengths
1.	To the best of my knowledge, this might be the first work to analytically characterize the admissible τ-interval that jointly guarantees robustness and uniqueness. The derivation is clean and verifiable, and surrogate pools + quantile relaxation elegantly make the bounds estimable without violating the theory.
2.	Consistently highest AUC on all six modification attacks and four datasets, while showing low sensitivity to pool size/quantile settings.
3.	Extensive results demonstrate the superior performance of AnaFP over other methods, which can generalize to CNN, MLP, and GNN models among diverse model IP attacks.
4.	While casting fingerprint generation as a “distance-to-boundary” control problem is not new (MarginFinger, IPGuard), AnaFP’s main novelty is adding analytical bounds on the stretch factor τ. I believe this idea and perspective could motivate more follow-up work in the community.

### Weaknesses
1.	My first concern is about the cost. Every anchor requires a full targeted C&W optimization (≈ 3,000 steps) and a 500-point grid search over τ. Complexity is O(N_f × 3,000 × 500) forward-backward passes—impractical for ImageNet-scale models. No speed-up (e.g., early termination, bisection search, Jacobian-free solvers) is discussed.
2.	Missing baselines. The experimental comparison is restricted to adversarial-example methods; recent non-adversarial ownership schemes are omitted, which can provide more necessary information on AnaFP’s superior performance.
3.	The evaluated surrogate piracy pool is limited to six handcrafted attacks (fine-tune, KD, prune). Stronger attack combinations (like prune→KD or adversarial training with fingerprint-aware data) are not covered.

### Questions
1.	Could the authors provide more information on the fingerprints’ generation cost rather than only the detection performance? Like time and memory size compared to other existing methods
2.	KD is known as the most challenging IP attack in the real-world scenario. I wonder about the detailed attack settings and detection performance(e.g., detection rate and FPR) on this attack, which could significantly help readers to buy these results.

Overall, although the current version of this paper lacks some important results and details, I believe the theoretical perspective presented is somewhat reasonable and could inspire more future work in similar directions. Therefore, I am currently slightly inclined to accept it. I hope the author will provide point-by-point responses to the issues I raised during the rebuttal phase. If these issues can be adequately clarified, I would be willing to raise my score and support the acceptance of this paper.

### Soundness
3

### Presentation
3

### Contribution
3
