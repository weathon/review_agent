# Conformalized Hierarchical Calibration for Uncertainty-Aware Adaptive Hashing

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
Unsupervised domain adaptive hashing transfers knowledge from labeled source domains to unlabeled target domains, addressing domain shift challenges in real-world retrieval tasks. Existing methods face two critical limitations: target domain noise severely misleads model training, and indiscriminate domain alignment strategies treat all target samples equally, potentially distorting essential feature structures. We propose an uncertainty-aware adaptive hashing approach that addresses these challenges through a hierarchical conformal calibration framework. At the semantic level, we employ conformal inference to generate confidence prediction sets, replacing single pseudo-labels with set-based predictions whose sizes directly quantify sample reliability for weighted pseudo-label learning and domain alignment. This enables the model to focus on reliable samples while suppressing noise. At the representation level, we predict the stability of individual hash bits, where bit-level confidence guides a robust weighted quantization loss and enables dynamic weighted Hamming distance during retrieval, fundamentally enhancing hash code quality and retrieval robustness. Through this hierarchical calibration mechanism, our method achieves more adaptive and robust cross-domain knowledge transfer. Extensive experiments on multiple benchmark datasets demonstrate significant improvements over existing approaches, validating the effectiveness and superiority of our method.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Conformal Hierarchical Calibration Adaptive Hashing, a framework for Unsupervised Domain Adaptive Hashing. The core problem it addresses is that existing UDAH methods suffer from noisy pseudo-labels and heuristic uncertainty handling, e.g., softmax thresholding, when adapting from a labeled source domain to an unlabeled target domain. At the semantic level, it uses conformal prediction set sizes to weight pseudo-label and alignment losses. At the representation level, it predicts individual hash bit stability to guide a weighted quantization loss and a novel weighted Hamming distance during retrieval. Experiments show COLA outperforms state-of-the-art methods on benchmark datasets.

### Strengths
The framework extends uncertainty quantification to the representation level by modeling the stability of individual hash bits. The proposed self-regulating mechanism intelligently uses quantified uncertainty signals to balance the multi-objective loss. The method achieves state-of-the-art performance, outperforming recent strong baselines on multiple benchmarks.

### Weaknesses
1. The paper's main solution adopts current theoretical results and methods in the literature of conformal prediction under domain shift, limiting its novelty.
2. The paper uses CP-based set size $1/|\mathcal{C}(x_t)|$ as its uncertainty measure. However, the paper does not compare it against a baseline that uses simpler heuristics (e.g., SoftMax threshold) within the COLA framework.
3. The retrieval efficiency could be low. Despite a comparison to dense vector in D.5, the current hash code weighted by floating points can show substantial time consumption compared to common binary hashing methods, violating the basic goal of hashing retrieval.
4. Formatting issues and typos:
 - Figure 5 obscures some text above it.
 - Typo around Line 231: "learnin"

### Questions
1. How does the domain shift severity influence the best threshold $r_{cal}$? This seems also heuristic.
2. Hashing retrieval are also designed for multi-label scenarios. Does the proposed method apply to them?

### Soundness
2

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
3

### Summary
This paper proposes an uncertainty-aware deep hashing method for cross-domain retrieval, addressing the problem of domain shift through conformal prediction. The conformal prediction strategy transforms the predicted classes with the highest softmax probabilities into a pseudo-label set, which serves as a surrogate for potentially noisy ground-truth labels. The resulting confidence estimates of the hash codes are then used to guide domain alignment, hash learning, and retrieval. Experimental results demonstrate consistent improvements over existing baselines.

### Strengths
1. Presentation. 
The paper is well-structured and clearly written, with a logical flow and well-defined sections. The mathematical notation is consistent and easy to follow, making the technical content accessible to readers familiar with the field.

2. Empirical Evaluation. 
The experimental evaluation on benchmark datasets is comprehensive and convincing. The inclusion of detailed plots and visualizations effectively illustrates the results and provides clear insights into model behavior. The figures are well-designed and aesthetically appealing.

3. Originality and Significance. 
While several prior studies have explored uncertainty-aware hashing strategies, the application of conformal prediction to deep hashing remains relatively underexplored. This work thus presents a novel and interesting direction that could inspire further research in uncertainty modeling for retrieval under domain shift.

### Weaknesses
1. Need for a Brief Introduction to Conformal Prediction. 
The paper would benefit from a short subsection introducing the basic concepts and principles of conformal prediction. This would make the paper more self-contained and accessible to readers who may not be familiar with this framework.

2. Missing Discussion of Related Work
The literature review omits several recent works that have explored uncertainty-aware deep hashing or uncertainty modeling in retrieval, which are directly relevant to this study. In particular, the following papers should be cited and discussed for a more comprehensive positioning of the proposed approach:

- Warburg, Frederik, et al. "Bayesian triplet loss: Uncertainty quantification in image retrieval." Proceedings of the IEEE/CVF International conference on Computer Vision. 2021.

- Warburg, Frederik, et al. "Bayesian metric learning for uncertainty quantification in image retrieval." Advances in Neural Information Processing Systems 36 (2023): 69178-69190.

- Wang, Yucheng, and Mingyuan Zhou. "Uncertainty-aware unsupervised video hashing." The 26th International Conference on Artificial Intelligence and Statistics. PMLR, 2023.

- Wang, Yucheng, Mingyuan Zhou, and Xiaoning Qian. "Hashing with Uncertainty Quantification via Sampling-based Hypothesis Testing." Transactions on Machine Learning Research.

- Tang, Haomiao, et al. "Modeling uncertainty in composed image retrieval via probabilistic embeddings." Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2025.

3. Some design choices lack sufficient motivation. Please refer to Question 3 for details.

4. Missing Complexity Analysis: Section 3.3.1: Uncertainty-aware Weighted Hamming Distance.
While the introduction of a weighted Hamming distance adds modeling flexibility, the paper does not provide a computational complexity analysis. This omission is problematic because the weighting mechanism changes the operational nature of the distance metric — from bitwise operations to floating-point vector arithmetic — which may significantly impact runtime performance and scalability in large-scale data retrieval. A formal complexity discussion, or at least empirical runtime analysis, is necessary to demonstrate the method’s feasibility in real-world retrieval settings.

5. Could the authors provide an ablation experiment in which all three components (SC, RC, and SR) are removed? This would help clarify the individual contribution and overall effect of each module.

### Questions
1. Line 144-146: "The size of these sets rigorously quantifies model uncertainty and is directly converted to weights that adaptively suppress the harmful effects of high-uncertainty samples in pseudo-label learning and domain alignment. " The authors should clarify why this uncertainty is categorized as model uncertainty rather than data uncertainty. Since the uncertainty appears to arise from the variability or ambiguity in the data samples (e.g., ambiguous labels or overlapping class features), it might be more appropriately described as data uncertainty. A brief justification or theoretical reasoning distinguishing the two types of uncertainty in this context would strengthen the conceptual rigor of the paper.

2. The paper employs a conformal prediction strategy to transform high-confidence softmax predictions into pseudo-label sets that replace the original noisy labels. However, at the early stages of training — when the model is randomly initialized — the softmax probabilities are effectively random. This implies that the generated pseudo-label sets could be even noisier than the original labels, potentially leading to unstable or misleading supervision.
Could the authors provide a discussion or empirical justification addressing this issue? For example, is there any mechanism (e.g., warm-up phase, confidence thresholding, or delayed pseudo-labeling) to mitigate the effect of unreliable predictions during early training?

3. Section 3.3: Proxy Task for Bit Stability. It appears that the stability measure $v_{i,k}$ could be obtained directly by adding perturbations (e.g., Gaussian noise) to the feature representation and simulating the corresponding bit responses. If that is the case, could the authors clarify why a separate confidence prediction head, $G_{bit}$, is necessary?

4. Experimental Setup and Fairness of Comparison
It is unclear whether the proposed model and the baseline models are evaluated using the proposed Uncertainty-Aware Weighted Hamming Distance (UWHD) or the standard Hamming distance. If the proposed method uses UWHD while the baselines rely on the original Hamming distance, this may introduce a fairness issue in performance comparison.
The authors may need to clarify this point and provide additional experiments where: (1) Baseline models are also evaluated using the proposed UWHD to ensure a fair comparison under consistent retrieval metrics. (2) An ablation study isolates the contribution of UWHD itself to quantify its effect on retrieval accuracy and runtime.
This clarification is important because the proposed UWHD may both increase retrieval cost and contribute independently to performance improvements.

### Soundness
3

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
This paper proposes a conformalized hierarchical calibration method called COLA for unsupervised domain adaptive hashing that integrates uncertainty quantification at both semantic and representation levels. The method leverages conformal prediction to produce confidence-calibrated pseudo-labels and introduces bit-level reliability modeling for hash code robustness. Additionally, a self-regulating mechanism dynamically balances learning objectives based on real-time uncertainty. Experiments on some benchmarks demonstrate improvement over state-of-the-art baselines.

### Strengths
1.	The use of conformal prediction for uncertainty estimation in domain adaptive hashing is innovative and grounded in theory, supported by formal coverage guarantees under domain shift.

2.	The hierarchical semantic and bit-level calibration is well-structured and addresses both label noise and representation robustness.

3.	Extensive experiments and clear performance gains across diverse benchmarks demonstrate effectiveness and robustness.

### Weaknesses
1.	The method includes introduces multiple components inducing semantic-level conformal calibration, dynamic threshold, bit-level reliability modeling, and a self-regulating optimization mechanism. While each is well-motivated, it would be beneficial if the authors could more explicitly clarify which part constitutes the core contribution and novelty of the method.

2.	The model contains many weighting mechanisms such as semantic-level sample weights, batch-level confidence weights, bit-level stability weights, and dynamically regulated loss coefficients. While each seems intuitively reasonable, the true necessity and effectiveness of these weights remain unclear. Although the conformal calibration module is theoretically grounded, the other weighting mechanisms also appear heuristic.

3.	Some notations and equations need further explanations. For example, how to compute $\hat{q}^W$ in Eq. 1 is unclear.

### Questions
see weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the problem of uncertainty estimation in unsupervised domain adaptive hashing (UDAH). It introduces COLA, a hierarchical framework that unifies semantic-level conformal calibration and representation-level (bit-wise) reliability modeling under a self-regulating training scheme. At the semantic level, COLA employs conformal prediction to generate multi-label prediction sets instead of single pseudo-labels, ensuring controlled coverage even under domain shift. At the representation level, COLA estimates per-bit reliability by measuring feature stability under perturbations. These bit confidences reweight quantization loss and are also applied during retrieval as a query-weighted Hamming distance, enabling uncertainty-aware retrieval. Extensive experiments on Office-Home, Office-31, and MNIST-USPS datasets show consistent improvements in mAP and robustness, supported by ablations for each module.

### Strengths
- A clear strength of this paper is its two-level uncertainty design, which handles noise both in pseudo-label assignment and binary code learning. The semantic-level conformal predictor generates calibrated prediction sets to control pseudo-label confidence, while the representation-level reliability estimator measures bit-level stability and integrates these confidences into the learning process. This combination enables consistent management of uncertainty from prediction to representation.

- The paper provides an explicit theoretical framing via Theorem 3.1, which establishes a coverage bound of 1 - \alpha_t - \text{TV}(P_s, P_t) under domain shift. The theorem connects conformal calibration with target-domain generalisation, grounding the semantic calibration step in probabilistic coverage terms. Although the reduction of the TV term has not been empirically proven, the presence of an analytical coverage expression distinguishes the paper from heuristic uncertainty-weighting works.

- The inclusion of a self-regulating mechanism that reweights alignment and quantisation losses according to aggregated confidence metrics represents another notable design feature. By dynamically adjusting these weights, the framework avoids the need for manual tuning of trade-off hyperparameters and adapts its learning emphasis based on reliability feedback.

### Weaknesses
1) **Retrieval efficiency and scalability untested.** The proposed uncertainty-weighted Hamming distance (Eq. 9) replaces the standard bitwise XOR + popcount kernel that underpins hashing’s speed with a per-bit weighted sum. While accuracy gains are clearly demonstrated on Office-Home, Office-31, and Digits benchmarks, these datasets are only moderate in size and do not represent million-scale retrieval scenarios. The paper does not report retrieval latency, throughput, or index compatibility for this weighted metric, and its complexity analysis covers only calibration (sorting O(n_{\text{cal}}\log n_{\text{cal}})). Thus, the impact on large-scale system efficiency remains unquantified.

This is because the proposed distance fundamentally changes the computation model from binary bit-operations to floating-point per-bit weighting. Therefore, it alters the very assumption that gives hashing its appeal: sub-millisecond similarity search using bitwise logic. Demonstrating that the new retrieval kernel still scales efficiently is therefore not an auxiliary test but a technical validation step. It confirms that COLA preserves hashing’s defining property of efficiency while adding uncertainty awareness. Without such timing and scalability evidence, the proposed method’s theoretical soundness is intact, yet its practical completeness is uncertain. At least, it is unclear to me whether the algorithm remains a true hashing framework or behaves more like a compact continuous embedding model.

Adding runtime and scalability experiments on standard large-scale retrieval corpora such as NUS-WIDE, MS-COCO etc, along with a UDA-at-scale check on VisDA-2017 or DomainNet would directly verify that the uncertainty-weighted Hamming distance maintains the computational advantages of hashing.


2. **Calibration-set construction is heuristic.**
The paper defines D_{\text{cal}} as the top r_{\text{cal}} of source samples nearest to the target centroid (default 20%). While this is a reasonable heuristic and sensitivity to r_{\text{cal}} is analyzed, there is no comparison with alternative calibration-set selection rules (e.g., per-class, random, or density-aware sampling). Since Theorem 3.1’s coverage guarantee assumes that the calibration distribution approximates the target-domain score distribution, could the authors clarify how representative the chosen D_{\text{cal}} is of the target domain in practice? Have the authors tested whether alternative calibration-set constructions produce similar coverage and retrieval performance, or could provide evidence (e.g., distributional similarity or coverage plots) to verify that the conformal bound remains valid under this heuristic?

3. The adaptive \alpha_t is an elegant idea, but its contribution is not explicitly ablated against a fixed-\alpha baseline. That is, the paper introduces an EMA-based dynamic \alpha_t, but there is no analysis of why a time-varying \alpha_t yields better or more stable coverage than a fixed \alpha, nor is there an ablation comparing the two. Since \alpha_t directly controls prediction-set size and, through it, the confidence-weighted alignment, please provide: (i) a fixed-\alpha baseline on all three datasets; (ii) a plot of empirical coverage vs. time for both fixed and dynamic \alpha; and (iii) an explanation of how the EMA parameters were chosen. Without this, the dynamic schedule appears ad hoc rather than necessary.

4. The theoretical result (Theorem 3.1) depends on a reduction of the TV distance between source and target conformity-score distributions through alignment. However, the paper does not empirically estimate or proxy this TV term, so the connection between the alignment loss and actual coverage improvement remains unverified experimentally. Please (i) justify why this bound is meaningful under the non-exchangeable source/target feature distributions in these datasets, and (ii) provide an empirical proxy (e.g., pre-/post-alignment MMD on the conformity scores) to support the claim that alignment makes the conformal coverage guarantee non-vacuous.

### Questions
1. The current ablation toggles semantic calibration (SC), representation calibration (RC), and self-regulation (SR), and concludes that each helps. However, this does not isolate which part of the proposed conformalization is responsible for the final mAP improvements, because all components are evaluated under the same centroid-based D_{\text{cal}} and dynamic \alpha_t. To make the causal story convincing, please provide an ablation that (i) fixes D_{\text{cal}} to a random subset and (ii) fixes \alpha, and then measures the marginal gain from adding SC and RC separately. This will show whether the benefit truly comes from conformal calibration, rather than from generic confidence weighting.

2. All reported experiments are on Office-Home, Office-31, and MNIST↔USPS, which are small-to-moderate in size. Yet the method introduces a retrieval rule that is more expensive than standard binary hashing. Please clarify the intended scope of the claim: is COLA meant only for moderate-scale UDAH (as in the current experiments), or is it intended to be a scalable, uncertainty-aware hashing method? If the latter, additional experiments on larger-scale benchmarks (DomainNet, VisDA-2017, or NUS-WIDE/MS-COCO for retrieval timing) are needed to support that scope.

### Soundness
3

### Presentation
3

### Contribution
3
