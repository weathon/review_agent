# From Acceleration to Saturation: Scaling Behavior of Bootstrapped Language Model Pretraining

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Bootstrapped pretraining, i.e., the reuse of a pretrained base model for further pretraining, such as continual pretraining or model growth, is promising at reducing the cost of training language models from scratch.
However, its effectiveness remains unclear, especially when applied to overtrained base models.
In this work, we empirically study the scaling behavior of bootstrapped pretraining and find that its scaling efficiency diminishes in a predictable manner: The scaling exponent with respect to second-stage pretraining tokens decreases logarithmically with the number of tokens used to pretrain the base model.
 The joint dependence on first- and second-stage tokens is accurately modeled by a simple scaling law.
 Such saturation effect reveals a fundamental trade-off in multi-stage pretraining strategies: the more extensively a model is pretrained, the less additional benefit bootstrapping provides.
 Our findings provide practical insights for efficient language model training and raise important considerations for the reuse of overtrained models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper investigates the scaling behavior of bootstrapped language-model pretraining. The authors discover a predictable saturation effect: the scaling exponent with respect to second-stage tokens D_2 decays logarithmically with first-stage tokens D_1. They introduce a multiplicative scaling law which accurately captures this interaction across model sizes and bootstrapping variants. Extensive fits show their law outperforms additive or hybrid forms and remains valid under different learning-rate schedules, replay ratios, and growth factors. By pairing the law with Chinchilla-style compute scaling, they derive practical thresholds that indicate when training from scratch overtakes bootstrapping, offering actionable guidance for compute-optimal pretraining.

### Strengths
1. This paper is the first to identify and quantify the saturation effect in bootstrapped pretraining, providing a theoretically grounded scaling law with an interaction term that precisely models the phenomenon.

2. Large-scale empirical validation across diverse model sizes, training strategies, and configuration variants demonstrates the robustness and general applicability of the proposed scaling law.

3. The work extends the data-only scaling law to a joint data-and-model-size formulation, delivering practical criteria for choosing between bootstrapping and from-scratch training under compute budgets.

### Weaknesses
1. Limited Scale of Experiments: The study's largest model is 1.1B parameters, trained on hundreds of billions of tokens, which is significantly smaller than modern LLMs (often 7B to trillions of parameters). The observed scaling saturation may not hold at these larger, more relevant scales, limiting the practical applicability of the proposed laws for state-of-the-art model training.

2. Narrow Scope of Domain Shift: The analysis of saturation is primarily conducted on domains (e.g., code, math) that are relatively similar to the original pre-training data, especially in the model growth experiments. The work does not investigate if the same saturation effects and scaling laws apply to highly dissimilar domains or cross-modal tasks, raising questions about the generalizability of the findings.

3. Lack of Mechanistic Explanation: The paper provides a descriptive scaling law and a high-level hypothesis about loss of plasticity but lacks a deeper investigation into the underlying cause of the saturation. Without analyzing the mechanistic reasons, the work offers limited guidance on how to potentially mitigate the saturation effect.

### Questions
As described in Weakness. Expanding the experimental scope to include larger models (7B+) and more rigorous cross-domain benchmarks would greatly enhance the generalizability and broader impact of the findings.

### Soundness
2

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
4

### Summary
This paper proposes a new scaling law to describe bootstrapped pretraining—a unified term encompassing both continual pretraining (CPT) and model growth.
Unlike the classical power-law scaling, the proposed formulation introduces a logarithmic interaction term with respect to the first-stage dataset size D1:
L(D1,D2) = AD_1^(-alpha1) * D_2^(-alpha2+alpha3 * log(D_1)) + E
The key insight is that when the base model is over-trained (large D1), subsequent bootstrapped training suffers from saturation, yielding diminishing returns.
Extensive experiments confirm that this form achieves the lowest RMS fitting error among tested alternatives.
The paper further discusses practical implications for balancing bootstrapping versus training from scratch and for compute-optimal allocation.

### Strengths
1. The experiments convincingly demonstrate that including the log(D1)​ term improves fit quality across both continual pre-training (domain shift) and model-growth (no shift) settings.
2. The proposed scaling law reveals the impact of saturation, which contributes to more in-depth research on the best practice of multi-staged pre-training
3. The paper is clearly written, with well-structured sections and thorough comparisons against additive and hybrid baselines.

### Weaknesses
1. The mathematical intuition is not convincing enough: if the power laws are accepted (as condition 1 and condition 2), it should also be accepted that equation (2) should reduce to equation (6) when no distribution shift is involved. However, equation (2) does not model the impact of distribution shift (as the paper claims that it is also suitable for model growth with no distribution shift), so it seems not consistent with equation (6). 

Perhaps the work will benefit from proving that (2) is better than (6) even when there is no distribution shift (perhaps with experimental results), and then it can be self-consistent. If this is the case, the top-down derivation part can be removed.

2. The practical implications are extrapolated from relatively small scales. Validation at larger model and data scales is necessary before treating these findings as practically predictive.

### Questions
1. As noted above, could you clarify the theoretical necessity of Eq. (2) over Eq. (6) when no distribution shift is present, and ideally provide experimental confirmation?

2. Regarding the second-stage training, is the learning-rate schedule re-warmed from scratch, or does it continue the remaining portion of the stage-1 schedule? (CPT can be vulnerable to learning rate setting for extremely large models (>30B), does eq.(2) still hold with less reasonable learning rate schedules?)

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper addresses the reuse of pretrained base models (bootstrapped pretraining) for improving a subsequent run on more data (continual pretraining, CPT) or more parameters (model growth). The authors build on the idea of *ossification*  under CPT, and discuss a general notion of *saturation* of a base model, such that any subsequent training does not provide a proportional marginal gain.
The power law loss relation is expanded with a combined effect on loss from tokens seen during the base model training *and* that seen in the second stage of bootstrapped pretraining.
Both analytically and empirically, the proposed functional forms for the scaling laws appear to match and accurately capture the core behaviour observed: overtraining a base model with more tokens leads to a *lesser* gain in loss in the second stage of training.
Thus, a practical guide emerges for when one should train a model from scratch versus start from a previously trained model (same size or smaller).

### Strengths
* Well motivated scope, and of relevant timing given the spurt in model growth and continual learning literature with scaling becoming more ubiquitous. 

* The novel functional form to model loss given parameters and data from two training stages appear well established through empirical evidence.

* Training details and various design choices, along with the scaling fit procedures, are well reported.

* The finding can be reduced to a simple rule-of-thumb for practical deep learning under bootstrapped pretraining.

### Weaknesses
* Some of the notations used, especially with the overloading of $\alpha$ slow down the accurate parsing of the paper and thereby the understanding of parts of the analysis (refer to *Questions* below).

* Clarity: some of the claims and assumptions in the text are without citations, references, or more details (refer to *Questions* below).

* The equivalence of both continual pretraining (CPT) and model growth as bootstrapped pretraining procedures is not adequately explained. Despite the apparent adherence to the proposed functional form of the power-law loss, Table 2 and Figure 4 show the quality of fit being better for CPT.
  * This can veer into a longer discussion of the complete difference in both the CPT and Stacking second-stage training w.r.t. data and thus the learning scope. Therefore, it questions the interpretation of $L_{D_1}(D_2)$ as equivalent.

### Questions
Various questions and suggestions are enumerated below.

Please note, the rating will go up contingent on the following points, with higher weightage on: 3, 4, 6, 9.


1\.1. L47-49: Could the authors intuit here more, how and where was the adverse effect of overtraining on second-stage pretraining seen?

1\.2 L47-49: We are still observing a different scaling behaviour, as captured with the fitted parameters to the functional form (Eq. 2). Other than the power law requirement, what other `scaling behaviour` do we want to keep consistent (especially in the widely accepted *token-per-parameter* parlance)?

2\. L39,L109: The citation needs fixing.

3\.1. Figure 1: What is the model size here?

3\.2. Figure 1 (bottom left) and Figure 3 (right): Does this indicate that as long as the second stage training is *overtrained*, the training of the smaller model is irrelevant?

3\.3. How does this relate to the N, the parameters of the base model, in consideration? 
* Could the main finding here be presented in relation to the Chinchilla, 20-tokens-per-parameter ratio for N and D?

4\. More generally, how does Table 7 or 8 compare to larger extrapolations of the scaling fits, beyond the validation data available? Especially when comparing the projected loss of more general functional forms, not capturing the interaction term of $D_1$ and $D_2$.
Alternatively, if possible, comparing projected loss (and a ($D_1$, $D_2$) recipe) from the fits obtained, to losses reported on training from scratch at similar (N, $D_2).

5\. L222-223: The Condition 2 for $L_{D_2}(D_1)$ should also be an equation for consistency. Adding subscripts to the $\alpha$s in both the equations should be useful.

6\. L241-244: The inline equality presented is hard to understand given $\alpha_1$ is not present in the RHS. Could the authors please clarify? 

7\. The footnote 3 reveals a rather nice design choice in terms of a $\sqrt{2}$ scaling of the *aspect ratio* (depth/width) for model growth. This point could indeed be more explicit.

8\. L417: This comparison of training from scratch is only applicable for model growth and not CPT. Should the paragraph title reflect so and not bootstrapped pretraining generally?

9\. L436-446: Is there a limit to the performance of Stacking? That is, is there a slow down in convergence rates to the point a $2N$ model trained from Scratch achieves better loss with the same tokens seen?

10\. To extend 9 above, are the conclusions drawn, based on the empirical evidence collected, specific to the choice of methods used for model growth and CPT? Especially given that both growth and CPT are by now a varied body of literature. What do the authors think about this and therefore the claims made on *general bootstrapped pretraining*?

11\. Appendix B.3: What is the *standard practice* used here for batch size scaling? How are the warmups in Table 5 chosen, w.r.t. L805?

12\. Table 3: Is it normal for function-preservation on width-expansion under inconsistent head sizes (d_{model} / n_{head})?
* Could the authors clarify which method is being used in the experiments?

13\. Figure 10: The labels on the plot are confusing based on their alignments. Could that please be explained better in text or caption?

### Soundness
3

### Presentation
2

### Contribution
3
