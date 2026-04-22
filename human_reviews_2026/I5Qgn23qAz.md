# Robust Domain Generalization under Divergent Marginal and Conditional Distributions

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 6, 4

## Abstract
Domain generalization (DG) aims to learn predictive models that can generalize to unseen domains.
Most existing DG approaches focus on learning domain-invariant representations under the assumption of conditional distribution shift (i.e., they primarily address changes in $P(X|Y)$ while assuming the label marginal $P(Y)$ remains stable). 
However, real-world data seldom satisfy this assumption.
Multiple domains often differ in more complex ways, where both the label distribution $P(Y)$ and the conditional distribution $P(X|Y)$ vary simultaneously.
In this work, we propose a new framework for robust domain generalization under divergent marginal and conditional distributions. 
We introduce a novel risk bound for unseen domains by explicitly decomposing the joint distribution into marginal and conditional components and characterizing risk gaps arising from both sources of divergence. 
To operationalize this bound, we design a meta-learning procedure that minimizes and validates the proposed risk bound across seen domains, ensuring strong generalization to unseen ones. 
Empirical evaluations demonstrate that our method achieves state-of-the-art performance not only on conventional DG benchmarks but also in challenging Multi-Domain Long-Tailed Recognition (MDLT) settings where both marginal and conditional shifts are pronounced.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper studies domain generalization under a compound shift in both $P(Y)$ and $P(Z|Y)$. A risk bound is provided, which decomposes the generalization gap into 2 parts: a prior shift term consisting of class distribution weighted risks, and a feature shift term consisting of Wasserstein distances between class-conditional feature distributions. Building on this, the paper proposes RC-Align, a meta-learning method that regularizes on a domain-class distribution alignment loss. Experiments on standard DG and MDLT benchmarks show strong average and worst-domain performance.

### Strengths
* This paper conducts a solid theoretical analysis of the domain generalization error. The decomposition into prior and feature shift terms is intuitive and insightful. The algorithm design is also closely connected with these theoretical impressions.
* Experimental results show decent performance improvement on both standard DG and MDLT benchmarks.

### Weaknesses
* From my current understanding, the current theoretical framework cannot guarantee generalization on an arbitrary target domain. In Theorem 3, the performance depends on the Wasserstein distance between the target data distribution and its best approximation via interpolation between source domains. Hence, it only guarantees generalization under the condition that the target domain is an interpolation of source domains, which is known to be well-resolved by ERM. The cases where the target domain is an extrapolation cannot be handled by the current results. Also, the idea of building generalization bounds via controlling the inter-domain feature distribution alignment is not new.
* The current hyperparameter selection scheme does not follow the DomainBed standard, which requires a sweep among a predefined joint distribution of all hyperparameters. The learning rate for RC-Align is fixed to 5e-5 as stated, which differs from the standard settings and may result in unfair comparison.
* The meta-learning scheme and the DA loss design are both borrowed from existing works. The methodological contribution is thus somewhat limited to a combination of existing methods. Also, the meta-learning scheme may introduce a significant computational burden compared to other baselines. This weakness is already acknowledged by the authors, which is good. However, there is no running time/memory comparison provided in the current version, so it is unclear whether the performance improvement is worthwhile with increased computational costs.

### Questions
* According to Theorem 3, it seems that it suffices to minimize the empirical risk $R_{D_i}$ on each source domain to achieve generalization, since the second term is intractable under the DG setting. I don't see a clear motivation for why we need additional upper bounds for $R_{D_i}$ in Theorems 1-2, since it can be directly optimized via ERM. Can the authors further explain this point?

### Soundness
3

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
4

### Summary
This paper addresses the challenging problem of domain generalisation under compound distribution shifts, where both the marginal label distribution and the conditional feature distribution differ across domains. The authors first derive a new theoretical upper bound, then prove that the practical Domain-Class Distribution Alignment loss can upper-bound the Wasserstein distance term in this bound, enabling a tractable optimisation objective. They propose RC-Align, a meta-learning framework that minimises this composite risk bound through a combination of cross-entropy and DA losses within a leave-one-domain-out training protocol.

### Strengths
The paper introduces a principled and interpretable risk decomposition that explicitly separates the effects of prior and feature distribution shifts. Empirically verify the correlation between DA loss and generalisation gap, and conduct ablations to show the complementary effects of the DA loss, meta-learning, and Manifold Mixup.

### Weaknesses
1. The generalization under concurrent marginal and conditional distribution shifts has been extensively studied in prior works (e.g., Hu et al., 2020; Tan et al., 2024), suggesting that this might not be a critical gap. However, this does not detract from the systemic and insightful theoretical framework presented by the authors.

2. A primary theoretical concern is that the PL condition represents a strong assumption regarding the non-convex loss landscapes inherent to deep neural networks (as acknowledged by the authors in the Appendix). This analysis should therefore be perceived as offering descriptive insights into the algorithm's dynamics within an idealised context, rather than providing stringent guarantees.

3. The DA loss is optimised by comparing feature distances between similar and dissimilar classes, whereas the Wasserstein distance evaluates differences in global distributions. The introduction of intermediate quantities and constants in the proof could render the bound relatively loose. This may reduce the theoretical robustness of the assertion that minimising the DA loss directly equates to minimising feature distribution mismatch.

4.  The analysis presented in Fig. 1 shows a strong correlation between DA loss and the generalisation gap. However, additional ablation studies isolating DA loss’s causal impact on robustness would further substantiate the claim, as potential confounding variables might be present.

5. Table 2 does not include the DomainNet results as presented in Table 3, leading to potential inconsistency in the results representation. Including these results would enhance the comprehensiveness of the analysis.

### Questions
1.While the generalization under concurrent marginal and conditional distribution shifts has been explored in previous works, emphasize how your theoretical framework offers a unique perspective or addresses gaps that these studies have not fully tackled.

2. Discuss the implications of the PL condition as a strong assumption in your analysis. Discuss any potential limitations and how these insights can still advance understanding in the field.

3. In regards to optimizing the DA loss, provide further clarification on the role of intermediate quantities and constants in your proof. 

4. In response to the suggestion for additional ablation studies in Fig. 1, propose any current or future experiments that could better isolate the causal impact of DA loss on robustness. 

5. Explain the discrepancy in the results between Table 2 and Table 3. If available, provide the missing DomainNet results in Table 2.

### Soundness
3

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
3

### Summary
This paper addresses domain generalization (DG) under compound distribution shifts where both the marginal label distribution P(Y) and conditional distribution P(X|Y) vary across domains. The authors propose RC-Align, a meta-learning framework grounded in a novel theoretical upper bound that explicitly decomposes generalization risk into prior shift and feature shift components. The method uses a Domain-Class Distribution Alignment (DA) loss combined with cross-entropy in a MAML-style meta-learning procedure. Experiments on standard DG benchmarks and Multi-Domain Long-Tailed Recognition (MDLT) settings demonstrate state-of-the-art performance.

### Strengths
1. The paper provides a clean theoretical decomposition of domain generalization risk into interpretable components (prior shift and feature shift).
2. Good performance on standard DG and MDLT benchmarks.

### Weaknesses
1. While the decomposition is useful, the individual components (domain alignment, meta-learning for DG) are well-established. The main contribution is combining them with theoretical justification, but the theoretical tools (Wasserstein distance bounds, InfoNCE decomposition) are standard.

2. The definition of $\pi$ in Theorem 1 is missing.

3. Although the theory motivates minimizing Wasserstein feature distance, the implemented DA loss is a heuristic contrastive loss that aligns features with class centroids. The connection between this loss and the Wasserstein bound is qualitative, not quantitative. The actual training objective may not truly minimize the theoretical upper bound.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
