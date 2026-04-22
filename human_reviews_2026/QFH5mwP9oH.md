# Adaptive Regularization for Large-Scale Sparse Feature Embedding Models

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 6, 8, 6

## Abstract
The one-epoch overfitting problem has drawn widespread attention, especially in CTR and CVR estimation models in search, advertising, and recommendation domains. These models which rely heavily on large-scale sparse categorical features, often suffer a significant decline in performance when trained for multiple epochs. Although recent studies have proposed heuristic solutions, the fundamental cause of this phenomenon remains unclear. In this work, we present a theoretical explanation grounded in Rademacher complexity, supported by empirical experiments, to explain why overfitting occurs in models with large-scale sparse categorical features. Based on this analysis, we propose a  regularization method that constrains the norm budget of embedding layers adaptively. Our approach not only prevents the severe performance degradation observed during multi-epoch training, but also improves model performance within a single epoch. This method has already been deployed in online production systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors study the one-epoch overfitting phenomenon in pCTR and pCVR models. They first conduct a theoretical analysis using Rademacher complexity, indicating that the norm of sparse embeddings leads to poor generalization. The authors propose an adaptive regularization method (AdamAR / AdagradAR)  with a dynamic strength of the embedding regularization terms. Experiments on several public datasets demonstrate the effectiveness of the proposed method.

### Strengths
1. Motivation. The paper studies a well-known and critical problem in pCXR prediction - the "one-epoch" overfitting problem.

2. Theoretical Grounding. The analysis links overfitting to Rademacher complexity, showing that embedding norm growth drives generalization error, giving a principled basis for adaptive regularization.

3. Technical Novelty. The proposed method is well-derived and elegant, with a limited impact on the model training.

### Weaknesses
1. Extension to Explicit Interaction Models. The analysis is only in DNNs, whose effectiveness is limited. It would be great to extend the theoretical discussion to explicit interaction models such as FM or CrossNet in DCN V2.

2. Experiments. Besides Wukong (BTW, the citation of the Wukong paper is wrong), other models in the experiment section are a bit out of date. It would be great to add experiments on the new SOTAs, e.g., DCN V2, xDeepFM.

3. Writing. The abstract contains limited concrete content. It would be great to be a bit more specific on the proposed theoretical analysis and method.

### Questions
1. Can the complexity analysis be extended to explicit feature interaction models like FM? It would be great to include such analysis since most modern recommendation systems heavily rely on explicit interactions.

2. How does the rewriting from Eq. 5 to Eq. 6 happen?

3. I'm curious about the effect of the proposed regularization method on the dimensional robustness of the learned embeddings. There are several recent works discussing the dimensional collapse (DC) issue in recommendation models [1,2], and I wonder whether there is a connection between the one-epoch phenomenon and DC. Can you provide an analysis of the singular spectrum of sparse embeddings?

[1]. On the embedding collapse when scaling up recommendation models. ICML 2024.

[2]. Balancing Embedding Spectrum for Recommendation. 2025.

[3]. Towards Mitigating Dimensional Collapse of Representations in Collaborative Filtering. WSDM 2024.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates the prevalent phenomenon of "one-epoch overfitting" in large-scale sparse feature embedding models, especially pertinent to click-through rate and conversion rate estimation in online advertising and recommendation systems. The authors provide a theoretical analysis rooted in Rademacher complexity, pinpointing the growth of embedding norms as the root cause of deteriorating generalization when training persists beyond a single epoch. To address this, they propose an adaptive regularization technique that dynamically adjusts regularization strength per embedding based on occurrence intervals, integrating seamlessly with standard optimizers like Adam and Adagrad. Comprehensive experiments across public and industrial datasets demonstrate that the proposed method outperforms existing baselines in mitigating overfitting and promoting stable, improved performance during multi-epoch training.

### Strengths
* Theoretical Insights: The paper offers a coherent theoretical framework using Rademacher complexity to explain one-epoch overfitting in sparse embeddings, clarifying a widely observed yet under-theorized phenomenon in deep CTR/CVR models (see Section 2.2 and 3).
* Principled Adaptive Method: Instead of relying on heuristics, the proposed adaptive regularization derives directly from the theoretical analysis, offering an intuitive and practical approach that is compatible with common optimizers (Section 3.1, Algorithm 1).
* Comprehensive Empirical Evaluation: The method is validated across a diverse set of datasets (iPinYou, Amazon, Avazu, proprietary LZD) and backbone architectures (DNN, WDL, DeepFM, WuKong), demonstrating strong and consistent improvements over multiple baselines (Tables 1, 2).

### Weaknesses
* Limited Positioning Relative to Broader Regularization Literature: While the paper covers baseline regularization approaches (L1, L2, weight decay), it does not sufficiently engage with recent or foundational work on sparsity-driven regularization at both the architectural and optimization levels. For instance, no discussion is provided relating their adaptive method to pruning/growth strategies or thresholding-based approaches, even though such techniques are central to the sparsity literature.
* Lack of Ablations: There is little to no ablation study dissecting critical components of the proposed method, such as the sensitivity to the base regularization coefficient $\alpha$ beyond Figure 3, or the explicit contribution of occurrence interval estimation. It remains unclear how much each modeling choice contributes to the performance gains.
* Hyperparameter Selection Transparency: Although hyperparameters are said to be chosen by grid search (Section 4.1.3), the details are sequestered in the appendix and visualization (Figure 3), limiting transparency and reproducibility. The main text should provide a concise summary of grid search procedures and best practice recommendations.
* Ambiguity in Some Notation and Algorithm Details: Certain notations, particularly around frequency and interval estimation for embeddings (Section 3.1), could be more precise. For instance, the paper uses $I_{ij}$ as both a stochastic occurrence interval and an empirical quantity with little justification for the estimation method used during optimizer steps.
* Insufficient Theoretical Analysis of Method Limitations: The theoretical section does not address possible drawbacks or edge cases (such as whether the adaptive regularization could lead to underfitting for extremely rare or never-seen values), nor the impact of frequency misestimation on convergence or generalization.
* Lack of Comparison to other LLM-based models: There are some new algorithms which are LLM-based recommendation algorithm, and this paper does not consider them.

### Questions
* How Sensitive is the Proposed Method to Hyperparameter $\alpha$ Across Datasets?
Figure 3 provides some evidence, but would the authors provide additional experimental data on extreme values of $\alpha$, and practical guidelines for selection especially in large-scale real-world datasets?
* Handling Never-seen or Extremely Rare Features:
Does the adaptive regularization approach systematically bias against tail classes? How does it perform in cold-start (new feature value) scenarios, and is there risk of underfitting for features with vanishing sample frequency? Can the method adapt dynamically if sparsity levels change after deployment?
* Ablation on Update Interval Estimation:
Would the authors run ablation studies to decouple the effect of interval-based regularization vs. embedding-based regularization overall?
* Comparison with Other Adaptive/Hybrid Regularization Approaches:
How does the proposed method perform compared to thresholding or pruning-based strategies, as well as non-uniform L1/L2 variants found in the sparse modeling literature? Would Table 1/2 results meaningfully shift under such baselines?
* Computational Overhead and Scalability:
Are there notable compute or memory implications to tracking per-embedding update intervals, especially as feature cardinality scales beyond the reported datasets?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper targets a common phenomenon in large-scale CTR/recommender models with highly sparse features: performance is often best after the first epoch, but further training leads to overfitting and degradation. The authors provide an explanation from the perspective of generalization bounds through Rademacher complexity as follows:

* most of the model capacity is concentrated in the huge embedding matrix, and in particular, in the very infrequent IDs’ embedding rows. 
* the embedding rows with very infrequent IDs are updated on a much slower schedule than the dense part / MLP, so their norms can grow uncontrollably and thus inflate the overall capacity. 

Based on this, the authors propose an adaptive regularization scheme that assigns stronger weight decay to infrequent embedding rows . They integrate this row-wise adaptive weight decay into Adam/Adagrad (termed AdamAR/AdagradAR), so that the sparser and less frequently updated a row is, the stronger the regularization it receives. 

Experiments on multiple datasets and CTR architectures show that this approach can mitigate or even eliminate the “multi-epoch performance drop,” while slightly improving AUC.

### Strengths
* Problem importance. The paper tackles an industry-recognized issue in large-scale CTR models—a core component of nearly all advertising systems.
* Theoretical grounding. It links the empirical “train only one epoch” practice to a capacity-control perspective, giving the phenomenon a clearer theoretical justification.
* Practical simplicity. The proposed technique is easy to integrate: it only requires tracking the last valid update step for each embedding row and can be directly plugged into existing optimizers.

### Weaknesses
1. The connection to existing “frequency-aware” or “epoch-level reset” approaches (e.g., MEDA-style methods) is not elaborated systematically.
2. The experiments mainly report global metrics such as AUC/Logloss, but lack fine-grained bucket analyses (by feature frequency, long-tail IDs, cold-start features) to directly demonstrate that “low-frequency rows were actually controlled.”

### Questions
1. Does applying strong L2 to low-frequency rows in adamW reproduces the main gains? If so, that would validate the causal story more strongly.

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
3

### Summary
This paper analyzes the one-epoch overfitting phenomenon in large-scale CTR/CVR models with sparse embeddings. Using Rademacher complexity, the authors show that overfitting arises from unconstrained embedding norm growth for infrequent features. They propose an adaptive regularization method that dynamically adjusts regularization strength by feature update intervals, forming optimizers. Experiments on public and industrial datasets demonstrate consistent performance gains, improved generalization, and easy integration into existing training pipelines.

### Strengths
A substantive assessment of the strengths of the paper, touching on each of the following dimensions: originality, quality, clarity, and significance. We encourage reviewers to be broad in their definitions of originality and significance. For example, originality may arise from a new definition or problem formulation, creative combinations of existing ideas, application to a new domain, or removing limitations from prior results.

1. Sound and well-motivated analysis linking embedding sparsity and overfitting.
2. Adaptive rule is intuitive and easy to implement.
3. Multi-dataset, multi-optimizer validation with consistent gains.
4. Addresses a critical and under-theorized issue in real-world recommender systems.
5.Provides a clear explanation for why rare features require stronger regularization.

### Weaknesses
1. The analysis assumes i.i.d. sampling and bounded feature norms, which may not hold in practice for long-tail industrial data.
2. The derivation of adaptive coefficients relies on approximating feature frequency by occurrence intervals; this estimation may introduce stochastic noise not formally analyzed.
3. The differentiability of \phi(\tau_{ij}) and the KKT-based derivation rest on idealized smoothness assumptions that may not hold in real deep networks.
4. The paper doesn’t rigorously study how the adaptive coefficient \lambda_{ij} affects optimizer dynamics or convergence stability, especially in non-convex regimes.
5. A theoretical guarantee on convergence rate or bias induced by adaptive decay would strengthen the work.
6. Tracking the “last valid update step” for every embedding vector may introduce memory overhead in billion-scale models. The paper does not discuss memory/time complexity or optimization strategies for this in production.
7. While the baselines are relevant, comparisons with more modern adaptive regularization or implicit bias control methods  e.g., SAM, AdaNorm, or adaptive dropout  would enhance credibility.
8. No statistical significance testing is reported for AUC gains, though the improvements are small (≈0.002–0.01).
9. The paper briefly mentions online deployment but does not analyze potential trade-offs e.g., regularization lag under distribution shift, interpretability implications, or fairness across infrequent users/items.

### Questions
Please refer to the weaknesses.

### Soundness
4

### Presentation
3

### Contribution
3
