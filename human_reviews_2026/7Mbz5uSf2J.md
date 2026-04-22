# Decoupling Dynamical Richness from Representation Learning: Towards Practical Measurement

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 4, 4, 8

## Abstract
Dynamic feature transformation (the rich regime) does not always align with predictive performance (better representation), yet accuracy is often used as a proxy for richness, limiting analysis of their relationship. We propose a computationally efficient, performance-independent metric of richness grounded in the low-rank bias of rich dynamics, which recovers neural collapse as a special case. The metric is empirically more stable than existing alternatives and captures known lazy-to-rich transitions (e.g., grokking) without relying on accuracy. We further use it to examine how training factors (e.g., learning rate) relate to richness, confirming recognized assumptions and highlighting new observations (e.g., batch normalization promote rich dynamics). An eigendecomposition-based visualization is also introduced to support interpretability, together providing a diagnostic tool for studying the relationship between training factors, dynamics, and representations.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces a new metric for quantifying rich (feature) learning. The authors motivate this by distinguishing between two perspectives on feature learning: a representation perspective (focus on usefulness of features for performance/generalization) and a dynamics perspective (focus on the transformation of features). They show that although it is often associated with representational usefulness, rich learning doesn’t necessarily lead to generalization and instead reflects an inductive bias towards certain solutions. Thus, the paper focuses on dynamic feature learning, introducing a computationally-efficient and performance-independent metric that effectively compares the network activations before and after the last layer. They show that this metric generalizes neural collapse, a phenomenon associated with rich learning. By comparing to prior measures of rich dynamics, they illustrate special cases where their metric captures dynamical richness, while others do not. They also conduct experiments across several different models and datasets, demonstrating how different training conditions affect performance and richness separately. Finally, the authors provide complementary visualization methods, make it easier to see the low-rank feature bias of rich dynamics.

### Strengths
The presentation and clarity of the paper is excellent. It is well-written, clear, and accessible. The paper is well-motivated, with a comprehensive introduction and background. 

The soundness of the paper is also excellent. The authors take care to support each of their claims with ample evidence and solid experiments/methodology. The paper identifies an important distinction in perspectives on feature learning (dynamics- versus representation-focused) that challenges implicit assumptions about rich learning necessarily leading to improved performance/generalization and provides experiments to support this. The metric introduced has good potential for studying feature learning, as it is efficient to compute and captures rich dynamics in settings where prior metrics fail. The authors provide well-thought experiments to demonstrate the utility of their metric and support their central claims, showing that it captures grokking, neural collapse, and other known feature learning phenomenon. They also provide helpful visualization methods, which they show the utility of in interpreting different network behavior. The paper offers a good contribution to the ICLR community and I recommend it for acceptance. It was a pleasure to read and I thank the authors for their solid work.

### Weaknesses
To clarify, I think these points are relatively specific (possibly only applying to certain edge cases) and that the work as a whole still stands well despite some potential limitations.

One potential weakness of the metric introduced is that it’s limited in its generality (to orthogonal and isotropic target functions). However, this is properly acknowledged by the authors and the metric still captures many classification tasks. 

As the authors also state, their metric depends on a comparison between the last two layers. Thus, two networks with different feature learning behavior in earlier layers may appear identical if their final two layers are similar. I’m wondering if there would be any principled way to applying the same approach to other layers in the network? And are there concrete settings where the comparison between the last two layers is a major limiting assumption?

As far as I understand, the metric is based on the low-rank bias of rich dynamics. I know that this might be rare in practice, but would the metric still work if the target function is full-rank? Is the assumption of low-rank dynamics potentially limiting?

The authors state that in Figure 5 “feature quality correlates with feature intensity during training, with larger features improving faster… the correlation between quality and intensity during training has not been previously observed or studied.” Isn’t this expected, as it’s known that large features (singular values) are learned first (Saxe et al., 2014)? Or, do you mean that eigenvector alignment is occurring simultaneously with singular vector scaling? In Atanasov et al., 2022, they show that for anisotropic data, “the NTK must necessarily change its eigenvectors when the loss is significantly decreasing, destroying the silent alignment phenomenon.” I’m a bit skeptical of the statement that this correlation has not been observed or studied.

### Questions
1. Would the metric still work if the target function is full-rank?
2. Would be any principled way to applying the same approach to other layers in the network? And are there concrete settings where the comparison between the last two layers is a limiting assumption?

Suggestion: there are a few typos but nothing major

### Soundness
4

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
The paper proposes a performance-independent metric for dynamical richness (DLR) that compares the feature kernel built from penultimate-layer activations with a “minimum projection” operator determined by the network’s learned function. Intuitively, truly rich dynamics should learn only the minimal features needed to span the learned function space; DLR quantifies the gap to this ideal and is normalized in [0,1] (lower is richer). The authors show that when the feature operator attains this minimum form, neural collapse conditions follow as a special case, linking the metric to established phenomena. Empirically, DLR is lightweight to compute and tracks lazy-to-rich transitions across MLPs/CNNs/Transformers, and an eigendecomposition-based visualization complements the scalar score.

### Strengths
1. Originality. Recasts richness as low-rank alignment between features and the learned function via a principled MP-operator; this is a fresh angle relative to NTK-deviation or label-based collapse metrics. The reduction to neural collapse provides a clean conceptual bridge.
2. Quality. The metric is simple, normalized, and computationally cheap, enabling use on standard vision models. Comparative experiments are thoughtful and reveal expected patterns.
3. Clarity. The paper is well structured and the three-panel eigendecomposition plots (quality/utilization/eigenvalues) are helpful.
4. Significance. A lightweight, performance-independent diagnostic for dynamics could become a standard tool, much as CKA did for representational comparisons.

### Weaknesses
1. DLR inspects only the final-layer features; rich dynamics might manifest earlier and be attenuated by a constrained head. Consider a hierarchical variant (layer-wise DLR or block-DLR) and show whether conclusions persist will be helpful.
2. The technical background is not much sufficient for readers to capture the essence. More straightforward interpretation and introduction are helpful.
3. The MP-operator and some guarantees rely on orthogonal/isotropic targets and supervised, one-hot settings; this narrows immediate applicability (e.g., class imbalance, multilabel, regression, self-supervised).
4. Broader settings (imbalance/SSL), deeper ablations (layer-wise, NTK-feasible baselines), and stronger analysis of assumptions would raise confidence and impact.
5. Writing/format polish. Minor issues distract: e.g., “batch nomralization” typo in Table 3; occasional spacing/notation inconsistencies around figures/equations; some insufficient definitions/clarifications on heavy-weight mathematical notations..

### Questions
1. Can DLR be generalized to non-isotropic/imbalanced tasks (e.g., via class-reweighted inner products or whitening), and does the neural-collapse link still hold?
2. What happens if you compute DLR per layer (or per block) and aggregate? This could localize where richness emerges and inform architecture design.
3. Can you theoretically connect decreasing DLR with transition conditions in lazy-to-rich analyses/grokking, beyond empirical alignment?
4. Why CKA over alternatives (e.g., centered HSIC variants)? Any cases where CKA’s centering removes signal that matters for DLR?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes DLR (Dynamical Low-Rank measure), a computationally efficient and performance-independent metric for quantifying dynamical richness in neural networks. The metric is grounded in the low-rank bias characteristic of rich dynamics and compares activations before and after the last layer via a functional kernel operator. The authors show that DLR reduces to neural collapse as a special case and empirically validate that it captures known lazy-to-rich transitions (e.g., grokking) without relying on accuracy. They further introduce an eigendecomposition-based visualization tool to enhance interpretability.

### Strengths
- Important problem: Decoupling dynamical richness from representation quality addresses a fundamental issue in understanding neural network training. The observation that rich dynamics ≠ better generalization (Figure 1) is compelling.
- Computationally efficient: The O(p²C) complexity is a significant improvement over NTK-based methods, making the metric practical for modern architectures.
- Strong theoretical grounding: The connection to neural collapse (Propositions 1 and 2) provides solid theoretical foundation, while extending applicability beyond classification (e.g., regression with scalar output).
- Performance-independent: Unlike existing metrics (Sinit, parameter norm, NC1), DLR doesn't rely on accuracy, initial kernel, or class labels, making it more robust (Tables 1 and 2).

### Weaknesses
- Limited scope: The current formulation only applies to orthogonal and isotropic target functions. While this covers many classification tasks, the restriction is significant. The authors acknowledge this but don't provide a clear path to generalization.
- Empirical validation: While the authors demonstrate DLR's utility across diverse settings (grokking, learning rate variations, weight decay, batch norm) the experiments conducted are of small scale and somewhat artificial. It would be interesting to see this applied to more modern scenarios, i.e. bigger models and more difficult datasets.
- Last-layer focus: By only examining last-layer features, the metric misses dynamics in earlier layers. While this is a trade-off for efficiency, it's unclear how much information is lost.
- Limited theoretical analysis: Beyond the connection to neural collapse, there's limited theoretical characterization of when DLR accurately reflects "richness" or what DLR values imply about learning.
- Comparisons could be broader: NTK-based measures are set aside due to computational cost; although justified, a scaled-down NTK comparison on smaller models (which the authors already use) would strengthen claims that DLR is a better proxy rather than merely cheaper.
- Clarity: While the method is well motivated theoretically and relative to prior work, the paper is occasionally difficult to follow. The bra-ket notation and operator formalism, while mathematically precise, may limit accessibility for a broader audience. The empirical approximations (Appendix E) are much clearer and could be introduced earlier to improve intuition. At times, the paper presents symbols and equations with insufficient context or motivation. Furthermore, the term dynamical richness is used somewhat loosely throughout; a dedicated subsection defining and contrasting rich vs lazy regimes would help.
- Unexplored temporal dynamics: The paper treats DLR at convergence or snapshot points, but since it is dynamical, a time-series analysis (e.g., how DLR evolves over training) would provide stronger evidence of what “rich” dynamics actually look like.

### Questions
- Typos: 049 "a dynamical richness (metric?) that", 373 "Our visualizes (visualizations)"
- How does the metric behave with different loss functions beyond MSE? The cross-entropy results are mentioned but not thoroughly analyzed.
- Metric sensitivity/ablation: More detail on estimation stability (sample size n for Nyström, dependence on width p, class count C) and hyperparameter sensitivity would help practitioners choose sampling parameters confidently.
- The batch normalization finding (last row of Table 3) is correlational. The paper doesn't establish that batch norm causes rich dynamics or explain the mechanism.
- The paper doesn't provide clear guidance on what intermediate DLR values mean.
- Interpretability link not quantified: The eigendecomposition visualization is qualitative; quantitative metrics (e.g., subspace alignment or effective rank) could have strengthened claims of interpretability.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
In this paper, the authors develop the ***low-rank measure*** \$\mathcal{D}_{LR}\$ to measure the richness of the representations learned by a neural network, independently of its performance. Using it, and comparing it with the metrics from the existing literature, they can efficiently measure whether a model is in the *lazy* or *rich* regime.

The authors first introduce \$\mathcal{D}_{LR}\$ theoretically, basing it on the concept of low-rank bias where in rich dynamics, the rank of the features learned by the model before the final linear layer should be similar to the dimensionality of the output, indicating well-separated classes. The metric itself is simple, as it is based on the CKA between the representations after the penultimate and the ultimate layer. It is therefore also inexpensive to compute.

They then compare their metric with other ones from the rich dynamics literature. Across two different test cases, they demonstrate it better illustrates rich-vs-lazy dynamics than other metrics. They further test their metric on realistic use-cases, showing with different examples how it represents richness, and where it is related (or not) to model performance.

The last part of the paper focuses on visualisations to explain shifts in the learning dynamics of neural networks. The authors separate three different metrics through eigendecomposition of kernel: cumulative quality, cumulative utilisation and relative eigenvalue. They demonstrate how they relate to rich dynamics.

### Strengths
I think that this paper is a strong paper, which proposes a new metric for dynamic richness and demonstrates how it works better than other examples from recent literature.

S1: The *low-rank metric* is simple to understand and inexpensive to compute. The authors give good intuition on how it works, why it works, and bounds on computation complexity.

S2: Theoretical foundations for the *low-rank metric* look solid. Although the authors make some assumptions to make calculations tractable, formulations are very general and encompass a large portion of neural networks.

S3: Empirical evidence covers the validity of the metric, comparison to existing other metrics from the literature, and practical use cases. The authors further provide statistical significance results. I think it strongly supports the main claims of the paper. Furthermore, the authors propose visualisation methods for increased interpretability of their results.

S4: The literature review seems extensive, and most claims are backed through proofs or citations. This work is well contextualised within the existing literature.

S5: I strongly commend the authors’ work on making their paper clear and easy to read. Writing and presentation are of high quality. It allows readers to more easily understand the theoretical framework around the development of the *low-rank metric*.

### Weaknesses
I have not found major weaknesses in this paper. The following points are either minor or nitpicks.

W1: I feel like the authors should find a clearer name and/or an acronym to designate the *low-rank metric*, which is often designated as “the metric” or as its mathematical notation \$\mathcal{D}_{LR}\$ throughout the paper. Such a name/acronym would make it easier to designate and reference in text, discussions and future work that will rely on it.

### Questions
Q1: It is not immediately clear to me how exactly the metrics introduced in Section 5 relate to the *low-rank metric*. They are derived from \$\mathcal{T}\$’s eigendecomposition and the learned/target functions, but I have difficulties directly linking them to \$\mathcal{D}_{LR}\$.

Q2: The authors mention two main limitations: focus on last-layer dynamics, and validity constrained to orthogonal and isotropic target functions. Regarding the second point: the authors mentions that “while this covers most classification tasks, a more general setup would be preferable”. Could the authors please explain in what cases their method would not work, and/or would not be theoretically justified, and give some examples? Also, are there practical scenarios where this limitation could lead to misleading conclusions if the metric is applied without caution? If relevant, including such details in the paper could help practitioners understand when to avoid or adapt the use of this metric in their work.

### Soundness
4

### Presentation
4

### Contribution
4
