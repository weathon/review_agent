# When Flatness Does (Not) Guarantee Adversarial Robustness

- Decision: Accept (Poster)
- Scores: 6, 2, 4, 6

## Abstract
Despite their empirical success, neural networks remain vulnerable to small, adversarial perturbations. A longstanding hypothesis suggests that flat minima, regions of low curvature in the loss landscape, offer increased robustness. While intuitive, this connection has remained largely informal and incomplete. By rigorously formalizing the relationship, we show this intuition is only partially correct: flatness implies *local* but not *global* adversarial robustness. To arrive at this result, we first derive a closed-form expression for relative flatness in the penultimate layer, and then show we can use this to constrain the variation of the loss in input space. This allows us to formally analyze the adversarial robustness of the entire network. We then show that to maintain robustness beyond a local neighborhood, the loss needs to curve *sharply* away from the data manifold.
We validate our theoretical predictions empirically across architectures and datasets, uncovering the geometric structure that governs adversarial vulnerability, and linking flatness to model confidence: adversarial examples often lie in large, flat regions where the model is confidently wrong. Our results challenge simplified views of flatness and provide a nuanced understanding of its role in robustness.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper provides a formal analysis of the relationship between flatness in the loss landscape and adversarial robustness in neural networks. While flat minima are often believed to enhance robustness, the authors demonstrate that this connection holds only locally, not globally. They derive a closed-form measure of relative flatness in the penultimate layer and use it to constrain input-space loss variation, enabling a theoretical assessment of network robustness. Empirical results across models and datasets support these findings, showing that adversarial examples often occupy flat regions where models are confidently incorrect.

### Strengths
1. This work presents a framework for analyzing adversarial robustness through relative flatness, a concept introduced in prior studies.
2. While the conclusion that “flatness implies local but not global adversarial robustness” is not surprising, formally establishing this insight contributes meaningfully to the theoretical understanding of adversarial robustness.
3. The finding that adversarial examples often lie in large, flat regions is intriguing.

### Weaknesses
1. The theoretical analysis appears to overlook the generalization from training to unseen test data. Even if the connection between relative flatness and adversarial robustness holds on the training data, it remains unclear how flatness measured on training examples translates to robustness on unseen test inputs.
2. The paper lacks actionable insights for improving adversarial robustness, limiting its practical impact despite its theoretical contributions.

### Questions
1. Given the similarity between the finding that adversarial examples often lie in large flat regions and the observation of a downward trend in the input loss landscape slope (as shown by IG in Fig. 1 of [1]), it would be valuable to explore whether the theoretical framework presented in this work can explain or align with that trend. Establishing such a connection could strengthen the theoretical grounding and unify observations across studies.

2. Since adversarial training is known to significantly enhance adversarial robustness, it is important to examine how the proposed analysis interacts with or adapts to models trained adversarially. Specifically, how does relative flatness behave under adversarial training, and does the established relationship between flatness and local/global robustness still hold? Addressing this would clarify the scope and applicability of the theoretical insights.

[1] Li, Lin, and Michael Spratling. "Understanding and combating robust overfitting via input loss landscape analysis and regularization." Pattern recognition 136 (2023): 109229.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper re-assesses prior claims about the relationship between the flatness of the loss landscape and adversarial robustness. Consistent with previous claims the paper finds that flatness enhances robustness around particular training samples. However, in contrast to previous claims the current paper suggests that adversarial examples can also lie in flat regions. Hence, increasing flatness does not necessarily improve robustness.

### Strengths
The paper is well structured and generally clearly written.

The paper combines both theoretical and empirical research.

### Weaknesses
All the analysis in the paper is based on characterizing successful adversarial attacks through changes in loss rather than changes in predicted label (section 3.2). Specifically, it is claimed that an adversarial perturbation will increases the loss beyond a threshold. However, this is not true for cross-entropy loss. For example, consider a neural network that performs a 3-way classification task. If the true label of a sample is 0 and this network outputs logits [0.6, 0.1, 0.1], then the sample is classified correctly and the cross-entropy loss is 0.7944. If the sample is perturbed in such a way as to produce logits [0.6, 0.7, -5], then the predicted class label is wrong (i.e. the perturbation constitutes a successful adversarial attack), yet the loss decreases to 0.7462. The reverse is also the case: a large increase in loss does not necessarily indicate a successful attack. For example, if the same sample was perturbed so that the network produced logits [0.6, 0.5, 0.5], then the attack would be unsuccessful, but the loss would be increased to 1.0331.

### Questions
Given the lack of correspondence between classification accuracy and cross-entropy loss described above, which of the claims/results in the paper are still valid?

### Soundness
1

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
3

### Summary
This paper investigates the long-standing hypothesis that flat minima in the loss landscape imply increased adversarial robustness. Through a rigorous theoretical formulation, the authors show that flatness guarantees only local robustness rather than global robustness. Empirical evaluations across different architectures further support this finding, revealing that adversarial examples often reside in large, flat regions of the loss landscape.

### Strengths
The authors rigorously extend Petzka et al. (2021)’s notion of relative flatness to the setting of adversarial robustness, providing formal derivations that are technically sound and mathematically non-trivial. The paper presents elegant formulations that establish a clear analytical link between adversarial robustness and the Hessian of the loss.

### Weaknesses
1. While overall well-written, the paper is dense and somewhat derivative-heavy. Key ideas, such as the geometric mapping between feature-space and input-space curvature, could be illustrated more clearly using diagrams or intuitive explanations. A more intuitive introduction or motivating example would help readers better grasp the high-level ideas before delving into the detailed derivations. In particular, the terms “relative flatness” and “relative sharpness” are used somewhat inconsistently, which may confuse readers.

2. The empirical evaluation is somewhat limited. While the experiments illustrate the theoretical claims qualitatively, they rely on relatively weak PGD attacks (PGD-$l_2$ with $\epsilon = 0.025$ in page 8) and lack comparisons with adversarially trained baselines (e.g., TRADES, AWP, or SAM-trained models). As a result, it remains unclear whether the observed relationships between sharpness and robustness persist under stronger or more realistic adversarial conditions.

### Questions
1. I think the first paragraph on page 1 could be divided into multiple shorter paragraphs to improve clarity. Also, in Figure 1 (mentioned on the first page), the notation $\phi^{-1}$ is a bit confusing — it’s not immediately clear how the authors relate the feature space to the input space. It would be better to explain this mapping more intuitively in the introduction.

2. On page 2, lines 90–93, the first and second listed contributions appear somewhat redundant. In the first contribution, you state that you theoretically establish the link between flatness and adversarial robustness, while in the second, you restate this point with more detail, specifying that the link is derived through the penultimate layer. I think these two points should be combined into a single, unified contribution for clarity.

3. On page 4, in Definition 4 (Loss-change adversarial example), you define adversarial examples as those that increase the loss by more than $\epsilon$. Then, in line 183, you state that “by using a conservative $\epsilon > \log(k)$ for cross-entropy loss, we can ensure a prediction flip.” However, in practice, when the number of classes $k$ is large, this threshold can correspond to very high loss values. Such cases may represent “strong” adversarial examples (in terms of loss magnitude). Moreover, the reverse implication is not guaranteed, when $l(f(x), y)$ is not close to zero, a prediction flip might occur with much smaller loss changes. Therefore, the relationship you establish between loss increase and prediction change may not hold universally. Could you provide more discussion or clarification on this limitation?

4. On page 7, line 363, and page 8, line 378, there are two separate “Setup” paragraphs in Section 6 From Theory to Practice—one ending with a full stop and another continuing without. It would be clearer to merge them into a single, continuous setup description, as the current separation is somewhat confusing.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper challenges the theoretical hypothesis that flatness increases the robustness of neural networks and discovered that flatness implies local but not global adversarial robustness. Flatness tends to emerge in regions where the model is highly confident.

### Strengths
1. The Uncanny Valley analysis helps explain why adversarial examples can appear deceptively safe, because they often lie in flat, vast, high-confidence region.
2. The paper is generally well-written and well-presented.
3. The relation between relative sharpness and adversarial robustness is clearly explained, and a precise robustness bound is given

### Weaknesses
1. A large part of the paper is used to justify that relative flatness at the penultimate layer is sufficient, which seems to have been stated in Petzka (2021).
2. The entire analysis was built on the local flatness at the penultimate layer; did the authors rule out the effect of the geometry of the input space on the adversarial robustness?

### Questions
1. Does the metric relative flatness possibly ignore the correlated curvature directions across layers?

### Soundness
3

### Presentation
3

### Contribution
2
