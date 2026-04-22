# Feature compression is the root cause of adversarial fragility in neural networks

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
In this paper, we uniquely study the adversarial robustness of deep neural networks (NN) for classification tasks against that of optimal classifiers.  We look at the smallest magnitude of possible additive perturbations that can change a classifier's output.  We provide a matrix-theoretic explanation of the adversarial fragility of deep neural networks for classification. In particular, our theoretical results show that a neural network's adversarial robustness can degrade as the input dimension $d$ increases.  Analytically, we show that neural networks' adversarial robustness can be only $1/\sqrt{d}$ of the best possible adversarial robustness of optimal classifiers. Our theories match remarkably well with numerical experiments of practically trained NN, including NN for ImageNet images. The matrix-theoretic explanation is consistent with an earlier information-theoretic feature-compression-based explanation for the adversarial fragility of neural networks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper investigates the root cause of adversarial fragility in neural networks and proposes that feature compression, which is the tendency of neural networks to depend on a reduced subset of features, fundamentally limits their robustness. Through a matric theoretic and geometric analysis, the authors show that neural networks are inherently more fragile than optimal classifiers, especially as input dimensions increase. They extend their theory from linear to multilayer nonlinear models and verify the predictions with synthetic and small scale real data experiments. The results align well with the theoretical framework, offering a clear and mathematically grounded explanation for why adversarial perturbations can easily fool deep networks.

### Strengths
The paper introduces a matrix-theoretic perspective on adversarial fragility, offering a rigorous mathematical foundation to the widely discussed but poorly formalized “feature compression hypothesis.” The authors connect earlier information theoretic intuition with precise linear algebraic analysis. The proofs are carefully structured, and theoretical predictions are supported by corresponding numerical validations. The idea is well conceived.

### Weaknesses
· Model Selection and Scope Limitations: The theoretical analysis focuses primarily on linear or shallow architectures with Gaussian assumptions. Although the paper claims to extend results to nonlinear deep networks, the derivations rely on strong linearity or independence assumptions that may not hold for modern architectures (e.g., transformers or convolutional models). The leap from linear matrix theory to practical deep models remains under-justified.
· Limited Robustness Evaluation: The experiments demonstrate numerical consistency but do not include robustness comparisons under standard adversarial benchmarks (e.g., FGSM, PGD on CIFAR-10 or ImageNet). Without such empirical robustness metrics, it is difficult to assess how well the proposed “feature compression factor” explains real-world adversarial behavior beyond toy examples.
· Dataset Diversity and Representativeness: Most numerical results are based on synthetic Gaussian data or simple setups , with limited exploration of diverse or structured datasets. The extension to MNIST or ImageNet is only briefly mentioned and lacks quantitative reporting. Consequently, it remains unclear whether the compression phenomenon generalizes across modalities and realistic.
· Insufficient Discussion on Defensive Implications: While the paper identifies feature compression as the root cause of fragility, it does not provide actionable strategies for mitigating it. The lack of defense-oriented discussion (e.g., modifying training objectives, architectural choices, or regularization methods to counter compression) weakens the practical contribution.
· Assumption Verifiability and Practical Relevance: Many theoretical conclusions depend on strong statistical independence or idealized random matrix properties that are difficult to verify in trained neural networks. This gap between the theoretical setup and actual deep learning practice raises questions about how well the results explain real-world adversarial behavior.

### Questions
1. Generality of the Theory: Your analysis is derived primarily under Gaussian and linear assumptions. How sensitive are the theoretical conclusions to deviations from these assumptions — for example, when dealing with real-world, structured, or non-Gaussian data distributions?
2. Feature Compression Measurement: In practice, how can one quantitatively measure “feature compression” in a trained deep neural network? Are there empirical indicators or diagnostics that could verify the compression–fragility relationship beyond the controlled synthetic setup?
3. Extension to Modern Architectures: Have you tested whether the same compression–fragility correlation holds for modern architectures such as transformers or convolutional networks trained on large-scale datasets? If not, what challenges do you foresee in extending the theoretical framework to them?

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
3

### Summary
This paper investigates the adversarial fragility of neural networks from a matrix-theoretic perspective. By analyzing the adversarial robustness of neural networks, the authors conclude that their robustness can be only $1/ \sqrt{d}$ of the best possible robustness achievable by optimal classifiers. The theoretical results are further supported by numerical experiments that empirically validate the proposed analysis.

### Strengths
The authors provide a thorough and mathematically rigorous theoretical analysis of the adversarial robustness of neural networks. The paper offers clear derivations grounded in matrix theory and connects them to the concept of feature compression, presenting a coherent framework that links theoretical insights with empirical observations.

### Weaknesses
1. You provide a thorough theoretical analysis, but there is a lack of clear discussion and verification of the underlying assumptions. This omission affects the credibility and correctness of your conclusions.

2. Regarding the experiments, on the last page you mention "Feature compression on deep non-linear networks trained on MNIST and ImageNet in the supplemental materials". I believe these real-world experiments should be included in the main body of the paper rather than placed in the appendix. They are crucial for demonstrating that your findings are not only theoretically valid under idealized assumptions but also meaningful and applicable in practical settings.

### Questions
1. In *Theorem 1 (page 3)*, you assume a hard label formulation for the neural network such that $f_j(x_i) = 1$ when $j = i$ and $f_j(x_i) = 0$ otherwise. However, in line 150 on the same page, you define $f_i(x) = w_i^{T}\sigma(H_1 x + \delta_1)$. I am concerned about the validity of this assumption, even though *Remark 1* mentions that the constant “1” can be replaced by any positive number. Specifically, what would be the impact if we instead adopt the typical classifier formulation, where $f_j(x_i) = v_j$ and $\arg\max_j v_j = i$? Would the main result of Theorem 1 still hold under this more general setting? I think a more detailed discussion of how this relaxation affects the theoretical conclusion would strengthen the paper.


2. I am a bit confused about the assumption regarding the training dataset described on 
page 3, lines 143--156 (the second paragraph of Section 2). It seems that your primary theorem is based on training data points $(x_i, i)$, where the input dimension, the number of labels, and the number of data points are all equal ($d$). I think this setting should be clarified further. In particular, why is such an assumption reasonable, or do your theoretical derivations only work on these assumptions?  Although you mention that the number of training data points can be extended to be exponential in the input dimension $d$, it would still strengthen the paper to elaborate on this setting in more detail, for example, by citing previous works that adopt similar assumptions and by discussing why this exponential scaling is theoretically meaningful or realistic in your context.

3. In page 5, line 220, you mention that, by random matrix theory, the eigenvalue distribution follows certain properties, which are then used to derive concentration results. However, as far as I know, such claims from random matrix theory rely critically on the assumption that the data entries are i.i.d. Gaussian random variables. If this assumption changes, for example, if $x$ is not Gaussian or its components are not independent, the results may no longer hold, and, to the best of my knowledge, the corresponding theoretical guarantees remain unknown. If relevant results exist, please provide appropriate citations. Otherwise, this reliance on strong i.i.d. Gaussian assumptions appears to substantially limit the applicability of your theory to real-world data.

### Soundness
3

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
4

### Summary
This paper proposes a framework to explain the long-standing mystery of adversarial fragility in neural networks.
The central claim is that feature compression—where neural networks effectively rely on a small subset or projection of the input feature space for classification—is the fundamental cause of adversarial vulnerability.

Main ideas and results:

Formal comparison with optimal classifiers:
The authors analytically show that neural networks’ adversarial robustness can be as small as O (1/sqrt(d)) of the optimal classifier’s robustness, where d is the input dimension.

Derivations:
Using QR decompositions of weight matrices, they prove that a successful adversarial perturbation can be achieved by modifying only the compressed subspace of features (Theorem 1–4).

For both linear and multi-layer networks, they characterize the small perturbation magnitude (constant-scale) that flips classification, while optimal classifiers need perturbations of O(1/sqrt(d)).

Generalization to nonlinear networks:
The paper extends to multi-layer nonlinear networks (Theorem 6), showing that adversarial perturbations are aligned with the projection of the true discriminative direction onto the gradients’ span — i.e., the compressed subspace.

### Strengths
Conceptual originality: This work claims to move beyond heuristic explanations (e.g., linearity or gradient magnitude) by introducing a precise matrix–QR decomposition view of compression in feature space. This work claims to link geometric and information-theoretic interpretations into a unified mathematical framework.

Theoretical originality: The derivations seem to bridge random matrix theory with robustness analysis — a fresh angle rarely seen in adversarial ML work. The paper claims novel results on the distribution of QR decompositions of products of Gaussian matrices (Lemma 7).

Intuition: Figure 2 (p. 16) provides an intuitive geometric picture—attacks succeed by perturbing along compressed feature directions. The “compression ratio = robustness ratio” insight is crisp and testable. I find the intuition interesting.

### Weaknesses
**Clarity of formal statements:** The theoretical statements are very dense and unclear. I will list below some of the things that I find dubious/unclear. If you are able to clarify, I will be willing to increase my score. 

l.143-146: the input dimension is $d$ and the number of training points is also $d$ ? That's an very unsual assumption. And also there are $d$ class labels ? I don't understand. 

l.153: does that mean that all the results hold in the limit d to infinity ? so for example in Theorem 1, the perturbation $e$ is a vector whose size goes to infinity? Same question for Theorem 3, the input dimension goes to infinity ? what about the widths of the network ? I'm surprised you can apply random matrix theory without taking all dimensions to infinity. It's not clear what is infinite and what is not, making it difficult to apprehend what is the precise mathematical claim.

Theorem 6:  "let the closest point in that class to x be denoted by x + x_i", so you are introducing a new notation "+"? In that case, what does "x + \epsilon x_1" mean ? Furthermore, the gradient $\nabla f_i (x)$, is it a gradient with respect to the parameters, with respect to the input ?

l.336-339, the sentence makes no sense to me.

**Scope of theoretical assumptions:** The main theorems assume Gaussian inputs and idealized linear or piecewise-linear models. It remains unclear how much these results extend to more realistic training dynamics, nonlinearities, and non-Gaussian natural data. For example, the assumption of exponential number of data points (Theorem 5) is not realistic at all, and taking such assumption does not seem to make much sense. Please explain where this is coming from. 

**Comparative analysis with existing theories:** The paper differentiates itself from “non-robust feature” and “curvature” explanations, but the relationship with these existing theory is not clear. For example, can compression be quantified independently of gradient alignment? Do non-robust features and compressed features coincide ?

### Questions
1) Can “feature compression” be quantitatively measured for arbitrary networks? e.g., via singular value decay or layerwise Jacobian rank?

2) Does adversarial training reduce compression (and thus improve robustness) in measurable ways?

3) How sensitive are results to activation type (ReLU vs. tanh) or normalization layers?

4) In nonlinear models, how does compression evolve layer-by-layer during training?

5) Can you comment on how this framework could guide architectural design and optimization to avoid compressive fragility?


Overall I think the idea of the paper is interesting, but the theoretical statements need clarification.

### Soundness
2

### Presentation
1

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
This paper introduces a theoretical framework that attributes the adversarial fragility of neural networks to the phenomenon of feature compression. The authors analyze the effect of dimensionality reduction within network representations and show that such compression significantly reduces adversarial robustness compared to an ideal optimal classifier. The study combines formal derivations with controlled numerical experiments and demonstrates that the degree of compression observed in trained models aligns with theoretical predictions.

### Strengths
1. Strong theoretical foundation. The paper presents a clear and rigorous mathematical framework that links the structure of neural networks with their lack of robustness under adversarial perturbations.

2. Novel conceptual insight. The feature compression hypothesis offers a fresh and intuitive way to interpret why modern neural networks are vulnerable, distinguishing itself from gradient-based or geometric explanations.

3. Internal consistency. The theorems and conclusions are internally consistent and appear sound within the given assumptions.

4. Empirical alignment. The observed experimental behavior supports the proposed theory, indicating the analysis captures an essential property of neural networks.

### Weaknesses
1. Idealized assumptions. Most theoretical results depend on simplified settings such as Gaussian input data and independent linear layers, which restrict their direct applicability to modern architectures like convolutional networks or transformers.

2. Limited experiments. The experimental validation focuses on synthetic or simplified datasets. The lack of large-scale experiments on realistic benchmarks makes it difficult to assess the practical relevance of the conclusions.

3. Incomplete practical implications. While the theory identifies the cause of fragility, it does not provide clear guidance for designing more robust models or training procedures.

4. Presentation density. Some proofs and derivations are overly long, which may distract from the main conceptual contributions. Streamlining these sections would improve readability.

### Questions
1. Idealized assumptions. Most theoretical results depend on simplified settings such as Gaussian input data and independent linear layers, which restrict their direct applicability to modern architectures like convolutional networks or transformers.

2. Limited experiments. The experimental validation focuses on synthetic or simplified datasets. The lack of large-scale experiments on realistic benchmarks makes it difficult to assess the practical relevance of the conclusions.

3. Incomplete practical implications. While the theory identifies the cause of fragility, it does not provide clear guidance for designing more robust models or training procedures.

### Soundness
3

### Presentation
3

### Contribution
3
