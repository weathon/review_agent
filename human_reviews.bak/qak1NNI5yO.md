# The Uncanny Valley: Exploring Adversarial Robustness from a Flatness Perspective

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 8, 3

## Abstract
Flatness of the loss surface not only correlates positively with generalization, but is also related to adversarial
robustness, since perturbations of inputs relate non-linearly to perturbations of weights. In this paper, we empirically
analyze the relation between adversarial examples and relative flatness with respect to the parameters of one layer.
We observe a peculiar property of adversarial examples in the context of relative flatness: during an iterative first-order
white-box attack, the flatness of the loss surface measured around the adversarial example *first* becomes sharper
until the label is flipped, but if we keep the attack running, it runs into a flat *uncanny valley* where the label remains
flipped. In extensive experiments, we observe this phenomenon across various model architectures and datasets, 
even for adversarially trained models. Our results also extend to large language models (LLMs), but due to the discrete
nature of the input space and comparatively weak attacks, adversarial examples rarely reach truly flat regions. Most
importantly, this phenomenon shows that flatness alone cannot explain adversarial robustness unless we can also
guarantee the behavior of the function around the examples. We therefore theoretically connect relative flatness to
adversarial robustness by bounding the third derivative of the loss surface, underlining the need for flatness in
combination with a low global Lipschitz constant for a robust model.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper reports an ``uncanny valley'' phenomenon and provides a theoretical connection between relative flatness and adversarial robustness.

### Strengths
This paper revealed an interesting phenomenon: the relative flatness first goes up and then down as the adversarial attack progresses.

### Weaknesses
1. The existing analysis and discussion are insufficient to demonstrate that the phenomenon observed in the paper is significant and well-motivated. Specifically, I have the following concerns:

     (1.a) The summary of conclusions in the existing literature on the relationship between landscape flatness and adversarial robustness is vague. Therefore, I have not established any intuition about this problem by reading the background part.

     (1.b) The author claimed multiple times that the valley is counterintuitive, which is very confusing to me. My questions are, intuitively, why adversarial examples should not lie in a flat region, and why the trend of flatness first increasing and then decreasing is unusual. 
 
     (1.c) What are the takeaways from the experiments on adversarially trained models?


2.  The theoretical part is not well connected to the empirical observation and is not well explained.

     (2.a) The motivation of the theoretical part (Lines 406-412) is puzzling, especially the ``counterintuitive behavior''.

     (2.b) The theoretical results involve the Lipschitz condition concerning input, which is directly related to the adversarial robustness of the feature extractor. 

     (2.c) The flatness analysis based on the penultimate layer is rather limited.

### Questions
1. If standard adversarial training is replaced with flatness-related adversarial training methods, such as Wu et al., 2020, how would the phenomena in Figures 4 and 5 change?


2.  Regarding the robust generalization problem, should we consider the flatness of the adversarial loss function, which refers to the maximal standard loss within a neighborhood?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper investigates the vulnerability of deep learning models to adversarial attacks by examining the connection between adversarial examples and the flatness of the loss landscape with respect to the model's parameters. To assess the flatness, the authors use a metric termed "relative sharpness." Through empirical observations, the paper identifies a pattern that the relative sharpness values tends to be lower ---- suggesting the loss landscape is flatter ----  when it is measured on perturbed examples generated during later iterations of adversarial attacks. This flat landscape is named uncanny valley in the paper. The paper also presents a theoretical analysis that links the relative sharpness metric to the model's robustness against adversarial perturbations.

### Strengths
The paper reveals an interesting phenomenon that the model's loss landscape w.r.t its parameters becomes progressively flatter when measured on "stronger" adversarial examples—those generated during the later iterations of an adversarial attack.

This potentially brings valuable insights for understanding the vulnerability of DNNs to adversarial perturbations.

### Weaknesses
Although the theoretical analysis links the relative sharpness measure, $\kappa_{Tr}^{\phi}$ , with the robustness of the model, it does not directly explain the emergence of the uncanny valley phenomenon. The theoretical results in Proposition 5 and Corollary 6 indicate that a model with a flatter loss landscape would generally be more robust, which does not have an explicit connection with the uncanny valley phenomenon. Also the relative sharpness measure $\kappa_{Tr}^{\phi}$ in Proposition 5 and corrolary 6 is considered to be computed on unperturbed examples whereas the experiments apply it to adversarial examples.



The theoretical results lack experimental validation. In line 484, the paper asserts that a model with smaller relative sharpness will achieve both good generalization and adversarial robustness. However, this claim requires experimental verification to substantiate it.



I also have concerns regarding Definition 2 as proposed in this paper. In Definition 2, the notion of adversarial examples depends on the choice of the loss function, which may be arbitrary and potentially unrelated to the classifier $f$. This construction could be misleading, as an example $x$ that alters the output of $f$  would be considered adversarial under Definition 1 but be treated as non-adversarial under Definition 2 if $l$ is chosen as a constant function or as a function with a very small Lipschitz constant. Could you clarify why Definition 2 offers an improved framework for defining adversarial examples compared to Definition 1, as established in prior work?



Multiple statements and claims in the paper are unclear and lack theoretical support. For example:

- Line 053, " a vicinity of such an adversarial sample will be filled with similar adversarial examples". Could you clarify what is meant by "similar adversarial examples" in this context?
- Line 170, "adversarial examples are a consequence of non-smooth directions". What is "non-smooth direction"?
- Line 256, "as expected the loss and sharpness increase as the attack progresses". Why is this increase expected, and on what basis?
- Line 261, "meaning there are adversarial examples in a flat region for good performing". What does "good performing" refer to?
- Line 266, "Thus, the uncanny valleys truly extend away from the original example, supporting the intuition that adversarial examples live in entire subspaces of the input space." Could you provide a more detailed explanation of the causal reasoning behind this statement?

### Questions
See weaknesses.

### Soundness
2

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
3

### Summary
This paper empirically analyzed the behavior of the loss function with respect to the parameters and adversarial examples. The loss surface initially becomes sharper as the attack progresses and the prediction is flipped. If the attack continues, the adversarial example often moves into a flat region (called an uncanny valley in the paper). This phenomenon appeared in various model architectures, including LLMs, datasets, and adversarially trained models.

### Strengths
There are abundant empirical results provided across different datasets and measures. The presentation of the overall paper is also concise.

### Weaknesses
Did not see a formal mathematical definition of "valley", but only a definition for measuring sharpness?

### Questions
How do you think the "uncanny valley" phenomenon would affect the developing defense techniques?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper empirically analyzes adversarial examples through the property of sharpness. As an adversarial attack gets progressively stronger, the loss surface first gets progressively sharper, and then flatter after the label is flipped. A measure of flatness can be used to theoretically bound the adversarial robustness for Lipschitz models, providing a theoretical connection between flatness and robustness.

### Strengths
The authors identify an empirical phenomena relating two areas, adversarial robustness and flatness of the loss landscape, potentially providing a path to increased adversarial robustness. They demonstrate this phenomena on a wide variety of models and datasets.

### Weaknesses
It is unclear what the bigger picture is with regards to the uncanny valley phenomena identified in this paper. In particular, consider the form of the trace of the Hessian on line 228. For adversarial examples if the feature extractor does not change much (since by definition we induce small perturbations when finding adversarial examples), then $\sum_{i=1}^d \phi_i^2$ is roughly constant, so the sharpness measure is essentially just $\sum_{j=1}^k \hat{y}_j(1-\hat{y}_j)$, from which all of the empirical results are easily explained. The relative sharpness is large when there are label probabilities that are not certain (far from 0 or 1). In the process of flipping the prediction from one label to another, the probabilities smoothly interpolate to a regime where the label probabilities are less certain, which explains why the relative sharpness first increases, and then decreases once the label is flipped. With this explanation the phenomena is not surprising, as this would just be a natural consequence of adversarial examples existing.

### Questions
- What practical effect does this empirical phenomena have? What can be done with this information?
- In the empirical results, do the adversarial examples have $\sum_{i=1}^d \phi_i^2$ roughly constant, so that the relative sharpness is mostly determined by $\sum_{j=1}^k \hat{y}_j(1-\hat{y}_j)$?

### Soundness
2

### Presentation
3

### Contribution
2
