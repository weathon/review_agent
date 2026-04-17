# Class-Conditional Neuron Pre-Activation Divergence to rule out validation set in label noise early stopping

- Decision: Reject
- Scores: 4, 2, 6

## Abstract
Label noise poses a major challenge in supervised deep learning: models tend to memorize corrupted labels, leading to poor generalization. Early stopping can mitigate this but usually requires a clean validation set, reducing training data. We introduce Class-Conditional Neuron Pre-Activation Divergence (CND), a metric that measures the divergence between class-conditional and marginal distributions of neuron pre-activations. We show that pre-activations naturally form class-dependent modes, which collapse during memorization of noisy labels, making distributions less class-dependent. Leveraging this insight, we propose a validation-free early stopping criterion that relies only on training data. Specifically, we observe that the CNDs of the last layer peak at the point of maximum generalization, enabling training to be halted without a held-out validation set and thus preserving all available data for learning. Across benchmarks with symmetric and instance-dependent label noise, our method consistently outperforms other validation-free approaches—especially on datasets with many classes and at low noise levels. These results highlight the value of analyzing pre-activation distributions for understanding memorization and improving generalization.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This work studies the relation between neuron pre-activation distributions and label-noise memorization. The authors hypothesize that the class-dependent structure in these distributions weakens as memorization sets in.

To capture this transition, they introduce a metric, Class-Conditional Neuron Pre-Activation Divergence (CND). Based on the observation that CNDs of the last hidden layer peak in alignment with generalization, they propose a validation-free early stopping criterion.

Their experiments reveal that CND outperforms baseline method.

### Strengths
1.  The work proposes a novel approach by linking neuron pre-activation distributions to label noise memorization, offering an interesting perspective on identifying the generalization peak.
2.  The core hypothesis is intuitive and well-grounded.
3.  The proposed CND metric demonstrates strong performance against PC in their settings.

### Weaknesses
My overall view is that this appears to be a first paper written by a very junior author and has a considerable number of flaws. After discussing the paper, I will provide some suggestions for junior authors on academic writing, offered with my best wishes. (I have read the paper in its entirety. These suggestions are not necessarily for you to revise in the rebuttal, but are fundamental advice for your future work).

Returning to the paper's content, I think the topic and the idea are interesting and have some insight. At a time when many choose quick a+b+c papers on LLMs or somehow ChatGPT can do something we are already known as an ACL-style entry point, we should encourage ML researchers like this, who are genuinely interested in discussing a small and refined topic about deep networks themselves.
Out of encouragement, I am giving a borderline reject before the rebuttal and will keep an open mind. 

I encourage the authors to get more research training (in reading literature, experimental design, and paper writing) before submitting to a top-tier conference again.

### Questions
**Major:**

1. I could not find the details of the experimental setup in the main text. The authors state, "For details about the datasets, see Appendix C, and for the simulation settings, see Appendix D," yet I was unable to find the appendix (it was not in the Supplementary Material either). Even if it were provided, this critical information should be included in the main body of the paper for self-containment.

2. The choice of noise settings is not well-justified. For instance, why is the maximum noise rate for CIFAR-100 only 40%? Why was only one type of synthetic noise evaluated? Furthermore, the baseline comparison is incomplete; it lacks a crucial comparison against standard early stopping with a small validation set.

3. What is PMF in Table 1(c)? The term is used without any prior definition in the text.

4. The claimed equivalence from Eq. (4) to Eq. (5) does not seem to hold under the given context. It appears to depend on the relationship between the true marginal distribution P(z) and the uniform mixture M(z) used in the JSD definition (I do not really understand it, although I have try my best to follow their flow).

5. While the paper links CND to the phenomenon of memorization, it lacks a deeper analysis of why memorizing noise would cause the class-conditional distributions to collapse toward the marginal distribution. The authors should provide at least a self-consistent reason for this core claim.

**Minor:**

1. On line 249, the index [i] is used. However, only the index [j] has been previously defined. It is unclear where i originates from.

2. The method for estimating the continuous probability density functions from data needs to be described more clearly.

**Suggestions:** 

(Not necessarily for you to revise in the rebuttal)

1. The paper has many short paragraphs, some consisting of only a single line ("orphans"). These should be consolidated into more developed paragraphs to improve readability and flow. A paragraph should ideally contain more than just a few sentences.

2. The equation around line 333 is unnumbered. The caption for Figure 1 contains the phrase, "...highlighting the multimodal behavior induced by backdoor triggers." This is confusing.

3. Placing all the experimental tables on Page 8, a page with no accompanying text, is not appropriate. Tables and figures should be placed near the text that discusses them.

4. The paper does not make full use of the page limit, suggesting a lack of depth. A more thorough evaluation is expected for a top-tier venue. For example, what ablation studies are missing? The authors should consider ablating the CND aggregation rule (e.g., mean vs. quartile) or the specific layer chosen.

5. The figures themselves are very large (maybe too large to leave space in paper for more information), yet the font size for labels and text within them is too small, which harms readability.

6. Have you considered adding a simple, illustrative diagram on page 2 to visually summarize your key finding? This could be referenced in the introduction and would significantly help readers quickly grasp the core concept of your work.

---

As a final suggestion, I recommend you carefully read the following papers to study their writing, visualization, experimental settings, and overall paper layout:

[1] Toneva, Mariya, et al. "An Empirical Study of Example Forgetting during Deep Neural Network Learning." International Conference on Learning Representations, 2019.

[2] Pleiss, Geoff, et al. "Identifying mislabeled data using the area under the margin ranking." Advances in Neural Information Processing Systems 33 (2020): 17044-17056.

[3] Li, Junnan, Richard Socher, and Steven CH Hoi. "DivideMix: Learning with Noisy Labels as Semi-supervised Learning." International Conference on Learning Representations, 2020.

[4] Frankle J, Carbin M. The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks[C]//International Conference on Learning Representations, 2019.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses the problem of early stopping under label noise without using a clean validation set. The authors propose a novel metric, Class-Conditional Neuron Pre-Activation Divergence (CND), which measures the divergence between class-conditional and marginal distributions of neuron pre-activations. They also show that when a network begins to memorize noisy labels, the class-specific neuron pre-activation collapses toward the marginal distribution.

Based on this observation, the authors design a validation-free early stopping criterion to stop training when the last-layer CND reaches its peak. Experiments on multiple benchmarks (MNIST, CIFAR-10/100, CIFAR-10N, and NEWS) demonstrate that CND consistently identifies optimal stopping epochs more accurately than prior validation-free methods such as Prediction Changes (PC) and Known Polluted Accuracy (KPA), thereby improving the performance.

### Strengths
1. The paper introduces a novel perspective for early stopping through neuron pre-activation distributions.
2. The authors propose Class-Conditional Neuron Pre-Activation Divergence (CND) to measure the divergence between class-conditional and marginal pre-activation distributions.
3. The authors conduct experiments on several datasets to verify the effectiveness of the proposed method. Compared to the previous method, the proposed method can better identify the early stopping epoch.

### Weaknesses
1. The experiments are mainly conducted on small or medium-scale datasets such as MNIST and CIFAR. Evaluating the proposed method on larger and more realistic datasets (e.g., Clothing1M, WebVision) would better demonstrate its scalability and practical utility under real-world settings.
2. The failure cases at very high noise rates (≥60%) are mentioned but not deeply analyzed. Additional discussion on why CND fails would improve completeness.
3. The comparison mainly focuses on validation-free baselines. It would be better to compare with other baselines published in recent years.
4. The current version appears incomplete. Multiple cross-references to appendices are unresolved, and the corresponding supplementary materials also do not include the Appendix.

### Questions
1. The meanings of the x-axis and y-axis in Figure 1 are not clearly described. Could the authors provide more details about them?
2. Why are the lines on Subfigure (f) more than those on the other Subfigures?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a novel metric called Class-Conditional Neuron Pre-Activation Divergence (CND), which measures the divergence between class-conditional and marginal pre-activation distributions of neurons. This metric is used to develop a validation-free early stopping criterion to mitigate label noise in deep learning models. The method claims to be particularly effective in datasets with multiple classes and lower noise levels, offering better generalization compared to existing methods.

### Strengths
1. The CND metric offers a new perspective on how neuron activations evolve during training, providing a novel tool for detecting memorization due to noisy labels.
2. The paper is well-structured, with clear explanations of the key concepts and results.

### Weaknesses
1. Some details are not clearly explained. The implementation details of the figures and formulas in the paper, such as the distinction between memorization and generalization in Figure 2, need more detailed explanations. This is particularly important for readers who may not have related background knowledge.
2. Limited experimental scope. The paper only evaluates on very small models using a relatively single-type dataset. Larger models generally have better noise tolerance, and the paper does not demonstrate the potential of the method on larger models. Additionally, the datasets used, CIFAR-10 and CIFAR-100, are actually quite similar, being both image classification tasks. The main difference between them is the number of classes (10 vs. 100), with CIFAR-100 being an extension of CIFAR-10, adding more categories. However, the feature distributions and task nature are not fundamentally different. Moreover, the results on the extended NEWS dataset are not particularly significant. The comparison methods are also limited (in fact, only one comparison is made). Testing CND on more diverse real-world datasets would make the results more valuable.

### Questions
1. Are there any limitations to using CND in models with very deep architectures, where the interpretation of pre-activations might become more complex?

2. Has there been previous research on class-conditional distributions? Could you provide more details on how your work differs from existing research?

### Soundness
3

### Presentation
2

### Contribution
3
