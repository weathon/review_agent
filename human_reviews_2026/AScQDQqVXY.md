# CARPRT: Class-Aware Zero-Shot Prompt Reweighting for Vision-Language Model

- Decision: Accept (Poster)
- Scores: 6, 4, 2, 4

## Abstract
Pre-trained vision-language models (VLMs) enable zero-shot image classification by computing the similarity score between an image and textual descriptions, typically formed by inserting a class label (e.g., "cat") into a prompt (e.g., "a photo of a").
Since the score for a given image-class pair is sensitive to the choice of prompt, existing studies ensemble multiple prompts using a weighting vector to aggregate scores across different prompts.
Yet, in current strategies, the weighting vector assigned to each prompt is shared across all classes, implicitly assuming that prompts are conditionally independent of classes, which often does not hold in practice, as a prompt like "an aerial view of" might be apt for "airport" but ill-suited for "apple".
To address this, we propose class-aware zero-shot prompt reweighting (CARPRT). 
This scoring scheme adjusts the weighting vector for each class label by capturing the class-specific relevance of different prompts in a training-free manner. 
For each class label and every available prompt, we quantify their class-specific relevance by averaging image–text relevance scores over images predicted to that class under the given prompt. These estimates are then normalized to derive class-specific weights.
Evaluations on standard image classification benchmarks show that CARPRT outperforms existing class-independent reweighting methods, confirming that modeling prompt-class dependencies is crucial for effective zero-shot prediction and even broader VLM-based application settings that rely on prompt ensembling. Our code is available at https://github.com/tmlr-group/CARPRT.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the limitations of class-agnostic prompt reweighting in zero-shot image classification with vision-language models (VLMs). The authors propose CARPRT, a training-free method that estimates class-specific prompt weights using only unlabeled data and VLM similarity scores. CARPRT introduces a principled probabilistic framework that justifies the need for class-aware weighting and provides an efficient implementation by aggregating maximum image-text similarity scores per prompt-class pair. Experiments show that CARPRT consistently outperforms class-agnostic reweighting baselines on various datasets, confirming the crucial role of class-aware prompt reweighting for improved zero-shot performance.

### Strengths
* The paper is well-organized and clearly written. The motivation is compelling, and the methodology is both conceptually sound and practically simple.

* The probabilistic framework offers strong theoretical justification for class-specific prompt weighting, addressing assumptions made by previous works (e.g., WPE).

* The method is training-free, requiring only inference-time access to unlabeled images, making it broadly applicable and computationally efficient.

* Empirical results show consistent gains across datasets, validating the method's generality.

### Weaknesses
* Limited Performance Gain on ImageNet
While CARPRT shows consistent improvements on fine-grained datasets, its advantage over WPE on the general-purpose ImageNet benchmark is marginal. This raises concerns about the method’s scalability or effectiveness in broader, more diverse domains.

* Lack of Real-World Use Case Discussion
The paper focuses primarily on academic benchmarks. Without examples or discussions related to real-world scenarios (e.g., medical or industrial domains), it is hard to evaluate the method's broader applicability where labeled data is truly scarce.

### Questions
* Why is the performance gap on ImageNet so small?
Given that ImageNet is closer to real-world image distributions, why does CARPRT offer minimal gains over WPE in this setting? Could this be due to prompt-template mismatch, semantic ambiguity, or dataset scale?

* Is CARPRT effective in real-world, low-resource domains?
Can the authors comment on whether CARPRT can be effective in domains like medical imaging or satellite imagery, where labels are scarce and prompts may carry domain-specific semantics?

* Can CARPRT be combined with lightweight prompt tuning?
Would it be beneficial to use CARPRT-derived weights as an initialization or prior for prompt tuning frameworks? Could this combination yield further performance gains in narrow domains?

### Soundness
4

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
This paper proposes Class-Aware Zero-Shot Prompt Reweighting (CARPRT), which leverages unlabeled test data to enhance existing zero-shot prompt reweighting methods. Specifically, it estimates a distinct weight for each template–class pair based on the unlabeled test images and utilizes this weight matrix to construct zero-shot text classifiers.

### Strengths
This paper is readable, and I enjoyed its presentation. It introduces a clear and well-motivated method for prompt reweighting using unlabeled test images. The proposed approach is grounded in probabilistic principles.

### Weaknesses
The paper introduces unlabeled test images to aid prompt reweighting. As far as I know, this should not be considered a zero-shot method. In fact, it should be regarded as unsupervised learning. This distinction raises two issues. First, it contradicts the zero-shot claim made in the paper. Second, it makes the comparison against the three baselines unfair, as those are genuine zero-shot methods.

Another critical point is the absence of discussion regarding the sampling methods used in the experiments. It is crucial to address this aspect, as the sampling techniques, as well as the number and distribution of samples, can have a substantial impact on the final performance.

Furthermore, the assumption that we have access to unlabeled test images is somewhat unconventional, as it implies a data collection phase prior to the testing stage. If such a phase exists, one might reasonably ask why the labels are not collected as well. I also notice that the authors discuss transductive methods and methods using external resources in the appendix; however, these operate under different—and arguably more practical—settings than the one adopted in this paper.

### Questions
I believe that incorporating the following experiments would further strengthen this paper.

## Unsupervised baselines:


1. Pseudo-labeling approach: Generate pseudo-labels for the unlabeled samples, obtain a visual classifier based on these pseudo-labels (via class mean), and then combine it with the original CLIP text classifier in a 50-50 manner to predict the remaining samples.


2. Reference [1]: This work presents a published unsupervised method that can serve as a suitable baseline.


3. Reference [2]: Although this is a transductive approach, it constructs a parametric model for classification and can therefore be adapted to the current setting.


## Sampling settings:


1. N-shot sampling: Evaluate performance with varying numbers of labeled samples per class (e.g., 1-shot, few-shot, and larger-N settings).


2. Partial class coverage: Use only a subset of downstream classes to assess generalization ability to unseen classes.


3. Random sampling: Conduct multiple runs of random sampling with a specified proportion of the test dataset. Note that the resulting class distribution may vary significantly across runs.


It is recommended that the authors include experiments comparing their proposed methods against the aforementioned baselines under these different sampling configurations.


## References:
[1] Liang, Jian, et al. “Realistic Unsupervised CLIP Fine-tuning with Universal Entropy Optimization.” Proceedings of the 41st International Conference on Machine Learning, 2024.

[2] Zanella, Maxime, Benoît Gérin, and Ismail Ayed. “Boosting Vision-Language Models with Transduction.” Advances in Neural Information Processing Systems 37 (2024): 62223–62256.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces CARPRT, a training-free method for zero-shot prompt ensembling in vision–language models. Unlike previous approaches that assign global weights to prompts, CARPRT estimates class-specific prompt weights from unlabeled data. The method computes image–text similarities for all prompt–class combinations, assigns pseudo-labels based on these similarities, and averages the scores for images pseudo-assigned to each class to derive class-specific prompt weights. These weights are then used to construct text embeddings for zero-shot classification. Experiments on fine-grained datasets and ImageNet variants show consistent improvements over mean and weighted prompt ensembling baselines.

### Strengths
- **Practical simplicity**

    The proposed approach is conceptually simple yet effective. It operates entirely at inference time, without additional training or fine-tuning, and requires no architectural modification to the underlying vision–language model. This makes CARPRT easy to integrate into existing zero-shot pipelines.

- **Clear motivation**

    The paper addresses a well-identified limitation of class-agnostic prompt weighting by explicitly modeling the dependence between classes and prompts. The intuition is sound and well justified: prompts that work well for one class may not be equally suitable for another, and accounting for this improves prediction reliability.

- **Good empirical results**

    The empirical results show consistent gains across a range of architectures and datasets.

### Weaknesses
- Theoretical analysis does not provide substantial insight and formalizes already intuitive facts.
The theoretical material feels disproportionate to the simplicity of the idea, and several results restate expected behavior rather than offering explanatory or predictive value.

    - Proposition 1 is a standard Chernoff-Hoeffding bound on the convergence of the empirical distribution to the  true distribution. It does not hold when using pseudo-labels.  To see this, consider a case when the VLM systematically under-predicts or doesn't predicts at all a certain classe which can happen on very specialized downstream dataset (e.g. Aircraft). Then the pseudo-label frequency for this class (zero) do not converge to the true class probability.
    - Proposition 2: Shows that class-specific weighting has greater representational capacity than class-agnostic weighting. This result is largely self-evident: allowing class-dependent weights simply provides more degrees of freedom in the model, and formalizing this point does not substantially deepen understanding of the method or its behavior.
    -  EBM formulation & Lemma 1: The EBM framing does not lead to algorithmic consequences, and Lemma 1 effectively restates that prompt weights scale text embeddings and thus linearly affect log-likelihood. This follows directly from the modeling assumptions and does not yield new theoretical insight.

- No analysis of robustness to pseudo-label errors. This limitation is evident in the iterative extension, which underperforms when the VLM is less accurate, suggesting sensitivity to noisy pseudo-labels.

- The contribution, while useful and well-executed, feels incremental. Moving from global to class-specific prompt weighting is a modest conceptual step.

### Questions
1. What is the size of the unlabeled dataset used to estimate prompt weights, and how does performance scale with this size?

2. How does the approach behave under noisy or imbalanced pseudo-labels?

3. Have the authors considered filtering low-confidence pseudo-labels to reduce the influence of uncertain assignments?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes a new scheme for prompt ensembling. The weights of each prompt for a given class $c$ are based on the average similarity score between the pseudo-label related to class $c$ and the image embedding. Experiments show improvement in accuracy and weights related to meaningful concepts.

### Strengths
The method is simple and interpretable by design. The improvements are consistent across multiple datasets and backbones.

### Weaknesses
The contribution is relatively incremental, as it can be summed up as a class-wise weighting of prompts based on their cosine similarity with the input image. Furthermore, recent works also explore the prompt ensembling strategy and should be discussed and compared [1,2]

Ensembling is known for its inference overhead, which might be prohibitive on many applications. Ablations showing the complexity and inference time would be required to evaluate the practicability of the approach.

The Bayesian framework is relatively artificial, as the core of the approach does not rely on the predictive distribution.

[1] Liao, Ning et al. “Rethinking Visual Prompt Learning as Masked Visual Token Modeling.” Artif. Intell. 348 (2023): 104417.
[2] Huang, Chen, et al. "Aggregate-and-adapt natural language prompts for downstream generalization of clip." Advances in Neural Information Processing Systems (2024)

### Questions
How does the proposed approach compare to few-shot prompt learning VLM adaptation methods both in term of performances and complexity?

### Soundness
3

### Presentation
2

### Contribution
2
