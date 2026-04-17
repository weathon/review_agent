# Focus on Likely Classes for Test-Time Prediction

- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
We ask: Can focusing on likely classes of a single, in-domain sample improve model predictions? Prior work argued no.  We put forward a novel rationale in favor of yes: Sharedness of features among classes indicates their reliability for a single sample. We aim for an affirmative answer without using hand-engineered augmentations or auxiliary tasks. We propose two novel test-time fine-tuning methods to improve uncertain model predictions. Instead of greedily selecting the most likely class, we introduce an additional step, focus on the likely classes, to refine predictions. By applying a single gradient descent step with a large learning rate, we refine predictions when an initial forward pass indicates high uncertainty. The experimental evaluation demonstrates accuracy gains for one of our methods on average, which emphasizes shared features among likely classes. The gains are confirmed across diverse text and image domain models.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors propose a test-time fine-tuning approach designed to improve model predictions under uncertainty. Their method encourages the model to focus on the most likely classes in order to refine its outputs. Concretely, they introduce two complementary strategies: doFo (Decrease Out-of-Focus), which suppresses the logits of unlikely classes, and iFo (Increase Focus), which amplifies the logits of likely ones. Extensive experiments across diverse datasets and architectures—covering both text generation and image recognition tasks—demonstrate the effectiveness and generality of the proposed approach.

### Strengths
1. The authors conduct extensive experiments across multiple models and datasets on both text and image tasks to validate the effectiveness of doFo and iFo, demonstrating the comprehensiveness of their evaluation.
2. Figure 1 and Figure 2 effectively illustrate the core ideas and workflow of the proposed approach, making the methodology easy to understand for readers.

### Weaknesses
1. In the field of test-time adaptation, there exists a related approach called PASLE [1], which partitions data into confident and uncertain subsets—assigning one-hot labels to confident samples and candidate label sets to uncertain ones. Its mechanism of assigning 1 to likely classes and 0 to unlikely ones is conceptually similar to the authors’ strategy of “focusing on the likely classes.” However, this prior work is not discussed or compared in the paper. I recommend the authors include a discussion and experimental comparison with PASLE to clarify the novelty and advantage of their method.
2. The experimental section lacks comparisons with any baseline methods, which makes it difficult to assess the absolute performance of the proposed approach. The authors are encouraged to identify and include several relevant methods from the literature for empirical comparison, especially from adjacent domains such as test-time adaptation.
3. The manuscript contains several overly long paragraphs that affect readability. The authors are advised to split long paragraphs into shorter ones to improve clarity, logical flow, and overall presentation quality.

[1] Selective Label Enhancement Learning for Test-Time Adaptation. ICLR 2025

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies whether focusing on likely classes can improve model predictions during test time. The authors introduce two simple fine-tuning strategies—iFo (increasing focus) and doFo (decreasing out-of-focus)—that apply one gradient descent step only when the model shows high uncertainty (measured by the top-1/top-2 probability gap). iFo aims to enhance shared features among likely classes, while doFo suppresses unlikely ones. Experiments across 70+ model-dataset pairs (vision and language) show that iFo consistently improves accuracy, while doFo often does not.

### Strengths
Novel yet simple idea – The “focus on likely classes” concept is intuitive and differs from classical entropy minimization or confidence-based TTA.

The method is tested on diverse image (CNNs/ViTs) and text (GPT-2, LLaMA-3, Gemma-3, etc.) models, showing broad empirical coverage.

Only one gradient step and per-sample adaptation make the method computationally efficient.

### Weaknesses
Although the paper cites Tent and related works, it does not directly compare with them in experiments (e.g., Tent, TTT++, CoTTA, etc.). As a test-time method, more quantitative comparisons with test-time adaptation approaches would strengthen claims.

The reported gains are relatively modest (e.g., +0.1–0.3%), which raises concerns about their practical significance. While the aggregated metrics (mean, standard deviation, and p-values) provide some support, they are not entirely convincing. In this context, comparisons with direct most-likely-class-based approaches such as Tent and ReCap [Region Confidence Proxy for Wild Test-Time Adaptation, ICML 2025] would be particularly important to better demonstrate the method’s advantage.

### Questions
How does the proposed method perform on out-of-distribution (OOD) datasets (e.g., ImageNet-Corruption), which are commonly used for evaluating test-time learning approaches?

More detailed and clearer ablation studies analyzing the effectiveness of iFo and doFo would also be preferred.

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
5

### Summary
This paper proposes two test-time fine-tuning methods, Increasing outputs for Focus classes (iFo) and Decreasing outputs of Out-of-Focus classes (doFo), to improve predictions on uncertain samples by focusing on likely classes via a single large-step gradient descent. An uncertainty assessment based on probability differences triggers optimization only when needed. Theoretical analysis highlights how iFo amplifies shared features among likely classes.

### Strengths
1. Efficient single-step optimization with large LR approximates multi-step results, minimizing computational overhead.
2. Comprehensive evaluation across diverse models (e.g., ViTs, ResNets, LLMs like GPT-2, Llama) and datasets (ImageNet, OpenWebText, etc.), demonstrating consistent gains for iFo (e.g., up to 2.2% on WideResNet).
3. Ablations on hyperparameters (LR, uncertainty threshold, iterations) and comparisons (e.g., input tuning) provide thorough insights.
4. Practical applicability: No auxiliary tasks or source data needed, works on pre-trained models.

### Weaknesses
1. Main concern: The method's primary motivation and approach seem to have appeared in prior TTA work [1] (Selective Label Enhancement Learning for Test-Time Adaptation, ICLR); authors need to further explain and strengthen the novelty and advantages of their method.

2. Images should be optimized for display and layout, preferably using vector graphics.

3. Some typos exist, e.g., line 266 has an extra ".".

### Questions
1. Can the method integrate with existing TTA techniques (e.g., entropy minimization) for further gains?

2. What architectural factors (e.g., transformers vs. CNNs) influence gains, and why?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes test-time fine-tuning to improve uncertain predictions by focusing on likely classes. When uncertainty is high (difference between top-2 probabilities less than 0.16), two methods are applied: (1) iFo increases outputs of focus classes by maximizing their weighted logits, and (2) doFo decreases outputs of unlikely classes by minimizing their average logits. Single gradient step with large learning rate modifies logits. Evaluated on 70+ model-dataset pairs (ImageNet with CNNs/ViTs, 6 LLMs on 6 text corpora), iFo shows consistent 1-2% improvements while doFo fails.

### Strengths
**Simple and Practical**: Elegantly simple - single gradient step on logits when uncertainty is high. Uncertainty measure (difference between top two probabilities) requires no calibration. Architecture-agnostic with easy implementation (code in Appendix B.2). Requires only one extra forward-backward pass.

**Broad Evaluation**: 70+ model-dataset pairs across vision (ImageNet on ResNet/DenseNet/EfficientNet/MobileNet/ViT) and language (GPT-2, Llama, QWEN, Fox-1, StableLM, Gemma on diverse corpora). Output spaces range 1K-100K+ classes. Honest reporting of doFo failures adds credibility.

### Weaknesses
**Limited Novelty**: Test-time gradient adaptation is established in TTA/domain adaptation. Main distinction (multiple likely classes vs. single class) is incremental. No comparison with existing TTA methods (Tent, TTT, MEMO) or calibration methods (temperature scaling, Platt scaling). Single-step optimization is a practical trick, not a conceptual advance.

 **Modest Gains Without Context**: Consistent 1-2% improvements but missing: (a) wall-clock time overhead measurements, (b) comparison with simple baselines (ensembles, calibration), (c) whether gains justify deployment complexity, (d) failure rate analysis - what percentage worsen? Figure 6 aggregates results without showing variance or per-sample effects.

### Questions
Please refer to the Weakness

### Soundness
2

### Presentation
2

### Contribution
2
