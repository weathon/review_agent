# LANE: Label-Aware Noise Elimination for Fine-Grained Text Classification

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
In this paper, we propose Label-Aware Noise Elimination (LANE), a new approach to learning with noisy labels. At its core, LANE introduces a new metric---label-aware margin---aimed at quantifying the degree of noise of each training example (or quality thereof). LANE leverages the semantic relations between classes and monitors the training dynamics of the model on each training example to dynamically lower the weight of training examples that are perceived to have noisy labels. We test the effectiveness of LANE on multiple text classification tasks and benchmark our approach on a wide variety of datasets with various numbers of classes and amounts of label noise. LANE considerably outperforms strong baselines on all datasets and settings, obtaining significant improvements ranging from an average improvement of 2.88% in F1 on manually annotated datasets to a considerable average improvement of 4.75% F1 on datasets with high level of injected label noise. We carry out a comprehensive analysis of LANE and identify the key components that lead to its success.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces LANE, a training framework designed to enhance model robustness under noisy labels. LANE dynamically adjusts the contribution of training examples by reducing the weight of samples likely to be mislabeled. To identify such samples, it leverages training and semantic relationships. Extensive experiments across multiple fine-grained text classification tasks demonstrate consistent gains, showcasing LANE’s effectiveness in handling label noise.

### Strengths
The reviewer notes the following strengths:
- The paper clearly articulates the limitations of prior methods and justifies the need for LANE.
- The proposed LANE methodology is intuitive and technically sound.
- The authors provide extensive experiments on multipole fine-grained text classification datasets that demonstrate consistent and meaningful improvements over multiple baselines.
- The authors also includes a large set of ablations alongside thoughtful exploration of weight distributions that help provide insight into the underlying methodology.

### Weaknesses
From the reviewer's perspective, LANE relies heavily on the foundational capabilities of the underlying model. Therefore additional evaluations on language models beyond BERT would add to the real-world applicability of LANE. But in general, the reviewer finds the overall methodology to be sane and showcase valuable improvements.

### Questions
See weakness above.

### Soundness
3

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
3

### Summary
LANE provides a method for learning with label noise. The authors suggest that label noise could arise from error-prone labelling processes or just genuinely ambiguous labels and they must be weighed differently. The key idea is to assign high weights to hard but clean examples that have high semantic similarity of class labels and down-weigh truly mislabelled examples.

### Strengths
* Addresses an important and novel aspect of handling noisy labels - distinguishing between ambiguous or similar labels and erroneous ones
* The method identifies and retains hard but clean examples with higher weight, taking into consideration the semantic similarity of label names
* Extensive empirical analysis has been done comparing LANE with other existing methods

### Weaknesses
* Requires additional network training to learn the weights - a computational overhead
* The label-aware supervise contrastive loss could be explained a bit more intuitively, right now, it requires readers to have a strong prior understanding of contrastive learning
* The method is motivated to handle ambiguous or semantically close label noise, but the experiments only consider 20% random noise settings. 
* L95 to L99: piling up such a larger number of references without explaining the individual contributions is not helpful at all

### Questions
* Have the authors considered the case of class imbalance? 
* How much compute power is required to learn the weights?
* Could the authors provide analysis of the learned weights in relation to semantic similarity?
* How would LANE perform for increased label ambiguity and not just random noise?
* Edit suggestions for better readability
1. Algorithm 1 - lines 170-173 could be written more legibly 
2. Font too small for all tables
3. Line 245 typo ‘mislabeled’

### Soundness
3

### Presentation
3

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
This paper introduces Label-Aware Noise Elimination (LANE), a novel approach to enhance deep learning model robustness against label noise in fine-grained text classification—a task where classes are semantically similar (e.g., distinguishing between subtypes of news articles or product reviews), making label noise particularly detrimental and common. LANE’s core design integrates two complementary signals to identify and downweight noisy training examples: (1) semantic relations between classes and (2) model training dynamics .

### Strengths
1. Fine-grained text classification is ubiquitous in real-world applications (e.g., e-commerce product categorization) but uniquely vulnerable to label noise—human annotators often confuse semantically similar classes.
2. LANE’s fusion of dynamic training dynamics (capturing how the model learns an example over time) and static class semantics (capturing inherent class ambiguities) is innovative.
3.  The reported F1 improvements (2.4–4.5%) are meaningful for fine-grained tasks

### Weaknesses
1. it is not clear how the supervised contrastive loss helps in Eq (8)? Is there any intuitive illustration?

### Questions
see weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Label-Aware Noise Elimination (LANE), a method that introduces a novel metric called label-aware margin. By incorporating inter-class semantic similarities and model training dynamics, LANE quantifies the label noise level of each training sample. It dynamically down-weights suspected noisy samples, thereby mitigating the harmful effects of noisy labels while retaining all training samples (including hard but clean ones). Experiments across ten text classification datasets demonstrate LANE's effectiveness under varying noise levels.

### Strengths
1. Clear Motivation: The paper effectively identifies two key limitations in existing methods: (1) valuable samples below a low AUM threshold are unnecessarily removed from the training set, and (2) AUM computation treats labels independently, ignoring semantic similarities between them. The authors propose LANE to address these specific shortcomings.
2. Clear Methodology: The theoretical analysis and step-by-step description of the proposed method are clear and well-presented.
3. Extensive Experiments: The comparative experiments are comprehensive, covering numerous datasets spanning most text classification tasks. The proposed LANE method achieves near-universal improvements over baselines.

### Weaknesses
1. Several works on noisy label learning emerged in 2025. As a submission to a 2026 conference, the lack of comparison with new methods such as [1], [2], and [3] detracts from the overall novelty of the work.
2. The approach requires training two BERT models for text classification, which is inefficient. Furthermore, while LLM-related experiments are included, the LLMs rely solely on context. The work could be strengthened by incorporating parameter-efficient fine-tuning techniques like LoRA or integrating noisy label learning methods specifically designed for LLMs, such as [3], to enhance persuasiveness.
3. The Related Work section is disorganized. Noisy label learning encompasses various approaches, but the authors resort to simple listing. This issue is also reflected in the main experiments (Table 2), which lack intuitive grouping. It is recommended to group and present comparisons by methodology type.
4. This paper lacks the reproducibility statement required by ICLR.

References:

[1] Pan et al., Enhanced Sample Selection with Confidence Tracking: Identifying Correctly Labeled yet Hard-to-Learn Samples in Noisy Data, AAAI, 2025.

[2] Xu et al., Revisiting Interpolation for Noisy Label Correction, AAAI 2025

[3] Ye et al., Calibrating Pre-trained Language Classifiers on LLM-generated Noisy Labels via Iterative Refinement, SIGKDD 2025.

### Questions
1. The experiments use BERT as the base model. Is LANE compatible with other architectures and large language models?
2. Why wasn't PLF compared in the 20% noise setting experiments?
3. The complexity of this work is not low. Why not include a main framework diagram? This would make the overall workflow and contributions of the work clearer.

### Soundness
3

### Presentation
3

### Contribution
3
