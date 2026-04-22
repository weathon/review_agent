# Consistency-Driven Calibration and Matching for Few-Shot Class Incremental Learning

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Few-Shot Class Incremental Learning (FSCIL) is crucial for adapting to the complex open-world environments. Contemporary prospective learning-based space construction methods struggle to balance old and new knowledge, as prototype bias and rigid structures limit the expressive capacity of the embedding space. Different from these strategies, we rethink the optimization dilemma from the perspective of feature-structure dual consistency, and propose a Consistency-driven Calibration and Matching (ConCM) framework that systematically mitigates the knowledge conflict inherent in FSCIL. Specifically, inspired by hippocampal associative memory, we design a memory-aware prototype calibration that extracts generalized semantic attributes from base classes and reintegrates them into novel classes to enhance the conceptual center consistency of features. Further, to consolidate memory associations, we propose dynamic structure matching, which adaptively aligns the calibrated features to a session-specific optimal manifold space, ensuring cross-session structure consistency. This process requires no class-number priors and is theoretically guaranteed to achieve geometric optimality and maximum matching. On large-scale FSCIL benchmarks including mini-ImageNet, CIFAR100 and CUB200, ConCM achieves state-of-the-art performance, with harmonic accuracy gains of up to 3.41% in incremental sessions. Code is available at: https://github.com/wire-wqz/ConCM

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes a consistency-driven calibration-and-matching approach that calibrates biased new-class prototypes and stabilizes the feature-space structure, thereby reducing confusion between old and new classes. The approach demonstrates strong performance across several benchmarks.

### Strengths
1. The motivations are clear. Biased new-class distributions caused by limited training samples, and adjustments to the feature space to accommodate new classes, are key challenges in FSCIL, as recognized by the community.
2. Leveraging common attributes to augment the features of new classes is valid and reasonable; WordNet provides rich semantic information beyond visual cues.
3. The proposed approach achieves strong performance on several datasets, including mini-ImageNet, CIFAR-100, and CUB-200.

### Weaknesses
1. The idea of “Attribute Separation” is similar to PA [1], which decouples common attributes within a family and transfers them to new species. It would be helpful to discuss the commonalities and differences between your approach and PA [1].  
[1] Prototype antithesis for biological few-shot class-incremental learning.
2. Neural collapse theory has been introduced into the FSCIL task by NC-FSCIL. The dynamic structure matching in this paper builds on that work; for example, it replaces the hard enforcement of a computed prototype distribution with a softer approach that reduces the distance between the original and optimal distributions. While this is indeed more reasonable and effective, the core idea and methodology are not fundamentally different from NC-FSCIL; in my view, this is an incremental improvement.
3. When performing prototype augmentation, this paper makes a strong assumption, namely, that feature distributions are Gaussian. However, many real-world class distributions do not align well with this assumption. Furthermore, based on this assumption, the authors infer the covariance of new classes by assuming that classes with similar means also have similar covariances. Is there theoretical justification for this assumption?
4. The MPC module appears highly dependent on the base-class distribution and on WordNet. If the base classes differ substantially from the new classes, or if the names of the new classes are not present in WordNet’s knowledge base, the model’s generalization ability may be greatly affected.
5. There are a few typos. For example, “mini-IamgeNet” at line 432 and “SOAT” at line 360 (these should be “mini-ImageNet” and “SOTA”).

### Questions
Please refer to the 'weaknesses'. I would consider to raise my rating if the authors could address my concerns.

### Soundness
3

### Presentation
2

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
The paper investigates how to maintain representation consistency in Few-Shot Class-Incremental Learning (FSCIL). The authors propose a framework that calibrates new class prototypes inspired by human associative memory. The calibration module separates and completes semantic attributes to refine prototype representations for new classes, ensuring better alignment. Furthermore, a geometric optimization strategy is introduced to preserve structural consistency during incremental updates. Experimental results on multiple FSCIL benchmarks demonstrate consistent improvements over existing methods.

### Strengths
1. The paper is well written, and the motivation is clear.
2. The proposed consistency-driven dynamic structure matching method is theoretically grounded and achieves excellent performance.
3. The ablation experiments are comprehensive and convincing.

### Weaknesses
1. The paper missed several important and highly related references and comparative results. 

    [1]  Mamba-FSCIL: Dynamic Adaptation with Selective State Space Model for Few-Shot Class Incremental Learning.

    [2] Learning With Fantasy: Semantic-Aware Virtual Contrastive Constraint for Few-Shot Class-Incremental Learning.

    [3] Learning optimal inter-class margin adaptively for few-shot class-incremental learning via neural collapse-based meta-learning.

    [4] Towards Better Representation Learning for Few-Shot Class-Incremental Learning

2. The pipeline of novel class prototypes calibration is the same as paper [1] except for the design of MPC network (both adopt the encode–aggregate–decode architecture). The novelty of this component should be further clarified.

    [1] Prototype completion for few-shot learning. 

3. More recent research on CIL mainly focuses on pre-trained ViT or CLIP models. Can the proposed method be transferred or adapted to pre-trained ViT models or CLIP models?
 
   [1] Pre-trained Vision and Language Transformers Are Few-Shot Incremental Learners.

### Questions
1. It is unclear whether the proposed method can be generalized to other tasks, such as few-shot incremental semantic segmentation.
2. What is the ratio hyper-parameter between the loss $L_{match}$ and $L_{Cont}$? Is it set to 1? Would it make a difference in performance when setting different values?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the FSCIL problem by identifying two key challenges — feature inconsistency and structure inconsistency.
The proposed ConCM framework introduces (1) a Memory-aware Prototype Calibration (MPC) module that leverages semantic attributes to calibrate prototypes, and (2) a Dynamic Structure Matching (DSM) module that dynamically updates class geometry to maintain global consistency.
Experiments on multiple benchmarks show that ConCM achieves better stability and accuracy than prior FSCIL methods.

### Strengths
1. The proposed modules (MPC and DSM) are conceptually sound and complementary, leading to consistent and clear improvements across multiple benchmarks.

2. The paper provides clear theoretical motivation and connects the design of DSM with neural collapse geometry, giving the framework better interpretability.

### Weaknesses
1. The Memory-aware Prototype Calibration (MPC) relies on semantic attribute extraction from WordNet or class names, which might not generalize to datasets without clear textual labels.

2. Conceptually, ConCM extends previous geometry-based or orthogonality-driven FSCIL ideas (e.g., OrCo, NC-based approaches), so its originality mainly lies in how these components are unified.

3. Consider including at least one recent 2025 method to strengthen the comparison with up-to-date FSCIL approaches.

### Questions
How sensitive is the performance to the semantic quality of the extracted attributes in MPC? Would ConCM still work well if semantic information is noisy or unavailable?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces ConCM, a novel two-stage framework for few-shot class-incremental learning that explicitly addresses the “dual-consistency” dilemma. By emulating hippocampal associative memory, the MPC module first calibrates few-shot prototypes with semantically related attributes extracted from base classes; the DSM module then dynamically updates the embedding geometry to satisfy both equi-distant separation and maximal matching with the previous structure.

### Strengths
1.	The illustration is clear.
2.	The method and the theoretical analysis seem solid.
3.	The reported performance improvement is considerable.

### Weaknesses
1.	The proposed ConCM uses WordNet to extract semantic attributes from class names, to calibrate the prototypes of the new classes. It works fine in the standard benchmarks such as cifar, imagenet, but if there are no semantic label names for each class, how would such calibration work?
2.	Lack of results on benchmarks with more classes. ConCM relies on explicitly calibrating the feature space for each class. The paper does not discuss the influence of the increased number of classes, especially when the benchmarks in this paper contain at most 200 classes.
3.	Typo: Caption of Table 4 “mini-imagenet”.

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
2
