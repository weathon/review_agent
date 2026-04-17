# Beyond Cropping and Rotation: Automated Evolution of Powerful Task-Specific Augmentations with Generative Models

- Decision: Reject
- Scores: 4, 6, 2, 6

## Abstract
Data augmentation has long been a cornerstone for reducing overfitting in vision models, with methods like AutoAugment automating the design of task-specific augmentations. Recent advances in generative models, such as conditional diffusion and few-shot NeRFs, offer a new paradigm for data augmentation by synthesizing data with significantly greater diversity and realism. However, unlike traditional augmentations like cropping or rotation, these methods introduce substantial changes that enhance robustness but also risk degrading performance if the augmentations are poorly matched to the task. In this work, we present EvoAug, an automated augmentation learning pipeline, which leverages these generative models alongside an efficient evolutionary algorithm to learn optimal task-specific augmentations. Our pipeline introduces a novel approach to image augmentation that learns stochastic augmentation trees that hierarchically compose augmentations, enabling more structured and adaptive transformations. We demonstrate strong performance across fine-grained classification and few-shot learning tasks. Notably, our pipeline discovers augmentations that align with domain knowledge, even in low-data settings. These results highlight the potential of learned generative augmentations, unlocking new possibilities for robust model training.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This author proposes EvoAug, a pipeline that automatically learns task-specific data augmentations by integrating generative models (controlled diffusion and zero-shot NeRF) with an evolutionary search over stochastic augmentation trees. Each tree hierarchically composes operators (e.g. classical transforms, ControlNet-conditioned diffusion via Canny/segmentation/depth/color cues, and 3D NeRF rotations). The method includes novel fitness metrics for low-data scenarios: a K-fold cross-validated loss for few-shot, and an unsupervised clustering-based score (silhouette minus a cluster-size penalty) for one-shot cases.

### Strengths
1)  The key idea of using conditional generative models (diffusion with ControlNet cues and NeRF rotations) in an automated augmentation search is creative and timely.
2) The paper carefully engineers a pipeline with diverse augmentation operators and fit-for-purpose fitness functions, showing strong methodological rigor.
3) Evaluation on multiple fine-grained datasets, with three architectures and various few-shot settings, gives confidence in the findings. The authors even run multiple seeds (reporting means and std dev) and include a random-tree baseline to show the importance of search.
4) Concepts are clearly introduced (e.g. “stochastic augmentation trees”), and the paper flows logically. Background and related work are well-cited and motivate the design choices.

### Weaknesses
1) The performance improvements over strong baselines are modest and dataset-dependent. In particular, on Flowers102 (a canonical fine-grained task) EvoAug underperforms standard augmentations. This raises questions about when the complex pipeline is worth its cost.
2) Using diffusion and NeRF models for each augmentation is expensive. The experiments already took up to 24 hours per setting, making wider adoption challenging.
3) As admitted in Section 5.1, learning augmentations for full-scale datasets is prohibitively slow.
4) The unsupervised clustering metric, while clever, relies on knowing true labels (so not entirely unsupervised).

Missing relevant references in literature review

1) Context-guided Responsible Data Augmentation with Diffusion Models

2) Effective Data Augmentation With Diffusion Models

3) Diffusion models: A comprehensive survey of methods and applications

4) GenMix: Effective Data Augmentation with Generative Diffusion Model Image Editing

5) DiffuseMix: Label-Preserving Data Augmentation with Diffusion Models

### Questions
1) Include ablations of key components. For instance, show performance with vs. without generative operators, or with deeper trees than depth=2. Also compare the one-shot fitness variants directly.
2) Add figures illustrating example augmented images produced by EvoAug and the corresponding augmentation tree policies.
3) Consider comparing to other modern augmentation/search techniques (e.g. ASHA, PBA, or even simpler pretrained generative data augmentation methods) to contextualize gains.

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
3

### Summary
This paper introduces EvoAug, an automated augmentation learning framework that combines classical data augmentations (e.g., cropping, color jitter) with generative augmentation operators (based on ControlNet diffusion models and NeRFs). The method is designed for low-data regimes, including few-shot and one-shot classification tasks. The authors propose several fitness functions (K-fold validation, clustering-based unsupervised evaluation, and double augmentation) to evaluate candidate augmentation policies without extensive supervision. Experiments on six fine-grained datasets (Caltech256, Flowers102, Stanford Dogs, Stanford Cars, Oxford-IIIT Pets, and Food101) show that EvoAug improves robustness and accuracy, often outperforming classical and automated augmentation baselines such as AutoAugment and RandAugment in few-shot and one-shot settings.

### Strengths
1. The combination of controlled diffusion, NeRF-based transformations, and evolutionary search represents a significant methodological innovation in data augmentation.

2. Addressing augmentation learning in few-shot and one-shot settings is both timely and important, as most prior work assumes large datasets.

3. The introduction of clustering-based and loss-based fitness functions for augmentation policy learning without labels is a thoughtful and practical contribution.

4. Algorithms are explicitly described, with evolutionary operations (mutation, crossover) and augmentation tree structures well illustrated.

### Weaknesses
1. The evolutionary search settings (population size, mutation rate, depth-2 trees) are fixed; no sensitivity analysis is presented.

2. The contribution of each augmentation operator type (diffusion, NeRF, classical) is not isolated; it’s unclear how much generative components actually contribute over strong classical baselines.

3. While the paper focuses on classification, there is little empirical evidence suggesting EvoAug’s adaptability to other tasks (e.g., detection, segmentation), despite the claim of broader applicability.

### Questions
1. How sensitive are EvoAug’s results to the evolutionary algorithm’s hyperparameters (population size, mutation rate, tree depth)?

2. Could EvoAug transfer augmentation trees learned on one dataset to another (cross-domain transferability)?

3. Do diffusion-based augmentations introduce artifacts or biases (e.g., unrealistic textures) that could hurt generalization on certain classes?

4. Could the method be adapted to non-vision modalities (e.g., audio, text) given generative models exist there as well?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces EvoAug, an automated data augmentation framework that combines generative models (ControlNet diffusion and NeRF-based operators) with evolutionary algorithms to learn task-specific augmentation strategies.
 The method constructs stochastic augmentation trees composed of both classical and generative operations, optimized through an evolutionary search with customized fitness functions for few-shot and one-shot learning.
 Experiments on six fine-grained datasets and three model architectures demonstrate that EvoAug achieves competitive or superior results compared to AutoAugment and RandAugment, particularly under low-data conditions.

### Strengths
The paper addresses a very important and cutting-edge problem: how to use powerful generative models (Diffusion, NeRFs) as data augmentation operators and automatically integrate them into the learning pipeline. This is a challenging but high-potential direction that goes beyond traditional augmentation methods.

The authors employed a commendable and rigorous approach in their experimental setup (Section 4.1) by intentionally selecting the subset with the lowest baseline accuracy from 10 subsets via preliminary experiments to serve as the benchmark task. This avoids cherry-picking and ensures the evaluation is conducted in the most challenging scenarios.

### Weaknesses
The paper's core premise is that integrating expensive generative operators (Diffusion, NeRF) provides an advantage on fine-grained tasks. However, in the 2-shot and 5-shot experimental results (Tables 1 and 2), the performance of EvoAug (Learned) is not superior to (and often worse than) strong baselines like RandAugment. The authors themselves admit the results are "mixed". Given the significant additional computational overhead EvoAug introduces in both search (EA) and application (generative models), these "mixed" (or worse) results call the necessity and practicality of the method into serious question.

The one-shot setting is the paper's most methodologically interesting contribution, but the results are similarly not overwhelmingly positive. The authors claim in Section 4.3 that "EvoAug consistently outperforms the baselines". This contradicts the data in Table 3. For example:

- On Caltech256/ResNet50, RandAugment (82.92) outperforms EvoAug (81.83).

- On Flowers102/ResNet50, AutoAugment (65.84) significantly outperforms EvoAug (56.70).

- On Stanford Dogs/ResNet50, AutoAugment (77.22) outperforms EvoAug (71.54). This disconnect between the empirical results and the conclusions severely undermines the paper's credibility.

The paper introduces multiple expensive generative operators (Canny, Color, Depth, Segment, NeRF) but provides absolutely no ablation studies to demonstrate that these operators are necessary or beneficial. Did the evolutionary algorithm actually learn to use these generative operators? Or did it primarily rely on the "Classical" operator? If an EvoAug search including only "Classical" and "NoOp" operators achieved similar performance, the paper's core premise about integrating generative models would be unsubstantiated.

The justification for the 1-shot clustering fitness function's necessity (i.e., why simple metrics fail) (Appendix A.5) is a core motivation for the paper's methodological design. Hiding this in the appendix makes the introduction of this metric in the main text (Section 3.3.2) appear unmotivated and somewhat arbitrary.

### Questions
Above

### Soundness
2

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
This paper present an automated augmentation strategy that leverages advanced generative models, specifically controlled diffusion and NeRF operators, in combination with classical augmentation techniques. The authors propose a new pipeline called EvoAug for automated data augmentation using generative models and evolutionary search. The key ideas:

1. Traditional augmentation methods (e.g., crop, flip, rotate) are limited in diversity. The authors argue that the recent availability of generative models (e.g., diffusion or few-shot NeRF) allows richer, learned augmentations rather than hand-coded ones.

2. They build an evolutionary algorithm that searches over augmentation trees: hierarchies of stochastic transformations (including generative-model based ones) tailored to a downstream task (e.g., few‐shot classification).

3. Their experiments show that EvoAug can discover augmentations aligned with domain knowledge and yield improved accuracy in fine-grained and few-shot classification tasks under low-data regimes.

A creative, promising direction that bridges generative modeling and automated augmentation. Needs stronger empirical and scalability validation to move from “promising idea” to “established method.”

### Strengths
Timely and interesting direction — Combining generative models with augmentation search is appealing, especially for low-data regimes where augmentation matters a lot.

Focus on low-data regime — The fact that they tackle few-shot / fine-grained classification, rather than just standard large‐scale regimes, gives practical relevance.

Domain-alignment — The finding that discovered augmentations “align with domain knowledge” is interesting: suggests the search may rediscover meaningful transformations rather than random noise.

### Weaknesses
Limited scalability / generality — The work appears focused on low-data tasks with few-shot classification; it’s unclear how well this would scale to large-scale datasets, high resolution images, or many‐class settings.

Baseline comparisons — It’s unclear if the comparisons include state‐of‐the‐art augmentation methods (e.g., AutoAugment, RandAugment, AugMix) combined with strong generative models, or whether the generative augmentation is compared fairly.

### Questions
How is the search space of “augmentation trees” formally defined—are nodes differentiable transformations, or arbitrary generative modules?

### Soundness
3

### Presentation
3

### Contribution
3
