# Debiased Contrastive Learning with multi-resolution Kolmogorov-Arnold Network for Gravitational Wave Glitch Detection

- Decision: Reject
- Scores: 5, 5, 3, 6

## Abstract
Time-series gravitational wave glitch detection presents significant challenges for machine learning due to the complexity of the data, limited labeled examples, and data imbalance. To address these issues, we introduce Debiased Contrastive Learning with Multi-Resolution Kolmogorov-Arnold Network(dcMltR-KAN), a novel self-supervised learning (SSL) approach that enhances glitch detection, robustness, explainability, and generalization. dcMltR-KAN consists of three key novel components: Wasserstein Debiased Contrastive Learning (wDCL), a CNN-based encoder, and a Multi-Resolution KAN (MltR-KAN). The wDCL improves the model’s sensitivity to data imbalance and geometric structure. The CNN-based encoder eliminates false negatives during training, refines feature representations through similarity-based weighting (SBW), and reduces data complexity within the embedding. Additionally, MltR-KAN enhances explainability, generalization, and efficiency by adaptively learning parameters. Our model outperforms widely used baselines on O1, O2, and O3 data, demonstrating its effectiveness. Extending dcMltR-KAN to other time-series benchmarks underscores its novelty and efficiency, marking it as the first model of its kind and paving the way for future SSL and astrophysics research.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper introduces a new architecture and self-supervised learning method dedicated to gravitational wave glich detection. Three modules are introduced as the novel components of the proposed method: wDCL, CNN-based encoder, and Multi-Resolution KAN. The author conducted one experiment to show its superiority to existing SOTA methods and another for ablation study. The method is also tested in emotional speech recognition. The results show the proposed method performs the best among the presented baselines

### Strengths
- The structure of the paper is easy to follow. Motivation of using each method is clearly written at the top of each section 
- Three methods proposed for the glich detection task, when used all together, perform SOTA in objective metrics

### Weaknesses
- Tables are not clear: 1) Results of the baseline model w/o two methods (wDCL, mltR-KAN) should be listed in Table 2 to make readers easy to find the difference. 2) Difficult to know if the numerical differences between methods in Table 2 are statistically significant
- Task selection is not clear: EMODB dataset is for speech emotion recognition and not suited for event detections. If the author wishes to claim the effectiveness of the proposed method in the audio domain, I recommend including an experiment with a sound event detection dataset like DESED, MAESTRO, most of which you could find at the official IEEE DCASE challenge website
- Presentation of figures: Figure 1 and 2 are too small. Figure 2 is not even cited in the paper. I suspect Figure 1 is mistakenly cited instead of Figure 2. Still one of them is not cited

### Questions
Why the SOTA comparison in the audio experiments are not included in Table 3?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
The authors present dcMltR-KAN, consisting of three components. Wasserstein Debiased Contrastive Learning (wDCL), a CNN-based encoder and multi-resolution KAN (MltR-KAN) layers. wDCL replaces the Euclidian distance with the Wasserstein distance to enhance the sensitivity to the geometric structure of the data. MltR-KAN can capture patterns at different scales. The model outperforms fully supervised baselines in classifying glitches in gravitational glitches and in speaker emotion classification.

### Strengths
1. Application of recent KAN architecture in suitable application fields and novel combination with debiased contrastive learning. Ablation studies show the importance of both contributions to a base CNN.
2. Significant improvements compared to current state of the art on gravitational wave dataset and speaker emotion detection dataset.

### Weaknesses
1. The presented experiments are done in somewhat niches and therefore give limited insides of the significance of the work for the ML community. More experiments could be provided, e.g. on the task of environmental sound classification (ESC50 dataset).
2. The KAN architecture is a essential part of your model and could also be motivated in the introduction.

### Questions
1. What are meaningful features?
2. Is the explainability still given if the CNN first encodes the input data?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper applies debiased contrastive learning (DCL) on wave gravitational data for the task of wave glitch detection. The authors add an auxiliary Wassterstein DCL loss to address data imbalance. Additionnally, they refine the feature representations by weighting using the similarity embedding matrix. Finally, they introduce Multi-Resolution Kolmogorov-Arnold Network (KAN) layers on top of a CNN encoder to capture multi-scale patterns.

### Strengths
1. The paper is clearly written and easy to follow. 

2. The application of contrastive learning for wave glitch detection is interesting, and is promising for extending to large unlabelled dataset.

3. The use of False Negative Elimination (FNE) at multiple levels with the multi-resolution KAN is an interesting idea.

4. The ablation studies effectively demonstrate the importance of different model components.

### Weaknesses
1. Unsupported claims


1.1. The authors make strong claims about KAN layers enhancing explainability, efficiency, and generalization without providing supporting evidence. The KAN layers, in fact, introduce additional complexity, and the paper does not demonstrate the claimed explainability.

1.2. Propositions and theorems are stated without proof or detailed explanation. These could have been included in an appendix for clarity and rigor.

2. Insufficient Literature Review

2.1. The literature review lacks coverage of class imbalance approaches, especially relevant methods in self-supervised contrastive learning.

2.2. Comparisons are limited to the authors' own baselines; there is no evaluation against established methods from the literature ( including on the EmoDB audio dataset).

3. The proposed approach is complex and appears unsuitable for scaling to large datasets, potentially limiting its practical applicability for large-scale learning on unlabeled data.

### Questions
In Section 3.1, please define L_wass for clarity. Figure 1 is very small; consider increasing its size to enhance readability. The legends in Figure 2 are very small and too difficult to read.

1. For L_wdcl (Eq. 2), why not use a single hyperparameter to balance the two losses, allowing one to scale relative to the other instead of using two hyperparameters?

2. I don't see the point of comparing two losses computed on different examples in Proposition 1, can you clarify its purpose? (also true for Theorem 3, and probably Theorem 2 too). Additionally, could you define the Rademacher complexity to improve clarity for readers unfamiliar with this concept?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents a new KAN-based model to solve time-series gravitational wave glitch detection problem. It is characterized by its multi-resolution KAN architecture and debiased contrastive learning loss that improves the CNN-based encoder. The paper presents experimental results on the benchmark datasets (also including an audio dataset), showcasing performance improvement over common supervised-learning methods.

### Strengths
- The paper provides rigorous mathematical definitions of their methods. Easy to follow and understand the proposed concept. 
- Although the reviewer is not an expert in that particular domain area, the proposed method seems to have improved over the reasonable supervised-learning models. 
- The paper did ablation tests to prove that each of the newly proposed methods is meaningful.

### Weaknesses
- It is not clear why Wasserstein distance needs to be used. The explanation implies the imbalanced data could be an issue by comparing the distributions, instead of Euclidean distance. But wouldn't the same argument be made via KL-divergence?

- It could be based on my lack of experience with the dataset, but the way the dataset was divided into training and testing folds, while the information on validation is missing, is not too informative. More detail is needed, as to how the model was trained. 

- The construction of the similarity matrix seems to be very computationally demanding. Meanwhile, it also means that the features can be based on the original similarity defined in the CNN's output. The additional contrastive learning part does introduce some performance improvement, but it is a little difficult to wrap my head around how these two different approaches interact with each other.

### Questions
- In eq (2), what's the meaning of x(\alpha)? I understand what \alpha actually means, but x as a function of \alpha doesn't make sense to me. 

- Is there any better way to construct the positive pair for contrastive learning? Curious if adding Gaussian noise generalizes well enough to unseen examples. For example, in text processing, positive word pairs can be constructed by using context windows. Is there any way to do something like this in this dataset?

### Soundness
3

### Presentation
3

### Contribution
3
