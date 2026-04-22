# SELECT: SELEctive Context Transfer for Class Incremental Semantic Segmentation

- Avg Score: 2.67
- Decision: Reject
- Scores: 4, 2, 2

## Abstract
Class-Incremental Semantic Segmentation (CISS) is fundamentally challenged by catastrophic forgetting and background shift, where learning new concepts degrades performance on previously seen classes. While existing methods attempt to balance stability (retaining old knowledge) and plasticity (learning new knowledge), they often fail to leverage prior knowledge effectively. These approaches typically rely on indiscriminate knowledge transfer or ambiguous initializations, which can dilute crucial semantic information. To overcome this limitation, we propose SELECT, a novel approach for Selective Context Transfer. SELECT intelligently transfers knowledge by focusing on inter-class semantic similarity. Its core is a Context Transfer Attention mechanism that identifies semantically related past classes and selectively transfers their contextual information to guide the learning of new ones. To ensure this transfer does not corrupt the original representations, we introduce a controlled perturbation and novel discriminative loss that explicitly enforces separation between the latent spaces of the new class and its influential old-class counterparts. Extensive experiments on PASCAL VOC 2012 and ADE20K demonstrate that SELECT significantly mitigates catastrophic forgetting and background shift, providing a robust and effective solution to the stability-plasticity dilemma.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes SELECT, a method for Class-Incremental Semantic Segmentation (CISS) that addresses catastrophic forgetting through selective knowledge transfer. The key idea is to identify semantically similar classes from previous tasks and use Context Transfer Attention (CTA) to selectively transfer their knowledge when learning new classes. The method measures representational perturbation to identify similar classes, then uses an attention mechanism to aggregate knowledge from these classes. A context transfer loss enforces separation between new and source class representations. Experiments on PASCAL VOC and ADE20K show improvements over recent methods.

### Strengths
## 1. Well-motivated
* The selective knowledge transfer approach based on inter-class similarity is intuitive and differentiates from prior work that either uses background initialization (MiB, SSUL) or transfers from all classes indiscriminately (NeST)
* The similarity metric (Eq. 1) offers a novel way to measure semantic alignment between old and new classes
* The combination of CTA with controlled noise (Eq. 5) and discriminative loss is a reasonable approach to balance knowledge transfer and separation

## 2. Well-written
* Well-structured paper with clear motivation and problem formulation
* Figure 3 effectively illustrates the overall framework
* The writing is generally clear

### Weaknesses
## 1. Unfair Comparison with NeST
The paper compares SELECT (using ViT-B/16 backbone, stated in Section 4.1) against NeST baselines that appear to use the Swin-B backbone. According to the original NeST paper, ViT-B achieves 76.5 mIoU (All classes) on VOC 15-1, significantly higher than the 71.4-72.2 reported in Table 2. The reported NeST scores in Table 2 match the Swin-B results from the original paper. This creates an unfair comparison where SELECT (ViT-B) is compared against a weaker baseline (NeST with Swin-B), potentially overstating the contribution.

## 2. Substantial Reproduction Gaps for Main Baseline, MBS
The reproduced MBS results differ significantly from reported values, especially on the ADE20K dataset:
* VOC 15-1: 80.6 (reported) → 79.0 (reproduced)
* ADE20K 100-5: 42.8 (reported) → 38.8 (reproduced)

While reproduction discrepancies can occur due to various factors (random seeds, library versions, hardware differences), the paper would benefit from providing more detailed information about the reproduction process:

* Environmental setup details (PyTorch/CUDA versions, GPU specifications)
* Number of random seeds used and variance observed
* Whether reproduction was verified with the original authors
* Discussion of potential sources of discrepancy

This additional information would strengthen confidence in the experimental comparisons and help the community better understand and reproduce the results.

## 3. Unvalidated Core Assumption
* Lines 279-280 state: "A small deviation between these two vectors indicates that the content of the image xi is contextually aligned with class k." 
* This is the foundation of the method, yet this critical assumption lacks proper validation: The similarity metric (Eq. 1) is foundational to the entire method, yet Figure 4 only shows that perturbations vary across classes without demonstrating correlation with actual semantic similarity.
* The paper needs to demonstrate that this metric reliably captures semantic similarity (e.g., with empirical or statistical analysis)

## 4. Sensitive Hyperparameters
The method introduces several hyperparameters that appear to have a substantial impact on performance: $\sigma$ (Table 9), $\alpha$ (Table 10), and $L_{ct}$.
These sensitive hyperparameters may suggest the method is brittle and may be difficult to tune for new datasets or scenarios. 

Please provide the sensitivity analysis for the $L_{ct}$ and threshold value.

### Questions
## 1. Aggressive Language
The Phrases below are unnecessarily strong for academic writing. They could be softened while still conveying the limitations being addressed.

- Line 52: "This approach is fundamentally misguided ..."
- Line 80: "this brute-force-like strategy ..."
- Line 81: "This is akin to learning a new skill by reading an entire library instead of consulting a few expert texts"

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
4

### Summary
This paper proposes a method for class-incremental semantic segmentation (CISS) by transferring only semantically relevant prior knowledge from a small set of past classes to each new class rather than relying on background or indiscriminative transfer. The authors propose to identify similar past classes via a representational perturbation test and aggregate their tokens with a context transfer attention (CTA) module. They also introduce a consensus-based class selection, which, for each new-task image, picks the past class with minimum token perturbation. It then keeps only past classes that frequently win across the dataset. The CTA module forms a guided token by attending over tokens of the selected prior classes, where query is the mean context with softmax weighting. This gives initialization for a new class. Controlled perturbation is added to the CTA token to create a buffer zone in representation space and mitigate overlap with its sources, and a margin-based context transfer loss to explicitly separate new and source classes. Experiments on standard datasets show consistent improvements across disjoint and overlapped scenarios and incremental learning settings over strong baselines. Detailed ablations are given for the data selection, CTA, noise, and loss terms.

### Strengths
This paper provides a nice motivational discussion on why background-centric or global transfer is suboptimal. It then proposes to use per-class selective transfer guided by a concrete, measurable stability criterion. The token selection, CTA, noise and margin loss are conceptually simple and easy to integrate. Perturbation and context transfer loss explicitly enforce separation from source classes which addresses a common source for forgetting when transferring from semantically close classes. Experiments are done on various overlapped settings, and detailed ablations make the contribution of each part convincing.

### Weaknesses
Data selection is based on token perturbation, but its robustness under shift or noise is not clearly explained. Experiments focus on ViT with a Segmenter-like head; it’s unclear whether selection and CTA behave similarly for other architectures. There are many hyperparameters like frequency threshold, noise mix and margin need to be fine-tuned. It is not clear to me how the proposed method addresses the specific problem of background shift in CISS.

### Questions
1 What is the per-step cost of computing perturbations for all past classes, CTA calculation and context transfer loss? How does complexity of the proposed method scale to larger models? 
2. Do the performance gains hold for other model architecture and longer continual learning task sequences? Does drifting and instability happen as the number of *selected* prior classes grows over time?

### Soundness
2

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
4

### Summary
The paper introduces a new strategy to select relevant contextual information from the past classes to improve the learning of new classes. It introduces Context Transfer Attention and context transfer loss to select useful information.

### Strengths
1. The paper revisits the problems of background leakage and how to leverage past class information when learning new classes. This problem is relevant in incremental segmentation.
2. The paper presentation and motivation are clear.

### Weaknesses
1. The Introduction claims that "current methods often fail to leverage prior knowledge effectively". This is not true. Several works such as REMINDER [1*], MBS [Park et al.] already leverage prior knowledge about the background (MBS), or about the relational structures between classes for optimal learning/forgetting trade-off (REMINDER).
2. It's unclear how the perturbation prevents damaging the past knowledge. How does the perturbation "creates a small gap for adapting a new class". And how "creating a small gap for adapting a new class" prevents knowledge loss?
3. The contribution (3) of context transfer loss aims to push away old and new class tokens/weights. This separating loss objective is already introduced in MBS as a form of orthogonality loss. As such, it lacks novelty. The only difference is the similarity function. What are the problems of MBS's orthogonality loss? How the proposed loss fixes those problems? What experiments we need to run to validate those problems and how the proposed loss can fix them?
4. According to Table 2, the performance of proposed method is lower than baselines on two tasks in terms of all metrics. On the other hand, performance gain is minimal (less than 1 p.p) on ADE20k. Overall, the performance gain is minimal. As the accuracy improving is minimal, how is the cost of training SELECT compared with other lightweight techniques such as NeST and INC?
5. What is the core difference when both REMINDER and SELECT aim to learn relevant information from old classes? How is the performance gain compared with similar past-class-learning methods such as REMINDER [1*]? 
6. According to Table 4, removing contrastive loss and feature distillation loss yields a higher (all) mIoU on 15-5. It shows that the contribution of contrastive loss is minimal. How shall we interpret this performance drop results?


[1*] Class Similarity Weighted Knowledge Distillation for Continual Semantic Segmentation, CVPR2022.

### Questions
Refer to Weaknesses section.

### Soundness
3

### Presentation
4

### Contribution
2
