# Few-Shot Class-Incremental Learning based on Hierarchical Dual-Stream Interaction and Associative Memory Fusion

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 6

## Abstract
Few-Shot Class-Incremental Learning (FSCIL) aims to learn novel classes from limited examples while preserving previously acquired knowledge. Current methods face two challenges: (1) Collapsed intra-class variance, where enhancing base-class separability limits generalization; and (2) Boundary instability, where few novel samples distort feature distribution and cause catastrophic forgetting. To address these challenges, we propose a cognition-inspired framework that employs a dual-stream network to extract a unified representation space with strong generalization and a hierarchical fusion mechanism with associative memory to improve old and new feature distribution. This framework comprises two key modules for rapid adaptation and long-term stability. The Hierarchical Dual-Stream Interaction Network (HDIN) decouples feature learning into a ResNet-based local stream for fine-grained detail extraction and a ViT-based global stream for long-range semantic dependencies. These streams are dynamically integrated via channel-adaptive attention to harmonize multi-scale information, simulating cognitive-level feature integration. The Associative-Enhanced Hierarchical Memory Fusion (AE-HMF) module simulates cortical memory consolidation by Gaussian sampling from class prototypes as associative memories and performing cross-layer feature interactions. Experiments on CIFAR100, miniImageNet, and CUB200 show that under the setting of no large-scale pretraining or data expansion techniques, our approach achieves the lowest Performance Decline Rates (DR) across all benchmarks, delivering a state-of-the-art balance between accuracy and forgetting. This work establishes a cognition-inspired, unified framework that effectively promotes the generalization capability and reduces catastrophic forgetting in FSCIL.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a new Few-Shot Class-Incremental Learning method by adopting a Hierarchical Dual-Stream Interaction Network (HDIN), which consists of a ResNet-based local stream and a ViT-based global stream. Associative-Enhanced Hierarchical Memory Fusion (AE-HMF) module performs cross-layer fusion and information-driven Gaussian sampling to enhance long-term memory retention and semantic stability. Experiments are conducted on CIFAR-100, miniImageNet, and CUB200 datasets.

### Strengths
HDIN has a well-motivated architecture, which decouples local and global cues and recombines them via a reasonable attention fusion strategy. AE-HMF’s hierarchical cross-layer fusion with mutual information-guided Gaussian sampling is described with clear mathematical details. HDIN reports a consistently low decline rate on CIFAR-100 and miniImageNet.

### Weaknesses
Weakness

1. Figures 1–3 do not clearly convey how modules connect or how information flows. The relationships and contribution paths are hard to follow. The statement around Lines 203–205 is also unclear and should be rewritten for precision.

2. The information-driven Gaussian sampling needs more details to clarify.

3. Top-k selection of base classes via MI can be fragile to few-shot covariance. The paper lacks sensitivity studies. It is unclear how the training samples distilled by the sampling strategy are incorporated into the training stage.

4. The quantitative performance of the proposed HDIN is not significant. On CUB200, CoDF outperforms HDIN in both aACC and DR.

5. Given the dual-stream and fusion complexity and the claim of better efficiency than CoDF, the proposed HDIN needs more quantitative evidence (e.g. FLOPs, training and inference time, GPU memory, parameter counts).

### Questions
For general suggestions, please refer to the weakness part. Here are several specific points to improve the article:

1. Revise Figures 1-3 to specify the workflow of HDIN, especially the interaction between the proposed modules. Add necessary details, such as what the dotted line indicates.

2. Are the empirical mean and covariance matrix fixed after the first round of computation? Regarding the description in lines 332-335, better to specify the update rule and provide a short pseudocode block if possible.

3. Study how sensitive HDIN results are to the selection of k in IGS, for example, report performance vs. k for MI-guided sample selection. If possible, discuss robustness under higher variance.

4. Please provide training and inference time, GPU memory, and parameter counts for each module vs. methods for comparison, particularly CoDF.

5. To align with the description of the cognition-driven learning paradigm in Figure 1, include Figure 6 (currently in the supplement) as supporting evidence referenced in the main manuscript.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper addresses the FSCIL problem and proposes a cognition-inspired framework which uses a hierarchical Dual-Stream Interaction Network (HDIN), consisting of a CNN and a ViT, and associative-Enhanced Hierarchical Memory Fusion (AE-HMF) which uses Gaussian samples and prototypes for reducing forgetting. Experiments on standard benchmarks show SOTA performance.

### Strengths
1) The paper is clearly written and the cognition inspired motivation is sound.
2) The proposed method shows SOTA performance on multiple benchmark FSCIL datasets.
3) Pairing local (CNN) and global (ViT) cues can help genearlize from limited data available in FSCIL.

### Weaknesses
1) The novelty of the paper is limited. There have been a large number of FSCIL methods that are cognition inspired, or use prototypes and Gaussian pseudo-samples. How is this paper contributing further than the large literature of FSCIL methods? Some examples of older FSCIL works that used these ideas: [a,b]
2) The method combines a large number of modules with many hyperparameters, making it more complex compared to prior methods. Have the authors analyzed the model compexity, the memory efficiency and computational efficiency? The gain in accuracy compared to many prior methods seems to be minimal (and less than CoDF). Does such a gain in accuracy warrant such a complex method that might incur more computational burden and require tuning more hyperparameters?
3) It was stated in the appendix that for CUB-200, the paper used pre-trained ImageNet features on both the CNN and the ViT. This should be clarified in the main paper. The paper should also compare against simpler works that use pre-trained ImageNet features and achieve SOTA performance. e.g. [b].
4) Gaussian sampling around prototypes risks class drift or over-smoothing of fine-grained classes. What variance schedule, class-adaptive covariance, and safeguards against prototype collapse are used?

[a] FearNet: Brain-Inspired Model for Incremental Learning, ICLR 2018.
[b] Ayub, A., & Fendley, C. (2022). Few-shot continual active learning by a robot. Advances in Neural Information Processing Systems, 35, 30612-30624.

### Questions
Please see the weaknesses section for my questions.

### Soundness
2

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
This paper investigates the challenging problem of Few-Shot Class-Incremental Learning (FSCIL), where a model must incrementally learn new classes from few examples while retaining previously acquired knowledge. The authors propose a cognition-inspired dual-module framework comprising a Hierarchical Dual-Stream Interaction Network (HDIN) that fuses local (ResNet-based) and global (ViT-based) representations through channel-adaptive attention, and an Associative-Enhanced Hierarchical Memory Fusion (AE-HMF) module that utilizes Gaussian sampling of class prototypes for cross-layer associative memory consolidation. Experiments on CIFAR100, miniImageNet, and CUB200 benchmark datasets demonstrate their method’s effectiveness in achieving lower performance degradation rates and competitive accuracy, without relying on large-scale pretraining or data augmentation.

### Strengths
1. Intuitive approach to deal with the FSCIL problem.
2. Strong performance improvement in terms of the lowest performance drop for CIFAR100 and miniImageNet datasets

### Weaknesses
1. Poor performance on the CUB dataset. The authors should discuss the reason behind this clear gap from the best method. Is it because of larger image size or a finegrained setup, etc.
2. The CoDF method seems to perform better in terms of accuracy, even though, as the authors mention, it requires more epochs for training. However, the authors should describe in more detail why the proposed approach should be preferred over CoDF apart from the training epochs.

### Questions
1. What is the reason behind the clear gap from the best method on the CUB dataset. Is it because of larger image size or a finegrained setup, etc.
2. Why the proposed approach should be preferred over CoDF apart from the training epochs.?
3. Does the proposed approach use any additional information that CoDF doesnot use?

### Soundness
3

### Presentation
3

### Contribution
3
