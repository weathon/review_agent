# Fly-CL: A Fly-Inspired Framework for Enhancing Efficient Decorrelation and Reduced Training Time in Pre-trained Model-based Continual Representation Learning

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Using a nearly-frozen pretrained model, the continual representation learning paradigm reframes parameter updates as a similarity-matching problem to mitigate catastrophic forgetting. However, directly leveraging pretrained features for downstream tasks often suffers from multicollinearity in the similarity-matching stage, and more advanced methods can be computationally prohibitive for real-time, low-latency applications. Inspired by the fly olfactory circuit, we propose Fly-CL, a bio-inspired framework compatible with a wide range of pretrained backbones. Fly-CL substantially reduces training time while achieving performance comparable to or exceeding that of current state-of-the-art methods. We theoretically show how Fly-CL progressively resolves multicollinearity, enabling more effective similarity matching with low time complexity. Extensive simulation experiments across diverse network architectures and data regimes validate Fly-CL’s effectiveness in addressing this challenge through a biologically inspired design. Code is available at https://github.com/gfyddha/Fly-CL.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes FlyCL, which addresses multicollinearity in continual learning. They use a mixture of sparse random projection, top-k sparsity and streaming ridge classification. They demonstrate impressive computational speedups (90%), while maintaining comparable performance. 

These are real and practical speedups, and this is definitely good engineering. The bio-inspiration however does not add much substance beyond motivation.

### Strengths
1. Very strong practical results, the paper shows very significant speedups with barely any loss in accuracy.
2. The method is clearly general enough to adapt to different architectures and datasets, in a plug-and-play manner.
3. The work has solid experimental evidence, good ablations, and statistical reporting.

### Weaknesses
1. The fly brain parallel is mostly surface-level (it looks like), and only inspires the sparse projection component.
2. Some hyperparameters seem to require architecture-specific tuning, although that does not necessarily negate the proposed generality.
3. Would be useful to examine if this scales to tasks at the scale of modern foundational models.

### Questions
1. How sensitive is the performance to m, p and k? The defaults chosen seem somewhat arbitrary.
2. When 10k dimensions are not sufficient, how does this scale to larger models?
3. Is there a benefit to keeping the projection matrix random as opposed to learning it?

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
5

### Summary
This paper presents Fly-CL, an efficient framework for continual learning with frozen pre-trained encoders. It introduces two key components: (1) a sparse random projection with top-k activation sparsification to decorrelate features and improve prototype separability, and (2) a streaming ridge regression classifier with adaptive regularization via generalized cross-validation for stability and low computational cost. Experiments across ViT-B/16 and ResNet-50 backbones show that Fly-CL achieves comparable or higher accuracy than prior representation-based methods (e.g., RanPAC, F-OAL) while reducing post-extraction training time.

### Strengths
1. The paper is clear motivated and formulated, easy-to-follow
2. High efficiency with strong accuracy trade-off
3. Simple and generalizable design, broadly applicable to backbones and datasets.

### Weaknesses
1. The proposed framework shares conceptual similarities with earlier representation-based approaches such as RanPAC and F-OAL, both of which employ random projections and analytic updates. While Fly-CL introduces additional sparsification and adaptive regularization, the methodological advancement over these predecessors appears incremental rather than fundamentally novel.

2. This paper does not empirically demonstrate the effect of reduced prototype correlation.
3. The experiments primarily report average accuracy across tasks, but omit standard CL metrics such as the final-task accuracy ($\mathbf{A}_T$) and forgetting measure. 
4. The study focuses on representation-based and prompt-based baselines but do not compared with recent lora/adapter-efficient continual tuning methods such as InfLoRA [A], SEMA [B], and MoE-Adapters [C].
5. Discuss why Fly-CL slightly underperforms RanPAC on CIFAR-100 despite achieving substantial gains on other datasets. 
6. It is not specified whether the ViT-B/16 model is initialized from ImageNet-21K or ImageNet-1K pre-trained weights. 

[A] Liang, Y. S., & Li, W. J. Inflora: Interference-free low-rank adaptation for continual learning. CVPR2024.

[B] Wang, H., et al. Self-expansion of pre-trained models with mixture of adapters for continual learning. CVPR2025

[C] Yu, J., et al. Boosting continual learning of vision-language models via mixture-of-experts adapters. CVPR2024.

### Questions
see weakness.

### Soundness
3

### Presentation
2

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
This paper proposes Fly-CL, a continuous learning framework inspired by the Drosophila olfactory circuit, designed to address multicollinearity issues in representation learning based on pre-trained models while reducing training time. Fly-CL achieves feature decoupling and efficient classification through mechanisms including sparse random projection, Top-k activation filtering, and streaming ridge classification. Experiments demonstrate that this method achieves or surpasses state-of-the-art performance across multiple datasets and backbone networks while significantly reducing training time, exhibiting particularly strong advantages during post-feature extraction processing stages.

### Strengths
1. The paper addresses the issue of “persistent feature decoupling in pre-trained models.” While the constituent components (random projection, Top-k, ridge regression) are established techniques, their creative combination and application constitute the contribution.
    
2. The paper provides a solid theoretical foundation, demonstrating the information retention capability of sparse projections. The experimental design encompasses multiple architectures (ViT, ResNet), datasets, and evaluation metrics.
    
3. This work provides a solution for efficient continuous learning in resource-constrained scenarios.

### Weaknesses
1. I believe the core mechanism of the paper—“random projection + Top-k sparse activation”—shares striking similarities with the fundamental concept of Kanerva's Sparse Distributed Memory (SDM). SDM is similarly inspired by neuroscience and employs high-dimensional sparse representations and similarity matching to address memory and learning challenges. The paper omits discussion with this classic approach.

2. Despite significant efficiency gains, projecting dimensions of m=10,000 may still impose memory constraints on extreme edge devices.

### Questions
1. Could the author establish a simple baseline by using only Fly-CL's projection and Top-k layers, followed by a straightforward linear classifier or k-nearest neighbors classifier, to demonstrate the necessity of streaming ridge regression in your problem setting?

2. Although Fly-CL claims to mitigate forgetting, how does it affect old tasks at the feature space level? When learning new tasks, do the class prototypes of old tasks drift or distort in the high-dimensional space after Fly-CL processing?

3. Theorem B.1 aims to prove that sparse projection matrices W are almost certainly full-rank, which is considered an argument for their ability to preserve information. However, in machine learning, the fact that a random matrix is full-rank does not directly equate to it being a “good” feature mapper. More importantly, the matrix's isometric property or distance-preserving characteristic is guaranteed by the Johnson-Lindenstrauss (JL) lemma. Can the authors provide evidence that your sparse matrix W indeed preserves pairwise distances between feature vectors with high probability?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Taking inspiration from the fly's olfactory system, the authors introduce FLY-CL, a way to extract and use features of a pretrained model for class incremental learning.

The features of a pretrained model are expanded with a fixed random projection, in a similar way to the a layer from "projection neurons" to "Keynyon cells" in the olfactory circuit of a fly. A top-k activation simulates lateral inhibition. A learned similarity matching down projection emulates the projection to "mushroom body output neurons". 

They show how this method reduces catastrophic forgetting when using pretrained models with a set of image benchmarks.

### Strengths
The high-dimensional sparse layer does seem to effectively prevent catastrophic forgetting. 

No task identity is required at inference. 

works with unmodified pre-trained models 

ablation studies do show the need for each part of the model (random projection / ridge regression / normalisation)

Figure 5 shows how important the high dimension layer is, with performance saturating at m>10k. This is crucial since I believe this is the main novel contribution.

### Weaknesses
There can be high memory costs associated with the large sparse layer (the authors do discuss this).

They do need the model to receive task boundaries and to store "class prototypes" for use during inference. 

No adaptation to the backbone so it is dependent on a good pretrained model (a weakness or a strength depending on the circumstance).  

The paper would benefit from a more detailed analysis of how the key hyperparameters (m, p, and k) scale with task complexity, the number of tasks/classes, and different pretrained backbones.

minor typo:
037 "generalization in downstream tasks for downstream tasks"

### Questions
you show the performance saturates beyond 10k. Do you expect that saturation point to remain stable as the number of tasks/classes grows much larger?

Have you run experiments that study how the required projection dimensionality scales with increased task complexity or number of tasks/classes? Or if it changes if you use a different pretrained backbone?

### Soundness
3

### Presentation
3

### Contribution
3
