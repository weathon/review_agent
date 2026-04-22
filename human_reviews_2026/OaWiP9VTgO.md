# BALF: Budgeted Activation-Aware Low-Rank Factorization for Fine-Tuning-Free Model Compression

- Avg Score: 5.33
- Decision: Reject
- Scores: 6, 2, 8

## Abstract
Neural network compression techniques typically require expensive fine-tuning or search procedures, rendering them impractical on commodity hardware. Inspired by recent LLM compression research, we present a general activation-aware factorization framework that can be applied to a broad range of layers. Moreover, we introduce a scalable budgeted rank allocator that allows flexible control over compression targets (e.g., retaining 50\% of parameters) with no overhead. Together, these components form BALF, an efficient pipeline for compressing models without fine-tuning. We demonstrate its effectiveness across multiple scales and architectures, from ResNet-20 on CIFAR-10 to ResNeXt-101 and vision transformers on ImageNet, and show that it achieves excellent results in the fine-tuning-free regime. For instance, BALF reduces FLOPs on ResNeXt-101 by 45\% with only a 1-percentage-point top-1 accuracy drop.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces BALF, a fine-tuning-free compression pipeline that unifies activation-aware low-rank factorization across fully connected, convolutional, and grouped-convolution layers, and couples it with a lightweight budgeted rank allocator based on Lagrangian relaxation that directly targets user-specified FLOPs or parameter budgets. The method estimates uncentered-whitening transforms from a small calibration set, performs truncated SVD in the whitened domain to minimize expected layer-output distortion, and then selects per-layer ranks globally to satisfy a compute/size constraint, replacing layers with efficient two-stage low-rank modules only when they yield real savings.

### Strengths
- General, principled activation-aware factorization applicable to grouped convolutions, not just FC layers; the method uses whitening to minimize expected layer-output distortion.
- Budget-aware rank allocation: sets layer-wise ranks to meet FLOPs or parameter budgets via a Lagrangian relaxation.
- Broad empirical coverage across CNNs and ViTs, with competitive fine-tuning-free trade-offs.

### Weaknesses
- Innovation scope: activation-aware factorization is not entirely new; the paper’s novelty lies in generalizing it to conv/grouped-conv and coupling it with an efficient budgeted allocator. This is a strong systems contribution but less of a theoretical breakthrough.
- The model-level bound (Theorem 3 in the main text; formal version in the appendix) is derived for a sequential L-layer network and does not explicitly handle residual/skip connections typical of ResNets. While the empirical section includes ResNet-family models, the paper does not clarify how residual blocks are treated in the theoretical bound (e.g., block-level aggregation or additional terms due to additive skips) nor in the practical error accounting within a residual structure.

### Questions
- Can BALF be composed with post-training quantization?
- When applying BALF to ResNet families, do you fuse BN into Conv before factorization, or do you factorize Conv and then correct BN statistics?

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
5

### Summary
This paper presents BALF, an efficient pipeline for compressing models using low-rank matrix decomposition without fine-tuning.

### Strengths
+ The motivation of this paper is clear, i.e., to select the optimal ranks using SVD to factorize the matrices of the target DNN models.
+ This paper provides sufficient theoretical analysis for the proposed method.
+ The proposed method shows effectiveness on several CNN models in the experiments.

### Weaknesses
- Limited novelty. The idea of compressing neural networks through matrix factorization is long-established, and many prior works have investigated rank selection strategies. In addition, the use of augmented Lagrangian formulations for enforcing low-rank constraints in network compression has already been explored [R1]. The paper does not seem to introduce a fundamentally new principle or formulation.

- Lack of large-scale validation. Although the motivation centers on compressing large models, no experiments are conducted on modern large-scale architectures such as LLMs. Instead, the evaluation focuses on relatively small and dated models (e.g., ResNet, ResNeXt, ViT). Given the large body of existing work applying matrix factorization to such models, the contribution appears incremental.

- Missing comparison with alternative compression methods. The paper does not compare its method against other major compression approaches, such as pruning or quantization, making it difficult to assess the relative performance and practicality of the proposed technique.

- Insufficient baselines in experiments. The set of comparison methods in the experimental section is very limited, and several state-of-the-art approaches, e.g., [R1] mentioned in the related work section, are not included in the empirical evaluation. This weakens the paper’s claim of superiority or generality.

- Unvalidated theoretical claims. Although the paper presents several theoretical results, the lack of strong experimental evidence and comprehensive comparisons makes it hard to verify the practical value of these theoretical contributions.

[R1] Low-rank compression of neural nets: Learning the rank of each layer.

### Questions
Please see the Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper suggests a unified and efficient fine-tuning-free compression framework for deep neural networks that combines activation-aware low-rank factorization with a budgeted rank allocation mechanism.

Traditional low-rank factorization techniques reduce model size by truncating singular values of layer weight matrices, but they often require fine-tuning or heuristic search to recover accuracy. BALF addresses this by (1) extending activation-aware decomposition to general layer types (including grouped convolutions) and (2) introducing a scalable, zero-overhead budgeted allocator that meets user-specified parameter or FLOPs constraints without retraining.

The method computes uncentered whitening matrices of layer activations to perform activation aware SVD, minimizing output distortion rather than parameter distortion. A Lagrangian relaxation formulation then determines per-layer rank allocations to meet global resource budgets efficiently.

Empirical results are also provided, outperforming existing fine-tuning-free baselines in both accuracy and runtime. On an RTX 2070, compression of ImageNet-scale models completes in minutes without fine-tuning.

### Strengths
Strengths

Conceptual innovation - 
BALF reframes low-rank compression through an activation-centric lens, ensuring that the projection minimizes functional output distortion rather than raw parameter deviation. This distinction aligns compression with representational behavior.

General framework-The authors generalize activation-aware SVD to a more general expressible layer, encompassing dense, convolutional, and grouped convolutional layers in a unified algebraic formulation.

Mathematical rigor: Theorems 1 and 2 show equivalence between activation aware and direct output truncation schemes, with a closed-form expression for activation distortion in terms of singular values.

I especially likes the rank allocator that transforms a combinatorial multiple choice knapsack problem into a linear-time Lagrangian relaxation, which enables control over global compression budgets.

Empirical performance: The improvement is consistent compared to standard SVD baselines, maintaining high accuracy at large compression ratios.

Reproducibility and presentation:
Implementation details, pseudocode, and open-source repository are provided, which enable accessibility for less theory researchers.

### Weaknesses
- The proof assumes ungrouped layers and bounded Lipschitz constants without empirical validation.
- The theoretical bounds may be too loose for larger networks.
- The method is not so strong on already-optimized architectures (e.g., MobileNet-V2). The adjustment might be adapted for each architecture.
The description of FLOP estimation and calibration sampling could be expanded for full reproducibility.

### Questions
The theoretical bounds depend on Lipschitz constants. Can't they be too high in practice?

### Soundness
3

### Presentation
3

### Contribution
3
