# IDAP++: Advancing Divergence-Based Pruning via Filter-Level and Layer-Level Optimization

- Avg Score: 4.67
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 6

## Abstract
This paper presents a novel approach to neural network compression that addresses redundancy at both the filter and architectural levels through a unified framework grounded in information flow analysis. Building on the concept of tensor flow divergence, which quantifies how information is transformed across network layers, we develop a two-stage optimization process. The first stage employs iterative divergence-aware pruning to identify and remove redundant filters while preserving critical information pathways. The second stage extends this principle to higher-level architecture optimization by analyzing layer-wise contributions to information propagation and selectively eliminating entire layers that demonstrate minimal impact on network performance. The proposed method naturally adapts to diverse architectures, including convolutional networks, transformers, and hybrid designs, providing a consistent metric for comparing the structural importance across different layer types. Experimental validation across multiple modern architectures and datasets reveals that this combined approach achieves substantial model compression while maintaining competitive accuracy. The presented approach achieves parameter reduction results that are globally comparable to those of state-of-the-art solutions and outperforms them across a wide range of modern neural network architectures, from convolutional models to transformers. The results demonstrate how flow divergence serves as an effective guiding principle for both filter-level and layer-level optimization, offering practical benefits for deployment in resource-constrained environments.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a novel theoretical and practical framework for neural network compression that operates jointly at the filter and layer levels. The core idea builds upon a formal analysis of information flow divergence, a mathematically defined metric quantifying the evolution of information content as it propagates through a network’s computational graph.

Leveraging this concept, the authors design a two-stage compression pipeline. In the first stage (filter-level pruning), filters or attention heads that exhibit high flow divergence—indicating minimal contribution to effective information transmission—are removed, thereby retaining the most information-preserving structures. In the second stage (layer-level pruning), entire layers with limited impact on overall information propagation are pruned while maintaining an acceptable trade-off between model compactness and accuracy.

The unified framework, IDAP++, implements these two compression stages under a controlled accuracy budget. Experiments across multiple architectures (CNNs, Transformers, hybrid models) and datasets demonstrate that IDAP++ achieves state-of-the-art compression performance, significantly reducing FLOPs, model size, and inference latency with only marginal accuracy loss.

### Strengths
- The paper is clearly written and well-structured, making it easy to follow. The integration of theoretical derivations with algorithmic descriptions and pseudocode effectively bridges intuition and implementation.

- The introduction of the information flow divergence metric provides a rigorous mathematical framework for quantifying information propagation within deep neural networks.  The concept of divergence is explicitly formulated for the most fundamental types of layers of neural networks including FC, CNN and self-attention layers.

- The proposed Unified two-stage compression IDAP++ method combines filter-level and layer-level pruning under a common theoretical criterion, achieving multi-granular compression in a unified process.

- The experimental evaluation is extensive, covering diverse architectures (e.g., CNNs, Transformers, hybrid models) and a wide range of application domains (computer vision, image classification, object detection, image segmentation, and NLP). Results consistently demonstrate strong compression performance with minimal accuracy degradation.

- The framework appears architecture-agnostic and could be applied broadly across model families, indicating potential impact beyond the tested configurations.

### Weaknesses
- Some key mathematical statements—such as Lemma 1, Lemma 2, and Theorem 1—are presented without formal proofs or sufficient derivations. This omission weakens the theoretical rigor of the proposed information flow divergence formulation and leaves open questions about the validity and generality of the results. Providing at least proof sketches or detailed references would enhance credibility.

- The process of testing multiple configurations to find the optimal pruning ratio $\rho^*$ likely requires repeated retraining or validation passes, which could significantly increase compression time. The paper should quantify this overhead.

- It remains unclear whether the Adaptive Replacement Strategy is complementary to divergence-guided pruning or primarily compensates for its aggressive weight removal. An ablation isolating each mechanism’s contribution would strengthen the empirical argument.

- The replacement of layers using Identity* or projection mappings may alter gradient flow, especially in networks with batch normalization or residual branches. The stability of training during these structural substitutions is not theoretically or empirically studied.
 
- The “error-driven” selection metric (line 264) is conceptually appealing but under-specified. It is unclear whether it measures local loss change, gradient magnitude, or validation error difference—and how it interacts with divergence-based pruning decisions.

Overall, the paper would benefit from stronger theoretical grounding and empirical validation. The divergence metric, while intuitively appealing, lacks rigorous theoretical justification. The study also presents limited ablation and robustness analyses, leaving uncertainty about parameter sensitivity and stage-wise contributions. Moreover, the framework’s computational overhead and scalability to very large models remain insufficiently quantified. There are notable fairness and reproducibility gaps in comparisons with baseline methods, and the paper provides minimal qualitative or interpretability analyses, especially for generative architectures.

### Questions
1) How does the proposed information flow divergence formally relate to established measures such as mutual information or Fisher information? Can it be theoretically justified as a valid proxy for layer or filter importance?

2) How does the divergence metric behave in the presence of non-differentiable components (e.g., ReLU, max pooling, attention masking)? Does it remain stable or require smoothing approximations?

3) Theorem 1 provides a relative deviation bound but not a convergence or optimality proof. Under what conditions does this bound hold, and is the resulting compression guaranteed to be near-optimal?

4) How much of the overall performance gain arises from Stage 1 (filter pruning) versus Stage 2 (layer truncation)? Have you conducted ablation studies to isolate their respective impacts?

5) How was the non-linear schedule $\rho_k = f(k, \alpha)$ chosen? Is there a theoretical or empirical rationale for its shape (e.g., exponential vs. polynomial growth), and how sensitive is performance to $\alpha$?

6) Would a moving-average or confidence-based threshold be more stable than a fixed (a predefined threshold $\tau$)?

7) How many configurations are typically evaluated to determine $\rho^*$? Could this search be approximated or learned adaptively (e.g., via reinforcement learning or Bayesian optimization) to reduce computational cost?

8) How does the $\text{Identity}^*$ mapping handle mismatched channel dimensions or incompatible tensor shapes in architectures with skip connections or multi-branch structures? Is the projection mapping trainable or fixed?

9) Does error-driven selection of $\delta_E$ depend on batch size or data variability, and how does it correlate with actual validation loss reduction?

10) What is the additional runtime introduced by recomputing flow divergence, fine-tuning adjacent layers, and evaluating $\delta_E$?

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
4

### Summary
This paper introduces IDAP++, a two-stage pruning framework grounded in information flow divergence. The method first removes redundant filters/heads, then prunes entire layers based on their contribution to information flow. This framework generalizes across diverse architectures (CNNs, transformers) and is validated on various vision and language tasks. Extensive experiments show substantial reductions in parameters, FLOPs, and inference time, while maintaining competitive accuracy against state-of-the-art methods.

### Strengths
- The paper is well-written and clearly structured.
- The paper proposes a novel, mathematically grounded framework that uniquely combines both filter-level and layer-level pruning, setting it apart from much of the heuristic-driven prior work.
- A large number of experiments on different models and datasets in the paper have proved the effectiveness of IDAP++.

### Weaknesses
- There is no detailed profiling of actual computational overhead (e.g., for models with hundreds of layers or very high input/output dimensions), especially during repeated recomputation after every pruning phase.
- This paper lacks ablation experiments and more comprehensive comparisons with the baseline.
- The paper proposes a unified framework but employs two different types of metrics: a norm-based metric for filter pruning and a difference-based metric for layer pruning. The paper does not explain the rationale for this divergence.

### Questions
- Why are different metrics used for filter pruning and layer pruning? What is the theoretical justification for this?
- Can the authors provide a more granular ablation study quantifying the independent effects of filter-level and layer-level pruning?
- What are the concrete computational overheads for flow-divergence computation and recomputation, particularly for larger transformer models? And the paper only compared the inference speed of the base model and IDAP++ pruned model but avoided comparing it with methods such as LTH and RigL. 
- The paper mentions that removing the filter first and then the layer is empirical. I would like to see more discussions on this aspect and ablation experiments.

### Soundness
2

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
3

### Summary
This paper introduces IDAP++, a theoretically grounded, two-stage neural network compression framework that jointly prunes models along both width (filter-level) and depth (layer-level) dimensions. The key contribution lies in formalizing deep networks as continuous dynamical systems and proposing the Information Flow Divergence (IFD) metric, which quantifies how information evolves as it propagates through the network.The first stage, Iterative Divergence-Aware Pruning (IDAP), removes redundant filters while preserving essential information pathways, whereas the second stage, Flow-Guided Layer Truncation, eliminates entire layers with minimal contribution to the global information flow. The unified optimization process integrates adaptive thresholding, flow recomputation, and fine-tuning under an accuracy budget constraint.The framework is evaluated extensively across diverse architectures—including CNNs, Vision Transformers, object detection and segmentation models, generative architectures (VQGAN, Stable Diffusion), and NLP models (BERT, GPT-2, T5)—and on multiple benchmark datasets.

### Strengths
1.	Theoretical Novelty and Mathematical Rigor: The paper’s greatest contribution is its rigorous theoretical foundation. By modeling neural networks as continuous dynamical systems and introducing the Information Flow Divergence metric, the authors provide a principled and architecture-agnostic means to quantify information propagation. This formulation bridges information theory and dynamical systems analysis, moving beyond empirical heuristics commonly used in pruning literature. The inclusion of mathematical properties (e.g., scale invariance and additive composition) and derivations for different layer types (fully connected, convolutional, and self-attention) adds strong theoretical depth.
2.	Holistic Compression Strategy: The proposed two-stage optimization—filter pruning followed by layer truncation—represents a meaningful step toward multi-level structural optimization. The idea that pruning along width simplifies subsequent depth optimization is well-motivated and empirically validated. The framework’s adaptive thresholding, flow rebalancing, and fine-tuning mechanisms form a coherent, unified pipeline that efficiently balances sparsity and performance.
3.	Comprehensive Empirical Validation: The experimental section is notably broad and thorough. Evaluations span vision, language, and generative domains, confirming the generality of the approach. For example, the method achieves a 75% FLOPs reduction on ViT-Base/16 with only 1.6% accuracy loss, and a 67–69% parameter reduction for large NLP models such as BERT and GPT-2. The inclusion of comparisons with multiple strong baselines and the public release of code further enhance reproducibility and credibility.

### Weaknesses
1.	Computational Overhead and Practicality: While the paper claims that flow computation has O(L) complexity, the full pipeline involves iterative divergence evaluation, fine-tuning at multiple stages, and layer-level recomputation, which could lead to significant computational overhead for very large models (e.g., GPT-scale). The paper would benefit from a quantitative comparison of latency and resource costs against the baseline, such as LTH and RigL.
2.	Hyperparameter Sensitivity and Usability: The framework introduces several critical hyperparameters—such as the target accuracy drop ∆max, Pruning hyperparameters α&β. Although default ranges are provided, the paper lacks a systematic sensitivity analysis. 
3.	Limited Discussion of Limitations and Failure Cases: While the paper’s results are strong overall, certain models (e.g., generative models and NLP architectures) exhibit higher performance drops (up to 5% accuracy / 9% FID degradation). These cases are acknowledged but not deeply analyzed. A more detailed discussion of why flow-based pruning may struggle in these settings—perhaps due to noise amplification or information bottlenecks—would strengthen the work’s completeness.

### Questions
1.	The divergence measure ( D(s) = \frac{d^2T}{ds^2}(s) \cdot (\frac{dT}{ds}(s))^\top ) is central to the framework. Could the authors provide **further intuition** for why this *second-derivative–based* formulation effectively identifies redundancy? How does it capture information loss differently from first-order measures such as gradient magnitude or signal variance?
2.	The paper claims that flow computation scales with O(L) complexity, yet the overall pipeline involves iterative divergence evaluation, fine-tuning, and recomputation at multiple stages. Could the authors provide **quantitative comparisons** of runtime and memory costs between IDAP++ and the pruning baselines, such as LTH and RigL?
3.	The framework introduces several key hyperparameters, including the pruning aggressiveness coefficients (α, β) and the target accuracy drop (∆max). How **sensitive** is the method to these choices across different architectures?

### Soundness
3

### Presentation
3

### Contribution
3
