# Fine-Grained Activation Steering: Steering Less, Achieving More

- Decision: Accept (Poster)
- Scores: 4, 4, 6

## Abstract
Activation steering has emerged as a cost-effective paradigm for modifying large language model (LLM) behaviors. Existing methods typically intervene at the block level, steering the bundled activations of selected attention heads, feedforward networks, or residual streams. However, we reveal that block-level activations are inherently heterogeneous, entangling beneficial, irrelevant, and harmful features, thereby rendering block-level steering coarse, inefficient, and intrusive. To investigate the root cause, we decompose block activations into fine-grained atomic unit (AU)–level activations, where each AU-level activation corresponds to a single dimension of the block activation, and each AU denotes a slice of the block weight matrix. Steering an AU-level activation is thus equivalent to steering its associated AU. Our theoretical and empirical analysis show that heterogeneity arises because different AUs or dimensions control distinct token distributions in LLM outputs. Hence, block-level steering inevitably moves helpful and harmful token directions together, which reduces efficiency. Restricting intervention to beneficial AUs yields more precise and effective steering. Building on this insight, we propose AUSteer, a simple and efficient method that operates at a finer granularity of the AU level. AUSteer first identifies discriminative AUs globally by computing activation momenta on contrastive samples. It then assigns adaptive steering strengths tailored to diverse inputs and selected AU activations. Comprehensive experiments on multiple LLMs and tasks show that AUSteer consistently surpasses advanced baselines while steering considerably fewer activations, demonstrating that steering less achieves more.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes AUSteer, a method for fine-grained activation steering in large language models. Instead of steering at the block level, AUSteer operates at the Atomic Unit (AU) level, corresponding to individual activation dimensions. The authors show that block-level activations mix helpful and harmful components, making coarse interventions inefficient. AUSteer identifies discriminative AUs through an activation momentum metric computed from contrastive pairs and applies adaptive per-AU scaling. Experiments on seven benchmarks and three model families demonstrate consistent improvements over block-level steering with far fewer activations, suggesting that steering less can achieve more.

### Strengths
The problem is clearly defined and relevant. The idea of decomposing block activations into AUs is intuitive and well motivated. AUSteer is simple, interpretable, and does not require retraining. The experiments are broad and consistent across tasks and models, and the analysis convincingly shows heterogeneity within block activations.

### Weaknesses
1) Efficiency claim lacks evidence:

The paper’s argument that a smaller steering footprint improves efficiency is not empirically verified. No inference-time or computational measurements are provided, and efficiency is used only in a representational sense.


2) Lack of comparison with broader control variants. 

The paper assumes that steering only a subset of AUs is inherently superior, but does not test a broader or fully generalized steering scheme where all AUs are jointly optimized or selectively weighted. Without such a comparison, it remains unclear whether partial AU control offers unique advantages beyond being a constrained version of more general steering

### Questions
1) Does efficiency refer to computational speed or representational precision?
2) Have you measured inference cost, latency, or stability?
3) Would steering all AUs with selective suppression perform similarly?
4) How scalable is activation momentum computation for very large models?

### Soundness
3

### Presentation
3

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
This paper introduces AUSteer, a novel fine-grained activation steering method for LLMs that operates at the atomic unit level rather than the traditional block level (e.g., attention, FFN, or residual blocks). The authors identify a key limitation in existing steering methods: block-level activations are heterogeneous, mixing beneficial, irrelevant, and harmful components. As a result, conventional approaches (like CAA, SADI, or ITI) that steer all dimensions of a block simultaneously are coarse, inefficient, and potentially harming model performance.

To address this, AUSteer decomposes each block into fine-grained AU-level activations, where each AU corresponds to a single column of the weight matrix and each activation is a scalar. The method consists of two main components:

-AU Localization via Activation Momentum: A metric that measures the discriminative power of each AU across positive and negative contrastive samples. It identifies which AUs consistently promote or suppress desirable activations.
-Adaptive Steering : Instead of applying a fixed vector, AUSteer adjusts steering strength per input and per AU, scaling the intervention by the activation’s current value and discriminative score.

Experiments are conducted on various LLMs (LLaMA2, Gemma2, Qwen3) and tasks, including commonsense reasoning, math problem-solving, and open-ended generation.

### Strengths
- Clearly identifies a fundamental issue: heterogeneity in block activations—and systematically decomposes it into atomic units.

- Introduces the concept of activation momentum to measure discriminative importance without training.

- Extensive experiments across three model families (LLaMA, Gemma, Qwen) and multiple tasks (reasoning, math, safety, alignment).

- No retraining or fine-tuning required.

- Ablation studies isolate the contribution of both components.

### Weaknesses
- The formal derivation connecting activation momentum to discriminative causality is unclear.

- AUSteer requires carefully curated positive–negative pairs, which may not be available or trivial to construct for all tasks.

- While steering itself is efficient, computing activation momentum across many AUs and samples may still be computationally intensive for very large models.

- Hyperparameter sensitivity is unclear and needs further demonstrations and explanations.

- One wonders what is the runtime overhead for AU localization and steering per sample compared to block-level methods like SADI?

### Questions
Please see the weaknesses.

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
In this paper, they present AUSteer, a more fine-grained activation steering technique, to control LLM's behavior during the inference time. First, they recognize the heterogeneity in block activation and explain this through comprehensive experiments. Inspired by these experiments, they developed a more fine-grained activation steering algorithm. In detail, first, use activation momentum to recognize the important atomic unit on the target tasks. Then, steer these atomic units' activation adaptively. They did comperihensive experiments to evaluate AUSteer. And the results are convincing.

### Strengths
1. They first use two sections to recognize and interpret the heterogeneity in block activation, which gives insight and inspiration for AUSteer.
2. The method is natural and effective. 
3. The experiments are comprehensive, spanning three LLMs with different architectures and three different tasks.

### Weaknesses
1. The biggest model used is 27B. Evaluating AUSteer on bigger models, e.g., 32B and 72B, and sparse models, e.g., MoE, even multi-modal models would be better.
2. The optimal hyperparameters \alpha and k are task-specific; how to set the hyperparameters for every tasks? And what is the hyperparameters used in Table. 1?

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
3
