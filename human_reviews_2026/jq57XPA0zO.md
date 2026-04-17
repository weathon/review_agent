# NdLinear: Preserving Multi-Dimensional Structure for Parameter-Efficient Neural Networks

- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
In deep learning, processing multidimensional inputs (e.g., images, medical scans, and time series) is an important task that often requires flattening the inputs. We introduce *NdLinear*, a drop-in replacement for linear layers that operates directly on tensors, requiring no flattening. By applying transformations separately along each dimension, NdLinear preserves native data structure while achieving dramatic parameter reductions, often by orders of magnitude, with minimal memory overhead. We prove NdLinear maintains expressivity through structured Tucker decomposition while preserving VC-dimension scaling. Extensive experiments demonstrate NdLinear's capacity to achieve significant parameter reductions with substantial wall-clock efficiency gains and minimal memory overhead. For instance, our *NdLinear-LoRA* matches or exceeds standard LoRA on language reasoning tasks using up to $9\times$ fewer parameters. Experiments across CNNs, RNNs, Transformers, and MLPs on vision, language, time-series, and tabular tasks consistently demonstrate NdLinear's efficiency gains. While excelling at axis-separable tasks, NdLinear has limitations with entangled spatial interactions. By processing data in its original N-dimensional form, NdLinear provides a theoretically grounded, practical component for building more efficient neural architectures.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces NdLinear, an alternative to standard linear layers that operates on N-dimensional tensors by applying sequential, dimension-wise linear transformations. The method achieves dramatic parameter and computational efficiency, with some theoretical analysis about VC-dimension provided. Analysis and controlled studies show that NdLinear excels with axis-separable data but degrades when faced with highly entangled cross-dimensional patterns. The paper conducts extensive experiments across a variety of architectures (CNNs, RNNs, Transformers, DiTs, MLPs) and domains (vision, language, time-series, tabular data), where NdLinear typically matches or outperforms baseline models with significantly fewer parameters and lower resource consumption.

### Strengths
- Overall, the idea of providing an approach that takes into account the input geometry (a tensor structure) is interesting and worth further investigation.

- The NdLinear layer achieves a dramatic reduction in parameter count, up to several orders of magnitude, while maintaining or even surpassing the accuracy of standard linear layers across a diverse range of tasks and data modalities.

### Weaknesses
- The paper claims that NdLinear is a native operator on tensors by drawing connections to mode-k tensor–matrix products and rank-1 Tucker decomposition. However, the paper does not clearly demonstrate how these connections relate to the proposed method or how they influence the effectiveness and characteristics of NdLinear.

- The theorems provided in the paper (both in the main text and the supplementary material) are stated and proved ambiguously. Specifically, they lack precise definitions (e.g., VC-dimension), omit references to prior results that are used (e.g., Line 1042), and do not provide clear insights into the meaning or implications of each theorem.

- The experiments offer limited insight into why the proposed method succeeds or fails on each task.

- While the method claims to dramatically reduce the number of parameters, the actual memory usage and training time are not improved. This limitation reduces the practical applicability of the method.

- In some benchmarks and tasks, the evaluations are thorough (e.g., Table 4 for CNNs with TRL/TCL and TT baselines). However, in more challenging or entangled-structure settings (e.g., generic image classification or dense MLPs for non-axis-aligned data), the empirical evidence for when to prefer NdLinear over alternatives such as mixture models, block-sparse methods, or structured matrix approaches is missing.

### Questions
- Is there a comparison with continuous structured matrix bases in practical settings (e.g., Potapczynski et al., *NeurIPS* 2024)? The authors should report empirical comparisons or explain why such baselines are not applicable.  

  [1] Potapczynski, A., et al. "Searching for efficient linear layers over a continuous space of structured matrices." *Advances in Neural Information Processing Systems* 37 (2024): 3857–3881.
- Could the authors present the theoretical results more rigorously and thoroughly? In particular, please (i) state all definitions and assumptions clearly, (ii) state prior results used in the proofs, and (iii) provide intuition or discussion on the practical implications of each theorem.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes NdLinear, a tensor–matrix linear transform that preserves N-D structure and greatly reduces parameters.

### Strengths
- The paper does a great job testing NdLinear in a wide range of settings, from small MLPs with thousands of parameters to large-scale Transformer models with billions of parameters, which is very important for a work that aims to improve such a fundamental building block as the linear layer. These experiments show that NdLinear isn’t just a niche idea, but a flexible, general-purpose replacement for standard linear layers. It’s nice to see that it performs consistently well (or at least on par) across such different types of models and data.

- The paper thoughtfully explores several theoretical dimensions of NdLinear, not limiting itself to empirical validation. It includes a VC-dimension analysis showing that the model’s capacity scales comparably to dense layers, an expressivity discussion analyzing how mode-wise factorization influences representational power, and a complexity analysis providing clear asymptotic comparisons for both parameters and computation. In addition, the authors discuss bias propagation and rank interpretation, explaining how separability and $1$-homogeneous activations help preserve functional structure. Altogether, these components indicate a well-rounded theoretical grounding that complements the empirical results.

### Weaknesses
- Although the paper discusses VC dimension and provides some intuition about expressivity, it does not rigorously compare the function class of NdLinear to that of fully dense layers. It remains unclear how much representational capacity is lost when enforcing mode-wise separability, or under what conditions NdLinear can approximate arbitrary linear maps.

- The paper includes one quantitative test, which is the “Dial-a-Bias” experiment, which links the separability assumption to performance as the data varies from separable to entangled. However, other theoretical ideas like rank preservation, bias propagation, and stability effects of 1-homogeneous activations are only discussed qualitatively with no direct experiments verifying them.

- The empirical section compares mainly to LoRA and baseline dense layers. However, stronger baselines such as TRL, TT, MONARCH, or Kronecker-structured linear layers are not included, making it difficult to gauge whether NdLinear actually outperforms prior structured approaches rather than merely matching them.

- The paper does report runtime and memory results, but only on small to medium tasks like CIFAR CNNs and MLPs, where the overhead is minimal, around 0.6-1.6% extra training time and 1-2% more memory. That’s good to see, but there aren’t any measurements for large-scale models such as Transformers or LLMs. Since NdLinear works through sequential per-dimension operations, it’s unclear whether the same efficiency holds at billion-parameter scale, especially under distributed training. It would be more convincing if the authors included runtime profiling on large models to confirm that the small overhead still applies.

### Questions
- Please provide a more rigorous comparison between the function class of NdLinear and that of a standard dense layer. In particular, under what conditions can NdLinear approximate an arbitrary linear map, and what expressivity (if any) is lost due to the mode-wise separability constraint?

- Please consider adding an experiment that empirically verifies one of the key theoretical claims, for example, by measuring rank changes or activation correlations across layers. Given the limited time of the rebuttal phase, even a single well-designed experiment would be sufficient to strengthen the paper’s theoretical support.

- Please consider adding results against stronger structured baselines such as TRL, TT, MONARCH, or Kronecker-structured layers, to more clearly position NdLinear within the broader family of structured linear operators.

- Please include runtime profiling or throughput measurements on large-scale models to demonstrate that the reported low overhead holds in practice.

### Soundness
4

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces NdLinear, a structured linear layer that operates on N-dimensional tensors without flattening. By performing sequential dimension-wise linear transformations, NdLinear preserves the multi-dimensional structure of input data, reducing parameter counts and computational complexity wrt the standard linear layer. The authors provide theoretical analysis showing that expressivity is preserved via VC-dimension scaling. Experimental results across a wide variety of architectures and application domains demonstrate comparable or improved performance while reducing trainable parameters as well as time complexity and memory consumption.

### Strengths
- The approach is simple, intuitive, and well-motivated, addressing a clear inefficiency in handling multi-dimensional data across neural architectures.
- Broad and comprehensive evaluation spanning diverse domains (language, vision, time-series, tabular) and model types (CNNs, RNNs, Transformers, MLPs), including large-scale models up to 8B parameters.

### Weaknesses
- The concept of applying separate linear transformations for each dimension is not novel; for example, [1] employs a similar approach in LoRA. Please compare your approach, especially in the context of fine-tuning tasks, to clarify the novelty or advantages.
- The experiment section needs a centralized summary table or section detailing key configurations across tasks, such as input tensor dimensionality, precise architecture modifications. This would greatly aid in comprehending the diverse experiments.
- It seems that the experiments primarily focus on low-dimensional data (mostly 2D/3D tensors like images or time-series). Evaluating on higher-dimensional inputs (e.g., 4D/5D/... tensors from video data or scientific simulations) would strengthen claims of generality for N-D data.
- Figure 1 could be improved for clarity: use shaded areas to represent standard deviation instead of error bars, and include more alpha values (e.g., finer granularity like 0.05 steps) to better visualize the transition in performance between NdLinear and dense MLPs. Additionally, provide explicit definitions of f_separable and f_entangled in the main text for better understanding.

Reference:

[1] Si et al. Maintaining Structural Integrity In Parameter Spaces for Parameter Efficient Fine-tuning. ICLR 2025.

### Questions
Please refer to weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The main idea of the paper is to introduce a new type of linear layer that directly operates on N-dimensional tensors instead of flattened vectors. The proposed approach performs sequential, dimension-wise linear transformations, allowing neural networks to preserve the inherent multi-dimensional structure of their input data.

The authors show that NdLinear drastically reduces parameter counts, from quadratic to linear growth in the number of tensor dimensions while maintaining expressivity through Tucker decomposition and matching the VC-dimension scaling of standard linear layers. 

The authors then suggest an extension, NdLinear-LoRA, that demonstrates that the proposed module can also serve as a plug-in replacement for LoRA layers.

### Strengths
Novelty - The odea of replacing the flattening option is new, and the authors offer a theoretically grounded and elegant alternative.

Mathematical rigor - The theoretical analysis includes clear formalization of mode-wise tensor transformations, and proofs via VC-dimension bounds, and explicit computational complexity comparisons.

Extensive empirical validation - Experiments cover a broad landscape (LLMs, CNNs, ViTs, RNNs, DiTs, and MLPs) with over 20 datasets. The improvements are consistent

Insightful inductive bias analysis - I liked the “dial-a-bias” experiment which clearly quantifies the advantage of NdLinear is advantageous

Reproducibility - The paper includes complete algorithmic pseudocode, detailed appendices, and a reproducibility statement with code availability in the appendix.

### Weaknesses
Theory - The proofs are quite short and trivial. The approximation error and convergence approximations are not sufficiently clear.
Experiments - Some graphs (Fig 1, for example) are not so professional and convincing, while using few number of samples.
Comparison - the comparison to existing baselines seems without optimal tuning or using the best state-of-the-art versions.

### Questions
In Theorem C.1, what are the bounds for a fixed d? What is the convergence rate?

### Soundness
2

### Presentation
1

### Contribution
2
