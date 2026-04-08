## Human Reviewer 1

### Summary
WASI is a novel method that applies subspace-based training to Transformer models, primarily aiming for highly efficient on-device fine-tuning by restricting parameter updates to a low-rank subspace that captures the model's essential information. The method overcomes the memory bottleneck of backpropagation and decreases inference latency in transformer models important for edge devices. The results show that WASI maintains accuracy comparable to vanilla training while reducing memory usage by up to 62× and computational cost (FLOPs) by up to 2×. On a Raspberry Pi 5, WASI achieves roughly 1.5× faster training and inference compared to vanilla training.

### Strengths
WASI solves an important problem, which is the memory bottleneck of backpropagation. Previous parameter-efficient methods often ignored the activation memory footprint, which scales linearly with batch size and context length. This comes from the fact that WASI iteratively maintains a subspace that efficiently handles both the weights and the intermediate activations.

The method has been shown to reduce memory usage by up to 62 times compared to vanilla training. This is crucial for enabling on-device fine-tuning of large models with severely limited RAM.

By operating within a low-rank subspace, the overall number of floating-point operations (FLOPs) required for both forward and backward passes is significantly reduced. Research reports up to a 2x reduction in computational cost, leading to demonstrably faster training and inference speedups on constrained hardware (e.g., 1.5x faster on devices like the Raspberry Pi).

Unlike methods that aggressively prune models, WASI aims to preserve the model's learning capacity by training within a space believed to contain the essential information. The key strength is maintaining task accuracy comparable to vanilla training, which means the huge efficiency gains do not come at a major performance cost.

I find the writing is clear and the paper is well structured. 

The experiment on Raspberry PI is good. 

The choice of baseline models, such as VIT, SwinT, on a couple of datasets is satisfactory.

### Weaknesses
The method is novel yet incremental. The fundamental idea of restricting model updates to a low-dimensional subspace is established. We see this in earlier methods like Stochastic Weight Averaging (SWA) and subsequent research focused on finding paths or subspaces of high-accuracy models.

The major gains in efficiency are obtained when compared to vanilla and ASI methods. The gains compared to SVD-LLM are minimal, i.e, only for memory consumption during training. 

No power consumption results on the device.  

The results are shown to work for seed 233. It is mentioned that the variance between the results for different seeds is not high, but there are no results showing that.

### Questions
What was the variance between the results with diffence seeds? What is average of results if you try 5 different seeds.

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
4

---

## Human Reviewer 2

### Summary
The paper proposes WASI, a weight–activation subspace iteration framework that trains and runs transformers entirely in a learned low-rank subspace. WASI couples (i) Weight Subspace Iteration (WSI)—initial SVD with explained-variance threshold ε and iterative subspace updates—with (ii) a redesigned Activation Subspace Iteration (ASI) (dynamic-programming rank selection, 3D/4D activations).

### Strengths
+ Unlike LoRA-style PEFT that re-inflates at inference, WASI keeps both weights and activations in low rank throughout forward/backward, with explicit FLOPs/memory formulas and subspace-space equations, yielding predictable savings end-to-end. 
+ The paper motivates stable layer ranks during fine-tuning  and shows singular-value/rank stability, enabling subspace iteration instead of per-step SVD; empirically WSI dominates repeated SVD in FLOPs/accuracy.
+ The on-device results seems interesting.

### Weaknesses
- The edge device used in this paper is only Rasperry pi 5. More edge devices like Jeston Nano should be included and evaluated.

- The final accuracy should be highlighted and reported with Tables. Now it is hard to find it in the draft.

- Main comparisons focus on MLP/linear blocks for fairness with baselines; attention-layer coverage is deferred to appendix. Non-IID, partial-participation, and straggler/client-heterogeneity studies (key for on-device contexts) are not central, leaving external validity to real deployments somewhat open.

### Questions
- Can you provide a micro-benchmark of one WASI iteration (basis update, orthogonalization, matmuls) on Pi-5 vs. GPU to attribute the speedups precisely?

- Can you adopt more concrete edge devices for expriments?

- How sensitive are results to ε and the number of subspace-iteration steps? Any auto-tuning strategy that targets a latency or memory cap directly?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 3

### Summary
This paper proposes Weight–Activation Subspace Iteration (WASI), a method that performs model training entirely within a low-rank subspace of both weights and activation. Experimental results demonstrate that WASI significantly reduces memory usage and roughly halves the computational cost (FLOPs), making on-device training feasible even on constrained hardware such as the Raspberry Pi.

### Strengths
- The paper is clearly written and easy to follow.

- The proposed method can be applied to various architectures, including ViT, Swin Transformer, and TinyLlama.

- Experiments convincingly show that WASI drastically reduces memory consumption and FLOPs while maintaining comparable accuracy to full fine-tuning.

### Weaknesses
- The core ideas, low-rank approximation of parameters and activations via subspace iteration, have been explored in prior work. Thus, the novelty is somewhat limited, and the main contribution is largely engineering integration rather than conceptual innovation.

- The ablation study indicates that the method can be sensitive to certain hyperparameters and datasets, which may affect stability.

- Experiments are conducted on relatively small datasets, which could lead to higher variance and limit generalization.

- Minor: The figures could be improved, for example, by labeling the ε thresholds more clearly to make the trends easier to interpret.

### Questions
- Could the authors evaluate the proposed method on a larger-scale dataset, such as training from scratch on ImageNet-1K? Even if direct on-device testing is infeasible in this scenario, a simulated experiment on full GPUs would provide stronger evidence of the method’s scalability and the effectiveness of the low-rank approximation at large scale.
- Is there a principled way to choose the ε  other than grid sweeping? For example, can it be adaptively determined or estimated from training dynamics?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3