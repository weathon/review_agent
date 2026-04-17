# PartInfer: Enabling LLM Inference On Edge Devices

- Decision: Reject
- Scores: 2, 4, 2, 4

## Abstract
Large Language Models (LLMs) have demonstrated remarkable capabilities across a range of Natural Language Processing (NLP) tasks, but their high computational and memory demands pose significant challenges for deployment on resource-constrained edge devices. Existing approaches to model compression and optimization often rely on coarse-grained pruning or quantization, which can compromise accuracy or require re-training and fine-tuning. In this work, we introduce PartInfer, a neuron-level optimization framework that enables efficient LLM inference on edge devices by exploiting the task-specific activation patterns of neurons. By profiling and identifying both task-specific and general-purpose neurons using an offline LLM profiler, PartInfer implements two key optimizations: Partial Loading, which reduces memory footprint by loading only a subset of neurons that were identified to be most important during the offline stage, and Partial Computation, which dynamically computes only the most relevant neurons at runtime. Evaluation across multiple NLP tasks shows that PartInfer achieves significant reductions in memory footprint and computation while preserving task performance, making it a practical step towards enabling LLM deployment on edge devices.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces PartInfer, a method for accelerating LLM inference through adaptive sparse activation. PartInfer is composed of two components, *partial loading* and *partial computation*. *Partial loading* loads only a subset of neurons to reduce memory burdens. Then *partial computation* selects important neurons within the loaded subset for computation, yielding a more efficient computation. PartInfer predicts the critical neurons with both offline statistics and online top-k selection during prefilling, which accelerates the LLM generation while maintaining the performance. Experimentsal results across various tasks validate the effectiveness of the strategy.

### Strengths
1. Intriguing observations regarding sentence semantics: The observations regarding the common neurons and task-specific neurons are insightful.
2. Improved neuron selection: The 2-step approach, *i.e.* partial loading and partial computation, reduce the memory burden and inference cost simultaneously.
3. Boosted efficiency: PartInfer can achieve non-trivial inference speedups on real devices while maintaining accuracy on various tasks.

### Weaknesses
1. **Generalization concern**: The major concern is the generalization ability of the proposed method. For long prefilling contexts or multi-turn dialogue, it is difficult to ensure the dynamic neurons are relevant. The computation cost for sorting neurons is also nonnegligible for long prefilling contexts. From my point of view, this method might not be suitable for general purpose, but applicable for specific tasks like stream LLM, which requires short context only.
2. **Insufficient experiment**: 
   - Limited benchmarks against state-of-the-art static and dynamic pruning methods, such as PowerInfer. 
   - Tasks like WinoGrad, ARC, MMLU, and language modeling perplexity are important benchmarks for the research in inference speedup.
   - Furthermore, given the claim that this paper focus on Llama family, it is expected to conduct experiments on more model sizes and discuss the generalization ability on other model types, such as Gemma, Qwen, etc.
   - Lack of comparison with other methods .on the decoding speed.
3. **Constraint on MLP**: This method is constrained on MLP block. However, as the development of dynamic activation such as MoE, the process of predicting critical neurons might be native to the latest architectures. A detailed discussion on the necessity of PartInfer is critical for assessing this paper.

### Questions
1. **Inappropriate metric**: The paper presents the normalized metric in its experiments. How is the metric computed? Why not setting the dense model performance as 1.0? The experimental results are much confusing and reduce confidence in its advantages.
2. **Hyperparameters**: How is the ratio set in Table 1? Why the overlap is counted on 40% critical neurons in Figure 1&2? There are too many magic numbers in this paper which reduce the readability.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes PartInfer, a framework designed to optimize LLM inference on edge devices by leveraging offline analysis of neuron activation patterns. The method claims to distinguish between model-specific (cross-task general) and task-specific (task-dependent) neurons. During online inference, it employs Partial Loading to reduce memory usage and Partial Computation to lower compute cost. The authors report a 13× speedup when deploying Llama 3.2–3B on an NVIDIA Jetson device.
However, the paper has several critical weaknesses. First, it lacks essential ablation studies: while it demonstrates that computing fewer neurons increases inference speed, it completely omits any accuracy–efficiency trade-off analysis. As a result, the chosen configuration (e.g., 40% computation) appears arbitrary and unsupported. Second, the experimental comparisons are limited and unfair—the claimed 13× improvement is measured only against disk offloading, which is an extremely weak baseline. The absence of comparisons with standard SOTA methods such as 4-bit quantization (GPTQ/AWQ) or pruning makes it difficult to assess the true effectiveness of PartInfer. Finally, all empirical findings are based solely on two models from the Llama 3.2 family, raising concerns about generality across architectures and tasks.
Overall, while the idea of leveraging neuron-level analysis for adaptive inference is interesting and relevant to the ICLR community, the paper’s experimental validation and comparative rigor fall short of publication standards.

### Strengths
a) Significance of the Problem: This paper tackles the highly challenging yet practically significant problem of deploying large language models (LLMs) on edge devices with limited memory and computational capacity. This topic is perfectly aligned with the core interests of the ICLR community.
b) Decoupled Optimization: The proposed PartInfer framework decouples memory optimization (via partial loading) from computation optimization (via partial computation). Theoretically, this design allows flexible trade-offs according to device constraints and application requirements.
c) Empirical Deployment Validation: The authors report successfully deploying a Llama3.2-3B model—previously unable to run due to insufficient memory—on an NVIDIA Jetson Orin Nano (8GB) device, achieving a 13× speedup compared to disk offloading.

### Weaknesses
Implementation Ambiguity: In the Transformer architecture, the Feed-Forward Network (FFN) layer contains multiple weight matrices (e.g., gate_proj, up_proj, down_proj in Llama). However, the paper does not clarify what “loading only a subset of neurons” means in engineering terms. It remains unclear whether this requires modifying the computation graph, dynamically slicing the weight tensors, or implementing custom CUDA kernels to support unstructured sparse computation.
Missing Key Ablation Studies: The core claim of the paper is that it can reduce memory and computation while maintaining task performance. However, no key ablation studies are provided to substantiate this claim. Figures 9 and 10 show that loading or computing fewer neurons leads to faster inference, but there is no analysis of how accuracy changes with respect to the percentage of computed neurons or the percentage of loaded neurons.
Insufficient and Unfair Baseline Comparisons — Missing SOTA Methods: The paper cites quantization and pruning as primary competing approaches in the introduction, yet Sections 6.4–6.6 do not compare PartInfer against any standard quantization (e.g., 4-bit GPTQ or AWQ) or pruning baselines. Furthermore, the paper uses disk offloading as the only speed comparison baseline, which is an extremely weak (even strawman) choice. While PartInfer (70% loading, 40% computation) achieves 9.85 tokens/s, it is not compared against other sparse inference approaches (e.g., CoreInfer or PowerInfer) or even a runnable dense baseline (e.g., a 1B model). As a result, the reported 13× speedup lacks meaningful context.
Extremely Limited Model and Architecture Coverage: As noted by the authors themselves, all conclusions in this paper are drawn solely from experiments on two models in the Llama 3.2 family (1B and 3B). It remains entirely unknown whether the proposed notion of model-specific and task-specific neurons generalizes to other architectures such as Mistral, Gemma, or Mixture-of-Experts (MoE) models. This narrow experimental scope substantially weakens the generality of the conclusions. Although the authors mention this as future work, such a limitation is quite significant for an ICLR submission.

### Questions
For specific details, see the weaknesses section.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper addresses a critical challenge in LLM deployment: enabling efficient inference on resource-constrained edge devices (e.g., NVIDIA Jetson Orin Nano) while preserving task accuracy. The core insight is that LLMs exhibit structured neuron activation patterns—distinguishing between model-specific neurons (universally active across tasks) and task-specific neurons (selectively active for individual tasks). Building on this, PartInfer introduces an offline LLM profiler to identify these neurons, paired with two online optimizations: (1) partial loading (only loading critical neurons to reduce memory footprint) and (2) partial computation (dynamically computing task/input-relevant neurons to cut overhead). Evaluations on Llama 3.2-1B/3B models show promising results: 13× speedup over disk offloading, 1.26GB memory savings for Llama 3.2-3B, and competitive accuracy across QA, translation, and summarization tasks.

### Strengths
1. Edge deployment of LLMs is critical for privacy-sensitive (e.g., healthcare on-device diagnostics) and low-latency (e.g., industrial IoT) applications. Unlike prior work that compromises accuracy or requires expensive retraining, PartInfer’s neuron-level optimization avoids these tradeoffs—filling a key gap in existing edge LLM tooling.

2. The authors systematically quantify cross-task overlap and intra-task consistency using diverse datasets.

### Weaknesses
1. The authors only evaluate Llama 3.2-1B/3B and mainly on  translation tasks. The paper omits recent popular downstream tasks for LLM evaluation such as commensence benchmark or code generation or few-shot learning, etc.

2. The offline profiler is central to PartInfer, but the paper provides no details on its computational cost, data requirements, or scalability. 

3. The paper compares PartInfer to CoreInfer and disk offloading but omits critical baselines that practitioners currently use for edge LLMs such as quantization or popular edge frameworks like llama.cpp or TensorRT-LLM include optimizations (e.g., KV caching, kernel fusion).

### Questions
1. How long does it take to profile a task (e.g., QA with SQuAD) for Llama 3.2-3B? On what hardware (edge device vs. cloud) is profiling intended to run?

2. If the profiling dataset includes only formal text, would informal text (e.g., social media) change the active neuron set enough to break partial computation?

3.  How were the parameter values (δ=30%, γ=70%, φ=40%) determined, and how does accuracy/speed change if γ is reduced to 50%


4. Can you add comparisons between PartInfer and 4-bit quantized Llama 3.2-3B (e.g., AWQ) or inference via llama.cpp on Jetson Orin Nano?

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
The paper proposes PartInfer, a system that enables efficient and accurate inference of large language models (LLMs) on low-resource edge devices without retraining, by selectively computing and loading only important neurons. PartInfer accelerates LLM inference by partially loading only critical neurons into memory and computing a subset dynamically based on input activations.

### Strengths
The paper addresses a important problem in the field: deploying LLMs on memory- and compute-constrained edge devices, which is crucial for applications requiring offline inference, privacy, and low latency.

### Weaknesses
- Terminology inconsistency: There appears to be an inconsistency in the definition of base and secondary neurons. In the Introduction, base neurons are described as “general-purpose” and secondary neurons as “task-specific.” However, in Section 4 (line 235), the roles are reversed—base neurons are defined as task-specific. The authors should clarify and maintain consistent terminology throughout the paper.

- Incomplete memory footprint analysis: In Section 6.5, the reported memory footprint reduction is based solely on the size of FFN parameters excluded via partial loading. However, this estimate does not account for actual memory usage during inference, which also includes KV cache, activations, and framework overhead. Moreover, the paper does not report total memory consumption before and after optimization, making it difficult to assess the practical impact of the reported 1.26 GB reduction. I strongly recommend including empirical measurements of end-to-end memory usage and reporting the percentage reduction.

- Lack of absolute performance metrics: The paper reports only normalized performance values (e.g., accuracy relative to the full model), without providing absolute scores. This limits the ability to evaluate the effectiveness of the compressed models across tasks or datasets. Including raw metric values would significantly improve transparency, allow for cross-task comparison, and help readers assess real-world usability.

- Limited trade-off analysis in Figure 9: Figure 9 presents decoding throughput across different neuron loading percentages, but it omits corresponding accuracy or task performance metrics. Without these, it's hard to evaluate the trade-off between speed and accuracy. I  suggest including performance curves and comparisons with baselines (e.g., CoreInfer, full model) to provide a more complete picture.

### Questions
- Clarification on Table 3 results: In Table 3, the combination of CoreInfer and Partial Loading yields inconsistent results—performing worse than CoreInfer alone in some cases, but better on XSum with LLaMA-3B. What accounts for this behavior? Do these results suggest that Partial Loading does not consistently guarantee performance gains?

- Concerns about task generalization: The use of task-specific base and secondary neurons, derived via offline profiling, raises concerns about generalization to unseen or mismatched tasks during online inference. The paper does not evaluate performance when test-time inputs differ significantly from the profiling workload. This limits our understanding of PartInfer’s robustness in real-world multi-task or zero-shot settings. Have the authors considered evaluating on out-of-domain tasks?

I am open to discussing this further during the rebuttal and will be happy to increase my score if my concerns are addressed.

### Soundness
2

### Presentation
2

### Contribution
3
