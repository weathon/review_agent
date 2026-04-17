# Analytical Restructuring of Feed-Forward Networks for Accelerated LLM Inference

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 4

## Abstract
Scaling large language models (LLMs) improves performance but dramatically increases inference costs, with feed-forward networks (FFNs) consuming the majority of computational resources. 
While sparse architectures like mixture-of-experts (MoE) can mitigate this, inducing sparsity in existing dense models typically requires extensive, resource-intensive retraining (often hundreds of billions of tokens), creating a prohibitive barrier to practical deployment. 
We propose a broadly applicable post-training framework that improves this performance–cost trade-off by enabling the rapid, analytical restructuring of FFNs into a sparse, efficient architecture.
The framework operates by analyzing neuron activation patterns from a small calibration dataset, then analytically rebuilding the FFN into a Mixture-of-Experts-style architecture with always-active ``shared'' experts and conditionally activated ``routed'' experts. 
Critically, this process can restructure dense FFNs into sparse MoE architectures and can also be applied recursively to the experts within existing MoE models to create finer-grained hierarchical sparsity for further acceleration. 
We construct a differentiable router directly from activation statistics, enabling immediate deployment with a useful training-free baseline and serving as a robust foundation for optional, lightweight fine-tuning.
Experiments validate our approach across diverse settings, delivering practical speedups reaching up to $1.17\times$ in compute-bound scenarios while providing consistent gains across all configurations. This is achieved with only minutes of processing time and minimal fine-tuning (2k samples), which favorably contrasts with methods requiring orders of magnitude more computational resources.
By providing an efficient, analytical path to high-performance sparsity, the framework makes accelerated LLM deployment practical and accessible for resource-constrained environments.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work introduces a post-training framework that analytically restructures FFNs into a sparse MoE-style architecture using only a small calibration dataset. It identifies activation patterns, then builds shared and routed experts and constructs a differentiable router without full retraining. The method can also recursively introduce hierarchical sparsity in existing MoE models. It achieves practical speedups up to 1.17× with only minutes of processing and minimal fine-tuning, enabling efficient LLM deployment in resource-limited settings.

### Strengths
- The proposed method only introduces very limited computing requirement.
- Across multiple base models and tasks, the approach outperforms existing MoE restructuring baselines.
- The paper is overall well-written in terms of writing.

### Weaknesses
- To measure "speedup", usually one should mention the cut in term of TFLOP by percentage, not just by speedup (could be implementation variations talking).
- The paper focuses on sparsity. There are many strong-performing baselines that are not taken into account. For instance, training-free SOTAs [1], [2]

[1] WINA: Weight-Informed Neuron Activation for Accelerating Large Language Model Inference

[2] Training-free activation sparsity in large language models

- Lack of therectical results supporting the claim.

- In industrial application, the results indicate that the sparse activity does pose random effect (can be average out by multiple sampling). Why not do this experiment in common academic benchmarks?

### Questions
see weaknesses.

### Soundness
2

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
4

### Summary
The paper proposes an analytical, post-training restructuring of transformer FFNs into a sparse MoE-style layer with a small set of always-on shared experts and Top-K routed experts. Neuron activation patterns gathered from a tiny calibration set are used to (i) choose shared neurons by high activation rate, (ii) cluster the rest into routed experts via a balanced assignment, and (iii) build a differentiable router from “representative” neurons, enabling a training-free baseline and optional lightweight LoRA fine-tuning. Reported full-model speedups reach up to 1.17× with drooped performance.

### Strengths
- **Training-light path to sparsity.** Clear pipeline: activation profiling → balanced clustering → router from activation statistics. The training-free start point plus small LoRA fine-tuning (2k samples) is lightweight.
- **Claimed universality/hierarchy.** the same procedure is described for intra-expert restructuring in existing MoE layers (hierarchical sparsity).
- **Evidence of end-to-end speedups.** across configurations and context lengths; a table reports up to 1.17× in compute-bound settings.

### Weaknesses
- **Limited conceptual novelty.** Clustering existing FFN neurons and partitioning them into experts has been explored in prior dense→MoE conversions (e.g., MoEfication, LLaMA-MoE). The paper should explicitly articulate what is new (e.g., routing construction, balancing, shared-expert selection) and quantify the incremental gain over these baselines with matched budgets and identical fine-tuning protocols.
- **Overclaim of differentiable routing.** The manuscript labels the router “differentiable,” but the implementation seems to keep a hard Top-K mask and simply multiply by a zero-initialized scaling term. This does not furnish a differentiable selection mechanism. The paper also omits discussion and head-to-head comparisons with existing differentiable-routing methods (e.g., ReMoE, Lory).
- **Hierarchical MoE claim lacks empirical validation.** The paper describes intra-expert restructuring for existing MoEs but does not present dedicated experiments isolating quality/throughput of the two-level hierarchy.
- **Evaluation breadth.** Downstream accuracy is limited to five zero-shot tasks (PIQA, WinoGrande, ARC-E/C, HellaSwag); no MMLU/coding/math quality comparisons are reported. This makes the quality story incomplete for LLMs.

### Questions
- How sensitive are clustering and router initialization to the size and domain of the calibration set? Please include sweeps and domain-shift tests.
- What is the exact update rule for the adaptive bias terms, and how do utilization entropy and per-expert load variance evolve compared to standard aux-loss balancing?
- Does lowering sparsity (e.g., increasing the number of active parameters or Top-k) avoid quality degradation? Please provide accuracy–throughput trade-off curves across sparsity levels and identify the break-even point versus the dense baseline.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a post-hoc approach to restructure the feed-forward network (FFN) layers of a pretrained large language model (LLM) into mixture-of-experts (MoE) layers, enabling faster inference with no training or light fine-tuning. The proposed algorithm exploits activation sparsity to distinguish between shared experts and routed experts. Shared experts are formed by selecting neurons that consistently exhibit high activation values, while the remaining neurons are uniformly partitioned into routed experts which are sparsely activated during inference. The router weights are then constructed by extracting the most representative weights from the original FFN for the corresponding experts. Additionally, the algorithm provides an efficient fine-tuning procedure to further refine the router and recover model performance.

### Strengths
+ The proposed method appears lightweight and efficient, with a well-designed overall pipeline. The paper provides implementation details for each stage of the approach.

+ Some of the theoretical insights are particularly interesting, especially those discussed in Appendices A.1 and A.4. The hypothesis is clearly stated, and the results are reasonably supported by light mathematical analysis.

+ The inclusion of some interesting experiment analyses is a plus. For instance, the authors examine load balancing and discuss performance under both memory- and compute-bound inference scenarios.

### Weaknesses
- The overall writing of the paper could be improved. Some notations are confusing (see Question), and several key formulations are missing. For example, it remains unclear how A is computed exactly. Text descriptions alone are insufficient given its central role in the proposed method. Additionally, some discussions in the appendix are insightful; the authors may consider moving part of these into the main text to improve logical flow and strengthen the justification for each design choice.

- The novelty of constructing MoE layers through activation clustering appears limited. The overall idea and pipeline are similar to [1], yet the paper lacks a detailed discussion or comparison with this prior work. The claim that [1] is concurrent to this study (Lns 100-103) is misleading, as the first version of [1] was published in 2021.

  [1] Zhang et al., MoEfication: Transformer Feed-forward Layers are Mixtures of Experts

- There are also a few missing baselines and related works that should be discussed, such as:

  [1] Zhang et al., MoEfication: Transformer Feed-forward Layers are Mixtures of Experts

  [2] Cai et al., Refactorizing LLMs as Router-Decoupled Mixture of Experts with System Co-Design

- While the proposed method shows promising results compared with the reported baselines, several concerns remain:

  (1) The baselines do not appear to undergo any fine-tuning, whereas the proposed method leverages LoRA-based fine-tuning. It is unclear whether the performance gain primarily stems from this fine-tuning step.

  (2) One of the main claims in the abstract is that the proposed MoE construction process can be completed within minutes. However, Fig. 2 lacks any comparison with baseline methods to verify the claimed efficiency.

  (3) The chosen benchmarks seem somewhat outdated. It would strengthen the paper to include evaluations on more challenging and widely adopted benchmarks, such as MMLU or MATH500.

### Questions
1. In Eqs (1) and (3), the vector appears to be multiplied on the left side of the matrices, which is unconventional. If the intended operation involves transposed vectors, all corresponding notations should be adjusted accordingly: $x$ -> $x^\top$, etc.

2. Could the authors also clarify the choice of distance metric defined over the $c_i$’s. A more natural option might be the Hamming distance. Does the chosen distance metric affect the expert assignment algorithm or the resulting model performance?

3. It is unclear why the experiments require lightweight fine-tuning with LoRA. The proposed method seems to require fine-tuning only for the router. Is the router fine-tuned jointly with other model weights, or is it optimized separately?

4. The potential impact of the calibration or fine-tuning dataset choice is not clear. Using WikiText as the calibration dataset appears rather preliminary; it would be helpful to evaluate whether the method’s performance is sensitive to this choice (with harder benchmarks being considered).

### Soundness
3

### Presentation
2

### Contribution
2
