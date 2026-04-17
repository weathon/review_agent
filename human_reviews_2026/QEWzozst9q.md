# HA-PAT: Hierarchically-Adaptive Pruning-Aware Tuning for Large Language Models

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
The enormous size of large language models (LLMs) limits their deployment and application. Some research utilizes structural pruning to alleviate this by removing redundant weights in a hardware-agnostic manner. However, existing methods tend to apply a uniform pruning strategy across all layers, ignoring the layer-wise functional diversity and risking the removal of essential model components. To tackle this challenge, we propose a Hierarchically-Adaptive Pruning-Aware Tuning (HA-PAT) method. Based on the pruning-aware tuning framework, HA-PAT employs Hierarchical Pruning Ratio Scheduling (HPRS) to derive optimal layer-wise sparsity guided by each layer's unique functionality. It preserves the general linguistic functions of shallow layers, while aggressively pruning the deeper layers that primarily encode task-specific features. To better preserve model performance, HA-PAT introduces a magnitude vector into the compensation mechanism, enabling the reconstruction of pruned weights based on a broader information space. Experimental results show that our method consistently outperforms the baseline both in average accuracy and inference efficiency. On LLaMA2-13B with 25\% pruning ratio, our approach surpasses the PAT baseline by 4.01\% in average accuracy across 14 benchmarks, along with a 30\% inference speedup. Further experiments on downstream tasks indicate that HA-PAT better preserves the pre-trained language understanding capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces HA-PAT, a hierarchical pruning-aware fine-tuning framework for large language models. Building on existing pruning-aware tuning methods, HA-PAT incorporates three key innovations: (1) Layer-wise Independent Masks (LIM) to capture functional diversity across layers, (2) Hierarchical Pruning Ratio Scheduling (HPRS) based on the information bottleneck principle, and (3) an Adaptive Compensation Operator (ACO) for stable knowledge recovery. Experiments across multiple LLMs (LLaMA2, Gemma) and 14 downstream tasks demonstrate consistent performance gains (up to +7% over PAT) with reduced memory and inference latency.

### Strengths
1. Methodological Novelty: The combination of layer-wise masking and hierarchical pruning ratios represents a substantial step forward beyond uniform pruning, offering theoretical and practical benefits.

2. Theoretical Grounding: The design of HPRS guided by the information bottleneck principle gives the framework interpretability and aligns well with observed empirical behavior.

3. Comprehensive Evaluation: Extensive experiments across diverse LLMs and datasets, together with detailed ablations, convincingly validate the effectiveness and generality of the approach.

### Weaknesses
1. Limited Learnability of HPRS: The pruning ratio schedule is predefined rather than learnable, potentially restricting adaptability to diverse architectures or tasks.

2. Scalability Analysis Missing: Although results on 7B–13B models are reported, there is no exploration on models >30B or inference-time deployment on real hardware, leaving uncertainty about large-scale efficiency.

3. Limited Theoretical Analysis: Although the information bottleneck principle is mentioned, there is no formal derivation or empirical evidence linking mutual information flow to pruning ratio distribution.

4. Hyperparameter Sensitivity: The method introduces several additional hyperparameters (mask learning rate, pruning ratio schedule parameters, ACO scaling vector), but their sensitivity or tuning cost is not analyzed.

### Questions
See weaknesses

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
PAT proposes a structural-pruning paradigm that couples pruning with fine-tuning. HA-PAT extends this framework by introducing a layer-wise sparsity schedule: it exploits inter-layer information disparities to assign adaptive pruning ratios to different depths.

Main contributions:  

1.  Building on PAT’s MASK mechanism, HA-PAT introduces LIM so that every layer owns an independent, learnable mask, enabling each layer to discover its own sparse structure adaptively.  

2.  It devises HPRS, a heuristic that prescribes low pruning ratios for shallow layers and high ratios for deep layers.  

3.  The ACO optimizer—adopted verbatim from PAT—is employed to co-optimize the masks and weights.

### Strengths
1. This paper departs from the uniform-pruning paradigm by deriving and empirically validating a layer-specific sparsity-schedule hypothesis. The resulting strategy not only proves effective in extensive experiments but also opens a new design dimension for future pruning schemes.

2. Ablations on every proposed module confirm that each contributes non-trivial gains.

3. Evaluations are conducted on two model families, each at two scales, providing preliminary evidence of generality. Nevertheless, a broader matrix of experiments—spanning more sizes within each family and additional architectures—would further strengthen the claim that the observed improvements are architecture-agnostic.

### Weaknesses
The HPRS scheduling rule is currently presented as an empirical trend-driven heuristic; a formal theoretical justification is absent. Consequently, readers have no systematic basis to judge whether the schedule remains near-optimal when the network depth, width, or residual topology changes, limiting confidence in its universality across architectures.

### Questions
1. The ACO procedure described in the paper is identical to the “scaling-and-rotation decomposition” already introduced in PAT; listing it again as a separate methodological item is therefore redundant.

2. According to the architectural diagram, the proposed approach still inserts a binary mask immediately after each HSM module—only now every layer owns an independent mask. If these layer-wise masks are further scheduled by the proposed HPRS (i.e., low sparsity for shallow layers and high sparsity for deep ones), how does the framework guarantee inter-layer feature alignment after pruning?

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
This paper focuses on structured model pruning for LLMs with a special focus on optimizing the sparsity levels for different layers. The main motivation of this paper is that most existing LLM pruning methods adopt the same sparsity ratios for all layers. The authors demonstrate that lower layers should be pruned less while upper layers can be more sparse from both intuitive perspective and information bottleneck.

Given this motivation, the authors set different target sparsity ratios for different layers and add an extra pruning loss, $\mathcal{L}_{\mathrm{ratio}}$, which will encourage the pruning masks to prune the model to the target sparsity ratios. For the other pruning techniques, they are mostly built upon the previous work, PAT, with slight modification (change the compensation matrix $D'$ in the method formulation). Experiment results on several LLMs demonstrate the proposed method outperforms two structured pruning approaches.

### Strengths
From my perspective, this paper is a nice paper with rigorous method development and experiment design. Specifically,

- Clear motivation. Applying different sparsity ratios to different LLM layers is reasonable, and the authors have made valuable explorations in this direction.
- Method design. Given the motivation, the authors make corresponding revision to the original PAT method to enable different sparsity ratios across LLM layers. The overall design also makes sense.
- Experiment design. This paper includes experiments on multiple LLMs to demonstrate the applicability of the proposed method. The ablation study in Table 2 also clearly show the effectiveness of different components of the method.

### Weaknesses
However, despite the aforementioned strenghts, I still think this paper suffers from several major weaknesses.

- Insufficient technical contributions. Although the method makes sense, the main concern is that these techniques are already widely used in existing LLM pruning approaches. For example:

  - layer-wise independent masks. Most existing LLM structured pruning [1-4] approaches does not constrain the mask to be consistent across all layers. Instead, the pruning masks are automatically optimized indepently across layers by default.
  - Hierachical pruning ratio loss. It is straightforward to adopt such a loss, which is also widely used by existing methods to prune LLMs to a target sparsity ratio such as SheardLlama [1].
  - Adaptive compensation operator (ACO). I believe this component, together with the original PAT formulation, is unnecessary. It prune a specific linear layer (attention head QKV projections and FFN matrices) by:

  $$Z = (M\odot D)\cdot WX$$

    Here M is a $d$ dimension vector, D is a $d\times d$ matrix, and $d$ is the hidden dimension. The nature of this equation, is to remove the $i$-th row/column of the origina parameter $W$ where $M_i=0$, while allowing scale up/down the other row/columns of $W$ by $(M\odot D)_i$. This is because, $M\cdot D$ will result a row/column vector where some elements are 0 and others can be any real numbers. This is different from conventional model pruning, where the elements of the fianl pruning mask is either 0 or 1. That is, it allow the the non-zeros elements to scale up or scale down the original parameters.
  
  From this perspective, I did not see why the proposed method has more expressive power compared to the original formulation of PAT. The ablation study in Table 2 also show that ACO brings the least performance improvement.
  
  In summary, the technical contribution of the paper is insufficient.

- Missing baselines & overlap with existing methods that allow different sparsity ratios across layers. Besides technical contributions of the 3 moduels above, the main contribution and breakthrough of this paper is the flexibile sparsity ratio for different layers. However, this is already explored by existing work [5, 6]. The Puzzle method already delivers different sparsity ratios for different layers, achieving much stronger performance than this paper. It is hard to distinguish the contribution of this paper given the strong work [5, 6].

- Unclear writing and presentation. For Section 3.2, the notations are very confusing. Although I can understand them, they are very unfriendly to non-experts. For example, the shape of these matrices, the upper-cased letter to represent a vector ($M$), and did not explain why Eq. 3 is equaivalent to the forward pass of a pruned model.

  (note: it should be a standard to use uppercased letters for matrices, lowercased letters (unbolded) for scalers, and bolded lowercased letters for vectors. The notations does not follow the standard.)

- (Minor issue) The models used in this paper are out-dated. Also, the baselines are not strong enough.



[1] Xia, Mengzhou, et al. "Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning." The Twelfth International Conference on Learning Representations.

[2] Sreenivas, Sharath Turuvekere, et al. "Llm pruning and distillation in practice: The minitron approach." arXiv preprint arXiv:2408.11796 (2024).

[3] Wang, Ziheng, Jeremy Wohlwend, and Tao Lei. "Structured pruning of large language models." arXiv preprint arXiv:1910.04732 (2019).

[4] Hou, Bairu, et al. "Instruction-Following Pruning for Large Language Models." Forty-second International Conference on Machine Learning.

[5]  Bercovich, Akhiad, et al. "Puzzle: Distillation-Based NAS for Inference-Optimized LLMs." Forty-second International Conference on Machine Learning, 2025.

[6] Bercovich, Akhiad, et al. "Llama-nemotron: Efficient reasoning models." arXiv preprint arXiv:2505.00949 (2025).

### Questions
Please refer to the weakness above.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Hierarchically-Adaptive Pruning-Aware Tuning (HA-PAT), a new method for structurally pruning Large Language Models (LLMs). The key insight is that existing methods, even advanced ones like Pruning-Aware Tuning (PAT), make a critical mistake by pruning all layers uniformly. HA-PAT corrects this by recognizing that different layers have different jobs. It introduces a hierarchical pruning schedule that preserves the general linguistic knowledge in the shallow layers by pruning them lightly, while more aggressively removing redundancy from the deeper, task-specific layers.

### Strengths
1. The idea that different layers should be pruned at different rates based on their function is intuitive and well-supported by established research on neural network feature hierarchies. It directly addresses a clear limitation in prior work.

2. The paper breaks down its improvements into distinct, understandable components: Layer-wise Independent Masks (LIM), Hierarchical Pruning Ratio Scheduling (HPRS), and the Adaptive Compensation Operator (ACO).

3. The results are compelling. Achieving a 4.01% average accuracy improvement over the PAT baseline while also delivering a 30% inference speedup is a significant achievement that clearly demonstrates the method's value.

### Weaknesses
1. The Hierarchical Pruning Ratio Scheduling (HPRS) is described as applying "progressively increasing pruning ratios." This might not be truly optimal for all models or tasks and could be less "adaptive" than the name implies.

2. The assumption that "shallow = general" and "deep = specific" is a powerful heuristic but not a universal law. A fixed hierarchical strategy might perform poorly on tasks that unexpectedly rely on complex features learned in early layers.

### Questions
1. Does allowing each layer to learn its own mask with Layer-wise Independent Masks (LIM) introduce any significant computational overhead during the pruning-aware tuning phase compared to the simpler global mask in PAT?

2. Have you explored whether the optimal hierarchical pruning schedule is task-dependent? For example, would a syntax-heavy task benefit from a different schedule than a task requiring more abstract, high-level reasoning?

### Soundness
3

### Presentation
3

### Contribution
3
