# MoNE: Replacing Redundant Experts with Lightweight Novices for Structured Pruning of MoE

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Mixture-of-Experts (MoE) enables efficient scaling of large language models by activating only a subset of experts per input token.
However, deploying MoE-based models incurs significant memory overhead due to the need to retain all experts in memory. 
While structured pruning is promising to reduce memory costs, existing methods often show suboptimal performance and unstable degradation in three dimensions: model architectures, calibration data sources, and calibration sample sizes.
This paper proposes \textbf{M}ixture-\textbf{o}f-\textbf{N}ovices-and-\textbf{E}xperts (\textbf{MoNE}), a novel expert pruning method that replaces redundant experts with lightweight novices to achieve effective and robust model compression. 
MoNE evaluates expert redundancy based on two metrics: access frequency and output variance. 
Experts exhibiting low usage and stable outputs are pruned and replaced with lightweight novices—unbiased estimations of their original outputs—minimizing performance degradation. 
Extensive experiments demonstrate that MoNE consistently outperforms baseline methods with minimal accuracy degradation across the three dimensions, confirming its effectiveness and robustness. 
Notably, it outperforms baselines by up to 2.72 for the average zero shot accuracy across nine downstream tasks under 25\% pruning ratio, with only 0.14 performance drop for Qwen2-57B-A14B. 
The code is available at \url{https://github.com/zxgx/mode-pd}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces MoNE, a pruning method for Mixture-of-Experts models. It identifies experts with low routing frequency and low output variance, and replaces them with constant vectors called novices. This reduces memory cost while keeping model accuracy stable. Experiments across multiple MoE architectures show that MoNE performs better than existing pruning baselines and remains stable across different calibration settings.

### Strengths
The idea of replacing experts with constant novice vectors is simple, easy to implement, and maintains the computational benefits associated with pruning.

The redundancy score combining frequency and output variance is well motivated and avoids relying solely on routing frequency, which previous methods often do.

The experimental evaluation is broad, covering multiple MoE architectures, different pruning ratios, and variations in calibration data, showing consistent robustness.

### Weaknesses
Replacing experts with constant vectors may reduce the model's expressive power, and test cases, due to their limitations, may not be able to cover the negative impacts of the evaluation.

Pruning strategies depend on the distribution of the calibration dataset. Different application scenarios may have different dependencies on experts, and differences between the distribution of the calibration dataset and the distribution of the real-world scenario may lead to the erroneous removal of important experts.

### Questions
No further questions, see above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The research focus of this paper is expert pruning in Mixture-of-Experts (MoE) models. To address this issue, the paper proposes Mixture-of-Novices-and-Experts (MoNE), a novel expert pruning method that replaces redundant experts with lightweight novices to achieve effective and robust model compression. Experiments demonstrate that MoNE consistently outperforms baseline methods across three dimensions—model architectures, calibration data sources, and calibration sample sizes—with minimal accuracy loss, validating its effectiveness and robustness.

### Strengths
1. This paper propose a novel expert pruning method named MoNE which replaces redundant experts with lightweight novices to compress MoE models with minimal performance loss
2. This paper uses expert access frequency and output variance to measure redundancy, and unbiased output estimation to minimize post-pruning discrepancy, yielding effective and robust pruning.

### Weaknesses
1. The combinatorial forms of frequency and variance adjacency matrices require ablation, such as weighted summation.
2. Replacing experts with constant vectors may reduce expressiveness; could learnable vectors or biases be used instead?
3. Could comparative experiments on pruning strategies (without finetuning) be provided to demonstrate the superiority of the proposed frequency- and variance-based pruning strategy?

### Questions
Refer to Weaknesses

### Soundness
3

### Presentation
3

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
MoNE proposes to prune MoE models by replacing selected experts with constant novices (per-expert mean outputs) instead of simply deleting or merging them. Redundancy is estimated by two metrics computed on a small calibration set—access frequency and output variance—and the novice for a pruned expert is its unbiased mean output. The method is evaluated across several MoE architectures (OLMoE, Moonlight, DeepSeek-V2-Lite, Qwen2-57B-A14B, Qwen3-30B-A3B), calibration sources/sizes, and pruning ratios, with ablations on the metrics and novice replacement.

### Strengths
* Simple, compute-friendly pruning primitive (constant novices) that retains router behavior and keeps overhead close to removal. 

* Consistent gains/robustness across models and calibration setups; headline numbers are competitive.

### Weaknesses
* The ablation in Figure 4 is intersting, seems like the variance metric can bring improvement without the novice. It would be helpful to include more comprehensive ablation, i.e. more combination (e.g. only frequency and only variance) to show the gain from each part.


* The novice is the unbiased mean output of a pruned expert (a constant vector), similar to FLAP’s use of averaged activations for compensation but at a different granularity. The paper should more explicitly discuss the relation with FALP and isolate the contribution.

 
* The redundancy score is the product of variance and frequency. This hard-coded fusion may be scale-sensitive; it would be helpful to include normalized scores, log-sum, or a learned weight λ and report stability.


* Some related works worth mentioning [1,2]

[1] MOE-PRUNER: PRUNING MIXTURE-OF-EXPERTS LARGE LANGUAGE MODEL USING THE HINTS FROM ITS ROUTER
[2] SlimMoE: Structured Compression of Large MoE Models via Expert Slimming and Distillation

### Questions
* After pruning, do tokens get routed more often to the remaining real experts or to novices? Any trends per layer?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes MoNE, a structured pruning approach that replaces redundant experts with lightweight "novices" - essentially constant vectors representing averaged expert outputs. The key idea is identifying redundant experts using two metrics: access frequency (how often an expert is selected) and output variance (how stable an expert's outputs are). Experts with low frequency and low variance get replaced by their mean output. The authors test this on five MoE models (OLMoE, Moonlight, DeepSeek-V2-Lite, Qwen2-57B-A14B, Qwen3-30B-A3B) at 25% and 50% pruning ratios, showing better performance than existing methods like MC-SMoE, RS, Angular, and FLAP.

### Strengths
1. The core idea is intuitive, simple, and training-free. The fused metric (frequency + variance) is well-justified, and the "novice" replacement (the expert's mean output) is an effective closed-form solution to minimize output discrepancy.

2. The experimental validation is a major strength. Testing on five different MoE architectures with varying sizes (7B to 57B parameters) demonstrates the method works across scales. The robustness evaluation across model architectures, calibration data sources (Zyda2 vs C4), and sample sizes (100, 500, 1000) is thorough.

### Weaknesses
1. There is a lack of specialized tasks (e.g., coding, math) in evaluation. It's unclear if the redundancy metric, calibrated on general text, might inadvertently prune experts that are critical for these specialized capabilities.

2. The paper doesn't explain or ablate the benefit of computing a dynamic, per-token gate for a static, constant "novice" vector. This appears computationally redundant.

### Questions
1. How does MoNE perform on specialized benchmarks like Math or GSM8K? The current evaluation on general tasks may not be sufficient to prove that specialized experts are not being harmed.

2. What is the justification for per-token routing to a static novice vector? Would a simpler approach, like adding the novice as a scaled, static bias, perform comparably while saving routing computation?

3. Are novices trainable during continued pretraining?

### Soundness
3

### Presentation
3

### Contribution
3
