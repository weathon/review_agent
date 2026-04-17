# DTP: Delta-Guided Two Stage Pruning for Mamba-based Multimodal Large Language Models

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 4

## Abstract
Multimodal large language models built on the Mamba architecture offer efficiency advantages, yet remain hampered by redundant visual tokens that inflate inference cost, with the prefill stage accounting for the majority of total inference time. We introduce Delta-guided Two stage Pruning (DTP), a method that progressively reduces token redundancy through selective pruning at early layer and complete pruning at late layer. Unlike Transformer-oriented pruning methods, our approach derives token importance directly from Mamba’s internal parameters. The statistical distribution of these importance scores, combined with implicit attention patterns, then provides the basis for determining both the pruning layers and the tokens to be removed. Extensive evaluation across diverse benchmarks shows that DTP cuts computation by nearly 50\%, maintains higher task performance than existing pruning methods, and further achieves over a 35\% reduction in prefill latency. Beyond efficiency, our analysis reveals previously underexplored behaviors of visual tokens within Mamba layers, suggesting a principled perspective for designing future pruning techniques in Mamba-based Multimodal Large Language Models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper targets the inference-efficiency bottleneck of Mamba-based Multimodal Large Language Models (MLLMs) and proposes **Delta-guided Two-stage Pruning (DTP)**: early **top-k** retention of visual tokens and late **complete removal**, performed entirely at inference without retraining. Guided by the $\Delta_t$ based token importance, the authors select **layer 15** for selective pruning and **layer 45** for complete pruning. Experiments on **Cobra** and **RoboMamba** indicate that DTP can reduce FLOPs by **approximately 50%** while maintaining competitive accuracy.

### Strengths
1. **Architecture-aligned pruning signal:** Avoids modifying model structure or additional training; the target layers are determined via forward passes only.
2. **Two-stage design:** Preserves sufficient early visual information while removing later-layer redundancy, substantially improving efficiency.

### Weaknesses
1. **Methodology flaws:**
   a. The 15/45 choices hinge on layer-wise standard deviation of $\Delta_t$ -derived token importance computed on a **calibration dataset**; the authors mention a **VQAv2 subset** but disclose neither its size nor sampling strategy. Please detail these and validate on **multiple, distinct calibration datasets**.
   b. A natural **ablation by perturbing the pruning layers** is missing: plots show a $Std_\ell$ valley near **layer 15** and another around **layers 30–35**, yet no experiments/discussion evaluate these alternatives.
   c. Reporting the **post-pruning** standard-deviation profiles would further substantiate the effectiveness of the proposed selection.
2. **Experimental clarity and completeness:**
   a. In **Table 5**, clarify the comparison between *complete pruning* and *disable complete pruning*: in the latter, what pruning rate is used, or is it the **vanilla** model?
   b. It is recommended that the authors provide experimental results with different pruning rates to demonstrate the effectiveness of their approach; the main results only include (r=0.9) and (r=0.5).
   c. Beyond **FLOPs**, include **wall-clock latency** (with **prefill/decoding** breakdown) and **memory footprint**; the two-stage paradigm may affect prefill and decode differently even under the same global token compression.
   d. The authors should provide details such as the sampling parameters of the specific experiments to improve the reproducibility of the paper.

### Questions
Refer to Weakness

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces DTP (Delta-guided Two stage Pruning), a novel and effective token pruning framework designed for Mamba-based Multimodal Large Language Models (MLLMs) to reduce the high computational cost of the prefill stage, which is dominated by a large number of visual tokens. The core idea is to leverage an internal, input-dependent parameter of the Mamba architecture, ∆t, to estimate the importance of each visual token without requiring any retraining. The proposed DTP method employs a two-stage strategy, beginning with selective pruning at an early layer (the 15th), where a portion of the least important visual tokens are discarded. This is followed by complete pruning at a late layer (the 45th), where all remaining visual tokens are removed, a decision justified by the observation that their contributions become negligible in deeper layers as implicit attention patterns diminish. Extensive experiments on two Mamba-based MLLMs (Cobra and RoboMamba) demonstrate that DTP can reduce computation (FLOPs) by nearly 50% while incurring minimal performance degradation, significantly outperforming adapted Transformer-based pruning methods.

### Strengths
- The paper addresses a relevant and important problem: the inference inefficiency of Mamba-based MLLMs. It is intrinsically tied to the Mamba architecture by using the ∆t parameter for importance scoring, a concept not applicable to Transformers.
- The design choices, particularly the selection of the 15th and 45th layers for pruning, are not made ad-hoc. They are convincingly supported by a careful analysis of the model's internal state, which adds a layer of interpretability and principle to the method.
- The method achieves a remarkable balance between computational reduction and performance preservation. A nearly 50% reduction in FLOPs with an average performance drop of less than 1 point (on the Cobra model) is a very strong result. The comprehensive ablation studies further solidify the paper's claims.

### Weaknesses
- The specific layers for pruning (15th and 45th) were empirically identified for the Cobra and RoboMamba models, which appear to have around 64 layers. It is unclear how these specific layer indices would generalize to Mamba-based models of different depths (e.g., a 48-layer or 96-layer model). While the methodology for finding these layers (analyzing the standard deviation of ∆t) seems general, the paper could be strengthened by framing it as a general "recipe" and discussing how it would apply to other architectures.
- The evaluation relies exclusively on FLOPs as a metric for computational cost. While FLOPs are a good hardware-agnostic proxy, they do not always translate linearly to wall-clock speedup due to factors like memory access patterns and GPU kernel optimizations. Including actual latency measurements ( ms/token or similar) would provide a more practical and complete picture of the efficiency gains.
- The experiments show results for keep ratios r of 0.9 and 0.5. A more detailed analysis of the trade-off between performance and the keep ratio r would be beneficial. A plot showing how the average performance gracefully degrades as r is decreased would give readers a better sense of the method's sensitivity to this hyperparameter.
- It is recommended to include a comparison and discussion with these methods [1-4].

[1] Arif, Kazi Hasan Ibn, et al. "HiRED: Attention-Guided Token Dropping for Efficient Inference of High-Resolution Vision-Language Models." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 39. No. 2. 2025. \
[2] Xing, Long, et al. "Pyramiddrop: Accelerating your large vision-language models via pyramid visual redundancy reduction." arXiv preprint arXiv:2410.17247 (2024).  \
[3] Wen, Zichen, et al. "Stop looking for important tokens in multimodal language models: Duplication matters more." arXiv preprint arXiv:2502.11494 (2025).  \
[4] Ye, Weihao, et al. "Fit and prune: Fast and training-free visual token pruning for multi-modal large language models." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 39. No. 21. 2025.

### Questions
- Regarding the choice of the 15th and 45th layers: Could you elaborate on the generality of this finding? If a researcher were to apply DTP to a new Mamba-based MLLM with a different number of layers, should they re-run the standard deviation analysis to find the new optimal "early" and "late" layers? Is there a rule of thumb, e.g., pruning at \~25% and \~70% of the model's depth?
- The paper's efficiency claims are based on FLOPs reduction. Have you conducted any experiments to measure the actual wall-clock inference speedup (e.g., in terms of tokens/second or total latency)? This would be a valuable addition to confirm the practical benefits of DTP.
- How sensitive is the DTP framework to the choice of the early-stage keep ratio r? Could you provide a brief analysis or a curve illustrating the performance-computation trade-off as r is varied continuously between, for example, 0.5 and 1.0?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Delta-guided Two Stage Pruning (DTP), a pruning framework designed for Mamba-based multimodal large language models (MLLMs). The method estimates token importance using the Mamba-specific internal parameter ∆t, enabling selective pruning in early layers and complete pruning in late layers. Experiments on Cobra and RoboMamba demonstrate that DTP can reduce FLOPs by nearly 50% with minimal accuracy degradation. The study further analyzes implicit attention patterns in Mamba, providing insight into token dynamics across layers.

### Strengths
- Proposes a new pruning paradigm specifically designed for Mamba-based models rather than Transformer-based ones.
- Experiments are comprehensive, covering both Cobra and RoboMamba across multiple benchmarks.
- The two-stage pruning strategy is intuitively reasonable and empirically validated.
- Provides interesting analysis of implicit attention patterns in Mamba, offering new perspectives on token behavior.

### Weaknesses
- Limited theoretical justification for ∆t as a universal token importance indicator; the claim is mostly empirical.
- The computational overhead of computing ∆t during inference is not discussed; clarity on latency gain beyond FLOPs would be beneficial.
- Comparison with non-pruning efficiency techniques (e.g., token merging, KV cache optimization) is missing.
- Some figures and analysis (e.g., implicit attention) could be better interpreted to help readers grasp the practical implications.

If authors address my concerns, I will consider raising my score.

### Questions
- Can ∆t be efficiently extracted in real inference pipelines without additional latency?

- How sensitive is DTP to the choice of pruning layers (15th and 45th)? Would adaptive layer selection improve performance?

- Could the ∆t-based importance be combined with other statistics (e.g., gradient-based) to further enhance robustness?

- How does DTP perform under extremely high pruning ratios (>60%)?

- Are the observations about implicit attention patterns consistent across all Mamba-based architectures?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Delta-guided Two-stage Pruning (DTP), a training-free framework for visual token pruning in Mamba-based multimodal large language models.

### Strengths
[1] Adaptation of pruning for Mamba’s state-space mechanism, clearly distinct from Transformer-specific attention-based approaches.   
[2] Two-stage pruning strategy is empirically well-motivated by variance and implicit-attention statistics.   
[3] Extensive ablations validating design choices and demonstrating robustness.   
[4] The method performs pruning during inference without any retraining or fine-tuning, which makes it practically deployable.

### Weaknesses
[1] The layer selection heuristic (15th & 45th) is empirical; a more formal justification or adaptive strategy could strengthen generality.   
[2] The analysis depth of implicit attention remains qualitative; quantitative correlation with pruning behavior would improve rigor.   
[3] While FLOPs reduction is well-documented, real-world inference latency (wall-clock time) is not reported.   
[4] Limited baselines are included. There are many works that should be considered for comparison, e.g. PyramidKV [a], VL-cache [b], etc.  
[5] Novelty is limited, where the methods/motivation are borrowed from the transformer-based research work. 


[a] Cai, Zefan, et al. "Pyramidkv: Dynamic kv cache compression based on pyramidal information funneling." arXiv preprint arXiv:2406.02069 (2024).   
[b]   Tu, Dezhan, et al. "VL-cache: Sparsity and modality-aware KV cache compression for vision-language model inference acceleration." arXiv preprint arXiv:2410.23317 (2024).

### Questions
How were the 15th and 45th layers chosen for selective and complete pruning? Were these empirically optimal for all models or do they depend on model depth or dataset characteristics?

### Soundness
2

### Presentation
3

### Contribution
2
