# Towards Global Expert-Level Mixed-Precision Quantization for Mixture-of-Experts LLMs

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Mixture-of-Experts large language models (MoE-LLMs) achieve state-of-the-art performance across diverse language tasks but incur substantial memory overhead due to their massive expert parameters.
Mixed-precision quantization, which allocates different bit-widths to experts according to their importance, has emerged as a promising technique for reducing the memory consumption of MoE-LLMs.
However, we identify two key limitations in existing MoE-LLMs quantization methods: (1) expert importance is estimated only locally within each MoE layer, failing to capture global importance across the model and leading to suboptimal bit-width allocation; and (2) expert quantization substantially alters the dynamics of MoE routers, yet this effect is often overlooked, resulting in suboptimal routing.
In this work, we propose Global Expert-level Mixed-precision Quantization (GEMQ) to overcome these limitations and enable extreme low-bit quantization. First, we introduce a global expert bit-width allocation method that formulates a linear programming model based on quantization error analysis to capture global expert importance. Second, we propose an efficient global router fine-tuning approach that adapts routers to quantized experts, enabling optimal routing. 
Additionally, we integrate the two techniques into a progressive quantization framework that leverages the previously quantized and fine-tuned model for expert importance estimation, enabling more accurate allocation and improved performance.
Extensive experiments show that our approach substantially reduces memory usage and improves inference speed while incurring minimal performance degradation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes GEMQ to tackle the high memory overhead of MoE-LLMs via global expert-level mixed-precision quantization. It addresses key flaws of existing methods (ignoring inter-layer importance differences and post-quantization router distortion) through three core components: global LP-based bit-width allocation, parameter-efficient router fine-tuning, and a progressive quantization framework.

### Strengths
1. The method proposed in the paper delivers practical acceleration gains, achieving a speedup of over 1.6× with negligible impact on accuracy.  
2. The paper is well-written with a complete and coherent logical chain.

### Weaknesses
See "Questions" below.

### Questions
1. Figure 1 shows that loss gradients vary across layers, leading to the claim that layers differ in importance. However, the significance of this observation is highly dependent on input data. It remains unclear whether the inter-layer importance differences are consistent across different inputs, or if they vary with input changes. If inter-layer importance does differ across inputs, how should the bit-width of different layers be allocated?  

2. The paper mentions using "task loss gradients" to globally measure expert importance, but it does not specify the **exact object of gradient calculation**—whether it refers to the gradients of expert weights or the gradients of layer outputs.  

3. The paper sets a constraint that "each layer must contain at least one 3-bit expert and one 2-bit expert" to avoid information bottlenecks. However, it fails to explain why the combination of 3-bit/2-bit is chosen (instead of other combinations like 2-bit/1-bit) and lacks ablation experiments to verify the constraint’s impact on performance.  

4. The paper only compares its method with PMQ as a baseline. However, there are many mixed-precision quantization methods for LLMs (e.g., SpQR). It is not addressed whether these methods also fail when applied to MoE models.  

5. The ablation experiment in Table 2 shows that accuracy improves significantly after gate fine-tuning. Without fine-tuning, the model’s PPL is worse than that of PMQ. This indicates that the paper’s performance gains rely on fine-tuning, while the mixed-precision effect of the pre-quantization stage is weaker than that of PMQ.  

6. The proposed method appears to be a combination of engineering tricks: extending intra-layer mixed-precision to inter-layer mixed-precision and adding fine-tuning after quantization. Both techniques are common in engineering practice.  

7. Robustness tests are missing, and the reproducibility of results needs to be enhanced.  


Based on the above issues, I would recommend a "Reject" decision. I would be happy to revise my rating if the authors can address these concerns in the revised manuscript.

### Soundness
2

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
To address the deployment memory bottleneck of large Mixture-of-Experts language models (MoE-LLMs), a Global Expert-level Mixed-Precision Quantization (GEMQ) method is proposed. Existing MoE-LLM quantization approaches face two key limitations: (1) they estimate expert importance locally within a single MoE layer, failing to capture global importance across layers, which leads to suboptimal bit allocation; (2) they ignore the impact of expert quantization on the router’s dynamic behavior, resulting in suboptimal routing. GEMQ addresses these challenges through three core designs: Global expert bit allocation: a linear programming model based on quantization error analysis to optimize bit distribution across all experts; Router-aware adaptation: efficient fine-tuning of router parameters accounting for less than 0.04% of total weights, ensuring compatibility with quantized experts; Progressive quantization framework: integrates the above two components and leverages fine-tuning from previous quantization rounds to improve expert importance estimation in low-bit scenarios. GEMQ achieves significant memory reduction and inference acceleration across various MoE-LLMs, while incurring minimal performance loss, outperforming existing methods such as PMQ.

### Strengths
1. The study clearly identifies two critical limitations of existing MoE quantization: local bit allocation and neglect of router dynamics, and experimentally validates their impact. Each GEMQ module is designed to directly address these shortcomings, forming a coherent logical framework.

2. Global bit allocation is formulated as a linear programming model based on task loss rather than heuristic coefficients, eliminating manual tuning and enabling adaptation to different MoE models. Router fine-tuning optimizes only a very small fraction of parameters.

3. The relationship between quantization error and task loss is derived using Taylor expansion and the Fisher information matrix (Eqs. 4–6), providing a theoretical foundation for global bit allocation. Additionally, GPTQ is adopted as the underlying quantization method, compatible with the HQQ library for model storage, and engineering considerations such as router quantization (Table 8) are incorporated, balancing theoretical rigor and practical feasibility.

### Weaknesses
1. Parameter Dependency and Progressive Framework Impact: The approach exhibits strong parameter dependency and is influenced by “previous-round model selection” in the progressive framework. If LP assigns too many low-bit experts to a layer, the router fine-tuning requires a higher learning rate to adapt; otherwise, router dynamics may drift more severely. However, the paper does not provide coordinated tuning rules across modules and relies solely on trial-and-error experiments (e.g., only one learning rate is tested). In practical applications, this would require traversing many parameter combinations, making tuning highly inefficient.

2. Limited Bit-Width Budget and Design Choices: The study only considers three bit budgets (2.5, 2.0, 1.5 bits/expert), which is a rather subjective design. It does not specify how iteration count (K value) or budget intervals are determined, nor does it analyze how different settings affect performance or efficiency. As a result, the method lacks clear guidance for parameter selection in different scenarios.

3. Performance on GSM8K and Calibration Sensitivity: On the GSM8K mathematical reasoning task, GEMQ performs significantly worse than the full-precision model, and its degradation is more severe under low-bit settings compared to PMQ. The paper attributes this to calibration-set fitting, suggesting that the quality of the MoE calibration set may be more critical than in mixed-precision setups. To ensure effective fine-tuning, GEMQ requires a strict match between calibration data and quantized data.

### Questions
1. Potential Optimization of Global Bit-Allocation Constraints: The linear programming model for global bit allocation imposes the constraint that “each layer must contain at least one 3-bit expert and one 2-bit expert” to avoid information bottlenecks. This constraint is adopted from the empirical setting in PMQ (Huang et al., 2024a) and has not been theoretically derived or validated via ablation studies. For small MoE models with very few layers or sparse experts (e.g., Qwen1.5-MoE with only 24 layers and 64 experts), this “fixed-type bit expert” constraint may waste bit budget. Would removing the constraint or changing it to “at least one high-bit expert every two layers” further reduce task loss under the same bit budget?

2. Undefined Failure Boundary of Taylor Approximation at Low Bits: The failure boundary of the Taylor expansion approximation under low-bit settings is not quantified. In Equation 4, the paper only mentions that “the approximation is unreliable at low bits” but does not define a specific threshold. For example, when the bit budget decreases from 2.0 bits to 1.5 bits, by how much does the Taylor approximation error increase? Is there a clear mathematical metric to determine in advance whether “the progressive framework should be enabled under the current bit budget”?

3. Theoretical or Empirical Justification for Calibration Synergy: Can theoretical or experimental evidence be provided to demonstrate synergy between the calibration set and this method—ensuring that experts are correctly selected, expert bit-widths are appropriately assigned, and post-calibration performance reaches optimal results across various tasks?

4. Can the method’s effectiveness be demonstrated on some multimodal MoE models, given that different modalities may have different impacts on expert selection?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes Global Expert-level Mixed-precision Quantization (GEMQ) for MoE-LLMs. The method estimates expert importance with a task-loss–motivated proxy and solves a global binary LP to allocate expert bit-widths under a budget. It then fine-tunes only router parameters to adapt routing after quantization, and further adopts a progressive quantization schedule that re-estimates importance from a nearby quantized-and-finetuned model. Experiments on DeepSeekV2-Lite, Qwen1.5-MoE, Qwen3-30B-A3B, and Mixtral‑8×7B show improved perplexity and competitive zero-shot accuracy under aggressive bit budgets, with large memory savings.

### Strengths
- The paper frames expert-bit assignment as a global optimization over all experts, rather than per-layer, yielding a principled resource–accuracy trade-off under a single LP.

- The empirical validation spans multiple MoE-LLMs (DeepSeekV2‑Lite, Qwen1.5‑MoE, Qwen3‑30B‑A3B, Mixtral‑8×7B) with consistent perplexity gains at 1.5–2.5 bpe, and strong memory reduction.

- The paper carefully analyzes inter-layer variations in expert importance and shows why a global allocator outperforms naive globalizations of local schemes.

- The paper is well written and easy to reproduce.

### Weaknesses
- Bit choices and the trade-off surface are narrow. Experiments constrain experts to {1, 2, 3} bits and attention to 4 bits which limits the observed efficiency–accuracy Pareto. There is little exploration of richer candidate sets (e.g., {0(prune), 1, 2, 3, 4}), mixed attention precisions, or hardware-aware constraints, making it hard to fully characterize cost–quality trade-offs.

- Importance is estimated on generic corpora (C4/WikiText2) and routers are fine-tuned with the same small calibration set. The paper itself notes worse GSM8K math results and attributes this to calibration bias toward general language modeling. This underscores a potential fragility: globally optimized bit layouts (and router readjustments) may overfit the calibration/task distribution and not transfer to domains with different expert usage.

- For Mixtral‑8×7B at 2.5 bpe, GEMQ’s average zero-shot accuracy (65.13) slightly trails the uniform schedule (65.49) despite improved perplexity (Table 1), suggesting that global LP + router FT does not uniformly dominate in accuracy at milder compression.

### Questions
- How does GEMQ behave with richer candidate sets, e.g., allowing 0-bit (expert skipping/pruning) or 4-bit experts, and varying attention precision?

- How robust are the LP coefficients to calibration sampling noise? Please report the variance of Eq. 6 costs across multiple random calibration draws and its impact on the final allocation and accuracy.

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
This paper proposes a Global Expert-level Mixed-Precision Quantization (GEMQ) framework for large-scale Mixture-of-Experts Language Models (MoE-LLMs).

It mainly consists of three core modules: Global Linear Programming (Global LP) bit allocation: jointly optimizing the bit width of all experts; Router Fine-tuning (RFT): lightweightly adjusting routing parameters after quantization to correct routing offsets; and Progressive Quantization: gradually reducing the bit width and quantizing in stages to ensure stability at low bit widths. Experiments on multiple models show results that outperform PMQ and uniform quantization methods.

### Strengths
1. This paper proposes a practical solution to address the memory and bandwidth bottlenecks faced by MoE-LLMs during deployment.
2. The logical relationships between the three modules (Global LP, RFT, and Progressive Quantization) are clearly defined, and the theoretical motivation and loss surface analysis are presented in Figure 3.
3. The paper covers multiple mainstream MoE architectures, and evaluations include perplexity, zero/few-shot accuracy, and various ablation experiments.
4. Routing fine-tuning (RFT) requires only less than 0.05% parameter adjustments to significantly restore performance at low bit precision.

### Weaknesses
1. Insufficient theoretical support: Although the method is based on linear programming (LP), it does not provide any form of theoretical analysis or upper bound on the error.

2. Lack of quantitative ablation in incremental quantization: The authors visualize the smoothness of the loss surface in the appendix, but do not provide quantitative indicators to explain the contribution or error accumulation at each stage.

3. Insufficient comparison with state-of-the-art methods: Only PMQ and Uniform are compared, without comparison with recent representative methods such as MoeQuant and EAQuant.

[1]: EAQuant: Enhancing Post-Training Quantization for MoE Models via Expert-Aware Optimization

[2]: MoEQuant: Enhancing Quantization for Mixture-of-Experts Large Language Models via Expert-Balanced Sampling and Affinity Guidance

### Questions
See weeknesses

### Soundness
3

### Presentation
3

### Contribution
2
