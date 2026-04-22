# SINQ: Sinkhorn-Normalized Quantization for Calibration-Free Low-Precision LLM Weights

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 6

## Abstract
Post-training quantization has emerged as the most widely used strategy for deploying large language models at low precision. Still, current methods show perplexity degradation at bit-widths $\leq 4$, partly because representing outliers causes precision issues in parameters that share the same scales as these outliers. This problem is especially pronounced for calibration-free, uniform quantization methods. We introduce SINQ to augment existing post-training quantizers with an additional second-axis scale factor and a fast Sinkhorn–Knopp–style algorithm that finds scales to normalize per-row and per-column variances, thereby minimizing a novel per-matrix proxy target for quantization: the matrix imbalance. Our method has no interactions between layers and can be trivially applied to new architectures to quantize any linear layers.
We evaluate our method on the Qwen3 model family and DeepSeek-V2.5. SINQ improves WikiText2 and C4 perplexity significantly against uncalibrated uniform quantization baselines, incurs a 0-2% compute overhead, and can be further enhanced by combining it with calibration and non-uniform quantization levels. Code is available in the supplementary.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents SINQ, a post-training quantization method for large language models that uses dual-axis scaling and a modified Sinkhorn–Knopp algorithm to minimize "matrix imbalance," improving perplexity on models like Qwen3 and DeepSeek-V2.5 while being compatible with mainstream paradigms such as NF4 and AWQ. However, it suffers from critical flaws—including unclear experimental details, unproven core metrics, limited innovation, and missing key comparisons—resulting in a "Reject" score, though addressing these issues could lead to reconsideration.

### Strengths
1. The necessity of dual-axis scaling has not been verified: The paper fails to compare the performance of "row-only scaling", "column-only scaling", and "dual-axis scaling". This makes it impossible to prove the advantage of "dual-axis scaling" — for instance, if column-only scaling can achieve similar performance, the additional complexity of dual-axis scaling becomes meaningless.  
2. It is a non-isolated solution and can be combined with mainstream quantization paradigms to expand application scenarios (e.g., NF4, AWQ).

### Weaknesses
1. Lack of experimental details: For the SINQ algorithm, the "specific selection rule for σmin" and "threshold setting for early-stopping" are not provided. In A-SINQ, the calibration dataset for AWQ (e.g., sample count, source) is not mentioned. While AWQ typically relies on 128–512 samples from the C4 dataset, the paper does not confirm consistency with this practice, nor does it explain whether calibration samples affect the optimization results of SINQ.  

2. The rationality of using matrix imbalance as a surrogate metric is unproven: The paper defines matrix imbalance as $I(W)=\sigma_{min}(W)/\sigma_{max}(W)$ (the ratio of the minimum to maximum standard deviations of rows and columns) and claims that minimizing $I(W)$ improves quantization accuracy. However, it does not establish a mathematical connection between "matrix imbalance" and "quantization error" — for example, why a smaller $I(W)$ leads to lower post-quantization MSE or perplexity. The paper only observes through Figure 2 that "minimizing $I(W)$ reduces kurtosis", but fails to analyze the relationship between kurtosis and quantization error (e.g., whether reduced kurtosis necessarily decreases distribution overlap under low-bitwidth conditions), leaving the core assumption without theoretical support.  

3. The paper modifies the standard Sinkhorn-Knopp algorithm to normalize row and column standard deviations, but does not prove the convergence of the modified algorithm (e.g., whether iterations enter cycles or if a unique fixed point exists). Additionally, it does not explain the basis for selecting the number of iterations $n_{iter}$ (e.g., why a fixed number is chosen instead of dynamic stopping based on the convergence threshold of $I(W)$), casting doubt on the algorithm’s stability. While Figures 2(a)(b) show that $I(W)$ stabilizes after 10 iterations, the paper does not explain "why 10 iterations are optimal" nor conduct ablation studies on the impact of $n_{iter}$ on performance.  

4. The paper adopts the sequence "SINQ normalization → AWQ scaling → quantization" (Section 2.2.2). However, AWQ’s core lies in "activation-aware weight scaling", which relies on the distribution characteristics of original weights (e.g., correlation between activations and weights). Prior SINQ normalization alters the weight distribution, potentially disrupting the correlation relied on by AWQ. The paper does not compare the performance of alternative sequences such as "AWQ first, then SINQ" or "joint optimization of SINQ and AWQ", making it impossible to verify the rationality of the current sequence. It also fails to decompose contributions from "SINQ alone", "AWQ alone", and "their combination", leaving uncertainty about whether performance gains stem from dual-axis scaling or AWQ.  

5. SINQ’s innovations are more akin to "combinatorial optimization of existing technologies" with limited breakthroughs: SINQ merely expands the scaling dimension from "weight-activation" or "single-axis weight" to "dual-axis weight", which is essentially an extension of the scaling target rather than a fundamental innovation. The standard Sinkhorn-Knopp algorithm normalizes row and column sums; SINQ only replaces the target with "row and column standard deviations" while retaining the algorithm framework (alternating iterative normalization), making this a routine modification rather than an innovative design.  

6. Key comparative methods are missing: Representative methods such as FlatQuant and OSTQuant are not included in the comparisons.  

7. The necessity of dual-axis scaling has not been verified: The paper fails to compare the performance of "row-only scaling", "column-only scaling", and "dual-axis scaling". This makes it impossible to prove the advantage of "dual-axis scaling" — for instance, if column-only scaling can achieve similar performance, the additional complexity of dual-axis scaling becomes meaningless.  

8. For common datasets (HellaSwag, PIQA, MMLU), accuracy is the standard evaluation metric, while "Flip rates" are uncommon. The paper provides no justification for selecting this metric.  

Based on the above weakness, I would assign a Reject score. I look forward to the authors addressing the aforementioned problems in future revisions, and I would be happy to reconsider and raise my score accordingly.

### Questions
See "Weaknesses"

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
4

### Summary
The paper proposes SINQ, a weight-only PTQ scheme that applies  dual scaling one scale per row and one per column, to each weight tile. The aim is to mitigate outliers along both dimensions and make ≤4-bit uniform quantization easier. The authors also introduce a proxy metric, matrix imbalance (the ratio of the largest to smallest row/column standard deviations), and a Sinkhorn–Knopp–style iteration that alternately normalizes row and column standard deviations to reduce this imbalance prior to quantization.

### Strengths
- **Simple, calibration-free** recipe that improves perplexity compared to strong uniform PTQ baselines at 3–4 bits across model sizes.
- Clear ablations comparing **imbalance** vs **kurtosis** as proxies for quantization difficulty.
- **Competitive results**, outperforming HQQ, GPTQ and AWQ in many settings.

### Weaknesses
- **Hardware evidence.** There are no end-to-end **inference throughput/latency** results or **kernel-level utilization** measurements; only **quantization-time** is reported. Without runtime data on common backends, deployment value is hard to assess.
- **Weight-only scope.** Although activation quantization is discussed, the experiments are weight-only. It remains unclear how dual scaling interacts with **W×A** low-precision matmuls and whether common kernel fusions remain intact.
- **Baselines.** Since the focus is weight quantization, a head-to-head with **codebook/rotation** approaches (e.g., **QuIP#**, **QTIP**) would strengthen the empirical case; these are mentioned in related work but not featured in the main tables.
- **Further empirical results.** Results would be more convincing with additional families (e.g., **LLaMA**, **Phi**) to demonstrate generality.
- CrossQuant appears closely related but is missing from the citations; it would help to explain the methodological differences and compare performance.

CrossQuant: A Post-Training Quantization Method with Smaller Quantization Kernel for Precise Large Language Model Compression., Liu, Wenyuan, et al. (2024).

### Questions
check the weakness, 

1- how would dual scaling be implemented in hardware for W×A low-precision matrix multiplications, and what impact would it have inference speed?

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
5

### Summary
This paper presents a post-training weight quantization method on a dual-scale matrix quantization scheme.  
Comparison to existing calibration-free PTQ methods including rotation-based methods is conducted empirically.

### Strengths
+ Clearly presented idea. 
+ Potentially significant practical value.

### Weaknesses
- Practical overhead of dual-scaling on actual HW is not comprehensively discussed, except for memory efficiency.

### Questions
* I am not sure the comparison against Hadamard rotation, etc. is also based on dual-scale scheme or not--it should be for a fair comparison.  
* Random rotation is purported to mix channels and thereby eliminate outliers--this seems to be doing similar things as dual-scale.  Could you do an ablation study with (1) Hadamard + single-scale, (2) Hadamard + dual-scale, and (3) SINQ?

### Soundness
2

### Presentation
3

### Contribution
3
