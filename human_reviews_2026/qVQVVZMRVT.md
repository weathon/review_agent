# No outlier channels but with outlier blocks

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 4, 8

## Abstract
With the rapid scaling of large language models, achieving efficient compression while maintaining model performance has become a critical challenge. To address the limitations of existing non-uniform quantization methods, which typically rely on fixed codebooks and require costly optimization, we propose a novel arbitrary bit-width non-uniform Quantization (NuBitQ). The framework enables flexible, layer-specific quantization strategies, significantly enhancing adaptability and efficiency. Notably, traditional outlier compensation methods used in uniform quantization are ill-suited for the anomalous distribution characteristics encountered in our context. To address this, we design a novel outlier evaluation metric that integrates weight perturbation, activation distribution, and perturbation propagation. Based on this metric, we further develop an Outlier Compensation Plugin (OCP) that implements multi-level, fine-grained outlier compensation strategies, effectively mitigating performance degradation caused by outliers. Our approach avoids direct complex Hessian computation and fine-tuning, offering strong applicability and scalability. Extensive experiments on multiple tasks and across various model series demonstrate the effectiveness of the proposed approach.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work studies the (1) Non-uniform quantization, (2) outlier impact, and (3) outlier compensation by minimization. Overall, the ultimate goal of this work is to push the performance boundary in non-uniform quantization. However, I found the organization of this work to be unclear and missing.

### Strengths
+ Thorough evaluation of non-uniform quantization across various models and benchmarks.

### Weaknesses
- The figures in this paper are not informative. I cannot understand them if not reading the papers. 

- I have several concerns regarding OCP. First, the final OCP methodology seems like the standard calibration with different levels of granularity. This resembles the AQLM's multi-phase optimization that first tackles the layer-wise quantization and then tackles the inter-layer error. Conceptually, there is nothing new about OCP, given that it has been extensively applied in dozens of PTQ works. 

- Moreover, the OCP method seems to be compatible with any quantization algorithms, not just non-uniform quantization. If this is the case, then the authors need to compare a lot of uniform baselines. 

- Third, I did not understand the necessity of Section 3.2.2. I have checked Appendix D for the detailed derivation of the outlier score. However, I should emphasize that the derivation is definitely not rigorous. The logarithm makes no sense to add. To me, this is just to prevent the exploding effect of the subsequent layers' weight norm. The Jacobian should contain the gradients of na on-linear function, while this work directly ignores them. Furthermore, this outlier score has nothing to do with outliers; it is just a metric to analyze importance. Again, this score seems to be applicable to all kinds of parameters and activations, not just outliers. I could not see how this is *outlier-centric*.

### Questions
I am curious to see if this non-uniform weight-only quantization can be accelerated on the modern LLM inference engine such as vLLM and SGLang.

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the challenges of quantizing LLMs, particularly focusing on the limitations of non-uniform quantization methods. The authors argue that while non-uniform quantization reduces overall error compared to uniform methods, it creates novel and less-understood outlier patterns that are not concentrated in specific channels but rather distributed across what they term "outlier blocks." To tackle this, they propose a two-part framework. The first part is NuBitQ, a flexible non-uniform quantization scheme supporting arbitrary bit-widths and layer-specific configurations without relying on costly Hessian computations. The second contribution is the OCP, a module designed to mitigate performance degradation from these outlier blocks. The OCP is guided by a novel, efficiently computed outlier score that approximates the impact of quantization error by considering weight perturbation, activation statistics, and error propagation through subsequent layers. Based on this score, OCP applies a hierarchy of compensation strategies at the sub-layer, block, and model levels. The authors conduct extensive experiments on LLaMA3, Qwen, and Gemma models, demonstrating that their framework achieves state-of-the-art performance, especially in ultra-low-bit settings.

### Strengths
1.The paper provides a valuable analysis of how outlier patterns differ between uniform and non-uniform quantization. The identification of "outlier blocks" rather than just "outlier channels" as the primary challenge in non-uniform quantization is a key insight that effectively motivates the need for new compensation strategies.

2.The Outlier Compensation Plugin is designed as a modular, plug-and-play component. This highlights its general utility.

3.The paper presents comprehensive experimental results across multiple model families and sizes. The framework demonstrates near-lossless performance at 4-bit and shows particularly strong gains in the challenging ultra-low-bit regimes, outperforming state-of-the-art baselines in both perplexity and downstream task accuracy.

### Weaknesses
1.The paper claims that OCP is a generalizable, "plug-and-play" module. While perplexity results in Table 1 support this, the downstream task accuracy results in Table 2 present a more mixed picture. When OCP is added to AQLM and GPTVQ, the average MMLU score shows negligible improvement or even a slight decrease. This suggests that the OCP's effectiveness may be dependent on the underlying quantization scheme, potentially conflicting with the claim of broad applicability.

2.The complete NuBitQ-OCP framework involves a considerable number of hyperparameters, including the quantization parameters, the number of compensation levels, and the thresholds that govern the selection of compensation strategies. Although the paper includes some ablation studies, it lacks a broader discussion on the sensitivity and practical guidance for setting these parameters, especially the compensation thresholds, which appear crucial for balancing performance and overhead. This complexity may hinder the method's ease of adoption.

### Questions
1.The results in Table 2 indicate that applying OCP to AQLM and GPTVQ does not consistently improve, and sometimes slightly degrades, performance on downstream tasks like MMLU. Why does OCP fail to provide a clear benefit in these cases, unlike the significant perplexity improvements shown in Table 1? Does this imply that OCP is specifically tuned for the error patterns produced by NuBitQ or that it might conflict with the optimization strategies already present in methods like AQLM and GPTVQ?

2.The compensation strategy relies on two thresholds, $\theta_1$ and $\theta_2$, to switch between model-level, Transformer-level, and linear-level compensation. How were these thresholds determined for the experiments reported in the paper? How sensitive is the final performance to their values, and would they need to be re-tuned for different model architectures or bit-widths, potentially limiting the "plug-and-play" nature of the OCP?

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
This paper proposes a **fine-grained, multi-codebook, multi-vector quantization** strategy that adapts bit-width and codebook design on a per-layer basis. In addition, an **outlier-based compensation** mechanism is introduced to further improve the overall quantization performance.

### Strengths
1. The proposed outlier compensation plugin effectively enhances the performance of the quantized model.
2. The evaluation covers multiple types of LLMs and diverse downstream tasks, demonstrating general applicability.

### Weaknesses
1. The discussion in the appendix omits some important state-of-the-art baselines, such as **DuQuant** and **ResQ**.
2. Certain results are not state-of-the-art but are marked as “best” in Table 2, which may cause confusion.
3. The proposed non-uniform quantization may introduce additional inference overhead, which is not clearly discussed.
4. The abbreviation **OCP** is introduced in the Introduction without prior definition or explanation.

### Questions
1. Could you include additional baselines such as **DuQuant** and **ResQ** in **Table 4** for fairer comparison?
2. What is the additional computational or inference cost introduced by **NuBitQ**?
3. Could you provide end-to-end measurements of speedup and memory usage to validate the claimed efficiency?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes NuBitQ (a vector-quantization-style, layer-wise, residual codebook design) and OCP (a three-level compensation guided by a novel block-level outlier score). It achieves near-lossless 4-bit and strong 2–3-bit results on several LLMs, without Hessian or fine-tuning.

### Strengths
1. Identifies a different outlier pattern for non-uniform quantization (block-level), and formalizes a usable score.

2. Practical PTQ: no Hessian, no fine-tuning; clear knobs (𝑟, 𝑐, 𝑑, 𝑔) with an explicit compression formula.

3. Consistent empirical gains at 4/3/2 bits across model families; ablations on hyper-params and compensation granularity are informative.

### Weaknesses
1. While the method avoids fine-tuning/Hessian, it still requires a nontrivial search over (r,c,d,g) and threshold(s) to balance accuracy vs. memory/latency. The paper shows trends, but lacks a budget-constrained, step-by-step selection procedure (e.g., how to pick (r,d) first, then threshold, under a fixed memory/latency cap). This limits immediate reproducibility in large-scale deployments.

2. The paper does not discuss systems constraints: current vendor fast paths primarily target uniform INT/FP formats; non-uniform (codebook/LUT) typically needs custom kernels or on-the-fly dequant, making throughput sensitive to group size, batch size, and cache behavior. A brief analysis of runtime/throughput vs. baseline INT8/FP8 would clarify practicality.

3. Prior work shows that rotation/affine transforms can mitigate outliers before quantization. The paper compares methods side-by-side, but does not systematically evaluate combinations (e.g., rotation → non-uniform weight quant → OCP). A combined study could reveal whether gains are additive or saturating, and where each component contributes.

### Questions
1. How stable is the outlier score ranking under domain shift and smaller calibration sets?

2. Can OCP selection be cast as a budgeted optimization to auto-derive Pareto frontiers?

3. What is the end-to-end latency/throughput impact on 70B-scale serving?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes NuBitQ, a flexible non-uniform quantization framework for large language models (LLMs), along with a plug-and-play Outlier Compensation Plug-in (OCP) to address outlier-induced performance degradation in low-bit settings. Extensive experiments showing near-lossless 4-bit performance and strong results at 2–3 bits across multiple LLMs.

### Strengths
- Introduces a novel outlier score that integrates both local sensitivity and global propagation effects, which is more tailored to non-uniform quantization than prior Hessian-based methods.

- The OCP module is a lightweight, tuning-free compensation mechanism that is also applicable to other quantization methods.

- Supports arbitrary bit-widths and layer-wise customization, offering more flexibility than fixed-bit approaches.

- Comprehensive evaluations on multiple models (LLaMA, Qwen, Gemma) and tasks (perplexity, MMLU, QNLI, MNLI, etc.) demonstrate consistent improvements over SOTA methods.

### Weaknesses
- Lack of Hardware-Aware Latency Evaluation : Although Table 3(b) compares Ada vs. Ampere architectures, there is no end-to-end latency or throughput comparison against FP16 or other quantization methods.

- Lack of Direct Validation:The experimental results do not directly validate these specific theoretical claims.

### Questions
- Can you provide the end-to-end latency or throughput comparison against FP16 or other quantization methods.

- Could the authors present quantitative or visual evidence (e.g., density plots or clustering metrics) supporting this claim?

### Soundness
3

### Presentation
3

### Contribution
3
