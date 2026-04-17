# CodeQuant: Unified Clustering and Quantization for Enhanced Outlier Smoothing in Low-Precision Mixture-of-Experts

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Outliers have emerged as a fundamental bottleneck in preserving accuracy for low-precision large models, particularly within Mixture-of-Experts (MoE) architectures that are increasingly central to large-scale language modeling. Under post-training quantization (PTQ), these outliers induce substantial quantization errors, leading to severe accuracy degradation. While recent rotation-based smoothing techniques alleviate the problem by redistributing outlier magnitudes, residual errors remain and continue to impede reliable low-precision deployment.

In this work, we tackle this challenge by introducing CodeQuant, a unified quantization-and-clustering scheme that contains smoothing activation outliers via learnable rotation and absorbing weight outliers into fine-tuned cluster centroids for MoE. This design reduces the influence of extreme values by fitting them within cluster centroids, thereby lowering quantization error while maintaining expressive capacity. Coupled with a dedicated kernel design for GPU and CPU, CodeQuant achieves up to $4.15\times$ speedup while delivering significantly higher accuracy than state-of-the-art quantization approaches across diverse MoE models. Our results highlight CodeQuant as a promising direction for efficient and accurate deployment of MoE-based large language models under low-precision constraints. Our code is available at https://github.com/SAI-Lab-NYU/CodeQuant.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates the efficiency of MoE-based LLMs and proposes a vector quantization method named CodeQuant, which incorporates learnable rotation and clustering operations to achieve competitive W4A4 performance. The authors further design optimized LUT kernels to demonstrate improvements in execution latency.

### Strengths
1. The proposed hardware-oriented implementation effectively enhances the efficiency of MoE-based LLMs.
2. The specially designed component for MoE, such as the KL loss term, is validated to be effective through ablation studies.

### Weaknesses
1. Since learnable rotation has been extensively explored in prior LLM post-training quantization (PTQ) works such as SpinQuant and OSTQuant, the paper should more clearly articulate its key differences and advantages over these baselines.
2. Some of the compared baselines appear outdated and do not represent the current state-of-the-art.
3. The proposed kernel is evaluated only within the Accel-Sim framework, lacking validation on real hardware

### Questions
1. What is the additional training cost introduced by the learnable rotation matrix?
2. Could you include comparisons with more competitive rotation-based quantization baselines mentioned in the Experiments section, such as DuQuant and SpinQuant?
3. Could you report actual latency and memory measurements on real Nvidia GPUs (e.g., RTX 4090 or A100) to support the Accel-Sim results?

### Soundness
2

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
2

### Summary
The paper introduces CodeQuant, a quantization method for MoE models. It combines Activation-oriented Outlier Smoothing (AOS), Adaptive Weight Clustering with Centroid Finetuning (ACCF), and Permutation-Invariant Outlier Grouping (POG), and provides kernels for LUT-driven execution. The experiments evaluate both quantization accuracy and latency.

### Strengths
1. AOS directly optimizes activation quantization error on calibration data. The use of Cayley transforms to enforce orthogonality is simple and numerically stable compared with naive orthogonalization schemes.

2. ACCF alternates optimization over centroids and assignments, which is well motivated. The additional KL term for router logits is also reasonable and conceptually sound.

3. POG targets a very practical issue: improving block-wise clustering quality under tight centroid budgets.

4. The LUT-based kernel is evaluated via simulation to estimate speedups and memory savings, and the reported numbers appear reasonable.

5. Overall, the motivation is clear, and both the method and the experimental design are appropriate to support the paper’s claims.

### Weaknesses
1. Some of the bolded entries in the tables appear incorrect. For example: (i) W4A4 Embedding, Phi-mini-MoE-Instruct, A-c dataset; (ii) W8A4 Embedding, Phi-mini-MoE-Instruct, WG dataset.

2. The GPU evaluation of the proposed method is based solely on simulation, whereas QuaRot and SqueezeLLM are benchmarked on real GPUs. In addition, there may be non-trivial overhead, especially from POG, which could introduce extra computation or memory traffic. The authors should discuss these potential costs in more detail.

3. The experiments are limited to perplexity and commonsense QA benchmarks. Additional benchmarks such as GSM8K or MATH would provide a more comprehensive assessment of the method’s impact on reasoning-heavy tasks.

4. The paper does not report the pre-processing (quantization runtime). Including this would help clarify the practical cost of applying the method in real deployments.

### Questions
See weaknesses above.

### Soundness
2

### Presentation
2

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
This paper proposes a codebook-based clustering method for quantizing Mixture-of-Experts (MoE) models. To mitigate the outlier issue, it combines 1) activation smoothing, which transfers outliers to weights, and 2) a training-based clustering method to reduce weight quantization error. The paper focuses on MoE LLMs, and the training loss is carefully designed to align the token routing behavior of the quantized model with the original model.  The authors validated inference performance benefit via a simulation framework and proposed insights for hardware and software co-design.

### Strengths
-  The work focus the timely and challenging problem of quantizing Mixture-of-Experts (MoE) models. The training loss is well-tailored for these architectures, giving it practical significant relevance for industry deployment.
- It provides a analysis on the computational patterns for clustering-based GEMM operations, offering a perspective for future hardware and software co-design.

### Weaknesses
- The technical innovation appears limited, as the method primarily combines previously established techniques of outlier transformation and clustering.
- The benchmarking and accuracy evaluation are conducted on relatively small models. Since MoE models are typically very large, validating the method on larger-scale models (e.g., Qwen-235B, DeepSeek-R1) would strengthen the claims. Furthermore, evaluation is limited to next-token prediction tasks; testing on more complex reasoning benchmarks (e.g., GSM8K, MATH500) would be more compelling for modern LLM capabilities.
- While inference performance is compared, a more detailed breakdown is lacking. It would be beneficial to provide detailed testing parameters and analyze performance separately for the prefill and decoding stages, as they have different computational characteristics (memory bandwidth vs. compute density).

### Questions
- What is the quantitative impact of the KL loss on final accuracy? Is there evidence that preserving the original token-expert assignment is necessary after quantization alters the weights?
- Computational Overhead:** What percentage of the total inference time is spent on the clustering operations? How does the performance benefit differ between the prefill and decoding stages?

### Soundness
3

### Presentation
2

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
The paper proposes CodeQuant, a unified clustering + quantization framework tailored for Mixture-of-Experts (MoE) models. Empirical results on several MoE models show improved perplexity. The paper claims up to 4.15× speedup with negligible accuracy loss.

### Strengths
- Addressing outliers in MoE quantization is highly relevant since activation and weight spikes are key bottlenecks in scaling low-bit inference.
- Combining activation smoothing (AOS), column reordering (POG), adaptive clustering (ACCF), and custom LUT kernels forms a complete end-to-end quantization pipeline.

### Weaknesses
- The evaluation omits strong baselines such as Duqant and GPTQ.
- Applying learned rotations within MoE layers risks disturbing the routing behavior. Although the paper mentions avoiding direct interference with router inputs, no quantitative analysis or bounds on routing consistency are provided.

### Questions
- On which layers or experts does CodeQuant fail? Are there cases where AOS rotation worsens quantization error or diverges?
- How is the trade-off between the number of centroids and quantization precision chosen?
- What is the formal connection between ACCF loss and quantization error？How do you ensure centroid fine-tuning does not simply overfit the calibration set?

### Soundness
2

### Presentation
3

### Contribution
2
