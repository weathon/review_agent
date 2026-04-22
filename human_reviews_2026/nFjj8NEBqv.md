# SERQ: Saliency-Aware Low-Rank Error Reconstruction for LLM Quantization

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 4, 8, 4

## Abstract
Post-training quantization (PTQ) has emerged as a prevailing technique for deploying large language models (LLMs) efficiently in terms of both memory and computation, across edge devices and server platforms. Existing PTQ methods primarily aim to reduce precision in weights and activations by mitigating quantization errors caused by channel-wise outlier activations (e.g., pre-quantization scaling, online transformations, or low-rank error reconstruction). Among these approaches, error reconstruction with low-rank adaptation (LoRA) has proven particularly effective, as it introduces a lightweight auxiliary computation path without requiring heavy optimization or additional online layers. However, prior studies reveal severe accuracy degradation under W4A4 settings, and conventional low-rank adaptations rely on two sequential factors, necessitating intermediate quantization during inference and thereby limiting low-precision efficiency. In this work, we propose SERQ, a saliency-aware error reconstruction method for low-bit LLM inference that employs a single low-rank compensation matrix. SERQ preserves efficient 4-bit matrix multiplication in linear layers by jointly mitigating quantization errors arising from both activation and weight saliency through three stages: (1) static activation flattening, (2) saliency-aware error reconstruction, and (3) offline weight permutation. The method incurs additional computation only for low-rank error reconstruction via a single decomposition, while all other operations are performed offline, thereby keeping latency overhead minimal. Empirically, SERQ outperforms prior error reconstruction methods under both W4A8 and W4A4 settings, and achieves higher accuracy than state-of-the-art rotation-based W4A4 approaches, while substantially reducing calibration complexity.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces SERQ. While existing low-rank adaptation methods are effective, they often face severe accuracy degradation in low-bit W4A4 (4-bit weight, 4-bit activation) settings and require two sequential factors, which limits inference efficiency. SERQ addresses this by employing a single low-rank compensation matrix to jointly mitigate quantization errors arising from both activation and weight saliency. The method operates in three stages: (1) static activation flattening, (2) saliency-aware error reconstruction, and (3) offline weight permutation. This design preserves efficient 4-bit matrix multiplication, keeps latency overhead minimal by performing most operations offline , and empirically outperforms prior error reconstruction and state-of-the-art rotation-based W4A4 approaches.

### Strengths
1. The motivation of this paper is clear, quantizing both main branch and lora branch to W4A4, as well as reducing the overhead of lora branch through saliency channel detection.
2. The ablation studies is comprehensive.

### Weaknesses
1. More recent state-of-the-art baselines—such as DuQuant [1], OSTQuant[2] and FlatQuant[3] for W4A4KV4 quantization—should be included for comparison.
2. SVDQuant [4] also use channel-wise scaling (static activation flattening in this paper) and LoRA error compensate, though it was used in diffusion model. This paper should disscuss more about such similar work.
3. End-to-end prefilling and decoding speedup should be included to demonstrate the efficiency of proposed method. 
4. The comparison results in Table 2 seems unfair. Flattening methods and proposed SEQR use different quantization formats. Additionally, the SEQR-MXFP4 average bit-width is higher than QuaRot and SpinQuant.
[1]  DuQuant: Distributing Outliers via Dual Transformation Makes Stronger Quantized LLMs, NeurIPS 2024
[2] OSTQuant: Refining Large Language Model Quantization with Orthogonal and Scaling Transformations for Better Distribution Fitting, ICLR 2025
[3] FlatQuant: Flatness Matters for LLM Quantization, ICML 2025
[4] SVDQUANT: ABSORBING OUTLIERS BY LOW-RANK COMPONENTS FOR 4-BIT DIFFUSION MODELS, ICLR 2025

### Questions
What quantization format is used for flattening methods in Table 2?

### Soundness
3

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
The paper proposes SERQ, a method for 4-bit quantization of large language models that uses a single low-rank compensation matrix to correct quantization errors. The approach combines three components: (1) static activation flattening via SmoothQuant-style scaling, (2) saliency-aware error reconstruction using only the most significant weight rows, and (3) offline weight permutation to avoid runtime overhead. The authors claim their method enables true end-to-end 4-bit computation while outperforming prior low-rank error reconstruction methods like L2QER and rotation-based approaches like QuaRot and SpinQuant.

### Strengths
● Novel Design for Latency Reduction: The paper proposes a scheme that combines a single compensation matrix with an offline permutation, with the goal of eliminating the latency overhead found in conventional two-factor error correction methods.
● Presents Experimental Results on Multiple Models: The paper reports experimental results across several modern LLMs and includes latency measurements on recent hardware to support its efficiency claims.
● Addresses a Significant Problem: The work targets the challenging and important problem of W4A4 quantization, which is of high interest to the community for efficient LLM deployment.

### Weaknesses
While the paper presents some interesting ideas, it suffers from significant flaws in its theoretical grounding, experimental validation, and practical considerations, which severely undermine the credibility and value of its contributions.
1. Critically Flawed and Incomplete Experimental Validation
● Flawed Experimental Comparisons: The paper's main results are built on unfair comparisons. For instance, SERQ is allocated a higher bit-budget (4.37 bits) than its main competitors (~4 bits), which may in itself account for its accuracy advantage. Furthermore, the performance of the key L2QER baseline is suspiciously low on certain models, casting doubt on the entire set of comparative results.
● Critically Limited Scope and Unproven Generalization: The paper’s experimental validation is fundamentally disconnected from modern, real-world LLM scenarios. The evaluation is confined to small-scale (<13B), dense models, failing to address three critical dimensions: (1) Scale: It lacks validation on large-scale models (70B+), where quantization challenges are most severe. (2) Architecture: It provides no evidence on diverse, prevalent architectures like Mixture-of-Experts (MoE), where its core saliency heuristic may not hold. (3) Context Length: It fails to evaluate performance in long-context scenarios (e.g., 8K-32K), which are vital for modern applications and where quantization errors are known to accumulate. This critically narrow scope makes it impossible to assess the method's true scalability and practical utility.
2. Dismissal of Practical Issues and Lack of Transparency
● Dismissal of Practical Drawbacks: The paper touts its offline permutation as a "zero-latency" solution while completely ignoring the immense software engineering challenges and loss of modularity it introduces, especially concerning its interaction with distributed inference techniques like tensor parallelism. This dismissal of practical drawbacks makes the method's utility questionable.
● Missing Analysis of Quantization Cost: The paper completely omits an analysis of its quantization cost. As a Post-Training Quantization (PTQ) method, the time and compute resources required for calibration are critical efficiency metrics. The absence of this data prevents a full assessment of its overall cost-effectiveness.
● Poor Explanation of Key Components: The paper fails to clearly explain how its best-performing variant (SERQ-GPTQ) works. The description in the appendix is contradictory and confusing, which is a major reporting weakness for a paper that claims top results with this variant and impacts its reproducibility.
3. Weak Theoretical Underpinnings and Incremental Contribution
● Incremental Contribution and Overstated Novelty: The method is essentially a patchwork of existing techniques (SmoothQuant, AWQ's saliency metric, LoRA-style correction). Its primary novel component—the offline permutation—is more of an engineering trick than a fundamental algorithmic breakthrough. The paper conveniently ignores the incremental nature of its contribution and overstates its novelty.
● Limited Theoretical Justification for Saliency Metric: The method relies entirely on a heuristic for saliency borrowed from AWQ (identifying rows by activation scale). The paper fails to provide any theoretical justification as to why this simple statistical metric is optimal for error reconstruction, nor does it rigorously compare it against other potential channel selection paradigms.

### Questions
Please address my concerns in Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work proposes SERQ, a post-training quantization method that uses a single low-rank term to only compensate the error for salient weight rows in a pretrained linear layer, which further improves the inference throughput over baselines.

### Strengths
- The motivation is clear and the method is straightforward but effective
- The evaluation includes both model performance and hardware results

### Weaknesses
- The evaluation lacks model larger than 13B: all results were collected on model between 1B and 13B. It will be convincing if the author could offer results on larger models like 70B to verify the scalability of SERQ
- Details of runtime/hardware performance needs further clarification. Could the author elaborate more how a SERQ layer is accelerated on GPU? are there kernel fusions?

### Questions
Apart from the questions in Weakness section, 

1. What does the "Quarot" block in Figure 2(b) mean? Is this for rotating activations?
2. SERQ is a heuristic based method. An ablation study of the static activation flattening seems needed here. 
    - SmoothQuant was originally proposed for INT8 quantization, while SERQ uses MXFP.
    - In line 262, "However, unlike standalone prior approaches, the combined use of low-rank reconstruction enables effective compensation for the induced weight errors.". Thus the static activation flattening transfers the challenge of activation quantization to weights using the scaling $s$, but the term to compensate to this compensate weight error, $R$, is in low-precision in SERQ. I wonder whether the activation flattening is really helpful in this case considering MXFP has shared scales to accommodate outliers.
3. The GPU performance evaluation is very helpful. Could the author offer more details of the setup of GPU performance, like the batch size, token numbers, and tokens per second throughput?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes a saliency-aware error reconstruction method named SERQ for LLMs in W4A4 using a single low-rank compensation matrix. The main contributions are threefold: static activation flattening, saliency-aware error reconstruction, and offline weight permutation. Experimental results show accuracy improvements with small computation overhead.

### Strengths
* Well-written with good presentation
* Systematic evaluation and good results

### Weaknesses
* Missing related work
* Missing theoretical justification

### Questions
* Ablation study: how much does static activation flattening affect your results?
* What are the design space of choosing saliency metrics? What do the authors mean by "sufficient"?
* Could the authors comment on how their work compares to QERA (ICLR 2025)?

### Soundness
2

### Presentation
3

### Contribution
2
