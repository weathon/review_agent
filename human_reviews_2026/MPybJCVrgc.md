# SQ-format: A Unified Sparse-Quantized Hardware-friendly Data Format for Large Language Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
Post-training quantization (PTQ) plays a crucial role in the democratization of large language models (LLMs). However, existing low-bit quantizaiton and sparsification techniques are difficult to balance accuracy and efficiency due to the limited hardware support. For example, W4A8 can only achieve the same peak TOPS as W8A8 whereas the GPU-supported sparse data format (2:4 semi-structure sparse) is seldomly adopted due to the loss of accuracy. To bridge this gap, in this paper, we propose the Sparse-Quantized Format (SQ-format), which is a unified data format for quantization and sparsification potentially easily supported by new hardware and existing GPUs. SQ-format makes use of the fact that sparse matrix can be accelerated in high-precision, and low-precision matrix multiplication can also be accelerated accordingly. As such, SQ-format is proposed to achieve Pareto improvement between performance and throughput. This format is particularly suitable for activations with outlier inequality status and makes their static compression possible. We show the state-of-the-art PTQ performance with SQ-format, propose the hardware required to support it, and further offer the design exploration and insights for the next-generation Al acceleractors.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces SQ-Format for post-training quantization (PTQ). The format divides a tensor into blocks, and in each block, a portion of elements (controlled by sparsity ratio s) are quantized to a higher precision such as INT8, while the rest are quantized to a lower precision such as INT4. The goal is to leverage the sparsity of high-precision elements and achieve speedups close to full low-precision matmuls. The authors propose two algorithms for compressing weights and activations to SQ-Format. Their accuracy results are promising (both for weights and static/dynamic activations).

### Strengths
- The paper is well-written.
- Extensive experiments + good set of baselines.
- Interesting algorithmic contributions, including the static activation masks.

### Weaknesses
- The idea is not novel. Quantization + higher precision outlier formats have been considered by multiple works including SpQR (which the paper cites) and QUIK [1].
- The runtime improvement are more like a promise (if such hardware is designed), hence calling the method hardware-friendly in the title is a bit questionable.
- I am not convinced by the speedups. While there seems to be a kernel implemented for the static version, its throughput is only reported on a single matmul.

[1] https://arxiv.org/pdf/2310.09259

### Questions
1. Regarding the first two contribution listed in the intro (SQ-Format definition and implementation / Pareto improvement): I believe the definition is not novel, as I mentioned above. Additionally, can the authors provide a clear plot of accuracy vs runtime to show pareto superiority against other baselines (static activations would suffice, as you already have kernels for them)? In the current state, I'm not convinced merely from Tables 1 and 2 that there's pareto improvement.

2. More broadly, can you include speedup numbers in Tables 1 and 2 (for weights and static activations)?

3. Would it be possible to apply the SQ-Format to both operands? What are the challenges? A brief discussion would suffice.

4. The algorithm picks the most "important" elements to keep in higher precision. Aside from importance, looking at the low-precision quantization error could also interesting here. For example, if an important element can be perfectly captured by the low-precision component, then there is no reason to keep it in high precision. As a suggestion, wouldn't a combination of importance and low-precision error potentially achieve better results?

5. Any idea why the static activation strategy prefers smaller bank sizes? To me it seems counter-intuitive.

In general, I believe the paper's novelty is somewhat questionable, and the pareto superiority is not convincing. I would be open to increasing my score, depending the authors' rebuttal regarding novelty and questions 1 and 2.

### Soundness
3

### Presentation
4

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
Post-training quantization (PTQ) is crucial for LLM deployment, but current hardware makes low-bit quantization and sparsification hard to balance for accuracy and efficiency (e.g., W4A8 offers similar peak TOPS as W8A8; GPU 2:4 sparsity often hurts accuracy). The authors propose a unified Sparse‑Quantized Format (**SQ‑format**) that leverages high‑precision sparse acceleration and extends it to low‑precision matmul, enabling static compression for outlier‑skewed activations and yielding Pareto gains in performance vs. throughput on existing GPUs and future hardware. They report state‑of‑the‑art PTQ results, specify required hardware support, and provide design insights for next‑generation AI accelerators.

### Strengths
1. This paper is well organized.
2. This paper proposes a novel quantization format.
3. The proposed method shows SOTA performance.

### Weaknesses
1. I think the results of this paper are not easy to reproduce, since the authors do not include code. It is hard to believe the training-free approach can achieve much better performance than training-based SpinQuant, as demonstrated in the paper.
2. The baseline performance of this paper is inconsistent with their original paper.
3. The authors do not include E2E speedup results and memory costs of the proposed method. This is very important for the application of the proposed format. Only theatrical analysis is not reasonable.
4. This paper has claimed that the proposed method supports FP quantization. I believe they should include results for such a setting.

### Questions
N/A

### Soundness
2

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
This paper addresses the challenge of efficiently deploying large language models under both quantization and sparsity constraints, aiming to reduce model size and computation while maintaining accuracy. The authors propose SQ‑format, a unified sparse-quantized data format that encodes weights and activations in mixed precision and uses masks to indicate high-precision elements, allowing block-wise efficient computation. They introduce algorithms to determine which elements to quantize at low precision for weights and activations, using either static or dynamic mask strategies. Extensive experiments on LLaMA‑3 and Qwen‑3 models show that SQ‑format achieves comparable or better accuracy than prior quantization or sparsity methods while improving throughput and hardware efficiency. SQ‑format provides a hardware-friendly approach for mixed sparse-quantized LLMs, enabling practical deployment on accelerators without sacrificing performance.

### Strengths
- SQ‑format combines sparsity and quantization in a single representation, facilitating efficient computation on modern hardware.

- The paper carefully considers practical deployment, proposing both static and dynamic mask strategies to balance accuracy and efficiency.

- Evaluations on multiple LLMs (8B–70B) with standard benchmarks demonstrate that SQ‑format preserves accuracy while improving throughput.

- Addresses a key deployment challenge for LLMs, bridging the gap between algorithmic innovations and hardware execution.

### Weaknesses
- Limited comparison to extreme low-bit settings: The paper mainly evaluates INT4–INT8 and moderate sparsity; performance in ultra-low bit scenarios (e.g., W4A4) is unclear.

- Complexity of mask design: Dynamic mask selection may introduce runtime overhead, and static masks require careful calibration, which may complicate practical adoption.

- Specific to current hardware: While hardware-friendly, the proposed format is tuned to modern GPUs; applicability to other accelerators (TPU, AI chips) is not fully validated.

- Additional storage overhead: Maintaining masks for sparse/high-precision elements increases memory usage, which may be non-trivial for very large models.

- The algorithm design is simple and lacks novelty, but it imposes a very heavy burden on deployment. Although the final results are indeed good, there are still concerns about the future prospects of this method.

### Questions
Please refer to the weaknesses above.

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
The paper introduces SQ‑format, a unified, hardware‑friendly data format that combines sparsification and quantization for large language models (LLMs). It encodes a tensor as a sparse high‑precision component for critical values and a dense low‑precision component for the rest, thereby achieving a better trade‑off between accuracy and throughput. The authors present algorithms to apply SQ‑format to weights and activations, and provide hardware design insights. Empirical results on multiple LLMs show that SQ‑format reaches near W4A8 accuracy while maintaining W4A4‑level throughput.

### Strengths
This paper introduces the SQ-format, which integrates sparsification and quantization into a unified data representation, bridging the gap between algorithmic compression and hardware efficiency for improved throughput and accuracy.

This paper introduces SQ-format for weights and activations, achieving significant throughput gains while maintaining near W4A8-level accuracy, effectively balancing efficiency and performance.

This paper introduces a static activation splitting strategy that reduces runtime overhead, making SQ-format more practical for deployment on current AI accelerators.

### Weaknesses
1. The paper proposes a unified sparse + quantized format (SQ‑format) combining high‑precision for critical values and low‑precision + sparsity for the rest. However, previous work, such as SpQR: A Sparse‑Quantized Representation for Near‑Lossless LLM Weight Compression (Dettmers et al., 2023), already investigates the idea of preserving a small subset of weights in high precision and quantizing the remainder. While SQ‑format adds the “bank” structure and hardware‑mapping discussion, the paper could do more to clearly highlight what is novel beyond those prior methods.

2. The authors argue that SQ‑format is “hardware‑friendly” and outline required hardware support, but they offer only simulation or theoretical throughput estimates—not measured latency, power, or memory‑bandwidth results on real GPUs or accelerators.

### Questions
1. How do you select the ratio of high‑precision elements in the “bank” structure (bank size  b, sparsity rate s) as model size increases (e.g., 8B → 70B)?

2. Can you quantify the extra memory bandwidth or branching overhead introduced by decoding the SQ‑format (high/low precision mix + sparsity mask) compared to a uniform low‑precision format?

3. If activation distributions shift (e.g., instruction‑tuned or domain‑adapted LLMs), how robust is the static activation split strategy, and what is the accuracy or latency impact?

### Soundness
3

### Presentation
2

### Contribution
3
