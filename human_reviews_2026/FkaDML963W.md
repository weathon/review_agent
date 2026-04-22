# KVLinC: KV Cache Quantization with Hadamard Rotation and Linear Correction

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
Quantizing the key-value (KV) cache is a promising strategy for improving the inference efficiency of large language models (LLMs). However, aggressive quantization to very low precision (e.g., 2 bits) introduces significant errors in the stored key and value tensors, which propagate through the dot-product attention mechanism and ultimately degrade generation quality. To address this, we propose KVLinC, a framework to mitigate attention errors introduced by KV cache quantization in the extreme low-precision regime. KVLinC combines a Hadamard rotation, which reduces quantization error in values, with lightweight linear correction adapters that explicitly compensate for errors introduced by quantized keys. Across extensive evaluations on the LLaMA, Qwen2.5, and Qwen3 model families, KVLinC consistently matches or surpasses strong baselines while achieving higher KV-cache compression. Furthermore, we implement a custom attention kernel that results in upto 2.55x faster inference compared to Flash Attention baseline, enabling efficient long-context LLM inference.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work proposes a framework for KV-cache compression involving Hadamard rotation to simplify the quantization problem, combined with a lightweight correction adapter. The authors provide an ablation study on the choice of quantization axis and rotation, adopting the best-performing option. To compensate for quantization error, they train dedicated correction modules. The approach is validated on several transformer model families for both short and long context tasks.

### Strengths
* The proposed method achieves a favorable compression-performance trade-off, outperforming several known baselines from the literature.

* The method is accompanied by a custom attention kernel that speeds up inference.

### Weaknesses
* The overall idea of applying rotation followed by linear correction is not novel. The AQUA-KV [4] method also involves training predictors and cache rotation. While the predictor in this work is implemented differently, the core concept is similar.

* The ablation study on the choice of KV-cache quantization axis, despite being useful, is not novel and was previously explored in [1] and [2], where channel-wise key quantization was also found to be preferable.

* The comparisons in Tables 1 and 2 are not entirely fair. The bitwidth difference between methods may be up to **0.5 bits** on average, placing methods with smaller bitwidths at a disadvantage relative to those with higher bitwidths.

 * I suggest adding comparisons with additional baselines:
    * KVQuant [2] is an established cache quantization method
    * QJL [3] demonstrates strong performance at 3-bit compression and provides kernels for faster inference
    * AQUA-KV [4], being closest in terms of formulation, is a natural baseline

---
References

[1] Liu, Zirui, et al. "Kivi: A tuning-free asymmetric 2bit quantization for kv cache." arXiv preprint arXiv:2402.02750 (2024).

[2] Hooper, Coleman, et al. "Kvquant: Towards 10 million context length llm inference with kv cache quantization." Advances in Neural Information Processing Systems 37 (2024): 1270-1303.

[3] Zandieh, Amir, Majid Daliri, and Insu Han. "Qjl: 1-bit quantized jl transform for kv cache quantization with zero overhead." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 39. No. 24. 2025.

[4] Shutova, Alina, et al. "Cache Me If You Must: Adaptive Key-Value Quantization for Large Language Models." arXiv preprint arXiv:2501.19392 (2025).

### Questions
This work primarily targets KV cache compression of 2.71 bits per parameter on average. How well does the method perform under even stronger compression—around 2 bits or below?

### Soundness
3

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
This paper addresses the challenge of KV cache compression in large language model (LLM) inference, where key–value tensors dominate memory and bandwidth costs as context length grows.
The authors propose KVLinC, a method that combines (1) Hadamard rotation for distribution smoothing and (2) linear correction adapters to mitigate quantization-induced bias in the attention mechanism.

### Strengths
1. Empirical thoroughness: Evaluated on multiple model families, sizes, and long-context benchmarks (RULER, GSM8K, LongBench)
2. Strong performance at extremely low precision (2-bit): Outperforms several state-of-the-art quantization baselines across diverse tasks.

### Weaknesses
1. Ablation clarity: The paper would benefit from explicit comparisons between “rotation only,” “correction only,” and “both” configurations on the same setup.
2. The contribution is incremental, as both Hadamard rotation and linear correction have been introduced in prior work; this paper mainly demonstrates an effective combination of the two techniques.

### Questions
1. How does KVLinC interact with KV cache sparsification or compression-based pruning?
2. Furthuer ablation study needed, See weakness1.

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
3

### Summary
- Asymmetric Hadamard rotation and quantization. The authors analyze several rotation and axis configurations and find that quantizing raw keys channel-wise and Hadamard-rotated values token-wise yields the best trade-off. This differs from QuaRot, which apply Hadamard rotations to both keys and values before token-wise quantization. I believe the rationale is that keys exhibit heavy-tailed channel distributions that are sensitive to rotation, while values benefit from rotation.
- Linear correction adapters. Lightweight trainable modules are added to compensate for distortions in the attention weights induced by quantized keys. These adapters add <1% parameters and introduce constant memory overhead, scaling linearly with sequence length.
- A custom Triton kernel integrates quantization and correction for efficient decoding. Experiments across Llama-3, Qwen-2.5, and Qwen-3  and KIVI, Gear, and ResQ on short- and long-context benchmarks (Wikitext, GSM8K, BBH, RULER, LongBench).

### Strengths
- Clear motivation and strong practical relevance. The paper addresses a key bottleneck in LLM inference, the KV-cache memory footprint, and focuses specifically on the challenging 2-bit quantization regime, which is highly relevant for deployment.
- Well-engineered and empirically thorough. I like that the authors implemented a dedicated Triton kernel to show that there are gains rather than just arguing the same, but failing against optimized benchmarks.
- Effective combination of existing ideas. Although both Hadamard rotation to KV and lightweight adapters are known concepts, their combination and tuning for extreme low precision is original in execution and yields consistently strong results.
- Clarity and reproducibility. The paper is clearly written, with helpful figures and detailed methodology.

### Weaknesses
- Limited novelty relative to prior work. The use of Hadamard rotation for quantization was already explored in QuaRot and TurboAttention, which both apply rotations to keys and values to mitigate outliers. KVLinC’s main advance is the asymmetric configuration (rotating only values and quantizing keys channel-wise), which is an empirical refinement rather than a new conceptual mechanism
- Correction adapters require retraining. The proposed linear correction mechanism is elegant but depends on fine-tuning small adapter modules on a calibration dataset. This breaks the plug-and-play nature of tuning-free quantizers such as KIVI or QuaRot, and may limit practicality in deployment across models or domains. Evaluating how sensitive the adapters are to the choice of calibration data or whether they can generalize across architectures would strengthen the claim.

### Questions
- The implementation modify attention calculation, is the proposed approach compatible with modern attention implementation such as LeanAttention (https://arxiv.org/abs/2405.10480).
- is there a way of avoiding the training/fine-tuning, and what would be the accurcay impact? What is the sentisivity on selection of calibration dataset?
- Interaction with attention kernel optimization. The paper presents a custom Triton kernel for decoding. Have the authors looked at integrating it into existing frameworks?

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
4

### Summary
The paper introduces KVLinC, a framework for quantizing KV cache to 2-bit precision while mitigating attention errors. It combines Hadamard rotation to reduce quantization errors in values with lightweight linear correction adapters that compensate for errors in quantized keys, achieving higher compression (using a calibration dataset) and performance than baselines like KIVI, QuaRot, ResQ, and Gear. Evaluations on LLaMA-3, Qwen-2.5, and Qwen-3 families show improved perplexity and accuracy on short and long-context benchmarks, plus a custom Triton kernel enabling up to 2.55× faster inference.

### Strengths
- The paper's originality lies in the hybrid approach: analyzing Hadamard rotation axes (observations 1-3, lines 189-211) to find an optimal config without extra compute overhead, and introducing trainable adapters (eq. 4-6, lines 216-257) that add <1% parameters but scale efficiently (O(dD) decode cost). Quality is high, with rigorous ablations (Fig. 3) and error reduction evidence (Fig. 5). Clarity is strong in explaining KV stats (Fig. 2, lines 108-161) and motivations (e.g., error accumulation in long sequences, lines 44-46). Significance is evident for efficient LLM deployment, as KV quantization enables larger batches/long contexts
- End-to-end systems story. A Triton kernel that streams bit-packed KV blocks and applies correction yields concrete throughput/batch-size gains on real models.
- Code is well-written and solid (reproducible).

### Weaknesses
- Missing SpinQuant baseline. Given the use of a calibration set (Alpaca for base and RedPajama for instruct), a fairer comparison should include SpinQuant (KV-only) at the same 2-bit setting, keeping weights/activations FP to isolate the KV effect. The paper compares to QuaRot (random Hadamard) but omits SpinQuant’s learned rotation.
- Minor: the evaluation scope is limited to smaller models (up to 8B).

### Questions
- In Sec. 3.1 (lines 189-194), Observation 1 states pre-multiplication with Hadamard yields worse performance due to token mixing; can you provide quantitative error metrics (e.g., MSE on attention logits) comparing pre- vs. post-multiplication on a specific model like Qwen-2.5-3B to support this?

- For linear correction adapters in eq. 5, how sensitive is performance to the feature map rank D=256 (line 320)? Could you report perplexity on Wikitext for D=128/512 to assess parameter overhead trade-offs?

- In Fig. 5, what is the corresponding downstream impact on a long-context benchmark like RULER (Sec. 4.1), e.g., accuracy drop without adapters?

### Soundness
3

### Presentation
3

### Contribution
3
