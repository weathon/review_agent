# LoPRo: Enhancing Low-Rank Quantization via Permuted Block-Wise Rotation

- Avg Score: 5.33
- Decision: Reject
- Scores: 6, 6, 4

## Abstract
Post-training quantization (PTQ) enables effective model compression while preserving relatively high accuracy. Current weight-only PTQ methods primarily focus on the challenging sub-3-bit regime, where approaches often suffer significant accuracy degradation, typically requiring fine-tuning to achieve competitive performance. In this work, we revisit the fundamental characteristics of weight quantization and analyze the challenges in quantizing the residual matrix under low-rank approximation. We propose LoPRo, a novel fine-tuning-free PTQ algorithm that enhances residual matrix quantization by applying block-wise permutation and Walsh-Hadamard transformations to align columns of similar importance, while explicitly preserving the quantization accuracy of the most salient column blocks. Furthermore, we introduce a mixed-precision fast low-rank decomposition based on rank-1 sketch (R1-Sketch) to further minimize quantization costs. Experiments demonstrate that LoPRo outperforms existing fine-tuning-free PTQ methods at both 2-bit and 3-bit quantization, achieving accuracy comparable to fine-tuned baselines. Specifically, LoPRo achieves state-of-the-art quantization accuracy on LLaMA-2 and LLaMA-3 series models while delivering up to a 4x speedup. In the MoE model Mixtral-8x7B, LoPRo completes quantization within 2.5 hours, simultaneously reducing perplexity by 0.4 and improving accuracy by 8%. Moreover, compared to other low-rank quantization methods, LoPRo achieves superior accuracy with a significantly lower rank, while maintaining high inference efficiency and minimal additional latency.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper tackles the Post-training quantization (PTQ) problem and presents a LoPRo method for the challenging sub-3-bit regime. The key idea is to enhance residual matrix quantization by applying block-wise permutation and Walsh-Hadamard transformations. This combination achieves the goal of aligning columns of similar importance while simultaneously preserving the quantization accuracy of the most salient column blocks. Extensive experimental results are reported for LLaMA-2 and LLaMA-3 series models while delivering up to a 4x speedup.

### Strengths
Originality: I have found the contributions in Sec. 3.2 (partial rotation quantization) and 3.3 (R1SVD) sufficiently novel.
Clarity: The paper is well written and easy to understand. 
Quality: I think the author did a good job of both technical contribution and literary presentation.
Significance: PTQ for LLMs remains one of the hot topics in AI research these days. This work seems a valuable contribution to an already-mature field.

### Weaknesses
- The idea of rotation in PTQ already exists in the literature (e.g., SmoothRot).
- Sec. 3.4 is a bit sloppy and too concise for a detailed analysis.
- Grammatical errors need more careful proofreading. For example, lines 153-154.

### Questions
how does LoPro compare against “Achieving Binary Weight and Activation for LLMs” [1] and SmoothRot [2]?
[1] https://aclanthology.org/2025.findings-acl.459.pdf?utm_source=chatgpt.com#page=6.55
[2] https://arxiv.org/abs/2506.05413?utm_source=chatgpt.com

### Soundness
4

### Presentation
4

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
This work proposes low-rank + quantization compression method for Large Language models. The low‑rank component is obtained with a modified randomized SVD algorithm. The residual is quantized after applying a column permutation and a block‑wise orthogonal transformation. These operations make the subsequent quantization easier. LoPRo can be combined with a variety of quantization schemes, including both scalar and vector quantizers. The approach is evaluated on the Llama‑2/3 families and on Mixtral‑8×7B.

### Strengths
* The proposed method yields pretty strong performance, outperforming state-of-the-art competitive approach at 2-bit quantization by a significant margin. 

* Adding the low‑rank adapter increases latency by only ~10 % compared with a baseline that does not use the low‑rank component.

* The paper includes detailed ablation studies on (i) the choice of rotation matrix and (ii) the LoRA rank (see the appendix).

### Weaknesses
**Quantization cost**

* It is claimed that the method is almost as fast as GPTQ for the scalar quantization case and faster than GPTVQ for the vector quantization case. However, GPTQ baseline seems to be unoptimized. GPTQModel repository  in fact quantizes model  that the provided numbers. Specifically, in my experience quantization of 7-8B Llama model takes ~8 minutes on single L40S. LoPRo_v is claimed to be faster than GPTVQ, but the LoRPo_v is in fact and enhancement of GPTVQ and should take at least a long. Nevertheless, the runtime is worth the resulting quality. 

**Baselines**

* An important baseline is missing. QTIP [1] achieves state-of-the-art performance for 2-bit quantization even without tuning (1MAD, 3INST codes). I think it worth adding it for the fairness of comparison.

**(minor editoral)**

* Tags with colors are an interesting way to categorize methods but it may be quite for the reader to memorize. Same holds for O1, F1, F2. Probably would be better to refer in some other, more explicit way. 

* **it** in the Stage A, Stage B is not defined, One could guess that it is Moore-Penrose pseudoinverse, but I would recommend to define it explicitly. 

* Figure 1 is quite nice, but requires strong zooming to discern what is going on it. I would recommend enlarging it in the camera-ready revision.

---
References

[1] Tseng, Albert, et al. "Qtip: Quantization with trellises and incoherence processing." Advances in Neural Information Processing Systems 37 (2024): 59597-59620.

### Questions
* Given that the method appears to be scalable to large models how difficult would it be to apply it on a larger and more powerful MoE model - DeepSeek V3 or Qwen-3-235B? 

* The provided list of benchmarks, despite standard in the compression literature, may be not exhaustive to evaluate capabilities of the model. I would suggest trying OpenLLM v1 / OpenLLM v2 leaderboard following the setting in [1] or some reasoning tasks for Qwen3 models or MoE.

---
References

[1] Kurtic, Eldar, et al. "" Give Me BF16 or Give Me Death"? Accuracy-Performance Trade-Offs in LLM Quantization." arXiv preprint arXiv:2411.02355 (2024).

### Soundness
3

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
The paper proposes LoPRo, which applies blockwise Walsh-Hadamard transformers to a low rank weight matrix’s approximation’s residual. LoPRo permutes the columns before applying blockwise rotation to maintain the numerical stability of the salient columns. The authors also propose fast low-rank decomposition based on rank-1 sketch to minimize the SVD costs. LoPRo exhibits improvements over several state of the art algorithms on both perplexity and downstream evaluations.

### Strengths
* LoPRo significantly improves over state of the art algorithms on perplexity and downstream benchmarks.
* Quantization runtimes don’t shoot up.
* Does not require finetuning unlike other low-bit quantization schemes.

### Weaknesses
* If W is dense in formation, the low rank approximation may not scale, offloading the majority of the error correction to quantization.
* Experiments are conducted on Llama-2, Llama-3 and Mistal. The results can be made stronger with results on frontier open source models such as Qwen2.5/Qwen3 or DeepSeek.

### Questions
* How much does the performance improve if a rank greater than 1 is used?
* Why can’t the model be quantized along the column axis? In that case the blockwise rotation should not be required, a full rotation matrix should work.
* How do the inference latency numbers look in comparison to the approaches listed in Table 1.

### Soundness
3

### Presentation
2

### Contribution
2
