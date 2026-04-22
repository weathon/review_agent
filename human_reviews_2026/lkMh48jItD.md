# SpecVLM: Fast Speculative Decoding in Vision-Language Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Speculative decoding is a powerful way to accelerate autoregressive large language models (LLMs), but directly porting it to vision-language models (VLMs) faces unique systems constraints: the prefill stage is dominated by visual tokens whose count scales with image resolution and video length, inflating both compute and memory, especially the key-value (KV) cache. We study speculative decoding for VLMs and introduce SpecVLM, a practical system that (1) establishes a strong EAGLE-2-style baseline, EagleVLM, delivering 1.5--2.3x end-to-end speedups over full autoregressive inference, and (2) further accelerates VLM inference with an elastic visual compressor that adaptively selects among pruning, pooling, convolution, and resampler primitives to balance FLOPs/parameters and accuracy per input. To avoid costly offline distillation corpora, we propose an online-logit distillation protocol that trains the draft model with on-the-fly teacher logits and penultimate features using a combined cross-entropy and Smooth L1 objective, eliminating storage and preprocessing while remaining compute-efficient. This protocol reveals a training-time scaling effect: longer online training monotonically increases the draft model's average accepted length, improving speculative efficiency. Empirically, SpecVLM achieves additional acceleration, culminating in 2.5--2.9x end-to-end speedups within 5 epochs across LLaVA and MMMU, consistently over resolutions and task difficulties, while preserving the target model's output distribution (lossless decoding).

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces SpecVLM, a framework for accelerating vision-language models (VLMs) through fast speculative decoding. Traditional speculative decoding speeds up language models but struggles with multimodal inputs due to large visual token overhead. SpecVLM tackles this using two innovations: an elastic visual compressor that adaptively combines pruning, pooling, convolution, and resampling to balance speed and accuracy, and an online-logit distillation method that trains draft models efficiently without large offline datasets. Built on an EAGLE-2 baseline (EagleVLM), SpecVLM achieves 2.5–2.9× end-to-end speedups on LLaVA and MMMU benchmarks while maintaining output fidelity and demonstrating strong training-time scaling effects

### Strengths
1. The paper is clearly written and well-structured.
2. SpecVLM achieves 2.5–2.9× end-to-end speedups, which is remarkable.

### Weaknesses
1. The performance of LLaVA-1.6 is relatively weak, so conducting experiments solely on it is less convincing. You could consider trying Qwen2.5-VL or Qwen3-VL instead.

2. From an overall perspective, the proposed method seems to be a forced combination of “Speculative Decoding” and a token compression strategy.

3. This approach requires additional training. I would like to see a discussion in the paper comparing it with widely recognized training-free token reduction methods in the field, such as FastV and PyramidDrop. Please discuss their limitations and explain why your visual compressor performs better.

### Questions
see Weaknesses

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
5

### Summary
Integrating speculative decoding into VLMs while improving draft model accuracy for visual context is a significant challenge. This paper introduces SpecVLM, which leverages an elastic visual compressor to reduce the number of visual tokens to ease the computation burden of draft model, and employs online-logit distillation to enhance training efficiency. Extensive experiments demonstrate substantial speedups and increased average accepted lengths over EagleVLM.

### Strengths
1. This paper explores the novel domain of accelerating vision-language models (VLMs) by integrating speculative decoding with visual token compression.
2. To further enhance training efficiency, the authors propose online-logit distillation, which streamlines the distillation process and improves alignment.
3. Extensive experiments demonstrate that the proposed SpecVLM consistently outperforms the strong EagleVLM baseline in both speed and average accepted length.
4. The paper is well-structured and clearly written, making it easy to follow and understand.

### Weaknesses
1. While the paper combines speculative decoding, visual token compression, and online training, these components have been widely explored in prior work. The specific challenges and innovations **unique to VLMs** are not clearly explored. For example, excessive number of prompt token is not unique to VLMs, but also a key pain point for long-context LLMs, what is the challenge specific to visual task should be clarified. 
2. The work lacks direct comparisons with other state-of-the-art visual compressors, such as [FastV](https://arxiv.org/pdf/2403.06764),  [SparseVLM](SparseVLM). Adding these comparisons would strengthen the empirical evaluation.
3. The paper does not provide a theoretical and deep explanation for key observations.
4. For section 4.2 results, it is not immediately clear from the results that the SpecVLM–EagleVLM performance gap widens with increasing model size. A visualized figure would help readers grasp this trend.
5. For VLM benchmarks, it would be beneficial to report the average number of visual tokens before and after compression to quantify the impact of the elastic compressor. Also, after changing the prompt to "Please give a detailed and reasonable explanation for the answer," the effect on average generated token length should be explicitly reported.
6. Except end-2-end speedup number, it would be better to report the decomposed latency, such as time to first token (prefill), inter-token latency (decoding), and other additional overhead. 
7. More detailed experimental setup descriptions would help readers reproduce the results.

### Questions
refer to weaknesses

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
3

### Summary
This paper introduces SpecVLM, a framework that accelerates vision-language models (VLMs) using speculative decoding. It first builds a strong baseline called EagleVLM, which integrates a lightweight draft model with the full target model for 1.5–2× speedups. To further reduce latency, SpecVLM adds an elastic visual compressor that adaptively selects among pruning, pooling, convolution, and resampler operators to compress visual tokens efficiently. It also proposes an online-logit distillation method that trains the draft model on-the-fly using teacher logits and features, eliminating the need for large offline datasets. Experiments on LLaVA and MMMU benchmarks show that SpecVLM achieves 2.5–2.9× faster inference without loss in output quality, and exhibits a training-time scaling effect where longer online training improves speculative efficiency.

### Strengths
This paper is well-written and easy to follow. The paper’s main strengths lie in its systematic extension of speculative decoding to vision-language models. It presents a strong and well-implemented baseline (EagleVLM) and builds upon it with two innovative components: an elastic visual compressor that dynamically balances accuracy and efficiency, and an online-logit distillation method that eliminates the need for costly offline data generation.

### Weaknesses
The novelty of this work is somewhat limited, as it primarily extends the Eagle-2 speculative decoding framework from text-only LLMs to vision-language models. While the adaptation to multimodal inputs and the introduction of the elastic visual compressor and online-logit distillation are practical, they represent incremental rather than fundamental innovations. Moreover, as shown in Figure 1, the major latency bottleneck lies in the LLM prefill stage, which has already been largely addressed by EagleVLM. The additional improvements provided by SpecVLM appear relatively modest compared to the gains that EagleVLM itself achieves over LLaVA-1.6-7B.

### Questions
1. What are the key differences in applying speculative decoding to VLMs compared to LLMs? It appears that the main modification is the introduction of visual token compression, while the core speculative decoding mechanism remains unchanged. Is there any genuine innovation beyond adapting existing techniques to multimodal inputs? 

2. In Section 3.4, the authors state that they combine heterogeneous compression scales to preserve both global context and local details, selecting Pruning/Pooling at 20×, Convolution at 3×, and a Resampler with 2 queries. Could the authors clarify how these configurations were determined? Were they derived from theoretical considerations or simply chosen through empirical hyperparameter search? 

3. In Section 3.3, the authors state that penultimate-layer features from the target model are fused with the draft model’s input embeddings to enrich representations and mitigate feature mismatch. However, the impact of this fusion is unclear, and additional ablation studies would help clarify its effectiveness.

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
3

### Summary
The paper adapts speculative decoding to vision–language models by pairing a deployment-aligned elastic visual compressor with online logit distillation. The compressor dynamically selects among pruning, pooling, convolutional downsampling, and a resampler to tailor visual tokens per instance, while a lightweight single-layer draft decoder is trained against a frozen target VLM and used with the same compression path at inference. SpecVLM achieves ~2.5–2.9× end-to-end speedups at batch=1 without accuracy loss, outperforming a strong EAGLE-style baseline. Limitations include hand-tuned compression ranges, no dynamic compression of draft KV, and limited tuning of draft depth/hyperparameters.

### Strengths
1. This paper proposed a novel coupling of speculative decoding and compression to jointly improve accepted-span length and forward cost.
2. The paper also provided a systematic coverage of compression primitives including dynamically selects/combines pruning, pooling, convolutional downsampling, and a resampler.
3. The experimental results shows promising acceleration speedup without

### Weaknesses
1. The method introduces a dynamic expert selection compressor. It remains unclear whether such complexity is necessary versus a simpler compressor scheme that might yield similar gains with lower engineering risk. Some compressors have an impact on KVCache. The analysis of different compressors on inference performance is unclear. 
2. In practice, draft and target models often run on separate GPUs [1], and at larger batches the proposed compression could further reduce inter-GPU communication and alleviate memory-bound operations. Thishe paper focus on batch=1, low latency, the setting of which might not be ideal for the proposed method.

[1] Xupeng Miao, Gabriele Oliaro, Zhihao Zhang, Xinhao Cheng, Zeyu Wang, Zhengxin Zhang, Rae Ying Yee Wong, Alan Zhu, Lijie Yang, Xiaoxiang Shi, Chunan Shi, Zhuoming Chen, Daiyaan Arfeen, Reyna Abhyankar, and Zhihao Jia. 2024. SpecInfer: Accelerating Large Language Model Serving with Tree-based Speculative Inference and Verification. In Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3 (ASPLOS '24), Vol. 3. Association for Computing Machinery, New York, NY, USA, 932–949. https://doi.org/10.1145/3620666.3651335

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
