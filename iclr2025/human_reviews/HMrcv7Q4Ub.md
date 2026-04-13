## Human Reviewer 1

### Summary
The paper tackles the challenge of accelerating vision-language models (VLMs) by leveraging a sparse Key-Value (KV) cache. Unlike existing compression methods for large language models (LLMs), which are less effective for VLMs, the proposed VL-Cache takes advantage of unique sparsity patterns in visual and text tokens. It uses layer-adaptive budget allocation to allocate cache resources based on sparsity levels and modality-aware token scoring to determine token importance with post-vision text tokens. By retaining only the most crucial cache entries, VL-Cache effectively reduces memory usage and latency. Experimental results on three lmms-eval tasks demonstrate the method’s effectiveness.

### Strengths
The paper is well-written and easy to follow. The sparsity analysis of prefilling and decoding effectively motivates the use of adaptive sparsity. The proposed method achieves significant speedup in decoding, particularly with long prefill tokens.

### Weaknesses
Please see the questions.

### Questions
1.	Some notations in the proposed method lack clarity. For instance, the meanings of $\Gamma$ and $Z$ are not sufficiently explained, making it difficult to understand their roles in the algorithm. Providing more detailed explanations would improve the clarity of the presentation.

2.	The authors state that the maximum batch size is constrained by peak memory usage during prefill rather than by the KV cache size, suggesting that compressing the KV cache does not increase the batch size. Does this imply that the proposed KV cache compression does not actually reduce overall memory usage? If so, why is that the case? This raises concerns about whether the title “KV Cache Compression” accurately reflects the method’s ability to save GPU memory, as the proposed approach appears to be more of a sparse computation technique applied during decoding.

3.	In Table 3, the reported speedup is based on generating a total of 100 output tokens. It would be helpful if the authors could provide the exact prefill and decoding times separately. Given that with a prompt length of 128k and only 100 output tokens, the prefill time is likely to dominate the overall processing time, having this detailed breakdown would clarify the impact of the proposed method on different stages of the inference process.

4.	In Section 4.2, the authors point out that accumulated attention has the limitation of assigning disproportionately high scores to earlier tokens due to summation over the full query dimension. To address this, they propose accumulated post-vision attention. However, there is another approach [1] that addresses the same issue using normalized attention. It would be helpful if the authors could include more discussion on this alternative method and provide additional experimental comparisons to better illustrate the benefits of their approach.

5.	The experimental evaluation is insufficient, as it lacks comparisons with two important baselines for VLM acceleration: FastV [1] and HiRED [2]. Additional comparisons with these methods are needed to strengthen the evaluation.

6.	Although the proposed method targets vision-language models, the current evaluation lacks coverage of a comprehensive range of tasks. To better demonstrate its effectiveness, it would be beneficial for the authors to include additional results on video benchmarks, such as Video-MME [4]. This would help showcase the method’s applicability to video tasks.

Reference:

[1] ZipCache: Accurate and Efficient KV Cache Quantization with Salient Token Identification. NeurIPS 2024.

[2] An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models. ECCV 2024.

[3] HiRED: Attention-Guided Token Dropping for Efficient Inference of High-Resolution Vision-Language Models in Resource-Constrained Environments. arXiv 2024.

[4] Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis. arXiv 2024.

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
4

---

## Human Reviewer 2

### Summary
The authors aim to accelerate VLMs by reusing KV caches for encoding visual contexts. The authors observe the limitations of existing KV cache compression methods and propose a specific version for VLMs. Based on the observation of sparsity patterns in prefilling and decoding stages, the main idea is to design a cache allocation method that balances the cache size and model accuracy. Evaluations based on vision-language benchmarks show the performance of model accuracy and speedups.

### Strengths
1.	The authors identify the I/O bottleneck between GPU’s HBM and SRAM in the decoding stage of VLMs, motivating the innovative design of KV cache compression. This understanding helps to address the challenge of scaling VLMs and optimizing their performance.
2.	Through Figure 1, the authors illustrate the distinctive attention sparsity patterns between VLMs and LLMs, emphasizing the need for modality-aware compression techniques tailored for VLMs.
3.	I have learned a lot from the author's observations. The KV cache budget and compression ratios differ in layers and modalities. This provides good insights into the optimization of VLM cache compression.
4.	To compress the VLM cache, the authors propose two key methods: a sparsity-aware cache budget allocation strategy and a modality-aware token scoring policy, to optimize the cache budget allocation and token score measurement, respectively.
5.	The experimental results show that the proposed methods achieve significant inference speedups and GPU memory savings.

### Weaknesses
1.	In Figure 1, the authors measure attention scores to verify the modality boundary, but the specific methodology for calculating these scores remains unclear. Please provide additional details on the measurement process.
2.	The term “post-vision attention” is frequently used but not explicitly defined, making the relevant sentences difficult to follow (“the attention patterns from the output language tokens is much closer aligned with the language tokens that follow the visual tokens in the prompt (the post-vision attention) rather than the visual tokens themselves”). A clear description of post-vision attention at its first occurrence would enhance the paper's readability.
3.	The threshold filter in Section 3.1 utilizes the hyper-parameter p to control sparsification. Conducting ablation studies on the impact of different p values would provide deeper insights into the robustness and sensitivity of the proposed method.
4.	To Algorithm 1, while it dynamically allocates the cache budget based on each prompt, its computational complexity raises my concern. Please analyze the computational overhead (in lines 11 to 14).
5.	Besides, Algorithm 1 does not explicitly reflect the modality-aware design as has been highlighted in this paper. Please clarify how the algorithm incorporates modality properties.
6.	In experiments, the I/O traffic amount caused by KV caching should be reported.

### Questions
Please refer to the questions in the weakness part.

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper introduces a modality-aware KV cache compression method for Vision-Language Models (VLM). It introduces a dynamic cache budget allocation mechanism and employs post-vision language tokens to compute attention scores.

### Strengths
The insight that the attention patterns of generated tokens closely align with language tokens is valuable.

### Weaknesses
1.	Algorithm 1 only outlines two procedures without detailing how these procedures are integrated into the algorithm.
2.	Definition 3.1 and the following text is unclear. The terms Psi(S) and S_Psi are used without clear explanation. A precise definition of Cache Hit Rate is essential.
3.	Figure 4 lacks clarity. It does not effectively illustrate how the budget is allocated or how post-vision attention is utilized.
4.	In the related work, ZipCache proposes a policy using normalized attention scores. This should be considered and compared.
5.	The KV cache for generated tokens is not addressed.
6.	The performance of VL-Cache is not convincingly demonstrated in Table 2. VL-Cache does not significantly outperform other methods, while H2O has achieved the strongest performance on many benchmarks. It is recommended to test VL-Cache on more challenging benchmarks.

### Questions
See weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
5

---

## Human Reviewer 4

### Summary
This paper addresses the challenge of efficiently storing and accessing large Key-Value (KV) caches in Vision-Language Models (VLMs), which are crucial for handling visual contexts like images or videos. Existing KV cache compression methods developed for Large Language Models (LLMs) don't perform optimally when applied to VLMs. To solve this, the authors propose VL-Cache, a compression method designed specifically for VLM inference. VL-Cache utilizes a layer-adaptive, sparsity-aware cache allocation and a modality-aware token scoring approach to reduce cache size without losing accuracy. Experimental results show that this method retains comparable accuracy with only 10% of the original cache, accelerates token generation by up to 2.33x, decoding by up to 7.08x, and reduces GPU memory usage by 90%.

### Strengths
1. The experimental results are very good; the performance is significantly better compared to other KV-cache methods.
2. Sparsity-aware layer-wise KV cache allocation and modality-aware token scoring are meaningful and have also been proven effective in the experiments.

### Weaknesses
1. The writing is somewhat complex and unclear; it takes a few careful readings to understand. Adding more illustrations to explain the method would improve it.
2. For MathVista, it seems that the KV-cache isn’t important? Even with only 1% of it, the performance is quite good, and sometimes the performance with 100% KV-cache isn’t even the best. Could you explain that?
3. In terms of writing, Section 3's PRELIMINARY EXPERIMENT is too long, while there’s relatively little written about the methodology. This writing style increases the burden on the reader. Readers likely want to see the method as soon as possible, so the focus should be on discussing your approach.

### Questions
Please see the section on weakness.

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
6

### Confidence
4