# Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding

- Avg Score: 7.00
- Decision: Accept (Poster)
- Scores: 6, 8, 8, 6

## Abstract
Diffusion-based large language models (Diffusion LLMs) have shown promise for non-autoregressive text generation. However, the practical inference speed of open-sourced Diffusion LLMs often lags behind autoregressive models due to the lack of Key-Value (KV) Cache and quality degradation when decoding multiple tokens simultaneously. To bridge this gap, we introduce Fast-dLLM, a method that incorporates a novel block-wise approximate KV Cache mechanism tailored for bidirectional diffusion models, enabling cache reuse with negligible performance drop. Additionally, we identify the root cause of generation quality degradation in parallel decoding as the disruption of token dependencies under the conditional independence assumption. To address this, Fast-dLLM also proposes a confidence-aware parallel decoding strategy that selectively decodes tokens exceeding a confidence threshold, mitigating dependency violations and maintaining generation quality. Experimental results on LLaDA and Dream models across multiple LLM benchmarks demonstrate up to 27.6× throughput improvement with minimal accuracy loss, closing the performance gap with autoregressive models and paving the way for practical deployment of Diffusion LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates the slow inference speed of diffusion-based large language models, stemming from the lack of Key-Value (KV) cache support and quality degradation in parallel decoding. The proposed Fast-dLLM method introduces a block-wise approximate KV cache tailored for bidirectional attention and a confidence-aware parallel decoding strategy. This strategy dynamically selects tokens above a confidence threshold, enabling efficient cache reuse and safe, training-free parallel token generation.

### Strengths
1. The method is training-free, making it an easily applicable plug-in for compatible diffusion language models.
2. It includes a comprehensive set of ablation studies analyzing the impact of key components and hyperparameters, such as cache block size, confidence threshold, cache variants, and generation/prefill length. 
3. The empirical results are significant, showing marked improvements in throughput across multiple benchmarks with minimal accuracy loss.

### Weaknesses
1. The mechanism of 'DualCache' is not clearly explained; it is unclear how caching suffix tokens saves computation and accelerates inference. Additionally, more details on the extra memory overhead introduced by the KV cache need to be disclosed.  In addition, the paper claims that the cache update introduces “negligible overhead” in Figure 2, but it provides no concrete explanation or timing comparison. 

2. The novelty of the block-wise KV cache contribution is unclear. The manuscript fails to sufficiently differentiate its caching method from existing work like Block-diffusion, and the experimental section lacks relevant diffusion model baselines. 

3. The reported speedup ratios raise concerns about potential metric inflation. Furthermore, the datasets used are primarily focused on math and code problems, lacking experimental data on benchmarks specifically designed for evaluating inference acceleration.

4. The conceptual relationship and practical distinction between threshold- and factor-based strategies are insufficiently clarified. It is unclear whether the factor-based rule is a relaxed, adaptive, or independent variant, or under which conditions one should be preferred. A clearer rationale linking these strategies would improve comprehension and applicability.

5. While the paper qualitatively illustrates KV similarity, it does not provide quantitative measures showing how approximation errors change with block size, decoding length, or modality, making it difficult to assess cache mismatch risks. It would be helpful to report KV similarity decay curves across decoding steps for different block sizes and tasks, correlate these similarities with downstream task metrics (e.g., accuracy, EM, BLEU) to establish a “similarity threshold → refresh” rule.

6. The assumption of a “well-defined joint PMF with self-consistent marginals” may not hold for real diffusion LLMs trained approximately. The implications of this idealization are not discussed in detail, limiting interpretability of theoretical guarantees.

7. The paper does not include a comparison with autoregressive (AR) models. As a result, it remains unclear whether the diffusion + Fast-dLLM approach is competitive with state-of-the-art AR systems in realistic serving scenarios.

### Questions
1. The paper does not clarify how to automatically tune the commonly used 0.9 confidence threshold across different model scales, temperatures, or tasks, nor whether a simple calibration method (e.g., based on confidence–accuracy curves) can be applied. It also leaves open how to select the factor-based hyperparameter \(f\), and whether a universal default exists or task-specific tuning is required.

2. The paper does not quantify the computational cost (FLOPs or wall-clock time) of recomputing all blocks after completing one, nor does it explore whether incremental updates—such as refreshing only neighboring blocks or subsets of Keys/Values—could achieve comparable accuracy more efficiently. Profiling or throughput measurements would help clarify these trade-offs.

3. In the MathVista experiments, Fast-dLLM exhibited a noticeable performance degradation. Could you give a detailed analysis?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper proposes a novel training-free acceleration framework Fast-dLLM, which aims to address two key bottlenecks in Diffusion LLMs: the lack of KV caching and the quality degradation associated with parallel decoding. First, the paper introduces a block-wise approximate KV Cache mechanism tailored for bidirectional diffusion models. This approach is justified by demonstrating the high similarity of KV activations across adjacent inference steps, which enables cache reuse with a negligible performance drop. Second, the paper posits that the quality degradation in parallel decoding stems from the conditional independence assumption, which disrupts critical token dependencies. To mitigate this, it proposes a confidence-aware parallel decoding strategy that selectively decodes only those tokens exceeding a confidence threshold. This approach is shown to reduce dependency violations while maintaining generation quality. The authors validate their method through comprehensive experiments on the LLaDA and Dream models across multiple LLM benchmarks, and results demonstrate that Fast-dLLM achieves significant throughput improvements (up to 27.6x) while incurring negligible degradation in generation quality.

### Strengths
1.The paper introduces a novel, training-free framework that tackles two foundational challenges in dLLM inference.

2.The proposed methods are well-justified and supported by solid theory and experiment phenomenon. The approximate KV cache is empirically validated by the high similarity of KV activations in adjacent steps. The parallel decoding strategy is theoretically supported by Theorem 1, which proves the equivalence of greedy parallel and sequential decoding under high-confidence conditions, motivating the threshold-based approach. Furthermore, the identification of these two phenomena is a valuable contribution, offering clear insights and new avenues for solving these challenges in dLLM inference.

3.The paper provides extensive and sufficient experiments. Main results validate the method's effectiveness. A wide range of ablation studies on variables (e.g., cache block size, confidence thresholds, generation lengths) assess scalability and robustness, with clear analysis and interpretation of the results.

4.This paper is writen well and clear. The structure is logical and easy to follow, making the core concepts and contributions readily understandable.

### Weaknesses
The experimental evaluation is primarily limited to comparisons against the baseline dLLM pipelines. The paper lacks an comparison to other existing acceleration techniques for Diffusion LLMs.

### Questions
1.The paper's appendix C.4 demonstrates that the 'Factor' decoding strategy significantly outperforms the 'Threshold' strategy in terms of throughput, with only a minor accuracy trade-off. However, the main experimental results (e.g., Table 1) are reported using the slower 'Threshold' strategy. Could the authors clarify why the seemingly superior 'Factor' strategy was not used for the main results? Does the 'Factor' strategy have other un-discussed drawbacks that led to this decision?

2.Could the authors comment on the robustness of the approximate KV cache strategy for very long sequence generation? Is there a risk that approximation errors will accumulate over many blocks, leading to a loss of attentional fidelity to the initial prompt?

3.The qualitative comparisons in appendix B are based on a simple arithmetic prompt, could the authors provide more diverse qualitative examples (e.g., generating idioms or logical pairs) where the risk of dependency failure is more prominent?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces Fast-dLLM, a training-free acceleration framework for diffusion-based LLMs that addresses their slow inference speed compared to AR models. The method incorporates two key innovations: (1) a block-wise approximate KV Cache mechanism tailored for bidirectional attention that enables cache reuse with negligible performance drop, and (2) a confidence-aware parallel decoding strategy that selectively generates multiple tokens simultaneously based on confidence thresholds to preserve generation quality. Experimental results on LLaDA and Dream models across multiple benchmarks demonstrate remarkable throughput improvement with little accuracy loss, effectively closing the performance gap with autoregressive models.

### Strengths
1. Novel adaptation of KV Cache to bidirectional diffusion models via block-wise approximation, with insightful analysis showing high cosine similarity between adjacent steps.
2. Theoretical foundation with Theorem 1 proving the equivalence between greedy parallel and sequential decoding under certain conditions.
3. Comprehensive ablation studies covering key hyperparameters (block sizes, thresholds, generation lengths) and evaluation across models and benchmarks.

### Weaknesses
The evaluation of models relies on only four benchmarks (GSM8K, MATH, HumanEval, MBPP for LLaDA), which primarily focus on math reasoning and code generation, missing important capability dimensions like commonsense reasoning (HellaSwag), factual knowledge retrieval (TriviaQA), and real-world code generation (LiveCodeBench, BigCodeBench) that would provide a more comprehensive understanding of the method's generalization and potential failure modes across diverse task types.

### Questions
1. Why does the throughput plateau when the block size continues to increase in Fig4 ?
2. Is the KV activation similarity pattern largely consistent, or can it be different among layers?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Fast-dLLM, a training-free inference acceleration framework for diffusion-based large language models (dLLMs). Unlike autoregressive LLMs, diffusion LLMs cannot directly leverage KV caching or efficient parallel decoding due to their bidirectional denoising process. The authors introduce two key techniques to address this:

* Block-wise Approximate KV Cache: exploits temporal similarity between diffusion steps to reuse attention states across steps, enabling cache reuse in bidirectional attention without retraining.

* Confidence-Aware Parallel Decoding: decodes multiple tokens simultaneously when their predicted confidence exceeds a threshold, with theoretical justification showing approximation to sequential decoding under high-confidence conditions.

Experiments on several diffusion LLMs (e.g., LLaDA, LLaDA-V, Dream) across text, code, and multimodal reasoning tasks demonstrate up to 27.6x speedup with negligible accuracy loss.

### Strengths
1. Timely and practically relevant: addresses the major efficiency bottleneck of diffusion-based LLMs, a direction gaining interest.

2. Training-free approach: requires no retraining or model modification, making it directly applicable to existing dLLMs.

3. Comprehensive evaluation: covers both text and multimodal reasoning tasks, with consistent gains across benchmarks.

4. Strong empirical results: large acceleration factors (up to 27.6x) with small degradation make the method attractive for deployment.

### Weaknesses
1. Applicability to distilled or few-step diffusion LLMs. It remains unclear whether the proposed caching and confidence-aware decoding strategies would remain effective for distilled diffusion LLMs that operate with only a few or even a single denoising step (e.g., dParallel, arXiv:2509.26488; One-Step Diffusion LLM, OpenReview:P7OzWxOUHK). The reviewer acknowledge that these are concurrent works, while such aggressive timestep reduction is becoming a key trend, similar to continuous diffusion distillation in image/video models. Caching-based acceleration mainly benefits multi-step teacher models, but may offer limited or no gain for student variants without hurting the accuracy, restricting practical adoption.

2. Memory overhead. The block-wise KV caching mechanism likely introduces memory costs, especially for long sequences or large models. The paper does not quantify the memory–speed trade-off or report actual VRAM usage, which is important for understanding deployment feasibility.

3. Baseline comparisons. Recent concurrent and closely related works, such as dLLM-Cache (arXiv:2506.06295) and dPad (arXiv:2508.14148), are not compared or discussed. Even if concurrent, a conceptual comparison highlighting methodological similarities and distinctions would help position Fast-dLLM within this rapidly evolving research landscape.

### Questions
How sensitive are the results to the cache window size and confidence threshold? Could a learning-based mechanism adaptively set these parameters?

What is the additional memory footprint introduced by storing bidirectional caches compared to AR caching?

Will the proposed caching mechanism be compatible with TensorRT deployment?

Could the proposed methods extend to multimodal diffusion transformers (e.g., text-to-image or video diffusion models)?

### Soundness
4

### Presentation
4

### Contribution
3
