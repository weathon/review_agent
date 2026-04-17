# FlexibleLLM: Making Low-Bit Quantization for Large Language Models More Flexible and Efficient

- Decision: Reject
- Scores: 2, 6, 6, 6, 4

## Abstract
Low-bit quantization is crucial for deploying Large Language Models (LLMs) on resource-constrained hardware. However, existing Post-Training Quantization (PTQ) methods are limited by a monolithic view of outliers, failing to address their dual spatial distribution (both discrete and clustered) and overlooking "attribute outliers"—weights that are sensitive to quantization but not numerically large. Furthermore, these methods generally ignore the critical issue of quantization errors accumulating and amplifying across layers. To overcome these challenges, we introduce FlexibleLLM, a novel finetuning-free, weight-only PTQ framework founded on a new theoretical analysis of outliers. FlexibleLLM holistically addresses the outlier problem through three synergistic components: (1) To handle clustered outliers, the Self-Adaptive Block-Level Greedy Bit Search (SBGBS) module enables highly flexible, fractional-level bit-width allocation (e.g., 2.1 bits), optimizing the trade-off between hardware utilization and model accuracy. (2) For discrete outliers, the Discrete Outlier Suppression and Aware (DOSA) module employs a dual strategy: it innovatively uses Hadamard transforms for computationally efficient suppression of numerical outliers and a Hessian-aware mechanism to precisely handle overlooked "attribute outliers”. (3) To combat error propagation, the Layer-Level Feedback and Denoising (LFD) module introduces a dynamic correction mechanism that mitigates the accumulation of ``activation noise'' from a global, cross-layer perspective. Extensive experiments demonstrate that FlexibleLLM achieves state-of-the-art performance, significantly outperforming not only existing finetuning-free methods but also many finetuning-based approaches, all while requiring substantially fewer computational resources. Code is available at https://anonymous.4open.science/r/FlexibleLLM.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This work presents FlexibleLLM, a weight-only post-training quantization method. At its core, three components are introduced: (1) A mixed-precision within each layer, (2) an outlier reduction mechanism with Hadamard transform, (3) asymmetric calibration. Unfortunately, I haven't found any of these to appear novel, given that they are already proposed in other works.

### Strengths
The name of each submodule looks fancy.

### Weaknesses
My major concern for this work is its novelty. 

+ For SBGBS, the sailency metric has been proposed in SliM-LLM [1], in which the Hessian diagonal values and the weight norm are used. In this work, the authors used the weight norm, not the quantization error norm, which makes less sense given that it is not pruning. In the meantime, this metric is not ACCURATE since the weights are updated after every iteration. This problem has been discussed in the SparseGPT Algorithm, but I do not see any deep discussion in this paper. 

+ For NOR, this is the same as QuaRot. And more ironically, even the code implementation is the same as QuaRot without citing them in Section 3.2.1. Apparently, the authors know what they are doing. 

+ For LFD, the core idea and methodology are exactly the same as GPTAQ [2]. Another proof is that they used the code from GPTAQ by slightly modifying the function name. Yet they did not compare the GPTAQ in the experiments. 

I would reconsider my rating if the authors could acknowledge they used the implementation of QuaRot and GPTAQ, clarify the difference, if any, and provide the file `flexiblellm_utils.py` for a detailed evaluation. 
 
## Minor weakness

+ The notation system in this work is very chaotic. Hessian is represented with both $H$ and $\mathrm{H}$. 

+ The evaluation is primarily about perplexity, which is less trustworthy in the current LLM community. I would like to see MMLU and reasoning task accuracy. 

## Reference

[1] SliM-LLM: Salience-Driven Mixed-Precision Quantization for Large Language Models, ICML 2025. 

[2] GPTAQ: Efficient Finetuning-Free Quantization for Asymmetric Calibration, ICML 2025.

### Questions
None.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper tackles the challenge that existing post‑training quantization (PTQ) methods fail to adequately handle both “discrete outliers” and “attribute outliers,” and tend to ignore error accumulation across Transformer layers. To address this, the authors propose FlexibleLLM, a finetuning‑free, weight‑only PTQ framework, comprised of three key modules: SBGBS (self‑adaptive block‑level greedy bit search) for flexible fractional bit allocation, DOSA (discrete outlier suppression & awareness) combining Hadamard transforms and Hessian signals, and LFD (layer‑level feedback & denoising) to counter cross‑layer error propagation. Experimental results on LLaMA / LLaMA‑2 / LLaMA‑3 models under 2‑ and 3‑bit settings show that FlexibleLLM significantly outperforms both finetuning-free and many finetuning-based baselines in perplexity and zero-shot accuracy, while using substantially lower computation cost. The results validate that more flexible bit allocation, better outlier handling, and cross-layer error correction lead to more robust and efficient low-bit quantization.

### Strengths
- The paper identifies key limitations in existing PTQ methods, including discrete outliers, attribute outliers, and cross-layer error accumulation, making the motivation very clear.

- Introducing SBGBS allows adaptive, fractional bit allocation at the block level, improving precision without full finetuning. DOSA and LFD further enhance robustness by handling discrete outliers and mitigating cross-layer errors.

- Evaluations span LLaMA, LLaMA‑2, and LLaMA‑3 models, under extremely low-bit settings (2-bit and 3-bit), demonstrating consistent improvements in perplexity and zero-shot tasks.

- FlexibleLLM is weight-only and finetuning-free, achieving better accuracy than many finetuning-based methods while keeping computational cost low, making it practical for large-scale LLM deployment.

### Weaknesses
- Combining SBGBS, DOSA, and LFD involves multiple stages and careful bookkeeping, increasing implementation difficulty.

- Block-level bit search and Hadamard-based outlier suppression may require tuning for different models or layers.

- Although finetuning-free, some steps like block-level greedy bit search and Hadamard transforms introduce extra computation compared to simpler PTQ methods.

- While the paper emphasizes the advantage of being finetuning-free, in practice the cost of light finetuning is often acceptable even for large models. 

- The baselines used for comparison are relatively outdated and do not include more recent and stronger methods such as OSTQuant[1] or SpinQuant[2], which limits the significance of the claimed improvements.

- Anonymous code links seem to be inaccessible.

[1] Hu, Xing, et al. "Ostquant: Refining large language model quantization with orthogonal and scaling transformations for better distribution fitting." arXiv preprint arXiv:2501.13987 (2025).

[2] Liu, Zechun, et al. "Spinquant: Llm quantization with learned rotations." arXiv preprint arXiv:2405.16406 (2024).

### Questions
Please refer to the weaknesses above.

### Soundness
3

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
The paper proposes FlexibleLLM, a finetuning-free, weight-only Post-Training Quantization (PTQ) framework designed to make low-bit quantization of Large Language Models (LLMs) more accurate, flexible, and efficient, especially for deployment on resource-limited devices. It rethinks the nature of outliers in quantization, builds a modular finetuning-free framework with adaptive and fractional bit-width control, and sets a new benchmark for low-bit, high-accuracy LLM quantization.

### Strengths
1. Contribution of the Paper：The paper makes an original and technically solid contribution to the low - bit quantization of LLMs. It introduces FlexibleLLM, a finetuning - free PTQ framework that holistically addresses key limitations of existing methods through three modules.

2. Modular design and High integration：FlexibleLLM use Self-Adaptive Block-Level Bit Search (SBGBS) module, Discrete Outlier Suppression and Awareness (DOSA) module, and Layer-Level Feedback Denoising (LFD) module to factilitate integtation into LLMs.

3. Theory and Concept:The work is notable for its conceptual originality. It offers a new theoretical perspective on outliers by distinguishing between discrete, clustered, and “attribute” outliers.It also models the cumulative effects of quantization noise across layers.

4. Methodology: The methodology is rigorous and empirically well - supported across multiple model families. It delivers strong performance improvements over SOTA baselines with minimal computational overhead.

### Weaknesses
1. Lack of Direct Validation:The experimental results do not directly validate these specific theoretical claims.

2. Impressive but Unclear Performance Improvements: The reported performance improvements are impressive but could be due to general optimization and calibration effects rather than the specific theoretical mechanisms proposed.

3. Missing Controlled Experiments:There are no controlled experiments that:

-  Isolate the influence of “attribute outliers” on quantization error.

- Visualize or quantify the dual distribution of outliers.

- Explicitly measure cross-layer noise accumulation before and after the LFD module.

### Questions
1. The paper claims that outliers exhibit both discrete and clustered distributions. Could the authors present quantitative or visual evidence (e.g., density plots or clustering metrics) supporting this claim?

2. Can the authors show how quantization noise propagates before and after applying LFD? For instance, is there a measurable reduction in activation variance or output deviation at deeper layers?

### Soundness
4

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
2

### Summary
This paper introduces FlexibleLLM, a finetuning-free, weight-only PTQ framework for large language models. It combines three key components: SBGBS for adaptive bitwidth allocation， DOSA for outlier suppression using a Hadamard transform and Hessian-aware calibration, and LFD for cross-layer error correction. The method targets both clustered and discrete outliers and reports consistent improvements.

### Strengths
- The paper is well-motivated: it focuses on real problems—mixed outlier types and error accumulation—that most PTQ works overlook.
- The integration of SBGBS, DOSA, and LFD forms a complete and coherent pipeline, not just a single heuristic.
- Experimental coverage across LLaMA and Qwen models shows strong potential.

### Weaknesses
- The inverse-Hessian correction might be heavy for large models. How is $H^{-1}$ approximated?
- Many recent PTQ works (e.g., SpinQuant, QuaRot) also perform rotation or error compensation. The novelty boundary between DOSA and these methods should be better articulated.

### Questions
- Weakness 1&2
- What’s the absolute per-token latency and throughput with and without DOSA and LFD on A100/H100?
-  Can you show empirical per-layer error amplification curves after applying LFD?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes FlexibleLLM, a finetuning-free, weight-only PTQ framework for LLMs. It tackles (i) clustered vs. discrete outliers and (ii) cross-layer error accumulation via three components:

- SBGBS: a self-adaptive, block-level greedy bit search that allocates fractional bit widths (e.g., 2.1 bits) based on importance.

- DOSA: a two-part module—(NOR) numerical outlier reduction via Hadamard transforms, and (HGC) a Hessian-guided mechanism to address “attribute outliers.”

- LFD: layer-level feedback/denoising to mitigate activation-noise propagation with an error-aware objective.

Experiments show improvements over several PTQ baselines on some metrics and models; the method is presented clearly with ablations.

### Strengths
1. The framework is easy to follow; writing is organized and readable.

2. Reasonable benchmarks and ablations; measurable gains on some tasks.

3. Attention to both numerical and “attribute” outliers: Explicit handling is practically relevant.

### Weaknesses
1. Limited novelty / incremental combinations: Each module largely extends known ideas with modest variations:

- SBGBS: Importance-guided mixed-precision allocation is well-studied; the greedy search and block granularity feel incremental.

- NOR (in DOSA): Hadamard transforms for outlier suppression/energy spreading are mature and previously explored.

- HGC (in DOSA): Using Hessian information (or approximations) to guide quantization or error reconstruction is established.

- LFD: Using output-error/activation-noise–aware objectives echoes prior PTQ works (e.g., QDrop-style criteria and related error-aware calibration).

Overall, the contribution reads more as a package of small updates than a distinct conceptual advance.

2. Many moving parts: Four submodules and several hyperparameters risk over-engineering, making it harder to attribute gains to any single idea and increasing deployment complexity.

3. Ablation depth: While ablations exist, they don’t isolate what is truly new versus what prior techniques would already deliver under matched calibration/tuning.

4. Theory claims: The “new theoretical analysis of outliers” is not clearly distinguished from existing analyses; formal novelty remains unclear.

### Questions
Do gains persist across different seeds, calibration set choices/sizes, and task distributions (generation vs. perplexity vs. alignment tasks)? Any evidence of overfitting to the calibration set despite being “finetuning-free”?

### Soundness
3

### Presentation
3

### Contribution
3
