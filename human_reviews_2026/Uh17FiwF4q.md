# Accelerating Diffusion Large Language Models with SlowFast Sampling: The Three Golden Principles

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 8, 4, 4, 6

## Abstract
Diffusion-based language models (dLLMs) have emerged as a promising alternative to traditional autoregressive LLMs by enabling parallel token generation and significantly reducing inference latency. However, existing sampling strategies for dLLMs, such as confidence-based or semi-autoregressive decoding, often suffer from static behavior, leading to suboptimal efficiency and limited flexibility. In this paper, we propose SlowFast Sampling, a novel dynamic sampling strategy that adaptively alternates between exploratory and accelerated decoding stages. Our method is guided by three golden principles: certainty principle, convergence principle, and positional principle, which govern when and where tokens can be confidently and efficiently decoded. We further integrate our strategy with dLLM-Cache to reduce redundant computation. Extensive experiments across benchmarks and models show that SlowFast Sampling achieves up to 15.63× speedup on LLaDA with minimal accuracy drop, and up to 34.22× when combined with caching. Notably, our approach outperforms strong autoregressive baselines like LLaMA3 8B in throughput, demonstrating that well-designed sampling can unlock the full potential of dLLMs for fast and high-quality generation.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper presents SlowFast Sampling, a dynamic decoding framework for diffusion-based large language models (dLLMs). Guided by three principles—Certainty, Convergence, and Positional—the method alternates between exploratory and accelerated decoding phases, achieving efficient token generation. Integrated with dLLM-Cache, the method achieves up to 34× speedup with negligible accuracy loss. Experiments on LLaDA and Dream across various benchmarks demonstrate consistent and substantial improvements in throughput and efficiency.

### Strengths
+ Proposes a novel and practical dynamic sampling strategy with a clear underlying rationale.
+ Demonstrates significant acceleration (up to 34×) with minimal loss in accuracy.
+ Offers conceptual clarity via the three principles, making the approach interpretable.
+ Shows real-world relevance, outperforming strong autoregressive baselines in throughput.

### Weaknesses
+ Although the results are strong, the paper lacks a detailed analysis of computational efficiency at different diffusion step counts. Providing tps statistics would make the acceleration claims more transparent and easier to validate.
+ The paper does not discuss any failure cases or corner scenarios. Including such examples would increase trust and reproducibility.
+ While the integration with dLLM-Cache is effective, its contribution is only reported numerically. A clearer explanation or figure showing how cache reuse interacts with SlowFast stages would enhance interpretability.

### Questions
+ Could the authors provide a step-wise breakdown of inference time to show which parts of the pipeline (exploration vs. acceleration) contribute most to overall speedup?
+ Could the authors report tokens-per-second (tps) across different diffusion step counts to make the acceleration claims more transparent and easier to validate?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a training-free inference-time acceleration method named SLowFast sampling for masked diffusion models. The approach is inspired by three proposed principles: high-confidence tokens should be preserved, each token tends to converge over iterations, and the decoded tokens admit certain positional bias. The algorithm leverages these ideas by first identifying convergence regions and then refining the regions in parallel. Experiments on LLaDA and dream on several common benchmarks demonstrate that the proposed approach achieves notable speedups without substantial performance drop.

### Strengths
1. This paper tackles an important problem of training-free decoding acceleration for masked diffusion models. More importantly, it helps improve the accuracy-efficiency frontier of masked diffusion models.

2. The experiments cover both LLaDA and Dream on a suite of benchmarks, which is a comprehensive evaluation.

### Weaknesses
1. As the experiment results show, it seems like most of the speedup comes with the help of dLLM-cache. Compared with Fast-dLLM (parallel) in the two scenarios (with or without cache) the gain in speedup is not that substantial.

2. The paper proposes to leverage three principles. However, there is no ablation study that separately investigates the role/contribution of each principle as well as some detailed designs (e.g. top-k refinement) in terms of enhancing the speedup. As a heuristic approach, the proposed method requires more in-depth analyses to help understand how each of the components contribute to the improved efficiency.

3. Some of the presentation should be improved. The paper claims to achieve 15x and up to 34x acceleration in abstract and introduction. However, the paper should be cautious of such claims and I think a less misleading way would be adding the model and dataset that these numbers were obtained from (GPQA in this case). The remarkable speedup might come only in certain datasets where the accuracy is less affected (e.g. the base model already achieves a pretty low accuracy so the negative effect on the accuracy when using acceleration techniques would be marginal).

### Questions
1. Could the authors explain why the accuracy on certain benchmarks for the base model and the acceleration algorithms are lower than originally reported? For instance, in original LLaDA paper they report ~37 accuracy on MMLU-pro but the authors report ~23 here in this paper.

2. It would be necessary to include another baseline that naively reduces the number of sampling steps w/o any modification on the sampling algorithm.

3. It is a bit confusing that the speedup reported in table 4 is ~15x, while it is only ~8x in table 2. I understand that they achieve different scores (31 vs 33), but still feel that the discrepancy here is confusing. An ideal way would be to report the curve with two axes being speedup and score, and compare the curves of different approaches.

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
The paper proposes a decoding algorithm, SlowFast sampling, for dLLMs that alternates between a slower exploratory decoding phase and a fast parallel decoding phase. Evolving around three empirical “golden principles” observed, SlowFast sampling consists of: a slow decoding phase — (1) confidence-based cautious decoding, (2) end point of convergence prediction with confidence thresholding, (3) stability check using variance of candidate tokens as a termination condition; and a fast decoding phase — (1) bidirectional predicted value caching (for out-of-span positions), (2) in-span parallel decoding, (3) fallback top-k refinements.

### Strengths
1. FastSlow decoding consistently reports better speedups than Fast-dLLM at comparable accuracy, indicating the design improves efficiency rather than merely trading off performance.
2. Clear presentation: the method, the case study and confidence-map visualizations make the two-phase “slow/fast” dynamics easy to understand and help readers see why the method works.

### Weaknesses
1. Overly complex and hyperparameter-heavy. The method introduces many knobs to tune (e.g., stability window/variance, multiple confidence thresholds, top-k settings). The paper only ablates the stability check; ablations for the other choices are missing and necessary.

2. TPS reporting (e.g., Table 4) labels both AR and LLaDA as “1$\times$” which makes it unclear what the true reference baseline is. Please normalize to one reference (e.g., AR or vanilla LLaDA) and report consistent relative and absolute TPS. As shown, the incremental TPS gains versus AR appear minimal.

3. The paper reads like a bundle of techniques rather than a single, principled mechanism with good motivation. more analysis explaining how much each component drives the gains would strengthen the contribution.

### Questions
1. When comparing “+cache”, are the authors using dual cache or prefix cache? Please specify the exact cache policies.

2. Could you provide sensitivity analyses for (i) the end-point-of-convergence criterion the parallel-decoding confidence threshold, and (iii) the cautious-decoding confidence threshold? These seem central to both correctness and speed.

3. How transferable is SlowFast to other dLLMs like Dream? Since dLLM is a fast evolving field, do your principles still hold as remasking strategies deviate from LLaDA, or does the method risk becoming obsolete? A discussion on compatibility layers or auto-tuning would be valuable.

### Soundness
2

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
This paper addresses the significant inference latency of dLLMs. The authors introduce "SlowFast Sampling," a new dynamic sampling strategy designed to accelerate generation without substantial loss in quality. 

The method is motivated by three empirically-derived "Golden Principles": 
(1) The Certainty Principle, which states that high-confidence tokens are likely correct and stable; 

(2) The Convergence Principle, noting that token predictions stabilize over diffusion steps; and 

(3) The Positional Principle, observing that confident tokens often appear in contiguous blocks. 

The proposed sampler operates in two alternating stages: a slow "Exploratory Stage" to cautiously identify stable regions of the sequence, and a fast "Accelerated Decoding Stage" that rapidly decodes these stable regions in parallel. The authors conduct extensive experiments on two dLLMs (LLaDA 8B and Dream 7B, both are SOTA dLLMs) across a wide range of benchmarks, demonstrating significant speedups (up to 15.6x, and 34.2x when combined with a caching mechanism) with minimal performance degradation.

I am not opposed to accepting this paper. It tackles a well-defined and highly relevant problem: making parallel-decoding dLLMs practically fast. The paper's core contribution is a framework for thinking about dynamic sampling in this context.

### Strengths
- **Significant Performance Improvement:** The paper deliver on its primary promise of acceleration. Speedups of over 10x with minimal accuracy drops are impressive and represent a major practical contribution to the field of efficient dLLMs inference.

- **Well-Motivated Approach:** The grounding of the SlowFast sampler in the three observed principles is a major strength. It shows that the design is not an arbitrary collection of heuristics but a thoughtful response to the underlying behavior of the generative process.

- **Comprehensive Evaluation:** The experiments are thorough. Using two different SOTA dLLM architectures demonstrates the generalizability of the approach. The wide array of benchmarks, spanning reasoning, general knowledge, and code, provides a holistic view of the method's impact.

- **Strong Baseline Comparisons:** The paper includes comparisons not only to vanilla diffusion sampling but also to other accelerated strategies (Fast-dLLM, semi-autoregressive) and, most importantly, to a state-of-the-art autoregressive model (LLaMA3 8B). Showing a throughput advantage over LLaMA3 is a powerful statement.

- **Clarity of Presentation:** The paper is well-written and structured. The figures, particularly the conceptual overview in Figure 1 and the pipeline in Figure 3, are very effective at communicating the core ideas.

### Weaknesses
- **Methodological Complexity and Hyperparameters:** The SlowFast sampling method introduces a non-trivial number of new hyperparameters (`Kmax`, `Whist`, `σ²_stable`, `τ_min_conf`, `τ_high_conf`, etc.). While the authors provide an ablation study (Figure 5 & 6), it raises questions about the method's sensitivity and the effort required to tune it for new models or tasks. The description in Section 3.3 is dense and would benefit greatly from a formal algorithm block (pseudocode) to improve clarity and reproducibility.

- **Oversimplification of the "Positional Principle":** The paper observes that confident tokens often cluster (Positional Principle), but the proposed method only acts on a single contiguous block `[scycle, ecycle]` at a time. It's plausible that a model might become confident about multiple, non-contiguous regions simultaneously (e.g., the beginning and end of a sentence). The current method does not seem equipped to exploit this.

- **Limited Analysis of Method Dynamics:** The paper presents strong final results, but provides little insight into the internal dynamics of the SlowFast sampler. For instance, what is the typical length of an accelerated "fast" span? How often does the fallback top-k refinement (in the fast phase) get triggered? Such analysis would provide a deeper understanding of *why* the method works so well.

### Questions
1.  Could you please comment on the process for selecting the hyperparameters? Were they found via a systematic search, and how sensitive is the performance to their joint configuration? How well would you expect the chosen values to transfer to a completely different dLLM architecture or a different data domain?
2.  To improve clarity and reproducibility, would it be possible to provide a pseudocode algorithm for the main SlowFast Sampling loop in the appendix? This would greatly help in understanding the interplay between the slow and fast phases and the various conditions.
3.  Regarding the Positional Principle: did you consider or experiment with strategies that could identify and decode multiple non-contiguous stable blocks in parallel during the "fast" phase? Do you believe this could offer further speedups?
4.  Can you provide any statistics from your experiments on the dynamic behavior of the sampler? Specifically, what was the average number of cycles (slow-fast pairs) per generation, and what was the average length of the `[scycle, ecycle]` spans?

### Soundness
3

### Presentation
3

### Contribution
3
