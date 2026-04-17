# Locality-aware Parallel Decoding for Efficient Autoregressive Image Generation

- Decision: Accept (Oral)
- Scores: 8, 6, 8, 6

## Abstract
We present Locality-aware Parallel Decoding (LPD) to accelerate autoregressive image generation. Traditional autoregressive image generation relies on next-patch prediction, a memory-bound process that leads to high latency. Existing works have tried to parallelize next-patch prediction by shifting to multi-patch prediction to accelerate the process, but only achieved limited parallelization. To achieve high parallelization while maintaining generation quality, we introduce two key techniques: (1) Flexible Parallelized Autoregressive Modeling, a novel architecture that enables arbitrary generation ordering and degrees of parallelization. It uses learnable position query tokens to guide generation at target positions while ensuring mutual visibility among concurrently generated tokens for consistent parallel decoding. (2) Locality-aware Generation Ordering, a novel schedule that forms groups to minimize intra-group dependencies and maximize contextual support, enhancing generation quality. With these designs, we reduce the generation steps from 256 to 20 (256×256 res.) and 1024 to 48 (512×512 res.) without compromising quality on the ImageNet class-conditional generation, and achieving at least 3.4× lower latency than previous parallelized autoregressive models.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes Locality-aware Parallel Decoding (LPD), an efficient method for parallel autoregressive image generation. By introducing position query tokens, it enables flexible, location-independent parallel decoding, while a locality-aware generation order improves consistency and quality. Experiments show that LPD achieves comparable or better image quality with up to 10× fewer generation steps and 3–4× faster decoding speed.

### Strengths
Here are the main strengths of the paper:

Significant Speedup – LPD greatly reduces the number of decoding steps (up to 10–20× fewer) while maintaining or improving image quality.

High-Quality Generation – The locality-aware scheduling preserves spatial coherence, producing consistent and detailed images even under high parallelism.

Flexible Decoding Framework – The position query token design allows generation in arbitrary orders, enabling diverse tasks like inpainting and outpainting without retraining.

Strong Empirical Validation – Extensive experiments on ImageNet (256×256 and 512×512) demonstrate clear improvements in both FID and latency over prior autoregressive baselines.

### Weaknesses
The paper is technically strong and the proposed method is clearly effective for autoregressive image generation. However, I think the evaluation could be further strengthened by extending it beyond ImageNet. In particular, it would be interesting to test the method on text-conditioned generation at higher resolutions (e.g., 1024²) to see whether the parallel decoding and locality assumptions still hold under more complex, long-range dependencies.
Moreover, applying LPD to video generation or temporal sequence modeling could highlight its scalability in spatiotemporal domains. Even a small-scale video experiment (e.g., UCF-101) would make the paper more convincing in terms of generality and impact.

### Questions
see above

### Soundness
3

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
5

### Summary
The paper introduces Locality-aware Parallel Decoding (LPD) with two techniques to accelerate autoregressive image generation. First, Flexible Parallelized Autoregressive Modeling leverages position-aware query tokens to indicate the tokens to be generated, enabling arbitrary generation ordering and degrees of parallelization. Second, a Locality-aware Generation Ordering is proposed to minimize mutual dependencies during parallel generation. Experiments on ImageNet 256$\times$256 and 512$\times$512 demonstrate the effectiveness of the proposed LPD.

### Strengths
1. The proposed Flexible Parallelized Autoregressive Modeling overcomes the constraint of a fixed generation order by allowing images to be synthesized in an arbitrary sequence. This capability holds the potential for discovering more effective generation orders in the future.
2. When equipped with the proposed Locality-aware Generation Ordering strategy, LPD demonstrates improved FID scores and greater generation efficiency on the ImageNet dataset.
3. The paper is easy to read and the figures are informative.

### Weaknesses
1. The paper's core algorithm (Algorithm 1) is presented in the appendix. While space constraints are understandable, the most critical algorithm should ideally be included in the main text, or at the very least, its underlying principles should be explained there.
2. The computational cost of the model increases compared to traditional fixed-order autoregressive models due to the use of additional positional query tokens. However, an analysis of this overhead is absent from the paper.
3. In line 208, the authors state that in previous methods, "tokens generated within the same parallel step are produced independently of one another." However, the paper does not rigorously analyze the issue of conditional independence in LPD's parallel sampling. It appears that while LPD ensures visibility among all target positions predicted concurrently, it may not fully resolve the underlying conditional independence of the parallel-generated tokens. A theoretical justification for this aspect would be beneficial.
4. The authors train the LPD model for 450 and 500 epochs on ImageNet at 256×256 and 512×512 resolutions, respectively. In contrast, most baseline methods (e.g., LlamaGen, PAR, and NAR) are trained for only 300 epochs. This discrepancy in training budgets may lead to an unfair comparison.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a method to accelerate AR visual generation, which is typically bottlenecked by sequential token prediction and memory bandwidth. It introduces two core innovations: (1) Flexible Parallelized Autoregressive Modeling, which allows arbitrary generation order and parallel prediction using learnable position query tokens and (2) Locality-aware Generation Ordering, a scheduling strategy that groups spatially related tokens to minimize dependencies and maximize contextual support. These techniques reduce generation steps dramatically and achieve at least 3.4× lower latency than prior parallel AR models without compromising image quality on ImageNet benchmarks.

### Strengths
The paper conducts a genuinely deep analysis of the key factors that affect both performance and generation quality in parallel autoregressive decoding (e.g., group size, dependency structure, attention visibility), and then turns those observations into a coherent, end-to-end parallelization method rather than a single heuristic component.

Through careful comparisons with recent parallel AR implementations (e.g., encoder–decoder style SAR/ARPG and decoder-only RANDAR), the authors show that their design, especially the “mutual visibility among concurrently generated tokens + cache only generated tokens” part, is not just faster but architecturally better motivated, and they support this with latency as well as quality numbers.

The paper is very readable: the motivation is clearly laid out, figures are aligned with the text (the attention-mask figures in particular), and the training vs. inference formulations are described in a way that makes reproduction and reimplementation realistic even for non-authors.

### Weaknesses
Experiments are limited to image generation; since current AR models are increasingly used for multimodal I/O (image–text, video tokens, layout, even audio tokens), it would strengthen the claim of “general AR parallelization” to show at least one non-image setting (e.g., CLIP-conditioned image tokens, image+text joint decoding, or video latents).

The paper does not compare against the newest AR acceleration lines such as speculative decoding, speculative Jacobi-style decoding, or draft/verify variants; even a small-scale experiment would help position LPD as complementary vs. strictly better.

While there is throughput analysis at batch 64, the work does not fully characterize memory consumption and scaling beyond that point; since the method introduces extra query tokens and fused encode–decode steps, reporting peak GPU memory and how it scales with batch size and resolution would make the efficiency story more convincing.

### Questions
Could the proposed LPD framework be extended or adapted to multimodal generation (e.g., image–text, video, or audio tokens)? Given its reliance on spatial locality, how might token dependencies behave for non-visual modalities?

How would LPD compare in efficiency and quality against speculative decoding or speculative Jacobi decoding? Including at least one experiment could clarify whether LPD complements or surpasses these methods.

Since the fusion of encoding and decoding steps introduces additional query tokens, could you quantify peak GPU memory usage and performance scaling with batch size and resolution (beyond batch 64)?

The method partially decouples context and generation tokens. Could the authors clarify how this affects the probabilistic consistency of the AR factorization (Eq. 2 in the paper)? In addition, the paper highlights speedups, but how does varying the group size quantitatively affect image fidelity and diversity?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces Locality-aware Parallel Decoding (LPD), a method designed to speed up autoregressive image generation. LPD uses two key techniques—Flexible Parallelized Autoregressive Modeling for arbitrary parallelization and Locality-aware Generation Ordering for quality enhancement—and reduces generation steps significantly (e.g., 256 to 20 for 256×256 images) on ImageNet while cutting latency by at least 3.4× compared to prior parallel models.

### Strengths
* Sound Method Design：The learnable position query tokens decouple context modeling from decoding, enabling generation at arbitrary target positions and boosting flexibility. The exploration of two locality principles, particularly the second one, offers meaningful insights for the community.

* Strong Performance：The method achieves clear reductions in generation steps and latency with good quality.

### Weaknesses
1. Overclaimed Contributions in Writing

   - For "Flexible Parallelized Autoregressive Modeling", decoder-only works like PAR/ZipAR/NAR already treat previously decoded tokens as KV Cache and the queries are decoded in parallel ensuring the mutual visibility among tokens generated concurrently; the key difference lies only in LPD’s position query tokens (enabling arbitrary target positions), which should be clarified to avoid overstating contributions.
   - For "Locality-aware Generation Ordering", the first principle is well-studied (Sec. 3.2 descriptions can be simplified), while the second principle ("low proximity among concurrent tokens") has been explored by Wang et al. (2024b) and Besnier et al. (2025)—its underexploration in prior work needs more detailed explanation in Sec. 3.2.

2. Insufficient Ablation Studies
   - Sensitivity analysis of critical thresholds $\tau$ and $\rho$ is missing, despite their importance to query position performance.
   - A key ablation is absent: for Flexible Parallelized AR Modeling (excluding adaptive generation order), testing whether replacing query tokens with LPD’s "position query tokens" would clarify the source of performance gains.

3. Confusing Details
    - Inference process ambiguity: It is unclear if sampled tokens need a forward pass to store KV Cache, and if this can be done in the final decoding step.
   - Figure 3 confusion: The figure shows queries attending only to the latest decoded tokens, conflicting with the claim that queries causally attend to all previously generated tokens.

4. Minor Structural Issue：The dynamic generation order, a core technical contribution, should be included in the main text.

### Questions
See weaknesses

### Soundness
3

### Presentation
2

### Contribution
3
