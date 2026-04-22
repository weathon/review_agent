# Stopping Computation for Converged Tokens in Masked Diffusion-LM Decoding

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 8

## Abstract
Masked Diffusion Language Models generate sequences via iterative sampling that progressively unmasks tokens. However, they still recompute the attention and feed-forward blocks for every token position at every step---even when many unmasked tokens are essentially fixed, resulting in substantial waste in compute.
We propose **SureLock**: when the posterior at an unmasked position has stabilized across steps (our *sure* condition), we *lock* that position---thereafter skipping its query projection and feed-forward sublayers---while caching its attention keys and values so other positions can continue to attend to it.
This reduces the dominant per-iteration computational cost from $O(N^2d)$ to $O(MNd)$ where $N$ is the sequence length, $M$ is the number of unlocked token positions, and $d$ is the model dimension.
In practice, $M$ decreases as the iteration progresses, yielding substantial savings.
On LLaDA-8B, SureLock reduces algorithmic FLOPs by 30--50\% relative to the same sampler without locking, 
while maintaining comparable generation quality.
We also provide a theoretical analysis to justify the design rationale of SureLock:  monitoring only the local KL at the lock step suffices to bound the deviation in final token probabilities. Our project page is available at https://daioba.github.io/surelock.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
SureLock locks converged tokens during masked diffusion LM decoding, skips their query projection + FFN for the rest of the sampling steps, and just reuses cached K/V so other tokens can still attend to them. This is supposed to drop per-step cost from O(N2d) to O(MNd), where M is the number of still-active tokens. Authors report ~$30–50$% FLOP reduction on LLaDA-8B with comparable quality. Authors also demonstrate TPS gains relative to the baseline for large batches and longer generation.

Overall, I like the idea -- it is simple and intuitive,  but the contribution feels modest without wall-clock comparisons to prior methods that tackle the same problem -- inference latency. I rank this paper at 4 for now and open to raising my score if the authors add solid baselines (and ideally more shorter generation results).

### Strengths
- Simple idea (to its credit), training free
- Proposed method results in TPS  speed ups for large-batch/long-generation settings.
- The design-theoretic bound helps motivate KL as a convergence signal (even if not yet operationalized)

### Weaknesses
- Table 2, perplexity for WikiText-103: it seems that this method results in considerable perplexity drop even for 256 tokens length generation (~4-8%) and even more for shorter context.
- For low-batch, short-generation $(N <= 256)$ proposed method shows little to no wall-clock benefit.Note that $N=256$
is not very short (it’s a few sentences), so the absence of speedup here is concerning for interactive use.
- Despite method is positioned as orthogonal to existing diffusion-LM accelerators (e.g., KV-cache reuse/adaptive recompute, fewer denoising steps, slow/fast schedulers), I do believe that the experiments lack direct wall-clock comparisons to at least one prior work. This makes it hard to judge the contribution.

### Questions
- Do authors have quantitative results for the optional unlocking variant? (Appendix H)?
- Since a locked token cannot be revised, can you share a few cherry-picked bad cases where premature locking leads to noticeable quality degradation?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes SureLock, a decoding-time method for masked diffusion language models (MDLMs) that permanently locks token positions once their posterior is locally "stable", thereby skipping Q-projection and FFN for those positions while caching K/V. This changes the dominant per-step cost from $O(N^2 T)$ to $O(MNT)$, where $M$ is the number of still-active positions. The paper establishes a local step-wise KL threshold (with optional confidence gating) for locking, and a theoretical bound for the terminal log-probability error. Experiments on LLaDA-8B (Base/Instruct) show 30–50% algorithmic FLOPs reduction with small changes in generation quality.

### Strengths
The study is very relevant and practical for discrete diffusion language model deployment.

The algorithm has simple, orthogonal develop towards temporal/reuse accelerations and integrates with K/V reuse and selective refresh. These are novel development in my opinion.

The algorithm is clearly presented and the paper is well written.

### Weaknesses
The bound in Theorem 1 relies on many assumptions, such as geometric tail contraction and Lipschitzness. These conditions are not carefully verified.

The base model shows non-trivial PPL degradation for short outputs ($N_{\rm gen}$ small). It may be interesting to propose adaptive locking schedules.

Empirical results are on LLaDA-8B Base/Instruct and two benchmarks. Evaluating long-context discrete diffusion models or other tasks would make the results more convincing and probe the robustness of the method.

### Questions
Can you estimate the constants in Assumptions (A2)-(A4) so that Theorem 1 becomes more explicit and interpretable?

The proposed locking is monotone, is it possible or beneficial to consider later free to "locked" tokens? This might matter for short length generations.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes SURELOCK, an efficient sampling method for Masked Diffusion Language Models that reduces algorithmic FLOPs. The method is based on a simple yet effective idea: once tokens stabilize during the diffusion process, they can be "locked," allowing their query projection and feed-forward sublayers to be skipped. The locking criterion is determined using step-wise KL divergence. Experimental results demonstrate that SURELOCK reduces algorithmic FLOPs by up to 50% without compromising generation quality.

### Strengths
1. The proposed method is simple yet effective, with experimental results demonstrating a substantial reduction in FLOPs without compromising generation quality.

2. Theoretical and experimental analyses further confirm the soundness and robustness of the proposed method.

### Weaknesses
1. While the experiments demonstrate the proposed method's effectiveness, incorporating relevant baselines for comparison would further validate the soundness of this work.

2. The experiments are limited to basic tasks such as language modeling and instruction following. Results on more complex tasks would further validate the method's effectiveness.

### Questions
Masked Diffusion Language Models are already known for faster inference compared to autoregressive models. Is trading off generation quality for better inference efficiency a worthwhile choice for Masked Diffusion Language Models?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces SURELOCK, a novel method to accelerate the iterative decoding process of Masked Diffusion Language Models (MDLMs). The core problem addressed is the computational redundancy in standard MDLMs, where self-attention and feed-forward networks are recomputed for all token positions at every decoding step, even for tokens that have already converged. SURELOCK proposes to dynamically identify and "lock" these converged tokens. The locking criterion is based on the step-wise KL divergence of a token's posterior probability distribution falling below a certain threshold. Once a token position is locked, its Query and FFN computations are permanently skipped in subsequent steps, while its Key and Value vectors are cached to allow other active tokens to continue attending to it. This reduces the dominant per-iteration computational cost. The authors provide a rigorous theoretical analysis to justify their use of a local KL-divergence criterion, proving that it bounds the terminal log-probability error. Empirical evaluations on LLaDA-8B demonstrate that SURELOCK achieves a 30–50% reduction in algorithmic FLOPs while maintaining comparable generation quality on language modeling and instruction-following tasks.

### Strengths
1.  **Well-defined Problem and an Intuitive Solution:** The paper effectively identifies a clear source of inefficiency in MDLMs—the redundant computation for already stable tokens. The proposed solution, SURELOCK, is intuitive and directly targets this issue by progressively reducing the set of actively computed tokens.

2.  **Principled Algorithm Design with Theoretical Backing:** A key strength of this work is that it goes beyond a purely heuristic approach. The authors provide a theoretical justification (Theorem 1) for their locking criterion, linking the local, step-wise KL divergence to a bound on the final log-probability error. This analysis provides a solid rationale for the algorithm's design and increases confidence in its stability.

3.  **Solid Empirical Validation:** The experimental evaluation is well-conducted. The authors report both theoretical efficiency gains (algorithmic FLOPs) and practical runtime performance (TPS), which is a good practice. The method is tested on both a foundational language modeling task and a more practical instruction-following task. The use of strong external LLMs for quality evaluation (LLaMA-3, GPT-4o) adds credibility to the claims that generation quality is largely preserved.

### Weaknesses
While the paper is strong overall, there are a few areas where the analysis could be deepened or clarified.

1.  **Empirical Grounding of Theoretical Assumptions:** The theoretical analysis relies on several assumptions, particularly (A2) "Geometric tail contraction," which states that the KL divergence decays at a geometric rate. This is a fairly strong assumption. The paper would benefit from an empirical investigation into how well this assumption holds for the models and tasks tested. Without this, the practical applicability of the derived error bound remains somewhat abstract.

2.  **Gap between Algorithmic FLOPs and Wall-Clock Time:** The paper acknowledges that the significant reduction in algorithmic FLOPs does not always translate to a proportional reduction in wall-clock time, especially in non-compute-bound scenarios. While the analysis of this gap is appreciated, a more in-depth discussion on the specific implementation overheads (e.g., managing the cache, irregular memory access) and the potential for hardware-specific optimizations would make the work more impactful from a practical systems perspective.

### Questions
1. Your locking criterion is based on the stability of the posterior distribution. Does this "distributional stability" always coincide with "semantic stability"? For instance, could a token position's distribution remain stable (low KL divergence) while oscillating between two semantically similar but different tokens (e.g., "happy" and "joyful")? If so, would locking it prematurely harm the final semantic nuance of the generated text?

2. The experiments were conducted with a default temperature of 0. How does sampling temperature interact with the SURELOCK mechanism? Intuitively, a higher temperature would lead to flatter, less certain posterior distributions, potentially delaying the locking of tokens and reducing the efficiency gains. Have you investigated this trade-off, and is there an optimal interplay between temperature and the locking threshold ε?

3. Theorem 1 provides a bound on the final error. From a qualitative perspective, how do errors from a potentially premature lock manifest in the generated text? Do they tend to be localized to the area around the locked token, or can they cause cascading failures that affect the global coherence of the sequence? Understanding the typical failure modes would be very insightful.

4. The SURELOCK principle of locking converged elements seems generalizable to other modalities where diffusion models are used, such as image or audio generation. Have you considered the applicability of this method to continuous data domains? What would be the main challenges in adapting the KL-divergence-based criterion from discrete token posteriors to continuous data representations?

### Soundness
4

### Presentation
3

### Contribution
3
