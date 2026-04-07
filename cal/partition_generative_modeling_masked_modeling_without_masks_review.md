=== CALIBRATION EXAMPLE 77 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract

The title "Partition Generative Modeling: Masked Modeling Without Masks" is accurate and memorable. The abstract's core claims are largely supported, but with important caveats that the abstract elides:

- The "5–5.5× higher throughput" claim for text is stated without noting that this is achieved with **more parameters** (203M for PGM 8/8 vs. 170M for MDLM; 268M for PGM 6/6 dim 1024) and not at strict parameter parity. Similarly, the ImageNet 7.5× throughput comparison uses a model trained for only 500k steps vs. MaskGIT's 2M steps. Neither caveat appears in the abstract.
- The claim "producing samples with lower Generative Perplexity" is true but only for the base model, not after distillation (where nucleus sampling is needed to match MDLM).

---

### Introduction & Motivation

The motivation is well-articulated and the problem is real. The observation that MGMs waste compute on [MASK] tokens at every step is correct and important. The positioning relative to prior art (Block Diffusion, distillation, confidence-sampling improvements) is mostly accurate.

One tension: the paper frames the inefficiency of MGMs as a crucial disadvantage for test-time compute scaling, yet the comparison against GPT-2 Small with KV-caching in Figure 1 (right) shows PGM achieving ~5,500 tok/sec vs. GPT-2's ~10,000–20,000+ tok/sec. It's not discussed whether PGMs are actually competitive with well-optimized autoregressive baselines that motivated the problem.

---

### Method / Approach (Sections 3 & 4)

**The core mechanism is elegant.** Replacing [MASK] with a partition is a natural and clean generalization of MGMs that eliminates the need for a special vocabulary token at inference.

**Theoretical connection to MDLM (Section 3.1):** The paper argues that PGM evaluates the MDLM training objective at two complementary masking rates in a single forward pass, yielding Equation (7). However, this connection is presented as an analogy rather than a rigorous derivation. Specifically:

- MDLM optimizes a *variational lower bound on log-likelihood*. The PGM loss (Eq. 7) is not shown to be a valid ELBO of any well-defined generative model. The generative model underlying PGM — what is the joint density *p*(x) that PGM is implicitly learning? — is never defined.
- Treating group 0 as "clean" and group 1 as "masked" is consistent with a left-to-right analogy, but at inference the model doesn't start from fully masked sequences; it starts from no tokens and reveals them one group at a time. A more precise treatment of the implied generative process would strengthen the paper.

**GroupSwap (Section 4.1):** The mechanism is clearly described. The choice of data-independent over data-dependent queries is justified empirically (Table 5), but the *why* is unexplored. Data-dependent queries aggregate group information via logsumexp/mean and add it back to positions in the opposite group — it is surprising that this adds no benefit over position-only queries, and some analysis of what the GroupSwap layer actually learns would be valuable.

**Attention mask structure:** The encoder uses group-wise block-diagonal attention, which the paper acknowledges is currently implemented with dense masks and `torch.sdpa`, resulting in a ~25% training throughput penalty. The paper defers efficient kernel implementation to future work, but this means training cost is currently non-trivial for a method that claims efficiency as its primary advantage.

---

### Experiments & Results (Section 5)

**Language Modeling (Section 5.1):**

- **LM1B:** PGM 6/6 achieves 26.80 vs. MDLM's 27.67 — a 1.95 PPL reduction, but the comparison with 25.72 (MDLM + complementary masking) shows that most of the improvement comes from complementary masking, not the architecture. The PGM still trails MDLM + complementary masking by ~1 PPL point. This is acknowledged but its implications for the architectural contribution are understated.

- **OWT:** At the same parameter count (170M, 12 layers), PGM *underperforms* MDLM (PGM 6/6 would be ~171M but is not shown at 1M steps; only 200k steps in Table 5, where PGM 6/6 gets 26.96 vs MDLM's 25.35). Matching MDLM requires 203M or 268M parameters. The throughput comparison at different parameter counts is not isoflop.

- **Downstream tasks (Table 2):** Differences are uniformly small and mixed. PGM 8/8 beats MDLM on 6/8 tasks, PGM 6/6 (1024) on 5/8. The overall picture is "similar," which undercuts the claim of quality advantage on language modeling.

- **Distillation (Section 5.1, Table 7):** This is a significant concern. After 5 rounds of SDTT, at 128 steps: MDLM achieves Gen. PPL 45.06 (FP32), while PGM 6/6 achieves 60.06 (FP32) or 43.84 with nucleus sampling (p=0.9). So the post-distillation comparison requires nucleus sampling to match MDLM, but this adds latency overhead (the throughput advantage decreases from 5× to ~4.6×). The distillation strategy is explicitly sub-optimal (one group is treated as masked), which is a fair acknowledgment, but this is presented as an implementation choice rather than a hard limitation. No experiment shows a better distillation strategy.

**Image Modeling (Section 5.2):**

- The 7.5× throughput improvement with comparable FID (5.54 vs 5.35) is impressive and is the paper's strongest result. However, MaskGIT was trained for 2M steps while PGM only for 500k due to "computational constraints." Given that FID typically improves with more training steps, this comparison is not apples-to-apples. The authors should at minimum discuss whether the FID gap is likely due to reduced training or the architecture.

- The confidence sampler results are relegated to Appendix D.3, where PGM 12/12 actually achieves *better* FID than MaskGIT (5.54 vs 5.92 at 32 steps with confidence). Reporting both samplers in the main text would give a more complete picture.

**Section 5.3 — Isolating Complementary Masking:**

This ablation is well-conceived. The result that complementary masking improves LM1B (27.67 → 25.72) much more than OWT (23.07 → 22.98) is notable and unexplained. Appendix D.1 attributes it to context length (128 vs 1024) rather than dataset, but only ~200k-step preliminary results are shown. This is an important finding that warrants more investigation: if the architectural benefit of PGM mainly manifests at short context lengths, the scalability of the approach is in question.

**Missing baselines:**

- No comparison with Block Diffusion on shared benchmarks. While their capabilities differ (BD sacrifices any-order generation), including a throughput vs. quality scatter plot across methods would help situate PGMs.
- No comparison with MaskGIT trained for 500k steps (the same training budget as PGM). This would isolate whether the FID difference is architectural or due to training duration.
- For the throughput measurements, the hardware specs (A100-SXM4-80GB) are given, but batch size choices are somewhat arbitrary (BS=32 for text, BS=32 for images). Throughput profiles for different batch sizes would help readers understand the practical setting.

---

### Writing & Clarity

The paper is generally well-written and well-organized. The Partition Transformer's three-component structure (encoder, GroupSwap, decoder) is clearly explained. One source of confusion: Algorithm 2 (simplified PGM sampling) starts the sequence with a BOS token and uses a random row permutation for positions, but the training procedure partitions at each timestep *t* with probability *αt*. The connection between training-time group assignment and inference-time token reveal schedule is not made explicit — how does the training-time stochasticity in group assignments correspond to the inference-time fixed sampling schedule?

---

### Limitations & Broader Impact

The limitations section is honest about the parameter overhead and training cost. However, some additional limitations deserve explicit mention:

1. **Scalability of GroupSwap:** The paper notes that "the gap remains between PGM and MDLM with complementary masking" and attributes it to a suboptimal GroupSwap. For longer sequences (1024 tokens), PGMs require either more layers or wider embeddings. How this scales to 4k, 8k, or longer contexts is only briefly touched in Table 10 (latency measurement only, no quality evaluation at 4096 tokens).

2. **The distillation quality gap is a first-order concern**, not a future work item. PGMs' primary claim is inference efficiency; if distilled PGMs require nucleus sampling to match MDLM, this substantially complicates the picture for practitioners who want to push sampling speed to the extreme.

3. **Generative Perplexity as a metric:** The paper correctly notes precision issues (Appendix C.4) and uses unigram entropy as a secondary metric. However, GPT-2 Large as an evaluator is itself a relatively weak autoregressive model, and Gen. PPL on 1,024 samples is noisy. The downstream task evaluation is a helpful complement, but the lack of any human evaluation or BPB (bits-per-byte) computation on a held-out set limits the quality comparison.

---

### Overall Assessment

This paper makes a genuine and well-executed contribution: partitioning replaces masking in a way that is conceptually clean, architecturally concrete, and delivers substantial throughput improvements (5× for text, 7.5× for images) without a large quality penalty. The compatibility with existing MGM samplers and distillation methods is a practical advantage. However, several issues limit the paper's impact as currently presented. First, the throughput gains versus MDLM are demonstrated at non-equal parameter counts, which partially muddies the efficiency story. Second, the post-distillation quality gap is significant — nucleus sampling is required to recover competitive Gen. PPL, which partially erodes the speed advantage. Third, the ImageNet comparison is confounded by a 4× training step difference. Fourth, the complementary masking benefit is context-length-dependent in a way that is not well understood, raising questions about how PGM scales to longer sequences. Fifth, the underlying generative model is not rigorously specified: the paper does not show PGM optimizes a valid ELBO or define what joint density it is learning. The contribution stands and is interesting for the MGM community, but the paper is closer to a strong workshop paper than a clear ICLR acceptance without revisions addressing at minimum the parameter-count-controlled throughput comparison, the distillation gap, and the ImageNet training-step confound.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces Partition Generative Models (PGMs), a framework that replaces the [MASK] token paradigm in Masked Generative Models with a partitioning strategy where sequences are split into two non-attending groups. By utilizing a specialized Partition Transformer architecture with a "GroupSwap" mechanism, PGMs claim to eliminate the processing of masked tokens during inference, achieving significant throughput improvements (5–7.5× faster) compared to standard MGM baselines like MDLM and MaskGIT while maintaining comparable sample quality.

### Strengths
1.  **Substantial Inference Speedup:** The paper provides convincing empirical evidence of significant throughput improvements. On OpenWebText, PGM achieves ~5.4× higher sampling throughput than MDLM with lower Generative Perplexity (Table 1), and on ImageNet, it matches MaskGIT's FID with a 7.5× speedup using the Halton sampler (Table 9).
2.  **Practical Compatibility:** The authors demonstrate that PGMs are compatible with existing MGM optimization techniques, such as Self-Distillation Through Time (SDTT) and alternative samplers like Halton sequences, without requiring major architectural modifications to these downstream methods (Section 5.1).
3.  **Comprehensive Ablation Studies:** Section 5.3 successfully isolates the effect of "complementary masking" (using double supervision) from the architectural changes, revealing that while supervision helps, the partition architecture is required to match the speed benefits. This strengthens the claims regarding the efficacy of the proposed design.

### Weaknesses
1.  **Training Efficiency and Architecture Overhead:** Although inference is faster, training PGMs is currently slower than MGMs due to suboptimal kernel implementations of the partition attention. Table 3 and Appendix E note that PGM training throughput is lower (e.g., 68 seq/sec vs 80 seq/sec on OWT for MDLM+Compl. masking), and the Appendix explicitly states training is "slower... for simplicity." This offsets the inference gains for the pre-training phase.
2.  **Inductive Bias Limitation:** The architectural constraint where Group 0 never attends to Group 0 (and vice versa) during the encoder pass imposes a strong inductive bias. This may limit the model's ability to capture local dependencies within the same partition compared to standard MGMs where any token position can theoretically depend on its neighbors via the masking process. Section 5.1 admits PGMs require larger parameters/embedding dimensions on OpenWebText to surpass MDLM initially, suggesting the bias may hamper parameter efficiency.
3.  **Comparison Scope:** While MDLM and MaskGIT are strong baselines, the paper could benefit from a more detailed comparison with other recent MGM acceleration methods like Block Diffusion (Arriola et al., 2025) or FlexMDM, which also address similar inference bottlenecks but with different trade-offs (Section 6 mentions these but lacks empirical comparison).

### Novelty & Significance
*   **Novelty:** The core idea of replacing [MASK] tokens with a partitioning strategy to enforce information flow is technically novel. The "GroupSwap" layer provides a specific mechanism to enable cross-group communication, distinguishing it from simple block-based masking.
*   **Clarity:** The paper is well-structured with clear motivation and logical flow. The Partition Transformer architecture is described with sufficient detail in Section 4 and Figure 3.
*   **Reproducibility:** The authors commit to releasing code (MIT license) and provide extensive hyperparameter details in the Appendix (Sections C, E), facilitating reproduction.
*   **Significance:** The work addresses a critical bottleneck in MGMs (inference efficiency) without sacrificing the parallel/asynchronous generation capability. If the training overhead can be mitigated (as suggested in Appendix E), this could be a significant step toward making MGMs viable for real-time, production-level applications.

### Suggestions for Improvement
1.  **Optimize Training Kernels:** Explicitly discuss or provide benchmarks for a block-diagonal attention implementation (as mentioned in Appendix E) to show the potential training throughput improvements, as the current overhead is a significant barrier.
2.  **Analyze Dependency Coverage:** Provide an analysis or example showing how the bipartite constraint affects modeling short-range dependencies, which are typically captured well by bidirectional masks in MGMs. Does the model struggle with specific linguistic or visual patterns?
3.  **Expand Baseline Comparisons:** Include a comparison with Block Diffusion or other KV-caching compatible MGM variants to better contextualize the trade-off between "any-order" flexibility and throughput.
4.  **Clarify Sampling Flexibility:** The paper claims "any-order" generation. Please clarify if the sampling algorithm must adhere to the specific group constraints (e.g., alternating groups) to maintain consistency with the training distribution, or if the "GroupSwap" truly allows arbitrary unmasking orders without distribution shift.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Iso-Param Pareto Frontier:** Provide a comprehensive sweep of model sizes where PGM and MDLM have identical parameter counts; current results show PGM underperforms MDLM at iso-params on LM1B (26.80 vs 25.72 PPL), undermining the efficiency claim.
2. **Total Compute Break-Even Analysis:** Quantify the training overhead versus inference savings to determine the usage threshold where PGM becomes computationally cheaper than MDLM, as training is currently slower (Appendix E).
3. **Stronger Image Baselines:** Compare against recent masked image generators (e.g., MagViT, TiTok) rather than solely MaskGIT (2022), as ICLR reviewers expect SOTA comparisons for image generation claims.
4. **Long-Context Training:** Train and evaluate on context lengths >1024 (e.g., 4k, 8k) to substantiate the claim that PGMs are suited for test-time compute scaling, rather than relying on inference-only proxies.
5. **GroupSwap Mechanism Ablation:** Test alternative information exchange mechanisms (e.g., simple concatenation, learned routing) to prove the specific GroupSwap design is necessary and not just one viable option.

### Deeper Analysis Needed (top 3-5 only)
1. **Gradient Variance Verification:** Provide empirical evidence (e.g., gradient norm histograms) to support the claim that complementary masking reduces variance, as current evidence relies solely on final perplexity metrics.
2. **Within-Group Dependency Limits:** Analyze whether preventing cross-group attention in the encoder creates an information bottleneck that harms modeling long-range dependencies within a single partition.
3. **Distillation Workaround Justification:** Explain why native PGM distillation was not developed and analyze the performance cost of reverting to MGM-style masking during the distillation phase.
4. **Training Stability Risk Assessment:** Investigate the loss spikes observed in Figure 6 to determine if they pose a divergence risk when scaling to larger models or batch sizes.

### Visualizations & Case Studies
1. **Attention Mask Verification:** Visualize the attention matrices in the encoder to explicitly confirm zero information flow between groups, ensuring the architectural constraint is strictly enforced.
2. **Qualitative Failure Cases:** Display side-by-side generated samples where PGM fails to maintain coherence compared to MDLM, exposing potential artifacts caused by the partitioning strategy.
3. **Throughput vs. Sequence Length Scaling:** Plot inference throughput against sequence length (up to 4096+) to demonstrate that the speedup advantage does not degrade at longer contexts.

### Obvious Next Steps
1. **Implement Sparse Kernels:** Develop and benchmark custom CUDA kernels that exploit the block-diagonal attention structure to eliminate the admitted training overhead before claiming efficiency.
2. **Develop Native Distillation:** Create a distillation objective that leverages the two-group structure rather than mimicking MGM masking, ensuring the architectural benefits persist in distilled models.
3. **Establish Scaling Laws:** Compare PGM, MDLM, and ARM scaling trends to determine if the partitioning approach offers fundamental advantages in compute-optimal training regimes.

# Final Consolidated Review
## Summary

Partition Generative Models (PGMs) replace the  token paradigm in Masked Generative Models with a partitioning strategy: sequences are split into two non-attending groups that predict each other during training. The Partition Transformer architecture enforces this constraint via group-wise self-attention and a GroupSwap cross-attention layer. At inference, PGMs process only clean tokens (like ARMs) while retaining parallel, any-order generation (like MGMs), achieving 5–7.5× throughput improvements over MDLM and MaskGIT on text and image benchmarks with comparable sample quality.

## Strengths

- **Elegant solution to MGM inference inefficiency**: The key insight—partitioning eliminates  tokens entirely, enabling inference on clean tokens only—is conceptually clean and well-motivated. The architecture (encoder with group-wise attention, GroupSwap layer, decoder with cross-attention) is carefully designed to enforce the bipartite information flow constraint.

- **Strong empirical efficiency gains**: On OpenWebText, PGM achieves 5.3–5.5× higher sampling throughput than MDLM with competitive or better Generative Perplexity (Table 1, 6). On ImageNet, PGM matches MaskGIT's FID with 7.5× speedup using the Halton sampler, and achieves better FID (4.56) with 2× more steps while remaining 3.9× faster (Table 9). These are substantial practical improvements.

- **Compatibility with existing MGM infrastructure**: PGMs work with existing samplers (Halton, confidence-based) and distillation methods (SDTT) without modification, demonstrated empirically in Sections 5.1 and 5.2. This makes adoption straightforward.

- **Iso-param evidence on LM1B**: At near-identical parameter counts (171M PGM vs 170M MDLM), PGM achieves lower validation perplexity (26.80 vs 27.67) with ~1.8× higher throughput (Table 1), providing some evidence that the architectural contribution is not purely parameter-driven.

- **Ablation isolates complementary masking effect**: Section 5.3 and Table 5 cleanly separate the training signal benefit (complementary masking) from the architectural benefit, showing that both contribute but PGM still underperforms MDLM+complementary masking, suggesting room for architectural improvement.

## Weaknesses

- **Non-iso-param comparisons on OpenWebText**: The main throughput comparisons require larger PGM models (203M or 268M parameters) to match MDLM's perplexity (170M). The paper reports PGM 8/8 (203M) achieves 22.61 PPL vs MDLM's 23.07, and PGM 6/6 (dim 1024, 268M) achieves 21.43. While the LM1B results show near-iso-param improvements, the OWT results—where the efficiency claim is most meaningful for practical language modeling—are not presented at strict parameter parity.

- **ImageNet training duration confound**: PGM is trained for 500k steps while MaskGIT is trained for 2M steps (Appendix C.3), with FID of 5.54 vs 5.35. The paper acknowledges this limitation, but it complicates interpretation of whether the quality gap is architectural or simply due to reduced training.

- **Post-distillation quality gap requires nucleus sampling**: After SDTT distillation, PGM achieves higher Generative Perplexity than MDLM (Table 7: 60.06 vs 45.06 at 128 steps FP32). Matching MDLM requires nucleus sampling (p=0.9), which adds latency overhead and partially erodes the speed advantage (from 5× to ~4.6×). The distillation objective treats one group as masked, which the authors acknowledge is suboptimal, but no native PGM distillation is provided.

- **Training overhead not fully resolved**: Appendix E.1 shows PGM training throughput is lower than MDLM (e.g., 68 seq/sec vs 81 seq/sec on OWT) due to dense tensor masks rather than efficient sparse kernels. The paper acknowledges this but defers kernel implementation to future work, which is a limitation for a method whose primary claim is efficiency.

- **Complementary masking benefit diminishes at longer contexts**: On LM1B (context 128), complementary masking improves MDLM from 27.67 to 25.72 PPL (Table 1). On OpenWebText (context 1024), the improvement is negligible (23.07 → 22.98). Appendix D.1 hypothesizes this is due to context length, but the mechanism is not well-understood, raising questions about how PGM scales to longer sequences relevant for modern LLMs.

- **Theoretical grounding is incomplete**: Section 3.1 presents the connection to MDLM's ELBO as an analogy rather than a rigorous derivation. The paper does not prove that the PGM loss (Eq. 7) optimizes a valid ELBO of a well-defined joint density, leaving a theoretical gap in understanding what distribution PGM implicitly learns.

## Nice-to-Haves

- Comparison with Block Diffusion (Arriola et al., 2025) on throughput vs. quality trade-offs, even though Block Diffusion sacrifices any-order generation.
- Empirical gradient variance analysis to support the variance reduction claim beyond final perplexity metrics.
- Results at longer context lengths (4k+) beyond the inference-only latency measurements in Table 10.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **GPT-2 with KV-cache comparison**: The harsh critic notes that GPT-2 achieves 10k-20k tok/sec vs PGM's ~5,500. However, Figure 1 (right) already includes this comparison, and PGMs are positioned as an alternative to MGMs (not ARMs). The GPT-2 comparison provides useful context but does not invalidate the MGM efficiency claims.

- **Demand for SOTA image generation baselines**: Requesting comparison with MagViT or TiTok rather than MaskGIT (2022) is scope creep. MaskGIT remains a representative and widely-cited baseline for masked image generation, and the paper's contribution is the efficiency mechanism, not beating SOTA FID.

- **Demand for human evaluation**: Generative Perplexity with GPT-2 Large and unigram entropy are standard evaluation metrics for discrete diffusion language models. Adding human evaluation is not standard for this venue and would be excessive for a methods paper focused on efficiency.

- **"No comparison with MaskGIT trained for 500k steps"**: This criticism misunderstands the experimental design. The paper makes clear that MaskGIT was trained for 2M steps, providing a stronger baseline. Comparing against a weaker baseline would not strengthen the paper.

## Novel Insights

The bipartite attention constraint in PGM reveals an interesting trade-off: while it enables dramatic inference speedups by eliminating masked token processing, it may create an information bottleneck for within-group dependencies. The fact that complementary masking (which provides denser supervision) helps more on short sequences (LM1B, ctx=128) than long ones (OWT, ctx=1024) suggests the partition structure interacts with sequence length in ways not yet understood. The GroupSwap layer's role as a learned information exchange mechanism—and the surprising finding that data-independent queries perform comparably to data-dependent ones—warrants further investigation into what structural properties the partition constraint actually imposes on the learned representations.

## Suggestions

1. **Report iso-param results on OpenWebText**: Train PGM 6/6 (171M, same layers/dimensions as MDLM) for 1M steps on OWT and report throughput at matched perplexity to enable fair efficiency comparisons.

2. **Train PGM on ImageNet for 2M steps**: Even with computational constraints, this would clarify whether the small FID gap is architectural or training-related.

3. **Develop native PGM distillation**: An ablation comparing the current MGM-style distillation to a method that leverages PGM's two-group structure would clarify whether the post-distillation quality gap is fixable.

4. **Analyze the context-length effect**: A focused experiment varying context length while holding other factors constant would help explain why complementary masking helps at short contexts but not long ones.

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 6.0, 6.0]
Average score: 7.0
Binary outcome: Accept
