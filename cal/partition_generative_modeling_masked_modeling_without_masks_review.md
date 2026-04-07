=== CALIBRATION EXAMPLE 70 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title ("Partition Generative Modeling: Masked Modeling Without Masks") clearly reflects the core contribution. The abstract succinctly states the problem (MGMs process [MASK] tokens), the solution (partitioning into non-interacting groups), and the key results (5-5.5x text throughput, 7.5x image throughput with comparable quality). The claims are specific and appear supported by the data presented. However, the abstract does not mention the increased parameter count sometimes needed to match baseline quality (as seen later for OWT), which is a relevant trade-off.

### Introduction & Motivation
The introduction effectively frames the inference efficiency problem of MGMs and clearly differentiates PGMs from prior work (distillation, block generation). The contributions are stated explicitly. The motivation is strong, given the importance of test-time compute scaling. One minor point: the sentence "Addressing the inference inefficiency of MGMs is not trivial..." could be tightened, as the following sentence about bidirectional architectures is the core reason, not an additional point.

### Method / Approach
**Training Objective and Connection to MDLM (Sec. 3.1):** The derivation linking the PGM objective to two complementary MDLM losses is clear and theoretically sound. The variance reduction argument is plausible but could be strengthened with a brief theoretical justification or citation beyond the empirical result.
**The Partition Transformer (Sec. 4):** The architecture is the novel core. The description of group-wise self-attention, GroupSwap, and decoder-only cross-attention is logically consistent with the goal of processing only clean tokens. However, the paper is somewhat light on intuition for *why* this specific architecture works. For instance, why is a cross-attention-based GroupSwap necessary versus a simpler mechanism? The claim that data-independent queries work as well as data-dependent ones (Sec. 5.1) is important but presented without exploration. This is a positive finding for simplicity, but a brief hypothesis would be valuable.
**Sampling (Sec. 3.2):** The description of sampling is clear and correctly notes compatibility with existing MGM samplers. The switch to a fixed schedule (`k` tokens per step) for text is pragmatic but is presented as an empirical finding rather than a motivated design choice. A short explanation (e.g., simplifies batching, improves quality) would help.
**Reproducibility:** The method is described in sufficient detail for replication, including the objective (Eq. 7, 8), architecture diagram (Fig. 3), and sampling algorithms (Appendix E.2). The connection to established MGM theory (variational bound) aids credibility.

### Experiments & Results
**Language Modeling (Sec. 5.1):**
- **Baseline Parity:** The results are comprehensive. Table 1 shows that on LM1B (length 128), PGM (6/6) beats MDLM in perplexity (26.80 vs. 27.67) and is much faster. On OWT (length 1024), the base PGM (8/8) slightly underperforms MDLM in perplexity (22.61 vs. 23.07) but is 5.3x faster. To surpass MDLM on OWT, PGM needs more parameters (dim 1024). This is a critical trade-off: the efficiency gain is not entirely "free" in all contexts. The paper acknowledges this in limitations but should highlight it more in the main results discussion.
- **Downstream Tasks (Table 2, 4):** The results show PGMs are competitive, often slightly better, which is encouraging. The adaptation of `lm-eval-harness` using the NELBO is correctly justified.
- **Distillation (Fig. 4, Table 6, 7):** The distillation experiments are thorough. The finding that PGM with nucleus sampling matches MDLM quality while remaining ~4.6x faster is strong. However, the narrative is slightly confusing: the text states the setup "naturally favors MDLM" because distillation is applied to a single group, but the results show PGM does well. Clarifying whether this is expected would be helpful.

**Image Modeling (Sec. 5.2 & Appendix D.3):**
- The results are compelling. With the Halton sampler, PGM achieves a slightly worse FID (5.54 vs. 5.35) at 7.5x the speed of MaskGIT at 32 steps. With 64 steps, PGM achieves a *better* FID (4.56) while remaining 3.9x faster. This is a clear win. Tables 8 and 9 provide exhaustive results across schedulers and guidance strengths.

**Ablation and Analysis (Sec. 5.3, Table 5):**
- The complementary masking ablation is essential and well-executed. It shows that part of PGM's gain comes from denser supervision (lower gradient variance), but an additional gap remains attributable to the architecture. This is good scientific practice.
- Table 5's ablation on layer balance and query initialization is useful. The finding that balanced encoder/decoder layers and data-independent queries work best simplifies future implementations.

**Potential Concerns:**
1.  **Training Stability:** Appendix D.4 and Figure 6 note training loss spikes for complementary masking and PGM. This is a non-trivial engineering concern for scaling. The paper notes all runs converged, but this deserves a sentence in the main limitations.
2.  **Computational Cost:** Appendix E and Table 3 show that PGM training throughput is lower than MDLM's (~75% for the best OWT model). While the inference gains are the primary focus, this training overhead is a practical consideration that should be noted in the main text's trade-off discussion.
3.  **Long Context Evaluation:** The claim that PGMs are suitable for test-time scaling (Sec. 5.1) is supported only by extrapolated latency numbers at length 4096 (Table 10), not by quality evaluations. This is understandable given cost, but the claim should be tempered.

### Writing & Clarity
The paper is generally well-written. The core idea (partition vs. mask) is effectively communicated through Figure 2. Some parts suffer from PDF parsing artifacts (e.g., garbled tables in the abstract, misplaced text in Figure 1), but these are not the authors' fault and do not impede understanding. The algorithmic pseudocode in the appendix is clear.

### Limitations & Broader Impact
The limitations section (Appendix A) correctly identifies the need for more parameters on OWT and the training inefficiency. It could be expanded:
- The training instability (loss spikes) should be mentioned.
- The paper notes PGMs are a "drop-in replacement" but the need for a new architecture (Partition Transformer) is a non-trivial change for practitioners.
- Broader impact is not discussed, which is standard for this type of technical work but sometimes expected by ICLR. A brief statement that the work advances efficient generative modeling without specific negative societal impacts would suffice.

### Overall Assessment
This paper presents a novel and impactful contribution. Partition Generative Modeling (PGM) offers a principled and effective method to decouple the training and inference costs of Masked Generative Models, achieving substantial speedups (5-7.5x) with minimal quality loss. The work is thorough, with rigorous experiments across text and image domains, careful ablations, and compatibility demonstrations with distillation and advanced samplers. The main weaknesses are the occasional need for increased parameters to match baseline quality, training-time inefficiency and instability, and some under-explained architectural choices. These do not undermine the core contribution but are important for a complete understanding. For ICLR, this is a strong submission that introduces a compelling new direction for efficient non-autoregressive generation. The empirical evidence is solid, and the conceptual advance is clear. Acceptance is recommended, contingent on addressing the minor issues raised, particularly a more balanced discussion of the parameter/speed/quality trade-offs.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces Partition Generative Models (PGMs), a novel approach to masked generative modeling that replaces masking with partitioning. Tokens are split into two non-interacting groups, enabling the model to predict each group conditioned on the other without using [MASK] tokens during inference. This yields significant speedups (5–5.5× on text, 7.5× on images) while maintaining sample quality comparable to existing masked generative models (MGMs) like MDLM and MaskGIT.

### Strengths
1. **Novel and impactful idea**: Replacing masks with partitions elegantly eliminates the need to process uninformative [MASK] tokens during sampling, directly addressing a key inefficiency of MGMs. The proposed Partition Transformer architecture enforces group separation while allowing information exchange via a GroupSwap layer.
2. **Strong empirical results**: PGMs achieve 5–5.5× higher throughput than MDLM on OpenWebText with comparable or better generative perplexity, and 7.5× higher throughput than MaskGIT on ImageNet with similar FID. Increasing sampling steps further improves FID to 4.56 while remaining 3.9× faster.
3. **Compatibility and generality**: PGMs are designed as a drop-in replacement for MGMs, maintaining compatibility with existing samplers (e.g., Halton), classifier-free guidance, and distillation methods (e.g., SDTT). Experiments confirm successful distillation without performance degradation.
4. **Thorough analysis**: The paper includes ablations (e.g., complementary masking, architecture variants), variance reduction analysis, and evaluations across multiple metrics (perplexity, FID, downstream tasks) on both language and image datasets.

### Weaknesses
1. **Parameter inefficiency**: On OpenWebText (context length 1024), PGMs require increased parameters (more layers or larger embedding dimension) to match MDLM’s validation perplexity (Table 1, Table 5), suggesting the current architecture is less parameter-efficient.
2. **Training overhead**: PGM training is slower due to dense attention masks; the paper notes that block-diagonal kernels could improve efficiency but are not implemented (Appendix E). Complementary masking also introduces occasional loss spikes (Figure 6), indicating potential stability issues.
3. **Limited comparison to autoregressive models**: While PGMs are compared to MGMs in speed and quality, no direct comparison to autoregressive models (ARMs) is provided, making it harder to assess their standing relative to the dominant text-generation approach.
4. **Ablation limitations**: The complementary masking experiment (Sec. 5.3) shows gains on LM1B but not on OpenWebText, and the explanation (context length difference) is only briefly explored in Appendix D.1 without conclusive analysis.

### Novelty & Significance
The core idea of partitioning instead of masking is novel and addresses a well-known inference bottleneck in MGMs. The significant speedups (5–7.5×) without quality loss represent a meaningful advance toward practical non-autoregressive generation. The work is likely to influence future research on efficient generative modeling, especially for applications requiring low-latency or test-time compute scaling.

### Suggestions for Improvement
1. **Compare with autoregressive models**: Include a direct comparison to ARMs (e.g., GPT-2) in terms of throughput, sample quality, and downstream task performance to better position PGMs in the broader landscape.
2. **Optimize training efficiency**: Implement and evaluate block-diagonal attention kernels to reduce training overhead, and investigate techniques to stabilize training (e.g., loss clipping, gradient normalization) when using complementary masking.
3. **Conduct deeper architectural analysis**: Explore more parameter-efficient variants of the Partition Transformer (e.g., lighter GroupSwap mechanisms) to close the parameter gap on OpenWebText, and study the impact of context length on complementary masking more systematically.
4. **Extend to other modalities**: Demonstrate PGM’s applicability to additional domains (e.g., audio, video) to strengthen claims of generality and inspire follow-up work.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Direct comparison with autoregressive models (ARMs) on quality and speed.** The paper claims PGMs combine ARM efficiency with MGM flexibility but only compares to MGMs. Without benchmarking against a standard ARM (e.g., GPT-2) on perplexity, downstream tasks, and throughput, the claim that PGMs get "the best of both worlds" is unsupported.
2. **Ablation study on the Partition Transformer architecture.** The novel GroupSwap layer and encoder/decoder split are core to the method. An ablation is needed to show the necessity of each component (e.g., removing GroupSwap, using a standard Transformer) to prove the design choices are critical for performance.
3. **Demonstration of any-order generation capability.** The paper claims PGMs retain any-order generation like MGMs, but no experiment validates this. Showing that sampling with different token orders yields similar quality (e.g., via perplexity) is essential to support this claimed advantage.
4. **Scalability to longer context lengths.** The paper notes PGMs require more parameters to match MDLM at length 1024. Experiments at longer contexts (e.g., 2048, 4096) are needed to assess if the throughput gains hold and how quality scales, which is critical for real-world applications.

### Deeper Analysis Needed (top 3-5 only)
1. **Analysis of the speed-quality trade-off curve.** The paper reports fixed step counts. A systematic analysis of how quality (perplexity/FID) degrades as sampling steps are reduced (to further increase speed) would show the practical limits of PGMs and inform optimal operating points.
2. **Quantitative verification of information separation and exchange.** The core mechanism relies on groups not attending to each other in the encoder and controlled exchange via GroupSwap. Analysis (e.g., attention weight visualization, probing classifiers) is needed to confirm no leakage and effective cross-group information flow.
3. **Measurement of gradient variance reduction.** The paper claims training on two complementary copies reduces gradient variance, but provides no quantitative evidence. Comparing gradient norms or variance between PGM and MDLM during training would substantiate this claimed benefit.

### Visualizations & Case Studies
1. **Visual progression of image generation.** For ImageNet, showing intermediate steps (like MaskGIT papers do) would reveal whether the partition-based generation produces coherent structures or artifacts, and how the two groups interact during sampling.
2. **Examples of generated text.** Beyond metrics like Generative Perplexity, displaying sample text from PGM, MDLM, and an ARM would allow qualitative assessment of fluency, coherence, and potential failure modes, which metrics may miss.

### Obvious Next Steps
1. **Implement and benchmark efficient kernels for block-diagonal attention.** The paper notes training is slower due to dense attention masks, but block-diagonal sparsity is inherent. A preliminary efficient implementation (even if not fully optimized) is necessary to demonstrate training efficiency is achievable.
2. **Explore more than two partitions.** The two-group partition is a natural first step, but using more groups (e.g., hierarchical partitioning) could further improve speed or quality. This is an obvious extension that should be discussed with preliminary results.

# Final Consolidated Review
## Summary
This paper introduces Partition Generative Models (PGMs), a novel framework for masked generative modeling that replaces the use of [MASK] tokens with a partitioning scheme. Tokens are split into two non-interacting groups, enabling the model to predict each group conditioned on the other during training, while during inference it processes only the already-generated ("clean") tokens. This leads to substantial sampling speedups (5–5.5× on text, 7.5× on images) compared to standard masked generative models (MGMs) like MDLM and MaskGIT, with minimal loss in sample quality.

## Strengths
- **Architectural innovation enabling efficient inference:** The proposed Partition Transformer, with its group-wise attention, GroupSwap layer, and decoder-only cross-attention, successfully eliminates the processing of uninformative [MASK] tokens during sampling. This is the core mechanism behind the significant throughput gains, as evidenced by the >5x speedup on text and >7.5x on images while matching or nearly matching baseline quality (Tables 1, 6, 9).
- **Strong and comprehensive empirical validation:** The paper demonstrates the effectiveness of PGMs across two domains (language and images) using standard benchmarks (OpenWebText, LM1B, ImageNet). Results show consistent and substantial speedups (5–5.5× for text, 7.5× for images) with competitive perplexity and FID. The work also shows compatibility with existing MGM techniques like advanced samplers (Halton) and distillation (SDTT), and includes informative ablations (e.g., complementary masking, architecture variants).
- **Clear connection to existing MGM theory:** The training objective is formally derived as evaluating two complementary MDLM losses, linking PGMs to the established variational framework for discrete diffusion. This theoretical grounding strengthens the methodological contribution.

## Weaknesses
- **Parameter inefficiency on longer sequences:** To match the validation perplexity of the MDLM baseline on OpenWebText (context length 1024), the PGM architecture requires either more layers or a larger embedding dimension (Table 1, Table 5). This indicates the current Partition Transformer design is less parameter-efficient than standard MGMs for this setting, a trade-off that should be clearly weighed against the inference speed gain.
- **Training instability and overhead:** The complementary masking strategy, integral to PGM training, introduces occasional spikes in the training loss (Fig. 6, Appendix D.4). While all runs converged, this indicates a potential stability concern for scaling. Furthermore, training throughput is currently lower than for MDLM (~75% for the best model on OWT, Table 3) due to the use of dense attention masks, though this could be mitigated with optimized kernels.
- **Incomplete validation of core flexibility claim:** The paper claims PGMs retain the "any-order" generation capability of MGMs, but this is not empirically validated. A simple experiment showing that sample quality (e.g., generative perplexity) is invariant to the order in which token groups are generated would strengthen this foundational claim.

## Nice-to-Haves
- **Direct comparison with Autoregressive Models (ARMs):** While the primary comparison is rightly against MGMs, a head-to-head comparison with a comparable ARM (e.g., GPT-2) on throughput, sample quality, and downstream tasks would better contextualize PGMs within the broader generative modeling landscape.
- **Deeper architectural analysis:** Further exploration of the GroupSwap mechanism (e.g., its necessity, alternatives) and a more systematic study of why the parameter inefficiency arises on longer contexts (OWT) but not shorter ones (LM1B) could provide valuable insights for future architectural improvements.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Strengths:** Generic praise like "the paper is well-written" or "the topic is important."
- **Weaknesses:** Criticisms about missing comparisons to ARMs or extensive architectural ablations as *core* weaknesses, as the paper's primary claim is an improvement over MGMs, and some ablation is provided (Table 5).
- **Weaknesses:** The request for "quantitative verification of information separation" is an interesting analysis but not a standard requirement for establishing the method's empirical efficacy.
- **Weaknesses:** The point about "limited comparison to autoregressive models" is moved to Nice-to-Haves, as the paper's scope is improving MGMs.
- **Weaknesses:** Nitpicks about writing style or minor phrasing in the introduction.

## Novel Insights
The paper's central novel insight is that the inefficiency of processing [MASK] tokens in MGMs can be circumvented by reformulating the training task as mutual prediction between two partitioned, non-interacting token groups. This partitioning allows the model to be trained on all tokens (reducing gradient variance via complementary masking) while architecturally enforcing that during sampling, each forward pass processes only the subset of tokens currently being conditioned on. This decouples the model's bidirectional training from the need to process placeholders at inference, a significant conceptual shift that directly enables the demonstrated order-of-magnitude inference speedups.

## Suggestions
- Add an experiment to validate the "any-order" generation claim, for example by reporting generative perplexity when sampling tokens with different pre-defined orderings (e.g., random, spatial).
- In the limitations or discussion, include a brief analysis or speculation on the cause of the training loss spikes associated with complementary masking and potential mitigation strategies.
- Consider adding a note in the main results discussion (e.g., Section 5.1) explicitly stating the parameter versus speed trade-off observed on OpenWebText, rather than only in the appendix.

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 6.0, 6.0]
Average score: 7.0
Binary outcome: Accept
