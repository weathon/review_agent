=== CALIBRATION EXAMPLE 87 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- **Accuracy & Clarity**: The title accurately captures the core methodological shift and the domain. The abstract clearly articulates the problem (AR pseudo-labelling bottleneck and decoupled supervision in USR), the proposed solution (CTC-driven teacher forcing + mixed sampling), and the empirical outcomes (2× faster training, improved OOD robustness, SOTA across multiple benchmarks).
- **Supported Claims**: The claim that the method "halves training time" is directly substantiated by Table 13 and Figure 5. The robustness and SOTA claims are well-supported by Tables 1–3 and Tables 7–9. The abstract correctly frames the method as a *training-time* optimization rather than an inference-time decoder, avoiding over-claiming on latency during deployment. No unsupported statements are present.

### Introduction & Motivation
- **Motivation & Gap**: The motivation is strong and grounded in practical semi-supervised learning constraints. The identification of two specific pain points in the prior USR framework (1) computational overhead of AR decoding at every step, and (2) self-reinforcing error propagation under distribution shift) clearly motivates the need for a new pseudo-labelling strategy. The gap in prior self-supervised/semi-supervised speech work (often decoupled fine-tuning or computationally prohibitive iterative decoding) is accurately positioned.
- **Contributions**: Clearly stated: CTC-driven teacher forcing, aligned target generation, mixed sampling to mitigate exposure bias, and empirical validation of efficiency/robustness. The contributions match the actual technical content.
- **Claim Calibration**: The introduction appropriately scopes the work to the self-training regime, noting that global coherence of CTC-driven outputs is unnecessary for knowledge transfer. This is a nuanced and accurate framing. No over-selling is detected.

### Method / Approach
- **Clarity & Reproducibility**: The method is well-structured. Equations 3–6, combined with Figure 2, provide a clear procedural description. Appendix A supplies necessary hyperparameters, architecture details, and preprocessing. Reproducibility is high.
- **Assumptions & Justification**: The method relies on two key assumptions: (1) that collapsed CTC tokens provide a sufficiently stable prefix for teacher forcing the attention decoder, and (2) that "matched conditioning" (teacher and student seeing the same CTC prefix) neutralizes sequence-level incoherence. The justification in Section 4.1 is conceptually sound but lacks quantitative validation of the "matched conditioning" hypothesis (e.g., tracking teacher-student token agreement or gradient alignment over training).
- **Logical Gaps / Technical Ambiguities**: 
  - **Equation 6 (AR Mode)**: The loss $L_{CTC,m} = 0.5 CTC(\hat{y}_{CTC,m}, \tilde{y}_{CTC}) + 0.5 CTC(\hat{y}_{CTC,m}, \tilde{y}_{Att})$ applies the CTC loss to attention pseudo-labels. Standard CTC expects target sequences to optionally include blanks and relies on dynamic programming over alignments to variable-length frame sequences. Attention pseudo-labels contain no blanks and are significantly shorter. How does the implementation handle this alignment mismatch? Does the CTC loss silently force excessive blank emissions, and are there gradient instability risks? This requires explicit clarification.
  - **Loss Weighting**: The 0.5/0.5 split in Equations 5 and 6 is presented as a default. While ablated in Table 10(a–b), the underlying assumption that CTC and attention targets provide orthogonal/complementary gradients is not analyzed. Gradient conflict or interference between these dual targets could impact convergence dynamics.
- **Edge Cases**: The method explicitly addresses the train-test exposure bias via mixed sampling (Section 4.2) and sequence incoherence via Appendix C.4. However, failure modes where the CTC branch catastrophically fails (e.g., severe clipping, non-speech audio, or domain shifts that break CTC alignment entirely) are not discussed. In such cases, teacher forcing would propagate degenerate prefixes to the decoder, potentially corrupting both branches.

### Experiments & Results
- **Claims Testing**: The experiments directly target the stated claims. Section 5.1–5.3 rigorously tests robustness to length, noise, and OOD datasets. Section 6 validates in-distribution performance and training efficiency. The evaluation covers the exact failure modes the method aims to fix.
- **Baselines & Fairness**: Baselines are appropriate and strongly competitive (AV-HuBERT, BRAVEn, prior USR, plus extensive SOTA tables 7–9). Appendix A.2 confirms official codebases were used where possible, ensuring fair comparison.
- **Ablations**: Section 7 and Appendix C.2 provide necessary ablations (loss weights, collapse operation, sampling schedule, inference decoding strategies). The constant 0.5 mixed sampling is justified. A missing but valuable ablation would be varying the confidence threshold $\tau$ under CTC-driven pseudo-labelling, as collapsed token sequences change the confidence estimation mechanism (Eq 12–13 vs standard frame-wise masking).
- **Statistical Rigor**: Table 13 reports mean ± std over 3 seeds, and Figure 8 includes error bands. However, the main comparative tables (Tables 1–3, Table 2 SOTA comparisons) report single-run numbers. Given ICLR's emphasis on statistical reliability, especially for claims of "outperforms all baselines by a wide margin" on OOD data, providing variance or confidence intervals for the primary results would strengthen the evidence.
- **Cherry-picking & Metrics**: Results are not cherry-picked; Table 4 ablations show nuanced trade-offs (e.g., removing CTC targets harms OOD, removing Attention targets harms ID). WER is the standard and appropriate metric. The decoding speed claim in Figure 1 (~40× faster) measures pure decoding latency on an H200; however, the proportion of total USR training time spent on pseudo-labelling vs. gradient computation is not quantified, which would better contextualize the observed 2× wall-clock reduction.
- **Datasets**: Well-chosen, covering ID (LRS3), OOD audio/noise (VoxCeleb2 synthetic noise, LibriSpeech), and challenging OOD video (WildVSR). Appropriate for the claims.

### Writing & Clarity
- **Clarity of Explanations**: The paper is generally well-written. The transition from decoupled USR to coupled USR 2.0 is logically sequenced. Figure 2 effectively visualizes the architectural/dataflow shift.
- **Figures & Tables**: Figure 1 and Figure 3 are informative and clearly labeled. Table 4’s ablation matrix is slightly dense; the notation for enabled/disabled PL types in CTC vs Decoder heads requires careful reading but is ultimately decipherable. Figure 7’s visualization of AR inconsistencies is helpful for building intuition, though the caption could explicitly state that these are *teacher* outputs used as *targets*, not model predictions.
- **Impediments**: The primary clarity hurdle is the lack of explicit discussion on how CTC loss handles attention tokens (Eq 6) and how the teacher-student "matched conditioning" mathematically stabilizes training despite token-level incoherence. Minor notation clarifications in these areas would improve readability.

### Limitations & Broader Impact
- **Acknowledged Limitations**: The authors explicitly note three key limitations in Appendix D: (1) longer wall-clock training than pure supervised/SSL fine-tuning, (2) greedy decoding bottleneck limits pseudo-label quality for already-strong branches (ASR/AVSR), and (3) CTC-driven forcing is restricted to self-training regimes. These are honest and well-calibrated.
- **Missed Limitations**: The dependency on initial CTC robustness is not formally framed as a limitation. If the base model's CTC alignment degrades sharply on highly reverberant or multi-speaker OOD audio, the teacher forcing pipeline loses its anchor. Additionally, the mixed sampling probability is treated as a static hyperparameter; while adaptive schedules didn't help empirically, the lack of theoretical grounding for why the 0.5 balance is optimal leaves it somewhat heuristic.
- **Broader Impact & Failure Modes**: The impact statement appropriately covers accessibility benefits and surveillance/privacy risks. It also acknowledges demographic bias risks. Given the unified multimodal nature of the system, a brief discussion of cross-modal failure modes (e.g., visual-only inference degrading more severely than audio under poor lighting, despite the unified representation) would align with ICLR's expectations for multimodal safety.

### Overall Assessment
This is a strong, well-executed paper that addresses a genuine bottleneck in semi-supervised unified speech recognition. The core insight—repurposing CTC outputs as teacher-forced prefixes for parallel attention pseudo-labelling—is elegant, well-motivated, and empirically validated across multiple robustness axes and scales. The methodological contribution is clear, and the 2× training efficiency gain combined with OOD robustness improvements makes it highly relevant for scalable speech model development. The primary concerns are technical clarifications around applying CTC loss to attention targets (Eq 6), a need for consistent statistical reporting across main comparative tables, and a deeper quantitative analysis of why "matched conditioning" neutralizes token incoherence during training. Addressing these points would significantly strengthen the methodological rigor without altering the paper's core contribution. I believe the work meets ICLR's standards for novelty, empirical thoroughness, and practical impact, and recommend acceptance pending minor revisions to clarify the loss formulation and add variance estimates to primary results.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces USR 2.0, a semi-supervised framework for unified audio, visual, and audiovisual speech recognition that replaces costly autoregressive pseudo-label generation with CTC-driven teacher forcing. By feeding greedily decoded CTC outputs directly into the attention decoder and alternating with standard autoregressive decoding via mixed sampling, the method enables parallel pseudo-label creation in a single forward pass. The approach cuts training time by nearly half, substantially improves robustness to long sequences, noise, and out-of-distribution domains, and achieves state-of-the-art performance across multiple benchmarks with a single unified model.

### Strengths
1. **Effective and Well-Motivated Architectural Innovation:** The core idea of using collapsed CTC outputs to drive the attention decoder during self-training directly addresses the autoregressive bottleneck in pseudo-labelling. Section 4.1 clearly explains why global sequence incoherence in the generated pseudo-labels does not harm learning, as teacher and student share identical conditioning during training, a sound insight for self-training dynamics.
2. **Rigorous and Comprehensive Empirical Validation:** The paper thoroughly evaluates OOD robustness across three critical axes: sequence length (Figure 3), noise (Table 1), and cross-dataset generalisation (Table 3). The results consistently show USR 2.0 outperforming strong self-supervised baselines (AV-HuBERT, BRAVEn) and the original USR, particularly under greedy decoding where pseudo-labelling operates.
3. **Strong Practical Efficiency Gains:** The method delivers ~2× faster training while maintaining or improving accuracy. The ablation in Appendix C.5 and Figure 5 clearly demonstrate both faster per-step computation and accelerated convergence (50 vs. 75 epochs), making it highly relevant for scaling semi-supervised speech models.
4. **Transparent Reproducibility:** The appendices provide exhaustive details on architecture variants (Table 5), hyperparameters (Table 6), confidence filtering adaptations, loss formulations, and training protocols. Code availability is explicitly promised, and baseline comparisons are conducted using official repositories, ensuring fair and reproducible benchmarking.

### Weaknesses
1. **Attribution of Efficiency Gains Could Be Clearer:** The ~2× speedup claim combines faster forward passes (from CTC-driven teacher forcing) with fewer total epochs. While Appendix C.5 disentangles this, the main text conflates the two, which may overstate the per-iteration computational gain. A clearer breakdown in the main body would improve precision.
2. **Sequence-Level Confidence Filtering Trade-offs:** USR 2.0 replaces frame-level confidence filtering with sequence-level filtering for CTC pseudo-labels (Section B.2, Eq. 12-13). While mathematically necessary due to CTC collapse, this coarse-grained thresholding may discard high-confidence partial sequences or retain low-confidence full sequences. The impact of this design choice is not ablated.
3. **Limited Architectural Generalisation Discussion:** The method is tightly coupled with the USR transformer encoder-decoder and joint CTC-attention framework. The paper only briefly mentions potential extensions to RNN-T or streaming models in the conclusion. Without analysis or even theoretical discussion of compatibility with non-autoregressive or state-space architectures, broader impact remains speculative.
4. **Fixed Mixed Sampling Probability:** The default 0.5 AR-sampling probability is motivated empirically, and the appendix shows linear/cosine schedules perform similarly. However, the paper does not investigate why a fixed schedule works robustly across diverse datasets and modalities, nor does it address potential benefits of task- or modality-adaptive mixing, leaving room for deeper analysis.

### Novelty & Significance
**Novelty:** The paper presents a clear, incremental novelty over the authors' prior USR work. Repurposing CTC outputs as teacher-forcing inputs for parallel attention pseudo-label generation is a clever and non-trivial adaptation of joint CTC-attention training principles to the self-training loop. The mixed sampling strategy adapts scheduled sampling concepts to a novel exposure bias in pseudo-labelling.
**Clarity:** The manuscript is exceptionally well-structured. The progression from problem identification (Section 3) to method (Section 4) and comprehensive ablations (Section 7) is logical and easy to follow. Mathematical notation is consistent, and qualitative examples (Table 11, Figure 7) effectively ground theoretical claims.
**Reproducibility:** High. Detailed hyperparameters, dataset licenses, training protocols, loss weighting schemes, and explicit references to baseline codebases are provided. The modular design of CTC-driven vs. AR modes with clear probability sampling makes straightforward implementation feasible.
**Significance:** High for the speech, multimodal, and semi-supervised learning communities. Halving training time while improving OOD robustness addresses a major bottleneck in scaling unified models. The insights on leveraging CTC's monotonic alignment to stabilise attention decoders during self-training have potential applicability to other sequence-to-sequence tasks where pseudo-labelling is computationally prohibitive.

### Suggestions for Improvement
1. **Disentangle Training Efficiency Metrics in the Main Text:** Explicitly separate per-step speedup from convergence improvements (e.g., "X% faster iterations, Y fewer epochs required") in Section 4 or 6 to provide a more transparent computational accounting.
2. **Ablate Confidence Filtering Granularity:** Add a brief evaluation comparing sequence-level vs. frame-level (or token-level) confidence masking for CTC pseudo-labels to quantify any performance degradation or filtering inefficiency introduced by the collapse operation.
3. **Expand Architectural Generalisation Analysis:** Include a short subsection or appendix discussing how CTC-driven teacher forcing would interact with alternative decoders (e.g., RNN-T, SSM-based, or non-autoregressive parallel models), noting which constraints (e.g., alignment assumptions, teacher forcing compatibility) are essential vs. flexible.
4. **Motivate the Fixed 0.5 Mixed Sampling Theoretically or Empirically:** Provide a brief analysis linking the fixed probability to exposure bias literature or gradient variance during self-training, and clarify whether modality-specific mixing ratios (e.g., higher AR for ASR, higher CTC-driven for VSR) could offer marginal gains.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3 only)
1. Replace Whisper auto-transcripts with human-verified or multi-model consensus ground truth for all OOD benchmarks. Without verified references, robustness claims under domain shift and noise are confounded by the oracle's own degradation, making WER gaps unreliable.
2. Compare against a USR baseline trained for the exact same number of optimization steps or GPU-hours, not just wall-clock time. The SOTA and efficiency claims cannot be disentangled from the fact that faster steps allow the model to see more pseudo-labelled batches per unit time.
3. Include a modern AR pseudo-labelling baseline utilizing KV-caching, batched generation, and early stopping. The current speedup comparisons omit standard PL acceleration techniques, artificially inflating the perceived novelty of the training efficiency gains.

### Deeper Analysis Needed (top 3 only)
1. Ablate the coupled supervision objective from the generation mechanism to prove that robustness improvements stem from architectural joint training rather than merely faster convergence allowing more data iterations. The method's core theoretical contribution remains unisolated.
2. Quantify autoregressive error propagation depth with and without mixed sampling by measuring token-level accuracy decay across generation steps. Final WER alone cannot validate the claimed mitigation of train-test mismatch and exposure bias.
3. Analyze how the fixed 0.8 sequence-level confidence threshold interacts with long/noisy inputs. If the filter disproportionately prunes high-error OOD samples, reported WER improvements likely reflect selective data curation bias rather than genuine decoder robustness.

### Visualizations & Case Studies
1. Plot cross-attention alignment heatmaps for the decoder under CTC-driven forcing versus standard AR mode. This directly reveals whether the decoder learns stable monotonic mappings (explaining robustness) or collapses to diagonal/self-attending artifacts (exposing fragility).
2. Map error types (insertions, deletions, substitutions) binned by input sequence length to pinpoint exactly where cascading AR failures occur and whether CTC-driven forcing specifically prevents early-step drift or merely caps late-sequence degradation.
3. Visualize the loss landscape or validation WER as the mixed sampling probability sweeps from 0.0 to 1.0 under strict compute normalization. This exposes whether the default 0.5 probability is a genuine optimum or an arbitrary compromise hidden by unequal training steps.

### Obvious Next Steps
1. Report all training dynamics and efficiency metrics normalized to FLOPs or GPU-hours rather than epochs/steps, which is the standard expectation for validating algorithmic speedup in modern self-training frameworks.
2. Evaluate sensitivity to subword vocabulary size and merge strategies, as the correctness and length of collapsed CTC outputs are heavily dependent on tokenizer granularity and directly impact decoder input quality.
3. Test a lightweight AR correction pass post-CTC-driven decoding to quantify the exact inference gap caused by train-test mismatch, providing a principled path toward low-latency decoding that bridges the paper's main limitation.

# Final Consolidated Review
## Summary
This paper introduces USR 2.0, a semi-supervised framework for unified audio, visual, and audiovisual speech recognition that replaces costly autoregressive pseudo-label generation with CTC-driven teacher forcing. By feeding collapsed CTC outputs directly into the attention decoder and alternating with standard autoregressive decoding via mixed sampling, the method enables parallel pseudo-label creation in a single forward pass. The authors report nearly halved training time, substantially improved robustness to long sequences and distribution shifts, and state-of-the-art performance across multiple benchmarks.

## Strengths
- **Clean and Practically Impactful Methodological Insight:** The core idea of repurposing collapsed CTC outputs as deterministic, parallelizable scaffolds for the attention decoder directly targets a known computational bottleneck in iterative self-training. The observation that global sequence coherence is unnecessary when teacher and student share identical conditional prefixes is both conceptually sound and elegantly implemented.
- **Rigorous and Multi-Axis Empirical Validation:** The paper thoroughly evaluates out-of-distribution robustness across sequence length, additive noise, and cross-dataset generalization. The consistent gains under greedy decoding directly address the failure modes that limit prior USR frameworks, with strong evidence supporting improved stability in self-reinforcing training loops.
- **High Reproducibility & Transparent Engineering:** Extensive appendices detailing architecture variants, hyperparameter schedules, confidence-filtering adaptations, and loss formulations, combined with the use of official baseline codebases, ensure the work is highly reproducible and practically deployable for scaling unified speech models.

## Weaknesses
- **Lack of Compute-Controlled Baselines Undermines Efficiency Claims:** The reported ~2× wall-clock speedup conflates per-iteration latency reduction with faster convergence (50 vs. 75 epochs). Without a strictly compute-normalized comparison (fixed GPU-hours or matched iteration count), it remains unclear whether the gains stem from the proposed architectural coupling or merely from the model seeing more pseudo-labelled batches per unit time. This leaves the core algorithmic efficiency claim partially unvalidated.
- **Statistical Reporting Gap in Primary Results:** While variance across seeds is provided for the epoch-comparison ablation, all primary OOD and SOTA tables report single-run metrics. Given the strong claims of "wide margin" improvements under domain shift and noise, the absence of error bounds or multi-seed aggregation for these central results obscures the statistical reliability of the robustness gains at ICLR standards.
- **Unvalidated Assumptions Around Matched Conditioning & Loss Coupling:** The paper asserts that sharing the CTC prefix between teacher and student neutralizes sequence-level incoherence, yet provides no empirical tracking (e.g., token agreement trajectories, gradient alignment, or error propagation depth). Similarly, the fixed 0.5 mixed sampling ratio and dual-objective loss weighting are presented heuristically, with no analysis of potential gradient interference between CTC and attention targets or why a static schedule generalizes across modalities.

## Nice-to-Haves
- Cross-attention alignment maps or error-type breakdowns (insertions/deletions/substitutions binned by length) would concretely illustrate whether CTC forcing stabilizes monotonic decoder mappings versus merely capping late-sequence degradation.
- A brief ablation of sequence-level vs. token-level confidence filtering could quantify any performance trade-offs introduced by the CTC collapse operation in highly noisy regimes.
- Explicit clarification on how the standard CTC loss handles autoregressive pseudo-labels (shorter token sequences without blanks) would address potential gradient instability concerns in AR mode.

## Novel Insights
The paper successfully reframes a fundamental constraint of joint CTC-attention self-training: global coherence of pseudo-labels is unnecessary when teacher and student operate under matched conditional prefixes. By treating collapsed CTC outputs as a robust, parallelizable teacher-forcing scaffold, the method decouples pseudo-label generation speed from autoregressive error accumulation, leveraging CTC's monotonic alignment to anchor attention decoding under distribution shift. Crucially, the demonstration that OOD robustness gains directly elevate in-distribution performance reveals a systemic dependency in semi-supervised speech learning: decoder stability on noisy or long-tail sequences is a prerequisite for unlocking the full utility of unlabelled corpora.

## Potentially Missed Related Work
- Speculative decoding and KV-cache optimized autoregressive generation pipelines for pseudo-labelling; contrasting against modern parallelized AR techniques would better isolate the novelty of the efficiency gains.
- Confidence-based curriculum and scheduled sampling methods in self-supervised speech pre-training (e.g., Noisy Student variants, momentum-based confidence filtering), which provide theoretical grounding for the mixed sampling strategy.

## Suggestions
- Run a strictly compute-controlled baseline (fixed total iterations or GPU-hours) comparing USR and USR 2.0 to disentangle per-step algorithmic efficiency from convergence acceleration and regularization effects.
- Report mean ± standard deviation across ≥3 random seeds for all primary out-of-distribution and cross-dataset tables to substantiate robustness claims with appropriate statistical rigor.
- Provide a theoretical or empirical analysis of the fixed 0.5 mixed sampling and dual-target loss weighting, ideally including gradient norm or cosine similarity ablations to rule out objective interference and explain modality-agnostic generalization.

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 8.0, 8.0]
Average score: 8.0
Binary outcome: Accept
