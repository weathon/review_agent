The paper text is in the user message. Let me re-read the relevant sections about the dataset construction more carefully.

From the paper (lines ~378-431):
- "we opt to extend HCSTVG-v2 (Tang et al., 2021) for creating new datasets for LF-STVG. The reason for choosing HCSTVG-v2 only for extension is that it is the only dataset which provides available source videos, thus allowing for extension with longer videos."
- "The extensions are based on original YouTube videos, not concatenated clips, and we manually review the extended videos to ensure their quality."
- Only the validation set is extended (because test set annotations are not publicly available).

The harsh critic raises a valid question: where does the ground-truth event fall in the extended videos? The paper doesn't explicitly clarify whether the original annotated 20-second segment's temporal boundaries were re-labeled or whether they remain the same. This is a genuine weakness.

**Claim 2 (Harsh Critic #2): "Incomplete specification of the loss"** — The paper says in Sec 3.5: "Due to limited space, please see our loss function in supplementary material." This is a genuine omission from the main paper.

**Claim 3 (Harsh Critic #3): "Autoregressive / streaming claim not matched by training"** — Training uses Nf=64 frames at 3.2 fps ≈ 20 seconds. The paper does not describe how inference handles 1-5 minute videos (sliding windows? full sequence with memory growing?). The memory banks grow without bound. This is a valid concern.

**Claim 4 (Harsh Critic #4): "Temporal modeling underspecified"** — The temporal prediction pipeline from hi to final tube is indeed not described. This is valid.

**Claim 5 (Harsh Critic #5): "Baseline comparisons not fair"** — This is partially valid but needs nuance. The baselines are run with their provided source codes, and they are inherently not designed for long videos. The paper's claim is precisely that existing methods are not suited for LF-STVG. That the baselines degrade on long videos is the *point*. However, the absence of sliding-window or chunked adaptations of baselines is a valid concern.

Now let me write my final review.

## Summary

This paper introduces Long-Form Spatio-Temporal Video Grounding (LF-STVG), extending the STVG task from short videos (~20 seconds) to 1–5 minute videos. The authors propose ART-STVG, an autoregressive transformer that processes frames sequentially with memory-augmented spatial and temporal decoders, selective memory strategies, and a cascaded spatio-temporal design. The framework is evaluated on newly extended versions of HCSTVG-v2 (LF-STVG-1min through -5min), showing substantial improvements over existing STVG methods on long videos while remaining competitive on short-form STVG.

## Strengths

- **Important and well-motivated problem formulation.** The gap between short-form STVG benchmarks (avg. 20–35 seconds) and real-world videos spanning minutes or hours is significant and underexplored. Figure 2 compellingly demonstrates that existing methods degrade sharply as video length increases (e.g., TA-STVG drops from 38.4% m_tIoU at 1 min to 7.7% at 5 min).

- **Coherent and novel architectural design.** The autoregressive streaming approach is a natural fit for long videos, avoiding the need to process all frames simultaneously. The selective memory banks and cascaded spatio-temporal decoder are well-motivated and demonstrate clear ablation improvements (temporal memory selection: 9.6% → 23.0% m_tIoU; cascaded vs. parallel: +1.5% m_tIoU).

- **Strong empirical gains on LF-STVG benchmarks.** ART-STVG consistently outperforms all prior methods across all five LF-STVG benchmarks with especially large margins on longer videos (e.g., +7.3%/5.5% m_tIoU/m_vIoU over TA-STVG on 5-min videos). The trend showing larger improvements on longer videos supports the core design motivation.

- **Systematic ablation studies.** Tables 2–5 provide clean ablations of temporal memory, spatial memory, the cascaded design, and the number of selective memories, demonstrating the contribution of each component.

## Weaknesses

### Major

- **Benchmark construction limitations.** The LF-STVG benchmarks are extensions of only the HCSTVG-v2 validation set, using the original YouTube videos to extend clips from 20 seconds to 1–5 minutes. Critical details are missing: (1) Are the ground-truth temporal boundaries still within the original 20-second annotated window, or were they re-labeled for the longer context? (2) Do the extended portions contain semantically similar distractors or just irrelevant padding? The paper states extensions are "based on original YouTube videos, not concatenated clips, and we manually review the extended videos to ensure their quality" but provides no annotation protocol or analysis of what information the longer context adds beyond more irrelevant content. If the target event always lies within the original 20s window and the rest is pure distractor, then "LF-STVG" here is effectively "short-form grounding with long irrelevant padding"—a much narrower task than the claimed goal of "locating the target in long-term videos." This undermines the paper's central claim of being "the first to explore LF-STVG."

- **Unboundedly growing memory banks.** The paper explicitly states "we update the memory bank by simply adding the query as a new memory, without removing any existing memories" (Sec. 3.3). For a 5-minute video at 3.2 fps, this means ~960 memory entries per partition per decoder block. No analysis of memory growth, computational complexity at inference, or any memory pruning/forgetting strategy is provided. This directly affects the scalability claims—without bounding memory, the approach does not "naturally" handle arbitrarily long videos as claimed.

- **Missing loss function and temporal prediction details.** The loss function is explicitly deferred to supplementary material (Sec. 3.5), and the process for converting per-frame start/end probabilities (h_i ∈ R²) into a final predicted temporal tube is never described. For a core algorithmic paper, these omissions make it impossible to fully understand or reproduce the method, and hinder assessment of whether improvements come from the proposed architecture or from a particular loss/matching strategy.

- **No comparison with adapted/streaming baselines.** All compared methods (TubeDETR, STCAT, CG-STVG, TA-STVG) are non-streaming approaches fed entire long videos at once, which they were not designed for. A natural and more informative baseline would be to adapt these methods with chunked/sliding-window processing (the obvious engineering solution for handling long inputs), which is never attempted. This means the paper shows that methods not designed for long videos fail on long videos, but does not show whether a simpler streaming adaptation could work comparably.

### Minor

- **The baseline (without memory) performs worse than TubeDETR on LF-STVG-1min (30.1 vs 32.5 m_tIoU) and ties on 2min (23.0 vs 23.0).** This means the bare autoregressive design without memory is not beneficial over even non-streaming methods, contradicting the framing that streaming processing itself is advantageous. The gains come specifically from the memory and selection mechanisms, not from the autoregressive paradigm per se. This should be discussed explicitly.

- **Only one dataset used for evaluation.** All long-form results come from extensions of HCSTVG-v2, a human-centric dataset. No second STVG dataset (e.g., VidSTG) is evaluated, limiting evidence of generalizability. The paper explains that HCSTVG-v2 is the only dataset with available source videos for extension, which is a practical constraint, but it still limits the conclusions.

- **No computational efficiency analysis.** The paper motivates ART-STVG partly by claiming it "resolves the computational bottleneck" (Sec 1) of processing all frames at once, yet provides no GPU memory usage, inference time, or FLOPs comparisons for different video lengths. Without this, the efficiency advantage remains an unverified claim.

- **Temporal memory selection is heuristic and under-specified.** The TextTiling-inspired approach (compute similarities between adjacent frame memories, take low-similarity points as boundaries) is described qualitatively but lacks algorithmic details (similarity function, thresholding). The ablation (Table 2) shows selection is necessary but does not compare against simpler alternatives (e.g., recency window, fixed-size FIFO).

### Trivial

- The paper claims to be "the first" to explore LF-STVG (Sec 1, Conclusion). Given the limited benchmark definition (validation-set-only extension, possible short-form-with-padding), this claim is somewhat overreaching.

## Nice-to-Haves

- Train ART-STVG on longer videos (beyond the 40s explored in Table 6) and evaluate the impact—this would be a strong selling point for the streaming design.
- Provide sliding-window adaptations of existing STVG methods as additional baselines.
- Analyze error propagation across the autoregressive chain (does spatial grounding error in early frames compound?).
- Release the LF-STVG extended datasets publicly and include detailed annotation documentation.

## Removed Points

- **"The loss specification makes comparisons unfair because baselines use different losses."** — While the loss should be in the main paper for completeness, the baselines are run with their own published code which includes their own matching/loss designs. Each method uses its standard setup; the comparison is not obviously unfair in this regard. The more legitimate concern is the missing specification itself, not comparative unfairness.

- **"Training-evaluation length mismatch undermines all claims."** — This is partially true but overstated. The 40-second training experiment (Table 6) shows all methods improve with longer training data, and the paper acknowledges this. The training mismatch is a real limitation but is explicitly acknowledged, and ART-STVG still outperforms others when trained on the same 40s data.

- **"The autoregressive/streaming claim is overstated because VidSwin also uses previous frames."** — VidSwin's use of neighboring frames for motion features is a local temporal window for feature extraction, not contradictory to the streaming claim. The key streaming property is that the decoder processes one frame at a time, which is accurate.

- **Demand for VidSTG extensions.** — The paper explains that VidSTG source videos are not available, making extension impossible. This is a practical limitation, not a methodological flaw.

## Novel Insights

The most important insight from synthesizing the reviews is that the paper's contribution is best understood as demonstrating that *selective memory mechanisms*—not the autoregressive paradigm itself—are the key to handling long-form STVG. The baseline without memory underperforms even non-streaming methods on shorter long-form benchmarks, while adding selective memory produces dramatic improvements (temporal: 9.6 → 23.0 m_tIoU). However, the benchmark definition raises questions about whether the task genuinely tests long-form temporal reasoning or primarily tests robustness to irrelevant context padding. The distinction matters: if ground-truth events are always in the original 20s window, selective memory may mainly function as an attention mechanism to filter noise, rather than truly modeling long-range temporal dependencies.

## Suggestions

- **Define the LF-STVG task more rigorously:** Provide explicit annotation documentation for the extended datasets, including whether temporal boundaries were re-labeled, the distribution of target events within the extended window, and whether the extended context contains semantically similar distractors. If the task is "short-form grounding with long irrelevant padding," frame it as such rather than claiming general long-form reasoning.

- **Include the loss function and temporal tube extraction procedure in the main paper.** These are core algorithmic details essential for reproducibility and understanding.

- **Add computational cost analysis:** Report GPU memory usage and inference time as a function of video length for both ART-STVG and baselines. Address memory bank growth explicitly, and consider evaluating on videos longer than 5 minutes.

- **Compare against sliding-window baselines:** Adapt at least one strong STVG method (e.g., TA-STVG) with overlapping window processing and temporal merging to demonstrate that the architecture's advantages go beyond avoiding full-video processing.

## Evaluation Dimensions

**Originality:** The problem formulation of LF-STVG is novel and important, though individual components (memory banks, autoregressive transformers, cascaded decoders) have precedent. The integration is creative. Moderate novelty.

**Importance of research question:** High. The gap between short-form benchmarks and real-world long videos is significant and underaddressed.

**Claim support:** Partially supported. The empirical gains are strong, but the benchmark definition is imprecise and key methodological details (loss, temporal prediction procedure) are missing. The claim of being "the first LF-STVG framework" is weakened by the narrow interpretation of what the benchmark tests.

**Soundness of experiments:** The ablations are clean and informative, but the lack of adapted baselines, efficiency analysis, and second dataset limits the conclusions. No variance or statistical significance is reported.

**Clarity:** Generally well-written, but methodological details (particularly Sections 3.3–3.4 and the loss function) are insufficiently specified for independent reproduction.

**Value to community:** Moderate to high. The LF-STVG problem is important, and the datasets (if properly documented) could catalyze further research. However, the current benchmark definition needs refinement.

## Score and Decision Calibration

Calibrating against retrieved papers:
- **TA-STVG (WOzffPgVjF)**, a strong STVG method accepted as Oral (scores 8,6,8,8): established SOTA on standard benchmarks with clear methodology and full details.
- **Adaptive Memory for Long-form Video (1DEHVMDBaO)**, rejected (scores 5,3,5,5,5): shared similar weaknesses (limited novelty of memory mechanism, limited benchmarks, lack of long-form evaluation despite claims, missing important comparisons). This paper has a stronger problem formulation and larger empirical gains, but similar concerns about benchmark validity and missing efficiency analysis.
- **SAM2Long (Ze49bGd4ON)**, withdrawn (scores 5,3,8,5): memory-based method for long videos with heuristic strategies and limited novelty, similar concerns about engineering vs. research contributions.
- **Self-Supervised STVG (eEtfBIjzWi)**, withdrawn (scores 5,5,5,3): established a new benchmark but with limited novelty and weak evaluation.

This paper is stronger than the rejected AMM paper due to its more substantial empirical gains and better-motivated architecture, but suffers from significant weaknesses in benchmark definition (are we testing long-form reasoning or robustness to padding?) and missing core methodological details. It is weaker than TA-STVG which had complete methodology, multiple benchmarks, and thorough evaluation. The paper lies in the borderline range—interesting problem and approach, but the evaluation framework needs more rigor.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>