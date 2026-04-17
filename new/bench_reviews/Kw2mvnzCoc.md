Now I have enough calibration context. Let me synthesize my final review.

**Papers for calibration:**

- FITS (8,8,8,8 - Accept spotlight): Lightweight TS model using frequency domain, 10k params, strong results. Clean architecture, well-validated.
- ROSE (5,6,6,6 - Reject): Register-assisted TS pre-training with decomposed frequency learning. Similar to TSPulse in some aspects (registers, frequency decomposition). Rejected due to limited novelty, insufficient ablation, and task-specific limitations.
- DAM (8,6,6,8 - Accept poster): Foundation model for forecasting, strong zero-shot transfer, novel ideas.
- LPTM (3,5,5,3,3 - Reject): Cross-domain TS pre-training, weak baseline comparisons, insufficient evidence.
- PTE4TS (3,3,5,3 - Reject): TS pre-trained encoder, lacking ablation, no clear contribution.

**Assessment of TSPulse:**

TSPulse has genuinely strong empirical results across four tasks with a very small model. The hybrid masking trick is simple but effective. However, there are genuine issues with:

1. The "disentanglement" claim - it's essentially multi-head/multi-objective training without architectural enforcement of separation. The evidence is synthetic perturbation tests only.
2. Task-specific pre-training - the paper positions this as a "family of models" but the strongest results come from task-specific variants.
3. Ablation doesn't isolate disentanglement from capacity/multi-objective effects.
4. The zero-shot AD claim uses labeled tuning data for head selection.
5. Comparison fairness questions (smaller baseline variants, different fine-tuning setups).

These are similar to ROSE's weaknesses (insufficient ablation, register token novelty questions), but TSPulse has stronger empirical results across more tasks. It's above ROSE (5-6 range) but below FITS (8) due to the overclaimed disentanglement and comparison fairness issues.

I'd say a score of 5.5-6 is fair. The paper has real engineering merit and practical value, but the core conceptual claim of "disentanglement" is oversold relative to the evidence.</think>## Summary
TSPulse introduces a family of ultra-light (1M parameter) pre-trained models for time-series diagnostic tasks—classification, anomaly detection, imputation, and similarity search—using a TSMixer backbone with multi-objective masked reconstruction across temporal, spectral, and semantic embedding spaces. The paper proposes hybrid masking strategies, lightweight post-hoc fusers (TSLens, MHT), and identity-initialized channel mixers, reporting strong gains over much larger pre-trained models across 75+ datasets.

## Strengths
- **Strong and consistent empirical performance** across four distinct diagnostic tasks with a 1M-parameter model. The reported gains—including +20% on TSB-AD (ranking first), +25% on similarity search, +50% on imputation, and +5–16% on multivariate classification—are substantial and span well-established benchmarks (TSB-AD, UEA, LTSF, UCR).
- **Practical efficiency**: GPU-free deployment at 1M parameters with TSMixer backbone is a genuine advantage. The model is dramatically smaller than baselines like MOMENT (~340M) and UniTS (~37M), which matters for real-time and edge applications.
- **Hybrid masking is well-motivated**: The 79% drop in imputation when trained with only block masking (Table 1c) clearly demonstrates that matching pre-training corruption patterns to realistic missingness matters. This is a simple, practical contribution.
- **Sensitivity analysis (Table 2)** provides interpretable evidence that different embedding types respond differently to perturbations (temporal embedding: 130% distortion under phase shift vs. semantic: 12%), supporting the functional utility of multi-view representations even if not proving full disentanglement.
- **Comprehensive ablation studies** (Table 1) cover key design choices across tasks, showing consistent drops of 7–16% when individual components (dual-space learning, TSLens, masking, channel mixing) are removed.

## Weaknesses

### Major:
- **The central "disentanglement" claim is oversold relative to evidence.** The paper frames multi-output heads on shared backbone+decoder segments as producing "explicitly disentangled" temporal, spectral, and semantic embeddings. However, the mechanism is simply applying separate losses to different output slices—there are no architectural isolation constraints (e.g., separate encoders, orthogonality penalties, capacity bottlenecks), making this better characterized as *soft specialization via multi-task heads*, not disentanglement. The quantitative evidence is limited to synthetic perturbation tests (Table 2) and ablations that conflate removing an entire output head (and its supervision) with removing "disentanglement." No probing, mutual information, or conditional prediction analyses verify that each embedding type truly excludes information from other factors. The model may simply be a well-tuned multi-head autoencoder—still useful, but the conceptual framing is misleadingly strong.

- **Task-specific pre-training undermines the "family of versatile models" framing.** Section 3.1 states "we specialize the pre-training for every task through reweighting loss objectives." This means the four-task performance comes from *separately pre-trained model variants*, not a single versatile model. The paper's title and abstract imply a unified model, but the real cost is four separate pre-training runs (1B samples each). The paper should transparently report results from a single universal model with equal loss weighting to clarify the trade-off.

- **Ablation does not isolate architectural novelty from training recipe effects.** With 1M parameters trained on ~1B samples, it is impossible to tell whether the gains come from the proposed architecture (dual-space heads, register tokens, TSLens), the data scale, or the hybrid masking strategy. No comparison against a same-size, same-data baseline using standard single-space masked reconstruction is provided. The closest ablation (removing "dual-space learning" in Table 1b/c) simultaneously removes heads, objectives, and capacity, changing training dynamics rather than just isolating "disentanglement."

- **"Zero-shot" anomaly detection uses labeled data for head selection.** For TSB-AD, the paper reports "zero-shot" results where multi-head triangulation selects the best head on a labeled tuning set. While this follows benchmark protocol, calling it "zero-shot" is misleading—this is supervised model selection. The truly zero-shot ensemble results (Head-ensemble, no labels) should be reported as prominently. The degree to which the headline +20% improvement depends on this label access is unclear.

### Minor:
- **Comparison fairness across tasks is uneven.** For similarity search, baseline models (MOMENT, Chronos) are evaluated only at their smallest variants "to match embedding size," which disadvantages them on representation quality. For imputation, baseline comparison mixes different evaluation protocols (MOMENT zero-shot vs. UniTS prompt-tuned with 10% data). The consistent claim of "outperforming models 10–100× larger" cannot be rigorously assessed without apples-to-apples capacity-matched comparisons.
- **Synthetic-only sensitivity analysis limits generalizability claims.** Table 2 validates disentanglement on synthetic signals with controlled perturbations, but does not extend to real downstream data where distribution shifts, noise types, and missingness patterns are more complex. The claim that embeddings are "effective across four TS diagnostic tasks" rests on task performance, not on demonstrated disentanglement in those tasks.
- **The mini-decoder size and register token count lack sensitivity analysis.** The decoder is stated to be 10–20% of backbone size, and register tokens R appear fixed, but no analysis explores how these hyperparameters affect the quality of semantic embeddings or overall performance.

### Trivial:
- The optional next-point prediction head (Section 2) is mentioned but never ablated in isolation, making its contribution unclear.

## Nice-to-Haves
- A same-parameter, same-data comparison against a plain masked autoencoder baseline (single reconstruction head, single masking strategy) would decisively isolate the architectural contribution.
- Reporting a single universal pre-trained model's performance across all four tasks (with equal loss weighting) would clarify the real cost of task specialization.
- t-SNE/UMAP visualizations of temporal, FFT, and semantic embeddings on real (not synthetic) datasets would strengthen the disentanglement narrative.
- CPU latency/throughput benchmarks for downstream inference (not just pre-training speed) would substantiate the "GPU-free deployment" claim.

## Removed Points
- **"Pre-training data may overlap with evaluation data"** — The paper explicitly states (Section 4 and Reproducibility Statement) that pre-training datasets do not overlap with any evaluation datasets, with datasets listed in Appendix A.8. No evidence suggests otherwise.
- **"Missing recent baselines for imputation"** — The paper already compares against strong supervised imputation methods (TimesNet, FedFormer) and pre-trained baselines (MOMENT, UniTS). Demanding additional recent methods falls to scope creep; the comparison set is sufficient for the claims made.
- **"Similarity search uses non-standard benchmarks"** — The paper constructs evaluation datasets from UCR with well-defined augmentation protocols (time shifts, magnitude changes, noise) and defines clear metrics (PREC@k, MRR@k). This is a reasonable evaluation for similarity search, which lacks established benchmarks comparable to TSB-AD or UEA.
- **"The paper should report statistical significance and confidence intervals"** — Large-scale benchmark evaluations across 40+ (AD) and 29 (classification) datasets are standard in the field without per-dataset confidence intervals. This is a nice-to-have, not a standard requirement.
- **"Formatting/style nitpicks"** — Removed per instructions.

## Novel Insights
The hybrid masking strategy—mixing block and point-level masking during pre-training to align corruption patterns with realistic missingness—is a practical insight that goes beyond TSPulse: any masked autoencoder for time-series imputation or anomaly detection should consider hybrid corruption. The finding that identity-initialized channel mixers stabilize fine-tuning after univariate pre-training is an underappreciated practical detail for multi-channel adaptation.

## Suggestions
- Run and report a single universal pre-trained model (equal loss weights, no task specialization) across all four tasks to transparently quantify the cost of specialization.
- Add a controlled "entangled" baseline: same total parameter count, same data, same backbone, but with a single pooled embedding and one reconstruction objective. This directly tests whether multi-head disentanglement helps beyond multi-objective supervision.
- Report truly zero-shot AD results (Head-ensemble, no labeled data) alongside the triangulation results in the main text, not just as one entry in Figure 4.

## Score and Decision

**Calibration references:**
- FITS (8,8,8,8) — lightweight TS model with clean architecture, strong empirical results, clear contribution. TSPulse has similarly strong results across more tasks but with overstated conceptual claims.
- ROSE (5,6,6,6) — register + decomposed frequency pre-training for TS, similar conceptual territory. Rejected for insufficient ablation/novelty. TSPulse is clearly stronger than ROSE in empirical coverage and ablation depth.
- DAM (8,6,6,8) — foundation model for forecasting, strong zero-shot, novel design. TSPulse has more tasks but weaker conceptual novelty.
- LPTM (3,5,5,3,3) / PTE4TS (3,3,5,3) — weak TS pre-training papers with major evidence gaps. TSPulse is well above these.

TSPulse delivers genuinely useful engineering with strong empirical results, but the core "disentanglement" claim is substantiated primarily by synthetic perturbation tests and ablations that conflate multi-objective training with factor separation. The task-specific pre-training is a practical choice that undercuts the universality narrative. These are significant but not fatal problems—the model works well, the hybrid masking insight is valuable, and the efficiency story is real. This is above the ROSE threshold but below the FITS threshold.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>