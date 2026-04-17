## Summary

This paper introduces Continuous Online Action Detection (COAD), a task formulation where models not only detect actions online from streaming video but also continuously learn in a single causal pass without data storage. To support this task, the authors curate Ego-OAD, a large-scale egocentric OAD benchmark derived from Ego4D (87 classes, 22,991 instances, 263h), and propose combining orthogonal gradient projection, non-uniform loss weighting, and state continuity with an RNN-based detection head. Experiments on Ego-OAD and EPIC-KITCHENS show improvements in both in-stream adaptation and out-of-stream generalization over pretrained-only and naive fine-tuning baselines.

## Strengths

- **Well-motivated and timely problem**: The gap between offline-trained OAD models and real deployment on wearable devices is real. The COAD formulation—causal, single-pass, no-replay learning from streaming video—addresses a genuinely important challenge for egocentric vision systems.

- **Useful benchmark contribution**: Ego-OAD provides a large-scale, diverse egocentric OAD benchmark with 87 fine-grained multi-label classes, 36% overlapping actions, and long-form videos (~472s average). This fills a clear gap in egocentric OAD resources and the three-way evaluation split (pretraining/in-stream/out-of-stream) cleanly separates adaptation from generalization.

- **Clear ablation structure**: Table 3 provides a systematic ablation of each COAD component, and Figures 3–4 analyze the stride/learning-rate trade-off and performance evolution over training. The finding that COAD steadily approaches the IID training upper bound (Figure 4) is compelling.

- **Demonstrates value of egocentric pretraining**: Consistent improvements from egocentric (EgoVLP) vs. exocentric (Kinetics) features across both backbones (Tables 1 and 4) provide actionable, well-supported guidance.

## Weaknesses

### Major

- **Limited baselines undermine claims of COAD's effectiveness**: The only comparisons are "Pretrained Only" and "w/o COAD" (identical architecture without the three proposed strategies). There are no comparisons with established continual learning methods (EWC, SI, LwF, experience replay with bounded buffers), parameter-efficient fine-tuning approaches (e.g., adapters), or even other OAD architectures adapted to this setting. Without these, it is impossible to determine whether the gains come from the specific COAD strategies or from the simple fact of performing any online fine-tuning. A small replay buffer (e.g., storing 1–2 past windows) is trivially feasible on-device and could be a stronger baseline. This significantly weakens the empirical case.

- **Headline improvement claims conflate data access with method contribution**: The abstract claims "up to 20% in top-5 accuracy" for adaptation and "up to 7%" for generalization. The 20% figure comes from the exocentric-pretrained in-stream comparison (Pretrained Only 57.5 → COAD 80.0), where much of the gain is attributable to having *any* in-stream training data. COAD-specific improvements over "w/o COAD" are much smaller (e.g., +4.2 Top-5 Recall on ego out-of-stream, and actually −2.2 mAP on ego in-stream). The paper should clearly distinguish improvements from data access vs. from the proposed strategies, and report representative rather than cherry-picked numbers.

- **Limited algorithmic novelty**: All three technical components—orthogonal gradient projection (Han et al., 2025), non-uniform loss (An et al., 2023), and state continuity (standard RNN inference)—are adopted directly from prior work. The contribution is their combination and application to the COAD setting. While this combination is shown to be effective, the paper overstates novelty by framing COAD as a "new task formulation" when it is more accurately a new training and evaluation protocol for an existing task (OAD). The paper would benefit from deeper analysis of *why* this particular combination works, rather than just showing *that* it improves performance.

- **No error bars or multiple runs**: All results are single-run scalar values. Given that online SGD on temporally correlated streams is known to be high-variance, and given that some reported improvements are small (e.g., +0.5 mAP differences in Table 2 action category), the reliability of these results cannot be assessed without variance estimates.

### Minor

- **On-device feasibility claims are unsubstantiated**: The introduction repeatedly frames COAD for resource-constrained wearable devices, yet no memory, latency, FLOPs, or energy measurements are reported. The backbone (TimeSformer) is frozen with precomputed features, so only the GRU head is trained online. This is a reasonable first step, but the "on-device" narrative is stronger than the experimental setup supports.

- **Catastrophic forgetting is not explicitly analyzed**: COAD trades some in-stream adaptation for better out-of-stream generalization (e.g., ego in-stream mAP drops from 39.0 without COAD to 36.8 with COAD). This is a key characteristic of the method, but no explicit analysis of *what* is forgotten (which classes, which segments) is provided. Per-class performance change analysis would clarify the adaptation–forgetting trade-off.

- **EPIC-KITCHENS in-stream results are weak and under-analyzed**: Both COAD and w/o COAD sometimes degrade in-stream performance relative to Pretrained Only (e.g., verb mAP drops from 11.4 to 10.7). The paper attributes this to "fine-grained nature of actions" but provides no supporting analysis (e.g., per-class breakdowns, correlation between class frequency and adaptation gain).

- **Ego-OAD dataset specification could be stronger**: The manual grouping of semantically similar free-form labels into 87 classes is described at a high level, but no inter-annotator agreement, grouping validation protocol, or analysis of the impact of grouping granularity on evaluation is provided. The class imbalance is acknowledged but not analyzed in terms of its effect on results.

### Trivial

- The paper has a few typos (e.g., "Countinuous" in the contributions list).

## Nice-to-Haves

- Comparison with standard continual learning baselines (EWC, replay, LwF, adapters) to properly contextualize COAD's contribution.
- Per-class analysis of adaptation vs. forgetting during continuous training.
- Computational cost profiling (training time per step, peak memory) to substantiate the on-device deployment narrative.
- Investigating whether lightweight backbone fine-tuning (e.g., LoRA) during COAD could close the gap to the IID upper bound shown in Figure 4.

## Removed Points

These points were flagged for removal or significant weakening:

- **"COAD task formulation is not substantively new" (Harsh Critic, structural)**: While the harsh critic argues that COAD is merely existing continuous learning applied to OAD, the paper's contribution in defining the evaluation protocol, data splits, and the specific adaptation/generalization trade-off framework does constitute a meaningful problem formulation contribution, even if calling it an entirely "new task" overstates novelty. Kept as a softened version under algorithmic novelty.

- **"Pretrained Only is an unfair baseline" (Harsh Critic)**: The paper explicitly identifies Pretrained Only as a lower bound ("serves as a lower bound and reflects the model's initial performance under limited supervision"). Since this comparison framework is transparent and the lower bound favors the baseline's weakness (not the paper's method), this is not an unfair comparison per the guidelines. Removed.

- **"Missing related works" (Spark, others)**: Per guidelines, I do not confirm existence of uncited works. Removed.

- **"Uniform loss naming confusion" (Harsh Critic, section-by-section)**: This is a minor naming inconsistency, not a substantive issue. Removed as a formatting/style nitpick.

- **"Reproducibility concerns about implementation details" (Harsh Critic)**: Per guidelines, complaints about undisclosed hyperparameters and implementation details are removed as nitpicks about reproducibility. Removed.

- **"Results on exocentric benchmarks (THUMOS, TVSeries)" (Spark)**: This asks for scope expansion beyond the paper's stated egocentric focus. Removed as scope creep.

- **"Partial backbone fine-tuning during COAD" (Spark)**: This is a nice-to-have suggestion for future work, not a core flaw. Moved to Nice-to-Haves.

## Novel Insights

The paper reveals an interesting tension between adaptation and generalization in continuous video learning: COAD's components (especially orthogonal gradients and non-uniform loss) consistently improve out-of-stream generalization, sometimes at the cost of in-stream adaptation. This is a known phenomenon in continual learning (stability–plasticity trade-off), but seeing it manifest so clearly in the streaming video context—with the specific mechanism being gradient decorrelation rather than parameter regularization—suggests that continuous video learning may require different mitigation strategies than traditional class-incremental scenarios. The finding that supervision as sparse as one label per ~68 seconds can still produce meaningful adaptation (Figure 3 discussion) is also noteworthy and underexplored.

## Suggestions

- Report representative improvement numbers (e.g., the ego-pretrained out-of-stream gains of ~4.7% mAP / 6.9% Top-5) rather than cherry-picked extremes, and clearly separate gains from data access vs. method contributions.
- Add at least 2–3 standard continual learning baselines (EWC, LwF, bounded replay) to contextualize whether COAD's specific strategies are necessary or whether simpler approaches suffice.
- Provide per-class analysis to characterize the adaptation–forgetting trade-off, showing which action classes benefit and which degrade during continuous training.
- Run 3+ seeds and report standard deviations for key results to establish statistical reliability.

## Score and Decision

**Calibration**: I compared against papers with similar patterns:
- Large Scale Video Continual Learning (7L2bpe7lfm, avg ~4.5): incremental novelty, limited baselines, similar task formulation contribution.
- Bayesian KD for OAD (8iojQVLLWb, avg ~4): same domain, limited novelty, similar evaluation gaps.
- VidEgoThink (Z5nqeTH24j, avg ~4.4): egocentric benchmark paper with evaluation gaps.
- POET (WGLu9Mv8mn, avg ~3.5): continual learning with weak baselines and overclaiming.
- Anomalies are Streaming (Y7jJN0VQ4y, avg ~5.7): better-executed continual learning paper.

This paper has a genuinely useful benchmark and a well-motivated problem, which places it above papers like POET. However, the limited baselines, overclaimed novelty ("new task formulation" for what is essentially a training/evaluation protocol), cherry-picked headline numbers, and absence of error bars place it below the better-executed papers in the ~5.5–6 range. The method novelty is incremental (combining three existing techniques). The paper is stronger on the benchmark/problem-definition side than on the method side. I rate it below the anomalies streaming paper (~5.7) but above the purely incremental OAD papers (~4).

MY FINAL SCORE: 4.5
MY FINAL DECISION: Reject