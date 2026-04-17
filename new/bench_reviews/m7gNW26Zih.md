Now I have enough context. Let me compile the final review.</think>## Summary

This paper presents a system for language-based audio retrieval on CLOTHO, combining a dual-encoder architecture with three enhancements: (1) soft-label distillation from an ensemble of pretrained audio-text retrieval models, (2) LLM-driven caption augmentation (back-translation and LLM mix for mixed audio), and (3) cluster-guided auxiliary classification using BERTopic-derived labels. The best single model achieves mAP@16 of 46.62 on the CLOTHO development test split, and a weighted ensemble of multiple backbone/SID combinations achieves 48.83.

## Strengths

- **Strong empirical performance with competitive results.** The system achieves the highest published numbers on the CLOTHO audio retrieval benchmark, apparently derived from or aligned with a top DCASE 2024 Task 8 system. The improvement from baseline to distillation (SID 1 → SID 2) is substantial, e.g., PaSST mAP@16 goes from 42.08 to 46.62, demonstrating the practical effectiveness of soft-label distillation.

- **Clear and well-structured pipeline description.** The three-stage training setup (pretraining on CLOTHO+AudioCaps+WavCaps, finetuning with distillation+augmentation, re-finetuning with cluster guidance) is described with sufficient detail for replication, including learning rate schedules, batch sizes, and temperature settings.

- **Systematic multi-backbone evaluation.** Testing across three audio encoders (PaSST, EAT, BEATs) and systematically constructing SIDs 1–5 provides a useful resource for analyzing trade-offs, even if the analysis itself falls short (see weaknesses).

## Weaknesses

### Major:

- **The claim that cluster-guided auxiliary classification improves retrieval is contradicted by the paper's own Table 2.** The abstract states that "ablations indicate consistent improvements under high correspondence ambiguity," but Table 2 shows that adding cluster guidance (SID 4 vs. SID 3 using finetuned clusters, SID 5 vs. SID 3 using BERTopic clusters) *hurts* performance for most configurations: EAT drops from 46.05 to 45.34 under both cluster variants; BEATs drops from 44.66 to 44.58/43.88; PaSST drops from 46.41 to 46.39 or gains a marginal 0.09 to 46.50. Even compared to SID 2 (no augmentation, no cluster), cluster variants often perform worse. The paper includes no analysis of "high correspondence ambiguity" examples — no partitioning, no stratified metrics, no evidence whatsoever for this claim. This is a direct contradiction between the empirical data and a core stated contribution of the paper.

- **Confounded ablations prevent attribution of gains to individual components.** Table 1 shows cumulative additions (SID 1: everything on; SID 2: +distillation; SIDs 3–5: + augmentation and +cluster), but there is no baseline row for contrastive-only training, no "augmentation only" condition, no "cluster only" condition, and no factorial 2³ design. The marginal gains from SID 2→3 (adding augmentation) are tiny (PaSST: 46.62→46.41, actually a *drop*), and cannot be disentangled from the simultaneous change in training schedule (additional 20 epochs of finetuning). Since the paper claims three distinct contributions, the failure to isolate each one's effect is a significant methodological gap.

- **The central narrative about "robustness to non-binary correspondences" is not operationalized or measured.** The paper motivates all three contributions as addressing non-binary audio-text correspondences (where a caption could validly match multiple audio clips). However, evaluation uses standard mAP@k and R@k metrics that assume binary relevance. No metric reflecting graded relevance, no analysis of ambiguous vs. unambiguous queries, and no partition by correspondence ambiguity is provided. Without such analysis, the claimed mechanism — that these methods improve *robustness to non-binary correspondences* — is unsupported. The data shows these methods improve *overall mAP*, but *why* they improve it is not demonstrated.

- **Missing comparison with existing published methods on CLOTHO.** The paper presents only internal ablations with no comparison to prior published results on the same benchmark. Methods like m-LTM (mini-batch Learning-to-Match), CLAP-based retrieval baselines, or the DCASE 2024 Task 8 baseline are not included. Since this work explicitly builds on Primus et al. (2024), comparing against that specific system is essential for establishing incremental contribution.

### Minor:

- **Ensemble weighting obscures true methodological contribution.** The headline 48.83 mAP@16 comes from a grid-searched weighted combination of 12 models across 4 SIDs (Table 3). More transparent would be a comparison with simpler ensembling (e.g., uniform averaging of the best backbone per SID). The final evaluation score of 0.421 mAP@16 is reported without context — if this is 42.1%, it represents a notable drop from the 48.83 dev-test result, and this generalization gap is not discussed.

- **Teacher ensemble specification is incomplete.** Section 2.2 and 3.4 mention "three audio models" as teachers for distillation, but do not specify whether these are the same PaSST/EAT/BEATs models used as students, whether they are trained with the same data/augmentations, or how similarity matrices from heterogeneous backbones are normalized before averaging. This matters because if teachers and students share training data and architecture families, the distillation signal may be dominated by self-consistency rather than capturing meaningful soft correspondences.

- **Proprietary LLM (GPT-4o) limits reproducibility of augmentation.** While the paper flags this as a limitation, no prompts, language selection criteria for back-translation, or post-processing details are provided, making the claimed "reproducible pipeline" for LLM augmentation not actually reproducible.

- **No variance or statistical significance reported.** All results appear to be single runs. Given the tiny differences between many systems (e.g., PaSST SID 3 vs. 5: 46.41 vs. 46.50), it is impossible to assess whether observed differences are meaningful.

## Nice-to-Haves

- Factorial ablation study (2³ design: {±distillation, ±augmentation, ±cluster}) on at least one backbone to cleanly isolate each contribution.
- Stratified evaluation on ambiguous vs. unambiguous audio-caption pairs to substantiate the "non-binary correspondence robustness" narrative.
- Qualitative analysis of clusters (number, quality, representative captions) and analysis of when cluster guidance helps vs. hurts.
- Comparison with uniform-averaging ensemble to assess the added value of grid-search weighted combinations.

## Removed Points

- **Missing references to prior work (e.g., CLAP, m-LTM):** While relevant baselines are missing from experiments, I should not fabricate or confirm the existence/non-existence of specific uncited references. The lack of external baselines in the experiments is already captured in Major Weakness 4.
- **Demands for evaluation on a second dataset:** The paper frames its contribution within the DCASE 2024 Task 8 challenge, so single-dataset evaluation on CLOTHO is within scope. Generalization would be nice but is not a core flaw.
- **Criticism that the three-stage training is overly complex without justification:** The paper describes the stages clearly. Requiring justification for every design choice (e.g., why not joint training) is scope creep given the engineering nature of the contribution.
- **Nitpick on equation numbering errors (e.g., Eq. 5 vs. "(ℋ)"):** These are parsing/formatting artifacts, not substantive issues.

## Novel Insights

The key insight in this paper — that soft labels from an ensemble of retrieval models can improve audio-text retrieval by capturing non-binary correspondences in the training data — is sound in principle. However, the empirical evidence reveals an ironic finding: the cluster-based auxiliary classification, which the paper positions as addressing correspondence ambiguity, actually *harms* most backbone configurations, suggesting that imposing hard cluster assignments may conflict with the very soft correspondence structure the other components are designed to exploit. This tension between soft labels (distillation) and hard auxiliary labels (cluster classification) is an important experimental observation embedded in the data but unacknowledged by the authors.

## Suggestions

1. **Remove or substantially revise the "consistent improvements under high correspondence ambiguity" claim.** The current Table 2 data directly contradicts it. Either add a proper analysis partitioning by ambiguity level, or acknowledge that cluster guidance yields mixed results.

2. **Add a plain contrastive baseline** (no distillation, no augmentation, no clustering) trained with the same multi-stage schedule to establish a clear lower bound and enable clean attribution of gains.

3. **Report results with standard deviation across multiple runs.** The differences between SIDs 2–5 are within 1 mAP point for most configurations, making statistical significance critical for claiming any improvement.

4. **Clarify the evaluation gap.** Explain whether 0.421 on the evaluation set represents a meaningful generalization concern and provide context (e.g., comparison with DCASE 2024 leaderboard scores).

## Score and Decision

**Calibration:** This paper is most comparable to competition-system papers with engineering contributions but limited novelty and confounded ablations. The microRTS competition-winning paper (scores 3/5/5/6, rejected) similarly combined existing techniques with weak ablations and was essentially an engineering description. The Generalized Category Discovery with Hierarchical Label Smoothing paper (scores 3/3/3/3, rejected) shared the pattern of soft-label + clustering claims with insufficient novelty and overclaiming. The GIFT paper on soft label distillation (scores 6/6/8/6, accepted poster) had clearer novelty and cleaner ablations. The m-LTM audio-text retrieval paper (scores 8/6/6, accepted poster) had genuine methodological novelty (optimal transport formulation).

This paper falls clearly below the m-LTM and GIFT papers in novelty (all three contributions adopted from prior work) and below them in empirical rigor (confounded ablations, overclaimed results). It is similar to the microRTS competition paper in being primarily an engineering description with a serious overclaiming problem (the cluster guidance claim is contradicted by the paper's own data). While the engineering achievement is solid, the paper overclaims contribution 3 and does not cleanly substantiate contribution 2 — the distillation contribution (contribution 1) is well-supported and clearly effective but is directly adopted from Primus et al. (2024).

Score: **4.0**

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>