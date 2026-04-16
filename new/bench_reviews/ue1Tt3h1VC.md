Now I have enough context to write the final review. Let me carefully verify the key reviewer claims against the paper before synthesizing.

## Summary

MoMoK proposes a Mixture of Modality Knowledge Experts framework for multi-modal knowledge graph completion. It introduces three components: (1) relation-guided modality knowledge experts (ReMoKE) that learn K relation-conditioned embeddings per modality, (2) multi-modal joint decision (MuJoD) that trains a separate TuckER scoring function for each modality plus a joint "modality" and sums their scores at inference, and (3) expert information disentanglement (ExID) that minimizes mutual information between intra-modality experts via the CLUB upper bound. MoMoK achieves state-of-the-art results on four MMKG benchmarks, with particularly large improvements on DB15K and KVC16K.

## Strengths

- **Clear and relevant motivation.** The observation that different relations require different modality perspectives (Figure 1) is intuitive and well-illustrated. The proposed architecture directly addresses this by making entity representations relation-conditional.

- **Strong empirical performance.** MoMoK achieves substantial improvements on DB15K (+21% MRR, +34% Hit@1) and KVC16K (+6-10% on most metrics) over prior baselines. Even on the more competitive MKG-W and MKG-Y, it achieves consistent gains. The comparison against 20 baselines is comprehensive.

- **Robustness analysis.** The experiments under noise, missing modality, and link sparsity conditions (Figure 3) show that MoMoK degrades more gracefully than baselines, which is practically important.

- **Well-designed ablation study.** Table 2 systematically isolates the contribution of each modality and each design choice, providing clear evidence that the joint training and multimodal fusion contribute meaningfully.

## Weaknesses

### Major:

- **The core empirical gains are confounded with multi-model ensembling.** The MuJoD module trains four TuckER scoring functions (structure, image, text, joint) and sums their scores at inference. Ablation (2.4) "w/o joint training" drops MRR from 35.89→32.73 on MKG-W and 39.57→37.62 on DB15K—the single largest ablation effect. The paper acknowledges this: "joint training has the most profound effect on the final performance, as it trains a separate MMKGC model for each modality, resulting in decision fusion." Yet the paper frames its contribution as the MoE routing and disentanglement architecture. Without a simple baseline that trains 4 TuckER models (one per modality + joint) with standard per-modality embeddings (no ReMoKE, no ExID) and fuses their scores, it is impossible to attribute the SOTA gains to the proposed innovations rather than to the ensembling effect. This does not invalidate the results but significantly undermines the claimed novelty contribution.

- **The ExID disentanglement claim is overstated relative to evidence.** The paper's second contribution claims that CLUB-based MI minimization "forces different experts to specialize in different relational contexts" and presents "detailed theoretical analysis." However: (a) CLUB provides an *upper bound* on MI; minimizing it decorrelates expert outputs but does not guarantee semantic disentanglement or relation-specific specialization—this inference is asserted without proof. (b) The ablation shows ExID contributes ~1 MRR point (35.89→34.99 on MKG-W, 39.57→38.42 on DB15K), which is modest relative to the theoretical framing. (c) There is no direct measurement of MI reduction, expert specialization metrics, or clustering analysis quantifying whether experts genuinely capture distinct relational perspectives. (d) Figure 5's attention weight visualizations do not isolate ExID's contribution (no comparison to a model without CLUB). The gap between the conceptual claim and empirical support is substantial.

- **Unexplained performance variance across datasets.** MoMoK improves MRR by ~21% on DB15K but only ~0-3% on MKG-Y. The paper does not discuss this discrepancy. Understanding what dataset properties (graph density, relation types, modality quality) drive the differential gains would be important for practical applicability and for understanding when the method is most beneficial.

### Minor:

- **No variance or significance testing across multiple runs.** Results are reported as single numbers. Some improvements are small (e.g., MKG-W MRR: 35.89 vs. 35.03 = +0.86 absolute). Without standard deviations or confidence intervals, it is difficult to assess whether these margins are statistically meaningful.

- **Misleading efficiency claim.** Section 5.6 claims "high efficiency and less GPU memory usage," but Table 3 shows MoMoK uses more memory (5900MB) and time (9.8s/epoch) than MMKRL (4504MB, 7.5s) and some other baselines. The comparison should accurately state that MoMoK achieves better performance with moderate additional computational cost.

- **ExID implementation is underspecified.** The variational network Q_θ is trained alternatively with the main model via L_exid (Eq. 9), but the paper does not specify training frequency, optimizer settings, or gradient flow between Q_θ and the expert networks. If Q_θ is poorly trained, the CLUB estimate can be unreliable, undermining the theoretical motivation.

- **Relation-aware temperature ε_r is introduced but not analyzed.** Section 4.1 presents the learnable temperature as a key design choice, but no analysis of learned temperature values per relation is provided. If all temperatures converge to similar values, this design is vacuous.

### Trivial:
- The "Improvements" row in Table 1 leaves MKG-Y MRR as "-" and has inconsistent percentage calculations.

## Nice-to-Haves

- A simple ensemble baseline (K=1 per modality, 4 TuckER models fused, no ReMoKE/ExID) to isolate the MoE and disentanglement contributions from ensembling.
- Quantitative measurement of expert specialization (e.g., entropy of gating weights per relation, MI estimation before/after ExID) to support the disentanglement claim.
- Experiments with more than two non-structure modalities (e.g., video) to validate the generality claim.
- Experiments with alternative score functions (e.g., RotatE, ComplEx) to show the proposed modules are not TuckER-specific.

## Removed Points

These points are flagged to be removed; treat them with caution.

- *Harsh Critic Point 4 (negative sampling regime vs. MANS/MMRNS):* The paper uses standard cross-entropy loss (Eq. 5) with uniform negative sampling (Eq. 1), which is clearly described. MANS and MMRNS have their own specialized negative sampling strategies that are part of their methods. Comparing under different sampling regimes is standard practice in KGC; the comparison is between complete methods, not ablated components. This is not a confound.

- *Neutral Reviewer #4 (limited scope of modalities):* The paper works with standard, widely-used MMKG benchmarks that only have image and text modalities. Demanding evaluation on additional modalities is scope creep beyond the paper's stated scope, and the architecture generalizes in principle.

- *Claim that feature extractors (VGG, BERT) are dated:* This is a style nitpick. The choice of feature extractor is standard in the MMKG community and consistent with baselines. Changing extractors would not test the proposed method's contribution.

- *Request for K analysis on multiple datasets:* While desirable, the paper provides K analysis on at least one dataset. This is a minor limitation, not a methodological flaw.

- *Demand for per-epoch training time normalization by performance:* Table 3 provides raw time/memory figures, which is the standard reporting approach. Normalization to "time-to-target-MRR" is non-standard.

## Novel Insights

The key insight from this paper—that modality utility in MMKGs varies by relational context and should be modeled adaptively—is well-motivated. However, the review analysis reveals that the paper's strongest empirical gains appear to come from a well-known ensembling strategy (training separate models per modality and fusing predictions) rather than from the novel architectural components (ReMoKE, ExID). The ablation starkly reveals this: removing joint training causes the largest drop, while removing the novel ExID contributes only ~1 MRR point. Future work in this space should carefully decompose ensembling effects from architectural innovations.

## Suggestions

1. **Add a simple multi-TuckER ensemble baseline** (4 models, 1 per modality + joint, no ReMoKE or ExID) to properly attribute gains to the proposed components vs. ensembling.
2. **Measure and report expert specialization quantitatively** (e.g., entropy of gating distributions per relation, MI estimates between expert pairs before/after training with ExID).
3. **Report standard deviations across multiple random seeds** for all main results, especially where margins are thin.
4. **Discuss the DB15K anomaly**—explain why gains are much larger on this dataset than others.

## Score and Decision

**Calibration:**
- M4oE (multimodal MoE, medical): scores 5-6-6-6, Accept Poster — similar architectural MoE contribution, overclaiming concerns, but solid results
- SM4 (sparse MoE multimodal): scores 3-5-8-5, Reject — limited novelty, ad-hoc design
- DMI (multimodal info-theoretic disentanglement): scores 5-6-6-5, Reject — marginal improvements, limited experiments
- MIDR (disentangled multimodal representations): scores 3-3-5-6, Reject — limited novelty, implementation concerns

MoMoK has stronger empirical results than DMI and MIDR, and a clearer architectural story than SM4. It is comparable to M4oE in conceptual novelty. Compared to M4oE (accepted at ~5.75), MoMoK has more substantial weaknesses: (1) the ensembling confound is significant and undermines the claimed novelty attribution, and (2) the ExID theoretical contribution is overstated. However, it also has stronger benchmark results (4 datasets, 20 baselines, large improvements). The paper demonstrates a working system that advances SOTA, but the attribution of gains to the novel components is muddied by the ensembling. I place this slightly below M4oE.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>