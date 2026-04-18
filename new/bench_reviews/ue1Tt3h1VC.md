Now I have enough calibration context. Let me synthesize the final review.

Let me carefully verify the key critical claims from the harsh reviewer against the actual paper text:

1. **CLUB / mutual information minimization issue**: The paper states in Eq. (7) that CLUB is an upper bound on MI and says "By optimizing such an objective, we can achieve information disentangling." Then in Eq. (8), they define $\mathcal{L}_{\text{club}}$ and in Eq. (10) add $\lambda \mathcal{L}_{\text{club}}$ to the loss. The question is: is minimizing CLUB sufficient to minimize MI? The paper says CLUB upper-bounds MI, so minimizing CLUB would push down the upper bound, which does push down MI (at least as long as the variational network $\mathcal{Q}_\theta$ approximates the true conditional well enough). This is actually a valid application of CLUB - it's established practice to minimize the CLUB upper bound as a proxy for minimizing MI. The claim from the harsh reviewer that "CLUB is normally used by maximizing this upper bound to estimate MI" is technically true in some contexts (e.g., when you want to *estimate* MI, you first learn a good variational approximation and then evaluate the bound), but for *minimizing* MI, minimizing the CLUB upper bound is a standard approach from the original CLUB paper. However, the harsh reviewer's point about the conflation of optimization objectives for θ vs. the expert parameters is well-taken. The paper does say that $\mathcal{Q}_\theta$ is separately optimized with $\mathcal{L}_{\text{exid}}$ and the main model is optimized with $\lambda\mathcal{L}_{\text{club}}$, but the details are insufficiently specified. Let me check more carefully.

The paper says in §4.4: "Besides, during each round of training, $\mathcal{Q}_{\theta,m}$ is also optimized with the loss $\mathcal{L}_{\text{exid}}$, separated from the MMKGC model." So they do separate the two optimizations. This is actually consistent with the standard CLUB approach. Still, it's true that the paper could be clearer about this.

2. **Relation-guided gating mechanism**: Looking at Eq. (2), the gating function $G_i(V_{m,i}^e, r) = \text{softmax}((\mathcal{U}_m(V_{m,i}^e) + \delta_{m,i})/\sigma(\varepsilon_r))$. The relation $r$ only enters through the temperature $\varepsilon_r$. This is indeed a very weak form of relation conditioning. The claim of "relation-guided expert selection" is overstated compared to what the mechanism actually does. This is a valid criticism.

Let me now write the consolidated review.

## Summary

The paper proposes MoMoK, a Mixture of Modality Knowledge Experts framework for multi-modal knowledge graph completion (MMKGC). The key idea is to use relation-guided modality experts (ReMoKE) that adaptively combine multiple intra-modality embeddings based on relational context, a multi-modal joint decision module (MuJoD) that integrates predictions across modalities, and an expert information disentanglement module (ExID) based on CLUB mutual information minimization to encourage specialization among experts within each modality. Experiments on four public MMKG benchmarks demonstrate state-of-the-art results and robustness in noisy, missing, and sparse settings.

## Strengths

- **Well-motivated problem**: The observation that different relational contexts require different modality information (Figure 1) is intuitive and addresses a genuine limitation of existing MMKGC methods that perform static entity-level multi-modal fusion.

- **Strong empirical results**: MoMoK achieves consistent improvements across four benchmarks against 20 baselines, with particularly notable gains on DB15K (21.1% relative MRR improvement) and KVC16K (10.6%). The Hit@1 improvements are substantial, suggesting genuine gains in ranking accuracy.

- **Comprehensive evaluation beyond standard benchmarks**: The robustness experiments under noisy, missing modalities, and link-sparse settings (Figure 3) go beyond standard evaluation and provide practical insights.

- **Modular and interpretable design**: The decomposition into ReMoKE, MuJoD, and ExID allows clear ablation analysis (Table 2), and the attention visualizations (Figure 5) provide some interpretability into how different relations leverage different experts and modalities.

## Weaknesses

### Fatal
None.

### Major

- **The "relation-guided" gating mechanism is significantly weaker than claimed**: As defined in Eq. (2), the relation $r$ enters the gating function only through a per-relation scalar temperature $\varepsilon_r$, which controls the sharpness of the softmax over experts but does not change *which* experts are preferred. The routing decision between experts is driven entirely by $\mathcal{U}_m(V_{m,i}^e)$ and random noise $\delta_{m,i}$, independent of the relation. This means the model does not implement relation-conditioned expert selection — it implements a relation-agnostic MoE with relation-dependent temperature. The paper consistently claims that "different sections of varied modality information emphasize their respective significance when making predictions based on different relationships" (§1) and that experts "specialize in different relational contexts" (§3), but the architecture does not support this. While the downstream score functions per modality (Eq. 4) do use relation embeddings $r_m$, the expert routing itself is not relation-guided. This gap between the core narrative and the actual mechanism undermines a central claim of the paper.

- **The ExID module's theoretical justification is incomplete and potentially misleading**: The paper claims that minimizing the CLUB-based loss forces experts to "minimize the mutual information between decisions" (§4.3). While minimizing the CLUB upper bound is a valid approach to minimizing MI, the paper conflates two optimization problems: (1) updating $\mathcal{Q}_\theta$ to tightly approximate the true conditional (via $\mathcal{L}_{\text{exid}}$), and (2) updating the experts to reduce MI (via $\lambda\mathcal{L}_{\text{club}}$). The description in §4.4 that $\mathcal{Q}_\theta$ is "also optimized with $\mathcal{L}_{\text{exid}}$, separated from the MMKGC model" is vague about the specifics (e.g., alternation schedule, convergence criteria). More importantly, the ablation (Table 2, row 2.5) shows removing ExID drops MRR by only ~0.9 on MKG-W (35.89→34.99) and ~1.15 on DB15K (39.57→38.42), which is relatively modest compared to removing joint training (35.89→32.73, a 3.16 drop). This suggests ExID contributes marginally, and the paper's heavy theoretical framing around mutual information is disproportionate to its empirical impact. Furthermore, minimizing mutual information between expert outputs enforces statistical independence, not alignment with distinct relational contexts — the connection between "independence" and "relation-specialization" is assumed but not established.

- **Ablation raises questions about the true source of gains**: The ablation study (Table 2) shows that removing joint training (row 2.4) causes the largest performance drop (MRR from 35.89 to 32.73 on MKG-W, and 39.57 to 37.62 on DB15K). This suggests that the primary benefit comes from training separate scoring functions per modality and combining them, rather than from the proposed MoE routing or ExID mechanisms. The paper does not compare against a simpler multi-task baseline (e.g., independent single-expert per modality + late fusion), which would isolate whether the MoE and MI machinery actually help over this simpler alternative. This makes it difficult to attribute the gains to the novel contributions versus the multi-task training setup.

### Minor

- **No standard deviations or statistical significance tests**: All results in Table 1 appear to be from single runs. For margins as small as the MKG-W improvement over MMRNS (35.89 vs 35.03, ~0.86 MRR points) and the MKG-Y improvement over AdaMF (37.91 vs 38.06 for AdaMF Hit@1), statistical significance is unclear. While the larger gains on DB15K and KVC16K are likely robust, reporting variance would strengthen the claims.

- **Ambiguity between Eq. (1) and Eq. (5)**: §3 describes negative sampling for training, but Eq. (5) defines a full softmax over all entities. It is unclear whether the actual implementation uses sampled negatives or full softmax, and this affects comparability with baselines using different training regimes.

- **Missing modality handling**: The paper does not discuss how MoMoK handles entities that lack image or text modality information — a common scenario in real-world MMKGs. The robustness experiments (Figure 3) partially address this by randomly removing modalities, but a systematic evaluation of how often this occurs and how the model handles it at training time would be useful.

### Trivial
- The notation in Eq. (3) uses both $m$ and $n$ in a way that makes it unclear whether a summation over modalities is intended in the denominator; this seems to be a minor notational issue but hinders reproducibility.
- In Table 1, the row labeled "MRNRS" should likely be "MMRNS" (consistent with the reference to Xu et al., 2022).

## Nice-to-Haves

- Comparing MoMoK against a simpler multi-task baseline (single expert per modality + late fusion) would isolate the contribution of the MoE and ExID components more clearly, as joint training alone appears to be the dominant factor.
- Testing K=1 within the full MoMoK framework would directly validate the title claim that "multiple heads are better than one"; the current K sweep in Figure 4 only varies K with all other components active, and K=1 is not reported.
- Providing quantitative analysis of expert specialization (e.g., expert activation patterns clustered by relation type) rather than only qualitative donut-chart visualizations.

## Removed Points

- **Claim that CLUB should be maximized, not minimized**: The harsh reviewer argues that CLUB is "normally used by maximizing this upper bound to estimate mutual information," implying minimization is incorrect. This mischaracterizes the intent: when the goal is to *minimize* MI (as in disentanglement), minimizing the CLUB upper bound is a standard and valid approach from the original CLUB paper (Cheng et al., 2020). While the optimization details could be clearer, the general approach of minimizing CLUB to reduce MI is sound.
- **Baseline comparability concerns about model size/parameter budget**: The reviewer demands identical or comparable base encoders and parameter budgets across all methods. This is an unreasonable standard for a paper with 20 baselines. The paper uses VGG and BERT as feature extractors, which are the same as or comparable to what other MMKGC methods use. This is not a major fairness concern.
- **VGG and BERT as "dated"**: Using established pre-trained encoders is standard practice in MMKGC literature and is not a weakness.
- **Missing important baseline comparisons**: This is flagged for removal since we cannot confirm existence of unspecified "recent" methods not cited by the paper.
- **Efficiency analysis only on one dataset with limited baselines**: The efficiency comparison in Table 3, while limited, is adequate for showing that the method doesn't catastrophically increase computational costs. This is a minor concern, not a major one.
- **Robustness experiments only on DB15K**: While extending to all datasets would be better, the DB15K evaluation is the most challenging and largest dataset, and results there are representative.

## Novel Insights

The paper reveals an interesting empirical finding that is somewhat at odds with its own narrative: the dominant source of performance improvement appears to be multi-task training of independent scoring functions per modality (Table 2, row 2.4), rather than the relation-guided MoE routing or the mutual-information-based disentanglement. This suggests that the key insight for MMKGC may be less about "which experts to route to" and more about "ensuring each modality has its own dedicated scoring pathway" — a simpler architectural principle that doesn't require the MoE/ExID apparatus. The paper does not confront this implication.

## Suggestions

- **Correct or substantially tone down the "relation-guided" claims**: Either introduce genuine relation-conditioned routing (e.g., a relation embedding in the gating function) or revise the narrative to accurately describe the temperature-based mechanism as "relation-aware sharpening" rather than "relation-guided expert selection."
- **Add a simple multi-task ablation**: Include a "K=1, no ExID, no relation temperature" variant that still uses per-modality scoring functions with joint training. This would quantify how much of the gain comes from the multi-task setup alone versus the MoE and ExID components.
- **Clarify the training objective and negative sampling**: Specify whether Eq. (5) uses full softmax or sampled negatives, and reconcile this with the negative sampling described in Eq. (1).
- **Report standard deviations**: Run experiments with multiple random seeds and report mean ± std to strengthen statistical claims, especially for the smaller improvements on MKG-W and MKG-Y.

## Evaluation on Axes

- **Originality**: Moderate. The application of MoE with intra-modality experts to MMKGC is novel, but MoE, Tucker scoring, and CLUB are established techniques. The key novelty claim (relation-guided expert routing) is weakened by the architecture's actual design.
- **Importance of research question**: Good. MMKGC is an important and active area, and the problem of context-dependent modality utilization is well-motivated.
- **Claims support**: Partial. The empirical gains are strong, but the mechanistic claims (relation-guided expert specialization, MI-based disentanglement) are not well-supported by either the architecture or the ablations.
- **Soundness of experiments**: Adequate but with gaps. Strong on breadth of benchmarks and baselines; weak on isolating the mechanism and statistical robustness.
- **Clarity of writing**: Moderate. The paper is generally readable but has notational ambiguities (Eq. 3) and overclaims relative to the architecture.
- **Value to research community**: Moderate. The multi-task per-modality training insight has value, but the specific MoE/ExID mechanism may be less impactful given the weak relation-guiding and marginal ExID contribution.

## Score and Decision Calibration

I compared against several calibration papers:
- **M4oE (NJxCpMt0sf)**: Multi-modal MoE with MI loss in medical imaging — similar pattern of MoE + MI disentanglement. Accepted as poster with scores 6/6/6/5.
- **GraphMETRO (QQ5eVDIMu4)**: MoE for graph distribution shift — similar pattern of MoE with theoretical claims that reviewers found incomplete. Rejected with scores 6/6/3/5.
- **SD-HC (4HAXypZfsm)**: MI-based disentanglement with theoretical gaps between claims and implementation — rejected with scores 6/3/5/6.
- **Self-Distilled Disentanglement (L0b0vryZRX)**: MI-based with incorrect definitions — rejected with scores 6/6/3/3.

This paper has stronger empirical results than GraphMETRO and SD-HC, but shares similar issues of overstated theoretical claims about disentanglement and questionable mechanistic interpretation. The "relation-guided" gating being merely a temperature scaler is a substantive architectural concern, not just a presentational issue. The ablation shows the novel components contribute modestly compared to the simpler multi-task baseline implication.

The paper is below the acceptance threshold primarily because: (1) the core novel mechanism (relation-guided expert routing) is not actually implemented as described, (2) the theoretical contribution (ExID via MI minimization) is both incompletely specified and empirically marginal, and (3) the main empirical gains likely stem from the simpler multi-task training setup without the MoE machinery. The empirical results are genuinely strong, but they don't clearly validate the claimed contributions.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>