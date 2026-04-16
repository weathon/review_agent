## Summary

The paper proposes MoMoK, a Mixture-of-Experts framework for multi-modal knowledge graph completion that introduces (1) relation-guided modality knowledge experts (ReMoKE) within each modality, (2) a multi-modal joint decision module (MuJoD) that treats a fused joint embedding as an additional modality, and (3) expert information disentanglement (ExID) via CLUB-based mutual information minimization to encourage diverse expert specialization. Experiments on four benchmarks show state-of-the-art results, particularly on DB15K and KVC16K, along with robustness experiments under noise, missing modalities, and link sparsity.

## Strengths

- **Clear and well-motivated problem identification**: The observation that different relational contexts require different modality emphasis (Fig. 1) is intuitive and important, and the paper proposes a concrete architectural response to this observation.
- **Strong empirical performance**: Consistent and often substantial improvements over 20 baselines across 4 benchmarks. The +21.1% relative MRR and +33.8% relative Hit@1 on DB15K are notable margins.
- **Comprehensive evaluation**: Beyond the main leaderboard (Table 1), the paper includes robustness experiments under noise/missing/sparsity scenarios (Fig. 3), ablation studies (Table 2), parameter sensitivity (Fig. 4), efficiency analysis (Table 3), and interpretability case studies (Table 4, Fig. 5).
- **Interpretability via attention visualization**: Figure 5 provides plausible evidence that different relations activate different experts and modalities, lending qualitative support to the design intent.

## Weaknesses

### Major

- **Gap between the "relation-guided" narrative and the actual gating mechanism**: The paper frames ReMoKE as producing "relation-guided" expert specialization, but Eq. (2) shows the gating logits $\mathcal{U}_m(V_{m,i}^e)$ depend only on the entity embedding, not on relation content. The relation $r$ enters only as a temperature parameter $\varepsilon_r$ via $\sigma(\varepsilon_r)$, which modulates how peaked or flat the softmax over experts is — it does not determine *which* expert is preferred. Consequently, an entity's relative expert preferences are largely fixed across relations, merely sharpened or smoothed. This significantly weakens the claim that experts learn "relation-aware" embeddings. The MI minimization in ExID also does not condition on relations (Eq. 8 operates over entities and experts within a modality), so it encourages statistical independence between experts but does not directly enforce the claimed relation-specific specialization. The paper's conceptual framing is stronger than its mechanism delivers.

- **Lack of a fair, tightly controlled baseline isolating the MoE/ExID contribution**: The scoring function for every modality is Tucker decomposition, and the overall training objective is a sum of per-modality cross-entropy losses. Several baselines in Table 1 use older or simpler scoring functions (TransE, DistMult, etc.), making it unclear how much of MoMoK's gain comes from the MoE/ExID design versus simply having a stronger backbone with multi-task training across modalities. The ablation study (Table 2) only removes components *within* the MoMoK family — there is no comparison against a vanilla "Tucker + multimodal fusion" baseline that shares the same backbone, training setup, and negative sampling but omits the expert/CLUB machinery. The substantial drop when removing joint training (setting 2.4: −3.16 MRR on MKG-W) suggests a significant portion of the gain may come from the multi-task training paradigm rather than the expert architecture itself.

- **Modest empirical contribution of ExID relative to its conceptual prominence**: ExID is presented as one of the paper's key innovations, but removing it causes only ~0.9 MRR reduction on MKG-W and ~1.15 on DB15K (Table 2, setting 2.5). Furthermore, no robustness experiments compare MoMoK with and without ExID under noise/missing modality scenarios — precisely the setting where disentangled experts should help most. No alternative regularizer baselines (e.g., orthogonality penalties, dropout-based diversity) are compared to isolate the benefit of CLUB specifically.

### Minor

- **No statistical significance testing or variance reporting**: All results appear to be single runs. For some improvements (e.g., +0.86 MRR on MKG-W vs. MMRNS, which is ~2.5% relative), confidence intervals would help establish reliability. This is standard practice in the field but its absence reduces confidence in marginal improvements.

- **Misleading efficiency claim (Section 5.6)**: The text states the approach achieves "less GPU memory usage" while maintaining SOTA, but Table 3 shows MoMoK uses 5900MB — higher than MMKRL (4504MB) and OTKGE (2540MB). Only MMRNS (25582MB) uses substantially more. The efficiency claim should be revised.

- **Redundancy in score aggregation**: The joint embedding $\tilde{e}_{joint,r}$ is a weighted sum of individual modality embeddings (Eq. 3), yet the final inference score sums over all modality scores *plus* the joint modality score ($\sum_{m \in \mathcal{M} \cup \{J\}} \mathcal{S}_m$). Since $S_J$ is derived from embeddings that are themselves weighted combinations of the individual modality embeddings, some redundancy or double-counting may occur. The paper does not analyze the correlation between $S_J$ and $\sum S_m$ or whether the joint modality alone suffices.

- **Incomplete theoretical justification for ExID**: The variational approximation $\mathcal{Q}_{\theta,m}$ assumes a Gaussian conditional distribution, but no justification or empirical validation is provided for whether this assumption holds for the learned expert embeddings. The optimization procedure for alternating between $\mathcal{L}_{club}$ and $\mathcal{L}_{exid}$ is described informally ("alternatively optimized") without pseudocode or a clear schedule. This matters for the correctness of CLUB as an MI upper bound.

### Trivial

- The number of experts $K=3$ is chosen empirically (Fig. 4) but no principled discussion connects $K$ to dataset properties (e.g., number of relation clusters).
- Robustness experiments (Fig. 3) are conducted only on DB15K with only 3 baselines, limiting generalizability claims.
- The "Improvements" row in Table 1 mixes different baselines across columns without clarifying which method serves as the reference for each improvement percentage.

## Nice-to-Haves

- A simple "Tucker + per-modality encoder + attention fusion" baseline (no experts, no CLUB) would cleanly isolate the contribution of each architectural innovation and considerably strengthen the empirical argument.
- Sparse (Top-K) routing instead of dense softmax gating could improve expert specialization and computational efficiency, and a comparison would strengthen the architectural analysis.
- Quantitative disentanglement metrics (e.g., CKA similarity between expert outputs, routing entropy statistics, MI estimates before/after ExID) would directly verify the claim that ExID produces genuinely specialized experts.
- Running robustness experiments on all four datasets rather than just DB15K, and including an "MoMoK without ExID" variant, would strengthen the robustness claims.

## Removed Points

- **"Baselines predating contemporary architectures invalidate comparisons"**: Removed because several recent baselines are included (AdaMF, VISTA, QEB, etc.). The concern is more specifically about the lack of a matched Tucker baseline, which is kept above.

- **"Sparse routing would be better than dense routing"**: Removed as a core weakness — this is a design choice, and dense routing is legitimate. Moved to Nice-to-Haves.

- **"Missing related works on relation-conditioned fusion in multimodal KGC"**: Removed per instructions — I cannot confirm the existence of specific uncited works.

- **"No confidence intervals is fatal"**: Downgraded from harsh critic's framing. In this field, single-run reporting is unfortunately common. Kept as a minor point rather than a fatal flaw.

- **"The case study is anecdotal"**: While true, this is a standard limitation of case studies. Kept as context for nice-to-haves about quantitative disentanglement metrics.

- **"Future work mentions LLMs which is speculative"**: Removed — this is a standard conclusion remark and not a weakness.

## Novel Insights

The key architectural insight—that treating the joint fused modality as an additional "senior" modality alongside the original ones and scoring each independently—creates a hierarchical decision structure that is conceptually appealing. However, the empirical analysis reveals that this hierarchical structure's contribution is entangled with the simpler mechanism of multi-task training across modalities (ablation 2.4 shows the largest drop), suggesting that the independent per-modality training paradigm may be more important than the expert routing itself. The disconnect between the "relation-guided" narrative and the actual mechanism (relation as temperature, not content) is a notable gap: the model effectively learns entity-specific expert preferences that are modulated in sharpness by relations, rather than learning genuinely relation-specific routing.

## Suggestions

- Add a vanilla "Tucker + multimodal" baseline (single encoder per modality, same training objective, no experts, no CLUB) to isolate how much the MoE architecture contributes beyond backbone strength and multi-task training.
- Replace or augment the temperature-only relation conditioning in the gating network with a more expressive relation-dependent routing mechanism (e.g., concatenate relation embedding to expert logits) to genuinely realize the "relation-guided" framing.
- Report mutual information estimates or expert dissimilarity metrics before and after ExID training to directly verify that CLUB produces meaningful disentanglement.
- Clarify the optimization schedule for the variational network and include pseudocode for the full training procedure.

## Score and Decision

**Calibration**: I compared against:
- M4oE (NJxCpMt0sf.md, scores 5-6, Accept Poster): Similar MoE multimodal framework with MI-based disentanglement. MoMoK has stronger empirical results but similar conceptual weaknesses around expert specialization justification.
- Mowst (wYvuY60SdD.md, scores 5-8, Accept Poster): MoE on graphs with scalability concerns. MoMoK is more comprehensive in evaluation.
- CLIP-MoE (e4em5klSEw.md, scores 3-6, Reject): MoE with novelty/designed concerns, weaker empirical setup. MoMoK is stronger empirically.
- SM4 (bIHyMpzeuI.md, scores 3-8, Reject): Ad-hoc MoE design concerns. MoMoK has better motivation and results.

MoMoK has genuinely strong empirical results and a well-motivated problem, but the gap between the "relation-guided expert specialization" narrative and the actual mechanism (temperature-only gating, relation-agnostic CLUB), combined with the lack of a fair controlled baseline, prevents the methodological claims from being fully convincing. The paper is above the rejection threshold due to its empirical strength and comprehensive evaluation, but below the threshold for a clear accept due to these conceptual and experimental gaps. It sits in the borderline-accept range.

MY FINAL SCORE: 5.5
MY FINAL DECISION: <orange>Accept</orange>