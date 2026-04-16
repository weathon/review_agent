Now let me look at the referenced human review files for calibration.Now I have enough information to write the consolidated review. Let me verify one last important point from the paper.Now I have everything I need. Let me write the final consolidated review.

---

## Summary

MoMoK is a multi-modal knowledge graph completion (MMKGC) framework that integrates three components: (1) Relation-guided Modality Knowledge Experts (ReMoKE), which constructs K per-modality expert networks gated by a relation-conditioned temperature; (2) Multi-modal Joint Decision (MuJoD), which trains per-modality Tucker-based scoring functions and combines them in a multi-task fashion; (3) Expert Information Disentanglement (ExID), which applies the CLUB mutual-information upper-bound estimator to discourage redundancy across experts within each modality. Experiments on four MMKG benchmarks against 20 baselines show strong overall results, with particularly large improvements on DB15K and more modest gains on MKG-W/MKG-Y.

---

## Strengths

- **Comprehensive empirical evaluation**: 20 baselines across four public MMKG benchmarks, with additional robustness stress tests, ablations, efficiency comparison, and interpretability analysis. Few MMKGC papers match this breadth.
- **Large and credible DB15K improvement**: MoMoK achieves ~21.1% relative MRR and ~33.8% relative Hit@1 improvement over the next-best method. The margin is large enough to rule out lucky runs and clearly demonstrates that the approach has genuine merit on this benchmark.
- **All ablated components contribute positively**: Table 2 demonstrates that each of relational temperature ε_r, tunable noise δ, adaptive fusion, joint training, and ExID individually contribute; no component is "free." The multi-task ensemble ("joint training") is the single biggest driver, which is at least clearly documented.
- **Code and data released**: Reproducibility is supported.
- **Well-structured motivation**: Figure 1 and the introduction compellingly illustrate that different relational contexts call for different modality information, providing an intuitive grounding for the MoE design.

---

## Weaknesses

### Fatal
*None identified. The approach is internally sound and the DB15K results represent genuine progress.*

### Major

1. **Relation-guidance is thinner than claimed** — The paper's central pitch is *relation-guided* expert specialization, but examining Eq. (2) closely reveals that relation information enters only through a scalar learnable temperature ε_r for each relation, applied as a softmax temperature. This is a very weak form of relational conditioning: it modulates routing sharpness per relation but does not steer *which* expert is preferred for a given (entity, relation) pair. There is no direct conditioning on the relation embedding in the gating logits. The gap between the marketing ("relation-guided modality knowledge experts") and the actual mechanism deserves an honest acknowledgment and, ideally, a comparison against relation-embedding-conditioned gating.

2. **Uniform summation at inference contradicts the core motivation** — The final inference rule is S(h,r,t) = Σ_m S_m(h,r,t) (Eq. 4.4), treating all modalities equally regardless of relation. This directly contradicts the paper's opening argument (Figure 1, §1) that *different relations require different modality emphasis*. Having invested effort in relation-aware intra-modality routing through ReMoKE, the method then discards this at the inter-modality aggregation step. The neutral reviewer independently raised this. The authors should at minimum explain why uniform summation is preferable over relation-conditioned inter-modality weighting, or acknowledge it as a limitation.

3. **SOTA claim is partially overstated: MoMoK does not lead on MKG-Y MRR** — Table 1 shows MoMoK's MKG-Y MRR = 37.91 vs. AdaMF's 38.06. The "Improvements" row correctly records "-" for this metric but the abstract, introduction, and conclusion all assert broad SOTA. The claim should be scoped more carefully.

4. **Robustness experiments are too narrow to support broad claims** — §5.4 reports robustness under noise, missing modalities, and link sparsity (Figure 3), but exclusively on DB15K with only three baselines (AdaMF, QBE, TBKGC). The abstract and §1 assert robustness "under complex scenarios" in general; the experiment merely shows MoMoK is more stable than those three baselines on one dataset. Replicating on at least one additional benchmark (e.g., MKG-W) with the top-performing baseline (MMRNS) included would substantially strengthen this claim.

5. **Ablations do not isolate expert specialization from parameter scaling and ensembling** — Table 2 removes components but never compares K=1 expert versus K=3 experts under matched parameter budgets, nor relation-guided gating (ε_r temperature) versus fully ablated relation-agnostic MoE. The largest performance drop in ablation is from removing "joint training" (multi-task modality scoring), which is a straightforward ensemble effect. The paper does not engage with the possibility that most gains come from ensembling rather than from expert specialization per se. Figure 4 left (K study) shows K=1 is already competitive, which is a concerning signal that the MoE architecture's incremental contribution over a strong single-expert baseline is limited. This warrants honest discussion.

6. **No direct measurement of disentanglement** — ExID's stated purpose is to minimize mutual information between expert outputs. Ablation (2.5) shows removing ExID hurts, but this is consistent with ExID acting as generic regularization rather than true disentanglement. There is no measurement of MI between expert outputs before vs. after training, no clustering of expert activations by relation type, and the variational Q_θ network is trained alternately but its convergence and MI estimates are never reported. The claim that ExID achieves semantic disentanglement is therefore not directly evidenced.

### Minor

- **Inconsistent metric reporting across datasets**: MKG-W and MKG-Y report only MRR and Hit@1, while DB15K and KVC16K also include Hit@3 and Hit@10. No justification is given for the omission.
- **Shared W_attn in MuJoD fusion** (Eq. 3): The adaptive inter-modality fusion weight uses a single learnable vector W_attn shared across all entities, modalities, and relations. This cannot capture entity- or relation-specific modality preferences, which is precisely what the paper argues is important. A straightforward relation-conditioned replacement would be internally consistent.
- **Disparity in improvements across benchmarks is unexplained**: DB15K sees +21% relative MRR while MKG-W sees +2.5% and MKG-Y is not SOTA at all. No analysis is provided of why DB15K benefits so much more. Understanding this is important for knowing when MoMoK applies.

### Trivial

- Table 3 efficiency comparison does not specify the dataset, making it impossible to fully interpret. "9.8s" vs "7.5s" per epoch depends entirely on dataset size and batching configuration.

---

## Nice-to-Haves

- A t-SNE/UMAP visualization of expert activations colored by relation type, and/or per-relation expert assignment entropy statistics, would provide genuine quantitative evidence for the claimed expert specialization (beyond the three anecdotal donut charts in Figure 5).
- A per-relation performance breakdown comparing MoMoK vs. top baselines would show which relation types most benefit from relational context, directly validating the central hypothesis.
- Reporting trained Q_θ mutual-information estimates before vs. after training (or as training progresses) would substantiate the ExID disentanglement claim rather than relying solely on the downstream performance gap from ablation (2.5).
- Extending robustness experiments (Figure 3) to MKG-W or KVC16K with MMRNS included would make the robustness claim generalizable.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

**Harsh Critic — "Full-softmax vs. negative sampling inconsistency"**: The critic claims "full-softmax and sampled negatives are different training regimes." Looking at Eq. (5), the training objective is unambiguously a full-softmax CE loss over all entities. The paper labels it "the negative sampling process mentioned before," which is confusing terminology but §3 explains that NS is generally used in KGC while MoMoK's actual Eq. (5) computes scores against the entire entity set. This is a terminology clarity issue at most, not a methodological flaw. The paper uses Tucker-style full-softmax consistently with TuckER baselines.

**Harsh Critic — comparing reproduced vs. borrowed baselines**: The critic says it's unclear which results come from MMRNS and which are reproduced, but "some of the baseline results refer to MMRNS (Xu et al., 2022)" is standard practice in MMKGC papers where the cited benchmark paper provides a controlled reproduction environment. This is not unusual for the field and does not invalidate the comparison.

**Human Finder W2 — Limited comparison with stronger multi-modal baselines**: This criticism requests comparison with unnamed recent methods. Per hard rules, we cannot introduce references to works not in the paper without external verification that they exist and postdate the baselines. Removed.

**Human Finder W4 — Assumption of shared expert structure across modalities**: This is speculation about a design choice (using the same K=3 for all modalities). The paper does include a parameter sensitivity study (Figure 4). This is a theoretical conjecture without evidence that different K per modality would help. Removed as speculative.

**Harsh Critic — Table 3 is "too limited to support a strong efficiency claim"**: The paper does not make a very strong efficiency claim; it says "within reasonable limits." The table is indeed underdocumented, but this is a minor presentation issue already captured under Trivial weaknesses above rather than a separate removal.

---

## Novel Insights

The paper's most genuinely interesting empirical finding—largely underdiscussed in the text—is the dramatic dataset-dependent variation in improvement magnitude (+21% on DB15K vs. +2.5% on MKG-W vs. no improvement on MKG-Y MRR). This pattern is itself informative about the conditions under which relation-guided modality experts provide benefit, and connecting this to dataset properties (relation cardinality, structural density, modality coverage) would be a scientifically valuable contribution that none of the reviewers fully pursued. The ablation finding that "joint training" is the single dominant contributor (Table 2, row 2.4) also raises an underexplored question: how much of the DB15K gain is attributable to the multi-task Tucker ensemble (a simple and replicable trick) versus the relation-guided routing specifically? Disentangling this—even approximately—would substantially sharpen the paper's contribution claims.

---

## Suggestions

1. **Eq. (2): Augment relation-conditioning** beyond a scalar temperature. Add r_m embedding to gating logits directly so that routing becomes (entity features, relation embedding) → expert weights. This makes "relation-guided" accurate.
2. **Replace uniform inference summation** with an attention-weighted combination using the learned modality attention weights from MuJoD, conditioned on the relation. This trivial change closes the motivation–design gap at inference time.
3. **Analyze what makes DB15K uniquely sensitive** to the proposed method (number of relations, relation diversity, image vs. text coverage), and include one sentence explaining this in §5.3.
4. **Report MI estimates from Q_θ** at the start and end of training (at least for one dataset) to directly validate that ExID achieves disentanglement, not just regularization.
5. **Add K=1 vs K=3 parameter-matched comparison** to Table 2 ablations, and be transparent about whether the improvement justifies the added complexity.
6. **Scope the SOTA claim precisely**: report that MoMoK is SOTA on 11 of 12 metrics tested (not MKG-Y MRR), which is accurate and still impressive.

---

## Score and Decision

**Calibration anchors used:**
- *bIHyMpzeuI* (Multi-modal Sparse MoE, Reject, scores 3/5/8/5, avg ~5.3): Heuristic design, limited novelty, adequate experiments. MoMoK is stronger empirically and more coherent methodologically.
- *3n4RY25UWP* (Disentanglement accept poster, scores 6/6/8/5, avg ~6.3): Stronger theoretical grounding, similar empirical scope. MoMoK has larger performance gains but weaker theoretical validation of its disentanglement claim.
- *Pu3c0209cx* (MoE Poster Accept, scores 6/8/8/6, avg ~7.0): Well-theorized MoE routing paper with principled design. MoMoK is more empirically driven and its routing mechanism is less principled.
- *lBrLDC7qXF* (KGC, Withdrawn, avg ~3.6): Much weaker; missing key baselines and has fundamental errors. MoMoK is substantially stronger.

MoMoK sits above bIHyMpzeuI (avg 5.3) due to its more comprehensive evaluation and genuinely impressive DB15K results. It falls short of Pu3c0209cx (avg 7.0) due to the gap between the claimed relation-guidance mechanism and its actual implementation (scalar temperature only), the inference inconsistency, and the limited ExID validation. The paper lands in the borderline-accept zone, closer to the 3n4RY25UWP-level paper but with methodological overclaiming that needs to be addressed.

**Evaluation summary:**
- *Originality*: Moderate — competent adaptation of MoE + Tucker + CLUB to MMKGC; no single component is novel, but the combination is well-targeted.
- *Importance*: Moderate-high — MMKGC is an active area and the DB15K result is a real contribution.
- *Claim support*: Weak-to-moderate — DB15K claims are well-supported; SOTA claim is slightly overreached; ExID and relation-guidance claims are under-evidenced.
- *Experimental soundness*: Moderate — comprehensive in breadth but narrow for robustness (1 dataset) and missing key ablations.
- *Clarity*: Good — well-structured and readable.
- *Community value*: Moderate — the method provides a reasonable new baseline for MMKGC research.

**Score: 6.0** — Borderline accept. The paper makes a genuine empirical contribution, especially on DB15K, and the framework is coherent. However, the central "relation-guided" claim is mechanistically weaker than presented, the inference design contradicts the core motivation, and ExID remains insufficiently validated. These issues should be addressable in revision.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>