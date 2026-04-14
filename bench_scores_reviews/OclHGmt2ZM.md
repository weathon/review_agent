## Summary

CMamba is a multivariate time series forecasting model that adapts the Mamba state space model to better handle cross-channel dependencies. It introduces three components: M-Mamba (a modified Mamba with feature-independent A and data-dependent D), GDD-MLP (a lightweight channel mixing module using pooled descriptors to generate data-dependent affine modulation weights), and Channel Mixup (an augmentation strategy that linearly combines channels within a sample to improve generalization of channel-dependent models). The model is evaluated on seven benchmark datasets and the channel modules are shown to transfer to other architectures.

---

## Strengths

- **GDD-MLP and Channel Mixup as transferable modules (Table 4):** The demonstration that these two components improve four architecturally diverse models (iTransformer, PatchTST, RLinear, TimesNet) by an average of ~5% is a concrete and specific result that goes beyond the usual claim of a standalone model. The improvements for CI models (e.g., +17.8% MSE reduction for PatchTST on Electricity) are especially notable.

- **Negligible computational overhead for channel modeling:** Table 5 shows that GDD-MLP adds only 1.35% FLOPs even for the 862-channel Traffic dataset, making the channel-dependent approach practical at scale. This distinguishes the work from heavier attention-based channel mixing like iTransformer.

- **Channel Mixup is a genuinely novel adaptation of the mixup paradigm:** Rather than interpolating across samples from different time windows (which disrupts temporal structure), the method mixes channels within the same sample, preserving shared temporal characteristics while synthesizing virtual training channels. This is a principled, non-obvious design choice backed by the ablation in Table 3 and the training curve analysis in Figure 4.

- **The data-dependence + global receptive field diagnosis is constructive:** The identification that standard MLP channel mixing is position-dependent and locally receptive (unlike self-attention) is a concise and action-guiding characterization. It motivates both GDD-MLP and Channel Mixup in a unified way, and Table 3 provides quantitative support—particularly the striking failure of GDD-MLP alone on Traffic (MSE 0.525 → 0.444 when combined with Channel Mixup), which is consistent with the overfitting/distribution-shift hypothesis.

---

## Weaknesses

### Fatal
None.

### Major

- **No direct comparison to Mamba-based time series forecasting baselines (S-Mamba, Bi-Mamba+, Time-SSM).** The paper explicitly positions itself as improving Mamba for multivariate forecasting and discusses these models in the related work section, yet the main results table (Table 1) contains zero Mamba-based competitors—only Transformer, Linear, and CNN models. The core claim that CMamba is a superior Mamba adaptation is therefore entirely unsubstantiated relative to its direct competition. This is the single most damaging gap for the paper's stated framing.

- **Traffic result in Table 1 contains a likely error that compromises the paper's empirical claims.** Table 1 reports CMamba on Traffic as MSE 0.444 / MAE 0.645 (highlighted as best), while iTransformer shows MSE 0.428 / MAE 0.262. On both metrics, iTransformer clearly outperforms CMamba on Traffic. Furthermore, CMamba's own ablation table (Table 3) reports MAE 0.265 for the full model on Traffic—inconsistent with the 0.645 in Table 1. This is almost certainly a table formatting error in Table 1, but as presented, it makes CMamba's "top 1 in 13/14 settings" claim untrustworthy. The authors must reconcile this discrepancy: if CMamba does not actually win on Traffic, the claim should be corrected.

- **Baseline results largely inherited from iTransformer without unified rerunning.** The paper explicitly states it "reuses most baseline results from iTransformer." For ICLR-level empirical claims, this introduces uncertainty: different implementations, preprocessing, tuning budgets, and normalization can materially affect results, particularly for models like PatchTST and DLinear whose optimal settings may differ from those used in iTransformer's paper. At minimum, the strongest Transformer-based competitors (iTransformer, PatchTST) should be rerun under the paper's exact protocol.

### Minor

- **M-Mamba ablation (Table 2) is too narrow to support architectural conclusions.** The ablation studies the three modifications only on the Weather dataset, with differences between cases ranging from 0.001 to 0.003 MSE—margins indistinguishable from noise given that no standard deviations are reported. These are insufficient to conclude "the convolution operation and the gated z-branch are redundant" in general. At minimum, two to three additional datasets should be covered.

- **GDD-MLP is affine modulation (SE-style), not full cross-channel mixing—but is described as the latter.** Equation 6 implements H'_t = Weight_t ⊙ H_t + Bias_t where Weight and Bias are generated from pooled per-channel descriptors. This is channel recalibration: each channel's features are rescaled and shifted based on that channel's own summary statistics and shared MLP, with no direct channel-to-channel interaction. The paper repeatedly claims GDD-MLP "captures cross-channel dependencies," but that interaction is only implicit through the shared MLP weights and pooled global descriptors. The paper should describe the mechanism more precisely to avoid overclaiming.

- **No controlled ablation isolating the two claimed failure modes of standard MLP.** The paper argues MLP fails due to lacking *both* data-dependence and a global receptive field. However, the ablation only compares (a) plain MLP vs. (b) GDD-MLP, which adds both properties simultaneously. An intermediate ablation—e.g., a data-dependent but patch-local MLP, or a global but static (non-data-dependent) pooling-based MLP—would determine whether both properties are genuinely necessary or whether one dominates.

- **Generalizability experiment (Table 4) covers only 2 of 7 datasets.** Table 4 tests GDD-MLP + Channel Mixup on Electricity and Weather, but Table 3 already shows that these modules behave very differently on Traffic (GDD-MLP alone causes +9.6% MSE degradation before Channel Mixup rescues it). Omitting Traffic and the ETT datasets from the generalizability claim leaves the most challenging case unaddressed.

- **No vanilla Mamba forecasting baseline anywhere in main tables.** The paper argues vanilla Mamba components are unsuitable for multivariate forecasting, but never provides forecasting numbers for a vanilla Mamba applied to the same benchmarks. The central architectural claim lacks a quantitative starting point.

### Tiny

- **σ hyperparameter for Channel Mixup is not reported or ablated.** Algorithm 1 uses λ ~ N(0, σ²), and the choice of σ controls the degree of channel perturbation. Neither the value used nor a sensitivity analysis appears in the main text. This matters because too large σ could meaningfully distort labels.

- **Training objective (loss function) is never stated explicitly.** The paper reports MSE and MAE as metrics but never writes down the training loss. It is presumably MSE, but this should be stated.

- **Notation inconsistency in Section 3.1:** The paper writes "X_{v,v} as the entire sequence of the channel indexed by v," but the intended notation should be X_{:,v} (all time steps, channel v). As written, both subscripts use the same index name, which is ambiguous.

---

## Nice-to-Haves

- **Visualization of GDD-MLP weights as a channel dependency matrix.** For a dataset with interpretable channels (e.g., ETT with physically meaningful quantities like HULL/MULL), visualizing which channels receive high/low modulation weights would validate that GDD-MLP learns semantically meaningful dependencies rather than acting purely as a regularizer.

- **Per-horizon Traffic breakdown.** Given the anomalies in the Traffic results, showing MSE and MAE across the four prediction horizons (96, 192, 336, 720) on Traffic would reveal whether any issues are horizon-specific and build confidence in the reported numbers.

- **Comparison of CMamba vs. iTransformer/PatchTST as look-back length grows (Figure 5).** Figure 5 only shows CMamba's trajectory. Showing that CMamba benefits more from longer look-back than Transformer baselines would validate the claimed long-range modeling advantage of the SSM design.

- **Explicit limitations section** discussing: (a) Channel Mixup's potential unsuitability when channels represent physically incompatible quantities; (b) the observed fragility of GDD-MLP alone on distributional-shift datasets like Traffic; (c) whether M-Mamba's inductive bias (feature-independent A) could fail on datasets with highly heterogeneous channel dynamics.

---

## Removed Points

*These points are flagged for removal; treat them with caution.*

- **"No asymptotic complexity analysis given"** (Harsh Critic): Removed as a standalone weakness. The paper provides FLOPs analysis in Table 5 which is sufficient for an empirical systems paper. Requesting formal Big-O complexity beyond what is standard in this community is a scope-creep demand.

- **"Channel Mixup target mixing may violate domain semantics"** (Harsh Critic): Partially removed. The paper explicitly uses λ ~ N(0, σ²), meaning E[Y'] = Y_i — the original target is preserved in expectation. The concern about negative λ creating unphysical channels is valid but is mitigated by the distribution's properties; this is a novel design with reasonable justification.

- **"Removing convolution may reduce expressive power rather than help"** (Harsh Critic): Removed. The ablation (Table 2, Cases ①–③) does show that removing conv improves or matches performance. The critic's concern about suboptimal tuning applies equally to all ablation studies and is not a specific fault here.

- **"Algorithm 1 replaces rather than augments"** (Harsh Critic): Removed. Replacing the original sample with the augmented one is standard augmentation practice (e.g., standard Mixup replaces both training examples with the mixed one). This is not unusual.

- **"Claim that channels-as-sequences is impractical lacks formal complexity comparison"** (Harsh Critic): Removed as a stand-alone weakness. The paper's strategy is to sidestep the complexity argument entirely by using a different mechanism; a formal complexity comparison to S-Mamba would be nice-to-have, not a core weakness.

- **"MLP-Mixer-style architectures not discussed in related work"** (Harsh Critic): Removed. The paper engages appropriately with the relevant forecasting literature for its claims. Demanding comprehensive cross-domain related work coverage is excessive scope creep.

- **"Requesting statistical significance tests (Diebold-Mariano, t-tests)"** (Review 2): Removed. Single-run or few-run evaluation without formal significance tests is the norm in the MTSF benchmarking community (iTransformer, PatchTST, TimeMixer all follow this convention). The paper does report 3-run means, which is already above the community standard.

---

## Novel Insights

The most genuinely novel observation across the three reviews is the **interaction between GDD-MLP and Channel Mixup as a co-dependent stability mechanism**: GDD-MLP alone severely degrades performance on the Traffic dataset (MSE jumps from 0.479 to 0.525), but when combined with Channel Mixup, performance recovers to 0.444—the best reported. This suggests that data-dependent channel modulation is inherently prone to distribution shift overfitting when applied without regularization, and that channel-within-sample augmentation specifically addresses this failure mode in a way that cross-sample augmentation (vanilla Mixup, shown in Figure 4) cannot. This is a non-obvious finding: the two modules are not independently beneficial but rather mutually enabling. A deeper theoretical analysis of *why* this co-dependence arises—why data-dependent channel gating is unstable without channel-space augmentation—would be a valuable direction for understanding the mechanism rather than merely its empirical effect.

---

## Suggestions

1. **Add S-Mamba and Bi-Mamba+ as baselines in Table 1.** These are direct competitors and their absence is the clearest vulnerability of the paper's claims. Even if the comparison is unfavorable on some datasets, transparency strengthens the paper.

2. **Resolve and explain the Traffic MAE discrepancy between Table 1 (0.645) and Table 3 (0.265).** If this is a formatting error in the PDF, correct it; if it reflects a real difference in experimental conditions, explain why. Update the "top 1 in 13/14 settings" count accordingly.

3. **Extend Table 2 (M-Mamba ablation) to at least 2–3 additional datasets** (e.g., ETTh1, Electricity, Traffic) with the same prediction horizons, to establish whether the architectural conclusions generalize beyond Weather.

4. **Rephrase GDD-MLP's mechanism more precisely** in the abstract and methodology: it performs data-dependent affine modulation (channel recalibration) rather than explicit cross-channel mixing. This is not a lesser contribution—SE-like recalibration is a well-validated paradigm—but overstating it invites justified skepticism.

5. **Extend Table 4 (generalizability) to all 7 datasets**, particularly Traffic, given that GDD-MLP alone degrades there. Showing that the combined module still helps (or understanding when it does not) would substantially strengthen the versatility claim.

6. **Report the σ value used for Channel Mixup** and include a sensitivity analysis (e.g., σ ∈ {0.1, 0.3, 0.5, 1.0}) in the appendix.