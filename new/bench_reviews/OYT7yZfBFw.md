Now let me do calibration searches to anchor the score.Now I have enough calibration data. Let me synthesize the final review.

**Calibration summary:**
- MOTOR (NialiwI2V6): avg 7.5, spotlight — large-scale EHR pre-training, 55M patients, 19 downstream tasks, cross-site validation, clean comparisons. Much stronger experimental setup than TrajGPT.
- TimelyGPT (2sCcTMWPc2): avg 5.5, reject — TrajGPT's direct predecessor; similar scope but simpler; reviewers flagged limited datasets and limited theoretical analysis.
- Stable Neural SDEs for irregular TS (4VIgNuQ1pY): avg 6.67, spotlight — more rigorous ODE-based modeling with stability theory.
- Large pre-trained TS models (KJ1w6MzVZw): avg 3.8, reject — pre-training for TS without strong baselines or fair comparisons.
- ZkEsEFFUyo (pre-training TS CloudOps): avg 4.33, reject — pre-training study without strong empirical validation.

TrajGPT sits between TimelyGPT (5.5, reject) and MOTOR (7.5, accept/spotlight). It has more contributions than TimelyGPT (data-dependent decay, ODE interpretation, time-specific inference, more tasks), but doesn't reach MOTOR's scale (55M vs 489K patients) or experimental rigor (19 tasks vs 5, cross-domain vs in-domain only). The pre-training asymmetry concern and thin ablation evidence for the main contribution keep it from being a strong accept.

My estimate: ~5.0–5.5. The paper makes genuine technical contributions (time-specific inference clearly helps by ~6 pts, large dataset, multi-task evaluation) but the architectural contribution (data-dependent decay) is poorly differentiated from the pre-training advantage, the zero-shot terminology is misleading, and the ablation is only done on one task. Still, it's clearly above the TimelyGPT paper it extends, since it adds meaningful innovations.

I'll score this at **5.0** — borderline below acceptance.

---

## Summary
TrajGPT is a Transformer model for irregularly-sampled health time-series that introduces a Selective Recurrent Attention (SRA) mechanism with data-dependent decay, interprets the recurrence as a discretized ODE via zero-order-hold (ZOH) to enable time-specific inference at arbitrary future timesteps, and is pre-trained on large-scale EHR data. The model is evaluated on diagnosis forecasting, drug usage prediction, phenotype classification, and sepsis prediction across two datasets (PopHR: 489K patients; eICU: 139K patients). The primary practical contribution — time-specific inference — consistently outperforms autoregressive inference by 4–6 percentage points across datasets and tasks.

---

## Strengths

- **Time-specific inference produces consistent, quantified performance gains.** Across both datasets and all recall thresholds, time-specific inference outperforms autoregressive inference: +6.2% at K=10 on PopHR (71.7% vs 65.5%), +3.7% at K=10 on eICU (57.8% vs 54.1%), and Table 3 confirms it accounts for the largest single-component drop in the ablation. This is the paper's most concrete and well-supported contribution.

- **Large-scale pre-training corpus for healthcare.** PopHR covers 489K patients over 17 years (drawn from 1.3M individuals), and eICU covers 139K admissions. Evaluated at this scale, TrajGPT distinguishes itself from prior irregular time-series work limited to small ICU datasets and validates the pre-training paradigm for this domain.

- **Interpretable trajectory analysis (Fig. 4).** The ability to extrapolate disease risk trajectories over a patient's lifetime, identify age windows of rapid risk growth, and surface associated phenotypes (e.g., chronic IHD + hypothyroidism preceding diabetes onset at 59–62) constitutes a clinically meaningful use case that is absent from the prior irregular-time-series literature.

- **Multi-task zero-shot evaluation across two independent datasets.** TrajGPT achieves top zero-shot AUPRC across all zero-shot tasks: 67.2% insulin prediction, 72.8% CHF, 45.1% sepsis — outperforming all baselines in the zero-label setting. Even granting that "zero-shot" here means no task-specific labels (not true out-of-distribution generalization), this is a concrete demonstration of representation quality.

- **Dual recurrent/parallel formulation of SRA.** The equivalence between Eq. 2 (recurrent, O(1) inference) and Eq. 3 (parallel, O(N) training) is cleanly established and practically useful, enabling efficient training without sacrificing inference-time efficiency.

---

## Weaknesses

### Fatal
None.

### Major

- **Pre-training asymmetry confounds the architectural comparison with specialized baselines.** TrajGPT is pre-trained on 489K (PopHR) / 139K (eICU) patients before evaluation. Transformer baselines receive 20 epochs of masking-based pre-training (Section 4.4). However, the specialized irregular-TS models (mTAND, GRU-D, RAINDROP, SeFT, ODE-RNN, HeTVAE, MGP-TCN) "were trained from scratch" with no access to the same unlabeled corpus. When TrajGPT's zero-shot or fine-tuning results are compared against these models' fine-tuning results, the comparison includes both architectural and data-access differences. The paper acknowledges this asymmetry only indirectly ("these methods lack a representation learning paradigm"), but this caveat is not prominently flagged in the results section. Notably, mTAND without any pre-training already reaches 83.7% Recall@15 on PopHR forecasting (vs. TrajGPT's 84.1%) and outperforms TrajGPT on CHF fine-tuning (85.4% vs 83.9%), and outperforms TrajGPT on eICU sepsis fine-tuning (52.5% vs 51.3%). The paper cannot claim architectural superiority over these models without controlling for this data imbalance. This matters because the paper's central comparative claim rests partly on these numbers.

- **Evidence for data-dependent decay is thin and potentially within noise.** The primary architectural novelty over TimelyGPT/BiTimelyGPT is the data-dependent γ_n. In the ablation (Table 3), removing decay gating reduces Recall@10 by only 1.4% (71.7 → 70.3). The ablation does not report standard errors for the ablation rows; the full model's SE in Table 1 is ±2.6, making the 1.4% gap potentially within noise. More importantly, this ablation is only reported on one task (forecasting) on one dataset (PopHR) at one threshold (K=10), with no equivalent for classification or eICU. Removing RoPE causes a larger drop (3.9%). The evidence that data-dependent decay is the key differentiator — as opposed to the inference procedure or RoPE — is insufficient given this single-condition ablation.

### Minor

- **"Zero-shot" terminology is imprecise.** The paper repeatedly uses "zero-shot" to mean "inference without task-specific fine-tuning labels," but TrajGPT has been pre-trained on the *same patient population* as the test set. Strictly, zero-shot generalization typically implies evaluation on a distribution not seen during training. The paper's usage is consistent with a growing convention in healthcare AI, but the claim "TrajGPT enables zero-shot forecasting without requiring fine-tuning" (Section 4.2) and its framing as a key contribution should be scoped more precisely to avoid overclaiming. Out-of-distribution zero-shot evaluation (e.g., train on PopHR, evaluate on eICU) would provide a much stronger demonstration and is explicitly left to future work.

- **Time-specific inference requires a known target timestep.** Section 3.2 states: "TrajGPT utilizes both the target timestep t_{n'} and the last observation." For future medical visits, the exact future timestamp may not be known in advance. The paper does not address how it handles cases where the forecast horizon is specified only in terms of a time delta (e.g., "within 6 months"), nor does it discuss the sensitivity of results to timestamp misspecification. This is a meaningful practical gap given the clinical deployment framing.

- **The "?" in Table 3 is unexplained.** The row "TrajGPT (without Pre-training)" contains a literal "?" in the auto-regressive inference column. This either indicates an omitted experiment or an unreported failure. While likely a minor omission, it should be either filled in or explicitly noted as "not run / not applicable."

- **Cherry-picked case studies.** The trajectory analysis in Section 5.3 reports patient-level Recall@10 of 90.1% and 84.7% — substantially above the population average of 71.7%. No justification is given for why these patients were selected, and no distributional analysis of per-patient recall is presented to demonstrate that the qualitative behavior is typical rather than exceptional.

### Trivial
None warranting mention beyond what is listed above.

---

## Nice-to-Haves

- Pre-train mTAND or ODE-RNN on the same unlabeled corpus and re-run comparisons. This single experiment would substantially clarify whether TrajGPT's gains come from its architecture or from having a pre-training paradigm at all.
- An ablation isolating data-independent decay (TimelyGPT-style γ) + time-specific inference vs. TrajGPT's full data-dependent decay + time-specific inference would cleanly separate the two main contributions.
- Wall-clock timing comparisons against ContiFormer and standard attention variants to empirically support the O(N)/O(1) efficiency claims.
- An empirical analysis of learned γ values across chronic vs. acute condition tokens to substantiate the mechanistic claim in Section 3.1 (currently asserted without evidence).

---

## Removed Points
*These points are flagged for removal; treat them with caution.*

- **"Fig. ??" broken cross-reference (Section 5.1):** Flagged by harsh reviewer. The "Fig. ??" reference points to a figure visualizing head-specific decay vectors. Given that the parser strips all appendix content, this is a parser artifact — the figure and its cross-reference exist in the original submission. **Removed per the rule on parser artifacts.**

- **ODE derivation running "backwards":** Harsh reviewer claims the paper "starts with the recurrence and reverse-engineers an ODE." This is standard practice in the SSM/S4 literature (Gu et al. 2022) and the paper cites this work explicitly. Deriving the continuous-time ODE from the discrete recurrence via ZOH is not methodologically wrong; it is how ZOH discretization connections are standardly made. The paper says Appendix C provides the proof. **Removed as a mischaracterization of standard methodology.**

- **"Zero-shot" = "standard self-supervised pre-training" (framing it as fundamentally invalid):** The harsh reviewer calls this a structural flaw, but "zero-shot" in the healthcare AI community has come to mean "no task-specific labels required" — a usage consistent with GPT-style generative models. The paper does not falsely claim it generalizes to unseen distributions; the concern is valid as a terminological precision issue but not fatal. **Downgraded from fatal to minor.**

- **Pretraining asymmetry "invalidates headline comparisons":** The harsh critic frames this as fatal. However, the paper explicitly positions pre-training *as a capability* — i.e., the point is that TrajGPT *can* be pre-trained, while specialized models *cannot*. The asymmetry is partly the point. The issue is legitimate as a caveat that limits architectural superiority claims, not as a full invalidation. **Downgraded from fatal to major.**

- **Claims about missing appendix proofs, absent appendix figures:** Removed per rules — parser strips appendix content.

---

## Novel Insights

The clearest novel insight beyond the paper's own stated contributions is the interplay between the time-specific inference mechanism and the ODE framing: the ablation shows this inference procedure is responsible for ~6 percentage points of gain — substantially more than the architectural novelty of data-dependent decay (~1.4%). This suggests that for healthcare trajectory forecasting, the *inference strategy* (directly targeting a future timestep using ODE state propagation) may matter more than the specific attention mechanism used to build the state. If this generalizes, it argues for separating the modeling objective (building a continuous-time state) from the architectural novelty (how that state is built), and prioritizing inference-time flexibility in future EHR foundation model designs.

---

## Suggestions

1. **Control for pre-training data access in baselines.** Either give mTAND and ODE-RNN access to the same unlabeled corpus (even if only via a simple masking task adapted to their architecture), or clearly relabel Table 1 comparisons as "with vs. without pre-training." The current table conflates two questions — architecture vs. training paradigm — that are both interesting but should be answered separately.

2. **Expand the ablation.** Repeat Table 3 ablations on classification tasks and on eICU, and include standard errors. A single recall metric on one dataset is insufficient to characterize the contribution of each component. Add a row for "data-independent decay + time-specific inference" to isolate the two claimed novelties.

3. **Scope the zero-shot claims precisely.** Replace "zero-shot" with "zero-label" or "pre-training-only inference" unless the paper demonstrates true out-of-domain transfer (PopHR → eICU or vice versa).

4. **Add an out-of-domain experiment.** Pre-train on PopHR, evaluate on eICU, or vice versa. This is mentioned as future work but would be the decisive test of the paper's generalizability claim.

5. **Explain the "?" in Table 3** and provide the missing patient-level recall distribution for the trajectory analysis.

---

## Score and Decision

**Calibration anchors:**
- `/home/wg25r/review_agent/human_reviews/NialiwI2V6.md` (MOTOR): avg 7.5, accept (spotlight). Pre-trained EHR foundation model at 55M patients, 19 tasks, cross-site validation, clean comparisons. TrajGPT is clearly below this — smaller scale, confounded comparisons, in-domain only.
- `/home/wg25r/review_agent/human_reviews/2sCcTMWPc2.md` (TimelyGPT): avg 5.5, reject. TrajGPT's direct predecessor — reviewers flagged limited evaluation and theoretical depth. TrajGPT is meaningfully above TimelyGPT: more contributions (ODE framing + time-specific inference), more tasks, better ablation coverage.
- `/home/wg25r/review_agent/human_reviews/4VIgNuQ1pY.md` (Stable Neural SDEs): avg 6.67, spotlight. Rigorous continuous-time modeling with stability theory. TrajGPT's ODE framing is less rigorous but compensated by practical scale.
- `/home/wg25r/review_agent/human_reviews/KJ1w6MzVZw.md` (Large pre-trained TS models): avg 3.8, reject. Pre-training TS work with weaker baselines and limited empirical validation — TrajGPT is above this level.
- `/home/wg25r/review_agent/human_reviews/ZkEsEFFUyo.md` (CloudOps pre-training): avg 4.33, reject. Less-developed pre-training study without the multi-task depth TrajGPT has.

**Assessment:** TrajGPT has more contributions than TimelyGPT (5.5) and delivers concrete, measurable gains from its primary mechanism (time-specific inference). However, it does not approach MOTOR (7.5) in experimental rigor, scale, or clarity of comparison. The confounded pre-training comparisons and thin ablation for the main architectural novelty are meaningful weaknesses that prevent a strong recommendation for acceptance. The paper is better than the rejected pre-training TS papers (3.8–4.3) but does not resolve the key question of whether its architecture — beyond having a pre-training paradigm — is the driver of gains. Positioning it slightly above TimelyGPT (5.5), at the borderline, gives a score of **5.0**.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>