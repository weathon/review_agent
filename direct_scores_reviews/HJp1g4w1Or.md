## Summary
This paper applies adversarial harmonization (an ADDA-style framework from Dinsdale et al., 2021) to MEG-based speech decoding, claiming to be the first feature-level deep learning harmonization for MEG neuroimaging data. Two models are evaluated: Brainmagick (Défossez et al., 2023) and MEGalodon (Jayalath et al., 2024), pooling across four MEG datasets. Results are clearly positive for Brainmagick but decidedly mixed for MEGalodon, with harmonization hurting speech detection while marginally improving voicing classification. As a side contribution, the authors release an open-source PyTorch/Lightning reimplementation of Brainmagick.

---

## Strengths

- **Statistically significant, cross-dataset improvement for Brainmagick:** Adversarial harmonization achieves 71.0% ±0.2 top-10 accuracy on the Gwilliams split and 68.6% ±0.2 on MOUS, versus 68.8% and 66.8% for naive pooling, confirmed significant at p<0.05 (one-sided t-test over 3 seeds). Crucially, the harmonized model outperforms the original single-dataset baselines (70.7%, 68.5%), demonstrating genuine benefit from cross-dataset pooling when domain shift is addressed.

- **Empirical evidence for age as a strong confound in MEG decoding:** The controlled comparison of balanced vs. random subsets in Table 3 provides concrete, quantified evidence that participant age distributions across datasets significantly affect model behavior and domain separability — a finding that has practical implications for how the neuroimaging community designs and pools studies.

- **Quantitative domain alignment evidence beyond t-SNE:** The domain classifier accuracy reduction from 99.9% to 79.7%/67.9% (full dataset/subset respectively) for Brainmagick provides a direct numeric measure of harmonization effectiveness, not just qualitative visualization.

- **Open-source Brainmagick reimplementation with verified bug fix:** Replacing internal Facebook Research tooling (Flashy/Dora) with standard PyTorch/Lightning and fixing a sensor-labeling bug meaningfully lowers the barrier to entry for the field. The corrected implementation still reproduces baseline performance within ~1%, establishing its reliability.

---

## Weaknesses

- **Abstract overclaims for MEGalodon:** The abstract states "We successfully improve the performance of both models when training across multiple datasets." However, Table 3 shows that dataset harmonization *reduces* speech detection from 57.29% to 55.04% (best case), and the voicing improvement is only 0.05 percentage points (52.65% vs. 52.60% control) — within any plausible noise margin. The warm-up-only condition (57.76%) actually outperforms all harmonized variants on speech detection. This overclaim is not minor: it misrepresents the central finding for one of the two models.

- **MEGalodon fine-tuning evaluated on only 3 subjects (Armeni dataset):** The paper explicitly notes the Armeni dataset contains three subjects, yet all MEGalodon fine-tuning conclusions rest on it exclusively. Any performance differences in Table 3 (e.g., 0.05% for voicing) are statistically meaningless at this scale. No confidence intervals or significance tests are reported for Table 3, in contrast to Table 2 — making it impossible to distinguish signal from noise. This undermines all MEGalodon-related claims.

- **No comparison to simpler harmonization baselines:** The paper compares adversarial harmonization only against naive pooling and a pre-training scheme, but provides no comparison to ComBat (the standard neuroimaging harmonization tool), z-score normalization per dataset, or other lightweight domain adaptation approaches. Without such baselines, it is impossible to determine whether the adversarial complexity is necessary or whether simpler approaches would suffice.

- **Training instability acknowledged but unresolved:** The paper admits adversarial harmonization is "extremely unstable, with task loss diverging sharply when the harmonization phase begins" and that "equivalent hyperparameter testing" for age harmonization could not be completed. The best speech detection result (57.76%) actually comes from the warm-up-only condition, suggesting the adversarial phase itself is counterproductive for MEGalodon's primary task. This leaves the reader unsure whether positive MEGalodon results are reproducible or lucky survivors of instability.

- **Scope limited to ~15% of available subjects:** Computational constraints restrict experiments to approximately 15% of subjects per dataset, yielding roughly 30–96 subjects per dataset. The "big data" motivation of the paper requires demonstrating that performance scales with pooling large datasets, yet the experiments demonstrate this at only small scale. The paper acknowledges this limitation, but it significantly weakens the generalizability claims.

- **Shallow vs. deep fine-tuning explanation is post-hoc and untested:** Section 5's explanation that harmonization hurts speech detection (shallow fine-tuning) but helps voicing (deep fine-tuning) because of the protocol difference is entirely speculative. The hypothesis is plausible but is presented as an explanation without any experimental verification (e.g., switching the fine-tuning protocol to confirm the causal mechanism).

---

## Nice-to-Haves

- Ablation of σ=10 for the Gaussian spreading in age binning; the choice is not justified and the sensitivity to this parameter is unknown.
- Ablation of α=0.25 scaling factor for aggregated domain classifier losses in MEGalodon.
- Quantitative domain alignment metrics (e.g., MMD, proxy A-distance) to complement t-SNE visualizations.
- Loss convergence curves across seeds to characterize training instability more precisely and help practitioners reproduce stable runs.
- Confusion matrices or per-class accuracy breakdowns for speech/voicing decoding to understand which phoneme categories benefit or are harmed by harmonization.
- A leave-one-dataset-out evaluation to more rigorously assess cross-dataset generalization.
- Explicit fine-tuning protocol ablation (freezing vs. unfreezing encoder for speech detection) to empirically test the shallow/deep hypothesis rather than leaving it speculative.

---

## Removed Points

*These points are flagged for removal — treat them with caution.*

- **[REMOVED — style/nitpick] Title is misleading:** The harsh critic argues "representations of speech" implies self-supervised representation learning. This is a reading preference issue; the paper's title is interpretable as "MEG-based speech decoding features," not strictly representation learning in the SSL sense.

- **[REMOVED — addressed by paper] Single-GPU vs. multi-GPU performance gap:** The paper explicitly notes and accounts for the GPU-count effect on contrastive loss scaling, and uses single-GPU results as its primary comparison baseline. This concern is adequately addressed.

- **[REMOVED — factual check on Armeni subject count] "3 subjects" as a new criticism:** The harsh critic is correct that Armeni has 3 subjects (confirmed: "three subjects each listening to 10 hours of speech"). This is kept as a legitimate weakness above.

- **[REMOVED — scope creep] Criticizing lack of methodological novelty as a standalone ML contribution:** The paper's target audience includes both the ML and neuroimaging communities. Within the paper's stated scope — demonstrating the first feature-level harmonization for MEG — algorithmic novelty at the level of inventing a new domain adaptation method is not required. The critique that "novelty lies almost entirely in application" is valid context but should not be the sole basis for rejection, given the genuine domain-specific value.

- **[REMOVED — non-standard requirement] Requesting theoretical proofs or formal convergence guarantees for adversarial harmonization:** This is an empirical systems paper; theoretical proofs are not standard expectations in this setting.

- **[REMOVED — minor engineering nitpick] α=0.25 called "ad hoc":** The α=0.25 value is directly motivated by the paper's four-feature-vector design (3 pretext tasks + original input). Treating it as an unmotivated ad-hoc constant is unfair.

- **[WEAKENED → Nice-to-Have] Open-source implementation not being a "scientific" contribution:** While it is unusual to list an implementation as a primary scientific contribution at ICLR, it is a genuine practical contribution with community value (bug fix + accessibility), and is appropriately scoped as such by the authors.

---

## Novel Insights

The most genuinely novel observation in this work — beyond the paper's technical results — is the empirical demonstration that participant age distribution is a dominant source of apparent "dataset bias" in MEG speech decoding: the MEGalodon control performs better when subsets are age-balanced, and this effect is strong enough that removing it partially decouples the domains even before adversarial training. This points to a previously underappreciated confound in cross-dataset MEG studies and has concrete implications for study design (targeted recruitment of older participants). The secondary insight — that deep fine-tuning (unfreezing the encoder) can recover task-specific performance lost during harmonization, while shallow fine-tuning cannot — is plausible and important for practitioners applying harmonization to pretrained models, though it remains unverified experimentally and should be treated as a hypothesis to test rather than an established finding.

---

## Suggestions

1. **Correct the abstract.** "We successfully improve the performance of both models" overstates MEGalodon results. The abstract should honestly characterize the differential outcomes across models and tasks.

2. **Add statistical testing for Table 3.** At minimum, run Table 3 over 3 seeds with confidence intervals, as done for Table 2. Even 3 seeds would allow readers to determine whether the voicing improvement is significant or noise.

3. **Include ComBat or a simpler DA baseline.** A single run of ComBat or z-score-per-dataset normalization applied before standard pooling would establish whether adversarial complexity adds anything over trivial corrections.

4. **Experimentally verify the shallow/deep fine-tuning hypothesis.** Run speech detection with both encoder and task head unfrozen and report the result. If this recovers performance, it strongly validates the paper's main explanatory claim for the MEGalodon results.

5. **Flag the 3-subject evaluation as a named limitation.** The Limitations section should explicitly state that the Armeni evaluation is insufficient for statistical reliability, and propose that future work replicate on a larger held-out dataset (e.g., a held-out MOUS subset).

---

## Evaluation

- **Novelty:** Moderate-low from an ML perspective (direct application of Dinsdale et al., 2021 to a new modality with minor engineering choices); moderate from a neuroimaging perspective (first MEG application of this class of harmonization).
- **Technical soundness:** Adequate for the Brainmagick experiments; weak for the MEGalodon experiments, which suffer from a 3-subject evaluation, missing significance testing, and an acknowledged failure to complete hyperparameter search.
- **Empirical support:** Strong for Brainmagick (statistically significant, multiple seeds, scaling confirmed). Poor for MEGalodon (no CIs, 3 subjects, primary task harmed by harmonization).
- **Significance:** Meaningful for the MEG/BCI/neuroimaging community, particularly the age confound finding. Limited for the core ML community as a methods contribution.
- **Clarity:** Generally clear in motivation and high-level description; the "shallow" vs. "deep" fine-tuning distinction is introduced only in the discussion without being defined in methods, and some computational details (four-pass training cost for MEGalodon) are underspecified in the main text.

Overall, this is a paper with a credible contribution for one model and a weak, overinterpreted contribution for the other. The Brainmagick half meets a reasonable publication bar; the MEGalodon half does not, yet is used to support a central claim in the abstract. The paper would need significant revision — particularly honest characterization of MEGalodon results, statistical testing for Table 3, and at least one simpler baseline — to be considered strong enough for ICLR.

MY FINAL SCORE: <pineapple>4.3</pineapple>