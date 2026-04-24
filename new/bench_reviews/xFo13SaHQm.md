Now let me run calibration searches in parallel to anchor the score.Now I have sufficient information to write the final consolidated review.Now I have everything I need to write the final review.

---

## Summary
WithAnyone tackles identity-consistent (ID-consistent) image generation by addressing the **copy-paste** failure mode, where models replicate reference face appearance rather than synthesizing genuine identity variations. The paper contributes (1) MultiID-2M, a large-scale dataset of 500k paired group photos with ~400 reference images per celebrity identity; (2) MultiID-Bench with a principled M_CP copy-paste metric; and (3) a FLUX-based model trained via a 4-phase pipeline with GT-aligned ID loss and InfoNCE-style contrastive loss over an extended negative pool. The key empirical finding is Figure 5: prior methods cluster along a regression curve where higher identity fidelity entails stronger copy-paste, while WithAnyone breaks this trade-off.

---

## Strengths

- **Formalizing and quantifying the copy-paste artifact** (Figure 5, Eq. 2): The M_CP metric is conceptually well-motivated and the scatter plot revealing a systematic field-wide fidelity–copy-paste trade-off across 14 baselines is the most compelling empirical observation in the paper. This finding alone has value for the research community, as it exposes a blind spot in prior evaluation protocols.

- **MultiID-2M dataset and paired training** (Section 3, Table 3 FFHQ ablation): The construction of 500k group photos with per-identity reference banks (~400 images/identity) fills a genuine gap. The ablation shows that replacing this with FFHQ-only data collapses Sim(GT) from 0.405 to 0.224, demonstrating that the dataset enables an entirely different training regime. The construction pipeline (clustering ArcFace embeddings, cosine-matching to reference centers) is described at reproducible level of detail.

- **GT-aligned ID loss with full-timestep supervision** (Section 5.1, Eq. 4, Figure 7): The observation that landmark extraction from noisy diffusion intermediates is unreliable, and the solution of using GT landmarks for alignment, is a practical and non-obvious engineering insight. It enables ID supervision across all noise levels at negligible overhead—a meaningful improvement over PortraitBooth (t < 0.25 only) and PuLID (full denoising required). Figure 7 provides mechanistic evidence that this yields lower denoising error at low noise and more informative gradients at high noise.

- **Comprehensive baseline evaluation** (Table 1): Comparison against 14 methods spanning general-purpose (OmniGen, FLUX.1 Kontext, GPT-4o, DreamO) and face-specific models (PuLID, InstantID, UniPortrait, ID-Patch) is more thorough than typical in this sub-field.

- **MultiID-Bench standardization** (Section 4): The benchmark uses rare long-tail identities with no overlap with training, and the primary metric is SimGT rather than SimRef—an explicit design decision that penalizes copying when natural variation is expected. This design is principled and addresses a known confound in prior evaluation (e.g., PuLID, UniPortrait sampling from CelebA without fixed splits).

---

## Weaknesses

### Fatal
None.

### Major

- **Unexplained numerical discrepancy between the ablation table and the main results table**: Table 3 reports the "Full Setting" as Sim(G) = 0.405, Sim(R) = 0.551, CP = 0.161, while Table 1 reports the same model as Sim(GT) = 0.460, Sim(Ref) = 0.578, CP = 0.144—a gap of 0.055 in the primary identity metric. No footnote or clarification is given. The most likely explanation is that Table 3 evaluates the model after Phase 3 while Table 1 uses the Phase 4 (quality-tuned) final checkpoint; if so, the ablation does not characterize the final submitted model and the component analyses (GT-aligned loss, extended negatives, Phase 3) are understated or misattributed. The paper must either clarify explicitly that Table 3 reflects an intermediate checkpoint, or ensure the ablation is run on the same checkpoint as Table 1. In its current form, the ablation evidence for the paper's core technical contributions cannot be reliably interpreted.

- **Data-budget asymmetry confounds attribution of improvement**: WithAnyone trains on 2M images in a purpose-built 4-phase curriculum drawn from the same distribution as the benchmark, while all baselines are evaluated off-the-shelf with no retraining. The ablation (Table 3) only compares data scales via the crude "FFHQ only" condition, which strips both data scale and paired supervision simultaneously. There is no controlled experiment that holds data fixed and varies only the proposed loss modifications. As a result, it is impossible to determine whether the gain over baselines stems from the GT-aligned ID loss and contrastive objective, or from having 4× more in-distribution training data. This is a structural attribution gap. A minimal experiment—e.g., re-training a competitive baseline (PuLID) on the same MultiID-2M data with and without the proposed loss—would substantially strengthen the causal claim.

### Minor

- **Incorrect claim that WithAnyone achieves "the highest face similarity with regard to GT"** (Section 6.1): Table 1 shows InstantID Sim(GT) = 0.464 vs. WithAnyone 0.460. The claim is thus false by the paper's own numbers, even if the margin is small. The correct framing—"highest among face-specific models while maintaining substantially lower copy-paste"—would be accurate and still compelling, since InstantID's CP is 0.337 vs. 0.144 for WithAnyone.

- **Non-standard contrastive loss formulation not acknowledged** (Section 5.1, Eq. 5): The paper describes L_CL as following the "InfoNCE formulation," but standard InfoNCE places the positive term in both the numerator and denominator. Eq. 5's denominator sums only over negatives, and the numerator uses the GT embedding t rather than a reference embedding r. This is a valid loss formulation but it is not InfoNCE as conventionally defined. The paper should either correct the terminology or explain the design choice. The difference in optimization dynamics (the loss does not include a positive self-similarity term in the denominator) may matter in practice and deserves acknowledgment.

- **CP metric threshold selection is unjustified** (Section 6.1 caption): Computing CP rankings only for cases with Sim(GT) > 0.40 (Table 1) or > 0.35 (Table 2) without any principled basis is a concern. The paper does not report robustness to alternative thresholds, nor does it document what fraction of each baseline's samples is excluded. If methods with lower SimGT are systematically excluded from the copy-paste ranking in a non-uniform way, the CP comparison could be biased.

- **OmniContext ranking presentation in abstract/introduction**: WithAnyone ranks 9th of 14 overall on OmniContext (6.52), with GPT-4o (8.12), OmniGen2 (8.34), and FLUX.1 Kontext (7.94) substantially ahead. Section 6.1 accurately limits the claim to "highest among face customization models," but the abstract and introduction language ("maintains state-of-the-art identity similarity") could mislead readers about overall OmniContext standing. The caveat should appear earlier.

### Trivial

- **User study sample size**: 10 participants ranking 230 groups is modest. The full statistical analysis is deferred to the appendix, which is acceptable, but the text's claim of "moderate positive correlation" between M_CP and human judgments should at minimum cite a correlation coefficient (Pearson r or Spearman ρ) in the main text.

- **M_CP degenerate-regime analysis absent**: When θ_tr is very small (reference and GT nearly identical), the denominator shrinks and the metric amplifies minor differences. When both θ_gt and θ_gr are large and similar (wrong face generated), M_CP ≈ 0 registers as neutral rather than failure. A brief discussion of the metric's behavior in edge cases, with a histogram of θ_tr values in the test set, would improve interpretability—but this is a clarification rather than a flaw.

---

## Nice-to-Haves

- Controlled data-equivalence ablation: train a competitive baseline on MultiID-2M with/without the proposed loss modifications to isolate data contribution from loss contribution.
- CP sensitivity analysis across threshold values (0.30, 0.35, 0.40, 0.45) to establish that rankings are robust to threshold selection.
- Analysis of non-celebrity identities: all training and evaluation uses celebrities; whether the model generalizes to private individuals is the primary deployment scenario and is entirely untested.
- Qualitative failure-mode analysis: the paper shows only favorable results; showing hard cases and failure distributions would clarify the method's operating regime.

---

## Removed Points
*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic: "SimGT conflates identity fidelity with instance-level appearance"** — Removed. This is inherent to any single-GT evaluation design and is partially acknowledged by the paper's own Figure 2. The paper explicitly chose SimGT over SimRef precisely because it penalizes over-copying. Criticizing the metric for using a single GT image is a generic concern applicable to virtually all evaluation in this subfield, not a specific flaw of this paper.

- **Harsh Critic: Benchmark contamination for VLM/general baselines** — Removed. The paper explicitly notes in Table 2's footnote that GPT-4o exhibits prior knowledge of TV series identities and handles this case. The claim that all foundation models benefit equally from celebrity pre-training knowledge is speculative and would affect SimRef (which the paper de-emphasizes) rather than SimGT in a predictable direction.

- **Harsh Critic: Ablation of extended negatives "contradicts paper's narrative"** — Partially removed. The critic argues that removing extended negatives lowers both CP and SimGT, which is "consistent with the trade-off the paper claims to break." However, the comparison is not within the ablation table but against external baselines: WithAnyone (full) achieves 0.460/0.144 vs. InstantID 0.464/0.337—similar SimGT but 2.3× lower CP. The ablation shows the extended negatives are necessary to achieve that high-fidelity, low-CP operating point; without them the model has neither property. This is not contradictory. Retained only as a minor presentation clarity issue.

- **Harsh Critic: M_CP formulation "has a degenerate regime not discussed"** — Retained as Trivial (moved to Trivial tier) as a clarification, not a fatal flaw.

- **Strength Finder: "Fully open-sourced project" as a standalone strength** — Removed as generic; it is a nice property but not a research contribution.

- **Strength Finder: "Thorough ethics statement"** — Removed as generic praise without bearing on scientific contribution.

---

## Novel Insights

The paper's most genuinely novel conceptual contribution is the empirical demonstration that the entire sub-field of identity-consistent generation has been optimizing a metric (SimRef) that inadvertently rewards copy-paste, and that virtually all methods lie on a Pareto frontier where increasing measured similarity requires increasing copy-paste. Figure 5 makes this visible for the first time, and the M_CP metric operationalizes it. This reframing—from "maximize SimRef" to "maximize SimGT while minimizing M_CP"—is a meaningful contribution to how the community should evaluate identity generation systems, independent of whether the WithAnyone model itself is the best possible solution.

---

## Suggestions

1. **Resolve the Table 1/Table 3 discrepancy explicitly**: Add a footnote or sentence stating that Table 3 evaluates the Phase 3 checkpoint while Table 1 uses the Phase 4 final model. If possible, run ablations on the same checkpoint as Table 1.

2. **Add a data-controlled baseline**: Fine-tune PuLID on MultiID-2M with and without the GT-aligned ID loss + extended negatives. This one experiment would dramatically strengthen the attribution of improvements to the technical contributions.

3. **Correct the "highest SimGT" claim** in Section 6.1 to accurately state that WithAnyone achieves the highest SimGT among face customization models while InstantID is marginally higher overall (0.464 vs. 0.460) but with 2.3× more copy-paste.

4. **Report the Spearman ρ/Pearson r** for the M_CP vs. human CP judgment correlation directly in the main text, not just in the appendix.

5. **Justify or sweep the CP threshold**: Show CP rankings under at least two alternative thresholds (e.g., 0.35 and 0.45) in an appendix table to establish robustness.

---

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Human Score | Comparison |
|---|---|---|---|
| DisenBooth | `/human_reviews/FlhjUkC7vH.md` | 7.5 (Accept/Poster) | Similar task (ID-preserving generation), cleaner ablations, stronger causal attribution of technical contributions; sets the high anchor |
| PersonalVideo | `/human_reviews/ndtFyx7UWs.md` | 4.5 (Withdrawn/Reject) | ID-fidelity video customization, similar scope, weaker results, fewer baselines; low anchor for this domain |
| ConceptFlow | `/human_reviews/EhSUM1FcJw.md` | 4.0 (Reject) | Multi-concept generation, rejected for insufficient novelty and weak evidence; low anchor |
| jw7P4MHLWw (Personalized Representation) | `/human_reviews/jw7P4MHLWw.md` | 5.8 (Accept/Poster) | Personalized generation with synthetic data, moderate novelty; medium anchor |
| DreamBench++ | `/human_reviews/4GSOESJrk6.md` | 6.0 (Accept) | Benchmark+evaluation paper for personalized generation; comparable in scope to the dataset/benchmark contribution here |

**Reasoning:** WithAnyone has more novel contributions than ConceptFlow (4.0) and PersonalVideo (4.5): the copy-paste metric, the large-scale paired dataset, and comprehensive 14-baseline evaluation are genuine community contributions. It is comparable in ambition to DreamBench++ (6.0), which is likewise a benchmark+model paper, but DreamBench++ had a cleaner evaluation design. It falls short of DisenBooth (7.5) because (a) the attribution gap between data and technical contributions is unresolved, and (b) the unexplained Table 1/Table 3 discrepancy weakens the ablation evidence. The correct comparison is to DreamBench++ (6.0) and jw7P4MHLWw (5.8) as the closest anchors in quality. The table discrepancy is a significant but potentially resolvable issue that pulls the score down from the 6 range. I place it at **5.5**.

**Evaluation on key axes:**
- *Originality*: Moderate-high. Copy-paste framing is novel; GT-aligned ID loss is practical; dataset is large-scale and genuinely enables new training.
- *Importance*: High. Copy-paste is a real and practical failure mode in deployed identity generation systems.
- *Claim support*: Moderate. The system-level results are supported, but attribution to specific technical components is not cleanly established.
- *Experimental soundness*: Moderate. 14-baseline comparison is strong; ablation is undermined by the checkpoint discrepancy and data confound.
- *Writing clarity*: Good overall, with the notable exception of the unclarified Table discrepancy.
- *Value to community*: High for dataset and benchmark; moderate for model.

**Decision: Borderline Reject.** The paper's dataset and benchmark contributions are genuine and the copy-paste framing is valuable. However, the unexplained numerical discrepancy between the main table and the ablation table, combined with the unresolved attribution gap between data scale and technical contributions, mean the central technical claims are not adequately supported in the current form. With clarification of the checkpoint discrepancy and a data-controlled ablation, this paper would be above the acceptance bar.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>