Now let me do calibration searches to anchor my score.Now I have enough information to write the final consolidated review. Let me synthesize the findings.Now I have all the information I need to write the final review. Let me compile it.

---

## Summary

TTVD (Test-Time adaptation by Voronoi Diagram guidance) reformulates neighbor-based TTA methods through the lens of computational geometry. The paper establishes that existing neighbor-based TTA methods implicitly implement Voronoi Diagrams, then extends this framework with (I) Cluster-Induced Voronoi Diagrams (CIVD) using rotation-augmented multi-site influence to jointly incorporate self-supervision and entropy minimization, and (II) Power Diagram (PD) subtraction to filter noisy boundary-adjacent samples during adaptation. Experiments on CIFAR-10/100-C, ImageNet-C, and ImageNet-R using the TTAB evaluation toolkit show consistent improvements over prior methods.

---

## Strengths

1. **Geometric reformulation is genuinely novel**: The paper identifies a structural equivalence between neighbor-based TTA methods and Voronoi Diagrams (Definition 3.1, Eq. 2–3), then extends this in two principled directions (CIVD and Power Diagram). This conceptual reframing is distinct from prior TTA work and grounded in established computational geometry.

2. **Figure 2a provides concrete, non-trivial diagnostic insight**: The visualization showing that entropy-based sample filtering only identifies noisy samples near decision boundaries — leaving interior noisy samples undetected — is a genuine empirical observation motivating the PD-subtraction mechanism, not a restatement of prior work.

3. **Progressive ablation cleanly decomposes contributions** (Table 2): VD (28.4%) → CIVD (22.7%) → CIPD (20.5%) on CIFAR-10-C demonstrates each geometric extension provides an additive gain, making the components interpretable.

4. **Standardized evaluation via TTAB with grid-search tuning**: Using the peer-reviewed TTAB toolkit (Zhao et al., 2023) with rigorous hyperparameter search is a genuine commitment to reproducibility and fair comparison. Reporting both classification error and ECE is appropriate for real-world deployment.

5. **Robustness to class mean precision** (Table 4): Using only 1% of ImageNet for class mean computation yields essentially identical performance to 10%, making the approach practical.

---

## Weaknesses

### Fatal
None.

### Major

- **Information asymmetry between TTVD and the neighbor-based baselines it claims to surpass.** Section 4.1 explicitly states: *"We use the full training set of CIFAR-10, CIFAR-100 to compute the class means for Voronoi sites and 10% of ImageNet for similar calculation."* T3A and TAST, however, construct prototypes exclusively from unlabeled test data with no access to training labels. This is a fundamental information advantage. The ablation confirms the problem: the simple VD baseline — which is just distance-to-training-class-mean plus entropy minimization — already achieves 28.4% error, well below T3A (40.3%) and TAST (39.6%). A large portion of the TTVD-vs-T3A gap (roughly 20 percentage points) is almost certainly attributable to having accurate, label-informed class centers rather than to the geometric framework itself. The paper's comparative narrative attributes these gains to Voronoi geometry, which is misleading. The paper never provides the obvious control: running T3A with training-class-mean prototypes to isolate what the geometry contributes independently of the information advantage. Note: TTVD's improvements over methods that do **not** use training class means (TENT 24.0→20.5%, SAR 24.2→20.5% on CIFAR-10-C) are real and the appropriate comparison, but these are not the comparisons emphasized in the paper.

- **CIVD's key claimed property — avoiding negative transfer — is asserted without mechanistic evidence.** Section 3.2 states: *"The joint label $\tilde{y}_k^{(\alpha)}$ avoids the negative transfer since the objective is now unified."* Operationally, CIVD creates 4 Voronoi sites per class by applying rotation augmentations to class means, then classifies based on a sum of power-distance terms (Eq. 4). This is functionally an ensemble of rotation-augmented prototype classifiers. The paper provides no gradient conflict analysis (e.g., cosine similarity between self-supervision and entropy minimization gradients), no convergence argument, and — critically — no ablation separating rotation augmentation from the specific CIVD aggregation formula. If simply adding rotation augmentation to vanilla T3A achieved similar gains to the VD→CIVD improvement (5.7%), the CIVD geometric framing would contribute no mechanistic value beyond the augmentation. This ablation is essential to validate the paper's core claim about CIVD.

### Minor

- **No variance reporting for the claimed marginal improvements.** Table 1 improvements over the best non-neighbor baselines are 0.8%, 0.7%, 1.6%, and 1.0% across datasets. All are single-run results. Without standard deviations across multiple runs or seeds, it is not possible to assess whether these differences are statistically meaningful. This matters because methods like SAR and TENT are non-deterministic in the online setting.

- **PD weight specification in main text is incomplete.** Definition 3.3 introduces weights $v_k$ and Lemma 3.1 connects them to logistic regression parameters $(W, b)$, but the main text does not state whether the operational PD uses frozen model weights or updated weights during adaptation. If the former, the PD structure is fixed and only applied for filtering; if the latter, interaction between PD boundaries and the model update step needs description. This is presumably in Algorithm 3 (Appendix H), but the gap leaves the filtering mechanism ambiguous in the main paper.

- **Adaptation curve interpretation (Figure 4)** has a confound: TTVD benefits from stable training-class-mean anchors over 750 online batches, which naturally resist the distributional drift that degrades entropy-based methods. The paper presents this as evidence that TTVD "continues to learn," but it may reflect the anchor's stabilizing role rather than the geometric framework's adaptive properties.

### Trivial

- The hyperparameter $\gamma$ in Eq. 4/6 and temperature $\tau$ in Eq. 3 are not reported in the main text. The sensitivity of these is unaddressed.

---

## Nice-to-Haves

- An ablation comparing T3A (or plain VD) with and without rotation-augmented sites, to directly test whether CIVD's gain comes from the augmentation or from the specific influence function aggregation.
- A fairness control: T3A seeded with training-class-mean prototypes (in addition to the usual pseudo-label update), to estimate how much of TTVD's advantage over T3A is due to information access rather than geometry.
- Evaluation on transformer/ViT backbones; all experiments use ResNet-26 and ResNet-50. Lemma 3.1 connects PD to linear classifiers, making extension to ViT architectures non-trivial and worth at least a brief discussion.
- Computational cost comparison: TTVD requires 4× inference passes per batch (for rotations) plus offline class mean precomputation. Wall-clock comparison to Tent and SAR is absent.

---

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Harsh Critic: "Comparison to Recent Baselines (RoTTA, ViDA, AdaContrast, LAME)"** — Removed per rule: do not mention missing related works/baselines we cannot independently confirm exist in the submission.

- **Harsh Critic: Table 4 robustness results trivialize the approach** — The claim that near-identical results at 10%/5%/1% proves that more data does not help is a straw-man reading. The paper frames this correctly as demonstrating robustness to imprecise class means, which is a practical strength. Removed.

- **Strength Finder: "Evaluation under challenging practical settings (Appendix B, label shift)"** — Removed because the appendix is stripped and we cannot verify details. The main text only briefly references these results.

- **Harsh Critic: "VD formalization does not apply to T3A since sites differ"** — Partially valid, but the paper clearly separates the general VD framework (which can use any sites) from the specific choice of training class means as sites. This is a framing observation, not a structural error. Removed as a standalone weakness.

- **Strength Finder: "Link between logistic regression and Power Diagram (Lemma 3.1)"** — Retained in principle but partially undercut by the missing specification of how $v_k$ is computed in practice. Moved to informational context rather than standalone strength.

---

## Novel Insights

The paper's most genuinely novel insight — beyond the geometric reformulation itself — is the diagnostic finding in Figure 2a: entropy loss decays sharply once a sample moves away from decision boundaries, meaning entropy-based sample filtering is geometrically blind to noisy samples deep within incorrect Voronoi cells. This is a structural limitation of entropy as a filtering criterion that has not been explicitly characterized this way in prior TTA literature, and it provides a principled motivation for boundary-aware filtering via Power Diagram subtraction. If the PD mechanism were better operationalized and ablated, this insight could anchor a stronger contribution.

---

## Suggestions

1. **Add the critical fairness control experiment**: Run T3A with training-class-mean prototypes (replacing its pseudo-label-based prototypes) and report results in the ablation. This is the single most important experiment to add.
2. **Ablate rotation augmentation**: Run the VD baseline (training class means, entropy minimization) with rotation-augmented class means but without the CIVD influence function, and compare to full CIVD. This isolates the geometric contribution of multi-site influence from the augmentation effect.
3. **Reframe the paper's narrative**: The primary story should be TTVD vs. entropy-based and self-supervised methods (TENT, SAR, TTT, BN-Adapt, SHOT) — the fair comparison set. The improvement over T3A/TAST should be clearly contextualized as coming partly from the use of training class means, which is an explicit design choice, not purely from the geometric framework.
4. **Report variance across runs** for the main Table 1 results.

---

## Score and Decision

**Calibration anchors used:**
- *DART* (rejected TTA, avg 5.67): Used labeled data at intermediate time, creating an information advantage over baselines — rejected partly for this. TTVD has a similar but less severe issue (training class means used in precomputation, not dynamic labeling).
- *PASLE* (accepted TTA poster, avg 6.4): Progressive pseudo-label enhancement with good but incremental contribution, accepted despite modest novelty concerns.
- *DeYO* (accepted spotlight TTA, avg 7.0): Strong motivation, novel sample selection criterion, good results, accepted despite some missing baselines.
- Low-scoring papers (scores 1–3): These suffer from no empirical contribution, trivial technical novelty, or fundamental conceptual flaws — none of which apply here.

**Positioning:** TTVD sits between DART and PASLE. Like DART, it has a genuine information asymmetry issue that undermines its primary comparative narrative. Unlike DART (where the labeled data use was central to the method), TTVD's use of training class means is one design choice among several, and improvements over methods that don't use class means (TENT, SAR) are real. However, unlike PASLE (accepted), TTVD lacks the critical ablation that would confirm its geometric claims are not explained by rotation augmentation alone. The improvements over fair baselines (0.7–1.6%) are modest, and no variance is reported. The paper needs reframing and two key missing ablations to support its core narrative.

**Score: 5.0 — Borderline Reject**

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>