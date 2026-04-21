## Summary

TTVD formulates test-time adaptation from a geometric perspective by identifying that neighbor-based TTA methods correspond to Voronoi Diagrams (VD). The paper extends this foundation in two directions: (1) Cluster-induced Voronoi Diagram (CIVD), which replaces single Voronoi sites with clusters of augmented prototypes (via rotation self-supervision) and uses a multi-source influence function for soft labeling; and (2) Power Diagram (PD), which introduces weighted cells to identify noisy samples via PD-VD boundary disagreement. The combined method (CIPD) achieves state-of-the-art error rates and ECE on CIFAR-10-C, CIFAR-100-C, ImageNet-C, and ImageNet-R, with a progressive ablation showing each extension contributes to performance gains.

## Strengths

- **Clean geometric formalization of neighbor-based TTA**: The observation that nearest-prototype TTA methods are instantiations of Voronoi Diagrams (Section 3.1, Definition 3.1, Eq. 2) provides a principled and interpretable grounding for this family of methods, connecting them to well-studied computational geometry structures.

- **Consistent improvements across all benchmarks**: Table 1 shows TTVD achieves the lowest error rates on all four datasets, with particularly notable ECE improvements (e.g., 11.8% on CIFAR-10-C vs. 16.4% for the next-best SHOT; 21.0% on ImageNet-C vs. 25.1% for the next-best TAST), indicating substantially better-calibrated predictions.

- **Clear progressive ablation**: Table 2 shows each geometric extension provides measurable gains—VD (28.4%), CIVD (22.7%, +5.7%), CIPD (20.5%, +2.2%)—making the contribution of each component transparent.

- **Practical robustness analyses**: Table 4 shows TTVD is robust to class mean precision (59.8%→59.9% when reducing ImageNet training data from 10% to 1%). Appendix B examines batch size and label shift effects, both practically relevant for real-world TTA deployment.

- **Superior adaptation dynamics**: Figure 4 shows TTVD continues decreasing error across online batches on ImageNet-C noise corruptions while TENT and SAR stagnate, demonstrating resistance to the overfitting problems common in TTA.

- **Strong improvements over neighbor-based baselines**: Table 3 shows TTVD substantially outperforms prior neighbor-based methods on ImageNet-C blur corruptions (e.g., 53.2% on Zoom vs. 60.6% for AdaNPC; 68.6% on Motion vs. 72.3%).

## Weaknesses

### Fatal

None.

### Major

- **Disconnect between geometric framing and actual operational mechanisms**: The paper's central narrative is that Voronoi Diagram generalizations drive TTA improvements. However, the actual mechanisms are recognizable as combinations of known techniques: CIVD is rotation-augmented prototypes with distance-based soft labeling (a standard self-supervised prototype expansion), PD-based filtering is classifier-prototype disagreement filtering (since by Lemma 3.1, PD boundaries are determined by the original classifier weights and biases), and the VD loss (Eq. 3) is softmax over negative distances—a standard prototype-based classification loss. The geometric formalism provides a principled *connection* between these ideas, but it is unclear whether the geometry itself drives the improvements or simply repackages known techniques. The paper does not isolate the contribution of the geometric formulation from the contribution of the underlying mechanisms. Without a comparison to a naive combination of self-supervision + prototype-based soft labeling + classifier-prototype disagreement filtering (without the CIVD/PD formalism), it is impossible to assess whether the geometric framework adds value beyond combining known ideas.

- **Overstated "unification" claim**: The paper repeatedly claims CIVD "unifies" self-supervision and entropy minimization (Section 3.2, line 140: "The joint label $\tilde{y}_k^{(\alpha)}$ avoids the negative transfer since the objective is now unified") and that it "enables a seamless integration" (Section 1, line 58). In reality, self-supervision is used as a *preprocessing step* to construct augmented prototypes (rotation augmentation of $\mu_k$ to generate $C_k$), and then a single entropy-like loss is applied to CIVD-derived soft labels. There is no joint optimization of two objectives, no gradient conflict resolution mechanism, and no analytical or empirical demonstration that gradient conflicts are reduced compared to naive joint training. The paper identifies conflicting gradients as a key challenge (Section 1, line 49) but does not show its method addresses this challenge beyond simply having one loss instead of two—which is avoidance, not resolution.

### Minor

- **Foundational VD underperforms standard baselines, undermining the narrative**: Table 2 shows VD alone achieves 28.4% error on CIFAR-10-C, which is worse than TENT (24.0%), SHOT (21.9%), TTT (21.3%), and SAR (24.2%) from Table 1. The paper claims VD "already surpasses that of other neighbor-based methods" (line 199), which is accurate (T3A: 40.3%, TAST: 39.6%), but it does not acknowledge that the geometric foundation alone is weaker than non-geometric approaches. Since virtually all performance improvement comes from adding self-supervision (CIVD) and filtering (CIPD), the claim that geometry drives the improvements is weakened.

- **No ablation of the γ hyperparameter in the influence function**: The influence function $F(z, C_k) = -\text{sign}(\gamma) \sum_\alpha (d(\mu_k^{(\alpha)}, z))^\gamma$ (Eq. 4) has a critical exponent γ that controls the entire behavior of CIVD/CIPD—negative values yield inverse-distance weighting, positive values yield polynomial weighting. While γ was presumably tuned via grid search (Section 4.1), no ablation is provided, making it unclear how sensitive the method is to this choice and whether the geometric structure or hyperparameter tuning drives performance differences.

- **PD-based noise filtering lacks quantitative justification beyond 2D visualization**: The claim that "subtracting PD from VD" identifies noisy samples (Section 3.3, Figure 2) is supported only by a 2D MNIST visualization and the entropy landscape in Figure 2a. By Lemma 3.1, PD boundaries are determined by the classifier weights and biases, while VD boundaries are determined by class means—so PD-VD disagreement is simply classifier-prototype disagreement, a straightforward idea that requires no Power Diagram formalism. The paper provides no precision/recall analysis of filtering quality and no comparison to simpler filtering heuristics beyond the entropy landscape visualization, which the paper argues (but does not quantitatively demonstrate) is insufficient.

### Trivial

- The phrase "remarkable improvements" in the abstract overstates improvements of 0.7–1.6 percentage points on error rate for the full method (Table 1), though the ECE improvements are more substantial.

## Nice-to-Haves

- A comparison to a simple combined baseline (e.g., TTT-style self-supervision + TENT-style entropy minimization + entropy-based filtering, without the CIVD/PD formalism) would clarify whether the geometric framework adds value beyond combining known mechanisms.
- Quantitative analysis of PD-based filtering quality (precision/recall of the noise filter, or comparison to entropy-based filtering in isolation) would strengthen the PD contribution claim.
- Reporting standard deviations or confidence intervals across multiple runs, while not standard in the field, would strengthen the statistical claims.
- An analytical or empirical demonstration that CIVD reduces gradient conflicts compared to naive joint training of self-supervision and entropy minimization would substantiate the "unification" claim.

## Removed Points

These points are flagged to be removed; treat them with caution:

- **Harsh Critic: "CIVD is standard self-supervised prototype expansion, PD filtering is classifier-prototype disagreement, VD loss is standard cross-entropy"** — The identification of these mechanisms as known techniques is valid and retained as a Major weakness (the disconnect between framing and mechanism). However, the critic's phrasing that these are *merely* repackaged known techniques is partially softened because the geometric formalism does provide a principled *unifying perspective* connecting these ideas, even if the operational mechanisms are familiar.

- **Harsh Critic: "γ is never experimentally specified"** — Reproducibility concerns about undisclosed hyperparameters are removed per rules. The methodological concern about γ ablation is retained as a Minor weakness.

- **Harsh Critic: "Algorithm 3 relegated to Appendix H, sample filtering criterion not defined mathematically"** — Removed per rules about missing appendix content (the parser strips appendices).

- **Harsh Critic: "No standard deviations or confidence intervals"** — Moved to Nice-to-Have, as single-run evaluation is standard practice in the TTA benchmarking field.

- **Harsh Critic: "Missing baselines: ViDA, EcoTTA, CoLA, OTA"** — Removed per rules; cannot confirm these works exist or their relevance.

- **Harsh Critic: "VD already surpasses that of other neighbor-based methods — referencing only T3A and TAST, not stronger methods"** — The paper's claim is specifically about neighbor-based methods and is factually accurate. The broader concern that VD underperforms entropy-based methods is retained separately as a Minor weakness.

- **Strength Finder: "CIVD effectively unifies self-supervision and entropy minimization"** — Dropped as a strength because this conflicts with the verified Major weakness that the "unification" claim is overstated. Self-supervision is a preprocessing step, not a jointly optimized objective.

- **Strength Finder: "PD-based boundary analysis reveals limitations of entropy-based filtering — more principled filtering criterion"** — Partially dropped. The entropy landscape observation is valid but the claim of "more principled" filtering is not well-supported, since PD-VD disagreement reduces to classifier-prototype disagreement. The visualization insight is retained implicitly in the robustness of CIPD results.

## Novel Insights

The most insightful observation emerging from cross-examining the paper against the reviews is that TTVD's core contribution may be less about geometric structures driving novel mechanisms and more about geometric structures providing a *coherent organizational principle* for combining three known TTA ingredients (prototype-based classification, self-supervised augmentation, and classifier-prototype disagreement filtering). The question is whether an organizational principle alone—absent a mechanism that could not be derived without it—constitutes a sufficient contribution. The paper would be significantly strengthened by demonstrating that the CIVD influence function (Eq. 4) produces soft label assignments that differ meaningfully from what a naive combination of augmented prototypes would yield, or that PD-VD boundary subtraction captures something beyond simple classifier-prototype disagreement.

## Suggestions

- Add a direct comparison to a "naive combination" baseline (self-supervised prototype augmentation + nearest-prototype soft labeling + classifier-prototype disagreement filtering) without the CIVD/PD formalism. This is the single most impactful experiment for validating the geometric framework's contribution.
- Ablate the γ parameter in the influence function (Eq. 4) and report how soft label assignments change with different γ values, to clarify whether the influence function structure or hyperparameter tuning drives the CIVD→CIPD gains.
- Tone down the "unification" language: clarify that CIVD incorporates self-supervision as a prototype construction step rather than jointly optimizing two objectives, and acknowledge that gradient conflict avoidance comes from having a single objective rather than from conflict resolution.
- Report precision/recall of the PD-based noise filter against a ground-truth noisy sample identification (e.g., samples where the model prediction disagrees with a post-hoc correct label), to quantitatively validate the PD-based filtering criterion.

<context>
**Paper summary**: TTVD proposes a geometric framework for test-time adaptation by identifying that neighbor-based TTA methods correspond to Voronoi Diagrams, and extending this to two generalizations: Cluster-induced Voronoi Diagram (CIVD) which uses rotation-augmented prototypes with a multi-source influence function for soft labeling, and Power Diagram (PD) which uses weighted cells to identify noisy samples via PD-VD boundary disagreement. The combined CIPD method achieves SOTA error and ECE on CIFAR-10-C, CIFAR-100-C, ImageNet-C, and ImageNet-R, with progressive ablation showing each extension contributes gains (VD 28.4% → CIVD 22.7% → CIPD 20.5% on CIFAR-10-C).

**Original reviewer signal**: Harsh Critic views the paper as repackaging known techniques (self-supervised prototype augmentation, classifier-prototype disagreement filtering, distance-based soft labeling) in geometric formalism that obscures rather than illuminates, with overstated "unification" claims and a foundational VD that underperforms baselines. Strength Finder views the geometric unification as a genuine contribution with strong empirical results and clean formalization.

**What was dropped and why**: (1) Reproducibility concern about γ not being specified — removed per rules on undisclosed hyperparameters; the methodological concern about γ ablation is retained. (2) Missing appendix (Algorithm 3) — removed per rules about parser-stripped appendices. (3) Missing baselines (ViDA, EcoTTA, etc.) — removed per rules on unconfirmable related works. (4) No standard deviations — moved to Nice-to-Have as single-run evaluation is standard in TTA benchmarking. (5) "VD surpasses only neighbor-based methods" criticism — partially dropped because the paper's claim is factually accurate and specifically scoped; the broader concern retained separately. (6) Strength Finder's "CIVD unifies self-supervision and entropy minimization" — dropped as it conflicts with the verified overclaim weakness.

**Cross-checks performed**: (1) Verified VD baseline (28.4%) vs. entropy-based methods from Table 1 — confirmed VD is indeed weaker, but the paper's claim about neighbor-based methods is accurate. (2) Verified "unification" claim at line 140 — confirmed it says "the objective is now unified" and "avoids negative transfer," but self-supervision is indeed a preprocessing step (rotation augmentation to construct C_k), not a jointly optimized objective. (3) Verified Lemma 3.1 connects PD to classifier weights — confirmed, meaning PD-VD disagreement is classifier-prototype disagreement. (4) Verified γ is mentioned as hyperparameter but no value or ablation is provided in main text. (5) Confirmed Table 2 provides some variability measure for CIVD (1.57) and CIPD (1.23) but not for VD or Table 1 methods.

**Review construction notes**: The two Major weaknesses stem from the same root issue — the disconnect between geometric framing and operational mechanisms. They are kept separate because one is about the general framing-vs-mechanism gap (affects the entire paper's contribution claim) and the other is about a specific overclaim ("unification") that could be corrected with language changes. The paper's scope is explicitly about geometric perspectives for TTA, so criticizing absence of non-geometric analyses is limited to Nice-to-Haves.
</context>