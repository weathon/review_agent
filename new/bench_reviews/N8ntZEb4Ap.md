Now let me search for calibration anchors.Good. Now let me run calibration searches while I finish reading the paper.Now I have enough information to write the consolidated review. Let me finalize it.

---

## Summary

AutoNFS is a differentiable end-to-end feature selection method that combines a masking network (using Gumbel-Sigmoid sampling with temperature annealing) with a task network to jointly learn which features to select. The key claimed novelty is that AutoNFS automatically determines *how many* features to retain (not just *which* ones), using a cardinality penalty term L_select = (1/D)Σmⱼ in the total loss, rather than requiring the user to specify a budget k. It is evaluated on 11 tabular OpenML datasets under three noise-injection scenarios and 24 metagenomic datasets, consistently outranking 10 baseline FS methods.

---

## Strengths

- **Zero misselection errors for random and Gaussian-corrupted noise scenarios (Figure 3a):** AutoNFS perfectly excludes injected noise features in two of three benchmark scenarios, while all baselines show substantially higher misselection rates. This is a concrete, specific empirical result with a clear quantitative advantage.

- **Strong rank performance across all three corruption scenarios (Figure 2):** AutoNFS achieves best average rank in the random and second-order scenarios and rank 2.1 in the corrupted scenario — consistent across all three distinct settings and 11 datasets.

- **Breadth of evaluation:** Experiments cover both synthetic benchmarks (Cherepanova et al. 2023 protocol), 24 metagenomic datasets, MNIST interpretability visualization, and computational scaling analysis. This multi-angle validation is above average for a feature selection paper.

- **Generalization across classifiers in the metagenomic experiments:** The AutoNFS-selected features improve not just MLP (the training classifier) but also Random Forest (+1.2 pp), addressing the concern of classifier-specific selection bias.

- **Clean, well-motivated architecture:** A single globally-learned binary mask trained end-to-end through Gumbel-Sigmoid relaxation is a principled and interpretable design for tabular data FS, and the paper explains the exploration-exploitation curriculum well.

---

## Weaknesses

### Fatal

None.

### Major

- **The "automatic feature count" claim is partially overstated.** The paper's central distinguishing claim is that AutoNFS "automatically determines the minimal set of features" without user specification, unlike methods that require k. However, the number of selected features is entirely controlled by λ (the cardinality penalty weight in L_total = L_task + λ·L_select). The paper explicitly acknowledges this in the Conclusion: "the balance between sparsity and accuracy, controlled through a single λ parameter." The authors justify λ = 1 by empirical validation on the benchmark datasets. This is a user-defined hyperparameter that implicitly sets the feature budget, just through a different interface (regularization weight vs. cardinality k). The claim is better framed as "replacing a cardinality constraint with a regularization weight that can be fixed universally at λ = 1," which is a weaker but still meaningful claim. As written, the framing misleads readers into thinking the method requires zero feature-count-related user input.

- **Baseline comparison constrains competitors to a potentially suboptimal k.** The paper states: "all baseline methods select the same number of features as were in the initial representation (before corruption), whereas our method automatically chooses a much smaller subset." Baselines are locked to k = D_original while AutoNFS freely optimizes k. Baselines could likely achieve better performance with smaller k cross-validated on each dataset (Table 1 shows AutoNFS selects substantially fewer than D_original even after noise removal). The observed performance gap therefore conflates (a) the quality of AutoNFS's selection mechanism and (b) the benefit of choosing an appropriate k. A clean comparison would include at least one baseline with k tuned via cross-validation. Without it, the experiment demonstrates that "using fewer features is often better" (a known result) rather than "AutoNFS is a better feature selector." This ambiguity weakens the central performance claim, though automatic k-determination is itself a valuable contribution.

### Minor

- **The masking network design is unexplained and unablated.** The masking network f: R^{D_e} → R^D takes a single fixed (but learned) embedding e as input. Since e is a constant across examples, f(e) is functionally equivalent to a trainable vector w ∈ R^D reparameterized through an MLP. No ablation compares this design against directly training a logit vector w ∈ R^D. The architecture of f (depth, width, activations) is not described in the main text. This is the most structurally unusual design choice in the paper and goes entirely unexplained.

- **Near-constant computational scaling claim (α ≈ 0.08) lacks architectural justification.** Section 4.3 presents α ≈ 0.08 as a "significant algorithmic advancement," but provides no architectural explanation for why the masking or task networks would be near-constant in D. The final output layer of f must produce D logits, and the task network g receives D-dimensional inputs — both scale at least O(D) in a standard MLP. The near-constant behavior may stem from GPU parallelism saturating across tested D values rather than an intrinsic algorithmic property. Empirical curve-fitting over an unspecified D range is not a substitute for an architectural argument.

- **Minimality conclusion in Figure 3b is technically imprecise.** The paper states the "average decrease of 0.313 means that the returned set cannot be further reduced." This is a leave-one-out argument — it shows each individual feature is necessary, but does not rule out joint redundancies where removing a pair simultaneously would not hurt performance. The claim of a "minimal" set is technically unsupported by this analysis alone.

- **No statistical test for metagenomic improvements.** The +0.7 pp (MLP) and +1.2 pp (RF) average improvements across 24 metagenomic datasets are reported without a significance test (e.g., Wilcoxon signed-rank). Table 2 shows multiple datasets where AutoNFS underperforms (e.g., KeohaneDM, JieZ, FengQ, ZhuF, ThomasAM_2018b). Without a test, these average differences cannot be claimed to be significant.

### Trivial

- The abstract claims AutoNFS "consistently outperforms both classical and neural FS methods." Whether neural FS baselines (STG, LassoNet, Concrete Autoencoders, INVASE — discussed at length in related work) are included in the 10 compared methods is not stated in the main text. This should be clarified explicitly.

---

## Nice-to-Haves

- Provide a version of the comparison where at least one strong baseline (e.g., STG or Concrete Autoencoder) has its k cross-validated, isolating whether AutoNFS's gains come from mechanism quality or k-determination.
- Ablate f(e) vs. direct logit training (w ∈ R^D trained directly) to verify the masking network contributes beyond parameterization convenience.
- A sensitivity analysis showing how k varies as λ is changed across a range (0.1–10) would clarify the robustness of the λ = 1 default claim.
- Statistical test (Wilcoxon signed-rank) over 24 metagenomic datasets for MLP and RF.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Reviewer: "Automatic" claim is structurally false** — Partially removed from Fatal tier. The concern is legitimate and retained as Major, but downgraded: the paper does determine k through optimization rather than direct specification; the issue is that λ substitutes for k as the effective budget parameter. This is worth flagging prominently but does not fully invalidate the contribution.

- **Strength Finder: "Near-constant computational scaling validates scalability"** — Moved to Nice-to-Have / weakened: the α ≈ 0.08 claim is empirical and architecturally unexplained in the main text; it cannot be accepted uncritically as evidence of a fundamental algorithmic advance.

- **Strength Finder: "Reproducibility and integration with Cherepanova et al. codebase"** — Removed as too generic (merely citing that code is shared and that the benchmark codebase was extended doesn't constitute a scientific strength).

---

## Novel Insights

The most interesting observation synthesized across reviewers is the tension between two competing framings of AutoNFS's contribution: (1) as a *feature selection mechanism* (which features to pick), and (2) as a *feature budget estimator* (how many features to pick). The experimental setup, by fixing baselines to k = D_original, primarily demonstrates that AutoNFS excels at (2), but conflates (2) with (1). The correct test for (1) — comparing AutoNFS against a baseline given the same k that AutoNFS chose — is absent. Interestingly, if one accepts the Cherepanova benchmark protocol as standard, AutoNFS's automatic k-determination is genuinely the key innovation, and the benchmark result legitimately shows that fixing k to the original feature count is suboptimal. The more precise and defensible story for this paper is not "better selector" but "the first practically useful automatic-k neural selector," which would actually be a compelling and honest contribution.

---

## Suggestions

1. Reframe the core claim from "automatically determines the minimal set" to "eliminates the need to specify k by replacing it with a regularization weight λ that can be fixed at 1 across datasets" — this is honest and still compelling.
2. Add one comparison where a strong baseline (STG or Concrete Autoencoder) has k tuned by cross-validation, to isolate selection mechanism quality from k-determination advantage.
3. Add an ablation table comparing f(e) → mask vs. direct w ∈ R^D → mask (i.e., remove the masking network, keep only the trainable logit vector with Gumbel-Sigmoid). This would clarify the role of the masking network.
4. Add a Wilcoxon test over the 24 metagenomic datasets.
5. Clarify in the main text which of the 10 baselines are neural FS methods.

---

## Score and Decision

**Calibration anchors retrieved:**

| Paper | Avg Score | Relevance |
|---|---|---|
| `lt6xKGGWov.md` — Feature selection with neural MI estimation | 2.33 | Low anchor — feature selection paper, far weaker (no real datasets, poor presentation, unsound method) |
| `Ai4L058yoO.md` — Unsupervised Feature Selection vs. Extraction | 4.50 | Medium-low anchor — feature selection comparison paper; broader scope but weaker methodology than AutoNFS |
| `1MRfyGLCcU.md` — Graph-enhanced Semi-supervised Feature Selection | 4.75 | Medium anchor — end-to-end neural feature selector; comparable contribution scope but weaker benchmark |
| `xtTut5lisc.md` — EASE iterative feature space optimization | 5.00 | Medium anchor — feature space optimization, good experiments but presentation issues; comparable overall quality to AutoNFS |
| `0SOhDO7xI0.md` — DeepDRK knockoff-based feature selection | 5.75 | Medium-high anchor — more theoretically grounded FS paper, strong experiments; rejected at 5.75 |
| `vNdOHr7mn5.md` — Deep Weight Factorization sparse learning | 7.00 | High anchor — sparse neural learning with theory + strong experiments; substantially stronger theoretical contribution |

**Positioning:** AutoNFS is clearly above the low anchor (lt6xKGGWov). It is stronger than EASE (5.0) and the semi-supervised FS paper (4.75) in terms of experimental breadth and clarity, but shares with them the weakness of evaluation ambiguity and overclaimed contributions. It is below DeepDRK (5.75) in rigor and below Deep Weight Factorization (7.0) in theoretical depth. The two Major weaknesses — partially misleading "automatic" claim and the baseline comparison that does not isolate mechanism quality — are real and substantive, but they don't invalidate the core results. The paper has genuine value in a well-established benchmark setting and is more comprehensive than most feature-selection papers in the corpus.

**Score: 4.5** — The experimental setup provides evidence for AutoNFS's k-determination advantage but does not cleanly establish superiority of the selection *mechanism* itself. Combined with the overclaimed "automatic" framing, this sits in the lower-borderline range, below DeepDRK (rejected at 5.75) and comparable to EASE/graph-FS papers.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>