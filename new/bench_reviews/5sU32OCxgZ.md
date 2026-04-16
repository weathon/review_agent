## Summary
The paper proposes TTVD, a test-time adaptation (TTA) framework that reinterprets neighbor-based adaptation through Voronoi diagrams in feature space. Building from a basic Voronoi-based entropy loss (VD), it introduces Cluster-induced Voronoi Diagrams (CIVD) to incorporate self-supervised augmentations as multiple “sites” per class, and Cluster-induced Power Diagrams (CIPD) to add weights and a boundary-based filtering heuristic for noisy samples. Evaluations on CIFAR-10-C/100-C, ImageNet-C, and ImageNet-R (via TTAB) show consistent improvements over several existing TTA baselines in both error and calibration.

## Strengths
- **Clear and reasonably principled base objective (VD).** Section 3.1 defines a simple, reproducible loss: a softmax over negative distances to class means with entropy minimization (Eq. 3), updated on the feature extractor only (Algorithm 1). This connects cleanly to a Voronoi partition view and is easy to implement.
- **Strong empirical performance under a standardized benchmark.** Using TTAB with grid search and oracle/non-oracle reporting, TTVD attains the best error and ECE on CIFAR-10-C, CIFAR-100-C, ImageNet-C, and ImageNet-R (Table 1). The gains on harder settings (e.g., 1.6% error and large ECE reductions on ImageNet-C/-R) are meaningful in the TTA context.
- **Progressive ablation of geometric variants.** Table 2 systematically compares VD, CIVD, and CIPD on CIFAR-10-C; even the basic VD variant beats prior neighbor-based methods, and each added structure yields further gain (≈5.7% from VD→CIVD, ≈2.2% from CIVD→CIPD).
- **Thoughtful use of computational geometry concepts.** The link from linear classifiers to power diagrams (Lemma 3.1) is correctly invoked, and the use of CIVD/CIPD extends earlier Voronoi-based work into the TTA setting. The 2D visualizations on MNIST-C (Figs. 1–2) make the geometric intuition accessible.
- **Broad robustness checks (though mostly in appendix).** The paper examines adaptation curves over time (Fig. 4), robustness to class-mean estimation with subsampled ImageNet data (Table 4), and mentions experiments under varying batch size and label shift (Appendix), supporting the claim that TTVD is reasonably stable in realistic online settings.

## Weaknesses

### Fatal
None. The paper proposes a coherent algorithm with clear base components and solid empirical benefits; there is no evidence of a flaw that invalidates its main empirical claims.

### Major
- **Underspecified CIVD/CIPD objectives and “unified” use of self-supervision and entropy minimization.**  
  CIVD and CIPD are central to the claimed contributions (multi-site influence, unified objective, mitigation of negative transfer), but the exact test-time loss and update rules for these variants are not spelled out in the main text:
  - For CIVD (Sec. 3.2), the influence function \(F(z, C_k)\) in Eq. (4) and the definition of clusters via rotated augmentations are given, but the corresponding adaptation loss (the analogue of \(\mathcal{L}_{VD}\) in Eq. (3)) is not explicitly written. The text says “similar to Equation 3, the soft label given by CIVD can be calculated from the influence function… The joint label \(\tilde{y}_k^{(\alpha)}\) avoids the negative transfer since the objective is now unified”, but does not show whether they minimize entropy of \(\beta(F(z,C_k))\), average over \(\alpha\), or use some other construction.
  - For CIPD (Def. 3.4, Eq. (6)), the influence incorporates weights \(v_k\), but the paper does not specify in the main body how \(v_k\) are obtained in practice (fixed from the pretrained linear head via Lemma 3.1, updated during TTA, or something else). Nor is the exact form of the loss given. It is stated only that “we infer and adapt the model accordingly by CIPD (Algorithm 3 in Appendix H) using Equation 6.”
  - The claim that CIVD “unifies self-supervision and entropy minimization” to avoid negative transfer is not given a concrete mathematical formulation (no explicit joint loss), and the mechanism is not compared to a straightforward multi-task combination of rotation loss + entropy loss.
  
  As a result, a reader cannot fully reconstruct what is optimized for CIVD/CIPD at test time or rigorously assess the “unified objective” claim. The empirical ablation (VD→CIVD→CIPD) is thus somewhat opaque: we know performance improves, but we do not know precisely which change in the loss or gradient field is responsible.

- **Power diagram–based noisy sample filtering is conceptually underdefined and not isolated experimentally.**  
  Section 3.3 attributes CIPD’s advantage to using PD’s flexible boundaries to identify and filter noisy samples via “diagram subtraction”:
  - The phrase “By subtracting the PD from the VD, we can extract a larger region from the resulting differences…” is not formalized. It is unclear whether samples are filtered based on disagreement between VD and PD cell assignments, set differences of regions, or some other criterion; Fig. 2b is illustrative but not algorithmic.
  - While Fig. 2a shows high entropy near VD boundaries, there is no rigorous argument that the additional regions flagged by PD–VD subtraction correspond to truly harmful samples rather than simply harder points, nor is there a guarantee they correspond to high-gradient-noise regions.
  - There is no dedicated ablation that isolates PD-based filtering. Table 2 conflates (i) moving from VD to cluster-based CIVD, and (ii) adding PD-based weights and filtering in CIPD. We never see, for example, CIVD with entropy-based filtering vs CIVD with geometric PD-based filtering under matched filtering rates.
  
  Since the narrative repeatedly highlights PD/CIPD as a geometric cure for “noisy samples” and “negative model updates,” the lack of precise definition and targeted experiments substantially weakens that conceptual contribution, even though overall performance gains remain.

- **Geometric reinterpretation of existing neighbor-based TTA methods is only partially justified.**  
  The paper states that “the underlying geometric structure of these neighbor-based methods is Voronoi Diagram” and uses this to motivate TTVD as a more advanced geometric framework. However:
  - The mapping from concrete methods like T3A, TAST, or AdaNPC to an explicit VD or PD partition is not derived; it is asserted at a high level. For instance, T3A’s dynamic update of prototypes and thresholds does not correspond directly to the fixed-site VD in Eq. (2)–(3).
  - Lemma 3.1 connects linear classifiers to power diagrams, but the interaction between updating only BN affine parameters under \(\mathcal{L}_{VD}\) and the resulting partition of feature space is not made explicit.
  
  This does not affect the correctness of TTVD as a new TTA algorithm, but it does mean that the paper overstates the extent to which it provides a unified geometric theory of existing neighbor-based methods.

- **Key mechanistic claims are not empirically tested.**  
  Beyond aggregate error/ECE improvements, the specific reasons the method is claimed to work are not directly validated:
  - The “avoiding negative transfer via unified objective” is not evaluated against a naïve joint rotation + entropy loss baseline, nor is any gradient-conflict analysis presented.
  - The claim about PD-based filtering improving robustness to noisy samples is not backed by experiments that contrast different filtering policies.
  - The discussion of overfitting/negative model updates in Fig. 4 is somewhat speculative; the curves show TTVD improving and TENT/SAR relatively flat under TTAB-tuned hyperparameters, but they do not directly exhibit the classical overfitting behaviour the introduction describes, nor show TTVD under adverse hyperparameters to demonstrate superior stability.
  
  Consequently, the paper is strong as an empirical TTA method but weaker as an explanatory geometric framework.

### Minor
- **Limited clarity on some implementation choices.**  
  A few design decisions that plausibly matter are only sketched:
  - Equation (4) uses distance to the 7th power; \(\gamma\) is described as a “scale” parameter, but why exponent 7 (and the sign convention) is chosen is not justified, nor is any sensitivity analysis shown.
  - The text says “Similar to Equation 3, the soft label given by CIVD can be calculated from the influence function, incorporating the expanded sites \(\mu_k^{(\alpha)}\)”, but does not state explicitly whether the softmax is over aggregated influences, separate augmented samples, or some pooled structure.
  - For PD/CIPD, while Lemma 3.1 suggests a way to map from classifier parameters to \((\mu_k, v_k)\), the paper does not specify whether this mapping is actually what is used in the experiments, or whether \(v_k\) remain fixed from pretraining or change during adaptation.

- **Base VD formulation is conceptually closer to known prototype-based TTA/SFDA approaches than acknowledged.**  
  Equation (3) effectively enforces concentration of features around class means with an entropy-like loss, which is quite similar in spirit to prototype-based pseudo-labeling and feature-clustering methods such as SHOT. The paper mentions SHOT as a baseline but does not situate VD relative to this prior line of work, leaving some novelty on the algorithmic side more incremental than the geometric framing implies.

- **Ablations limited to CIFAR-10-C.**  
  The important VD/CIVD/CIPD decomposition in Table 2 is only reported for CIFAR-10-C. While the main results on other datasets are strong, we do not see whether the relative contributions of CIVD vs CIPD carry over to CIFAR-100-C or ImageNet-C, or whether, for example, CIPD’s gains shrink on higher-resolution and more complex datasets.

- **No statistical variability reported.**  
  Table 1 and Table 2 present single-point estimates without error bars or confidence intervals. Some reported gains (e.g., ≈0.7% error) are modest, and it is hard to tell how robust they are across runs. This does not affect reproducibility (hyperparameter choices are reasonably described), but it does limit how strongly one can interpret “remarkable improvements.”

- **Computational overhead is not quantified.**  
  The paper notes that computing class means for ImageNet using 10% of training data takes less than 10 minutes, which is helpful, but there is no discussion of per-batch adaptation cost when using CIVD/CIPD (with multiple augmented sites per class and weighted distance computations), nor wall-clock comparisons to baselines. In practice this might matter for real-time deployment.

### Trivial
- **Scope limitations.**  
  The method inherently relies on class prototypes and classification heads; it does not directly address non-classification tasks (e.g., detection/segmentation). This is a natural scope choice rather than a flaw, but a short explicit remark would set expectations for applicability.
- **Some qualitative claims read stronger than warranted.**  
  Phrases like “avoids negative transfer” or “particularly effective in identifying noisy samples” are somewhat stronger than the evidence provided. Softening these to “helps mitigate” or “appears to improve” would better match the current experiments.

## Nice-to-Haves
- A clearly written main-text description of the CIVD and CIPD losses and updates (even if full pseudocode remains in the appendix) would substantially improve clarity and perceived rigor.
- A synthetic or low-dimensional controlled experiment where one can explicitly see CIVD’s multi-site influence rescuing points that a single-site VD misclassifies, and PD-based filtering excluding specific known noisy points, would strengthen the geometric narrative.
- Basic hyperparameter sensitivity plots (e.g., for temperature \(\tau\) and the influence exponent) would reassure readers that performance is not overly brittle to tuning.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“Source-data dependence of class means undermines TTA setting.”**  
  Some prototype-based TTA reviews worry that needing source data to compute prototypes violates a strict TTA assumption. Here, the paper explicitly uses the full CIFAR training set and 10% of ImageNet to compute class means and clearly states this setup (Sec. 4.1). It also shows robustness to using only 1%–10% of ImageNet data (Table 4). Within this paper’s stated setting (access to training features/prototypes, not raw labels at test time), this is not a correctness issue; it is a design choice. So concerns premised on “no source data can be accessed at all” do not apply and are removed.
- **“Baseline set is missing more recent methods, invalidating SOTA claims.”**  
  The harsh concern about missing very latest baselines cannot be verified from the paper text alone (we do not have an external list of all contemporaneous TTA methods), and the paper does compare against a broad, diverse set (neighbor-based, entropy-based, repurposed DA) standardized by TTAB. Without concrete evidence from the paper that an obviously stronger method was ignored, this criticism is speculative and removed.
- **“Identical numbers in Table 4 suggest sites have no effect at all.”**  
  Table 4 shows error changing from 59.8 to 59.9 when using less data to estimate class means. While the similarity is noteworthy, interpreting it as proof that prototypes have no effect goes beyond the data. It more reasonably suggests that 1% of ImageNet is sufficient to estimate class means well enough for this method; without further analysis, stronger inferences are unwarranted and removed.
- **“Generalization to non-classification tasks is required.”**  
  Some generic concerns about lack of applicability to detection/segmentation exceed the stated scope of the work, which clearly targets classification TTA on CIFAR/ImageNet variants. Demanding broader task support as a weakness would be scope creep, so this is reclassified as a minor scope note rather than a core flaw.

## Novel Insights
The strongest novel insight is that even a very simple, explicitly geometric distance-to-centroid entropy loss (VD) can already outperform more elaborate neighbor-based TTA methods when tuned fairly under a standardized benchmark, and that adding multi-site influences (via self-supervised augmentations) and weighted partitions (via power diagrams) produces further gains without changing the core optimization paradigm. Nonetheless, while the geometric lens is conceptually appealing, the paper’s current formulation does not yet fully substantiate the stronger ambition of a unified geometric theory of neighbor-based TTA.

## Suggestions
- **Clarify the CIVD/CIPD objectives in the main text.**  
  Add explicit equations for:
  - The CIVD-based prediction \(\tilde{y}_k\) (how \(F(z, C_k)\) feeds into a softmax and how the multiple \(\alpha\) are combined), and the exact loss minimized at test time;
  - The CIPD loss, including how weights \(v_k\) are computed from the pretrained classifier and whether they are fixed or updated.
  This should be visible without relying on the appendix.

- **Formally define the PD–VD filtering rule and ablate it.**  
  Specify something like: “we drop samples where the class assigned by VD and CIPD disagree” (or whatever is actually used), and then run an ablation comparing:
  - CIVD without any filtering,
  - CIVD with standard entropy-based filtering at a comparable retention rate,
  - CIVD with the proposed geometric filtering.
  This would directly test the noisy-sample filtering story.

- **Evaluate CIVD/CIPD contributions on at least one more dataset.**  
  Extend the VD/CIVD/CIPD ablation beyond CIFAR-10-C to at least CIFAR-100-C or ImageNet-C to show that the decomposition observed in Table 2 is not dataset-specific.

- **Add a simple joint-Objective baseline.**  
  Implement a baseline that directly minimizes \(\mathcal{L} = \lambda_1 \cdot \text{entropy}(p(y|x)) + \lambda_2 \cdot \text{rotation loss}\) with shared parameters during TTA. Comparing this to CIVD under matched hyperparameters would demonstrate whether the geometric “unification” helps beyond straightforward multi-tasking.

- **Include basic sensitivity and runtime analyses.**  
  Provide at least one plot showing performance vs. temperature \(\tau\) (and possibly the exponent in Eq. (4)), and a table with adaptation time per 1k test samples for TTVD vs a representative baseline, to document robustness and overhead.

- **Tone down or qualify some mechanistic claims.**  
  Adjust wording around “avoids negative transfer” and “identifying noisy samples near decision boundaries” to reflect that the current evidence is suggestive but not definitive, or add the experiments above to substantiate them.

### Overall evaluation on key axes
- **Originality:** Moderate. The base VD loss is close to known prototype-based objectives; the use of CIVD/CIPD for TTA is a reasonably novel transfer of existing geometric structures.
- **Importance of question:** High. Test-time adaptation under distribution shift on standard robustness benchmarks is a central and active topic.
- **Support for claims:** Strong for aggregate performance; weaker for the specific mechanistic/geometric claims (unified objectives, noise filtering).
- **Soundness of experiments:** Generally sound in design and breadth; missing some ablations targeting specific claims and no statistical variability.
- **Clarity of writing:** Good at a high level and for VD; less clear for CIVD/CIPD implementation and filtering details.
- **Value to the community:** Good. Even if some conceptual claims are overstated, TTVD as an algorithm and the geometric framing are likely to spur follow-up work.

## Score and Decision

**Calibration references (from provided human reviews):**
- Prototype-based / Voronoi-like TTA papers with mixed clarity and modest gains (e.g., eXrUdcxfCw.md, LQDJO7txyN.md) received scores mostly in the 5–6 range when contributions were incremental or underspecified.
- Stronger TTA conceptual papers with well-substantiated mechanisms and solid experiments (e.g., 9w3iw8wDuE.md “Entropy is not Enough for TTA”) received higher scores (6–8).
- Papers criticized for underspecified objectives and missing mechanistic validation tended to fall near 5.

Relative to these:
- This paper’s empirical performance is more convincing than many mid-range prototypes papers (better benchmark coverage, calibration metrics, TTAB adherence), pushing it above the weakest of those (3–4 range).
- However, the underspecification of its main advanced components (CIVD/CIPD) and the lack of targeted tests for central claims keep it below the strongest, more rigorous TTA works (7–8 range).

Balancing strong empirical value against the conceptual/clarity gaps, a **score around 6.0** seems appropriate: a solid paper with meaningful contributions but not yet at the level of clarity and mechanism-substantiation expected for acceptance in a top-tier venue.

Given the instructions to weigh fundamental issues over averaging, and here there is no fatal flaw—just underspecified but plausibly fixable components—I lean slightly positively but note that a revision would be highly beneficial.

MY FINAL SCORE: <pineapple>6.0</pineapple>  
MY FINAL DECISION: <orange>Accept</orange>