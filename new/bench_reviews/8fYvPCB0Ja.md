## Summary
This paper studies fairness in dataset distillation and argues that standard matching-based DD methods can inherit and even amplify protected-attribute imbalance in the source data. It proposes FairDD, a simple plug-in objective that replaces whole-distribution matching with protected-attribute-wise synchronized matching, and shows large empirical improvements in fairness metrics across several matching-based DD methods and benchmarks.

## Strengths
- **Important and timely problem formulation.** The paper identifies a genuinely meaningful failure mode of dataset distillation: condensed datasets can worsen protected-attribute bias rather than merely preserve it. This is well motivated in Sec. 1 and supported empirically in Table 1, where vanilla DD often has much worse DEO than training on the full dataset.
- **Simple, modular method.** FairDD is easy to understand and easy to graft onto existing matching-based DD objectives: the core change is replacing the aggregate class-level target in Eq. (4) with a sum over PA-wise targets in Eq. (7). The method does not require architecture changes and is demonstrated on DM, DC, IDC, and DREAM.
- **Strong empirical gains over vanilla DD baselines.** The fairness improvements are consistently large in Table 1, often dramatic, and accuracy is usually improved rather than hurt relative to the vanilla DD counterpart in Table 2. The cross-architecture results in Table 3 also suggest that the distilled sets retain some transferability beyond the architecture used for distillation.
- **Good evaluation instinct.** The paper does not only optimize/report fairness; it also tracks target-task accuracy, includes cross-architecture evaluation, and provides ablations on bias ratio and initialization, which is the right direction for evaluating distilled datasets.
- **Clear main mechanism.** The paper’s central insight—that matching to the overall class distribution can bias synthetic data toward majority PA groups, while per-group matching can counteract that—is coherent and well explained.

## Weaknesses

###: Fatal
None.

### Major:
- **The paper overstates what its theory proves about fairness.**  
  The theoretical results are about stationary points of the matching objective and the location of the **mean target signal** of synthetic samples (Theorem 4.1 / Eq. 8), plus an upper-bound relation between losses under convex distances (Theorem 4.2 / Eqs. 9–10). But the experimental fairness metric is **DEO of downstream classifier predictions** (Sec. 2), and the paper does not establish a theorem linking equal weighting of PA-wise signal means to equalized-odds fairness of the trained classifier. So claims such as “We also provide theoretical analyses to guarantee the fairness and accuracy of synthetic samples” and the conclusion’s “FairDD guarantees the fairness of synthetic datasets” are stronger than the actual theory supports.
- **The experimental design does not isolate whether the gains come from synchronized matching specifically, versus simply injecting PA-aware balancing into distillation.**  
  FairDD is compared primarily against vanilla DD and random sampling. But since the method explicitly uses PA labels and gives all PA groups equal weight in Eq. (7), a key missing control is a simpler PA-aware baseline, such as balanced subset selection, group-reweighted matching, or other straightforward PA-aware balancing during distillation/training. Without such controls, the experiments establish that **using PA information in distillation helps a lot**, but do not cleanly establish that the specific synchronized matching formulation is the decisive ingredient.
- **The empirical scope is still somewhat narrow for the paper’s broad framing.**  
  Most of the evidence comes from constructed bias settings: C-MNIST, C-FMNIST, and grayscale CIFAR10-S, where the PA is deliberately injected and test distributions are designed for fairness measurement. These are useful stress tests, but the main-paper real-world evidence is limited to CelebA with one TA/PA setup. The method looks promising, but the broad framing as a general fair dataset distillation framework for image recognition is supported more strongly on synthetic spurious-correlation benchmarks than on realistic fairness settings.
- **The core method requires protected-attribute annotations, and the paper’s framing should foreground this more clearly.**  
  This is not a fatal flaw—the paper openly acknowledges it in the limitation section (lines 534–536)—but it is a material scope condition because the method’s mechanism is to partition data by PA and optimize Eq. (7). Some claims in the abstract/introduction are phrased broadly (“fair dataset distillation,” “regardless of PA imbalance in the original data”) and would be more precise if they explicitly said this is in the setting where PA labels are available during distillation.

### Minor
- **The theoretical interpretation in Theorem 4.2 is too strong.**  
  The proof shows that \(\mathcal{L}_{\text{FairDD}}\) upper-bounds the vanilla matching loss under convex \(\mathcal D\), but the text then states that minimizing this upper bound “can guarantee the comprehensive distribution coverage” and even “ensures the minimization” of the original objective. That interpretation is stronger than what an upper-bound relationship alone typically justifies.
- **Some mechanistic claims are more intuitive than rigorous.**  
  For example, Sec. 4.1 describes a “pull-and-push” process, but Eq. (7) only contains attraction terms to group-wise targets; there is no explicit repulsion term. Likewise, the argument in Sec. 3 from weighted expectation matching to claims about sample-level collapse into majority groups is plausible, but stronger than what the equations strictly prove.
- **Variance/stability reporting is missing from the main paper.**  
  Dataset distillation is often sensitive to initialization and optimization randomness, and the paper reports point estimates only. Given the very large gains, this omission does not erase the empirical signal, but some measure of variability would make the conclusions more robust.
- **Some explanations of accuracy behavior are speculative.**  
  In the cross-architecture discussion, the explanation for why larger models can become fairer yet less accurate is plausible, but not empirically validated in the paper.

### Trivial
- **Main-text ablation presentation is slightly inconsistent.**  
  The text says the BR study uses \(\{0.85,0.90,0.95\}\), but the displayed main-paper table excerpt shows only 0.85 and 0.90. This is minor, but the presentation should align with what is shown in the main text.

## Nice-to-Haves
- Add PA-aware control baselines, especially simple ones: balanced subset sampling by (TA, PA), inverse-frequency/group-reweighted matching, and possibly fairness-aware training on top of a vanilla distilled set.
- Expand the real-world evaluation beyond one CelebA attribute pair, ideally including settings with multi-valued or intersectional protected attributes.
- Report mean/std across multiple runs for the main fairness and accuracy results.
- Include a more explicit empirical analysis of when FairDD helps most, especially in cases with many PA groups or weakly separable biases.
- Clarify the relationship between the theoretical surrogate guarantees and downstream EO-style fairness, ideally by weakening the wording if no stronger theorem is available.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The method is unfairly advantaged because it uses PA labels while baselines do not.”**  
  Removed as a main weakness in that form. This is not by itself a valid criticism: giving the proposed method extra supervision relevant to its stated goal is acceptable. The real issue is not asymmetry per se, but the absence of **PA-aware control baselines** that would isolate which part of the supervision matters.
- **“The paper should evaluate on TM / MTT and its exclusion invalidates generality.”**  
  Weakened/removed as a core criticism. The paper explicitly scopes itself to matching-based DD in the DMF and states why TM is excluded: “doing so would require extra model trajectories trained on minority groups...” This is a limitation, but not a fatal contradiction. It is fair to ask for narrower wording, not to treat TM omission as a decisive flaw.
- **“The visualizations prove / disprove the method.”**  
  Removed as a substantive weakness. The paper uses t-SNE/feature plots illustratively; while causal language around them should be toned down, their mere use is not a serious flaw.
- **Pure novelty complaint that the method is ‘just reweighting’ and therefore not a contribution.**  
  Weakened. The underlying balancing idea is indeed simple, but simplicity is not a flaw by itself, and in this paper the important contribution is also the problem formulation and its adaptation to DD.

## Novel Insights
The most interesting synthesis here is that FairDD’s strongest contribution is less the formal theory and more the empirical reframing of dataset distillation as a **fairness-sensitive data-generation process** rather than a pure compression problem. The paper convincingly shows that once protected-attribute imbalance exists, standard DD objectives can fail in a way that harms both fairness and even target accuracy, suggesting that DD objectives are more distribution-shaping than prior work may have acknowledged. At the same time, the paper’s dramatic gains over vanilla DD likely indicate that biased DD benchmarks can destabilize standard matching objectives much more severely than expected, so FairDD may be functioning both as a fairness intervention and as a representation-coverage correction. That dual role is scientifically interesting and worth making explicit.

## Suggestions
- Narrow the main claim language so it matches the evidence: present FairDD as a method for **PA-supervised fair dataset distillation for matching-based DD**, rather than implying a broader setting.
- Rephrase the theory claims conservatively. Theorems 4.1–4.2 support properties of the surrogate objective and group-balanced signal matching; they do **not** prove equalized-odds fairness of downstream classifiers.
- Add simple PA-aware baselines to establish whether synchronized matching beats more naive balancing strategies.
- Strengthen the real-world case with more natural-bias datasets or more attribute settings on CelebA.
- Report multi-run statistics for the main tables.
- Discuss more explicitly when representing all PA groups with a single \(\mathcal S_y\) may become difficult, e.g., many groups or multimodal/far-apart group distributions.

## Score and Decision
**Assessment by axis:**  
- **Originality:** Moderate. The fairness-in-DD problem framing is novel and important, while the method itself is simple and conceptually close to group balancing/reweighting ideas.  
- **Importance of the research question:** High. Fairness in distilled datasets is a meaningful and underexplored problem.  
- **Whether the claims are well supported:** Moderately supported. The core empirical claim that FairDD improves fairness over vanilla DD is well supported; the strongest theoretical and generality claims are not.  
- **Soundness of experiments:** Moderate. The experiments are broad within matching-based DD and the effect sizes are large, but missing PA-aware control baselines and limited real-world coverage prevent stronger conclusions.  
- **Clarity of writing:** Generally clear, though some claims are stronger than warranted.  
- **Value to the research community:** Good. Even if imperfectly scoped, this paper identifies an important issue and offers a simple, effective baseline that future work will likely need to consider.

**Calibration against human-reviewed anchors:**  
- I compared this paper most directly against **L5yq5KsnEZ (Mitigating Bias in Dataset Distillation; scores 5/3/5/5, reject)**, which also studies bias amplification in DD with large gains but was held back by limited validation and questions about the method’s incremental nature. FairDD is somewhat stronger than that anchor because it has a cleaner plug-in formulation, broader evaluation across multiple DD backbones, and clearer fairness framing.  
- I also compared it to **3JsU5QXNru (Group Distributionally Robust Dataset Distillation; scores 6/8/6/6, accept)**, which addressed subgroup robustness in DD with stronger perceived theoretical grounding and broader robustness framing. The current paper is below that anchor because its theory is overclaimed relative to what it proves, and its experiments do not isolate the proposed mechanism with the right control baselines.  
- Relative to higher-scoring dataset distillation papers such as **h57gkDO2Yg (scores 6/8/6/5/6, accept)**, this submission has a compelling new problem and strong empirical signal, but weaker support for its strongest claims and a less complete evaluation design.

Overall, this places the paper in the **borderline but slightly below accept** range for me: promising, useful, and likely influential, but not yet fully convincing in its current form.

**Score: 5.5 / 10**  
**Decision: Reject**

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>