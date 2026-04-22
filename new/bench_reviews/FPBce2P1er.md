Now I have a good understanding of the paper and the calibration anchors. Let me now carefully verify the key criticisms from the harsh reviewer against the actual paper content.

Key criticisms to verify:

1. **No experimental test of non-conjunction-wise-additive tasks in DNNs**: The paper tests only symbolic addition and context dependence (both conjunction-wise additive). The theory's most distinctive prediction is failure on transitive equivalence. Verified from the paper: Section 6 only tests symbolic addition and context dependence. This is a valid concern.

2. **Kernel regime assumption unvalidated for DNN experiments**: The theory depends on compositional structure + kernel regime. The DNNs are trained with backpropagation, not tested for kernel regime. The paper discusses this in Section 2 and the discussion acknowledges limitations. Valid concern but the paper does partially address this (measures S(1;2) in ConvNet layers), and the Discussion acknowledges the kernel model limitation.

3. **Apparent conjunction-wise additivity confounded by task structure**: Testing additive tasks shows additive behavior — but any reasonable model on an additive task would behave additively. The paper actually acknowledges this partially ("at least when trained on conjunction-wise additive tasks", line 216). This is a valid concern but the paper partially addresses it.

4. **Zero-mean assumption in Proposition 5.1**: The paper does state this assumption explicitly. This is a substantive assumption but it's clearly stated. Minor concern.

5. **Qualitative vs quantitative validation**: The paper shows qualitative trends matching Proposition 5.1 but doesn't plug measured S(1;2) into the formula for quantitative comparison. Valid concern.

Now let me synthesize my review.

---

## Summary

The paper develops a theory of compositional generalization for kernel models with compositionally structured representations, proving that such models are restricted to "conjunction-wise additive" computations (Theorem 4.2) — they can only sum values assigned to conjunctions of components seen during training. This yields a clean partition of the compositional task space into tasks kernel models can and cannot solve (e.g., transitive ordering is solvable, transitive equivalence is not). The paper further characterizes two failure modes on solvable tasks: memorization leak (Proposition 5.1 gives a closed-form formula) and shortcut bias (dependent on representational salience and training statistics). Finally, the paper empirically validates qualitative predictions of the theory in ConvNets, ResNets, and ViTs on conjunction-wise additive tasks.

## Strengths

- **Theorem 4.2 provides a clean, non-trivial characterization** of what compositionally structured kernel models can express on test inputs (Eq. 2). The conjunction-wise additivity constraint is precise and has clear, important consequences — most notably the impossibility of transitive equivalence generalization while transitive ordering remains solvable (Section 4.3). This impossibility result is the paper's most distinctive contribution.

- **Proposition 5.1 gives an exact, testable formula** for the memorization leak slope as a function of representational salience S(1;2) and training set size p (Eq. 3). The non-obvious prediction that interpolation vs. extrapolation does not systematically affect the slope is particularly interesting and empirically confirmed in DNNs (Fig. 5d).

- **The decomposition into two distinct failure modes** — memorization leak (a quantitative distortion) and shortcut bias (a binary success/failure driven by representational geometry and training statistics) — provides a principled, theory-grounded framework for understanding why compositional generalization fails, rather than post-hoc interpretation.

- **Proposition 4.1 grounds the theory in concrete neural architectures**, proving that infinite-width random networks preserve compositional structure (Appendix A.3), ensuring the theory's assumptions are not vacuous. The introduction of representational salience S(k;C) as a compact descriptor (reducing C+1 parameters to C-1 free parameters) makes the theory's predictions tractable and interpretable.

## Weaknesses

### Fatal
None.

### Major

- **The DNN experiments do not test the theory's most distinctive prediction.** The theory's strongest and most novel implication is that compositionally structured kernel models *cannot* solve non-conjunction-wise-additive tasks, most notably transitive equivalence (Section 4.3). Yet all DNN experiments (Section 6) test only conjunction-wise-additive tasks (symbolic addition and context dependence) — tasks the theory says kernel models *can* solve. Showing that DNNs behave conjunction-wise additively on additive tasks is weak evidence because any model that generalizes reasonably on an additive task would appear conjunction-wise additive, regardless of whether it is fundamentally constrained to be so. As the paper itself acknowledges (line 216), DNNs implement conjunction-wise additive computations "at least when trained on conjunction-wise additive tasks." Without testing whether DNNs also fail on tasks the theory says are impossible (e.g., transitive equivalence), the claim that the theory "captures the behavior of deep neural networks" (Abstract, Section 6 title) is overstated. The difference between validating the theory's explanatory power and showing consistency on easy cases is significant.

- **The kernel regime and compositional structure assumptions are unvalidated for DNNs.** The theoretical results depend on two strong assumptions: (a) the model operates in the kernel regime (no feature learning; Section 2), and (b) representations are compositionally structured (Definition 3.1). The DNNs in Section 6 are trained with backpropagation (not in the kernel regime) and their representations are not verified to be compositionally structured. The paper measures S(1;2) in ConvNet intermediate layers (Fig. 5a), which is a useful diagnostic, but measuring S(1;2) is only informative *if* the representation is compositionally structured — otherwise the metric may not capture the relevant structure. The Discussion acknowledges the theory is "limited to a particular learning mechanism (kernel models)" and that "Other learning mechanisms could overcome the limitations," but the empirical section is framed as validation rather than preliminary evidence. This gap between the theory's assumptions and the experimental setup means the qualitative matches in Figs. 5b–e are consistent with the theory but also consistent with other explanations.

### Minor

- **Qualitative trends are shown rather than quantitative predictions tested.** Proposition 5.1 gives an *exact* formula involving S(1;2) and p. The experiments show that slope increases with distance (Fig. 5c) and with training set size (Fig. 5d), but the paper does not test whether plugging measured S(1;2) values into the formula yields accurate quantitative slope predictions in DNNs. Without this, the experiments demonstrate qualitative consistency rather than strong validation. This is acknowledged in the Discussion ("we do not provide any quantitative bounds") but understates the opportunity.

- **The zero-mean assumption in Proposition 5.1** (line 176: "the average value in both V and W is zero") is substantive and could constrain the formula's applicability. When this assumption fails, the clean proportional scaling may not hold. The paper does not discuss how robust the formula is to violations of this assumption, which would clarify how broadly the memorization leak formula applies.

- **The Definition 3.1 assumption of compositional structure is highly restrictive.** It requires that the kernel depends *only* on the number of overlapping components, implying all components are symmetric and interchangeable within a component set. This excludes most learned representations except in specific setups. The paper mentions extensions (Appendices A.5, C) with preliminary evidence for randomly sampled and disentangled representations, but the gap between the restrictive theoretical setting and the empirical setting is not thoroughly discussed.

### Trivial
None.

## Nice-to-Haves

- **Test DNNs on transitive equivalence** (a non-additive task the theory says is impossible for kernel models). This is the single experiment that would most strengthen the paper: if DNNs also fail in the predicted way, it validates the theory's extension beyond kernel models; if they succeed, it clarifies the theory's limits.

- **Test quantitative predictions of Proposition 5.1 in DNNs** by plugging measured S(1;2) into the formula and comparing against observed slopes.

- **Diagnose the kernel regime assumption** by measuring how much the representation changes during training (e.g., CKA between initial and final layers) and correlating this with how well the theory's predictions hold.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **Overclaimed framing in abstract/introduction** (from Harsh Critic): The abstract states the theory "captures the behavior of deep neural networks" — this claim is partially valid as the paper does show qualitative consistency on additive tasks, but the claim is indeed too strong given the untested assumptions. However, the *specific* phrasing issue is a presentation/clarity concern that falls under the already-captured Major weakness about untested distinctive predictions, not a separate weakness about framing language.

- **"All-or-nothing generalization pattern could arise from other mechanisms"** (from Harsh Critic Section 5.2 note): This is a generic concern that any empirical phenomenon could have multiple explanations. The paper provides a mechanistic explanation (weight magnitude analysis in Fig. 4d) that is consistent with the theory. This is a standard "alternative explanation" concern that applies to any empirical paper and is already implicitly covered by the Major weakness about unvalidated assumptions.

- **"Section 7 limitations understate the problem"** (from Harsh Critic): The paper states "we do not provide any quantitative bounds" and "our theory is limited to a particular learning mechanism (kernel models)" — these are honest acknowledgments. The critique that the core issue is "lack of validation for the mechanism connecting the theory to DNNs" is already captured in the Major weaknesses above.

- **"Missing related works"** — removed per rules, as we cannot verify existence of works not cited.

## Novel Insights

The paper reveals an underappreciated asymmetry in compositional generalization: two tasks that appear superficially similar (transitive ordering vs. transitive equivalence) fall on opposite sides of the conjunction-wise additivity partition, making the case that formal analysis of task structure is essential for predicting when compositional generalization will succeed. This suggests that empirical studies reporting mixed results on compositional generalization may be confounded by not distinguishing between additive and non-additive task structures. The memorization leak phenomenon — that the same proportional distortion factor applies regardless of interpolation vs. extrapolation — is a counterintuitive prediction that challenges assumptions common in generalization research.

## Suggestions

- Restructure the DNN experiments section to test at least one non-conjunction-wise-additive task (e.g., transitive equivalence on MNIST/CIFAR-10), which would provide the clearest evidence for or against the theory's extension to DNNs. If results are mixed, this itself is a valuable finding that clarifies the theory's scope.

- Tone down the framing from "captures the behavior of deep neural networks" to "is consistent with the behavior of deep neural networks on conjunction-wise additive tasks" in both the abstract and Section 6 title, which would more accurately reflect what the experiments establish.

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| Spectral kernel generalization | /home/wg25r/review_agent/human_reviews/3SJE1WLB4M.md | 8.0 | Stronger theoretical contribution (exact, widely applicable); this paper has similar theory quality but weaker empirical bridge to DNNs |
| Provable compositional generalization (identifiability) | /home/wg25r/review_agent/human_reviews/7VPTUWkiDQ.md | 7.33 | Stronger theory-empirical alignment; this paper's theory is novel but the gap to DNNs is wider |
| Feature learning vs kernel scaling | /home/wg25r/review_agent/human_reviews/dEypApI1MZ.md | 7.2 | Similar kernel-regime limitations paper; this paper's Theorem 4.2 is cleaner but empirical validation is weaker |
| Impossibility of tight generalization bounds | /home/wg25r/review_agent/human_reviews/NkmJotfL42.md | 7.0 | Clean impossibility result, less empirical validation needed; this paper has comparable impossibility contribution but overclaims on the DNN side |
| GeoCon compositional geometric constraints | /home/wg25r/review_agent/human_reviews/dggRphAcCj.md | 6.33 | Withdrew; this paper has stronger theoretical foundations but shares the representational geometry + compositional generalization theme |
| Transformer compositional capabilities (synthetic) | /home/wg25r/review_agent/human_reviews/tHHzfZSP6T.md | 5.0 | Rejected; strong empirical but limited theoretical contribution; this paper is the reverse — stronger theory, weaker empirical DNN bridge |
| Compositional interfaces (COIN) | /home/wg25r/review_agent/human_reviews/D1w3huGGpu.md | 4.75 | Rejected; minimal novelty; this paper has substantially more theoretical novelty |
| Image-Guided Code Patch (nonsensical) | /home/wg25r/review_agent/human_reviews/OXIIFZqiiN.md | 1.5 | Clearly irrelevant (nonsensical paper) |

This paper sits between the high-scoring theory papers (7.0–8.0) and the medium-scoring ones with weaker theory-empirical connections (5.0–6.33). The theoretical contribution (Theorem 4.2 + Proposition 5.1) is genuinely solid and novel. However, the overclaimed empirical bridge to DNNs is a meaningful weakness: testing only additive tasks and not validating the theory's most distinctive prediction (failure on non-additive tasks) significantly weakens the empirical contribution. The paper is better than the medium anchors (5.0–6.33) because the theory is genuinely novel and the memorization leak formula is precise and testable, but it falls below the high anchors (7.0–8.0) because the DNN experiments primarily show consistency rather than validation, and the framing overstates what is established.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>