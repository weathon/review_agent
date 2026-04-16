## Summary
This paper proposes **NAMMD**, a normalized variant of MMD for kernel-based distribution closeness testing, motivated by the observation that equal MMD values can correspond to distribution pairs with different RKHS norms and different finite-sample testing behavior. The paper provides asymptotic analysis, Type-I error control, sample-complexity bounds, and empirical evaluations on two-sample testing, discrete closeness testing, and several dataset-shift case studies.

## Strengths
- **The paper identifies a real and nontrivial limitation of plain MMD for cross-pair comparison under a fixed kernel.** Figure 1 and the surrounding discussion make a legitimate point: for different distribution pairs, the same MMD can coexist with different RKHS norms and different estimator variance/p-values under the same kernel. The paper’s proposed normalization is simple and clearly motivated from this testing perspective.
- **The method is technically substantial rather than purely heuristic.** The submission includes asymptotic distribution results for the estimator (Theorem 2), a variance estimator (Lemma 4), Type-I error control (Theorem 5), concentration/sample-complexity results (Theorem 8), and comparison theorems against MMD (Theorems 10 and 12).
- **The proposal is computationally lightweight and compatible with standard kernel testing pipelines.** NAMMD is built from the same kernel quantities as MMD and is integrated into a fused-kernel two-sample test (NAMMDFuse), making the method easy to adopt where MMD-style tests are already used.
- **The empirical study is fairly broad.** The paper evaluates both standard two-sample settings (Figure 2, Table 1) and a more application-oriented “reference-pair” closeness-testing protocol on ImageNet variants, confidence margins, and adversarial perturbation settings.

## Weaknesses
###: Fatal
- **The paper’s central notion of “closeness” is not convincingly justified as a distributional target, only as a testing-motivated reweighting.**  
  The core rationale for NAMMD is that with fixed MMD, larger RKHS norms can yield smaller variance and lower p-values, so such pairs should be considered “less close.” The paper explicitly leans on this testing argument: “we separate two distributions more effectively at same MMD distance with larger norms” (Section 3, Remark), and the introduction motivates NAMMD through differing p-values in Figure 1c. This does support NAMMD as a **test statistic calibrated to detectability under a fixed kernel**, but it does **not** establish that NAMMD is a principled underlying notion of distributional closeness in the same sense that TV, Wasserstein, or even MMD itself are typically used. Since the paper’s framing is not merely “a more powerful test statistic” but a new **distribution closeness measure/test**, this conceptual gap substantially weakens the central claim.

- **The main “higher power than MMD” claim for distribution closeness testing is not a clean same-hypothesis comparison.**  
  In Section 4.3, Definition 11 and Theorem 12 compare
  \(H_0^N:\mathrm{NAMMD}(P_2,Q_2)\le \epsilon^N\) versus
  \(H_0^M:\mathrm{MMD}(P_2,Q_2)\le \epsilon^M\),
  where \(\epsilon^N\) and \(\epsilon^M\) are induced by the same reference pair but under different metrics. These are generally **different nulls**, since NAMMD and MMD can order pairs differently. Thus, the paper does not establish a standard power dominance result for two tests of the **same** hypothesis; rather, it shows that NAMMD tends to reject more often under its own induced ordering. That is useful, but weaker than the paper’s repeated claim that NAMMD has “higher test power than MMD” for distribution closeness testing in a directly comparable sense.

### Major:
- **The practical claim about assessing model-performance similarity without labels is only partially supported.**  
  The paper motivates NAMMD as helping decide whether a model “performs similarly across training and testing datasets without ground truth labels,” but the experiments in Section 5.2 mainly show that NAMMD tracks pre-ordered shift scenarios (ImageNet variants ordered by accuracy margin, confidence margins, adversarial perturbation levels). This is suggestive evidence of correlation, not a demonstration that NAMMD can reliably support downstream adaptation decisions or predict performance degradation in a calibrated way.
- **Same-kernel empirical gains over MMD are often small.**  
  Table 1 does show consistent improvements for NAMMD over MMD with the same kernel, but many are numerically modest (e.g., 0.563→0.566, 0.796→0.797, 0.332→0.334). These are directionally aligned with the theory, yet the practical significance is unclear from the main text.
- **The key DCT comparison theorem relies on a nontrivial condition whose practical scope is not well characterized.**  
  Theorem 12 assumes \(\|\mu_{P_1}\|+\|\mu_{Q_1}\| < \|\mu_{P_2}\|+\|\mu_{Q_2}\|\). The paper says this is “often met in practice,” but gives only informal justification. Since this assumption is tied directly to the claimed advantage of the normalization, the absence of a clearer empirical characterization weakens the practical reach of the theorem.
- **Theoretical “strict improvement” guarantees are weakly quantified.**  
  In Theorems 10 and 12, the “furthermore” statements only guarantee the favorable event with probability \(\varsigma \ge 1/65\). That is mathematically non-vacuous, but too weak to strongly support the paper’s broad practical superiority claims.

### Minor
- **The method is restricted to a particular kernel class, and the implications are under-discussed in the main text.**  
  The paper explicitly limits itself to kernels of the form \(\kappa(x,x')=\Psi(x-x')\le K\) with positive-definite \(\Psi\) and \(\Psi(0)=K\). This covers several important kernels used in practice, but the restriction is still meaningful and deserves more prominent discussion in the main body rather than mostly as a limitation note.
- **Kernel selection for the actual multi-pair closeness-testing setting remains unresolved.**  
  The paper openly notes that selecting an optimal global kernel for distribution closeness testing with multiple distribution pairs is an open problem. This is honest, but it also means a practically important part of the framework is not yet solved by the paper.
- **The distribution-closeness experiments do not isolate NAMMD’s contribution as cleanly as they could.**  
  The comparison to Canonne’s TV-based test in Table 2 mixes a different discrepancy notion with a kernel representation advantage. It therefore does not by itself establish that NAMMD, specifically, is the source of the improvement rather than kernelization more broadly.

### Trivial
- **A more direct synthetic validation of the motivating phenomenon would help.**  
  The paper’s central story is about pairs with identical MMD but different norms; Figure 1 illustrates this, but a dedicated experiment systematically varying norms while holding MMD fixed would have made the evidence more compelling.
- **A brief computational-cost discussion would improve usability.**  
  NAMMD appears to add negligible overhead relative to MMD, but the paper does not explicitly quantify this.

## Nice-to-Haves
- Add a calibration study linking NAMMD rejection decisions to actual downstream accuracy degradation or adaptation benefit, rather than only showing ordering consistency.
- Include a targeted experiment probing when Theorem 12’s norm condition holds or fails, and what happens empirically in the failure regime.
- Report paired significance tests or confidence intervals for the small Table 1 improvements.
- Provide a direct synthetic ablation where MMD is fixed and RKHS norms vary, to validate the paper’s motivating cross-pair ordering claim.
- Give a brief runtime comparison with MMD/permutation testing to confirm negligible additional cost.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The comparison in Figure 2 is unfair because some methods get different sample sizes.”**  
  Removed as a main weakness because the asymmetry stated in the paper does **not** disadvantage the proposed method alone: the paper says “we set the test sample size for NAMMDFuse, MMDFuse, MMDAgg, and ACTT to be twice that of other methods,” so the protocol is not a simple one-sided boost for the authors’ method. It still limits cleanliness of comparison, but not in the way implied by the criticism.
- **“No finite-sample calibration evidence is provided for nonzero \(\epsilon\).”**  
  Weakened/removed as a core criticism because the paper does provide some nonzero-threshold experimental evidence in Section 5.2 (e.g., reference-pair tests and confidence-margin experiments), even if this evidence is not fully decisive.
- **Pure requests for more baselines or missing related work.**  
  Not retained as core weaknesses under the review policy, unless tied to a specific substantive gap already captured above.

## Novel Insights
The most important synthesis is that the paper is strongest when read as proposing a **kernel test statistic that reweights MMD by self-similarity to better reflect finite-sample detectability across distribution pairs under a fixed kernel**. In that framing, the method is simple, technically developed, and plausibly useful. The paper becomes much weaker when it upgrades this testing intuition into a broad new notion of intrinsic “distribution closeness” and then claims higher power over MMD for distribution closeness testing, because Section 4.3 compares different metric-induced hypotheses rather than the same one. So the paper likely contains a useful idea, but it is over-framed relative to what is actually established.

## Suggestions
- Reframe the contribution more carefully: present NAMMD primarily as a **testing-oriented normalized discrepancy** rather than claiming it is a generally superior underlying notion of distribution closeness.
- Soften and clarify the Section 4.3 claims: explicitly acknowledge that Theorem 12 compares **different metric-induced nulls**, and avoid presenting it as a standard same-hypothesis power dominance result.
- Add experiments that directly test the motivating scenario of equal MMD with varying norms.
- Strengthen the practical use-case evidence by evaluating whether NAMMD better predicts actual downstream performance changes or adaptation decisions.
- Empirically characterize when the norm condition in Theorem 12 holds and whether improvements persist when it does not.
- Discuss kernel-class restrictions and unresolved global-kernel selection more prominently in the main text.

## Score and Decision
**Originality:** moderate-to-good. The normalization itself is simple, but the specific RKHS-norm-based adjustment for closeness testing is novel.  
**Importance:** moderate. Extending closeness testing beyond discrete 1D settings is valuable.  
**Claims support:** mixed. The theory is substantial, but the strongest conceptual and comparative claims are overstated.  
**Experimental soundness:** decent breadth, but not fully aligned with the strongest practical claims, and same-kernel gains are often modest.  
**Clarity:** generally understandable, though the paper sometimes blurs detectability, statistical significance, and semantic closeness.  
**Community value:** moderate. There is a useful testing idea here, but the framing should be more careful.

**Calibration.** I compared this paper against:
- **GPcSYm89wK** (“Practical Kernel Learning for Kernel-based Conditional Independent Test”, scores 5/5/3/5, Reject): similar pattern of nontrivial kernel-testing methodology with modest empirical gains and concerns about how strongly the theory supports the practical claims. The current paper is somewhat stronger technically and broader experimentally, so it should score slightly above that reject anchor.
- **bkzkCHSYp9** (“Learning Interpretable Characteristic Kernels via Decision Forests”, scores 3/5/5/3, Reject): another kernel-testing paper with some theory and experiments but concerns about significance and contribution strength. The current paper is more coherent and more technically developed than this low anchor.
- **z9j7wctoGV** (“Deep Kernel Relative Test for Machine-generated Text Detection”, scores 6/6/6, Accept): a positive anchor where the testing setup and practical claim are more tightly matched and the contribution is cleaner. The current paper falls below this accept anchor because its main conceptual framing and its MMD comparison for closeness testing are not fully supported as stated.

Relative to these anchors, this paper looks **borderline but below acceptance**: technically serious and not devoid of contribution, yet weakened by a foundational framing issue and overstated comparative claims.

**Final score: 4.5 / 10**

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>