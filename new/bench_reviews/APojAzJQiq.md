## Summary
This paper proposes ConFIG, a gradient aggregation method for multi-term losses in PINNs that constructs an update direction with positive alignment to each loss-specific gradient, equalizes projection lengths across losses, and adaptively scales update magnitude based on conflict. It also introduces M-ConFIG, a momentum-based approximation that alternates per-loss backpropagation to reduce runtime, and evaluates both on several PINN benchmarks plus a CelebA multi-task learning setting.

## Strengths
- **Clear and well-motivated problem formulation.** The paper identifies a real issue in PINN training: residual, boundary, and initial-condition losses can induce conflicting gradients and uneven optimization. The toy example in Fig. 1 and the discussion in Sections 1 and 4.1 make the motivation concrete.
- **Technically coherent core method.** The ConFIG construction in Eqs. (2)–(3) is principled: it aims to produce an update with positive dot product against each loss gradient while equalizing projection lengths. The positioning relative to PCGrad and IMTL-G in the two-loss case is particularly insightful and helps isolate where ConFIG differs.
- **Practical acceleration idea.** M-ConFIG is a useful contribution beyond the basic operator. Alternating momentum updates to avoid recomputing all loss gradients each step is a reasonable systems-oriented idea, and the wall-clock evaluation is relevant for PINNs where gradient-based multi-loss methods can be expensive.
- **Reasonably broad empirical coverage within the paper’s scope.** The paper evaluates on multiple PDE settings of different dimensionalities (Burgers, Schrödinger, Kovasznay, Beltrami), considers both two-loss and three-loss variants, includes runtime comparisons, and adds an auxiliary MTL experiment on CelebA.
- **Some honest discussion of limitations.** The paper explicitly notes that M-ConFIG degrades as the number of loss terms grows and that conflict elimination does not solve all PINN difficulties. This is a constructive and credible aspect of the presentation.

## Weaknesses

###: Fatal
- **Test-set leakage in model selection undermines the empirical claims.** Section 4 states: “every result is computed via averaging three training runs initialized with different random seeds, each using **the model with the best test performance during the training**.” This is a serious protocol flaw, not a minor reporting issue. Once the test set is used for checkpoint selection, the reported “test MSE,” relative improvements, runtime-vs-performance curves, and CelebA test comparisons are no longer valid unbiased evidence for generalization or superiority. Since the paper’s main claims are empirical (“consistently shows superior performance and runtime”), this materially weakens the core evidence base.

### Major:
- **The paper overstates the practical “conflict-free” guarantee, because it is established for ConFIG but not for M-ConFIG.** The exact ConFIG operator is designed so that the final update has positive alignment with each current loss-specific gradient. But the practically emphasized accelerated variant in Section 3.3 uses alternately updated momenta and stale information for most losses at each step. The paper does not show that the same guarantee holds for the true current gradients under M-ConFIG. This matters because the practical case is made heavily through M-ConFIG’s runtime-constrained results (Figs. 9–10), yet the main conceptual selling point does not clearly transfer to the method being advocated most strongly in practice.
- **The headline empirical claims are stronger than the presented evidence supports.** The abstract and conclusion claim broad consistency and superiority, but the main text itself reports exceptions. For example, in the three-loss setting: “PCGrad performs better for the Burgers and Schrödinger case,” and earlier the paper notes cases where IMTL-G is slightly better in the two-loss setting. Those are meaningful counterexamples to “consistently superior.” The MTL section is also presented only as an “outlook” and a “partial summary,” which is too thin a basis for claiming the method outperforms “SOTA methods.”
- **The empirical evaluation is somewhat thin for strong superiority claims.** Only three seeds are used, and the main paper emphasizes relative improvement over Adam rather than absolute margins. Relative improvement is useful, but it can obscure whether differences between advanced methods are practically large or modest. Given the variability often seen in PINN training, stronger statistical characterization in the main paper would be important for a claim of robust superiority.

### Minor
- **The adaptive magnitude scaling, presented as a key component, is not isolated cleanly in ablation.** In the two-loss case the paper argues that ConFIG, PCGrad, and IMTL-G share direction and differ in magnitude scaling, making this a good setting to validate ConFIG’s magnitude rule. But there is no explicit ablation that fixes the direction and swaps only the magnitude strategy under identical conditions. Fig. 8 studies direction weights, not the adaptive magnitude rescaling in Eq. (2).
- **Numerical conditioning of the pseudoinverse deserves more discussion.** The paper notes that the inverse operation is feasible when parameter dimension exceeds number of losses, but existence is not the same as good conditioning. If normalized loss gradients become nearly dependent, the pseudoinverse may be numerically fragile. This is not shown to be a breaking issue in the experiments, but it is an important practical caveat for a method centered on matrix inversion.
- **The “uniform decrease rate” interpretation is a bit stronger than what is strictly justified.** Equal projection lengths onto the loss gradients are a geometric property of the update direction; translating that directly into equal decrease of loss values relies on local linearization and smoothness assumptions. The paper cites Pareto-style motivation, which is reasonable, but the wording occasionally suggests a stronger optimization interpretation than is warranted.

### Trivial
- **Benchmark breadth could be expanded further in the main text.** The paper already includes several PINN tasks and mentions additional challenging PDEs in the appendix, so this is not a core flaw. Still, stronger emphasis on the hardest cases in the main paper would better support generality claims.

## Nice-to-Haves
- Add an explicit ablation that keeps ConFIG’s direction fixed and compares multiple magnitude-rescaling rules.
- Report absolute test metrics and uncertainty more prominently in the main paper, not only relative improvements.
- Include a more direct empirical diagnostic of conflict reduction, e.g., trajectories of pairwise cosine similarities among loss gradients over training.
- Provide a more explicit discussion of when equal direction weights may be suboptimal for PINNs with strongly asymmetric constraint importance.
- Analyze M-ConFIG’s approximation quality as a function of number of losses and update frequency.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“Missing related work / missing baseline X or Y.”** Removed per instruction. The paper already compares against multiple weighting and gradient-based baselines, and I should not assert omitted work without external confirmation.
- **“The paper should compare against concurrent dual-cone method / NTK weighting / causal PINNs / etc.”** Removed as a formal weakness for the same reason above. Such comparisons could be useful, but I cannot reliably penalize the paper for missing specific external baselines here.
- **“The method is not really for PINNs because the theory is generic.”** Removed as overstated. A generic multi-loss method can still be valuable for PINNs if the motivation and experiments are relevant, which they are.
- **“Wall-clock comparison may be unfair because implementation parity is unclear.”** Removed as insufficiently verified from the paper text.
- **“Only four PDEs means the benchmark is too small.”** Weakened rather than kept strongly. The paper actually includes multiple PDE families, two-/three-loss settings, runtime evaluation, and mentions additional appendix tests, so “too small” would overstate the weakness.

## Novel Insights
The most consequential issue is not simply that some claims are overbroad; it is that the paper’s strongest practical contribution and strongest conceptual contribution come apart. ConFIG’s appeal is its exact geometric guarantee, but the practically emphasized M-ConFIG is a heuristic approximation whose success is empirical rather than guaranteed. That separation would be acceptable if the evaluation were airtight—but because checkpoint selection uses the test set, the paper simultaneously weakens both the theoretical-to-practical bridge and the empirical validation of that bridge. In other words, the submission has a real methodological contribution, but the current version does not yet demonstrate with reliable evidence that the practically relevant algorithm inherits the claimed advantages strongly enough to support the paper’s headline conclusions.

## Suggestions
- **Fix the evaluation protocol first.** Re-run experiments with a proper validation set for checkpoint selection, or use a predefined stopping rule independent of test performance. This is the most important revision.
- Distinguish clearly between guarantees for exact ConFIG and empirical behavior of M-ConFIG; avoid implying that the latter inherits the former without proof.
- Soften the abstract/conclusion claims from “consistently superior” to wording that matches the actual mixed outcomes reported in Section 4.
- Add a direct ablation for adaptive magnitude scaling, since this appears to be the key differentiator in the two-loss case.
- Report absolute metrics and uncertainty in the main paper, especially where method differences are small.
- Add a short discussion of pseudoinverse conditioning and any stabilization used in practice.

## Score and Decision
**Originality:** good. The pseudoinverse-based conflict-free aggregation plus adaptive magnitude is a meaningful contribution, and the momentum-acceleration idea is practically interesting.

**Importance of the research question:** high. PINN training instability and multi-loss conflict are important problems.

**Whether the claims are well supported:** currently weak, due primarily to the test-set-based checkpoint selection and secondarily to overclaiming relative to mixed results.

**Soundness of experiments:** compromised by the evaluation protocol. The experiments are otherwise fairly extensive, but the protocol issue is severe enough that the conclusions cannot be trusted as stated.

**Clarity of writing:** generally good. The method and comparisons are explained clearly, especially in the two-loss case.

**Value to the research community:** potentially substantial if corrected, because the core idea is real and relevant; however, the present submission does not validate it rigorously enough.

**Calibration against human-reviewed anchors:**  
I calibrated this against lower-scoring papers where methodological flaws or unsupported claims substantially weakened otherwise interesting ideas, such as **Physics-Informed Neural Networks with Trust-Region Sequential Quadratic Programming** (scores 3/3/3; rejected) and **Physics Informed Neurally Constructed ODE Networks** (mostly 3s with one 6; rejected), as well as mid-range rejects like **PINNacle** (6/3/6/6) where contributions were meaningful but evidence or framing left notable gaps. I also considered stronger accepted papers such as **SCoRe** (all 8s), which had clear claims tightly matched to strong evidence. This ConFIG paper is well above the weak-reject end in idea quality and clarity, but the test leakage issue is serious enough to prevent a borderline-accept score; it lands closer to the reject side of the midrange because the main empirical claims are not presently trustworthy.

**Final score: 4.0 / 10.0**  
**Decision: Reject**

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>