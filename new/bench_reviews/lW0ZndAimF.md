## Summary
This paper studies **robust fairness** in adversarially trained classifiers, defined as improving the **worst-class robust accuracy** rather than only average robust accuracy. It derives a PAC-Bayesian upper bound on worst-class robust error via the robust confusion matrix, then proposes a training regularizer intended to reduce the spectral norm of a differentiable surrogate of that confusion matrix, with experiments on CIFAR-10/100 and Tiny-ImageNet showing consistent worst-class robustness gains.

## Strengths
- **Addresses an important and well-motivated problem.** The paper focuses on class-wise disparity under adversarial attack, which is a meaningful failure mode not captured by average robust accuracy alone. The framing around worst-class robust accuracy is practically relevant.
- **Clear conceptual link between worst-class error and confusion-matrix norms.** Section 2 cleanly connects worst-class robust error to the column sums of the adversarial confusion matrix and hence to its matrix norms. This provides a coherent lens for the problem.
- **Interesting theoretical contribution.** The paper extends prior PAC-Bayesian analyses to derive a bound for worst-class robust error in terms of the empirical robust confusion-matrix spectral norm plus a model/data complexity term. Even if not fully operationalized, this is a nontrivial theoretical extension.
- **Useful empirical observation about train/test class-wise divergence.** Figure 2 supports the paper’s motivation that the worst class on the training set need not match the worst class on the test set, and that naive explicit reweighting can worsen this mismatch. This is a valuable diagnostic insight.
- **Empirical results are broadly consistent.** Across fine-tuning and from-scratch training, multiple datasets, and both \(\ell_\infty\) and \(\ell_2\) settings, the method usually improves worst-class robust accuracy, often with small impact on average robust accuracy. For example, in Table 3 on CIFAR-10, worst-class AA improves from 20.7 (TRADES) to 34.9, and on CIFAR-100 from 1.0 to 4.0.
- **Reasonably strong experimental breadth.** The paper includes fine-tuning on strong pretrained robust models, from-scratch training, several architectures, and multiple attacks (AA, PGD-20, CW-20).

## Weaknesses

###: Fatal

None.

### Major:
- **The theory-to-method connection is substantially weaker than the paper’s “principled” framing suggests.**  
  The bound in Prop. 3.1 depends on \(\|C_{S',\gamma}^{f_w}\|_2\), but the practical method in Sec. 4.1 does not directly optimize this quantity, nor does it derive a standard surrogate loss with a clear consistency guarantee. Equation (11) defines
  \[
  \frac{\partial \Psi(f_w,S',\gamma)}{\partial w} = \sum_{i\neq j}\cdots
  \]
  i.e., effectively a custom gradient recipe rather than a clearly specified scalar objective with justified descent properties. The paper’s justification is the analogy that optimizing KL/cross-entropy can improve accuracy, but that does not establish that this update is a faithful or even monotone proxy for reducing the bound-relevant spectral norm. Since the paper’s main claim is a principled theory-to-algorithm pipeline, this gap matters.

- **The key surrogate approximation is insufficiently validated.**  
  The method relies on replacing the discrete confusion matrix entries with KL-based quantities in Eq. (10), and then using the heuristic in Eqs. (9)–(11) that the update direction is approximately aligned. This is plausible as a heuristic, but the paper provides neither theoretical analysis nor empirical validation of this alignment. An ablation or correlation study showing that the proposed regularizer actually reduces the empirical confusion-matrix spectral norm, and that such reduction tracks worst-class robustness improvements, is notably missing.

- **The comparative claim against reweighting-based methods is somewhat overstated given the evidence provided.**  
  The paper’s introduction argues that explicit reweighting is limited by train/test class-wise divergence, but the direct evidence for this is narrow: Figure 2 is a single CIFAR-10/WRN example, and the fine-tuning comparisons in Tables 1 and 4 do not isolate whether the gains arise specifically from the proposed mechanism versus differing optimization behavior or tuning protocol. The empirical pattern is suggestive, but the broader narrative that explicit reweighting is fundamentally limited is stronger than what is established here.

- **The practical relevance of the PAC-Bayesian bound is not demonstrated.**  
  Proposition 3.1 is an asymptotic-style upper bound with hidden constants, a multiplicative \(\nu\), and a large complexity term \(\Phi'(f_w)\). The paper does not evaluate the bound numerically, discuss whether it is non-vacuous, or show whether the empirical spectral term is dominant in practice. As written, the theory is useful as motivation, but the manuscript overstates how directly it explains the empirical gains.

### Minor
- **No analysis of computational overhead.**  
  The method requires constructing the surrogate confusion matrix and computing spectral-norm-related updates during adversarial training. The paper does not report runtime or training-cost comparisons, which is especially relevant for many-class settings.

- **Lack of variance / multi-seed reporting weakens confidence in smaller gains.**  
  Some improvements are large, but others are modest, especially on CIFAR-100 and Tiny-ImageNet. Since worst-class accuracy can be noisy, reporting multi-seed means/variance would make the empirical claims more convincing.

- **Results on harder settings are still quite limited.**  
  The Tiny-ImageNet result improves worst-class AA from 0.0 to 4.0 (Table 2), which is directionally positive but still indicates that the problem remains largely unsolved in harder many-class regimes.

- **The \(\nu\) discussion is not very persuasive as currently presented.**  
  The paper argues that \(\|C\|_1 \le \nu \|C\|_2\) is practically tight based on random confusion matrices for \(d_y=10\). But trained confusion matrices are highly structured, so validation on real learned models would be more informative.

### Trivial
- The paper could more clearly distinguish what is truly new in the proof chain versus what is adapted from Morvant et al., Neyshabur et al., and Xiao et al.
- Figure 3 is illustrative but does not strongly support the mechanism claim; a more quantitative visualization would be more helpful.

## Nice-to-Haves
- Report whether the proposed regularizer actually decreases the empirical robust confusion-matrix spectral norm over training.
- Add an ablation comparing the proposed update to simpler alternatives, e.g. class-balanced adversarial KL penalties or direct worst-class margin penalties.
- Include runtime overhead or per-epoch training cost.
- Provide multi-seed statistics for worst-class robust accuracy.
- Analyze \(\nu\) on actual trained confusion matrices, not only random ones.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“Missing related methods / works” style criticisms.**  
  Removed per instruction not to speculate about missing related work. The paper already compares against several relevant robust-fairness methods (FRL, FAAL, CFA, WAT in different settings), so this is not a strong paper-internal flaw anyway.

- **Complaints about unfair comparison because baseline fine-tuning epochs differ.**  
  Removed as a core criticism. The asymmetry here does **not** obviously favor the authors in a way that invalidates the result; in fact, showing stronger results in only 2 epochs than baselines using longer fine-tuning can be a stronger point, not an unfair one. A cleaner matched-budget study would help, but this is not a decisive weakness.

- **Parser-induced concerns about malformed equations in the extracted text.**  
  Removed because the user explicitly warned about extraction artifacts, so such verification issues should not be counted against the paper.

- **Pure reproducibility nitpicks about minibatch subset handling or implementation-level details.**  
  These are secondary details and not central enough to retain as weaknesses absent evidence they materially affect the conclusions.

## Novel Insights
The most interesting synthesis is that the paper’s real contribution may be stronger as a **diagnostic and heuristic** contribution than as a fully principled one. The train/test divergence of class-wise robust performance is a useful critique of explicit reweighting, and the empirical results suggest that confusion-structure-aware regularization is a promising direction. However, the paper currently presents this as if the PAC-Bayesian bound directly yields the training algorithm, while the actual method is better understood as a motivated heuristic inspired by the bound rather than a bound-optimizing procedure. Reframing the paper more honestly around this distinction would likely make it stronger.

## Suggestions
- Recast the method more carefully as a **theory-inspired heuristic** unless a stronger optimization justification can be provided.
- Add a direct empirical study showing whether the proposed regularizer reduces \(\|C_{S',\gamma}^{f_w}\|_2\) and whether that correlates with worst-class robust accuracy improvements.
- Provide an ablation comparing Eq. (11) against simpler surrogates to show that the confusional-spectral design, not merely extra regularization, drives the gains.
- Report multi-seed results, especially for CIFAR-100 and Tiny-ImageNet.
- Include runtime/training-cost measurements.
- If space allows, evaluate the numerical magnitude of the terms in Prop. 3.1 to clarify whether the bound is informative in practice.

## Score and Decision
**Originality:** good. The PAC-Bayesian worst-class robust error framing and confusion-spectrum perspective are novel enough to stand out.  
**Importance:** good. Worst-class adversarial robustness is an important problem.  
**Claims support:** mixed. The empirical claim that the method often improves worst-class robust accuracy is supported; the stronger claim of a principled theory-to-algorithm derivation is not fully supported.  
**Experimental soundness:** above average but incomplete. Breadth is good, but mechanism validation and variance reporting are missing.  
**Clarity:** reasonably clear overall, especially in motivation and setup, though the method section blurs gradient recipe vs objective.  
**Community value:** moderate. Even with the current gaps, the paper offers useful ideas and empirical observations for robust fairness research.

### Calibration
I calibrated this paper against the following human-reviewed anchors:

- **BRdEBlwUW6 (DAFA: Distance-Aware Fair Adversarial Training)** — scores **6, 8, 6, 5**, accepted. This is a close topical anchor: robust fairness under adversarial training with theory plus experiments. The current paper is comparable in motivation and empirical relevance, but weaker in the theory-to-method bridge.
- **2GwMazl9ND (Algorithmic Stability Based Generalization Bounds for Adversarial Training)** — scores **8, 6, 5, 6**, accepted. This is a theory-heavy adversarial training paper. Relative to it, the current paper has a similarly interesting theoretical angle but less convincing validation of the theoretical mechanism.
- **b87H1A3sxm (Enhancing Adversarial Robustness Through Robust Information Quantities)** — scores **6, 6, 5, 6**, rejected/withdrawn. This is a useful quality-pattern anchor: promising empirical results plus incomplete justification/mechanism support. The current paper is somewhat stronger because the problem is sharper and the empirical gains on worst-class robustness are more consistently relevant.
- **c1xnHAcMhv (Generating Less Certain Adversarial Examples Improves Robust Generalization)** — scores **5, 5, 5, 3, 3**, rejected. Relative to this lower anchor, the current paper is clearly stronger due to broader experiments and a more substantial conceptual contribution, despite still having a heuristic core.
- **XcClNiB17O / other low-end adversarial training overclaim papers** — mostly **3–5** range. The current paper is above these because it is not fundamentally broken and does deliver recurring empirical improvements.

Overall, this paper sits **below** the stronger accepted robust-fairness/theory papers because the “principled” method claim is overstated, but **above** clearly rejectable heuristic-only or weakly evidenced submissions because it tackles an important problem, contains a meaningful theoretical angle, and shows consistent empirical benefit. That places it around the **borderline accept / weak accept** range.

**Final score: 6.0 / 10**  
**Decision: Accept**

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>