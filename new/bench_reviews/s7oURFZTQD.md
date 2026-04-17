Now I have enough calibration data. Let me summarize my findings:

**Calibration anchors:**

1. **Block Coordinate Descent (n2RIkaf1S4)** - Scores 3,3,5,5 (avg ~4), Decision: Reject. Had convergence claims with flawed proofs, restrictive assumptions (d_in >= n, bijective activations), and limited practical value. The MGDL paper shares similar issues: convergence under unrealistic assumptions, restrictive convex reformulation, and questionable practical relevance of theory.

2. **Convex Formulations for Two-Layer ReLU (e0X9l4kecx)** - Scores 6,6,6,8 (avg ~6.5), Decision: Accept (Poster). Had a solid theoretical contribution (convex reformulation) but with limited practical value (small-scale experiments, quadratic complexity). The MGDL paper's convex reformulation is weaker since it directly adapts a prior result.

3. **Learning Dynamics Beyond Edge of Stability (J4Dvxv7WnG)** - Scores 6,8,6,8 (avg ~7), Decision: Accept (Poster). Good theoretical analysis of GD dynamics with empirical support, but limited to deep linear networks.

4. **Gradient Descent Converges Linearly (vxvgZ0kTFv)** - Scores 3,6,3,5 (avg ~4.25), Decision: Reject. Had overclaimed theoretical contributions and flawed proofs for a very simple model.

5. **Can Stability be Detrimental? (zPaTnGjgpa)** - Scores 5,3,3,5,5 (avg ~4.2), Decision: Withdrawn/Reject. This is particularly relevant because it argues the opposite of MGDL - that instability can be beneficial for generalization.

6. **Disconnect Theory and Practice (GqI4fTVUXC)** - Scores 5,5,8,6 (avg ~6), Decision: Reject. Highlighted the gap between theoretical assumptions and practical reality in deep learning optimization.

7. **Efficient stagewise pretraining (Y5LjYI4N6P)** - Scores 5,8,6,8 (avg ~6.75), Decision: Accept (Poster). Stagewise training framework with theoretical justification and practical improvements.

The MGDL paper has several significant issues:
- Overclaimed theoretical contributions (eigenvalue claim not proved, α_l ≪ α not proved, convex reformulation requires unrealistic overparameterization)
- Unfair empirical comparisons (architecture/capacity not matched, MSE for classification)
- Gap between theory (full-batch GD) and experiments (Adam/SGD)
- The convergence theorems assume iterates stay in a compact set without verification

On the positive side, it does have real ingredients: the per-grade convergence analysis, the empirical eigenvalue tracking, and the breadth of experiments. But the overclaiming is serious and pervasive.

Let me now write the final review.</think>Now I have enough calibration data. Let me compose the final review.

---

## Summary

This paper provides theoretical and experimental justification for Multi-Grade Deep Learning (MGDL), which decomposes end-to-end training into a sequence of shallow subproblems trained on residuals. The authors establish GD convergence guarantees for MGDL per-grade, show that single-layer ReLU grades reduce to convex subproblems, and analyze eigenvalue distributions of iteration matrices to explain MGDL's observed stability advantages over single-grade end-to-end training (SGDL). Experiments span image regression, denoising, deblurring, CIFAR classification, and time-series regression with transformers.

## Strengths

- **Thorough investigation of training dynamics**: The paper combines convergence theory, convex reformulation, eigenvalue diagnostics, and experiments across multiple tasks and architectures (FCN, CNN, Transformer). This breadth of investigation is commendable.

- **Per-grade convergence guarantee (Theorem 2)**: While the result itself is a standard GD convergence theorem applied to each MGDL subproblem, it provides a rigorous foundation for the per-grade optimization and correctly identifies the learning-rate range η ∈ (0, 2/α_l) as grade-dependent.

- **Convex reformulation (Theorem 3)**: Adapting Pilanci & Ergen (2020) to the MGDL setting to show that single-layer ReLU grades admit convex reformulations is a non-trivial theoretical observation, even if practically restrictive.

- **Consistent empirical pattern**: The observation that MGDL tends to produce smoother loss curves and broader usable learning-rate ranges is supported across multiple experiments and appears to be a genuine empirical phenomenon worth studying.

## Weaknesses

### Major:

- **The central theoretical claim about eigenvalues is overstated.** The abstract and conclusion state that "MGDL's eigenvalues lie within (−1, 1), ensuring stable convergence," implying this is a structural guarantee. What Theorem 4 actually proves is a conditional statement: *if* τ = sup ||I − ηH_F(W)|| < 1, then the linearized iteration converges. The paper never proves that MGDL *satisfies* this condition. The eigenvalue analysis in Section 7 is purely empirical on small models, and these observations do not generalize to a theorem. The gap between "MGDL keeps eigenvalues within (−1, 1) in our experiments" and "MGDL structurally ensures eigenvalues within (−1, 1)" is fundamental to the paper's narrative. This is not a minor phrasing issue: the paper's core story is that MGDL has a spectral advantage, but the provided analysis does not establish this as a general property.

- **The claimed "broader admissible learning-rate range" (α_l ≪ α) is asserted, not derived.** After Theorems 1 and 2, the paper claims that MGDL "allows a broader admissible learning-rate range (η_l ∈ (0, 2/α_l) with α_l ≪ α)." However, α_l ≪ α is never proved. Both theorems give identical sufficient conditions (η < 2/sup-Hessian-norm) applied to different objectives. Whether the per-grade Hessian spectral norms are systematically smaller than the joint Hessian depends on the specific architecture and data; it is not a consequence of the multi-grade decomposition alone. The empirical learning-rate sweeps in Section 6 provide useful anecdotal evidence but do not constitute a proof.

- **The convex reformulation (Theorem 3) requires m_l ≥ P_l, which is exponentially large in practice.** P_l is the number of distinct ReLU activation patterns over N data points in d_l-dimensional space, which grows combinatorially. The paper does not estimate P_l for any of its experimental settings, and none of the practical architectures satisfy m_l ≥ P_l. The claim that MGDL "reduces a highly nonconvex problem to a sequence of convex subproblems" is technically correct but only under conditions that are never met in the paper's own experiments, making the connection between this result and MGDL's practical advantages misleading.

- **Experimental comparisons do not match model capacity, compute, or standard training practices.** SGDL and MGDL use different architectures (e.g., SGDL with depth 8 vs. MGDL with 4 grades of depth 2). Parameter counts and FLOPs are not reported. The SGDL baselines use vanilla Adam or GD without learning-rate scheduling, warmup, or regularization — standard techniques known to mitigate instability. For CIFAR classification, the paper uses MSE loss instead of cross-entropy, which is nonstandard and disadvantages the SGDL baseline. No ablation isolates the effect of the training strategy from the architectural differences. Without these controls, it is impossible to attribute the observed improvements to the multi-grade principle rather than to different model sizes, training budgets, or the absence of standard stabilizers for SGDL.

### Minor:

- **The convergence theorems assume iterates remain in a compact convex set W**, but this is assumed rather than verified. For deep networks with potentially unbounded parameters, this is a strong assumption, and no justification is provided (e.g., via regularization or bounded data). This is common in optimization theory but worth acknowledging, especially since ReLU (the paper's focus) is not twice differentiable, creating a gap with the C² assumption of Theorems 1 and 2.

- **Gap between theory (full-batch GD) and practice (Adam)**: The theoretical analysis and eigenvalue tracking both use full-batch GD, but the practical experiments (Section 5, CIFAR) use Adam. The paper does not discuss whether the spectral stability arguments transfer to stochastic or adaptive optimizers, which is what practitioners actually use.

- **No statistical significance information**: All experimental results appear to be single-seed. No error bars, standard deviations, or confidence intervals are provided, making it difficult to assess robustness of the claimed improvements.

## Nice-to-Haves

- Compare MGDL against other staged/incremental training methods (e.g., greedy layer-wise pretraining, curriculum learning) to isolate whether MGDL's advantages come from the specific residual-based decomposition or any gradual training strategy.

- Run SGDL baselines with standard stabilizers (cosine LR schedule, warmup, weight decay) to test whether the oscillations are inherent to SGDL or artifacts of naive training.

- Report classification accuracy (not just MSE loss) on CIFAR with standard cross-entropy loss and modern training pipelines.

- Provide wall-clock time comparisons and total parameter counts for all experiments.

## Removed Points

- **"MGDL is not compared against greedy layer-wise training or other staged methods"**: While a valid suggestion, the paper's stated scope is specifically MGDL vs. SGDL, not an exhaustive comparison with all staged methods. This is a "nice-to-have" rather than a core flaw.

- **"The paper doesn't demonstrate scalability to large models (ResNet-50 on ImageNet, LLMs)"**: The paper experiments with FCNs, CNNs, and Transformers. While larger-scale experiments would strengthen the paper, the current scope is reasonable for a first theoretical/empirical study.

- **"Instability may be beneficial for generalization (from Edge of Stability literature)"**: This is an alternative perspective but doesn't invalidate the paper's empirical findings about MGDL's stability. The paper isn't claiming stability is universally desirable; it's claiming stability correlates with better performance in its experiments.

- **"No comparison with standard training practices like LR scheduling"**: Subsumed by the more fundamental fairness concern about capacity-matched baselines.

- **"The convergence theorems assume iterates remain in a compact convex set":** This is moved to Minor weaknesses as it's a standard assumption in convergence theory, though it should be acknowledged.

- **"CIFAR uses MSE loss which is nonstandard"**: Subsumed into the capacity/comparison fairness point; its primary impact is on fairness of comparison rather than intrinsic wrongness.

## Novel Insights

The eigenvalue-tracking diagnostic — monitoring the spectrum of I − ηH_F during training and correlating eigenvalue excursions outside (−1, 1) with loss oscillations — is a genuinely informative tool for understanding training dynamics. However, the paper's narrative turns this from a diagnostic into a structural explanation, which overshoots what the evidence supports. The most honest reading is: MGDL's shallower per-grade objectives often have better-conditioned Hessians in practice, leading to more stable GD, but this is an empirical pattern rather than a theorem.

## Suggestions

- Weaken the theoretical claims throughout: change "MGDL's eigenvalues lie within (−1, 1)" to "MGDL tends to have eigenvalues closer to the stable range in practice" and change the α_l ≪ α claim from a statement of fact to a conjecture supported by empirical evidence.

- Add at least one controlled experiment where MGDL and SGDL have matched parameter counts and training FLOPs, using SGDL with standard training practices (LR scheduling, warmup). This would isolate the contribution of the multi-grade strategy.

- Discuss the restrictive nature of the m_l ≥ P_l condition explicitly, and clarify that Theorem 3 applies only to the extremely overparameterized regime.

- Report multi-seed statistics (mean ± std) for key experiments, and report classification accuracy alongside MSE loss.

## Score and Decision

**Calibration reasoning:** Papers with similar patterns of overclaimed theory but real empirical observations:
- Block Coordinate Descent (n2RIkaf1S4): convergence claims with flawed/restrictive assumptions, limited practical value → scores 3,3,5,5, rejected
- Convex Formulations Two-Layer ReLU (e0X9l4kecx): solid convex reformulation with limited practical applicability but honest about scope → scores 6,6,6,8, accepted as poster
- GD Converges Linearly (vxvgZ0kTFv): overclaimed contributions for a simple model → scores 3,6,3,5, rejected
- Disconnect Theory and Practice (GqI4fTVUXC): demonstrated gap between theory and practice → scores 5,5,8,6, rejected
- Stagewise Pretraining (Y5LjYI4N6P): stagewise training with theory + experiments → scores 5,8,6,8, accepted as poster

The MGDL paper has real ingredients but systematically overclaims: the eigenvalue claim is presented as theoretical when it's empirical, the learning-rate advantage is asserted without proof, the convex reformulation is restricted to an infeasible regime, and the experiments don't control for confounds. This pattern is closer to the rejected BCD paper than to the accepted convex reformulation or stagewise pretraining papers, both of which were more careful about bounding their claims. The paper does make a genuine empirical observation about stability, but the gap between what's proved and what's claimed is too large.

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>