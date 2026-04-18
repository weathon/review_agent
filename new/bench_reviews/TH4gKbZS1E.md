Let me re-read the paper content from the user message to verify the contradictions.

The key claimed contradiction is about jump functions (Section 3.3). The paper text says:

- Section 3.3 text: "Results show that the MLP outperforms the KAN."
- Figure 3 description from the paper's figure alt text: "In all cases, KAN (red dashed line) fits the target function much better than MLP (blue dashed line)."

This is a genuine contradiction in the paper. The textual description in Section 3.3 says MLP outperforms KAN, but the figure caption/alt-text says KAN fits much better than MLP.

Now let me check the noisy function claims:

Section 4.1 says: "Our observations indicate that KAN achieves a lower test loss with low noise levels but performs worse under high noise conditions." Then: "MLP shows better performance with minor noise interference, but KAN rapidly outperforms MLP as the training sample size increases."

These two sentences seem contradictory: "KAN achieves lower test loss with low noise" vs "MLP shows better performance with minor noise interference." These could potentially be different things (low noise = noise level is low vs. low noise = small amount, minor noise = small amount), but they seem to say opposite things about the same condition.

Now let me verify the optimizer contradictions:

Section 3.4: "L-BFGS achieves faster convergence, whereas networks with Adam converges to a lower test loss."
Figure 7 caption: "In all cases, L-BFGS achieves lower test loss faster than Adam."
Section 5: "Adam exceeded L-BFGS in performance for both networks in every instance, except for function f9"

These are clearly contradictory. Section 3.4 says Adam converges to lower test loss. Figure 7 caption says L-BFGS achieves lower test loss faster. Section 5 says Adam exceeded L-BFGS in performance in every instance except f9.

Now let me synthesize everything.</think>Now I have all the information needed to write the final review. Let me carefully verify key claims and synthesize.

## Summary

This paper empirically compares Kolmogorov-Arnold Networks (KAN) and Multi-Layer Perceptrons (MLP) on approximating irregular and noisy 1D functions. The authors categorize functions into six types (regular, non-differentiable, jump discontinuities, singularities, coherent oscillations, and noisy), control for parameter counts and training sample sizes, and assess the impact of optimizer choice (Adam vs. L-BFGS) and noise on fitting performance. The main findings are that KAN does not universally outperform MLP—KAN excels on regular and singular/oscillatory functions, while MLP performs better on functions with local non-differentiabilities or jump discontinuities—and that both architectures struggle with noisy versions of irregular functions.

## Strengths

- **Well-structured experimental taxonomy**: Organizing functions by regularity type (smooth, non-differentiable, jump, singular, oscillatory, noisy) provides a clear, interpretable framework for comparing KAN and MLP. This systematic categorization is conceptually useful and makes the results easy to digest for practitioners choosing between architectures.

- **Fairness by parameter matching**: The paper matches parameter counts between architectures ([1,5,1] KAN ~120 params vs. [1,39,1] MLP ~118 params; [1,10,1] KAN ~240 params vs. [1,79,1] MLP ~238 params), ensuring that capacity differences do not confound architecture comparisons.

- **Practical value of optimizer and timing analysis**: Tables 3 and 4, showing that KAN takes 50–100× more wall-clock time than MLP for the same parameter budget, provide genuinely useful practical information for researchers considering KAN deployment. The comparison of Adam vs. L-BFGS for different function types adds practical depth.

- **Counter-hype findings**: The paper provides evidence that KAN is not universally superior to MLP, adding important nuance to the recent KAN literature, particularly the finding that MLP outperforms KAN on nonsmooth functions (f3–f6).

## Weaknesses

### Fatal
None that fully invalidate the paper, but the combination of major weaknesses below severely undermines the reliability of the conclusions.

### Major

- **Internal contradictions between text and figure descriptions**: The paper contains directly contradictory claims about which model performs better on key function types. Section 3.3 states "Results show that the MLP outperforms the KAN" for jump functions, but the Figure 3 description states "In all cases, KAN (red dashed line) fits the target function much better than MLP (blue dashed line)." Similarly, Section 3.4 states "networks with Adam converges to a lower test loss" while Figure 7's caption states "In all cases, L-BFGS achieves lower test loss faster than Adam," and Section 5 claims "Adam exceeded L-BFGS in performance for both networks in every instance, except for function f9." Without tables of numerical results (test MSE, number of runs), the reader cannot determine which claim is correct. This inconsistency across central comparative claims is a serious problem—it undermines the entire empirical contribution.

- **No statistical rigor—results from single runs**: All figures, tables, and qualitative conclusions appear to be drawn from single experimental runs with no variance reporting, no multiple random seeds, no multiple train/test splits, and no confidence intervals. This is especially damaging for the noise experiments (Section 4), where results depend on a single noise realization. Claims like "KAN outperforms MLP for regular functions" or "noise has minimal impact on fitting" cannot be trusted without replication.

- **Narrow experimental scope undermines generalization claims**: All test functions are 1D scalar functions (f: ℝ → ℝ) on compact intervals, each type represented by only 2 handpicked examples. Architectures are single-hidden-layer with only 2 width configurations. No multivariate functions, no real datasets, no deeper networks, and no systematic variation within regularity classes (e.g., varying jump magnitude, oscillation frequency, or singularity strength). The abstract and conclusion make type-level generalizations ("KAN exhibits superior performance over MLP for regular functions or functions with severe discontinuities") that the experimental scope cannot support—these are at best observations from a handful of specific examples, not robust conclusions about function classes.

### Minor

- **Dramatic computational cost disparity is reported but not integrated into conclusions**: Tables 3–4 show KAN takes ~70× longer than MLP (e.g., 588s vs. 8.3s for f7 with L-BFGS), yet the conclusion discusses only accuracy and convergence speed, not performance-per-compute-second. A model that is marginally more accurate but 70× slower may not be practically preferable. This limits the actionable value of the "KAN outperforms MLP" claims.

- **Inconsistent optimizer choices across experiments**: For some functions (e.g., f9), KAN uses L-BFGS while MLP uses Adam, chosen post hoc. This makes it impossible to disentangle whether observed differences are due to architecture or optimizer, undermining the "fair comparison" claim.

- **Function typology is informal**: The six categories are not grounded in approximation-theoretic notions (e.g., Hölder or Besov regularity), and no systematic within-category variation is tested. The paper's own description—"mathematical analysis typically categorizes into at least five distinct types"—is asserted without citation.

### Trivial

- Noise levels (SNR=0, 4, 10) and noise distribution (appears to be additive Gaussian) are not precisely defined in the main text.

## Nice-to-Haves

- **Extend to multivariate functions**: The Kolmogorov-Arnold representation theorem is fundamentally about multivariate functions; testing only 1D functions ignores KAN's core structural advantage. Including 2D+ functions would substantially strengthen the paper's relevance.

- **Deeper architectures**: All tested networks are single hidden layer. Testing multi-layer KAN and MLP would reveal whether the observed patterns persist with depth, which is crucial for practical relevance.

- **Visualization of learned KAN activation functions**: KAN's claimed strength is interpretability—showing what basis functions KAN learns for singular/oscillatory functions (vs. just the output fit) would provide insight into *why* KAN succeeds or fails on different regularity classes.

- **Pointwise error analysis**: Showing |f(x) − f̂(x)| across the domain, not just aggregate test loss, would clarify *where* each architecture fails (e.g., at discontinuities vs. smooth regions).

## Removed Points

These points are flagged to be removed, treat them with caution.

- **"KAN 2.0 / pykan not available"**: The paper cites pykan v0.2.6 and KAN 2.0; this is assumed real and available per review rules. (From harsh critic's implied concern about tool availability.)

- **"Missing comparison to KAN variants (Fourier-KAN, Wavelet-KAN, etc.)"**: The paper explicitly scopes itself to comparing vanilla KAN and vanilla MLP. Demanding comparison to every KAN variant is scope creep. (From human finder.)

- **"Limited novelty over prior work (Shen et al., 2024)"**: The paper itself acknowledges building on Shen et al. (2024) and extends it to more function types, optimizer comparisons, and noise experiments. This is incremental but constitutes a genuine extension. (From human finder and neutral reviewer.)

- **"No theoretical analysis of why these patterns occur"**: This is a suggestion for future work, not a weakness of an empirical comparison paper. The paper makes empirical claims and provides empirical evidence. Demanding theoretical explanation is scope creep. (From spark and neutral reviewer.)

- **"Missing regularization ablations for noisy data"**: The paper's stated contribution is comparing raw architectures, not proposing methods to handle noise. Regularization techniques are orthogonal to the claimed contribution. (From spark.)

- **"Not comparing deeper MLPs (3-4 layers)"**: The paper uses parameter-matched shallow architectures. While a deeper comparison would strengthen the paper, it is beyond the stated scope of comparing KAN vs. MLP at matched parameter counts in single-layer configurations. (From spark.)

- **"Extrapolation vs. interpolation behavior"**: The paper focuses on fitting quality, not generalization outside the training domain. This is outside scope. (From spark.)

- **"Kolmogorov-Arnold theorem irrelevant for 1D"**: This is technically true—the theorem is about multivariate decomposition—but the paper uses KAN's actual architecture (learnable spline activations), not just the theorem's motivation, so the experiments are still valid for what they test. (From harsh critic.)

## Novel Insights

The most noteworthy finding is that the comparative advantage of KAN vs. MLP reverses depending on function regularity: KAN excels on smooth or singular/oscillatory functions (where its B-spline basis can efficiently capture smooth global structure or rapid local variation), while MLP outperforms on nonsmooth functions (jumps, kinks) where ReLU's piecewise-linear inductive bias is well-suited. The 50–100× computational cost of KAN over MLP at matched parameter counts is a striking practical concern that substantially qualifies any accuracy advantage. The internal contradictions in the paper's own reporting (text vs. figures on jump functions, inconsistent optimizer claims) suggest that some of these conclusions may not be robust.

## Suggestions

- **Add numerical results tables** with test MSE (mean ± std over multiple seeds/runs) for each function and architecture to resolve the contradictions and provide verifiable evidence. This is the single most impactful change.
- **Reconcile contradictions** between Section 3.3 text vs. Figure 3 description, and between Section 3.4/5 vs. Figure 7 caption. Determine which is correct and make the paper internally consistent.
- **Standardize optimizer usage**: For every function, report results for both {KAN, MLP} × {Adam, L-BFGS} under identical conditions, or at minimum use the same optimizer for both architectures in each comparison.

## Score and Decision

**Calibration anchors:**

- **ydlDRUuGm9 (On the expressiveness and spectral bias of KANs)**: Accepted as poster, scores 5-8 (avg ~6.3). This paper provides *theoretical* analysis of KAN vs. MLP expressivity plus empirical validation—much deeper than the current paper.

- **Bb1ddVX8rL (Legendre-KAN)**: Rejected, scores 3-5. Similar empirical-only comparison of KAN variants, but with a novel variant (Legendre basis). Still got low scores for lacking theoretical justification and narrow evaluation scope.

- **K9xuqsaP0R (KAE)**: Withdrawn/Rejected, scores 3. Pure empirical KAN application paper with narrow scope and questionable comparisons.

- **l4jBHP4FPy (FAN)**: Rejected, scores 3-6. Novel architecture with empirical evaluation on toy + real tasks; criticized for shallow analysis and limited novelty.

The current paper has more serious problems than the typical KAN paper in this range: internal contradictions between text and figures, no statistical rigor (single runs), and extremely narrow scope (only 1D toy functions, 2 architectures per model class). Compared to Legendre-KAN (rejected, avg ~3.5) and KAE (rejected, avg 3), this paper has less novelty (no new architecture, no new method—just a comparison) but similarly narrow evaluation. Compared to the expressiveness paper (accepted, avg 6.25), the current paper has no theoretical contribution and much weaker empirical methodology.

This paper is below the Legendre-KAN tier because that paper at least proposed a novel variant; this paper is purely an empirical comparison on toy 1D functions with contradictory results. The internal inconsistencies alone significantly reduce confidence in the findings, and the lack of statistical rigor means even the non-contradictory results are unreliable.

MY FINAL SCORE: <pineapple>3.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>