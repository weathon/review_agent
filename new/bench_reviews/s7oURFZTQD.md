Now I have enough information to write a comprehensive review. Let me carefully consider the key issues.

The main criticisms from the harsh critic are:

1. **Conflation of architecture (residual connections) with training procedure**: MGDL's architecture has residual/skip-connection structure (each grade adds to cumulative output). SGDL baseline doesn't have this. Thus improvements could be due to architecture, not training procedure. Let me verify this claim against the paper.

Looking at the paper: MGDL builds networks in stages where each grade trains on residuals. The final output is $\sum_{l=1}^L g_l$. So the MGDL architecture IS a residual structure - each grade predicts the residual and adds to previous predictions. SGDL trains a standard deep network end-to-end without this additive structure. This IS a confound - the paper compares a residual-style architecture trained greedily against a vanilla deep network trained end-to-end. This is a valid major concern.

2. **Convex reformulation intractability**: Theorem 3 requires $m_\ell \geq P_\ell$ where $P_\ell$ is exponential in input dimension. This is indeed a valid concern - the convex reformulation is primarily of theoretical interest, not practical. But the paper presents it as theoretical justification, not necessarily as a practical algorithm.

3. **Convergence theorems are standard**: Theorems 1 and 2 are indeed standard GD convergence results. The key claim that $\alpha_\ell \ll \alpha$ is asserted without proof. This is a valid concern about the gap between theory and claimed advantage.

4. **MSE loss for classification**: The paper does use MSE for CIFAR classification, which is non-standard. This is a valid minor concern.

5. **No comparisons with improved baselines**: The paper only compares MGDL vs SGDL, both from the same group. No comparison with ResNets, standard training improvements, etc. This is a valid concern that limits the impact of the empirical claims.

6. **No error bars/statistical variance**: Valid minor concern.

7. **Theory-practice gap**: The theory uses full-batch GD, experiments use Adam. Valid concern.

Let me now write the review.</think>Now I have all the information needed. Let me produce the final consolidated review.

## Summary

The paper presents a theoretical and empirical study of Multi-Grade Deep Learning (MGDL), which decomposes end-to-end training of a deep network into a sequence of shallow "grades," each trained on the residual of previous grades. It provides convergence guarantees for gradient descent on MGDL, shows that single-layer ReLU grades reduce the nonconvex problem to a sequence of convex subproblems (extending Pilanci & Ergen 2020), and analyzes eigenvalue distributions of the iteration matrix $I - \eta H_\mathcal{F}(W)$ to explain why MGDL exhibits more stable training dynamics than Single-Grade Deep Learning (SGDL). Experiments on image regression/denoising/deblurring, CIFAR-10/100, and time series regression with transformers support the theoretical claims.

## Strengths

- **Systematic eigenvalue analysis as a diagnostic tool (Section 7):** The paper monitors eigenvalues of $I - \eta H_\mathcal{F}(W)$ during training across synthetic, image, and classification tasks, demonstrating that SGDL's smallest eigenvalues drop near $-1$ (causing oscillations) while MGDL's remain within $(-1, 1)$. This provides a concrete, checkable mechanistic explanation for MGDL's stability advantage. The correlation between eigenvalue behavior and loss dynamics is documented across multiple settings (Figures 4, 5, 6).

- **Comprehensive empirical benchmarking across six tasks:** The paper evaluates MGDL vs. SGDL on six distinct settings—synthetic regression, image regression (6 images, Table 1), image denoising (6 noise levels, Table 2), image deblurring (Table 3), CIFAR-100 classification (Figure 3), and CIFAR-10 eigenvalue analysis (Figure 6)—demonstrating consistent PSNR gains (0.42–3.94 dB on regression) and training stability improvements across all tasks.

- **Learning rate robustness analysis (Section 6):** The systematic sweep of learning rates for both synthetic and image regression tasks (Figure 2, Figure 20) shows MGDL maintains effective training over a substantially wider learning rate range than SGDL, providing direct empirical support for the theoretical claim about broader admissible learning rates.

## Weaknesses

### Fatal

None.

### Major

- **Confounding of training procedure with architectural differences (residual structure):** MGDL's final output is the cumulative sum $\bar{g}_L = \sum_{l=1}^L g_l$ and each grade trains on residuals $e_{(l+1)n} = e_{ln} - g_l(\Theta_l^*; x_n)$. This is architecturally a residual/skip-connection structure. The SGDL baseline is a standard deep feedforward network with no skip connections. Consequently, the empirical improvements cannot be cleanly attributed to the multi-grade *training procedure* rather than to the well-known benefits of residual architectures. The paper does not compare MGDL against an SGDL variant with additive skip connections trained end-to-end. This is the central methodological gap: without such a comparison, it is impossible to disentangle whether the gains come from the greedy residual training procedure or simply from having residual connections, which decades of work (He et al., 2016, cited in the paper) have established as beneficial. This affects all empirical claims in Sections 5–8.

- **The theoretical convergence advantage claim ($\alpha_\ell \ll \alpha$) is unproven:** Section 3 states that MGDL "allows a broader admissible learning-rate range ($\eta_\ell \in (0, 2/\alpha_\ell)$ with $\alpha_\ell \ll \alpha$)." This claim that the Hessian spectral norm of each grade-level subproblem is much smaller than that of the full network is the key theoretical mechanism explaining MGDL's advantage. Yet it is asserted without proof or quantitative bound. Without it, Theorem 2 merely says "MGDL converges with a small enough learning rate" — which Theorem 1 already says for SGDL. The empirical eigenvalue analysis (Section 7) is suggestive but uses linearization and a pointwise (not supremum-over-$\Omega$) measurement, leaving the theoretical gap unaddressed.

### Minor

- **The convex reformulation (Theorem 3) requires exponentially many neurons:** The condition $m_\ell \geq P_\ell$ where $P_\ell$ is the number of linear regions of a ReLU partition of $\mathbb{R}^{m_{\ell-1}}$, which scales as $O(\binom{N}{m_{\ell-1}})$, makes the convex program (Eq. 8) computationally intractable for practical networks. The paper lists this as Contribution 2 without acknowledging this limitation. While the result has theoretical value as an existence proof, it does not support the claim that MGDL's convex reformulation substantively improves trainability in practice.

- **MSE loss for classification:** The CIFAR-100 experiments (Section 5) use MSE loss rather than cross-entropy, which is non-standard and likely disadvantages SGDL, since modern deep learning training pipelines use cross-entropy precisely to avoid the optimization difficulties that MGDL is designed to address.

- **Limited baselines beyond SGDL:** All experiments compare MGDL against a single SGDL baseline from the same research group, with no comparison to established methods (e.g., BM3D for denoising, standard ResNets for classification) or modern training practices (learning rate scheduling, batch normalization, warmup). The SGDL baseline appears to use vanilla training with no stabilization techniques, making it unclear whether MGDL's advantages persist against properly tuned standard training.

- **No error bars or statistical significance:** Results in Tables 1–5 and Figure comparisons report single numbers or single runs without error bars or confidence intervals across random seeds.

### Trivial

None.

## Nice-to-Haves

- Compare MGDL against an SGDL baseline with equivalent residual/skip connections trained end-to-end — this single experiment would dramatically clarify whether the gains come from the architecture or the training procedure.
- Compare against standard architectures and training recipes (ResNet, cross-entropy + cosine LR schedule) to contextualize MGDL's improvements.
- Formally bound or empirically measure $\alpha_\ell$ vs. $\alpha$ across settings to substantiate Theorem 2's key claim.
- Report wall-clock time or FLOPs for comparable accuracy, since MGDL trains $L$ networks sequentially.

## Removed Points

These points are flagged to be removed; treat them with caution:

- **Claim that the convex reformulation invalidates Contribution 2 entirely**: The harsh critic argued this makes the convex result useless. While the condition $m_\ell \geq P_\ell$ is impractical, Theorem 3 still has conceptual value—showing that multi-grade decomposition can, in principle, convexify a deep nonconvex problem grade-by-grade extends Pilanci & Ergen (2020) from shallow to deep networks. The limitation is real but the result is not vacuous as a theoretical existence statement. Moved to Minor.

- **Claim that Theorems 1 and 2 are just "textbook results"**: While the convergence theorem form is standard (GD on smooth functions), Theorem 2 is applied in a novel context—sequential grade-level subproblems—and derives a new condition $\eta_\ell \in (0, 2/\alpha_\ell)$. The structure is standard but the application and the framework are contributions.

- **Theory-practice gap (full-batch GD vs. Adam)**: The harsh critic noted that theory uses full-batch GD but experiments use Adam. This is a legitimate observation but is standard practice in the optimization-for-deep-learning literature—almost all such papers analyze GD and then use Adam in experiments. Minor, not major.

- **Demand for comparison with modern training improvements (LR scheduling, batch norm, warmup)**: While a fair suggestion, this amounts to asking the paper to benchmark against an entire modern training pipeline, which is beyond its stated scope of comparing MGDL vs. SGDL. Moved to Nice-to-Haves.

## Novel Insights

The eigenvalue monitoring across grades (Section 7) provides a striking empirical regularity: across all tested settings, SGDL's smallest eigenvalues of $I - \eta H_\mathcal{F}$ converge near $-1$, while MGDL's remain strictly within $(-1, 1)$. This regularity is presented as a mechanistic explanation for MGDL's stability, and it holds consistently even though the linearization analysis (Theorem 4) requires a supremum condition that is not verified. The gap between this empirical observation and the unproven $\alpha_\ell \ll \alpha$ claim suggests an important direction for future theoretical work: formally characterizing why greedy residual training reduces the Hessian spectral norm.

## Suggestions

- Add one experiment comparing MGDL against a standard ResNet-style architecture (same additive structure) trained end-to-end with standard training. If MGDL still wins, the training procedure gets the credit; if not, the improvements are architectural.
- Acknowledge the computational intractability of Theorem 3's condition explicitly, and discuss whether approximate or partial convexification results are achievable.
- Provide $\alpha_\ell$ and $\alpha$ measurements (Hessian spectral norms at convergence) for at least the synthetic experiments where the networks are small enough to compute.

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| Adaptive Resolution Residual Networks | I1Gd2d1WXY | 3.75 | Similar confound (residual structure vs. training method), but that paper was weaker on theory. This paper has more substance. |
| Block Coordinate Descent for NNs | n2RIkaf1S4 | 4.00 | Similar structure—layer-wise/greedy training with theoretical convergence claims but restrictive assumptions. This paper has broader experiments but a similar core gap. |
| Progressive Subnetworks (RAPTR) | ZyH5ijgx9C | 5.75 | More complete experimental evaluation with modern architectures, proper baselines, and modest claims. This paper has a stronger theoretical contribution but weaker baselines. |
| Hessian alignment paper | MHjigVnI04 | 7.67 | Much stronger theoretical contribution (rigorous proofs of interesting spectral phenomena). This paper's theory is far weaker. |
| Visual Analysis of ResNN Landscapes | CgBhR1NSLM | 3.00 | Weak paper with opposite-conclusions on tiny datasets. This paper is clearly above this. |
| Sobolev training/YhT1ZemZow | YhT1ZemZow | 4.50 | Theory-meets-practice paper on training acceleration with limited practical impact. Comparable tier. |

The confound between residual architecture and training procedure is a structural problem that undermines the paper's core empirical claims, but the paper still makes useful contributions: the eigenvalue analysis, the convex reformulation (even if impractical), and the systematic learning rate robustness demonstration. However, the two major weaknesses—conflating architecture with training procedure, and the unproven $\alpha_\ell \ll \alpha$ claim—substantially weaken the paper's ability to deliver on its title ("Why Multi-Grade Deep Learning Outperforms Single-Grade"). The paper tells us *that* MGDL outperforms SGDL and gives suggestive reasons, but doesn't conclusively establish *why*.

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>