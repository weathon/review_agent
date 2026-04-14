=== CALIBRATION EXAMPLE 4 ===

# Final Consolidated Review
## Summary

This paper proposes the Curve Line Fitting (CLF) network, which replaces MLP linear transformations and activation functions with piecewise quadratic Bézier curve fitting. The single-layer model reduces to an additive sum of learned univariate functions; the multi-layer model extends this by grouping input dimensions via loss-covariance and fitting child curves conditioned on parent segment indices. The central claimed advantages are intrinsic interpretability (learned relationships are directly visualizable as curves) and training stability.

---

## Strengths

- **Tangible, evidence-backed interpretability in low-dimensional settings.** Figure 3 concretely demonstrates that after training on $y = f(x_1) + f(x_2) + \text{noise}(x_4)$, the noise dimension $x_4$ converges to a horizontal line while the signal dimensions recover the correct functional shapes. This is a specific, falsifiable demonstration of interpretability—not just a qualitative claim—that most black-box methods cannot match by construction.

- **Demonstrated training stability advantage.** Table 4 shows CLF achieving 0.07–0.21% deviation across repeated training runs versus 1.46–5.78% for MLP on the taxonomy task, with MLP requiring re-training due to non-convergence. For practitioners dealing with unstable MLP training on structured tasks, this is a genuinely meaningful property.

- **Analytically closed-form local updates are elegant.** Because each piecewise quadratic segment has an explicit derivative w.r.t. its three control points (Eq. 2), the update rule requires no autograd graph traversal and is naturally sparse—only 2–3 parameters are updated per sample per dimension. This is a self-consistent and clean design.

- **Multi-layer interpretability is insightful.** Figure 4's third row concretely shows that unrelated child dimensions retain the same curve shape across all parent segments (confirming independence), while Figure 4's second row shows related child dimensions change shape as a function of parent segment value—directly revealing the multiplicative interaction $\sin(x_1) \cdot \log(x_2+1)$. This is a genuinely novel visualization capability for learned interactions.

---

## Weaknesses

### Fatal
None identified.

### Major

- **No empirical comparison to KAN (already cited).** The paper explicitly cites Liu et al. (2024) as the most relevant prior work that makes "activation functions learnable," yet provides zero comparison—neither experimental nor mathematical. KAN likewise learns univariate functions per input dimension using B-splines on edges. The structural similarities are substantial and the paper needs to (a) formally characterize how CLF differs from KAN, and (b) provide head-to-head empirical comparison on at least one benchmark. Without this, novelty relative to the most closely related work cannot be assessed.

- **Generalization failure on MNIST is unexplained and unaddressed.** Table 5 shows CLF 2-layer at 94.97% test accuracy versus MLP 784-480-10 at 97.92%, with the 1-layer CLF exhibiting severe overfitting (96.93% train vs. 90.73% test). The paper acknowledges the problem in §3.5 but offers no mechanistic analysis of *why* piecewise Bézier fitting generalizes worse than ReLU activations in high-dimensional settings, and no regularization strategy is explored beyond "CLF+" (which itself is undefined—see below). For a method proposed as a general MLP replacement, a nearly 3-point gap on MNIST without explanation is a significant concern.

- **CLF+ is used in Table 5 but never defined.** The "CLF+" variant appears in both Table 5 and §3.5 but has no corresponding method description anywhere in the main paper. This is a reproducibility failure—results from an undefined variant cannot be evaluated or reproduced.

- **Automatic grouping algorithm (§2.3.1) is never quantitatively validated.** All multi-layer experiments in Table 3 use manually specified oracle groupings. The paper describes a covariance-based relation measure $\text{Relation}(i,j) = \text{Cov}(l_{:,i}, \hat{y}_{:,j})$ for automatic grouping but never evaluates: (a) how accurate this measure is at recovering true dimensional interactions, (b) what threshold distinguishes "related" from "unrelated" and how sensitive the model is to this threshold, and (c) whether the automatic procedure achieves comparable loss to oracle grouping. Since Table 3 shows that incorrect grouping raises loss from 0.1365 to 0.9201 at 20 segments—a 6.7× increase—the reliability of automatic grouping is critical to the method's practical utility.

- **The single-layer CLF is structurally an additive model (GAM) without acknowledgment.** The architecture $\hat{y} = \sum_i f(x_i)$ where each $f$ is a learned nonlinear function of a single input dimension is precisely the definition of a Generalized Additive Model—a well-established class studied for decades, with interpretability as their primary motivation. The paper presents this structure as novel without situating it relative to this body of work. This omission makes it impossible to assess what CLF contributes beyond replacing spline or kernel basis functions with piecewise quadratic Bézier.

### Minor

- **The "fully interpretable" claim is overclaimed.** The abstract states "removal of activation functions makes the CLF model fully interpretable." This is not definitionally true. Removing activation functions is neither necessary nor sufficient for interpretability—the 784-dimensional MNIST case itself shows that visualizing 784 learned curves is not interpretably informative at the human level. The interpretability advantage is real in low-dimensional settings but should be scoped accordingly.

- **Input space constraint $[0, \text{max}]$ with known $\text{max}$ is a hard architectural constraint not discussed in limitations.** The model requires the input domain to be pre-normalized to a fixed bounded interval. This is a genuine practical restriction (real-world inputs are not always bounded a priori or bounded in a known way), and it is relegated to an appendix without appearing in the main limitations section.

- **Multi-layer functional form is not precisely specified.** The description "each child dimension possesses multiple curves, specifically one for each segment curve of its parent dimension" (§2.3.2) is informal. The exact functional form—i.e., given root $x_i$ falls in segment $k$, the child computes $f_{j,k}(x_j)$, and the outputs are aggregated how?—is not written as an equation. The exponential parameter growth ($\text{seg}^{\text{layer}}$) noted in the conList formula is also never discussed as a scalability concern.

- **Grouping covariance measure has no theoretical grounding.** The measure $\text{Cov}(l_{:,i}, \hat{y}_{:,j})$ is presented without explanation of why this specific quantity captures genuine dimensional interaction. It could spuriously flag correlated noise dimensions as related.

### Tiny

- The taxonomy dataset is entirely synthetic and two-dimensional, making it a weak benchmark for comparing CLF and MLP at ICLR's standard.
- Training epochs/iterations and MLP learning rates are not reported, limiting reproducibility.

---

## Nice-to-Haves

- Evaluate on CIFAR-10 or a tabular benchmark with higher dimensionality to assess scalability beyond MNIST.
- Provide FLOPs or wall-clock timing measurements to substantiate the efficiency claims in §3.4, since the current argument is qualitative.
- Plot model behavior on inputs outside the training range (OOD extrapolation) to expose polynomial/Bézier instability—a known weakness of polynomial fitting.
- Provide a formula or empirical curve for parameter count as a function of input dimensions, layers, and segments so readers can assess the curse of dimensionality concretely.
- Provide quantitative interpretability metrics (e.g., feature attribution fidelity, interaction detection accuracy) rather than relying solely on visual plots.

---

## Removed Points

*These points are flagged for removal; treat them with caution.*

- **[REMOVED — formatting nitpick] Title redundancy.** "Curve" appearing in both "curve line fitting" and "Bezier curve" is a style observation, not a substantive flaw.

- **[REMOVED — misreads the paper's claim] "No backward function needed" is misleading.** The harsh critic argues that Eq. (2) *is* backpropagation in disguise. However, the paper's claim is specifically that CLF does not need an autograd/graph-traversal backward pass—which is technically accurate. The gradient of the quadratic output w.r.t. control points is computed analytically in closed form. This is a legitimate design choice. The novelty framing may be modest, but it is not a misrepresentation.

- **[REMOVED — unfairness benefits the baseline] Comparison to unregularized MLP is unfair.** The harsh critic argues a regularized MLP (with dropout, batch norm) would widen the gap further. This is true, but such a comparison would be unfair in the *opposite* direction: using an unregularized MLP is already giving CLF the best chance of competing. The asymmetry favors the baseline, not the authors. The gap remaining even against a plain MLP is itself informative.

- **[REMOVED — per instructions] Missing related work citations.** Both the GAM point and any other specific citation demands are flagged rather than cited since external literature cannot be verified here; the conceptual connection to additive models is preserved as a substantive architectural observation in the Weaknesses section.

---

## Novel Insights

The most genuinely novel observation across all three reviews—not present in the paper itself—is the following: the multi-layer CLF's visualization of child dimension curves conditioned on parent segments (Figure 4, second row) is, in effect, a form of **local interpretable interaction detection**. When child curves change shape across parent segments, the model is surfacing a multiplicative or conditional relationship; when they remain constant, it surfaces independence. This is a principled and visually compelling mechanism for interaction analysis that goes beyond what additive models can do and could, if validated on real-world high-dimensional data with reliable automatic grouping, constitute a meaningful contribution to interpretable machine learning. However, as currently implemented, the potential is undermined by the reliance on oracle groupings and the absence of quantitative validation of the grouping algorithm.

---

## Suggestions

1. **Define CLF+ precisely in the method section** — this is a minimum bar for reproducibility before any resubmission.
2. **Add a direct KAN comparison** — at minimum, a mathematical characterization of how piecewise quadratic Bézier nodes (CLF) differ from B-spline edges (KAN), plus one head-to-head empirical comparison (e.g., MNIST or a tabular task).
3. **Evaluate automatic vs. oracle grouping** — run Table 3's experiment using the §2.3.1 covariance procedure and report the resulting loss versus oracle grouping; this directly validates (or invalidates) the method's practical applicability.
4. **Provide mechanistic analysis of overfitting on MNIST** — e.g., inspect whether overfitting concentrates in specific pixel-dimension curves, and whether smoothness penalties on adjacent control points (e.g., penalizing $\|P_{k+1} - P_k\|$) serve as effective regularization.
5. **Explicitly scope the interpretability claim** — revise the abstract and conclusion to state that full interpretability holds for low-dimensional inputs where per-dimension curves are human-readable, while flagging that high-dimensional settings (e.g., 784 pixel inputs) require dimensionality reduction or feature selection before the visualization is useful.
6. **Move the input range constraint from the appendix to the main limitations section** — it is a fundamental architectural requirement and readers deserve to see it prominently.

---

**Evaluation summary:**
- *Novelty*: Modest to moderate. The use of piecewise Bézier curves as learnable univariate functions is a reasonable and clean idea, but its differentiation from KAN and from classical additive models is never established, leaving the contribution's novelty unclear.
- *Technical soundness*: Adequate for the single-node and single-layer components; the multi-layer structure lacks a precise mathematical specification and the grouping algorithm lacks theoretical grounding.
- *Empirical support*: Weak. Nearly all evidence comes from synthetic distributions designed to showcase CLF's strengths. The sole real-world evaluation (MNIST) reveals a meaningful generalization gap, and CLF+ results cannot be reproduced as the variant is undefined.
- *Significance*: Potentially interesting for low-dimensional interpretable regression/classification tasks, but not demonstrated to be significant at the scale or generality expected of ICLR contributions.
- *Clarity*: The single-node and single-layer descriptions are clear; the multi-layer structure is underspecified and the CLF+ omission is a significant clarity failure.

# Actual Human Scores
Individual reviewer scores: [3.0, 1.0, 3.0, 3.0]
Average score: 2.5
Binary outcome: Reject
