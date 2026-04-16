Let me re-read the paper content provided in the user message more carefully to verify the $L_6$/$M_6$ claim.

Looking at the paper content:

In the proof of Theorem 4.3, the text says:
```
f(x_{t+1}) \stackrel{(9)}{\leq} \Phi_{x_t, 2}(x_{t+1}) + \frac{L_6}{6} \|x_{t+1} - x_t\|^3 \stackrel{(7)}{\leq} \min_{y \in \mathbb{R}^n} \left\{ \Phi_{x_t, 2}(y) + \frac{M_6}{6} \|y - x_t\|^3 \right\}
```

Yes, the paper literally writes $L_6$ and $M_6$ where it should be $L_2$ and $M_2$. This is clearly a typo/notation error — in the CRN setting (p=2), the Lipschitz constant should be $L_2$ and the regularization parameter $M_2$, as defined in equations (2), (7), and (9). The paper then uses $M_2 + L_2$ in the next lines, and the final result (20) uses $M_2 + L_2$. So the $L_6$, $M_6$ are clearly typos for $L_2$, $M_2$ — the subsequent line uses the correct constants. This is a cosmetic typo, not a structural proof issue.

The harsh critic argues this is a structural problem. But it's clearly just a typo — the very next line uses $M_2 + L_2$ correctly, and the theorem statement uses $M_2$ and $L_2$. The constants $L_6$, $M_6$ don't appear anywhere else in the paper. This would be a minor typo, not a fatal inconsistency.

Now let me evaluate the harsh critic's primary claim about the vacuity of "global superlinear convergence":

The critic argues that the definition of global superlinear convergence (Definition 4.1) is vacuous because any method with linear convergence can be reparameterized to have a decreasing $\zeta_t$. Let me evaluate this carefully.

Definition 4.1 says: $\frac{f(x_{t+1}) - f^*}{f(x_t) - f^*} \leq \zeta_t$, where $1 > \zeta_t > \zeta_{t+1}$ and $\zeta_t \to 0$.

For gradient descent on strongly convex functions, we have $f(x_{t+1}) - f^* \leq (1 - \frac{\mu}{L_1})(f(x_t) - f^*)$, i.e., $\alpha_t \geq \frac{\mu}{L_1}$ for all $t$. So $\zeta_t = 1 - \frac{\mu}{L_1}$ is a constant, and it doesn't decrease to 0. So gradient descent does NOT satisfy this definition. The definition requires $\zeta_t \to 0$, meaning the ratio must decrease to 0, which is genuinely different from linear convergence.

The critic says you can define $\zeta_t = 1 - \alpha_t^{\text{best up to }t}$, but the definition requires that $\zeta_t$ bounds the actual ratio $\frac{f(x_{t+1})-f^*}{f(x_t)-f^*}$ at each step $t$, not that it's some post-hoc estimate. The paper proves that there exist $\zeta_t$ (depending on the iterates) such that the ratio is bounded by $\zeta_t$ AND $\zeta_t \to 0$. This is a substantive statement.

However, the critic raises a valid subtlety: the proof shows that the *allowed* $\alpha_t$ can be increased based on worst-case bounds, but does the algorithm actually achieve this? The paper proves that $f(x_{t+1}) - f^* \leq (1 - \alpha_t^*)(f(x_t) - f^*)$ where $\alpha_t^*$ is computed from $\|x_t - x^*\|$ (equation 20). So the method does guarantee this rate — it's not a post-hoc construction. The question is whether this constitutes something truly novel.

The key insight is: for CRN, the guaranteed convergence ratio improves over iterations because $\|x_t - x^*\|$ decreases (due to strong star-convexity), which makes $\kappa_t$ decrease, which makes $\alpha_t^*$ increase. For GD, $\alpha_t \geq \mu/L_1$ is the best fixed rate — the ratio cannot be shown to decrease over time in the same way. So this IS a substantive difference.

However, the critic is right that the result is somewhat "weak" in the following sense: the superlinear improvement comes from bounding $\|x_t - x^*\|$ linearly (using the linear rate from Theorem 4.3), and then observing that a linearly-decreasing distance makes the local convergence ratio improve. This is essentially saying: "the method converges linearly, and because it's making progress, the per-iteration guarantee improves." The *product* rate (28) is indeed faster than a fixed linear rate, but the characterization is not as clean as, say, showing $f(x_t) - f^* = O(\beta^{t^2})$ for some specific $\beta$.

Actually, let me think more carefully. From Theorem 4.5, we have:
$\alpha_t^{sl} = \frac{-1 + \sqrt{1 + 4\kappa_t^{sl}}}{2\kappa_t^{sl}}$ where $\kappa_t^{sl} = \frac{(M_2+L_2)\sqrt{2}}{3\mu^{3/2}}(1-\alpha^{low})^{t/2}(f(x_0)-f(x^*))^{1/2}$

As $t \to \infty$, $\kappa_t^{sl} \to 0$, so $\alpha_t^{sl} \to 1$ (from (24), $\alpha^*(z) \to 1$ as $z \to 0$). This means the per-iteration improvement factor $(1-\alpha_t^{sl})$ goes to 0. So indeed, $\zeta_t = 1 - \alpha_t^{sl} \to 0$, and the product of these factors decays faster than any fixed linear rate. This IS a genuine superlinear convergence result, not just a reparameterization of linear convergence.

But the characterization is through the product form (28), not a clean closed-form expression. The actual complexity benefit depends on problem constants, and the superlinear phase may only kick in significantly after many iterations. So while the result is technically valid and non-trivial, its practical significance compared to the linear rate may be limited for poorly conditioned problems.

The critic's strongest claim — that the definition is "essentially vacuous" — is incorrect. The definition genuinely distinguishes CRN/tensor methods from GD, where GD has $\zeta_t = \text{const}$. But the critic's concern about the construction being "self-referential" has some validity: the $\alpha_t^{sl}$ is derived by upper-bounding $\|x_t - x^*\|$ using the linear rate, which is a conservative bound.

Let me now verify the critic's claim about the NATA contribution:

The critic correctly notes that NATA's theoretical rate is the same as the original Nesterov method plus a log factor. So NATA's theoretical contribution is modest — it's mainly an empirical/practical improvement. The paper acknowledges this in Theorem 3.1 ("matches the classical convergence rates" with an additional log factor). This seems like a fair assessment, but it doesn't make the contribution worthless — practical improvements are valuable even when theoretical guarantees match.

Now let me check the critic's claim about the general p-th order theorem (Theorem 4.6) being "only sketched":

The main text states Theorem 4.6 with the key formulas but defers proofs to "Appendix B." Since the appendix is not included, we can't verify the derivation. This is a legitimate concern — the general case requires non-trivial extensions, and without the full proof, we can only evaluate what's in the main text.

Let me now think about scoring. The paper makes three contributions:
1. Global superlinear convergence result (novel, but the "superlinear" characterization may be less impactful than advertised)
2. NATA algorithm (useful practical improvement, modest theory)
3. OPTAMI library (useful software contribution)

Comparable papers:
- otU31x3fus (scores 6,6,1,8, Accept poster): Second-order method with lower bounds and convergence analysis
- Ty6TCjKNSF (scores 6,6,6,8, Accept poster): Newton with Hessian sketching, limited experiments
- FcxwXnYXWh (scores 5,5,3,6, Reject): Quasi-Newton with superlinear rate, but strong assumptions
- YwJkv2YqBq (scores 6,5,8,8, Accept spotlight): Nesterov acceleration for non-convex, strong assumptions
- 24CZaossxH (scores 5,5,3,3, Reject): Engineering library lacking novelty

This paper is stronger than FcxwXnYXWh (which had overclaimed results) and 24CZaossxH (which was pure engineering). The theoretical result, while potentially overclaimed, is genuine and novel. The superlinear convergence is real — it's just that the bound may not be as practically impactful as presented. The NATA contribution is practical. The library is useful.

Compared to otU31x3fus which got scores of 6,6,1,8 and accept poster — that paper had convergence rate results for second-order methods similar to this one. The theoretical contribution here is arguably more novel (global superlinear is a new type of result for CRN), but it also has stronger potential overclaiming issues.

I think this paper is in the 5-6 range. It has genuine contributions but some significant weaknesses around the overclaiming of "global superlinear convergence" and limited experimental scope. Let me settle on 5.5 — the theoretical result is novel and non-trivial, but the overclaiming and limited experiments bring it down.

Actually, let me reconsider. The harsh critic raises one point that I need to think more carefully about — whether the definition of "global superlinear convergence" is genuinely meaningful or not.

The key question: is $f(x_t) - f^*$ bounded by a product of decreasing factors $(1-\alpha_t^{sl})$ actually faster than linear?

If we have $f(x_t) - f^* \leq \prod_{i=0}^{t-1}(1-\alpha_i) \cdot (f(x_0)-f^*)$ where $\alpha_i \to 1$, then the product $\prod(1-\alpha_i)$ decreases supergeometrically (faster than any geometric rate). This IS genuinely faster than linear convergence. However, the rate depends on problem constants and the upper bounds may be conservative.

So the result IS technically correct and meaningful — it's just that:
1. The practical significance depends heavily on conditioning
2. The characterization through a product of time-varying rates is less clean than traditional Q-superlinear or Q-quadratic characterizations
3. The title and abstract may overstate the significance

Let me settle the scoring. I'll give it a 5 — borderline reject. The contributions are real but the overclaiming on the "global superlinear convergence" and the modest theoretical contribution of NATA, combined with limited experiments, push it below the acceptance bar for a top venue.

Wait, actually let me reconsider once more. Let me check what makes comparable papers get accepted vs rejected.

FcxwXnYXWh (5,5,3,6 Reject): Overclaimed superlinear rate, strong assumptions
This paper also has: overclaimed "global superlinear convergence" (though the result IS real), and star-convexity assumption. But it also has empirical contributions and a library.

YwJkv2YqBq (6,5,8,8 Accept spotlight): Strong assumptions but interesting results
The scoring range is wide here.

I think this paper is roughly a 5. It has novel theoretical results (global superlinear convergence, even if arguably not as impactful as claimed), a useful practical algorithm (NATA), and a library. But the experiments are very limited, the NATA theory is weak (same rate + log), and the "global superlinear convergence" framing may overstate the practical impact.

Actually, let me reconsider one more time. The global superlinear convergence result IS a genuine and novel theoretical contribution. It's the first time this has been shown for CRN and basic tensor methods. The proof technique (bounding ‖x_t - x*‖ using linear decay, then getting improved per-iteration rates) is clean and elegant. The fact that the practical significance depends on conditioning doesn't negate the theoretical novelty. And NATA clearly works well in practice. And the library is a useful community contribution.

I think the paper is around 5.5, which is borderline. Let me finalize at 5.5 and Reject (since borderline reject at many venues).

Hmm, but scores typically round to 0.5 or integers. Let me go with 5.

Actually, after careful consideration, I think the criticisms about the "global superlinear convergence" being somewhat weak are valid but not fatal. The result is genuinely new and the proof technique is clean. The $L_6$/$M_6$ issue is clearly just a typo. The experiments are limited but adequate to illustrate the main points. NATA's theory matches the original (not better), but the empirical gain is clear.

On the other hand:
- The "global superlinear convergence" framing potentially overclaims
- The Theorem 4.6 generalization is deferred to appendix
- Experiments are restricted to a9a and synthetic functions  
- No wall-clock comparisons

I'll give it a 5.</think>## Summary

This paper makes three contributions to high-order optimization: (1) a theoretical result showing that Cubic Regularized Newton and Basic Tensor Methods achieve global superlinear convergence for μ-strongly star-convex functions; (2) NATA, a practical variant of the Nesterov Accelerated Tensor Method with adaptive A_t scheduling; and (3) OPTAMI, an open-source PyTorch library implementing various high-order methods, acceleration techniques, and subsolvers.

## Strengths

- **Novel global superlinear convergence result.** The proof that CRN and basic tensor methods enjoy global superlinear convergence (in the sense that the per-iteration contraction factor improves over time) for μ-strongly star-convex functions is a genuine theoretical advance. The proof technique—observing that ‖x_t − x*‖ decreases linearly under star-convexity, which via (20) makes α_t increase over time—is elegant and well-presented. This genuinely distinguishes high-order methods from gradient descent, which has a constant contraction factor μ/L_1. Table 1 provides a clear comparison with prior per-iteration rate bounds.

- **Motivated by empirical observations.** The theoretical investigation is directly inspired by practical observations (Figures 1, 6, 7) showing faster-than-linear convergence of CRN and tensor methods, closing the theory-practice gap. This empirical-theoretical loop gives the work a compelling narrative.

- **Comprehensive practical comparison of acceleration techniques.** Figures 4 and 5 provide one of the first systematic empirical comparisons of multiple high-order acceleration schemes (Nesterov, near-optimal, proximal-point, optimal) for both p=2 and p=3, identifying that standard Nesterov acceleration can hurt performance and that the "optimal" methods underperform in practice due to conservative theoretical parameters.

- **NATA is a useful practical improvement.** The adaptive ν_t mechanism in Algorithm 2 is well-motivated and the empirical results clearly demonstrate improved performance over classical Nesterov acceleration and other acceleration schemes on the tested problems.

- **OPTAMI library is a valuable community resource.** Implementing these methods as PyTorch optimizers with a modular architecture (basic methods + subsolvers + accelerations) lowers the barrier for practical research on high-order methods.

## Weaknesses

### Major:

- **The "global superlinear convergence" contribution is overstated relative to its practical significance.** While Definition 4.1 is mathematically valid and the result is non-trivial—the product bound (28)/(31) does imply convergence that is asymptotically faster than any fixed linear rate—the practical impact depends heavily on problem conditioning. From (27), α_t^{sl} depends on (M_2 + L_2)/μ^{3/2} and (f(x_0) − f*)^{1/2}, so for poorly conditioned problems, the superlinear improvement may require prohibitively many initial iterations before becoming significant. The paper does not quantify when the superlinear phase becomes dominant, nor provide a clean iteration complexity bound in terms of ε (e.g., f(x_T) − f* ≤ ε with an explicit T(ε)). The product form in (28) does not directly translate to a standard iteration complexity rate that can be compared against known lower bounds. This gap between the empirical observation and the formal characterization is substantial.

- **Experiments are narrow.** All empirical results use logistic regression on a9a (d ≈ 123) and a synthetic Nesterov lower-bound function. There are no experiments on other convex problems, non-convex star-convex functions (despite the abstract claiming applicability to "some non-convex functions"), larger-scale settings, or neural network training problems (despite the library claiming neural network applicability). This severely limits the generalizability of claims like "NATA consistently outperforms all SOTA acceleration techniques."

- **NATA's theoretical contribution is modest.** Theorem 3.1 shows NATA matches the O((M_p R^{p+1}/ε)^{1/(p+1)}) rate of the original Nesterov Accelerated Tensor Method, plus an additional log_θ(ν^{max}/ν^{min}) factor for adaptation overhead. This is not a theoretical improvement—NATA is theoretically slightly worse. The practical gains come from aggressive ν choices (including tuned variants that lack convergence guarantees), and the paper does not provide tighter analysis explaining the empirical speedup.

### Minor:

- **Notation inconsistency in Theorem 4.3 proof.** The proof chain uses L_6 and M_6 where it should use L_2 and M_2 (consistent with the CRN setting for p=2). The subsequent lines and theorem statement use the correct L_2, M_2 constants. This is clearly a typo rather than a structural error, but it is confusing and should be corrected.

- **Unfair experimental comparison for "Optimal Acceleration" methods.** Section 3.2 notes that Optimal Acceleration methods perform worst because "we used the theoretical parameters in our implementation" without tuning, while NATA is given both adaptive and hand-tuned ν variants. This asymmetry makes the strong comparative claims in favor of NATA less convincing.

- **The general p-th order Theorem 4.6 proof is deferred entirely to Appendix B.** The specialization from the p=2 case to general p ≥ q ≥ 2 is non-trivial (involving the root condition h_κ(α) = 0 and degree-q star-convexity), and the main text provides no derivation. Without access to the full appendix, the reader cannot verify these important claims.

### Trivial:

- Algorithm 2, line 3 writes ν^t = ν^t θ, which should be ν^{t+1} = ν^t · θ (assignment update). This appears to be a typo/parsing artifact.

## Nice-to-Haves

- Wall-clock time or total FLOP comparisons alongside "Hessian computations" to give a more complete practical picture, especially since different methods have different subsolver costs.
- Experiments on at least one non-convex star-convex function to validate the claimed applicability beyond convexity.
- A quantitative characterization of when the superlinear rate becomes dominant (e.g., iteration count in terms of μ, L_p, Δ_0), or a clean closed-form iteration complexity bound from the product form (28).
- Comparison with quasi-Newton methods (L-BFGS, etc.) as practical second-order baselines.

## Removed Points

- **"Global superlinear convergence is vacuous" (Harsh Critic Point 1):** The critic claims Definition 4.1 is "essentially vacuous" because any linearly convergent method can be reparameterized with decreasing ζ_t. This is incorrect: for GD on strongly convex functions, the ratio (f(x_{t+1}) - f*)/(f(x_t) - f*) ≤ 1 - μ/(L+μ) is a fixed constant and does not decrease to 0. The definition genuinely distinguishes CRN/tensor methods from first-order methods. However, the critic's concern that the practical significance may be limited for poorly conditioned problems is valid and retained above.

- **"Theorem 4.3 has a structural inconsistency with L_6/M_6" (Harsh Critic Point 2):** The L_6 and M_6 constants in the proof chain are clearly typographical errors for L_2 and M_2. The definitions (equations 2, 7, 9) establish the p=2 setting, the very next line of the proof uses M_2 + L_2 correctly, and the theorem statement uses L_2 and M_2. This is a cosmetic typo, not a structural proof error. Downgraded from "Fatal" to "Minor."

- **"Theorem 4.6 is only sketched" (Harsh Critic Point 3):** The concern about deferred proofs has some validity but the general p-th order result is stated with explicit formulas. The issue is retained as a Minor weakness about missing derivation in the main text.

- **"Missing experiments on non-convex star-convex functions" (Spark Point 1):** The paper explicitly claims applicability to "some non-convex functions" in the abstract, yet provides no experimental validation of this claim. This is retained above.

- **"Wall-clock time/FLOP comparisons" (Spark Point 2, Human Finder Point 2):** Retained as a nice-to-have rather than a major weakness, since the paper's primary metric (Hessian computations) is standard in this literature and the computational cost discussion is acknowledged in the limitations section.

- **"NATA hyperparameters not systematically ablated" (Spark Point 4):** While valid, only two fixed-ν variants are shown briefly, and the adaptive version has a theoretical guarantee. This is retained as a minor concern within the experimental narrowness issue.

- **"The library lacks standalone evaluation" (Human Finder Point 5):** This conflates the paper with a pure systems contribution. The library is one of three contributions and its value is as research infrastructure. Retained as a strength rather than evaluated as insufficient.

## Novel Insights

The key insight that bridges the theory and practice in this paper is the following: the empirical observation that CRN and tensor methods exhibit "accelerating" convergence on strongly (star-)convex functions is not merely local quadratic convergence kicking in—it can be certified globally through a time-varying contraction factor that improves as ‖x_t − x*‖ shrinks. However, this global improvement is derived through worst-case bounds on ‖x_t − x*‖ that may be very conservative, creating a gap between the theoretical characterization and the empirical behavior. The practical lesson from NATA—that aggressive parameter adaptation can overcome the conservatism of theoretical step-size schedules while maintaining convergence guarantees—suggests that many high-order methods with conservative theoretical parameters may benefit from similar adaptive strategies.

## Suggestions

- Provide a clean iteration complexity bound in terms of ε from the product form (28)/(31), or at minimum give explicit conditions under which the superlinear rate dominates, to bridge the gap between the formal result and practical significance.
- Add experiments on 2-3 additional problem classes (at minimum one non-convex star-convex example) to validate the breadth of claims.
- Clarify in the abstract and introduction that "global superlinear convergence" refers to a time-varying contraction factor improving over iterations (rather than, e.g., Q-superlinear or quadratic convergence), to avoid overinterpretation.
- Fix the L_6/M_6 typo in the proof of Theorem 4.3 and the ν^t assignment notation in Algorithm 2.

## Score and Decision

**Calibration anchors:**

- otU31x3fus (scores 6,6,1,8, Accept poster): Second-order method with inexact Hessians, convergence analysis, similar scope. Our paper has a more novel theoretical result (global superlinear) but more limited experiments and overclaiming.
- Ty6TCjKNSF (scores 6,6,6,8, Accept poster): Newton with sketching, limited experiments. Similar level of theoretical contribution.
- FcxwXnYXWh (scores 5,5,3,6, Reject): Quasi-Newton with overclaimed superlinear rate and strong assumptions. Our paper's "global superlinear" claim is more justified than FcxwXnYXWh's overclaimed "closing the gap," but still pushes the framing.
- YwJkv2YqBq (scores 6,5,8,8, Accept spotlight): Nesterov acceleration with restrictive assumptions, interesting approach. Our paper is similar but with weaker experimental scope.
- 24CZaossxH (scores 5,5,3,3, Reject): Pure engineering library. Our paper has substantial theoretical content beyond the library.

This paper's theoretical contribution is genuine and novel (first global superlinear convergence for CRN/tensor methods), but the framing overstates its practical significance, NATA's theoretical contribution is minimal (same rate + log), experiments are narrow, and the general p-th order proof is incomplete in the main text. The OPTAMI library and practical comparison are valuable but don't compensate for the overclaiming on the main theoretical contribution. The paper is borderline: below otU31x3fus (which had cleaner theory-experiments alignment) and above FcxwXnYXWh (which had more severe overclaiming). I place it at 5 — below the acceptance threshold due to overclaiming on the primary theoretical contribution and narrow experimental evaluation.

MY FINAL SCORE: <pineapple>5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>