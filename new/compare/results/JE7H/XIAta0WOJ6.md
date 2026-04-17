---
job_id: 95c9e8a5-600f-4cfb-8cfc-df8444070651
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: XIAta0WOJ6.pdf
paper: Faster Gradient Methods for Highly-Smooth Stochastic Bilevel Optimization
main_score_norm: 0.8
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper is squarely about stochastic bilevel optimization algorithms and complexity, which falls under optimization and learning theory, fully within ICLR scope.

## Minimum Quality
Pass ✅.  
The paper is in English and has all core sections (Abstract, Introduction, related-work-style Section 2.2, method, theory, lower bound, experiments, conclusion). The theory is non-trivial and appears internally consistent; experiments, while limited, are present and relevant. No obvious fatal methodological or statistical flaws are apparent from the text provided.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I do not see any attempts to manipulate the review process or hidden instructions targeting automated reviewers.

---

# Expected Review Outcome:

## Summary

The paper studies the complexity of finding an $\epsilon$-stationary point in stochastic nonconvex–strongly-convex bilevel optimization under standard SGD-type first-order oracles. Building on the penalty-based F$^2$SA framework, the authors reinterpret F$^2$SA as a first-order finite-difference approximation of the hyper-gradient and generalize it to a $p$th-order finite-difference scheme, F$^2$SA-$p$, that leverages high-order smoothness in the lower-level variable. They prove that for $p$th-order smooth problems, F$^2$SA-$p$ achieves an SFO complexity of $\tilde{\mathcal{O}}(p\kappa^{9+2/p}\epsilon^{-4-2/p})$ and show, via a reduction to single-level problems, that an $\Omega(\epsilon^{-4})$ lower bound holds even for bilevel problems, making the method nearly optimal when $p$ is sufficiently large. A small-scale experiment on logistic “learn-to-regularize” demonstrates empirical improvements over prior bilevel solvers.

## Strengths

1. **Clear conceptual reinterpretation of F$^2$SA via finite differences.**  
   The paper gives a crisp reformulation of the existing F$^2$SA method by introducing the perturbed problems $g_\nu$ and the scalar function $\ell_\nu(x)$, and by using Eq. (8) to identify $\nabla\varphi(x)$ with $\frac{\partial^2}{\partial\nu\partial x}\ell_\nu(x)|_{\nu=0}$. Eq. (9) then makes explicit that the original F$^2$SA hyper-gradient estimator is nothing but a forward-difference approximation. This is an insightful perspective that clarifies *why* the penalty formulation behaves as it does and directly motivates higher-order schemes.

2. **Technically solid extension to high-order finite differences with a clean complexity story.**  
   Using Lemma 3.1 (general $p$-point finite differences) together with Lemma 3.2 (high-order Lipschitz control on $\partial^{p+1}\ell_\nu/\partial\nu^p\partial x$), the authors design Algorithm 1 (F$^2$SA-$p$) and derive Theorem 3.1. The choice $\nu\asymp (\epsilon/(\bar L\kappa^{2p+1}))^{1/p}$ and the resulting $\mathcal{O}(\nu^p)$ hyper-gradient error are carried consistently through to yield the complexity $p\kappa^{9+2/p}\epsilon^{-4-2/p}$, which smoothly recovers prior $\tilde{\mathcal O}(\epsilon^{-6})$-type rates at $p=1$ and approaches $\tilde{\mathcal O}(\kappa^9\epsilon^{-4})$ for large $p$. The derivation appears internally coherent and achieves a nontrivial improvement over earlier first-order bilevel methods.

3. **Lower bound tailored to the bilevel setting that actually respects the smoothness assumptions.**  
   The lower bound in Theorem 4.1 is obtained via a fully separable construction where $f(x,y)$ coincides with the hard instance $f_{\mathbf U}(x)$ from Arjevani et al. (2023) and $g(x,y)=\mu y^2/2$. This construction is simple but important: unlike earlier bilevel lower bounds, the authors explicitly argue that it satisfies their first- and high-order smoothness assumptions (Definition 2.2) and avoids the violations discussed for Dagréou et al. (2024) and Kwon et al. (2024a). This makes the $\Omega(\epsilon^{-4})$ bound genuinely comparable to their upper bounds and supports the “near-optimality” claim for highly smooth problems.

4. **Reasonable positioning w.r.t. prior bilevel optimization work and clean comparison of complexities.**  
   Section 2.2 carefully distinguishes different assumption regimes: stochastic Hessian oracles (Ghadimi & Wang; Ji et al.; Yang et al.), mean-squared smoothness assumptions, and joint high-order smoothness (Huang et al.). The authors are explicit about the differences between their Assumption 2.5 (high-order smoothness only in $y$) and joint $(x,y)$ smoothness used in prior work, as well as the fact that they do not require stochastic Hessians or mean-squared smoothness. Table 1 makes the comparison concrete, showing where F$^2$SA-$p$ fits relative to previous F$^2$SA analyses and known lower bounds.

5. **A useful tighter smoothness bound that refines earlier results.**  
   Remark 3.2 points out that Lemma 3.2 for $p=2$ implies $\frac{\partial^2}{\partial\nu\partial x^2}\ell_\nu(x)$ is $\mathcal{O}(\kappa^5\bar L)$-Lipschitz in $\nu$, tightening the $\mathcal{O}(\kappa^6 \bar L)$ bound in Chen et al. (2025b, Lemma 5.1a). The way they avoid explicitly computing $\nabla^2\varphi(x)$ and instead reason via the limit $\nu\to 0$ is a technically nontrivial improvement and of potential independent interest for analyzing bilevel hyper-Hessians.

6. **Some empirical support and a clear figure.**  
   Figure 1 (Page 10) plots test loss and test accuracy vs. outer-loop iterations on the 20 Newsgroup logistic “learn-to-regularize” task. The plots indicate that higher-order F$^2$SA-$p$ variants (especially $p=5,8,10$) consistently achieve lower test loss and higher test accuracy than F$^2$SA ($p=1$) and HVP-based baselines (stocBiO, MRBO, VRBO), while “w/o Reg” performs substantially worse. While limited, this provides some evidence that the high-order schemes can translate into practical gains when the high-order smoothness assumption is satisfied.

7. **Clarity of presentation overall.**  
   The paper is generally well written and logically structured: assumptions are clearly enumerated; the algorithm is stated explicitly in Algorithm 1; key equations (e.g., (3), (4), (8)–(10)) are introduced with context. The discussion around even vs odd $p$ is thoughtful, and the remark that F$^2$SA-2 is “almost free” relative to F$^2$SA is easy to understand.

## Weaknesses

1. **Limited empirical evaluation and no direct complexity / wall-clock scaling evidence.**  
   Experiments are restricted to a single convex logistic “learn-to-regularize” problem on the 20 Newsgroup dataset (Example 2.2), plus an MLP study relegated to the appendix. There is no exploration of other bilevel benchmarks (e.g., meta-learning, data cleaning, hyperparameter learning beyond this specific setup), and in particular no experiments with less benign or only moderately smooth lower-levels where high-order smoothness might be violated in practice.  
   Moreover, Figure 1 only reports *outer iterations*, not SFO calls or run time. Since per-iteration cost scales roughly linearly with $p$ (Algorithm 1 uses $p$ inner problems when $p$ is even, $p+1$ when odd), it is entirely possible that F$^2$SA-2 or F$^2$SA-3 are the best trade-offs in wall-clock time, while large $p$ is not actually faster once per-iteration cost is accounted for. The complexity claim is asymptotic in SFO, but without at least one plot of test loss vs. total gradient evaluations or vs. wall-clock time, it is difficult to judge how much of the theoretical speedup manifests in practice.

2. **Very strong smoothness assumptions in $y$ and somewhat optimistic practical scope.**  
   Assumption 2.5 requires Lipschitz continuity of all mixed derivatives $\frac{\partial^q}{\partial y^q}\nabla f$ and $\frac{\partial^{q+1}}{\partial y^{q+1}}\nabla g$ up to order $p$ in $y$. This is quite demanding, especially in stochastic deep models where the lower-level optimization is typically non-convex and piecewise linear (e.g., ReLU networks) or non-smooth (e.g., hinge losses).  
   While the authors provide meaningful examples (softmax logistic regression for data hyper-cleaning and learn-to-regularize) where such smoothness holds, they do not discuss in any detail how sensitive the algorithm is to violations of Assumption 2.5. For example, if the model is only second-order smooth but we run $p=10$, how much does the convergence degrade? Is the hypergradient estimator still stable, or does bias explode? Right now the theory is “knife-edge”: either high-order smoothness holds exactly, or no guarantees; practical implications for more realistic models are left entirely to the reader.

3. **Condition-number dependence is very poor and not empirically explored.**  
   The SFO complexity in Theorem 3.1 scales as $\mathcal{O}(p\kappa^{9+2/p}\epsilon^{-4-2/p})$. This is an extremely steep dependence on the condition number, and the authors themselves note a gap of $\Omega(\kappa^9)$ between upper and lower bounds (Table 1 and Page 3).  
   Two issues arise:
   - First, there is no empirical study that varies $\kappa$ (e.g., by adjusting regularization or the curvature of $g$) to see whether the observed scaling is indeed that bad or whether the analysis is mainly pessimistic.  
   - Second, the analysis in Lemma 3.2 is where most powers of $\kappa$ accumulate, but the paper does not give much intuition for whether $\kappa^{2p+1}$ is inherent or an artifact of estimation via finite differences in $\nu$. The current text acknowledges the gap but essentially defers all condition-number improvements to future work.

4. **Finite-difference analysis and its connection to $\ell_\nu$ could use more explicit detail.**  
   A central technical step is Lemma 3.2, which bounds the Lipschitz constant of $\partial^{p+1}\ell_\nu/\partial\nu^p\partial x$ in $\nu$ using a high-dimensional Faà di Bruno formula (Licht, 2024). However, the main text only states the result and notes that “the variables $x$ and $\nu$ play equal roles” without providing any intermediate structure of the derivatives or explicit dependence on the high-order derivatives of $g$ in Assumption 2.5.  
   This creates a bit of an opacity: we are told that high-order smoothness in $y$ suffices to control high-order derivatives in the scalar $\nu$, but the relationship between derivatives in $y$ and derivatives in $\nu$ is not fully transparent in the main body. For instance:
   - It would be useful to see at least a sketch of how $\partial^p\ell_\nu/\partial\nu^p$ can be expressed via derivatives of $f$ and $g$ in $y$;  
   - The bound $\mathcal{O}(\kappa^{2p+1}\bar L)$ appears nontrivial, and while the appendix may justify it, the main text does not hint which terms dominate and why the exponent $2p+1$ is natural.  
   Given that this lemma is the hinge for the transition from Lemma 3.1 to Theorem 3.1, a bit more mathematical transparency in the main text would significantly strengthen the paper.

5. **Algorithmic details and normalization step are under-justified and untested.**  
   Algorithm 1 normalizes the hypergradient estimator at each outer step: $x_{t+1} = x_t - \eta_x \Phi_t / \|\Phi_t\|$ (line 14). Remark 3.1 mentions that this normalization is only used to simplify analysis and that the same guarantees should hold for unnormalized steps with a more involved proof. However:
   - The experiments do not state whether they use normalized or standard gradient steps; this is ambiguous and potentially important for reproducibility.  
   - Normalized gradient descent changes the behavior significantly when $\|\Phi_t\|$ varies a lot over iterations; there is no empirical sensitivity analysis w.r.t. this choice, nor clarification on what happens if $\Phi_t$ is extremely small (e.g., any safeguard against division by nearly zero).  
   - From a theoretical standpoint, the analysis depends on controlling the movement of $y^*_{j\nu}(x_t)$ across outer iterations; it would be helpful to give a more concrete bound in the main text that shows exactly where the normalization is critical, and whether a simple clipping/scaling rule could replace it without harming complexity.

6. **Approximation quality of the inner loop and its interaction with $p$ not fully spelled out.**  
   The algorithm estimates each $y_{j\nu}^*(x_t)$ via $K$ steps of SGD on $g_{j\nu}(x,\cdot)$. Theorem 3.1 prescribes $K \asymp \kappa^2\sigma^2/(\nu^2\epsilon^2) \log(R L_1\kappa/(\nu \epsilon))$, which grows as $1/\nu^2 \approx \epsilon^{-2/p}$. This means that for larger $p$ (smaller $\nu$), the inner loop becomes more demanding.  
   While this is all embedded in the final expression $pT(S+K)$, the paper does not explicitly discuss the trade-off: at what $p$ does the additional inner-loop burden outweigh the gain in outer iterations? Especially since Lemma 3.1 guarantees $|j\alpha_j|\leq 1$, the norm of the estimator is well-controlled, but the contribution of inner-loop bias vs finite-difference bias across all $j$ is largely buried in the analysis. A more explicit high-level explanation of the bias-variance and inner-vs-outer tradeoff as a function of $p$ would help practitioners decide which $p$ values are sensible.

7. **Experiments lack ablations on $p$ and do not connect tightly to the theory.**  
   In Figure 1, F$^2$SA-$p$ is run for $p\in\{2,3,5,8,10\}$, but the authors only plot test curves; there is no report of total SFO calls or per-iteration cost to see whether the empirical behavior matches the predicted $\epsilon^{-4-2/p}$ scaling. Useful ablations would include:
   - Plotting performance at equal SFO budgets for different $p$;  
   - Reporting how the total number of gradient evaluations needed to reach a target loss changes with $p$;  
   - Examining whether the alleged “almost free” F$^2$SA-2 indeed matches or beats F$^2$SA-1 when per-iteration cost is fully accounted for.  
   Without such ablations, the experiments mainly show that “higher-order variants can be better” on one benchmark but do not really test the asymptotic story that is the main theoretical contribution.

8. **Minor issues and notational inconsistencies.**  
   - The notation for the function classes changes: $\mathcal{F}^{n\iota\cdot n\iota}$ in Definition 2.2 vs. $\mathcal{F}^{nc\cdot sc}$ or $\mathcal{F}^{nc\cdot \kappa}$ in later statements (e.g., Lemma 3.2, Theorem 3.1), which is confusing.  
   - In Eq. (9) there appears to be a missing opening parenthesis in the numerator: it is written as $\frac{\partial}{\partial x}\ell_\nu(x) - \frac{\partial}{\partial x}\ell_0(x)}{\nu}$ in the text, which should be $\big(\frac{\partial}{\partial x}\ell_\nu(x) - \frac{\partial}{\partial x}\ell_0(x)\big)/\nu$.  
   - Table 1 mixes different works and small notational slips (e.g., “F2SA-p | 1st-order | Theorem 3.1” is conceptually a bit confusing because the method uses $p$th-order smoothness in $y$; it might be clearer to phrase the “Smoothness” column with respect to the problem (upper vs lower) rather than “1st-order”).

9. **Missing or under-discussed related work on accelerated and universal stochastic methods.**  
   While the bilevel literature is reasonably covered, there is little discussion of more generic accelerated or universal methods for high-order smooth stochastic optimization that might be adapted to bilevel settings. This underplays how distinctive the proposed finite-differences construction really is versus simply importing higher-order single-level techniques.

Overall, the theoretical story is compelling and carefully executed, but the paper would be considerably stronger with more empirical validation, more transparent explanations of the key analytic steps (particularly Lemma 3.2), and a sharper discussion of the trade-offs in $p$ and $\kappa$.

## Potentially Missing Related Work

Below I list related works that appear absent from the current references and that seem directly relevant enough to warrant at least a brief discussion or positioning.

1. **Cao, Wang, Liu (2024): “An Accelerated Gradient Method for Convex Smooth Simple Bilevel Optimization.”**  
   - Relevance: Addresses bilevel optimization with convergence acceleration in a convex “simple bilevel” setting. While this paper focuses on nonconvex–strongly-convex settings, both works are about improving complexity guarantees for bilevel problems.  
   - Suggestion: Cite and briefly discuss in Section 2.2 (“Comparison to Previous Works”), clarifying how their convex assumptions differ and why the proposed F$^2$SA-$p$ extension is complementary (nonconvex upper-level, stochastic, high-order smoothness).

2. **Hu, Zhang, Lin (2023): “Contextual Stochastic Bilevel Optimization.”**  
   - Relevance: Deals explicitly with stochastic bilevel optimization, albeit in contextual formulations. Complexity and oracle assumptions are important parallels, and it would be useful to position the present work relative to their stochastic setting and assumptions.  
   - Suggestion: Add in Section 2.2 as another stochastic bilevel line, emphasizing that here the focus is on high-order smoothness and complexity lower bounds, whereas they target context-dependent structure.

3. **Liu, Wang, Zhang (2023): “Faster Stochastic Variance Reduction Methods for Compositional MiniMax Optimization.”**  
   - Relevance: While aimed at compositional minimax optimization, the structure is similar to bilevel and many variance-reduction ideas there could be relevant to extending F$^2$SA-$p$ with variance reduction, which the authors mention as future work.  
   - Suggestion: Mention in the discussion/future-work section (Section 6) as a candidate source of techniques to combine with F$^2$SA-$p$ to further improve rates.

4. **Zhang, Chen, Xu (2024): “Functionally Constrained Algorithm Solves Convex Simple Bilevel Problem.”**  
   - Relevance: Another bilevel optimization method, again in a convex/simple regime, that focuses on complexity and functional constraints.  
   - Suggestion: Add to Section 2.2, clarifying the connection and differences in problem class (convex vs nonconvex–strongly-convex, high-order smoothness) and oracle assumptions.

5. **Cao-related and Rodomanov et al. (2024): “Universal Gradient Methods for Stochastic Convex Optimization.”**  
   - Relevance: Universal gradient methods provide complexity guarantees across a range of smoothness orders $p$. Since this paper also leverages high-order smoothness to get better rates, explicitly contrasting with universal methods (which auto-adapt to smoothness without knowing $p$) would be insightful.  
   - Suggestion: Mention in Section 3.3 or the discussion as a conceptual comparison, clarifying that F$^2$SA-$p$ currently assumes a known order $p$ and is not universal.

6. **Chen, Xu, Luo (2023): “Faster Gradient-Free Algorithms for Nonsmooth Nonconvex Stochastic Optimization.”**  
   - Relevance: Explores gradient-free high-order methods in nonsmooth stochastic optimization. While this work is gradient-based and bilevel, the underlying idea of leveraging high-order structure for improved complexity is similar.  
   - Suggestion: Cite in the discussion (Section 6) as an example of alternative high-order acceleration in stochastic settings, especially if the authors wish to highlight that their finite-difference idea is conceptually compatible with gradient-free perspectives.

7. **Lin, Chen, Luo (2023): “Decentralized Gradient-Free Methods for Stochastic Non-Smooth Non-Convex Optimization.”**  
   - Relevance: Methodologically more distant (decentralized and gradient-free), but still within the broader theme of stochastic nonconvex optimization with weaker or different oracle models.  
   - Suggestion: Could be mentioned briefly in the related-work section to situate the work among non-standard oracle assumptions and decentralized settings, though this is less critical than the directly bilevel papers above.

8. **Liu, Chen, Luo (2024): “Decentralized Convex Finite-Sum Optimization with Better Dependence on Condition Numbers.”**  
   - Relevance: Focuses on improving condition-number dependence in decentralized convex optimization, which is precisely the type of improvement the authors identify as an open problem for their own method.  
   - Suggestion: Mention in the “Open problems” paragraph (at the end of Section 3.3) as an example of recent progress on condition-number refinement in related problems.

9. **Chen, Liu, Zhang (2025): “Second-Order Min-Max Optimization with Lazy Hessians,” and Doikov, Chayti, Jaggi (2023): “Second-Order Optimization with Lazy Hessians.”**  
   - Relevance: These works propose ways to exploit second-order information with reduced Hessian cost (“lazy Hessians”), which is directly related to trading off first-order vs second-order oracles under high-order smoothness.  
   - Suggestion: Add in Section 2.2 or Section 6 as alternative high-order strategies in nonconvex optimization, and contrast with the present finite-difference–based approach that remains fully first-order but uses more evaluations to approximate hyper-gradients.

Overall, most of these works do not invalidate the contributions but should be acknowledged and situated to provide a more complete picture of the landscape of accelerated and high-order stochastic optimization and bilevel methods.

## Questions

1. **Clarification of practical usage of $p$.**  
   In practice, how would you recommend users choose $p$? Do you anticipate that small $p$ (e.g., 2 or 3) is always preferable due to per-iteration cost, or are there realistic settings where very large $p$ (e.g., $p\approx \log(1/\epsilon)/\log\log(1/\epsilon)$) is actually beneficial? Any empirical study or rule-of-thumb based on your experience with the 20 Newsgroup and MLP experiments would help.

2. **Sensitivity to violations of Assumption 2.5.**  
   Have you tried running F$^2$SA-$p$ on problems that are *not* highly smooth in $y$ (e.g., ReLU-based lower-level models), especially for $p>2$? Does the algorithm still behave reasonably, or does it become unstable? Some empirical observations or a brief theoretical “robustness” argument would significantly clarify the real-world applicability of the method.

3. **Role and implementation of the normalized step in experiments.**  
   Did your implementation of F$^2$SA-$p$ in Figure 1 use normalized steps ($x_{t+1} = x_t - \eta_x\Phi_t/\|\Phi_t\|$) or standard steps ($x_{t+1}=x_t-\eta_x\Phi_t$)? If normalized, did you include any safeguards for very small $\|\Phi_t\|$? If not normalized, can you comment on whether you observed any issues that motivated the theoretical use of normalization?

4. **Breakdown of SFO usage between inner and outer loops.**  
   For the 20 Newsgroup experiment, can you provide numbers showing how many gradient evaluations are spent in the outer vs inner loops for different $p$ values, at the point where a given test loss is reached? This would help to concretely assess where the gains of higher $p$ come from and whether the trade-off you analyze theoretically is visible empirically.

5. **More intuition for Lemma 3.2.**  
   Could you add a high-level derivation sketch for Lemma 3.2 in the main text, particularly explaining why the Lipschitz constant of $\frac{\partial^{p+1}}{\partial\nu^p\partial x}\ell_\nu(x)$ scales as $\kappa^{2p+1}$? Even a schematic bound showing how each occurrence of the inverse Hessian of $g$ introduces a factor of $\kappa$ would enhance readability and help others build upon this argument.

6. **Choice of mini-batch size $S$ in practice.**  
   Theorem 3.1 prescribes $S\asymp\sigma^2/(\nu^2\epsilon^2)$, which can be large when $\nu$ is small (i.e., for large $p$). How did you choose $S$ in the experiment, and did you find that smaller-than-theoretical $S$ was sufficient? Any empirical guidance here would help practitioners adapt the method.

## Flag For Ethics Review

- No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The theoretical contributions (finite-difference interpretation, Lemma 3.2, Theorem 3.1, lower bound) are nontrivial and appear technically sound within the stated assumptions. The main caveats are the strong smoothness assumptions and condition-number dependence, plus some opacity in the high-order derivative analysis, but there are no evident fatal flaws.

## Presentation Rating

3: good.  
The paper is generally clear and well structured, with a logical flow from preliminaries to algorithm, analysis, lower bound, and experiments. Some notational inconsistencies, missing intuitive explanations (especially for Lemma 3.2), and limited experimental detail prevent an “excellent” rating but overall readability is solid.

## Contribution Rating

3: good.  
The reinterpretation of F$^2$SA through finite differences, the resulting F$^2$SA-$p$ method with improved SFO complexity, and the bilevel-specific lower bound collectively represent a meaningful and nontrivial advance in the theory of stochastic bilevel optimization. The main limitations are the strong high-order smoothness assumptions and modest empirical validation, but the paper should be valuable to the optimization community.

## Overall Rating

8: Accept, good paper (poster).  
The paper offers a well-motivated and technically solid refinement of first-order bilevel optimization methods, with a clear complexity improvement under high-order smoothness and a matching lower bound in the highly smooth regime. Despite limited experiments and strong assumptions, the theoretical contributions and insights are substantial enough to merit acceptance as a poster.

## Reviewer Confidence

4: confident.  
I am familiar with stochastic optimization and bilevel literature, and I have carefully checked the main derivations and complexity arguments at a high level. While I did not re-derive every step of Lemma 3.2 in full tensor detail, the overall structure and conclusions appear consistent with known techniques.