## Summary
This paper combines three contributions around high-order optimization: a PyTorch library (OPTAMI) for basic/accelerated high-order methods, a practical adaptive variant of Nesterov acceleration (NATA), and new convergence results claiming global superlinear behavior of Cubic Regularized Newton and Basic Tensor methods on strongly star-convex objectives. The topic is important and the software angle is useful, but the paper’s strongest comparative and theoretical claims are not supported as cleanly as the presentation suggests.

## Strengths
- **Addresses a real gap in practice for high-order methods.** The paper correctly identifies that second-/third-order methods and their accelerations are hard to compare and deploy due to multi-level structure, subsolvers, and implementation complexity. The OPTAMI decomposition into **basic methods / subsolvers / accelerations** is a sensible and potentially valuable systems contribution.
- **NATA is a practically motivated algorithmic idea.** Section 3.1 clearly explains that the classical accelerated tensor method can be slowed by conservative theoretical constants in the \(A_t\) schedule, and NATA proposes an adaptive mechanism to increase aggressiveness while preserving the same asymptotic order in Theorem 3.1.
- **The paper surfaces an interesting empirical phenomenon.** The experiments consistently illustrate that classical high-order accelerations may underperform their non-accelerated counterparts in the tested settings, and that the proposed adaptation can help substantially.
- **Theoretical development is nontrivial and clearly connected to the method structure.** The proof route in Theorem 4.3—building a descent factor from the upper model and strong star-convexity—is meaningful and gives a useful iteration-dependent contraction perspective for cubic/tensor methods.
- **The paper is generally clear about scope limitations.** Section 5 explicitly acknowledges computational and memory limitations of exact high-order methods in high dimensions, which is appropriate and helps frame the contribution more honestly.

## Weaknesses

###: Fatal
- **The headline “global superlinear convergence” claim is not as strong as presented, because the main theorem derives an improving contraction schedule from an already established global linear bound rather than proving a sharper direct recurrence on the actual one-step ratio.**  
  This concern is grounded in Section 4 itself. In Theorem 4.3, the paper first proves
  \[
  f(x_{t+1})-f^* \le (1-\alpha_t)(f(x_t)-f^*)
  \]
  for any \(\alpha_t \le \alpha_t^*\), where \(\alpha_t^*\) depends on \(\|x_t-x^*\|\). Then in Theorem 4.5, the paper upper-bounds \(\|x_t-x^*\|\) using the earlier linear rate (21):
  \[
  \|x_t-x^*\| \le \left(\frac{2}{\mu}(1-\alpha^{\text{low}})^t(f(x_0)-f^*)\right)^{1/2},
  \]
  and substitutes this into \(\kappa_t\) to define \(\alpha_t^{\text{sl}}\). This does produce a decreasing upper bound sequence \(\zeta_t=1-\alpha_t^{\text{sl}}\), but the result is materially weaker than the presentation in the abstract/contributions/Table 1 suggests. The theorem does not establish an explicit stronger global complexity than the known linear bound, nor a direct non-asymptotic superlinear recurrence on the actual contraction factor. Since this is the paper’s main theoretical headline, the overstatement materially affects the paper’s core claim.

### Major:
- **The empirical basis for claiming NATA “consistently outperforms all SOTA acceleration techniques” is not methodologically strong enough.**  
  Section 3.2 itself admits that some competing methods perform poorly because the authors used theoretical parameters and that these methods likely need tuning/adaptation: e.g., “Optimal Acceleration method performs the worst in practice… as we used the theoretical parameters in our implementation.” At the same time, NATA is explicitly built around adaptive parameter selection and is also shown in tuned fixed-\(\nu\) variants (“Cubic NATA with tuned \(\nu\)”, “Tensor NATA with tuned \(\nu\)”). That setup is informative, but it does not justify a broad superiority claim over all competing acceleration methods.
- **The experimental scope is too narrow for the breadth of the practical claims.**  
  The empirical section is dominated by logistic regression on a9a/a9b and a synthetic Nesterov lower-bound example. That is enough to demonstrate the phenomenon on a simple convex family, but not enough to support claims about broad practical superiority of NATA or that OPTAMI facilitates application to “a wide range of optimization problems,” especially “including neural networks.” The current evidence is too limited in task diversity, model class, and scale.
- **The evaluation metric of “Hessian computations” is incomplete as a practical cost measure.**  
  The paper itself notes substantial differences in inner-loop behavior across methods: line-search iterations, safe segment searches, and inner iterations for optimal methods. In that setting, plots against Hessian computations alone do not suffice to establish practical dominance. A wall-clock comparison or at least a fuller accounting of subsolver/line-search overhead is needed for the practical claims being made.
- **NATA appears sensitive to hyperparameters, and the paper does not provide enough guidance about robustness.**  
  Algorithm 2 introduces \(\nu^{\min}, \nu^{\max}, \theta\), and Figure 3 further shows tuned fixed-\(\nu\) variants that outperform the adaptive version. The text also states that aggressive choices “may diverge if \(\nu^t\) is not chosen carefully.” This makes the practical story less “plug-and-play” than the prose suggests; sensitivity analysis or usage guidance is needed.

### Minor
- **The theoretical notion is hard to interpret quantitatively.**  
  The final guarantees in Theorems 4.5/4.6 are given through a product form \(\prod_t (1-\alpha_t^{sl})\). While this is enough for the paper’s stated definition, it remains difficult to compare this result against standard global linear + local quadratic characterizations or to read off a clean \(\varepsilon\)-complexity interpretation.
- **The strong/star-convex assumptions deserve more discussion.**  
  The paper correctly states that strongly star-convex classes include strongly convex functions and “some non-convex functions,” but gives little intuition or concrete examples beyond that remark. A more careful discussion of when these assumptions are expected in practice would improve the paper.
- **Some rate statements are presented loosely.**  
  For example, in Section 2.1 the CRN “global convergence rate” is written as \(O(M_2D^3/L_2)\), which reads more like a quantity than a full iteration complexity statement. This is likely shorthand, but it is imprecise and can confuse readers.

### Trivial
- None.

## Nice-to-Haves
- Add wall-clock comparisons and/or a computational cost breakdown including inner-loop, line-search, and subsolver overhead.
- Include a broader benchmark suite: at least one higher-dimensional problem and one nonconvex PyTorch task to substantiate the library/practicality claims.
- Provide a sensitivity study for NATA over \(\theta,\nu^{\min},\nu^{\max}\), including failure cases for overly aggressive settings.
- Better explain how the product-form “global superlinear” guarantee relates to the familiar local quadratic regime.
- Include concrete examples or a short discussion of strongly star-convex but nonconvex objectives.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“Missing comparison to quasi-Newton methods such as BFGS/L-BFGS.”**  
  This could be a useful extension, but it is not a core flaw for the paper as written. The paper’s stated scope is comparison among high-order basic/accelerated tensor-style methods implemented in OPTAMI, not an exhaustive comparison to all practical second-order families.
- **Pure reproducibility concerns about release/availability of the library or cited tools.**  
  The paper cites the library explicitly; existence/availability should not be questioned.
- **Pure style/formatting complaints.**  
  Parser artifacts and cosmetic issues are not meaningful scientific weaknesses here.

## Novel Insights
The most important synthesis is that the paper is strongest when viewed as a **practical high-order optimization toolkit plus an adaptive acceleration heuristic**, not as a decisive theoretical breakthrough on global superlinear convergence. The experiments do support a useful message: conservative high-order acceleration schedules can be practically harmful, and adaptive relaxation of those schedules can help a lot. However, the paper currently packages this practical insight together with a theoretical claim whose formal content is subtler and weaker than the headline suggests. A more convincing version of the paper would narrow and sharpen its claims rather than trying to maximize them.

## Suggestions
- Reframe the theory claim more carefully: emphasize an **iteration-dependent improving contraction bound** for strongly star-convex functions, unless the authors can derive a genuinely stronger direct global rate statement.
- Tone down comparative claims in the abstract/introduction from “consistently outperforms all SOTA acceleration techniques” to a statement supported by the actual benchmark scope.
- Add wall-clock experiments and report total inner-loop effort, not only Hessian computations.
- Expand experiments beyond logistic regression and the synthetic lower-bound example.
- Add a NATA robustness section with sensitivity plots and recommended default settings.
- Clarify Section 2 rate statements so they read as standard complexity guarantees rather than loose shorthand.
- Add discussion/examples clarifying the practical meaning of strong star-convexity.

## Score and Decision
**Originality:** good. The combination of a high-order library, adaptive tensor acceleration, and convergence analysis is novel.  
**Importance of the research question:** high. Practical high-order optimization is underexplored and worth studying.  
**Whether the claims are well supported:** mixed to weak. Some claims are supported (NATA helps in the tested settings; the library is useful), but the broad superiority claims and the framing of the global superlinear result are overstated.  
**Soundness of experiments:** moderate. The plots are informative, but the scope is narrow and the comparison methodology is not strong enough for the breadth of the claims.  
**Clarity of writing:** generally good, though some theoretical claims are presented more strongly than the details warrant.  
**Value to the research community:** moderate. OPTAMI and the practical observations are useful, but the paper needs tighter claims and stronger empirical support.

**Calibration against human-reviewed anchors:**  
- Compared with **RSHTR** (`/home/wg25r/review_agent/human_reviews/tuu4de7HL1.md`, scores 8/8/6, accepted spotlight), this paper is clearly weaker: RSHTR paired strong theory with broader and better-supported empirical validation, while this submission overreaches more on both theory framing and experiments.  
- Compared with **LG-BFGS** (`/home/wg25r/review_agent/human_reviews/FcxwXnYXWh.md`, scores 5/5/3/6, reject), this paper is somewhat stronger in practical value and clarity, but similar in that a superlinear-convergence headline is less convincing than advertised.  
- Compared with **Truncated Newton for OT** (`/home/wg25r/review_agent/human_reviews/gWrWUaCbMa.md`, scores 5/6/6, accept poster), this paper is weaker on empirical substantiation: that paper supported practical claims with extensive runtime experiments over many datasets, whereas this submission relies on a much narrower benchmark.

Overall, this places the paper in the **borderline-below-threshold** range: promising ideas and useful artifacts, but not enough support for the headline claims in its current form.

**Score: 4.5 / 10**  
**Decision: Reject**

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>