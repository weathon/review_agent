## Summary

This paper proposes three one-step diffusion-based solvers (CMILP, SCMILP, MFILP) for integer linear programming, leveraging consistency models, shortcut models, and meanflow models to dramatically reduce inference time compared to multi-step diffusion approaches. A novel Iterative Integer Projection (IIP) layer is introduced to extend neural ILP solvers to non-binary integer variables without binary expansion, and a momentum-based gradient descent scheme is proposed for objective-guided sampling. Experiments on binary and non-binary ILP problems demonstrate speed advantages over prior diffusion-based methods (IP Guided DDPM/DDIM) and competitive feasibility on binary problems, though with notable optimality gaps.

## Strengths

1. **Practical speed improvement is significant.** On binary ILP, inference is reduced from hours/minutes (IP Guided DDPM/DDIM) to seconds (Tables 1-3). On non-binary problems, the gap is similarly dramatic (e.g., 2-3s vs. 28-48 min). This directly addresses a core limitation of diffusion-based ILP solvers.

2. **IIP is a creative mechanism for non-binary ILP.** The function f_proj(x) = x − sin(2πx)/(2π) is a differentiable, iteratively tightening projection that avoids the exponential blowup of binary encoding. Table 4 shows that binarization destroys feasibility (DDIM drops from 80% to 0% dataset feasibility), while IIP maintains compact representation. This is a meaningful contribution even if results remain imperfect.

3. **Comprehensive experimental evaluation.** The paper covers 3 binary problem types, inventory management problems with varying difficulty, and synthetic random ILP instances, comparing against traditional solvers (Gurobi, SCIP, COPT), heuristics (rins, feasibility pump), and prior neural methods. This breadth is commendable.

4. **Near-perfect feasibility on binary ILP.** All three proposed methods achieve 100% dataset feasibility on every binary ILP benchmark (Table 1), improving over IP Guided DDPM (95.7–99.8% sample feasibility).

## Weaknesses

### Major:

- **Large optimality gaps on non-binary ILP undermine the paper's main novel contribution.** The extension to non-binary ILP is the paper's primary novelty ("for the first time"), yet the results are weak on harder non-binary instances: MFILP achieves 107.1% gap and 68% sample feasibility on IM-(50,5,10); CMILP achieves 119.3% gap on IM-(50,5,10); various methods show 62–90% dataset feasibility across non-binary benchmarks. As the Zeng et al. predecessor reviewer noted, "If [solutions] are far from [optimal], having a high feasible ratio does not mean anything"—especially since the feasible region can be exponentially large with many poor solutions. The paper's own conclusion acknowledges "a relatively big optimality gap compared to traditional solvers," but this understates the severity—gaps exceeding 100% mean solutions cost more than twice the optimal, severely limiting practical utility.

- **Modified consistency loss (Eq. 6) departs from standard theory without justification.** The standard consistency model loss enforces self-consistency between two timesteps on the same trajectory. The paper replaces this with a loss directly regressing both trajectory points to a Dirac delta at x*. This collapses the distributional objective into supervised regression, removing the self-consistency guarantee. The paper states "Since the solution x* is explicit given the problem instance, we can integrate x* into the loss for better training instead of focusing on the gap between f_θ of two diverse timesteps"—but if x* is the optimal solution, why are "500 optimal and suboptimal solutions" used as training data? If suboptimal solutions are included, which x* is used? This inconsistency between the loss formulation and the training data generation is never resolved.

- **Missing ablations for core components.** There is no ablation isolating: (a) the IIP layer vs. simple rounding at test time; (b) the feasibility penalty coefficient λ_penalty; (c) the CLIP-style pretraining vs. end-to-end training; (d) the momentum guidance vs. no guidance. Table 5 only ablates GD vs. MGD on one dataset configuration. Without these, it is impossible to attribute performance gains to the claimed innovations. As noted in reviews of the Zeng et al. predecessor paper, "the functions of contrastive model and generative model are not showcased by ablation study."

- **Overstated claims in the abstract and introduction.** The abstract claims "our approach outperforms existing learning-based methods on both binary and non-binary instances." On non-binary ILP, this is not clearly supported—IP Guided DDIM often achieves better gaps and comparable dataset feasibility on several configurations (e.g., Random-(500,20,2): DDIM 0.7% gap vs. CMILP 0.0%, though CMILP is much faster). The claim of "nearly 100% feasibility" is properly qualified to binary ILP in the text, but the abstract's broader "outperforms" claim is misleading. The "strong scalability compared to traditional solvers" claim in the abstract is also problematic—Gurobi/SCIP/COPT consistently achieve 0% gap and 100% feasibility, and the neural methods' only consistent advantage is wall-clock inference time after offline training.

### Minor:

- **SCMILP and MFILP method details are deferred to the appendix** (mentioned only in passing in the main text), making it hard to assess whether these are correctly adapted from their generative model origins. The "step size" conditioning for shortcut models has no clear analog in ILP solution space, and this is not discussed.

- **Training data requirements are heavy and under-described.** The paper collects "500 optimal and sub-optimal solutions" per instance but does not specify how these are generated, their quality distribution, or the computational cost of producing them. This overhead is not counted in reported runtimes.

- **The 30-sample evaluation protocol inflates both feasibility and time cost.** While standard for diffusion methods, reported inference times (2-3s) reflect per-sample cost; the effective cost per instance with 30 samples is 30× higher, reducing the speed advantage over traditional solvers. Per-sample performance metrics would help clarify the trade-off.

- **No experiments on standard MILP benchmarks** (e.g., MIPLIB). All non-binary datasets are synthetic or constructed toy problems, limiting external validity of claims about general non-binary ILP.

- **Tang et al. (2025) is cited in related work as handling non-binary ILP with an integer correction layer,** yet no experimental comparison is provided. This is the most directly comparable prior work for the non-binary setting.

## Nice-to-Haves

- Formal convergence analysis of the IIP projection function, showing that repeated iteration converges to the nearest integer for all initial values on the relevant domain.
- Per-sample (rather than 30-sample) quality metrics to clarify the cost-quality trade-off.
- Ablation of IIP vs. simple rounding or other integer-valued surrogates.
- Sensitivity analysis on λ_penalty and momentum hyperparameters (γ, φ).
- LP-relaxation rounding as a trivial non-learning baseline for non-binary ILP, to establish whether the neural approach adds meaningful value over this simple heuristic.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh Critic's claim that "the integration of consistency/shortcut/meanflow training with ILP objectives is conceptually muddled" as a fatal structural issue.** While the modified consistency loss is a valid concern, characterizing the entire approach as fundamentally incoherent overstates the case. The paper's approach (training a denoiser to map noisy samples toward the solution distribution, then using one-step inference) is a reasonable adaptation, even if the specific loss formulation departs from standard consistency theory. This is a weakness but not a fatal one.

- **Harsh Critic's claim that IIP is "bordering on structural" failure.** IIP is a reasonable heuristic even without formal convergence proof. The empirical results show it works to some degree (62-90% feasibility on non-binary problems), and it clearly avoids the binarization blowup. The real issue is the lack of ablation and analysis, not that IIP is unsound.

- **Demand for explicit reproducibility details (hyperparameters, network sizes, schedulers)** classified as a nitpick about reproducibility. The paper states code will be released. The missing implementation details are a legitimate concern but not a review-killer for a conference submission.

- **Harsh Critic's claim that the CLIP-style pretraining is "under-specified" to the point of being a weakness.** The paper provides a reasonable description (contrastive learning between instance features and solution features, similar to the predecessor Zeng et al. 2024). While more detail would help, this is not a novel component—it follows established methodology.

- **Human finder's point about "missing T2T comparison."** T2T (Li et al., 2023/2024) operates on different problem types (TSP, MIS) and represents a different paradigm (training-to-testing). It is not a direct baseline for ILP, and its absence is understandable.

- **Human finder's point about "training cost not reported."** While useful, this is standard practice for neural solver papers. Inference cost is the more relevant comparison point for neural vs. traditional solvers.

## Novel Insights

The most interesting observation is that the binarization approach for non-binary ILP completely fails when used with diffusion-based solvers (Table 4: DDIM drops from 80% to 0% dataset feasibility, proposed methods from 88-90% to 3-9%). This is not just about variable count increase—it suggests that diffusion models learned on binary encodings may struggle with the sparse, high-dimensional solution space that binarization creates, making the IIP approach's avoidance of this expansion particularly valuable even if IIP itself has room for improvement. This finding also underscores that the gap between diffusion-based solvers and traditional solvers remains substantial on non-binary ILP, where the solution quality (not just feasibility) needs significant improvement before neural solvers can be practically useful.

## Suggestions

- **Tone down claims**: Replace "outperforms existing learning-based methods on both binary and non-binary instances" with a more nuanced statement distinguishing speed advantages from solution quality tradeoffs, and explicitly note that gaps on harder non-binary problems are large.
- **Add key ablations**: At minimum, compare IIP vs. simple rounding, and isolate the effect of the feasibility penalty. This would substantiate the paper's main novel component.
- **Report per-sample results**: Show single-sample feasibility and gap alongside 30-sample results, so readers can assess the true per-query cost-quality frontier.
- **Compare against Tang et al. (2025)** on non-binary ILP: this is the most directly comparable prior work and its absence weakens claims of being "the first" extension to non-binary neural ILP solving.

## Score and Decision

**Calibration:** The most directly comparable paper is Zeng et al. (2024) [IP Guided DDPM], the predecessor work, scored 5/6/6/8 (Reject). The current paper extends Zeng et al. with one-step inference (meaningful speed improvement) and IIP for non-binary handling (novel but with large gaps). It is a clear improvement over Zeng et al. in speed, but the non-binary results are weak and lack ablations. Compared to DISCO (5/6/6/6 Reject) and PDGNN (5/5/6/5 Reject), this paper has similar novelty concerns but also has the distinct issue of large optimality gaps on its primary novel contribution (non-binary ILP). Compared to CADO (5/6/6/6 Reject), this paper has more experimental breadth but significant methodology gaps (modified consistency loss, missing ablations). The Shortcut Models paper (8/8/8/8 Accept Oral) represents the high end—a clean, well-justified methodological contribution—which this paper does not approach. Given the significant weaknesses (large gaps on non-binary ILP, missing ablations for core components, overclaiming in the abstract, underspecified methodology), this paper falls below the acceptance threshold but has valuable ideas that could be improved with revision.

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>