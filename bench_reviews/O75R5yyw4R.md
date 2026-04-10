## Summary
This paper proposes Iterative Reward-Guided Refinement (IterRef), a test-time scaling method for discrete diffusion models. It frames iterative in-situ refinement of intermediate states as a Multiple-Try Metropolis (MTM) process with a custom noising-denoising transition kernel, providing a theoretical convergence claim. Empirically, it demonstrates improved reward scores over several baselines across text and image generation tasks under various compute budgets.

## Strengths
- **Novel algorithmic synthesis**: The adaptation of the Multiple-Try Metropolis framework to discrete diffusion, using a designed noising-denoising kernel for in-situ state refinement, is a creative and novel approach to the underexplored problem of test-time scaling for this model class.
- **Extensive empirical validation**: The method is evaluated across multiple modalities (text, image), model backbones (MDLM, LLaDA-8B, MaskGIT), and diverse reward functions (Toxicity, Sentiment, CoLA, Perplexity, CLIPScore), showing consistent improvements and robust performance.
- **Insightful practical analysis**: The analysis of where to apply refinement (effective timesteps) and the trade-off between iteration count (`k`) and candidate count (`N`) provides valuable, actionable insights into the dynamics of discrete diffusion and the method's operation beyond the core result tables.

## Weaknesses
### Major:
- **Theoretical foundation is inadequately substantiated**: Proposition 1 claims convergence to the optimal distribution under the assumption that `q` and `pθ` "form a reversible Markov kernel." The forward noising process `q` and the learned reverse process `pθ` in standard absorbing-state discrete diffusion are not reversible with respect to each other. The paper provides no proof or citation in the main text to justify that the specific composite kernel `K` (Eq. 2) satisfies the detailed balance condition required for MTM's convergence guarantee with respect to the target `p*(xt)`. This undermines a core claim of the paper.
- **Misleading and conflated efficiency metric**: All empirical comparisons use a single "Numbers of Function Evaluations (NFE)" metric that treats a generative model call and a reward model call as equivalent. As the paper itself notes (Section 3.3), for large models like LLaDA-8B, generative calls dominate cost, while for smaller models costs are comparable. Basing all scaling and "8x faster" claims on this aggregated metric is misleading, as it obscures the true computational bottleneck and latency profile. The paper's core contribution is an efficient scaling method, so this flaw is significant.
- **Incomplete and potentially unfair baseline comparison**: The related work section cites recent, directly relevant methods like DSearch (Li et al., 2025) and DTS (Jain et al., 2025) for inference-time alignment in discrete diffusion, and also mentions Wang et al. (2025) which addresses the same token irreversibility problem via re-masking. None are included as baselines. The claim that IterRef "consistently outperforms prior reward guidance methods" and represents the state-of-the-art is therefore unproven against the most relevant contemporary work.

### Minor:
- **Lacks analysis of sample diversity**: The method refines a single state iteratively, which could lead to mode collapse. The paper does not measure the diversity (e.g., via Self-BLEU or LPIPS) of the high-reward samples it produces, leaving a gap in understanding its practical utility.
- **Insufficient analysis of performance variance across models**: The paper notes that scaling dynamics differ between MDLM and LLaDA-8B (e.g., gains appear at low NFEs for one and high NFEs for the other) but only provides a speculative, post-hoc explanation for the CoLA result. A deeper investigation into why the method's effectiveness varies with model architecture is missing.
- **Practical implementation details are sparse**: The concept of an "effective timestep set U" is flexible but the paper offers no heuristic or algorithm for choosing `U` for a new task, leaving a gap between formulation and deployment.

## Nice-to-Haves
- A direct ablation comparing `k=1` iteration (with adjusted `N`) to the full iterative procedure would more cleanly isolate the benefit of iteration versus simply sampling more candidates.
- A visualization of how a text sequence evolves across IterRef iterations at a fixed timestep would vividly illustrate the claimed in-situ correction mechanism.
- Reporting separate counts for generative vs. reward model calls, alongside wall-clock time, would provide a transparent and realistic efficiency comparison.

## Removed Points
*These points are flagged to be removed, treat them with caution.*

**Strengths removed**:
- "The paper is well-written" (Generic, applies to many papers).
- "The topic is important" (Generic, applies to many papers).

**Weaknesses removed**:
- **"Figures suffer from severe formatting/OCR artifacts"**: This is an artifact of the PDF extraction for this review process, not a flaw of the submitted paper.
- **"Theoretical guarantee appears tautological or misapplied"**: This point is partially valid but is covered in more precise terms in the "Major" weakness above regarding the unsubstantiated reversibility assumption.
- **"The balancing function choice needs justification"**: The paper provides the balancing function `λ` (Eq. 2) and states it leads to the simplified acceptance ratio `β`. The derivation for this specific `λ` is in Appendix D.2. This is addressed in the paper.
- **"Claim that pool reuse preserves theoretical guarantees needs justification"**: The paper explains the pool reuse is valid because candidates are i.i.d. from the same kernel. This is a reasonable practical approximation, not a fundamental theoretical flaw.
- **"Demand for comparison to gradient-based guidance or user studies"**: This is outside the paper's scope (discrete, non-differentiable setting) and not standard practice for this type of algorithmic contribution.
- **"Request for confidence intervals on large-scale benchmarks"**: Not standard practice in this field for the reported metrics.
- **"Criticism that cited models (e.g., LLaDA-8B) do not exist"**: Hard Rule violation. The paper cites them, they are assumed to exist.
- **From Human Finder**: Points about "smoothness assumptions" and "zero-order optimization" are not directly relevant as this paper does not assume smooth rewards nor use zero-order optimization. Points about "limited data types" (music) and "incomplete baseline comparison" are partially valid but the latter is already covered in a major weakness; the former is a scope critique (paper covers two major modalities).

## Suggestions
1. **Strengthen the theoretical justification**: Either provide a proof in the main text that the kernel `K` satisfies detailed balance with respect to `p*(xt)` given the standard properties of `q` and `pθ`, or significantly qualify the convergence claim, acknowledging the practical approximation and focusing on the empirical effectiveness.
2. **Reformulate the efficiency analysis**: Report results using two separate cost metrics: number of generative model calls and number of reward model calls. Clearly state any "speedup" claims in terms of generative model calls, which are the true bottleneck for large models.
3. **Expand the baseline comparison**: Include direct comparisons to at least Wang et al. (2025) and one of the more recent search-based methods (e.g., DSearch or DTS) to properly situate IterRef's performance within the current state-of-the-art.
4. **Add a diversity metric**: Include a measure of sample diversity (e.g., Self-BLEU for text) in the main results to ensure high rewards are not achieved at the expense of variety.

## Evaluation
- **Novelty**: High. The adaptation of MTM for in-situ refinement in discrete diffusion is novel.
- **Technical Soundness**: **Low**. The theoretical guarantee is inadequately supported, and the empirical efficiency metric is flawed.
- **Empirical Support**: Medium. The experiments are broad but the cost comparison is misleading and key baselines are missing.
- **Significance**: Medium. The problem is relevant and the method shows promise, but the current presentation has significant flaws that limit its impact.
- **Clarity**: Medium. The algorithm and motivation are clear, but the theory is not rigorously presented, and the results are presented with a misleading metric.

**Overall**: The paper presents a novel and intuitively appealing idea with promising empirical results. However, it is marred by a critical, unsubstantiated theoretical claim and a misleading empirical analysis that undercut its core contributions. In its current form, the paper is not technically sound enough for acceptance. It requires major revisions to address the foundational theoretical issue and to provide honest, apples-to-apples efficiency comparisons against the most relevant contemporary work.