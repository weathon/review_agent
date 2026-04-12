## Summary
This paper proposes **SparseFW**, a layerwise post-training pruning method for LLMs that relaxes binary mask selection to a convex program over the mask polytope and solves it with Frank–Wolfe. The paper’s most distinctive technical contributions are (i) a clean optimization view of mask selection and of Wanda/RIA as greedy approximations to that objective, (ii) an efficient FW implementation via precomputing \(G=XX^\top\) and \(H=WG\), and (iii) empirical evidence that FW can substantially reduce local reconstruction error and modestly improve final perplexity/zero-shot accuracy over Wanda/RIA on several 7B–14B models.

The paper is novel and technically interesting, but the central empirical story is more fragile than the framing suggests: the version that works best is not vanilla FW, but a hybrid that fixes a large fraction of Wanda-selected weights and only optimizes the remainder. This does not negate the contribution, but it does materially change what the paper has shown.

## Strengths
- **A genuinely insightful optimization reinterpretation of existing pruning heuristics.** Section 2.1 does more than present another pruning method: it derives Wanda as minimizing the one-weight pruning objective without reconstruction and interprets RIA as the same greedy procedure on a rescaled weight matrix. This is a specific conceptual contribution that clarifies what these popular methods are actually optimizing.
- **The convex-relaxation formulation is principled and algorithmically well matched to FW.** The feasible set \(C_k=\{M\in[0,1]^{d_{out}\times d_{in}}:\|M\|_1\le k\}\) is a natural relaxation of binary mask selection, and the LMO is especially simple: select up to \(k\) most negative gradient entries. This makes the method mathematically clean while preserving sparse updates.
- **The implementation insight is practically meaningful.** The paper exploits that both objective and gradient depend on \(X\) only through \(G=XX^\top\), and precomputes \(G\) and \(H=WG\). This reduces dependence on calibration sequence length and sample count during iterative optimization, which is a concrete systems/engineering strength rather than a generic “efficient implementation” claim.
- **The method supports both unstructured and semi-structured sparsity within the same framework.** Appendix D shows how the LMO extends to \(n\!:\!m\) sparsity by separability over blocks; empirically, the paper reports results for both 50/60% unstructured and 2:4 sparsity across multiple modern GPT-family models.
- **The paper is unusually transparent about a key failure mode.** Section 2.3 and Appendix C explicitly state that unconstrained vanilla FW often improves the local pruning objective yet can worsen perplexity, and they provide the \(\alpha\)-ablation showing this. That honesty is valuable and helps readers understand the real scope of the method.

## Weaknesses
###: Fatal
None.

### Major:
- **The empirical success is driven by a hybrid “fix-most-of-Wanda” variant, not by vanilla FW alone, and the paper’s framing understates this.**  
  This is directly supported by the paper itself. Section 2.3 states:  
  > “setting \(\alpha = 0.0\) (full FW without any fixed weights) consistently yields worse results than the baselines.”  
  Appendix C further shows that the strongest gains often occur around \(\alpha=0.9\), i.e., fixing 90% of the kept weights by Wanda saliency and optimizing only the remaining 10%. This substantially weakens the headline narrative that FW “overcomes” greedy pruning. The evidence instead supports a more nuanced claim: **FW is useful as a constrained local refinement on top of a strong saliency prior**. That is still interesting, but materially narrower than the abstract/introduction suggest.
- **The main-text algorithm presentation obscures the method actually used in the strongest experiments.**  
  Algorithm 1 presents plain FW plus thresholding, while the practically necessary variant appears only later as a “caveat” in Section 2.3 and formally in Appendix B (Algorithm 2). Given that the paper itself reports that \(\alpha=0\) is consistently worse than baselines, the saliency-fixing mechanism is not an implementation detail; it is central to the empirical method. This affects clarity and claim calibration.
- **Theoretical guarantees are only partially aligned with the empirically successful algorithm.**  
  The main theory in Section 4 / Appendix E establishes guarantees for the relaxed problem plus top-\(k\) rounding of the FW iterate. But the experimentally strongest method adds an extra constraint: fixing a subset of high-saliency weights beforehand and optimizing only the complement. The paper does not extend the guarantee to this hybrid algorithm, even though that is the version supporting the main empirical claims. So the theory is sound as far as it goes, but it does not fully justify the method that actually matters most in practice.
- **The paper lacks a concrete compute-cost accounting despite using substantially more optimization than the baselines.**  
  The paper acknowledges this limitation (“SparseFW is clearly more compute-intensive than Wanda and RIA”) and reports using 2000 FW iterations per layer. But there is no wall-clock, FLOP, or pruning-time comparison to Wanda/RIA. Since the final perplexity gains are often modest in absolute terms, it is important to quantify the cost/benefit trade-off rather than discuss it qualitatively.
- **The connection between large local objective improvements and modest end-task gains remains underexplained.**  
  The paper convincingly shows sizable reductions in per-layer pruning error (Figure 2; often 20–40% average, up to 80% in some layers), yet final perplexity improvements in Table 1 are much smaller and sometimes mixed. The paper does acknowledge a “mismatch between local and global objectives,” but this becomes a central unresolved issue: the proposed optimization target is demonstrably improved, yet that does not reliably translate into corresponding model-level gains without additional inductive bias from Wanda.

### Minor
- **Main-result uncertainty is hard to assess because Table 1 omits standard deviations.**  
  The paper says “We omit standard deviations for legibility,” but some reported improvements are small enough that variability matters for interpretation. Figure 3 does include seed ranges for one ablation, which is helpful, but the main comparison table would be stronger with at least compact uncertainty reporting.
- **The paper’s “state-of-the-art” phrasing should be narrowed to the comparison class actually studied.**  
  Section 3 explicitly restricts comparisons to methods “that also aim to find a better pruning mask by solving (MASK SELECTION)” and excludes reconstruction-based approaches such as SparseGPT. That is a reasonable scoped evaluation choice, but then claims should be consistently phrased as improvements over strong **mask-selection** baselines rather than over LLM pruning methods broadly.
- **The mechanism behind the stronger gains in some sparsity regimes/patterns is not deeply analyzed.**  
  The 2:4 and higher-sparsity settings seem to benefit more consistently than 50% sparsity, but the paper does not provide much insight into when FW refinement is most valuable and why.

### Trivial
- **The role of thresholding dynamics could be explained more directly.**  
  Figure 4 is interesting and the discussion is plausible, but readers would benefit from a clearer practical takeaway: whether the thresholding plateau is mostly due to the FW step-size schedule, lack of vertex convergence, or an inherent property of the relaxation in this setting.

## Nice-to-Haves
- Add a pruning-time / accuracy Pareto analysis against Wanda and RIA.
- Report mask overlap or Hamming/Jaccard distance between the warmstart and final SparseFW masks to show how much the optimization actually changes.
- Move the \(\alpha\)-ablation from the appendix into the main paper, since it is central to interpreting the method.
- Analyze which layer types or matrix types contribute most to the final gains, given the local/global mismatch.
- If space allows, include at least one broader baseline with light reconstruction, while keeping the paper’s primary scoped comparison intact.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The theoretical guarantee does not apply to unstructured sparsity at all because it is only row-wise.”**  
  Removed in this strong form because it overstates the issue. The appendix explicitly says:  
  > “For simplicity, we work in the row-wise formulation; the proof for the full-matrix case follows by the same arguments.”  
  So it is not accurate to claim the theory is strictly limited to row-wise/separable settings. The valid concern is narrower: the theory does not cover the **hybrid fixed-mask variant** used for the best results.
- **Criticism based on absence of release / reproducibility of cited models or tools.**  
  Removed per instruction.
- **Pure formatting/style objections.**  
  Removed per instruction.
- **Demands for many extra downstream benchmarks as a core flaw.**  
  Weakened and not kept as a main weakness. The current evaluation on WikiText perplexity and EleutherAI zero-shot accuracy is not unusually narrow for this line of work; additional tasks would strengthen the paper but are not necessary to establish the paper’s scoped claims.
- **Claims that the method is not novel because it “just uses a different optimizer.”**  
  Removed in that simplistic form. Recasting mask selection as convex relaxation with FW, providing the greedy reinterpretation of Wanda/RIA, and deriving theory are real contributions. The better criticism is that the empirical gains depend heavily on the hybridization with Wanda.

## Novel Insights
The most important synthesis is that this paper’s real contribution is not “Frank–Wolfe replaces greedy pruning,” but rather **Frank–Wolfe exposes a useful separation between globally indispensable weights and locally refinable ones**. The experiments suggest Wanda is good at identifying a protected core set of weights, while FW improves the combinatorial search only on the residual degrees of freedom. In that sense, the paper uncovers a structural fact about LLM pruning: optimizing the local quadratic objective more faithfully helps, but only after injecting a strong prior about which weights must remain untouched. This makes the work more interesting scientifically than a simple “better benchmark number” paper, but also means the paper should be reframed around hybrid optimization rather than around pure replacement of greedy heuristics.

## Suggestions
- Reframe the paper’s main claim around **hybrid saliency-constrained FW refinement**, not pure FW replacing greedy heuristics.
- Move Algorithm 2 and the \(\alpha\)-ablation into the main text, and make clear upfront that this is the primary practical method.
- Either extend the theory to the fixed-mask variant or clearly separate “theory for vanilla relaxation” from “empirical gains for the hybrid algorithm.”
- Add a concise table with pruning wall-clock/runtime and memory overhead versus Wanda/RIA.
- Include uncertainty for Table 1, even if only for a representative subset or via compact ± values.
- Quantify how much SparseFW changes the warmstart mask, and where those changes occur across layers/matrix types.
- Clarify in the abstract/introduction that the method improves over strong **mask-selection baselines**, and that unconstrained FW alone does not reliably improve perplexity.