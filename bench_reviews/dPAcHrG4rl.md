## Summary
This paper combines an information-theoretic framing of single-pass LLM reasoning for multi-hop QA with a practical multi-call framework, InfoQA. The theory derives a Fano-style upper bound and uses a parametric demand model to argue for an “Accuracy Cliff” as hop count and context length increase; the empirical side introduces a controlled synthetic benchmark and shows that InfoQA, via decomposition plus query contraction/pruning, is substantially more robust than single-pass prompting baselines under those controlled conditions.

## Strengths
- **The paper turns an intuitive systems limitation into a concrete formal lens.** Theorem 1 is a valid combination of conditional Fano and an output-entropy bound, and the paper uses it to articulate a clear capacity-vs-demand perspective on why single-pass reasoning should fail as problem complexity grows. Even if some downstream modeling choices are debatable, the core formal statement is legitimate and gives the paper a sharper conceptual backbone than typical prompting papers.
- **The synthetic benchmark is unusually well-matched to the paper’s stated goal of controlled stress-testing.** The authors explicitly vary hop count (1–4) and context length (0.5k–10k), place evidence out of logical order, and populate contexts with semantically similar distractors rather than only irrelevant filler. This design is specifically useful for isolating how depth and noise interact, which is exactly the paper’s target phenomenon.
- **InfoQA shows strong, consistent gains in the regime the paper cares about most: deep, noisy, long-context MHQA.** On Qwen3-14B, the overall 2–4 hop average improves from 0.75 (best single-pass baseline, S-C) to 0.86, and the advantages are especially large at 3–4 hops and long contexts. The 4-hop numbers are particularly notable: InfoQA averages 0.80 vs 0.61 for S-C and 0.57 for CoT.
- **The ablations support the core design intuition rather than just the full pipeline.** Removing decomposition (“w/o D.”) hurts sharply, especially as contexts lengthen, and removing pruning also degrades performance, indicating that the gains do not come from a trivial multi-call wrapper alone. This is useful evidence for the specific roles of decomposition and contraction.
- **The paper identifies a practically relevant failure mode of the proposed framework.** The error analysis does not simply claim success; it points to semantic drift in iterative query contraction as the remaining dominant error source. That is a concrete and credible diagnosis that could guide follow-up work.

## Weaknesses
###: Fatal
- **The claimed empirical validation of the theoretical bound is much weaker than the paper presents, because the key quantities in the “predicted curves” are not independently measured but fit post hoc to the same benchmark results.**  
  Section 5.2 states that the authors “fit the parameters θ = (β0, α, γ, C) of our plug-in accuracy bound (Eq. 7) to empirical F1 scores” by minimizing MAE, and Appendix A.5 confirms a grid search over all four parameters. This means the paper does not independently estimate the theorem’s information demand \( \beta = H(A\mid Q,C) \) or capacity \( C = H(Y) \), then test whether the theorem predicts the observed collapse. Instead, it fits an effective demand/capacity model to the observed performance and then reports alignment. As a result, the experiments support the usefulness of a descriptive scaling law, but they do **not** constitute strong validation that the information-theoretic bound itself governs model behavior in the claimed predictive sense. This substantially weakens the paper’s central theory-to-experiment claim.

- **The bridge from the formal theorem to the MHQA demand model is conceptually underspecified and, as written, conflates entropy-based uncertainty with an empirically fitted notion of reasoning difficulty.**  
  The theorem defines information demand as \( \beta \triangleq H(A\mid Q,C) \). But in Section 3.1 the paper introduces a parametric model
  \[
  \beta(h,L)=\beta_0+\alpha L\gamma^{h-1},
  \]
  motivated by baseline complexity, context burden, and hop amplification. This is plausible as a *difficulty proxy*, but the paper continues to speak of it as the same \( \beta \) from the theorem. That identification is not justified. In fact, the paper’s own benchmark construction makes the answer a single entity drawn from a controlled synthetic space, so the entropy of the answer conditioned on query/context is not obviously the quantity that should grow super-linearly with hop count in the way Eq. 6 assumes. The issue is not that Eq. 6 is useless—it may be a reasonable empirical ansatz—but that the paper overstates it as a direct instantiation of the theorem’s information demand, when it is really an effective fitted surrogate. This mismatch undermines the technical soundness of the main explanatory story.

### Major:
- **The “capacity” quantity used in experiments is not tied back convincingly to the formal \(H(Y)\) introduced in the theory.**  
  In Section 2 and Appendix A.3, capacity is defined as \(C = H(Y\mid Q,C)\) (or upper bounded via output vocabulary/length). But in Section 5.2/A.5, \(C\) becomes a fitted scalar chosen to best match F1 curves. The paper never measures or even approximates output entropy from model generations, nor does it explain why the fitted \(C\) should be interpreted as an entropy quantity rather than just an effective nuisance parameter. This weakens statements such as certain prompting methods “increase capacity \(C\)” or “reduce hop inflation \(\gamma\),” because these are not established as identifiable, physically meaningful properties of the model; they are artifacts of a low-dimensional fit.
- **External validity is limited because all substantive claims are validated on a single synthetic benchmark.**  
  The benchmark is well-designed for controlled diagnosis, but the paper’s rhetoric is broader: it claims to expose a fundamental inadequacy of the single-pass paradigm for MHQA. Without any evaluation on standard real-world MHQA benchmarks, it remains unclear how much of the reported cliff behavior and InfoQA’s advantage depends on the benchmark’s templatic structure, evidence placement strategy, and synthetic distractor distribution. For a paper making both theoretical and practical claims, this omission matters.
- **The practical cost of InfoQA is under-analyzed relative to the reported benefit.**  
  The framework is explicitly multi-call, and the paper positions it as a proof-of-concept to transcend single-pass limitations. However, there is essentially no quantitative accounting of inference cost: number of calls, token consumption, latency, or cost-performance tradeoff. Since the main empirical gain comes from decomposing one hard call into multiple easier ones, the absence of a compute/latency analysis leaves the practical significance incomplete.

### Minor
- **Some baseline framing is potentially confusing, especially for methods whose original use often involves iterative interaction.**  
  The paper says all baselines were implemented as “zero-shot, single-pass methods,” including ReAct and Self-Ask. This is not necessarily invalid for the paper’s goal—indeed, the whole point is to compare single-pass paradigms—but the exact single-pass instantiation matters for interpretation. The paper should describe more concretely how these methods were adapted into a strictly single-generation setting and what functionality was intentionally removed.
- **The ablation analysis does not fully isolate all three claimed ingredients of InfoQA.**  
  Table 2 includes ablations for removing decomposition and removing pruning, but the “dependency-explicit workflow” is not isolated as cleanly as the other two. Given that the method is presented as a three-part design, the evidence for the third component is more indirect.
- **The paper’s own error analysis points to semantic drift during contraction, but this is not examined deeply enough.**  
  This seems to be the dominant residual failure mode, yet the paper provides no qualitative examples, per-hop failure statistics, or contraction-sensitivity analysis to verify the diagnosis.

### Trivial
- None.

## Nice-to-Haves
- Evaluate InfoQA on at least one established real-world MHQA benchmark to test whether the observed gains transfer beyond the controlled synthetic setting.
- Report token usage, number of model calls, and latency/cost tradeoffs for InfoQA versus single-pass baselines.
- Add uncertainty or sensitivity analysis for the fitted parameters \((\alpha,\gamma,\beta_0,C)\), since they are estimated from only 24 \((h,L)\) conditions per model.
- Show per-hop success rates or case studies of successful and failed query contraction, especially for the semantic-drift failure mode the paper itself highlights.
- Clarify more explicitly that Eq. 6 is an empirical effective-demand model rather than a direct measurement of \(H(A\mid Q,C)\), unless the authors can independently justify that identification.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The theorem itself is mathematically wrong.”** Removed because this is not supported by the paper text. Theorem 1 is a straightforward and valid consequence of conditional Fano plus \(I(A;Y\mid Q,C)\le H(Y\mid Q,C)\), and the appendix provides the standard derivation.
- **Complaints that certain cited models/datasets/references may not exist or be unavailable.** Removed per instruction.
- **Pure style/presentation praise such as “well-written” or “important topic.”** Removed as too generic.
- **Requests for generic reproducibility minutiae.** Removed because the paper already includes a reproducibility statement, code/data release claims, and sufficient implementation detail for the level of contribution.
- **Baseline unfairness framed as asymmetry against baselines.** Softened rather than kept as a core criticism. The paper intentionally evaluates many methods in a strict single-pass regime because that is its target setting; the main issue is lack of clarity in adaptation, not unfairness per se.
- **Strong claim that Eq. 6 is literally impossible because \(H(A\mid Q,C)\) must be small and constant.** Softened. The paper does not specify a fixed tiny answer set size in the main theorem, and the exact entropy of the synthetic answer distribution is not computed. The real issue is not formal impossibility but the unjustified identification of a fitted difficulty law with the theorem’s entropy quantity.

## Novel Insights
The most important synthesis is that this paper is stronger as a **capacity-aware empirical systems paper with a provocative theoretical framing** than as a clean theory-validation paper. The controlled benchmark and the InfoQA gains do support the practical claim that decomposition plus contraction helps avoid prompt-level overload in long, noisy multi-hop settings. However, the paper currently overclaims by presenting a fitted effective-demand/capacity model as if it were a direct empirical confirmation of a Shannon-style theorem. Reframing the contribution as: “a principled information-theoretic motivation plus an empirically validated effective scaling law and a strong proof-of-concept framework” would make the work more honest and substantially more convincing.

## Suggestions
- **Separate the formal theorem from the empirical surrogate model more explicitly.** State clearly that Eq. 6/7 is an effective phenomenological model inspired by the theorem, not a direct measurement of \(H(A\mid Q,C)\) and \(H(Y)\).
- **Reduce the strength of the validation claim in Section 5.2.** Replace “validate the Fano-style upper bound” with language closer to “test whether a theorem-inspired effective scaling law matches observed performance.”
- **Independently estimate at least one theoretical quantity.** For example, approximate output entropy or mutual information proxies from sampled generations, or calibrate an effective capacity with a separate probing task rather than fitting everything on the evaluation set.
- **Add one real-world benchmark.** Even a smaller-scale experiment on HotpotQA/MuSiQue-style data would materially strengthen the significance claim.
- **Report inference overhead.** Include average calls per question, total tokens processed/generated, and latency relative to the best single-pass baseline.
- **Deepen the analysis of semantic drift.** Provide failure traces showing what constraints are lost during contraction and whether those losses concentrate at specific hops or context lengths.