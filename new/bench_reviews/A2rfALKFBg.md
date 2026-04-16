## Summary
This paper proposes a new lens for mechanistic circuit tracing: decompose each attention head’s QK bilinear form via SVD, identify a small subset of singular-vector “slices” that dominate a given attention score, and use those slices to trace upstream head-to-head communication in GPT-2 small on IOI. The idea is novel and the empirical case study is interesting: the method appears to denoise upstream attribution substantially, recovers a meaningful fraction of the known IOI circuitry, and surfaces additional candidate structure such as redundant pathways and early feature injection.

## Strengths
- **Novel methodological angle with a clear mathematical object.** Rewriting attention scores as a bilinear form \(A'_{ij}=\tilde x_i^\top \Omega \tilde x_j\) and analyzing the SVD of \(\Omega\) is a clean and original framing for QK-side interpretability. This is not just standard low-rank compression; the paper uses the singular directions to study which input subspaces contribute to an attention score.
- **The denoising effect is one of the paper’s strongest results.** Section 5.2 / Figure 4 gives a convincing qualitative demonstration that tracing through the selected singular-vector subspaces suppresses a large amount of irrelevant upstream activity compared to using the full residual. The filtered maps highlight heads already known to be functionally important for name movers, which is good evidence that the decomposition is not arbitrary.
- **The paper asks a sharper mechanistic question than many tracing papers.** Rather than only asking which heads matter, it asks what low-dimensional features mediate communication between heads. That is a meaningful advance in framing, even if the present evidence is still partial.
- **Validation is better than purely correlational work.** The paper does perform interventions, including both local and global variants plus random-subspace controls, and shows that many traced components matter for IOI performance. The distinction between local and global interventions is especially useful because it acknowledges that end-task effects and direct downstream effects are not identical.
- **The recovered graph is not just re-reporting prior work.** The trace broadly overlaps prior IOI findings while adding candidate structure not emphasized before, including early contributors to the IO token and redundant/lattice-like pathways across layers 7–9.
- **The writing is generally clear and technically organized.** The method sections are readable, the role of \(\Omega\) is explained well, and the paper is transparent about some limitations, e.g., restriction to firing heads and exclusion of MLPs.

## Weaknesses

###: Fatal
None.

### Major:
- **The paper overstates the causal meaning of traced edges relative to what Eq. (7) actually measures.**  
  This is the most important issue. The paper repeatedly uses strong language such as “each edge represents a causal direct effect” and “causal communication paths.” But Eq. (7) is fundamentally a projection-based contribution score in the downstream head’s selected singular-vector subspaces, not a derived estimate of the actual change in the downstream attention score under the full computation. The paper itself acknowledges omitted complications: “processing that may affect the signal in between the upstream head and the downstream head,” feature removal/self-repair, and layer norm effects (§4.3). The interventions in §5.4 show that the identified subspaces are often behaviorally relevant, which is valuable, but they do **not fully validate** the stronger interpretation that a drawn edge in Figure 5 is an identified direct causal communication channel with the exact semantics assigned in the text. This should be reframed more cautiously as identifying plausible, intervention-supported communication candidates rather than direct causal edges by construction.
- **The central sparse-decomposition claim relies on a bespoke sign-based heuristic, and the paper does not show that the phenomenon is robust to alternative sparsification choices or alternative bases.**  
  The set \(S_{ij}\) is defined as keeping the strictly positive terms after labeling as “noise” the largest set whose sum is \(\le 0\) (§4.1). This is a consequential design choice, not a minor implementation detail: the whole signal/noise split and tracing pipeline depend on it. The current evidence shows that positive attention scores for firing heads can often be reconstructed from a small subset of positive slice contributions under this rule, but that is weaker than establishing that the attention computation is intrinsically sparse in the SVD basis in a robust sense. The paper would be much stronger with sensitivity analysis over other slice-selection rules (top-\(k\), magnitude thresholds, reconstruction-error targets, etc.) and with at least one comparison to alternative bases (e.g., random orthogonal or PCA-like baselines) to show that the SVD basis is doing something special rather than merely convenient.
- **The evaluation scope is too narrow for the paper’s broader framing.**  
  All substantive tracing results are on GPT-2 small and one task, IOI. The non-specific-input experiment in Figure 3 is a useful sanity check that some sparsity-like behavior appears off-task, but it does not establish that the tracing method remains faithful or useful beyond IOI. Given the paper’s broad framing around tracing attention-head communication generally, the evidence is still limited to a highly curated “model organism” setting where much of the circuit is already known.

### Minor
- **Several important heuristics are under-justified and lack sensitivity analysis.**  
  Beyond the definition of \(S_{ij}\), the graph depends on the 70% cumulative contribution threshold for selecting upstream heads (§5.3), the >50% attention threshold for a head to count as “firing” (§4.3), and the use of \(\sqrt{\sigma_k}\) in Eq. (7) to split contribution between source and destination. These may all be reasonable first choices, but because they materially shape the resulting graph, the lack of robustness analysis is a real weakness.
- **The validation target is mostly end-task logit difference rather than the specific downstream head behavior the trace claims to explain.**  
  Section 5.4 evaluates interventions mainly through IOI logit difference. That supports behavioral relevance, but a more direct test would examine whether modifying a traced edge predictably changes the specific downstream head/token attention score or attention pattern that the edge is supposed to mediate.
- **The traced circuit is incomplete because MLPs are entirely omitted.**  
  The paper explicitly scopes itself to attention-head communication, so this is not a fatal objection, but it still limits the interpretability of the resulting graph. Readers should not mistake Figure 5 for a full task circuit.
- **Claims about semantic interpretability of the identified signals are only lightly supported in the main text.**  
  The paper’s motivating narrative is about “features used to communicate between attention heads,” but the main-text evidence for feature semantics is limited; the concrete example given is that a signal for head (9,9) separates names from non-names, with details deferred to the appendix. This supports plausibility, but not yet a strong general claim that the traced subspaces are broadly semantically interpretable communication features.
- **Agreement with prior IOI circuitry is useful but only moderate by the paper’s own numbers.**  
  The paper notes appendix results of precision 0.52 and recall 0.69 against Wang et al. after additional filtering. That is encouraging, not dismissive, but it is not strong enough on its own to justify the paper’s more emphatic claims about tracing communication paths faithfully.

### Trivial
- **The “single forward pass” rhetoric is somewhat stronger than the actual end-to-end workflow.**  
  The core tracing computation avoids patching and counterfactual datasets, which is a genuine advantage. Still, the final graph is aggregated over 256 prompts and filtered by frequency/contribution thresholds, so the presentation should be a bit more precise about what is single-pass per example versus what is required for the reported graph.

## Nice-to-Haves
- Add at least one additional task or one additional model to test whether the sparse-decomposition phenomenon and tracing utility generalize beyond IOI in GPT-2 small.
- Compare the SVD basis against at least one alternative basis and compare the current \(S_{ij}\) heuristic against one or two simpler sparsification rules.
- Report robustness of the recovered graph to the 50% firing threshold and 70% contribution threshold.
- Include a more direct validation metric on downstream attention scores/weights, not only final IOI logit difference.
- Expand the semantic analysis of traced signals across several heads, not just one illustrative example.
- Clarify runtime/computational cost versus patching-based approaches; the qualitative efficiency argument is plausible, but quantitative timing would help.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Pure claims that the paper “does not use a single forward pass.”**  
  Removed as stated too strongly. The paper’s claim is about avoiding patching/counterfactual-heavy tracing; while the full reported analysis aggregates over many prompts, the core method per example is still a forward-pass-based tracing procedure. This is better treated as a wording/precision issue, not a substantive flaw.
- **Criticisms that the paper should provide extensive reproducibility minutiae or release-status concerns.**  
  Removed per instruction. The submission already provides the core method and experimental setup at the level expected for review.
- **Any criticism implying unfairness because the method does not compare symmetrically to baselines when the asymmetry favors the baseline.**  
  Removed per instruction. The real issue is not fairness to baselines; it is the lack of quantitative head-to-head comparison needed to establish fidelity.

## Novel Insights
The most interesting synthesis here is that the paper’s real contribution is stronger as a **feature-filtering / candidate-edge discovery tool** than as a fully validated causal tracer. The denoising evidence is genuinely compelling: the singular-vector subspaces appear to isolate a small set of residual directions that are much more diagnostic of functional upstream influence than the raw residual is. If the paper reframed itself around “intervention-supported discovery of candidate communication subspaces” rather than “direct causal edges by construction,” the empirical results would read as substantially more solid. In other words, the work seems closer to a promising interpretability instrument than to a fully faithful circuit extractor—and that is still a meaningful contribution.

## Suggestions
- Reframe the strongest claims: replace “each edge represents a causal direct effect” with a more careful statement such as “each edge represents a candidate communication channel supported by downstream-basis projection and intervention evidence.”
- Add robustness studies for:
  - the \(S_{ij}\) selection rule,
  - the 70% upstream-contribution threshold,
  - the 50% firing threshold,
  - and the \(\sqrt{\sigma_k}\) weighting in Eq. (7).
- Add at least one basis comparison. A random orthogonal basis baseline would already help; a PCA-like or other structured basis comparison would be even better.
- Evaluate direct downstream effects more explicitly: show how ablating/boosting a traced component changes the claimed downstream head’s attention score or attention distribution on the traced token pair.
- Broaden empirical coverage modestly with at least one more task or model.
- If space is limited, prioritize fewer claims with stronger quantitative support—especially around fidelity of the traced edges.

## Score and Decision
**Assessment across axes:**  
- **Originality:** good. The SVD-on-\(\Omega\) tracing angle is genuinely novel.  
- **Importance of question:** high. Understanding what features mediate head-to-head communication is a meaningful mechanistic interpretability question.  
- **Support for claims:** moderate. Some claims are well supported (denoising, behavioral relevance of selected subspaces), but the strongest causal-edge claims are overstated.  
- **Experimental soundness:** moderate. The experiments are thoughtful and nontrivial, but too narrowly scoped and too dependent on unablated heuristics.  
- **Clarity:** good. The paper is readable and technically organized.  
- **Value to the community:** moderate to good. Even if not yet a definitive tracer, this could be a useful tool and framing for future interpretability work.

**Calibration against human-reviewed papers:**  
- I compared this paper against **/home/wg25r/review_agent/human_reviews/fpoAYV6Wsk.md** (“Circuit Component Reuse Across Tasks in Transformer Language Models,” scores 8/6/6/6, accepted), which is stronger because it supports broader generalization claims across tasks/models and ties interventions more convincingly to that story. The current paper is more novel methodologically, but less complete empirically and more overstated in its causal claims, so it should score below that accept-range anchor.
- I also compared it against **/home/wg25r/review_agent/human_reviews/89wVrywsIy.md** (“Automatically Identifying and Interpreting Sparse Circuits with Hierarchical Tracing,” scores 5/3/1/5/3, rejected), which had more severe faithfulness concerns and weaker evidentiary grounding. The current paper is clearly stronger than that: it has a cleaner core mathematical object, clearer qualitative wins, and better-targeted interventions.
- As another nearby anchor, **/home/wg25r/review_agent/human_reviews/LphpWGimIa.md** (“Interpreting Attention Layer Outputs with Sparse Autoencoders,” scores 3/6/6) shows that mechanistic-interpretability papers with interesting empirical observations but limited systematic validation often land around the borderline. This submission feels somewhat stronger than that paper in methodological novelty and targeted intervention design, but still not solid enough for confident acceptance.

On that scale, this paper lands as a **borderline reject**: promising and interesting, but not yet sufficiently validated for its strongest claims.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>