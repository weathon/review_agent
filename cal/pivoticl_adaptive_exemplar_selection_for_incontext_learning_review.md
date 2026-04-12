=== CALIBRATION EXAMPLE 11 ===

# Final Consolidated Review
## Summary
This paper proposes Pivot-ICL, an exemplar-selection framework for in-context learning that builds a weighted bipartite graph between test queries and exemplar candidates, then uses graph scores (primarily HITS-style authority/hub scores) to choose between query-specific dynamic exemplars and task-level static exemplars. The central claim is that this adaptive routing helps when some queries are well covered by the exemplar pool while others are effectively out-of-distribution, and the paper provides evidence across planning, math, commonsense, and science QA tasks, plus several LLM backbones.

## Strengths
- **The paper identifies and empirically validates a genuinely useful failure mode of standard ICL exemplar selection:** dynamic nearest-neighbor retrieval is not uniformly best, and static/global exemplars can be preferable when the query is weakly covered by the exemplar pool. The cleanest evidence is the PDDL ID/OOD analysis in Section 4.4, where dynamic retrieval is better in-distribution while static exemplars become better out-of-distribution.
- **The graph perspective yields stronger task-level static exemplar selection than the clustering-style static baseline used here.** In Table 1, graph-based static methods such as Authority/PageRank consistently outperform Auto-CoT overall, which supports the claim that bilateral exemplar-query structure is useful rather than merely decorative.
- **The method is lightweight relative to loss-based exemplar selection approaches.** The paper’s contribution is not just “another selector,” but a zero-shot routing mechanism that uses embeddings plus graph scoring rather than repeated LLM scoring or bandit-style search. Appendix A.6 gives a concrete complexity discussion and makes a credible case that the extra cost is modest compared with generation.
- **The gains appear to transfer across substantially different backbones.** Table 2 shows Pivot-adapt outperforming static or dynamic-only choices on SQA/GPQA for Gemini 2.0 Flash, Llama 3.3 70B, and Qwen 2.5 7B, which strengthens the significance of the idea beyond a single model/task pairing.
- **The paper does more than present a headline method; it probes several design choices.** The node-construction and edge-construction ablations in Section 4.5 are useful and show that some seemingly richer hybrid constructions can hurt, which is a nontrivial empirical insight.

## Weaknesses

### Major:
- **Pivot-adapt is fundamentally transductive in its current form, which materially limits the claimed deployment setting.**  
  The paper explicitly defines the graph over the full query set and exemplar set, \( G=(C \cup Q, E, W) \), and computes query scores from that graph. The threshold for routing in Section 3.4 is also defined as \( t_\nabla = \alpha / (|C||Q|) \), so the routing rule depends on the full test set size. This is not a reviewer misread: the paper itself later acknowledges in Section 5, “sometimes, there is no observed full set of test examples,” and proposes a future workaround via k-fold validation over exemplars. That acknowledgement is useful, but it does not remove the limitation. As written, the main adaptive method is an offline batch method rather than a standard per-query ICL selection mechanism. This weakens the practical claim of “zero-shot adaptive treatment” for online or streaming inference.
- **The paper does not isolate whether HITS-style iterative bilateral scoring is necessary, versus a much simpler similarity-based routing rule.**  
  The core scientific claim is not merely that “switching between static and dynamic helps,” but that the graph-based bidirectional scoring identifies when to switch. However, there is no direct baseline that routes using simple query-level similarity statistics, e.g., max/mean cosine similarity to exemplars or a nearest-neighbor confidence threshold. The paper compares Pivot-adapt against static-only and dynamic-only methods, and Pivot-concat provides a different thresholding mechanism, but this still does not isolate the value added by iterative hub/authority scoring. Given that the dynamic selector itself is standard similarity-based Gecko retrieval, this missing ablation leaves the methodological contribution under-identified.
- **The thresholding used for Pivot-adapt is heuristic and insufficiently justified for cross-dataset robustness.**  
  Section 3.4 sets \( t_\nabla = \alpha/(|C||Q|) \) with a single empirically chosen \(\alpha=2000\), and footnote 1 states this “can be optimized with a development set.” This is a reasonable practical heuristic, but it undercuts the stronger framing of a plug-and-play zero-shot decision rule. The paper does not explain why HITS hub scores should scale compatibly with \(|C||Q|\) across tasks of very different graph sizes and densities, nor does it provide a sensitivity analysis for \(\alpha\). Appendix A.5 discusses thresholding more generally, but the evidence remains thin for the adaptive threshold specifically.
- **The evidence for the central “hub score detects poor exemplar coverage/OOD” hypothesis is convincing only on PDDL, and remains indirect on the other datasets.**  
  Section 4.4 is a strong motivating analysis, but it uses a specially structured setting where OOD is naturally defined by block count and all exemplars come from 3–7 blocks. For AIME24, SQA, and GPQA, the paper asserts that graph scores identify examples “more out of distribution,” but it does not quantify the relation between low hub scores and actual failure under dynamic retrieval, accuracy, or independent hardness/OOD measures. Since this is central to the mechanism, stronger per-example analysis outside PDDL would materially improve the paper’s technical support.

### Minor
- **Statistical uncertainty is not reported, which matters for some of the smaller evaluations.**  
  Table 1 and related results are presented as single point estimates. For datasets such as AIME24 (30 problems), small absolute differences can be unstable, especially with temperature 1 decoding. In large-scale LLM benchmarking, single-run reporting is common, so this is not a fatal flaw, but for a method paper making relatively modest margins on some tasks, some indication of variability would strengthen confidence.
- **The comparison to LENS/EXPLORA in Table 3 is only loosely informative because it is not under a matched protocol.**  
  The paper itself says these results follow the original EXPLORA settings with GPT-4o-mini and 5 exemplars, and the numbers for LENS/EXPLORA are extracted from prior work. This is acceptable as a rough positioning experiment, but the text should be more careful not to overstate this as a direct head-to-head validation of comparability.
- **Some graph-construction choices remain weakly motivated.**  
  Examples include retaining the top-100 edges per query and using the \(\mu_q + 2\sigma_q\) heuristic in Pivot-concat. The paper does provide some threshold discussion in Appendix A.5, so this is not an unaddressed omission, but these design choices still read as empirical heuristics more than principled components.

### Trivial
- **The practical trade-off versus simple embedding retrieval could be clearer.**  
  Appendix A.6 argues the graph overhead is small relative to LLM inference, which is plausible, but the paper could more explicitly compare runtime/latency against the strongest simple retrieval baseline in the main text.

## Nice-to-Haves
- Add a direct routing ablation that replaces hub scores with simple similarity statistics (e.g., max similarity, average top-k similarity, or similarity margin) to test whether iterative graph scoring truly adds value.
- Report per-example analyses showing the correlation between hub scores and downstream accuracy, or between hub scores and gains from static-over-dynamic routing, on AIME24/SQA/GPQA in addition to PDDL.
- Evaluate an inductive approximation for the no-full-test-set setting, since the paper already discusses this limitation in Section 5.
- Analyze ordering/position effects for Pivot-concat, since concatenating static and dynamic exemplars can interact with known prompt-position sensitivity.
- Provide a sensitivity study for \(\alpha\) in Pivot-adapt specifically, not only general threshold discussion.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Claim that the paper fails to distinguish itself from unspecified prior graph-based methods or has “limited novelty” because graph search for LLMs exists.**  
  Removed because this criticism relies on external related-work assertions not verifiable from the paper alone, and the cited examples in the reviews concern broad graph/tree augmentation for LLMs rather than a clearly established duplicate of this paper’s adaptive exemplar-routing setup.
- **Criticism of unreleased/non-verifiable models or tools.**  
  Removed per instruction. The paper cites the models/tools it uses, so their existence/release status should not be questioned.
- **Pure reproducibility nitpicks about implementation details and artifact release.**  
  The paper includes substantial implementation detail in Appendix A.2/A.7 and promises code release. Missing precomputed graphs, training logs, etc., are not appropriate core weaknesses here.
- **Generic strength claims such as “the paper is well-written” or “experiments are extensive.”**  
  Removed because they are too generic without paper-specific evidence.
- **Strong claim that macro-averaging in Table 1 is misleading because PDDL dominates the average due to scale.**  
  Removed because the table reports macro-average over four tasks, so each task contributes equally; this specific criticism is factually off.
- **Concern that the method is invalid because HITS may collapse on dense graphs or may not converge meaningfully.**  
  We do not see evidence in the paper that this causes a practical problem, and the experiments/ablations do not support elevating this beyond speculation.

## Novel Insights
The most compelling synthesis across the reviews is that the paper’s real contribution is less “HITS for ICL” in isolation and more an operational reframing of exemplar selection as a routing problem under imperfect exemplar coverage. The strongest evidence is not the average gain itself, but the PDDL analysis showing that static and dynamic exemplars dominate in different regions of the query distribution. This suggests a broader design principle for ICL systems: exemplar selection should be treated as conditional policy selection based on coverage/confidence, not just better retrieval. At the same time, the present paper has not yet shown that graph-based mutual reinforcement is the right or necessary confidence signal, which is the key unresolved scientific question.

## Suggestions
- Add a simple but decisive routing baseline: use Gecko dynamic retrieval unless the query’s top-1/top-k similarity score falls below a threshold, then switch to static exemplars.
- Quantify, on all datasets, whether low hub scores actually predict (i) worse dynamic-only performance and/or (ii) larger benefit from static routing.
- Either reframe Pivot-adapt explicitly as an offline batch/transductive method, or add an inductive approximation experiment for sequential/streaming inference.
- Provide a targeted sensitivity plot for \(\alpha\) in Pivot-adapt across at least two tasks with very different \(|C|\) and \(|Q|\).
- Tone down the direct-comparability claim around Table 3 unless a matched-protocol experiment can be added.
- If space permits, include a short case study of successful and failed routing decisions to show what kinds of queries are being identified as poorly covered.

# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 0.0]
Average score: 1.3
Binary outcome: Reject
