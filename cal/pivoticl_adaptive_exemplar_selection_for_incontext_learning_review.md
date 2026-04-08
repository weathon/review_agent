=== CALIBRATION EXAMPLE 28 ===

# Final Consolidated Review
## Summary

Pivot-ICL proposes an adaptive exemplar selection method for in-context learning that models bilateral interactions between test examples and candidate exemplars as a weighted bipartite graph, then applies HITS (Hyperlink-Induced Topic Search) to score both exemplars (authorities) and test examples (hubs). High-scoring test examples receive dynamic (input-specific) exemplars, while low-scoring examples receive static (task-generic) exemplars. Experiments across four challenging reasoning tasks and multiple LLM backbones show consistent performance improvements over purely dynamic or static selection strategies.

## Strengths

- **Principled adaptive mechanism with clear motivation.** The core insight—that test examples vary in how well they are covered by the exemplar pool, and thus benefit from different selection strategies—is well-grounded and empirically validated through the controlled ID/OOD analysis on PDDL (Figure 2), which cleanly shows dynamic selection excelling on ID examples while static selection helps on OOD examples.

- **Creative repurposing of HITS for ICL.** Applying a classic bipartite graph mining algorithm to score both exemplar and test-example nodes bidirectionally is a novel and elegant formulation. It provides a principled way to identify globally representative exemplars (authorities) and globally disconnected test examples (low-hub-score queries) simultaneously, going beyond simple similarity-based retrieval.

- **Efficient and practical.** The method operates purely on embeddings without requiring LLM forward passes for scoring, taking under 10 minutes for graph construction and scoring (Appendix A.6) versus 2–5 hours for loss-based methods like EXPLORA, while achieving comparable or better performance.

- **Consistent gains across diverse tasks and backbones.** Improvements hold across math (AIME24), planning (PDDL), commonsense reasoning (SQA), and PhD-level science QA (GPQA), and across Gemini, Llama, and Qwen models (Table 2), demonstrating genuine generalizability rather than task-specific tuning.

## Weaknesses

### Major:

- **The adaptive mechanism is opaque without per-example assignment analysis.** The paper never reports what percentage of test examples receive static vs. dynamic exemplars for each task, nor the per-group accuracy of examples routed to each strategy. Without this breakdown, it is impossible to verify that the hub-score threshold is making sensible routing decisions rather than, e.g., routing nearly all examples to one strategy. A simple table showing the static/dynamic split ratio and accuracy per group would be straightforward to produce and is essential for validating the core claim.

- **Transductive design limits real-world deployment, and this constraint is under-discussed.** The method requires access to the entire test set Q to construct the bipartite graph and compute hub scores. This means a single query arriving at an API cannot be processed without re-computing graph scores over the entire batch. The authors briefly mention k-fold validation as a workaround in Section 5, but this limitation—which fundamentally shapes the method's applicability—should be prominently acknowledged and discussed in the main text, not just the conclusion. The comparison with EXPLORA in Table 3 further obscures this: EXPLORA is inductive (learns a scoring function deployable per-query), while Pivot-ICL is transductive, making the efficiency comparison incomplete without this caveat.

- **Threshold formula lacks justification and exhibits counterintuitive properties.** The Pivot-adapt threshold $t_\nabla = \alpha / (|C||Q|)$ depends on the total number of test examples $|Q|$, meaning the routing decision for a given query changes if the batch size changes. This is theoretically unusual: the intrinsic "connectedness" of a query to the exemplar pool should not depend on how many other queries are being evaluated simultaneously. The paper provides no theoretical or empirical motivation for this normalization. The authors should either justify why the threshold should scale with $|Q|^{-1}$ or demonstrate that simpler alternatives (e.g., a fixed hub-score percentile) perform comparably.

### Minor:

- **The "+8.8% relative gain over the best baseline" claim in the abstract is not directly verifiable from Table 1.** Computing relative gain over the best individual baseline method's average (approximately 55.7 for Gecko dynamic) yields roughly (60.6 - 55.7)/55.7 ≈ 8.8%, which appears consistent, but the table formatting makes precise verification difficult. The authors should explicitly state which baseline the percentage is computed against, or report the gain computed over multiple baselines for transparency.

- **"Zero-shot" characterization is slightly misleading.** The paper describes Pivot-ICL as providing "zero-shot signals" (Section 1), but α is empirically set to 2000 and could require tuning via a development set (Footnote 1). While Appendix A.5 provides some threshold sensitivity analysis, the main text overstates the zero-shot nature. The method is better described as requiring minimal hyperparameter selection rather than being truly zero-shot.

- **Missing naive ensemble baseline.** Pivot-concat concatenates filtered dynamic exemplars with static exemplars, and Pivot-adapt chooses between the two. A simple baseline concatenating all dynamic + static exemplars (without graph-based filtering or thresholding) would isolate the contribution of the graph-based decision mechanism versus simply having more exemplars available. This comparison is absent.

### Trivial:

- The notation $t_\circ$ and $t_\nabla$ for thresholds is functional but could be more mnemonic (e.g., $t_{\text{concat}}$, $t_{\text{adapt}}$).

## Nice-to-Haves

- **Correlation between hub scores and example difficulty/OOD-ness beyond PDDL.** The ID/OOD analysis (Figure 2) is compelling but limited to one task with an explicit distribution boundary (block count). Showing that hub scores correlate with example difficulty or distributional distance on SQA, GPQA, or AIME24 would substantially strengthen the claim that the graph mechanism is detecting OOD examples.

- **Qualitative case studies.** Showing 3–5 concrete examples where Pivot-adapt routes to static vs. dynamic exemplars, with the actual exemplars selected and model outputs, would provide intuitive understanding of when and why the method works.

- **Exemplar pool size ablation.** Testing performance with varying candidate pool sizes (e.g., 100, 500, 1000 exemplars) would inform practitioners about robustness to different data availability regimes.

- **Embedding model sensitivity analysis.** The method relies on Gecko embeddings for edge weights. Testing with SimCSE or BM25 as the edge-weight function in the graph (not just as standalone baselines) would clarify how sensitive the adaptive mechanism is to embedding quality.

## Removed Points

These points are flagged to be removed; treat them with caution.

- **Formatting/notation complaints from PDF parsing artifacts** (garbled equations): Removed per the rule against pure formatting/style nitpicks. The actual paper would use proper summation notation.

- **Missing ConE comparison**: Removed per the rule against demanding comparisons the paper explicitly scopes out. The paper notes in Section 2 that ConE requires computing model conditional entropy from open-weight models, which is a different operational setting from the embedding-based approach this work targets.

- **Unfair comparison with EXPLORA because it is inductive**: Removed per the rule against complaints where the asymmetry favors the baseline. EXPLORA's inductive capability is an advantage for EXPLORA, not for the authors. The relevant concern (transductive limitation) is already captured above.

- **Claims about limited model diversity**: Removed as the paper tests on five different models (Gemini 1.5 Pro, Gemini 2.0 Flash, Llama 3.3 70B, Qwen 2.5 7B, GPT-4o-mini) spanning different sizes, architectures, and providers. This is adequate coverage.

- **Missing related work citations**: Removed per the rule against mentioning missing related works without external confirmation.

- **Computational cost concerns**: The paper explicitly addresses this in Appendix A.6 with concrete timing numbers (<10 minutes vs. 2–5 hours). Further demands for wall-clock timing in every table are unreasonable.

## Novel Insights

The bipartite graph formulation reveals a structural asymmetry that prior ICL work overlooks: exemplar selection methods implicitly assume that all test queries benefit from the same selection strategy (either dynamic or static), but the distribution of exemplar-test similarities is inherently bimodal. Well-connected queries are best served by their nearest exemplars, while poorly-connected queries are actively harmed by forcing similarity matches that may be misleading. The HITS authority/hub scoring provides an elegant, parameter-light mechanism to detect this structural divide without any training—effectively performing distributional outlier detection at inference time. This insight—that exemplar selection should be *conditional on query-exemplar connectivity rather than uniform*—could generalize beyond ICL to few-shot classification and retrieval-augmented generation, where similar connectivity dynamics likely exist.

## Suggestions

- **Add a per-example routing breakdown table** showing the percentage of examples routed to static vs. dynamic and the accuracy of each group across all four tasks. This is the single most important missing analysis for validating the core claim.

- **Replace or supplement the $|Q|$-dependent threshold** with a simpler alternative (e.g., percentile-based on hub scores within the exemplar set only, or a fixed hub-score percentile from k-fold validation on the exemplar pool) and report whether performance degrades. This would directly address the counterintuitive batch-size dependency.

- **Include a naive concatenation baseline** (dynamic + static exemplars without filtering) to quantify the contribution of the graph-based selection mechanism versus simply providing more exemplars.

- **Prominently acknowledge the transductive setting** in the Method section (not just the Conclusion), and discuss practical implications for streaming/incremental deployment scenarios.

- **Clarify the +8.8% claim** by specifying the exact baseline method(s) and computation in the main text, not just the abstract.

---

**Axis evaluations:**

- **Novelty**: Moderate-to-high. The bipartite graph formulation and HITS application to ICL exemplar selection is genuinely new and well-motivated, though the individual components (HITS, KNN selection, static exemplar sets) are established.

- **Technical soundness**: Moderate. The method is clearly described and largely sound, but the threshold design has unexplained properties, and the adaptive mechanism lacks empirical validation at the per-example level.

- **Empirical support**: Moderate-to-strong. Consistent gains across tasks and backbones, plus the compelling PDDL ID/OOD analysis. Weakened by the missing routing breakdown and limited ablation on hyperparameters.

- **Significance**: Moderate. Exemplar selection is a practical problem, and the adaptive insight is valuable, but the transductive constraint and opaque routing limit immediate practical impact.

- **Clarity**: Good. The paper is well-structured with clear motivation, method description, and experimental organization. The threshold notation could be improved but is not a barrier.

# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 0.0]
Average score: 1.3
Binary outcome: Reject
