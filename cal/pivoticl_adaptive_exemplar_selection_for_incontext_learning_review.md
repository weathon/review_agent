=== CALIBRATION EXAMPLE 18 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract

The title "Pivot-ICL: Adaptive Exemplar Selection for In-Context Learning" is accurate and descriptive. The abstract's central promise — that exemplar selection should adapt based on whether a test example is in- or out-of-distribution — is clearly articulated. However, the headline claim of "+8.8% relative gain over the best baseline" is computed as a macro-average over four tasks and obscures significant heterogeneity. On AIME24, the absolute difference between Pivot-adapt and the best dynamic baseline is 3.4 percentage points (23.4% vs. 20.0%), which on a 30-sample test set equals exactly **one additional question answered correctly**. This number should not be aggregated with confidence into a headline figure.

---

### Introduction & Motivation

The bilateral intuition is genuinely compelling: a test example that lies far from all available exemplars should receive generic, task-level demonstrations rather than noisy nearest-neighbor matches. The HITS framing (exemplars as authorities, test examples as hubs) is creative and reasonably well-motivated.

That said, the contributions are somewhat oversold. The first and second bullet points ("model bilateral interactions" and "design bilateral weighted graph") are nearly identical restatements of the same contribution. The third ("further analysis shows how performance gain comes from...") is not a contribution but an empirical finding. The actual novel contributions are: (1) the bipartite graph representation of the exemplar–test relationship and (2) using HITS scores to adaptively route queries.

The paper also does not acknowledge a transductive assumption at this stage: the method requires the **entire test set to be available upfront** to build the bipartite graph. This fundamentally limits applicability to batch-inference settings and is only briefly acknowledged at the end of the conclusion.

---

### Method (Section 3)

**Bipartite Graph and HITS (§3.2):** The paper borrows HITS to compute authority (exemplar) and hub (test example) scores via mutual reinforcement. This is technically sound. However, the paper misses an important mathematical connection: when HITS is run on a bipartite graph with symmetric weighted edges (cosine similarity), the authority and hub score vectors converge to the **dominant left and right singular vectors of the bipartite adjacency matrix W**. This is well-known in the spectral graph theory literature. The paper should acknowledge this equivalence and address whether a direct SVD or spectral embedding would yield similar results, since this would clarify *what HITS is actually doing* beyond the suggestive website analogy.

**Pivot-concat threshold (§3.4):** The threshold t◦ = μ_q + 2σ_q is applied per query to keep only "statistically significant" dynamic exemplars. However, edge weights are cosine similarities of embeddings — not samples from a distribution where the phrase "statistically significant" applies. The description is a heuristic masquerading as a statistical criterion. Table 5 (Appendix) shows performance is indeed sensitive to this cutoff (ranging from 53.0 to 62.1 on GPQA), suggesting this parameter matters and the "2σ" choice is not well-principled.

**Pivot-adapt threshold (§3.4):** The threshold t∇ = α/(|C||Q|) with α = 2000 is set empirically across tasks. Footnote 1 says α "can be optimized with a development set" — but then the claim of zero-shot adaptivity is weakened. The paper reports α = 2000 across all tasks, but never provides sensitivity analysis for this hyperparameter. Table 6 shows that using the mean as threshold instead gives clearly lower performance, which means the specific functional form chosen matters. Why α/(|C||Q|) rather than, e.g., a percentile of hub scores? This design choice needs deeper justification or ablation.

**Zero-shot claim:** The method is labeled "zero-shot" because it doesn't use LLM forward passes for scoring. But it requires: (i) the full test set upfront, (ii) a pre-specified embedding model (Gecko), and (iii) an empirically set α. The "zero-shot" label is misleading and should be qualified as "zero inference-call" or "zero LLM-call."

---

### Experiments & Results (Section 4)

**Test set sizes:** The most serious empirical weakness is AIME24. With **only 30 test examples**, Pivot-adapt achieves 23.4% vs. 20.0% for the best baseline — a difference of exactly **1 problem**. No statistical test can establish this as a meaningful result. Similarly, GPQA uses 198 test examples. The paper reports no confidence intervals, no bootstrap significance tests, and no p-values across all tables. For ICLR, this is a significant omission.

**Main results (Table 1):** The table is hard to parse cleanly (likely due to PDF artifacts), but the qualitative conclusions are: Pivot-adapt outperforms on PDDL, AIME24, and SQA, while Degree Centrality achieves the highest GPQA score (66.7%) — outperforming Pivot-adapt's 65.6%. This means Pivot-adapt is not universally best; it is the best on average. The paper does not adequately address why a simple graph-mining heuristic (Degree Centrality) outperforms the full Pivot-adapt on GPQA.

**Comparison with EXPLORA/LENS (Table 3):** This comparison uses a completely different experimental configuration: GPT-4o-mini backbone, 5 exemplars, and three different tasks (GSM8K, TabMWP, SQA*) than the main experiments. EXPLORA results are "extracted from the original paper" rather than re-run. Pivot-adapt trails EXPLORA on GSM8K (93.1 vs. 93.6) and SQA* (92.0 vs. 95.1), and only leads on TabMWP (94.0 vs. 90.1). The authors characterize this as "comparable," but the 3.1-point gap on SQA* is not trivial. More importantly, this comparison is confounded by model versions and API differences.

**ID vs. OOD analysis (§4.4, Figure 2):** This is the most convincing empirical analysis in the paper. The PDDL task provides a clean, ground-truth-defined ID/OOD split (block count), and the results clearly show that dynamic exemplars help in-distribution while static exemplars dominate out-of-distribution. This directly motivates the adaptive approach. However, a key question is left unanswered: **Do the hub scores actually identify OOD examples correctly?** The paper shows that the *strategy* is beneficial but never measures whether the hub-score threshold correctly classifies which PDDL test examples are ID vs. OOD (e.g., precision/recall of OOD detection). This direct validation would be far more convincing.

**Backbone generalization (Table 2):** Results with Gemini 2.0 Flash, Llama 3.3, and Qwen 2.5 are appreciated. However, Gemini 2.0 Flash shows essentially identical performance between dynamic and pivot on SQA (82.2% vs. 82.2%), with only a 2-point gain on GPQA. The gains are more pronounced for weaker models (e.g., Qwen 2.5 7B), which makes sense — weaker models benefit more from better exemplar selection.

**GPQA exemplar leakage concern:** GPQA Diamond (198 test examples) and its exemplar pool (250 questions) are drawn from the same 448-question dataset. Exemplars and test examples share the same domain-level distribution almost by construction. The OOD motivation is weaker here, yet GPQA is still used to validate the adaptive approach. This warrants explicit discussion.

**Missing competitor:** ConE (Peng et al., 2024) is mentioned in related work as a recent state-of-the-art that "leads to good ICL performance" but is explicitly excluded from comparison because it requires "open-weight models." However, the paper tests Llama 3.3 (open-weight) in Table 2 — why not include ConE there?

---

### Ablation Studies (§4.5)

The node construction ablations (Table 4) are informative. The key finding that heterogeneous node construction (ex+r, t) degrades performance while uniform constructions ((ex,t) or (ex+r,t+r)) work well is an important practical lesson. The edge construction ablation shows that the bipartite-only graph is competitive with including exemplar-only edges, which validates efficiency. However, the ablations are only on SQA and GPQA — it would strengthen the analysis to include PDDL where distributional effects are most controlled.

The iteration ablation (last row of Table 4) shows minimal gain from one additional generative resampling step (61.1 vs. 59.1 on GPQA). Given the significant computational overhead of additional LLM calls, this seems like a dead-end rather than a promising direction, and the paper's comment to "leave for future research" is appropriate.

---

### Writing & Clarity

The description of Pivot-concat in §3.4 is confusing. The paper says Pivot-concat "avoids explicit scoring of test examples" but it does implicitly score them via the per-query similarity threshold t◦. The distinction from Pivot-adapt — which uses hub scores — needs a cleaner description. An algorithmic pseudocode box for both variants would substantially improve readability.

The paper also refers to "Pivot-Aaapt" (sic) in Appendix A.5, a typo that reveals hasty proofreading in the appendix.

---

### Limitations & Broader Impact (Appendix A.1)

The limitations section acknowledges two issues: (1) treating exemplars as atomic units rather than subgraphs, and (2) limited exemplar pool sizes. Notably absent:
- The **transductive assumption** (full test set needed upfront) is not listed as a limitation.
- **Sensitivity to the embedding model** choice is not discussed; the method relies on Gecko, and it's unclear how performance degrades with weaker embedders.
- **Computational scalability**: For very large test or exemplar sets, computing all pairwise similarities is O(|Q|·|C|·d) where d is embedding dimension, which can be expensive. While the HITS step itself is fast, the pairwise similarity computation is not discussed.

---

### Overall Assessment

Pivot-ICL presents a genuinely interesting framing of ICL exemplar selection as a bipartite graph problem, and the core intuition — that OOD test examples should receive generic exemplars rather than noisy nearest neighbors — is well-motivated and supported by the clean PDDL ID/OOD experiment. The HITS-based scoring is elegant and the efficiency advantages over loss-based methods (EXPLORA) are real. However, the paper's empirical case is significantly undermined by the absence of any statistical significance testing across all results, and the reliance on AIME24 with 30 test examples as one of four headline tasks is particularly troubling. The adaptive routing mechanism's central claim — that hub scores correctly identify OOD examples — is never directly validated with a precision/recall analysis. The transductive assumption (full test set required) is underacknowledged. The comparison with EXPLORA uses different backbones and extracted numbers. Taken together, the contribution has a sound conceptual core but the empirical validation is too thin for ICLR's standard. In its current form, this paper is borderline; the authors would need to add significance testing, directly validate OOD detection quality, discuss the transductive limitation more prominently, and either re-run EXPLORA/LENS comparisons under controlled settings or clearly caveat the existing comparison.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes Pivot-ICL, a method for adaptive exemplar selection in In-Context Learning (ICL) that models interactions between test inputs and candidate exemplars using a bipartite graph. By applying the HITS algorithm to compute authority and hub scores, the method dynamically switches between input-specific dynamic exemplars and task-level static exemplars based on a test example's connectivity in the graph. The approach demonstrates robust performance gains (+8.8% relative) across multiple complex reasoning tasks and backbone models compared to standard static and dynamic baselines.

### Strengths
1.  **Clear Intuition and Adaptivity:** The method effectively addresses a known limitation in ICL: the trade-off between specific (dynamic) and general (static) exemplars. The distinction between in-distribution (ID) and out-of-distribution (OOD) examples (Section 4.4) is well-motivated, and the graph-based decision mechanism provides a principled, zero-shot way to handle this variance.
2.  **Strong Empirical Evidence:** The paper provides extensive experiments across four distinct challenging tasks (PDDL, AIME24, SQA, GPQA) and multiple LLM backbones (Gemini, Llama, Qwen). The consistent performance improvements over strong baselines like Gecko and Auto-CoT (Table 1 and Table 2) demonstrate the practical utility of the approach.
3.  **Efficiency and Zero-Shot Nature:** Unlike loss-based selection methods such as LENS or EXPLORA, Pivot-ICL does not require gradient computation or additional training on the task. The computational overhead is primarily graph construction, which the authors note is minimal (Appendix A.6) compared to inference costs.

### Weaknesses
1.  **Hyperparameter Sensitivity:** The performance depends on thresholds ($\alpha$ for Pivot-adapt, $\sigma$ for Pivot-concat). Appendix A.5 indicates these need optimization per task (e.g., 2$\sigma$ is a "sweet spot"), suggesting the method is not entirely plug-and-play without a development set tuning, which slightly contradicts the "zero-shot" claim.
2.  **Comparative Fairness:** In Table 3, comparisons with LENS and EXPLORA are made with different configurations (5 exemplars vs. 10 in Table 1, different tasks). While justified as extracting original values, a direct head-to-head comparison under identical settings would strengthen the claim of efficiency and performance superiority.
3.  **Graph Construction Constraints:** The method limits edges to the top-100 per test example (Section 3.2). While this saves compute, it risks ignoring weaker but potentially relevant long-tail connections that might be crucial for ambiguous OOD cases. The sensitivity to this threshold is not explored.

### Novelty & Significance
**Novelty:** The application of the HITS algorithm to model bilateral relations specifically for *adaptive decision-making* in ICL is a novel contribution. While graph-based retrievals exist in NLP, using authority/hub scores to trigger a "pivot" between static and dynamic exemplar strategies is a distinct architectural innovation.
**Significance:** The work addresses a critical reliability issue in deploying ICM-based reasoning systems—handling examples that fall outside the distribution of available demonstrations. This method improves robustness without the compute cost of retraining or loss-based scoring, making it significant for real-world application where exemplar pools are fixed.
**Clarity:** The mathematical formulation (Equations 1-2) is conceptually clear, though parsing artifacts obscure the summation symbols in the provided text. The experimental setup is generally well-documented.
**Reproducibility:** The paper commits to open-sourcing code (redacted URL). Details on embedding models (Gecko) and HITS iterations are provided. A strict reproducibility audit would benefit from the code release and an explicit latency benchmark in wall-clock time.

### Suggestions for Improvement
1.  **Clarify Tuning Requirements:** Explicitly quantify the "zero-shot" aspect. If $\alpha$ or $\sigma$ must be tuned via validation set, detail the overhead of this tuning or provide an empirical default that works across all domains without validation.
2.  **Enhance Baseline Comparisons:** For the comparison with LENS and EXPLORA (Table 3), consider providing the performance of your method on those exact tasks with 5 exemplars to ensure apples-to-apples comparison, or clearly justify why the 5-exemplar setting is fair for a method that suggests "adaptive" usage.
3.  **Analyze Graph Sparsity and Failure Cases:** Include a case study or analysis of *when* Pivot-ICL fails. Does it over-select static examples when dynamic ones were needed? Does the top-100 edge cutoff cause issues in very large exemplar pools?
4.  **Robustness of HITS:** Discuss the convergence stability of HITS on sparse graphs. Does the method perform consistently if the exemplar pool has low connectivity (a common scenario in specialized domains)?

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add experiments varying the batch size of test examples used to construct the graph (e.g., 1 vs. 100 vs. full set), because the current transductive setting assumes unrealistic access to all test inputs before inference.
2. Include OOD splits for AIME and GPQA (e.g., by topic) to verify the hub score mechanism detects distributional shift beyond the synthetic PDDL block counts.
3. Report statistical significance tests (e.g., bootstrap confidence intervals) for AIME24 and GPQA, as the small test set sizes (30 and 198) combined with stochastic decoding make gains susceptible to variance.
4. Provide a sensitivity analysis for the hyperparameter $\alpha$ across tasks, because if the optimal threshold varies significantly, the claim of "zero-shot" adaptive treatment is undermined.
5. Compare against stronger recent dynamic selection baselines (e.g., diversity-aware or LLM-scored), as basic KNN/Embedding baselines may underestimate the true performance gap.

### Deeper Analysis Needed (top 3-5 only)
1. Analyze the distribution of hub scores across tasks to verify if a natural bimodal split exists, because without it, the thresholding mechanism appears arbitrary rather than data-driven.
2. Compare HITS scores against simple bipartite degree centrality or one-step similarity sums to justify the computational cost of iterative graph mining.
3. Evaluate performance variance when using weaker embedding models (e.g., SimCSE vs. Gecko) to determine if the graph method is brittle to representation quality.
4. Quantify cases where Pivot-adapt incorrectly switches to static for in-distribution examples (false negatives) to understand the failure modes of the gating mechanism.
5. Provide a detailed breakdown of graph construction time vs. LLM inference time across varying test set sizes, as $O(|C||Q|)$ similarity computations may bottleneck large-scale deployment.

### Visualizations & Case Studies
1. Plot hub scores against accuracy for both Dynamic and Static methods to visually validate the claimed intersection point where static exemplars become superior.
2. Provide side-by-side examples where Pivot-adapt selected Static over Dynamic with model outputs to demonstrate *why* the switch improved reasoning.
3. Visualize a subgraph showing high-authority exemplars connected to low-hub test examples to illustrate the "generic support" mechanism claimed in the paper.
4. Show failure cases where low hub scores did not correlate with OOD status to expose limits in the graph's ability to detect difficulty.

### Obvious Next Steps
1. Develop an inductive variant that clusters exemplars offline without test set access, as the current transductive requirement severely limits real-world deployment.
2. Replace the fixed $\alpha$ hyperparameter with a validation-based or heuristic approach to make the method truly task-agnostic.
3. Evaluate on larger standard benchmarks (e.g., MMLU, BigBench) to demonstrate scalability beyond the current small-scale test sets.
4. Investigate alternative graph algorithms beyond HITS to determine if the specific mutual-reinforcement mechanism is necessary for the observed gains.

# Final Consolidated Review
## Summary

Pivot-ICL proposes modeling the relationship between exemplars and test examples as a weighted bipartite graph, using the HITS algorithm to compute authority scores for exemplars and hub scores for test examples. High-hub-score examples receive dynamic (input-specific) exemplars, while low-hub-score examples receive static (task-generic) exemplars. The method achieves consistent improvements across four reasoning tasks and multiple backbone LLMs.

## Strengths

- **Clear and well-motivated intuition:** The core idea—that test examples far from available exemplars should receive generic demonstrations rather than potentially misleading nearest-neighbor matches—is grounded in the ID/OOD analysis (Figure 2), which cleanly shows dynamic exemplars help ID cases while static exemplars help OOD cases in PDDL.

- **Elegant graph-based formalization:** Modeling exemplar-test relationships as a bipartite graph and applying HITS to jointly score both sides is a natural formulation that enables principled adaptive routing without LLM forward passes for scoring.

- **Consistent improvements across tasks and models:** Table 1 and Table 2 show Pivot-adapt achieving gains over baselines on PDDL (+8.0 absolute over Gecko), AIME24 (+3.4), SQA (+1.5-1.9), and GPQA (+3.5) across Gemini, Llama, and Qwen backbones. The relative gains hold across model families.

- **Efficiency advantage over loss-based methods:** Unlike EXPLORA or LENS, Pivot-ICL requires no gradient computation or LLM forward passes for scoring. Appendix A.6 notes HITS overhead is under 10 minutes vs. 2-5 hours for loss-based methods.

## Weaknesses

- **Small test set sizes raise statistical concerns:** AIME24 uses only 30 test examples; the 23.4% vs. 20.0% improvement over the best baseline is a difference of one correct answer. GPQA uses 198 examples. No confidence intervals, bootstrap tests, or p-values are reported for any results, which is problematic for ICLR standards given these sample sizes.

- **Headline aggregate metric obscures task-level heterogeneity:** The "+8.8% relative gain" is a macro-average over four tasks with very different baselines and scales. On GPQA, Degree Centrality (66.7%) actually outperforms Pivot-adapt (65.6%), yet this is averaged into the headline claim. The paper should report per-task significance and be clearer about where Pivot-adapt does not achieve best performance.

- **No direct validation of OOD detection accuracy:** The paper demonstrates that adaptive routing improves overall performance and that static exemplars help OOD examples in PDDL. However, it never measures whether hub scores correctly classify examples as ID vs. OOD (precision/recall analysis). Without this validation, the claim that hub scores identify distributional shift remains unsubstantiated.

- **Transductive assumption limits deployment scenarios:** The method requires the full test set to build the bipartite graph (Section 3.2: "weighted bipartite graph G = (C ∪ Q, E, W)"). The conclusion briefly mentions k-fold validation as an alternative for streaming settings, but this limitation is not prominently discussed despite being central to real-world applicability.

- **Hyperparameter sensitivity without thorough analysis:** The thresholds (α = 2000 for Pivot-adapt; t° = μ_q + 2σ_q for Pivot-concat) are set empirically. Appendix A.5 shows performance varies meaningfully with σ (GPQA: 53.0 to 62.1), yet no sensitivity analysis for α across tasks is provided. The method is described as "zero-shot" but requires per-task threshold tuning or a shared α that may not generalize.

- **Unfair comparison configuration in Table 3:** EXPLORA and LENS comparisons use 5 exemplars and different tasks (GSM8K, TabMWP, SQA*) versus the main experiments' 10 exemplars and tasks. Results are "extracted from the original paper" rather than re-run under controlled conditions, making efficiency claims harder to evaluate fairly.

## Nice-to-Haves

- **Sensitivity analysis for α across tasks:** If the optimal α varies substantially, the "zero-shot" claim is weakened; demonstrating a universal α or providing a principled selection mechanism would strengthen the method.

- **Edge sparsity ablation:** The top-100 edge cutoff per test example (Section 3.2) was not ablated. For large exemplar pools, this pruning may drop informative long-tail connections.

- **Analysis with weaker embedding models:** The method relies on Gecko embeddings. Testing with SimCSE or BM25 for graph construction would reveal robustness to representation quality.

- **Inductive variant for streaming deployment:** A practical extension would pre-compute static exemplar sets and thresholds offline, enabling single-example inference without batch availability.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **Criticism that "zero-shot" label is misleading:** The paper uses "zero-shot" to mean no LLM inference calls for scoring, which is reasonable terminology in the ICL literature. The transductive limitation is separate from the zero-shot claim.

- **Criticism about omitting HITS-SVD equivalence:** The paper correctly applies HITS for its intended purpose; not discussing the mathematical connection to SVD is an omission but not a flaw affecting validity.

- **Criticism that contributions are restated:** The contributions bullet points distinguish (1) modeling bilateral interactions, (2) designing Pivot-ICL methods, and (3) analysis showing performance source. These are distinct contributions.

- **Complaint about "Pivot-Aaapt" typo in appendix:** This is a minor proofreading error that doesn't affect the paper's substance.

- **Criticism about GPQA exemplar leakage:** GPQA Diamond (test) and exemplar pool are drawn from the same dataset, but this follows standard ICL evaluation protocols and doesn't constitute "leakage" in the problematic sense.

## Novel Insights

The bipartite graph framing reveals a fundamental insight: exemplar selection is not just about picking "good" exemplars, but about recognizing that different test examples have different needs based on their connectivity to the exemplar pool. This reframes ICL exemplar selection from a retrieval problem to a routing problem—deciding which selection *strategy* (dynamic vs. static) each test example requires. The ID/OOD analysis (Figure 2) provides compelling empirical grounding: dynamic exemplars excel when good matches exist, but harm performance when forcing "best available" matches that are actually poor. This insight—that matching quality matters more than match existence—has implications beyond ICL for retrieval-augmented systems broadly.

## Suggestions

1. **Report bootstrap confidence intervals** for all results, especially for AIME24 (n=30) and GPQA (n=198), to establish statistical significance.

2. **Add precision/recall analysis for OOD detection:** On the PDDL split, report how well hub scores classify examples as ID (3-7 blocks) vs. OOD (8-20 blocks). This directly validates the routing mechanism.

3. **Either re-run EXPLORA/LENS comparisons under identical settings** (same backbone, same tasks, same number of exemplars) or clearly state the comparison is illustrative rather than controlled.

4. **Clarify the transductive limitation** in the main paper (not just conclusion): acknowledge that real-time single-query deployment requires pre-computing static exemplar sets and thresholds offline.

# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 0.0]
Average score: 1.3
Binary outcome: Reject
