=== CALIBRATION EXAMPLE 9 ===

# Final Consolidated Review
## Summary

Pivot-ICL proposes modeling the bilateral interactions between candidate exemplars and test examples as a weighted bipartite graph, using the HITS algorithm to compute authority scores (for exemplars) and hub scores (for test examples). Based on these scores, the method adaptively assigns either dynamic (input-specific) or static (task-generic) exemplars to each test input. Experiments across four challenging reasoning tasks and multiple LLM backbones show consistent gains over purely dynamic or purely static selection baselines.

## Strengths

- **Novel graph-theoretic framing of the dynamic/static trade-off in ICL.** Reframing exemplar selection as a bipartite graph mining problem—where exemplars act as "authorities" and test examples as "hubs"—is a genuine conceptual contribution. This bilateral perspective captures mutual reinforcement between exemplars and test inputs that unidirectional similarity-based methods miss, and the observation that poorly-connected test inputs benefit from generic exemplars is both intuitive and empirically supported.

- **Compelling ID/OOD analysis on PDDL (Figure 2).** The controlled experiment leveraging the natural in-distribution (3–7 blocks) vs. out-of-distribution (8–20 blocks) split provides direct evidence that dynamic exemplars help ID cases while static exemplars help OOD cases. This analysis goes beyond reporting aggregate numbers and substantiates the core motivation for adaptive treatment.

- **Strong efficiency advantage over loss-based methods.** Pivot-ICL requires only embedding similarity computation and a lightweight graph scoring pass (under 10 minutes on CPU per Appendix A.6), compared to EXPLORA's 1000+ LLM calls and 2–5 hours. This is a meaningful practical advantage for the zero-shot setting the paper targets.

- **Consistent improvements across backbones (Table 2).** Gains are observed on Gemini 2.0 Flash, Llama 3.3 70B, and Qwen 2.5 7B, with particularly notable improvements on GPQA across all models (e.g., +4.6 on Qwen 2.5 7B). This suggests the benefit is not an artifact of a single model's behavior.

## Weaknesses

- **Transductive assumption limits practical applicability.** The bipartite graph requires the full test set Q to be known a priori (Section 3.2: G = (C ∪ Q, E, W)), and the Pivot-adapt threshold t∇ = α/(|C||Q|) explicitly depends on |Q|. In streaming or API-serving scenarios where queries arrive individually, the method cannot be directly applied. The paper acknowledges this in Section 5 ("sometimes, there is no observed full set of test examples") and proposes k-fold validation on exemplars as a workaround, but this alternative is not empirically evaluated, leaving the method's applicability outside batched settings unsubstantiated. This is a significant practical limitation for a method aimed at improving ICL deployment.

- **Statistical reliability on small test sets is not established.** AIME24 contains only 30 test problems. Pivot-adapt's reported 23.4% versus ~10–20% for baselines represents a difference of roughly 1–4 additional correct answers out of 30. No confidence intervals, standard deviations, or significance tests are reported for any task. Given the well-documented variance in ICL performance across exemplar orderings and selections, the reliability of gains on AIME24 is uncertain, and this concern extends to GPQA (n=198) and SQA (n=490) where variance in reasoning tasks can be substantial.

- **Comparison with loss-based methods (Table 3) uses different experimental conditions.** LENS and EXPLORA results are compared under a different setup: 5 exemplars (vs. 10 in main experiments), GPT-4o-mini backbone (vs. Gemini 1.5 Pro), and different task sets (GSM8K/TabMWP/SQA* vs. AIME24/PDDL/SQA/GPQA). The SQA variant is explicitly noted as easier (oracle facts provided). These confounds make the claim of "comparable performance" difficult to evaluate fairly.

- **Hyperparameter sensitivity is insufficiently analyzed.** The core adaptive decision in Pivot-adapt is governed by α = 2000 (set "empirically" per footnote 1), and the normalization t∇ = α/(|C||Q|) means α's effective decision boundary shifts with dataset size—making it dataset-dependent rather than truly zero-shot. The Pivot-concat threshold (μ + 2σ) is similarly heuristic. While Appendix A.5 provides limited ablations (Table 5 varies σ multiplier on GPQA only; Table 6 compares against mean threshold), no sensitivity analysis for α across tasks is provided. Given that the entire adaptive mechanism hinges on these thresholds, this gap weakens confidence in the method's robustness.

- **Justification for HITS over simpler graph scoring methods is limited.** Table 1 shows that Degree Centrality achieves the best SQA result (66.7%) and PageRank is competitive on several tasks. The mutual-reinforcement property of HITS is intuitively appealing, but the paper does not provide theoretical or empirical evidence that the iterative authority/hub computation is necessary rather than sufficient. The analogy between web hyperlink structure (directed, discrete) and ICL similarity (undirected, continuous cosine scores) is drawn without formal justification, and the undirected nature of the similarity edges removes the directionality that gives HITS its original interpretation.

## Nice-to-Haves

- **Analysis of hub score distributions and switch error rates.** A key implicit assumption is that hub scores are bimodal (separating ID from OOD examples), enabling a clean binary threshold. If scores are unimodal, the adaptive switch may be arbitrary. Plotting hub score distributions and quantifying how often Pivot-adapt makes the "wrong" choice (static where dynamic would be better, or vice versa) would substantially strengthen the analysis.

- **Correlation between hub scores and OOD-ness beyond PDDL.** Figure 2 validates the ID/OOD intuition only on PDDL, where OOD has a clean structural definition (block count). Verifying that low hub scores correlate with out-of-distribution difficulty on AIME24, SQA, and GPQA would generalize the core claim.

- **Ablation over embedding models.** The method depends entirely on Gecko embeddings for edge weights, yet no ablation tests alternative encoders. A comparison with a weaker encoder would clarify whether Pivot-ICL's gains are robust to embedding quality.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Weakness: Ambiguity of the "+8.8% relative gain" claim.** The abstract states this is a relative gain over the best baseline, and it is directly verifiable from Table 1: Pivot-adapt average (60.6%) vs. best baseline average MMR (55.7%) yields ~8.8% relative improvement. This is not ambiguous.

- **Weakness: Equation formatting issues (missing summation symbols).** This is a parser artifact, not a paper error. Removed per the formatting nitpick rule.

- **Weakness: Reproducibility concern about missing convergence criteria for HITS.** The paper states HITS runs iteratively (Section 3.2), and Appendix A.6 notes 10–50 iterations with O(k(|V|+|E|)) complexity. While more detail could help, this falls under trivial implementation details not expected in a submission.

- **Weakness: Computational cost of embedding the entire test set.** The paper provides overhead estimates in Appendix A.6 (under 10 minutes). Embedding cost scales linearly with |Q|, which is standard for any retrieval-based method. This is not a distinctive weakness of Pivot-ICL.

- **Weakness: Marginal gains on smaller models.** While Qwen 2.5 7B's SQA gain is modest (70.6→71.2), the GPQA gain is substantial (30.8→35.4). The gains are consistent in direction across all model-task pairs; characterizing them as "marginal" selectively focuses on the smallest improvement.

## Novel Insights

The bipartite graph framing reveals a subtle but important structural property: the value of an exemplar is not intrinsic but depends on the test distribution it serves, and conversely, the "difficulty" of a test example is relative to the available exemplar pool. This mutual dependence—formalized through HITS authority/hub scores—suggests that exemplar selection methods that treat these two sets independently (as all prior work does) are fundamentally misspecified. The PDDL analysis (Figure 2) provides a particularly clean demonstration: the crossover point where static exemplars begin to outperform dynamic ones could serve as a diagnostic for when exemplar pools are insufficient for a given test distribution, a concept that could inform future work on active exemplar acquisition.

## Suggestions

- Empirically evaluate the k-fold exemplar-only workaround for the transductive setting (Section 5) to demonstrate that Pivot-ICL can function without test-set access. Even a single-task experiment would address the most critical practical limitation.

- Replace the fixed α with an adaptive threshold derived from the hub score distribution itself (e.g., elbow detection or percentile-based splitting) to reduce dataset-specific tuning and strengthen the zero-shot claim.

- Report results with multiple random seeds or exemplar orderings on AIME24, or at minimum compute bootstrap confidence intervals, to establish that gains on this 30-example benchmark are not due to noise.

- Add a direct comparison with LENS or EXPLORA on the main experimental setup (AIME24/PDDL/SQA/GPQA with Gemini 1.5 Pro and 10 exemplars) to enable a fair head-to-head evaluation against loss-based methods.

# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 0.0]
Average score: 1.3
Binary outcome: Reject
