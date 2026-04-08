=== CALIBRATION EXAMPLE 6 ===

# Harsh Critic Review
Now I have read enough of the paper to write a thorough review. Let me compose it.

---

## Section-by-Section Critical Review

### Title & Abstract

The title "Pivot-ICL: Adaptive Exemplar Selection for In-Context Learning" is accurate and descriptive. The abstract's central claim — "+8.8% relative gain over the best baseline" — is a macro-average across four tasks with vastly different scales and test-set sizes (including AIME24 with only 30 examples), and presents this aggregate figure without noting that it masks uneven gains and is heavily influenced by a task with a tiny test set. The claim that the method is "zero-shot" is accurate and worth emphasizing more clearly, since it distinguishes the approach from loss-based competitors. Overall, the abstract is concise and honest about the method's scope.

---

### Introduction & Motivation

The core motivation — that exemplar candidates may not uniformly cover the test distribution, so a universal dynamic or static strategy is suboptimal — is well-conceived and practically important. The observation that OOD test examples may be hurt by misleading dynamic exemplars (noisy nearest neighbors) is well-supported later by Figure 2.

However, the introduction conflates two distinct ideas: (1) using HITS to identify *better static exemplars* (authority scores for exemplars), and (2) using HITS to detect *which test examples are OOD* (hub scores for test examples). These are logically separable contributions but are bundled together as if HITS is essential for both. It is not obvious from the framing that the bipartite structure is needed for (1); exemplar-only scoring (e.g., using the exemplar subgraph alone) might suffice. This point is partially revealed in the ablation (Table 4, "Authority (exemplar)": 82.4 on SQA vs. "Authority (bipartite)": 81.0), but it is never addressed head-on in the introduction.

---

### Method (Section 3)

**Graph construction (Sec. 3.2):** The use of HITS on a bipartite graph is the paper's core technical ingredient. The authors should state a crucial mathematical fact that they omit: in a bipartite graph with no intra-set edges, HITS converges to the *leading singular value decomposition* of the edge-weight matrix **W** (i.e., hub score ∝ **W**·authority, authority score ∝ **Wᵀ**·hub). This means Pivot-ICL's authority and hub scores are essentially the principal left and right singular vectors of the similarity matrix. This has two implications:
  1. The "iterative" framing is misleading for the bipartite-only setting — the result is equivalent to a one-shot spectral computation.
  2. The "bidirectional" benefit claimed over simple mean-pooling is real but limited: the weighting of exemplars by their hub-weighted importance is different from a straight average, but whether this specific reweighting is the right inductive bias for ICL is not theoretically justified.

**Threshold design (Sec. 3.4):** Two separate thresholds are introduced — t◦ = μ + 2σ (for Pivot-concat) and t∇ = α/(|C||Q|) with α=2000 (for Pivot-adapt). The former is a heuristic with no ablation on the 2σ choice. The latter has α "empirically set to 2000 across tasks" and noted as something that "can be optimized with a development set." No development set is mentioned in the experimental setup, raising the question of whether α was tuned on the test set. This is a significant issue: the threshold determines which examples get dynamic vs. static exemplars, and without a held-out validation set, the risk of inadvertent test-set overfitting is real. The paper should clarify exactly how α was set, or provide an ablation (e.g., Table in Appendix A.5) showing robustness to its value.

**Requiring full test-set access:** Both variants require the full test set Q to be available at scoring time to build the bipartite graph. This is a significant deployment constraint that the paper acknowledges only briefly in Section 5. It rules out online, streaming, or single-example inference scenarios. The proposed k-fold workaround (using exemplars as proxies) is mentioned but never evaluated.

---

### Experiments & Results

**AIME24 test set size (Section 4.1):** AIME24 has exactly **30 test problems**. At temperature=1, the stochasticity of Gemini 1.5 Pro means that a single correct/incorrect response shifts accuracy by ~3.3%. In Table 1, Pivot-adapt reports 23.4% on AIME24 vs. the best dynamic baseline's 20.0% — a difference of exactly **1 correct answer**. Claiming this as evidence of the method's effectiveness is not statistically justified. No confidence intervals, no multiple-seed averaging, and no significance testing are reported anywhere in the paper. For AIME24 especially, this renders the reported improvements essentially uninterpretable.

**Statistical rigor across all tasks:** None of the reported results in Tables 1, 2, 3, or Figure 2 come with standard deviations or significance tests. Given temperature=1 generation and the small size of several test sets (AIME: 30, GPQA: 198), variance in model outputs can be substantial. At minimum, the authors should run each condition with multiple random seeds and report intervals.

**Comparison with EXPLORA (Table 3):** The paper compares Pivot-adapt against EXPLORA and LENS from the *original paper* (Purohit et al., 2024), rather than re-running those baselines. This is acceptable for a fair description of the comparison setting, but the paper should explicitly confirm that the same backbone (GPT-4o-mini) and prompt format were used. The result on TabMWP (Pivot-adapt: 94.0 vs. EXPLORA: 90.1) is surprisingly large — a ~4-point absolute improvement over a carefully tuned bandit-based algorithm using a simple graph heuristic. This warrants more scrutiny: is the SQA* task here the same "easier" variant (with oracle facts) as described in Footnote 3? If so, is a comparison with a different SQA variant than the main paper misleading?

**GRAPHIC (Fu et al., 2025):** The reference list includes GRAPHIC (Fu et al., 2025), which also uses a graph-based ICL retrieval model for multi-step reasoning. This is a closely related concurrent work that is not compared against or adequately discussed. Given that ICLR reviewers will note this, the authors should either compare or explicitly characterize the differences.

**Ablation study (Sec. 4.5 / Table 4):** The ablation is thoughtful and covers node construction, edge construction, and the iteration variant. However, a critical ablation is missing: **comparing hub scores against simpler OOD proxies** for the Pivot-adapt decision. For example, does using max(similarity to exemplars) or mean top-k similarity as the threshold work as well as hub scores? This would isolate whether HITS specifically is needed, or whether any reasonable OOD detector would yield similar results. As noted, the ablation result "Authority (exemplar): 82.4" outperforming "Authority (bipartite): 81.0" on SQA is evidence that the bipartite graph may not be adding much for *static* selection. The paper should directly acknowledge this and explain why the bipartite formulation is still preferable.

**Pivot-concat on PDDL:** Pivot-concat performs below Gecko on PDDL (60.3 vs. 61.3). The explanation — "an abundant set of exemplars is designed to be in the same distribution" — is circular: if distribution is well-covered, concatenating static exemplars consumes context slots that would otherwise be used for high-quality dynamic exemplars. This is a real failure mode of Pivot-concat that deserves more careful analysis and is a genuine limitation.

**Backbone ablation (Table 2):** The backbone ablation on SQA and GPQA is useful and shows consistent improvements. However, improvements on Gemini 2.0 Flash for SQA (82.2 for both dynamic and pivot) suggest Pivot-adapt provides no additional gain for already-capable models on some tasks, which is an interesting null result worth discussing.

---

### Writing & Clarity

Section 3.4's description of Pivot-concat is somewhat confusing: the paper says Pivot-concat "does not score test examples explicitly" yet applies an edge-weight threshold t◦. Clarifying that t◦ is a per-query local filtering step (not a global test-example score) would reduce confusion. The comparison between Pivot-concat and Pivot-adapt (at the end of Sec. 4.3) would also benefit from a clearer framing.

The paper also does not define "cover-EM" (the metric for SQA) in the main text, deferring to a citation. For self-containedness, a one-line description would help.

---

### Limitations & Broader Impact

The paper's conclusion acknowledges the test-set-access constraint and the opportunity for future self-evolving exemplar generation. However, several failure modes are not discussed:
- **Embedding model dependency:** The method relies on Gecko embeddings. If the embedding space is miscalibrated for a domain (e.g., highly technical math), hub scores may be unreliable. There is no analysis of what happens when the embedding model is weak.
- **Exemplar pool quality:** If the exemplar pool is uniformly bad, HITS will identify which exemplars are "most representative" of a bad pool, not which are inherently useful. The method presupposes that good exemplars exist.
- **Token budget:** With Pivot-concat, the concatenation of dynamic + static exemplars may exceed the context window or be impractical with 10 exemplars from each set in long-context tasks. Token counts are not reported.

---

### Overall Assessment

Pivot-ICL presents a clean and practically motivated idea: model the bilateral relationship between exemplars and test examples as a bipartite graph, then use HITS-based scoring to both identify strong static exemplars and detect OOD test examples that should receive static rather than dynamic treatment. The core intuition is validated convincingly by Figure 2's controlled PDDL experiment, and the method is genuinely zero-shot (no LM calls for scoring), which is an important practical advantage over LENS and EXPLORA. The multi-backbone evaluation and the breadth of tasks add credibility.

However, the paper has notable weaknesses that limit confidence in the reported gains. Most critically: the AIME24 test set has only 30 examples, making any reported difference there statistically meaningless, and **no result in the paper includes confidence intervals or significance tests despite stochastic generation at temperature=1**. The ablation in Table 4 reveals that using only exemplar-to-exemplar edges (without test examples) matches or beats the bipartite graph for static selection, which undermines the central claim that bidirectional test-exemplar interaction is necessary. The threshold hyperparameter α=2000 is empirically set with no validation procedure described, and the requirement for full test-set access at scoring time is a serious deployment constraint that is underemphasized. For ICLR, where the bar is high for both empirical rigor and methodological novelty, the paper as currently written would need (1) proper statistical analysis across all results, (2) a cleaner experimental separation of which gains come from the HITS bipartite structure versus simpler alternatives, (3) clarification of the α hyperparameter setting protocol, and (4) an engagement with the concurrent GRAPHIC paper. The contribution is real but currently oversold relative to the evidence provided.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces Pivot-ICL, a zero-shot adaptive exemplar selection framework for in-context learning that models query-exemplar interactions as a weighted bipartite graph. By applying mutual-reinforcement scoring algorithms (e.g., HITS), it computes "hub" scores for test queries and "authority" scores for candidates, automatically routing in-distribution queries to dynamic exemplars and out-of-distribution queries to a static, representative exemplar set. Extensive experiments across math, planning, commonsense, and scientific reasoning tasks demonstrate that this adaptive strategy consistently outperforms purely dynamic or static selection methods while maintaining low computational overhead.

### Strengths
1. **Well-grounded motivation with clear empirical validation**: The paper correctly identifies that dynamic exemplar selection degrades when test queries lack close matches in the candidate pool. This intuition is rigorously validated via a controlled ID/OOD experiment on PDDL (Figure 2, Section 4.4), which explicitly shows dynamic methods succeed on in-distribution queries while static methods generalize better to out-of-distribution ones.
2. **Efficient, zero-shot formulation**: The use of classic graph mining algorithms on a similarity-weighted bipartite graph avoids the expensive LLM calls or loss computations required by prior methods (e.g., LENS, EXPLORA). Appendix A.6 convincingly shows that graph scoring adds minimal overhead (<10 mins CPU) compared to hours of LLM inference, making it highly practical for deployment.
3. **Robust cross-task and cross-model generalization**: The method is evaluated on four diverse, complex reasoning benchmarks and demonstrates consistent performance gains across multiple backbone LLMs (Gemini, Llama 3.3, Qwen 2.5), confirming that the adaptive mechanism is not model- or task-specific (Tables 1 & 2).

### Weaknesses
1. **Indirect baseline comparisons undermine empirical rigor**: Table 3 claims Pivot-ICL is "comparable with loss-based methods," but results for LENS and EXPLORA are extracted from their respective papers rather than re-run under the same experimental conditions (Section 4.3). Without a controlled, head-to-head comparison on identical datasets, seeds, and LLM versions, the efficiency vs. performance trade-off remains partially unsubstantiated.
2. **Heuristic thresholding lacks theoretical or unsupervised grounding**: Pivot-adapt relies on a fixed hyperparameter $t_\nabla = \alpha / (|C||Q|)$ with $\alpha=2000$ empirically set across tasks. While k-fold CV is mentioned, this introduces dependence on a development set and offers no guarantee of robustness when query distributions shift, limiting the "zero-shot" claim in open-world deployments.
3. **Limited analysis of edge-weight robustness and noise propagation**: The bipartite graph relies exclusively on cosine similarity of Gecko embeddings. The paper acknowledges this but does not analyze how semantic similarity noise (e.g., lexically similar but logically distinct reasoning steps) affects authority/hub score convergence or leads to negative transfer in complex tasks like math or planning.
4. **Missing statistical significance testing**: Performance improvements are reported as point estimates without variance measures, multiple random seeds, or statistical tests (e.g., bootstrap, paired t-test). Given the ~1-2 point absolute gains on some tasks, it is difficult to assess whether differences are statistically significant or within LLM inference variance.

### Novelty & Significance
**Novelty**: Moderate. The conceptual framing of ICL exemplar routing as a bipartite graph problem is fresh, and applying mutual-reinforcement scoring to query-exemplar relations is a clever adaptation. However, the method does not modify the underlying algorithms; it directly applies established graph mining techniques (HITS, PageRank) to a pre-computed similarity matrix. The contribution lies more in system design and empirical validation than algorithmic innovation.
**Clarity**: High. The methodology is logically structured, well-illustrated (Figure 1), and clearly bridges graph theory concepts with ICL intuition. Mathematical formulations and pipeline steps are transparent.
**Reproducibility**: Good. Dataset splits, LLM configurations, embedding models, graph construction parameters, and baseline implementations are explicitly detailed. The authors also promise public code. Threshold values and hyperparameters are documented, though the fixed $\alpha$ limits plug-and-play reproducibility on unseen tasks.
**Significance**: High practical impact. Adaptive, zero-shot exemplar selection addresses a real bottleneck in deploying ICL for open-domain, complex reasoning. The ID/OOD analysis and low computational overhead provide actionable insights for researchers and practitioners aiming to scale ICL without costly fine-tuning or dev-set tuning.

### Suggestions for Improvement
1. **Conduct direct, controlled comparisons with loss-based baselines**: Re-run LENS, EXPLORA, and ConE on the same four tasks/models/evaluation settings used in Table 1, using identical seeds and inference parameters. This will substantiate claims of comparable accuracy with superior efficiency.
2. **Replace fixed thresholds with an unsupervised, data-driven mechanism**: Explore adaptive thresholding techniques such as percentile-based routing (e.g., top/bottom 30% of hub scores), Gaussian Mixture Modeling on the hub score distribution, or knee-point/elbow detection. This would strengthen the zero-shot claim and reduce reliance on dev-set tuning.
3. **Add failure-case analysis and embedding sensitivity studies**: Systematically evaluate how varying the quality of the initial similarity metric (e.g., swapping Gecko for BGE, OpenAI embeddings, or instruction-tuned retrievers) impacts final ICL accuracy. Include qualitative examples where the graph routing succeeds vs. fails to clarify the method's boundaries.
4. **Include statistical validation and scalability analysis**: Report results across multiple random seeds with standard deviations and perform statistical significance testing. Additionally, discuss computational/memory scaling when $|Q|$ reaches tens of thousands, and propose approximate strategies (e.g., nearest-neighbor subgraph construction or block-wise graph processing) to ensure the method remains viable for large-scale deployment.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Inductive/Online Setting Evaluation:** The method requires access to the entire test set ($Q$) to construct graph edges, creating a transductive setting incompatible with standard online ICL. Evaluate performance when processing $Q$ in streams or batches to verify viability beyond offline batch processing.
2. **Threshold Sensitivity Analysis:** The adaptive switch relies on a fixed hyperparameter $\alpha=2000$ without per-task tuning. Report performance variance when $\alpha$ is perturbed (e.g., $\pm 20\%$) to validate the robustness of the "zero-shot" claim.
3. **Graph Complexity Ablation:** Compare HITS against a baseline using simple average similarity scores instead of iterative graph mining. If gains are negligible, the added algorithmic complexity of the bipartite graph is unjustified.

### Deeper Analysis Needed (top 3-5 only)
1. **Hub Score vs. Accuracy Correlation:** Plot hub scores against individual example accuracy to verify the core mechanism. Without this, the claim that low scores identify "difficult" or "OOD" examples remains unsubstantiated anecdotal evidence.
2. **Switching Error Analysis:** Quantify how often Pivot-adapt incorrectly selects static exemplars for examples that would have benefited from dynamic ones (and vice versa). This exposes the reliability of the graph signaling mechanism.
3. **Bipartite vs. Exemplar-Only Value:** Table 4 shows Exemplar-only graphs perform similarly or better than Bipartite on GPQA/SQA. Analyze specifically what bilateral interaction adds, as current results suggest it may introduce noise rather than signal.

### Visualizations & Case Studies
1. **Score Distribution Histogram:** Show overlapping distributions of hub scores for correctly vs. incorrectly answered examples. This reveals if the scoring mechanism actually provides separable signal for decision making.
2. **Decision Boundary Case Studies:** Visualize specific test examples near the threshold $t_\nabla$ to show whether the switch decision aligns with semantic similarity or actual task difficulty. This exposes whether the graph captures meaningful structure or artifacts.

### Obvious Next Steps
1. **Incremental Graph Updates:** Develop a mechanism to update node scores incrementally as test examples arrive, removing the requirement for full test set access which limits deployment.
2. **Edge Weight Alternatives:** Experiment with uncertainty-based or loss-based edge weights instead of cosine similarity to determine if stronger signals improve the bipartite scoring reliability.
3. **Oracle Upper Bound:** Include an "Oracle" switcher baseline that knows the optimal exemplar type per example. This establishes the theoretical headroom and shows how much signal the graph is actually capturing.

# Final Consolidated Review
## Summary

Pivot-ICL proposes an adaptive exemplar selection framework for in-context learning that models query-exemplar interactions as a weighted bipartite graph. By applying HITS-style mutual reinforcement scoring, the method computes "authority" scores for candidate exemplars and "hub" scores for test queries, enabling automatic routing: well-connected (in-distribution) queries receive dynamic, similarity-based exemplars, while poorly connected (out-of-distribution) queries receive static, representative exemplars. Experiments across planning, math reasoning, commonsense QA, and scientific QA tasks show consistent improvements over purely dynamic or static selection baselines.

## Strengths

- **Clear motivation with controlled validation:** The core intuition—that dynamic exemplar selection fails when test queries lack close matches in the candidate pool—is rigorously validated through the PDDL experiment (Figure 2, Section 4.4). The controlled ID/OOD split (3-7 blocks vs. 8-20 blocks) directly demonstrates that dynamic methods excel on in-distribution examples while static methods generalize better to out-of-distribution ones, providing strong empirical grounding for the adaptive approach.

- **Efficient zero-shot formulation:** The method requires no LLM calls for scoring, using only embedding-based similarity and graph algorithms. Appendix A.6 confirms the computational overhead is minimal (~10 minutes for graph construction) compared to loss-based methods like EXPLORA that require 1000+ LLM calls. This practical advantage is significant for deployment scenarios where inference costs matter.

- **Consistent multi-task and multi-backbone gains:** Table 1 shows improvements across all four tasks (PDDL, AIME24, SQA, GPQA), and Table 2 demonstrates these gains generalize to Gemini 2.0 Flash, Llama 3.3 70B, and Qwen 2.5 7B. The method is not tuned for a specific model architecture, supporting its general applicability.

## Weaknesses

- **AIME24 test set is too small for meaningful claims:** AIME24 has exactly 30 test problems (Section 4.1). At temperature=1 generation, a single correct/incorrect answer shifts accuracy by ~3.3 percentage points. The reported improvement (23.4% vs. 20.0% for best baseline) represents a difference of approximately one correct answer. Without significance testing or multiple runs, this result is statistically uninterpretable and should not be treated as evidence of method effectiveness.

- **No statistical validation across experiments:** None of the results in Tables 1, 2, 3, or Figure 2 include standard deviations, confidence intervals, or significance tests. This is particularly problematic for GPQA (198 examples) and AIME24 (30 examples) where variance can substantially affect conclusions. The claimed "+8.8% relative gain" is a macro-average that masks this uncertainty.

- **Threshold hyperparameter methodology unclear:** Pivot-adapt uses $t_\nabla = \alpha/(|C||Q|)$ with $\alpha=2000$ "empirically set across tasks." The paper states this "can be optimized with a development set" but no development set is described in the experimental setup. Appendix A.5 shows some threshold ablation but not for the critical $\alpha$ parameter. This raises concerns about whether $\alpha$ was implicitly tuned on test performance.

- **Bipartite structure may not add value for static selection:** Table 4 shows that "Authority (exemplar)" (using only exemplar-to-exemplar edges derived from test-example proxies) achieves 82.4 on SQA, outperforming "Authority (bipartite)" at 81.0. This suggests the core technical contribution—the bipartite graph structure—may not be necessary for static exemplar selection. The paper should directly address why the bipartite formulation is preferable despite this ablation result.

- **Full test-set access requirement limits deployment:** Both Pivot variants require access to the entire test set Q to construct the bipartite graph, making the method incompatible with online, streaming, or single-example inference scenarios. While Section 5 briefly mentions k-fold validation as a workaround, this alternative is not evaluated, leaving a significant practical limitation unaddressed.

- **Pivot-concat fails when distribution is well-covered:** On PDDL, Pivot-concat (60.3) underperforms compared to pure Gecko (61.3). The explanation offered—that the exemplar set is designed to match the test distribution—identifies a real failure mode: concatenating static exemplars consumes context budget that would otherwise go to higher-quality dynamic matches. This limitation deserves explicit acknowledgment in the method description.

## Nice-to-Haves

- **Hub score calibration analysis:** Plotting hub scores against per-example accuracy would validate whether low hub scores genuinely identify "difficult" or "OOD" examples. Currently this mechanism is asserted but not empirically verified at the example level.

- **Embedding model robustness:** The method relies exclusively on Gecko embeddings. An ablation with alternative embedding models (e.g., BGE, OpenAI embeddings) would clarify whether the gains are robust to the choice of similarity metric or artifact-specific to Gecko.

## Removed Points

These points are flagged to be removed; treat them with caution.

- **"Missing GRAPHIC comparison"**: The paper cites GRAPHIC (Fu et al., 2025) but does not compare against it. However, GRAPHIC focuses on multi-step reasoning retrieval, which differs from this paper's scope of adaptive static/dynamic routing. Comparing against every graph-based retrieval method is scope creep; the current baselines (BM25, SimCSE, Gecko, MMR for dynamic; Random, Auto-CoT, Degree, Authority, PageRank for static) reasonably cover the exemplar selection space.

- **"Theoretical justification for HITS weighting"**: Demanding theoretical proof that HITS-style reweighting is the "correct" inductive bias for ICL is unreasonable for an empirical methods paper. The controlled PDDL experiment provides sufficient empirical justification.

- **"Inductive/online setting missing"**: This is already covered by the full test-set access weakness. Demanding additional experiments for streaming scenarios when the method's limitation is clearly stated is excessive.

- **"Generic claim that the paper is well-written"**: This generic positive does not identify anything specific this paper does better than others and is removed per guidelines.

- **"Significance testing for all large-scale benchmarks"**: While significance testing is warranted for small test sets (AIME24, GPQA), demanding it for large benchmarks where single-run evaluation is standard practice is excessive. The paper follows community norms for large-scale evaluation.

## Novel Insights

The ID/OOD analysis in Figure 2 provides a valuable methodological template for future ICL exemplar selection research: rather than reporting aggregate performance, explicitly partitioning test examples by distribution overlap with the candidate pool reveals when and why adaptive methods help. This diagnostic approach should become standard practice. Additionally, the observation that exemplar-only graphs perform comparably to bipartite graphs for static selection suggests that the core contribution may be the OOD detection mechanism (hub scores) rather than the bilateral scoring for exemplars—a distinction worth disentangling in follow-up work.

## Suggestions

1. **Report multi-seed results for small test sets:** For AIME24 (30 examples) and GPQA (198 examples), run each condition with at least 3-5 random seeds and report mean ± standard deviation. This is essential for claims to be statistically credible.

2. **Clarify the threshold setting protocol:** Explicitly state how $\alpha=2000$ was determined. If it was tuned on test performance, acknowledge this as a limitation. Better yet, provide a held-out validation procedure (e.g., using a subset of exemplar candidates as proxies) in a revision.

3. **Address the exemplar-only ablation result:** Explain why Table 4's "Authority (exemplar)" outperforms "Authority (bipartite)" on SQA, and clarify whether the bipartite structure's primary value is for hub-score-based query routing rather than authority-score-based static selection.

4. **Include failure mode discussion for Pivot-concat:** Explicitly note that Pivot-concat can underperform when the exemplar pool already adequately covers the test distribution, as shown on PDDL, and recommend Pivot-adapt for uncertain deployment scenarios.

# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 0.0]
Average score: 1.3
Binary outcome: Reject
