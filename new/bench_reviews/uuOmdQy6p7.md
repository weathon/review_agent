## Summary
This paper proposes CEMA, a black-box adversarial attack framework for multi-model multi-task learning systems. The method uses clustering on input-output embeddings from 100 auxiliary queries to generate pseudo-labels, trains substitute models on these cluster labels, and selects adversarial examples that successfully attack the most substitute models. The paper claims state-of-the-art attack success rates (>60% ASR for classification, BLEU <0.36 for translation) against both open-source models and commercial APIs (Baidu, Ali Translate) with only 100 total queries.

## Strengths
- **Practical threat model addressing an underexplored setting**: The paper targets black-box multi-model multi-task scenarios (separate APIs for classification and translation) rather than the more commonly studied white-box shared-parameter MTL. This is a realistic deployment scenario that prior work largely ignores, as noted in Section 1 (lines 21-22) where the authors correctly identify that "most existing adversarial methods, designed to attack shared parameter models, are ineffective against these systems."

- **Demonstrated effectiveness against commercial APIs**: Table 2 shows CEMA achieving BLEU scores of 0.15-0.35 against Baidu Translate and Ali Translate with only 100 queries, outperforming baselines like Morphin and TransFool. This empirical validation against closed-source systems adds practical relevance beyond synthetic evaluations.

- **Unified framework for heterogeneous tasks**: The method successfully handles both classification (ASR metric) and translation (BLEU metric) within a single attack pipeline without requiring task-specific modifications, as shown in Table 1 where CEMA attacks dis-sst5, dis-emotion, and opus-mt simultaneously.

## Weaknesses

### Fatal
// None - the paper demonstrates real empirical effects, even if the explanation is incomplete.

### Major

- **Unverified core mechanism: cluster-task alignment is assumed, not demonstrated**: The paper's central claim rests on the assertion that "if an adversarial attack on the substitute model successfully changes the cluster label of text x_i from 0 to 1, the label y_i^A shifts accordingly" (lines 91-92). However, cluster IDs are arbitrary artifacts of partitioning 100 samples into two groups via spectral clustering (Section 4.2, line 226: "we set the number of clusters to 2...ensuring cluster sizes are as close to 50 as possible"). There is no empirical evidence or theoretical justification that the hyperplane separating Cluster 0 and Cluster 1 approximates the victim model's actual decision boundary (e.g., Positive vs. Negative sentiment). Without this alignment, adversarial examples crafted to flip cluster labels should not reliably flip task labels. The paper reports >60% ASR but provides no visualization (e.g., t-SNE plot colored by cluster vs. true label) or metric (e.g., cluster purity with respect to task labels) to verify this critical assumption. This gap undermines the methodological contribution—the results may stem from generic perturbation strength rather than the proposed cluster-discriminability mechanism.

- **Misleading query efficiency metric prevents fair comparison**: Table 1 reports CEMA's queries as 0.045 and 0.05, calculated by amortizing the 100-query setup cost over the entire test set (2,210 and 2,000 samples respectively, per line 280). In contrast, baselines (BAE, TextBugger) report per-sample query costs (e.g., 21.43 queries). This compares a one-time fixed cost (CEMA) against a marginal per-instance cost (baselines). If an attacker wishes to attack a single sample, CEMA costs 100 queries while BAE costs ~21. By reporting amortized costs, the paper artificially inflates CEMA's efficiency by approximately 400× in the tables. This misrepresentation undermines the "few-shot" superiority claim in practical, sample-specific attack scenarios and makes direct comparison with standard baselines impossible.

### Minor

- **Suspicious transferability from low-capacity substitute lacks analysis**: The substitute model is trained on only 100 samples with pseudo-labels from a 2-cluster split (Section 4.2). Standard transfer attack literature indicates that substitute models require significant capacity and data to approximate victim decision boundaries. Achieving >60% ASR against diverse models (BERT, T5, Commercial APIs) via transfer from such a weak substitute is counter-intuitive. The paper provides no analysis of whether the substitute is overfitting the 100 auxiliary samples, no ablation on cluster number beyond showing 2 works best (Section 5.3), and no discussion of why this low-capacity model transfers effectively. This discrepancy raises concerns about potential evaluation issues that are not addressed.

- **Algorithm description contains imprecise terminology**: Algorithm 1, Line 6 states "refine the internal parameters of the clustering model f_c." Spectral clustering is typically non-parametric (eigenvalue-based) and does not have trainable internal parameters that are "refined" during clustering. This phrasing suggests either a misunderstanding of the algorithm or an unspecified neural clustering component that is not described elsewhere in the paper.

### Trivial
// None identified beyond the Major/Minor issues above.

## Nice-to-Haves
- Provide a visualization (t-SNE or similar) showing auxiliary data colored by cluster ID vs. true task label to verify cluster-task boundary alignment.
- Report per-sample query cost including proportional setup cost for fair baseline comparison.
- Add semantic similarity metrics (e.g., BERTScore) alongside BLEU for translation tasks to verify low BLEU corresponds to semantic divergence.
- Include ablation on the number of auxiliary samples to show how performance scales with query budget.

## Removed Points
These points are flagged to be removed, treat them with caution:

- **Harsh Critic: "Missing appendix, missing proofs in appendix"** - REMOVE per hard rules. The parser strips appendix sections from all papers; they exist in the original submission.

- **Harsh Critic: "Baseline methods are not optimized for the multi-task constraint"** - WEAKEN. This asymmetry actually favors the baselines (they're single-task methods applied to multi-task victims), which per hard rules should not be counted as a weakness against the author's method.

- **Strength Finder: "Robustness to data distribution shifts"** - This is partially supported by Table 6 (zero-shot attack), but the claim that CEMA "maintains effectiveness even when auxiliary data does not match" is overstated—Table 6 shows ASR drops from 73.57% to 64.00% when using Emotion data to attack SST5. The strength is kept but tempered.

- **Strength Finder: "Low sensitivity to specific implementation choices"** - Table 4 and 5 do show <3% variance across clustering/vectorization methods, which is concrete evidence. This strength is retained.

- **Harsh Critic: "Typos, spelling, grammar"** - REMOVE per hard rules. These are parser artifacts, not author errors.

- **Harsh Critic: "Requesting confidence intervals for large-scale benchmarks"** - MOVE TO NICE-TO-HAVE. Single-run evaluation is common in adversarial attack papers; demanding confidence intervals is not standard practice in this subfield.

## Novel Insights
The paper's core innovation—converting multi-task black-box attacks into single-task classification via clustering on input-output embeddings—is conceptually interesting but lacks mechanistic validation. The most novel observation is that ensemble selection across multiple substitute models (Section 4.5) improves transferability, with Table 3 showing ASR increasing from ~50% (1 candidate) to ~73% (3 candidates). However, this ensemble benefit is well-established in transfer attack literature and does not compensate for the unverified cluster-task alignment assumption. The zero-shot transfer results (Table 6) suggesting attackers need only "related" rather than identical auxiliary data is practically significant but under-analyzed.

## Suggestions
1. **Validate the cluster-task alignment assumption**: Provide empirical evidence (visualization, purity metric, or correlation analysis) showing that the 2-cluster boundary approximates the victim's task decision boundary. Without this, the method's core mechanism remains unverified.

2. **Correct the query efficiency reporting**: Report both the one-time setup cost (100 queries) and the per-sample marginal cost separately. Add a metric showing query cost for attacking a single sample (including proportional setup cost) to enable fair comparison with iterative baselines.

3. **Analyze substitute model capacity**: Investigate whether the substitute model overfits the 100 auxiliary samples and provide analysis of why transfer from such a low-capacity model succeeds. Consider ablation on auxiliary sample size to show performance scaling.

4. **Clarify Algorithm 1**: Fix the imprecise terminology about "refining internal parameters" of spectral clustering, or specify if a different clustering method with trainable parameters is used.

## Calibration and Score

I retrieved the following calibration anchors:

| Paper Path | Avg Score | Comparison to Current Paper |
|------------|-----------|----------------------------|
| /home/wg25r/review_agent/human_reviews_2026/OkjB6PWJEA.md | 3.00 | Similar efficiency claims with misleading query accounting; rejected for incomplete evaluation and unfair baselines. |
| /home/wg25r/review_agent/human_reviews_2026/PL4aPRtr3R.md | 3.50 | Black-box attack with overstated claims and insufficient theoretical justification; rejected. |
| /home/wg25r/review_agent/human_reviews_2026/OsXr7S8X4x.md | 3.00 | Adversarial attack with misleading efficiency comparisons and missing baselines; withdrawn. |
| /home/wg25r/review_agent/human_reviews_2026/M72B8jb7cA.md | 4.00 | Jailbreak attack with weak black-box experiments and motivation-experiment disconnect; rejected. |
| /home/wg25r/review_agent/human_reviews_2026/mTsWEVhcZM.md | 5.00 | Black-box privacy attack on MTL with clear threat model but missing quantitative metrics; accepted as poster. |
| /home/wg25r/review_agent/human_reviews_2026/ibXhUapwcz.md | 4.80 | Black-box generative attack with incremental contribution but comprehensive experiments; accepted as poster. |
| /home/wg25r/review_agent/human_reviews_2026/UQK3tUsouK.md | 6.50 | Transferability study with strong empirical validation and clear mechanism; accepted as poster. |
| /home/wg25r/review_agent/human_reviews_2026/T5hD0as3jb.md | 6.00 | Jailbreak attack with comprehensive experiments including commercial APIs; accepted as poster. |

**Score Reasoning**: The paper under review shares critical weaknesses with the 3.0-4.0 range anchors (OkjB6PWJEA, PL4aPRtr3R, OsXr7S8X4x, M72B8jb7cA): misleading efficiency metrics, unverified core mechanisms, and insufficient analysis of why the method works. However, it differs from these papers in two ways: (1) it demonstrates real empirical effects against commercial APIs (similar to T5hD0as3jb at 6.0), and (2) the threat model is genuinely underexplored (similar to mTsWEVhcZM at 5.0). The fatal flaw in OkjB6PWJEA and OsXr7S8X4x was misleading efficiency claims—this paper has the same issue with amortized query reporting. The mTsWEVhcZM paper (5.0) had a similar black-box MTL setting but was accepted because its claims were more modest and better supported. This paper makes stronger claims with weaker support.

Positioned relative to anchors: Below mTsWEVhcZM (5.0) due to unverified core mechanism; above OkjB6PWJEA (3.0) due to demonstrated commercial API effectiveness. The cluster-task alignment gap is a more severe methodological flaw than the weaknesses in mTsWEVhcZM, but the empirical results are stronger than OkjB6PWJEA.

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>