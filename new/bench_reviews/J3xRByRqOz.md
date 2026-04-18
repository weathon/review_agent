## Summary

The paper proposes Reliability-Aware RAG (RA-RAG), a multi-source RAG framework that estimates source reliability without ground truth labels using an iterative weighted majority voting (WMV) algorithm adapted from crowdsourcing, then uses the estimated reliabilities to selectively retrieve from reliable sources (κ-RRSS) and aggregate responses via WMV. The paper also introduces a synthetic benchmark for multi-source RAG with heterogeneous source reliability and demonstrates RA-RAG's effectiveness over simple baselines (Naive RAG, MV) on three QA datasets with three LLMs.

## Strengths

- **Principled formulation.** The paper provides a clean formalization of multi-source RAG as a WMV problem (Section 2.2, Eq. 1), drawing on established crowdsourcing theory (Li & Yu, 2014). This is a natural and well-motivated mapping from a known theoretical framework to the problem of heterogeneous source reliability in RAG.

- **Well-designed mechanism suite.** The two-stage approach (iterative reliability estimation → efficient inference) together with practical components (keyword-based prompts for response normalization, ROUGE-based misalignment filtration, κ-RRSS for scalability) addresses the stated challenges coherently. The ablation studies (Tables 1 and 2, Figures 5–7) demonstrate that each component contributes meaningfully.

- **Comprehensive ablation analysis.** The paper provides thorough ablations: effectiveness of misalignment filtration (Table 1, Table 2), impact of κ in κ-RRSS (Figure 5), cost comparison (Figure 6), and importance of relevance in source selection (Figure 7). The spammer–hammer experiments (Figure 4) provide useful intuition about robustness.

- **Honest limitations.** The paper acknowledges key limitations (keyword-based aggregation, short-form generation focus, inability to detect incorrect-but-grounded answers, need for existing queries for reliability estimation) in Section 6.

## Weaknesses

### Fatal

None.

### Major

- **All empirical evaluation is on a synthetic benchmark, limiting confidence in real-world applicability.** The benchmark (Section 4) constructs artificial "sources" by partitioning GPT-4o-mini-generated factual and misinformation documents, with source reliability $p_i$ sampled from designed priors (beta or spammer-hammer). Every source's errors are i.i.d. given $p_i$, there are no correlated errors, no adversarial content, and no topic-dependent reliability variation—all of which are core challenges in real information ecosystems. The paper claims this "reflects real-world scenarios with heterogeneous source reliability" (Abstract, Section 4), but the i.i.d. Bernoulli model of source correctness exactly matches the assumptions of the WMV estimator, making the favorable results somewhat circular. No evaluation on naturally occurring multi-source corpora (e.g., news from outlets with varying credibility, or Wikipedia + social media) is provided. Without such evaluation, the claim that RA-RAG is robust and effective in "realistic, heterogeneous, and adversarial information environments" is unsupported beyond the controlled simulation.

- **Missing comparisons with cited robust RAG baselines.** The paper explicitly positions itself as improving over several misinformation-aware RAG methods (Pan et al., 2023; Weller et al., 2024; Xiang et al., 2024; Deng et al., 2024; Pan et al., 2024) and discusses their limitations in the introduction, yet none of these methods appear as experimental baselines. The baselines used are Naive RAG, MV, WMV (with estimated weights), and Oracle WMV. WMV without RRSS is an ablation of the authors' own method, not an independent baseline. This leaves the paper without demonstrating that RA-RAG outperforms any existing robust RAG approach, only that it outperforms generic RAG and simple voting.

### Minor

- **No systematic evaluation of reliability estimation quality.** The paper evaluates final answer accuracy (EM) but never directly measures how well the estimated reliabilities $v_i$ correlate with true source correctness rates. The one qualitative example (Figure 3) shows good agreement, but no quantitative metric (e.g., rank correlation, MSE between $v_i$ and $p_i$) is reported. Since reliability estimation is the core contribution, this gap weakens the evidence that the iterative algorithm works well beyond the synthetic setting.

- **Restricted to closed-ended/short-form QA.** The entire framework—from keyword-based prompts to exact-match WMV aggregation—depends on short, unambiguous answers. The paper acknowledges this (Section 6) and discusses extensions to long-form generation via decomposition (Appendix J), but no evidence is provided that the approach generalizes. This significantly limits applicability.

- **Scalability concerns with limited empirical demonstration.** The paper motivates κ-RRSS for scalability, but experiments only use 3–9 sources with κ=4. Real-world multi-source settings could involve dozens or hundreds of sources. It is unclear how well the iterative reliability estimation converges with 50+ sources or with sparse per-source data. The scalability claim is asserted but not convincingly demonstrated.

- **No convergence analysis or hyperparameter sensitivity.** The iterative estimation algorithm (Section 3.3) has no convergence guarantees or empirical analysis of convergence behavior (number of iterations, sensitivity to initialization). The ROUGE-1 threshold of 0.9 and $\bar{w}=0.6$ are fixed without exploration. The scaling $v_i = N\hat{w}_i - 1$ can produce negative weights, which is intended to penalize unreliable sources but is not analyzed for potential failure modes (e.g., when most sources are unreliable).

### Trivial

- The qualitative example (Figure 3) shows results for a single data point; a few additional examples or a quantitative analysis of estimation accuracy would strengthen the presentation.

## Nice-to-Haves

- Evaluation on at least one real-world multi-source corpus (e.g., news aggregation data with known source credibility) to validate that the gains transfer beyond the synthetic benchmark.
- Direct comparison with at least one of the cited misinformation-robust RAG methods.
- Scatter plot of estimated $v_i$ vs. true $p_i$ to quantitatively validate reliability estimation.
- Experiments with larger numbers of sources (20–50) to substantiate scalability claims.

## Removed Points

- **"LLM-generated misinformation may not reflect real-world misinformation patterns."** This overlaps with the synthetic benchmark concern (already captured in Major weakness 1). More importantly, the paper explicitly creates a controlled experimental environment; criticizing the realism of simulated misinformation is part of the same broader issue already addressed. However, the specific point that GPT-4o-mini generates both the data and is then evaluated on it (potential contamination) is worth noting but not a standalone fatal flaw since different prompts and paraphrasing are used.

- **"Data contamination: GPT-4o-mini generates the benchmark and is also an evaluation model."** This is a valid concern but secondary to the larger issue that the benchmark is entirely synthetic. It does not independently invalidate the results on Llama3-8B and Phi3-mini.

- **"The definition of 'source' doesn't correspond to realistic RAG pipelines."** This is partially addressed by the paper's explicit formalization in Section 2.2 and the use case scenarios described. Source metadata is often available in practice (news outlets, API providers). This is a scope limitation rather than a fundamental flaw.

- **"No formal convergence analysis."** This is a nice-to-have for an empirical paper rather than a substantive weakness. The practical convergence is demonstrated in experiments.

- **"Negative weights in $v_i = N\hat{w}_i - 1$."** This is by design following Li & Yu (2014)—negative weights down-weight unreliable sources' answers. It is an intentional feature, not a bug, though the paper could discuss failure modes more clearly.

- **"Incremental novelty: the core algorithm is adapted from crowdsourcing."** The paper explicitly and transparently cites Li & Yu (2014). While the adaptation is straightforward, the application to multi-source RAG with the associated practical mechanisms (keyword prompts, filtration, κ-RRSS) constitutes a reasonable systems-level contribution.

- **"Request for confidence intervals/error bars."** The paper reports averages over 10 random trials. While error bars would improve presentation, single-run or few-run evaluation is standard in this community, and the consistent trends across models and datasets provide some robustness.

## Novel Insights

The mapping from crowdsourcing label aggregation (WMV) to multi-source RAG is natural but had not been explicitly formalized and operationalized in the RAG literature. The key insight that a separate per-source LLM call enables source-level reliability estimation—with subsequent selective retrieval guided by estimated reliability—is a clean architectural contribution. However, the evaluation reveals a tension: the method essentially assumes an i.i.d. Bernoulli model of source correctness, and the synthetic benchmark embeds exactly this model, making the evaluation favorable by construction. The real test of this approach will be whether the WMV-based reliability estimator holds under non-i.i.d., correlated, or adversarial source error patterns—which remain untested.

## Suggestions

1. **Add at least one experiment on a real-world multi-source corpus** (e.g., construct a dataset from news articles of known outlets with fact-checked claims, or use existing datasets with source metadata). Even one such experiment would substantially strengthen the claims.
2. **Include at least one contemporary robust RAG baseline** (e.g., CRAG, Self-RAG, or one of the methods cited in the introduction) to demonstrate that RA-RAG's improvements hold beyond simple baselines.
3. **Report rank correlation or MSE between estimated $v_i$ and true $p_i$** to quantitatively validate the reliability estimation component.
4. **Discuss failure modes explicitly**: What happens when most sources are unreliable? When sources have correlated errors? When the ROUGE filter fails?

## Score and Decision

**Calibration**: I compared against papers with similar profiles:
- *WinnowRAG* (scores 5,5,5,5, rejected): Multi-source/noisy RAG, limited novelty, missing baselines, compute concerns — similar profile.
- *AstuteRAG* (scores 5,6,5,5, rejected): Robust RAG method, limited baselines compared to, short-form focus — similar profile.
- *MDBench* (scores 5,3,3,5, rejected): Synthetic benchmark without real-world validation — shares the evaluation concern.
- *InstructRAG* (scores 8,8,5,6,8, accepted poster): RAG denoising method with comprehensive baselines and strong empirical results — clearly stronger.
- *CLA-RA* (scores 3,3,5,3, rejected): Crowdsourcing-style reliability estimation, small experiments — weaker version of similar idea.

This paper sits between CLA-RA (weaker) and AstuteRAG/WinnowRAG (comparable). It has a cleaner formulation than AstuteRAG but shares the missing-baselines and limited-applicability weaknesses. The synthetic-only evaluation is a significant gap relative to the claims made. Comparable papers scored 5.

MY FINAL SCORE: <pineapple>5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>