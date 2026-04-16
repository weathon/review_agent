## Summary
This paper studies offline RL in episodic Regular Decision Processes (RDPs), a structured non-Markovian setting where history dependence is captured by a finite automaton. Its main contribution is a new language-based metric for comparing suffix-trace distributions inside state-merging RDP learners, yielding PAC identification bounds that can replace the unfavorable \(L_\infty^p\)-distinguishability dependence of prior work with a more refined \(L_X\)-distinguishability; on some instances such as T-maze, this gives an exponential improvement in the relevant distinguishability term. A second contribution uses Count-Min-Sketch to reduce memory costs for storing empirical counts.

## Strengths
- **Clear and nontrivial theoretical insight on why prior distinguishability bounds are too pessimistic.** The paper does more than tweak constants: it identifies a real pathology of prior \(L_\infty\)-style testing and formalizes it in Theorem 1, showing a family of RDPs where \(L_\infty\)-distinguishability is \(\mathcal O(2^{-N})\) while the proposed language-based distinguishability remains \(\Omega(1)\). The T-maze example makes this concrete and compelling.
- **Original language-metric construction with a meaningful hierarchy.** The introduction of \(L_X\) and the hierarchy \(\mathcal X_{i,j}\), built from basic trace patterns and the \(C_k^\ell\) operator, is the paper’s most novel idea. It gives a principled interpolation between very local tests and richer pattern-based tests, rather than treating traces only as atomic suffixes.
- **Theoretical results are substantive and not merely qualitative.** Theorem 3 gives a PAC bound for ADACT-H using the language metric with dependence \(\tilde{\mathcal O}\!\left(\frac{C_{\mathbf R}^*\log(1/\delta)\log |\mathcal X|}{d_m^*\mu_0^2}\right)\), and the discussion correctly notes that when \(j\) is constant, \(\log |\mathcal X_{i,j}|\) can stay small even as horizon grows. This is a real improvement over the prior support-size-driven dependence when the language family is well chosen.
- **Good transparency about limitations of prior analysis and the authors’ own bounds.** The paper explicitly states that its new analysis uncovers an error in Cipollone et al. (2023) and that both the prior and corrected sample complexities incur an additional factor. That sort of correction is valuable to the community.
- **Experiments are aligned with the paper’s motivating example for computational scaling.** Figure 2 does support the claim that the language-metric variant scales much better than the \(L_\infty\)-style suffix-based implementation on T-maze-like domains, and Table 1 suggests the approach can produce better downstream policies than the compared alternatives on several benchmark tasks.

## Weaknesses

###: Fatal
- None.

### Major:
- **The paper is framed as an offline RL advance, but the main new theorems stated in the paper are model-identification guarantees, not end-to-end near-optimal policy guarantees.** Section 2.3 defines the offline RL objective as outputting an \(\varepsilon\)-optimal policy, but Theorems 2 and 3 state that ADACT-H “returns the minimal RDP.” Section 3 says these methods “can be incorporated” into RegORL, but the main text does not state a full theorem translating the new identification result into a final \(\varepsilon\)-optimal policy guarantee for the proposed variants. This does not invalidate the technical contribution, but it does make the title/abstract framing somewhat stronger than what is directly established in the visible main text.
- **The empirical section does not validate the headline sample-efficiency claim.** The abstract and introduction emphasize improved sample efficiency, and Theorem 3 is the core theoretical sample-complexity result. However, Section 5 reports automaton size, return, and runtime for a fixed data regime, and Figure 2 varies corridor length with fixed \(K=100\) episodes. There are no learning curves versus dataset size and no empirical study of how many samples are needed to reach near-optimal return. For a theory-forward paper this is not fatal, but it does mean the experimental evidence mainly supports computational scaling and policy quality on small tasks, not empirical sample efficiency.
- **The choice of language family \(\mathcal X_{i,j}\) is central yet underdeveloped algorithmically.** The theory requires Assumption 1: the chosen family must yield \(L_{\mathcal X_{i,j}}\)-distinguishability at least \(\mu_0>0\). But the paper gives no adaptive procedure, selection rule, or practical diagnostic for choosing \(i,j\) in an unknown environment. Since too small a family may fail to distinguish states and too large a family increases complexity, this is a genuine practical gap.
- **The overall sample complexity may still be poor because of the \(1/d_m^*\) term.** The discussion after Theorem 3 correctly acknowledges this: “The constant \(1/d_m^*\) depends exponentially on \(H\) if there exists an RDP state that is very hard to reach.” Thus, the paper resolves one important exponential dependence—through \(\mu_0\)—but does not eliminate all routes to exponential sample complexity. The exponential improvement claim is therefore conditional, not universal.

### Minor
- **The empirical baseline story is weaker than ideal for the paper’s stated claims.** The paper’s theoretical advance is relative to prior \(L_\infty^p\)-based RDP learning, yet experiments compare primarily against FlexFringe and the CMS variant. Figure 2 compares CMS vs. language metric, which is useful for showing computational scaling, but an explicit comparison to the original exact \(L_\infty^p\)-based ADACT-H/RegORL implementation would have better matched the paper’s main claim.
- **The CMS contribution is clearly less impactful than the language-metric contribution.** This is visible in both theory and experiments. Theorem 2 largely preserves the original distinguishability dependence, and the experiments show CMS remains slow because it still iterates over suffixes. CMS is a reasonable memory-saving add-on, but the paper currently presents it almost on equal footing with the more important language-metric result.
- **Some key formal definitions in the main text are harder to parse than they need to be.** In particular, the transition from the abstract operator \(C_k^\ell\) to the concrete hierarchy \(\mathcal X_{i,j}\) is dense, and the estimator paragraph near the end of Section 4.1 is imprecise: \(\hat p_1 := \sum_{e \in \mathcal Z_1}\mathbb I(e \in \mathcal X_{i,j})/|\mathcal Z_1|\) is not well-typed as written because \(\mathcal X_{i,j}\) is a set of languages, not a single event. The intended meaning is recoverable, but this is exactly the point where the statistical test should be most concrete.
- **Some claims should be phrased more carefully.** For example, the conclusion says the language-restricted approach “removes the dependency on \(L_\infty^p\)-distinguishability,” but more precisely it replaces that dependence with a different distinguishability notion, \(L_X\)-distinguishability. That is still a strong result, just not quite the same as removing distinguishability assumptions altogether.

### Trivial
- **A small ambiguity appears in Theorem 2’s definition of \(d_m^*\).** The theorem writes \(d_m^* = \min_{u,a,o} d_t^*(u,a,o)\), leaving \(t\) free in the displayed definition. This is likely shorthand or a presentation slip rather than a conceptual flaw, but it should be fixed in the final version.

## Nice-to-Haves
- Add dataset-size curves showing return or exact reconstruction quality as a function of number of episodes; this would directly connect the theory to empirical sample efficiency.
- Include an ablation over hierarchy choices \((i,j)\), especially since the paper uses \(\mathcal X_{3,1}\) throughout experiments.
- State explicitly in the main text the end-to-end offline RL guarantee obtained when plugging the new state-merging tests into RegORL, if this is indeed available in the appendix or follows straightforwardly from prior results.
- Provide a short computational-cost discussion for evaluating \(L_X\) tests in practice, not only their statistical benefit.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The experiments are too weak because they do not compare against the original exact ADACT-H, only against CMS/FlexFringe.”** This is directionally fair as a *minor* limitation, but it should not be overstated into an unfair-comparison criticism. The absence of the exact baseline is not evidence of cherry-picking, and the current comparisons still provide useful computational evidence.
- **“The paper’s experimental evidence effectively validates the theoretical sample-complexity claim.”** This positive reviewer claim is too strong. The experiments validate scaling/runtime/model-size behavior and some downstream return, but not sample complexity per se.
- **Any criticism implying the paper should also solve cyclic RDPs.** The paper explicitly scopes itself to episodic acyclic RDPs and even says extending to cycles is future work; criticizing it for not addressing that broader setting would be scope creep rather than a core flaw.
- **Pure formatting/parser issues.** The extracted PDF has artifacts, and minor notational glitches should not be inflated beyond the few places where they genuinely obstruct understanding.

## Novel Insights
The paper’s most interesting broader implication is that in non-Markovian offline RL, the right statistical object may not be the full suffix distribution nor individual trace probabilities, but a task-structured family of languages that captures semantically relevant trace events. This reframes representation learning in RDPs as choosing the right event algebra for state distinguishability. That perspective is more general than the specific hierarchy proposed here and may be the most lasting conceptual contribution: it suggests that sample efficiency can improve not by better generic density estimation over histories, but by designing structured tests that preserve only behaviorally relevant temporal regularities.

## Suggestions
- State an explicit theorem in the main text that connects minimal-RDP recovery with \(\varepsilon\)-optimal offline RL for the proposed variants, or soften the framing from “offline RL” to “RDP model learning for offline RL.”
- Add empirical curves versus dataset size \(K\), ideally on T-maze and one or two other domains, to substantiate the practical meaning of the sample-complexity improvement.
- Include an ablation over \(\mathcal X_{i,j}\) and discuss practical selection of \(i,j\), even if only heuristic.
- Reframe CMS as a secondary engineering contribution unless stronger practical evidence can be shown.
- Rewrite the estimator/test-statistic paragraph in Section 4.1 so that the empirical estimate of \(p(X)\) for each \(X\in\mathcal X\) is formally clear.
- Clarify when the remaining \(1/d_m^*\) factor is benign versus when it dominates, so readers do not overgeneralize the exponential-improvement claim.

## Score and Decision
**Assessment on the main axes:**  
- **Originality:** high. The language-metric formulation and hierarchy are genuinely novel.  
- **Importance:** good. Offline RL in structured non-Markovian settings is important, and the paper addresses a real bottleneck in prior RDP theory.  
- **Claims support:** mixed. The theory supports an improved identification sample bound under the new distinguishability notion, but the offline-RL framing and empirical “validated efficiency” language are somewhat stronger than what is directly demonstrated.  
- **Experimental soundness:** adequate but limited. The experiments support computational scaling and some downstream return quality, but not the central sample-efficiency claim.  
- **Clarity:** moderate. The main ideas are strong, but parts of the formalism are dense and one key estimator definition is not clearly written.  
- **Value to the community:** solid, especially for researchers at the intersection of RL, automata learning, and non-Markovian sequential decision-making.

**Calibration against human-reviewed anchors:**  
- Compared with **/home/wg25r/review_agent/human_reviews/B5kAfAC7hO.md** (scores 5,5,6; reject): this paper has a sharper and more compelling core theoretical insight, but similarly suffers from some presentation density and limited empirical support for broad claims. I view the current submission as somewhat stronger than that reject anchor.  
- Compared with **/home/wg25r/review_agent/human_reviews/GnOLWS4Llt.md** (scores 5,5,5; accept poster): both are theory-forward offline RL under partial observability/history dependence with limited empirical validation. The present paper has a crisper theoretical novelty than that anchor, but also a similar mismatch between broad framing and the exact experimental evidence.  
- Compared with **/home/wg25r/review_agent/human_reviews/jId5PXbBbX.md** (scores 6,5,6,8; accept poster): that paper earned acceptance due to strong theoretical novelty despite some concerns about computational interpretation and lack of experiments. This submission feels somewhat below that level because the offline-RL framing is more indirect and the empirical section does not fully support the central headline claim, but it is in the same general quality band.  
- Compared with **/home/wg25r/review_agent/human_reviews/U6Qulbv2qT.md** (scores 6,6,8,6,8; accept poster): that accepted theory paper also had concerns about terseness and computational aspects, but was judged to make a strong enough conceptual/theoretical contribution. The present paper is somewhat narrower and less fully rounded empirically, so I place it lower.  
- Compared with **/home/wg25r/review_agent/human_reviews/67t4ikhlvP.md** (mixed 5,8,1,1; reject): this paper is substantially more coherent and technically grounded, with a clearer central contribution and fewer foundational concerns.

Overall, this is **a good but not fully polished theory paper**: the main idea is strong and worthy of attention, but the framing should be tightened and the empirical validation better aligned with the headline sample-efficiency claim. I land slightly above the bar.

MY FINAL SCORE: <pineapple>6.5</pineapple>  
MY FINAL DECISION: <orange>Accept</orange>