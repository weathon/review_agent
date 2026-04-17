---
job_id: df681504-c28a-4ab1-971a-be0d112d2213
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: DZUehXNiBn.pdf
paper: Efficient Causal Structure Learning via Modular Subgraph Integration
main_score_norm: 0.4
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The work is on scalable causal structure learning / DAG discovery, well within ICLR’s scope (causal reasoning, graph learning, optimization).

## Minimum Quality
Pass ✅.  
The paper is in English and contains all core sections (Abstract, Introduction, Preliminaries / Related Work, Methodology, Experiments, Conclusion). The methods and theorems are non‑trivial, the experimental section is substantial, and there are no obvious fatal statistical or experimental violations, although there are conceptual and theoretical weaknesses discussed below.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I did not detect any attempts to manipulate automated reviewing (no hidden prompts or suspicious instructions).

---

# Expected Review Outcome:

## Summary

The paper proposes VISTA, a modular divide‑and‑conquer framework for causal DAG learning. For each node, a Markov Blanket (MB) is estimated, a base learner is run on the induced subgraph, and the resulting local DAGs are merged via a weighted voting scheme followed by a Feedback Arc Set (FAS) heuristic to enforce acyclicity. The authors provide finite‑sample error bounds and an asymptotic consistency result under assumptions on local vote probabilities, and show experimentally that wrapping existing solvers (NOTEARS, GOLEM, DAG‑GNN, GraN‑DAG, SCORE, CAM) with VISTA can often improve accuracy and reduce runtime on synthetic graphs and on the Sachs protein network.

## Strengths

1. **Clear and modular algorithmic idea.**  
   The core algorithm (Algorithm 1 on Page 14 and Figure 3 on Page 5) is conceptually simple: MB‑based node‑centered decomposition, local learning, then a matrix‑level weighted vote plus a greedy FAS pass. This is an appealing wrapper because it does not depend on the internal form of the base learner and can exploit existing solvers “as is”.

2. **Model‑agnostic integration and plug‑and‑play nature.**  
   The framework truly only requires that the base learner output directed edges on a given node subset. The aggregation step uses only counts of directed edges, so the same pipeline works with differentiable DAG learners (NOTEARS, DAG‑GNN, GraN‑DAG), likelihood‑based methods (GOLEM), and ordering‑based SCORE / CAM. This broad applicability is well supported empirically: Tables 1–3 (Pages 8–9) and Tables 9–14 (Pages 28–30) show results for a wide range of base learners and graph regimes.

3. **Weighted voting formulation and probabilistic interpretation.**  
   The weighted score in Equation (2) on Page 4,  
   \[
   s(X\to Y) = (1-e^{-\lambda m})\frac{A}{m},
   \]  
   is a nice refinement of naive majority voting; Appendix D.1 shows it can be viewed as a posterior mean under a Beta prior whose strength decays with $m$. This provides an interpretable way of penalizing low‑support edges and is more principled than ad‑hoc thresholding.

4. **Some theoretical analysis of aggregation.**  
   The paper goes beyond “we do a vote and it works”: Theorem 3.2 (Page 5) and Lemma E.1 (Page 19) derive concentration‑based bounds on edge‑level error versus the number of local subgraphs $m$ and the vote probability $p$, and Theorem 3.5 (Page 7 / Appendix E.3) proves asymptotic consistency when $m = C\log n$. While assumptions are idealized, they do provide a first‑order understanding of how weighted voting behaves.

5. **Runtime gains via divide‑and‑conquer.**  
   Tables 3, 6–8 (Pages 8–9, 27–28) clearly demonstrate substantial speedups when wrapping heavy base learners. For example, in Table 3 with ER3 graphs and $n=300$, NOTEARS takes $12515$ s vs $2136$ s with VISTA, and DAG‑GNN drops from about $17714$ s to $1960$ s. These are meaningful improvements that matter to practitioners.

6. **Empirical evidence that weighted voting is better than naive voting.**  
   Table 1 (Page 8) nicely illustrates that naive voting (+VISTA‑NV) can drastically inflate FDR (e.g., NOTEARS ER5: FDR jumps from 0.21 baseline to 0.87 with NV), while weighted voting (+VISTA‑WV) pushes FDR back down (0.08) and improves F1 (0.76 → 0.79). Figures 4(a–c) (Page 9) further show precision‑recall trade‑offs as a function of $\lambda$, matching the qualitative behavior predicted by Theorem 3.4.

7. **Insightful figure on MB vs learner degradation.**  
   Figure 1 (Page 3) shows F1 vs number of nodes for MB identification, NOTEARS baseline, DAG‑GNN baseline, and their VISTA‑wrapped versions. MB F1 is relatively flat while base learners degrade sharply with $n$; VISTA curves lie above the baselines. This supports the central design choice of trusting MB discovery more than full‑graph learning and leveraging overlapping MB subgraphs.

8. **Comparison to another divide‑and‑conquer framework.**  
   Appendix F.2 and Table 5 (Page 26) compare VISTA against DCILP under a common base learner (DAGMA), showing VISTA‑WV yields markedly better FDR and SHD in all listed settings (e.g., ER5, $n=50$: DCILP F1 0.29 vs VISTA‑WV F1 0.86). This strengthens the case that their aggregation scheme is competitive among modular methods.

## Weaknesses

I list several substantial concerns that, in aggregate, make me doubt the readiness of the paper for ICLR.

1. **Heavy reliance on idealized independence and correctness assumptions in theory.**  
   The main theoretical results (Theorem 3.2, Corollary 3.3, Theorem 3.5) assume (a) that votes from different local subgraphs are independent, and (b) a fixed inclusion probability $p$ for true edges and $q$ for false edges. In practice, all subgraphs are estimated from the *same* dataset and share variables, so vote correlations can be strong, especially around hubs. The authors briefly acknowledge this on Page 6 (“votes … can induce correlations… the bound should be interpreted as a qualitative guide”), but then use these bounds to argue rigorous finite‑sample error control and asymptotic consistency. At minimum, the current statements are mathematically correct only under a stylized model that is far from the actual algorithm’s behavior. This gap matters because the entire selling point of VISTA is calibrated, theoretically justified aggregation; the calibration guarantee is significantly weaker once dependence is acknowledged.

2. **MB coverage assumptions are unrealistic and not analyzed under error.**  
   Proposition 3.1 (Page 3) shows that if the true MBs are known, every true edge is covered by at least two subgraphs. However, the paper never analyzes what happens when MB estimators are imperfect, which is inevitable in finite samples. There is no theoretical characterization of how MB recall / precision affect weighted voting performance, and no explicit conditions linking MB error rates to global edge error. Given that MB discovery is nontrivial and may fail in high‑dimensional noisy settings, treating Proposition 3.1 as the “foundation” of VISTA feels optimistic. An error analysis that combines MB mis‑specification with the voting bounds would be essential for a convincing story.

3. **Theoretical conditions on $m$ and $\lambda$ are decoupled from how $m$ actually arises from MBs.**  
   Multiple results require that each candidate edge appears in $m=C\log n$ *independent* local subgraphs (Theorem 3.5) or at least in $m\ge m_{\min}$ supports (Lemma E.1) with explicit lower bounds, but there is no proof that MB‑based neighborhood construction yields such $m$ in typical graphs. The ER/SF analyses in Appendix E.2 (Theorems E.4, E.5) treat $m_{ij}$ as a Poisson‑perturbed constant (mostly 2), under *true* MBs; in the actual algorithm, $m_{ij}$ depends on both MB *estimation* and on which MB algorithm is used, which is left entirely abstract. Thus the asymptotic consistency claim (Theorem 3.5) is more a statement about an oracle model “we get $C\log n$ i.i.d. votes per edge” than about the implemented VISTA with MBs.

4. **Mathematical issues / looseness in some derivations.**  
   A few specific points:
   - **Theorem 3.4 (Page 6, Appendix E.1)**: the theorem claims that choosing $\lambda$ in  
     \[
     -\frac{1}{m}\ln(1-t) < \lambda \le -\frac{1}{m}\ln \epsilon
     \]  
     “guarantees error control under the union bound”, but the proof in Appendix E.1 mainly analyzes the monotonic behavior of an upper bound $\mathcal L(\lambda)$ and argues that too large $\lambda$ makes $1-e^{-\lambda m}\approx 1$. There is no explicit step showing that *for all edges* the conditions $p>r_\lambda$ and $q<r_\lambda$ simultaneously hold within that interval, nor that $\mathcal L(\lambda)\le\epsilon$. The link between the bound and the interval is heuristic rather than rigorous.
   - **Corollary 3.3 (Page 6 / Appendix D.3)**: the derivation uses a first‑order Taylor expansion in $y=e^{-\lambda m}$ and then drops $O(y^2)$ terms, effectively giving an approximate, not guaranteed, lower bound for $m$. This approximation is then promoted in the main text as an explicit sufficient condition (Equation (4)), which is technically inaccurate since the approximation may be loose when $y$ is not very small.
   - There is some notational confusion: in Theorem 3.5 (Page 7) the margins are defined as $\delta_p=p-t$, $\delta_q=t-q$ (without weighting) whereas earlier analyses use margins relative to $r_\lambda(m)$. This change is not clearly justified and hides the dependence on $\lambda$ that is crucial to the earlier lemmas.

   These issues do not completely invalidate the intuitions, but they weaken the claim of “finite‑sample error bounds” that rigorously characterize VISTA’s behavior.

5. **Empirical behavior is not uniformly positive; VISTA often degrades baselines.**  
   While the text emphasizes improvements, several tables show that even the *weighted* version can significantly hurt performance:
   - In **Table 1 (Page 8)**, for GraN‑DAG on ER5, F1 drops from 0.06 (baseline) to 0.17 (WV), which is an improvement, but TPR remains very low (0.10) and SHD is still large; for SCORE, F1 is low across the board (0.14 → 0.31).
   - In **Table 2 (Page 8)**, for DAG‑GNN on ER5, VISTA‑NV is catastrophic (FDR 0.85, SHD ≈ 610), and while VISTA‑WV improves over NV, its FDR is still worse than baseline (0.14 vs 0.16) and SHD only modestly improves. For SCORE, VISTA‑WV lowers FDR somewhat but still leaves F1 much lower than NOTEARS or GOLEM.
   - In large‑scale settings in Tables 12–14 (Pages 29–30), SCORE+VISTA‑WV can have extremely poor FDR (e.g., Table 14 SF5 $n=300$: FDR 0.93) with tiny F1. Several GOLEM + VISTA‑WV entries at $n=300$ also have very low recall.

   These mixed results suggest the framework is not robustly beneficial and can make weak base learners worse. The paper does not provide a systematic analysis of *when* VISTA helps vs harms, beyond a few qualitative comments about “large graphs”.

6. **Real‑data evaluation is very limited.**  
   The only real dataset used is the 11‑node Sachs network (Table 4, Page 9). While this is standard, it is too small to convincingly demonstrate the claimed large‑scale benefits, and the improvements are modest: e.g., GOLEM FDR 0.80 → 0.57 and SID 50 → 48, but TPR *drops* from 0.26 to 0.18. For GraN‑DAG, SID improves from 48 to 45 but TPR drops from 0.53 to 0.29. There is no evaluation on any medium‑scale real‑world graphs (e.g., fMRI, gene expression, or known 50–100‑node networks), which makes it hard to judge practical utility.

7. **Hyperparameter selection and fairness of comparison are under‑discussed.**  
   The weighted voting requires choosing both $\lambda$ and threshold $t$. The authors fix $\lambda=0.5$, $t=0.7$ “for all main tables” (Page 8), arguing this lies inside Theorem 3.4’s interval. However:
   - The sensitivity plots in Figure 4 show considerable variation in precision/recall with $\lambda$, and the chosen point is not clearly justified as near‑optimal across tasks.
   - It is unclear whether baselines are tuned to a similar extent (e.g. regularization strengths, score penalties), or whether VISTA benefits from a de facto extra hyperparameter layer.
   - In some settings (e.g., Tables 9–11), different learners seem to use the same $\lambda,t$, but no discussion is given about whether that is appropriate for very different error profiles.

   A more systematic tuning protocol (with validation splits) or at least a sensitivity study over $(\lambda,t)$ beyond a single plot would strengthen the empirical claims.

8. **Naive voting is surprisingly poor and might be a strawman.**  
   Across Tables 1–3 and 9–14, VISTA‑NV often has FDR > 0.8 and huge SHD, sometimes far worse than the base learner (e.g., Table 1 NOTEARS ER5: FDR 0.21 → 0.87; Table 3 GOLEM $n=300$ FDR not shown but SHD skyrockets). This raises the question whether NV is an artificially fragile baseline that makes WV look good. The text refers to NV mainly as a “coverage validation,” but many readers will interpret it as a realistic voting strategy. More nuanced baselines (e.g., majority vote with per‑edge support threshold) could provide a more meaningful comparison.

9. **Limited analysis of MB estimation and MB solver choice.**  
   The MB identification step is treated almost as a black box, with a brief comment that VISTA “provides a flexible interface to plug any MB estimator” (Page 3) and a generic reference to Wu et al. (2022, 2023). However:
   - The specific MB algorithm and its parameters are not clearly described in the main text, nor is its independent performance reported (Figure 1 only shows aggregated MB F1).
   - There is no ablation comparing different MB solvers or examining sensitivity to MB quality (e.g., degrade MB F1 synthetically and see how VISTA reacts).
   - Given that MB discovery is central to both coverage and runtime, this feels under‑analyzed.

10. **Some clarity issues and minor bugs in pseudo‑code / exposition.**  
    - Algorithm 2 (Page 16) for FAS: Line 4 says “if $\mathcal G$ contains a source then choose the sink $u$ with maximum $\delta(U)$”; this is inconsistent (source vs sink) and makes it unclear whether they meant “source” or “node with positive imbalance.” This matters because FAS heuristics can change behavior substantially based on that choice.
    - In Algorithm 1 (Page 14) they use `EdgeCount + EdgeCount^{⊤}` to form the occurrence matrix; this assumes symmetric treatment of $X\to Y$ and $Y\to X$, which may double‑count in some configurations. It would help to state explicitly that for each unordered pair $\{i,j\}$ the occurrence is $A+B$.
    - Section 3.1 occasionally has typos that hinder precision (“However, while NV does not distinguish…” missing “it” etc.), and the transition between naive and weighted voting could better emphasize what is *proven* vs what is intuition.

11. **Figures could be better integrated into the narrative and interpreted more critically.**  
    - Figure 3 (Page 5) is a good high‑level overview of the pipeline, but it omits any indication of how MB size or graph sparsity affect complexity; readers might wrongly assume near‑linear scaling from the visual.
    - Figures 5–10 in Appendix F.4 show detailed performance trends (e.g., DAG‑GNN / GOLEM / NOTEARS vs graph size and degree), but the main text scarcely discusses cases where VISTA‑WV performs worse than the baseline (e.g., in some SF5 settings, see subplots in Figures 6 and 7). A more balanced discussion of these plots would improve scientific honesty.

Overall, while the idea is attractive and results are promising in many regimes, the combination of idealized theory, missing MB error analysis, and mixed empirical behavior makes the work feel not yet fully mature.

## Potentially Missing Related Work

1. **Squires & Uhler, “Causal Structure Learning: A Combinatorial Perspective”, 2022.**  
   This survey systematically discusses combinatorial aspects of DAG learning, including divide‑and‑conquer and structural constraints. It would be natural to cite and briefly position VISTA in relation to the broader combinatorial viewpoint in Section 2 (Preliminaries / Related Work), to clarify how their MB‑based decomposition fits into existing taxonomies of structure learning.

2. **Yu, Hou, Liu, “A Novel Constraint‑Based Structure Learning Algorithm Using Marginal Causal Prior Knowledge”, 2024.**  
   This work integrates local causal prior information into constraint‑based learning, conceptually related to merging local structures into a global graph. It should be discussed in the “Existing Modular Causal Discovery Paradigms” paragraph on Page 3 or in Appendix B, as another example of leveraging local information for global structure learning, and possibly as a baseline or conceptual comparison.

3. **Zhang, Yuan, Che, “Subgraph Information Bottleneck with Causal Dependency for Stable Molecular Relational Learning”, 2025.**  
   This paper explicitly studies causal dependencies at the subgraph level to improve stability in relational learning. While focused on molecules, its subgraph‑centric causal modeling is directly relevant to VISTA’s subgraph integration philosophy. A brief discussion in Section 1 or Appendix B about how VISTA’s goal (global causal discovery) differs from but relates to subgraph causal modeling in representation learning settings would help situate the contribution.

## Questions

1. **Dependence of votes and robustness of theory.**  
   Can the authors clarify how strongly dependent local votes actually are in practice for their MB construction (e.g., empirical correlation of $A_{ij}$ across subgraphs)? Are there existing concentration inequalities for weakly dependent Bernoulli sequences that could be applied to tighten Theorem 3.2 beyond the i.i.d. assumption?

2. **MB error and coverage in practice.**  
   Could you report quantitative MB metrics (precision/recall) vs graph size and then stratify VISTA performance by MB quality? For example, what happens if you plug in an oracle MB vs a noisy MB? This would greatly help understand the sensitivity of the framework to MB inaccuracies.

3. **When does VISTA hurt, and can we detect it?**  
   Several entries in Tables 9–14 show degradation. Is there any diagnostic (e.g., distribution of $m$ or vote entropy) that correlates with harmful aggregation, and could VISTA be adaptively turned off (or fallback to baseline) in such cases?

4. **Hyperparameter selection protocol.**  
   How were $(\lambda,t)$ chosen in practice? If you sweep $\lambda$ and $t$ on a validation set, do the qualitative advantages over baselines remain? It would be helpful to see at least one setting where $(\lambda,t)$ are tuned via a principled procedure and then applied to a held‑out test set.

5. **FAS ordering details and impact.**  
   In Algorithm 2, could you clarify the intended behavior regarding sources vs sinks and imbalance? Did you experiment with alternative FAS heuristics (e.g., directly using edge weights in the order construction) and, if so, how sensitive are the final graphs to this choice?

6. **Real‑world scalability beyond Sachs.**  
   Are there any larger real‑world datasets you attempted (even without full ground truth), such as large fMRI graphs or gene networks, where runtime can be measured and qualitative graph properties inspected? Even partial ground truth or intervention probes could strengthen the case for applicability.

Clarifications or additional experiments addressing these points would positively influence my assessment, especially a more thorough analysis of MB errors and empirical dependence of votes.

## Flag For Ethics Review

- No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

2: fair.  
The algorithm itself is sensible and empirically validated on synthetic data, but the theoretical guarantees rest on strong independence and oracle MB assumptions, and several derivations (Theorem 3.4, Corollary 3.3) are somewhat heuristic rather than fully rigorous. Empirical results are mixed in some regimes and the conditions for success are not fully characterized.

## Presentation Rating

3: good.  
The paper is overall readable, with helpful figures like Figure 1 and the schematic in Figure 3, and detailed appendices. However, there are some clarity issues in pseudo‑code, a few notational inconsistencies, and the narrative could more honestly discuss failure cases and limitations.

## Contribution Rating

2: fair.  
The modular MB‑based wrapper with weighted voting and FAS is a useful engineering contribution and the analysis is a first step towards understanding such aggregation, but the conceptual novelty is moderate and the theoretical + empirical story is not yet strong enough for a top‑tier venue.

## Overall Rating

4: Marginally below the acceptance threshold. But would not mind if paper is accepted.  
The idea is attractive, the framework is practical, and many results are encouraging, especially the runtime improvements and some large‑scale gains. At the same time, the theoretical claims are built on idealized assumptions, MB errors are not analyzed, and empirical benefits are inconsistent, with several instances where VISTA hurts performance. With stronger theoretical grounding under realistic dependencies or a more systematic empirical characterization of when VISTA helps, I would be more inclined toward acceptance.

## Reviewer Confidence

4: confident.  
I am familiar with causal structure learning and vote‑based aggregation, and I carefully checked the main equations and tables. Some details of specific MB solvers and all experimental configurations could not be fully verified, but overall I am reasonably confident in the assessment.