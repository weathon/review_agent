## Summary
This paper proposes EdgePrompt and EdgePrompt+, graph prompt tuning methods that attach learnable prompts to edges rather than nodes, with EdgePrompt+ producing edge-specific prompts via anchor prompts and an attention-like scoring function. The idea is intuitive and potentially useful: by intervening at message passing rather than only at node features/readout, the method opens a meaningful new axis for graph prompt design, and the experiments across ten datasets and four pretraining strategies show the approach is often strong.

## Strengths
- **Clear and meaningful methodological idea.** The paper identifies a genuine limitation of node-prompt methods—prompts attached to a node are propagated uniformly to all its neighbors in common message-passing GNNs—and proposes edge-level prompting as a direct fix. Figure 1 and the discussion in Section 4.3 make this motivation concrete.
- **Reasonable practical design for few-shot prompt tuning.** EdgePrompt+ does not naively learn one free vector per edge; instead it parameterizes edge prompts through a shared set of anchor prompts and edge-dependent mixture weights (Eq. 4–6), which is a sensible way to make the method trainable in few-shot settings.
- **Broad experimental grid within the paper’s chosen setting.** The evaluation covers 5 node-classification datasets and 5 graph-classification datasets under 4 pretraining strategies, with comparisons against multiple prompt-tuning baselines. This is a fairly comprehensive benchmark within the few-shot prompt-tuning setting.
- **Empirical results are generally strong, especially for EdgePrompt+.** In Tables 2 and 3, EdgePrompt+ is frequently the best or runner-up across many settings, especially on graph classification where it is consistently among the top methods.
- **Some theoretical effort is made to justify the method.** The CSBM-based analysis in Theorem 1 at least attempts to formalize why edge prompts could improve separability for node classification, which is better than a purely heuristic presentation.

## Weaknesses

###: Fatal
- **Theorem 2 / the associated universality claim appears materially overstated and is not credible as stated.** The paper claims in Section 4.4:  
  *“Given an input graph \(\mathcal G=(X,A)\) and its transformation \(\mathcal G'=(X',A')\) by an arbitrary transformation function \(\mathcal T\), there exists a set of edge prompt vectors ... such that \(f(X,A,\{p^{(1)},...,p^{(L)}\}) = f(X',A')\) for any pre-trained GNN model \(f\).”*  
  This is an extremely strong statement. As written, it says edge prompts on the original graph can reproduce the representation of **any** transformed graph under **any** pre-trained GNN. That goes far beyond what is justified in the main text, and the subsequent claim that EdgePrompt is universally capable of matching “any prompt strategies” depends on it. Since this is one of the paper’s headline theoretical claims, the paper should either substantially narrow the theorem statement/interpretation or provide a much more carefully qualified result. As is, this undermines an important conceptual pillar of the work.

### Major:
- **The broad architecture-compatibility claim is stronger than the evidence provided.** The abstract claims the method is *“compatible with prevalent GNN architectures pre-trained under various pre-training strategies.”* However, the experiments only use a 2-layer GCN for node classification and a 5-layer GIN for graph classification (Section 5.1). Moreover, Eq. (2) modifies the aggregation behavior of the backbone itself; this is not merely attaching an external prompt while leaving the forward computation untouched. That does not invalidate the method, but it does mean the paper has demonstrated compatibility for the tested settings, not established a broad architecture-agnostic claim.
- **The empirical claims of “superiority” are somewhat overstated relative to the tables.** The results are good, but not uniformly dominant. There are several settings where gains are very small, within reported standard deviations, or where another method wins (e.g., Table 2, GraphCL on Flickr where GraphPrompt is best; EP-GraphPrompt on ogbn-arxiv where EdgePrompt outperforms EdgePrompt+; several Table 3 cells where GPF-plus or GraphPrompt are essentially tied or better). The evidence supports “often best / consistently competitive,” more than blanket superiority.
- **No computational cost or scalability analysis is given, despite EdgePrompt+ being edge-wise.** EdgePrompt+ computes edge-dependent scores and prompt mixtures for every edge at every layer (Eq. 4–6), which is a real extra cost compared with node-prompt alternatives. The paper does not report runtime, memory, or parameter-count comparisons, even on larger graphs like ogbn-arxiv. Since the method’s main novelty is edge-level prompting, this omission matters for practical assessment.
- **The theory-to-practice connection is limited.** Theorem 1 is an existential result: there *exist* anchor prompts and score vectors that improve class-centroid separation under a 2-class CSBM and pre-trained GCN. That is useful intuition, but it does not show the actual parameterization in Eq. (4–6), optimized from few labels via Eq. (7), will reliably learn such prompts in practice. The theorem therefore provides suggestive support, not strong validation of the practical learning procedure.

### Minor
- **Theoretical scope is narrow relative to the paper’s breadth of claims.** Theorem 1 is restricted to a two-class CSBM setting and explicitly discusses pre-trained GCNs. That is acceptable as a stylized analysis, but it only partially supports claims spanning multiple tasks, datasets, and architectures.
- **The score-function choice in EdgePrompt+ is under-ablated.** Eq. (6) adopts a GAT-style attention scoring function, and the paper explicitly defers investigation of alternatives to future work. Since this score function is central to how edge prompts specialize, an ablation over alternative \(\phi\) choices would strengthen the method substantially.
- **Convergence claims are only partially evidenced.** Figure 2 shows selected curves and suggests faster convergence, but there is no quantitative summary and only a subset of settings is shown. This is supportive but not enough to establish a general convergence advantage.
- **The comparison between EdgePrompt and GPF partly weakens the practical case for the simpler variant.** The paper itself notes the performance gaps between GPF and EdgePrompt are often small. This does not hurt EdgePrompt+, but it suggests the shared-prompt version may not by itself be a compelling advance over strong node-prompt baselines.

### Trivial
- **Table 1 and naming introduce slight ambiguity.** The table highlights “EdgePrompt+ (Ours)” while the paper presents both EdgePrompt and EdgePrompt+ as contributions. This is minor but mildly confusing about which method is the primary object of comparison.

## Nice-to-Haves
- Add fine-tuning baselines to contextualize how far prompt tuning remains from full adaptation.
- Report performance across varying shot counts (e.g., 1/3/5/10/20) rather than only 5-shot and 50-shot.
- Add parameter-matched comparisons against GPF-plus or other baselines to isolate whether gains come from edge-level design rather than extra capacity.
- Include visualizations of learned edge prompt assignments / anchor mixtures to validate the claimed mechanism.
- Study heterophilic graphs or settings with edge attributes to clarify the method’s boundary conditions.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“Lack of significance testing” as a core flaw.** The paper already reports mean ± standard deviation over five runs, which is a reasonable empirical practice in this setting. It is fair to say the superiority claims should be toned down because some gains are small relative to variance, but requiring formal significance testing would be more of a nice-to-have than a core defect here.
- **“No experiments on graphs with existing edge features” as a decisive weakness.** This would certainly broaden applicability, but the paper’s goal is to design edge prompts for prompt tuning, not to solve all edge-feature settings. This is outside the core claimed contribution.
- **Baseline unfairness due to asymmetry in method design.** The comparisons do not obviously favor the proposed method in a way that invalidates conclusions, and the paper standardizes the downstream classifier and optimization setup. It would still help to report parameter counts, but the current evidence is not enough to call the baseline comparisons unfair.
- **Pure reproducibility nitpicks about unspecified implementation details.** The paper gives the backbone choices, hidden size, optimizer, LR, batch size, epochs, shots, and anchor counts in the main text; lacking further low-level details is not a substantive review point.

## Novel Insights
The most important synthesis is that this paper is strongest as an **empirical and conceptual extension** of graph prompt tuning—moving prompts from nodes/readout to message-passing edges—rather than as a theory-heavy universality paper. The experiments and intuition justify edge prompting as a meaningful design direction, especially EdgePrompt+, but the paper overreaches when it tries to elevate this into a very broad architecture-compatibility and universality claim. In other words: the method itself is more convincing than the strongest claims made on its behalf.

## Suggestions
- Narrow or restate Theorem 2 and any downstream universality language so that the theoretical claim matches what can actually be justified.
- Rephrase the compatibility claim to align with the demonstrated evidence: compatibility with the tested backbones/pretraining settings, rather than all prevalent GNN architectures.
- Tone down “superiority” claims to “often best and consistently competitive,” unless stronger statistical or broader empirical support is added.
- Add runtime, memory, and parameter-count comparisons, especially for EdgePrompt+ versus GPF-plus on large graphs.
- Include ablations on the score function \(\phi\) and possibly layer-wise edge prompting to clarify what parts of EdgePrompt+ matter most.

## Score and Decision
**Evaluation by axis:**  
- **Originality:** Good. Edge-level prompt tuning is a simple but genuinely novel perspective in this literature.  
- **Importance of research question:** Good. Bridging pretraining/downstream gaps for GNNs is worthwhile, and prompting is relevant.  
- **Whether the claims are well supported:** Mixed. The empirical claim that EdgePrompt+ is often strong is supported; the strongest universality/compatibility claims are not.  
- **Soundness of experiments:** Above average overall, with broad coverage, but weakened by the lack of efficiency analysis and some overinterpretation.  
- **Clarity of writing:** Generally clear and well organized.  
- **Value to the research community:** Moderate to good. The method is likely useful, but the paper needs tighter claim calibration.

**Calibration against human review anchors:**  
I compared this paper primarily against:
- **/home/wg25r/review_agent/human_reviews/VBeLiRkZMP.md** (“Instance-Aware Graph Prompt Learning,” scores 6/5/5/5, reject/withdrawn): similar space, with practical promise but concerns about complexity/fair comparison and limited empirical separation. The present paper is somewhat stronger in breadth of evaluation and cleaner motivation, but it also overclaims theoretically.
- **/home/wg25r/review_agent/human_reviews/XrtFVM1f6w.md** (“value of edge embeddings in GNNs,” scores 5/5/5/6, reject): relevant because it shares the edge-centric motivation and practical concern about computational overhead. The current paper is more application-driven and empirically broader, but similarly leaves efficiency underexplored.
- **/home/wg25r/review_agent/human_reviews/C1wSR50nYf.md** (“Does Graph Prompt Work?”, scores 3/5/6, reject/withdrawn): useful as a theory-heavy graph prompting anchor. Compared with this, the present paper has stronger empirical grounding, but its strongest theorem claim also appears overstated.
- As a higher-end contrast, **/home/wg25r/review_agent/human_reviews/wFs2E5wCw6.md** (accept poster, mostly 6–8) shows what stronger support for claims and more complete validation looks like; this submission falls below that bar because its headline claims outrun its evidence.

Relative to these anchors, this paper lands in the **borderline reject / weak reject** range: more substantive than clearly weak submissions, but not convincing enough for acceptance in its current form because one major theoretical pillar appears overstated and the empirical framing is too aggressive for the evidence shown.

**Final score: 4.5 / 10**  
**Decision: Reject**

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>