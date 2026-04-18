Now let me read some calibration papers from the human review database to calibrate my score.Now I have sufficient information to write the final consolidated review.

---

## Summary

HG-Adapter proposes the first "pre-train, adapter-tuning" framework for pre-trained Heterogeneous Graph Neural Networks (HGNNs). The method introduces dual structure-aware adapters — a homogeneous adapter that learns a k-NN adjacency from node features, and a heterogeneous adapter that learns edge-type attention weights — applied on top of frozen pre-trained HGNNs. It additionally designs a label-propagated contrastive loss and two self-supervised losses (feature reconstruction, margin loss) to exploit unlabeled nodes as "potential labeled data." The paper presents a generalization bound intended to ground the design choices theoretically, and reports consistent improvements over prompt-tuning baselines on four heterogeneous graph datasets across three pre-trained backbones.

---

## Strengths

- **Novel problem formulation**: This is the first work to systematically apply parameter-efficient adapter tuning to pre-trained HGNNs. Existing prompt-tuning methods (HetGPT, HGPrompt) focus exclusively on feature-level modifications; designing adapters that explicitly model structural topology (both homogeneous and heterogeneous) is a meaningful and underexplored contribution in this space.

- **Coherent dual-adapter architecture**: The homogeneous adapter adaptively learns a class-homophilic graph structure from features without requiring expert-designed meta-paths, while the heterogeneous adapter assigns learned importance weights to neighbor edge types. The low-rank decomposition (W = W_down × W_up, t ≪ d) keeps parameter overhead small. Figure 2(a) and 2(b) provide empirical evidence that the learned structures are semantically meaningful — homophily ratios increase steadily, and removing max-weight neighbors significantly hurts performance.

- **Consistent empirical gains**: HG-Adapter outperforms both fine-tuning baselines and prompt-tuning methods (HetGPT) across four datasets (ACM, Yelp, DBLP, Aminer) and three pre-trained backbones (HDMI, HeCo, HERO), for both node classification and node clustering. Gains on clustering (+2.69% on average vs. HetGPT) are particularly notable.

- **Plug-in flexibility**: The framework is agnostic to the pre-trained backbone, requiring only that the pre-trained model exposes a frozen representation and message-passing structure. This generality is a practical virtue.

---

## Weaknesses

### Fatal
*None.*

### Major

- **Theoretical claims significantly outrun the actual derivations.** Theorem 2.3 is a standard uniform convergence bound of the form U(ε_M) = ε̂_M + O(√|P_M|/n_M)), presented as a restatement of results from Arora et al. (2018) and Aghajanyan et al. (2021). It is not tailored to HGNNs, adapters, or graph-structured data; the complexity measure is hidden in Big-O notation with no explicit constants, assumptions on loss/hypothesis class, or characterization of the function class. The paper then claims — in the abstract, introduction, and conclusion — to "theoretically demonstrate that the proposed method achieves a lower generalization error bound than existing prompt-tuning-based methods." The formal comparison, however, is pushed to Appendix C.2 (not present in the main text), and the visible argument reduces to two informal claims: (1) the adapters "decrease training error" by getting "closer to the optimal parameters" — but closeness to the optimal is circular since the optimal is defined as whatever minimizes the RHS, and (2) pseudo-labeled nodes "increase n_M" in the bound — but replacing ground-truth labels with noisy propagated ones does not simply scale the clean sample count in standard generalization theory. The paper acknowledges uncertainty in pseudo-label confidence ("the confidence... may be insufficient during the early stages of training") but still reuses n_M from the original bound verbatim. As currently presented, the theory functions as intuitive narrative rather than a rigorous, mechanism-specific bound, and cannot sustain the strong generalization guarantee claims in the abstract and conclusion.

- **No experiments probing the generalization narrative.** Despite the paper's heavy emphasis on "generalization ability," all evaluations use standard fixed transductive splits on four small benchmark datasets. There are no label-rate variation experiments (e.g., 1%, 5%, 10% labeled nodes), no cross-graph transfer evaluations, and no robustness tests. The claim that the method specifically reduces the generalization gap — not just achieves better task performance — is left untested. Figure 2(c) shows test error curves for three tuning methods, but this is simply final task performance on a fixed split; it does not decompose training error vs. generalization gap, which is the central theoretical narrative.

- **Ablation table (Table 2) has a clear formatting error that undermines interpretation.** Two rows in Table 2 are labeled identically as "— ✓ —" (rows 2 and 3 in the table), yet produce radically different results: one row yields Macro-F1 ≈ 32.3% (near-random) while the other yields 87.9% on ACM. The second row likely corresponds to "✓ — —" (only L_con), making it effectively a critical ablation result — but as presented it is uninterpretable. Similarly, the "— ✓ ✓" row (without L_con) collapses to ≈ 32% Macro-F1, suggesting the method is completely unstable without L_con. The paper's discussion does not explain this brittleness, noting only that "label information is necessary." These erratic configurations warrant deeper analysis, not silence.

### Minor

- **No ablation on the adapters themselves.** Table 2 ablates only the three loss terms (L_con, L_rec, L_mar) while keeping the dual-adapter architecture fixed throughout. There is no comparison against a version without the structure-aware adapters (e.g., standard MLP adapter or LoRA-style feature adapter with the same parameter budget). This means it cannot be determined whether the performance gains come from structural tuning or simply from adding any trainable module.

- **No computational complexity or efficiency analysis.** The homogeneous adapter requires pairwise similarity computation for k-NN construction (O(n²) in principle), label propagation is over all nodes, and the margin loss sums over node pairs with different pseudo-labels. For large-scale heterogeneous graphs, this could be a bottleneck. No training time or memory comparison is provided.

- **Numerous hyperparameters without sensitivity analysis.** The method introduces at least seven hyperparameters (τ, λ, η, μ, α, β, γ) plus architectural choices (k, t, t'). No sensitivity analysis or practical guidance is provided for tuning these, which is important given how brittle some ablation configurations appear to be.

- **Concatenation fusion without justification or ablation.** The final representation Z is formed by simple concatenation of homogeneous and heterogeneous representations. No ablation or justification compares this against alternatives (e.g., gated fusion, attention-weighted combination), and the paper presents this as a fixed design choice without discussion.

### Trivial

- **Table 2 row-label typo** (two rows identically labeled "— ✓ —"): This needs to be corrected with proper labels in the final version.
- **Notation inconsistency in Eq. (5)**: The equation uses $\tilde{\mathbf{h}}_i$, while the text refers to $\hat{\mathbf{h}}_i$ and $\hat{\mathbf{h}}_j$ for the frozen representations used in similarity computation. This discrepancy creates momentary ambiguity about which representation is used.

---

## Nice-to-Haves

- **Label-rate variation experiments**: Running HG-Adapter at 1%, 5%, and 10% label fractions would directly validate the paper's core motivation (helping under limited labeled data) and provide evidence for or against the generalization gap reduction claim.
- **Visualization of learned edge-type weights S**: Showing which neighbor types receive high vs. low weights per dataset would reveal whether the heterogeneous adapter learns semantically meaningful patterns.
- **Training curve decomposition**: Showing training error and test error separately for HG-Adapter vs. baselines (as opposed to just test error in Fig. 2(c)) would directly test the theoretical narrative that adapters reduce training error while data extension reduces the gap.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic – "Adapter interleaving under-specified across HDMI/HeCo/HERO"**: The paper's description of adapters as a general plug-in on top of frozen pre-trained representations is sufficient for understanding the mechanism. Implementation details for specific backbones are a minor reproducibility concern properly relegated to the appendix, not a substantive weakness.

- **Harsh Critic – "Claiming contrastive pre-training tasks can be reformulated as subgraph similarity (hand-wavy)"**: The paper cites Liu et al. (2023b) and Yu et al. (2024b) for this claim and provides its own formulation in Eq. (12). This is a motivational argument for the contrastive loss design, and the claim exists in the cited literature. Criticizing its brevity crosses into scope-creep.

- **Spark – "No comparison with fine-tuning at matched parameter count"**: The paper's explicit framing is adapter-tuning as an alternative paradigm to both fine-tuning and prompt-tuning, not a parameter-matched comparison with fine-tuning. Evaluating the method on whether it is better than a parameter-controlled fine-tuning variant is outside its stated scope.

- **Human Finder – "Novelty is incremental; all components are well-known"**: While the individual components (low-rank adapters, k-NN structure, label propagation, InfoNCE-style loss) are each established, their specific combination and application to the novel problem of adapter-tuning pre-trained HGNNs with joint structural tuning is not. Per the paper, this is the first such work. Calling it "pure combination" mischaracterizes the architectural integration.

- **Human Finder – "Only 4 datasets / 2 tasks are insufficient"**: Four heterogeneous graph datasets with two downstream tasks is standard in the HGNN pre-training literature (e.g., HERO, HeCo, HGPrompt all use similar setups). Criticizing this as insufficient scope applies equally to all prior work and is not a differential weakness for this paper.

---

## Novel Insights

The paper's most genuinely interesting (if underexplored) contribution is the joint treatment of homogeneous and heterogeneous structural learning within the adapter-tuning paradigm. Rather than asking pre-trained HGNN representations to generalize via feature-level prompts, HG-Adapter restructures the aggregation topology at test time — learning a class-homophilic adjacency and importance-weighted heterogeneous graph simultaneously, without requiring meta-path knowledge. The label-propagated contrastive loss that reformulates downstream objective functions as subgraph similarity is also a flexible idea worth generalizing: if contrastive pre-training tasks can indeed be unified under subgraph similarity (as the cited literature argues), then this loss function provides a clean bridge between arbitrary pre-training objectives and semi-supervised tuning. Neither insight is deeply analyzed in the paper, but both represent potentially useful primitives for the broader pre-training literature on heterogeneous graphs.

---

## Evaluation on Core Axes

- **Originality**: Moderate-to-high within the narrow HGNN adapter-tuning niche. The dual structural adapter design is not a direct combination of prior work; there is no prior art on adapter-tuning pre-trained HGNNs. However, the individual components are standard.
- **Importance of research question**: High. Parameter-efficient tuning of pre-trained models is a central problem, and structural information is indeed neglected in existing HGNN prompt-tuning methods.
- **Claims well-supported**: Partly. Empirical claims are well-supported; theoretical claims of "lower generalization error bound" are not rigorously established.
- **Soundness of experiments**: Adequate for a conference paper, but gaps in the ablation (no adapter-only ablation), the Table 2 formatting bug, and the absence of label-rate experiments are notable. Results are modest in some configurations (e.g., +0.4% Macro-F1 on ACM when building on HERO).
- **Clarity of writing**: Generally clear and well-organized. Minor notation inconsistencies exist.
- **Value to community**: Moderate. The adapter-tuning framework is practically useful and the plug-in nature is appealing. But the overreaching theoretical claims dilute credibility.

---

## Score and Decision

**Calibration anchors:**
- `3FJOKjooIj.md` (HERO paper, the backbone this work builds on) — Scores 6,8,8,6,6,8 → **Accept**. Stronger theoretical analysis (grouped effect proved), same datasets, arguably more foundational contribution.
- `6j0oKBo196.md` (graph OOD adapter, combination of known techniques) — Scores 5,5,3,3 → **Reject**. Comparable novelty concerns, weaker results.
- `1JiIKjcwrr.md` (BFTNet heterogeneous graph, incremental novelty) — Scores 6,3,3 → **Reject**. Stronger by one reviewer, but similar incremental concerns and no complexity analysis.
- `tolvZ5BS50.md` (ELU-GCN, generic theory + structure learning + contrastive loss) — Scores 5,3,5,3 → **Withdrawn**. Very close analog: graph structure learning + label propagation + generic generalization bound, but for homogeneous GCN. That paper had similar theoretical weaknesses.

HG-Adapter sits between these reference points. It is more principled and empirically consistent than 6j0oKBo196 and tolvZ5BS50, and addresses a genuine gap (structural neglect in HGNN prompt-tuning). However, it falls short of 3FJOKjooIj (HERO) because its theoretical story is weaker and its ablations are incomplete. The core architectural contribution is genuine and the results are broadly positive, but the theory overclaims substantially and the missing experiments (label-rate variation, adapter-only ablation) leave the central narrative only partially validated. The Table 2 formatting issue is also a concern.

**Score: 5.5** — Borderline reject. The empirical and architectural contributions merit consideration, but the paper cannot currently support its headline theoretical claims, the ablation has a clear error, and key validation experiments are absent. Revisions that (i) substantially clarify or weaken the generalization bound narrative to match what is actually proved, (ii) fix Table 2, (iii) add label-rate variation experiments, and (iv) include an adapter-only ablation would meaningfully strengthen the submission.

## Suggestions

1. **Revise or properly scope the theory**: Either derive a bound that is genuinely specific to the HG-Adapter architecture (incorporating the graph structure, pseudo-label noise, and contrastive loss), or reframe Section 2.2–2.4 as "theoretical motivation" rather than a formal proof of generalization superiority over prompt-tuning methods. The current framing misleads readers about what has been shown.
2. **Fix Table 2 immediately**: Correct the duplicate row labels. Consider explaining why configurations without L_con yield near-random results — this brittleness is potentially informative and deserves a sentence of explanation.
3. **Add label-rate variation experiments**: Even one dataset with 1%, 5%, 10% labeled nodes would directly validate the paper's stated motivation and strengthen the "labeled data extension" narrative.
4. **Add an adapter-only ablation**: Compare the full method against a version using only the dual adapters (no L_rec, L_mar, no label propagation) to isolate the structural contribution from the semi-supervised one. Also consider a flat MLP adapter baseline to isolate the structure-aware design choice.
5. **Report training error alongside test error** in Figure 2(c) to directly verify the theoretical claim that adapters reduce training error while label extension reduces the generalization gap.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>