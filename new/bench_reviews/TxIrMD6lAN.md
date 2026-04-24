## Summary  
This paper proposes an incremental learning architecture that uses task-specific adapters with co-training of the backbone network to separate invariant and task-specific representations, aiming to improve both stability and plasticity. The authors demonstrate that adapter-enhanced variants of several regularization-based methods (EWC, MAS, PathInt, LwF, LwM) yield higher average accuracy on CIFAR-100 compared to their non-adapter counterparts, and show benefits on ImageNet and when combined with DualPrompt/iTAML.

## Strengths  
- **Novel architectural split via adapters** – Repurposing adapters as feature modifiers while co-training the backbone is a fresh perspective for incremental learning, clearly described in Section 3.2 and Figure 2.  
- **Consistent gains on CIFAR-100** – Figure 3 shows adapter-augmented methods outperform baselines by 3–5% on average across 10 tasks, demonstrating a strong empirical signal.  
- **Robustness to task ordering** – Figure 5 confirms the advantage persists under coarse-grained and iCaRL random orderings, highlighting resilience to inter-task diversity.  
- **Broad compatibility** – Adding adapters to DualPrompt and iTAML yields >1% improvements (Table 2), and the proposed Adapter+LwF surpasses TAMiL (74.0% vs 71.4%).  
- **Scalability to larger datasets** – Adapter variants of LwF and LwM show non-trivial gains on ImageNet-Subset (Table 1), though results are mixed for EWC.

## Weaknesses  

### Fatal  
None.  

### Major  
- **Unsupported primary-driver claim** – The abstract states that “inter-task differences are the primary driver of catastrophic forgetting,” but the only evidence is that coarse-grained ordering (higher inter-task diversity) harms LwF performance (Figure 1). This shows correlation, not primacy over other factors like architecture, optimization, or data shifts. The central motivation is therefore not validated.  
- **Overstatement regarding the stability-plasticity dilemma** – The paper claims the approach “effectively addressing” (abstract) and “resolve[s]” (conclusion) the dilemma, but Figure 3 shows all methods—including adapter variants—exhibit declining accuracy as tasks increase. The trade-off persists; adapters only shift the curve upward. This misrepresents the contribution as eliminating a fundamental dilemma rather than improving it.  
- **Adapters not isolated as the source of improvement** – Adapter-enhanced methods differ from baselines in three ways: (1) the adapter modules themselves, (2) co-training the backbone (vs. frozen backbones in prior adapter work), and (3) method-specific regularization adjustments (e.g., extra backbone distillation for LwF, exclusion of adapter parameters from EWC penalties). The “frozen backbone” ablation (LwF-A vs. LwF-A-FrB in Table 2) isolates (2), but no experiment isolates (1). Without comparing baseline + co-training + identical regularization versus baseline + co-training + adapter, the attribution of gains to the adapter architecture itself is uncertain; the improvements could stem from the extra regularization alone.  
- **Inconsistent performance on ImageNet** – EWC-A underperforms the non-adapter EWC baseline on all later tasks (Table 1), contradicting the claim of universal superiority. The authors attribute this to hyperparameter transfer from CIFAR-100, but this reveals a lack of robustness and suggests the adapter approach may not generalize without careful tuning.  
- **Missing contemporary SOTA comparisons** – While the paper benchmarks against some recent techniques (DualPrompt, iTAML, TAMiL), it omits dominant rehearsal-based methods such as DER++ and GDumb. Without these, the reader cannot judge whether the adapter framework advances the state of the art in incremental learning.  

### Minor  
- **No statistical analysis** – All results report means over 10 runs but provide no standard deviations or confidence intervals. For marginal gains (1–3%), statistical significance is unclear.  
- **Backbone regularizer lacks justification** – The additional distillation term on backbone outputs (Eq. 1) is introduced without theoretical motivation and without an ablation showing its necessity. It might be that co-training alone suffices, or that this regularizer could benefit non-adapter baselines.  
- **Ambiguous TAMiL comparison** – The statement “aligning our setup with theirs” does not clarify whether training epochs, learning rates, or data splits were identical. Without identical protocols, the reported 74.0% vs 71.4% may not be a fair comparison.  
- **Class‑IL results confined to appendix** – The more practical class-incremental setting is relegated to Appendix B without any discussion in the main text, making it harder to evaluate the method’s real-world applicability.  

### Trivial  
- Minor notation and phrasing nits.  

## Nice-to-Haves  
- Theoretical analysis of why the adapter bottleneck dimension c = number of classes is effective.  
- Per-class accuracy heatmaps or confusion matrices to illustrate forgetting prevention.  
- Visualization of adapter activation patterns across tasks and layers.  
- Quantification of computational overhead (training time, memory) beyond parameter count.  
- Ablations separating adapter effect from backbone regularization (e.g., LwF with backbone regularizer but no adapter; EWC with adapters but full regularization).  
- Use of consistent architectures across datasets (e.g., ResNet-32 for CIFAR, ResNet-50 for ImageNet).  
- Statistical significance testing for reported improvements.  
- Direct comparison with recent SOTA (DER++, GDumb) on standard class‑IL benchmarks.  
- Relocate class‑IL results to main paper with proper discussion.  

## Removed Points  
These points are flagged to be removed because they violate the review rules (e.g., questioning existence/availability of cited material, factual inaccuracies, or nitpicks). Treat them with caution.

1. *“Evidential: The evaluation is incomplete for the most important incremental learning setting—class-incremental learning (class-IL). … (unavailable).”* – The paper explicitly states class‑IL results are included in Appendix B, so claiming “unavailable” is factually wrong. (Rule: REMOVE criticisms questioning cited material’s existence.)  
2. *“Section 4.1: Network architecture choice is inconsistent… This makes cross‑dataset comparisons potentially invalid.”* – Using ResNet‑34 for CIFAR‑100 and ResNet‑18 for ImageNet is a common practice given input resolution differences; this does not invalidate internal adapter vs. non‑adapter comparisons.  
3. *“Section 4.1: The validation set construction… needs justification that it doesn’t advantage their method.”* – Following a standard class‑balanced 10% split from prior work is reasonable and not inherently advantageous.  

## Novel Insights  
None beyond the paper’s own contributions.  

## Suggestions  
- Include a rigorous ablation that isolates the adapter’s contribution: compare baseline + co‑training (same as current baseline) against baseline + co‑training + adapters while keeping all other conditions identical.  
- Re‑tune hyperparameters on ImageNet for each adapter‑enhanced method to ensure fair comparison; report both CIFAR‑tuned and ImageNet‑tuned results.  
- Move the class‑IL results into the main paper and discuss them explicitly.  
- Add standard deviation bars or confidence intervals to all plots and report p‑values for key comparisons.  
- Correct all overstatements (e.g., “eliminate/resolve the stability‑plasticity dilemma” → “improve the stability‑plasticity trade‑off”; “primary driver” → “a key factor”).  

## Score and Decision  
MY FINAL SCORE: <pineapple>4.0</pineapple>  
MY FINAL DECISION: <orange>Reject</orange>  

*Calibration note*: This score is anchored against the following human-reviewed papers:  
- /home/wg25r/review_agent/human_reviews/5U1rlpX68A.md (avg 7.50, accept) – combines theoretical analysis with strong multi‑benchmark results; our paper lacks theory and has significant replication/ablation issues.  
- /home/wg25r/review_agent/human_reviews/dOAkHmsjRX.md (avg 7.50, accept) – introduces a budget‑aware framework with adaptive freezing; our paper’s evaluation is less comprehensive and omits key SOTA.  
- /home/wg25r/review_agent/human_reviews/H6pf70GZVU.md (avg 5.0, reject) – novel IL idea but poor reproducibility and insufficient ablation; our paper shares these flaws and additionally overstates claims and shows inconsistent ImageNet results.  
- /home/wg25r/review_agent/human_reviews/WReszdNNdP.md (avg 5.25, reject) – limited novelty and missing diverse datasets/SOTA; our paper similarly lacks SOTA breadth.  
- /home/wg25r/review_agent/human_reviews/Sxi6gBtJcI.md (avg 3.00, withdraw) – overclaim (“eliminate hallucinations”) with incremental method; our overclaim about “resolving the stability‑plasticity dilemma” parallels this fatal presentation issue.  

The paper demonstrates promising empirical gains on CIFAR‑100 but suffers from serious overstatements, an incomplete ablation that questions the attribution of improvements to adapters, inconsistent ImageNet performance, and missing SOTA comparisons. These weaknesses collectively undermine confidence in the stated contribution, leading to a reject decision.