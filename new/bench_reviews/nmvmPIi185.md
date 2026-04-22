Now I have thoroughly read the paper and the calibration anchors. Let me write the final review.

## Summary

The paper introduces Neural Causal Graph (NCG), a classification framework that constructs a directed acyclic graph from WordNet's hypernym taxonomy, estimates edge weights using propensity score methods (PSM/DRL) to deconfound the shared input variable X, and performs multi-label classification with graph-based message passing. An intervention training method simulates do-operations during training by clamping prior concept logits, enabling test-time interaction where users can fix concept values and observe downstream effects. Experiments on a custom Bird dataset and ImageNet show improvements over multi-class, multi-label, and CBM baselines.

## Strengths

- **Clean reformulation of hierarchical classification as multi-label task over a taxonomy (Section 3.2.2):** Converting single-label classification into multi-label classification by including all WordNet ancestor nodes is a straightforward but effective way to inject structural priors, and is clearly implemented.

- **Thorough ablation of edge weight estimation methods (Table 3):** Comparing six weight variants (PSM, DRL, One, Random, Zero, Learn) provides solid evidence that the PSM/DRL estimated weights outperform alternatives, especially the "Learn" baseline which underperforms on ResNet50 — supporting the claim that naive gradient-based weight learning is insufficiently stable.

- **Systematic ablation of intervention training and learnable scaled weight (Table 4):** Showing that both components contribute, with intervention training providing ~2% improvement on its own, establishes that each design choice carries weight and is not redundant.

- **Test-time intervention is a genuinely useful capability (Figure 4):** The left panel of Figure 4 demonstrates that NCG with intervention training reaches ~95% top-1 accuracy on ImageNet as ground-truth prior concept labels are progressively provided — a form of interactive human-AI classification that goes beyond post-hoc interpretability.

- **Statistical rigor in reporting:** The paper consistently reports standard deviations across multiple runs and p-values from t-tests (Tables 2–4), which is above average for classification papers.

## Weaknesses

### Fatal

None. The paper makes a real, working contribution; the issues are about framing and scope rather than fundamental invalidation.

### Major

- **The "causal" framing is significantly overclaimed relative to what the graph structure actually provides.** The NCG is constructed from WordNet (Section 3.2.1, line 89), whose edges are hypernym (IS-A) taxonomic relationships — not causal relationships. "Animal" does not *cause* "Bird" in any Pearlian sense; it is a superordinate category. The paper treats these taxonomic edges as causal and applies do-calculus, backdoor adjustment, and ATE estimation to them. While one can argue that deconfounding X's shared influence on related concept nodes (Eq. 1–2) is a reasonable technique for estimating influence weights in a structured graph, calling these weights "average treatment effects" and this process "causal inference" stretches the terminology beyond its standard meaning. This does not invalidate the method — the weighting procedure may well produce useful edge parameters — but it inflates the perceived theoretical contribution. The paper's central claim of "integrating causal inference with neural networks" would be more accurately stated as "applying propensity-score-based deconfounding to estimate influence weights on a label taxonomy."

- **The "~95% top-1 accuracy on ImageNet" claim in the abstract is misleading.** The abstract states NCG "achieves nearly 95% top-1 accuracy on the ImageNet dataset by employing a test-time intervention method." A reader would reasonably interpret this as a comparable ImageNet accuracy result. In reality (Figure 4), this 95% figure requires progressively providing *ground-truth labels* for prior concept nodes at test time — an oracle experiment. The standard (no intervention) performance is ~84% with CLIP and ~74% with ResNet50 (Table 2). Presenting the oracle-assisted number as a headline result without this critical qualification in the abstract misrepresents the contribution.

- **Missing comparison to hierarchical classification baselines.** The paper positions NCG as a novel "classification paradigm" (Figure 1) yet only compares to Multi-class, Multi-label, and CBM — none of which exploit label hierarchy. Since NCG's graph comes from WordNet, the most natural comparison class is hierarchical deep classification methods (e.g., HD-CNN, hierarchical softmax, DAG-based label embedding) that also use WordNet or similar ontologies to impose structural priors. Without this comparison, it is impossible to determine whether NCG's gains come from the novel components (PSM/DRL weight estimation, intervention training) or simply from exploiting the label hierarchy — which is a well-studied technique. This gap directly undermines the paper's ability to establish its claimed contribution.

### Minor

- **ImageNet improvements are practically small:** ResNet50 goes from 73.08 → 73.75 (+0.67%), CLIP from 83.49 → 84.44 (+0.95%). While statistically significant, these small gains raise questions about the practical return on added architectural complexity (graph construction, multi-head reasoning, intervention training). The Bird dataset shows larger gains but is small and self-collected (11,700 train, 450 test, 9 posterior classes), making it a weaker basis for strong claims.

- **Figure 4 (right panel) shows DRL+CLIP accuracy *declining* with more interventions when intervention training is absent, and the explanation is inadequate.** The paper dismisses this as "anticipated" because the models "are capable of learning causal dynamics and performing interventions inherently" (line 290). But if correct interventions (ground-truth labels) degrade accuracy, this undermines rather than supports the claim of inherent causal reasoning. A better explanation would be that without intervention training, the model's concept representations become miscalibrated when clamped, which is precisely why intervention training is needed — but the paper's language overreaches.

- **No sensitivity analysis for intervention training hyperparameters.** The intervention rate p=0.15 and confidence value v=5 are stated as "empirically determined" (Section 3.3.4) without any analysis of how results change across different values. Given that these parameters directly affect the training distribution shift, a small sensitivity study would strengthen the work.

- **The propensity score model L(X) is not specified in the main text.** Equation 2 depends on the propensity score L := L(X) satisfying the conditional independence required by the backdoor criterion in Figure 3(b). The paper defers implementation details to the appendix and does not discuss what L(X) is or validate the required assumptions, making it hard to assess whether the "unbiased" label is warranted.

### Trivial

- The exogenous variable U_Cj ~ N(0,1) in Eq. 3 is added as isotropic Gaussian noise without justification, but this is a minor modeling choice that could be easily clarified.

## Nice-to-Haves

- **Comparison to at least one hierarchical classification baseline** (e.g., HD-CNN or label-embedding methods) to isolate the contribution of PSM/DRL weights and intervention training from the contribution of simply using the hierarchy.
- **Noisy (non-oracle) interventions at test time** to establish practical utility beyond the idealized setting.
- **Concrete intervention case studies** showing how fixing a specific prior concept (e.g., "is a water bird") changes posterior predictions — this would be the most compelling demonstration of "intervenable classification."
- **Replace or soften causal language** where taxonomic/structural language would be more precise (e.g., "structural influence weights" rather than "average treatment effects," "label hierarchy" rather than "causal graph").

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"CBM already supports intervention so NCG doesn't add anything"**: The paper explicitly compares to CBM in Figure 4 and shows NCG benefits from intervention while CBM degrades (right panel). The difference is that NCG has explicit graph structure for propagating interventions, while CBM has a flat bottleneck. This critique ignores the paper's own evidence.

- **"The paper claims to release code/data but this can't be verified"**: Per the hard rules, if the paper cites a GitHub repository, we treat it as existing.

- **"C ⊥ X | L assumption cannot be verified, so Equation 2 is incorrect"**: This is the standard unconfoundedness/ignorability assumption required in all propensity score methods; it is always assumed, never fully verifiable. Flagging it as unique to this paper is misleading; it is a standard caveat.

- **"Existing models lack interactive interventions" misrepresents the landscape**: While CBM supports intervention, the paper's claim is about a *new paradigm* (graph-based, not bottleneck-based), which is a different structural approach. The distinction is reasonable.

- **"Bird dataset is small and self-collected" as a fatal flaw**: This is a valid concern but the paper also validates on ImageNet with full-scale experiments. The Bird dataset's size is a minor limitation, not a fatal one.

- **Formatting nitpicks and minor notation issues**: Removed per hard rules.

## Novel Insights

The paper's most interesting empirical finding is that PSM/DRL estimated weights consistently outperform learned-from-scratch weights (Table 3, "Learn" variant), particularly on ResNet50 where Learn actually underperforms. This suggests that the propensity-score-based deconfounding provides a better initialization than gradient-based optimization for graph edge weights — possibly because the graph structure creates a difficult optimization landscape. This finding is underappreciated in the paper and could have been the basis for a stronger, more focused contribution about why and when deconfounded weight estimation helps in structured classification.

## Suggestions

- Add at least one hierarchical classification baseline (e.g., DAG-based label embedding or HD-CNN) to isolate the contribution of the NCG-specific components.
- Qualify the 95% ImageNet claim in the abstract as "with ground-truth concept interventions" or similar wording.
- Consider reframing the contribution more honestly: rather than "integrating causal inference," present it as "hierarchical classification with deconfounded edge weight estimation and intervention-capable training" — this accurately describes the contribution without the semantic overreach.
- Add a brief sensitivity analysis for intervention training hyperparameters (p, v) to strengthen the empirical contribution.

## Score and Decision

**Calibration anchors referenced:**
- `/home/wg25r/review_agent/human_reviews/mb9oOA3rD9.md` (avg 4.50, Withdrawn): Misused causal terminology for pairwise classification tasks; this paper similarly stretches causal language for taxonomic structures but has stronger empirical backing.
- `/home/wg25r/review_agent/human_reviews/OatZMyMuIo.md` (avg 4.00, Reject): Overclaimed causal invariance with technical issues in the SCM framework; this paper has similar overclaiming but a working system.
- `/home/wg25r/review_agent/human_reviews/Zju6U5CiM7.md` (avg 4.50, Withdrawn): Questioned whether model truly performs causal discovery; analogous concern here.
- `/home/wg25r/review_agent/human_reviews/9ljHiYuRHl.md` (avg 4.25, Withdrawn): Misused "anti-causal" terminology; similar linguistic overreach concern.
- `/home/wg25r/review_agent/human_reviews/I4e82CIDxv.md` (avg 8.0, Accept Oral): Sparse Feature Circuits with genuine causal intervention and editing; this paper falls well below this standard.
- `/home/wg25r/review_agent/human_reviews/4P76wCt9N5.md` (avg 3.0, Reject): DAG-based generative regression with weak methodology and overclaimed gains; this paper is somewhat better as it has a working system with ablations, but shares overclaiming tendencies.

This paper sits in the same band as the overclaimed-causal-terminology papers (4.0–4.5 range). It has a real, working method with thorough ablations and genuine (if small) empirical improvements, which puts it above the truly weak papers at 2–3. However, the misleading 95% accuracy claim and the overclaimed causal framing are significant issues that inflate the perceived contribution. Compared to the 4.0–4.5 anchors, this paper is comparable — it has similar issues with causal overclaiming but arguably more empirical substance. Score: 4.0.

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>