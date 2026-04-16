Now I have a solid understanding of the paper and the calibration landscape. Let me construct the final review.

Key calibration anchors:
- **Causal CGMs** (lmKJ1b6PaL.md): Accepted poster, scores 6,6,8,6,8 (avg ~6.8). Similar topic (causal concept graphs for interpretable classification with intervention). Had novelty concerns (similar to CausalVAE), and the causal framing was questioned, but experiments were solid and matched black-box performance.
- **Counterfactual CBMs** (w7pMjyjsKN.md): Accepted poster, scores 8,8,6,5 (avg ~6.75). Strong intervention methodology, well-evaluated.
- **Energy-Based CBMs** (I1quoTXZzc.md): Accepted poster, scores 6,8,6,6,6 (avg ~6.4). Novel architecture for concept interaction + intervention.
- **Beyond CBMs: Intervenability** (5oJlyJXUxK.md): Rejected, scores 5,8,8,5 (avg ~6.5 but rejected). Concept intervention on black boxes.
- **SupCBM / Hierarchical CBMs** (Q9Z0c1Rb5i.md): Rejected, scores 3,6,5,6 (avg ~5). Hierarchical concepts with info leakage issues, poor clarity.

Now let me assess the NCG paper against these anchors:

The NCG paper has a meaningful core idea — using WordNet hierarchy as a concept graph and propagating information through it via causal-estimation weights. However, the paper has serious issues:

1. **The causal framing is substantially overstated**: WordNet IS-A relations are treated as causal, which is a misapplication of causal language. PSM/DRL are applied to estimate "causal effects" between concepts, but the identification assumptions are neither verified nor discussed.

2. **Missing critical baselines**: No comparison with hierarchical classification methods (hierarchical softmax, DAGGER), GNN-based label relation methods (ML-GCN), or with the same graph structure + simpler weight schemes as proper ablations.

3. **The 95% ImageNet claim is misleading**: It relies on oracle ground-truth intervention labels, not a realistic human-AI interaction scenario.

4. **Modest improvements on ImageNet**: ~0.5-1% over multi-class baseline, which could be explained by hierarchical label supervsion alone.

Compared to Causal CGMs (accepted, ~6.8), NCG has a weaker theoretical grounding (Causal CGMs properly define causal models with identifiability guarantees) and weaker evaluation (only 2 datasets, both ImageNet-derived; Causal CGMs used proper intervention experiments with human-in-the-loop). NCG also lacks the qualitative analysis and user study that concept-based intervention papers typically include.

Compared to the rejected SupCBM (avg ~5), NCG has similar issues with conceptual clarity around the "causal" framing but has stronger experimental methodology.

I would rate this around 4.5-5. It has a reasonable idea (structured concept graph for classification), but the causal overclaim, missing baselines, and misleading 95% claim pull it down. The core contribution when stripped of its causal framing is just "hierarchical label propagation," which is not novel enough.

Let me write the final review now.</think>Now I'll compile my final review, carefully verifying each claimed weakness against the paper text.

## Summary

The paper introduces Neural Causal Graph (NCG), a framework that constructs a directed concept graph from WordNet hierarchies (prior concepts as ancestors, posterior concepts as leaf labels), estimates edge weights using propensity score matching (PSM) or doubly robust learning (DRL), and performs classification via message-passing over this graph. An intervention training method allows test-time "what-if" reasoning by clamping concept logits, and experiments on ImageNet and a self-collected Bird dataset show improvements over multi-class, multi-label, and CBM baselines, with near-95% top-1 accuracy on ImageNet under test-time intervention.

## Strengths

- **Well-motivated goal**: The paper tackles the important problem of making classification models interpretable and intervenable at the concept level, going beyond post-hoc explanation. The idea of a structured concept graph that supports human-AI interaction is timely and practically relevant.

- **Clever use of WordNet hierarchy**: Leveraging WordNet's hypernym/hyponym structure to provide a prior on concept relationships for classification is a reasonable and implementable approach. The multi-label assignment (Section 3.2.2) — treating ancestor labels as additional supervision — is a clean formulation.

- **Comprehensive ablations**: Tables 3 and 4 provide useful diagnostics on the role of causal weight types (PSM, DRL, One, Random, Zero, Learn) and design components (intervention training, learnable scaled weights). The statistical testing (p-values, standard deviations) is a positive.

- **Test-time intervention experiment is creative**: Figure 4, despite its limitations (discussed below), demonstrates an interesting property — that incremental oracle concept information systematically improves classification accuracy — which is a meaningful capability to study.

## Weaknesses

### Major:

- **Causal framing is substantially overstated**: The paper repeatedly claims "unbiased causal effect estimation" and that edges represent "causal relationships," but the NCG graph is constructed from WordNet's IS-A/taxonomic relations (Section 3.2.1). That "canine" is a hypernym of "dog" does not mean "canine" *causes* "dog" in any SCM sense. The paper never justifies this conflation of ontological hierarchy with causal mechanism. This is not merely a terminological issue: the entire PSM/DRL machinery (Sections 3.2.3–3.2.4) is justified via do-calculus and backdoor adjustment on a graph that is assumed causal, but the backdoor identification assumptions (ignorability, positivity) are neither discussed nor verified. The propensity score $L(X)$ is introduced to reduce dimensionality (Eq. 2), but the paper does not argue that $L$ satisfies the conditional independence required for the backdoor criterion. Without these assumptions, the "causal effects" stored in $A_{ij}$ are association measures, and do-calculus-based reasoning and interpretability claims do not hold as presented. The paper would be stronger if it framed the contribution as structured hierarchical classification with principled edge weight estimation, rather than making unjustified causal claims.

- **Missing critical baselines for isolating the contribution**: The paper compares NCG against multi-class, multi-label, and a basic CBM, but does not compare against methods that also leverage label hierarchies — such as hierarchical softmax, hierarchical classification methods (e.g., DAGGER, HRN), or GNN-based label-relation models (e.g., ML-GCN). This matters because the improvements over multi-class/multi-label could be entirely explained by the hierarchical multi-label supervision + graph structure (which is well-known to help), rather than by the causal weight estimation. Table 3 shows that "Zero" and "One" weights perform surprisingly well (e.g., CLIP + Zero = 93.69 vs. CLIP + PSM = 94.22 on Bird Acc), suggesting much of the benefit comes from graph structure alone. Without a proper hierarchical classification baseline, it is impossible to attribute gains to causality specifically.

- **The "95% imageNet" claim is misleading without caveats**: The abstract prominently states NCG "achieves nearly 95% top-1 accuracy on the ImageNet dataset by employing a test-time intervention method." Section 4.5 clarifies that this uses ground-truth labels for intervened prior concepts, which is effectively providing the model with privileged hierarchical information at test time. This is not a standard classification metric and should not be presented as a headline result without immediate qualification. The paper also does not compare against simpler alternatives given the same privileged information (e.g., label propagation on the WordNet graph, or hierarchical soft labels), making it impossible to assess whether NCG's causal structure adds value beyond having correct ancestor labels.

### Minor:

- **Limited evaluation diversity**: Both datasets (Bird and ImageNet) share the same domain (natural images) and the same taxonomy source (ImageNet's WordNet-derived labels). The Bird dataset is small (only 450 test samples) and is a subset of ImageNet. No evaluation on datasets with different label structures or domain-specific ontologies (e.g., CUB with attribute annotations, iNaturalist, medical imaging) is provided, limiting confidence in generalizability.

- **No qualitative analysis of learned causal graphs or concept reasoning**: Despite claiming interpretability and intervenability, the paper provides no visualization of the NCG's estimated causal weights, no case studies of concept-level reasoning chains, and no analysis of whether the learned edge weights are semantically meaningful. This is a significant gap for a paper whose central claim is interpretability.

- **Linear SCM assumption vs. non-linear reasoning**: Section 3.2.3 explicitly assumes a linear structural equation model to justify PSM/DRL, but the actual reasoning function (Section 3.3.2) uses multi-head attention and MLPs (for ImageNet, a 3-layer perceptron with tanh activation). The mismatch between the linear assumption for weight estimation and the non-linear implementation is not discussed.

### Trivial:

- The intervention rate (p=0.15) and confidence value (v=5) in Section 3.3.4 are set without exploration of their sensitivity.

## Nice-to-Haves

- Evaluation under noisy or adversarial test-time interventions (not just oracle ground-truth), which would demonstrate the practical utility of the interventable claim.
- Comparison with hierarchical classification methods and GNN-based label models to isolate the contribution of causal weight estimation vs. graph structure.
- Visualization of estimated edge weights to assess whether they encode semantically meaningful relationships.
- Datasets beyond natural images and WordNet-derived labels.

## Removed Points

- **"The confounding argument for propensity scores is unmotivated because X is observed"**: The paper does address this — Section 3.2.3 explicitly discusses X as a confounder and the difficulty of conditioning on high-dimensional X directly, motivating the propensity score dimensionality reduction. Whether one agrees with the causal framing or not, the paper does argue for why PSM is used rather than direct conditioning. This is a methodological choice, not an unsupported omission.

- **"Reproducibility concerns about PSM/DRL implementation details deferred to appendix"**: Nitpicking about implementation details being in the appendix rather than main text is not a substantive weakness. The appendix exists and is cited.

- **"Only two datasets are used, both derived from ImageNet"**: This is retained as a minor point (limited evaluation diversity), but the harsh critic's demand for OOD/robustness evaluation is scope creep — the paper's stated scope is interpretable classification on structured label spaces, not domain generalization.

- **"CBM implementation is under-specified"**: The paper describes CBM as predicting prior concepts through a bottleneck and then using those to predict labels, which is standard. The comparison uses the same backbone and supervision, making this a reasonable baseline. This is not a fatal issue.

- **"The exogenous noise U_Cj ~ N(0,1) is not studied in ablations"**: This is a minor implementation detail; the noise adds stochasticity during training. Lack of an ablation on this is a nice-to-have, not a substantive weakness.

## Novel Insights

The paper surfaces an interesting empirical finding that has not been well-studied: when you provide hierarchical label supervision through a concept graph during training (even with uniform or zero weights), classification performance improves over flat multi-class prediction. This suggests that the structured label prior itself carries significant value, and the marginal benefit of carefully estimated "causal" weights (PSM/DRL over One/Zero) is relatively small. This aligns with findings in hierarchical classification literature but casts doubt on whether the causal framing adds substantial value beyond the hierarchical structure. The intervention training method is also a creative form of structured data augmentation that deserves more analysis as augmentation rather than causal reasoning per se.

## Suggestions

- **Reframe the contribution**: Present NCG as a hierarchical concept graph method that leverages WordNet structure and principled statistical weight estimation, rather than claiming "unbiased causal effect estimation." This would better match what the method actually delivers and would still be a valuable contribution.
- **Add hierarchical classification baselines**: Compare NCG against at least one hierarchical classification or GNN-based label relation method to properly isolate the contribution of the causal weight estimation.
- **Contextualize the 95% intervention accuracy**: Clearly state in the abstract and results that this uses ground-truth concept labels, and provide analysis of how much information the interventions are providing (e.g., what fraction of prior nodes are needed to reach 95%, and how semantically close are they to the target label).

## Calibration

I compared this paper against:
- **Causal CGMs** (lmKJ1b6PaL.md): Accepted poster, scores 6,6,8,6,8 (avg 6.8). Similar domain (causal concept graphs for interpretable models). Causal CGMs had stronger theoretical grounding with proper SCM definitions, broader evaluation with proper intervention experiments, but also faced novelty concerns (similarity to CausalVAE). NCG has weaker causal justification (WordNet ≠ causal graph) and narrower evaluation.
- **Energy-Based CBMs** (I1quoTXZzc.md): Accepted poster, scores 6,8,6,6,6 (avg 6.4). More novel architecture with energy-based concept interaction. Better evaluation across multiple datasets.
- **Beyond CBMs: Intervenability** (5oJlyJXUxK.md): Rejected, scores 5,8,8,5 (avg 6.5 but rejected). Similar intervenability focus but lacked baselines and had limited practical improvements over CBMs.
- **SupCBM / Hierarchical CBMs** (Q9Z0c1Rb5i.md): Rejected, scores 3,6,5,6 (avg 5). Shared issues with NCG around unclear causal/conceptual framing and limited novelty.

NCG is below Causal CGMs and ECBMs (which were accepted with scores ~6.4-6.8) due to: (1) significantly weaker causal justification, (2) narrower evaluation, (3) missing key baselines, and (4) the misleading 95% claim. It is comparable to or slightly above SupCBM (rejected, avg 5) in terms of technical execution but shares similar conceptual weaknesses. The "Beyond CBMs" paper (rejected, avg 6.5) had more focused methodology but insufficient baselines — NCG has a similar problem. Overall, NCG is in the range of rejected CBM extension papers, roughly corresponding to a score of 4-5.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>