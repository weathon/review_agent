## Summary

The paper introduces Neural Causal Graph (NCG), a classification framework that constructs a directed concept graph from WordNet, estimates edge weights using propensity-score-based causal inference methods (PSM/DRL) to deconfound from input features, and performs message-passing reasoning over the graph. An intervention training method simulate do-operations during training to enable test-time concept interventions. Experiments on ImageNet and a self-collected Bird dataset show accuracy gains over multi-class, multi-label, and CBM baselines, and near-95% top-1 ImageNet accuracy when oracle concept labels are injected at test time.

## Strengths

- **Novel integration of causal inference with neural classification.** The paper proposes a creative architecture that bridges SCM/POM frameworks with deep learning for structured concept reasoning. Using propensity scores to deconfound edge estimation from input features (Eqs. 1–2) is an interesting idea even if its causal interpretation is debatable, and the intervention training mechanism (Sec. 3.3.4) is a practical contribution for enabling test-time interaction.

- **Clean, pluggable architecture.** The concept proposer + DAG message-passing design is modular and can be applied to different backbones (ResNet50, CLIP), which the authors demonstrate. The framework is well-structured and clearly presented.

- **Consistent empirical improvements.** NCG outperforms multi-class, multi-label, and CBM baselines across both datasets and both backbones in Table 2, with statistically significant margins (p < 0.05) in most configurations. The ablation studies (Tables 3–4) provide useful insights into the contribution of causal weights and intervention training.

- **Test-time intervention experiment.** Figure 4 demonstrates that oracle concept interventions can boost accuracy to ~95%, and the ablation showing that intervention training is necessary for models to benefit from interventions is informative.

## Weaknesses

### Fatal
None.

### Major

- **The "causal" framing is overclaimed — the graph and edge weights are not causal in the Pearl/Rubin sense.** The graph is extracted from WordNet hypernyms (Sec. 3.2.1), which encode taxonomic ("is-a") relations, not causal mechanisms. Calling these "causal edges" and labeling PSM/DRL estimates as "unbiased causal effects" (Sec. 3.2.4) stretches terminology: these are debiased association weights on a semantic hierarchy, not interventional quantities. The PSM/DRL estimation assumes unconfoundedness — that conditional on propensity score L(X), treatment assignment is independent of potential outcomes — which is unverifiable here since "treatment" (concept node activation) is a deterministic function of the same input X used to compute L. The method is a reasonable heuristic for edge weight estimation on a concept graph, but the causal inference machinery (do-calculus, backdoor criterion, ATE) is invoked without satisfying its required assumptions. This overclaim is significant because it constitutes the paper's central novelty卖点.

- **The ~95% intervention accuracy relies on oracle ground-truth concept labels, inflating the intervenability claim.** Section 4.5 describes intervening on prior concepts "using the corresponding labels" — i.e., feeding the model correct concept states. This is equivalent to revealing partial ground truth to the model at test time, which would boost any hierarchical classifier. The abstract's claim of "nearly 95% top-1 accuracy on ImageNet by employing a test-time intervention method" omits this critical caveat. There is no evaluation with noisy, partial, or incorrect interventions, which would be necessary to demonstrate meaningful human-AI collaboration. Without this, the intervenability contribution is essentially showing that "giving a model correct answers for sub-tasks improves the final answer."

- **Missing comparison with hierarchical classification methods.** NCG's core mechanism — propagating information over a label taxonomy — is the defining feature of hierarchical classification. Yet no such baselines are included (e.g., hierarchical softmax, DAGGER, label-graph GNNs). Table 3 shows that learned weights ("Learn") perform comparably to PSM/DRL on Bird with CLIP (94.18 vs. 94.49), suggesting the graph structure and multi-label training may be doing most of the work, not the "causal" estimation. Comparing against hierarchical classification methods would clarify whether the PSM/DRL estimation specifically adds value beyond simply using the graph structure.

- **Modest accuracy improvements on ImageNet relative to framework complexity.** On ImageNet, NCG improves over standard multi-class classification by +0.67% (ResNet50) and +0.95% (CLIP) — small margins given the added complexity of graph construction, causal effect estimation, and intervention training. The training regime (3 epochs, no data augmentation, frozen backbone) is also non-standard, making it unclear how NCG would fare in a more typical fine-tuning setup.

### Minor

- **No evaluation of interpretability.** The title and abstract emphasize "interpretable" classification, but the paper provides no user study, interpretability metrics, or even qualitative examples showing that NCG's concept predictions are human-meaningful. The interpretability claim rests entirely on the structure having concept-named nodes, which is not sufficient.

- **The Bird dataset is small and unconventional.** With only 450 test samples and 16 prior concepts, Bird is limited for reliably evaluating the framework. Standard concept-based benchmarks (CUB-200-2011, AwA2) widely used in the CBM literature are not included.

- **Connection between intervention training and formal do-calculus is motivational, not rigorous.** The intervention training (Sec. 3.3.4) replaces logits with ±5 scaled ground-truth values at a 15% rate — inspired by masked language modeling rather than derived from the SCM framework. The claim that this "simulates the effects of the do(·) operator" is an analogy, not a formal equivalence.

### Trivial
- The hyperparameters p=0.15 and v=5 for intervention training are set empirically without sensitivity analysis.

## Nice-to-Haves

- Evaluate with noisy or partially incorrect interventions to demonstrate practical intervenability.
- Add a comparison with at least one hierarchical classification or label-graph GNN method.
- Analyze which parts of the concept graph contribute most to the accuracy gains.
- Visualize estimated edge weights overlaid on WordNet structure to show what PSM/DRL learn beyond topology.

## Removed Points

- **Availability/reproducibility of Bird dataset and OpenCLIP model.** The paper cites a GitHub repository and OpenCLIP-H-14; these are assumed to exist and be available per the hard rules.
- **Formatting/style nitpicks.** Several reviewers noted presentation issues; these are removed per the rules.
- **Demand for larger datasets or more models.** ImageNet is already a standard large-scale benchmark; requesting additional datasets is a generic demand outside the paper's scope, though evaluating on standard CBM benchmarks would strengthen the contribution.
- **Criticism that CBM comparison is unfair to baselines because CBM is stronger but performs worse.** The paper shows CBM performs worse than multi-class in some settings (e.g., Bird+CLIP: 90.67% vs. 93.33%), meaning the comparison favors baselines, not NCG. Per hard rules, this criticism is removed.
- **Unfamiliarity with WordNet as a resource.** WordNet (Miller, 1995) is a well-established lexical database; its use here is appropriate.

## Novel Insights

The paper identifies a genuine and underexplored problem: making neural classifiers amenable to interactive, concept-level intervention. The architecture of embedding a structured concept graph into the classification pipeline and using propensity-score-debiased weights is an interesting design. However, the core insight — that a taxonomic hierarchy with debiased edge weights can improve classification — is better characterized as structured prediction than causal reasoning. The gap between the method's contributions (a well-designed hierarchical classification head with intervention capability) and its claims (unbiased causal effect estimation, causal graph reasoning) is the paper's central tension.

## Suggestions

- Reframe the contribution honestly: describe NCG as a concept-graph-structured classification framework with debiased edge estimation and intervention capability, rather than as a causal model. The method is interesting without overclaiming causality.
- Add a simple hierarchical classification baseline (even label smoothing with the WordNet hierarchy) to isolate the contribution of the causal weight estimation from the contribution of the graph structure itself.
- Evaluate with partially incorrect interventions (e.g., flip 10–30% of prior concept labels) to demonstrate that intervenability works under realistic noise, not just in the oracle setting.

## Score and Decision

**Calibration comparison:**
- "Beyond Concept Bottleneck Models" (5oJlyJXUxK): Similar intervenability theme, similar issues with practical evaluation of interventions — scored 5/8/8/5 (Reject).
- "Boosting CBM with Hierarchical Concepts" (Q9Z0c1Rb5i): Similar reliance on external concept structure, limited baselines — scored 3/6/5/6 (Reject).
- "Concepts' Information Bottleneck Models" (2xRTdzmQ6C): Marginal improvements, intervention concerns — scored 6/6/1/6/3 (Reject).
- "Causal Representation Learning" (OatZMyMuIo): Overclaimed causal structure, limited baselines — scored 5/5/3/3 (Reject).

This paper shares core weaknesses with these rejected papers: the causal framing is overstated relative to what the method delivers, the practical intervenability claim is inflated by oracle evaluation, and missing baselines make it hard to attribute gains. It does have cleaner architecture and more thorough ablations than some, but the modest ImageNet improvements and the overclaimed causal interpretation place it below the threshold. Relative to papers scored 5 (borderline), this paper has similar methodological overclaims but somewhat better empirical work; relative to papers scored 3 (weak), it has more substance. I place it at 4.0.

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>