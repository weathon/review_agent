Now let me check the calibration papers referenced by the human reviewer.Now I have enough information to produce the consolidated review. Let me synthesize.---

## Summary

The paper introduces the Neural Causal Graph (NCG), a classification framework that extracts a concept hierarchy from WordNet, estimates edge weights as average treatment effects (ATE) between concept nodes using propensity score matching (PSM) and doubly robust learning (DRL), and augments training with an intervention scheme to support human-guided test-time concept corrections. Experiments on ImageNet and a self-collected Bird dataset show modest but statistically significant improvements over multi-class, multi-label, and CBM baselines, plus a test-time intervention curve that reaches ~95% accuracy when oracle concept labels are incrementally provided.

---

## Strengths

- **Principled causal estimation machinery.** The reduction from backdoor adjustment over high-dimensional X (Eq. 1) to a propensity-score scalar L (Eq. 2), connecting the do-calculus (SCM) and POM frameworks, is technically coherent and clearly presented.
- **Effective intervention training.** Section 3.3.4 proposes a practical way to align train/test distributions under interventions by fixing concept logits using ground-truth multi-label values; Table 4 shows this drives the largest performance gains and is the most substantive novel mechanism in the paper.
- **Comprehensive ablations.** Table 3 systematically evaluates six edge-weight strategies (PSM, DRL, One, Random, Zero, Learn), and Table 4 separately ablates intervention training and learnable scaled weights, providing genuine insight into each component's contribution.
- **Statistical rigor.** Results include standard deviations and two-sample t-tests over repeated runs, and most gains reach p < 0.05.
- **Two-backbone evaluation.** Testing on both ResNet50 and CLIP-ViT-H gives a fair picture of how the framework generalizes across encoder strengths.

---

## Weaknesses

### Fatal
*(No single weakness rises to the "not even a paper" level — the system works and produces reproducible results — but the two issues below, taken together, substantially undermine the core claims.)*

### Major

- **WordNet taxonomy ≠ causal graph.** The entire "causal" framing rests on treating WordNet is-a (hypernym) relations as directed causal edges (Section 3.2.1: *"we retain the corresponding directed edges in WordNet as the node relationships of NCG"*). Lexical hypernymy encodes semantic subsumption, not mechanism: "animal" does not *cause* "bird" in any interventional sense. The paper never justifies this identification, and the standard backdoor-adjustment derivation (Eq. 1–2) cannot convert a taxonomic hierarchy into an identified causal effect — it requires substantive causal assumptions about the data-generating process that are neither stated nor testable here. As a result, calling the ATE estimates "causal effects" and framing the whole pipeline around do-calculus is conceptually misleading throughout. The contribution is more accurately described as: hierarchical concept graph classification with taxonomy-derived structure and ATE-weighted edges.

- **95% ImageNet claim is misleading.** The abstract states *"NCG achieves nearly 95% top-1 accuracy on the ImageNet dataset by employing a test-time intervention method"* without qualification. Section 4.5 reveals that this figure is obtained by intervening on up to 1,357 prior concept nodes *"using the corresponding labels"* — i.e., ground-truth concept information derived from the true class is fed to the model at test time. This is an oracle-assisted regime, not standard classification. The result measures how well the model exploits leaked label information through the concept graph, not autonomous recognition ability. Placing this number in the abstract as a classification achievement is materially misleading.

- **Gains do not support the claimed "robustness" contribution.** The abstract and introduction claim NCG provides *"enhanced robustness"* (Introduction, Conclusions), but no robustness benchmark, out-of-distribution evaluation, or adversarial test is ever conducted. The standard classification improvements on ImageNet (+0.67–0.95% over the multi-class baseline, Table 2) are modest and statistically significant but do not constitute evidence for robustness. The paper over-claims on this dimension.

- **Ablation does not isolate causal estimation from graph structure.** Table 3 shows that Zero-weight edges (which propagate no causal information) outperform One-weight edges and come close to PSM with CLIP (Zero: 93.69 vs. PSM: 94.22 vs. DRL: 94.49). Table 4 shows intervention training alone accounts for the majority of the performance gain (e.g., DRL+ResNet50 with IT only: 92.09 vs. without either: 90.09). This means a non-causal hierarchical graph baseline with the same multi-label supervision and intervention training would likely capture most of the gain, but no such baseline is tested. The paper cannot attribute the improvements specifically to unbiased causal weight estimation rather than to (a) hierarchical multi-label supervision and (b) intervention training.

### Minor

- **Linear SEM assumption is not maintained.** Section 3.2.3 asserts *"the neural causal graph can be defined as a linear structural equation model,"* which grounds the backdoor-adjustment derivation, but Section 3.3.2 and 4.1 use tanh-activated 3-layer MLPs as the update function φ on ImageNet. The paper does not acknowledge or justify this inconsistency; the linear-SEM framing is used as motivation but abandoned in practice.

- **No evaluation on standard concept-annotated benchmarks.** The interpretability claims cannot be verified because neither ImageNet (no per-sample concept annotations) nor Bird (450 test samples, self-collected) provides ground-truth concept labels for evaluation. Standard concept-based benchmarks (CUB-200, AwA2) are standard in this line of work and would allow direct validation of concept accuracy and intervention effectiveness against known ground truth.

- **Intervention experiment does not test realistic human input.** The test-time intervention in Section 4.5 assumes a perfectly informed user who provides exactly the correct ancestor concept labels. Real human collaborators will sometimes intervene with incorrect values. Without any experiment studying noisy or partial interventions, the practical "human-AI interaction" claim is unsubstantiated.

### Trivial

- The exogenous noise term U_{C_j} ~ N(0,1) in Eq. (3) is introduced with no ablation and no explanation of its effect during inference; its role is unclear.
- The five-hop truncation rule for graph construction (Section 3.2.1) is unexplored by sensitivity analysis; whether the prior concept set size significantly affects performance is unknown.

---

## Nice-to-Haves

- Sensitivity analysis over intervention hyperparameters (rate p, confidence v) and graph construction parameters (hop limit), to demonstrate robustness of design choices.
- Visualization of learned causal weights (PSM/DRL) vs. uniform weights on the concept graph, to show whether the estimated ATEs carry meaningful semantic signal.
- Visualization of per-sample intervention traces — showing which concept logits shift and how the prediction changes — to give qualitative evidence of causal reasoning rather than mere pattern matching.
- Experiments on standard concept-annotated benchmarks (CUB-200, AwA2) with per-sample concept labels to enable rigorous evaluation of concept accuracy.
- Computational overhead analysis comparing NCG to simpler hierarchical baselines.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic: Positivity/overlap issues for PSM/DRL on label-determined concepts.** While this is a theoretically valid concern (when treatment assignment is near-deterministic, PSM can be unstable), the paper is doing this in a practical ML estimation context and the tables show stable empirical results. The critic did not show that the estimates actually collapse or produce degenerate values. Removed as unverified and too speculative given the empirical evidence of functioning results.

- **Harsh Critic: t-tests with only 3 runs on ImageNet are weak evidence.** While low-n t-tests are indeed conservative, this is a common computational constraint for large-scale experiments and is not atypical for the setting. The standard deviations are very small, lending additional confidence. Removed as a nitpick about methodology that does not undermine the claims.

- **Harsh Critic / Spark: Missing related works (hierarchical classifiers, DAG-Net, etc.).** Per hard rules, missing related works are not flagged as we cannot confirm their existence with external sources. Removed.

- **Harsh Critic: Sensitivity of 5-hop rule.** Potentially interesting but minor and not shown to be harmful; moved to nice-to-have.

- **Harsh Critic: The Gaussian exogenous noise U is under-specified.** Trivially addressable; no evidence it harms performance; already filed under Trivial above.

- **Human Finder: Scalability and computational complexity.** Valid as a practical question, but not a flaw in the core claims; moved to nice-to-have.

- **Human Finder: Human evaluation for interpretability.** A valid desideratum but not standard in classification papers of this type; filed as nice-to-have.

---

## Novel Insights

The most genuinely novel observation across reviewers is the three-way tension revealed by Table 3: Zero-weighted edges outperform One-weighted edges, yet PSM/DRL still outperform Zero — suggesting that the concept graph topology itself (structuring message-passing over the ancestor hierarchy) provides independent inductive bias beyond just switching edges on or off, while the calibrated ATE weights provide an additional, separable benefit. This implies that future work should separately study (a) graph structural bias from knowledge hierarchies and (b) data-driven weight estimation, rather than collapsing both into a single "causal" claim. The intervention training mechanism (Section 3.3.4) is the paper's most concrete and practically transferable contribution: it is a clean, generalizable technique for training models that can exploit expert-provided concept corrections at test time, independent of whether the underlying graph is truly causal.

---

## Suggestions

1. **Reframe the causal claim**: Replace "causal graph" and "causal effect" with "knowledge-structured concept graph" and "weighted hierarchical edge." Apply PSM/DRL as data-informed weight estimators (which is accurate), not as identifiers of causal effects (which is not established). This reframing preserves the technical contribution without the unjustified causal interpretation.
2. **Fix the abstract's 95% claim**: Clearly state in the abstract that the 95% figure is obtained under oracle-assisted concept intervention (ground-truth prior concept labels provided at test time), not under standard autonomous classification.
3. **Add a non-causal hierarchical baseline**: Include a GNN-based baseline over the same WordNet graph with uniform or learnable edge weights + the same multi-label supervision + the same intervention training. This would properly isolate the contribution of ATE-estimated weights.
4. **Evaluate on a concept-annotated dataset**: Validate concept accuracy and intervention quality on CUB-200 or AwA2 where ground-truth concept labels are available, to substantiate the interpretability claims.
5. **Test noisy interventions**: Run Section 4.5's experiment with 10–30% of intervention values flipped or randomly corrupted, to characterize robustness to imperfect human input.

---

## Score and Decision

**Calibration:**

- **Graph Concept Bottleneck Models** (`qPH7lAyQgV`): Scores 5/6/6/6, **Rejected**. Similar contribution (concept graph over a hierarchy for CBM), but with fewer conceptual integrity issues — mostly cited for marginal improvements and unclear design choices. NCG has a comparable contribution level but adds the misleading 95% claim and the undefended causal framing.
- **Generating Explanations From Linear SCMs** (`V42LZPlorE`): Scores 3/3/3/5/3, **Rejected**. A paper that similarly conflates a modeling convenience (linear SCM) with legitimate causal identification, with weak empirical validation. NCG has stronger empirical results but comparable conceptual issues.
- **Energy-Based CBMs** (`I1quoTXZzc`): Scores 6/8/6/6/6, **Accepted**. Clearly superior: well-scoped claims, rigorous evaluation on standard benchmarks, cleaner conceptual framing. NCG falls well short of this bar.
- **EE-CBM** (`42TXboDg3c`): Scores 5/3/5/5, **Rejected**. Comparable weakness level (overclaimed novelty, missing baselines, weak ablations) but NCG has more complete experiments and a more novel angle.

**Assessment:** NCG sits between Graph CBMs (5–6, rejected) and V42LZPlorE (3, rejected). The working system and statistically significant gains push it above V42LZPlorE. But the misleading 95% headline, the undefended causal interpretation of a lexical taxonomy, and the failure to isolate causal estimation from hierarchical multi-label supervision are more severe than the issues that sank Graph CBMs. The paper is better described as "a hierarchical concept-graph classifier with ATE-weighted edges and an intervention training mechanism" — a legitimate but more modest contribution than claimed. Overall: **4.5, Reject**.

**Originality:** Low-Moderate — the combination of PSM/DRL for graph weighting and intervention training is novel, but the causal framing borrows heavily from standard SCM language without the identification work.  
**Importance:** Moderate — human-guided test-time concept correction is a practically valuable direction.  
**Claims vs. Support:** Poor — the headline claim is misleading; the causal framing is unjustified; robustness is asserted but not tested.  
**Soundness:** Moderate — the technical steps within the stated assumptions are coherent, but the key assumption (taxonomy = causal graph) is indefensible.  
**Clarity:** Good — the writing is clear and the figures are informative.  
**Value to Community:** Low-Moderate — the intervention training trick is reusable; the causal framing as presented is a negative contribution in that it misrepresents what the method achieves.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>