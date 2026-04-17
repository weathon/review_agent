# Peeling Context from Cause for Multimodal Molecular Property Prediction

- Decision: Reject
- Scores: 4, 4, 4, 2

## Abstract
Deep models are used for molecular property prediction, yet they are often hard to interpret and may rely on spurious context rather than causal structure, degrading reliability under distribution shift and harming predictive performance. We introduce **CLaP**, *Causal Layerwise Peeling*, a framework that separates causal signal from context in a layerwise manner and integrates diverse graph representations of molecules. At each layer, a *causal block* performs a soft split into causal and context branches, fuses causal evidence across modalities, and *peels* batch-coupled context to concentrate on label-relevant structure, limiting shortcut signals and stabilizing layerwise refinement. Across nine molecular benchmarks, including OOD benchmarks, CLaP reliably reduces MAE and MSE relative to competitive baselines. We also obtain atom-level causal saliency maps that highlight substructures for a prediction, providing actionable guidance for targeted molecular edits. Case studies confirm the accuracy of these maps and their alignment with chemical intuition. By peeling context from cause at every layer, the model delivers predictors that are accurate and interpretable for molecular design.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper tackles spurious “context” in molecular property prediction and proposes CLaP, a layerwise framework to extract causal information from irrelevant contexts. At each depth, representations are split into a causal branch and a trivial (context) branch, and the causal path is trained to satisfy an increasing target for within-batch label correlation with a monotonicity regularizer. The trivial path fits the residuals. Multimodal fusion (2D graphs, 3D geometry, peptide strings) occurs on the causal path, yielding atom-level saliency. On ESOL, FreeSolv, Lipo, and CycPeptMPDB, CLaP improves the regression error compared to the baselines; ablations confirm each component’s contribution, and re-batching tests suggest robustness to batch artifacts.

### Strengths
- The authors successfully implement the idea of separating useful information from irrelevant background.

- By enforcing a gradually increasing correlation coefficient between intermediate-layer representations and the label, which enabled informative saliency maps.

### Weaknesses
- The empirical performance needs to be more convincing: the compared baseline models are largely outdated (pre-2022), and head-to-head evaluations against more recent methods [3][4] are necessary to validate the claimed gains. Moreover, reporting results on more tasks such as QM9 [1] and BACE [2], among others, would provide a more comprehensive assessment.

- The observed improvements may stem primarily from the introduction of multimodality rather than from the proposed causal-context peeling itself, and this confound should be disentangled.

- Since the proposed approach relies on optimizing the correlation coefficient with continuous label values, it appears to be difficult to apply directly to classification task settings.

### Questions
- Since Equation (7) already enforces deeper layers to achieve higher correlation with the ground-truth labels, isn’t $L_{mono}$ a redundant training objective? How do you explain the substantial performance gap observed in Table 4 when $L_{mono}$ is included versus omitted?

- If the trivial branch is intended to capture information deemed irrelevant and therefore “peeled” away, why are these trivial signals accumulated and reused in the final prediction? What theoretical or empirical justification supports re-introducing them at the output?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Causal Layerwise Peeling, a framework for multimodal molecular property prediction that explicitly separates causal signals from contextual noise in a layerwise manner. CLaP introduces: 1) a causal–trivial split at each layer, 2) a batch-wise invariance principle to isolate sample-intrinsic signals, 3) a depth-wise correlation schedule and monotonicity regularizer, and 4) multimodal fusion across 2D SMILES graphs, 3D geometries, and peptide notation. Empirical results on four molecular benchmarks show consistent gains over strong baselines. The model also provides atom-level causal saliency maps aligned with chemical intuition.

### Strengths
1. The proposed method effectively distinguishes causal and contextual information structurally, overcoming limitations of traditional single-stage causal subgraph selection methods. This approach is conceptually novel and offers strong interpretability. The authors also provide formal derivations and theoretical proofs, which elucidate the model's rationale for progressively removing contextual noise.
2. The ablation studies are thorough, systematically validating the impact of key modules (causal-trivial branch, correlation scheduling, monotonicity regularization). This forms a logically complete validation loop.
3. The work introduces multimodal fusion into causal representation learning. By adaptively gating and fusing 2D, 3D, and sequential modalities, the model enhances robustness and cross-modal consistency.

### Weaknesses
1. The comparative experiments are not sufficiently comprehensive. The authors have only evaluated on four datasets, which are limited in size. We recommend augmenting the comparison with experiments on the GOOD benchmark. Additionally, the causality-oriented baselines are only compared against OOD methods, without inclusion of molecular causality-oriented baselines[1]. Please include relevant comparative experiments.
2. Causal invariance is validated solely through "re-batching" experiments. This needs to be compared with other OOD settings to further justify the chosen approach. The assumption of independent and identically distributed mini-batch sampling may not hold in practice for chemical data, where clustering or scaffold biases are common. Such biases may not adequately represent real-world distribution shifts (e.g., scaffold splits or property distribution shifts).
3. Although the paper claims performance improvements from multimodal fusion, it lacks an analysis of robustness under conditions where one or more modalities are missing or noisy.

Ref: 

[1] Learning Substructure Invariance for Out-of-Distribution Molecular

### Questions
1. Given the limited number of benchmarks used in this work, what are the training costs and convergence stability of the model when trained on larger datasets or more complex molecular systems?
2. If all modalities contain significant batch-dependent noise, will the proposed causal-trivial partition still converge effectively?
3. Could you report results on OOD settings (e.g., scaffold-based splits) to further validate the causal generalization capabilities of your model?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces CLaP (Causal Layerwise Peeling), a novel multimodal framework for molecular property prediction. Its primary objective is to explicitly disentangle the causal signal—the intrinsic molecular substructures driving a property—from spurious contextual shortcuts often learned by deep models, thereby enhancing reliability, especially under distribution shift. CLaP is built upon a batch-wise invariance principle, treating the natural feature fluctuations across mini-batches during training as contextual variation that the true causal features must ignore. The framework employs multimodal fusion, integrating three distinct molecular representations (2D topology, HELM notation, and 3D geometry) to leverage complementary structural information.

### Strengths
The framework effectively utilizes the principle of batch invariance, where the causal features ($Z_c$) produced by a molecule ($X$) must be able to stably predict the label ($Y$), regardless of which batch ($B$) the molecule is randomly assigned to. This constraint forces the model to shift its attention from batch-coupled spurious correlations (i.e., shortcut signals) to the sample-intrinsic, true causal structure.

### Weaknesses
The central weakness lies in the potential for confounding variables in the performance comparison: the reported performance gains might be primarily attributable to the richness of the input data (using three modalities: 2D, HELM, and 3D) rather than the effectiveness of the proposed Causal Layerwise Peeling (CLaP) mechanism itself. I think the authors must perform a detailed ablation study on the input modalities to cleanly isolate the contribution of the CLaP architecture.

### Questions
The comparison models largely use only 2D information, while CLaP uses 2D, HELM, and 3D modalities. Have the authors performed a direct comparison between CLaP and the strongest baseline, where both models use the exact same set of input modalities (e.g., 2D + 3D)?

The capacity of the trivial branch is crucial for soaking up context without contaminating the causal path. Could the authors have performed a sensitivity analysis showing how performance changes when the trivial branch capacity (e.g., number of layers or width) is varied?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes CLaP, a framework for interpretable regression in molecular property prediction for small molecules and cyclic peptides. The core idea is to split the representation at every layer into a “causal branch” and a “trivial branch,” and to progressively peel away batch-specific contextual signal so that only signal that is stably aligned with the target property remains. Each layer produces a causal readout that is encouraged to correlate more strongly with the target in deeper layers, while the trivial branch absorbs residual variation that is considered context-dependent rather than property-driven. The model integrates multiple types of molecular information (2D connectivity, peptide-like sequence descriptions, and 3D structure) and is trained to emphasize features that remain predictive across different batch conditions. CLaP is evaluated on four benchmark regression tasks and outperforms strong baselines. In addition, CLaP produces atom- and substructure-level saliency maps indicating which parts of a molecule are considered causal drivers of the predicted property, and these highlighted regions are reported to align with established chemical intuition, suggesting possible guidance for targeted molecular modification.

### Strengths
1. This work proposes a progressive, layerwise separation of causal and contextual signals. Two parallel branches are trained end to end, with their alignment to the target strengthening at deeper layers. The resulting “peeling” design contrasts clearly with invariant-subgraph heuristics while remaining easy to optimize.

2. The advantages of the model framework are demonstrated through evidences on both small molecules and cyclic peptides, which indicate transferability beyond a single regime.

3. Outputs are interpretable and chemically plausible in  atom- and substructure-level saliency maps.

### Weaknesses
1. The notion of “causal signal” is still operational rather than experimentally causal. The method defines causal features as those that are stable across batches and aligned with the target, and treats the rest as contextual noise. This is practical, but it is not yet demonstrated through true evidence such as editing chemistry and/or observing measured changes.

2. The proposed method relies on manually choosing a layer depth threshold 𝐿 that determines where causal “peeling” is applied. This threshold may need to be re-tuned for different model architectures and datasets, suggesting that the causal effect has a strong dependence on heuristic, experience-driven selection. In addition, training epoch progression appears to affect when the “peeled” representation emerges and stabilizes, which could further increase the sensitivity and difficulty of selecting 𝐿.

3. The interpretability results are mostly qualitative. The paper shows atom- and group-level saliency maps that align with chemical intuition, but does not yet provide a quantitative analysis where suggested edits lead to predicted directional changes in the property.

4. The paper states that the framework can be applied at every layer of the model and incorporates multi-modal gating. However, inserting this additional causal/trivial decomposition and gating at all layers may introduce nontrivial computational and memory overhead. The authors do not report these overheads or compare them to a baseline, so it is difficult to assess the scalability of the approach in larger models or fully multi-modal settings.

5. Although there is no citation for the data set, it looks the small molecular sets are from MoleculeNet, which has a lots of bias discussed in the literature.

6. The paper lack enough related work citations. the citations are also out of date, most of which are from the published work before 2022.

### Questions
1. How are the 3D structures obtained or selected, especially for flexible systems? If multiple conformations are possible, does the method fix one, or average over them? How sensitive is the model to noisy or low-quality 3D input, and have you compared single-modality vs multi-modality specifically on the harder peptide dataset?

2. On batch-wise invariance assumptions: The method relies on the idea that stable, property-driven signal should persist across different batch compositions, while batch-specific co-occurrences are filtered out. In practice, batching can itself introduce structure (for example, similar scaffolds ending up together). Have you examined whether the “trivial” branch is in fact capturing batch-level statistics such as scaffold frequency or label distribution, as opposed to chemically meaningful signal?

3. Corresponding to the previously mentioned weakness #2, do different model architectures or datasets require different choices of 𝐿? Is there any principled way to determine 𝐿, rather than relying on researcher intuition? In addition, the paper suggests that the “peeled” representation emerges and stabilizes over training epochs. Do you observe that the effective peeling depth drifts over training, i.e. that the same nominal 𝐿 corresponds to different behavior at different epochs? If so, does that imply that choosing 𝐿 depends not only on architecture/data but also on training stage?

### Soundness
2

### Presentation
2

### Contribution
3
