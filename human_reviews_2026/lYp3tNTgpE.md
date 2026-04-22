# RAPTORGraph: Graph-Based Pathway Modeling for Causal Discovery in Single-Cell Perturbations

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Experiments involving the perturbation of individual cells are central to understanding cellular mechanisms and can accelerate drug discovery.
Causal representation learning (CRL) allows us to uncover the latent factors that regulate biological systems and predict the impact of novel perturbations.
Unfortunately, existing methods fail to address intervention spillover in a closed-world setting where intervention targets are known a priori, such as in Perturb-seq experiments, due to their reliance on dense encoders.
Furthermore, incorporating curated biological pathways into the model imposes a confirmatory bias, forcing it to explain the data through preexisting pathways and reducing the set of hypotheses the model can explore, while discarding novel signals that lie outside the annotated pathways.
In this work, we introduce RAPTORGraph, a $\beta$-VAE with a GraphPathway encoder that explicitly models complex gene-to-gene interactions within learned pathways.
Moreover, our model's preconditioning isolates the influence of perturbed genes, yielding clean, single-node latent interventions required for identifiable causal discovery and eliminating spillover.
Finally, we train the model on data preprocessed with optimal-transport alignment, which guarantees a well-defined mapping between control and perturbed samples and further stabilizes the learned latent representations.
We demonstrate that RAPTORGraph improves state-of-the-art performance on downstream analyses of unseen perturbations, such as non-additive interactions, while outperforming other approaches on objective metrics, such as MSE and MK-MMD.
The code will be made publicly available upon publication of this paper.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces **RAPTORGraph**, a causal and interpretable framework for predicting cellular responses to genetic perturbations in single-cell data (e.g., Perturb-seq). The model combines a *preconditioned GraphPathway encoder* that enforces one-to-one mappings between perturbed genes and latent “meta-pathways,” a **DAGMA-based causal graph layer** to infer pathway interactions, and **optimal transport preprocessing** to mitigate the mean-collapse problem arising from unpaired single-cell measurements. The authors claim that this architectural design addresses two key issues in causal representation learning for perturbation data: (i) *intervention spillover* caused by dense encoders and (ii) *mean collapse* due to random pairing. Experiments on the Norman et al. (2019) Perturb-seq dataset show a strong overall balance between reconstruction quality, distributional fidelity (MK-MMD), and prediction of non-additive gene interactions compared to several recent baselines (dVAE, SENA, scGPT). The model also includes a reverse-perturbation analysis, identifying the genetic perturbations responsible for a given cellular gene expression profile. The paper further provides theoretical justification of identifiability and includes ablation analyses for its core components.

### Strengths
- **Clear motivation and well-defined problems.** The paper identifies two concrete issues (intervention spillover and mean collapse) and ties them to theoretical limitations of current causal representation learning frameworks.
- **Architectural novelty.** The preconditioned block-diagonal GraphPathway encoder is an elegant way to enforce sparse, single-node latent interventions, bringing the implementation closer to the identifiability assumptions in CRL theory. Similar ideas of sparse Jacobian or disentangled representations exist, but their adaptation to pathway-based causal inference appears new and well-motivated.
- **End-to-end causal discovery.** The encoder and DAGMA layers are trained jointly in an end-to-end fashion, which is a good design choice as it reduces human bias in predefined causal structures and enables the model to learn causal relations directly from data.
- **Solid empirical evaluation.** The paper benchmarks against strong baselines on a widely used dataset with multiple complementary metrics (MSE, MK-MMD, Precision@10, Hit Rate@K), showing balanced performance across tasks.
- **Biological relevance.** The framework aligns well with real-world perturbation experiments and is potentially valuable for hypothesis generation in drug discovery.

### Weaknesses
- **Limited dataset diversity.** All evaluations are performed on a single dataset (Norman-CPA). Although justified for compatibility, this limits claims of generalization across cell types or experimental protocols.
- **Potential rigidity of preconditioning.** The fixed block-diagonal structure may restrict the model’s flexibility in cases where gene–pathway relationships are overlapping, context-dependent, or nonlinear. Literature on over-constrained causal structures suggests that such rigidity can sometimes bias downstream inference in systems with pleiotropy or shared regulators.
- **DAGMA interpretability.** While DAGMA ensures an acyclic causal graph, it does not guarantee biological validity. The paper provides no examples linking learned edges to known gene interactions, so it remains unclear whether the resulting DAG corresponds to meaningful pathways. Moreover, even a DAG satisfying acyclicity may not represent true causality if hidden confounders or feedback loops exist.
- **Ablations and sensitivity.** The contribution of each component (OT preprocessing, DAGMA, preconditioning) could be evaluated more systematically. For example, how much does OT pairing alone improve results relative to random pairing?

### Questions
1. Since the authors state that the encoder and DAGMA modules are trained end-to-end, could they elaborate on how they manage the trade-off between enforcing acyclicity and maintaining reconstruction fidelity? For example, how sensitive is performance to the weighting of the DAGMA loss term?
2. How robust is the preconditioning approach when gene–pathway mappings are not clearly one-to-one? Could the method handle partial overlaps or shared pathway membership?
3. Have the authors verified whether the learned causal edges correspond to known gene–gene or pathway–pathway interactions? If not, how should readers interpret the learned DAG biologically?
5. Would the OT pairing approach generalize to other datasets where the control–perturbation relationship is less structured or contains additional confounders?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents RAPTORGraph, a β-VAE–based causal representation learning framework designed for interpretable modeling of single-cell perturbation responses. The authors identify two fundamental issues in current causal generative models:
Intervention spillover, where dense encoders cause perturbations to affect all latent variables, breaking causal identifiability assumptions.
Mean collapse, where random pairing of control and perturbed cells during training erodes biological heterogeneity.
To address these, RAPTORGraph introduces:
A GraphPathway encoder with block-sparse preconditioning that enforces one-to-one mappings between perturbed genes and latent “meta-pathways,” enabling clean single-node interventions
A DAGMA-based causal discovery layer that learns directed acyclic dependencies between latent pathways
An optimal transport alignment step between control and perturbed cells to mitigate mean collapse
The framework achieves state-of-the-art performance on the Norman et al. (2019) Perturb-seq dataset across multiple evaluation metrics (MSE, MK-MMD, and Precision@10 for non-additive interactions)

### Strengths
Solid Empirical Validation
Comprehensive evaluations on the Norman-CPA dataset demonstrate strong reconstruction fidelity and distributional accuracy. RAPTORGraph outperforms state-of-the-art methods like scGPT, dVAE, and SENA on both MSE and MK-MMD metrics

Causal Interpretability
The model’s ability to perform reverse perturbation analysis—predicting causal genes from observed phenotypes—is impressive. The Hit Rate@K results show clear interpretability benefits over baseline VAEs and transformers

Methodological Rigor
The integration of OT-based alignment and DAGMA acyclicity constraints reflects careful methodological design rather than ad hoc architectural engineering.

### Weaknesses
1 Single-Dataset Evaluation
While the paper justifies using the Norman-CPA dataset for fairness, relying on one dataset limits generalization claims. It’s unclear how the approach performs on multi-modal or cross-species data.
2 Ablation Scope
The results show aggregate performance but lack fine-grained ablations—e.g., what is the quantitative contribution of OT alignment versus DAGMA or encoder preconditioning?
3 Biological Validation
While statistical metrics are strong, the paper doesn’t showcase concrete biological discoveries. Were any inferred causal relationships validated against known regulatory pathways or experimental literature?
4 Computational Overhead
The DAGMA layer and OT alignment are computationally heavy. The paper doesn’t discuss scaling behavior for larger perturbation graphs or real-time inference feasibility.

### Questions
1. Identifiability Assumptions
The paper claims that block-sparse encoder preconditioning enforces atomic interventions, improving causal identifiability.
What theoretical conditions guarantee that this structure yields unique causal factors rather than simply disentangled ones?
Is there any formal justification (e.g., through identifiable β-VAE or SCM identifiability theorems)?
Can intervention spillover still happen if correlations between genes violate the one-to-one encoder mapping?

2. β-VAE Regularization
The method uses a β-VAE style latent prior.
How sensitive are results to β?
Is there an empirical tradeoff between disentanglement and reconstruction accuracy?
How does β interact with DAGMA constraints (since both promote sparsity/independence)?

3. Dataset Diversity
The model is only evaluated on the Norman-CPA dataset. Have you tried other single-cell perturbation datasets (e.g., Replogle, Dixit, or sci-Plex)?
Can RAPTORGraph generalize across cell types or perturbation modalities (knockdown → overexpression)?

4. Statistical Significance
Are reported performance gains statistically significant (e.g., over multiple random seeds)?
What is the variance in MK-MMD or Precision@10 across runs?

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
The paper continues a line of work on causal representation learning where a causal model is induced over latent factors (pathways) with the help of an encoder that maps observables (genes) to these latent factors. One key issue is identifiability as many latent causal models can result in the same distribution over the observables, necessitating interventions and also mapping them appropriately to represent interventions in the underlying factors. The authors introduce a sparse (pre-conditioned) encoder to mitigate ``causal spillover'' effect, requiring that each gene that involves perturbations is associated with a separate (single) latent factor. 
The model is learned with multiple losses, including encoder-decoder reconstruction loss, interventional prediction, and a causal graph loss. The authors also introduce a pre-processing step where control cells are paired with perturbed cells via optimal transport with the idea that this reduces the extent to which cell to cell variation would confuse perturbation effects. The primary contribution in the paper is formulation / architectural since significant theoretical results are borrowed from prior work.

### Strengths
The paper is quite well-written and includes a good discussion on challenges with causal representation learning, including intervention spillover and ``mean collapse'' resulting from random pairing of control cells with perturbed ones. The proposed approach seems clear, well-motivated albeit straightforward without significant technical innovations.

### Weaknesses
The verbose discussion of effects takes away from technical clarify (steps missing). E.g., the decoder itself or how the effects of perturbations are predicted are not explicated. The underlying causal graph appears to be linear, resolved with the help of DAGMA loss during training. Each intervention in the data has to be associated with a different latent factor. If factors are pathways, this does not hold in practice. Is there a way to mitigate?

### Questions
Since the encoder maps observables x to exogenous variables z, how are x reconstructions carried out? By mapping x to z, then z to u via the causal graph, and finally from u back to x? 

Do interventions follow the same above steps, starting from a (paired) control cell x, mapped to z, causal graph modified due to intervention on the particular u_i, then resolving the remaining u from the graph, followed by a mapping back to x? 

How many samples are there per condition? Is MMD calculated across conditions?

### Soundness
3

### Presentation
3

### Contribution
3
