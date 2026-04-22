# scCBGM: Single-Cell Editing via Concept Bottlenecks

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 6, 6, 4

## Abstract
How would a cell behave under different conditions? Counterfactual editing of single cells is essential for understanding biology and designing targeted therapies, yet current scRNA-seq generative methods fall short: disentanglement models rarely support interventions, and most intervention-based approaches perform conditional generation that synthesizes new cells rather than editing existing ones. We introduce Single-Cell Concept Bottleneck Generative Models (scCBGMs), unifying counterfactual reasoning and generative modeling. scCBGM incorporates decoder skip connections and a cross-covariance penalty to decouple annotated concepts from unannotated sources of variation, enabling robust counterfactuals even under noisy concept annotations. Using an abduction–action–prediction procedure, we edit cells at the concept level with per-cell precision and generalize zero-shot to unseen concept combinations. Conditioning modern generators (e.g., flow matching) on scCBGM embeddings preserves state-of-the-art fidelity while providing precise controllability. Across three datasets (up to 21 cell types), scCBGM improves counterfactual accuracy by up to 4×. It also supports mechanism-of-action analyses by jointly editing perturbation and pathway-activity concepts in real scRNA-seq data. Together, scCBGM establishes a principled framework for high-fidelity in silico cellular experimentation and hypothesis testing in single-cell biology.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces scCBGM, a model for performing counterfactual perturbation prediction on single-cell RNA-seq data. scCBGM individuates concepts in the latent space, which it splits into two components: known concepts and unknown residual factors. Known concepts represent annotated covariates, like cell type. Unknown concepts are unobserved axes of cellular variation. The authors train scCBGM on a combined task, where the standard VAE loss is combined with a concept loss. The authors also combine the latent space in scGBM with flow matching, using factor representations as conditioners. The main use of the methods is performing counterfactual predictions of perturbation effects on single cells, which the authors benchmark on the task of simulating state shifts in multiple biological settings.

### Strengths
I find the paper enjoyable and the scientific question quite compelling. The attempt to infuse interpretability into perturbation models is relevant to step closer to their deployment, and I am happy to see the authors develop their research in this direction. Everything is described clearly, the Appendix is very polished, and the code is readable; I could trace back multiple questions I had to the analysis notebooks.

### Weaknesses
Currently, my questions and remarks suggest a negative score. However, I also admit that I might have missed a few points or hold misconceptions, and I remain more than happy to improve my rating during the rebuttal phase. 

**Major**

- **Scope and formulation** My main concern regards the way concepts are formulated. If I understand correctly, the factors associated with known concepts are trained to approximate exactly these concepts, so the best possible outcome of the training process is for a cell to be encoded to its own conditioner. I do get that by doing it, one does not obtain the exact values for the concept, but more of an "activation". However, I feel that this is not that different than having a standard conditional VAE's (cVAE) latent space with some dimensions dedicated to representation and some others to covariate-based conditioning. I also acknowledge the originality of the cross-covariance loss to further remove the effect of known concepts from the latent space, but a similar effect is, again, to some extent present in cVAEs as the decoder regresses out conditioning variables (hence why such a method is so popular for batch correction). I am not hinting at the fact that the model is no different from a cVAE, but I believe that eventually it may have a similar effect, especially when conditioning the Flow Matching model. 

-  **Editing and Flow Matching.** In my opinion, it is a bit inaccurate to define what has been done with Flow Matching as counterfactual/editing. Starting from the decoding, I think I would need an example showing that if you start from a noise variable $z \sim N(0,I)$ and generate from it conditioned on the concepts  $[u_i, c_i]$ of a cell $x_i$ (i.e., no editing), you effectively retrieve something close to the original cell $x_i$. The reason why I am saying this is that I am not sure to what extent generating from noise violates the "cell identity" information embodied by $u$. A similar concept applies to the editing procedure. You assume that inverting the flow to noise preserves the cell identity, but in a way that is already encoded by the unknown factors in $u$. Maybe the performance gain by using flow matching arises since the model is evaluated exclusively based on distributional metrics, at which the flow model may be better, as it smooths predictions around the data points. 

-  **Table 2.** The results in Table 2 are not significant. I think this ablation really speaks against the worth of the method against the reference approach. Maybe you could find another way to assess the improvement? 

- Unfortunately, I am not very convinced by the results of the case study. My main concern is the quality of the evidence, which, for now, is limited to a new cluster on the UMAP. I think this result and the whole study in general would be more interesting if more insight into the gene expression changes were provided. For example, are there any interesting genes upregulated by your synthetic simulation process? Do they agree with the state of the art? 

**Minor**

- L52-53: "Existing perturbation...": I am not 100% sure about this statement. I thought the idea behind CPA, Biolord, STATE, and similar methods is to predict a state shift from a basal/initial cell state. I feel this is quite akin to the counterfactual task described here. 

- It would be great if you could add subscripts referring to the considered distribution for the expectations. 

- L201-211: Unless you are familiar with the concepts, it is not clear what the difference between a standard bottleneck model and a concept embedding model is. 

- L277-278: I would write "data generating process" rather than "generative model".

- I think the MMD analysis could be complemented by something like subtype classification. In other words, pre-train a subtype classifier and evaluate how many times the subtype is preserved in translated cells. 

[1] Lotfollahi, Mohammad, et al. "Predicting cellular responses to complex perturbations in high‐throughput screens." Molecular systems biology 19.6 (2023): e11517.

[2] Piran, Zoe, et al. "Disentanglement of single-cell data with biolord." Nature Biotechnology 42.11 (2024): 1678-1683.

[3] Adduri, Abhinav K., et al. "Predicting cellular responses to perturbation across diverse contexts with State." bioRxiv (2025): 2025-06.

### Questions
1. Are the MMDs computed on decoded data? If yes, I find the use of the rMMD a bit strange. Since you have the same ground truth for all models you are benchmarking, you could just compare them based on that, right? In many cases (e.g. Monocytes, Dendritic cells, NK cells), the response predicted by scCBGM is much worse than the population with the lowest distance to the perturbed subtype. I deem this result a bit concerning with respect to the reliability of the model. 

2. You refer to Vanilla FM as CellFlow here. But CellFlow is a state-to-state transition model using OT-based Conditional Flow Matching. Provided you reimplemented CellFlow, which does not exploit noise in its formulation, how do you perform editing and decoding here? Or how is the Vanilla model different from the scCGMB one? I looked for this information in the appendix, but I could not locate it.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
Authors with this manuscript presents scCBGM, a generative model designed for the counterfactual editing of single-cell RNA-sequencing data. It combines a concept bottleneck architecture with a generative decoder to allow for interpretable, cell-specific interventions.

### Strengths
***S1:*** The paper's core strength is its focus on the nuanced but critical difference between conditional generation and counterfactual editing. This is a more scientifically valuable goal, and the entire framework is built in service of it.

***S2:***  The concept bottleneck approach is a nice fit for this problem. It demystifies the latent space and turns the model into an interactive tool for in silico experiments, allowing researchers to directly test hypotheses in a causal manner.

***S3:*** The additions of skip connections and the cross-covariance penalty are solutions to the specific challenges of preserving cell identity and achieving robust disentanglement, which are essential for reliable counterfactuals. The ablation results support their inclusion.

### Weaknesses
***W1:*** The model's greatest strength is also its primary weakness. Its effectiveness is entirely contingent on the user providing a comprehensive and accurate set of biological concepts. If a crucial biological process is not included in the bottleneck, the model will be unable to reason about it, and its predictions for edits involving that process may be unreliable. Authors need to work on this, finding a key workaround.

***W2:*** The paper relies on synthetic data where ground-truth counterfactuals are known, and population-level metrics on real data. This is a reasonable and standard approach. However, for real cells, a true, cell-specific counterfactual is impossible to observe. The paper would be strengthened by a more thorough discussion of this fundamental limitation and the potential pitfalls of relying on population-level metrics to validate cell-level edits.

***W3:*** The experiments use a reasonable number of concepts. It is unclear how the model's performance and training stability would be affected as the number of concepts scales to the hundreds or thousands, which might be necessary to capture the full complexity of some biological systems.

### Questions
Some extra questions to solve my concerns and help me raising the score, upon weaknesses rebuttal.

***On Concept Leakage:*** The cross-covariance regularizer aims to prevent unannotated factors from leaking into the concept representations. How do you diagnose such leakage if it still occurs? Could the model, for instance, learn to encode information about cell cycle state within the "cell type" concept, even if they are meant to be separate?

***On Concept Completeness:*** How would a user know if their predefined concept set is "good enough"? Have you explored any diagnostic tools to identify when the model's residual (unexplained variance) is still highly structured, suggesting that important concepts are missing?

***On the Abduction Step:*** The abduction step infers the concept values for a given cell. How does the model handle cases where the concepts are biologically entangled (e.g., a specific perturbation is known to always activate a certain pathway)? Does the model correctly assign causality in these cases, or does it simply reflect the correlations seen in the training data?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Conventional conditional generation models predict population-level changes under altered conditions, whereas cell editing allows prediction of individual-cell level changes. Conducting experiments that combine single cells with multiple conditions (treatments, exposures, doses) results in an enormously large combinatorial space, making exhaustive experimentation impractical. Therefore, computational models aim to bridge this gap. This is particularly important in settings where cell heterogeneity plays a key role, such as diverse cell states and developmental trajectories. The authors propose scCBGM, an extension of CBGM that incorporates skip connections and cross-covariance penalty, designed to enhance robustness against noise. Using real datasets from Kang et al., Cui et al., and TCDD exposure, including IFN-β, various cytokine, and dose conditions, they demonstrate that scCBGM not only improves performance over baselines but also shows further gains when integrated with scCBGM-FM.

### Strengths
1. Clear conceptual distinction and integrative applicability

    The model clarifies the distinction between conditional generation and counterfactual cell editing, enhancing conceptual clarity in scRNA-seq research. Moreover, scCBGM supports both precise cell-level editing and population-level expression inference, enabling integrated single-cell and group-level analysis.
    
2. High compatibility with diverse generative frameworks

    scCBGM integrates seamlessly with state-of-the-art generative modeling frameworks such as Flow Matching, and its flexible design ensures scalability and adaptability to future generative models, maintaining long-term applicability.
    
3. Performance improvement

    The incorporation of skip connection and cross covariance penalty mechanisms leads to enhanced predictive performance and improved robustness across various experimental settings.

### Weaknesses
1. The overall flow is somewhat disjointed due to Section 3, and some details are missing, making it difficult to fully understand the methodological and experimental connections across sections.
2. The technical contribution is marginal.  The use of skip connections and Lcc (cross-covariance loss) is not particularly original.
3. In the introduction, the authors state: *“Existing perturbation modeling methods focus on conditional distributions of cell states across treatments, but they do not enable counterfactual editing of individual cells.”*
Given this claim that prior methods focus on distribution-level modeling and fail to support individual-cell editing it’s unclear why the authors chose to use rMMD instead of an OT-based (Optimal Transport) metric, which would seem more aligned with their stated motivation, especially considering that the baseline model, CINEMA-OT, employs an OT-based metric.

### Questions
1. In the introduction, the authors state: *“Existing perturbation modeling methods focus on conditional distributions of cell states across treatments, but they do not enable counterfactual editing of individual cells.”*
Given the claim that prior methods focus on distribution-level modeling and fail to support individual-cell editing, it would be valuable to include an additional experiment comparing rMMD with an OT-based (Optimal Transport) metric, since the latter appears more consistent with the authors’ stated motivation, particularly considering that the baseline model, CINEMA-OT, employs an OT-based metric.


2. Potential inconsistency between the VAE structure and the cell-editing objective:
    
    The scCBGM framework is based on a Variational Autoencoder (VAE), which enforces the latent space to conform to a prior distribution through regularization. This can cause the latent variable z to collapse toward minimal information content, potentially losing fine-grained details of the unknown factors.
    
    However, since cell editing fundamentally requires preserving informative latent representations for precise perturbation, an Autoencoder (AE) architecture, without such regularization, might have been a more suitable alternative.
    It would be helpful to clarify the rationale for choosing the VAE-based architecture over an AE-based one.
    

3. In Table 1, the results across models overlap within standard deviation, raising questions about the statistical significance and meaningfulness of the reported ablation improvements.


4. Around line 443, the paper states that scCBGM-FM outperforms vanilla-FM, yet for some cell types, vanilla-FM performs better. The authors should discuss why this occurs and whether it indicates model instability or dataset-specific behavior.



### Minor

1. In D.1.1, the authors mention using all but megakaryocytes among nine broad cell types in the Kang et al. dataset. Wouldn’t that mean eight cell types remain? Yet, Table 2 lists only seven cell types. This discrepancy needs clarification.
2. In Cui et al., the dataset includes 17 cell subtypes and 86 cytokine-based stimulations, but the authors only tested seven combinations. It’s unclear what criteria were used to select these seven as test conditions.
3. Figure 4 lacks any explanation in the main text, and based on the reported rMMD scores, it appears to correspond to Cui et al.’s dataset rather than Kang et al.’s. This should be explicitly clarified.

### Typo

1. Line 53 is missing a period (‘.’).
2. In the Figure 1 caption, the acronym “DAG” is used without first providing its full name.
3. In lines 253 and 259, as well as in Table 2 and Table 3, the model is referred to as “scCBM-FM”, but it appears this should be “scCBGM-FM.”

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes scCBGM, a modification of Concept Bottleneck Generative Models (CBGMs) to single-cell data. Specifically, scCBGM uses “(i) a standard concept bottleneck model rather than a concept embedding model, (ii) [...] skip connections to the decoder to maintain persistent concept conditioning; and (iii) [...] a cross-covariance loss instead of the cosine similarity loss for orthogonality. The main evaluation lies in perturbation response prediction for which the paper proposes (i) the vanilla scCBGM, (ii) scCBGM with sampling from the latent (decode), and (iii) scCBGM with encoding an unperturbed x and editing it (edit). scCBGM shows overall strong performance on rMMD scores in perturbation response prediction.

### Strengths
The paper presents a methodologically sound latent disentanglement model to predict perturbation response with an interesting addition to enable both fully-connected decoding with skip connections and as a conditioning for flow-matching models. Originally arises from combining the existing CBGM model with tweaks for single-cell data and novel decoders that recently proved powerful in the field. The paper is clearly written and enjoyable to read. It presents a solid contribution to the field of single-cell disentanglement methods and perturbation response prediction models.

### Weaknesses
I find three key technical contributions: (1) the changes on the CBGM model for single-cell data, specifically, (a) the standard concept bottleneck model instead of concept embedding and (b) cross-covariance loss instead of the cosine similarity loss. (2) the changes on the decoder of single-cell VAE models, specifically the skip connection. (3) the adaptation of flow-matching models on top of disentangled latent spaces.

While technically sound, (1) and (2) seem incremental and lack ablation study - the ablation of scCBGM vs CBGM may mix concepts together. I find (3) as a key contribution interesting. However, given the breadth of disentanglement methods in ML for single-cell - some of which are referenced to in this submission - a more rigorous comparison is required in my opinion. For example, how does an edit-flow-matching decoder perform on the disentangled latent space of another existing disentanglement method?

The empirical evaluation seems to show the overall good performance of flow-matching based models, which is generally supported by recent literature. I think a harder comparison may be how the scCBGM and CellFlow flow-matching models perform on, e.g., scVI latent spaces, disentangled representations of single-cell data, etc. 

Last, the evaluation mainly relies on rMMD and qualitative assessment, a more rigorous evaluation would help practitioners find a realistic assessment of scCBGM.

### Questions
I kindly suggest that the reviewers ablate individual components, especially addressing if the disentanglement of scCBGM is really superior to that of existing single-cell disentanglement methods.

### Soundness
3

### Presentation
3

### Contribution
2
