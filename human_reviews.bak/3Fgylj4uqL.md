# Interpretable Causal Representation Learning for Biological Data in the Pathway Space

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Predicting the impact of genomic and drug perturbations in cellular function is crucial for understanding gene functions and drug effects, ultimately leading to improved therapies. To this end, Causal Representation Learning (CRL) constitutes one of the most promising approaches, as it aims to identify the latent factors that causally govern biological systems, thus facilitating the prediction of the effect of unseen perturbations. Yet, current CRL methods fail in reconciling their principled latent representations with known biological processes, leading to models that are not interpretable. To address this major issue, in this work we present SENA-discrepancy-VAE, a model based on the recently proposed CRL method discrepancy-VAE, that produces representations where each latent factor can be interpreted as the (linear) combination of the activity of a (learned) set of biological processes. To this extent, we present an encoder, SENA-$\delta$, that efficiently compute and map biological processes' activity levels to the latent causal factors. We show that SENA-discrepancy-VAE achieves predictive performances on unseen combinations of interventions that are comparable with its original, non-interpretable counterpart, while inferring causal latent factors that are biologically meaningful.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces SENA-discrepancy-VAE, a novel model in causal representation learning (CRL) designed to make biological data analysis—especially from Perturb-seq experiments—more interpretable. A key innovation of this work is how it integrates biological processes (BPs) as prior knowledge, directly linking the model’s latent factors to known biological pathways. This approach fills an important gap in existing CRL methods, which often struggle with interpretability since they don't directly associate learned representations with actual biological mechanisms, making them less useful for real research applications.

SENA-discrepancy-VAE builds on the standard discrepancy-VAE by introducing a pathway-based masking strategy within a new encoder, SENA-δ. This encoder uses a two-layer masked MLP where the first layer maps gene expression values to BP activity levels, with a tunable parameter that adjusts the influence of genes outside predefined pathways, giving the model flexibility in gene-pathway associations. The second layer models latent factors as combinations of these BP activities, which is a more realistic approach since biological interventions often impact multiple pathways. This setup stays true to the CRL assumption that each intervention targets a single latent factor but does so in a way that aligns with biological realities.

The authors evaluate SENA-discrepancy-VAE on a Perturb-seq dataset of leukemia lymphoblast cells. They show that the model performs as well as the original discrepancy-VAE on unseen perturbation combinations while providing greater interpretability by identifying specific BPs associated with each latent factor. This interpretability is validated through pathway-specific analysis, demonstrating the model’s ability to reveal biologically meaningful patterns in response to genetic interventions.

### Strengths
1) The proposed model’s integration of biological pathway data as a prior in the causal representation learning (CRL) framework is interesting and practical, offering a biologically grounded solution to gene expression analysis. Through the direct alignment of the latent factors with biological pathways, SENA-discrepancy-VAE addresses the common limitation in CRL models of producing uninterpretable latent factors. 

2) The paper presents thorough experiments across multiple perturbation types, including single and double-gene knockouts, demonstrating the model’s robustness. Its generalization to unseen perturbations is compelling, and the ablation studies, which explore interpretability-reconstruction trade-offs, further validate the model’s design.

3) The authors have done a good job communicating the importance of embedding biological processes into latent spaces, with visuals that illustrate BP-specific activity levels influencing latent factors. The use of causal graphs and the differential activation (DA) metric enhances transparency, allowing readers to trace latent factors back to biological functions. 

4) The model’s ability to provide interpretable predictions on cellular responses can aid in experimental design and offer insights into the potential effects of genetic or drug interventions. This approach addresses a pressing need in biomedicine for interpretable causal representation learning (CRL) models that can shed light on the intricate causal relationships underlying gene function and cellular processes.

### Weaknesses
1) While the model effectively identifies single-point perturbations, it does not accommodate multi-step perturbations or capture the progression of cellular responses over time. In biological experiments, the cellular response often evolves in phases, with gene activity showing distinct transitions that are essential for understanding the effect of interventions.

2) The model’s validation on a single dataset (K562 cell line data) restricts insights into its generalizability across different cell types or conditions. Testing on additional datasets, such as those from other cell lines or cellular environments, would offer a stronger assessment of robustness and applicability across a wider range of biological data.

3) The model assumes static pathway relevance across all tasks, which may limit its adaptability in varied biological contexts where pathway importance changes with cell type or condition.

4) The paper assumes that each intervention corresponds to a single latent factor, limiting the model's ability to capture complex interactions where multiple latent factors might be influenced by a single intervention. This simplification restricts the model’s interpretability in representing overlapping or interacting biological processes, which are common in gene expression dynamics.

5) The model lacks a detailed exploration of how varying the number of latent dimensions impacts the interpretability and causal mapping of latent factors. Larger or smaller dimensions can influence the granularity of the factors and thus affect the biological insights the model can provide.

### Questions
1) Could the authors clarify whether multi-label or probabilistic pathway activity labels were considered as an alternative to binary labels? Binary labels may oversimplify gene activity levels, especially when certain pathways exhibit gradations rather than discrete on/off states.

2) How would the model perform if pathway relevance were dynamically adjusted based on specific task contexts or experimental conditions? Pathway importance often varies, and adapting pathway relevance could improve model flexibility.  A task-specific analysis to explore whether dynamically adjusting pathway selection enhances generalizability across datasets and biological contexts could provide insights into the model’s robustness.

3) In the causal graph (Figure 3), how sensitive are the edge weights to the choice of λ and latent dimension? Please provide a sensitivity analysis.

4) The mask matrix M (equation 2) assumes binary gene-pathway relationships. Have you considered using weighted relationships based on pathway databases' confidence scores?

5) How does D_KL evolve during training for different λ values? Training curves would help understand if this is an optimization or regularization effect. Does this phenomenon persist if you randomize the pathway annotations while maintaining the same sparsity structure? This would determine if the benefit comes from biological knowledge or just sparsity.

6) Have the authors empirically validated that the expectation in Equation (10) aligns with the observed experimental outcomes? Demonstrating this match would reinforce the theoretical assumptions and provide additional confidence in the model's causal interpretability.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents SENA-discrepancy-VAE, an extension of the discrepancy-VAE framework that incorporates biological pathway knowledge to produce interpretable causal latent factors. The authors modify the encoder architecture to map gene expression through biological processes (BPs) while maintaining the theoretical guarantees of the original model. The approach achieves comparable predictive performance to the original discrepancy-VAE while providing biologically meaningful latent representations and interpretability.

### Strengths
* Clarity: The paper is well written and easy to follow.

* Novel Technical Contribution: The paper successfully extends causal representation learning to incorporate domain knowledge while preserving theoretical guarantees. The SENA-δ encoder architecture is a clever solution to balance interpretability and performance.

* Practical Impact: The work addresses a significant gap in current causal representation learning methods for biological data, where interpretability is crucial for scientific insights. The ability to map latent factors to biological processes makes the model more useful for domain experts.

### Weaknesses
* Limited Biological Validation: While the authors show statistical associations between perturbations and biological processes, there could be more validation using external biological knowledge or experimental validation of the discovered causal relationships.

* Hyperparameter Sensitivity: The model introduces an additional hyperparameter λ that significantly impacts performance. While ablation studies are provided, more guidance on selecting this parameter would be valuable (this is important given that there's some large impact on the performance of the method)

* Restricted Evaluation: The empirical evaluation is limited to a single dataset (Norman et al., 2019). Additional validation on different types of biological data would strengthen the claims of generalizability.

### Questions
Can you provide examples of BPs in the appendix? 

Regarding DAR evaluation: What happens if unnaffected pathways have very low action? Also, more general, how would you deal with impalanced pathways, which might lead in measuring large noise levels? 

In table 2, for SENA λ=0.1, latent dim 105, the variance compared to original MLP and λ=0 is significantly lower (0.000081 vs 0.001087). Just double checking if this is corrent.

L237: During filtering you end up with a (biased) set of BPs. How much do you think this can influence interpretability? Is there a risk of removing useful BPs?

Suggestions
L100: the word faithfully here gets confused with causal faithfulness. Please consider an alternative adverb if possible.
L105: target instead of targets (remove final s)

### Soundness
2

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
This work proposes to extend the discrepancyVAE interventional causal representation learning framework to biological processes applications. Specifically, the authors propose to embed prior knowledge about biological processes (BPs) through a framework called SENA-discrepancyVAE, which recovers latent factors that are a linear combination of a set of biological processes (pathways). The main idea presented in this work is to design a more flexible encoder class (SENA-\delta) specific to mapping biological pathways to latent causal factors for interpretability. Empirical results show that the framework is shown to improve performance in predicting the effect of unseen perturbations.

### Strengths
- The paper is well written with strong motivations behind using CRL techniques for biological applications.
- The metrics proposed (differential activation, Hits@N) seem to be robust indicators of perturbation effects on BPs and downstream effects. I believe these evaluation metrics are one of the key interesting contributions of this work.
- The empirical evaluation is exhaustive and illustrates some interesting observations, especially the representational capacity of the VAE-based SENA method compared to the traditional discrepancyVAE.
- The interpretability analysis of the reparameterization layer is interesting and reveals which genes were affected the most upon perturbations. I do believe that exploring real-world applications of CRL is a very important direction.

### Weaknesses
- Although the application in gene regulatory networks is quite interesting, this work seems to be more of an evaluation study of the discrepancy-VAE framework proposed by Zhang et al. I do not see much of an added contribution beyond the original paper besides highlighting the application.
- The difference in performance between the SENA variant and the original discrepancyVAE seems to be quite marginal in terms of representation in the double-perturbation scenario. For instance, in Table 2, the KL-divergence for double-perturbation prediction is only marginally better than the original MLP-based discrepancyVAE.

### Questions
What is the intuition behind the $\lambda$ hyperparameter to tune small influences of a gene on a biological process? Should this be a constant value throughout the mask matrix or would it be better to learn this influence via some type of attention weights?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper addresses the challenge of learning interpretable causal representations for Perturb-seq data (gene expression in cells). The primary contribution is the novel introduction of masking to incorporate biological process (BP) knowledge into an existing method for causal representation learning (discrepency-VAE), which is named SENA-discrepancy-VAE. The masking ensures that latent factors can be interpreted as linear combinations of the activity of BPs. Since this modification is compatible with the discrepancy-VAE, the original model's theoretical guarantees for causal representation learning remain.

The method and ablated variants are evaluated on a Perturb-seq dataset collected from one particular cell line and is set up to minimize the overlap between the BPs. The results demonstrate that SENA performs similarly to discrepency-VAE in terms of reconstruction yet results in sparser and more interpretable results. Furthermore, by studying the contrast between inferred activity levels on perturbed and control samples the authors show that the latent factors can be associated with BPs and are therefore interpretable.

### Strengths
- Clear technical contribution that bridges causal representation learning with biological interpretability while maintaining theoretical guarantees
- The paper contributes to causal representation learning for Perturb-seq data by introducing biological interpretability through pathway information, while maintaining the theoretical guarantees of discrepency-VAE.
- Well written and clear presentation of the method and results.
- Thorough ablation studies
- Demonstrates interpretability of latent factors with concrete biological examples.

### Weaknesses
- Experimental validation is limited to one dataset and no baselines other than their own ablations and discrepancy-VAE. The paper would benefit from comparisons to at least one of the other listed related works.
- No comparison with simpler approaches like post-hoc interpretation of standard discrepancy-VAE latent factors.
- While the link between latent factors and BPs is investigated, the quality of the discovered causal graph is not.
- Given that the latent factors group a large number of BPs into a small number of latent factors there should be a deeper investigation of the biological plausibility and practicality of this result beyond the contrasting activations.
- Readability of several figure texts should be improved.

### Questions
- How sensitive is the model to the quality and completeness of the pathway knowledge used? Have you tested with different pathway databases or subsets of pathways?
- How does the computational complexity scale with the number of biological processes? Is there a practical limit to how many processes can be incorporated?
- Have you explored whether the causal relationships discovered by the model align with known biological pathway interactions beyond the examples provided?
- Can you confirm that all genes involved in the double perturbations were also present in your single-perturbation training data?
- Has $N$ and $\tau$ been mixed up in the Hits@N metric?
- How does table 2 show that “both models tend to assign most interventions to a small number of latent factors”?

### Soundness
3

### Presentation
4

### Contribution
3
