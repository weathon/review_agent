# Discovering interpretable biological concepts in single-cell RNA-seq foundation models

- Decision: Reject
- Scores: 6, 4, 2, 6

## Abstract
Single-cell RNA-seq foundation models achieve strong performance on downstream tasks but remain black boxes, limiting their utility for biological discovery. Recent work has shown that sparse dictionary learning can extract concepts from deep learning models, with promising applications in biomedical imaging and protein models. However, interpreting biological concepts remains challenging, as biological sequences are not inherently human-interpretable. We introduce a novel concept-based interpretability framework for single-cell RNA-seq models with a focus on concept interpretation and evaluation. We propose an attribution method with counterfactual perturbations that identifies genes that influence concept activation, moving beyond correlational approaches like differential expression analysis. We then provide two complementary interpretation approaches: an expert-driven analysis facilitated by an interactive interface and an ontology-driven method with attribution-based biological pathway enrichment. Applying our framework to two well-known single-cell RNA-seq models from the literature, we interpret concepts extracted by Top-K Sparse Auto-Encoders trained on two immune cell datasets. With a domain expert in immunology, we show that concepts improve interpretability compared to individual neurons while preserving the richness and informativeness of the latent representations. This work provides a principled framework for interpreting what biological knowledge foundation models have encoded, paving the way for their use for hypothesis generation and discovery.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduce an interpretability framework for single-cell RNA-seq models based on concept extraction through a decomposition approach, which relies on sparse auto-encoders. To evaluate these, they propose an attribution-based method with conterfactual perturbations, an attribution-based gene set enrichment analysis (GSEA) and a web-based visualization tool with an expert interpretation. Experiments on Tabula Sapiens Immune and Cross-Tissue Immune Cell Atlas datasets show that these concepts are more interpretable and stable than individual neurons while retaining downstream predictive power.

### Strengths
1. Paper is well written and structured.

2. Addresses interpretability of large scRNA-seq foundation models, which is still an area of high interest.

3. Combines sparse dictionary learning with counterfactual-based attributions, going beyond correlation-based analyses.

4. Uses large immune cell datasets; includes cross-dataset stability analysis.

5. Incorporates an area expert, which adds credibility and biological realism.

### Weaknesses
1. The novelty of the proposed top-k sparse auto-encoder to extract meaningful concepts is limited. 

2. Expert evaluation involves only one annotator; inter-expert variability or reproducibility is not assessed.

3. The “interpretability gain” is mostly qualitative, no standardized metric is provided.

4. Both datasets are immune-cell focused; generalization to other tissues or modalities remains untested.

### Questions
1. Is the code and visualization platform available at submission? I can't find it in the paper.

2. Could you highlight the main difference between other models that provide an interpretable latent space such as VEGA (https://doi.org/10.1093/bioinformatics/btad387), GSAE (https://doi.org/10.1186/s12918-018-0642-2) or SENA-discrepancy-VAE (https://openreview.net/pdf/f8c0bd3ea256d2af7cfe66c6a46b511cab5ba995.pdf)? Couldn't we forward the single-cell samples through any of these and extract similar features?

2. How does the attribution-based GSEA differ quantitatively from traditional DGEA-based enrichment? Can you provide examples of pathways uniquely discovered?

3. How stable are concepts when changing SAE random seeds or hyperparameters? Could you include std in metrics where multiple seeds have been tested?

### Soundness
3

### Presentation
3

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
This paper introduces a framework for improving the interpretability of single-cell RNA-seq foundation models. The manuscript is clearly written, the methodology is sound, and the experiments are thorough.

The core of the contribution is the gene attribution method based on counterfactual perturbations. By identifying genes that distinguish cells activating a concept from similar cells that do not, this approach moves beyond simple correlational analysis (like DGEA) to better pinpoint influential genes.

A couple of weaknesses (context within the broader field of interpretable ML, justification of methodological choices, depth of the counterfactual analysis, limitations of the expert study, hypothesis generation claim) should be addressed.

### Strengths
The core contribution (gene attribution method based on counterfactual perturbations) is a well-grounded methodological advancement.

The empirical validation is sound: The authors provide evidence that concepts extracted via TopK SAEs are more interpretable than individual neurons from the original models scGPT and scVI. Also, the study evaluates the stability of learned concepts across different datasets, a known challenge for SAEs.

### Weaknesses
Context within the broader field of interpretable ML: The introduction frames the problem primarily as one of post-hoc explanation for "black box" models. However, it would benefit from acknowledging and contrasting its approach with the branch of causal representation learning, e.g., discrepancy-VAE (Zhang et al.), SENA (de la Fuente et al.), and GEARS (Roohani et al.). Discussing why a post-hoc concept extraction approach might be preferable or complementary (e.g., applicable to any pre-trained foundation model without architectural changes) would provide a more complete picture of the landscape.

Justification of methodological choices: The manuscript states that TopK SAEs simplify tuning over vanilla SAEs with L1 regularization. This is a practical reason, but the authors should provide a more thorough justification for choosing this specific dictionary learning architecture over other methods for learning disentangled representations. For instance, why not use a VAE with a structured prior (e.g., beta-VAE) to learn the concepts directly from the embeddings?

Depth of the counterfactual analysis: The counterfactual for a sample is defined as the closest sample in the embedding space that does not activate the concept. This is a reasonable heuristic, but it has limitations. The "closest" cell might differ in biologically important ways beyond the concept in question, potentially confounding the attribution scores. The authors should discuss this limitation.

Limitations of the expert study: The study was conducted with a single domain expert. While valuable, this is an N=1 study, and the generalizability could be questioned. The authors should explicitly state this as a limitation and perhaps soften their conclusions slightly. Was there a process to measure intra-expert consistency (e.g., showing the same concept twice)? This should be clarified.

Hypothesis generation claim: The examples show that the framework can recover well-established biological concepts. To fully substantiate the claim of enabling hypothesis generation and discovery, it would be powerful to showcase an example of a concept that is either novel, refines existing knowledge, or reveals a non-obvious connection between genes that could lead to a testable biological hypothesis.

Typos:
- Page 2, line 104: \sigma(\cdot)
- Page 2, line 107: L1 regularization

### Questions
The questions directly relate to the identified weaknesses:
- How does the post-hoc concept extraction framework complements or offers advantages over methods from causal representation learning which aim to learn disentangled representations directly as part of the model architecture?
- What are the primary methodological advantages of using TopK SAEs for learning concepts from embeddings compared to alternative approaches for learning disentangled representations?
- What are the potential limitations of defining a counterfactual as the nearest neighbor that does not activate a concept, particularly the risk that confounding biological differences between the two cells might influence the gene attribution scores?
- Which measures were taken to assess intra-expert consistency?
- Can you provide an example where your model identifies a concept that reveals a previously non-obvious connection between genes or refines existing knowledge in a way that leads to a new, testable hypothesis?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose a concept-based interpretability framework for single cell RNA-seq data to uncover genes that activate different (biological) concepts, enabling interpretation of known single-cell emeddings. They also provide an interactive interface to facilitate the concept analysis by experts, as well as an enrichment method based on uncovered attributes.

The authors show that the purpose model leads to better interpretability compared to individual neurons.

### Strengths
The paper provide a novel way of interpreting the embeddings provided by well known scRNA-seq models such as scVI and scGPT. They also provide an intuitive framework to analyze the obtained results by experts that may not be able to run the model themselves. I think this work can have important relevance on the biomedical community, as interpretability is a key factor while working with single-cell data.

### Weaknesses
I don't think this work provide enough sound advances in representation learning or machine learning for being suitable to ICLR. The method applies known SAE for dictionary learning based on known representation learning models from scRNA-seq data (scVI, scGPT). While the idea of interpreting the concepts inferred by the SAE is very interesting, they are more suitable to more specialized conferences/journals.

In addition to that, I believe the results are somehow incomplete. While they provide good analyses on the SAE across the two embedding spaces, a more in-depth benchmark is required to fully understand the benefits of the proposed model. For example, there are other integration/embedding models that are widely used that should be compared such as CCA (Seurat), Harmony, the new U-space by MrVI, etc. The inclusion of these models could provide a more "universal" view of the proposed interpretability framework.

Additionally, more there are other works based on SAE for interpretability (eg. https://doi.org/10.1093/nar/gkae197, and reference therein) that should be mentioned (and maybe compared).

Finally, the expert analysis (the soundness of the biology inferred by the model) should be more thoroughly analyzed (see for example, the work mentioned above).

### Questions
How does the model perform when using other integration/embedding models for scRNA-Seq data?

How does it compare against traditional GSEA analysis on the same comparisons? I understand that through GSEA genes themselves may not have "interpretability" but are both models uncovering the same biological processes?

### Soundness
2

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
2

### Summary
This ms mainly focuses on exploring the cell embeddings given by scGPT and scVI. The idea is to perform a low-rank decomposition of the embedding matrix (cell x emb) using sparse autoencoder (SAE), yielding a loading matrix U and factor matrix D. The factor matrix is called "concept" by the authors. Then they try to see which genes are relevant for the concepts. This is done by masking each input gene on chosen cells and check how they change the loading of a given concept. The change is called "attribution" by the authors. In two experiments, they show that the concepts are biologically interpretable, which are associated with cell types, GO pathways. This work is a step further than traditional tasks of using cell embeddings for clustering or batch effect correction.

### Strengths
1. The idea of introducing concept seems to make biological sense.
2. I think the association between the concepts and GO terms (Fig. 3B and Fig. 4C) are very interesting.  It could be further explored in the future.
3. Although missing some details, the method descriptions are solid.

### Weaknesses
Maybe I have missed, but In section 4.2, it is unclear to me
1. the distribution of the attribution scores, why 0.05 is selected as the cutoff
2. what are the cells used for calculating the attribution scores

Lack of a baseline method for the benchmark. How does it compare with direct low rank decomposition of the cell embedding matrix? In theory, you can treat the factors as the "concepts" and loadings as the "concept activations".

### Questions
line 179, why not directly mask gene l in x^p ?
line 288, what is "expansion factor" and why it matters here?
line 300, what is "neurons" in scGPT? could you be more specific about the layers?

### Soundness
3

### Presentation
3

### Contribution
3
