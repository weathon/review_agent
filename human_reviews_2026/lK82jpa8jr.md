# HEIST: A Graph Foundation Model for Spatial Transcriptomics and Proteomics Data

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
Single-cell transcriptomics and proteomics have become a great source for data-driven insights into biology, enabling the use of advanced deep learning methods to understand cellular heterogeneity and gene expression at the single-cell level. With the advent of spatial-omics data, we have the promise of characterizing cells within their tissue context as it provides both spatial coordinates and intra-cellular transcriptional or protein counts. Beyond transcriptomics, proteomics offers a complementary view by directly measuring proteins, which are the primary effectors of cellular function and key therapeutic targets. However, existing models either ignore the spatial information or the complex genetic and proteomic programs within cells. Thus they cannot infer how cell internal regulation adapts to microenvironmental cues. Furthermore, these models often utilize fixed gene vocabularies, hindering their generalizability to datasets with different genes than pretraining. In this paper, we introduce HEIST, a hierarchical graph transformer foundation model for spatial transcriptomics and proteomics. HEIST models tissues as hierarchical graphs. The higher level graph is a spatial cell graph, and each cell in turn, is represented by its lower level gene co-expression network graph. Rather than using a fixed gene vocabulary, HEIST computes gene embeddings from its co-expression network and cellular context. HEIST  achieves this by performing both intra-level and cross-level message passing to utilize the hierarchy in its embeddings and can thus generalize to novel datatypes including spatial proteomics without retraining. HEIST is pretrained on 22.3M cells from 124 tissues across 15 organs using spatially-aware contrastive and masked autoencoding objectives. Unsupervised analysis of HEIST embeddings reveals spatially informed subpopulations missed by prior models. Downstream evaluations demonstrate generalizability to proteomics data and state-of-the-art performance in clinical outcome prediction, cell type annotation, and gene imputation across multiple technologies.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes "HEIST" a hierarchical graph transformer model for spatial omics data. This model consists of two pars: upper level (cell-cell) and lower level (gene-gene co-expression within a cell). The model was trained on constraining loss and masked autoencoding loss.

### Strengths
- Rigorous design to account for cell-cell spatial information and gene-gene co-expression signals. 
- Experiments demonstrated HEIST's superior performance in several downstream tasks.

### Weaknesses
- Contribution is vague. Computational novelty (hierarchical graph transformer) is a bit limited, as the proposed model is somewhat a standard method. Then it should have some contribution from the biological side. But biological contribution is also limited. The downstream task analysis (e.g., tissue classification, gene imputation, cell type annotation) is also very routine and limited in claiming biological contribution as a “foundation” model like scGPT. 
- Not sure HEIST is a truly foundation model in that its zero-shot performance is very low. The claim of “foundation model” might be overstating. HEIST focuses on validating its biological realism (e.g., how spatially grounded embeddings work in downstream tasks), but it fails to show generalizability to unseen tasks in a zero-shot setting. 
- Training size is large and I appreciated that, but it’s still limited in the number of organs and tissues to claim the model as “foundation model”.

### Questions
see weakness for improvement

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
5

### Summary
The paper introduces HEIST, a hierarchical graph foundation model for spatial omics that jointly models a spatial cell graph and cell type specific gene co-expression graphs, with bidirectional cross-level message passing. HEIST is pretrained on a large corpus (reported as 22.3M cells across 124 tissues/15 organs) and evaluated on cell clustering, cell-type annotation, gene imputation, and tissue-level clinical outcome prediction, including transfer to spatial proteomics without retraining.

### Strengths
- A clear hierarchical formulation that couples intra-cell gene programs with inter-cell spatial context via cross-level message passing.  
- Ambitious pretraining scope (22.3M cells across multiple organs).
- Demonstrates SOTA-level results on clustering, annotation, imputation, and outcome prediction, with ablations highlighting the contributions of hierarchy, cross-level passing, and loss design.

### Weaknesses
Although the idea is interesting, several components are not clearly motivated, and some key method and experiment settings are missing.

### Questions
1. How does the use of MAGIC denoising before training affect model performance? Could this amplify batch effects or introduce spurious correlations?

2. The model employs global attention on the cell graph. What is the typical sequence length per slide, and how is the quadratic cost of O(n^2) handled in practice (batching, sparsification, etc.)?

3. For model initialization:
- How are cell embeddings defined as \mathrm{PE}+P given the mismatch between 128-d PE and 2-d coordinates?
- How are gene embeddings defined as \mathrm{PE}+X, where X is a scalar expression?

4. In the clustering task, why are classical baselines such as GraphST and STAGATE excluded?

5. Could clustering the embeddings from SCGPT-spatial or CellPLM yield results comparable to HEIST’s clusters in Fig. 3C?

6. In the gene imputation task, what is the number of genes per dataset and the fraction masked?

7. For tissue classification, how are tissue embeddings derived from cell embeddings?

8. Why do some baselines (SCGPT-spatial, CellPLM) perform worse in the aligned setting than in the unaligned one (Table 2)? Isn’t alignment expected to improve performance?

9. In the cell-type annotation task, how are data splits defined, and how is fine-tuning performed?

10. The paper notes SCGPT-spatial’s strong performance is explained by pretraining on MERFISH lung cancer data. For the brain dataset, HEIST achieves nearly 0.99 accuracy. Could this also result from pretraining overlap with brain data?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a graph foundation model for spatial transcriptomics and proteinomics data. In this work, two two-level graphs, e.g., the spatial cell graph and the gene co-expression graph, are introduced into a hierarchical graph transformer foundation model for embedding learning. Experiments on various datasets are conducted to study the effectiveness of the proposed method.

### Strengths
1. A graph foundation model is proposed for analysing spatial transcriptomics and proteomics data.

2. The method incorporates two hierarchical graph levels — the spatial cell graph and the gene co-expression graph — within a hierarchical graph transformer framework for effective embedding learning.

3. Extensive experiments on multiple datasets are conducted to evaluate the effectiveness of the proposed approach.

### Weaknesses
1. The detailed descriptions of the spatial cell graph and the gene co-expression graph are insufficient and should be clarified.

2. Figure 2 does not effectively illustrate the main concept or workflow of the HEIST architecture.

3. Regarding the contrastive learning design, the c–c, g–g, and c–g objectives are formulated separately. It is unclear whether combining these formulations in a joint loss function might lead to a decrease or not in performance.

4. Concerning the weighted sum of the contrastive and autoencoder reconstruction losses, it remains unclear how the weights are optimised or adaptively learned to achieve the best balance.

5. The computational complexity and reconstruction (or restoration) efficiency of the proposed method, as well as those of all compared baselines, should be analysed and discussed in detail.

### Questions
See Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces HEIST, a hierarchical graph transformer that couples spatial cell-cell graph with per-cell gene co-expression graph with  bidirectional message passing between the two. The training dataset is diverse, the model is trained on a combination of contrastive and MAE loss, and the authors show SoTA performance on a range of downstream tasks.

### Strengths
- Modeling spatial cell graphs with gene co-expression networks seems well-motivated, as previous work emphasized either one
- The inference time speed-up over other methods seems good
- Downstream results seem promising

### Weaknesses
Weaknesses/Questions:
- How is $\tau$ chosen? Is it different across cell-types? How does varying $\tau$ affect the results?
- What is $\sigma$ in L240?
- It would be interesting to see if  the model is capable of identifying known inter-cell interactions. Currently biological interpretability analysis seems limited 
- While the loss seems interesting, I am slightly skeptical if the MAE + CL work together; how does perform change on tasks if you train on either? 

Overall, I think the paper presents a strong and well-motivated approach. I would lean towards accept.

### Questions
Mentioned with weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
