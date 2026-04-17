# Biology-Guided Prototype Booster: Enhancing Latent Representations of Foundation Models for Gene Expression Prediction

- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Spatial transcriptomics (ST) is a cutting-edge technology that enables the measurement of gene expression while preserving spatial context and generating detailed tissue images. However, ST technology remains time-consuming and costly. The ability to predict ST gene markers of cancer from histology-grade H&E-stained tissue images is opening new horizons for precision and personalised pathology. Despite the success of foundation models in generating general-purpose embeddings of H&E-images, these representations are not optimized for gene expression prediction and lack task-specific adaptability. To address this limitation, we introduce Biology-Guided Prototype Booster (BP-Booster), leveraging biological prior knowledge to guide the construction and training of learnable prototypes for embedding reconstruction, thereby improving gene expression prediction. We demonstrate superior performance of BP-Booster across datasets, various cancer tissue types and different ST platforms. We also show that BP-Booster can flexibly integrate various foundation models to enhance their task-specific representations, enhancing explainability and applicability in clinically relevant tasks like predicting cancer biomarkers. Code will be released upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes BP-Booster (Biology-Guided Prototype Booster), a lightweight refinement module that enhances the latent representations of pathology foundation models for the task of gene expression prediction from H&E-stained histology images. The manuscript points out the limitation that foundation model embeddings, though powerful, are not tailored for gene expression prediction and often contain task-irrelevant information.
Experiments are conducted on multiple cancer types and ST platforms, showing consistent improvements over existing methods such as ST-Net, HisToGene, BLEEP, TRIPLEX, and Stem. BP-Booster also demonstrates robustness across different foundation models and applicability to predicting specific gene expression signatures.

### Strengths
- The paper makes a clear statement that general-purpose pathology embeddings are sub-optimal for tasks requiring alignment with gene expression data and the practice to solve the mis-alignment is to some extent of great importance to the community.
- The biologically-guided prototype learning with cross-attention and gating is an intuitive yet effective way to refine embeddings without re-training the foundation model.
- Extensive comparisons with SOTA methods on multiple datasets and ST technologies are provided and the results are satisfactory.

### Weaknesses
- The authors seem to misuse the tex commands of \citep{} and \citet{}, leading to incorrect citation formats. Please consider to correct these commands in a revised version.
- While the paper motivates BP‑Booster well from a computational pathology perspective, conceptually the approach can be interpreted as a variant of feature transformation / domain adaptation. The module takes source H&E patch embeddings from a frozen foundation model and maps them into a target latent space aligned with gene expression outputs using prior knowledge. This idea is related to existing techniques in domain adaptation, multimodal alignment, and feature recalibration. However, the manuscript fails to discuss these related works in the manuscript.

### Questions
- For the readers' information, have you tried fine-tuning the foundation model encoder jointly with BP-Booster? Would this further improve results or lead to overfitting on small datasets?
- How sensitive is BP-Booster to the number of prototypes and prototype dimension? Is there a risk of overparameterization in smaller datasets?
- My deepest concern towards this method is the positioning. Can you elaborate your method's contributions under the context of the concepts mentioned in the section of weaknesses? If not relevant, why the practice is unique? If yes, why existing methods cannot solve the issue?

### Soundness
3

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
This paper proposes BP-Booster, a small module that can be plugged in between a frozen pathology foundation model and a downstream regressor for spatial transcriptomics gene‑expression prediction from H&E images. BP‑Booster uses prototype‑guided cross‑attention (prototypes query image embeddings) and gating mechanisms to shape a refined latent representation. Prototypes are either randomly initialized or initialized with gene programs built by intersecting spatially variable genes (SVGs) with MSigDB pathways, then aggregating expression across spatial locations. The experiments show significant performance improvement on Hest-1k and demonstrate the practical value of this module.

### Strengths
1. The paper introduces a clear and modular idea: Using prototypes as queries over frozen embeddings and the auxiliary regression/reconstruction losses are simple and easy to implement. 
2. The cross‑model analysis is helpful to readers choosing backbones and also demonstrating the proposed methods work for different backbones

### Weaknesses
1. While I understand the problem setting is frozen image embedding + Ridge regression to predict the expression, I’m concerned the reported gains may largely come from additional params introduced by PGCA rather than from biological priors or prototype priors. Ridge cannot model nonlinearity, so a fair ablation should replace ridge with a parameter-matched MLP adaptor (same input/output dims and similar parameter count/FLOPs as PGCA) to isolate the effect of attention vs. capacity.
2. Improvements over a PCA adaptor are modest (Table 3) and not particularly compelling on their own.
3. Gene-set selection is under discussed. Relying only on MSigDB leaves some questions: gene sets vary widely in size and scope; what’s the principled selection strategy? With a ≥10-gene overlap threshold against SVGs, many pathways may pass by chance. For ST specifically, panels are often hand-curated, how does the method account for that?

### Questions
See my Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces Biology-Guided Prototype Booster (BP-Booster), a method designed to optimize H&E image embeddings from foundation models for improved task-specific adaptability in gene expression prediction. BP-Booster is based on a Q-former architecture, with a prototype initialization strategy proposed for the Q-former. Experimental results demonstrate that BP-Booster achieves better Pearson correlation coefficient (PCC) and mean squared error (MSE) performance compared to selected baselines, using different foundation models (UNI, CONCH, and Virchow2) as feature extractors.

### Strengths
- The paper focuses on an important problem, adapting representations from pre-trained H&E image foundation models for gene expression prediction.
- The paper shows better performance than the selected methods on the evaluated datasets.

### Weaknesses
- Unclear contribution

Adapting or fine-tuning parameters of foundation models typically enhances their task-specific performance. In the proposed BP-Booster, learnable parameters are added on top of the frozen foundation models to improve adaptability; however, the extent of this contribution to the overall task performance is not clearly analyzed. The method is closely related to parameter-efficient fine-tuning (PEFT), or even direct fine-tuning, but the paper does not provide a detailed study or comparison with existing PEFT approaches. Although the paper demonstrates better performance than the selected methods on the evaluated datasets, it does not clearly evaluate how the proposed method improves existing related approaches.

- Unclear setting:

There exist large-scale datasets collection such as STimage-1K4M and HEST-1K. It is unclear why the proposed method was not evaluated on these broader datasets to better demonstrate its generalization capability.

Based on the setup in Table 3, it is difficult to compare the method performance with and without the GP initialization, as the comparison is not clearly isolated.

### Questions
In Eq. 12, is any weighting applied to the reconstruction loss?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces the Biology-Guided Prototype Booster, a lightweight module designed to enhance general-purpose embeddings from pathology foundation models specifically for gene expression prediction. The module uses a novel prototype-guided cross-attention mechanism, where prototypes can be initialized using biological priors.

### Strengths
- The concept of a "Biology-Guided" prototype initialization is a nice contribution. Using Spatially Variable Genes and known biological pathways to create semantically meaningful, learnable prototypes is a novel method for injecting relevant prior knowledge into the model.
- The experimental validation is extensive and provides strong evidence for the method's generalizability. The authors demonstrate that BP-Booster consistently outperforms not only strong published baselines but also a strong PCA baseline across nine different foundation models.

### Weaknesses
- The paper's core biological contribution is the Gene Program-guided initialization. However, the results show that the simpler Random initialization is often competitive and, in some cases, even better (e.g., Virchow2 on PRAD in Table 2 , or CONCH on SKCM in Table 3). This seems to weaken the "Biology-Guided" claim. Can the authors elaborate on why a random initialization is so effective and sometimes superior?
- The methodology uses two different sets of genes: Spatially Variable Genes for the prototype initialization and Highly Variable Genes for the regression loss. The rationale for this split is not entirely clear. Why is one set better than other for initialization and training supervision? Have the authors tried to flip them or even find a common set?

### Questions
- The final prediction pipeline appears to be a two-stage process: 1) Train the BP-Booster to generate embeddings, and 2) Train a separate regression model on these frozen embeddings. Why was this approach chosen over an end-to-end solution? Have the authors tried an end-to-end approach?
- The Gene Program-Guided Initialization, which filters MSigDB pathways by intersecting them with spatially variable genes, is a critical step. The model's performance and the number of prototypes seem highly dependent on this filtering. How sensitive is the model to this intersection threshold, and how many prototypes did this process typically generate for the datasets (IDC, SKCM, etc.)? Do different disease types benefit from different number of prototypes? What do these prototypes mean/ comprise?

### Soundness
2

### Presentation
2

### Contribution
2
