# SAVE: A Generalizable Framework for Multi-Condition Single-Cell Generation with Gene Block Attention

- Decision: Accept (Poster)
- Scores: 8, 2, 4, 6

## Abstract
Modeling single-cell gene expression across diverse biological and technical conditions is crucial for characterizing cellular states and simulating unseen scenarios. Existing methods often treat genes as independent tokens, overlooking their high-level biological relationships and leading to poor performance. We introduce SAVE, a unified generative framework based on conditional Transformers for multi-condition single-cell modeling. SAVE leverages a coarse-grained representation by grouping semantically related genes into blocks, capturing higher-order dependencies among gene modules. A Flow Matching mechanism and condition-masking strategy further enhance flexible simulation and enable generalization to unseen condition combinations.  We evaluate SAVE on a range of benchmarks, including conditional generation, batch effect correction, and perturbation prediction. SAVE consistently outperforms state-of-the-art methods in generation fidelity and extrapolative generalization, especially in low-resource or combinatorially held-out settings. Overall, SAVE offers a scalable and generalizable solution for modeling complex single-cell data, with broad utility in virtual cell synthesis and biological interpretation.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents SAVE, a conditional generative framework for single-cell RNA sequencing (scRNA-seq) data based on a variational autoencoder with a Transformer backbone. SAVE introduces a biologically motivated gene-block attention mechanism that aggregates groups of genes into block-level tokens to capture co-regulation and pathway-level structure. Condition-specific effects are incorporated through Adaptive Layer Normalization (AdaLN), while a dual masking strategy on condition labels and latent codes enhances robustness and generalization. The framework is evaluated across three tasks—conditional single-cell generation, batch correction, and perturbation prediction and compared against established baselines including CFGen, scVI, and scDiffusion.

The paper includes a careful analysis of evaluation metrics (Wasserstein Distance, MMD, PCC, and MSE), extensive ablations on the gene-block design, and discussions of dataset- and model-specific limitations. Overall, SAVE aims to provide a unified, biologically grounded model for multi-condition single-cell data generation.

### Strengths
1. Biologically grounded architectural design.
The gene-block attention mechanism is an elegant way to encode biological priors into a Transformer framework. By grouping genes into functional or co-expression blocks, the model captures higher-level dependencies that would be difficult to learn from individual gene tokens. The accompanying ablation study (varying block size) provides strong empirical evidence that this inductive bias improves both performance and computational efficiency.

2. Rigorous and transparent evaluation.
The experiments are thorough and well-motivated. The authors provide detailed preprocessing pipelines, normalization strategies, and metric definitions, which help ensure reproducibility. 

3. Comprehensive comparisons and analyses.
The experimental section includes meaningful comparisons to strong baselines, alongside careful discussion of potential confounding factors such as size-factor grouping and normalization space. The analyses convincingly explain discrepancies between quantitative metrics and qualitative observations.

4. Good result interpretability.
Beyond standard distributional alignment, the study evaluates gene-level fidelity and provides insight into how SAVE captures biologically relevant structures, lending credibility to its claims of biological plausibility.

### Weaknesses
1. Limited evaluation scope on perturbation modeling.
While the paper demonstrates SAVE’s versatility across multiple tasks, the perturbation prediction experiments are limited to a narrow set of baselines. Comparisons to more recent flow-matching or optimal-transport–based approaches would further support claims of generalization in conditional settings.

2. Decoder design justification remains empirical.
The use of a fully connected MLP decoder is empirically justified but lacks a stronger probabilistic or biological rationale. While the arguments against negative binomial modeling are sound, exploring hybrid or auxiliary likelihoods could improve interpretability without compromising stability. For example, a flow-based decoder on top of a transformer embedding is a reasonable alternative.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a unified generative framework for scRNA-seq that (i) groups genes into semantically defined gene blocks using LLM-derived gene embeddings, (ii) applies a conditional Transformer with Adaptive LayerNorm (AdaLN) to inject multi-factor conditions (batch, cell type, disease, protocol, etc.), and (iii) couples this with flow matching in latent space to enable flexible conditional generation, including unseen condition combinations.

### Strengths
1. The paper correctly notes that many single-cell generative models treat genes as flat tokens, ignoring biological structure and struggling with combinatorial conditions. SAVE’s gene-block idea is a plausible way to add structure.
2. Same architecture handles conditional generation, batch integration, and perturbation prediction. That’s attractive for labs that don’t want three separate models.

### Weaknesses
1. Gene-block construction is underspecified for reproduction. The method relies on LLM-derived gene embeddings + optimal-transport clustering to form balanced blocks, but the paper does not fully quantify: what text fields from NCBI were used, which LLM/encoder, how sensitive performance is to mistakes in gene descriptions or to species-specific vocabularies.
2. SAVE uses a VAE + Transformer over blocks + flow matching + AdaLN. That’s quite a stack. There is a note that all experiments ran on a 3090 with shared hyperparameters, but no training time, memory profile, or comparison to lighter baselines is given. For large atlases (millions of cells), this matters.
3. Evaluation axes are mostly distributional. WD/MMD and scIB show that the clouds overlap, but users often care about gene-level faithfulness for downstream DE, pathway scores, or ligand–receptor analysis. Only in the perturbation section do we see gene-level metrics (PCC, R²), extending that style of evaluation to the other datasets would strengthen the paper.
4. The paper claims “broad utility in virtual cell synthesis and biological discovery,” but doesn’t show a biological discovery enabled by SAVE.

### Questions
The paper adopts affine flow matching in latent space and combines it with classifier-free guidance, which is elegant, but there’s no comparison to latent diffusion or to a simple conditional ODE on top of the VAE to show why Flow Matching was preferred (stability? speed? better extrapolation?).

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
SAVE is a method that uses conditional flow matching on latent embedding to denoise and reconstruct cell data. The gene is first clustered with llm embedding. Similar genes are then grouped together as a gene block. Batch and cell type are used as conditional information for double conditionals and single conditional information such as cell types. The latent is generated from the gene blocks and denoised with flow matching. The final latent is then passed through the decoder to form a reconstructed gene block. The methodology contributions from SAVE are Gene Block Attention where gene embeddings are clustered before passing to a transformer block, and masked conditionals injected to the transformer processing latent information. The model was evaluated on conditional generation, batch effect correction, and perturbation effect prediction with datasets from blood, brain in human and mice.

### Strengths
The methodology which combined several ML methods is interesting and some of the results are compelling in double conditional generation. There are several datasets used across different tissues and species, which provides a stronger support for the claim on generalization. The results are carefully separated by categories of cell type for better understanding the strength and weaknesses of the method. The performance was generally strong for all three tasks.

### Weaknesses
The author should provide more quantitative information for the purpose of evaluating the effectiveness of perturbation prediction. The metric used was PCC and $R^2$ (and MSE in the appendix), which are both correlation measures instead of WD and MMD. There are also several other models that predicts perturbation effects, such as [cellOT](https://doi.org/10.1038/s41592-023-01969-x), and [meta flow matching](https://arxiv.org/abs/2408.14608). It is unclear how SAVE compares to the current SOTA on perturbation prediction. 

There are also some formatting issue including missing section hyperlink in the results, as well as unclear labelling of the main diagram. The author should proof read and correct those mistakes.

### Questions
- The link to the code repository [here](https://anonymous.4open.science/r/submit-code-3E75) all showed "file not found". Please ensure the code is available for review.
- For double conditional generation, if the batch information is provided how would out-of-distribution generation be possible for a batch never included before? As well, is this additional batch conditional included in the train/test of other baselines?
- Do you have results of the perturbation experiments in other forms than correlation metrics (and MSE in the appendix), such as MMD, WD? 
- Why is CD14+Mono perturbation result in Table 16 and Table 5 missing for scGEN? How did it fail under default experimental setup?
- For the perturbation experiment, can ablations of gene block be done on the results on Table 5? Since the genes are already grouped via the block mechanism, predicting perturbation may become much easier than non-blocked mechanism.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposed a novel generative framework for single cell gene modeling. There are three main ideas: 1. gene block attention to aggregate semantically related genes into blocks and applies a Transformer over these blocks; 2. a latent VAE + Flow Matching module for conditional generation; 3. a condition-masking training strategy. Experiments cover conditional generation (single/dual/multi-condition), batch effect correction (scIB suite), and perturbation prediction (PBMC-IFN), with consistent gains over baselines (scVI, scDiffusion, CFGen, trVAE, etc.).

### Strengths
Clear modular architecture and formulation presentation.

Gene block construction is novel and biologically motivated. 

The proposed method outperforms baseline, which well-supports the claims in the paper.

Comprehensive ablation study -- suggesting (a) removing Gene Block Attention degrades WD/MMD and (b) condition masking improves extrapolation for perturbation prediction

Impressive training efficiency by using block attention.

Good reproducibility -- code and hyperparameter settings are publicly available.

### Weaknesses
1. The biological validity of LLM-based semantic grouping from NCBI descriptions is plausible, but risks text-space biases and possible drift from functional co-expression or pathway structure. A comparison against pathway/GRN-based groupings (e.g., MSigDB, Reactome) would strengthen the claim that semantics beats alternatives. 

2. WD/MMD at the distribution level and UMAP are helpful but can be insensitive to gene-level calibration.

3. The paper adopts the affine path and CFG-style guidance; however, comparisons to diffusion-style latent generators beyond scDiffusion (e.g., alternative probability paths or ODE solvers) are limited.

4. While baselines are strong and described, more clarity on tuning parity and parameter budgets would improve fairness claims (e.g., scDiffusion vs. SAVE parameter counts/training budgets). 

5. The method motivates blocks as biologically meaningful; an analysis that profiles top genes per block, enrichment for known pathways, and how specific conditions modulate block-level attention would amplify the biological insight.

### Questions
1. How sensitive are results to the source of text embeddings (e.g., different LLMs or just GSEA gene sets) and to the block count L / block size K?

2. What is the runtime/memory profile versus scDiffusion/CFGen at parity?

3. Can you report block-level attention maps and condition-wise shifts (e.g., which blocks drive perturbation responses) to support mechanistic interpretability claims?

### Soundness
3

### Presentation
3

### Contribution
3
