# GenAR: Next-Scale Autoregressive Generation for Spatial Gene Expression Prediction

- Avg Score: 4.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 6, 4

## Abstract
Spatial Transcriptomics (ST) offers spatially resolved gene expression but remains costly. Predicting expression directly from widely available Hematoxylin and Eosin (H&E) stained images presents a cost-effective alternative. However, most computational approaches (i) predict each gene independently, overlooking co-expression structure, and (ii) cast the task as continuous regression despite expression being discrete counts. This mismatch can yield biologically implausible outputs and complicate downstream analyses. We introduce GenAR, a multi-scale autoregressive framework that refines predictions from coarse to fine. GenAR (a) clusters genes into hierarchical groups to expose cross-gene dependencies, (b) models expression as discrete token generation over a fixed vocabulary of integer count tokens to directly predict raw counts, and (c) conditions decoding on fused histological and spatial embeddings. From an information-theoretic view, the discrete formulation operates directly on the physical count scale, and the coarse-to-fine factorization aligns with a principled conditional decomposition. Extensive experimental results on four ST datasets across different tissue types demonstrate that GenAR achieves state-of-the-art performance, offering potential implications for precision medicine and cost-effective molecular profiling. Code will be publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes and auto-regressive-based approach to predict gene expression from HE images. This model can predict gene expressions by clusters and also raw couts. Although the model is clear, I have some questions.

### Strengths
The prediction task is well defined, and the prediction of raw count sounds interesting.

### Weaknesses
There are several constraints and problems in experimental settings and interpretation.

1. I do not see the contribution of predicting raw counts. Current ST data analysis always needs to normalize the data to avoid the problems of sequencing depth. What is the accurate rate of GenAR in modeling sequencing depth? Other baseline models predict normalized data. How to ensure that the benchmarking system is fair?

2. The number of datasets is quite low. Methods such as STFlow, as well as datasets such as HEST/STImage, have considered 8 datasets or more from different tissues and diseases. The authors should include more data in the evaluation stage.

3. In Table 4, PCC-10 reaches the highest correlation, which is a bit weird. For example, predicting DEG is always difficult than predicting whole-genome, as some genes have nearly constant expression levels and can improve prediction metrics. What is the performance of prediction based on marker genes or highly variable genes?

4. How to determine the resolution of selecting clusters in predicting gene expression? If you change the resolution, will the detected gene pathways change? What if we randomly select genes into different clusters? Will the performance become lower or higher?

5. It seems that there are not many component-level contributions or innovations in model design. I believe the authors directly modify VAR for gene expression prediction, which lacks motivation. Moreover, I do not find hyperparameter details in the benchmark. How to ensure that the benchmark is fair? Baselines and proposed methods should be tuned to the best level.

### Questions
Please see the weaknesses.

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
3

### Summary
The paper presents a novel computational framework called GenAR, designed to predict spatial gene expression directly from widely available H&E-stained histology images.
 The core innovation of this method lies in reformulating gene expression prediction as a multi-scale autoregressive discrete token generation task.
 Unlike existing approaches that predict each gene independently or treat the problem as a continuous regression task, GenAR clusters genes into a hierarchical structure from coarse to fine and autoregressively predicts expression at each scale to model inter-gene dependencies.
 In addition, the method directly predicts raw gene expression counts, avoiding potential biases introduced by common logarithmic transformations and preserving the biological interpretability of the data.
 Extensive experiments on four distinct spatial transcriptomics datasets demonstrate that GenAR achieves state-of-the-art performance across multiple evaluation metrics.

### Strengths
Applying an autoregressive generative model to gene expression prediction, the paper presents comprehensive experimental evaluations and in-depth ablation studies, demonstrating strong credibility.
The direct use of raw count data aligns well with the requirements of downstream biological analyses.
The manuscript is well organized, and the figures are visually appealing and clearly presented.

### Weaknesses
1. The method adopts a multi-scale autoregressive generative framework, which may lead to increased computational and memory overhead as the scale grows.
 The paper does not provide a detailed discussion of training and inference time, nor a comparison with baseline methods in terms of computational efficiency.
2. To ensure fair comparison with other approaches, the final evaluation still applies a log2 transformation to the predicted results.
 This, to some extent, diminishes the advantage of directly predicting raw counts during the evaluation stage.

### Questions
None

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
5

### Summary
The paper introduces GenAR, a next-scale autoregressive model for predicting spatial gene expression (counts) directly from H&E patches plus spot coordinates. Genes are clustered into hierarchical groups (coarse to fine). At each scale the model predicts discrete “tokens” for group summaries and, at the last scale, per‑gene counts. The decoder is a causal Transformer conditioned on a fused histology and spatial embedding; gene‑identity embeddings modulate activations. The authors claim that (i) modeling counts as discrete tokens preserves biological meaning and avoids “log‑induced biases,” and (ii) the coarse to fine factorization better captures cross‑gene dependencies. Experiments on HEST1k show great performance improvement

### Strengths
1. This is a very interesting work of using autoregressive modeling [1] on predicting the H&E images
2. Benchmark on HEST-1k shows significant performance improvement

   [1] Tian, Keyu, et al. "Visual autoregressive modeling: Scalable image generation via next-scale prediction." *Advances in neural information processing systems* 37 (2024): 84839-84865.

### Weaknesses
1. The motivation for modeling raw molecule counts is unconvincing (lines 65–67). In transcriptomics, it is standard to apply log1p and library-size normalization because (a) sequencing depth varies across experiments so total counts differ by sample, and (b) variance across genes is high, log transforms help stabilize it. Using raw counts likely hurts transferability across cohorts and may, in practice, limit performance. (There are also no cross-dataset transfer experiments to counter this concern either.) Re line 73, I'm not understanding why raw counts are better for downstream analysis. 

2. I assume the authors make the design decisions following AVR [1], where outputs are tokens in a codebook suitable for cross-entropy optimization. But for count prediction, CE over discretized counts does not provide a direction-aware penalty (over- vs under-prediction) in the way regression losses (eg, MSE, Poisson/NB deviance) do. Im concerned about error calibration and directionality.

3. Several prior approaches (not essentially H&E -> gene prediction) already explore tokenizing molecular counts via binning, such as scGPT/scGPT-spatial [2, 3], and often trained with regression losses on binned targets

4. The benchmark is only done on one slide, no cross-validation etc.


[1] Tian, Keyu, et al. "Visual autoregressive modeling: Scalable image generation via next-scale prediction." *Advances in neural information processing systems* 37 (2024): 84839-84865.
[2] Cui, Haotian, et al. "scGPT: toward building a foundation model for single-cell multi-omics using generative AI." Nature methods 21.8 (2024): 1470-1480.
[3]  Wang, Chloe, et al. "scGPT-spatial: Continual pretraining of single-cell foundation model for spatial transcriptomics." bioRxiv (2025): 2025-02.

### Questions
See the Weaknesses

### Soundness
2

### Presentation
3

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
This paper introduces a multi-scale autoregressive framework for predicting spatial gene expression from H&E images. 

Gene expression counts subject to capture efficiency, amplification bias, and technical noise essentially noisy observations of underlying continuous mRNA abundances. The Poisson or negative binomial distributions commonly used in analysis treat counts as arising from continuous rate parameters. By discretizing to a fixed vocabulary and predicting tokens directly, the authors are imposing artificial quantization on data that is already a noisy discretization of continuous biology. The claim that this avoids log-induced bias is misleading—the log transformation is specifically designed to stabilize variance in count data and is standard practice precisely because it better reflects the multiplicative nature of biological processes.

### Strengths
- The multi-scale formulation provides a principled coarse-to-fine decomposition that progressively refines predictions from global transcriptional context to gene-level precision. The technical implementation combining causal transformers, FiLM conditioning on histological embeddings, gene identity modulation, and scale-specific loss functions is clearly specified and reproducible. The ablation study effectively validates that removing multi-scale structure causes substantial degradation, demonstrating the hierarchical architecture captures useful inductive bias beyond simply adding parameters.
- The paper evaluates on four well-chosen ST datasets spanning different species, spatial resolutions, tissue types, and pathological states. Consistent improvements across all datasets suggest the method captures generalizable spatial-molecular patterns rather than overfitting to specific tissue characteristics, which strengthens claims of robustness compared to prior work reporting on single datasets.
- The paper compares against five recent strong baselines plus foundation model ablations using complementary metrics. The evaluation captures both top-performing gene predictions and overall accuracy, with qualitative validation showing GenAR better preserves spatial expression boundaries and heterogeneity compared to oversmoothed baseline predictions. Testing on both cancer tissues and healthy tissues demonstrates the method's applicability across biological contexts rather than being specific to pathological states.

### Weaknesses
- The codebook-free framing is not accurate. The method still uses a fixed vocabulary, which is functionally equivalent to a codebook—it just doesn't learn it via VQ-VAE. The authors claim this avoids encode-decode reconstruction loss but this is not an advantage when the prediction target is fundamentally continuous. Authors simply replace one form of discretization (learned codebook) with another (fixed binning), without justification for why fixed binning is superior.
- The evaluation is fundamentally compromised by the log transformation applied post-prediction. The authors train on raw counts but then log-transform predictions to ensure consistent evaluation with other models. This makes it impossible to assess whether improvements come from the discrete formulation or simply from having more parameters and a different loss function. The claim to directly predict raw gene expression counts is not meaningful when evaluation requires transformation back to the continuous space that baselines operate in natively.
- The multi-scale loss function averages losses across K scales with equal weight, treating coarse group-level predictions as equally important as final gene-level predictions. This is arbitrary and not justified. Why should predicting 4 gene groups contribute 1/K to the loss when predicting 200 individual genes also contributes 1/K? The soft KL loss for intermediate scales uses temperature-smoothed targets but no temperature value or smoothing schedule is provided.
- The heteroscedastic Gaussian NLL with variance is applied only at the final scale. The choice of this specific variance form and the values of α and β are not justified. This looks like it was designed to match the mean-variance relationship of negative binomial data, which undermines the entire premise of avoiding distributional assumptions through discrete modeling.
- Table 1 and 2 show improvements but the margins are modest and could easily result from having more parameters, longer training, or the specific gene clustering used. The ablation in Table 3 shows w/o Multi-scale has dramatically worse performance, but it's unclear what this baseline actually is—does it predict all 200 genes at once? The comparison is not clearly specified. The w/ Cross-entropy ablation shows worse performance, but cross-entropy is not appropriate for count data regardless, so this doesn't validate the proposed loss functions.
- The paper fails to compare against methods that properly model count data distributions. Comparison to negative binomial regression, zero-inflated models, or probabilistic frameworks like scVI adapted for spatial data is missing.

### Questions
- Can you provide evaluation metrics computed directly on raw predicted counts without any log transformation? The current evaluation applies log2 transformation to predictions before computing metrics, which makes it impossible to assess whether the discrete formulation actually provides advantages for count-based predictions. Report MSE, MAE, & correlation on the original count scale, & demonstrate that predicted count distributions possess appropriate statistical properties (e.g., mean-variance relationships) for downstream analyses like differential expression testing with DESeq2 or edgeR.
- What happens if you train a continuous regression baseline with identical architecture (same transformer depth, hidden dimensions, multi-scale supervision structure) but predict log-transformed expression directly? This ablation is critical to isolate whether performance gains come from the discrete token formulation itself or simply from architectural choices like multi-scale training, gene identity embeddings, & having more parameters. Without this comparison, it's unclear if the core conceptual contribution—discrete generation—actually matters.
- Why is k-means clustering on spatial expression patterns the appropriate way to group genes for the hierarchical structure, & have you compared against biologically-informed groupings? Genes co-localized in space do not necessarily share regulatory relationships—did authors test pathway-based groupings (Gene Ontology, KEGG), transcription factor regulons, or protein-protein interaction networks? Provide evidence that your learned gene groups show enrichment for shared biological processes or that they outperform biologically-motivated alternatives.
- What is the actual vocabulary utilization in your trained models, & how does prediction quality vary across the expression range? Report how many unique tokens out of vocab_size are actively used, whether there is mode collapse, & provide stratified results for low-expression genes (mean count <10), medium (10-100), & high-expression genes (>100). If the discrete formulation is genuinely advantageous, it should show particular benefits for lowly-expressed genes where counts are truly discrete rather than effectively continuous.

### Soundness
3

### Presentation
3

### Contribution
2
