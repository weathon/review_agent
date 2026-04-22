# scDFM: Distributional Flow Matching Model for Robust Single-Cell Perturbation Prediction

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 2, 4, 6

## Abstract
A central goal in systems biology and drug discovery is to predict the transcriptional response of cells to perturbations. This task is challenging due to the noisy, sparse nature of single-cell measurements and the fact that perturbations often induce population-level shifts rather than changes in individual cells. Existing deep learning methods typically assume cell-level correspondences, limiting their ability to capture such global effects.
We present **scDFM**, a generative framework based on conditional flow matching that models the full distribution of perturbed cells conditioned on control states.
By incorporating an MMD objective, our method aligns perturbed and control populations beyond cell-level correspondences. 
To further improve robustness to sparsity and noise, we propose the Perturbation-Aware Differential Transformer architecture (PAD-Transformer), a backbone that leverages gene interaction graphs and differential attention to capture context-specific expression changes.
**scDFM** outperforms prior methods across multiple genetic and drug perturbation benchmarks, excelling in both unseen and combinatorial settings. In the combinatorial setting, it reduces MSE by 19.6\% over the strongest baseline.
These results highlight the importance of distribution-level generative modeling for robust $\textit{in silico}$ perturbation prediction. The code is available at https://github.com/AI4Science-WestlakeU/scDFM.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
scDFM is a generative framework for single-cell perturbation prediction based on conditional flow matching. To model population-level effects, it combines a flow matching objective with a Maximum Mean Discrepancy (MMD) loss. The model's backbone, the Perturbation-Aware Differential Transformer (PAD-Transformer), incorporates a gene co-expression graph to guide its attention mechanism. The method is evaluated on combinatorial genetic and drug perturbation datasets.

### Strengths
- The combination of flow matching for trajectory modeling and MMD loss for distributional alignment is a sound approach for this task.

- The PAD-Transformer architecture, which incorporates biological priors via a co-expression graph, is an interesting design choice.

- The evaluation incorporates several cell-eval metrics which is encouraged.

### Weaknesses
- MMD has already been considered in STATE [1], and flow-matching has already been covered in CellFlow [2]. Therefore the novelty needs to be carefully positioned. It is unclear why performing flow matching in the original space is a meaningful novelty and improvement over e.g. cell-flow.
- It is unclear how the results compare to CellFlow, given its similarity. Also, the paper’s citation on CellFlow [2] seems wrong.
- It is unclear how multi-step generation increases performance.
- There seems to be a discrepancy from [3]. In particular, in [3], the additive baseline performs better than Geneformer, but Fig. 3 here seems to show an opposite trend. Furthermore, it is unclear how the results compare to well-established benchmarks in [3] on unseen perturbation and drug perturbations.
- The intuition on the gene attention mask remains unclear. The exact same argument could be applied to image or language data, but bi-directional attention without masking is still widely adopted.

[1] Adduri, Abhinav K., et al. "Predicting cellular responses to perturbation across diverse contexts with State." bioRxiv(2025): 2025-06.
[2] Klein, Dominik, et al. "CellFlow enables generative single-cell phenotype modeling with flow matching." bioRxiv (2025): 2025-04.
[3] Ahlmann-Eltze, Constantin, Wolfgang Huber, and Simon Anders. "Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines." Nature Methods (2025): 1-5.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes scDFM, a conditional flow-matching model for single-cell perturbation prediction. The method extends standard flow-matching by adding a Maximum Mean Discrepancy (MMD) loss to align generated and empirical cell-state distributions and introduces a Perturbation-Aware Differential Transformer (PAD-Transformer) with a gene–gene co-expression attention mask. The model is evaluated on the Norman and ComboSciPlex datasets in both additive and hold-out regimes. Results show modest improvements over previous methods such as GEARS, CPA, Geneformer, and scGPT.

### Strengths
1. The paper is well-written and clear, with a strong visual presentation and reproducibility statement.
2. Applying flow matching directly in the expression space is a reasonable and technically clean adaptation of continuous generative models to the single-cell domain.
3. The idea of incorporating a population-level regularizer (MMD) is conceptually sound and aligns with the motivation to capture distributional, rather than per-cell, perturbation effects.
4. The inclusion of differential attention and gene-graph masking demonstrates awareness of biological structure.

### Weaknesses
1. Limited novelty and incremental contribution

Flow matching for biological state modeling has already appeared in multiple works. The MMD term is a straightforward sample-based regularizer with no new theoretical or algorithmic insight. The PAD-Transformer largely reuses existing building blocks (Differential Transformer + GEARS-style gene-masking). As a result, the paper reads as a combination of known components rather than a fundamentally new modeling principle.

2. Outdated benchmarking and missing key baselines

The evaluation follows the GEARS-style additive/holdout split, which has since been shown to be insufficient due to expression-vector overlap between training and test sets. Recent frameworks such as PertEval and Systema explicitly address these issues; none of these are considered here. Furthermore, the paper cites but does not evaluate against the most relevant contemporary models (CellFlow, State), making it difficult to assess progress over the true state of the art.

3. Lack of analysis of computational cost and scalability

Flow matching and MMD introduce non-trivial overhead compared to autoencoder or diffusion models, yet there is no runtime or memory comparison. Without such analysis, it is unclear whether the method is practical for larger or multi-tissue datasets.

### Questions
See my weaknesses

### Soundness
2

### Presentation
4

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
This paper represents a generative framework called scDFM based on conditional flow matching that predicts response to cellular perturbation (genetic and molecular). There are three parts to scDFM framework: (1) the control flow matching where it models the transition from initial state of the cell lines to perturbed state. (2) multi-kernel MMD regularizer which ensures population-level fidelity. (3) backbone design which is a perturbation aware transformer which addresses the noisy, sparse, and high dimensional
nature of the data. Finally, the model is evaluated on two tasks of genetic and molecular perturbation on two datasets: Norman and Sciplex3 and it marginally outperforms the baselines.

### Strengths
- The paper is well written and easy to understand.
- This paper addresses two interesting and important questions in a single framework: genetic and molecular perturbation.
- The framework uses a biological prior which strengthens the model.
- The authors use flow matching instead of diffusion models and autoencoder (like prior methods) which is an interesting architectural choice and the reason behind it is sound .

### Weaknesses
The main weakness regarding this paper is Section 4. Experiments:
- The experimental results are not strong. In most cases, scDFM just barely outperforms the baselines.
- The baselines used in this paper are not the latest and best in the field.
- The dataset is limited and since the model is not showing strong results, it is not clear how scDFM would perform on other datasets. 

The framework for molecular perturbation has some limitations:
- It cannot be generalized to unseen molecules.
- The experiments are done only on a subset of drugs in sciplex3

### Questions
- Are the results stated in the paper on the final prediction or the delta (prediction - control) ? Could you please report the eval on delta as well?
- Why dataset Sciplex3 is called ComboSciPlex in the paper? 
- Is there an ablation study where we could see the benefit of flow matching against diffusion models?
- Could you explain the experimental setting ? For example, how many genes are trained test on? do you only look at HVGs?
- I'm still not convinced why just predicting the mean expression vector is not good enough. In fact, the experimental results shows many of the mean based models have very strong results. Is it possible to design an experiment that shows mean expression vector is not enough?

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
This paper presents scDFM, a novel generative framework for predicting the transcriptional response of single cells to perturbations. It models the perturbation as a continuous-time flow that transforms the control cell distribution into the perturbed one, and it incorporates an MMD loss to force the entire distribution of generated cells to match the ground-truth distribution. The authors demonstrate through experiments on genetic and drug perturbation datasets that scDFM outperforms existing baselines.

### Strengths
The paper is well written and easy to follow. The idea is original.

### Weaknesses
The model does not learn the gene-gene interaction network but it is given it as a biologically grounded prior. This graph is constructed from simple absolute Pearson correlation on the training data, which prevents the model from discovering novel, non-obvious, or non-linear gene relationships that aren't captured by basic correlation.

The flow matching framework learns a path from control to perturbed. As the paper acknoledges, it uses a simple linear interpolant as the reference path, which is a significant oversimplification. Real biological processes follow complex, non-linear manifolds, and this assumed path may not be biologically realistic.

### Questions
1. The ablation study shows that removing the graph prior worsens performance. Does this suggest the model is critically dependent on this prior, limiting its ability to discover novel, non-obvious gene interactions that are not captured by simple correlation? How would the model perform if the biological prior was based on a more sophisticated measure than Pearson correlation?

2. What is the computational trade-off of using MMD?

### Soundness
3

### Presentation
3

### Contribution
3
