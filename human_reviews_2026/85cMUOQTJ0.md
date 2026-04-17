# A Hyperparameter Benchmark of VAE-Based Methods for scRNA-seq Batch Integration

- Decision: Reject
- Scores: 0, 2, 2, 4

## Abstract
We present the first systematic model architecture hyperparameter benchmark of variational-autoencoder (VAE)–based for single-cell RNA sequencing batch integration. We focused on models available under the scvi-tools framework, and compared the scVI, MrVI, and LDVAE models across four datasets with heterogeneous designs under two feature regimes: training with all, and utilizing only highly variable genes (HVGs). Our study executes 960 training runs spanning 120 configurations for the three models that vary latent size capacity, network depth/width, and evaluates with a comprehensive, standardized metric suite from the scib package capturing both batch removal and biological conservation (Batch ASW, PCR-batch, iLISI, graph connectivity, NMI, ARI, label ASW, isolated-label F1/ASW, cLISI, and trajectory conservation), qualitative analysis with UMAP and t-SNE, alongside PCA, random projection, and unintegrated baselines. We find trade-offs across datasets: scVI delivers the strongest overall integration, driven by superior batch correction; LDVAE shows dataset-specific gains in biological structure preservation; MrVI shows stability and batch correction superiority under multi-protocol datasets, however, it is more resource-intensive. Selecting for HVG features generally outperforms full-gene training for all models. Model architecture hyperparameter analysis indicates that moderate to high latent dimensionality (more than 30 dimensions) often yields the best balance, while sensitivity to latent size appears to be related to dataset heterogeneity (diverse tissues, laboratories, chemistries, and gene-coverage profiles), and larger latent spaces tend to improve batch mixing but can reduce biological conservation. We provide model and dataset-specific guidelines that translate our analysis into practical defaults and tuning rules for the practical deployment of VAE-based integration in single-cell studies.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
The authors present a systematic hyperparameter benchmark of three VAE-based methods (scVI, MrVI, LDVAE) for single-cell RNA-seq batch integration. The authors test 120 configurations on three datasets, varying latent dimensionality, network depth, and hidden width, and evaluate the learned latent representations with metrics introduced in [1].

### Strengths
- The paper addresses an important practical question. Many users use default hyperparameters without justification. This is a genuine problem, and a systematic investigation is warranted.

### Weaknesses
- The authors only test three methods. While these are definitely popular, this narrows the scope of the paper significantly.
- The authors show that HVG-selection improves method performance. This is well known in the single-cell community and has been studied previously by [1] and [2]. The authors do not cite [2] or compare their results to previous findings.
- The paper only investigates the effects of four hyperparameters. Missing: learning rate, dropout, batch size, max number of epochs, etc. It is unclear why the authors prioritized those over the others.
- The authors of scVI have done an ablation on latent space dimensions (Supplementary Figure 1) and number of epochs (Supplementary Figure 2). Contrary to the results presented on this paper, they show that there is little difference between using a 10, 20 or 30 dimensional latent space. This should be discussed.
- Training VAEs and evaluating latent representations is a noisy process. However, the authors only ran one random seed per configuration. Therefore, it remains entirely unclear how much of the results depend on randomness.
- The authors only use three datasets, all of which contain immune/PBMC cells. Therefore, it is completely unclear if these results generalize to other tissues, organisms, and modalities.
- No analysis of what dataset properties (size, batch structure, heterogeneity) predict which hyperparameter regime performs best.
- Figure 1 is copied and pasted together from other papers and websites without proper attribution, see https://docs.scvi-tools.org/en/stable/user_guide/models/scvi.html and https://academic.oup.com/bioinformatics/article/36/11/3418/5807606.
- 40+ pages is excessive for the results presented. The content could be substantially condensed. Especially the tables in the appendix are difficult to navigate.

[1] Luecken, M.D., Büttner, M., Chaichoompu, K. _et al._ Benchmarking atlas-level data integration in single-cell genomics. _Nat Methods_ **19**, 41–50 (2022).

[2] Zappia, L., Richter, S., Ramírez-Suástegui, C. _et al._ Feature selection methods affect the performance of scRNA-seq data integration and querying. _Nat Methods_ **22**, 834–844 (2025)

### Questions
- How well do the results presented generalize to different datasets?

- Given that VAE training and batch integration evaluation are stochastic, what is the variance across runs?

- What specific properties (batch structure, cell type diversity, sample size, technology) of a dataset lead to which hyperparameters performing best?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The authors systematically benchmarked three VAE-based models (scVI, MrVI, and LDVAE) for single-cell RNA-seq data integration, evaluating their performance in batch effect removal and biological information preservation. Rather than performing active hyperparameter optimization, they examined how predefined architectural settings, specifically latent dimension size, network depth, and network width, influence integration quality. The results show that scVI achieves the strongest overall integration, primarily through superior batch correction; LDVAE offers dataset-specific advantages in maintaining biological structure; and MrVI demonstrates stable performance across conditions. Based on these findings, the authors provide model- and dataset-specific guidelines to aid effective integration in future single-cell studies.

### Strengths
The paper presents a well-structured and focused benchmarking study, offering a clear and systematic comparison of how different architectural hyperparameters, specifically latent dimensionality, network depth, and network width, affect the performance of three prominent VAE-based models for single-cell RNA-seq data integration. This design allows the authors to isolate the impact of model architecture on integration quality, making the analysis both controlled and interpretable.
A major strength of the work lies in its practical relevance. By identifying distinct behaviors across models, such as scVI’s strong batch correction, LDVAE’s superior preservation of biological structure, and MrVI’s stability, the study provides actionable insights for practitioners selecting models and configurations for specific datasets or analytical goals.
The authors also employ a comprehensive and balanced evaluation strategy, incorporating metrics that assess both batch effect removal and biological conservation. This dual perspective ensures that integration performance is not judged solely by technical alignment but also by the biological fidelity of the resulting latent representations.
Finally, the study’s proposal of model- and dataset-specific guidelines represents a valuable practical contribution. These recommendations translate the benchmarking results into concrete guidance for researchers, enhancing the reproducibility and efficiency of future single-cell data integration efforts.

### Weaknesses
While the study is carefully designed and clearly presented, it has several limitations that slightly constrain the generality of its conclusions. First, the benchmark focuses on a relatively narrow set of architectural hyperparameters, namely, latent dimension size, network depth, and network width. Although these parameters are indeed central to model architecture, other influential factors such as learning rate, dropout rate, or batch size are not explored. Including a broader range of hyperparameters might have provided a more complete picture of model behavior and performance sensitivity.
Second, the comparison is limited to three VAE-based methods—scVI, MrVI, and LDVAE. While these are representative and widely used in the field, the absence of non-VAE or hybrid integration approaches restricts the broader applicability of the findings. Incorporating alternative paradigms, such as graph-based or contrastive learning models, could have offered valuable comparative insight.
Another limitation lies in the scope of the datasets. Although the study includes multiple scRNA-seq datasets, these primarily represent moderate-sized, well-characterized benchmarks. The results may not fully capture how the models behave in more complex integration scenarios, such as extremely large datasets, multi-species comparisons, low sequencing-quality datasets, imbalance datasets, or data with severe sparsity.
The paper also provides limited discussion of computational efficiency. While the integration quality is thoroughly evaluated, readers are given little information about the runtime or resource demands of different architectures. Since computational feasibility often influences method selection in practice, this omission slightly reduces the practical value of the results.
Finally, the study gives relatively little attention to latent space interpretability. Although the quantitative metrics effectively evaluate integration quality, they do not reveal how architectural changes might affect the biological meaning or interpretability of the latent representations. A deeper exploration of this aspect could further strengthen the connection between technical performance and biological insight.

### Questions
1. How consistent are the observed performance patterns (e.g., scVI’s strong batch correction, LDVAE’s biological preservation) across datasets with varying cell type diversity or batch imbalance?
2. Why were only latent dimension, depth, and width selected as benchmarked hyperparameters? Could other settings (e.g., learning rate, regularization) also play major roles in integration quality?
3. Are the proposed configuration guidelines applicable to unseen datasets, or should users expect to re-evaluate settings for each new dataset?
4. Can the authors quantify the trade-off between batch effect removal and biological structure preservation for different model configurations?
5. How do variations in latent dimension affect downstream analyses (e.g., clustering consistency, trajectory inference)?
6. How do the authors calculate the overall score? Is it the average of the biological conservation and batch correction?

### Soundness
2

### Presentation
2

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
This paper provides a benchmark of popular variational models in single-cell RNA-seq analysis, for the task of batch integration, sweeping through 120 hyperparameter configurations in a controlled way with 720 training runs. The paper recommends scVI as the go-to model for most datasets, at the same time highlighting that the hyperparameters of scVI do need to be tuned (as opposed to current practice of using the default hyperparameters).

### Strengths
- This paper provides the first benchmark of hyperparemter configurations for popular variational models in single-cell RNA-seq analysis, challenging a common assumption that the default hyperparmeters in scvi-tools can be used directly without tuning.

### Weaknesses
- For a benchmark paper, the datasets focus entirely on immune-related cells. Therefore, it is difficult to assess whether the conclusions from this paper can transfer to datasets with other cell types. I recommend the authors to include more datasets to enhance the diversity of the cell types considered.

- Do the t-SNE and UMAP plots offer additional insights compared to the numeric evaluation metrics? Currently, they seem to agree with the evaluation metrics, so it's unclear whether including these plots in a benchmark offers additional insights.

### Questions
- Do the hyperparameter recommendations transfer well to new datasets (especially a dataset not related to immune cells)? For example, can you fit a scaling law with respect to key hyperparameters and show that the scaling law also applies to new datasets?

- Since the variational models considered have modest computational requirement and mature tooling, a user can simply tune the hyperparameters for a given dataset of interest. What does this benchmark study offer in this setting?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents the first systematic study of how hyperparameters affect VAE-based models for single-cell RNA sequencing (scRNA-seq) batch correction. The authors benchmark scVI, MrVI, and LDVAE: three widely used probabilistic generative models implemented in scvi-tools across three heterogeneous datasets. The hypers they consider are: latent dimensionality, network depth, and hidden width, under two feature regimes: all genes vs. highly variable genes (HVGs). Performance is assessed using a standardized smetric suite measuring both batch removal and biological conservation (e.g. NMI, ARI, label ASW, cLISI, trajectory conservation), along with qualitative UMAP/t-SNE visualization. They provide practical defaults and rules of thumb for hyper initialisation in these VAE methods for scRNA.

### Strengths
- The paper performs the first systematic hyperparameter benchmark of VAE based integration models for scRNA-seq, across multiple datasets with a fully standardized pipeline.
- The study extracts actionable insights like latent dimensionality 30–50, shallow depth, HVG features.
- By harmonising preprocessing and feature selection they isolate model behaviour, the steps taken are reasonable. This helps achieve cross-modal comparability. 
- The planned public release of all the models may provide a useful reference resource for researchers in single cell generative models.

### Weaknesses
- To fit ICLR’s bar, the paper should Abstract beyond the domain, argue that the results reveal general principles of VAE behavior under latent capacity scaling or dataset heterogeneity rather than just focus on the narrow setting of scRNA-seq. I would consider such a dataset-specific comparison better suited for bioinformatics journals.

### Questions
- While scVI generally outperforms the others, were compute time and memory costs explicitly compared? For example, does the performance gain from higher latent dimensionality justify the additional training cost?
- LDVAE is designed for interpretability, yet the paper primarily quantifies integration performance. Can the authors comment on whether the more interpretable linear decoder representations preserved biologically meaningful gene programs even when quantitative integration scores were lower?
- Since VAE training is inherently stochastic .. how sensitive are the reported results to random seeds? Were multiple training replicates performed per configuration, and if not, could variance across runs materially affect the ranking of hyper. settings or models? Couldn't find this information anywhere easily.

### Soundness
2

### Presentation
2

### Contribution
2
