# PertEval-scFM: Benchmarking Single-Cell Foundation Models for Perturbation Effect Prediction

- Decision: Reject
- Scores: 3, 6, 3, 6

## Abstract
_In silico_ modeling of transcriptional responses to perturbations is crucial for advancing our understanding of cellular processes and disease mechanisms. We present PertEval-scFM, a standardized framework designed to evaluate models for perturbation effect prediction. We apply PertEval-scFM to benchmark zero-shot single-cell foundation model (scFM) embeddings against simpler baseline models to assess whether these contextualized representations enhance perturbation effect prediction. Our results show that scFM embeddings do not provide consistent improvements over baseline models, especially under distribution shift. Additionally, all models struggle with predicting strong or atypical perturbation effects. Overall, this study provides a systematic evaluation of zero-shot scFM embeddings for perturbation effect prediction, highlighting the challenges of this task and revealing the limitations of current-generation scFMs. Our findings underscore the need for specialized models and high-quality datasets that capture a broader range of cellular states. Source code and documentation can be found at: https://anonymous.4open.science/r/PertEval-C674/.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper introduces an evaluation framework for scRNA-seq foundation models applied to the downstream task of predicting the transcriptomic effect of unseen genetic perturbations. In order to adapt foundation models to this task, the authors first extract gene embeddings from a variety of foundation models by simulating a genetic knock-out in unperturbed cells and then embedding both unperturbed and in-silico perturbed cells using the foundation model. Then, they fit MLP adapters on top of these embeddings with a root mean squared error loss. They include two baselines to compare against: the mean baseline, (presenting no variation between different perturbations) and an MLP adapter fit on a simple co-expression based embedding.

The authors then proceed to evaluate these models. At its heart, the proposed evaluation suite relies on two previously established metrics, the mean squared error (MSE) over the top 20 differentially expressed genes (DEGs) per perturbation and the MSE over 2000 highly variable genes (HVGs). On top of this, a spectral splitting technique (SPECTRA) is applied to simulate the effects of increasingly stronger distributional shifts between train and test data, and the top 2000 HVG MSE is averaged over different splitting strengths to yield the area under the SPECTRA performance curve (AUSPC) metric. The authors overall find that foundation model embeddings show a marginal, but not stastitically significant improvement over simple baselines in these metrics.

Finally, they investigate properties of individual perturbations and of the different foundation models used that could explain the observed performance. First, they show that larger effect sizes as measured by the energy distance of the perturbation to the control population and uncommon effect distributions are correlated with higher MSE per perturbation. Second, they explore the effect of overlap between foundation model pre-training corpus and the downstream perturbation dataset on the overall performance, finding mostly a correlation with robustness at a very stringent data splitting regime, with little overall impact otherwise.

### Strengths
* **Impactful question:** Predicting the effect of unseen perturbations is at the heart of understanding the causal impact of genes and therefore an important biological question. Therefore, the scope of the paper, evaluating the impact of scRNA-seq foundation models on this task, is an important research topic.
* **Manuscript quality:** The manuscript is overall well-written and clearly organized.
* **Novelty:** The two elements that are highlighted as novel, simulating challenging datasets with SPECTRA and investigating contextual alignment, are, to my knowledge, novel in this domain.
* **Publishing negative findings:** The authors overall report negative findings as to the impact of foundation models on the task in question, which is is often  disincentivized by the scientific community. I find this commendable since it can help the community to make progress and improve these tools.

### Weaknesses
* **Benchmarking on inappropriate dataset**
    * Most importantly, the authors rely on a **single and quite small** (<300 overall perturbations) perturbation dataset for all of their evaluation, namely [Norman (2019)](https://www.science.org/doi/full/10.1126/science.aax4438). Despite the overall solid presentation, this leaves the impression of a rushed and unfinished work. At the very least, I would like to see results on [Replogle (2022)](https://www.cell.com/cell/fulltext/S0092-8674(22)00597-9?dgcid=raven_jbs_aip_email) included (>10,000 perturbations), but a comprehensive evaluation framework should ideally look at most available Perturb-seq datasets out there to be able to draw any insightful conclusions.
    * More than half of the perturbations in Norman (2019) are combinations of genetic perturbations. Figure E1 shows that the simplest data splits contain >200 perturbations, suggesting that these combinations were included in the analysis. However, I don't see any mention of how combinations are handled by the MLP adapters, and the authors never show any evaluation pertaining to the effect of combinations, as done, for example, in [Roohani (2024)](https://www.nature.com/articles/s41587-023-01905-6).
    * The Norman et al dataset contains CRISPRa perturbations, resulting in overexpression of the perturbed genes, i.e., a **gain-of-function perturbation**. The foundation model embeddings employed by the authors simulate **loss-of-function perturbations**. On a biological level, this is highly unintuitive. While it may still lead to meaningful and useful embeddings, this is a potential failure mode that could explain the overall findings of the paper.
* **Dataset splitting is not high-impact:** Since evaluation is performed on individual datasets (or more specifically, a single dataset here), generalizing to unseen perturbations could be construed as an experimental design question: can I generalize from the observed perturbations to unseen ones, and if so, which ones should I collect data for? Since Perturb-seq can already applied to the order of thousands of perturbations, if the task is to generalize to arbitrary perturbations, a **random selection of genes is always an admissible design**. While novel, that means that making the prediction task artifically harder through the SPECTRA split might overestimate the difficulty of the problem in an actual experimental design context. Moreover, the dataset that the authors consider is already very small and is further massively reduced (see Figure E1, less than 30 perturbations in the final split). Overall, I am not convinced that this splitting technique should be an integral part of the eval pipeline.
* **Correlative investigations of model performance are not convincing**
    * The observed correlation of E-distance to MSE (Figure 3a) is interesting, but not completely convincing to me. At a minimum, could the authors compute a measure of correlation in this plot? Moreover, it should be able to not only assess this question observationally, but also causally through ablation studies. Does changing the splits to specifically include similar E-distance perturbations change the observed performance? Overall, all this seems to say is that outliers are harder to predict.
    * Similarly, the different contextual alignment examples in Figure 5, while novel, do not actually show any good stratification of results (except for the 0.7 split which is very small, and even there it might be exacerbated by the box-plot excluding outliers for scBERT). Therefore, I'm not convinced as to why this is the right formulation to consider for assessing how appropriate the pre-training dataset is for the downstream task.
* **Summary and rating:** The above weaknesses by far outweigh the strenghths, leading to a recommendation to **reject**.

### Questions
* It is not clear to me how the mean baseline is computed. Is it the mean over all perturbations? This is not immediate from the notation in Equation (6).
* The error bars in Figure 2 are very large. Could this be improved by averaging over more model splits?
* In Figure 3, how are the MSEs per perturbation calculated over splits? Are they averaged here? Would it make more sense to investigate this on a per split basis?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces PertEval-scFM, a framework for evaluating how well single-cell foundation models (scFMs) can predict gene expression changes in response to genetic perturbations. The authors benchmark zero-shot embeddings from various scFMs against simple baseline models, finding that scFMs don't consistently outperform baselines. They indentify narrow training distributions as one of the potential culprits and suggest improvements.

### Strengths
- Creation of a standardized evaluation framework (PertEval-scFM) with well-documented codebase is good for field progress
- I like that there's multiple complementary evaluation methods (E-distance, AUSPC), including some novel metrics
- Evaluation includes all major single-cell foundation models with appropriate baselines
- I like how the paper provides constructive criticism of current models while suggesting concrete paths forward, by doing the distribution shift effect analysis.

### Weaknesses
- Limited dataset scope: Heavy reliance on single dataset (Norman et al.), restricted to CRISPRa perturbations in K562 cell line make me concerned about how well this generalizes
- My main issue with the paper is that there seem to be big parameter count disparaties betw. the MLPs because of the different embedding dimensions.
        - Base embedding dimensions vary widely (200 to 3,072)
        - Results in vastly different MLP sizes between models - by my calculations, scFoundation's MLP ends up with around 25M parameters while scBERT's has less than 0.5M
        - Such dramatic differences in model capacity could significantly affect training dynamics and make fair comparison difficult and this doesn't seem significantly considered / addressed. Though the authors tuned the hypers individually for each embedding.
- Minor note: In Table 1 scGPT is a Transformer, FlashAttention is just a way of efficiently implement normal attention.

### Questions
1. Could you elaborate on the methodological choices around MLP probe architecture? Specifically:
    - How sensitive are results to the choice of architecture?
    - How does the varying parameter count due to different embedding dimensions affect fair comparison between models?
    - Have you considered approaches to normalize model capacity across different embeddings?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This work introduces a benchmark framework for the evaluation of single-cell FMs in the context of genetic perturbation prediction. Multiple scFMs (scBERT, Geneformer, scGPT, UCE, and scFoundation) are compared to simple baselines through a linear probing approach and evaluated across multiple metrics aggregated from previous works. Results highlight the limitations of scFMs in this context, with negligible improvements over the baselines. Additionally, the authors highlight the need for better datasets across a broader and more diverse range of cellular states to better assess predictive models.

### Strengths
- The topic is relevant, as the prediction of genetic perturbations has critical biological applications and therapeutic potential, and previous work has shown how scFMs generally provide limited benefits in this context.
- The manuscript is easy to follow, and the figures are clear.
- A good number of scFMs is considered in this study. 
- The work introduces a general, open-source benchmarking framework.

### Weaknesses
- Setting: the major limitations of this work are its scope and predictive methodology. This work focuses on a very specific setting of "zero-shot" scFM embeddings for perturbation effect prediction. However, this is not necessarily the only (or the best) approach to leverage scFMs for this task. Notably, previous works on scFMs also include perturbation effect prediction results and often focus on a different setting. For example, scGPT, in the original publication (Cui et al., 2024), leverages finetuning for this task, reporting promising results. This leads to important limitations: 1) while it is possible to compare multiple methods in this study, it is not possible to compare results to those included in the original publications, and 2) given that this is the only approach tested in this paper for all methods, it is difficult to say if it is generally good enough for the task. Indeed, it could be possible that the way perturbed embeddings are generated (line 133) leads to out-of-distribution (and, in turn, low-quality) scFMs embeddings for all methods. Overall, this limits the generality of the conclusions of this work.
- Novelty: the novelty of this work appears to be limited, as it leverages previously-defined evaluation metrics, and both the dataset, task, and models have been extensively studied in previous works. 
- Data: the benchmark relies on a single, small dataset. This further restricts the generality of the conclusions, potentially limiting their robustness.


Overall, this work presents a general framework that, while potentially relevant, still has critical limitations in terms of modeling techniques explored and datasets analyzed.

### Questions
- The authors should include additional datasets, discussing trends across them.
- The authors should include multiple modeling paradigms, for example based on fine-tuning or exploring the space of different architectures for linear probing.
- The authors should include additional baselines, in particular specialized methods for perturbation effect prediction (e.g., GEARS), to better contextualize the results.

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
4

### Summary
This work presents PertEval-scFM for evaluating zero-shot single-cell foundation models (scFMs) in predicting cellular responses to genetic perturbations, using the Norman et al. dataset (2019).  Specifically, PertEval-scFM evaluates zero-shot embeddings from several existing scFMs—such as scGPT, scBERT, Geneformer, UCE, and scFoundation against baseline models based on gene expression matrix in the distribution shift scenario. The results suggest that zero-shot scFM embeddings do not consistently outperform simpler baselines, such as mean expression and MLP baselines across different evaluation metrics. As the sparsification probability increases, the performance of scFM embeddings declines more sharply than baseline models, suggesting that scFMs may be sensitive to distribution shifts. particularly in strong/atypical scenarios, with limited generalizability in zero-shot contexts of perturbation effect prediction.

### Strengths
**More scFMs in benchmarking and distribution shift scenarios**: Compared to existing works on gene perturbation benchmarking,  PertEval-scFM considers more available updated scFMs and introduces distribution shift scenarios to evaluate the generalization of scFM better.

**Comprehensive Metrics**: By combining AUSPC, E-distance, and contextual alignment,  PertEval-scFM provides a detailed analysis of model robustness, distribution shift sensitivity, and alignment between pre-training and application-specific datasets.

**Reproducibility**: The PertEval-scFM code has been provided transparently and accessiblely, with the potential for further development and use by the community.

### Weaknesses
**Novelty and Significance**: As the authors have noted in the introduction section, benchmarking scFMs on genetic perturbation tasks is not new, with similar conclusions being drawn.  The current benchmark indeed falls short in considering only one Perturb-seq dataset, making the results here rather preliminary. Using the datasets provided in existing benchmark works, such as Wu et al. (2024) could enhance the benchmarking breadth of current work. The author may also elaborate on addressing the key findings of PertEval-scFM by introducing distribution shift, especially in a biological context -- for example,  more quantitative and systematic analysis on the strong/atypical perturbations, and biological mechanism explanations/insights evidenced by bioinformatics analysis.

**The MLP Structure Selection**:  In PertEval-scFM， one hidden layer structure was used as an MLP probe to model the perturbation effects, and it served as a baseline. It's not clear how this selection influences the results since it's possible that the mapping between scFM embeddings and perturbation is highly complicated. Robustness tests could be added here. It is also insightful to investigate the role of scFM embeddings in more advanced or state-of-the-art deep learning-based prediction models such as GEARS[1] or CELLOT[2].

**Absence of Fine-Tuning Component**: While the zero-shot test provides insights into scFM performance, real-world applications frequently involve fine-tuning to adapt models. Without a fine-tuning component, the evaluation lacks a practical measure of scFM potential in real-world perturbation prediction settings, especially where significant distribution shifts are possible. Incorporating fine-tuning into the analysis, or at least comparing zero-shot performance to fine-tuned baselines, would yield a more comprehensive understanding of scFM effectiveness in applied perturbation prediction.

Overall, while the paper is not recommended to be accpeted by ICLR at current form, I would like to increase the score if the concerns are adequately addressed by the authors.

[1]Roohani, Y., Huang, K. & Leskovec, J. Predicting transcriptional outcomes of novel multigene perturbations with GEARS. Nat Biotechnol 42, 927–935 (2024). https://doi.org/10.1038/s41587-023-01905-6

[2]Bunne, C., Stark, S.G., Gut, G. et al. Learning single-cell perturbation responses using neural optimal transport. Nat Methods 20, 1759–1768 (2023). https://doi.org/10.1038/s41592-023-01969-x

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
2
