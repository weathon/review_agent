# CauKer: Classification Time Series Foundation Models Can Be Pretrained on Synthetic Data

- Avg Score: 6.00
- Decision: Accept (Oral)
- Scores: 6, 6, 4, 8

## Abstract
Time series foundation models (TSFMs) have recently gained significant attention due to their strong zero-shot capabilities and widespread real-world applications. Such models typically require a computationally costly pretraining on large-scale, carefully curated collections of real-world sequences. To allow for a sample-efficient pretraining of TSFMs, we propose CauKer, a novel algorithm designed to generate diverse, causally coherent synthetic time series with realistic trends, seasonality, and nonlinear interactions. CauKer combines Gaussian Process (GP) kernel composition with Structural Causal Models (SCM) to produce data for sample-efficient pretraining of state-of-the-art classification TSFMs having different architectures and following different pretraining approaches. Additionally, our experiments reveal that CauKer-generated datasets exhibit clear scaling laws for both dataset size (10K to 10M samples) and model capacity (1M to 783M parameters), unlike real-world datasets, which display irregular scaling behavior.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose CAUKER, a synthetic data generation algorithm that leverages Gaussian Process kernel composition and Structural Causal Models to produce diverse time series for augmenting training data in time series classification tasks. The paper evaluates the proposed approach against other synthetic data generation techniques for several time series foundation models.

### Strengths
- The paper introduces a novel synthetic data generation technique leveraging structural causal models (SCMs) for time series.
- The work is a focused study on synthetic data augmentation for time series classification, an understudied area in time series literature.
- Two time series foundation models (TSFMs) are evaluated with supervised and contrastive learning pre-training schemes
- Several synthetic data augmentation approaches are systematically compared in Table 1, highlighting relative effectiveness.
- Figure 3 effectively illustrates scaling laws and the relationship between model size and performance.
- Figure 4 provides an interesting analysis showing the diversity of principal components in synthetic data relative to non-synthetic datasets.
- The study demonstrates that fewer synthetic samples can achieve comparable performance to real-world pre-training datasets, highlighting practical efficiency benefits.

### Weaknesses
I would be happy to increase my score if the following concerns/points are addressed.

- Zero-shot evaluation methodology:
The study claims to evaluate TSFMs in a zero-shot setting, but the models are allowed to be pre-trained on the training set of the same dataset used for evaluation. This means the evaluation is not strictly zero-shot, as the train and test sets are likely in-distribution (Lines 122–124):
“In practice, if we evaluate a given TSFM on a test set from a UCR (Dau et al., 2019) dataset, we ensure that the TSFM was not pre-trained on it, but we allow for the train set of this same dataset to be used for pre-training.”


- Missing baseline comparisons: Results without synthetic data augmentation are not reported in Table 1. Including these and quantifying the lift from augmentation would be helpful.

- No text-based or experimental comparison with the synthetic data generation process used by TabPFN, which also leverages structural causal models.

- No comparison with non-foundation model baselines (e.g., random forecasts, XGBoost, logistic regression).

- Clarification on model pre-training: It is unclear whether the models are pre-trained from scratch on synthetic data or fine-tuned with synthetic data (using pre-trained models on real-world data). For example, the text states:
“In practice, if we evaluate a given TSFM on a test set from a UCR (Dau et al., 2019) dataset, we ensure that the TSFM was not pre-trained on it, but we allow for the train set of this same dataset to be used for pre-training.”

### Questions
1. Are the TSFMs pre-trained from scratch on synthetic data, or are they fine-tuned on synthetic data (using models already pre-training on real data)?
2. How do the models perform without any synthetic data augmentation?

Suggestion: It would be interesting to include the combined scaling laws for the UEA and Cauker datasets on the same plot in Figure 3 to show cross-dataset scaling laws.

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
This paper proposes CAUKER, a novel and sophisticated pipeline for generating synthetic time series data specifically tailored for the pre-training of classification-oriented Time Series Foundation Models (TSFMs). The core idea is to combine two methodologies: Gaussian Process (GP) kernel composition, which generates realistic temporal patterns (trends, seasonality), and Structural Causal Models (SCMs), which impose a causal graph structure to create complex, non-linear interactions and meaningful clusters. The authors conduct extensive experiments showing that TSFMs pre-trained on CAUKER data not only outperform models trained on other synthetic datasets but also nearly match the performance of models pre-trained on real-world corpora that are over an order of magnitude larger. A key finding is that CAUKER-generated data enables smooth and predictable scaling laws with respect to both dataset and model size, a property the authors show is absent when pre-training on standard real-world benchmarks.

### Strengths
*   **Novelty and Formulation:** The primary strength of this work is its well-motivated. Rather than creating a monolithic generator, the authors identify two key requirements for classification data—realistic temporal dynamics and discriminative clustering structure—and solve them by combining the strengths of two distinct fields. Using GP kernel composition (common in forecasting) for temporal patterns and SCMs (from the causality and tabular learning literature) for creating underlying class structures is a novel and highly effective synthesis. The design choices are clearly justified (Section 3.2), and the ablation-style comparison in Table 1 convincingly demonstrates that both components are necessary for optimal performance.

*   **Empirical Evidence of Scaling Laws:** The paper's most impactful result is the clear demonstration of scaling laws (Figure 3). The experiments showing that accuracy on downstream tasks increases smoothly and monotonically with more synthetic data and larger models are a significant contribution. By contrasting this with the erratic and non-scaling behavior of models trained on the real-world UEA benchmark, the authors make a powerful case for using high-quality synthetic data as a controlled "wind tunnel" to study and develop scalable TSFMs. This provides a valuable methodology for the community, independent of the CAUKER pipeline itself.

*   **Sample Efficiency and SoTA Performance:** The paper provides strong evidence that "quality over quantity" is important for pre-training data. The results in Figure 7 are particularly striking, showing that pre-training a model like Mantis on just 100K synthetic samples can achieve performance nearly identical to pre-training on its original 1.89M real-world sample corpus. This has significant practical implications, as it dramatically reduces the need for expensive and difficult data collection and curation. The fact that this performance is state-of-the-art for synthetic-data pre-training validates the effectiveness of the proposed approach.

*   **Experimemntal Validation:** The paper is written with outstanding clarity. The experimental validation is extensive and robust, covering comparisons to multiple baselines, scaling laws, qualitative analyses (PCA, CKA, non-linearity in Figures 4 and 5), and transferability to different benchmarks (UCR, WOODS) and even a different task (forecasting). The appendices are helpful, providing, detailed descriptions of the function banks, hyperparameter sensitivity analysis.

### Weaknesses
I have concerns about the evaluation process and specially related to the complexity of the proposed generator, the framing of its comparison to real-world data, and the scope of the architectural evaluation.

1.  **High Generator Complexity and Opaque Design Choices:** The CAUKER pipeline is a complex amalgamation of multiple components: three distinct function banks (kernel, mean, activation), random kernel composition, and random DAG generation. This introduces a large number of "meta-hyperparameters" (e.g., the specific contents and size of the banks, the distribution of DAG parameters). While the appendix provides a sensitivity analysis for a few of these, the process for designing the function banks themselves is not fully justified. It is unclear if the chosen set of 36 kernels or the specific activation functions are uniquely effective, or if a much simpler subset could achieve comparable results. This complexity could pose a significant barrier to adoption and reproducibility for researchers who do not have the authors' expertise in this specific setup.

2.  **Potential for a "Straw Man" Argument Against Real Data:** The paper's narrative strongly contrasts the clean scaling of CAUKER with the poor scaling of the UEA benchmark. While this is a powerful rhetorical device, it risks overgeneralizing the conclusion. The UEA archive, while a standard benchmark, is a heterogeneous collection of many small, domain-specific academic datasets; it was not designed as a large-scale, cohesive pre-training corpus in the vein of ImageNet or The Pile. The observed lack of scaling could be an indictment of the UEA dataset's specific properties (lack of diversity, domain mismatch) rather than a fundamental flaw of pre-training on real-world data in general. The paper lacks a discussion of this nuance.

3.  **Limited Diversity of Tested Model Architectures:** The experiments are exclusively focused on two Transformer-based models (Mantis, which is ViT-based, and MOMENT, which is T5-based). While these represent different pre-training objectives (contrastive vs. masked reconstruction), they share a core architectural paradigm. It is an open question whether the benefits of CAUKER's data structure are universally applicable or if they are particularly well-suited to the inductive biases of attention-based models. The rich, causally-linked structures might be more effectively captured by attention than by models with different biases, such as CNNs or State Space Models.

### Questions
Based on these weaknesses, here my questions to the authors:

*   **Question 1:** The CAUKER pipeline is composed of several stochastic modules and expertly curated function banks. How were the specific contents of these banks (e.g., the 36 kernels, the set of mean/activation functions) selected and validated? Is the performance highly sensitive to these specific choices, or is the framework robust to using a simpler, more generic set of components?

*   **Question 2:** The hyperparameter sensitivity analysis in Appendix C.3 is helpful. However, to better understand the generator's failure modes, have you investigated scenarios where deliberately poor choices (e.g., using only linear activations, forcing very shallow DAGs, or using only a single kernel type) cause the method to fail or degrade to the level of the simpler baselines in Table 1?

*   **Question 3:** To what extent do you believe the poor scaling on the UEA benchmark is a fundamental property of real-world time series data, versus a specific artifact of the UEA collection's composition and scale? How might CAUKER compare against a hypothetical, massive, and diverse real-world corpus curated specifically for pre-training (e.g., a "TimeNet")?

*   **Question 4:** The study convincingly demonstrates CAUKER's benefits for Transformer-based TSFMs. How do you hypothesize the generated data would interact with models possessing fundamentally different inductive biases, such as those based on CNNs (e.g., InceptionTime) or State Space Models (e.g., Mamba), which process information more locally or linearly?

*   **Question 5:** The causal graph propagation step seems central to creating discriminative structure. Does this structural property particularly favor the global receptive field of attention mechanisms? A deeper analysis of which components of CAUKER (GP vs. SCM) are most beneficial for which type of model architecture would be a valuable contribution.

*   **Question 6:** The paper successfully extends CAUKER to forecasting. Does this suggest that good classification data is a superset of good forecasting data, or were any modifications to the CAUKER pipeline necessary to achieve strong forecasting performance? Specifically, are the SCM-induced non-linearities as important for forecasting as they are for classification?

### Soundness
3

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
4

### Summary
The manuscript presents CAUKER, a synthetic data generation pipeline for pretraining classification time-series foundation models. CAUKER composes Gaussian-process kernels and mean functions within a structural causal model (SCM) graph, producing causally coherent sequences for self-supervised pretraining of contrastive (Mantis) and masked-reconstruction (MOMENT) encoders. Empirically, models pretrained solely on CAUKER data achieve competitive zero-shot accuracy on UCR and exhibit monotonic scaling with both dataset size and model capacity.

### Strengths
1. Integrating kernel composition with SCM-based propagation yields diverse dynamics and inter-series dependencies aligned with classification objectives.
2. Evaluation across contrastive and masked-reconstruction pretraining objectives increases the generality and external validity of the findings.
3. Experiments demonstrate data/model scaling laws and strong zero-shot transfer, offering a compelling empirical performance.

### Weaknesses
1. Pretraining on pure synthetic data and obtaining strong results is not particularly surprising, as prior work (e.g., TabPFN-TS) has already demonstrated the potential of synthetic data. This manuscript would benefit from sharper positioning of what is substantively novel in methodology part. 
2. This paper does not clearly articulate the challenges in transferring synthetic data generation methods designed for forecasting tasks to classification tasks—what the specific difficulties are and how they are addressed. The introduction reads largely as an integration of existing generators applied to classification, with empirical observations such as scaling laws, but as a research contribution this positioning feels insufficient.
3. The evaluation scope remains narrow (largely UCR-style, often univariate and fixed-length), with limited robustness analysis on generator hyperparameters and little evidence for multivariate, irregularly sampled settings.

### Questions
1. What are the concrete, theoretically grounded challenges when porting forecasting-oriented synthetic pipelines to classification (label generation, class balance, inter-class separability, invariance desiderata), and how does each CAUKER design choice mitigate them?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes CAUKER, a synthetic data generation framework combining Gaussian Process kernel composition and SCM for time series foundation models for classification tasks. Unlike most prior work focusing on forecasting, CAUKER targets classification and demonstrates that synthetic pretraining can yield competitive or superior performance to real world datasets. It also reveals scaling laws for synthetic pretraining in terms of dataset and model size.

### Strengths
* Addresses a clear gap, synthetic pretraining for classification TSFMs.
* The causal kernel composition is conceptually elegant and well motivated.
* Benchmarks across multiple models and datasets .
* Includes scaling law analyses for data, model, and compute.
* Outperforms real-data pretraining in several zero-shot setups.
* The method is explained clearly, with schematic diagrams and pseudocode.

### Weaknesses
* Both GP based and SCM based data generation already exist, the novelty lies mostly in combining them.
* Evaluation confined to zero-shot classification. would benefit from downstream fine-tuning or transfer learning results.
* The contribution of causal graph depth/branching remains unclear.
* While interesting, the scaling analysis is somewhat descriptive without deeper theoretical grounding

### Questions
* How does CAUKER handle multivariate dependencies beyond univariate channel concatenation?

* Can CAUKER generalize to forecasting or imputation pretraining tasks?

* How computationally expensive is CAUKER compared to kernel only methods?

### Soundness
3

### Presentation
3

### Contribution
4
