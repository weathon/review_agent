# LLM4GRN: Discovering Causal Gene Regulatory Networks with LLMs - Evaluation through Synthetic Data Generation

- Decision: Reject
- Scores: 3, 5, 5

## Abstract
Gene regulatory networks (GRNs) represent the causal relationships between transcription factors (TFs) and target genes in single-cell RNA sequencing (scRNA-seq) data. Understanding these networks is crucial for uncovering disease mechanisms and identifying therapeutic targets. In this work, we investigate the potential of large language models (LLMs) for GRN discovery, leveraging their learned biological knowledge alone or in combination with traditional statistical methods. We employ a task-based evaluation strategy to address the challenge of unavailable ground truth causal graphs. Specifically, we use the GRNs suggested by LLMs to guide causal synthetic data generation and compare the resulting data against the original dataset. Our statistical and biological assessments show that LLMs can support statistical modeling and data synthesis for biological research.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper titled "LLM4GRN: Discovering Causal Gene Regulatory Networks with LLMs – Evaluation through Synthetic Data Generation" presents a novel approach leveraging large language models (LLMs) for inferring gene regulatory networks (GRNs) from single-cell RNA sequencing (scRNA-seq) data. The authors propose using LLMs to generate complete GRNs or to provide prior knowledge to traditional statistical methods. A key contribution is the use of causal synthetic data generation as an evaluation strategy, which allows for the assessment of the inferred GRNs in the absence of ground truth causal graphs. The study demonstrates the potential of LLMs in supporting statistical modeling and data synthesis for biological research, with a particular emphasis on the combination of LLMs and statistical inference methods showing promise for scRNA-seq data analysis.

### Strengths
1. The methodology involving the use of causal synthetic data generation to evaluate GRNs is well-designed and addresses a significant challenge in the field, namely the lack of ground truth data for validation.
2. The paper is well-organized, with clear explanations of the methodology, experimental setup, and results. The use of visual aids such as figures and tables enhances the understanding of the presented material.
3. The potential of LLMs to capture complex biological interactions and support statistical modeling in genomics is a significant contribution, with broad implications for disease mechanism understanding and therapeutic target identification.

### Weaknesses
1. The paper could benefit from a more detailed discussion on the limitations of the current approach, particularly regarding the assumptions made in the GRN structure due to the constraints of the GRouNdGAN framework.
2. While the study highlights the potential of LLMs, it does not explore the reasons behind the performance differences observed between the LLM-based and statistical methods, which could provide further insights into the underlying mechanisms.
3. The paper might consider discussing the scalability of the proposed method, especially how it might perform with larger and more complex datasets.
4.The paper does not seem to have much theoretical innovation. It simply uses LLM as a plug-in to replace previous human prior knowledge or statistical analysis methods.
5. The results are not very convincing: Figure 2 aims to illustrate the robustness and superiority of the method by overleaping different methods. However, the small gap cannot reflect the advantages of the proposed method.

### Questions
1. Could the authors elaborate on how the performance of the LLM-based GRN inference might be affected by the quality and representativeness of the data used to train the LLMs?
2. It would be helpful if the authors could discuss potential future work on refining the GRN inference process, especially in addressing the noise and specificity issues mentioned in the biological plausibility analysis.
3. The paper mentions the use of GPT-4 and Llama-3.1-70B models. Are there any plans to compare the approach with other LLMs or to investigate the impact of model size on the performance of GRN inference?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper presents LLM4GRN, a method that integrates Large Language Models (LLMs) to discover causal Gene Regulatory Networks (GRNs) from single-cell RNA sequencing (scRNA-seq) data. The study employs a causal synthetic data generation approach, comparing two experimental setups where LLMs either generate GRNs directly or assist statistical methods (e.g., GRNBoost2) by providing transcription factor (TF) candidates. The authors test the approach on three datasets, highlighting LLMs' effectiveness in biological plausibility and similarity to real data. Notably, Llama-3.1 outperformed GPT-4 in some aspects, suggesting potential advantages for using certain open-source models over more advanced commercial ones.

### Strengths
1. Innovative Use of LLMs: Leveraging LLMs in GRN discovery demonstrates novel application potential in biological data analysis.
2. Comparison of Multiple LLMs: Provides comparative insights into GPT-4 and Llama-3.1 performances, showing valuable differences that inform model selection for scRNA-seq analysis.
3. Promising Hybrid Approach: Results show that combining LLM-derived TF lists with statistical models like GRNBoost2 can improve causal inference quality.

### Weaknesses
1. Limited Dataset Variety: Only three datasets (PBMC-All, PBMC-CTL, and Bone Marrow) are used, which limits generalizability to broader biological contexts.
2. Dependency on Pathway Completeness: The method’s performance may suffer in cases with incomplete pathway information, which isn’t fully addressed.
3. Biological Relevance of Generated Data: Some results indicate noise and inconsistency in cell-type differentiation, which could impact downstream analyses and applications.
4. Complexity of Methodology: The use of multiple knowledge bases and LLM models creates a complex pipeline that may be difficult for other researchers to reproduce.

### Questions
1. How might the LLM-based approach handle incomplete pathway information or limited TF-gene knowledge in certain datasets?
2. How does pathway information specifically improve GRN inference compared to simpler approaches that omit it?
3. What criteria were used to select the three datasets, and how might the results generalize to datasets with different biological characteristics?
4. How does the complexity of LLM4GRN impact reproducibility, and are there ways to simplify the model without losing performance?
5. Can you discuss how robust the model is to potential biases in the LLM-based knowledge base, especially given the dependency on prior information?
6. Why do you believe Llama performs better than GPT-4 in certain settings, and what does this imply for LLM-based GRN inference?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper explores the application of general-purpose foundation models (GPT-4, LLaMA) for identifying context-specific transcription factors (TFs) and constructing corresponding gene regulatory networks (GRNs). The authors evaluate the GRNs by contrasting them with knowledge base-derived GRNs and those inferred using GRNBoost2, then apply causal synthetic data generation based on the method of Zinati et al. (2024) to assess the biological plausibility of the generated networks.

### Strengths
- The study addresses a critical need for context-specific GRNs, as many practitioners rely on generic knowledge-base GRNs, which lack context specificity and often lead to less accurate inferences.
- A thorough analysis, evaluating both the TF lists generated by foundation models and the resulting GRNs with controls at each step

### Weaknesses
The primary limitation is the lack of objective metrics for validating the inferred GRNs. While the authors use synthetic data generation as an indirect evaluation method, results vary significantly across datasets, with the GPT-based GRNs performing well on PBMC_ALL but showing weaker, near-random performance on other datasets (as measured by MMD and RF AUROC). These results indicate low systematic performance on the author's suggested standards.

The above may also be a result of knock-effects caused by the Causal GAN synthetic data generation. To address these issues, it would be helpful to incorporate additional validation methods that test the sensitivity of the probing strategy to known, context-dependent variations within well-studied GRNs. For instance, starting with a validated GRN and introducing experimentally confirmed context-specific variations could help determine if the models correctly adapt the network structure. Although this approach still has limitations (e.g., it could have been in the training set of GPT4), it could provide orthogonal evidence of the probing strategy’s effectiveness in identifying/retrieving context-sensitive elements.

Similarly, for the initial TF lists generated by foundation models, I recommend testing their context specificity by comparing TF lists for pairs of contexts with low TF overlap based on expression in resources like the Human Cell Atlas. This could reveal whether the models distinguish between TFs active in distinct biological contexts.

### Questions
- Please consider adding other orthogonal direct ways of validating GRNs, such as the one suggested above by testing the exact same probing strategy on whether it can identify context specific variations of a well known and validated TF GRN
- Please include an orthogonal method to validate the context-specific TF list, such as distinguishing TFs based on uncorrelated expression patterns across distinct cell types, as mentioned in the Weaknesses section

### Soundness
2

### Presentation
1

### Contribution
2
