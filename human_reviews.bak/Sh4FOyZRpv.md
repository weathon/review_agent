# CTSyn: A Foundation Model for Cross Tabular Data Generation

- Decision: Accept (Poster)
- Scores: 6, 6, 5, 6

## Abstract
Generative Foundation Models (GFMs) have achieved remarkable success in producing high-quality synthetic data for images and text. However, their application to tabular data presents significant challenges due to the heterogeneous nature of table features. Current cross-table learning frameworks struggle because they lack a generative model backbone and an effective mechanism to decode heterogeneous feature values. To address these challenges, we propose the Cross-Table Synthesizer (CTSyn), a diffusion-based generative foundation model for tabular data generation. CTSyn comprises two key components. The first is an autoencoder network that consolidates diverse tables into a unified latent space. It dynamically reconstructs table values using a table schema embedding, allowing adaptation to heterogeneous datasets. The second is a conditional latent diffusion model that generates samples from the learned latent space, conditioned on the table schema. Through large-scale pre-training, CTSyn outperforms existing table synthesizers on standard benchmarks in both utility and diversity.  These results position CTSyn as a promising framework for synthetic table generation and lay the groundwork for developing large-scale tabular foundation models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper presents CTSyn, a generative model designed for synthesizing tabular data by unifying diverse datasets into a single latent space.
The proposed architecture combines a cross-table variational autoencoder model with a conditional diffusion model to generate flexible synthetic data for various domains. Extensive pre-training across real-world datasets enables CTSyn to outperform existing models in data diversity and fidelity, also enhancing downstream machine learning tasks, especially in low-data regimes, even *surprisingly* surpassing real data.

### Strengths
- Relevance. CTSyn focuses on the important area of generating high-quality tabular data, which is essential for improving machine learning models. Synthetic data can help when real data is limited or sensitive, such as in healthcare and finance.

- Novel Architecture Design. The model features a new design that combines a cross-table variational autoencoder with a conditional diffusion model. This approach effectively maintains data quality and flexibility across different types of tables, significantly enhancing the capabilities of previous methods in this field.

- Solid Baseline and Dataset Choice. CTSyn compares its performance with SOTA models and uses large, diverse datasets from the real world.

### Weaknesses
- CTSyn cannot handle datasets where columns include free text (e.g., in medical records with text entries), limiting its application to purely structured numerical or categorical data. 

- The evaluation lacks certain robustness tests for synthetic data quality, such as a discriminator test to assess whether a surrogate model can distinguish between real and synthetic data, or metrics like "distance to closest record" for authenticity. Additionally, the experimental setup for machine learning utility (e.g., cross-validation specifics) needs more transparency and detail to fully validate CTSyn's claims of enhancing downstream tasks.

- A more comprehensive ablation study is needed to assess the impact of each component in CTSyn’s architecture, such as the specific roles of the cross-table variational autoencoder and the conditional diffusion model in enhancing data fidelity and adaptability. Including these would offer clearer insights into model design choices.

### Questions
- The statement *"synthesizers cannot add information not included in the original training data"* may not fully capture the recent advances in synthetic data generation that utilize pre-trained LLMs for transfer learning. Specifically, methods like TabTab and LaTable, which are pre-trained on extensive tabular datasets, leverage LLMs' vast prior knowledge, extending the potential of data generation beyond what’s strictly in the training dataset. This aligns with approaches like GReaT, which explicitly incorporate pre-trained LLMs to enable effective transfer learning across tabular domains. To further clarify the novelty of CTSyn, it would be helpful if the authors addressed these points, including a comparison to methods that already exploit LLM-based architectures for tabular data synthesis. Additionally, on Line 099, the authors mention that existing methods struggle to capture "intrinsic structural properties of tables" and continuous values when using LLMs. Could the authors provide more context on this limitation? Recent studies indicate that fine-tuning can improve LLM performance for modeling continuous values, potentially overcoming some of these structural challenges [1]. 

 - CTSyn employs quantile transformation for handling numerical values, but it’s unclear why this choice was made over alternative scaling methods, such as min-max or standard normalization. Quantile transformation can improve handling of skewed distributions but might also introduce artifacts in certain cases. Could the authors elaborate on the rationale behind selecting quantile transformation? 

- Eq. 1 introduces the sequence $\mathbf{E}$, representing tokenized and embedded table values, but the meta embedding $e_m$ is excluded. Since $e_m$ likely contains crucial table-specific metadata, incorporating it directly into $\mathbf{E}$ might strengthen the alignment between table content and contextual information

-  Table 2 presents a *column-wise comparison* of synthetic and real data, which provides some indication of CTSyn’s data fidelity. However, this approach does not fully address the challenges of synthetic data generation, particularly regarding joint modeling of column relationships and complex dependencies. For synthetic data to be *realistic*, it must ideally preserve not just marginal distributions but also higher-order dependencies and potential causal relationships between columns. Could the authors consider additional evaluation metrics, such as multivariate measures or metrics assessing the preservation of conditional distributions, to more rigorously demonstrate CTSyn’s capability for joint data modeling? 

- In Fig. 2, the authors show that synthetic data generated by CTSyn outperforms real data in certain tasks, which is a quite surprising result in my view. How can authors elaborate on that? Please provide detailed explanation of the experiment settings here. 

- What is the final parameter size for the CTSyn model?


[1] Akhtar, Mubashara, Abhilash Shankarampeta, Vivek Gupta, Arpit Patil, Oana Cocarascu, and Elena Simperl. "Exploring the numerical reasoning capabilities of language models: A comprehensive analysis on tabular data." arXiv preprint arXiv:2311.02216 (2023).

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Authors propose Cross Table Synthesizer (CTSyn), a tabular data generation pipeline that leverages diffusion models and variational autoencoders to encode and generate data from diverse types of tables. Specifically, CTSyn comprises a pre-processing step where the table meta-data, table column names and column values are all embedded using Language Models (and quantile transformer in the case of continuous column values) into a common embedding space. Each table row is then represented using the meta-data, column-name and column-value embedding vectors. This sequence of embeddings is supplied to a $\beta$-variational autoencoder ($\beta$-VAE) for embedding multiple diverse types of tables in a common embedding space. Finally, a diffusion model is trained (conditioned upon the latent representation output from the encoder in the $\beta$-VAE) to generate synthetic table data. Results demonstrate that synthetic data generated by the proposed technique achieves good statistical similarity (evaluated using column-wise metrics) to real data. Further, results also demonstrate that the representation learned by CTSyn is statistically similar to the training data while not exactly replicating the training data. Overall, the experimental comparison to baselines and research questions investigated demonstrate the prowess of the proposed synthetic tabular data generation technique.

### Strengths
- The paper is well-written and the proposed technique is detailed clearly. 
- The results clearly demonstrate the prowess of CTSyn in terms of matching the training data at least as captured by column-wise statistical metrics (Table 2). 
- Figure 2 and Table 3 demonstrate the ability of CTSyn to maintain privacy of training data (compared to other non differentially-private synthetic tabular data generators) while still learning useful representations. This demonstrates that the model is capable of balancing learning rich / accurate representations of the tabular data without replicating training data.

### Weaknesses
- The training process employing a diffusion model requires costly pre-training and fine-tuning hence scaling the modeling pipeline to large tables (e.g., 100s of columns, millions of rows) may be challenging.

- One crucial facet of the paper that is lacking clarification is a description of the meta-data for the various tables employed. 

- Further, as the current model is termed as a foundation model for tabular data generation, it is crucial to demonstrate its effectiveness on noisy training / fine-tuning tabular data. The noise-free nature of datasets (at least during fine-tuning) cannot be guaranteed and an investigation of the robustness of the proposed technique to noisy training data is necessary but not presented.

### Questions
1. Why are `free text` and `date-time` containing tables dropped (line 280)? Does this have to do with the inability of the existing pipeline to encode linearly increasing data (e.g., dates / times) or was there some other reasoning?
2.  How was PE generated to be employed with different tables? Although the motivation of column order invariance is clear in the tabular data generation context, what is the intuition behind treating column embeddings and positional encoding as substitutes, i.e., could they not have complementary strengths (e.g., could PEs be designed in a way to encode information not captured in column embeddings, table metadata embedding?) due to which the model might benefit from usage of both PE, col. embeddings in conjunction with metadata embeddings?
3. How does the proposed CTSyn method perform in the context of missing column values in the training / fine-tuning tables? Has this investigation been conducted?
4. What is the intuition behind the result in Table 2a where F1 scores achieved by a classifier trained on CTSyn data outperforms the performance of a classifier trained on real data?

### Soundness
3

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
This paper introduces CTSyn, a foundational model designed to generate heterogeneous tabular data. CTSyn leverages an autoencoder that consolidates diverse tables into a unified latent space and reconstructs data based on the provided table schema embedding, allowing it to adapt dynamically to various table formats. Through large-scale pre-training, CTSyn outperforms existing data synthesizers and demonstrates superior performance compared to real data in low-data scenarios.

### Strengths
A strength of this work is the successful application of a straightforward approach to map embedding vectors of categorical variables back to their original space. Although simple, this method effectively demonstrates that returning to the original categorical space can be achieved without complex transformations, providing a useful baseline for handling categorical data embeddings.

### Weaknesses
A limitation of this work is that it primarily proposes a method for handling individual variables within a framework similar to LSGM [1] that trains a diffusion model in latent space. As such, the approach lacks substantial novelty and may have limited impact, given that it focuses on variable handling within an established generative model framework rather than introducing fundamentally new techniques.

Minors. Typos in line 129 (specific) and 266 (The)

[1] Vahdat, Arash, Karsten Kreis, and Jan Kautz. "Score-based generative modeling in latent space." Advances in neural information processing systems 34 (2021): 11287-11302.

### Questions
1. As part of the ablation study, I am also interested in seeing the results of experiments with only the column name condition applied.
2. While reviewing the provided code in the supplementary material, I encountered issues running vae.sh smoothly. Would it be possible to provide an updated .sh script that can reliably reproduce the main experimental results or specify the exact package versions used? Thank you for your help in ensuring reproducibility.

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
The authors propose CTSyn, specifically targeting generating heterogeneous tables. The author propose to do this by first using a cross table VAE, which can embed tables with different types of rows and projects them in a common latent space. Subsequently, they employ a DDPM based diffusion model (With classifier free guidance via a pre-trained LM) to generate samples for different types of tables. They also show the benefits of pre-training on a large number of diverse tables -- so as to serve as a starting point for different downstream tasks. The authors present results on multiple downstream tabular datasets with classification and regression tasks - and compare with multiple baselines -- with regard to fidelity, ML utility, and privacy.

### Strengths
1. While there have been prior works employing auto-encoders + latent diffusion models towards tabular data synthesis (e.g. https://arxiv.org/abs/2310.09656), prior works dealing with heterogenous table synthesis have been limited.
2. The authors provide good comparison against baselines on a good range of datasets -- for fidelity, privacy and ML utility of generated data.

### Weaknesses
1. While a common latent space across tables provides a strategy to work on heterogeneous tables, it certainly limits the ability to interpret what the embeddings in the space mean - and the authors have not studied this aspect (to clarify, this is different from the privacy plots)
2. The pre-trained LM to emit embeddings for rows - individually for each column type, while preserving tabular structure - raises questions about scalability to enterprise tables - which have thousands of columns associated with each table (and can become even bigger due to joins, lineage additions, etc). Since this directly impacts the dimension (Eq 1, Page 4) - it can have an impact on the representative power when reduced to a common low dimensional space (with tables with much smaller number of columns)
3. Many tables often have short form/ abbreviated or cryptic tabular headers - where the tokenizer of the pre-trained LM can suffer - the authors can potentially add a study on this.

### Questions
Please address the weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
3
