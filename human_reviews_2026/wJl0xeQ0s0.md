# TabPFN-Wide: Continued Pre-Training for Extreme Feature Counts

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 4, 2, 2

## Abstract
Revealing novel insights from the relationship between molecular measurements and pathology remains a very impactful application of machine learning in biomedicine. Data in this domain typically contain only a few observations but thousands of potentially noisy features, posing challenges for conventional machine learning approaches. While prior-data fitted networks emerge as foundation models for tabular data, they are currently not suited to handle large feature counts ($>500$). Although feature reduction enables their application, it hinders feature importance analysis. We propose a strategy that extends existing models through continued pre-training on synthetic data sampled from a customized prior. The resulting model, TabPFN-Wide, matches or exceeds its base model's performance while exhibiting improved robustness to noise. It seamlessly scales beyond $50{,}000$ features, regardless of noise levels, while maintaining inherent interpretability, which is critical for biomedical applications. Our results show that prior-informed adaptation is suitable to enhance the capability of foundation models for high-dimensional data. On real-world biomedical datasets many of the most relevant features identified by the model overlap with previous biological findings, while others propose potential starting points for future studies.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This mainly investigates how continued pre-training with a tailored HDLSS synthetic prior extends a tabular foundation model (TabPFNv2) to generalise to extreme feature counts (up to >50k) without sacrificing performance on standard ranges and while retaining per-feature interpretability.

### Strengths
1. **[Important] HDLSS synthetic prior + feature widening algorithm.** The paper proposes a masked linear projection with sparsity *p* followed by noise injection to widen moderate-dimensional synthetic datasets into *d*-dimensional HDLSS data. This produces correlated “clusters” of features that mimic real omics correlation structures.
2. **Empirical improvements on real HDLSS + benchmarks.** On 4 TCGA multi-omics datasets with 26k–60k features, TabPFN-Wide attains higher AUROC and outperforms TabPFNv2/TabICL, matching tree baselines and RealMLP-TD; on standard TabArena tasks (≤500 features) performance remains on par with TabPFNv2 (i.e., no catastrophic forgetting).
3. **Inherent interpretability via attention-to-label.** After disabling feature grouping so each token maps to a feature, average attention-to-label correlates with feature importance.

### Weaknesses
1. **[Important] Missing important benchmark methods for HDLSS datasets.** Although the results show that the proposed method could improve the performance of TabPFN, it would be more convincing to compare TabPFN-wide with those methods designed for HDLSS datasets. Only comparing TabPFN-wide with the existing methods may provide obscure conclusions as the other models may not achieve satisfying performance for meaningful comparisons. Specifically, I would highly suggest the authors refer to the corresponding literature for specific methods [1, 2, 3].
2. **[Important] The claimed “interpretability” could be misleading.** If I understand correctly, the “interpretability” of TabPFN-wide seems to be post-hoc analysis on the model behavior. Thus, I am unsure such post-hoc analysis can really be considered as a “characteristic” of the proposed model.
3. **Limited coverage of benchmark HDLSS datasets.** I would suggest the authors refer to [3] for more comprehensive dataset setups. In HDLSS settings, the conclusion could encode much greater variance than normal datasets, and thus it is more of a necessity to include comprehensive datasets given the paper claims TabPFN-wide to be a “foundation predictor”.
4. **Analysis on TabICL seems insufficient.** Failure to adapt TabICL may reflect prior mismatch rather than architectural limitation; I would appreciate it if the authors could expand on this.

[1] Yang, Junchen, Ofir Lindenbaum, and Yuval Kluger. "Locally sparse neural networks for tabular biomedical data." *International Conference on Machine Learning*. PMLR, 2022.

[2] Jiang, Xiangjian, et al. "ProtoGate: prototype-based neural networks with global-to-local feature selection for tabular biomedical data." *Proceedings of the 41st International Conference on Machine Learning*. 2024.

[3] Ye, Han-Jia, et al. "Revisiting Nearest Neighbor for Tabular Data: A Deep Tabular Baseline Two Decades Later." *The Thirteenth International Conference on Learning Representations*.

### Questions
Please refer to the "Weaknesses" section.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces TabPFN-Wide, an extension of TabPFN tailored for high-dimensional, low-sample-size (HDLSS) data. Specifically, the authors propose a continued pre-training on synthetic datasets generated from the existing SCM-based TabICL prior and widened through a sparse linear projection with added Gaussian noise. Moreover, the authors study the interpretability of TabPFN-Wide by looking at the attention maps computed within the transformer architecture.

### Strengths
1) TabPFN-Wide addresses an important constraint of TabPFN/TabICL-like models, meaning their inability to handle datasets with a large number of features. The addition of an interpretability analysis based on attention-to-label mechanisms provides an interesting complement to the original model.

2) The code is available and easy to use.

3) The authors provide an honest and transparent discussion of the methodology, acknowledging the limitations and potential weaknesses of their approach.

### Weaknesses
In my view, the contribution is partially incremental relative to the standards of ICLR: 
1) The proposed continued pre-training strategy is not really a contribution of the paper, but more an experimental necessity. 

2)  Interpretability analyses have been previously explored in TabPFN-like models (e.g., [1]).

3) While the proposed HDLSS prior is interesting and practically effective, its design remains largely engineering-driven and lacks a theoretical justification.

[1] Rundel, David, et al. "Interpretable machine learning for TabPFN." World Conference on Explainable Artificial Intelligence. Cham: Springer Nature Switzerland, 2024.

### Questions
1) Previous work has explored the fine-tuning of TabPFN-like models using synthetic data. I was wondering why the authors chose a continued pre-training strategy instead, given that the HDLSS datasets considered in this work already belong to a specific and well-defined data regime. 

2) From Figure A2, it seems that the performance plateau observed for TabICL during continued pre-training might not result from an architectural limitation, but rather from the fact that the HDLSS priors generated are already well represented within the priors seen during the original meta-training. 
Consequently, the lower performance on real HDLSS datasets might instead be explained by the feature reduction procedure rather than by architectural constraints. Could the authors comment on this interpretation?

3) How do the authors ensure that the HDLSS prior, which constructs $X_{\text{wide}}$ via sparse linear projections and additive noise, does not disrupt the original relationship between the original $X$ and $y$?

[1] Bühler, Magnus, Lennart Purucker, and Frank Hutter. "Towards Synthetic Data for Fine-tuning Tabular Foundation Models." 1st ICML Workshop on Foundation Models for Structured Data.

------

I find the paper interesting and technically solid, but not sufficiently novel for the standards of ICLR. Therefore, placing the work in a clear borderline area. This is why, for the moment, I propose a borderline reject.

### Soundness
3

### Presentation
4

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
The paper presents TabPFN-Wide, an extension of TabPFNv2 architecture, focusing on modeling high-dimensional, but low sample size (HDLSS) tasks. The main contribution of the method is the feature-widening prior, which leverages a tree-based structural prior (similar to TabICL) but applies a sparse linear projection to the input features, thereby widening them to the desired target (and much larger) dimension. This procedure generates thousands of new features that are highly correlated with the original feature set. TabPFN-Wide uses continued pre-training, starting from a TabPFNv2 and continuing. to train the model with the new wide priors. Experiments on 4 HDLSS datasets derived from TCGA show that TabPFN-Wide performs slightly better than baselines on cancer subtyping (?) tasks as well as on standard tabular data benchmarks from TabArena. The authors also provide a study on feature interpretability derived from TabPFN-Wide.

### Strengths
- Addressing the HDLSS problem in biomedical applications (e.g., cancer genomics) is well-motivated and of practical importance.
- The manuscript is clear, well-written, and generally easy to follow.

### Weaknesses
- W1: The paper addresses an important practical challenge in tabular data learning that relates to high-dimensional, but low sample size (HDLSS) tasks. However, there is neither a discussion nor a comparison of many approaches that address these tasks. To name a few [1-7].

- W2: The experiments (and results) on the HDLSS dataset seem overly optimistic. Given that the authors have a ~60K/ ~25K dimensional problem (LGG,OV/BRCA,SARC) with ~200 samples and are still able to achieve very high performance (even across baselines). This suggests that the benchmark is oversaturated ie. that the tasks are very trivial. Even the SD, which I assume is from the 5-fold CV, seems very small for such problems. Even more, it is unclear what the actual subtyping tasks are, as they are never documented. A better benchmark would be [8], which also discusses the challenges of these tasks. Many of [1-7] have also been assessed on these data. Tabpfn-wide should also be evaluated against traditional lasso/elastic-net. They typically perform very well in such scenarios, and also allow explainable post-hoc analysis.

- W3: The correlation assumption (Clarke 2008), while valid for certain domains, doesn’t necessarily generalize to other domains/tasks [8] (often there is low correlation between estimated and true error in HDLSS scenarios). Therefore, this might not be beneficial if Tabpfn-wide is applied to tasks other than omics. Moreover, this might also be the reason why there is no significant difference in performance between the size variants of TabPfn-Wide, as well as the lack of effect on TabICL.

---
[1] Romero et al "Diet networks: Thin parameters for fat genomics” 2017

[2] Singh et al “Feature selection network on high-dimensional biological data” 2023

[3] Ruiz et al. "Tabular deep learning when d >> n by using an auxiliary knowledge graph” 2023

[4] Margeloiu et al “Weight predictor network with feature selection for small sample tabular biomedical data” 2023

[5] Bhalm et al "Concrete Autoencoders: Differentiable Feature Selection and Reconstruction” 2019

[6] Liu et al “Deep neural networks for high dimension, low sample size data” 2017

[7] Feng et al “Sparse-Input Neural Networks for High-dimensional Nonparametric Regression and Classification” 2019

[8] Kuncheva et al  “Feature Selection from High-Dimensional Data with Very Low Sample Size: A Cautionary Tale ”2020

### Questions
- See weaknesses
- What are the classification tasks being solved in 5.1? What's the evaluation setup (eg. train/val/test, strata etc.)? How were the baselines tuned for these tasks?
- I would appreciate some clarification on how TabPFNv2 and TabICL were run for the experiments in Section 5.1? It is stated that “Unless stated otherwise, all models were evaluated using all features” (L300). Was feature reduction used as discussed in A2, and if so, which one is the one reported in Table 1?
- The authors should consider providing the full results of the TabArena benchmark, as the figures, while helpful, are insufficient for analyzing the methods' behavior.
- How substantial is the additional computational overhead of cont. pre-training?
- It would be interesting to see how the important features identified by Tabpfn-wide compare to the ones identified by XGB and RF as they also have similar performance.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper addresses the problem of learning from high-dimensional, low-sample-size (HDLSS) data, with applications in biomedicine. The authors propose a continued pre-training strategy to adapt the pre-trained foundation model TabPFNv2 for learning from HDLSS datasets.

Specifically, the paper introduces a widening procedure to generate synthetic HDLSS data, which is then used to continue pre-training TabPFNv2, resulting in an adapted model called TabPFN-Wide. The authors also employ attention maps to analyze feature importance and enhance model interpretability. Numerical experiments demonstrate the performance of TabPFN-Wide on several datasets.

### Strengths
1. The paper proposes a novel continued pre-training approach to adapt TabPFNv2 for HDLSS data.

2. Empirical results indicate improved performance over baseline methods and prior foundation models.

### Weaknesses
1. The contribution appears incremental, and the proposed method is demonstrated only on a single foundation model (TabPFNv2).

2. The approach lacks theoretical rigor and detailed justification for several design choices.

3. The evaluation of feature importance using attention matrices is unclear and insufficiently described.

### Questions
The paper presents a potentially interesting idea of adapting pre-trained foundation models for domain-specific tasks involving HDLSS data. However, the current version of the paper has several shortcomings that limit its impact and clarity. My detailed comments are as follows:

1. Incremental Contribution and Limited Scope

The proposed approach appears incremental and seems to work only for TabPFNv2. Similar ideas—such as generating synthetic data or using linear neural networks with priors to augment features—have been explored previously.

Moreover, the use of averaged attention maps across samples, heads, and layers for feature interpretability has been considered in earlier studies.

It is also unclear whether the proposed feature widening via random synthetic data and continued pre-training can generalize beyond TabPFNv2 to other foundation models such as TabICL. Without such demonstrations, the generality of the approach remains uncertain.

2. Lack of Rigor and Justification

Several aspects of the proposed approach appear ad hoc or insufficiently explained. For example:

i. What motivates the use of random samples drawn from directed acyclic graphs (DAGs)?

ii. What is the role of the mask in Algorithm 1?

iii. Why is Gaussian noise added to the linear layer outputs?

iv. How does Algorithm 1 ensure that the generated synthetic data accurately mimic real-world HDLSS characteristics?

v. The selection of parameters such as the number of features (d) and sparsity (p) also lacks justification.

vi. Additionally, key details about the model architectures (TabPFNv2, TabICL) are missing. It is also unclear what constitutes the input feature matrix X in Algorithm 1—are these features derived from real datasets or synthetically generated?

3. Unclear Use of Attention for Feature Importance

The explanation of feature importance via attention matrices is inconsistent and underdeveloped.

In Section 2, authors say "the role and interpretability of attention maps are controversial in the literature" ; "attention maps may provide a coarse indication of a model’s reasoning process, they are often noisy and can erroneously emphasize irrelevant tokens", and "For TabPFNv2’s attention specifically, earlier research shows that it evolves across layers, shifting from label-focused attention in the first layers to semantically relevant attribute attention in deeper layers". Yet, in section 4.3, attention matrix is used for feature importance, while also claiming "We acknowledge that attention maps can vary substantially across these dimensions."

This contradiction weakens the interpretability claims. A more rigorous justification would strengthen this section.

### Soundness
2

### Presentation
2

### Contribution
2
