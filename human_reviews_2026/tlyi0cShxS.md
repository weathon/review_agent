# MSFT: Mitigating Spurious Correlations in Text Classification via Feature Induction in Embedding Layers and Tensor Stretching

- Decision: Reject
- Scores: 2, 4, 2, 6

## Abstract
Text classification stands as one of the core tasks in natural language processing (NLP). However, it has long been plagued by the issue of spurious correlations, where models tend to learn associations between non-informative or spurious input features and class labels, thereby constraining generalization capability and classification performance. To tackle this challenge, we formalize the notion of a ‘semantic centroid' and, leveraging this construct, propose a novel plugin termed MSFT (Mitigating Spurious Correlations in Text Classification via Feature Induction in Embedding Layers and Tensor Stretching). The MSFT plugin first computes a semantic centroid—encapsulating the global semantic information of the entire dataset—by aggregating all embedding tensors within the dataset. During model training, it alleviates spurious correlations through tensor distance stretching in the embedding space, specifically targeting the subset of data that drives the formation of such spurious associations. Designed for seamless integration with existing classification architectures, MSFT boasts strong generality and scalability. We conduct experiments on two representative categories of language models: BERT-style masked language models (MLMs) and autoregressive large language models (LLMs; e.g., GPT-2). Extensive experiments across multiple datasets demonstrate that our plugin effectively mitigates the detrimental impacts of spurious correlations while consistently improving classification performance. Notably, it achieves or even surpasses state-of-the-art (SOTA) benchmarks in spurious correlation mitigation for text classification, regardless of the underlying model architecture.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a lightweight plugin, MSFT, that mitigates spurious correlations in text classification by manipulating embedding representations. It formalizes a “semantic centroid”, representing the global semantic information of a dataset, and introduces a tensor distance stretching mechanism in embedding space to weaken overrepresented, spurious associations. The authors claim MSFT can be seamlessly integrated into both BERT-style masked language models and autoregressive LLMs, achieving or surpassing state-of-the-art performance in mitigating spurious correlations.

### Strengths
(1) MSFT shows clear and repeatable gains in worst-group accuracy and overall accuracy across datasets and models.

(2) MSFT works effectively on both BERT-style and GPT-type models without modifying architecture.

(3) MSFT functions as a plug-in with only one main hyperparameter, making it simple and computationally cheap.

### Weaknesses
(1) The research gap is not clearly articulated. The introduction briefly mentions existing methods’ complexity and hyperparameter sensitivity but fails to explain why a new method is necessary, or how MSFT differs fundamentally from prior representation-level debiasing or centroid-based regularization approaches.

(2) The intuition -- that contracting embeddings toward a semantic centroid mitigates spurious correlation -- is presented without theoretical grounding or sufficient empirical justification.

(3) The definition of “excess samples” in Section 3.2.1 is ambiguous. It is unclear whether count(y) refers to class frequency within each cluster Ci or across the entire dataset.

(4) The notion that “excess samples are high-risk instances” is asserted without empirical evidence -- no visualization, distributional analysis, or ablation is provided to support that these samples indeed drive spurious correlations.

(5) The term “semantic centroid” is overclaimed as a “rigorously proven” construct. Its validation is purely empirical (via cosine similarity comparisons) and lacks mathematical rigor.

(6) Only one baseline family (NFL-CO/NFL-CP) is compared. The lack of broader baselines (e.g., GroupDRO, IRM, DFR, JTT) limits the credibility of “state-of-the-art” claims.

(7) The datasets used (Amazon, IMDB, Yahoo!) lack explicit spurious feature annotations. Hence, it is unclear what constitutes “spurious correlations” in these benchmarks.

(8) The evaluation metric (worst-group accuracy) is inadequately defined -- there is no explanation of how groups are constructed in datasets without subgroup labels, or why W-ACC can effectively demonstrate robustness to spurious correlation.

(9) Need a more detailed README file explaining how to use the shared code.

### Questions
(1) Clarify how count(y) is computed for “excess samples”.

(2) Explain why excess samples are considered high-risk.

(3) Specify which embedding layer is used for centroid computation and stretching.

(4) State whether centroids/clusters are computed once on training data or updated dynamically during training.

(5) Define what spurious features exist in the chosen datasets.

(6) Explain how subgroups for worst-group accuracy (W-ACC) are defined in datasets lacking group annotations. And justify why W-ACC demonstrates robustness to spurious correlations in this context.

(7) Include stronger baselines to validate comparative claims.

### Soundness
2

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
2

### Summary
"MSFT ... " is a paper that aims to combat spurious correlations by, defining a dataset-level semantic centroid in embedding space, and then stretching samples that are overrepresented towards that centroid via a fine-tuning procedure. This method is evaluated on Amazon Reviews, IMDB, and Yahoo datasets, with the models of GPT-2 and several BERT-style models.

### Strengths
1. The algorithm presented is to the best of my knowledge, novel and is an interesting method. 

2. The method doesn't require any specific data (e.g. counterfactual labels or spurious attribution data)

### Weaknesses
1. I think that the algorithm 1 in the appendix makes the algorithm much more clear, it might be worth moving this into the main text to help with clarity.

2. The models and datasets used are outdated, and do not appear to actually even be SOTA. I would consider raising my score if the authors were able to evaluate this method on more challenging datasets, such as the classification tasks presented in the MTEB (Massive Text Embedding Dataset).

### Questions
Minor Errata:
on line 221, there is a reference for (Textbook, (2010)) which I think is mislabeled. 

1. It is not fully clear how it is measured that this method actually reduces the amount of spurious correlations, how is this measured?

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
This paper introduces an approach which can be used as a plugin with modern deep learning models to mitigate spurious correlations that often bothers classifiers. They first divide the entire dataset into K clusters and then identify "excess samples" in each cluster which potentially contribute to spuriousness. For these samples, they stretch their sample embeddings towards the cluster centroid to debias them and show improvements in overall and worst group accuracy on 3 datasetes.

### Strengths
1. The idea of debiasing the embeddings by stretching them towards a common dataset-specific centroid is simple and interesting and also leads to improvements in tackling spurious correlations in data.
2. The authors show results across multiple models like BERT, GPT, Roberta, Deberta, etc. * 3 datasets and observe consistent improvements with their MSFT plugin.
3. Ablation study to show centroid has higher similarity to dataset samples compared to in-between sample similarity on average. Also post stretching label-wise data similarity drops.

### Weaknesses
1. Lack of baseline definitions or strong baselines. Authors report NFL baseline as SOTA but no justification for the same is given or description about it. Related Works need to be strengthened too with more recent works like - Elastic Representation: Mitigating Spurious Correlations for Group Robustness (https://arxiv.org/pdf/2502.09850), Improving Group Robustness on Spurious Correlation via Evidential Alignment (https://arxiv.org/abs/2506.11347). Add details how they compare with your work and if these could be potential baselines ?

2. Although the authors report consistent improvements, there's no justification or analysis provided for the same to understand what exactly is happening. I have many questions - (a) why is k-means expected to create clusters which inform us of "excess samples" or spurious-prone samples within the cluster ? This is a strong inductive bias assumption made and needs to be validated with theoretical proofs or some analysis (b) what if i simply drop the "excess samples" instead of stretching ? (c) why is the stretching done at input embedding layer vs final/intermediate/all layers ? (d) How will this approach fare for highly label-imbalanced data vs label-balanced data ?

3. In general, the paper needs to be more rigorous of the claims. Suggest doing some analysis on toy synthetic dataset in controlled fashion to uncover insights on why this approach helps rather than just treating de-biasing/stretching as blackbox high-level objective which is supposed to help. Or maybe try coming up with theoretical guarantees for MSFT.

### Questions
1. Table 3 caption: baseline indicates SR =1 or 0 ?
2. No need to waste space for Section 6.1 . It's trivial to show that centroid will be closer than other datapoints on average.
3. Also the methodology is currently taking up lot of real estate on the paper. No point in introducing so many notations. The approach can be explained in much lesser space and use that space to do more rigorous analysis as suggested above.
4. Also add standard deviation numbers for each table/result shown and the statistical significance of results. Such approaches usually tend to be very sensitive to hyperparamters.

### Soundness
3

### Presentation
2

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
The paper provides a a representation learning based data-manipulation strategy for training. To that end, it leverages the concept of semantic centroids of datasets, which is calculated by a mean of K-means centroids of the embedding representation of datapoints.  
The approach is a drop in function replacement at the [CLS] token layer (for BERT) and final embedding layer for other models.  
The approach is evaluated across six-pretrained models and three datasets.

Suggestions:  
1. The entire section Line 054-078 is a related work discussion and should be moved to the corresponding section on related work.  
2. Line 219: Cite this for elbow method.
@article{Thorndike_1953, title={Who Belongs in the Family?}, volume={18}, DOI={10.1007/BF02289263}, number={4}, journal={Psychometrika}, author={Thorndike, Robert L.}, year={1953}, pages={267–276}}
Also please go through:
@article{10.1145/3606274.3606278, author = {Schubert, Erich}, title = {Stop using the elbow criterion for k-means and how to choose the number of clusters instead}, year = {2023}, issue_date = {June 2023}, publisher = {Association for Computing Machinery}, address = {New York, NY, USA}, volume = {25}, number = {1}, issn = {1931-0145}, url = {https://doi.org/10.1145/3606274.3606278}, doi = {10.1145/3606274.3606278}, journal = {SIGKDD Explor. Newsl.}, month = jul, pages = {36–42}, numpages = {7} }
for a better K decision strategy  
3. Figure 1: stretched centroid -> semantic centroid  
Encoder#2 -> Encoder#1
4. A lot of sections in the paper have been edited/written by LLM. As per ICLR author guidelines, please state that in the paper. This can be seen from double-hyphenated text present throughout the paper.  
5. Table 2: Also mention what OSR is in the table description - or change it with O-$\alpha$ for consistency
6. Line 355-357: Details are needed for how Autoregressive models have been used for classification.

### Strengths
1. Section 6 is great in providing validation of the proof of concept of semantic centroid for NLP classification datasets.
2. Easy plug-and-play replacement model for robust classification model training.

### Weaknesses
1. Paper clarity in section 3. 3.1 does not provide much useful information in building up other parts when preliminary section is present. Also the section is confusing. Are the authors doing K-means multiple times? (3.2.1 and 3.2.2). If they are the same the compress this section.
2. Missing additional evaluation metrics like F1 scores, statistical significance testing.
3. The datasets are not particularly representative in the sense that the amazon and IMDB datasets are review datasets. Suggest the authors to look at more representative datasets for text classification (eg. starting with GLUE benchmark)

### Questions
1. Line 179: What is SR?  
2. What is the need for spurious correlation definition (section 3.1), Line 154-158  
3. Line 300-302: Please report the F1 scores as well. Accuracy is not a sufficient metric. Also please do a statistical significance test for the approach.  
Can you share the dataset statistics for training, validation and test for each of the datasets?  
4. Line 465: Should it not be SR = 1? (instead of SR=0) Also please use consistent notation throughout.  
5. Why is semantic centroid calculated as the mean of K-Means centroids and not the entire mean of the embedded dataset?

### Soundness
3

### Presentation
2

### Contribution
3
