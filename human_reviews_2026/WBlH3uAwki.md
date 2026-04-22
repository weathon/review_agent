# LLmFPCA-detect: LLM-powered Multivariate Functional PCA for Anomaly Detection in Sparse Longitudinal Texts

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Sparse longitudinal (SL) textual data arises when individuals generate text repeatedly over time (e.g., customer reviews, occasional social media posts, electronic medical records across visits), but the frequency and timing of observations vary across individuals. 
These complex textual data sets have immense potential to inform future policy and targeted recommendations. 
However, because SL text data lack dedicated methods and are noisy, heterogeneous, and prone to anomalies, detecting and inferring key patterns is challenging. 
We introduce LLmFPCA-detect, a flexible framework that pairs LLM-based text embeddings with functional data analysis to detect clusters and infer anomalies in large SL text datasets. 
First, LLmFPCA-detect embeds each piece of text into an application-specific numeric space using LLM prompts. 
Sparse multivariate functional principal component analysis (mFPCA) conducted in the numeric space forms the workhorse to recover primary population characteristics, and produces subject-level scores which, together with baseline static covariates, facilitate data segmentation, unsupervised anomaly detection and inference, and enable other downstream tasks. 
In particular, we leverage LLMs to perform dynamic keyword profiling guided by the data segments and anomalies discovered by LLmFPCA-detect, and we show that cluster-specific functional PC scores from LLmFPCA-detect, used as features in existing pipelines, help boost prediction performance. 
We support the stability of LLmFPCA-detect with experiments and evaluate it on two different applications using public datasets, Amazon customer-review trajectories, and Wikipedia talk-page comment streams, demonstrating utility across domains and outperforming state-of-the-art baselines.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents an approach on clustering and finding anomalies in sparse longitudinal data: data with irregular timestamps associated to one user. It combines LLM generated embeddings with mFPCA for clustering and anomaly detection. The authors argue that current methods either ignore temporal structure or miss guarantees for errors. The method is evaluated on two datasets and some insights are extracted from the clusters, based on the most frequent keywords.

### Strengths
- Timely problem: Anomalies in SL text is a relevant topic that has applications in several fields.

- Clear explanations and intuitions at the high level: In spite of the heavy formalizations, one can follow easily the ideas behind the paper. The high-level view of the pipeline is clear.

- It seems that adding the emotion mFPC scores improves predictions on some downstream tasks.

### Weaknesses
- Formalization is not very rigorous. Some concepts are not properly explained, which makes the reading hard. For example, Algorithm 2 uses Algorithm 7 and 8, but they are not explained. These are important components of the overall pipeline, so at least an intuition or a high-level formalization would improve the quality of the paper.
- The scientific contribution is not very novel. mFPCA is well established and appending the LLM embeddings plus Algorithms 2 and 3, which only apply some postprocessing to the output of mFPCA is not an idea that one can then transfer to other domains.
- Unclear explanations of experiments, see below for questions. There are some things that are not clear about the experiments, so it is hard to assess their quality.

### Questions
**Data heterogeneity and anomalies**

- If the Karhunen-Loeve expansion is applied as written, clusters differ only in the cluster mean. Why is it valid to assume all clusters share the same models of variation (eigenfunctions)? In the Amazon example, should we expect the "poor-quality reviews" cluster to share eigenfunctions with the "good-reviews" cluster?
- Line 164; What is T_0? Also, if anomalies are defined by $a_i(t) \neq 0$, how do you distinguish a true anomaly from measurement error when $a_i(t)$ is small?
- Line 170: You approximate $C_k \cup A_0$ by $A_0^{k, \epsilon}$, but later use the distribution of FPC scores from the "clean" subset $C_k \cup A^C_0$. How is that clean distribution obtained if you only had an approximation earlier?

**Methods: Pipeline and estimation**

- Line 237: What does "clustering their estimated mFPC scores jointly with static covariates" mean operationally? Are covariates appended as additional dimensions in the clustering vectors? If so, how are covariates encoded and scaled with respect to the mFPC scores?
- Algorithm 3: How are detection windows defined (length, stride)? How many windows are used per subject? Are they overlapping or disjoint, and how sensitive are results to this choice?
- The Dynamic Keyword Profiling description is quite short. How exactly are LLMs used for this? What text is passed to the model and how are outputs refined?

**Experiments**

- How do you choose the embedding function in practice? For Amazon, how were the LLM-derived emotion scores obtained? Why use zero-shot rather than few-shot prompts?
- Why was an older GPT-3.5 model used? Do you compare embeddings or labelers from other models and report performance differences?
- Line 373: What is Table 6? (It seems referenced but not present)
- How many anomalies were detected in total, and how are they distributed across clusters and time windows?
- On Wikipedia, you mention BERT baselines, but why isn't there a comparison of LLmFPCA-detect using BERT-based encoders versus GPT-based encoders?
- How were comments selected from the full Wikipedia dataset? For the 925 users, how many total comments and time points are included after filtering?
- Which GPT model produced toxicity/aggression scores, and how do these scores correlate with human annotations?

### Soundness
3

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
3

### Summary
This paper proposes LLMFPCA-DETECT, a novel framework combining LLM-based embeddings with multivariate Functional Principal Component Analysis (mFPCA) to analyze sparse longitudinal (SL) textual data—datasets where individuals generate text intermittently and irregularly over time (e.g., reviews, social media posts, medical notes).

### Strengths
1. The combination of LLM embeddings and functional data analysis (mFPCA) is both creative and technically elegant. It bridges modern NLP with classical statistical modeling, addressing an underexplored setting—sparse, irregular longitudinal text.
2. The paper clearly defines the challenge of SL texts, distinguishing them from standard time series or dense sequences. The motivation for type-I-controlled anomaly detection in irregular textual data is compelling.

### Weaknesses
1. The empirical evaluation mainly compares against Isolation Forests (on BERT and GPT embeddings). This feels weak given the paper’s ambition. 
2. The performance hinges on the quality of LLM-derived embeddings (e.g., emotion or toxicity scores). However, prompt variability or model drift could affect reproducibility.
3. mFPCA requires covariance estimation and eigen-decomposition, which may scale poorly with large numbers of subjects or long trajectories.

### Questions
1. Can this approach generalize to dense sequences or multimodal signals (e.g., text + physiological data)?
2. Are there theoretical guarantees for cluster recovery under sparse observation settings?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper addresses the challenge of analyzing sparse longitudinal (SL) textual data—settings where individuals generate text at irregular times (e.g., customer reviews, social media posts, or clinical notes). Existing text mining methods (e.g., BERT-based embeddings or time-series models) assume dense, regular sampling and thus fail to handle the irregular timing, heterogeneity, and noise in SL data. The proposed LLmFPCA-detect framework integrates Large Language Models (LLMs) with multivariate functional principal component analysis (mFPCA) to enable clustering and anomaly detection on SL text streams.

### Strengths
The paper presents a novel integration of LLM-derived embeddings with statistical functional analysis via mFPCA. It introduces outlier detection, featuring a two-stage anomaly calibration procedure (Algorithms 2 and 3) that employs sample splitting and multiple-comparison control. The authors also provide empirical studies demonstrating the effectiveness of the proposed approach.

### Weaknesses
The novelty of the work is not very clear, as the proposed method appears to combine elements of existing algorithms rather than introducing a fundamentally new approach. 

Moreover, the paper lacks strong theoretical justification. There are some guarantees in appendix, but their scopes seem limited. 

Given the absence of strong theoretical results, the empirical validation also feels limited. A broader set of simulations or comparisons would strengthen the paper’s overall contribution.

### Questions
Could the authors more clearly articulate the novelty of the proposed method? In its current form, the approach appears to be a combination of existing algorithms, and the unique conceptual or technical contribution is not immediately evident.

The overall algorithmic pipeline is quite long, and several key components are deferred to the appendix. It would be helpful to include a concise “prototype” version of the algorithm in the main text—summarizing the core steps and key ideas—so that readers can quickly grasp the main workflow without referring to supplementary materials.

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
2

### Summary
This paper presents a pipeline for the analysis of anomalies in sparse longitudinal texual data. The raw texts are converted to numerical vectors with prompting an LLM, and the mFPCA is used to extract the mFPC scores, which are the basis for later postprocessing steps, including clustering, anomaly detection, and keyword extraction. The proposed pipeline is demonstrated on an amazon review dataset and a Wikipedia request-comment dataset. Interesting patterns are shown by the clusters and anomalies are interpreted with the help of the extracted keywords. It is obvious that the proposed pipeline could be appiled to many more scenarios.

### Strengths
* The pipeline decribed in the paper shows an end2end procedure to analyze textual corpus with timestamps. The highlight of the algorithm is the interpretability of the anomalies found, which is very useful in real scenarios.
* The application of the pipeline is demostrated on two use cases in detail. The many qualitative results are interesting to read.
* The description of the algorithms in the many Alg blocks are clear and helpful for readers to understand.

### Weaknesses
* It is a complex system and have many steps. The design of some important steps are only presented but not justified. So, it reads somewhat like product white paper. (1) Why mFPCA instead of uFPCA, and what is the benefit? (2) Is Alg. 2 a standard way of anomaly detection (if so, what's the citation), or a novel one (if so, why is it a good design)? (3) Why clustering is required at all, and how to determine the number of clusters?
* The evaluation are mostly qualitative, and very few baselines are compared with.

### Questions
* At L157, the modeling uses "cluster-specific mean function" and "shared eigenfunctions across clusters", while at L240, cluster-specific mean and eigenfunctions are used.
* L191. Why is the output of LLM deterministic, and why does this determinism matter? Generally, LLM outputs are sampled with randomness.
* Why is the *functional* version of PCA required? What if the time axis is dropped (or divide into groups by time window if localization is wanted), and do regular PCAs; Will anomalies still identifiable?
* What does the boldface mean in Tab. 1?
* L317. Could you elaborate on how the baseline "by mean Automobile rating" works? Beside this baseline, are there more baselines?
* The lower two rows of Tab. 3 is identical. Is this a typo or by chance?
* L413. Which method is referred as the "state-of-the-art", any citation for the method?

### Soundness
3

### Presentation
2

### Contribution
2
