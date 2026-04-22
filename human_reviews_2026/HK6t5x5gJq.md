# FeDaL: Federated Dataset Learning for General Time Series Foundation Models

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 6, 2, 4, 4

## Abstract
Dataset-level heterogeneity introduces significant domain biases that fundamentally degrade generalization on general Time Series Foundation Models (TSFMs), yet this challenge remains underexplored. This paper rethinks the from-scratch training of TSFMs using the paradigm of federated learning. We propose a novel Federated Dataset Learning (**FeDaL**) approach to tackle heterogeneous time series by learning dataset-agnostic temporal representations. Specifically, the distributed architecture of federated learning is a nature solution to decompose heterogeneous TS datasets into shared generalized knowledge and preserved personalized knowledge. Moreover, based on the TSFM architecture, FeDaL explicitly mitigates both local and global biases by adding two complementary mechanisms: Domain Bias Elimination (DBE) and Global Bias Elimination (GBE). FeDaL`s cross-dataset generalization has been extensively evaluated in real-world datasets spanning eight tasks (including various regression and classification), against 54 baselines. We further analyze federated scaling behavior, showing how data volume, client count, and join rate affect model performance under decentralization. Our code is publicly available at https://github.com/shengchaochen82/FeDaL.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this work, the authors propose a novel Federated Dataset Learning (FeDaL) method to address heterogeneous time series data. FeDal mainly considers domain bias elimination (DBE) and global bias elimination (GBE), which alleviates the domain bias caused by heterogeneous time series datasets and enables learning domain-invariant time representations. Extensive experimental results on several datasets show that FeDaL outperforms existing methods in many downstream tasks.

### Strengths
1. The presentation of the proposed method is clear, and the paper is overall well-written.

2. This paper considered the challenge of time-series dataset heterogeneity, which is a significant and novel problem in FL.

3. The evaluation of the proposed method is comprehensive.

### Weaknesses
1. In this work, the foundation model is trained directly on the client without considering the computational burden and parameter-efficient fine-tuning (PEFT) method.

2. The paper proposes several types of bias (temporal resolution, physical constraint, pattern transition), but does not provide a reproducible quantitative metric (how to label a dataset as "high resolution bias"?). Therefore, it is impossible to determine which datasets are more affected by DBE.

3. The experiments were compared with a large number of time series models. However, these methods are not designed for federated learning. This is not a fair and convincing comparison. At the same time, among the baselines for federated learning, only FFTS is designed for time series. More baselines for federated time series and federated time series foundation models should be considered, such as [1][2].

4. It could be better to report the number of repetitions and standard error for each experiment and add a description of the variance.


[1] Liu, Qingxiang, et al. "Time-FFM: Towards LM-empowered federated foundation model for time series forecasting." In NeurIPS 2024.

[2] Xu, Ronghui, et al. "PeFAD: a parameter-efficient federated framework for time series anomaly detection." In KDD 2024.

### Questions
See in weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a solution to handle heterogeneity when learning representations for time series foundational models. They focus on the federated scenario, highlighting three sources of heterogeneity: mismatch in sampling frequencies, heterogeneity in the underlying physical processes, and divergence due to exogenous events. They propose two solutions: domain bias elimination (DBE) and Global bias elimination (GBE), at the client and server levels respectively. DBE decomposes a series into trend and seasonal components per client and learns a common representation across local datasets. GBE tackles the remaining misalignments through a moving-average like process to avoid client-specific patterns from dominating the global trend. They further refine it using a core-set fine-tuning. The method is evaluated on IID and non-IID scenarios showing improvements over other federated baselines at learning representations. Comparisons are also made against other time series foundation models on several downstream tasks using the learned representations, showing improvements over the baselines.

### Strengths
- The paper shows many experimental results and ablation studies to show how it performs on multiple aspects. The breadth of baselines covered is impressive.
-  Performance results are good.
- The research problem is very relevant for the community

### Weaknesses
- The novelty of the method is weak. The method simply combines existing works and plugs them into the framework. For e.g. DBE applies decomposition (Wu et al. 2021) and EMA from (Zhang et al. 2015). GBE uses drift correction from (Acar et al. 2021) and core set tuning from (Killamsetty et al 2021). 
- Many of the federated baselines in Table 1 are quite old, except for FFTS. Newer baselines like Time-FFM (Liu et al. 2024) are missing.  
- Table 1 also does not compare against strong centralized TSFM baselines, where data from all clients is pooled. For e.g. Moirai and Chronos. I understand that these comparisons have been made on downstream tasks, but it is also necessary to show these results in table 1 to see how close a distributed approach is to centralised representation learning.
- The method’s performance improvement over the baselines seems highly dependent on all components being present (Table 2). I’m wondering if adding just one or two components to an existing baseline e.g. FFTS would make it even stronger than the proposed method.
- The results on privacy-preservation are not clear. In the TSNE plots in the appendix and Fig. 6, the authors claim that the perturbed points (blue) are privacy-preserving. But visually, to me it looks like the green (initial core set) is further away from the original data (pink). These figures should be accompanied with a quantitative metric indicating how close or far the perturbed points are from the actual data, instead of relying on visualizations alone.

### Questions
See the weaknesses W1-W4. Additionally I have the following minor comments.
- The code link provided for the baselines in Appendix B page 26 is not working. 
- The imbalance factor of  H1 and H2 (Appendix page 21) seem quite small (0.1 and 0.3). I wonder how well the method would perform on UTSD if this is increased to say 0.7+. 
- The absolute number for the core-set size in table 8 is not sufficient. I think it is important to also indicate the percentage w.r.t the training data size.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces FeDaL, a federated learning framework designed to eliminate dataset-level biases in time series foundation model (TSFM) pretraining. The method tackles the limitations of prior approaches that treat heterogeneity too coarsely, neglecting dataset-level structural biases and compromising cross-domain generalization. Specifically, FeDaL integrates two complementary components—Domain Bias Elimination (DBE) and Global Bias Elimination (GBE)—operating at the client and server levels respectively. DBE disentangles local biases by decomposing masked input representations into trend and seasonal components, injecting a trainable bias vector during reconstruction. GBE introduces a server-side state vector to track accumulated client drift and applies gradient-level correction during aggregation. The paper further analyze federated scaling behavior, demonstrating that federated TSFM pretraining benefits more from data diversity and client participation than from model size. Overall, the work offers a modular and well-motivated approach to improving generalization in federated time series modeling.

### Strengths
1. The paper addresses a well-motivated problem in federated time series modeling, focusing on dataset-level bias and cross-domain generalization.
2. The proposed FeDaL framework is modular and clearly structured, integrating client-side and server-side bias elimination mechanisms (DBE and GBE).
3. The gradient-level correction mechanism via a server-side state vector is a meaningful extension of FedAvg, with clear mathematical formulation.
4. The scaling analysis in Section 4.3 provides valuable insights into federated pretraining dynamics.

### Weaknesses
1. Unclear Structure in Problem Statement: The introduction claims that existing works face “two major challenges,” but only one (coarse-grained treatment of heterogeneity) is explicitly developed. The second challenge is not clearly introduced or elaborated, which weakens the framing of the paper’s motivation.
2. The Global Bias Elimination (GBE) module introduces a cumulative server-side state vector sr, which may be prone to instability over long training rounds if not properly scaled. While the paper provides empirical evidence supporting the choice of β, further theoretical analysis or broader sensitivity exploration would strengthen confidence in its stability.
3. Occasional Conceptual Overreach in Terminology: Certain statements, such as “the distributed architecture of federated learning is a nature solution to decompose heterogeneous TS datasets,” are methodologically oversimplified and should be more rigorously justified. 
4. The caption of Figure 1 refers to four settings—FedAvg-Independent, FedAvg-Mixed, Stand-alone, and Transformer—yet the figure itself only visualizes the first three. The Transformer setting is neither shown in the bar plots nor explained in the figure annotations.

### Questions
1. The paper describes the DBE (Dataset Bias Estimation) module as “plug-and-play,” yet its integration requires latent feature decomposition, bias vector computation, and a modified reconstruction loss with an additional alignment term. Could the authors clarify in what sense the module is truly plug-and-play? Specifically, does it require architectural changes, joint optimization, or task-specific tuning? Has its generalizability across different backbone architectures or tasks been empirically validated?
2. The paper suggests that federated learning naturally decomposes heterogeneous time series data into generalized and personalized components. However, federated learning was originally designed to enable privacy-preserving collaborative modeling, not to perform automatic representation disentanglement. It remains unclear whether this decomposition is an emergent property of the federated setup or the result of explicit modeling choices. Could the authors clarify what mechanisms, if any, enable this decomposition and whether it goes beyond the standard federated architecture?

### Soundness
3

### Presentation
1

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
This paper presents a federated pre-training framework for learning dataset-agnostic temporal representations in heterogeneous settings.
The method involves two mechanisms: (1) Domain Bias Elimination (DBE), which decomposes local embeddings into shared and private components, aiming to mitigate client-specific distortions, and (2) Global Bias Elimination (GBE), which corrects cross-client gradient drift via bias-aware aggregation and core-set tuning. The paper reports experiments on several datasets with numerous baselines, empirically showing improved generalization under non-IID conditions.

### Strengths
The paper is well-written and presents results of a comprehensive experimental study. The problem that it addresses is timely and had previously not been explored enough. The proposed two-level design (DBE for local, GBE for global bias) appears modular and easy to integrate.

### Weaknesses
While the paper reports a strong engineering effort, it unfortunately lacks a clear algorithmic or theoretical innovation. The framework combines concepts from a number of prior works, including (Nie et al., 2022, Wu et al., 2021, Zhang et al., 2015, Acar et al., 2021, Killamsetty et al., 2021), to establish DBE and GBE but these are ultimately heuristic -- there is no theoretical analysis of "bias elimination", or of the convergence of the proposed federated scheme. Additionally, it is not clear to this reviewer whether the framework performs genuine bias correction or is the improvement achieved due to introduced regularization -- it seems plausible that a regularization could change how the model fits the data and thus improve generalization even when underlying distributional gaps across domains remains unaddressed.

### Questions
What distinguishes FeDaL from existing domain-invariant FL approaches such as FedBN, MOON, or Ditto?
Can you quantify the additional communication or computation cost introduced by DBE and GBE?
How sensitive is performance to the decomposition granularity (trend/seasonal split) or the hyperparameters controlling bias elimination?
A formal or empirical measure of bias reduction (perhaps feature distribution alignment?) should be included to substantiate the central claim.

### Soundness
3

### Presentation
3

### Contribution
2
