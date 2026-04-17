# Low-pass Personalized Subgraph Federated Recommendation

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Federated Recommender Systems (FRS) preserve privacy by training decentralized models on client-specific user-item subgraphs without sharing raw data. However, FRS faces a unique challenge: subgraph structural imbalance, where drastic variations in subgraph scale (user/item counts) and connectivity (item degree) misalign client representations, making it challenging to train a robust model that respects each client’s unique structural characteristics. 

To address this, we propose a Low-pass Personalized Subgraph Federated recommender system (LPSFed). LPSFed leverages graph Fourier transforms and low-pass spectral filtering to extract low-frequency structural signals that remain stable across subgraphs of varying size and degree, allowing robust personalized parameter updates guided by similarity to a neutral structural anchor. Additionally, we leverage a localized popularity bias-aware margin that captures item-degree imbalance within each subgraph and incorporates it into a personalized bias correction term to mitigate recommendation bias. Supported by theoretical analysis and validated on five real-world datasets, LPSFed achieves superior recommendation accuracy and enhances model robustness.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
In this paper, the authors introduce Low-Pass Personalized Subgraph Federated Recommendation (LPSFed) for personalized federated recommendation systems. LPSFed could address the challenge of subgraph structure imbalance and local popularity deviation. LPSFed uses low-pass spectral filtering on heterogeneous subgraphs to extract low-frequency structural signals. These signals are used to calculate the personalized structural similarity between each client and the global model. LPSFed also uses a local popularity bias-aware margin to capture the imbalance of item degrees in each subgraph and incorporates it into the personalized bias correction term to mitigate recommendation bias. Extensive results on multiple datasets demonstrate the effectiveness of the approach.

### Strengths
1. This paper is easy to follow and generally well presented.

2. It is a timely topic and a very interesting area to explore.

3. The theoretical analysis is solid.

### Weaknesses
1. The authors acknowledge in the Appendix and Limitations that spectral calculations are performed in the preprocessing stage. If the eigenvectors of the entire graph (or subgraph) are directly calculated on a very large graph (such as Tmall), it may not be feasible in terms of memory/time.

2. The author averages the local margins of all (u,i) of each client into a scalar $M^c$ and uploads it to the server. A single average value cannot reflect the skewness (such as long tail) or multimodal distribution within the client; the average value is sensitive to outliers and will mask important information.


3. Figure 1 shows a 15-client partition for Amazon-Book; however, D.2 in the main experiment states, "In the main experiment, we partitioned the full graph into four subgraphs." The paper does not clearly explain the correspondence between the different experimental settings and the reasons behind them.

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
6

### Rating Number
6

### Confidence
5

### Summary
The paper introduces LPSFed, a federated recommendation framework designed to handle structural imbalance across user–item subgraphs of different clients. This imbalance arises from variations in subgraph density and connectivity, which often lead to inconsistent representation learning when applying traditional GNN-based federated models. To address this issue, LPSFed employs low-pass spectral filtering to capture smooth, low-frequency graph signals, enhancing representation stability under heterogeneous structures. In addition, a popularity-aware contrastive loss is introduced to balance recommendations and reduce local popularity bias. Experiments on several datasets show consistent improvements over baseline federated models, and theoretical results provide further support for the effectiveness of the proposed spectral regularization strategy.

### Strengths
1. The paper offers a novel perspective by applying low-pass spectral filtering to mitigate structural imbalance in federated recommendation. The combination of spectral graph modeling and bias-aware optimization is original and novel.

2. The approach is conceptually clear. The appendix adds useful details on complexity analysis and spectral properties, improving transparency and reproducibility.

3. The study investigates an important but relatively underexplored problem in federated recommendation, namely the structural imbalance among decentralized user–item subgraphs. This imbalance is caused by variations in graph density and connectivity across clients. It can negatively influence the stability and fairness of model training in practical applications. By introducing spectral filtering and bias-aware optimization to address this issue, the paper provides a fresh perspective on improving structural consistency in federated recommender systems. The proposed direction may encourage further research on combining structure-aware learning with privacy-preserving mechanisms to develop more robust and equitable personalized recommendation models.

### Weaknesses
1. The core method of the paper is not sufficiently explained. The authors propose decomposing the embedding layer to generate a low-pass convolution kernel distribution k, which is then used for similarity comparison. However, the paper does not clarify why this specific k is chosen or why it is appropriate for measuring similarity. A more thorough discussion of both the theoretical and empirical reasoning behind this design would help readers understand its advantages over alternative approaches.

2. While the paper claims innovation in "structural imbalance," the uniqueness of the method compared to existing spectral federation methods (such as FedSSP) can be explained more clearly. 

3. Although the description of the main method is relatively clear, the workflow diagram(Figure 2) appears somewhat cluttered, which makes it harder to fully understand the overall process.

### Questions
1.  Please provide a detailed explanation of the motivation behind the core method, specifically the similarity comparison based on the low-pass convolution kernel distribution k. The authors should clarify why this kernel distribution is chosen for measuring similarity and discuss its advantages over other possible approaches from both theoretical and empirical perspectives.

2. The model uses spectral domain filtering (Graph Fourier Transform + low-pass filtering) to remove high-frequency noise while preserving low-frequency structural information. Why is it better to retain the low-frequency signal for similarity comparison?

3. While the goal of federated learning is to preserve user privacy, the paper does not discuss whether the client-uploaded information (such as statistical features, spectral kernel distributions, or structural similarity scores) might pose a risk of indirectly revealing the underlying graph structure. A discussion or analysis of potential privacy implications would enhance the completeness of the work and align better with the core principles of federated learning.

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
3

### Summary
The paper targets the challenge of subgraph structural imbalance in FRS. The authors propose LPSFed which uses graph Fourier transform and low-pass spectral filtering to extract stable structural signals, measures structural similarity to guide personalization, and introduce a popularity-bias-aware margin to mitigate feedback loops. Experiments across five datasets show advantages supported by theoretical analysis.

### Strengths
The authors provide a well-motivated spectral view and lots of theoretical analysis. The qualitative tests show advantageous performance with discussions on robustness, completed by the ablation study of model components and hyperparameter analysis. The limitation of reliance on spectral computations is acknowledged.

### Weaknesses
1. Although the motivation of selected algorithms and theories are well motivated, the work seems to be a nice combination of existing proposals applied to this specific challenge, i.e., subgraph structural imbalance. This is not to undermine the effort and importance of such combination, but the novelty by itself is thus unavoidably limited. 
2. As acknowledged by the authors in the appendix, the reliance on spectral computations raises the computation burden, I'd like to see if authors can provide some thoughts on how that could be alleviated. 
3. It seems only one baseline is from the spectral FL family, please include more baselines from this direction, or otherwise explain the reason. The same applies to bias performance tests, which didn't compare with sufficient fairness-aware FRS works. 
4. The cut-off frequency may have a considerable impact on system performance, yet the authors seem skip the discussion on the frequency value choice.  This also applies to the neutral structural anchor.
5. There seem to be quite some places in the math reasoning that are either vague or too idealized. For example, C1 and C2 only have vague description (actually only C1 has) without specifying how to determine their values; it idealizes the client graphs as having clear k-block community structures, which is quite rare in reality I'm afraid.

### Questions
Could the authors elaborate on the novelty besides combining existing algorithms and methods?
Could the authors provide some discussion on how the computation burden caused by spectral computation may be alleviated?
Can the authors include more fair comparisons with peering baselines such as spectral FRS and fairness-aware FRS?
Could the authors discuss more about the cut-off frequency and structural anchor determination and their impact and sensitivity?
Could the authors relax the idealization in math assumptions and give more specifics on value choices for example for C1 and C2 etc. ?

### Soundness
3

### Presentation
4

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
This paper introduces LPSFed, a federated recommender framework designed to handle subgraph structural imbalance and popularity bias across clients. It applies low-pass spectral filtering to extract stable, low-frequency structural signals and computes a structural similarity between each client and a server-generated reference graph for personalized aggregation. Additionally, it introduces a bias-aware margin loss that mitigates popularity bias by adjusting item ranking based on local degree distributions. Theoretical analysis justifies the stability and regularization of low-pass filtering, and experiments on five real-world datasets demonstrate consistent improvements in recall, NDCG, and robustness over multiple baselines.

### Strengths
- Clear and coherent framework integrating personalization and debiasing.

- Sound theoretical motivation for spectral similarity and low-pass regularization.

- Strong empirical results across multiple datasets and settings.

- Effective ablation studies confirming each module’s contribution.

### Weaknesses
- Novelty is incremental; relies heavily on previous spectral FL ideas.

- Theoretical assumptions may not align with real-world bipartite graphs.

- Ambiguity in privacy handling (what information the server receives).

- Sensitivity to anchor graph design and data partitioning not deeply analyzed.

### Questions
1 Can similarity computation be done entirely on clients to ensure privacy?

2 How sensitive is performance to the reference graph’s design or randomness?

3 Do empirical eigengaps correlate with observed improvements?

4 Can the bias-aware margin adapt automatically rather than be fixed?

5 Has the model been tested under naturally partitioned (non-synthetic) clients?

### Soundness
3

### Presentation
3

### Contribution
3
