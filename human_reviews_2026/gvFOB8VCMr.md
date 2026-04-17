# Rethinking GNNs and Missing Features: Challenges, Evaluation and a Robust Solution

- Decision: Reject
- Scores: 6, 6, 2, 8

## Abstract
Handling missing node features is a key challenge for deploying Graph Neural Networks (GNNs) in real-world domains such as healthcare and sensor networks. Existing studies mostly address relatively benign scenarios, namely benchmark datasets with (a) high-dimensional but sparse node features and (b) incomplete data generated under Missing Completely At Random (MCAR) mechanisms. For (a), we theoretically prove that high sparsity substantially limits the information loss caused by missingness, making all models appear robust and preventing a meaningful comparison of their performance. To overcome this limitation, we introduce one synthetic and three real-world datasets with dense, semantically meaningful features. For (b), we move beyond MCAR and design evaluation protocols with more realistic missingness mechanisms. Moreover, we provide a theoretical background to state explicit assumptions on the missingness process and analyze their implications for different methods. Building on this analysis, we propose GNNmim, a simple yet effective approach for node classification with incomplete feature data. Experiments show that GNNmim consistently outperforms specialized architectures across diverse datasets and missingness regimes.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper identifies two key challenges with existing datasets to evaluate missing feature-based GNNs: a) High-dimensional but sparse node features where most values are zero, b) Uniform MCAR: a missing mechanism applied to generate incomplete or missing features, which the authors claim to be overly simplistic. Moreover, they identify a more realistic missing setting where the missing feature distribution is different during training and testing.

To establish these challenges, authors first theoretically show that when feature sparsity is high, a large amount of missingness is required to degrade the model quality and produce meaningful loss of information. The author introduces 1 synthetic and 3 real-world datasets to establish a better benchmark.

Then they establish multiple missingness mechanisms for evaluating and benchmarking the methods and datasets. Namely, they introduce label-dependent MCAR, feature-dependent MNAR, and class-dependent MNAR.

Finally, they introduce from their theory, the Missing Indicator Method (NIM) to model missing features in graph machine learning, GNNmin, where they concatenate the missing feature mask as additional input to GNN.

The authors show that this simplistic method performs robustly across all datasets and various missing conditions.

### Strengths
Authors have identified a relevant problem for industry-level deployments of GNNs. The Authors have established the significance of their motivation well enough with good theoretical insights. They have proposed a novel evaluation design with a novel introduction of distribution shift between train and test feature missingness. The proposed method is simple and effective.

### Weaknesses
A) The proposed method is non-inductive, i.e., it won't be able to handle new nodes in the graph. This, I think, is its major limitation.
B) Overall, I think the positioning of the paper can be improved further if the authors can position the proposed method as an effective baseline, as the paper's core novelty lies in identifying the challenges with the current evaluation setup.  Moreover, this method can supplement existing baselines as well to improve their performance. 
C) Comparison with simple baselines like 0 imputation or mean/median imputation will be advisable to further strengthen the claims.
D) Moreover, with datasets like Cora, CiteSeer, etc, what happens if we ignore all zero feature values and apply missingness on non-zero values only? Will that solve the core issue highlighted with these datasets.

### Questions
I request the authors to respond to the questions raised in the weakness section.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper primarily addresses the issue of handling missing node features in the practical application of Graph Neural Networks and challenges the evaluation methods used in existing research. It ultimately proposes a more challenging benchmark and a simple yet effective model.

### Strengths
1.Theoretical proof: The paper reveals the limitations of traditional sparse datasets when evaluating the robustness of missing features, making the evaluation process more scientifically grounded.

2.Introduction of new datasets: New datasets with dense, raw, and semantically meaningful features are introduced, providing more challenging and representative benchmarks for future research.

3.Proposing a more robust model: GNNmim is a simple yet effective new approach that consistently outperforms complex existing specialized models across various datasets and missing mechanisms.

### Weaknesses
1.The proposed synthetic/real-world datasets have relatively few features, reflecting the nature of the original measurements, but this also limits the direct generalizability of the findings in high-dimensional sparse feature scenarios.

2.GNNmim uses zero-padding as a placeholder for missing values, which is an arbitrary replacement rather than a semantically meaningful feature attribution. More theoretical analysis is needed to justify the appropriateness of this approach. Additionally, the effectiveness of GNNmim depends on the accurate construction of the missing indicator matrix M. If the missing information contains errors or uncertainties, model performance may be affected.

3.While the proposed MNAR generation mechanisms (FD-MNAR, CD-MNAR) are more realistic than MCAR, they are still synthetic and may not capture all the complex non-random missing patterns encountered in real-world scenarios.

### Questions
Please refer to Weaknesses part.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper revisits learning on graphs with missing node features and argues that common benchmarks and MCAR-only masking protocols obscure the true difficulty of the problem due to extreme feature sparsity and simplistic missingness. It contributes (i) an information-theoretic analysis showing sparsity dampens the impact of additional missingness, (ii) one synthetic and three real-world, denser-feature datasets, (iii) more realistic MCAR/MNAR protocols with train–test shifts, and (iv) GNNmim, a missing-indicator GNN that treats the mask as input rather than imputing features. Experiments show GNNmim is consistently robust across datasets and mechanisms.

### Strengths
1. The motivation to investigate why existing benchmarks and missingness assumptions fail to reflect real-world scenarios is well justified.
2. The proposed GNNmim framework is simple yet effective, showing consistent robustness across various missing-feature settings.

### Weaknesses
1. The information-theoretic results (e.g., ∆ bounds/data-processing) are important but scattered; adding “takeaway” remarks after each theorem and clarifying practical conditions (e.g., bandwidth/feature distributions) would help adoption.
2. More direct comparisons and discussion versus imputation-free conditional models and density-oriented approaches are needed.
3. The experimental studies are not strong enough since they are conducted on limited benchmark datasets and lack evaluations on larger or more realistic scenarios that can demonstrate the practical effectiveness and robustness of the proposed approach.
4. The significance of this work in advancing or motivating future research directions in graph neural networks is not clearly established.

### Questions
1. I am curious about, how would you recommend practitioners detect and handle mask-distribution shift at test time when using GNNmim, and can a simple calibration or reweighting mitigate the R2 drop you observe?
2. Can you quantify when MIM beats imputation (e.g., under which MNAR strengths or graph homophily levels), and provide a small decision guide for method choice?
3. Could you report ablation on the mask channel (mask only, features only, both) to isolate the gain from explicit missingness modeling?

### Soundness
2

### Presentation
2

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
This paper presents the case that existing GNN evaluations with missing features and sparse datasets are fundamentally flawed. The paper then introduces a benchmark with dense datasets and reasonable missing feature patterns e.g. MNAR. Additionally, they introduce a simple GNN model that concats mask for missing features, and empirically show that on this new benchmark, this simple/baseline model outperforms most existing GNNs.

### Strengths
S1: The main premise is quite interesting, and having myself evaluated GNN models on sparse datasets, I can clearly see how it could have been flawed. The experiments and theoretical analysis (sparsity and information loss connection) make argument intuitive. 



S2: Contributes datasets that elicits the flaws of earlier benchmarks by showing that even a simple GNN (with mask for missing features) is competitive.

S3. Paper is well written, easy to follow, and claims are supported by corresponding experiments.

### Weaknesses
W1. The graph scale of real-world dataset is quite small. Most real datasets currently have millions of nodes. How do results generalize to large scale graphs with dense features? Also will GNNmim remain competitive in such graphs?
Further the benchmark consists of 3 graphs – they may not be representative of real graphs. How non-trivial it is to expand the benchmark?

W2. The paper emphasizes on GNNmim as a new method, however, the main contribution seems to be the benchmark developed to assess GNNs appropriately. 

W3. The paper could benefit from discussion on what makes GNNmim effective.What it learns, or exploits that is present in dense graphs.

### Questions
Q1. Do you think the wordings for GNNmim could be changed from consistently outperformance to is competitive? 
Q2. Given your results, I’d be curious to see how even more simpler baseline like zero-imputation GNN perform? Will it be comparable to GNNmim?

### Soundness
4

### Presentation
4

### Contribution
4
