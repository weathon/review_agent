# Is it Bigger than a Breadbox: Efficient Cardinality Estimation for Real World Workloads

- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
DB engines produce efficient query plans by relying on cost models, that estimate cardinalities of subqueries constituting the input query. Standard cardinality estimators rely on heuristics, with magic numbers tuned to improve average performance on benchmarks. Empirically, their estimation error increases with query complexity. Recently, learning-based neural estimators have been proposed. However, they are not adopted in-practice, due to added operational complexity. Training or tuning generally requires extracting or simulating training data, from database contents or sample queries. Rather than learning one monolithic network, we are motivated to capture the common practical scenario: query patterns repeatedly re-appear soon after their first introduction. We propose LITECARD, that online-learns many (simple) regressors, one per subquery pattern. The regressor corresponding to a pattern can be randomly-accessed using hash of directed acyclic graph encoding the subquery. LITECARD instantiates from cold-start, imposing negligible overhead, while competing with SoTA neural approaches on error metrics. More importantly, implementing LITECARD into PostgreSQL reduces query execution time by 30% of popular JOB-lite workload on IMDb, while incurring negligible overhead (37 seconds) for online learning. By achieving notable accuracy and runtime improvements over standard methods, and drastically reduces operational costs compared to neural estimators, LITECARD pushes Pareto frontiers balancing execution speedups and overhead time.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces a cardinality estimation for SQL queries based on kernel regression and handcrafted features. 
The method has quite small overhead and competes with SoTA learning-based approaches on error metrics. 
Further, amending PostgreSQL with the method achieves notable accuracy and runtime improvements over traditional methods and drastically reduces operational costs compared to other learned cardinality estimators, thereby offering the most practical and efficient solution on the Pareto frontier.

### Strengths
1. The language of the paper is fine, main ideas are easy to follow.
2. Empirical results show that the proposed method has quite fast inference (not sure about training?), when compared to alternatives.

### Weaknesses
1. The performance is lower than some baselines (Table 3).
2. The equation (6) is not clear. In ML, one should use train/test split of a dataset to avoid overfitting.
3. Figure 5 caption: missing reference 
4. It seems that your method is called "LITECARD". But it was never introduced formally.
5. The main idea of the proposed method is a specific vectorization of execution DAGs. The features (Appendix D) are very domain specific and lack generality. I propose to resubmit the manuscript to VLDB or KDD.
6. A comparison with standard GCN, GAT is not provided.
7. I can't find proofs that the proposed method  (i) can run from cold-start, requiring no upfront training; (ii) can adapt to changes in workloads or data shifts

### Questions
1. Can you compare your method with:
* P. Negi, R. Marcus, H. Mao, N. Tatbul, T. Kraska, and M. Alizadeh, ‘‘Cost-guided cardinality estimation: Focus where it matters,’’ in Proc. IEEE 36th Int. Conf. Data Eng. Workshops (ICDEW), Apr. 2020, pp. 154–157. 
* J. Sun and G. Li, ‘‘An end-to-end learning-based cost estimator,’’ 2019, arXiv:1906.02560.
* J. Sun, J. Zhang, Z. Sun, G. Li, and N. Tang, ‘‘Learned cardinality estimation: A design space exploration and a comparative evaluation,’’ Proc. VLDB Endowment, vol. 15, no. 1, pp. 85–97, 2021

2. How your paper is related to:
Woltmann, L., Hartmann, C., Thiele, M., Habich, D., & Lehner, W. (2019, July). Cardinality estimation with local deep learning models. In Proceedings of the second international workshop on exploiting artificial intelligence techniques for data management (pp. 1-8) ?

3. It is not clear how the model is trained. Is it trained online or query dataset is divided into train/test splits?

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
4

### Summary
This paper presents LITECARD, an online, pattern-based learned cardinality estimator for relational databases. The key idea is to exploit the repetitive nature of sub-query patterns in real-world workloads by grouping isomorphic subquery graphs and training lightweight regressors (e.g., locally weighted linear regression or decision forests) per pattern.

### Strengths
S1. Practical motivation: The paper addresses a genuine gap in learned cardinality estimation research, bridging the accuracy of learned models with the low overhead needed for production DBMS integration.

S2. Novel technical framing: The use of graph-local learning combined with hash-based partitioning of isomorphic subqueries is original and conceptually neat. The hierarchical (H1–H3) partitioning and incremental online updates are well-justified.

S3. System integration: Implementing the method in PostgreSQL and demonstrating measurable runtime improvement (e.g., 7.5 min, around 30% reduction on IMDb) is good.

### Weaknesses
W1. Unclear pattern detection pipeline.
It is unclear how the query patterns are discovered or clustered online. The paper seems to assume that the graph of subqueries is already extracted from query plans, but does not explain whether clustering or pattern identification contributes to the overhead. Clarifying the computational cost and scalability of this step is essential.

W2. Dependence on historical query plans.
The method appears to rely on a corpus of historical queries and their plans to build pattern-specific models. This raises several questions:

- Are the query plans generated by PostgreSQL itself?

- Is it fair to compare against methods that do not assume access to such historical execution traces?

- How is the cold-start phase (when no history exists) handled?

W3. Generalization and workload shift.
Since models are trained per pattern derived from past workloads, the approach seems inherently query-driven. How well does it transfer to new workloads or schema variations? If a new workload introduces unseen patterns, does the system re-hash and retrain? Are such adaptation costs included in the reported “37 s overhead”? Clarifying this is crucial for evaluating the claim of “negligible overhead.”

W4. Scalability with large workloads.
The method’s efficiency is evaluated on IMDb (about 5 k queries). How does performance scale when the number of query patterns grows (e.g., 50 k queries or more)? The memory cost of maintaining many regressors and hash tables may become significant, while data-driven baselines remain unaffected by query count.

W5. Experimental scope and baseline selection.
The evaluation focuses primarily on IMDb. It would strengthen the paper to include other complex benchmarks (e.g., STATS, StackOverflow, or TPC-DS) that feature multi-table joins and diverse predicates.
Also, the baselines are limited to DeepDB and FactorJoin as “data-driven” methods, but several  approaches (e.g., NeuroCard 2021, FLAT 2021, CardBench baselines) could provide a fairer SoTA comparison. The authors should justify why these were omitted.

Minor issues.

Figure 5 caption contains a typo: “Eq. ??” should reference the correct equation number.

The paper sometimes intermixes the terms LITECARD and ours; consistent terminology would improve clarity.

### Questions
Q1. How are query patterns identified? Is there an explicit clustering step, and is its cost part of the 37 s overhead?

Q2. How are query plans obtained for historical queries? If the system requires pre-existing plans, does that imply an extra source of information unavailable to other baselines?

Q3. How does LITECARD handle new query workloads or schema changes where the subquery graphs differ from historical ones?

Q4. How does performance and overhead scale as the number of unique query patterns increases (e.g., beyond 50 k)?

Q5. Why is IMDb the only dataset tested? Would the method generalize to datasets with more complex join paths or string predicates (e.g., STATS, TPC-DS)?

Q6. Why were only DeepDB and FactorJoin chosen as data-driven baselines, and are they considered current SoTA for this comparison?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This manuscript delineates an innovative approach to cardinality estimation called LITECARD, which addresses the trade-off between the inaccurate, fast heuristics of traditional database systems (like PostgreSQL's per-column histograms) and the accurate, slow deep learning models that are often too complex to deploy in practice. The LITECARD works by decomposing the query space and applying graph-local learning, which exhibits an amalgamation of pattern recognition, custom kernel, local models, and hierarchical fallback, a combination that endows the model with superior performance in comparison to baseline methods.

### Strengths
S1. The paper's significance is underscored by its contribution of a novel cardinality estimator that 1) can run from a cold-start (no upfront training), 2) can adapt to changes in workloads or data shifts, and 3) has negligible update and inference time.

S2. The estimator's novelty is encapsulated in its graph-local learning - an amalgamation of pattern recognition, custom kernel, local models, and hierarchical fallback.

S3. The evaluation is comprehensive, with comparisons to baseline methods providing a compelling demonstration of the superior performance of the proposed method.

### Weaknesses
W1. While the median error (P50) is highly competitive (1.70), the tail errors (worst-case errors) are significantly worse than the most accurate learned estimators, DeepDB and FactorJoin.

W2. The core efficiency and low overhead rely on the assumption that highly repetitive subquery patterns exist in the workload. For a workload with extremely low query template reuse or high churn in unique query patterns, the model would frequently fall back to the base PostgreSQL estimator or have to train local models on very sparse data, undermining its advantage.

W3. The local learning models are simple and established. While simplicity contributes to the low overhead, relying on models that only consider local feature proximity might limit their ability to capture complex non-linear feature interactions within a query pattern, potentially contributing to the high tail errors. The novelty rests more on the system-level integration and workload decomposition strategy rather than the machine learning models themselves.

### Questions
See "weaknesses" above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a lightweight online learning approach for query cardinality estimation in relational databases. Instead of using a single large model, it maintains multiple small regressors, each specialized for a subquery pattern represented as a graph and retrieved via hashing. The method continuously updates online with minimal overhead and integrates into PostgreSQL, achieving accuracy comparable to state-of-the-art learned estimators while reducing training cost and runtime overhead.

### Strengths
1. The problem studied in this paper is important.
2. The presentation is good.

### Weaknesses
1. Unclear novelty. The paper does not clearly explain how it differs from existing learned optimizers and cardinality estimation methods.
2. Insufficient scalability analysis. The paper lacks detailed discussion and evaluation of how the method scales with larger datasets and workloads.

### Questions
I have two main concerns about the paper: novelty and scalability.

1. Novelty
The paper discusses several existing learned optimizers and cardinality estimation approaches. However, the fundamental novelty of this work remains unclear. Please clarify what distinguishes the proposed method from prior works — for example, what new information or mechanisms are introduced in the learning process, and how these lead to fundamentally different behavior or advantages compared with existing models.

2. Scalability and Training Overhead
Section 4.1 reports a total overhead of about 37 seconds for 5k queries on the IMDb dataset, but it is unclear how this overhead scales with larger workloads. Specifically:
	•	How does the system perform as the number of stored subquery patterns or regressors increases (e.g., 100k queries or multi-terabyte datasets)?
	•	Does the memory footprint or lookup latency grow linearly with the number of hashed entries?
	•	Since each pattern maintains a separate model, could model management or hash collisions become a bottleneck for very large workloads?

### Soundness
3

### Presentation
3

### Contribution
3
