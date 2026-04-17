# Relatron: Automating Relational Machine Learning over Relational Databases

- Decision: Accept (Poster)
- Scores: 4, 8, 6, 4

## Abstract
Predictive modeling over relational databases (RDBs) powers applications in various domains,
yet remains challenging due to the need to capture both cross-table dependencies and complex feature interactions. Recent Relational Deep Learning (RDL) methods automate feature engineering via message passing, while classical approaches like Deep Feature Synthesis (DFS) rely on predefined non-parametric aggregators.
Despite promising performance gains, the comparative advantages of RDL over DFS and the design principles for selecting effective architectures remain poorly understood.
We present a comprehensive study that unifies RDL and DFS in a shared design space and conducts large-scale architecture-centric searches across diverse RDB tasks. Our analysis yields three key findings: (1) RDL does not consistently outperform DFS, with performance being highly task-dependent; (2) no single architecture dominates across tasks, underscoring the need for task-aware model selection; and (3) validation accuracy is an unreliable guide for architecture choice.
This search yields a curated model performance bank that links model architecture configurations to their performance; leveraging this bank, we analyze the drivers of the RDL–DFS performance gap and introduce two task signals—RDB task homophily and an affinity embedding that captures size, path, feature, and temporal structure—whose correlation with the gap enables principled routing. Guided by these signals, we propose Relatron, a task embedding-based meta-selector that first chooses between RDL and DFS and then prunes the within-family search to deliver strong performance. Lightweight loss-landscape metrics further guard against brittle checkpoints by preferring flatter optima. In experiments, Relatron resolves the “more tuning, worse performance” effect and, in joint hyperparameter–architecture optimization, achieves up to 18.5\% improvement over strong baselines with $10\times$ lower computational cost than Fisher information–based alternatives. Our code is available at https://github.com/amazon-science/Automating-Relational-Machine-Learning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents an empirical study comparing Relational Deep Learning (RDL) and Deep Feature Synthesis (DFS) for predictive modeling on relational databases. The authors construct a large-scale design space for both model families and build a "model performance bank" by running an architecture-centric search across numerous RDB tasks. Relatron uses diffferent signals in a meta-predictor to first choose between RDL and DFS (macro-selection) and then prune the search within that family.

### Strengths
- The paper targets an important problem, relational deep learning (RDL), in the machine learning.
- The experimental analysis across models and datasets is interesting.

### Weaknesses
- The recommendation task is not involved. Only classification and regression task types from Relbench.
- The results of a foundation model, KumoRFM, outperforms Relatron that selects between models. This questions the practical usage of Relatron. Since there is already a stronger foundation model, is it still necessary to design and train a model selector?

### Questions
- The paper provides a theoretical argument in Appendix C.2 for RDL's strength in low-homophily settings. It would be better to explain the other, more surprising half: why DFS is so strong in high-homophily regimes. 
- Could Relatron also include KumoRFM, the current state-of-the-art foundation model, into its selection pool?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper lays out a design space of modeling approaches for machine learning (ML) on relational database (RDB) tasks, and does a comprehensive benchmarking study on it. The findings suggest that different tasks require different design choices (including model architecture). To automate this choice the paper proposes a few metrics which have high correlation with the performance of various approaches on different tasks and can be used to predict the best method. This AutoML system (Relatron) achieves performance improvements over hyperparameter-tuning and auto-transfer baselines, while being more efficient computationally.

### Strengths
* The paper is thorough in its parameterization of the design space.
* The findings are interesting. I like the experiment and finding that validation metrics are unreliable as this is very important for temporal splits as in leading relational benchmarks.
* The metrics proposed for "task embeddings" are simple, clear, clever, and have high predictive power.
* The baseline comparisons are comprehensive.
* The paper is well-written overall.

### Weaknesses
I think it is important to include intuitions and analysis (theoretical as well as empirical) for why RDL is better at low-homophily tasks and DFS is better at high-homophily tasks in the main paper. Currently it is in Appendix C.2, but it is too long (4 pages!) and it is not clear what the intuitive takeaways are. It would be nice to have a discussion of the main intuitions and takeaways in the main paper. It would be ideal if the theory can be substantiated with some experiments.

### Questions
1. Why is RDL better at low-homophily tasks and DFS better at high-homophily tasks?
2. How can RDL be improved based on insights from 1?
3. Are there some kind of error bars for Figure 2?
4. What is "labeling tricks for RDL"?
5. L412: What is GraphGym similarity? How is it "ground truth"?

Minor:
* L040: repeated citation
* L059: remove space before footnote
* L362-363: You might be interested in [1] as they have similar findings/methodology and propose similar terminology "post-hoc selection".

[1] Post-Hoc Reversal: Are We Selecting Models Prematurely? NeurIPS 2024.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work conduct a detailed analysis between Relational Deep Learning (RDL) and DFS method with heuristic feature aggregator. It reveals that 1) DFS can outperforms  RDL on some tasks. 2)Different architecture are needed for different RDB tasks. 3) valid performance is not reliable. So this work propose an automl framework

### Strengths
1. Extensive experiments over various baselines and datasets. Figure 2, Table 2-4, shows various experiments.
2. Insightful observation for model selection. To me, “Correlation between homophily and RDL-DFS performance gap" is the most interesting finding. Though homophily is a common tool in graph learning, it is first used in RDB datasets to our best knowledge.

### Weaknesses
1. Griffin cited in this work should also be included as one important baseline on RDB tasks.
2. The comparison between these auto ml framework and vanilla baseline is missing. If this framework leads to large computation overhead, then vanilla baseline may be prefered in real-world application.

### Questions
1. While automl framework is useful, I still think a unified RDB foundation model is the future. Can observations in this work helps development of RDB foundation model?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper focuses on automatic architectural selection for predictive task on relational databases. The model proposes Relatron, a task embedding-based selector using novel task signals. The performance bank is a reusable resource, and findings like task-dependent RDL vs. DFS trade-offs challenge prevailing assumptions in the field. Experiments demonstrate meaningful gains (up to 18.5% improvement with 10× compute savings), making it relevant for real-world deployment.

### Strengths
The architecture search (180 configs/task for entity-level, 20 for DFS) is comprehensive. The findings are interesting as it show DFS wins on more tasks , attributing gaps to homophily.

### Weaknesses
1. The scope of the paper is limited, as it focuses on from-scratch models and defers foundation models(e.g., Griffin, KumoRFM) despite comparisons (Fig. 2). It would be interesting to see how Relatron can handle pretrained relational foundation models. 
2. The experiment dataset is limited, only covering most of RelBench and two additional tasks. It would be interesting to see the performance across multiple datasets. 
3. The correlation between homophily and RDL-DFS performance gains are strong, but non-parametric. It would be interesting to see if the authors can.generate synthetic data(tasks with varying homophily) to strengthen the claim.

### Questions
1. How sensitive is Relatron to bank size? With fewer tasks, does transfer degrade?
2. Could homophily be extended to link-level tasks (e.g., recs), and does it correlate with higher-order passing (RelGNN)?

### Soundness
3

### Presentation
3

### Contribution
2
