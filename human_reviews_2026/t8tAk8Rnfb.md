# Capturing Structure and Feature Signals in Graph Self-Supervised Learning

- Decision: Reject
- Scores: 4, 4, 4

## Abstract
This paper analyzes graph self-supervised learning (SSL) methods for node-level prediction tasks. First, we thoroughly evaluate several representative SSL methods on a diverse set of graph datasets. We observe that, contrary to prior literature, two popular generative methods MaskGAE and GraphMAE often fail to outperform well-tuned supervised baselines. At the same time, the contrastive methods BGRL and GRACE on average perform better than generative methods and supervised baselines. We hypothesize that this happens because BGRL and GRACE are able to capture the information about both graph structure and node features, while MaskGAE and GraphMAE concentrate on a single source of information. We support this hypothesis by conducting an analysis on carefully designed synthetic data. Motivated by our observations, we recommend designing SSL objectives that capture both feature and structure information. To verify the effectiveness of this approach, we propose a generative method that reconstructs both graph structure and node features. While being simple, this method outperforms all other considered approaches.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper primarily investigates methods for node-level prediction tasks in Graph SSL. Initially, the paper conducts a thorough evaluation of several representative Graph SSL methods, revealing a surprising finding: the performance of two popular generative methods, MaskGAE and GraphMAE, often fails to surpass that of carefully tuned supervised baselines. Meanwhile, the comparative methods BGRL and GRACE exhibit superior average performance compared to both generative methods and supervised baselines. The authors hypothesize that this is because BGRL and GRACE are capable of simultaneously capturing information about graph structure and node features. Based on these observations, the authors propose a method named grasp, which jointly processes graph structure and node features through a GNN encoder, and then employs three MLP decoders to reconstruct masked edges, reconstruct original features, and predict node degrees, respectively.

### Strengths
1. Rigorous benchmarking. The most notable advantage of the paper lies in its comprehensive and rigorous empirical evaluation. The authors conducted experiments on 10 diverse datasets, encompassing homogeneous graphs, heterogeneous graphs, and various types of node features.  
2. Fair tuning of the baseline: Unlike previous studies, this paper conducts a thorough hyperparameter search and architectural enhancement for supervised baselines. It is this rigor that reveals that the performance of SSL methods in previous studies may have been overestimated.  
3. Clear motivation and concise method: Based on the analysis, the suggestion of "capturing both structural and feature signals" is clear in motivation and instructive.

### Weaknesses
1. Limited innovation; this paper is more like an experimental report, which can provide some insight to researchers in graph SSL, but lacks theoretical and methodological innovation. The proposed method, GRASP, merely integrates several existing SSL tasks, which should be common in previous work and represents a relatively trivial innovation in methodology.

2. Scope limited to node-level tasks: The author explicitly states in the limitations section that this study is entirely focused on node-level prediction tasks. 
3. Tuning limitations of SSL methods: Although the authors emphasize the importance of tuning, they also acknowledge that due to the high computational cost, the supervised baseline was re-optimized 10 times, while the hyperparameters of the SSL method were only optimized once. This constitutes a potential weakness in the evaluation.

### Questions
no

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors systematically examine the effectiveness of Graph Self-Supervised Learning (GraphSSL) on node-level tasks, with a specific comparison between generative (e.g., MaskGAE, GraphMAE) and contrastive (e.g., GRACE, BGRL) paradigms. They hypothesize that superior performance is closely related to whether a model can simultaneously capture both "structural signals and feature signals." Based on this, the authors further propose a new method, GrASP, and validate its effectiveness on multiple datasets.

### Strengths
The paper conducts extensive experiments on both homophilic and heterophilic graph data, covering different types of graph structures and node features, which enhances the generalizability of the conclusions. Furthermore, the proposed GrASP method is conceptually and implementation-wise relatively simple, yet achieves performance improvements across several benchmark tasks, demonstrating certain practical value.

### Weaknesses
1. The analysis in the article largely relies on experimental results and lacks theoretical exploration into why simultaneously capturing structure and features is more effective.

2.  Previous research [1] has shown that simple baselines can achieve strong performance with sufficient hyperparameter tuning. How can the authors ensure that the re-run baselines used for comparison with GrASP were indeed sufficiently tuned? According to Table 6 provided in Appendix A, it seems the hyperparameter search space for some models might have missed their optimal settings. For instance, the optimal mask rate for GraphMAE on Cora is reportedly 0.75, but the authors only searched within [0.5, 0.9, 0.05]. Such settings raise concerns about the reliability of the baseline results and the true source of GrASP's improvements.


[1] Classical GNNs are Strong Baselines: Reassessing GNNs for Node Classification. NeurIPS 2024.

### Questions
1. For GraphMAE and MaskGAE, their official implementations often use GAT as a strong backbone. Why did the article not include comparisons using a GAT base?

2. In Table 2, GrASP seems to perform notably better on tabular (feature-focused) data. Is there a deeper analysis for this observation?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates graph self-supervised learning (GSSL) and argues that most existing methods capture either structure or feature information, but rarely both, first performing a systematic benchmark of representative GSSL methods  under a unified setup, including GRACE, BGRL, GraphMAE, and MaskGAE.

It shows contrastive methods tend to capture both structure and feature signals, while generative methods often focus on one type only.

Based on these insights, GrASP (Graph Attribute and Structure Prediction) is proposed, a simple generative framework that jointly reconstructs masked edges and node attributes, plus an auxiliary degree-prediction task.
GrASP achieves state-of-the-art performance across ten benchmark datasets while being simpler and more stable than prior generative GSSL approaches.

### Strengths
1.	Comprehensive empirical study of major GSSL families (contrastive vs generative) under consistent evaluation.
	2.	Insightful analysis revealing that the type of signal captured (structure vs feature) explains most performance gaps.
	3.	Proposed GrASP framework — a minimal joint prediction objective that unifies structure and feature reconstruction.

### Weaknesses
1. Lack of Large-Scale Benchmarks:  All core results are on moderate-scale node-classification datasets (Cora/Citeseer/Pubmed, LastFM, Facebook, Amazon-Photo/Computers, Tolokers, Questions, Ratings). While the set is diverse (homophily/heterophily; tabular vs. homogeneous features), it does not include truly large graphs typical in production or recent GSSL scaling studies. This makes it hard to assess scalability, stability, and efficiency of GrASP and the baselines under realistic constraints (GPU memory pressure, neighbor sampling variance, long training horizons) and to validate the paper’s claims about simplicity vs. performance at scale. The paper itself frames evaluation around node-level tasks with careful tuning, but within a transductive setup and the above dataset suite. Further, I suggest some large-scale dataset like ogbn-products (≈2.4M nodes / 61M edges) as standard transductive node classification and widely used as a “large but manageable” benchmark. Without a large-scale OGB evaluation, external validity remains uncertain.

2. Missing Comparison with Recent Graph SSL Models: The paper benchmarks classic GSSL methods such as GRACE, BGRL, GraphMAE, and MaskGAE, which are indeed canonical baselines. However, the field has recently shifted toward foundation-style graph representation learning, characterized by multi-modal pretraining, large-scale datasets, and Transformer-based architectures. These models, including GraphMVP, GraphMAE-2, GraphGPT and GraphFM / GROOV / UniGraph (2024–2025), aim to unify graph self-supervision under scalable, cross-domain objectives. Without comparison to such models, it remains unclear whether GrASP’s observed simplicity–performance advantage extends beyond the classical GNN-encoder regime. If these recent models cannot be evaluated, please illustrate the reason.

3. The paper’s central claim is that methods that jointly capture structural and feature signals outperform those that focus on a single source—is supported empirically but lacks a formal account of why and when this principle should hold. The current narrative connects performance gaps to what information a method “captures,” based on synthetic and real-data analyses, and then proposes GrASP as a simple joint-reconstruction objective. While convincing in practice, the argument remains predominantly empirical. A theoretical lens would clarify conditions under which joint structure feature pretraining yields provable benefits (e.g., linear probing guarantees, sample complexity improvements), and when it may not.

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
2
