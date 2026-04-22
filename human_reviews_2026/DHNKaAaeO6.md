# GLIM: Towards Generalizable Learning Representation for MILP

- Avg Score: 3.50
- Decision: Reject
- Scores: 6, 2, 2, 4

## Abstract
Mixed-Integer Linear Programs (MILPs) underpin a wide range of combinatorial optimization applications, and there has been many works in the field of MILPs based on machine learning. However, existing learning-based approaches often struggle to generalize beyond narrow training distributions or specific tasks. In this paper, we introduce GLIM (Generalizable Learning Representation for MILPs), a general-purpose embedding model designed to unify learning across diverse MILP classes and downstream tasks. GLIM is trained on a large corpus of roughly 78,000 instances spanning 2,000 problem classes. Motivated by the observation that problem type and problem scale are orthogonal factors whose interaction drives empirical difficulty, GLIM learns a joint representation that disentangles type, scale, and solving complexity. Each instance is encoded as a bipartite graph and processed by a hybrid architecture that couples GNN modules with Perceiver-like blocks.  We evaluate GLIM on two representative MILP tasks to probe representation quality: (i) MILP Instance Retrieval and (ii) MILP Solver Hyperparameter Prediction. Across in-distribution and distribution-shifted settings, including real-world MIPLIB benchmarks, GLIM outperforms strong baselines in most cases and exhibits robust transfer to new classes and sizes. These results indicate that a single, disentangled embedding can serve as a reusable backbone for MILP tasks, enabling broader generalization than task- or class-specific learned components.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a general-purpose embedding model that learns a joint representation to disentangle type, scale, and solving complexity of mixed integer linear programs (MILPs). The proposed architecture consists of (1) a bipartite GNN encoder to embed the MILP instance (2) a perceiver like mix-attention block that performs cross attention between three learnable latent vectors (type, scale, and complexity token) with all node embeddings, and (3) final projection head. The authors use type loss, scale loss, and complexity loss to encode complementary and interpretable factors within the embedding. The authors finally demonstrate the effectiveness of the architecture on MILP instance retrieval and MILP solver hyper-parameter prediction.

### Strengths
1. This is a well-written paper.

2. The idea of disentangling type, scale and complexity of MILP instances make a lot of sense, and to my knowledge, is quite novel. 

3. Empirical results on MILP instance retrieval and MILP solver hyperparameter prediction seems convincing.

### Weaknesses
1. The baselines compared in the paper (Linq-Embed-Mistral and Qwen-3-Embed) are purely LLM-based text embedding model and seems to be quite weak. There has been many linear attention architectures proposed in previous literature (e.g. the simplest would be self-attention with sliding windows). To strengthen the results, the author should consider comparing with those linear attention variants. 

2. Empirical experiments are only performed on graph-level prediction tasks. To strengthen the results, the author should consider node-level or edge-level prediction task (e.g. learning to branch, learning to cut) to show that the disentanglement loss with the proposed architecture can help generally for many learning for MILP tasks.

### Questions
See the weaknesses above. I have the following question to expand upon Weakness 1: 

1. Appendix D.1 architecture ablation. The authors cap the maximum token for self-attention to 512, which seem restrictive. How would self-attention perform with a larger maximum token limit? Furthermore, how would the self-attention baseline perform for hyper-parameter prediction?

### Soundness
3

### Presentation
4

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
This paper introduces GLIM (Generalizable Learning Representation for MILPs), a general-purpose embedding model for Mixed-Integer Linear Programs. The authors argue that existing learning-based MILP solvers lack generalization across tasks and distributions, and propose to learn a disentangled embedding that captures three orthogonal factors — problem type, scale, and complexity. Each MILP instance is represented as a bipartite graph and encoded through a hybrid GNN–Perceiver architecture. The model is trained on a large synthetic dataset with multi-task supervision. Two downstream tasks — MILP instance retrieval and solver hyperparameter prediction — are used to evaluate the learned representation, showing some improvement over text-based embedding baselines.

### Strengths
The paper tackles an important and under-explored direction: learning general-purpose representations for MILP problems, rather than training specialized models for a single solver component or problem class. This line of work is timely and could serve as a foundation for future multi-task or cross-domain learning in combinatorial optimization.

### Weaknesses
1. The paper lacks meaningful comparison with existing learning-based solvers that also involve instance embeddings. For example, VAE-based models like G2MILP [1], multimodal approaches such as MILP-Evolve [2], and learning-based solvers like Predict & Search [3] all involve representation learning of MILPs, though they are not focused on representation learning directly. A recent work [4] also explores cross-domain MILP training. The authors should position their contribution more clearly with respect to these works and provide quantitative or qualitative comparisons when possible.
1. The proposed decomposition of MILP representation into type, scale, and solving complexity appears intuitively reasonable but lacks empirical justification. These three factors are not truly orthogonal, as solving complexity is influenced by both type and scale, and the paper provides no analysis demonstrating independence or disentanglement among them. Moreover, the learned representation is global (one vector per instance) and thus may not generalize to node-level or constraint-level tasks (e.g., branching, diving, cut selection, or solution prediction), which limits practical applicability.
1. The evaluation tasks are quite limited and largely self-defined. Instance retrieval and hyperparameter prediction are not standard benchmarks in the ML4CO literature, and thus do not convincingly demonstrate the utility of the learned embeddings. There are no strong baselines available for these tasks, making it difficult to assess the real advantage of GLIM over simpler alternatives. Applying the learned representations to more realistic solver-level tasks (such as heuristic selection, branching, or solution prediction) would significantly strengthen the work.
1. From Tables 3 and 4, while GLIM achieves improvement over text-based embeddings, the gains are generally small and sometimes inconsistent across datasets. Given that the baselines are general text embedding models rather than architectures designed for MILPs, the empirical advantage appears marginal and does not yet support the claim of a robust, generalizable representation.

[1] Geng et al., A deep instance generative framework for milp solvers under limited data availability. NeurIPS 2023.

[2] Li et al., Towards foundation models for mixed integer linear programming. ICLR 2025.

[3] Han, et al. A GNN-Guided Predict-and-Search Framework for Mixed-Integer Linear Programming. ICLR. 2023.

[4] https://openreview.net/pdf?id=wRQmQ6UXYF

### Questions
See Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work tackles the compelling problem of learning a generalizable representation for MILPs. While the direction is timely and important, the empirical evaluation in its current state does not adequately substantiate the core claims. The performance improvements over the baseline are marginal. Furthermore, the claim of generalizability is critically underexplored, resting on a single meaningful downstream task. To convincingly demonstrate a general-purpose representation, the work must show its effectiveness across a broader spectrum of fundamental MILP tasks, such as branching, cutting plane selection, and—most importantly—generalization to unseen problem types.

### Strengths
The graph representation is interesting.

### Weaknesses
I have the following concerns:

- While the authors claim their framework is highly generalizable, the choice of downstream tasks does not sufficiently support this. The MILP Instance Retrieval task is non-standard and its relevance to practical solver performance is unclear. To convincingly demonstrate generalizability, the framework should be evaluated on core, well-established tasks such as learning to branch, solution prediction, or learning to cut. 
- Furthermore, the evaluation of the hyper-parameter prediction task raises concerns. On MIPLIB, GLIM outperforms the default solver in only half the instances, suggesting limited practical advantage. More critically, the use of the "Moderated Solving Time" metric is misleading, as it artificially inflates performance by taking the minimum of two runtimes. For a fair assessment, the raw runtime using the predicted hyper-parameters must be reported.

### Questions
- In the anonymous link, I do not see the code for the task of instance retrieval.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces an embedding model guided by three factors: problem type, scale, and complexity for mixed-integer linear programs (MILPs). The primary objective is to leverage the learned embeddings to enhance the generalization performance of downstream tasks. The authors provide empirical evidence of the advantages of their approach over baseline methods, although the baselines and considered instances could be strengthened to better demonstrate the method’s capabilities.

### Strengths
1. The paper is well-written and presents a clear, logical flow of ideas.
2. The concept of using embedding learning to enhance generalization performance is both reasonable and intriguing.
3. The empirical results show improvements compared to the selected baselines.

### Weaknesses
Below are my major concerns:

1. **Simplistic instances**: While the MILP instances used in the embedding experiments are large (up to 100k variables and 100k constraints), the filtering process retains only those instances that can be solved optimally within 1000 seconds. These instances are relatively simple compared to real-world MILPs, where even solving for hours or days may not guarantee an optimal solution. Similarly, the datasets used for hyperparameter prediction are relatively easy and can be solved by the default solver in an average of less than 100s (some even in under 5s). I recommend incorporating more challenging datasets to demonstrate the robustness of the approach in real-world scenarios.
2. **Limited baselines**: The hyperparameter tuning task is evaluated only against other embedding approaches. I would expect comparisons that demonstrate the effectiveness of the proposed embeddings when applied to existing hyperparameter tuning methods, which would offer a stronger evaluation. 

Since I am not deeply familiar with the field of generalization and embedding learning, I express a relatively low level of confidence in my review.

### Questions
Please see the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3
