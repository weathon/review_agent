# Graph Recurrent Attention Networks for Solving Satisfiability Problems

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 6, 2

## Abstract
In recent years, the use of deep learning for solving Boolean Satisfiability (SAT) problems has gained significant interest. This paper advances such neural-based methods by introducing a **G**raph **r**ecurrent **a**ttention **n**etwork for **SAT** (GranSAT). GranSAT employs two innovative steps to guide the network to search towards satisfaction of clauses: (1) evaluating the truth degree of each clause based on t-conorm fuzzy logic operators, and (2) updating assignments with attention mechanisms, closely aligning with distributed local search methods. Logical states are coupled with recurrently updated hidden states that are used to compute attention values, allowing the model to refine fuzzy assignments while retaining information from previous updates. Experimental results on crafted and random SAT benchmarks demonstrate that GranSAT outperforms existing neural SAT solvers in both performance and generalization. Furthermore, when combined with local search post-processors, GranSAT achieves state-of-the-art performance on random instances, showcasing its effectiveness in solving SAT problems.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper propose GranSAT, a GNN-based model for SAT solving. The soft assignments produced by the model can be refined by post-processing with a local search SAT solver. The experimental results show that GranSAT outperforms NeuroSAT and GAT-SAT on G4SATBench and achieves better results on SAT competition 2018 random track than the baseline solvers.

### Strengths
1. The unsupervised training objective makes model training more efficient. 
2. Combining with classical solvers yields better performance. For example, GranSAT+NuWLS improves PAR-2 and reduces timeouts over strong local-search baselines on the SAT’18 Random track.

### Weaknesses
1. Baselines appear dated. The comparisons focus on NeuroSAT and GAT-SAT, which are no longer state of the art among neural SAT solvers. More recent neural and hybrid approaches, SATformer[1], NeuroBack[2] and NLocalSAT[3] are not evaluated. Because these methods also use neural models both to act as neural solver and to guide CDCL or LS solving, comparisons against them should be important. 
[1] Satformer: Transformer-based unsat core learning
[2] NeuroBack: Improving CDCL SAT Solving using Graph Neural Networks
[3] NLocalSAT: Boosting Local Search with Solution Prediction

2. The experiment is not convincing enough. Experiments are limited to G4SATBench and the SAT’18 Random track, which are relatively easier. More evaluations should include the latest SAT Competition benchmarks.

### Questions
1. Please justify the above weakness. 
2. I wonder in Table 2, do all three solvers include post-processing? or are they just used to predict assignments without any LS solvers, like NeuroSAT default settings?

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
This paper presents GranSAT, a hybrid neural SAT solver based on Graph Recurrent Attention
Networks that operate on fuzzy logic variable assignments. Clause satisfaction is computed deterministically using t-conorm fuzzy operators (Gödel, Łukasiewicz, or Product), while variable assignments are updated via attention-based message passing combined with recurrent hidden states. This continuous relaxation of Boolean logic allows gradient-based learning while maintaining logical structure. When paired with local-search post-processing (NuWLS, MatSat), GranSAT achieves strong empirical performance on several benchmark families (G4SATBench and SAT Competition 2018).

### Strengths
• Clear and mathematically rigorous formulation.
• Elegant integration of fuzzy logic with attention and recurrence.
• Strong empirical performance on multiple benchmarks.
• Transparent experimental protocol and reproducibility.

### Weaknesses
• Most components are well-known; innovation lies mainly in using fixed fuzzy operators.
• The claimed relation between attention and local search is not supported by analysis.
• Outdated baselines; unclear definition of “state of the art.”
• Related Work section lacks critical comparison.

### Questions
• Can the authors justify or empirically demonstrate the claimed connection between attention
and distributed local search?
• Does fixing clause updates via t-conorms lead to measurable advantages over learned
updates?
• How sensitive is performance to the choice of t-conorm?

### Soundness
3

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
The paper introduces a method for solving SAT problems using a Graph Recurrent Attention Network (GranSAT).
The method:
1.	Evaluates each clause’s satisfaction state under fuzzy variable assignments using t-conorms.
2.	Updates the variable assignments via attention mechanisms.
GranSAT produces fuzzy assignments, which can be refined through local search–based post-processing methods.
The authors conduct experiments on various benchmarks and compare the performance of GranSAT with state-of-the-art neural and discrete SAT solvers.

### Strengths
The paper is well written and easy to read.

The authors claim that
•	GranSAT outperforms existing neural SAT solvers 
•	When combined with local search post-processors, GranSAT achieves state-of-the-art performance on random SAT instances.

### Weaknesses
Due to GPU memory limitations, only 195 out of 255 instances were evaluated.

While SAT problems are interesting, they are quite abstract. In my opinion, it would have been valuable to include some high-impact, real-world applications that can be formulated as SAT problems and solved using GranSAT.

### Questions
•	Line 194: Instead of using a t-conorm, could this mapping be replaced with a simple neural network whose weights are trained jointly with the Graph Neural Network?

•	Line 233: In many applications, Transformer- or Mamba-based architectures outperform GRU and LSTM models. Could these architectures be tried here, replacing the GRUs?

•	How sensitive is GranSAT to hyperparameters? I understand that in the experiments, the hidden/logical state dimension was set to 32 and the number of attention heads to 4. Were these values found to be optimal? How much do the results vary with different parameter choices?

•	It is interesting that in the experiments, GAT-SAT performs best on the hard-CA dataset but completely fails on some of the other tasks.

•	What were the wall-clock training times for the experiments?

•	Based on Appendix B, my understanding is that the authors used four GPUs. How were the training tasks distributed across these GPUs?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes, GranSAT, a graph neural network-based approach to solving the classic satisfiability problems. GranSAT uses the graph recurrent attention networks and alternates two updates: 1) clause's satisfaction state update using t-conorms; 2) variable assignment update using attention mechanisms. GranSAT can be used as a standalone solver and integrated with local search methods for further post-processing. Experimental evaluations are performed on the standard G4SATBench, which was designed to evaluate GNNs on sat solving, and GranSAT outperforms two baselines, i.e., NeuroSAT and GAT-SAT. When integrated with local search (NuWLS), GranSAT is close to or slightly outperforms state-of-the-art local search solvers such as Sparrow2Riss and YalSAT.

### Strengths
- the paper is generally well-written; particularly, visual illustrations are very helpful for conveying the essential idea
- the targeted problem, Boolean satisfiability, is of great importance, and relevant background about graph neural networks and attention mechanism are properly addressed

### Weaknesses
- the proposed method is fairly incremental, given that there are numerous attempts of applying graph neural networks for sat solving since the seminal work NeuroSAT (2019). 
- the chosen baselines (i.e., NeuroSAT and GAT-SAT) are quite limited, omitting many important baselines such as DG-DAGRNN, NLocalSAT, QuerySAT, Graph-QSAT, NSNet, to name a few. 
- the evaluation results of G4SATBench are inconsistent with the original evaluation of NeuroSAT
- problems of SAT Competition 2018 seem to be quite outdated
- the highlighted application for local search is not promising, i.e., the performance is close to the baseline NeuroSAT and there is a significant gap from the state-of-the-art  local search solver

### Questions
The accuracy numbers reported in this work are significantly different from the evaluation in the G4SATBench, particularly the results of NeuroSAT. Why is there such a big difference?

Why is SAT competition 2018 used, instead of the recent competitions (SAT Competition 2024)?

Minor writing issues: 
- Notations like $e_{ij+n}$ is confusing, which may refer to $e_k$ where $k=i*j+n$ or $e_{i, j+n}$. 
- Furthermore, $h_j$ and $h_{j+n}$ are also confusing. What are possible values of $N_i$? Shouldn't $h_{j+n}$ be used when considering $j \in N_i$.

### Soundness
2

### Presentation
3

### Contribution
1
