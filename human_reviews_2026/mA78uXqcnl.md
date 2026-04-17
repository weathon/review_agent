# Causal Structure Learning in Hawkes Processes with Complex Latent Confounder Networks

- Decision: Accept (Oral)
- Scores: 8, 8, 6, 6

## Abstract
Multivariate Hawkes process provides a powerful framework for modeling temporal dependencies and event-driven interactions in complex systems. While existing methods primarily focus on uncovering causal structures among observed subprocesses, real-world systems are often only partially observed, with latent subprocesses posing significant challenges. In this paper, we show that continuous-time event sequences can be represented by a discrete-time causal model as the time interval shrinks, and we leverage this insight to establish necessary and sufficient conditions for identifying latent subprocesses and the causal influences. Accordingly, we propose a two-phase iterative algorithm that alternates between inferring causal relationships among discovered subprocesses and uncovering new latent subprocesses, guided by path-based conditions that guarantee identifiability. Experiments on both synthetic and real-world datasets show that our method effectively recovers causal structures despite the presence of latent subprocesses.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes a causal discovery method for identifying latent subprocesses and their causal structures in partially observed multivariate Hawkes processes. The authors introduce a causal model for partially observed multivariate Hawkes processes to represent continuous-time event sequences. Based on this model, they leverage rank constraints on the covariance matrix to identify causal influences without prior knowledge of the existence or number of latent subprocesses.

### Strengths
1. The authors discretize the Hawkes process, transforming the multivariate Hawkes process causal model into a linear autoregressive model, and theoretically prove this conclusion.
2. It proposed a method for identifying latent subprocesses and causal structures solely by leveraging rank constraints on second-order statistics.

### Weaknesses
1. In Proposition 4.5, should it be that the rank constraint is both necessary and sufficient for the corresponding local independence in the graph, under the structure defined in Definition 4.4 and the data generated accordingly?
2. The results mentioned in the original SHP paper differ significantly from those presented in this paper. Additionally, could results for different time intervals be provided, similar to the approach in the SHP paper?
3. When the paper transforms the model into a linear autoregressive model, does it require the noise to be constrained as Gaussian?
4. The proposed two-phase iterative algorithm suffers from severe scalability limitations, which restrict its practical applicability.


Typo:

There is an extra } at line 300.

### Questions
What parameters were used for the method proposed in the real-world data experiments?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses the problem of causal structure discovery in multivariate Hawkes processes under partial observability. Authors consider Hawkes processes with elaborate latent structure, and derive the conditions under which the latent variables and the relationship between them is identifiable. Similar results exist in the context of linear autoregressive processes and as acknowledged by the authors, they inspire the latent structure discovery results in this manuscript (Theorem 4.7 and 4.8). By transforming the Hawkes process inference problem into a discrete-time linear autoregressive formulation (Theorem 4.1), the authors establish the results for Hawkes processes. 

The paper proposes a two-phase iterative algorithm that alternates between (i) discovering causal relations among existing subprocesses and (ii) inferring new latent subprocesses based on rank constraints of cross-covariance matrices. Necessary and sufficient conditions are derived for identifiability, including the introduction of path-based conditions (Definition 3.4) ensuring one-to-one correspondence between latent confounder structures and observable rank deficiencies. Empirical results on synthetic and real-world data show that the proposed method successfully recovers causal structures even when latent confounders exist.

### Strengths
Compared to the previous NeurIPS 2025 submission that I had reviewed, the manuscript has substantially improved in clarity of the claims and structure of the paper. Most importantly, the iterative algorithm is now explicitly defined, assumptions and identifiability conditions are more carefully motivated, and the connection to prior work—including an additional LPCMCI baseline in experiments, rank-based latent structure discovery methods, and INAR processes—has been expanded.

### Weaknesses
1. Novelty of Theorem 4.1 can still be debated. While the authors’ expanded discussion distinguishes their formulation from prior binning-based estimation approaches, the contribution could still benefit from more explicit formal comparison (e.g., showing in what sense their linear representation differs from INAR-based or EM-based formulations beyond the absence of likelihood modeling).

2. Motivation benefits from more discussion. In much of the classical literature on the broader causal discovery problem, the structure within them is not discussed, as the latent confounder are often treated as root nodes affecting the observables. The work is indeed interesting in a theoretical sense, yet, I'd like to question the motivation for latent structure discover: Since these variables are not observed, it is hard to imagine interventions on them, so why is it of interest to practitioners to identify the structure of the latent variables?

3. Assumptions and their implications. The identifiability results depend on assumptions about the Hawkes process and the structure of latent confounder. In which practical scenarios are these assumptions justifiable? On the other hand, in misspecified cases, how poorly is the latent structure discovered? A short illustrative example, even synthetic, where the assumption are valid and one where they fail would greatly benefit the reader.

4. Missing acknowledgement/comparison with recent work on causal discovery in Hawkes processes via compression schemes, e.g., [1,2]

[1] Hlaváčková-Schindler, K., Melnykova, A., & Tubikanec, I. (2024). “Granger causal inference in multivariate Hawkes processes by minimum message length.” JMLR 25(133): 1–26

[2] Jalaldoust, A., Hlaváčková-Schindler, K., & Plant, C. (2022). “Causal Discovery in Hawkes Processes by Minimum Description Length.” AAAI 36(6): 6978–87.

### Questions
Please address the weaknesses mentioned above.

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
The paper studies structure learning (causal discovery) for partially observed multivariate Hawkes processes (PO-MHP) and provides the first principled framework that identifies latent sub-processes and recovers causal structure in continuous-time event sequences without prior knowledge.

The authors make a key theoretical contribution (Theorem 4.1) by showing that a continuous-time multivariate Hawkes process can be represented by a discrete-time linear causal model when the event-count data is appropriately binned. They further prove that the low-rank constraints on the cross-covariance matrices induced by the linear representation can be used to (1) detect the presence of latent confounder subprocesses, and (2) identify parent–cause sets and causal edges under explicit path-based conditions (Definitions 4.4, Propositions 4.3/4.5, Theorems 4.7/4.8).

Based on this theoretical foundation, the paper proposes a novel two-phase iterative algorithm where Phase I identifies causal relationships among the currently known (observed and inferred) subprocesses and Phase I discovers new latent confounders via rank tests. The authors also prove that this method guarantees the identifiability of the causal graph. Experiments on both synthetic and real-world datasets show that the proposed method effectively recovers the ground-truth causal graphs, outperforming existing baselines, especially in settings with complex latent structures.

### Strengths
1. One of the paper's main strengths is its strong theoretical foundation. Theorem 4.1 is a powerful result, which innovatively establishes a connection between continuous-time Hawkes processes and a discrete-time linear autoregressive representation. Moreover, the *Definition 4.4 + Proposition 4.5 + Theorems 4.7/4.8* that links symmetric path structures to observable rank deficiencies is also original and enables finding latent confounders without prior information of the existence or number of latent subprocesses.

2. The paper addresses a critical challenge, where many previous causal discovery algorithms assume that all relevant variables are observed. This paper instead studies a more realistic and difficult scenario under partial observability, where they propose a novel framework to uncover causal structure with unknown latent subprocesses. It is scientifically important.

3. The proposed two-phase iterative algorithm is a direct and elegant consequence of the theoretical results. The experiments, while concise, are well-designed to validate the paper's core claims. Specifically, the synthetic experiments include multiple graph families, sample sizes, and sensitivity checks.

### Weaknesses
1. Strong structural assumptions. Definition 4.4 formalizes the Symmetric Acyclic Path Situation (the observed effects being connected to the latent via paths of equal length and acyclic intermediate latents), which is a somewhat special topology. However, in complex systems intermediate latents can have varying path lengths or additional cross-links, which would break the condition and make that latent unidentifiable by the method.

2. While the paper is theoretically rigorous, it is also extremely dense. The written could be polished with a motivating real world example in Figure 1, then expand the theoretical proof based on this step-by-step real world example. Moreover, some intuitions could be explained before show the theorems and proofs. For example, the transition from 4.2.1 to 4.2.2 could give some intuitions.

3. Only one real world dataset is limited. This small dataset (evaluation on a five-alarm subgraph) can not support the model effectiveness on large and noise real-world systems. Meanwhile, the reported results on Tables 1-4 do not show variance.

### Questions
1. How sensitive is the performance of your algorithm to the choice of the discretization interval Δ? Is there a principled way to select an optimal Δ, or is it purely an empirical choice? How does data sparsity affect this choice?

2. Some additional experiments could be added. For example, what if some latent confounders are removed (which violates the condition in Definition 4.4)?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies the problem of causal discovery in multivariate Hawkes processes (MHPs) with latent confounders. The idea is to represent a MHP with a specific form of excitation functions as a linear autoregressive model over discretised variables. Afterwards, the authors introduce a set of conditions under which the causal structure is identifiable using rank tests on covariance matrices of observed discretised variables.

### Strengths
The paper addresses an important and relevant problem, i.e., causal discovery in multivariate Hawkes processes (MHPs) with hidden confounders. 
The theoretical contributions provide valuable insights into causal discovery without assuming causal sufficiency in MHPs and represent an important step toward advancing research in this area.

### Weaknesses
The main result builds on representing an MHP as a linear autoregressive model through discretization. However, according to Theorem 4.1, this result holds only when the discretization parameter (\Delta) tends to zero. In practice, for small but finite (\Delta), this leads to model mismatch, which can also be observed in the sensitivity analysis with respect to (\Delta) in Table 1.

Moreover, all identifiability results are derived under the assumption that the linear representation holds, i.e., as \Delta -> 0. However, no guidance is provided on how to choose (\Delta) in practice to ensure consistent results.

The identifiability results further rely on an additional assumption that the excitation functions take the form (a_{i,j}w(s)), for example, the exponential decay function a_{i,j}\exp(-\beta s). While this is a common assumption in the MHP literature, it is often extended to cases where the decay rate \beta is also an unknown, node-specific parameter, i.e., (a_{i,j}\exp(-\beta_i s)). Although this may appear to be a minor modification, it is non-trivial to see how the results of this work extend to such more general excitation functions.

The proposed algorithm has exponential complexity, which limits its scalability. As discussed above, its performance is also sensitive to the choice of \Delta. Furthermore, the method relies on rank tests, which typically require large amounts of observational data. This raises a question regarding Figure 4: assuming the experimental setting is favorable to all baseline methods as well as the proposed approach (i.e., with no latent confounders), how would these methods perform with substantially fewer observations, e.g., significantly less than 30,000?

### Questions
Please see above comments.

### Soundness
3

### Presentation
4

### Contribution
3
