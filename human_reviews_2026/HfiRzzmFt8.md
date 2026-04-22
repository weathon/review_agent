# Amortized Bayesian Causal Discovery of Extended Factor Graphs

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 2, 4, 4

## Abstract
Learning a causal graph from interventional data is a challenging problem with broad applications. For example, a key goal in molecular biology is to identify gene regulatory networks from large-scale perturbation data. An ideal approach for this task should scale to thousands of nodes, incorporate interventions even when their targets are unknown, quantify uncertainty, and offer identifiability guarantees. Previous approaches using score-based optimization or approximate Bayesian inference do not meet all of these criteria. To address these limitations, we developed Amortized Bayesian Causal Discovery of Extended Factor Graphs (ABCDEFG). Our method guarantees exact acyclicity; scales to graphs with thousands of nodes; and incorporates interventions with known or unknown targets. Additionally, ABCDEFG estimates a posterior distribution whose maximum identifies the true causal graph up to an equivalence class. ABCDEFG achieves state-of-the-art accuracy on simulated datasets, estimating a well-calibrated posterior distribution while outperforming previous score-based and approximate Bayesian methods. Our approach also identifies known and novel genes targeted by growth factors in large-scale single-cell perturbation data.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents a new method, called ABCDEFG, for learning causal relationships from experimental (interventional) data with unkown targets. The key idea is to introduce the extended factor graph, which can handle uncertainty in which interventions act on which variables. The paper claims strong theoretical guarantees (identifiability) and demonstrates superior performance on both simulated datasets and a real-world biological dataset.

### Strengths
1. The proposed method is scalable, identifiable, and can handle uncertainty estimation. 

2. The experiments show consistent improvements in F1 score and SHD over many other methods on the synthetic dataset and lower MSE in the real-world dataset. 

3. The problem addressed is important, as learning causal relationships from complex biological data remains a significant and challenging task.

### Weaknesses
So far, the largest question from my side is the lack of motivation for the proposed framework. I’m not very familiar with the factor graph framework, so I’d appreciate more explanation on the motivation for using it in this paper. In causal discovery, most existing methods are based on DAGs or I-DAGs (for interventional data), while this paper follows the approach of Lopez by extending factor graphs to handle biological data. I’m curious to understand what unique advantages factor graphs offer in this setting, especially given that biological data often involve selection bias and latent variables. Is the idea that factors can, to some extent, represent latent variables or capture dependencies that DAGs cannot?

Another question is about the identifiability. I appreciate that the authors included a proof sketch, which nicely conveys the general intuition. However, it’s still difficult for me to grasp the role of each assumption. Specifically, how does each assumption contribute to establishing the identifiability result?

### Questions
Could the authors briefly discuss the interpretability of the learned causal factor graph shown in Figure 2(b)? It would be helpful to understand whether the discovered structure offers any biologically meaningful insights or aligns with known domain knowledge.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper argues for the lack of causal discovery methods that can handle thousands of variables and can incorporate intervention with unknown targets.  To address these limitations, they developed the Amortized Bayesian Causal Discovery of Extended Factor Graphs (ABCDEFG) to approximate the causal graph by the f-DAG under the differentiable Bayesian framework. Experimental results on simulated and real-world single-cell perturbation data verify its effectiveness.

### Strengths
1. This paper proposed ABCDEFG, which can do causal discovery for large-scale data with unknown intervention targets.
2. They approximate causal discovery with f-DAG under the differentiable Bayesian framework.

### Weaknesses
1. It takes me a long time to understand the contribution and writing logic. The motivation in the first two paragraphs is the need for high-dimensional causal discovery. However, all the contribution is related to modeling intervention. Intervention provides extra information, but how does it help causal discovery in factor-DAG?
2. The contribution is the Amortized Bayesian Causal Discovery in Section 2.5. However, I did not see the direct benefit of assuming the causal graph can be approximated by the f-DAG under the differentiable Bayesian framework.
3. The model of the ABCDEFG is in the appendix.

### Questions
1. What does the symbol $\Lambda$ mean in Line 241.
2. Line 252: in the Appendix is not clear.
3. Equation (1) is the same as the one below it (Line 269)
4. Only Equation (1) has the equation number, while others do not.
5. The setting for gene expression is not strong enough. We know that the gene regulatory network is a complex system with latent cofounders like non-coding RNAs and biological constraints. However, the ABVDEFG model has strong assumptions, including DAG, causal sufficiency, and no selection bias. Some work can handle latent confounders with unknown intervention targets, like Causal Discovery from Soft Interventions with Unknown Targets: Characterization and Learning.
6. Score-based causal discovery can handle thousands of genes as well, like FGES, A million variables and more: the Fast Greedy Equivalence Search algorithm for learning high-dimensional graphical causal models, with an application to functional magnetic resonance images.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper provides a new approach to causal discovery that's scalable, able to quantify uncertainty, capable of utilizing interventional data over unknown targets, and comes with identifiability guarantees.

### Strengths
- generally clearly written
- more general, efficient, and performant than existing deep generative models approaches to DAG learning
- theoretically nice approach to DAG learning, as far as deep generative models go: the paper cleanly presents the factor graph set-up from the literature, extends it to the general/unknown interventional setting, and derives the ELBO.

### Weaknesses
- not clear how practical the method actually is: the motivating gene regulatory network application doesn't really match the low-rank, causally sufficient DAG set-up (or if so, can the authors justify this?); also not clear how badly performance degrades when these assumptions are violated.
- unnormalized SHD in the tables are hard to interpret, especially without clear possible ranges and 'chance' baselines for comparison---consider [1]; F1 also hard to interpret without some 'chance' baseline when classes are unbalanced (i.e., edge probability not close to 0.5).
- no mention of or comparison against more traditional (i.e., MCMC-based) Bayesian approaches for causal discovery; there's a lot to choose from, going back to [2], but perhaps something more recent like a method in [3] is a reasonable starting point.
- unpolished writing, e.g., Figure 2 is illegible, math formatting throughout Section 2 is off ($KL$ instead of $\mathrm{KL}$, comma instead of colon in definition of $\Xi$, using $p(a|b)$ instead of $p(a\mid b)$, likewise for $\|$), after defining eq. (1) on L247, it appears to be identically repeated only 20 lines later, references [11] and [12] in the paper are duplicates

[1] Petersen, A. H. (2025) Are You Doing Better Than Random Guessing? A Call for Using Negative Controls When Evaluating Causal Discovery Algorithms. In The 41st Conference on Uncertainty in Artificial Intelligence.

[2] Heckerman, D., Meek, C., & Cooper, G. (1997). A Bayesian approach to causal discovery. Technical report msr-tr-97-05, Microsoft Research.

[3] Suter, P., Kuipers, J., Moffa, G., & Beerenwinkel, N. (2023). Bayesian Structure Learning and Sampling of Bayesian Networks with the R Package BiDAG. Journal of Statistical Software, 105(9), 1–31. https://doi.org/10.18637/jss.v105.i09

### Questions
I'm happy to increase my rating, depending on the answers to these questions:
1. Can the authors address the weakness mentioned above about real-world applicability and robustness to assumption violations?
2. Likewise, how about the metrics reported in the tables?
2. Likewise, how about discussing and comparing to some MCMC methods?
3. The definition on L147 seems to have removed the interventional Markov factorization described above on L137. Is this intentional?
4. The unspecified "equivalence class" mentioned on pages 1 and 2 is the interventional Markov equivalence class described later, right? Better to use the precise phrase upfront (and fine to leave the definition for later).
5. Is the $m$ in Table 1 the number of factor nodes? These aren't described until page 4, and then they're still not explicitly connected to the usage in Table 1.

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
3

### Summary
This work introduces a scalable causal discovery method that guarantees acyclicity, supports graphs with thousands of nodes, and incorporates interventions even when their targets are unknown. The proposed approach, ABCDEFG, estimates a full posterior distribution over causal structures, whose mode is provably identifiable up to the relevant equivalence class, allowing uncertainty quantification and principled recoverability guarantees. The key innovation lies in representing causal relationships using extended factor graphs, where feature nodes and intervention nodes are connected through auxiliary factor nodes. Building upon the f-DAG framework introduced by Lopez et al. [17], proposed method extends factor graph modeling to handle interventions with unknown targets by introducing factors that capture their influence on affected nodes. Through this extended formulation, the method achieves accurate and scalable distributional estimation of causal DAGs.

### Strengths
- The paper establishes clear identifiability guarantees by proving that the mode of the estimated posterior recovers the true causal graph up to the appropriate equivalence class.
- The amortized Bayesian framework enables efficient inference, allowing the method to scale to large graphs and large-scale datasets while maintaining exact acyclicity.

### Weaknesses
- The methodological framework is not novel that largely based on prior work, particularly Lopez et al. "Large-Scale Differentiable Causal Discovery of Factor Graphs" (NeurIPS 2022). 
- The experimental section would benefit from clearer characterization and comparative analysis of the datasets used. The current presentation lacks a clear description of dataset properties (e.g., underlying causal structure, mechanisms, and complexity), making it difficult to assess how these factors influence performance. Comparative analysis over these factors would help clarify the strengths and potential limitations of the proposed approach in different settings.
- Despite emphasizing scalability as a major advantage, experimental validation in large-scale scenarios is limited. The large-graph evaluation in Section 3.2 is difficult to interpret since results are primarily summarized in text (L429–473). Additionally, the visualization in Figure 2b lacks sufficient explanation, making its practical relevance unclear. It would be valuable to include a more systematic evaluation on real large-scale datasets, such as those used in “Large-Scale Targeted Cause Discovery via Learning from Simulated Data” (TMLR 2025), to more convincingly demonstrate scalability and practical impact.

### Questions
- In Appendix Table 14, what is the unit used in the time analysis?
- How does the performance change on simulated data when varying causal structure, mechanisms, and graph complexity?
- On which types of data does the proposed method encounter empirical limitations?
- How does the method compare to other applicable approaches on large-scale single-cell data, and to what extent are the reported performance metrics meaningful from the perspective of biological practitioners?
- What is the practical significance and interpretation of the visualization shown in Figure 2b?

### Soundness
3

### Presentation
2

### Contribution
2
