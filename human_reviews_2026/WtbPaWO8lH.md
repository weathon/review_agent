# Causal Discovery in the Wild: A Voting-Theoretic Ensemble Approach

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Causal discovery is a critical yet persistently challenging task across scientific domains. Despite years of significant algorithmic advances, existing methods still struggle with inconsistent outcomes due to reliance on untestable assumptions, sensitivity to data perturbations, and optimization constraints. To this end, ensemble-based causal discovery has been actively pursued, aiming to aggregate multiple structural predictions for increased stability and uncertainty estimation. However, current aggregation methods are largely heuristic, lacking theoretical guarantees and guidance on how ensemble design choices affect performance. This work is proposed to address there fundamental limitations. We introduce a principled voting-based framework for structural ensembling, establishing conditions under which the aggregated structure recovers the true causal graph. Our analysis yields a theoretically justified weighted voting mechanism that informs optimal choices regarding the number, competency, and diversity of causal discovery experts in the ensemble. Extensive experiments on synthetic and real-world datasets verify the robustness and effectiveness of our approach, offering a rigorous alternative to existing heuristic ensemble methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a principled ensemble framework for causal discovery by treating multiple causal structure learning algorithms (or multiple runs of one algorithm) as noisy experts. It introduces a voting-theoretic model based on Bayes-optimal weighted voting to aggregate DAG structures. Theoretical analysis  of the proposed method (including the version under estimated parameters) are provided.

### Strengths
- This paper provides first voting-theoretic error bounds for structure ensembling in causal discovery, particularly useful as many algorithm candidates are available.
- The error bounds offer insights into how ensemble size, diversity, and competency for algorithm design affect the performance of the ensembling.
- The experiments combine synthetic and real datasets, showing consistent performance and wide algorithm coverage.

### Weaknesses
- the motivation of this work (or ensembling) is not about the structure learning is bad (low $\mathbb{P}(\widetilde{G} | G,\mathcal{D})$) and we want to aggregate them. But the focus is on the fact that causal dscovery methods rely on very different identifiability assumptions, thus different data generating process $\mathbb{P}(\mathcal{D} | G)$. So the "competency" is often zero once the identifiability assumptions are violated. How do we evaluate them under the same probability measure? and how does this change the problem setup and estimation of $\theta$'s?
- It is helpful to include a runing examples when introducing "feature" level to illustrate how thinking in "feature" level instead of graph level helps reduce the complexity of the problem.
- See questions.

### Questions
- How is $\mathbb{P}(G | \mathcal{D})$ or $\mathbb{P}(\widetilde{G} | G,\mathcal{D})$ defined? What is the mariginal distribution $\mathbb{P}(G)$ and the data generating process $\mathbb{P}(\mathcal{D} | G)$? Why is $\mathbb{P}(G | \mathcal{D})$ called "prior" rather than "posterior"?
- Why does switching the focus to "feature" level from graph level make the computation for feasible? The "feature"s are essentially sub-structure in the graph, e.g. edges. How to compare the size of two levels quantatively?
- In proposition 1, it seems we do not need $\widetilde{\pi}$ to be close to $\pi$?
- In Theorem 2, what is the intuition that we need at least $2m-1$ experts? More experts leads to more parameters to estimate. Does the estimation/identification of $\pi$ come from number of experts?

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
4

### Summary
The authors study ensemble-based causal discovery, where the results of several causal discovery algorithms are aggregated into a single final graph. In particular, they divide the graph into different "features", such as possible edges, and perform feature-wise aggregation. Following a typical aggregation approach, they assume there are $n$ independent experts, and seek to approximate the Bayes optimal voting rule by estimating the transition probabilities of each expert on each feature and the prior over each feature. To estimate these quantities, they assume iid samples from each expert, and use minimum distance estimation.

The authors provide two sets of experimental results. In the first set of results, they use a simulation where, instead of using causal discovery algorithms, they specify each expert by a deterministic (but unknown) transition matrix from the true graph to its predicted graph, and show that their procedure outperforms both the best expert and the plurality voting rule. In the second set of results, they consider actual causal discovery algorithms such as ICA-LiNGAM, GES-BIC, PC, SCORE, NOTEARS, and DAGMA, on 9 benchmarks from the bnlearn repository, with the proposed method achieving the best results on the SHD metric and competitive results on Adjacency F1 and Orientation F1 metrics.

### Strengths
**Originality:** To the best of my knowledge, the use of more sophisticated aggregation techniques for causal discovery is relatively novel.

**Quality:** The paper is of good quality, with a good balance of theoretical motivation and experimental investigation.

**Clarity:** The motivation for going beyond plurality voting and other simple aggregation techniques is well-articulated.

**Significance:** The results are promising and may help push causal discovery towards more practical reliability.

### Weaknesses
**Theorem presentation/novelty:** Theorem 1 in this paper seems to be an extension of Theorem 1 in [1] to multi-class classification and unbalanced truths. Overall, I think the nature of both Theorems 1 and 2 are obscured by an early overcommitment to the causal discovery setting: the nature of the results is really about general aggregation of experts with categorical outputs. I'm not an expert in the area, but aggregation is a very well-studied subject in statistical learning theory, and the paper would benefit from contextualizing itself within this field before turning the attention to causal discovery. A better understanding of aggregation would also address some issues with the theoretical assumptions, discussed in the next point.

**Theorem assumptions:** A key assumption (Assumption 1) is that the experts make independent decisions (conditioned on the truth). However, in practice, each of the causal discovery experts would be using the same dataset (or, you would have to reduce the sample size and divide a dataset of size $M$ into $n$ datasets of size $M/n$), i.e., their decisions are only independent when conditioning on both the ground truth *and* the dataset. Further, to estimate the expert transition matrices, the theory requires another $N$ independent voting profiles from each expert. Overall, these assumptions are far different from how any aggregation method would be used in practice, and seem to be too strongly based on intuitions from crowdsourcing/voting theory. At the very least, I think it is essential that the paper emphasizes these limitations and better describes the mismatch between theory and practice (e.g., the experiments use bootstrapping to get different graph samples for each expert, which would not be independent). Ideally, the authors would be able to use more sophisticated analysis to handle these issues, but in reality I think it's fair to leave that direction for future work.

[1] Berend et al. (2015), A Finite Sample Analysis of the Naive Bayes Classifier

### Questions
1. Is the SID plot in Figure 1 an error?
2. Since you aggregate at the feature level, there could be issues with combining the aggregated features into a single graph. For example, with plurality voting, if expert 1 gives the graph (1->2, 2->3), expert 2 gives the graph (2->3,3->1), and expert 3 gives the graph (3->1,1->2), then the aggregated features would be 1->2, 2->3, and 3->1, so combining these features gives a cyclic (rather than acyclic) graph. How do you handle such issues? If you don't, what is the justification?

### Soundness
3

### Presentation
3

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
This work aims to address three fundamental limitations of ensemble-based causal discovery. They claim existing methods are largely heuristic, lack theoretical guarantees and guidance on how ensemble design choices affect performance. The paper proposes a principled voting-based framework for structural ensembling, establishing conditions under which the aggregated structure recovers the true causal graph. They concluded that their analysis yielded a theoretically justified weighted voting mechanism that informs optimal choices regarding the number, competency, and diversity of causal discovery experts in the ensemble. Their model was tested on both synthetic and real world data to verify effectiveness and robustness.

### Strengths
S1: Extends the novel estimation algorithm based on optimal transport in a clearly non-trivial way.

S2: Solid mathematical formulation with proofs or guarantees, with clear connection between intuition and formal results.

S3: Comprehensive empirical validation with experiments on diverse, well-chosen benchmarks

### Weaknesses
W1: The solution approach is hard to follow with very limited intuitive explanation of complex maths used.

W2: It is not clear how the paper addressed the well-defined problems with existing ensemble methods.

W3: Authors did not share the code therefore reproducibility is questionable.

W4: The problem the paper is trying to solve (existing methods are largely heuristic, lack theoretical guarantees) is not clearly motivated and lacks practical importance.

### Questions
Q1: When learning the structure from noisy experts or base learners, how is the competence of an expert measured, before creating the competence transition matrix?

Q2: Different experts/base methods use different parameters to identify causal links, how did you reconcile and consolidate these parameters? 

Q3: The proposed solution using the Optimal Transport framework is not clear.

Q4: Can this proposed approach identify nonlinear causal links, i.e often found in dynamic systems like climate systems with feedback loops?

Q5: Did you encounter situations were weighted linear voting rule say false positives, i.e the experts voted the presence of an edge in a graph when there exist no edge in the true graph? How did you reconcile such situations?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes an ensemble framework for causal discovery task. More specifically, the authors analyze the conditions under which the predicted graph matches the true graph. They also provide theoretical justification for different size, competency and diversity levels of causal discovery experts. Finally, they show performance on synthetic and real-world datasets.

### Strengths
The proposed method offers a strong contribution as it provides practitioners with a guide on how to configure ensemble strategies while combining outputs of multiple causal discovery experts for optimal performance. It’s interesting how the authors avoid searching the exponential space by decomposing the graphs into sub-structures.

### Weaknesses
Below I provide my comments:

## **Major**
* The paper is little convoluted. The steps of the proposed algorithm can be organized in a better way.   Section 3.1 is notation-heavy, with lots of notations, making it quite hard to keep track of all notations.  
* A simulation of the voting mechanism is needed, i.e., how $\hat{Y}(w)$ is constructed and what its relation is with $\hat{G}(\sigma, \mathcal{S})$, how the transition matrix looks like and how it contributes.
* Some of the discussion from Appendix D can be transferred to the main paper (Section 2) to provide readers with a better understanding of the concepts. Many concepts in the main paper cannot be connected without proper preliminaries or background.  
* How parameter estimation with optimal transport works is not clear. The authors should focus more on explaining the main algorithm intuitively.
* The authors did not compare with other existing distance-based or voting-based structural aggregation algorithms. For example, the authors cited papers such as Mio et al. (2025), Guo et al. (2021), (Tang et al., 2019), Aslani & Mohebbi, 2023. They should compare their performance with these ensemble frameworks.
* In figure 2, are the causal experts (black) executed once to record their performance? In such case, the proposed ensemble framework should perform better always. Would that be a fair comparison? I would request the authors to point me if I misunderstood this.

---

## **Minor**
* There are too many versions of the graph: $G$, $\mathbf{G}$, $\mathcal{G}$, $\tilde{G}$, $\tilde{G}_i$, $\hat{G}$.  
* Some visualization or example of $T$, $\theta$ would be nice.  
* Line 140: shouldn’t the incorrect prediction be a sum over all $q_{i, G_j | G_1}$?  
* It is not clear intuitively what the bias term indicates in Equation 3.  
* Lines 200–201: should it be an equal sign?

### Questions
Below I share my questions. 

## **Question**
* What does convergence mean in the “converging to the correct decision”?  
* Line 346: at least $2m - 1$ experts are needed. For a three-node directed graph having three features {(v1, v2), (v2, v3), (v1, v2)}, what will be the value of *m* and how many experts do we need?  Is it number of experts always feasible in real-world setups?
* If $\mathbf{G}_1$ is the true graph, does the proposed algorithm work with a vector of probabilities $T_i[:, G_1]$ instead of the matrix?  
* How large is the competence transition matrix $T_i$? Is it of size $|\mathcal{G}| \times |\mathcal{G}|$? Is it very large and hard to learn?  
* In Condition 1, the authors consider that two rows are distinct element-wise. To my understanding, each index of $T_i$ represents $T_i := [{P}(\tilde{G}_i = G_j |  G = G_k,  D)]$. A row represents the probability distribution of different true graphs while the prediction is fixed. On the other hand, a column represents the distribution of different predictions while the truth is fixed. 
Now, when the true graph is different, if the algorithm provides different probability distributions over the predicted graphs, that should indicate the algorithm’s capability to distinguish between two graphs, right? Why then do the authors use rows in condition 1 and not columns of the matrix?

### Soundness
3

### Presentation
1

### Contribution
3
