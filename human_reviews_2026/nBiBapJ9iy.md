# A Scalable Global Optimization Algorithm For Constrained Clustering

- Decision: Reject
- Scores: 4, 4, 4, 2

## Abstract
Constrained clustering leverages limited domain knowledge to improve clustering performance and interpretability, but incorporating pairwise must‑link and cannot‑link constraints is an NP‑hard challenge, making global optimization intractable. Existing mixed‑integer optimization methods are confined to small‑scale datasets, limiting their utility. We propose Sample-Driven Constrained Group-Based Branch-and-Bound (SDC-GBB), a decomposable branch‑and‑bound (BB) framework that collapses must‑linked samples into centroid‑based pseudo‑samples and prunes cannot‑link through geometric rules, while preserving convergence and guaranteeing global optimality. By integrating grouped-sample Lagrangian decomposition and geometric elimination rules for efficient lower and upper bounds, the algorithm attains scalability via embarrassingly simple parallelism. Experimental results show that our approach handles datasets with 200,000 samples for cannot-link constraints and 1,500,000 samples for must-link constraints, which is 200 - 1500 times larger than the current state-of-the-art under comparable constraint settings, while reaching an optimality gap of <= 3%. In providing deterministic global guarantees, our method also avoids the search failures that off‑the‑shelf heuristics often encounter on large datasets.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a new algorithm for optimally solving sum of squared errors clustering problems (k-means) in the constrained setting, where one includes certain must-link and cannot-link constraints on certain pairs of points. Since this is an NP-hard problem, optimal solutions for it are hard to compute and existing methods do not scale to very large datasets.

The new method proposed in this paper scales to much larger datasets than existing techniques, reaching up to instances with 1.5 million samples for must-link constrained cases and 200k samples for cannot link cases, with low optimality gaps. Many prior techniques struggle to obtain optimal solutions with hundreds of points, and the best existing results, according to this paper, only scale to instances around 1000 points.

The paper achieves these speed-ups by introducing geometric sample determination rules to eliminate cannot-links, and merge must link constraints into pseudosamples. These tricks speed up the branch and bound process for finding optimal solutions; the BB approach specifically branches on cluster-center variables rather than sample-to-cluster variables.

### Strengths
* The constrained clustering problem is well-motivated and is a very relevant topic; it is interesting to develop exact algorithms for this problem even though it is NP-hard
* The numerical experiments seem impressive and do appear to significantly improve in practice over previous techniques.
* The paper incorporates helpful figures in presenting the theoretical results.
* The structure of the paper is fairly good. There do not appear to be many issues with typos or grammar

### Weaknesses
There are two weaknesses:

W1: The key contribution in section 3.1 is unclear due to technical writing issues

In the Sections 2-3, there are many technical details that are not explained precisely to the point where the core technical approach is unclear to me. I am not intimately familiar with research on branch-and-bound algorithms for clustering, but I have good general familiarity with optimization methods for clustering problems, and there were many technical details I could not parse, often because notation was not properly explained or defined. 

Here are some questions I had in the technical details while reading specifically through Section 2 and Section 3.1

* Is Problem 2 supposed to be an equivalent reformulation of Problem 1? I'm pretty sure it is, but that's not explicitly stated, and you call it a semi-supervised MSSC, which is the first time you use that term (you did not use it to describe Problem 1) so at first it's not clear if you're introducing some equivalent formulation or a new problem. It's also not entirely clear to me what Problem 2 has to do with what is in section 3
* In Problem 2, are you missing a "for all k \in \mathcal{K}" in constraints b and c?
* What's a big-M constant? (line 121)
* In line 139, it's not clear to me why \rho is the current incumbent cost. The objective function is a sum of squares, whereas \rho is the maximum distance between a point and a centroid. These are not the same thing. I'm not sure what I'm missing, but this doesn't seem explained very clearly. 
* In line 145, you refer to (s,k) as a candidate pair, but you don't explain what that means and it's not immediately clear what you're saying.
* In the definition of \rho, you use the notation \mu_{k(s)}^\text{best}. This is used without a clear explanation or definition before being used. You do explain right afterward that this is the "centroid to which [a sample] is assigned in the current best solution", which is good, but it'd be helpful to have a clearer explanation beforehand including what k(s) represents. 
* You say M_k represents any region, but then specify you mean an axis aligned box. "region" is a much more general term than "axis-aligned box" though, so I don't think you really mean "any" region.
* The notation for M_k seems very close to M_0, which is used to mean something completely different later (the "solution space"). This is a bit confusing.
* In some places you write d_{min}(x_s, M_k), and in the figures you use d_{min}^i for some integer i. Can you give a more specific mapping between these notations so a reader can parse it more readily?

Obviously, some of these issues are larger than others. But when taken all together, I ultimately was not able to follow the core contribution in Section 3.1 and what Lemmas 3.1 and 3.2 are ruling out. 

Weakness 2: The technical contribution in 3.2 seems quite limited.

In particular, this section seems to just be making the observation that if you include a must-link constraint, you can just collapse points into a new sample and then solving an equivalent unconstrained problem. This is a pretty basic and standard observation in constrained clustering, and it's significance is maybe a bit overstated. In the summary of contributions in the introductions this is described as "We introduce a centroid-based pseudosample formulation for must-link subsets", but that seems like an overly complicated way to describe the simple thing going on. 

One final minor comment: the opening sentence in 3.3 is identical to the opening sentence in 3.4. Was this intended? It seems very repetitive.

### Questions
Can you please clarify my questions regarding the notation and points being made, especially in section 3.1?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The author introduces SDC-GBB, a novel branch-and-bound framework to solve the NP-hard Minimum Sum-of-Squares Clustering (MSSC) problem with constraints (e.g., must-link (ML) and cannot-link (CL)) that achieves global optimality on much larger instances. The algorithm branches only on the continuous centroid variables, avoiding the combinatorial explosion of assignment variables, and bounds are computed by a grouped-sample Lagrangian decomposition and a heuristic k-coloring approach.  

The paper proposes several techniques to improve efficiency, such as collapsing each ML-connected component into repeated “pseudo-samples” at its centroid, and the geometric sample pruning rule to eliminate infeasible cluster assignment from geometric and constraint checks. Thus preserving the exact optimum while reducing problem size. 

Empirically, SDC-GBB outperforms prior exact methods: it solves ML-only clustering on up to 1.5 million points and CL-only on up to 200,000 points.

### Strengths
- The pseudo-sample reformulation provably exact transformation with Lemma 3.3 and Theorem 3.4 shows that such a replacement shifts the objective by a constant, with the global minimum preserved.
- The method comes with a convergence proof that the BB scheme recovers the true optimum given exhaustive splitting.
- Geometric distance bounds to eliminate sample-to-cluster assignments, which prune the BB search tree aggressively before branching.
- SDC-GBB can handle huge datasets. The authors demonstrate orders-of-magnitude improvements in scale (up to 1.5M points) and achieve very low optimality gaps.

### Weaknesses
While the idea is interesting, some weaknesses in the proposed work's methodology need clarification:
- **Geometric Pruning Rules in High Cluster/High Dimensional Data**  Branch-and-bound on cluster centers lives in an $mK$-dimensional continuous space.  Geometric pruning rules are known to degrade in high dimensions or when K is relatively large;  they potentially weaken the elimination rules. The paper provides no theoretical justification or experimental analysis for $K>3$ or higher-dimensional data.  All experiments use low-dimensional data (2–8 features or 2D Gaussian mixtures) and a fixed small cluster count (K=3).
- **Computing Axis-bounding Box** For large feature spaces or larger $K$, $d_{\min}$ and $d_{\max}$ over each box $M_k$ may be loose if the region $M_k$ is large and can be costly to compute. The authors do not mention how the initial region $M_0$ is chosen or how boxes are subdivided in the methodology section.
- **High-variance ML components** The pseudo-sample approach relies on the covariance computations, which can be numerically unstable for large ML components. No theoretical or empirical analysis is given for more practical cases, such as a high-variance ML group.

Also, there are some things the authors can improve in the experiments:
- **Limited metric in experiments** The metric in the experiment only focuses on SSE. Since constraints often reflect domain needs, one might also evaluate clustering quality like the Rand Index. Also, the paper does not address whether the solutions actually respect all constraints.
- **Fraction of constraints** Also, the varying fraction of must-links vs. cannot-links in the dataset could dramatically affect difficulty, but no sensitivity analysis is presented.
- **Unclear statement** In the experiment design, some statement like “each dataset has $n/4$ samples bounded by ML/CL constraints” is unclear; it is not clear how many total constraint pairs this yields.
- **Reproducability** code and results are not provided in the current submission for reproducibility.

### Questions
-   How exactly are $d_{\min}(x_s, M_k)$ and $d_{\max}(x_s, M_k)$ computed for an axis-aligned box $M_k \subset \mathbb{R}^m$? 
  Please clarify whether you use the closed-form per-coordinate formula implied in the text and how this scales with dimension.
-   For $\beta_{\text{SG+LD}}(M) := \max_{\lambda} \beta_{\text{SG+LD}}(M,\lambda)$, do you solve this dual at every BB node or only at nodes whose bounds are too loose?
- you use a single global $N = \max_{s,k} \sum_i \max\{|x_{s,i} - \mu^\ell_{k,i}|^2, |x_{s,i} - \mu^u_{k,i}|^2\}$, which can suffer from numerical or solver-sensitivity issues from large $N$. How did you resolve this?
- How key parameters (tolerance $\epsilon$, time limit, grouping sizes for Lagrangian decomposition) affect outcome and runtime?
- How does the algorithm perform with larger $K$ and higher-dimensional data?
- ablation study on different elements of the algorithms; for example, consider adding a baseline that first collapses ML components and then runs a standard $k$-means. 
- it would be helpful for the authors to mention the trade-off between scalability and solution quality across methods. In particular, for the heuristic baselines, what are the corresponding wall-clock times and objective values as problem size increases, and how do these compare to the proposed approach?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper introduces SDC-GBB, a deterministic global optimization framework for constrained clustering. It combines geometric pruning, pseudo-sample aggregation, and Lagrangian decomposition to scale branch-and-bound methods to much larger datasets. The work is clearly written and technically careful.

### Strengths
Addresses a hard and relevant problem. The algorithmic framework is well motivated and mathematically sound. Most parts of experiments are extensive and the implementation seems professional. The writing quality is high overall.

### Weaknesses
The conceptual novelty is limited — most components (B&B structure, geometric elimination, decomposition) have appeared in previous global optimization literature. The claimed scalability relies heavily on large-scale hardware rather than clear algorithmic gains. Experimental comparisons are somewhat narrow, mostly against older baselines. The contribution feels more like a well-engineered variant than a new learning idea.

### Questions
1. Could the authors clarify what parts of SDC-GBB are genuinely novel compared to Cao & Zavala (2019) and Piccialli et al. (2022)? 2. How much of the reported speedup arises from parallelization versus algorithmic improvement? 3. What is the computational complexity per iteration or per branch node? Any empirical scaling curve (e.g., runtime vs n)? 4. Would the method still converge on moderate hardware (e.g., 32 cores, 64 GB RAM)? 5. Can the same framework be extended to non-Euclidean or kernelized distances? 

I would be glad to improve my scores if the authors could resolve my doubts.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents a new method to address the constrained clustering problem involving *cannot-link* (CL) and *must-link* (ML) constraints. The authors use a decomposable branch-and-bound (BB) framework that prunes cannot-link pairs using geometric rules. The used algorithm achieves highly scalable pairwise k-means constrained clustering through parallel computation.

### Strengths
1. The approach effectively combines mathematical rigor with computational efficiency, demonstrating potential scalability of the  MP(mathematical programming)-based method  to large datasets.

2. In general, the paper is easy to go through.

### Weaknesses
1. The proposed technique appears highly similar to [1], raising concerns about its originality and incremental contribution beyond existing work.

2. In prior research, must-link constraints can also be represented by representative points while preserving the problem's hardness.

### Questions
1. After introducing parallelism into the algorithm, how does the method maintain its optimality guarantees (as in the theoretical part)? Is the solution remains optimal? If not, what is the error bound?

2. Could the authors provide an analysis of the algorithm's running time complexity? Particularly, provide discussion of its scalability in terms of both data size and constraint density?

3.  Several critical questions for the experimental analysis:

a. Heuristic comparison:
   - Which specific heuristic methods were used in the experiments?
   - There exist many comparable approaches under the k-means setting (see Table 1 in [2]).
   - Could the authors compare against some of these methods and justify the selection of baselines?

b. Dataset consistency:
   - Tables 1, 2, and 3 appear to use different datasets. Could the authors provide complete results across all datasets for better comparison?

c. Table 4:
   - The authors describe certain observed phenomena without sufficient explanation.
   - Additionally, I believe TIME/NODE should be tested directly rather than reporting only *computing time per node*.

d. Table 5:
   - Why does the number of nodes decrease as the dataset size increases?

e. Fairness of comparison:
   - The experimental comparison appears potentially unfair.
   - For example, the paper compares BLPKM-CC and COP in terms of clustering cost, but metrics such as ARI, NMI, Purity, and running time are not reported. These should be included for completeness.

f. Table 6:
   - The results for SDC-GBB (CL) on the *iris* and *seeds* datasets remain identical across all metrics, suggesting the clustering assignments did not change.
   - Does this imply that the CL constraints have no influence (i.e., the solution remains unconstrained or was pruned by branch-and-bound)?
   - If so, in Table 8, why does BLPKM-CC achieve a lower cost than ENCODE k-means-post?
   - Generally, unconstrained costs are expected to be lower than constrained ones - please clarify this discrepancy.

[1] Kaixun Hua, Mingfei Shi, and Yankai Cao. *A scalable deterministic global optimization algorithm for clustering problems.* In *International Conference on Machine Learning (ICML)*, pp. 4391鈥?4401. PMLR, 2021.
[2] Philipp Baumann and Dorit S. Hochbaum. *An algorithm for clustering with confidence-based must-link and cannot-link constraints.* *INFORMS Journal on Computing*, 37(4):1044鈥?1068, 2025.

### Soundness
2

### Presentation
2

### Contribution
2
