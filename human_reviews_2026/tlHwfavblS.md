# Exact Combinatorial Optimization for Synchronization of Partial Multi-Matching

- Decision: Reject
- Scores: 2, 4, 6, 10

## Abstract
In permutation synchronization, the goal is to find globally cycle‐consistent correspondences from noisy pairwise matchings. In this work, unlike spectral relaxations that embed permutations into an orthogonal space and often result in inaccuracies, we maintain the problem in its original combinatorial form. By shifting the affinity spectrum to ensure positive semidefiniteness, we cast the trace‐maximization over partial permutations as a convex‐in‐P formulation. Our minorization-maximization scheme then replaces this with a sequence of exact linear‐assignment subproblems, the row-/column-sum constraints of which are totally unimodular, guaranteeing integral solutions with no rounding. This direct, combinatorial approach delivers a monotonic objective ascent, convergence to a KKT point, and achieves superior accuracy, cycle consistency, and runtime on image-matching benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper addresses the multi-graph matching problem, focusing on how to exploit cycle consistency to refine a set of reasonably good pairwise initial matchings. The authors formulate the task as a semidefinite quadratic assignment problem and solve it with a sequence of linear assignment problems. The proposed method efficiently improves matching quality within a relatively short computation time.

### Strengths
The paper is well written and easy to follow, with a clear structure and sufficient background provided to help readers outside the immediate research area. The theoretical analysis is relatively complete and logically sound. Experimental results demonstrate that the proposed algorithm achieves noticeable performance improvements, confirming its practical effectiveness.

### Weaknesses
1. The experimental design has a major flaw. The proposed method and the compared baselines are not solving the same problem setup: the proposed algorithm refines results that are already near-perfect (obtained by Stiefel), whereas the baselines start from scratch. This discrepancy significantly exaggerates the performance advantage of the proposed method. To fairly evaluate the contribution of the MM module, the authors should examine how MM improves results from different initialization methods, or alternatively, treat “MM + Stiefel” as an enhanced version of Stiefel and compare it with other approaches while including the total computation time.

2. Lemma 4.1 is overly simple and does not warrant a separate statement. It unnecessarily occupies space.

3. Section 1 contains only one subsection (1.1), which is structurally redundant and could be merged for clarity.

### Questions
1. It would be interesting to see how the algorithm performs when the initial matching quality is low.
2. What is the difference between problem 8 and problem 9?
3. Can the linear assignment problem be solved using the classic Hungarian method?

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper outlines a MM algorithm for permutation synchronization. Early work on this problem motivated by applications in cryo-EM and computer vision, seeks to take permutations (which give feature correspondences between a pair of images) and "synchronize" them globally which means that cycle consistency (i -> j -> i) is preserved. Initial approaches were spectral relaxations but more powerful methods have been developed. The approach described here argues to stay in the combinatorial setting. It first convexifies the trace obj and then solves a sequence of linear subproblems. The constraints for these subproblems involve a matrix that is totally unimodular. So integral solutions (similar to max-flow/min-cut setting) are guaranteed - at least for every iteration. This gives monotonic convergence to a KKT point. On several datasets experiments are shown, which suggest that the method performs well.

### Strengths
1. The paper synthesizes several standard ideas into something useful for the problem. Objective is convexified, MM helps linearize it and then TUM property is used to avoid using rounding etc. This gives an algorithm which is combinatorial avoiding continuous spectral or Stiefel relaxations. 

2. The analysis is relatively easy to follow. No major claims, basically monotonic progress to KKT. The experimental analysis shows that it works well and runtime etc are competitive. This is fine. 

3. Permutation synchronization is a mature problem. To that sub-community interested in this problem and/or its applications, the algorithm in this paper could offer value. Perhaps also limited hyperparameter tuning at the rounding stage?

### Weaknesses
1. The algorithm and its main findings are fine. But for a well studied problem, a strong technical result would include analysis of approximation ratio (under some assumptions) or convergence rate etc. This paper gives neither. The algorithm is warm-started from a continuous solution, so its not obvious how close to the reported solution this initialization already is. 

2. The experiment results are a little underwhelming. Yes it does cover some of the datasets in those original papers. But the paper should do more to convince the modern ML community that the problem is worth studying and the experiments enable important downstream applications. In its current form, I'm afraid that the results in the narrow scoping shows benefits but broadly, it does not make a strong case for which modern use cases will benefit. For example feature matching across views is now handled by much more sophisticated features/models. Is permutation synchronization still a valid issue there?

### Questions
1. Given that modern methods can give highly discriminative features, to what extent is a classical multi-image matching problem still relevant? Why not simply perform nearest-neighbor matching on these features, and show that for some downstream use cases the procedure actually helps? 

2. The analysis relies on assumption that solves a LP to optimality at each step. Will this work for the projection step in the appendix? If so, please describe. If not, what happens to the analysis. Does it work?

3. The procedure seems to warm start from a strong baseline. What about ablations where the initialization is the given set of noisy permutations?

4. does TUM hold under row-stacking? Try A = [1 1 0; 1 0 1] and B = [0 1 1].

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
5

### Summary
The authors propose a simple and elegant minorization-maximization algorithm for permutation synchronization. It employs standard tricks to change the subproblems into exactly solvable subproblems. Experimental evaluations shows promising results and fast execution.

### Strengths
Writing:
- The related work and introduction gives an exhausting picture of the state of the art in multi-graph matching.
- The technical section is written very well and is accessible. Possibly some illustrations for the majorization and the universe formulations might help additionally.

Conceptual:
- The algorithm is a nice contribution and is sound in itself. I also think it has additional potential when incorporating arbitrary linear and quadratic costs for solving the general multi-graph matching problem.

Experimental:
- On the reported metrics and the somewhat limited problem sets and restrictive experiment setup (see below for my critique on that) the results are good.

### Weaknesses
Writing: 
- The title implies suggests that you solve the permutation synchronization exactly, while what you do is you have a sequence of subproblems each of which is solved exactly, but the overall problem solution is only approximated. Please change the title, it is misleading and might be felt as overselling your approach.
- In the abstract you write "superior cycle consistency". Cycle consistency is either fulfilled or not, so you cannot be superior here and fulfill it more.
- The experimental evaluation is repetitive and tedious. I would like to have all results in one table, with best and second best results highlighted. Right now there is a lot of repetitive text that just repeats the results from the tables. Add the ETH3D results to the main part but have an average over all four subproblems. Improve the tables (highlight whether higher or lower is better).

Conceptual:
- A lot of space is taken up by the total unimodularity of (8). I think it is just another variant of the linear assignment problem, so one could have derived exactness from that without the more involved proving TU (which also is almost exactly like the proof of unimodularity for the linear assignment problem anyway).
- I am not sure about the projection approach to solving (8). First of all, it can be reformulated to be a linear assignment problem and offloaded to a combinatorial algorithm. Second, for first order methods I think the Sinkhorn Knopp Algorithm is state of the art.

Experimental:
- I think reporting the objective from (8) is the most important metric, since it directly measures the algorithm's performance. Other metrics like F-score, inlier ratio etc. take into account learning aspects that are outside the bounds of what the algorithm does.
- The experiments only convey an aspect of the potential performance. The proposed approach only does permutation synchronization, while other approaches can optimize w.r.t. arbitrary costs and can additionally incorporate quadratic terms that will typically result in better solutions. As such, while I believe that the proposed method can be better on pure permutation synchronization, this is less important when one can run other solvers like the one from Kahl et al on arbitrary linear and quadratic costs.
- Relatively few problems are considered. I think additionally the worms dataset also used in Kahl et al might be interesting and they are freely available.  The CMU datasets are rather easy and synthetic and basically this means that only the ETH3D datasets are real-world challenges.

### Questions
- Do you initialize all methods with the exact same pairwise assignments?
- Can you provide objective costs for each algorithm?
- Why is inlier ratio on the ETH3D datasets higher for GREEDA but F-score lower than for your method?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
The paper introduces a new strategy for permutation synchronization based on a minorization-maximization strategy and the integrality guarantee of solutions to linear programs with totally unimodular matrices. Performance on (small) standard matching benchmarks from computer vision is impressive.

### Strengths
Permutation synchronization is an important problem in several areas, including computer vision. The paper introduces a completely new idea to this field and shows that it empirically outperforms all previously used techniques on multiple benchmark datasets.

- The paper introduces a genuinely new optimization technique to an important problem
- The idea is crisp and clear and appears to be mathematically sound
-  Empirical performance is compared to all other competing algorithms and is found to be superior

### Weaknesses
At first I thought the authors evalauted the algorithm on just the "house" dataset, then I realized that there are several other sets of results in the Appendix. This could be made a bit clearer. It is not entirely clear where the other datasets come from, how many examples there are in each, how they were labelled, and so on.

Given the AI era that we live in, and the generality of their algorithm, it would be nice if the authors also tried out their algorithm on more ambitious, larger, and more varied data, for examples matching molecules to each other or parts of graphs.

### Questions
It seems like your algorithm is equally applicable to matching complete permutations. Are there other types of matching (more broadly, combinatorial optimization) problems where it could be used? Is there precedent for it?

### Soundness
4

### Presentation
3

### Contribution
4
