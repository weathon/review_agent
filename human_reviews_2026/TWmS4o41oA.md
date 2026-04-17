# Better Learning-Augmented Spanning Tree Algorithms via Metric Forest Completion

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 8, 4

## Abstract
We present improved learning-augmented algorithms for finding an approximate minimum spanning tree (MST) for points in an arbitrary metric space. Our work follows a recent framework called metric forest completion (MFC), where the learned input is a forest that must be given additional edges to form a full spanning tree. Veldt et al. (2025) showed that optimally completing the forest takes $\Omega(n^2)$ time, but designed a 2.62-approximation for MFC with subquadratic complexity. The same method is a $(2\gamma + 1)$-approximation for the original MST problem, where $\gamma \geq 1$ is a quality parameter for the initial forest. 
We introduce a generalized method that interpolates between this prior algorithm and an optimal $\Omega(n^2)$-time MFC algorithm. Our approach considers only edges incident to a growing number of strategically chosen "representative" points. One corollary of our analysis is to improve the approximation factor of the previous algorithm from 2.62 for MFC and $(2\gamma+1)$ for metric MST to 2 and $2\gamma$ respectively. We prove this is tight for worst-case instances, but we still obtain better instance-specific approximations using our generalized method. We complement our theoretical results with a thorough experimental evaluation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The submission improves upon the recent learning-augmented algorithm of Veldt, Stanley, Priest, Steil, Iwabuchi, Jayram and Sanders. It provides a tighter analysis which explains the previous gap between theoretical lower bounds and experimental performance, and also develops a related means of interpolating between slower approaches (with worst-case guarantees) and approximation.

### Strengths
The contributions are, as a whole, clearly articulated and nicely presented. The submission's contributions are relevant for Machine Learning research and provide a nice combination of theoretical results supported with auxiliary experiments. The results are non-trivial. Moreover, the submission is nicely written overall.

While not a strength per se, I found the openness and detail in which the authors discussed their use of LLMs refreshing.

### Weaknesses
The contributions could be seen as slightly incremental over the recent work of Veidt: the additional contributions "on top" of Veidt are not that novel/substantial (this applies in particular to Subsection 3.1). On the other hand, the motivation for the theoretical contributions of Subsection 3.2 and the experiments in Section 4 is weaker than for the improved bounds; these show that one can interpolate between the exact and approximate procedures by obtaining larger representative sets R (via a DP with approximation guarantees), which is not that surprising. 

I view both of the above as "minor" since in both respects, the submission only seems to be slightly below what I would consider the average for a combined theory+experimental ICLR paper.

*Typos and Minor Suggestions*

-Top of page 6: use \qedhere to place qed in correct location.

-Page 2: "approximations factors" ->  "approximation factors"

-Page 2: "... a disjoint set of trees called the initial forest, such that each of the n points belongs to one component in the forest." This sentence is unclear, as it does not specify what are the vertices of the forest. The formalization on page 3 is clear, however.

-Page 6: "This methods" -> "This method"

### Questions
On page 1, the submission attempts to provide a more formal footing for learning-augmented algorithms as follows:

"The goal (of learning-augmented algorithms) is to design an algorithm that is consistent, meaning that it produces near-optimal outputs when the prediction is good, and robust, meaning that it recovers the same worst-case guarantees as a prediction-free algorithm when the prediction is bad"

However, isn't it the case that *every* procedure which is "consistent" can be made "robust" by simply running the algorithm achieving worst-case running time in parallel? After all, worst-case guarantees are typically provided in O-notation (and this also holds for the lower bounds for MinTSP provided by (Indyk 1999)), so a factor-2 decrease in the running time is still tight w.r.t. the worst-case lower bound. If the above is indeed a gap, please suggest an alternative formulation of the goal of learning-augmented algorithms - would it simply be achieving the "consistency" property?

Additionally, the description of the purpose of the experiments in the Introduction was a bit unclear to me. Based on my current understanding, they demonstrate that the "novel" DP you develop towards achieving your theoretical guarantees also outperforms greedy/trivial solutions in practice, but feel free to comment on (and potentially correct) this.

### Soundness
3

### Presentation
4

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
This submission deals with the problem of computing an approximate minimum spanning tree (MST) in subquadratic time when a partial solution is already given. More precisely, we are given a metric instance of the MST problem and an initial forest. The goal is to compute approximately a minimum spanning tree in subquadratic time among all spanning trees that contain the given forest. This problem is known as the metric forest completion (MFC) problem and it is motivated in the context of learning-augmented algorithms in which, e.g., a learning algorithm provides already a partial prediction for the minimum spanning tree.

The submission is heavily based on the research by Veldt et al. (2025), who first considered the MFC problem. The quality of the initial forest is measured by a parameter γ, which corresponds to the overlap of the given initial forest with some MST and is defined as the ratio between the current weight of the given forest and an optimal choice of an optimal minimum spanning tree, where we only count the weight of edges if the corresponding endpoints are in the same component as in the given forest (γ=1 means an optimal overlap with some MST). Veldt et al. gave a 2.62-approximation for MFC and showed that if the initial forest has a γ-overlap then the final spanning tree is a (2γ+1)-approximation of an MST. They also conducted multiple experiments and concluded that even simple heuristics for the initial forest provide very good results on synthetic and real datasets.

As first main contribution, the paper at hand improves on the theoretical bounds by showing that the algorithm by Veldt et al. (2025) even achieves a 2-approximation for the MFC problem and a 2γ-approximation for the metric MST problem. Additionally, it is shown that these bounds are tight in the worst case by designing a family of instances with γ=1, where for a specific choice of representatives in each component of the initial forest the ratio between the MST and the optimal corresponding MFC solution approaches 2 in the limit. The second main contribution is a modification of the algorithm by Veldt et al. (2025). While the latter is based on choosing a single representative for each component of the initial forest, the generalized version considered in this submission, is allowed to choose multiple representatives in each component, where the total number of additional representatives is upper bounded by some value b. For this setting (called best representatives problem) the authors design a 2-approximation using a 2-approximation algorithm for k-center and dynamic programming. Additionally, they test their algorithms on some of the real-world instances used by Veldt et al. and show experimentally that slightly increasing the number of allowed representatives gives some improvement in solution quality with only a small increase in runtime.

The paper is generally well written and follows closely the notation and experimental setup of Veldt et al. The theoretical results are clearly written. The proof of the 2-approximation is similar to the proof ideas by Veldt et al., but uses a different technique to bound the final cost and show the corresponding approximation factors. Allowing more representatives in each component is a natural extension of the algorithm. The experimental results show a slight improvement in solution cost when using more representatives per component or allowing for a larger additional time budget beyond the initial case with having a single representative.

### Strengths
The authors improve the analysis of the existing approximation algorithm for the MCF problem and show tight bounds. The theoretical analysis is sound and the writing is good. The modified algorithm using additional representatives performs well in experiments.

### Weaknesses
The theoretical considerations rely very much on the work of Veldt et al. (2025) and neither the problem nor the algorithm is new. The extension towards additional representatives is quite canonical and does not provide better theoretical bounds. I have the impression that the results are slightly overstated: More representatives lead to better solutions but the difference seems to be rather small while the running time increases by a significant factor (see questions for more details).

### Questions
- The experimental section misses some additional information about the system on which the experiments were run. 

- Additionally, the significance of including additional representatives with respect to the runtime is not clear. For example, on the Cooking dataset it seems that the standard algorithm (b=0) did run in the experiments in approximately 100 seconds while the additional variants for selecting additional representatives can take more than 1000 seconds while improving the approximation factor only slightly. The improvement looks more severe due to the scaling and truncation of the diagram but if I understand Figure 2 correctly then already the original algorithm achieves an approximation factor of around 1,01. A summary in form of a table and different values of b could be useful here. 

- Also, in the experimental evaluation it is not exactly clear how the setup is precisely done, as the value b is not part of the description but only implicitly mentioned as being an effect of increasing values of b. I would have preferred an analysis as in the appendix, section E, where the value of b gradually increases, or even an additional plot with b on the x-axis and the corresponding runtime (increase) on the y-axis.

- In Figure 2, for some of the greedy plots we have at some point straight lines, though b increases. Though the explanation, written in the section regarding question 2 is also making sense, I am still surprised that no further decrease in cost is being observed, even if much more representatives are allowed and costs are greedily decreased.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper considers a learning-augmented variant of the classical metric minimum spanning tree (MST) problem. The authors study a setting proposed by Veldt et al. (2025), where the algorithm has a-priori access to an initial forest with the goal to augment the initial forest to a spanning tree with the objective of minimising the total edge weight. This problem is called metric forest completion (MFC). Since both MST and MFC are known to require a running time of $\Omega(n^2)$, the goal is to find approximation algorithms with a lower running time. The initial forest can be interpreted as a prediction of an MST, making MFC a learning-augmented problem. If there exists an MST that contains the initial forest, then optimally augmenting the initial forest leads to the computation of an MST. In general, the authors define a function $\gamma$ which measures "how close" the initial forest is to being augmentable into an MST.

As a main result, the authors give a $2$-approximation for a MFC, which implies a $2\gamma$-approximation for MST, with running time $o(n^2)$ if the number of components in the initial forest is $o(n)$. The algorithm is a generalisation of the algorithm by Veldt et al., which arbitrarily choses one representative from each component of the initial forest and finds the optimal augmentation that only uses edges incident to at least one representative. If the number of components is $o(n)$, then this algorithm yields a running time $o(n^2)$. The algorithm in this paper is based on the same idea but allows multiple representatives per component within a certain budget. As a first main result, the authors show that the algorithm is a $2$-approximation for a MFC and a $2\gamma$-approximation for MST, for a fixed arbitrary set of representatives.  This improves upon the guarantees of $2.62$ and $2\gamma+1$ by Veldt et al. As the second main result, the authors give a $2$-approximation for the problem of selecting the optimal set of representatives.This approximation first computes $b+1$ (the overall budget for representatives) potential representatives for each component by greedily solving $k$-center clustering instances. Then, it uses a knapsack DP to allocate the budget to the components. Finally, the authors give empirical experiments to compare the performance of the algorithm for different representative budgets and different representative selection methods.

### Strengths
* The proof of Theorem 1 is very clear and simple, and at the same time improves the guarantees by Veldt et al. In my opinion, this is a very nice contribution.
* The empirical experiments nicely illustrate how the choices (budget, selection of representatives) in the algorithmic framework impact performance of the algorithm.
* The paper studies a well-motivated problem. The framework of learning-augmented algorithms is still mainly studied for online problems: According to the online repository, there are more than 170 paper studying online problems and only 30 studying running time improvements. Further work on learning-augmented settings to improve running times is an important contribution.

### Weaknesses
* The impact of allowing more representatives seems to be mainly studied empirically. It would be nice to have a discussion of theoretical results depending on the budget.
* The new analysis of the algorithm does not require many new technical ideas.
* For the sake of transparency, it would be nice to already mention in the introduction that the running time depends on the number of components.

### Questions
* Do you think there is hope for theoretical results that quantify the impact of allowing a larger budget for representatives?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper considers MST with far fewer than $n^2$ distance queries via the Metric Forest Completion framework, generalizing the one-representative scheme to MultiRepMFC (multiple reps per component) and considering only edges touching reps; it proves an instance-specific bound $\alpha=1+\mathrm{cost}(P,R)/w(E_t)$, yielding an $\alpha$-approx for MFC and $\alpha\cdot \gamma$-approx for MST, which in particular improves the classic one-rep worst case from 2.62 to 2 for MFC and $(2\gamma+1)$ to $2\gamma$ for MST with a matching tight example; representatives are chosen via a shared-budget multi-instance $k$-center with a 2-approx.

### Strengths
1. Firstly, I appreciate that the paper is well-written and well-organized; I enjoy reading it. The authors provide sufficient intuition for each main theorem, making it easy to understand the main idea behind the improved algorithm.

2. This paper considers an interesting problem and uses a simple idea to get an improved algorithm. The algorithm is easy to implement, and thus, I expect it to have a positive impact in practice.

3. Casting representative selection as a shared-budget multi-instance k-center and giving a 2-approx (Gonzalez curves + DP allocation) is neat. This algorithm is simple but looks interesting to me, and it may be helpful for other related problems.

### Weaknesses
1. From the theoretical view, this work looks incremental. The idea of increasing the size of the representative set is standard and appears in many other classical problems (e.g., k-means++ and greedy k-means++). This is a weakness from a purely technical angle, but borrowing techniques from one problem to create a better approximation algorithm sounds like a good approach to me, especially for a non-theory conference.

2. Computing this larger representative set looks time-expensive to me, but the final running time is still in subquadratic time. Since several subquadratic Euclidean MST methods exist, a brief comparison explaining when your algorithm, MultiRepMFC, is preferable would help situate the work. I think it would be better to have a small table to compare distance query counts per algorithm variant or compare their precise running times.

### Questions
Are there conditional lower bounds (e.g., SETH-style or decision-tree/query-complexity) for achieving <2 for MFC with $o(n^2)$ queries?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper considers the learning augmented minimum spanning tree (MST) problem for points in metric space. The authors propose solving this using the metric forest completion problem (MFC), where given an input of forest, goal is to find edges to form a full spanning tree, where the forest can be considered to be provided by an oracle (the learning augmented model). Previous work shows $\Omega(n^2)$ runtime is necessary for optimally solving MFC, and gives a 2.62 approximate algorithm. This works extends the previous algorithm and obtains 2 approximation for MFC and $(2\gamma + 1)$-approximation for learning augmented MST, where $\gamma$ measures how good the input forest is w.r.t optimal MST.

### Strengths
The previous work solved the MFC problem by considering one representative point per partition (in the forest), and then finding the set of edges minimizing the cost. The extension in this work is to instead consider multiple points in each partition. The problem of finding the best such set of points across different partitions is formulated as a variant of the $k-$center problem. The authors then show an extension of the classical 2 approximation $k-$center algorithm for this problem. This in turn, helps improve the approximation ratio to 2. Experiments show a few additional representatives help obtain much better solutions.

### Weaknesses
The idea of using multiple representatives per partition is a natural one. The analysis is clean but the algorithmic contributions are incremental. The authors argue for tightness of their result, but in the hard instance, the number of representative points per partition is 1.

### Questions
Can the authors provide hard instance for choosing multiple points arbitrarily per partition and the best possible approximation ratio for this?

### Soundness
3

### Presentation
3

### Contribution
2
