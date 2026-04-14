# How to Find the Exact Pareto Front for Multi-Objective MDPs?

- Decision: Accept (Spotlight)
- Scores: 8, 6, 8, 8, 8, 6, 8

## Abstract
Multi-Objective Markov Decision Processes (MO-MDPs) are receiving increasing attention, as real-world decision-making problems often involve conflicting objectives that cannot be addressed by a single-objective MDP. 
The Pareto front identifies the set of policies that cannot be dominated, providing a foundation for finding Pareto optimal solutions that can efficiently adapt to various preferences.
However, finding the Pareto front is a highly challenging problem. Most existing methods either (i) rely on traversing the *continuous preference space*, which is impractical and results in approximations that are difficult to evaluate against the true Pareto front, or (ii) focus solely on deterministic Pareto optimal policies, from which there are no known techniques to characterize the full Pareto front. Moreover, finding the structure of the Pareto front itself remains unclear even in the context of dynamic programming, where the MDP is fully known in advance.
In this work, we address the challenge of efficiently discovering the Pareto front, involving both deterministic and stochastic Pareto optimal policies.
By investigating the geometric structure of the Pareto front in MO-MDPs, we uncover a key property: the Pareto front is on the boundary of a convex polytope whose vertices all correspond to deterministic policies, and neighboring vertices of the Pareto front differ by only one state-action pair of the deterministic policy, almost surely.
This insight transforms the global comparison across all policies into a localized search among deterministic policies that differ by only one state-action pair, drastically reducing the complexity of searching for the exact Pareto front. 
We develop an efficient algorithm that identifies the vertices of the Pareto front by solving a single-objective MDP only once and then traversing the edges of the Pareto front, making it more efficient than existing methods. Furthermore, the entire Pareto front can be found in $V$ iterations, where $V$ represents the number of vertices on the Pareto front.
Our empirical studies demonstrate the effectiveness of our theoretical strategy in discovering the Pareto front efficiently.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper is concerned with characterizing optimal Pareto front and a corresponding algorithm to efficiently build the optimal Pareto front for Multi-Objective MDPs. In contrast with the existing works using preference vector and traversing continuous space, this paper characterizes Pareto front using deterministic policies, and elucidating the relationship between neighboring vertices. The authors present the relevant algorithms to extract Pareto optimal faces. The numerical experiments and detailed proofs were provided.

### Strengths
I really like the presentation of this paper. Sections 4.1, 4.2, and 4.3 sequentially validate the algorithm in a very clear way. The figures help understand the characterization of the vertices and faces, and the algorithm description is very clear. 
The idea of identifying the vertices as deterministic policies through a linear transformation for Multi-Objective MDPs is impressive.
I believe that the concept of traversing the vertices (deterministic policies) is clearly distinguished from the previous works which typically uses approximation while searching continuous spaces.

### Weaknesses
I doubt if the author’s algorithm is really more efficient than the OLS in all cases. Generally, dealing with discrete sets such as the author's approach takes longer than handling continuous sets. I understand that the experiment demonstrates the efficiency of the algorithm, but this was done when S and A are small enough. For example, in Line 496, the author had clarified the complexity of the algorithm which is $O(NS(A-1))$, but if the state/action spaces are large, $S(A-1)$ may be larger than $N$ and it may have a higher complexity than $O(N^2)$. Can you present why the above situation may not be the case? If impossible, I suggest the authors discuss trade-offs between N, S, and A, and preferable scenarios where the algorithm would work well (e.g. propose certain situations where $N$ >> $S(A-1)$ and why).

### Questions
1. Line 905 says that $\Pi$ can contain non-stationary policies, while Line 684 defines $\Pi$ as a set of stationary policies. I am confused about whether the authors only consider stationary policies or not. Can you explain any intentional differences if they exist?

2. What is $L \Pi_D(\mu)$ in Line 749? Is it $\Pi_D (\mu)$? Or does it mean that the vertices are the occupancy measures of a set of policies $\Pi_D (\mu)$? (If it is a separate concept, the authors should define what $L\Pi$ is.) 

3. To recover “full” exact Pareto front, I believe that the “if” statement in Lemma 3 should be modified to “if and only if” (necessary and sufficient). If you intended to do so, can you modify that? (It seems that Line 409 talks about “only if” part, but it is not accurately stated in the lemma.) If not, please explain why the sufficient conditions are enough to prove the following theorems or propositions.
Also, I cannot locate the proof of Lemma 3.

4. I think the point of Lines 10-11 in Algorithm 2 is to include all normal vectors associated with lower-dimensional faces to apply (1). If that’s the case, shouldn’t Line 4 of Algorithm 2 specify that .pop() be done from high-dimensional faces to lower-dimensional faces (or specify that .push() is done in the order of dimension)? If the order matters, please modify the algorithm accordingly and if it doesn't, please explain why this is the case.

5. It would be better to add relevant existing algorithms in the appendix (e.g. Roijers, PPrune algorithm) for the completeness.

6. It would be better to note the corresponding proof locations below all theorems (e.g. Theorem 2, Proposition 1).

7. In the abstract, it would be better to use MO-MDPs inside the first parenthesis instead of MDPs to understand the 12th line of the abstract (I personally didn't get it at first glance).

I am giving the score of 6 for the current draft, but I am eager to talk with the authors and reconsider the score.

### Soundness
4

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
3

### Summary
The paper proposes an exact method for finding Pareto fronts (in turn Pareto optimal policies) for Multi-Objective Markov Decision Processes (MO-MDPs). Leveraging the geometric properties of the Pareto front, the authors demonstrate that it lies on the boundaries of a convex polytope, with neighboring Pareto front vertices differing by at most one state-action pair. The paper provides proofs showing that every deterministic policy on the Pareto front is connected to at least one deterministic Pareto optimal policy, which can be extracted via the convex hull of a Pareto optimal policy and its neighbors. Building on these findings, the paper introduces a search algorithm for Pareto fronts that requires solving each single objective of the MO-MDP once for initialization, with total iterations bounded by the number of Pareto fronts. The proposed algorithm is evaluated on small MDPs with 5 to 8 states, 5 to 7 actions, and 3 objectives. The benchmark used for comparison is the Optimistic Linear Support (OLS) method, which can find Pareto vertices but not the entire Pareto front. Results indicate that the proposed method can find exact Pareto fronts and is more computationally efficient than the OLS method.

### Strengths
* The paper makes a novel theoretical contribution by providing an exact solution method for finding optimal Pareto front in MO-MDPs. The proofs are rigorous, offering valuable insights into the geometric properties of the Pareto front, which could impact other optimization methods for MO-MDPs.
* The method reduces computational complexity compared to OLS, marking a notable advantage in computational efficiency for exact MO-MDP solutions. The iterative nature of the algorithm also suggests it could be more robust and reliable than blind search-based methods.
* The writing is clear and well-structured with good visual presentations.

### Weaknesses
* The practical relevance of finding exact solutions versus approximation is not thoroughly addressed. Since approximate solutions are computationally easier to obtain, the paper could benefit from providing scenarios and discussion where exact solutions are preferred.  Comparisons and discussions against approximation methods, such as those in [1,2], are missing, which could help contextualize the result.
* The experimental evaluation is relatively weak, relying on very small, toy MO-MDPs with only a single benchmark for comparison.  Is there any practical relevance or application of exact solutions to such problems on such small problem sizes?
* The scalability of the algorithm is not adequately discussed. From the results, it seems like running times are negligible for small MDPs, it's unclear how the method performs on larger problems, and when scalability could become an issue. 

[1] Xingchao Liu, Xin Tong, and Qiang Liu. Profiling Pareto front with multi-objective Stein variational gradient descent. Advances in Neural Information Processing Systems, 34:14721–14733, 2021.

[2] Xiaoyuan Zhang, Xi Lin, Bo Xue, Yifan Chen, and Qingfu Zhang. Hypervolume maximization: A geometric view of Pareto set learning. Advances in Neural Information Processing Systems, 36:38902–38929, 2023.

### Questions
Please address the concerns and questions raised in the weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper studies how to find the Pareto front of multi-objective Markov decision processes efficiently.  The authors first show that the Pareto front lies on the boundary of a convex polytope, with vertices representing deterministic policies, and neighboring vertices of the Pareto front differ by only a state-action pair in the deterministic policy. Based on this observation, the authors propose a Pareto front searching algorithm with three key steps: (1) neighbor search; (2) incident faces identification; (3) Pareto face extraction. A major advantage of this method is that the localized search among deterministic policies that differ only one state-action pair significantly reduces the computational burden. The authors also provide some empirical results to show the effectiveness of this proposed method.

### Strengths
- The key idea is clearly presented. Compared with the state of the art for multi-objective Markov decision processes, the authors highlight that the proposed method can find the entire Pareto front. 

- The authors derive a few interesting geometric properties of the vector value function: (1) distance one property; (2) sufficiency of traversing over edges; (3) locality property of the Pareto front. I believe this is the main theoretical contribution to the literature. The authors also show how these properties are used to find the entire Pareto front, which is new to me. 
 
- Based on the geometric properties of the Pareto front, the authors develop an efficient Pareto search algorithm. This algorithm initiates with a neighbor search of a deterministic policy, computes incident faces for a convex hull, and then extracts the Pareto front.

- The authors evaluate the performance of the proposed method in an experiment, showing better runtime compared to benchmark.

### Weaknesses
- A key contribution is to find the full exact Pareto front for multi-objective MDPs. Although it is of theoretical interest, it is less discussed the practical use of the entire Pareto front, such as multi-objective learning. It would be helpful if the authors could provide some examples that warrant the entire Pareto front. 

- The authors first introduce the algorithm and then the key properties. Without reading Section 4, it is difficult to understand why Algorithm 1 works. It would be helpful to reverse the order of Section 3 and Section 4. 

- The convergence study of Algorithm 1 is not provided. It seems that it is not justified that Algorithm 1 can find the entire Pareto front. Also, the stop criterion and computational complexity are not analyzed.  

- The scalability of Algorithm 1 is not analyzed regarding the size of MDP and dimension of reward. 

- Regarding the properties of the vector value function, it would be helpful if the authors could provide some examples or intuitions behind them. 

- The current analysis seems to be limited to finite MDPs. It would be useful to extending it to MDPs with continuous spaces.

- The example in experiments is artificial. It would be useful if the authors could provide some practical use cases.

### Questions
- The authors have focused on identifying deterministic policies in the Pareto front. What are specific reasons not considering the whole policy space including stochastic policies? 

- What are assumptions about MDPs made in analysis (e.g. finite state/action spaces, ergodicity, etc.)? Can the authors state them in theorems? 

- How sensitive of Algorithm 1 to the accuracy (e.g., errors or approximations in these subroutines) of calling subroutines like line 11, 12, and 13? 

- Can the authors provide some use examples of the entire Pareto front?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper investigates the geometric structure of the Pareto front in multi-objective MDPs and reveals key properties, including that the vertices on the Pareto front correspond to deterministic policies and that neighboring vertices differ by only one state-action pair, almost surely. 
The paper also presents an efficient algorithm that leverages these geometric properties to find the Pareto front, which could reduces the computational complexity compared to existing methods.
The empirical studies demonstrate the proposed algorithm maintains a much better run time than Optimistic Linear Support (OLS).

### Strengths
The paper introduces an original contribution to the field by uncovering key geometric properties of the Pareto front in multi-objective Markov Decision Processes.
The paper has good quality, and is supported by theoretical proofs and empirical evidence.
The paper provides a clear explanation of the geometric properties of the Pareto front and a clear demonstration of the proposed algorithm. The paper should ideally explain the empirical setup in more detail.
The paper makes a significant contribution to solving multi-objective MDP problems by offering insights into the geometry of the Pareto front and developing an efficient algorithm for finding the exact Pareto front.

### Weaknesses
In the evaluation of the algorithm, more details could be provided about the experiment setting to give context to the results. For instance, specifying the states, actions, transition kernel, reward function, and the number of time steps considered. Also, including a few more benchmark algorithms would help demonstrate the efficiency of the proposed algorithm more effectively.

### Questions
The evaluation of the algorithm demonstrates its efficiency with state space sizes of 5 and 8, action spaces ranging from 5 to 7, and a reward dimension of 3. If the state and action space cardinalities were significantly larger, would this algorithm remain more efficient compared to others? Specifically, could sampling the preference space be more efficient in certain cases?

Can the algorithm be adapted for use in continuous action spaces?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors provide an algorithm for finding the entire Pareto-optimal front for multi-objective MDPs. Their algorithm is based upon several structural/theoretical observations about the geometry of the Pareto front, relating neighboring vertices of the Pareto front to policies which differ in a single state-action pair. This leads to their local-search-based algorithm, which outperforms baselines and prior approaches.

### Strengths
I think the paper is written very clearly. In particular the proof sketches are structured in a helpful manner.
The paper makes good use of examples to explain technical points of the algorithm (ex. paragraph around 407).

The paper also seems to make a original and significant contribution to the study of multi-objective MDPs, not just outperforming prior work but also developing a fundamentally different algorithmic approach and presenting useful theoretical/structural results.

### Weaknesses
In other settings the assumption of ergodicity is a significant one. I think this should be highlighted a bit more.

This is subjective but I think the organization of section 3 would be improved if the problem setup (Section 3.2) was provided before the algorithm overview (Section 3.1). (I still think the algorithm overview is very helpful.)

I think the wording of the definition of ergodicity should be clarified. The provided definition sounds similar to the definition of a communicating MDP.

The plots in Figures 5 and 6 are hard to understand due to overlapping curves. I think changing the y-axis scale might be a better solution than the current approach. Also I think the caption of Figure 6 is incorrect? (Should be OLS not benchmark)

### Questions
Could you give an overview of the difficulties in generalizing your results beyond the ergodicity assumption and describe how it is used in the proofs?

Regarding the benchmark algorithm in Section 5.1, are there situations where the benchmark algorithm has comparable performance to the main algorithm, and what would they look like?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 6

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies the multi-objective MDPs and explores the properties of their Pareto front. By using the insight that the Pareto front is on the boundary of a convex polytope where each vertex is a deterministic policy, a fast algorithm is designed to compute the Pareto front efficiently.

### Strengths
- This paper is well written.

- The idea is interesting and novel, and the results are significant.

### Weaknesses
- The complexity analysis lacks clarity, particularly when compared to other algorithms. A more explicit comparison would significantly improve the discussion.

- The numerical evaluation is restricted to small instances, which are insufficient to demonstrate the scalability of the proposed algorithm.

### Questions
see weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 7

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents an efficient method to identify the exact Pareto front in multi-objective Markov Decision Processes (MO-MDPs). The authors use the geometric insight that the Pareto front forms on a polytope boundary, allowing for a simplified, localized search by adjusting one state-action pair at a time. This approach keeps computations efficient and complete.

### Strengths
1. The paper takes an original approach to MO-MDPs, revealing geometric properties of the Pareto front that support an efficient search algorithm, standing apart from previous methods that rely on scalarizing preferences or using approximations.
2. The algorithm provides a strong computational edge, needing only one single-objective solution to start, then systematically traversing edges between Pareto-optimal policies—ideal for applications requiring efficient navigation of complex Pareto fronts.

### Weaknesses
1. The complexity analysis could benefit from a more detailed comparison with existing algorithms, especially in larger, higher-dimensional state-action spaces. Testing on bigger benchmarks would clarify the algorithm's scalability and relevance for complex MO-MDPs.
2. The experiments use small, artificial setups. Adding tests in larger, more realistic MO-MDP scenarios would better show the algorithm’s practical applicability and effectiveness.

### Questions
Could the authors elaborate on potential modifications to improve the algorithm's scalability with larger action and state spaces?

### Soundness
3

### Presentation
4

### Contribution
3
