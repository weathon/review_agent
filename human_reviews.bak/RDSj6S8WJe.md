# Demystifying Linear MDPs and Novel Dynamics Aggregation Framework

- Decision: Accept (poster)
- Scores: 6, 6, 6, 5, 6

## Abstract
In this work, we prove that, in linear MDPs, the feature dimension $d$ is lower bounded by $S/U$ in order to aptly represent transition probabilities, where $S$ is the size of the state space and $U$ is the maximum size of directly reachable states.
Hence, $d$ can still scale with $S$ depending on the direct reachability of the environment.  To address this limitation of linear MDPs, we propose a novel structural aggregation framework based on dynamics, named as the *dynamics aggregation*.
    For this newly proposed framework,
    we design a provably efficient hierarchical reinforcement learning algorithm in linear function approximation that leverages aggregated sub-structures. Our proposed algorithm exhibits statistical efficiency, achieving a regret of $\tilde{O} \big( d_{\psi}^{3/2} H^{3/2}\sqrt{ NT} \big)$, where $d_{\psi}$ represents the feature dimension of *aggregated subMDPs* and $N$ signifies the number of aggregated subMDPs. 
    We establish that the condition $d_{\psi}^3 N \ll d^{3}$ is readily met in most real-world environments with hierarchical structures, enabling a substantial improvement in the regret bound compared to LSVI-UCB, which enjoys a regret of $\tilde{O}(d^{3/2} H^{3/2} \sqrt{ T})$.
    To the best of our knowledge, this work presents the first HRL algorithm with linear function approximation that offers provable guarantees.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this paper, the authors prove a lower bound of $d$ for the linear MDP to aptly represent the transition probability. Therefore, they claim that linear MDPs may have regret guarantees dependent of the state space. To address the issue, they propose a novel structural aggregation framework based on dynamics, named as the dynamics aggregation. For this framework, they design a provably efficient hierarchical reinforcement learning algorithm and provide a regret upper bound for this algorithm.

### Strengths
1. The problem of proving lower bounds for $d$ and considering hierarchical structure is very interesting and important.

2. The paper is solid, the proof looks correct to me. 

3. The lower bound is meaningful, demonstrating the limitations of the linear MDP. 

4. The presentation is clear in general. The simulation is interesting.

### Weaknesses
My main concern is about technical novelty. The dynamics aggregation is very similar to the misspecified linear MDP considered in [1]. Also, the algorithm is adapted from LSVI-UCB. The result can be expected given [1]. 

[1] Chi Jin, Zhuoran Yang, Zhaoran Wang, and Michael I Jordan. Provably efficient reinforcement learning with linear function approximation.

### Questions
Please refer to the weakness section. Generally speaking, I lean towards acceptance of this paper because of the interesting lower bound and its meaningful insights.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper provides a new perspective into the low-dimensional representation structures of MDPs. On the one hand, it casts reasonable doubt on the popular linear representation structure via a simple lower bound on the feature dimension $d$, showing that $d$ may actually scale up with $S$ when the direct reachability $U$ of the environment is limited. On the other hand, it proposes a novel dynamics-based hierarchical aggregation framework that leverages *known* mappings to aggregated sub-MDPs (each equipped with linear representation structure), and shows that it achieves a competitive regret *under certain assumptions*.

### Strengths
1. This paper provides a new angle for researchers to understand the fundamental limit of linear MDPs. The result by itself is technically simple and straightforward, but the valuable part of it is the motivation it provides to reflect upon a popular modelling option that is potentially subject to implementation issues.
2. The flow and writing of this paper is good. It provides the reader with adequate background knowledge, illustrates the key points clearly with concrete examples and figures, and accompanies the main results with intuitions and discussions.
3. The mathematical proofs seem correct to the reviewer in the form they are presented in the paper, though results cited from literature are taken for granted.

### Weaknesses
1. The authors claim that the dynamics-based aggregation framework proposed in the second part of the paper *addresses the limitation of linear MDPs*. The reviewer is skeptical about the contribution, in that:
    * The aggregation framework seems very artificial. There is not enough motivation why people would have the aggregation mapping $\psi^{i \to (n)}$ in hand *a priori*. Specifically, why don't people directly consider the aggregated MDP when they model real-world scenarios, but rather introduce a large-scale MDP and identifies the similarity between (unnecessarily differentiated) states in the meantime?
    * Apart from the novel idea of substructures, the contribution of the second part seems minimal to the reviewer since it looks like a simple application of LSVI-UCB in sub-MDPs. The proof structure is also similar to that of LSVI-UCB with minor changes.
    * The results will be very interesting if the aggregation structure can be learned (either online or offline) rather than given, but reviewer fails to come up with a quick fix that enables such learning. The reviewer would be more positive about this paper if the authors can, at least, illustrate a potential algorithm design idea to learn the structure.
2. The comparison against LSVI-UCB seems sketchy. The authors claim in the abstract that $d_{\psi}^3 N \ll d^3$ is "readily met in real-world environments", but the only discussion about this is a few conjectured inequalities on page 9 without any real-world data. This seems like too much overclaiming to the reviewer, and thus the authors are urged to provide real-world evidence for their claim.
3. The experiment design can be improved in the following ways:
    * The source code to reproduce the results is not publicly available.
    * The environment is designed to be tabular, which is reducible to linear MDPs, but only in a very inefficient way. The comparison is therefore unfair. It would be more convincing if the algorithms can be evaluated and compared in MDPs with intrinsic low-dimensional structures.
    * The MDPs used in the experiment are very small in size. Experiments are expected to, at least, show adequate scalability of the algorithm.

### Questions
1. Why would people have the aggregation mapping $\psi^{i \to (n)}$ in hand *a priori* in modelling?
2. In what kind of real-world environments would the condition $d_{\psi}^3 N \ll d^3$ be met? Please provide concrete examples.
3. In Algorithm 1, $n$ and $i$ seems to be variables that automatically get their values upon observation of states. Should it be written in a clearer way to show that $n$ and $i$ are actually calculated from the state?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper provides two interesting contributions:

1) It shows a lower bound on the feature dimension in linear MDPs which depends on the inverse of the maximum reachability of the environment.

2) It provides a novel algorithm with sublinear regret for hierarchical RL where each of the subMDP is a linear MDP.

### Strengths
I think that (albeit simple and potentially expected) the lower bound on the feature dimension of Linear MDP is an important result for the RL theory community.

The algorithm (UC-HRL) seems to be an interesting and novel contribution to hierarchical RL.

### Weaknesses
The regret bound in Theorem 2 has a linear term which can be made sublinear only if $\epsilon_P$ in Definition 4 is of order $\mathcal{O}(1 / poly(T))$. However it is not very clear from the paper how big $\epsilon_P$ can be for common choices of the approximate feature aggregation mappings $\psi$.

The fact that the aggregating functions $\psi$ are required to be known in advance seems rather strong but somehow common in hierarchical RL.

The discussion after Theorem 2 that justifies that $d^3_{\psi} N \leq d^3$ is unclear in my opinion in particular I do not understand why the regime $MN \leq S$ and $M^2 \leq S^2/U^3$ is a reasonable one. Maybe the author should consider expanding this discussion in their revision.

### Questions
1) Can you provide an example of aggregating functions $\psi$ such that the error $\epsilon_P \leq 1/\sqrt{T}$ ?

2) Another related question is: do you expect $\epsilon_P$ to decrease as the number of subMDP $N$ increase ? Is it possible in this way to find the value of $N$ which minimizes the regret bound? 

3) Could add an example of Linear MDP where the conditions $MN \leq S$ and $M^2 \leq S^2/U^3$ hold and therefore the hierarchical algorithm has a clear advantage ?

4) Can the assumption of known aggregating functions $\psi$ be relaxed ?

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper focuses on the problem in the linear Markov decision process and its linear representation to the transition probability kernel. It first shows that the current regret results form the literature has the dependency on the state cardinality, which comes from the fact that the dimension of the linear representation for the transition kernel is lower bounded by the rank of the matrix. Then it leverages the technique from the previous works on state aggregation and group mapping to propose a hierarchical version of the linear MDP algorithm, reducing the final regret dependency on the state cardinality to the grouped MDP dimension.

### Strengths
1. The paper shows that the dimension of the linear representation for probability transition kernel is lower bounded by |S|/|U|, where |S| is the cardinality of the states and |U| is the maximum size of directly reachable states. If |U| is not the order of |S|, the regret would depend on the state cardinality.
2. The paper develops a hierarchical linear MDP algorithm to reduce the state cardinality dependency in the final regret. It leverages the internal structure of the problem with the state aggregation and mapping from previous works. The final regret and examples show the effect of it.

### Weaknesses
1. The paper makes stronger assumption than previous linear MDP algorithms. For the sub-structure that is explored by the paper, it assumes that the dynamic aggregation is known and has the desired boundedness in Definition 4. 
2. For the final regret proven by the paper, although the regret seems to be improved in terms of the state cardinality theoretically, it also introduces another T-dependent term characterizing the aggregation gap w.r.t the original probability transition kernel. It's not clear to me whether the newly introduced term would cancel out the improvement from the first term in Theorem 2.
3. The algorithm seems to be a direct extension of the previous works in the tabular case by adding in linear representation and similar analysis.

### Questions
The assumption of known dynamic aggregation is strong to me. In reality, how could we extract such information without knowing the transition kernel?

============= After rebuttal  ============
I really appreciate the author/s effort in addressing my concerns and questions. After checking the author/s response and other reviewers' comments, I would like to increase my score from 3 to 5, since I am still not totally onboard with the additional stronger assumptions about the dynamic aggregation and the very similar algorithmic design as previous works.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 5

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
Recent advancements in reinforcement learning (RL) have spotlighted function approximation to address the generalization challenges in tabular Markov Decision Processes (MDPs). Linear MDP, a cornerstone model, has demonstrated that regret bounds are influenced more by feature dimension rather than state space size. However, the authenticity of this claim is examined in this paper. Researchers found that for appropriate representation of the probability space, the feature dimension is inevitably influenced by the size of the state space. A discrepancy was observed in the relationship between the feature dimension and state space size, especially as the latter expands. It's concluded that linear MDPs might not inherently allow learning detached from state space size. To counter this, the paper presents a new hierarchical framework called dynamics aggregation. It encompasses both state aggregation and equivalence mapping, promoting efficiency and adaptability. A hierarchical reinforcement learning (HRL) algorithm is proposed within this structure, which is statistically efficient and offers a comprehensive regret bound. The algorithm is validated against existing methods, showcasing its superior performance.

### Strengths
1. The paper questions the widely accepted belief about linear MDPs, delivering a comprehensive critique of its fundamentals.
  
2. The new framework, which fuses state aggregation and equivalence mapping, holds promise for addressing the limitations of linear MDPs, making it a significant contribution to the field.

3. The proposed HRL algorithm not only introduces an innovative approach to RL but is the first of its kind to provide proven guarantees in function approximation.

4. The inclusion of numerical experiments fortifies the theoretical claims, showcasing the algorithm's efficacy against existing counterparts.

### Weaknesses
1. While the new algorithm excels in controlled experiments, its scalability and performance in more complex, real-world scenarios are yet to be determined.

2. While numerical experiments are conducted, this paper mentions several examples in section 4 but does not include experiments and analysis in these examples.

### Questions
1.For the proof provided for Theorem 1: suppose that there exists a state-action pair$(s, a)$ for which the transition probabilities are non-zero for more than $U$ states. How would this affect the recursive logic applied in the derivation of $\operatorname{rank}(\mathbb{P}_h) \geqslant\lfloor S / U\rfloor$? Would the derived relationship between $d$, the rank of $\mathbb{P}_h$, and the relationship $\lfloor S / U\rfloor$ still hold? 

2. Feature Dimension vs. State Space Size: Given that the research reveals a deeper connection between feature dimension and state space size than previously thought. What is potential future work especially in contexts where the state space is vast?

3. Hierarchical Structures in Real-world Scenarios: With the proposed dynamics aggregation framework depending heavily on the hierarchical structure of problems, how feasible is it to identify or establish such hierarchies in complex, real-world scenarios, where the state dynamics might be more intricate and less structured?

4. This paper mentions several examples in section 4. How does this method work and does it perform well in these examples?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
