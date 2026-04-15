# TreeDQN: Sample-Efficient Off-Policy Reinforcement Learning for Combinatorial Optimization

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 6

## Abstract
A convenient approach to optimally solving combinatorial optimization tasks is Branch-and-Bound method. The branching heuristic in this method can be learned to solve a large set of similar tasks. The promising results here are achieved by the recently appeared on-policy reinforcement learning (RL) method based on the tree Markov Decision Process (tMDP). To overcome its main disadvantages, namely, very large training time and unstable training, we propose TreeDQN, a sample-efficient off-policy RL method that is trained by optimizing the geometric mean of expected return. To theoretically support the training procedure for our method, we prove the contraction property of the Bellman operator for the tree MDP. As a result, our method requires up to 10 times less training data, performs faster than known on-policy methods on synthetic tasks. Moreover, TreeDQN significantly outperforms the state-of-the-art techniques on a challenging practical task from the ML4CO competition.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
Authors propose an off-policy RL algorithm, TreeDQN, to scale the branch and bound algorithm for combinatorial optimization. The branching nodes are infered with a neural net forward pass. This approach is claimed to be faster than classical solvers such SCIP and CPLX because search trees are smaller with the proposed TreeDQN. To me, the contribution compared to previous work that also apply RL to branch and bound is not clear.

### Strengths
The use of RL for B&B algorithm is well-motivated (second paragraph of the intro). The experimental setup is detailed.

### Weaknesses
I believe the contriubtions are unclear. 

What is the difference between FMCTS and TreeDQN from a convergence point of view ? It seems to me that when you say on line 163 " This method [FMCTS] is sample efficient since training data can be sampled from a buffer of past experiences. However, it may not converge to the optimal policy because its training data was obtained by older and less efficient versions of the Q-function ", this aso applies to TreeDQN despite proving contraction of operators. 

Furthermore, I think Tree MDPs can be rewritten as classical MDPs and thus one can just use the classical Bellman Operators to do Q-learning. Can't you just say your state space the set of all sub-MILPs coupled with the current tree depth, your actions are the set of branching node and the reward is -1 ? I am not sure why one needs to define tree MDPs and prove contractions of tree operators.

I am very troubled by figure 1. In particular, I am not convinced that TreeDQN actually learns anything: it seems to me that solutions found at initialization of TreeDQN are already better than those of FMCTS. 

Furthermore, why can't you add tree sizes of solvers such as SCIP or CPLEX on your figure 1 ? This would be easier to compare your results with those of FMCTS (figure 2 of Etheve 2020).

You should add parenthesis to your citations in your introduction please (e.g. lines 50-54).

### Questions
Is it really necessary to define and prove results about a new Bellman Operator?

What is the difference between FMCTS and TreeDQN aside from the loss to optmize?

Why not include the tree sizes of SCIP or CPLX in your figure 1?

Are you sure TreeDQN is actually learning? 

Thank you in advance

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper studies RL-guided Branch-and-Bound (B&B) method for Mixed Integer Linear Programs. Specifically, they modeled the procedure of B&B (select which integer variable to do splitting) as a tree MDP which is first proposed by [Scavuzzo et al., 2022]. They proposed a deep q-learning algorithm for the tree MDP, which they named as TreeDQN to guide the variable selection of B&B. Experiments on both synthetic and practical tasks are conducted to show the effectiveness of TreeDQN.

### Strengths
1. They did experiments on both synthetic and practical mixed integer linear programming tasks to show the effectiveness of their proposed algorithm.

### Weaknesses
The theoretical basis seems not correct to me in this paper which is a big weakness.
1. The definition of value function (2) is not correct. First, the value function (excluding optimal value function) should be policy-dependent. In (2), on the lefthand side, it is policy-independent while on the righthand side, action $a_t$ appears suddenly. Second, the value function shouldn't depend on the node selection strategy since we model the problem as a tree MDP instead of a temporal MDP based on the definition of tree MDP in [Scavuzzo et al., 2022]. Therefore, all the following analysis based on the value function is also under question.
2. The definition of 'contraction in mean' sounds weird to me. What does it mean if an operator is stochastic? And 'contraction in mean' does not guarantee that under this operator, the value function can converge with high probability.
3. The contraction property can be a justification for value iteration methods. However, using it as theoretical backing for q-learning methods is fragile.

### Questions
As I have mentioned in the weaknesses part, why does the node selection strategy matter in value function?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces TreeDQN, an off-policy RL method that enhances the Branch-and-Bound approach for combinatorial optimization by learning efficient branching heuristics. With a proven Bellman contraction for stable training, TreeDQN requires up to 10 times less data and is faster than on-policy methods, achieving superior performance on both synthetic tasks and a challenging ML4CO competition task.

### Strengths
1. The paper presents TreeDQN, a novel sample-efficient off-policy RL algorithm designed specifically for combinatorial optimization, which addresses the limitations of high variance and slow training in existing on-policy methods.


2. Theoretical and Empirical Validation: TreeDQN’s theoretical foundation is strong, backed by the contraction property of the Bellman operator for tree MDPs. Empirical results show substantial improvements, including up to 10 times less training data and superior performance on both synthetic and practical tasks, notably the ML4CO competition task.

### Weaknesses
1. The experimental section primarily compares TreeDQN with basic methods and lacks extensive benchmarking against a wider range of state-of-the-art approaches in combinatorial optimization, which could better illustrate its relative strengths.


2. The method is tailored to similar MILP tasks, potentially limiting its generalizability to significantly different combinatorial optimization problems, which the paper acknowledges but does not address experimentally.

### Questions
1. Could the authors clarify if TreeDQN’s reliance on a contraction property holds under varying conditions or is sensitive to specific task characteristics, such as tree depth or branching factors?

2. What impact does the choice of geometric mean over arithmetic mean have on convergence stability, and are there cases where this choice might be disadvantageous?

3. Could the authors elaborate on how TreeDQN might perform on combinatorial tasks with differing structures, and are there any plans to extend the approach for more varied optimization tasks?

### Soundness
3

### Presentation
3

### Contribution
3
