# Realizable Abstractions: Near-Optimal Hierarchical Reinforcement Learning

- Avg Score: 6.25
- Decision: Reject
- Scores: 8, 3, 6, 8

## Abstract
The main focus of Hierarchical Reinforcement Learning (HRL) is studying how large Markov Decision Processes (MDPs) can be more efficiently solved when addressed in a modular way, by combining partial solutions computed for smaller subtasks.
Despite their very intuitive role for learning, most notions of MDP abstractions proposed in the HRL literature have limited expressive power or do not possess formal efficiency guarantees.

This work addresses these fundamental issues by defining Realizable Abstractions, a new relation between generic low-level MDPs and their associated high-level decision processes.
The notion we propose avoids non-Markovianity issues and has desirable near-optimality guarantees.
Indeed, we show that any abstract policy for Realizable Abstractions can be translated into near-optimal policies for the low-level MDP, through a suitable composition of options.
As demonstrated in the paper, these options can be expressed as solutions of specific constrained MDPs.
Based on these findings, we propose RARL, a new HRL algorithm that returns compositional and near-optimal low-level policies, taking advantage of the Realizable Abstraction given in the input.
We show that RARL is Probably Approximately Correct, it converges in a polynomial number of samples, and it is robust to inaccuracies in the abstraction.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper studies hierarchical MDP. To characterize how close the abstraction states to the ground truth, they proposed the idea of realizable abstractions. They further show that under this condition, the values of the abstract MDPs and ground MDP are close to each other.

Based on these properties, they proposed a new algorithm for hierarchical MDPs, and they demonstrate sample efficiency for the algorithm.

### Strengths
1. The setting of this paper is clear and the paper is well written. 

2. The algorithm proposed in this paper is sample and computational efficient.

3. The idea of realizable abstraction is natural and useful for characterizing the sample complexity of the algorithm.

### Weaknesses
1. This paper proposes an algorithm for hierarchical RL. It will be better if there is some numerical experiment which demonstrate that the hierarchical RL algorithm performs better than the normal RL algorithm in some specific domain.

### Questions
1. The author proposes the condition of realizable abstraction and the sample complexity of the algorithm depends on this condition. Is there any way to identify an abstraction which satisfies this condition?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper explores HRL and introduces a framework called Realizable Abstractions, which aims to improve the efficiency of solving large MDPs by breaking them into modular components. It defines a formal relationship between high-level (abstract) and low-level (ground) values and specifies conditions required to reduce the effective planning horizon within these abstractions. Furthermore, the paper presents a sample-efficient algorithm that operates based on the assumption of an admissible and realizable abstraction.

### Strengths
Realizable Abstractions offer a fresh theoretical foundation for HRL, opening up a promising way for potentially reducing sample complexity in reinforcement learning. I find it particularly intriguing that sparse rewards play a crucial role in ensuring admissibility (Proposition 6).

### Weaknesses
While the new concept of Realizable Abstractions is interesting, the paper lacks some key definitions, making it challenging to follow. As a result, it’s difficult to fully grasp the significance of the paper’s main contributions.

The followings are main weaknesses of this paper:

- The proposed algorithm requires a **known** abstraction and assumes admissibility, which feels like a rather strong assumption. However, there is not enough rigorous discussion regarding these assumptions and their implications.
It is also unclear how stringent these assumptions are compared to other assumptions, such as known state abstraction (Abel, 2020) or known equivalence mapping (Wen et al., 2020).

- Exploration, which is crucial yet challenging to design in RL, relies heavily on the admissibility assumption. Without this assumption, it is unclear how an optimistic policy could be constructed either in practice or theoretically.

- In Algorithm 1, functions like REALIZER, VALUEITERATION, ROLLOUT, and the algorithm $\mathfrak{A}$  are not clearly defined. They should be formally described, perhaps in pseudocode, to improve clarity and ensure precise understanding.


- In Theorem 7, the sample complexity scales with 
$\bar{A}$, but I could not find a formal definition for $\bar{A}$ in the paper. I guess it refers to the cardinality of the set of action sequences, which is typically much larger than $A$. If this is the case, it’s unclear whether this approach actually improves regret.

**Minor**
- Line 188: The sentence "Any set of options ..." appears incomplete.

### Questions
- Line 186: What is the definition of "relevant block"? This term is not clearly defined in the paper.

- Line 239: Could you clarify the statement, "For this reason, we only use 2-MDPs to
represent the abstract MDP, and never a k-MDP with k>2."? The reasoning here is unclear to me.

- Line 246: What is the formal definition of a "new absorbing state"? Additionally, why is this state necessary for defining $\mathcal{S}_{\bar{s}}$?

- Line 265: Could you provide the definition of $\bar{\gamma}$ in Equation (2) and (3)? Why does the inequality $\bar{\gamma} \leq \gamma$ hold? 
Furthermore, could you give a clear definition of  $\bar{\mathcal{A}}$?

- Line 403: How are the constraints in Equation (7) derived?



I will consider raising the score once my concerns and questions have been addressed.

### Soundness
2

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
This paper proposes Realizable Abstractions as a method of relating low-level MDPs to high level abstractions, in particular ones that are suitable for hierarchical reinforcement learning.
The approach provides near-optimality guarantees for low-level options which realize higher-level abstracted behavior and are the solutions to specially constructed constrained MDPs.
A PAC algorithm is presented which is modular with respect to a PAC-Safe online learning algorithm which is used as a subroutine.

### Strengths
The paper addresses an important problem in hierarchical RL dealing with abstractions which relate high-level and low-level representations.

Connecting hierarchical abstractions to constrained MDPs such that options can be extracted by solving the CMDPs with off-the-shelf algorithms is interesting and, to my knowledge, novel.

The paper is well-written and the intuitive explanations for the theory are fairly easy to follow, though it is extremely notation-heavy.

### Weaknesses
While an algorithm is proposed (RARL), there are no empirical results to support it and validate the assumptions that are made for the guarantees in Section 4. I would like to see RARL compared with existing methods, e.g., some form of option-critic (with specified options) or deep skill chaining (for a skill discovery comparison). Additionally, as it is not clear to me how reasonable Assumptions 1-3 are, the paper would be strengthened by experiments showing how RARL is affected by violations of those assumptions.

### Questions
Can you provide examples of algorithms for which Assumption 1 holds? What properties need to hold for such an algorithm?

How might one construct abstractions suitable for RARL ensuring that Assumptions 2 and 3 hold (or verify that they hold for a given abstraction)? Related to the experiment described above, how do things change with as violations of the assumptions become larger? Does RARL respond gracefully?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The goal of HRL is to replace an MDP with another simpler decision process, solve this MDP abstraction and from this solution, reconstruct a solution in the base MDP. The authors are motivated and propose a solution to what they identify as shortcommings of the currently proposed HRL schemes: first the MDP abstraction itself is often non-Markovian, and second there is rarely any theoretical guarantee on the optimality of the policy reconstructed from an optimal policy on the MDP abstraction.
The authors consider a type of MDP abstraction they call "realisable abstraction" that supposes any abstract action has a "local" realisation with similar occupancy measure and value. By compositionality, this is then shown (theorem 1) to imply a similar "global" statement in the form of a bound on the difference between the values of an abstract policy and its realisation.
The authors then present an HRL algorithm that is shown to PAC learn the optimal policy under some assumptions.

### Strengths
Very clear, very well written.

### Weaknesses
I appreciated the example illustrated in figure 1. I would suggest the authors refer to it when introducing new concepts and when discussing RARL.

### Questions
I would like to see how the realisability condition compares to the bisimulation relation for MDPs
https://arxiv.org/pdf/1207.4114

### Soundness
4

### Presentation
4

### Contribution
4
