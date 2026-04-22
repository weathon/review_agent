# Tackling GNARLy Problems: Graph Neural Algorithmic Reasoning Reimagined through Reinforcement Learning

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 6, 2

## Abstract
Neural Algorithmic Reasoning (NAR) is a paradigm that trains neural networks to execute classic algorithms by supervised learning. Despite its successes, important limitations remain: inability to construct valid solutions without post-processing and to reason about multiple correct ones, poor performance on combinatorial NP-hard problems, and inapplicability to problems for which strong algorithms are not yet known. To address these limitations, we reframe the problem of learning algorithm trajectories as a Markov Decision Process, which imposes structure on the solution construction procedure and unlocks the powerful tools of imitation and reinforcement learning (RL). We propose the GNARL framework, encompassing the methodology to translate problem formulations from NAR to RL and a learning architecture suitable for a wide range of graph-based problems. We achieve very high graph accuracy results on several CLRS-30 problems, performance matching or exceeding much narrower NAR approaches for NP-hard problems and, remarkably, applicability even when lacking an expert algorithm.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper reframes the NAR as an RL process and builds an architecture that is applicable to problems with multiple correct answers, NP-hard problems, and problems without expert algorithms. Performances are evaluated on several CLRS-30 problems and the robust graph construction task.

### Strengths
Redefining the algorithm learning process as a trajectory optimization problem is a promising paradigm. Experiments show that the proposed method solves key problems of NAR methods.

### Weaknesses
1. The introduction of the proposed method, model variants, and experimental metrics is not clear enough. Sometimes it's hard to understand the content due to a loss of details.

2. The time consumption compared with other methods is lacking.

3. The performance of the proposed method is still inferior compared to advanced Non-NAR methods.

### Questions
1. For problems without expert algorithms, how do you compute the rewards?

2. In your framework, how do you train GNARL-BC and GNARL-PPO in detail? I think it should be presented in a concentrated manner rather than scattered across different sections.

3. Line 316 says "meaning that not all CLRS-30 graph problems are representable for the time being". Can you show some examples in CLRS-30 that GNARL can not handle, and is this a limitation of GNARL?

4. How much computational overhead is brought by training the critic model? Can you compare the training time consumption of GNARL and other NAR methods?

5. What is the meaning of the metric "TSP percentage above optimal objective"? Is it the lower the better?

6. In the Limitation Section, it says "GNARL relies on the environment during execution, creating a performance bottleneck. ". Can you explain this in detail?

7. On evaluation of CLRS-30, it says "Rodionov & Prokhorenkova (2025) report 100% graph accuracy on the BFS, DFS, and MST-Prim problems. ". Why is this baseline removed from the comparison in Table 2?

8. Why is GNALR-PPO not considered in Table 2? 

9. In Table 3, why do you use only 10% of the training data? What is the performance comparison when using the full training data?

### Soundness
3

### Presentation
2

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
In this work the authors propose the Graph Neural Algorithm Reasoning with RL (GNARL) framework. This approach casts algorithmic processes as trajectories in MDPs. By formulating the problem as an MDP, they leverage RL techniques to improved performance in replicating performance on algorithmic reasoning tasks. In contrast to other NAR approaches, the authors use MDPs to ensure valid solutions, cast P and NP graph problems into the same formalism, and learn examples without an expert algorithm.

### Strengths
The core idea of this paper is good, and appears quite promising.

The description of the need the authors hope to fill is clear and well-motivated.

The paper’s results are very even-handed, with a clear assessment of the limitations of the current work.

### Weaknesses
If my understanding is correct, the key insight is the formulation of the problem as an MDP. Thus, section 1 is a critical section of the paper. This section was difficult to parse. It seems that the transition function is algorithm-dependent, but many of the other elements are not. Perhaps the authors could provide a compact definition of a GNARL MDP that is similar to the MDP tuple definition found at the beginning of 3.2. Then the GNARL MDP tuple could be defined in terms of graph elements, with notes about which elements are the same for all problems, and which vary according to problems.

In general, it strikes me as difficult to think about how to formulate an algorithmic problem as an MDP, even after reading this paper. It seems to me that a more general framing of the approach would be helpful. Or perhaps this difficulty is the core trade-off, as indicated in the limitations section?

For me, the second and third paragraphs of 3.2 felt superfluous, except perhaps the definitions of the acronyms BC and IL. If the authors need more room to describe their method, I think this section could be trimmed substantially.

### Questions
See weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Neural Algorithmic Reasoning (NAR) involves training neural networks to learn to exectue algorithms. This is typically done by executing the algorithm and collecting  supervision signal (e.g., intermediate variable values) at every step of the algorithm. This data can then be used to train the neural network by supervising the output at every step. This paper proposes a new approach which views algorithms as Markov Decision Processes with a state, a set of actions, and a transition function. This then allows to train NAR models to produce the right action at every step, hence moving the supervision signal from the domain of outputs to those of actions. Training then happens though reinforcement learning methods. In particular the authors propose to use either imitation learning (when an expert policy is available) or proximal policy optimization.
The proposed method is evaluated on standard algorithms for NAR from the CLRS dataset, a combinatorial optimization algorithm (travelling salesman), and an NP algorithm (minimum vertex cover). The baselines are provided by popular NAR methods for CLRS, and specialized approaches for the other scenarios.

### Strengths
- Viewing NAR as an MDP is a very intuitive yet novel idea
- The method can in principle be applied to several graph algorithms
- The experimental results show important improvements over previous methods

### Weaknesses
- There are some existing methods which train NAR models without supervising at every step that have not been considered (e.g., "Deep Equilibrium Algorithmic Reasoning", Georgiev et al., 2024l "Deep equilibrium models for
algorithmic reasoning", Xhonneux et al. 2024). These works should be cited at the very least 
- I found some parts of the text a bit unclear (see questions below)
- defining the MDP for a given algorithm is not always trivial and there may be more possible MDPs for any given algorithm

### Questions
- Are there ways to extend the method to algorithms in which the Markov property doesn't hold?
- Could you explain how the proto action works? I found it quite unclear from the current text
- Does the translation from "encode-process-decode" to "encode-process-act" lead to a much higher number of "steps"? From what is shown in Appendix B it seems that this is the case. Can this be an issue? I feel like there should be a discussion about this aspect in the paper
- Could the authors please expand on the statement that the proposed method "can handle multiple correct solutions"? Is this because instead of having an end-to-end "gold" trajectory, one can just apply 1 step from the current state and then use it for supervision?
- Reinforcement learning is notoriously challenging to apply in practice. Could you please add plots on the training stability of the proposed method and some discussion on the number of training samples required with respect to supervised training?
- Could the authors please expand on what is meant with "we estimate the graph accuracy as micro-F1|"? This last paragraph in Page 6 is quite unclear to me
- Is there an "automatic" procedure that can be used to define an MDP from a given algorithm?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper presents a rephrasing of the reinforcement learning or Markov Decision Process in the language of Neural Algorithmic Reasoning (NAR).

The authors present the general mapping and experiments on the CLRS-30 benchmark on Depth-first or Breadth-First Searches, on TSP instances, on Minimum Vertex Cover (MVC), and robust graph construction (RGC). 

The model can be trained similarly to RL, using Imitation training or using proper RL methods (e.g.PPO)

The paper's contributions are:
1. to show the mapping between NAR and MDP
2. to evaluate and compare the performance on 4 learning tasks

### Strengths
The paper is well written, the related works are mostly covered, and a relatively good number of evaluations.

### Weaknesses
There is no reference in the related work at least to the GFlowNet approach.

The approach is meant to be NAR, but the mapping depends on the problem: "A(s) are specified for each problem", "the horizon $h$ is defined by the problem".

In general, it is hard for me to really understand the difference with respect to using RL. The "framework" is to define the State space $\mathcal S=\mathcal T \times \mathcal F$, with $\mathcal T$ the imposta space and $\mathcal F$ the state space of the graph. 

I probably also have a problem understanding what is different in the general NAR framework, which seems to be the idea of having an encoder, a decoder, and some iteration in the latent space. 

While I find CO really fascinating, I am not sure I really appreciated the NAR perspective; therefore, it is hard for me to evaluate this work.

### Questions
Based on my previous analysis, the author shall therefore clarify better 1) what the actual contribution is compared to RL, 2), explain the difference to RL, and 3) also explain the difference with GFlowNet, which also defines an MDP over a graph of states.

### Soundness
1

### Presentation
2

### Contribution
1
