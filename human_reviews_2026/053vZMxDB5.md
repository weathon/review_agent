# TGPO: Temporal Grounded Policy Optimization for Signal Temporal Logic Tasks

- Decision: Reject
- Scores: 2, 8, 4

## Abstract
Learning control policies for complex, long-horizon tasks is a central challenge in robotics and autonomous systems. Signal Temporal Logic (STL) offers a powerful and expressive language for specifying such tasks, but its non-Markovian nature and inherent sparse reward make it difficult to be solved via standard Reinforcement Learning (RL) algorithms. Prior RL approaches focus only on limited STL fragments or use STL robustness scores as sparse terminal rewards. In this paper, we propose TGPO, Temporal Grounded Policy Optimization, to solve general STL tasks. TGPO decomposes STL into timed subgoals and invariant constraints and provides a hierarchical framework to tackle the problem. The high-level component of TGPO proposes concrete time allocations for these subgoals, and the low-level time-conditioned policy learns to achieve the sequenced subgoals using a dense, stage-wise reward signal. During inference, we sample various time allocations and select the most promising assignment for the policy network to rollout the solution trajectory. To foster efficient policy learning for complex STL with multiple subgoals, we leverage the learned critic to guide the high-level temporal search via Metropolis-Hastings sampling, focusing exploration on temporally feasible solutions. We conduct experiments on five environments, ranging from low-dimensional navigation to manipulation, drone, and quadrupedal locomotion. Under a wide range of STL tasks, TGPO significantly outperforms state-of-the-art baselines (especially for high-dimensional and long-horizon cases), with an average of 31.6% improvement in task success rate compared to the best baseline.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper presents a reinforcement learning (RL) approach for learning from signal temporal logic (STL) to make learning more feasible for long-horizon tasks. The novel model-free approach divides and flattens complex STL formulas and searches for time-variable actualizations via Metropolis-Hastings (MH) sampling to enable efficient learning. The proposed method is compared with a range of existing approaches across several environments. I believe the idea is original and shows promise for improving over existing methods for STL learning. However, the paper still needs substantial work; specifically, a more thorough technical analysis and a systematic description of the proposed approach, as well as clearer explanations and presentation of the experimental results.

### Strengths
To the best of my knowledge, the proposed approach is a novel way to address the sparsity and non-Markovian nature of STL rewards, an important problem in STL learning that existing methods do not sufficiently handle, especially for long-horizon tasks with complex STL formulas. In addition, the proposed approach is compared against a sufficiently representative set of baselines.

### Weaknesses
- **W1.** (Minor) Eq. (2) can be visually improved.
- **W2.** The problem statement is quite vague and somewhat confusing. I think a more comprehensive and consistent notation can be used. Also, constructing an augmented MDP should be part of the solution rather than the objective. Lastly, the probability-maximization statement with respect to the initial state is also ambiguous.
- **W3.** The STL decomposition is not systematically or algorithmically explained (Algorithm 1 is very high level) and is presented primarily through intuition and examples.
- **W4.** The number of different time-variable assignments can grow exponentially with the number of subgoals, which can be further complicated for nested and/or disjunctive formulas.
- **W5.** I think observed (continuous) time should be part of the augmented state space rather than the discrete step number, as STL is defined for continuous signals.
- **W6.** Exact time-variable assignments do not seem realistic; as I understand it, they represent the exact time at which a certain state must be reached.
- **W7.** It seems that a new MDP is constructed for every new assignment, which can technically impede RL.
- **W8.** The formal connection between the dense rewards and STL satisfaction is not established or discussed.
- **W9.** The authors do not provide sufficient technical analysis of the time-variable assignment procedure.
- **W10.** The flow and clarity of the experiments section can be improved.

### Questions
- **Q1.** See W2. What does $\mathbb{P}_{x_0\in\mathcal{X}_0}$ formally mean in the problem statement? Do we assume a distribution over the initial states? Is the objective maximization over success rates or average robustness score?
- **Q2.** See W3. Can you provide pseudocode for STL decomposition? What are the inputs and outputs? How are nested formulas formally divided? How are the time variables determined; for example, for $G_{[0,200]} F_{[0,10]} a$, where the number of time variables can vary, or for $F_{[0,100]} a \vee F_{[0,100]} b$, where one time variable might be sufficient?
- **Q3.** What are your thoughts regarding W4?
- **Q4.** See W6. Why isn't an ordering of subgoals sufficient?
- **Q5.** See W7. What are your thoughts on such a nonstationary environment? Why do you think your procedure transfers knowledge efficiently across MDPs?
- **Q6.** See W8. Does maximizing the return maximize the STL success rate or the robustness score? Is it an admissible heuristic?
- **Q7.** See W9. For example, assuming deep RL converges to an optimal policy that maximizes return, is the time-variable assignment procedure complete; i.e., if there is a feasible assignment, will the procedure eventually propose such an assignment? How fast could that be, given W4?
- **Q8.** See W9. Could you provide pseudocode for time-variable assignment with MH sampling? Also, why do you think the critic provides a good heuristic for MH sampling given that the MDPs are changing?
- **Q9.** I think some of the STL formulas used in the experiments should be discussed in the main paper. Could you provide a general description of the STL formulas? 
- **Q10.** In the figures in the experiments section, are training steps the same as environment steps?
- **Q11.** Could you compare the dimensionality of your augmented state space with history-stacked state spaces?
- **Q12.** Regarding Figure 6, did you use the same STL formulas with the same time intervals? Could you provide similar figures for different environments where the intervals are also scaled accordingly?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper proposes Temporal Grounded Policy Optimization (TGPO), a hierarchical reinforcement learning framework for solving control problems specified using Signal Temporal Logic (STL). STL enables rich task specifications with temporal and spatial constraints, but its non-Markovian structure and sparse reward signals make it difficult to handle with standard RL algorithms.  TGPO decomposes STL formulas into subgoals with invariant constraints, and introduces a two-level architecture: a high-level “temporal grounding” component assigns time variables to each subgoal, while a low-level time-conditioned policy learns to satisfy them using dense, stage-wise rewards. The framework includes a critic-guided Bayesian time allocation step using Metropolis–Hastings sampling, which focuses exploration on promising temporal schedules.
Experiments across five environments (2D navigation, unicycle, Franka Panda, quadrotor, and Ant) show that TGPO and its Bayesian variant (TGPO*) outperform several baselines—τ-MDP, F-MDP, RNN, Grad, and CEM—particularly on complex, high-dimensional, and long-horizon STL tasks.

### Strengths
Strengths
1. Solid statistical analysis and thorough results section.
The empirical evaluation is broad, well-structured, and uses multiple seeds, metrics, and ablation studies. The inclusion of both low- and high-dimensional systems helps demonstrate scalability.
2. The use of the learned value function to guide time allocation search is interesting 
3. The writing is clear for the most part (see weakness 1)

### Weaknesses
1. In sec. 4.1 the authors say “Our method of decomposing STL into subgoals with invariant constraints is inspired by Kapoor et al. (2024); Liu et al. (2025).”  From a clarity standpoint, it is unclear to what extent the proposed method for decomposition into subgoals differs from these prior works.  Can the authors elaborate on this?
2. The shaped reward defined in eq. 6 seems specific to locomotion-type problems.  It’s unclear how this would generalize to other domains.

### Questions
1. See weaknesses 1 and 2
2. In what sense is the time-allocation method Bayesian?
The Metropolis–Hastings sampling uses the critic as a heuristic energy function, but it’s unclear whether this constitutes a Bayesian inference procedure or simply a guided stochastic search.

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
4

### Summary
This paper presents a new reinforcement learning method to learn control policies for some types of STL specifications. The proposed method consists of first sampling time assignments for decomposed subgoals and then learn policies to achieve these subgoals conditioned on the time assignments.

### Strengths
This work presents a new reinforcement learning method for motion planning to achieve sequenced subgoals given time assignments. The formulation of several RL components including state space and reward function seems to be new.

This work proposes a new method for time domain planning in STL that is capable of sampling promising future timesteps for subgoal planning.

Experiments in five simulated environments demonstrate improved performance over some baseline methods.

### Weaknesses
Although the paper claims to solve *general* STLs, in fact the proposed method cannot be applied to some important STLs at the moment.
- Line 197, it is assumed that the type of invariance task after translation only contains *interval* for time variable. This makes modelling and learning for STL that involves infinite time impractical. Although this is mentioned in the limitation section, It should be clearly described at the introduction of STLs.
- This paper cannot handle STLs like G(F …) as mentioned in Line 212. In fact, many liveness properties of the STL cannot be handled by the current method. These kinds of temporal logic are quite important in some domain like maintenance tasks or repeated tasks that require the agent conduct task periodically.
Therefore, the current paper overclaims what types of STL the proposed method can handle. The paper is supposed to have a clear explanation on what the specific STLs are they are targeting, give a formal definition on what these STLs are and what common properties they have in common, how they are related to other types of STLs that the method cannot handle.

The proposed reward function lacks a direct link with respect to the original STL task. It is unclear under this combination of rewards, how the optimal policy behaves in the original STL task. Ideally, the conversion from STL task to RL task should be formulated in a way that both optimization tasks share the same optimal solutions. Similar to LTL2Action, it would be better to provide analysis showing that the optimal policy under the proposed rewards also guarantees the satisfaction of the original STL.

### Questions
How are the STLs designed?

What are the guidelines used for this designing?

Are there any practical scenarios where these STLs have been applied in real applications?

### Soundness
2

### Presentation
3

### Contribution
2
