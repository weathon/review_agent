# Action Dimension Coordination via Centralised Critics for Continuous Control

- Avg Score: 2.00
- Decision: Reject
- Scores: 2, 2, 2, 2

## Abstract
Continuous control tasks with large action spaces often demand coordination across action dimensions. Recent work has shown that factorising the action space enables deep Q-learning to tackle high-dimensional continuous control problems by leveraging value decomposition methods adapted from multi-agent reinforcement learning (MARL). However, these approaches treat action dimensions independently, which can result in sub-optimal policies when coordination is required. To overcome this, we propose a general framework that adapts centralised training with decentralised execution (CTDE) to single-agent continuous control with factorised action spaces. Our key insight is to reinterpret action dimensions as cooperative "agents" and enable them to exchange information via a centralised critic during training, leading to coordinated policies that can be executed in a decentralised manner at test time. We instantiate this framework with two algorithms, DAC-AC and DAC-DDPG, and evaluate them on 13 DeepMind Control Suite tasks, demonstrating that incorporating centralised critics improves both sample efficiency and asymptotic performance on a wide range of tasks. Using these two algorithms, we further show that our framework seamlessly integrates with existing offline RL methods, achieving state-of-the-art performance across multiple benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper aims to leverage action decomposition to address the coordination problem of action dimensions in high-dimensional continuous control tasks. The authors note that existing action space decomposition methods (such as DecQN) assume independence of action dimensions, which can lead to suboptimal policies in tasks requiring fine-grained coordination. To address this, this paper proposes applying the CTDE paradigm from MARL to single-agent control. During training, a centralized critic, $Q^i$, conditions actions $a_{C_i}$ in other dimensions to learn inter-dimensional dependencies and achieve coordination. At execution, a single, decentralized actor, $\pi_\phi(a|s)$, is responsible for outputting the complete joint action.

Based on this framework, the authors instantiate two algorithms: DAC-AC and DAC-DDPG. Experiments on DeepMind Control Suite tasks demonstrate that this approach outperforms DecQN, REValueD, SAC, and TD3 in an online setting. In addition, the authors also demonstrated the policy-based regularization expansion algorithm and experimental results of this framework in the offline setting, and compared the effects of layer normalization and context Ci in the ablation experiment.

### Strengths
1. The core contribution of this paper is to analogize the "action dimension" to the "MARL agent," highlighting the lack of coordination limitations of existing VDN-style action decomposition methods (such as DecQN). It also introduces a mature and widely used CTDE framework from the MARL field.

2. The actor-critic structure used in this framework makes it easily integrated with existing offline RL techniques, particularly policy-constrained methods. This extends the limitations of DecQN's value-based learning and facilitates the introduction of policy regularization constraints. The state-of-the-art results achieved by DAC-AC-BC on offline benchmarks demonstrate the effectiveness of this extension.

3. The experimental results in this paper are self-contained. On complex high-dimensional control tasks such as dog, fish, and humanoid, the proposed DAC-AC and DAC-DDPG significantly outperform the basic SAC and TD3 algorithms in learning speed and variance. They also demonstrate advantages over DecQN in offline scenarios.

### Weaknesses
1. In the introduction, this paper contrasts "value decomposition methods" with "CTDE frameworks." This formulation is nonstandard in the MARL field and can be misleading. Value decomposition methods (e.g., VDN, QMIX, QPLEX, and QTRAN) are themselves a type of CTDE. I understand the author's intention to contrast "hypothetically independent CTDEs (e.g., VDN-style DecQN)" with "explicitly coordinated centralized critic algorithms (e.g., MADDPG)," but the current formulation is inaccurate.

2. The contribution of this paper's approach is limited. While the proposed contextual Q-value framework mitigates the VDN-style limitations of DecQN, it still lags far behind the state-of-the-art in MARL. The proposed CTDE method is essentially equivalent to a simplified version of a fully cooperative MARL algorithm, where local observations are reduced to a shared global state and the policy network is a joint action policy. Decoupled Q networks are almost MADDPG-style centralized critics, where the context Ci is set to the joint action of all other agents. In this scenario, the difficulty of learning the critic is no different from directly learning a centralized critic, completely failing to demonstrate the necessity of decomposing the action dimension. The discretization of continuous actions is also puzzling, given the numerous mature policy gradient methods in the MARL community, such as FACMAC, that support this setting.

3. The baselines considered in this paper are not comprehensive, and the experimental setup is relatively simple, making it difficult to clearly support the core argument of this paper. The online comparison only considers the classic TD3 and SAC algorithms, severely lacking a comparison with the latest performance in the field. The offline setting only considers DecQN and BC, lacking classic offline RL benchmarks such as BCQ, IQL, and TD3+BC. Furthermore, only DAC-AC-BC performs well, while DAC-DDPG-BC performs very poorly. The authors attribute this to biases in Gumbel-Softmax, but this still means that a key instance of the framework is ineffective in offline scenarios.

### Questions
Q1: Regarding the stability of DAC-AC policy updates: DAC-AC's actor update mechanism appears to be potentially unstable. The mechanism first involves the actor $\pi_\phi$ sampling a joint action $a \sim \pi_\phi(\cdot|s)$; then, using the sampled action $a_{-i}$ as context, it computes a new "optimal" label $a_i^{opt}$; finally, $\pi_\phi$ is trained to imitate $a^{opt}$. Consider a simple XOR game where the optimal actions are (0,0) and (1,1). Could this update strategy lead to catastrophic oscillations? For example, the actor outputs (0,0), and the critic computes the optimal label (1,1) based on the context of (0,0); the actor is updated to output (1,1); and in the next step, the critic computes the optimal label (0,0) based on (1,1). Would this unstable feedback loop prevent the policy from converging to an optimal solution (e.g., (0,1) or (1,0))? Can you design a matrix game experiment of this type to verify and illustrate this?

Q2: Regarding the trade-off between coordination and cost: The ablation experiment in Figure 4 shows that on the Dog-Walk task, using a context with 15 dimensions (|\mathcal{C}_i|=15$) yields similar performance to using 30 dimensions (fully centralized). However, I observe that using 30 dimensions consistently outperforms 15 dimensions on all the tasks shown in the figure, indicating that centralized learning is sufficient to achieve high performance, and reducing the number of observation dimensions only degrades performance. Can you provide a more detailed performance comparison, such as a scaling curve from an empty context Ci to the full-dimensional context?

Q3: As well as the issues mentioned in the weakness section.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work re-formulates standard MDPs as a multi-agent problem under CTDE. Each action dimension is treated as an agent in a cooperative multi-agent system. Based on this framework, they introduce extensions to DDPG and actor critic, called DAC-AC and DAC-DDPG, which use a decentralized critic.

### Strengths
The connection between Factored Action MDPs and MARL is interesting, although the paper does not fully explore this connection in a principled way. Code is also provided in the supplementary material.

### Weaknesses
The fundamental motivation for this work is extremely confusing. In MARL, CTDE is a popular choice because the environment restricts us to learn a fully factorized policy where agents act independently. Thus, the goal is to maximize return over a factorized policy space. In single-agent RL, we do not have such restrictions and the policy is a joint policy defined on the joint action space, which can be represented by either a single policy or an autoregressive one. However, if we restrict the policy space to be factorized, then we are unnecessarily reducing the policy space, and making learning harder. 

Furthermore, there are no theoretical guarantees or even any justifications for why the decomposition is sound. For instance the target in Algorithm 1 is not justified and appears out of nowhere. Value decomposition is also quite restrictive in MARL. To make the connections between single-agent and multi-agent setting in a more principled way, I would suggest focusing on the FA-MDP and MMDP (CTCE) and trying to find deeper connections by deriving either policy iteration or value iteration first.

The experimental results are also mixed, and certainly not convincing enough to justify the lack of theoretical results.

### Questions
1. How do you specify the context $a_{C_i}$?
2. Why is the policy not conditioned on the context?
3. In lines 168-169, “the decomposed functions are referred to as critics”. Why is that the case? Is the decomposed Q-value an actual Q-value, i.e. does it have an interpretation as expected cumulative return, and does it satisfy any Bellman equations?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work proposes a method to tackle environments with large action spaces that require coordination across action dimensions. This work leverages principles from MARL value decomposition methods to improve learning in the single agent domain by factorizing the action space, following the popular centralised training with decentralised execution convention.

### Strengths
This paper addresses issues in single agent RL using important concepts from MARL, which is interesting. A large set of experiments is provided to show the performance of the proposed method. However, there are still some points that could be improved, please see below.

### Weaknesses
While this paper studies an interesting problem, there are still some points that are a bit unclear and could be improved. For example, the notation is sometimes a bit confusing and makes it difficult to follow the methodology. There are also some terms that are inconsistent from a MARL perspective. The authors could have provided more deeper details about how the proposed method is different from other methods that factorise actions spaces in single agent learning, such as the mentioned DecQN. The experiments are good, but the improvements in performance are not impressive either when compared to some of the baselines. In figure 2 it would be good to see the comparison with the baselines methods without the action decomposition too, i.e., DDPG for example. 

It is unclear to me if the size of C_i is environment dependent or not; if C_i can assume any size for any environment, then it is arguable to say that "full centralisation" is being used because if there is a possibility of extending the number of dimensions indeterminedly, then it means it is always possible to make it more centralised, with more dimensions.

A set of examples for C_i were introduced in lines 175-179; however, in the experiments it seems that the full set (2nd bullet point) is the one always being used.

Some minor points: 
- in line 59: missing full stop "."
- in lines 175-179: some points have "." at the end of sentence, others dont
- there are missing citations in the paper; for example QMIX (line 117)
- in line 659-660: "Unimodal algorithm published at IEEE in 2024" - is there a reference instead of writing?


Please find below some more questions that reflect my concerns.

### Questions
1. normally in methods such as VDN, the target to approximate is based on a $Q_{tot}$ which is a mix of the individual q-values of each agent and the target is computed against that; in algorithm 1, is the target calculated based on a mix of the action values from the indices in the set $C_i$ ? What is the meaning of $[a_i]$ int his context? is it calculating the target only for the dimension i? 
2. could the authors elaborate on how their method is different from other methods that factorise the action space such as DecQN?
3. could the authors elaborate on the meaning of "joint action" (line 160) in this single-agent context? does it mean that each agent can pick more than a single action? the meaning of the action dimensions in the presented context is not straightforward for the reader without deeper effort
4. in euqation $y^n$ in line 226, is the target calculated based always on the same action "$a^{opt}$"? it is unclear whether this actions changes for each value of j; it is also unclear whether this is a mix of the action dimensions or if it corresponds only to values of $a^{opt}$
5. in the experiments with different sets of $C_i$: can $C_i$ have any size? or is it dependent on the environment? have the authors considered how the right size of $C_i$ can be calculated without intervention? 
6. considering that the proposed DAC-based methods perform quite high in almost every scenario, is there a specific reason for the suboptimal performance of DAC-DDPG on Finger-Spin or Humanoid-Walk?
7. could the authors elaborate on the need for K ensemble members as shown in fig 1? i cannot also find discussions about the values used for K in the paper; this value is sometimes mistaken for N, when mentioning concepts such as number of critics, ensemble sizes K, "ensemble of N=10 critics" (line 331), etc; these concepts could be made more clear for the reader

### Soundness
2

### Presentation
2

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
This paper introduces two algorithms from multi-agent reinforcement learning (MARL) to address high-dimensional control in single-agent RL, following the work of Seyde et al. (2023) and others. Specifically, it proposes algorithms based on two MARL algorithms, MAAC and MADDPG, which are considered to be better than the MARL algorithm used in Seyde et al. (VDN) in the MARL literature. Experimental results on both online and offline settings show the proposed approaches are competitive to the selected baselines.

### Strengths
The paper has the following strengths:
1. To my knowledge, this is the first work that adopts centralized training with decentralized execution (CTDE) methods to the single-agent RL setting. While prior works have introduced MARL algorithms to address high-dimensional control problems in the single-agent case, a centralised critic approach has not been explored, which might provide better coordination across action dimensions.
2. The paper is well-written and easy to follow.

### Weaknesses
The paper also has some significant weaknesses:
1. The contributions are incremental. While the paper introduces two MARL algorithms with centralised critics, in an attempt to address the lack of action dimension coordination, the paper does not properly address *why* such coordination is needed. While Seyde et al. (2023) demonstrate coordination could happen across different time steps even with a decoupled value function, it’s unclear if a centralised critic provides further useful coordination. The paper does not provide any insight other than performance differences.
2. The motivation of using decentralized policies needs further justification. While the motivation of Seyde et al. (2023) is to use decentralized value function to address the high-dimensionality issue of *value-based* methods, what’s the motivation of using decentralized policies compared to a centralized policy under the actor-critic framework? Isn’t a centralized policy arguably better?
3. The paper overclaims that incorporating centralised critics improves both sample efficiency and asymptotic performance in some online RL settings. The paper only demonstrates two *actor-critic* algorithms with centralised critics perform better than a *value-based* algorithm with a decoupled Q-network. It is unclear if the occasional performance improvement is due to the difference in the critic/value function architecture (centralised vs. decoupled) or the RL algorithm difference (actor-critic vs. value-based). The experiments in Figure 4 partly address this but it is not convincing: 1) The results are based on only three environments and three seeds, which are quite limited, 2) they do not cover the case in which the condition set size is 0 (i.e., decoupled), and more importantly, 3) the difference between curves are small with overlapping shades (confidence intervals instead of standard deviation should be used for the plots).
4. The paper overclaims that the proposed two algorithms achieve state-of-the-art performance across multiple benchmarks. 1) The paper only performs empirical investigation in one benchmark with different settings (Beeson et al., 2024). 2) The paper uses standard error with five seeds as an error measure, which is not statistically significant to make such a strong claim. 3) Only one of the proposed methods appears to be better than the baselines in a subset of settings.
5. The empirical evaluation is limited to continuous control environments that do not seem to require cross action-dimension coordination (Seyde et al., 2023).

### Questions
Questions that might impact the rating:
1. Why is action coordination through a centralised critic needed? Seyde et al. (2023) have shown that DecQN can achieve coordination across different time steps. The results in their paper also suggest that the DeepMind Control suite (used in this paper) does not require action coordination. Could the authors provide justification for the choice of the benchmark?
2. Could the authors provide further insights or justification on the core motivation of the paper? Why would decentralized policies be better than a centralized policy under the actor-critic framework?
3. Why does SAC have such an unstable performance in humanoid tasks? What hyperparameters are used for it?
4. Are layer normalisation and huber loss applied to SAC/TD3 baseline as well?

Other minor questions that have little impact on the rating:
1. Line 387: Why would using a straight-through (ST) estimator necessarily inflate the critic’s input dimensionality? It should be possible to treat each action dimension independently as it is done in the DAC-DDPG-BC, which uses Gumbel-Softmax.
2. The paper hypothesizes that DAC-DDPG-BC suffers from bias introduced in Gumbel-Softmax. What temperature is used in Gumbel-Softmax in DAC-DDPG? Would reducing the temperature (and thus reducing the bias) improve performance in offline settings?

### Soundness
1

### Presentation
3

### Contribution
2
