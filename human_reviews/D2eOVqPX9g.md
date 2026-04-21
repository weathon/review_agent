# Finite-Time Analysis of On-Policy Heterogeneous Federated Reinforcement Learning

- Avg Score: 6.25
- Decision: Accept (poster)
- Scores: 6, 8, 5, 6

## Abstract
Federated reinforcement learning (FRL) has emerged as a promising paradigm for reducing the sample complexity of reinforcement learning tasks by exploiting information from different agents. However, when each agent interacts with a potentially different environment, little to nothing is known theoretically about the non-asymptotic performance of FRL algorithms. The lack of such results can be attributed to various technical challenges and their intricate interplay: Markovian sampling, linear function approximation, multiple local updates to save communication, heterogeneity in the reward functions and transition kernels of the agents' MDPs, and continuous state-action spaces.  Moreover, in the on-policy setting, the behavior policies vary with time, further complicating the analysis. In response, we introduce FedSARSA, a novel federated on-policy reinforcement learning scheme, equipped with linear function approximation, to address these challenges and provide a comprehensive finite-time error analysis. Notably, we establish that FedSARSA converges to a policy that is near-optimal for all agents, with the extent of near-optimality proportional to the level of heterogeneity. Furthermore, we prove that FedSARSA leverages agent collaboration to enable linear speedups as the number of agents increases, which holds for both fixed and adaptive step-size configurations.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes the federated version of SARSA algorithm and analyses its convergent performance with the existence of heterogeneity in both transition dynamics and reward functions.
Different from classical settings of federated reinforcement learning, the paper does not assume that agents have to share the same transition dynamics and reward functions.
The paper demonstrates that its proposed algorithm achieves a linear speedup for the convergence to the optimal answer in each local environment both theoretically and empirically.

### Strengths
1. The paper considers heterogeneity in both transition dynamics and reward functions. Moreover, it quantifies the degree of these heterogeneities and discusses their effect in the final convergence of FedSARSA.
2. The paper discusses the convergence region and linear speedup of FedSARSA. It is claimed that smaller learning rates and a larger number of participating agents will help tighten the convergence region, which matches the intuition.
3. The numerical experiment is carried out in settings with different degrees of heterogeneity.

### Weaknesses
1. What does MSE in the numerical experiments stand for? Does it mean the averaged MSE of current parameter to optimal parameters in different environments?
2. The explanation of numerical experiments is not enough. For example, why the MSE of FedSARSA with a large number of $N$ ($N=40$) increase along the training process when $\epsilon_p>0,\epsilon_r>0$? And where is the confidence bound for the numerical experiments?
3. The convergence MSE of FedSARSA with different number of agents are different from each other. What makes that difference?

### Questions
See Weaknesses.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work performs theoretical studies on federated reinforcement learning with clients facing heterogeneous environments. In particular, the classical SARSA algorithm is extended to the federated version. Theoretical analyses are established to demonstrate the finite-sample convergence of the proposed FedSARSA under linear function approximation. In particular, linear speedups are reported with the established results.

### Strengths
- This work follows the interesting line of work of extending federated learning to the domain of decision-making under environmental heterogeneity. This setting is well-motivated and has wide practical implications.

- The established results are solid and novel based on my reading. In particular, no similar results have been reported on FRL with heterogeneous clients in the planning task with both linear function approximation and linear speedup.

- Despite the theoretical nature, the overall presentation is clear and the key intuitions are provided. The listed sketch of the proof especially facilitates the readability.

### Weaknesses
- I am overall satisfied with this work. There is just one minor question I have for the authors. As mentioned at the end of page 8, the obtained results from federated RL can be leveraged as initialization points for finetuning with just local data. I imagine the analyses would not be different given existing works, and wonder whether it would be possible to state the finetuning results, which may better highlight the impact of cooperation on accelerating individual learning.

### Questions
Please see the weakness section.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies federated on-policy reinforcement learning with linear function approximation. It proposes FedSARSA that is a federated version of SARSA. It proves that FedSARSA converges to the neighborhood of the optimal parameter with a linear speed up.

### Strengths
+ FedSARSA is intuitive and reasonable.
+ The theoretical result that proves linear speedup is interesting.

### Weaknesses
- The theoretical results do not have the impact of periodic updating. 
- The authors do not specify the communication cost and how it trades off with the convergence.
- The FedSARSA is a straightforward of single-agent SARSA -- the only difference is to aggregate the parameter estimation from all agents, which is a straightforward average.
- The authors talked about how heterogeneity is captured in the convergence, but this relationship is not well articulated. In FL, one would first need to define the heterogeneity metric and then express the convergence bound as a function of this metric. Furthermore, such heterogeneity should be defined on the data, not the underlying distribution.
- Paper writing needs some work. It is strange to not have a Conclusion section.

### Questions
- The motivation is unclear to me -- why do we want to learn a single universal policy? Each agent may interact with his/her own environment and learn a personalized policy for that environment. Isn't that better to be deployed on that environment than the single averaged policy across all environments?
- I don't see the linear speedup in the simulation results?

### Soundness
3 good

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies the problem of on-policy federated RL with agents interacting with potentially different environments. A new algorithm FedSARSA is proposed, and shown to converge to a near-optima policy for all gents. Convergence speed analysis is also provided.

### Strengths
- The paper is well written. The formulation and ideas are explained clearly.

### Weaknesses
- It would be helpful if the authors could provide more intuition about where the speed up comes from. Specifically, what in the problem formulation/assumptions enable this speedup? Intuitively, this would be possible only when things are homogeneous (or close to that). 
- Is it possible to comment on the optimality of the finite-time error? Right now only upper bounds are provided.

### Questions
- It would be helpful if the authors could provide more intuition about where the speed up comes from. Specifically, what in the problem formulation/assumptions enable this speedup? Intuitively, this would be possible only when things are homogeneous (or close to that). 
- Is it possible to comment on the optimality of the finite-time error? Right now only upper bounds are provided.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good
