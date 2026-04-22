# Bridging the performance-gap between target-free and target-based reinforcement learning

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 8

## Abstract
The use of target networks in deep reinforcement learning is a widely popular solution to mitigate the brittleness of semi-gradient approaches and stabilize learning. However, target networks notoriously require additional memory and delay the propagation of Bellman updates compared to an ideal target-free approach. In this work, we step out of the binary choice between target-free and target-based algorithms. We introduce a new method that uses a copy of the last linear layer of the online network as a target network, while sharing the remaining parameters with the up-to-date online network. This simple modification enables us to keep the target-free's low-memory footprint while leveraging the target-based literature. We find that combining our approach with the concept of iterated $Q$-learning, which consists of learning consecutive Bellman updates in parallel, helps improve the sample-efficiency of target-free approaches. Our proposed method, iterated Shared $Q$-Learning (iS-QL), bridges the performance gap between target-free and target-based approaches across various problems while using a single $Q$-network, thus stepping towards resource-efficient reinforcement learning algorithms.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents iterated Shared Q-Learning, a deep reinforcement learning method designed to bridge the gap between target-free and target-based approaches. The proposed technique stores only the last linear layer of an old Q-network as a frozen target, while sharing the remaining parameters with the current online network. Coupling this shared features innovation with iterated Q-learning reduces the memory footprint compared to classic target-network approaches without sacrificing stability. Extensive experiments (Atari, DMC continuous control, language model RL) show that iS-QN recovers much of the performance lost by omitting target networks, sometimes even outperforming classic methods in both online and offline settings.

### Strengths
1. The paper is well-motivated, as reusing all but the final linear layer for target calculation cuts memory usage, making deep RL feasible even on resource-constrained hardware or for large architectures.

2. The framework is tested on diverse discrete and continuous control tasks, as well as language-based RL, complemented with analyses on training dynamics, expressivity of representations, and the stability of regression targets.

### Weaknesses
1. The number of parallel Bellman updates (heads, K) needs tuning, and optimal values may vary with domain and architecture.

2. While memory use is reduced, floating-point operations and wall-clock training time are not improved compared to baseline methods.

### Questions
1. Is there a way to adaptively select or anneal K during training?

2. Can you answer W2? Adding studies or discussions on continual learning/resource-constrained real-world deployments might be a good idea.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Iterated Shared Q-Network (iS-QN), a simple method that stores only the final linear layer of the target network while sharing all other layers with the online network. Combined with iterated Q-learning, iS-QN learns multiple Bellman updates in parallel, reducing memory use and improving sample efficiency. Experiments across Atari, DMC, and Wordle tasks show that iS-QN bridges or surpasses the performance gap between target-free and target-based RL with minimal additional cost.

### Strengths
The notion of retaining only the last linear layer of the target network (shared features) is conceptually simple but impactful. It effectively merges the strengths of both paradigms without introducing significant architectural or computational overhead.
Moreover, the evaluation spans multiple benchmarks (Atari, DMC, Wordle), architectures (CNN, IMPALA, SimbaV2, GPT-2), and settings (online/offline/continuous).
The breadth of validation strongly supports the claim of generality and practicality.
The paper achieves significant empirical improvements over many benchmark examples.
Detailed hyperparameters, FLOP analyses (Fig. 10), and appendices ensure reproducibility.
Code release and dependency listings are also promised upon acceptance

### Weaknesses
Despite solid empirical results, the paper provides no formal convergence or stability analysis. For instance, how the partial sharing of target parameters affects the contraction properties of the Bellman operator remains unaddressed.

The optimal number of parallel Bellman updates (K) varies across domains (e.g., K = 9 works for Atari, K = 1 for SAC). The paper admits this (Sec. 6) but does not analyze the scaling law or provide tuning guidance.
While works like Kim et al. (2019), Bhatt et al. (2024), and Gallici et al. (2025) are cited, the paper lacks direct quantitative comparison under identical settings.
The FLOP and runtime discussion (Fig. 10) shows that iS-QN doesn’t actually reduce compute per iteration. Clarifying whether sample efficiency translates into wall-clock speed-up would strengthen the practical argument.
The exposition occasionally interchanges terms like iS-QN, iS-DQN, iS-CQL, etc., without immediate redefinition. A concise schematic or unified notation table would improve readability.

### Questions
1. What is the empirical or theoretical trade-off between K and variance/stability?
2. How does feature sharing interact with normalization layers (LayerNorm vs BatchNorm) beyond empirical observation?
3. Can the authors provide ablation on the frequency T of head updates, similar to target-network refresh rate in DQN?
4. Future work: combining iS-QN with network pruning or quantization could yield compelling results for edge RL; a brief pilot experiment would add value.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes an alternative to target networks, namely keeping only a frozen copy of the last linear layer as the “target,” while sharing all upstream features with the online network. Building on this, the authors use iterated Shared Q‑Networks (iS‑QN), a method that uses multiple linear heads that learn consecutive Bellman updates in parallel. Across Atari (online/offline), DMC Hard (SAC), and Wordle (ILQL), this closes or even beats target‑based baselines while using far less memory. The paper also offer some empirical explainations why the method stabilizes learning.

### Strengths
The main strength of this work lies in the problem they are solving, which is removing the target network from deep value based RL methods. Their idea can be integrated into DQN code with minimal changes while preserving the memory usage of a target free method and enjoying the stability of a target based method. The experiments appear extensive and well executed.

### Weaknesses
The main weakness lie in this paper's similarity to that of Elsayed 2024 [1] and the follow up by Vasan 2025, which also claims to solve a similar problem, this work would benefit greatly from a more direct comparison with these works. By properly placing your contributions in the context of these works, it would greatly enhance and appropriately detail the contributions of your work. 

While somewhat orthogonal to this work, it would be nice if the authors could give some theoretical explains (even on simple mdps/toy problems) as to why their method enjoys a best of both worlds (target free memory, target based stability) thought this might be hard with deep networks. 

[1] Elsayed, Mohamed, Gautham Vasan, and A. Rupam Mahmood. "Streaming deep reinforcement learning finally works." arXiv preprint arXiv:2410.14606 (2024).

### Questions
1. How does your work compare with the work of Elsayed 2024 [1] and Vasan 2025, which as you mention in your work, remove the need for the target network? When/why should one be preferred over the other? Why did you not compare your method with theirs? 

Given this is my main issue with this work, I would consider raising my score if this is adequately addressed.

### Soundness
3

### Presentation
4

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
Proposes iterated shared Q-learning (iS-QL), which combines iterated Q-learning (prior work) with the novel approach of "Shared Features" for the target network. "Shared Features" replaces the standard target network with a single linear layer that shares its input features with the final linear layer of the online Q-network. Similar to a standard target network, the new target network head is a delayed copy of the online network head's parameters. This greatly reduces the memory required for training, since there is no longer a full target network, yet mostly retains the improved stability of a standard target network.

### Strengths
Experimental results suggest that iS-QL indeed increases return compared to target-free algorithms, but with significantly less memory usage than target-based algorithms.

The Fig 7 results on an LLM show iS-QL getting even higher return than the target-based algorithm, which is extra promising because that experiment is, among the experiments in the paper, debatably the closest to a real-world problem.

Large variety of experiments (online, offline, continuous control, and language-based RL)

The hyperparameter tuning protocol is clear (except for some things mentioned in "Weaknesses" of this review)

Code is included in the supplementary materials, and at a glance it looks fairly comprehensive

### Weaknesses
Introduces a hyperparameter, $K$, the number of Bellman iterations to learn in parallel. It appears to be mildly sensitive. Especially since, if I understand right, Fig 6 introduces an additional discounting hyperparameter for the iS-SAC $K = 9$ case.

As made clear in the paper (which is a strength), sometimes iS-QL does not match the returns of standard target-based algorithms.

It should be made clearer whether the target-based approach is given LayerNorm in continuous-action settings, or no normalization at all. If it is given no normalization at all, that would be unfair since iS-QL is given BatchNorm (which the paper notes does not help the target-based approach).

Fig 7 shows 600k gradient steps, but the appendix says 800k. Further, neither value is explained in the paper, and I don't think the original ILQL paper used either of those values.

Results use 5 or 10 seeds per task (but the large variety of experiments greatly mitigates this overall).

### Questions
> mitigate the brittleness of semi-gradient approaches and stabilize learning

Is this referring to two issues, or one issue?

&nbsp;

> Our proposed method, iterated Shared Q-Learning (iS-QL), bridges the performance gap between target-free and target-based approaches across various problems, while using a single Q-network, thus being a step forward towards resource-efficient reinforcement learning algorithms

You might split this sentence into two, and also say "stepping towards" instead of "thus being a step forward towards"

&nbsp;

> However, replacing look-up tables with non-linear function approximators and allowing off-policy samples to make the method more tractable

You might specify "tractable" more precisely here

&nbsp;

> This ultimately limits the size of the online network due to the constrained Video Random Access Memory (VRAM) of GPUs.

Memory limits also apply to, for example, TPUs

&nbsp;

> We propose storing only the smallest possible part of the target network, i.e., the parameters of the last linear layer,

This is not the "smallest possible part of the target network", because you could for example lag only half of the last linear layer. Even though I would guess that would not work well

&nbsp;

> Iterated Q-Network (Vincent et al., 2025)

"$Q$-network" and "Q-network" are used inconsistently. It would be better to stick with one

&nbsp;

> The optimal policy of a Markov Decision Process (MDP) with a discrete action space can be obtained by selecting for each state, the action that maximizes the optimal action-value function $Q^*$

This applies to continuous-action MDPs too

&nbsp;

> This is why Mnih et al. (2015) approximate the optimal action-value function with a neural network $Q_\theta$, represented by a vector of parameters $\theta$.

This confused me for a second, since I misread it as trying to explain why they use a neural network in particular

&nbsp;

> leveraging the contraction property of the Bellman operator $\Gamma$ to guide the optimization process toward the operator’s fixed point, i.e., the optimal action-value function $Q^*$.

The Bellman operator is often not a contraction (even when it can provably converge to $Q^*$). See for example Section 11.4.3 "Comparison between the two convergence proof techniques" in https://sites.google.com/view/rlfoundations/home

&nbsp;

> Where $\gamma$ is the discount factor linked to the MDP of interest.

"the discount factor linked to the MDP of interest" is a bit unclear, particularly since the discount factor used for training should often ideally be smaller than the discount factor used for evaluation. https://www.ijcai.org/Proceedings/16/Papers/626.pdf

&nbsp;

> Gallici et al. (2025) also develop a method for a streaming scenario, in which they rely on parallel environments to cope with the non-stationarity of the sample distribution

Would it be fair to say they also rely on LayerNorm for this?

&nbsp;

> gradient w.r.t. the

Is there a missing backslash in the LaTeX there? The space after the last "." looks a bit large

&nbsp;

> Importantly, we incorporate the insights provided by Gallici et al. (2025) to use LayerNorm (Ba et al., 2016) for the experiments with discrete action spaces, as we found it beneficial, even for the target-based approach. Similarly, we use BatchNorm (Ioffe & Szegedy, 2015), as suggested by Bhatt et al. (2024), to improve sample-efficiency in continuous action settings

This seems to imply BatchNorm helped more than LayerNorm for continuous-action settings? If so, it might help to make that even clearer.

&nbsp;

### Soundness
3

### Presentation
3

### Contribution
3
