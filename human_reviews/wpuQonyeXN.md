# Provably Efficient Exploration in Quantum Reinforcement Learning with Logarithmic Worst-Case Regret

- Avg Score: 6.00
- Decision: Reject
- Scores: 6, 6, 6, 6

## Abstract
While quantum reinforcement learning (RL) has attracted a surge of attention recently, its theoretical understanding is limited. In particular, it remains elusive how to design provably efficient quantum RL algorithms that can address the exploration-exploitation trade-off. To this end, we propose a novel UCRL-style algorithm that takes advantage of quantum computing for tabular Markov decision processes (MDPs) with $S$ states, $A$ actions, and horizon $H$, and establish an $\mathcal{O}(\mathrm{poly}(S, A, H, \log T))$ worst-case regret for it, where $T$ is the number of episodes. Furthermore, we extend our results to quantum RL with linear function approximation, which is capable of handling problems with large state spaces. Specifically, we develop a quantum algorithm based on value target regression (VTR) for linear mixture MDPs with $d$-dimensional linear representation and prove that it enjoys $\mathcal{O}(\mathrm{poly}(d, H, \log T))$ regret. Our algorithms are variants of UCRL/UCRL-VTR algorithms in classical RL, which also leverage a novel combination of lazy updating mechanisms and quantum estimation subroutines. This is the key to breaking the $\Omega(\sqrt{T})$-regret barrier in classical RL. To the best of our knowledge, this is the first work studying the online exploration in quantum RL with provable logarithmic worst-case regret.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work studies quantum reinforcement learning, where quantum means that the classical reward and state transition feedback is replaced by quantum pure states (see Eq. (3.1) (3.2)). This paper studies both the general MDP and linear MDP, and it shows that they can achieve logarithmic regret performance, which breaks the classical square-root regret lower bound.

### Strengths
- The authors did a good job of presenting this work and comparing it to known literature.

### Weaknesses
- Lack of novelty. I am familiar with the work of Wan et al 2022 for quantum multi-armed bandits and quantum linear bandits. As the main theoretical tools for multi-armed bandits and linear bandits are considerably similar to RL and linear RL respectively, this paper can be regarded as an extension from Wan et al 2022 to quantum RL. Although the author pointed out one new challenge in Remark 5.1, I did not see enough novel contributions in this work.



---
Zongqi Wan, Zhijie Zhang, Tongyang Li, Jialin Zhang, and Xiaoming Sun. Quantum multi-armed
bandits and stochastic linear bandits enjoy logarithmic regrets. In To Appear in the Proceedings
of the 37th AAAI Conference on Artificial Intelligence, 2022. arXiv:2205.14988

### Questions
Although leaning toward a negative evaluation of this work for its lack of contribution, I think this quantum RL topic is interesting and would suggest that the authors look into challenging issues around this topic, e.g., regret lower bounds for quantum RL which is not studied in quantum bandits in Wan et al 2022. 

If the authors think there are other nontrivial challenges (except for Remark 5.1) in this work than in Wan et al 2022, please take the chance of rebuttal to explain.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies quantum RL, which provides sample complexity for both tabular MDP and linear mixture MDP, based on several quantum estimation oracles. Compared with previous literature, this paper provides an online exploration method for quantum RL.

### Strengths
- This paper is well written and easy to follow
- Study quantum RL is novel in the literature, with limited prior works
- The proposed online exploration paradigm is more practical than previous work.

### Weaknesses
- The discussion on sample complexity is not enough. For example, it would be better to discuss why the $\sqrt{T}$ factor is removed. Is that because of Lemma 3.1 and Lemma 3.2 such that the previous $\epsilon^{-2}$ sample complexity can be reduced to $\epsilon^{-1}$ sample complexity so that the exploration can be more aggressive? 
- Besides the previous comment, I'm also looking for discussions about the lower bounds (or at least some conjectures). For example, if the dependency on $d, H$ within the lower bounds still match (Zhou et al., 2021) or not?

### Questions
Besides my concern about the weakness, I'm concerned about the cost of translating a classical RL task into a quantum-accessible RL task. Here are my questions
- Can one directly covert the observation in classical RL to a quantum-accessible RL? (e.g., changing the Atari games to quantum). If the quantum RL can be used in classical RL tasks, then how would the current $\log T$ bound break the classical $\sqrt{T}$ regret bound?
- If the current algorithm can only be used in quantum-accessible RL, and we cannot convert a classical RL task into quantum, then how will this algorithm contribute to real-world RL tasks?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work studies a quantum RL problem, where the objective is to explore the episodic MDP with quantum access and learn the optimal policy while minimizing the regret over $T$ episodes. The authors propose the Quantum UCRL and Quantum UCRL-VTR algorithms for tabular MDP and linear mixture MDP settings respectively. Their analysis of the algorithms gives $O(\text{poly}(\log T))$ regret upper bounds.

### Strengths
- This work is one of the pioneering effort in studying quantum reinforcement learning with theoretical guarantees.
- This work incorporates quantum *multi-dimensional/multivariate* estimation subroutines into UCRL-based algorithms. The insights in such incorporations may of interest to the emerging quantum machine learning community.

### Weaknesses
- Quantum regret lower bound is not discussed in the paper 
- The presentation is not clear -- some notations are used without clearly defined or explained
- A closely related work is missing from the literature review: Ganguly, Bhargav, et al. "Quantum Computing Provides Exponential Regret Improvement in Episodic Reinforcement Learning." arXiv preprint arXiv:2302.08617 (2023).
- This work does not have any empirical study of the proposed algorithms

### Questions
- Could the authors comment on why the binary oracle is not considered in the tabular setting? What is the main difficulty in generalizing the result of Lemma 3.1 to binary oracle?
- Is the regret lower bounds of the studied "quantum" exploration problems known? If not, could the authors comment on the difficulties of getting such lower bounds? 

The above two points may be worth mentioning in a future direction section/paragraph.

- The introduction/description of the Quantum UCRL algorithm is not clear enough. Specifically, I could not find the $\bar{\varphi}_{h+1}, \mathcal{D}_h(s^k_h, a^k_h)$ notations appeared in Algorithm 4 being defined anywhere. 
- If $\bar{\varphi}_{h+1}$ is as defined at the end of subsection 3.2, then it is a quantum state in superposition. How could Algorithm 4 line 9 update the counter according to the superposition? Please correct me if I missed anything.
- Why does Quantum UCRL divides the episodes into T/H phases while Quantum UCRL-VTR divides into K phases? How should the practitioners set the parameter K for Quantum UCRL-VTR?

I would love to see Algorithm 4 be presented in the main paper for the sake of clarity if the page limit allows.

- (minor wording issue)  The term "quantum state" is somehow ambiguous as the term "state" has its special meaning in RL problem.

### Soundness
3 good

### Presentation
1 poor

### Contribution
3 good

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies the online exploration problem in reinforcement learning. Specifically, two RL settings are considered: tabular Markov decision processes (MDPs) and linear mixture MDPs; and the goal is to learn the policy that minimizes regret. To achieve this, the authors propose two algorithms that adapt existing RL algorithms by using tools from quantum computing to get performance gain.

### Strengths
**The following are the strengths of the paper:**
1. Adapting recent tools from quantum computing to improve the performance of RL algorithms is a challenging and interesting contribution.

2. The authors consider two RL settings -- tabular MDPs and linear mixture MDPs; and propose algorithms (Quantum UCRL and Quantum UCRL-VTR) with logarithmic regret (in terms of episodes) for both problems due to quantum speedup.

### Weaknesses
**The following are the key weaknesses of the paper:**
1. Motivating examples: It is unclear if the assumptions (access to quantum oracles and their inverse, quantum state) made in the paper are practical or not. Adding a few motivating examples where these assumptions (will) hold will make the contribution even more significant.

2. The doubling trick to design lazy-updating algorithms with quantum estimators is already used in existing work (e.g., Wan et al., 2022), so saying this is a  novel technique proposed in the paper is an overclaim (Last paragraph on Page 2). However, I agree adapting this idea to MDPs is not that straightforward.

3. Since the learner does not observe the next state, it is unclear how the number of quantum samples ($n_h(s, a)$) is tracked by the learner. It is important as tracking $n_h(s, a)$ is needed to update the estimate of the transition kernel. Overall, adding a detailed explanation of how quantum computing tools are used will make it easier to understand the contributions.

### Questions
Please address the above weaknesses. I have a few more questions/comments:
1. Page 6, paragraph before 'Lazy updating via doubling trick': Is there any connection between phase length (H) and episode horizon (H)?

2. The quantum oracle for reward function is not used as it is assumed to be known for the problems considered in the paper. Is this right?

Minor comment:
 If possible, authors can add a few experiments using the Python library QisKit. It will make the paper stronger.

I am open to changing my score based on the authors' responses.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
