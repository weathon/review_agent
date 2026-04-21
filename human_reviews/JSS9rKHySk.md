# On the Role of General Function Approximation in Offline Reinforcement Learning

- Avg Score: 6.80
- Decision: Accept (spotlight)
- Scores: 8, 8, 6, 6, 6

## Abstract
We study offline reinforcement learning (RL) with general function approximation. General function approximation is a powerful tool for algorithm design and analysis, but its adaptation to offline RL encounters several challenges due to varying approximation targets and assumptions that blur the real meanings of function assumptions. In this paper, we try to formulate and clarify the treatment of general function approximation in offline RL in two aspects: (1) analyzing different types of assumptions and their practical usage, and (2) understanding its role as a restriction on underlying MDPs from information-theoretic perspectives. Additionally, we introduce a new insight for lower bound establishing: one can exploit model-realizability to establish general-purpose lower bounds that can be generalized into other functions. Building upon this insight, we propose two generic lower bounds that contribute to a better understanding of offline RL with general function approximation.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies the offline RL with general function approximation.

The central problem of this topic is to identify the structural assumptions that empower sample-efficient learning. Many previous works impose assumptions on the data coverage and/or assumptions on the function class/MDP structures. Recently, foster et. al. established the fundamental limit when the completeness is missing but there are also works like [1] and [2], bypassing the lower bound in foster et. al. with different function approximation target.

This work aims  to present a rather unified framework to connect the function approximation to the realizability, which is interesting and very relevant to the community.


[1] Offline reinforcement learning with realizability and single-policy concentrability
[2] When is realizability sufficient for off-policy reinforcement learning?

### Strengths
The topic is interesting. I have been curious about what we can do without strong completeness-type assumption and super-strong data coverage conditions since [1]. While there are works bypassing the lower bounds in [1] with different type of realizability assumptions and refined characterization of completeness [2, 3], the relationship between various assumptions in the literature is still not very clear. So it is exciting to see that the authors are trying to fill the gap.

However, the current version is still not ready for publication due to the writing quality and since the relationships to previous works are not clear. 

[1] Offline reinforcement learning: Fundamental barriers for value function approximation
[2] Offline reinforcement learning with realizability and single-policy concentrability
[3] When is realizability sufficient for off-policy reinforcement learning?

### Weaknesses
The main drawback of this paper is the writing. The authors state many things without concrete examples and arguments but only abstract summarizations (by using the word certain). For instance, the authors mention the previous works of ``Xie and Jiang''by their refined data assumptions, which assuming that the audience is familiar with these previous works. Meanwhile, many notations are used without a formal definitions although I can understand some of them as they have been used by the previous papers but this can still be a problem for general readers. 

1 It would be great if the authors could instantiate the definitions present in this paper or discuss the relationships with the previous assumptions like the Bellman completeness in [1] and the realizability in [2]. For instance, in the example 2, it would be better to prove that the standard Q-pi-realizability is a special case of definition 2 by explicitly specifying the $\mathcal{F}$ and $\mathcal{G}$.

2 In example 3, may I take \phi as the mapping from a Q-function to its greedy policy? So completeness w.r.t. $\mathcal{T}^\pi$ for all the functions whose greedy policy is $\pi$ is indeed completeness w.r.t. $\mathcal{T}$ for all the functions in $\mathcal{F}$?

3 While proposition 1 is intuitive and reasonable, it seems that many previous works are indeed constructing the lower bounds through a worst-case consideration on the MDP by constructing a family of MDP classes satisfying the assumptions (e.g. [3]). 

4 How did the conditions of theorem 1 compare to the assumptions in [1,2]? This should be highlighted so that the readers can evaluate the roles of the corollaries presented in section 5.1.

5 Most of the coverage assumptions are made with respect to the $\xi(M) \in \Pi$, which is more related to the coverage over $\Pi$, in contrast to the single-policy coverage assumption ($\pi^*$) commonly used in the previous works. I am trying to understand the case of $\pi^* \in \Pi$. But due to the presence of $\xi$, the obtained lower bound may not compete with $\pi^*$. Can you give an example of the unknown mapping $\xi(\cdot)$ so that we can understand the meaning of the target $J_M(\xi(M))-J_M(\hat{\pi})$?



[1] Bellman-consistent Pessimism for Offline Reinforcement Learning

[2] Offline reinforcement learning with realizability and single-policy concentrability

[3] is pessimism provably efficient for offline rl

### Questions
See weakness.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies offline RL with general function approximation, focusing on an information-theoretic understanding of different types of assumptions (such as completeness and realizability) and how they translate to assumptions on the underlying MDP. This results in a principled way of constructing lower bounds in offline RL with general function approximation via model-realizability, and with this method the authors prove the necessity of completeness assumptions.

### Strengths
- This a well-motivated and interesting work and I enjoyed reading the paper. The work puts common assumptions for PAC learning of offline RL with general function approximation under scrutiny and provides a clear overview and concrete definitions of completeness and realizability assumptions. 
- Connecting the model-free completeness assumptions to assumptions that impose restrictions on the underlying MDP is very useful as it unifies lower-bound constructions for different function approximation methods. Particularly given that worst-case information-theoretic lower bounds are specific to function-class assumptions—e.g., Foster et al. 2021 prove a lower bound for value-based and exploratory data settings while recent papers such as Zhan et al. 2022 overcame the difficulty by changing the function approximation method. 
- With the model-realizability technique introduced in this paper, the authors show the necessity of the completeness-type assumptions.

### Weaknesses
There are no major weaknesses.

### Questions
--

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
4 excellent

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies general function approximations in offline reinforcement learning (RL) settings. The authors firstly formally defined two types of usual assumptions on function approximations, i.e., realizability-type and completeness-type. They then discussed the relationship between assumptions on function approximations and the Markov decision processes (MDPs) those can be realized. Section 5 presented the main Theorem 1, which says that only realizability-type of assumptions are not enough to learn better policies. Corollaries 1-3 then assure similar results for different variants of realizability-type assumptions. The authors later discussed the limitations of the lower bound results, i.e., using over-coverage and not containing $Q^*$-realizability. Theorem 2 then shows a lower bound for $Q^*$-realizability but with partial coverage which may not cover optimal policy.

### Strengths
1. The presentation is clear, with enough discussion about the background and related work.
2. The problem is convincingly motivated and important.
3. The results are novel to my knowledge.

### Weaknesses
1. As the authors also noted, the main point of arguing that only realizability-type assumptions are not enough to learn better policies is kind of weakened by the fact that upper bounds for $Q^*$-realizability already exist, and Theorem 2 also feels short of compensating since as mentioned the covered policy is not optimal.

### Questions
Could you explain how it happens for $\xi(M)$ can be optimal while the value function class does not contain optimal $Q^*$. Example or intuition could be helpful.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper focuses on offline learning setting with general function approximation.
The authors first formalize different types of assumptions in offline RL (realizability, completeness).
In Section 3, the authors reveal that the lower bounds derived for model-realizability can also be applied to other types of functions.
After that, in Section 4 and 5, they establish lower bounds for learning in offline setting either without $Q^*$-realizability or weaker data coverage assumptions.

### Strengths
The paper writing is relatively clear to me. The authors contribute interesting lower bounds, which reveal new understanding for learning in offline setting with general function approximation.

### Weaknesses
1. Although lower bound results are also meaningful, it would be better if there are some upper bound results to understand what we can do.

2. As for the results in Section 6, can you explain more about the difference between the data coverage assumptions in (Xie & Jiang 2020) and the lower bound instance in Theorem 2?

    Besides, as for the data coverage assumption used in Theorem 2, how does it compare with the practical scenarios? Would it be too restrictive in practice (maybe in practice we may rarely encounter such bad case)?

### Questions
I'm still trying to understand more about Def. 2. Consider the completeness assumption that $\forall f\in\mathcal{F}$ we have $\mathcal{T}^* \in \mathcal{F}$. I wonder what the corresponding $\mathcal{F}$ and $\mathcal{G}$ in Def. 2 should be in this case?

It also seems unclear to me how Def. 2 "captures" what common completeness assumptions ensure: "the existence of a
function class that can minimize a set of loss functions indexed by another function class".

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 5

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies the role of function approximation in offline reinforcement learning (RL). Specifically, the paper first propose various types of function classes used in offline RL, and then show the learning lower bound under these types of function classes. Generally speaking, given functions with a policy class domain, it is impossible to learn a good policy.

### Strengths
1. The paper proposes a unified way to study general function approximation in offline RL.
2. The theoretical results are solid and interesting.

### Weaknesses
1. The presentation is not clear. Some new words, such as exploratory-accurate(accuracy) and data assumption are used without pre-defined.
2. While there are examples about those assumptions, I think a big table summarizing and connecting existing concrete examples and assumptions and the proposed general assumptions can make the paper easier to understand.

### Questions
It seems like function approximation on policy class or value function class results in impossible learning tasks, does it mean that learning with a model class might have positive results?

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good
