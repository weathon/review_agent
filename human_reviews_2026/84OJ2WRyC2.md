# Fault Tolerant Multi-Agent Learning with Adversarial Budget Constraints

- Avg Score: 3.33
- Decision: Reject
- Scores: 2, 6, 2

## Abstract
Multi-agent reinforcement learning (MARL) often assumes that agents can reliably execute their intended actions. In practice, however, malfunctions and unexpected failures are inevitable, leading to severe coordination breakdowns. We propose the \emph{Multi-Agent Robust Training Algorithm (MARTA)}, a plug-and-play framework for training MARL policies that remain effective under agent failures. MARTA introduces a novel adversarial game in which a \emph{Switcher} learns when and where to activate malfunctions in high risk-states, while an \emph{Adversary} controls the faulty agents. The remaining agents are trained to \emph{jointly} best-respond to such targeted malfunctions, yielding policies that are robust to critical failures. We provide theoretical guarantees that MARTA converges to a Markov perfect equilibrium, ensuring robustness against worst-case malfunctions under both cost and budget formulations. Empirically, MARTA achieves state-of-the-art fault tolerance across diverse MARL benchmarks, including Traffic Junction, Level-Based Foraging, and Multi-Agent Particle Environments, substantially improving performance under both aligned and distribution-shifted failure scenarios. Our results highlight MARTA as a principled and general approach for fault-tolerant MARL.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes an adversarial training framework for robust multi-agent reinforcement learning (MARL). The approach, called the Multi-Agent Robust Training Algorithm (MARTA), introduces a Switcher agent that decides at every time step when and how to intervene with an adversarial decision-maker. Therefore, each agent has an adversarial counterpart that can replace its corresponding functional agent. The functional agents and the adversarial agents are trained alternatingly using different QMIX instances, whereas the Switcher is trained with soft actor-critic. MARTA is evaluated in different benchmark environments with adversarial agents and compared to the standard MARL methods QMIX and VDN, which are not adversarial themselves.

### Strengths
The paper addresses an important (yet well-studied) problem in cooperative MARL.

The proposed method, MARTA, and setting are built on well-established foundations, such as zero-sum Markov games, and are therefore sound.

I appreciate the diversity of environments used to demonstrate the effectiveness of MARTA.

### Weaknesses
**Novelty**

The problem of faulty agents or where each agent has an adversarial counterpart has been addressed in prior work, which has neither been discussed in related work nor compared with in the experiments [1,2]. The problem setting defined in Section 2.1 has been formalized as a zero-sum Markov game or, more generally, as a mixed cooperative-competitive game in the literature [3,4].

The adversarial training scheme of two opposing QMIX instances has also been introduced in [1].

This leaves the Switcher agent being the only new addition to the overall adversarial framework, making the technical contribution somewhat incremental.

**Soundness**

The theoretical analysis may be sound, but I do not consider it a major contribution, since the whole setting is a straightforward instantiation of zero-sum Markov games [4], where robustness properties have been shown long before.

Minor comment: Despite focusing on a partially observable setup, the paper assumes observations to be Markov as the policy conditions on them. To make the definition theoretically sound, the policies should condition on the action-observation history [5].

**Significance**

The paper aims to evaluate the robustness by integrating faulty agents. However, I couldn't find any information on how these faulty agents are created (e.g., from MARTA?) to check for potential biases during testing.

Despite discussing some prior work on robust MARL, the experimental comparison does not include any of them. Instead, the work is only compared with QMIX and VDN, which are standard MARL methods that are known to fail in adversarial settings [1,2]. Without any comparison with alternative adversarial MARL methods, I cannot confirm the claim that MARTA is less conservative and more effective than prior work.

**Literature**

[1] Phan et al., "Learning and Testing Resilience in Cooperative Multi-Agent Systems", AAMAS-20

[2] Li et al., "Byzantine Robust Cooperative Multi-Agent Reinforcement Learning as a Bayesian Game", ICLR-24

[3] Lowe et al., "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments", NeurIPS-17

[4] Littman et al., "Markov Games as a Framework for Multi-Agent Reinforcement Learning", ICML-94

[5] Oliehoek et al., "A Concise Introduction to Decentralized POMDPs", 2015

### Questions
How are the faulty test agents created for the experimental test?

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper tackles the problem of fault tolerance in multi-agent reinforcement learning (MARL). The authors propose MARTA, a modular and plug-and-play training framework that enables MARL systems to remain robust when some agents malfunction. The framework introduces two additional learners: a Switcher, which decides when and which agent to trigger a malfunction on, and an Adversary, which determines the faulty agent’s behavior. The interaction between the Switcher, Adversary, and cooperative agents is formulated as a stochastic game with either a per-activation cost or a fixed malfunction budget. Through theoretical analysis, the authors demonstrate that the learning dynamics converge to a Markov perfect equilibrium. Extensive experiments across several benchmark environments show consistent improvements in robustness without altering the underlying MARL architectures.

### Strengths
1. The paper addresses an important and relatively unexplored problem of fault tolerance in multi-agent reinforcement learning, with a clear and well-motivated problem statement.

2. The proposed MARTA framework introduces a novel Switcher–Adversary formulation that models both the timing and behavior of agent malfunctions in a unified way.

3. The work offers theoretical guarantees and consistent empirical results across various MARLs, demonstrating clear practical benefits and robustness improvements.

### Weaknesses
1. The presentation is quite dry and mathematically heavy, making the paper difficult to follow in parts. The figures are limited and do not clearly illustrate the key intuitions behind the proposed framework. 

2. The title and abstract do not accurately convey the main emphasis of the work. They make the paper appear as a system or framework for fault-tolerant MARL, whereas the actual contribution lies more in the theoretical formulation and convergence analysis of the proposed approach. 

3. Although the framework is described as involving both a Switcher and an Adversary, the analysis and experiments mainly focus on the Switcher component, leaving the Adversary’s role and interaction underexplored.

### Questions
1. Some modeling choices appear primarily made for mathematical tractability, like allowing only one agent to malfunction at a time. While understandable, it would be valuable to include a short discussion or limitation section on how these assumptions affect real-world applicability and whether extensions to multiple concurrent failures are feasible.

2. Although the framework introduces both the Switcher and Adversary learners, the paper effectively fixes the Adversary during analysis and training. Given the symmetry of roles, would it be possible to alternatively fix the Switcher and optimize the Adversary, then combine the two solutions for better equilibrium behavior?

3. In the MARTA-B variant, the budget constraint is expressed as a fixed number n of allowed activations. This formulation raises several questions: 
(1) Does the learned policy tend to exhaust its budget regardless of context? 
(2) If the remaining steps become fewer than the remaining budget, does the Switcher always activate faults? 
(3) Conceptually, MARTA encourages selective fault triggering to balance cost and robustness, whereas MARTA-B seems to encourage full budget utilization. Would it be more consistent to introduce a total-cost constraint (e.g., ∑cᵢ ≤ B) with a penalty beyond B rather than a hard count-based budget?

4. It is unclear whether MARTA-B is empirically evaluated. The experimental section seems to focus solely on MARTA; providing results for MARTA-B would strengthen the overall validation of the proposed variants.

5. While the paper claims modularity and “plug-and-play” usability, there is limited discussion of the computational overhead. The appendix briefly reports total CPU hours, but a more explicit analysis of computational cost would help substantiate the claimed practicality.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes MARTA, a framework for training MARL policies that are robust to an adversary that can at any step adversarial control one of the agents. A Bellman operator for the game is proposed, and various theoretical properties are shown. A variant of the framework with a budget on the number of malfunctions is also considered. Experiments on three simple MARL problems show that the addition of the proposed method improves fault tolerance compared to unconstrained MARL methods.

### Strengths
- The exact problem setting tackled seems to be novel, to the best of my knowledge, although there are other works (though uncited) that tackle very similar problems
- Experimental results suggest the proposed framework improves robustness compared to without applying the proposed framework

### Weaknesses
- Related works [A, B, C] tackling very similar attack vectors as the one proposed in this paper are not cited or discussed.
- The experimental section is a bit lacking. In particular, the proposed method is not compared to any other MARL methods that try to improve fault tolerance, including both methods discussed in the paper (e.g., M3DDPG) as well as ones that the authors did not identify but are highly relevant (e.g., [A]).
- The domains that are tested on seem quite simple
- Also, details on the environments tested are lacking
    - How many agents does each environment have?
- Line 429: The paper claims that shielding and backup policy methods “often sacrifice performance and scale poorly as the number of agents increases.” Does the proposed method improve upon the scalability of these methods? For example, the cited (Qin et al., 2021) has experiments with up to 1024 agents. Was the method deployed on even larger scale environments compared to prior work that “scaled poorly”?
- Many parts of the proofs in the appendix are poorly written are difficult to read
    - All the math in Proposition 3 uses inline mathematics which is too excessive. This hampers readability and made it very difficult to follow the logical flow of the proof.

[A] Li, Simin, et al. "Byzantine robust cooperative multi-agent reinforcement learning as a bayesian game." ICLR 2024.

[B] Zhou, Ziyuan, and Guanjun Liu. "Robustness testing for multi-agent 
reinforcement learning: State perturbations on critical agents." *arXiv preprint arXiv:2306.06136* (2023).

[C] Zheng, Haibin, et al. "One4all: Manipulate one agent to poison the cooperative multi-agent reinforcement learning." *Computers & Security* 124 (2023): 103005.

### Questions
- How do existing methods of fault tolerant MARL perform in terms of robustness compared to the original VDN/QMIX baseline even though they do not target the exact same problem formulation?
- The citations for switching controls (Mguni, 2018a; Mguni et al., 2023) seem very strange
    - In Mguni (2018a), the term “switching controls” does not even appear. Instead, it references very old and established literature on **impulse control**, as well cites the appropriate literature such as [$\alpha$]
    - It’s not clear why the term “switching controls” was chosen instead of the more established “impulse control”
- I’m confused by how / whether the existence of the value $v^*$ is shown. Specifically, I’m not sure where it is proven that $\min_{g} \max_{\pi} v(s | \pi, g) = \max_{\pi} \min_{g} v(s | \pi, g)$.

[$\alpha$] Bensoussan, A. "Contrôle Impulsionnel et Inéquations Quasi Variationelles." International Congress Of Mathematicians. 1975.

### Soundness
2

### Presentation
2

### Contribution
2
