# Beware Untrusted Simulators -- Reward-Free Backdoor Attacks in Reinforcement Learning

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 10

## Abstract
Simulated environments are a key piece in the success of Reinforcement Learning (RL), allowing practitioners and researchers to train decision making agents without running expensive experiments on real hardware. Simulators remain a security blind spot, however, enabling adversarial developers to alter the dynamics of their released simulators for malicious purposes. Therefore, in this work we highlight a novel threat, demonstrating how simulator dynamics can be exploited to stealthily implant action-level backdoors into RL agents. The backdoor then allows an adversary to reliably activate targeted actions in an agent  upon observing a predefined "trigger", leading to potentially dangerous consequences. Traditional backdoor attacks are limited in their strong threat models, assuming the adversary has near full control over an agent's training pipeline, enabling them to both alter and observe  agent's rewards. As these assumptions are infeasible to implement within a simulator, we propose a new attack "Daze" which is able to reliably and stealthily implant backdoors into RL agents trained for real world tasks without altering or even observing their rewards. We provide formal proof of Daze's effectiveness in guaranteeing attack success across general RL tasks along with extensive empirical evaluations on both discrete and continuous action space domains. We additionally provide the first example of RL backdoor attacks transferring  to real, robotic hardware. These developments motivate further research into securing all components of the RL training pipeline to prevent malicious attacks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This manuscript proposes a novel action-level backdoor attack paradigm, where the adversary is the developer of the simulator or training environment - a threat that compromises the RL supply chain. The proposed attack is evaluated using simulation benchmarks as well as real robotic hardware.

### Strengths
- The concept of performing backdoor injection via the simulator is particularly compelling.

- Although the experiments on real robotic hardware remain preliminary, they constitute an good first step.

### Weaknesses
- Several claims and discussions in the manuscript are not presented with sufficient logical rigor.

- Some important concepts and descriptions lack clarity, leading to potential confusion.

### Questions
- It is unclear how Algorithm 1 is implemented in practice. Is it explicitly integrated into the released simulator’s code? Are there any examples that illustrate this process? If the answer is yes, it would be helpful to clarify how this is consistent with the claim that “the attack does not modify the underlying dynamics of the simulator, but merely severs the relationship between actions chosen by the agent and those executed in the environment.” (P5, Line 266)

- The use of third-party simulators seems more aligned with research settings rather than real-world applications. It would be helpful if the authors could provide examples of real-world cases where a victim relies on a third-party simulator to train an agent before real deployment. Even if real-world cases exist, since agents typically require post-training or fine-tuning to adapt to real environments, it would be interesting to discuss how such processes might affect the effectiveness of Daze.

- One key point that needs clarification is the assumption that the simulator does not have access to the reward. In standard benchmark simulators (e.g., Atari, MuJoCo), the developer naturally knows the entire MDP, including the reward function. In contrast, for user-configurable simulators such as Isaac Gym, the simulator provider cannot know the specific task definition (including state space, action space, and reward), which would make the proposed attack impractical. It would therefore strengthen the paper to clearly define which simulator settings are considered and to explain the rationale behind assuming that the simulator cannot observe or alter the reward, while the state and action can be observed, and the state can even be altered.

- The effectiveness of Daze in continuous action spaces may be highly dependent on the choice of the target action. For example, if the current target action is frequently selected by the agent when encountering a triggered state, the attack works well; however, if the target action is rarely chosen, the attack’s effectiveness may decrease significantly. In cases where the action space is very large, the probability of the target action being selected could approach 0, making it practically impossible for the agent to learn the backdoor. Lines 380-382 and Figure 3 in the manuscript seem to provide indirect evidence for this issue. I suggest that the authors more precisely discuss the attack’s boundaries and limitations.

- Regarding the statement “only receive actions and return environment states, while rewards are computed externally. Therefore, this scenario, while realistic, violates a critical assumption of all prior backdoor attacks - the adversary can no longer alter or even observe the agent’s rewards”(P1, Line 53), I have two concerns: 
1.The phrase “rewards are computed externally” should be explained more clearly, as it forms a key part of the manuscript’s motivation. 
2.The logic seems somewhat inaccurate. Even if rewards are computed externally, the adversary may still be able to manipulate them. Since rewards are tied to the agent’s task objective, an adversary who knows the task goal can still approximate the reward and perform coarse-grained reward manipulation, even if the values are not perfectly precise.

- The statement “Without altering rewards, traditional backdoor attacks are unable to sufficiently bias the agent’s expected returns and ...” (P2, Line 75) is inaccurate. Even if the adversary cannot alter the reward, it can still select high-reward transition data and tamper with the corresponding states and actions to inject a backdoor. 

- The five claimed contributions appear somewhat ambitious. I suggest reorganizing them for clarity and conciseness - for example, combining points 1 and 2, and points 3 and 4. 

- I have a different view regarding the statement “In line with prior work, $\delta$ must be designed such that $S_\delta$ is disjoint with S” (P3, Line 112). In realistic attack scenarios, $S_\delta$ and S do not necessarily need to be disjoint. If $s^{-1} \notin S$, it is considered anomalous and can be easily detected or filtered. On the other hand, if $s^{-1} \in S$ occurs frequently, it may affect the agent’s performance on benign tasks, reducing the attack’s stealthiness. This clearly represents a trade-off.

- It would be clearer to simply write R’ (P3, Line 113) as R to simplify the formulation and avoid potential confusion.

- Why is it necessary to define $\mathcal{U}$ (S) as a uniform distribution in Eq. (1)? Could other distributions be used, and if so, how would that affect the formulation or results? 

- Will $\mathcal{L}_{adv}$ be added to the RL algorithm’s loss function? If not, I recommend not referring to it as a “loss function” here, as this could create confusion for the reader.

- In my understanding, the simulator directly modifies the states (insert trigger). Then, how should we understand the statement “since δ and ϕ only introduce superficial alterations to states, all rewards will be computed with respect to true transitions that occur in M’”？(P5, Line 247)

- Theorem 1 may appear unintuitive, as its validity seems directly related to the choice of $p_\phi$ and the target action. For example, consider a simple counterexample: suppose there are two states, I and D, and two actions, “stay” and “move right”. The initial state is I, and executing “move right” transitions the agent to D, ending the task. If the target action is “move right”, Theorem 1 holds; however, if the target action is “stay”, Theorem 1 fails, because in this case the Daze state is ineffective - the agent will not execute the target action when observing the trigger in state I. This example does not violate the assumption that the random policy is not optimal. Therefore, I guess additional assumptions are needed for Theorem 1 to hold.

- The statement “The only requirement for the malicious simulator developer is to choose which trigger and daze functions” (P6, Line313) seems somewhat inaccurate. As discussed above, the choice of the target action and $p_phi$ are also crucial for the attack’s effectiveness.

- The constant C has a significant impact on the performance of TrojDRL and SleeperNets. Therefore, using the same fixed value across different tasks makes it difficult to conclusively demonstrate that Daze outperforms reward-alteration-based attacks. It would be valuable to include a comparison with the work “UNIDOOR: A Universal Framework for Action-Level Backdoor Attacks in Deep Reinforcement Learning”, which, to my knowledge, performs fully automatic reward alteration. 

- There are several missing definitions in the manuscript:
P5, Line 240: $P_\phi$ appears for the first time but is not defined.
P5, Line 256: $\tau$ appears for the first time but is not defined.
P6, Line 297: the daze factor k appears for the first time but is not defined.

- Some Typos：
P2, Line 105: state spaces A.
P3, Eq (2): Discrete is inconsistent with the text above (line 133). 
P5, Line 257: is the use of $\leq$ before the second $\tau$ incorrect?
P6, Line 291: DLR

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper studies a new framework of attack called Dazed attack where the attacker essentially has control over state transition of the environment and can perturb the action of the agent using a malicious simulator. The authors propose this as a "reward-free" threat model which contrasts to prior works like TrojDRL, SleeperNets which rely on reward poisoning. The attack mechanism work by punishing agent on non-compliance(on not following the target action) by forcing them to a "Dazed" state where agents actual actions are ignored and an uniform policy is played on behalf of agent thereby leading to poor performance. The authors provide theoretical guarantees on their attack framework and demonstrate the effectiveness of it through empirical experiments on Atari and Mujoco environments.

### Strengths
1. The paper is well written and most of concepts are presented in an organized way.
2. The paper claims to present a novel threat model which I think is misleading - more on this in the weakness section.
2. The authors provide good theoretical guarantees on their attack algorithm.
3. The attack framework leads to high empirical success rate on both environments outperforming reward-poisoning counterparts.

### Weaknesses
1. The paper's central claim that this is "more constrained" or "weaker" form of attack is quite misleading. The argument is highly one-dimensional focusing only of lack of reward access but it ignores the fact that the attacker is granted the complete control over the environment transition and action taken by agent to random action. This is an extremely powerful attack capability and it is not convincingly argued why this is 'weaker' than simply altering a scalar reward.
2. The paper misses critical recent related works on state and action perturbations and the work should rightfully be compared to those papers rather than reward-poisoning attack papers. Since authors fail to compare against there more relevant baselines, the paper's positioning is not very clear and its claims on "weaker" form of attack appear highly overstated to the reviewer.
3. The state perturbation attacks are highly detectable and the authors have failed to discuss any defense mechanism in the paper.

References:

[1]. Optimal Attack and Defense for Reinforcement Learning, McMahan et. al, AAAI-24.

[2].  Robust Reinforcement Learning on State Observations with Learned Optimal Adversary, Zhang et. al, ICLR21. (Also look at their follow up works)

### Questions
1. Can you please provide a more robust justification of why your framework is "weaker"?
2. Why is related work, experiments section completely miss comparisons to prior works on state and action perturbation attacks?
3. For a defender, it would be easy to detect the presence of a dazed state by investing the fact that any action taken by agent there leads to similar poor performance, thereby declaring the environment to be malicious. How the attacker would protect themselves against such strong signal of identification?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a novel and highly practical threat model for backdoor attacks in Reinforcement Learning (RL). Instead of assuming an attacker can poison an agent's rewards (a common but strong assumption), this work considers a more subtle attacker: the developer of an "untrusted simulator". The paper provides formal proofs that this mechanism guarantees both attack success (the agent must learn the backdoor) and attack stealth (the agent's policy on the benign task remains optimal). Empirically, Daze is shown to outperform existing SOTA attacks (which require reward access) in continuous domains and match their performance in discrete domains.

### Strengths
Novel and Practical Threat Model: The paper correctly identifies that as RL becomes more reliant on third-party simulators (MuJoCo, PyBullet, etc.) and cloud services, the "untrusted simulator" is a major, realistic security blind spot. It is the first to formally define and attack the "reward-free" threat model, which is a significant contribution to the field of RL security.


Strong Theoretical Guarantees: The attack is not just a heuristic. The authors provide formal proofs (Theorem 1 and Theorem 2) that an agent optimizing its own returns must converge to a policy that is both successfully backdoored and stealthy. The detailed proofs in the appendix (Appendix E) appear sound.

### Weaknesses
1） Would the "Daze" attack successfully transfer in a complex, uncontrolled, "in-the-wild" environment that isn't almost identical to the training simulator?

2） The paper claims this assumption is "fairly weak". However, one could argue that in highly stochastic or complex environments, a random or exploratory action in a specific state might be part of a near-optimal policy. If the "punishment" of random actions isn't severe, the agent won't be as strongly incentivized to learn the backdoor, and the attack would fail.

### Questions
Please see the weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
10

### Rating Number
10

### Confidence
3

### Summary
In this paper, the authors propose DAZE, a reward-free backdoor attack against RL agents. DAZE creates triggered and dazed states: when the agent encounters a trigger, it is forced to perform suboptimal actions. Agents that take the target action (i.e., the adversary's desired behavior) avoid a penalty during training; therefore, they learn to execute the backdoor behavior. Experimental results (in both continuous and discrete action spaces) show that DAZE achieves a high attack success rate while inducing no drop in benign return. The authors also validated DAZE on real robotic hardware. 

I believe that the work is strong and well motivated. It explores simulator-level vulnerabilities in RL with a solid theoretical framework and experimental results.

### Strengths
S1. Novelty: The authors highlight realistic adversary capabilities when the simulator is malicious: no read or write access to the agent's rewards, or cannot alter the rewards. Therefore, their attack, DAZE, operates under constrained but realistic assumptions unlike the prior backdooring work with the reward manipulation assumption.

S2. Impactful Demonstration: In addition to empirical results in the MuJoCo and Atari domains, the real-world robotic example (also supplemented by a video) effectively shows the practical security implications of the attack.

### Weaknesses
W1. Practical Defenses: Although the attack is realistic, the authors have limited discussion on potential detection or mitigation strategies.

W2. Trigger Design: The visual or input-level triggers are somewhat artificial. These triggers might not go unnoticed during deployment.

### Questions
Q1. How sensitive is Daze to the shape of the trigger? Can a slightly perturbed trigger still activate the backdoor?

Q2. Page 8, line 419, type (ore -> or, Interesction -> Intersection)

### Soundness
4

### Presentation
3

### Contribution
4
