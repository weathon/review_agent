# Adversarial Inception for Bounded Backdoor Poisoning in Deep Reinforcement Learning

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 6, 3, 5

## Abstract
Recent works have demonstrated the vulnerability of Deep Reinforcement Learning (DRL) algorithms against training-time, backdoor poisoning attacks. These attacks induce pre-determined, adversarial behavior in the agent upon observing a fixed trigger during deployment while allowing the agent to solve its intended task during training. Prior attacks rely on arbitrarily large perturbations to the agent's rewards to achieve both of these objectives - leaving them open to detection. Thus, in this work, we propose a new class of backdoor attacks against DRL which achieve state of the art performance while minimally altering the agent's rewards. These ``inception'' attacks train the agent to associate the targeted adversarial behavior with high returns by inducing a disjunction between the agent's chosen action and the true action executed in the environment during training. We formally define these attacks and prove they can achieve both adversarial objectives. We then devise an online inception attack which significantly out-performs prior attacks under bounded reward constraints.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors interrogate the drivers of current attack performance in RL, and in doing so identify a simple mitigation that affects multiple extant attacks, but, crucially, not their own novel attack framework.

### Strengths
The authors take an interesting perspective on RL attacks that highlights a weakness (vis a ve detectability) in current attack frameworks, that when resolved, opens new opportunities for highlighting more realistic attacks. This is then used to demonstrate the impact of their approach. 

While there is a lack of clarity and specificity relating to the mathematical details of their work in the opening chapters, the remainder of the work is well written and cleanly presented. The authors have clearly thought about how to convey their intended message,

**Edit (2 December 2024)** - score has been increased to a 6 based upon the rebuttal period, however this is a soft 6. The authors have improved their work in terms of the rebuttal revisions, and have demonstrated a significant desire to improve their paper. However the rebuttal comments have also raised additional concerns regarding the clarity of explanations around core concepts (see discussion regarding the outer-loop threat model), and on if the authors have fairly captured relative performance. 

The discussion period also did illuminate additional concerns regarding data handling practices that I am going to be charitable and assume have all been resolved. However for future reference I think that the authors need to fundamentally re-think how they're managing their data, as having multiple cut and paste related errors suggests serious process failings. Please generate data once, consistently, and then copy from that. Your problems have been created by your own piecemeal approach.

### Weaknesses
To begin with, I'm quite suspicious about the results in Table 5. Take for example Q-Incept, with a BRR of 99.29% and a SD of 17.27%. To maximise the SD assume that the first 3 trials were 100, and leave the last two trials as free variables. There's no solution in [0,100] that yields an average of 99.29% with a standard deviation of 17.27%. I don't see how some of the results are achievable for both CAGE-2 and Q-BERT - the standard deviations are just too high to be realisable.

Another major concern is the assumption that all attacks are evaluated under bounded reward poisoning (line 441 and 442). I appreciate that this is a simple and effective approach when demonstrated in Figure 2, but I am concerned about the lack of specificity to it, and the potential for this to be a highly environmentally sensitive approach (see Questions). The impact of this issue is amplified when Figure 4 demonstrates how intrinsic this approach is for the observed performance deltas. 

I also find the argument about access to the machine and access to ram being a likely attack vector to be highly optimistic. These listed system intrusions are not contextualised by the number of systems that could be compromised, nor if they are at all related to any machine learning related systems. This is not to say I don't disagree about making white-box assumptions, but I would argue that there are far better arguments for it than system penetration statistics. 

Speaking of arguments regarding the threat model, I also find the argument that offline trajectory manipulations are easier to detect as the relationship between states and actions would need to be well defined - in many systems of interest establishing such a model would either be a) trivial or b) potentially an important part of model development. In both cases, this would appear to undercut the argument being made. 

Beyond these points, the authors show a consistent lack of specificity when it comes to their mathematics terms (see below), and while they mention testing at other beta ratios, but don't show them in the main body of the text.

Beyond these points, I also have a number of minor quibbles:
- Figure 1 attempts to demonstrate how the approach works. But it fails to detail how the agent would behave in the absence of a trigger? Moreover a reader may be confused as to why the agent chooses an action that is different to the transition action. In the context of trajectory manipulation it makes more sense, however this point is not well made in the opening of the paper. 
- Table 1 should cite the other techniques - especially when this is the first time in the paper that they have been discussed. 
- The authors perspective on the literature is heavily weighted towards reward poisoning attacks, but there's also more general poisoning attacks, or offline specific attacks. There's a broad topology of attacks against RL, including those that manipulate rewards, obversations, policies, and actions. 
- S3 "The adversary attempts to instead train the agent on adversarial MDP" - is ambiguous and poorly phrased. For starters, it should be "on the adversarial MDP", as you're introducing a formal definition of it afterwards. But by discussing this as both an adversary (who is doing the action) and an adversarial MDP, there is an ambiguity about what the adversarial MDP is adversarial to. Moreover, this whole framing of "attempting to instead train" may be confusing to readers - how are they attempting to train this? Why is this only an attempt, is this not a guaranteed process? Does this MDP mechanicistically replate to M? Is it that M is replaced - if so, then a different phrasing would be more appropriate. 
- "We model the trigger" - the trigger is not yet defined, so you're discussing a concept in your problem formulation for which the reader has no context to understand its meaning, nor how it relates to the sentences that have just been introduced. This definition on line 114 becomes extra confusing around line 125 - when $\delta$ is now the poisoned state, rather than being a trigger. It may well be that the trigger being activated converts $s$ to its poisoned state equivalent, but the current way this is presented does not make that point cleanly. This issue pops up later in S4.2 as well - and I guess it comes down to a differences in perspective. The authors are presenting the trigger as the system that is triggered to change between states, but I would think that the average reader would interpret a trigger as being something that induces a second system to change state behaviour. 
- Repeated and superfluous definition of MDP's from line 87 and line 110. 
- $\mathbb{S}$ is never introduced as (presumably) the set from which S can be drawn from until 6 lines after S is introduced, and 5 lines after $\mathbb{S}$. 
- $\pi^{+}$ is not defined. 
- Section 3 uses $S$, but Section 3.1 uses $s$ - seemingly for the same concept. 
- Line 133 - V is not defined. 
- $\mathbb{1}$ is not defined - although most would get what it is, it is still a point of notational complexity. 
- Statistics presented as "in 2024 alone" for a paper that was written at some time in 2024 are impossible to parse by a reader, as they have no context to understand how far into the year these numbers were collected. 
- Line 237 "are standard throughout [the] poisoning literature"
- When talking about $\beta$ - this is referred to as a rate/proportion, but it is only discussed in terms of states.  But is this the proportion of states across a single trajectory, or across all trajectories? There is no specificity to this point. 
- Line 240 "decreasing the agent's performance in the benign task" - there's an argument to be made about why this is important, but instead it's left to the reader to understand the context here. 
- L266 "they receives" -> receive.

### Questions
- The authors make a point about other approaches allowing unbounded perturbations. But part of this work is that stealthiness is a concern in prior works, and that multiple definitions exist. Do these alternate approaches allow unbounded perturbations /while/ maximising stealthiness? If so, does the fact that prior works allow unbounded perturbations have any practical impact? 
- Figure 2 appears to be a very simple, but powerful argument against some of the tested comparisons. However, from my understanding, rewards for highway merge are highly sparse and spiky - so wouldn't these seem spikes be observed in the case of a weakly performing agent? If so, is it really that Figure 2 is showing anomalous spikes, rather than it just demonstrating the inherent spikiness in the rewards for this environment? Is this same behaviour seen with these attacks in other environments, or over longer time periods? How do these markers change between trained, untrained, and partially trained models?
- Could you explain the standard deviations in your table of results (see in the weaknesses for more details about this point of concern)?

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
4

### Summary
This paper proposes a new set of backdoor attacks that minimize the altering of the target agent's reward function. The paper demonstrates the effectiveness of the proposed method both empirically and theoretically.

### Strengths
+ The paper proposes a new attack against DRL under the constraint of reward mutations. 

+ The paper provides theoretical justifications for the proposed attack.

### Weaknesses
- I would appreciate the authors showcase or discuss the generalizability and scalability of the proposed attacks. The authors can consider reevaluating their attacks on more complex DRL environments, such as MuJoCo or Atari games. 

- I would suggest the authors evaluate the proposed method against existing backdoor detection and defenses, such as [1].


[1] BIRD: Generalizable Backdoor Detection and Removal for Deep Reinforcement Learning

### Questions
1. How well the proposed attack can be generalized to more complex DRL environments? 

2. Is the proposed attack applicable to multi-agent RL environments?

3. Is the proposed attack resilient to existing backdoor defenses for DRL?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper introduces "inception attacks," a novel approach to DRL backdoor poisoning that overcomes the limitations of reward-based attacks by manipulating the relationship between chosen and executed actions during training, rather than relying on suspicious reward modifications. The method is formalized through the adversarial inception attack framework and implemented via "Q-Incept," which demonstrates superior attack success rates under bounded reward constraints. The effectiveness is theoretically proven and empirically validated while maintaining the agent's normal performance on its intended task.

### Strengths
1. Identifies a fundamental limitation in reward-based backdoor attacks and proposes a novel solution through action manipulation
2. Provides theoretical analysis for adversarial inception, with results generalizable across MDPs,
3. Proposes formal guarantees for maintaining dual objectives - normal task performance and backdoor effectiveness

### Weaknesses
1. The core concept of action manipulation between chosen and executed actions isn't novel, being previously explored in adversarial RL literature ([1], [2])
While the application to backdoor attacks is new, the paper fails to properly acknowledge and discuss these related works.
2. The paper's description of experimental scenarios appears overclaimed - **should not characterize simple gym environments as "robotics navigation"**. While the abstract and introduction claim diverse experiments, the evaluations are actually conducted in very simple Gym environments with minimal difficulty, raising serious concerns about empirical effectiveness and practical scalability.
3. The ablation studies on attack budgets are notably incomplete, lacking comprehensive exploration of different parameter settings and their impacts.

[1] Action Robust Reinforcement Learning and Applications in Continuous Control, Tessler et al, ICML 2019

[2] Game-Theoretic Robust Reinforcement Learning Handles Temporally-Coupled Perturbations, Liang et al, ICLR 2024

### Questions
See weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper studies the problem of backdoor attack in Deep Reinforcement Learning termed as “Adversarial Inception”. Unlike prior works that require large perturbation in reward to achieve attack goal, this paper proposes a novel attack method in bounded reward poisoning setting but allowing the attacker to also perturb the actions taken by the agent(hence the name inception). The authors propose a training algorithm called Q-incept to install the backdoor behavior in the agent which can be triggered during deployment using predefined trigger functions. The paper provides some theoretical results on their attack algorithm and supplement it with various empirical results to prove the efficacy of their method on various simulation environment.

### Strengths
1. The paper proposes novel attack strategy in bounded reward domain that require changing the action taken by the agents.
    
2. The paper appear to provide both theoretical to guarantee effectiveness of their training algorithm.
    
3. The paper  conducts empirical analysis comparing their algorithm with prior works like SleeperNets, TrojDRL to illustrate superiority of their method in bounded reward setting.

### Weaknesses
1. The problem setup is poorly formulated. There are a bunch of issues with notation and objectives defined in the paper. See the question section for more info.
    
2. Threat model is pretty strong and makes it less practical in real world setting - it requires the adversary to break into the training system and access RAM values and change all of state, actions and rewards. Specifically, unconstrained inception attack is very strong. If the adversary is that powerful, why wouldn’t it directly modify the final policy itself?
    
3. The action based attack are not new and has been previously studied more rigorous RL attack setting. See [1].
    
4. Experimental results are very poorly states. The authors have just stated a plethora of results without highlighting any interesting takeaway points. Metrics like Attack Success Rate and Benign Return Ratio are not defined anywhere in the paper.

5. The paper does not discuss sufficient details on defenses against their attack method.
    

[1]. [Optimal Attack and Defense for Reinforcement Learning](https://ojs.aaai.org/index.php/AAAI/article/view/29346)

### Questions
1. Why is a good policy a map from superset of states of actions? How does actions not in $M$ affect the transitions in $M$?

2. What does maximizing over $M’$ even mean equations (1) and (2)? None of terms in expectation depend on $M’$. Please clarify.
    
3. The MDPs $M$ and $M'$ appear to be totally different except sharing their action spaces. What does $V^M_{\pi^+}(s)$ mean?
    
4. Table 1 is not very clear to me. What is the difference between static and bounded rewards? Are they mutually exclusive and is one better than the other?
    
5. Can you provide a comparison between the reward perturbation made by your algorithm vs prior methods?

### Soundness
2

### Presentation
2

### Contribution
2
