# Partner-Aware Hierarchical Skill Discovery for Robust Human-AI Collaboration

- Decision: Reject
- Scores: 4, 4, 4, 2

## Abstract
Multi-agent collaboration, especially in human-AI teaming, requires agents that can adapt to novel partners with diverse and dynamic behaviors. Conventional Deep Hierarchical Reinforcement Learning (DHRL) methods focus on agent-centric rewards and overlook partner behavior, leading to shortcut learning, where skills exploit spurious information instead of adapting to partners’ dynamic behaviors. This limitation undermines agents' ability to adapt and coordinate effectively with novel partners. We introduce Partner-Aware Skill Discovery (PASD), a DHRL framework that learns skills conditioned on partner behavior. PASD introduces a contrastive intrinsic reward to capture patterns emerging from partner interactions, aligning skill representations across similar partners while maintaining discriminability across diverse strategies. By structuring the skill space based on partner interactions, this approach mitigates shortcut learning and promotes behavioral consistency, enabling robust and adaptive coordination. We extensively evaluate PASD in the Overcooked-AI benchmark with a diverse population of partners characterized by varying skill levels and play styles. We further evaluate the approach with human proxy models trained from human–human gameplay trajectories. PASD consistently outperforms existing population-based and hierarchical baselines, demonstrating transferable skill learning that generalizes across a wide range of partner behaviors. Analysis of learned skill representations shows that PASD adapts effectively to diverse partner behaviors, highlighting its robustness in human-AI collaboration.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper is motivated by the challenge in Human-AI collaboration where traditional Hierarchical Reinforcement Learning (HRL) agents fail to adapt to diverse partners due to agent-centric skill discovery, which often leads to "shortcut learning." To address this, the authors introduce Partner-Aware Skill Discovery, a DHRL framework that learns skills conditioned on partner behavior. They achieve this by proposing a novel contrastive intrinsic reward to align skill representations for similar partners while maintaining discriminability for diverse strategies. Evaluating PASD in the Overcooked-AI environment with diverse self-play and human proxy partners, the authors found that their method consistently outperforms existing population-based and hierarchical baselines, demonstrating superior generalization and robustness across a wide range of collaborator behaviors.

### Strengths
* PASD introduces a novel contrastive intrinsic reward that conditions skill learning on partner behavior that is quite interesting
* Generalization is validated using a diverse partner population across various skills
* Analyses of mean skill switches and policy entropy was a nice qualitative insight into learned adaptive behavior

### Weaknesses
* Comparisons against established cooperative baselines, specifically Cross Environment Cooperation (CEC) and E3T, are absent and necessary for full validation.
* The paper lacks in-depth analysis of error modes and failure cases for baselines (FCP, HiPT) versus PASD, which is needed to fully justify the claims of robust coordination and mitigation of shortcut learning.
* Section 4.2 is unclear and could benefit from an explanation more grounded in the context of partner-adaptive dynamics
* The approach relies on sampling from a predefined partner population; the paper should briefly discuss the implications for zero-shot generalization to truly novel human partners in scaled up settings beyond Overcooked

### Questions
- Can the authors offer a more detailed, qualitative analysis of failure cases? Specifically, demonstrate instances where FCP or HiPT fall victim to shortcut learning or coordination failure, and contrast these with how PASD's partner-aware skills resolve the issue.
- What are the practical implications for zero-shot generalization? Can the authors speculate or provide preliminary results on performance when paired with a truly novel, unmodeled human partner policy in realistic settings beyond Overcooked?
- How does the assumption guarantee that the InfoNCE objective captures meaningful partner-conditioned information rather than merely maximizing skill-to-state diversity?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a novel method for learning diverse, partner-consistent skills for a hierarchical RL algorithm set within a MARL problem setting. Their method for learning diverse skills extends the mutual-information family of diverse skill learning strategies to the MARL setting, where they maximize a lower bound on the mutual information between skills and sub-trajectories. This encourages consistent representations (and presumably behaviors?) across partner interactions, i.e. skills that are discriminative across partners but consistent for partners with similar behaviors. 

They evaluate their method in the standard Overcooked-Ai environment and show superior performance to another method that transfers to humans without human data, Fictitious Co-play, a hierarchical MARL method, HiPt, and DIAYAN. They show improved performance adapting to a policy cloned from human data, and show that their method switches skills more than HiPT---the other Hierarchical MARL algorithm.

### Strengths
- The writing is relatively clear 
- The idea of applying mutual-information based skill discovery to the multi-agent setting is a really interesting application of the idea. The method. It's interesting that their method, in principle, allows for more diverse skills to be learned in a hierarchical MARL setting.
- They show good results in the overcooked-AI domain
- They show good results when transferring to a behavior clone from human data

### Weaknesses
- right now, I think "ultimately supporting adaptive human-AI coordination" is over-claiming since you don't actually test with transfer/adaptation to humans
- why not compare against human player? The original overcooked codebase [1] has code for running human experiments. Why don't you use that? [2] also studies transfer to human players, why not use that? Behaviorally cloned policies are rarely as adaptive as ones learned online (without very large, diverse training sets). Thus, it's hard to imagine that Table 2 is representative of transfer to human partners.
- "We assume that distinct sub-trajectory views of the same skill encode a consistent partner-adaptive strategy" - can you motivate this? One agent is adaptive to another agent, so based on what part of the task the other agent is doing, I could see that a sub-trajectory for the same ostensible skill would encode a different partner-adaptive strategy, since its adapting to the partner. Do you demonstrate this somehow? Figure 4 shows more skill switches by PASD tha HIPT. Is that evidence for this? If so, why? If not, what is your evidence for this? 
- Regardless, by construction, I can see why different sub-trajectories of the same skill will encode the same behavior (regardless of another partner) because of how mutual-information based Rl methods work. Maybe this is what your method is exploiting for skill learning?
- The method is not that easy to read and understand given all of the indexing. This is a challenge in both HRL and MARL settings generally, which probably compounds for your method. A summary figure would be really helpful.
- your related work should discuss [2]
- Table 1 and your standard deviations are a bit deceptive. $101.3 \pm 8.5$ should be be bold in reference to $96.0 \pm 1.3$, since those clearly overlap. Your first sentence of this paper is "Developing intelligent agents that can coordinate effectively with humans and other novel partners has long been a central challenge in multi-agent reinforcement learning". Given this motivation, shouldn't you care about generalization to competent partners? Your evaluation doesn't seem like the best one given your motivation.
- You say "Analysis of learned skill representations shows that PASD adapts effectively to diverse partner behaviors, highlighting its robustness in human-AI collaboration." What analysis shows this? Figure 3? This shows that your method switches skills more (which doesn't indicate being more adaptive) and that your method maintains a higher entropy of switching. 
- The size of the plots (e.g. Figure 3) make them really hard to read.


[1] https://github.com/HumanCompatibleAI/overcooked_ai/tree/master/src/overcooked_demo

[2] Cross-environment Cooperation Enables Zero-shot Multi-agent Coordination

### Questions
- not sure that the overcooked domain is sufficiently rich for an HRL method. What kinds of skills are you learning? You show no demonstration or visualization of the kinds of skills your method is learning. [1] suggests that the space of skills for coordination is quite small in overcooked environments.
- why should we care about how well a partner evaluates across different levels of skill or diversity? Even if we do, why would you do the mean across all of these? 


[1] Cross-environment Cooperation Enables Zero-shot Multi-agent Coordination

### Soundness
3

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
4

### Summary
The paper introduces PASD, a Hierarchical RL method for Skill Discovery/Learning for Human-AI Collaboration. The authors argue that previous skill learning methods are agent centric and fail to capture information about the partner conditioned dynamics in the Multi-Agent Cooperative setting. The authors that proposes a Contrastive Loss-like shaped reward term to the objective of both the high and low level policies to maximize mutual information between trajectory segments sampled from the same skill. They then evaluate PASD on Overcooked-AI and compare it with FCP, HiPT and DIAYN.

### Strengths
- The method presented is well motivated and presents a principled approach to skill learning in the Multi-Agent Cooperative/ZSC setting.
- The paper is well written and structured.

### Weaknesses
- Unfortunately the experimental section of the paper is a faily weak at the moment
   - Lack of evaluations against real human partners. This is a major omission in the experimental section considering that the paper is proposing a method for Human-AI Collaboration. Furthermore, previous works (Carroll et al., Strouse et al. and Loo et al.) all conducted experiments with real human partners.
   - Limited evaluation partners. The authors evaluate only on one type of SP partners (TrajeDi plus past checkpoints) when there are a few other diverse partner generation methods (MEP, CoMeDi and HSP). 
   - Lack of qualitative analysis of skills. Though the authors provide some analysis of the learned skills in term of skill switching frequency and overall entropy. It is unclear is the skills learnt by PASD have any significant behavioural difference. It would to interesting to see some visualisations of different skills learnt by PASD in Overcooked.
- Minor typo Line 227 “collectoing” → “collecting”
- In Table 2 what does “CoSkill” refer to? Is that supposed to be PASD?

### Questions
- What do the skills learned by PASD look like in Overcooked?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work presents an HRL algorithm that is aware of other agents in cooperative MARL settings by introducing an intrinsic reward based on a contrastive metric to prevent skill collapse. The algorithm is evaluated on Overcooked to highlight its strength over prior HRL work.

### Strengths
- The intrinsic reward is well motivated and sound.

### Weaknesses
- Overcooked (v1) is a bad evaluation environment for this paper (see "OvercookedV2: Rethinking Overcooked for Zero-Shot Coordination").
  - The results reported in this paper underperform the naive IPPO baselines (and state-augmented IPPO baseline) reported in the OvercookedV2 paper (for Overcooked-v1)
  - OvercookedV2 already demonstrates that there is no zero-shot coordination challenge in Overcooked aside from state coverage
  - Since Overcooked-v1 is fully observable, an LSTM is unnecessary
  - Hierarchical RL in general is unnecessary for Overcooked, since it can be quickly solved with standard IPPO
  - OvercookedV2 would be a better environment to validate your results on, but I still have the concern that HRL unnecessarily complicates the learning process.

### Questions
Why is the intrinsic reward also applied for training low-level policies?

### Soundness
1

### Presentation
2

### Contribution
1
