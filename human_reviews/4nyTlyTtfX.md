# Heterogeneous Decision Making towards Mixed Autonomy: When Uncertainty-aware Planning Meets Bounded Rationality

- Decision: Reject
- Scores: 3, 6, 3, 5

## Abstract
The past few years have witnessed a rapid growth of the deployment of automated vehicles (AVs). Clearly,
AVs and human-driven vehicles (HVs) will co-exist for many years to come, and AVs will have to operate around HVs, pedestrians, cyclists, and more, calling for fundamental breakthroughs in AI designed for mixed traffic to achieve mixed autonomy.  Thus motivated, we study heterogeneous decision making by AVs and HVs in a mixed traffic environment, aiming to   capture the interactions  between human and machine decision-making and develop an AI foundation  that enables vehicles to operate safely and efficiently. There are a number of challenges to achieve mixed autonomy, including 1) humans drivers make  driving decisions with bounded rationality, and it remains open to develop accurate models for HVs' decision making; and 2)  uncertainty-aware planning plays a critical role for AVs to take safety maneuvers in response to the human behavior. In this paper,  we introduce a formulation of AV-HV interaction, where the HV makes  decisions with bounded rationality and the AV employs uncertainty-aware planning based on the prediction on HV's future actions. We conduct a comprehensive analysis on AV and HV's learning regret to answer the questions: 1) \"How does the overall learning performance depend on HV's bounded rationality and Av's planning?"; 2) "How do different decision making strategies impact the overall learning performance?"  Our findings reveal some intriguing phenomena, such as Goodhart's Law in AV's learning performance and compounding effects in HV's decision making process. By examining the dynamics of the regrets, we gain insights into the   interplay between human and machine decision making in  mixed autonomy.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper considers heterogeneous decision-making by human-driven and automated vehicles in 2 player game scenarios. The paper introduces two main lines of inquiry. The first hypothesis (H1) is understanding the relationship between learning performance and HV’s bounded rationality and AV’s planning horizon. The second hypothesis (H2) relates the impact of decision–making strategies on the overall learning performance. To evaluate H1, the paper provides a theoretical analysis through a derivation of the upper bound of the regret. To evaluate H2, the authors provide an empirical ablation over hyperparameters.

### Strengths
+ The paper provides a formalization of a problem that has not had extensive theoretical analysis.
+ The introduction shows this has a useful application despite being a smaller setup (2-player).
+ There is a clear description of the method assumptions and limitations throughout the paper.

### Weaknesses
-The main argument of the paper assumes that identifying and formalizing this problem is a significant contribution. 

-Since there are only two agents, there may be methods outside of autonomous driving that may be relevant. [1] may be a good starting point to find such literature.
[1] Natarajan, M., Seraj, E., Altundas, B., Paleja, R., Ye, S., Chen, L., ... & Gombolay, M. (2023). Human-Robot Teaming: Grand Challenges. Current Robotics Reports, 1-20.

-There are many bounds proved, but it is unclear what the intuition is. Is this bound good? Could it be better? The paper is quite difficult to understand the takeaways from these bounds. This weakness would no longer be an issue after clarifying the writing.

### Questions
How should these theoretical results be interpreted? It would be beneficial to the reader to add some intuition regarding whether this bound is tight enough for performance.

What is the main takeaway from the empirical performance? There exists a set of parameters for a setting to minimize the regret, but then how should a practitioner use these results? Can they be extended beyond a 2-player setting or do these hold empirically with humans? Many of these results ask more questions (due to the nature of a paper formalizing a problem). Thus, I would expect much of the paper to discuss the implications of these results, which it is currently lacking.

Some minor points:
The metaphor in line 1 was a bit confusing.

Many of the abbreviations in the introduction should be introduced before using the abbreviated form. What is “NHTSA”, “HV” (only defined in abstract, may do well to just repeat once in the introduction such as Automated Vehicle (AV).

In Section 5, line 2, I believe “Av” should be “AV”

Please link to the proofs in the appendix from the main document.

### Soundness
1 poor

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper delves into the challenges and advancements associated with mixed autonomy in traffic scenarios, where both automated vehicles (AVs) and human-driven vehicles (HVs) coexist. 
The primary objective of the study is to gain insights into the interactions between HVs and AVs by framing these interactions as a process wherein AVs predict the decisions made by HVs. 
Regret analysis serves as a metric for evaluating the learning performance of both systems, and the authors conduct a comprehensive analysis to unveil how the overall system performance is influenced by HVs' bounded rationality and AVs' planning horizon.

### Strengths
The reviewer has limited familiarity with the field of autonomous driving research. Based on the reviewer's best judgment: 

- The paper offers a novel perspective on mixed autonomy, emphasizing the diverse decision-making processes involving both human and machine behaviors. 

- The paper's introduction and problem formulation are well-defined and effectively presented. The paper's structure is logical and facilitates easy comprehension. 

- The formulation of interactions between human-driven vehicles (HVs) and automated vehicles (AVs) appears innovative and of significant relevance. The regret analysis applied to both systems is robust, and it is intriguing to observe how the results reveal the overall system performance's dependence on HVs' bounded rationality and AVs' planning horizon.

### Weaknesses
The study relies on a set of assumptions such as: 

- Both HVs and AVs operate within the same action space;

- HVs plan actions over a shorter time horizon, whereas AVs adopt a longer horizon for decision-making; 

The practicality of these assumptions raises questions. 

The formulation of HV-AV interactions leans on Goodhart’s law to evaluate AV performance and incorporates a model wherein HVs react to AVs based on bounded rationality. At this stage, I am unable to comment on the robustness of such a formulation. 

Some comparative experiments and analysis would strengthen the paper's contributions.  

I will refine my viewpoint upon reviewing the feedback from other reviewers. 


Minor: 

Please use the ICLR2024 template.

### Questions
Could the authors elaborate on the foundational assumptions made for the formulation presented in the paper？

### Soundness
3 good

### Presentation
4 excellent

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper is concerned with a setting where autonomous vehicles must deal with humans that have bounded rationality. Under this scenario, this work provides a theoretical upper bound on the resulting regret of a planning-based autonomous vehicle with a finite planning horizon. The upper bound is then utilized to analyze the impact of various factors (e.g. the human's bounded rationality, the autonomous vehicle's finite planning horizon, the discount rate used during planning, etc.) towards the regret upper bound produced by the human-autonomous vehicle interaction. Finally, the authors claimed that they use their proposed upper bound formulation to analyze the impact of different learning strategies in human-autonomous vehicle interaction.

### Strengths
**Major Strength - Originality - Novelty of Regret Analysis**

To the best of my limited knowledge of the subject, I have not seen a regret analysis in a human-autonomous vehicle interaction where humans have bounded rationality. While the proposed analysis seems to be based on very limiting assumptions, it could provide an important basis for future works in human-autonomous vehicle interaction. Perhaps from a multiagent systems researcher's perspective, the closest work to this paper is from Loftin & Oliehoek (2022), who also explored regret bounds when dealing with adaptive partners (that can be humans). Nonetheless, this work does not account for a human partner's bounded rationality in their theoretical analysis and is limited to fully cooperative settings.

**Major Strength - Clarity**

In general, I find the paper to be clearly written. I especially appreciate the authors pointing out which factors of the human or autonomous vehicle's decision-making each theorem/analysis addressed. That helped the readers to stick with the overall flow of the paper, which is always challenging with papers proposing lots of theoretical analysis.

Citations:

Loftin & Oliehoek. 2022. "On the Impossibility of Learning to Cooperate with Adaptive Partner Strategies in Repeated Games". ICML 2022.

### Weaknesses
**Minor Weakness - Quality - Missing Citations to Related Works**

While not limited to their applications to autonomous vehicles, there are works in multiagent systems concerned with interaction against partners (including humans) whose decision-making processes (which encompass many things, such as their beliefs, goals, policies, bounded rationality, etc.) are unknown. These works can usually be found in the literature on ad hoc teamwork [1], zero-shot coordination [2], or opponent modelling [3].  Following how human-autonomous vehicle interaction can be modelled using the formulation described in these research areas, there should be a reference to these works in the paper.

**Minor Weakness - Quality - Problem Formulation**

The decision to model the interaction as an MDP is also questionable. It specifically goes against the original formulation of MDPs to have the environment output two reward functions (one function each for the human and autonomous vehicle). In this case, would it not have been better to model the problem as a Dec-POMDP (or any other multiagent systems formulation)?

**Major Weakness - Quality - Non-Adaptive Policy Assumption**

As highlighted by the authors themselves, a major weakness of this paper lies in the assumption that human and autonomous vehicle policies do not change/adapt during interaction. This assumption plays an important role in the analysis where teammate actions are assumed to be fixed when establishing the regret upper bounds of the autonomous vehicle's decision-making process. But this, overall, seems like an unrealistic assumption in real-world human-autonomous vehicle interaction.

In general, I highly recommend the authors to consider the concept of adaptive regret [4] in their analysis. Unlike the notion of regret used in this work, adaptive regret accounts for the possible change in a partner's decision-making process during interaction.

**Minor Weakness - Quality - Misrepresented Contribution**

In the final sentence in the introduction, the authors claimed that they investigated the effects of different learning strategies on the learning performance achieved through the human-autonomous vehicle interaction. Unfortunately, the analysis in Section 4 did not exactly achieve that and instead settled with analyzing the effects of an autonomous vehicle's approximation errors on the overall regret. If the authors wanted to stick with the original claim, it would have been better if they formulated different types of learning algorithms and analyzed the effects on the regret bounds (see Section 5 of [4]). Otherwise, the authors should adequately adjust the reader's expectations in the introduction by being more direct about which factors' effects on the regret bounds are analyzed.

**Major Weakness - Significance**

I am concerned with the significance of this work. First, it seems based on highly restrictive assumptions that would not hold in real-world scenarios, where humans will adapt to the autonomous vehicle's policy. While this could have been fine had there been no previous theoretical analysis on dealing with adaptive partners, Loftin and Oliehoek (2022) [4] have introduced concepts for this kind of analysis. To improve the significance of this work for the community, I would expect the authors to build their analysis from the adaptive regret concept [4].

At the same time, there isn't a sufficient analysis of the effects of different autonomous vehicle learning algorithms on the overall regret in this paper. This generally limits the usefulness of this work for other people working on learning algorithms for autonomous vehicles. All in all, I doubt this work can significantly influence future work in this field.

Citations:

[1] Mirsky et al. (2022). "A Survey of Ad Hoc Teamwork Research". EUMAS 2022

[2] Hu et al. (2020). "Other-Play for Zero-Shot Coordination". ICML 2021.

[3] Albrecht et al. (2017). "Autonomous Agents Modelling Other Agents: A Comprehensive Survey and Open Problems". AIJ

[4] Loftin & Oliehoek. 2022. "On the Impossibility of Learning to Cooperate with Adaptive Partner Strategies in Repeated Games". ICML 2022.

### Questions
1. What are the challenges to extending the existing theoretical analysis to an adaptive setting where both human and autonomous vehicle change their policies during an interaction?

2. Can this analysis be extended to other learning strategies that are not model-based?

3. Why is it appropriate to model the system as an MDP? Doesn't the environment output two rewards at each timestep (which goes against the definition of MDPs)?

4. Can you explain why the non-linear model (Section 3) is suitable to model human-autonomous vehicle interaction?

### Soundness
1 poor

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This is a theory paper, analyzing a heterogeneous system of an automated vehicle (AV) and a human vehicle (HV) modeled as a joint MDP with continuous actions. The AV's model of the HV is N-step look ahead optimization plus Gaussian noise. Authors derived upper bounds for the following cases:
1. AV's performance Gap with linear dynamics
2. AV's performance Gap with non-linear dynamics
3. AV's Regret with non-linear dynamics
4. HV's Regret with non-linear dynamics
5. Joint HV-AV's Regret with non-linear dynamics

### Strengths
+ Great theoretical analysis that lays foundation for mixed human and autonomous systems
+ In addition to the upper bound derivation, authors plotted the upper bound for several conditions for better explanation of their impact

### Weaknesses
- Generalizability: Assumptions 1 and 2 can limit the applicability of the theory in practice. While I understand Assumption 2 has been widely used in the past, it would be great to discuss limitations these two assumptions would play in practice. 
- Writing and clarity: there are writing issues with the paper (see minor comments for examples). Also I could not find the values for certain setups (e.g. settings 1-5).

Minor comments:
In definition of Q^π(x,u_A,u_H), the expectation needs \pi,
optimization on an proxy object => on a proxy
Let Let => Let
et al. (2022)Sadigh et al. (2016) => add ,

### Questions
Can elaborate on the limitations of your assumptions in practice?

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good
