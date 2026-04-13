## Human Reviewer 1

### Summary
This paper studies a tendency in existing GCRL methods with goal relabelling that prioritises optimisation toward closer achieved goals. To mitigate this bias, the authors propose an actor-critic objective without goal relabelling, incorporating a KL-divergence constraint towards a novel prior policy. This prior policy ensures that the learned policy behaves consistently towards both the final desired goal and intermediate subgoals. In the experiments, the authors demonstrate that the proposed method achieves strong performance with fewer environment interactions, although the experimental design remains contentious.

### Strengths
1. The motivation behind the proposed method is insightful and articulated clearly, providing a strong rationale for the approach.
2. The method developed from this motivation is based on a promising and well-supported assumption.

### Weaknesses
The main concern with this paper lies in the experimental design:
1. Section 6.1: It lacks comparison with suitable baselines (see Question 5).
2. Section 6.2: This section appears redundant, as GCQS's advantage in sample efficiency is already demonstrated in Figure 5 and Section 6.1.

Some minor issues are noted in the questions section.

### Questions
1. The definition of Q-BC should be clarified upon its first appearance (line 80).
2. There is a typo in line 312-313 - 'policy objective that reaching achieved goals'.
2. Section 2 discusses GCWSL methods like GoFar (Ma et al., 2022) and WGCSL (Yang et al., 2022) without mentioning their offline nature. Yang et al. prove the theoretical guarantees in the offline goal-conditioned setting—do these guarantees still hold in the online setting?
3. In Section 5.2, the relabelled (achieved) goal $g'$ is redefined as subgoals $s_g$, but both $g'$ and $s_g$ are used in the text, which causes some confusion. Would it be clearer to consistently use the notation $g'$ throughout the paper?
5. The comparison in Section 6.1 appears unfair, as some baselines, like Actionable Models, WGCSL, and GoFar, are designed for offline goal-conditioned RL and may not perform as well in the online setting.

**References**

Ma, J. Y., Yan, J., Jayaraman, D., \& Bastani, O. (2022). Offline Goal-Conditioned Reinforcement Learning via f-Advantage Regression. NeurIPS.

Yang, R., Lu, Y., Li, W., Sun, H., Fang, M., Du, Y., … Zhang, C. (2022). Rethinking Goal-conditioned Supervised Learning and Its Connection to Offline RL. ICLR.

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
5

### Confidence
3

---

## Human Reviewer 2

### Summary
This paper works on goal-conditioned reinforcement learning tasks and introduces the idea of leveraging subgoals in hindsight relabeling to improve learning efficiency. The proposed method is mainly heuristic, and it is justified with theory on the performance. Empirical results are provided to support the method.

### Strengths
The authors appropriately used illustrative figures to explain some concepts in their paper. Motivating examples are provided to demonstrate the problem of previous approaches. The experiments are conducted on multiple tasks and results are compared with multiple baselines.

### Weaknesses
There are several main weaknesses in the current paper:

1. on the high level, the necessity of the proposed method is not clear. I understand the key insight the authors aimed to convey using Figure 2. However, from my perspective, isn't this a *property* rather than a *problem* of hindsight relabeling? e.g. when the number of steps required to achieve different goals is uniformly sampled from 1-50, then during the learning process, hindsight relabeling using trajectories having a length of 50 will generate a uniformly distributed length from 1-50, and as the learning proceeds, the policy will learn to achieve closer goals, and averaged trajectory length will be smaller than 50 --- this will lead to a compounding effect in increasingly having more shorter paths. 


2. to solve the above challenge, why is the proposed method "necessary"? For instance, why can not one just to re-sample or down-sample on some of the state-goals? In the current write-up, it is not very clear why the multiple designing factors introduced are well-motivated or necessary. 


3. according to the theory, using a smaller $\eta$ will lead to a tighter bound, does this mean that we need the $\eta$ to be as close to 0 as possible?

4. In Figure 5, the averaging/smoothing method for different figures seems different, and some results lack error bars. The FetchSlide performance is noticeably lower than what has been achieved in the literature.
From my perspective, it is not necessary to demonstrate a method is always better than baselines in all experimental cases. What is more important is to clearly show what are the realistic motivations for introducing the techniques/designing components of the paper, and then use experiments to highlight --- the settings where those challenges exist, the method shines and on the other hand, when those challenges (controllably) alleviated, the performances of different methods would converge. 


Minor:

5. On presentation. In Figure 1, it would be great if the authors could give some concrete examples and make sure the notations are self-consistent. I acknowledge the authors' effort in explaining their method using Figure 1 and it could be a great idea if the clarity can be further enhanced. For instance, what is "QBC" objective in this figure, what does it mean by "subgoals come from achieved goals", and why the policy can take either s and g or s and s_g as its inputs?

### Questions
please refer to weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
5

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper presents GCQS, which utilizes achieved goals as subgoals and iteratively updates by considering behavior cloning schemes for prior policy and KL-constraint for policy.

### Strengths
This paper is generally well-written, easy to follow, and provides promising results on some benchmark problems.
The paper also highlights neglected aspects, such as the issue of short trajectory usage in subgoal generation, and presents theoretical support for the proposed objective.

### Weaknesses
The lack of comparison with other benchmark problems, such as complex AntMaze tasks presented in [1, 2], where the original HER showed significantly degraded performance compared to similar methods such as PIG, raises questions about whether omitting high-level planning for subgoal generation is valid for other complex and long-horizon GCRL tasks.

See questions for others.

[1] Kim, Junsu, et al. "Imitating graph-based planning with goal-conditioned policies." arXiv preprint arXiv:2303.11166 (2023).

[2] Kim, Junsu, Younggyo Seo, and Jinwoo Shin. "Landmark-guided subgoal generation in hierarchical reinforcement learning." Advances in neural information processing systems 34 (2021): 28336-28349.

### Questions
(1) This is the question for clarification. What is the key difference between the proposed methods compared to existing methods [1,2] which update subgoals throughout training? Specifically, how does relabelling achieved goals as subgoals differ from landmark or subgoal updates in PIG and HIGL? 

(2) This method discards the necessity of high-level planning for subgoal generation but I'm curious whether GCQS works well in long-horizon tasks such as AntMaze [1,2,3] compared to methods utilizing high-level planning, such as HIGL and PIG.

(3) It is unclear how GCQS addresses the issue of short trajectory updates since HER already conducts a similar relabeling scheme. Which component of GCQS handles this? 

(4) How important is the SAC component in Eq.12 for GCQS? This omitted ablation leaves the question of the importance of the BC term.

(5) Presenting more details on the derivation of Eq. 25 from Eq. 24 would be helpful for readers.

[1] Kim, Junsu, et al. "Imitating graph-based planning with goal-conditioned policies." arXiv preprint arXiv:2303.11166 (2023).

[2] Kim, Junsu, Younggyo Seo, and Jinwoo Shin. "Landmark-guided subgoal generation in hierarchical reinforcement learning." Advances in neural information processing systems 34 (2021): 28336-28349.

[3] Lee, Seungjae, et al. "Cqm: Curriculum reinforcement learning with a quantized world model." Advances in Neural Information Processing Systems 36 (2023): 78824-78845.

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
5

### Confidence
3

---

## Human Reviewer 4

### Summary
This paper studies hindsight relabeling in goal-conditioned reinforcement learning (GCRL). The key finding is that previous GCRL methods have a bias towards closer achieved goals during training, which results in the learned policy being less aligned to reach long-term goals. To address this issue, this paper proposes a novel GCRL method, GCQS, which first learns a goal-reaching policy and then uses the KL-divergence between the learned policy to final goals and goal-reaching policy to subgoals along the trajectories to guide policy optimization for reaching the long-term goals. This method enjoys a performance guarantee and demonstrates better performance than state-of-the-art GCRL baselines.

### Strengths
1. The proposed method shows strong empirical results.
2. The method is well-motivated.

### Weaknesses
Major questions:

1. If prioritizing the closer achieved goals is the main issue of previous GCRL methods, can we simply use data augmentation to rebalance the relabeling trajectory dataset and improve the performance? I think this would be an interesting result to see and could make the paper more solid.
2. The connection between GCQS and GCWSL: while the authors do not classify GCQS as one GCWSL method, I think there are some similarities to discuss between the two methods. Actually, in offline settings, both Equation 12 and Equation 15 should have a closed-form solution, which is an exponentially weighted form of $\pi_{relabel}$ and $\pi^{prior}$, as discussed in [1]. This makes me wonder if GCQS is still a GCWSL method (like exponential advantage weight mentioned in WGCSL [2]) and if the closed-form solution could further improve the method.
3. Several baselines in the experiments are offline methods, so I am wondering how authors compare them with other online baselines and GCQS to make sure the comparison is fair.

Minor points:

1. The cumulative function should be defined in Theorem 4.1, as it is slightly different from the commonly used probability cumulative function.
2. I believe Section 6.1 has a typo: 16 CPUs -> 16 GPUs.

Despite the questions mentioned above, I am recommending weak acceptance due to the motivation and strong performance of the paper at this stage, however I hope the authors can address my concerns.

[1] Nair, Ashvin, et al. "Awac: Accelerating online reinforcement learning with offline datasets." *arXiv preprint arXiv:2006.09359* (2020).

[2] Yang, Rui, et al. "Rethinking goal-conditioned supervised learning and its connection to offline rl." *arXiv preprint arXiv:2202.04478* (2022).

### Questions
There are some questions and concerns, which I have outlined in the previous section.

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
8

### Confidence
4