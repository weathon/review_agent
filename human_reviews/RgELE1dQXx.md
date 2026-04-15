# Learning to Make Adherence-aware Advice

- Decision: Accept (poster)
- Scores: 5, 6, 5

## Abstract
As artificial intelligence (AI) systems play an increasingly prominent role in human decision-making, challenges surface in the realm of human-AI interactions. One challenge arises from the suboptimal AI policies due to the inadequate consideration of humans disregarding AI recommendations, as well as the need for AI to provide advice selectively when it is most pertinent. This paper presents a sequential decision-making model that (i) takes into account the human's adherence level (the probability that the human follows/rejects machine advice) and (ii) incorporates a defer option so that the machine can temporarily refrain from making advice. We provide learning algorithms that learn the optimal advice policy and make advice only at critical time stamps. Compared to problem-agnostic reinforcement learning algorithms, our specialized learning algorithms not only enjoy better theoretical convergence properties but also show strong empirical performance.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies the problem of learning to give advice to humans assuming humans may take the advice or not according to some underlying adherence level. The paper presents a formal problem definition of the problem, two UCB-style algorithms, and a theoretical analysis with convergence bounds.

### Strengths
1. **Problem formulation:** This paper proposes a new formulation of advising by assuming a fixed human player with a fixed probability of taking advice. This problem formulation to me is novel by considering the human adherence level. 

2. **Theoretical analysis:** The paper adopts a few theoretical analysis frameworks to this advising problem and derives much-improved convergence bounds by leveraging the problem structures. 

3. **Empirical studies:** It is nice to see the algorithm really works on the flabby bird domain although it is simple.

### Weaknesses
My biggest complaint about this paper is its presentation. A few comments are listed below.

1. It is weird that there is no citation at all in the introduction section. The introduction part is also particularly short with the contribution statements even deferred after the related work section. I would strongly encourage the authors to expand the introduction with more detailed explanations and more intuitions. 

2. I think the "**Theoretical reinforcement learning**" section should be expanded a bit, e.g., to be an independent background section. I think this part is the most relevant literature. It would be better if the paper could spend more texts explaining the connection and differences between the existing literature and the current work from a more technical perspective. 

3. It is strange to me that the convergence bounds are introduced even before the algorithms. This makes it difficult for the readers to follow the insights.

4. It is so tough for me to fully follow section 4.2. Equation (5) looks like a magical number to me and I have no idea how it is derived and why such a structure helps improve the bound from reward-free learning. I'm not a theoretical person. Carefully checking every detail of the proof is out of my capacity, so it would be great for me if the paper could explain well the insights of why the structure helps improve the bound for an outsider. 

5. It is stated in the experiment section that two testbeds are considered, including the Flabby Bird domain and the Car Driving domain. However, I cannot find the Car Driving results.

### Questions
I don't have any specific questions. 

I think this paper should be ultimately evaluated according to the strength of its theoretical results, which is out of my expertise. My current rating is based on its presentation but I'm okay to raise the score if other reviewers would champion its theoretical contributions.

### Soundness
3 good

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies a reinforcement learning model that takes into account the human’s adherence level and incorporates a defer option so that the machine can temporarily refrain from making advice. The authors provide novel algorithms based on the principle of optimism in the face of uncertainty, which can learn the optimal advice policy and make advice only at critical time stamps. The authors further present the theoretical guarantee of the proposed algorithms and show their empirical performance.

### Strengths
This work investigates a novel reinforcement learning model taking into account new realistic factors, including the human 's adherence level and the defer option. The corresponding algorithm design and the theoretical analysis are novel to reinforcement learning. The empirical studies also verify the algorithms' performance. In addition, the paper is well-structured, and the main idea of this work is easy to follow.

### Weaknesses
(1) The major concern about this work is that there is a gap between the result for $\mathcal{E}_1$ in Algorithm 1 and that for $\mathcal{E}_2$ in Algorithm 3. The dependence on $S$ is typically more important than $H$ in the reinforcement learning problem. As claimed by the authors, the sample complexity bound for Algorithm 3 is sharper in the dependence on S than for Algorithm 1. But $\mathcal{E}_2$ should be a harder problem than $\mathcal{E}_1$, and thus the sample complexity bound for $\mathcal{E}_1$ is expected to be better. Can the authors discuss why there is such a gap and whether it is possible to modify Algorithm 1 to have a better sample complexity bound matching the one for Algorithm 3 in terms of $S$?

(2) Can the authors point out which part of the supplement is for the theorem and the corresponding proof of Algorithm 3? It appears not explicitly presented in the paper, although the authors have shown its sample complexity result in the main text.

(3) Eq.(1) is not clearly explained, especially the third case in Eq.(1). Can the authors provide some more explanation on how the probability of the third case is obtained?

### Questions
Please see the sections above.

### Soundness
3 good

### Presentation
2 fair

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
This paper formalizes the setting of a static policy taking advice from another policy as an MDP. The authors use this abstraction to study the setting of a human taking advice from a machine in order to perform better on a decision-making task. Crucially, this MDP assumes the human (i.e. the static policy that receives machine advice) only follows the advice (i.e. a recommended action) with a probability $\theta$ per state. The paper then 1. introduces UCB-based algorithms to solve the case where the environment dynamics and human policy are know, but $\theta$ is unknown, and 2. adapts reward-free exploration methods to solve the case where the dynamics and advice-taking policy are additionally unknown, whereby the MDP is modeled as a CMDP. The experiments focus on a toy environment based on Flappy Bird and two toy, hard-coded policies ("Greedy" and "Fixed"). In this latter setting, the authors show their reward-free expoloration method, RFE-$\beta$ can successfully provide advice to improve the performance of the two fixed policies.

### Strengths
- This paper deals with the important topic of human-AI collaboration, a topic that is largely overlooked by the greater ML community in favor of fully-autonomous approaches. In particular, advice-taking is an important setting of human-AI collaboration, and the formalization of this setting as a CMDP, while straightforward, presents an important step toward making progress on this important problem. 
- Overall, the presentation is highly legible and the key ideas are explain clearly.

### Weaknesses
While this paper studies an important topic, the paper should be improved before acceptance to a top-tier conference like ICLR:
- The experimental setting is extremely simplistic. The Flappy Bird MDP effectively consists of just 2 actions (up or down). The exact layout of MDP also appears fixed. Likewise, the advice-taking policies are fixed as two hard-coded policies. Effectively, the problem then reduces to learning a policy to solve 2 static MDPs with very small action spaces. Ideally the study can look at Flappy Bird under a procedurally-generated setting as well as look at other environments with higher-dimensional observation and action spaces.
- It would also be ideal to include an environment with continuous action spaces.
- The experimental setting also buckets all states into 2 coarse groupings of adherence levels for the advice-taking policy. In practice, the adherence levels are likely more complex and also time-varying. It would improve the paper to include experiments with a co-adapting advice-taking policy.
- In extending this work to higher-dimensional environments, the authors should consider comparing their approaches to existing deep RL methods for exploration [1,2] as baselines.

**Minor comments**
- In the related works section, the authors should include citations to the line of works studying human-AI cooperation in Hanabi: 
    - Bard, Nolan, et al. "The hanabi challenge: A new frontier for ai research." Artificial Intelligence 280 (2020): 103216.
    - Foerster, Jakob, et al. "Bayesian action decoder for deep multi-agent reinforcement learning." International Conference on Machine Learning. PMLR, 2019.
    - Hu, Hengyuan, et al. "Off-belief learning." International Conference on Machine Learning. PMLR, 2021.


**References**

[1] Ciosek, Kamil, et al. "Better exploration with optimistic actor critic." Advances in Neural Information Processing Systems 32 (2019).
[2] Osband, Ian, et al. "Deep Exploration via Randomized Value Functions." J. Mach. Learn. Res. 20.124 (2019): 1-62.

### Questions
- Does this method handle higher-dimensional environments or continuous action spaces?
- How does this method handle more complex adherence functions?
- How does this method handle co-adapting advice takers?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
