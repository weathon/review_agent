# Multi-Agent Guided Policy Optimization

- Decision: Accept (Poster)
- Scores: 4, 4, 6

## Abstract
Due to practical constraints such as partial observability and limited communication, Centralized Training with Decentralized Execution (CTDE) has become the dominant paradigm in cooperative Multi-Agent Reinforcement Learning (MARL). 
However, existing CTDE methods often underutilize centralized training or lack theoretical guarantees. 
We propose Multi-Agent Guided Policy Optimization (MAGPO), a novel framework that better leverages centralized training by integrating centralized guidance with decentralized execution. 
MAGPO uses an autoregressive joint policy for scalable, coordinated exploration and explicitly aligns it with decentralized policies to ensure deployability under partial observability. 
We provide theoretical guarantees of monotonic policy improvement and empirically evaluate MAGPO on 43 tasks across 6 diverse environments. 
Results show that MAGPO consistently outperforms strong CTDE baselines and matches or surpasses fully centralized approaches, offering a principled and practical solution for decentralized multi-agent learning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes Multi-Agent Guided Policy Optimization (MAGPO), a framework for cooperative multi-agent reinforcement learning that unites centralized training with decentralized execution. MAGPO uses a centralized autoregressive guider for coordinated exploration while keeping it aligned with decentralized policies, ensuring deployability. It offers theoretical guarantees of monotonic improvement and shows state-of-the-art results across 43 tasks, outperforming CTDE baselines and matching fully centralized methods.

### Strengths
The paper’s main strength lies in its conceptually coherent integration of centralized and decentralized learning in multi-agent reinforcement learning. By introducing the Multi-Agent Guided Policy Optimization (MAGPO) framework, it provides a principled way to leverage centralized guidance during training while maintaining decentralized deployability.

### Weaknesses
I have several concerns regarding this study, as summarized below.

1. The paper argues that the autoregressive guider policy alleviates the exponential growth of the joint action space. However, this sequential modeling still requires conditioning on all preceding agents’ actions and therefore may not scale well in the number of agents. The computational and communication costs of this approach under large agent populations are not analyzed, and all experiments are limited to small- or medium-scale settings (mostly under ten agents). Consequently, the claim of effective scalability remains largely qualitative and lacks empirical validation on truly large-scale MARL problems.

2. MAGPO’s approach to addressing policy asymmetry by regularizing the centralized guider to stay close to decentralized learners via a KL constraint (parameterized by δ). This seems to be an intuitive mechanism but not a principled solution. The theoretical guarantee of monotonic improvement holds only in the tabular and fully observable setting, leaving its applicability under partial observability and function approximation uncertain. In practice, the δ constraint appears to serve mainly as a tunable heuristic, and the paper provides limited analysis of how sensitive performance is to its choice. Thus, while MAGPO may empirically reduce the imitation gap, it does not theoretically resolve the structural mismatch between centralized and decentralized policy spaces.

3. The theoretical analysis of MAGPO assumes full observability, effectively reducing the problem to a centralized Markov game. While this simplification allows for a clean monotonic improvement proof, it undermines the practical relevance of the result. Realistic multi-agent systems operate under partial observability, where theoretical guarantees often break down due to non-Markovian local histories. Without an extension of the theory to Dec-POMDPs, the presented result is largely of limited practical utility and does not substantiate the paper’s claims of theoretical robustness under decentralized execution.

4. The role of the “guider backtracking” step (resetting the guider to the learner policy each iteration) is not well justified empirically. While it appears necessary for the theoretical monotonic improvement proof, this requirement arises only under restrictive assumptions (tabular, fully observable setting). In practice, backtracking may slow learning by erasing the guider’s accumulated coordination knowledge. The paper would benefit from an in-depth ablation study or clarification on whether strict backtracking is essential for stability, or if a softer synchronization mechanism could yield faster convergence (and better performance) without violating practical stability.

5. The addition of the RL auxiliary loss in the learner update is a pragmatic design choice rather than a theoretically required one. It helps the learner recover from the imitation gap when the guider’s centralized behavior is not perfectly realizable under decentralized observations. However, the motivation is somewhat heuristic. The paper includes a limited ablation (Fig. 5a) demonstrating that λ influences performance, but offers no systematic sensitivity or interaction analysis. In particular, the relationship between λ and the guider–learner KL constraint (δ) remains unexplored. The effect of λ is treated empirically rather than analytically, leaving uncertainty about the stability and generality of this design choice.

6. The experimental results are broad and methodologically sound, showing that MAGPO is a competitive CTDE framework. However, the improvements over strong baselines such as MAPPO, HAPPO, and Sable are often moderate rather than decisive. While MAGPO demonstrates robustness and conceptual coherence, the evidence does not clearly establish a substantial empirical leap over other SOTA methods. Particularly, MAGPO’s performance sees to heavily depend on the guider’s backbone (Sable vs. MAT) and the δ hyperparameter. This makes the results appear fragile, strong when tuned, weaker otherwise. This paper doesn’t present qualitative examples of improved coordination, communication efficiency, or emergent behaviors. So I don’t really see how MAGPO achieves its advantage.

### Questions
How well does the proposed autoregressive guider policy actually scale to large multi-agent systems, given that it still requires sequential conditioning over agents’ actions and has not been empirically validated beyond small- or medium-scale settings?

Since the monotonic improvement proof assumes full observability and tabular settings, how meaningful are the theoretical guarantees for realistic partially observable environments, and can the authors extend or justify the theory’s applicability to Dec-POMDPs and function-approximation cases?

To what extent are the key algorithmic components, including guider backtracking, the δ constraint, and the RL auxiliary loss (λ), essential for achieving stable learning, and how sensitive is MAGPO’s performance to these design choices and to the selection of the guider backbone (e.g., Sable vs. MAT)?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work proposes MAGPO, a Centralized Teacher with Decentralized Students (CTDS) approach for multi agent RL, that leverages a sequentially executed guider for joint coordinated exploration while constraining it to be close to the decentralized learner policies. The authors identify and aim to address two problems with previously proposed CTDS approaches: (i) training a centralized agent raises scalability issues; and (ii) under partial observability, the students may fail to replicate the teacher's behavior. MAGPO addresses both the scalability and policy asymmetry issues by learning a centralized, autoregressive guider policy that is constrained to remain aligned with decentralized learners policies. The authors prove monotonic improvement of MAGPO under tabular settings. The authors provide experimental results comparing MAGPO against CTCE and CTDE methods, as well as a vanilla implementation of on-policy CTDS, across several domains.

### Strengths
The paper features a good discussion of related work in Sec. 2.2. The paper seems to do a good job motivating the research gap it aims to address. The experimental protocol seems solid, with the authors performing multiple runs and reporting confidence intervals. The experimental results seem reproducible. The experiments are also extensive, considering several different environments. The authors provide an ablation study. The work appears to feature a sufficient degree of novelty, even though I am not very familiar with previously proposed CTDS approaches.

### Weaknesses
I think the clarity of the paper needs to be further improved in some places. While the experimental results seem to support that the proposed method outperforms pervious baselines, I'm not sure whether the included baselines are well-representative of the previously proposed works; namely, I wonder if the authors are properly comparing their method against previously proposed *CTDS* methods for MARL (see my questions for additional information regarding this matter). This is my main point of concern regarding the article.

### Questions
- I think the authors should better emphasize since the beginning that they are focused on *autoregressive* policies. The authors should also note that autoregressive policies do not entirely cover the space of all possible joint policies.
- 176: "Agents act sequentially, observing previous agents’ actions before choosing their own." - I guess the authors wanted to say that the actions of the agents are jointly selected, right? Why are the actions *sequentially* selected? In centralized execution what matters is that they are *jointly* selected, even tough in practice some methods may sequentially select them. And, if the actions are jointly selected, then why the agents cannot solve this task in the first place? Even if the actions are sequentially selected, why is the third agent selecting 3 instead of 4 prior to the update?
- line 222: How is this KL distance minimized? I guess we minimize it under a given state distribution, right? Under which state distribution? I think this step needs to be further explained in the context of this article.
- Eq.2: I believe this equation needs to be better explained in the context of this article. Does this hold for any state $s$? I think $Q^{\mu_k}$ is the Q-function for policy $\mu_k$, but this is undefined in the article. D_KL is the KL-divergence? Maybe the authors could talk a bit more about Policy Mirror Descent in the background section of the work. I can see that later the authors specify the updates, but I believe the discussion should be clearer in the text close to eq. 2.
- I suggest the authors to find a way to group the different classes of methods in the plots in some way (so that it is easier to understand which class of methods, e.g., CTDE, CTDS, etc., a given line corresponds to). For example, CTCE can be plotted with a dashed line, CTDE with a solid line, etc. Because I think this could make it easier to interpret the plots.
- My key question regarding the experimental results is: as far as I understood, this work in not the first proposing a CTDS approach to MARL. Thus, I would expect the authors to also compare their method (MAGPO) against other CTDS approaches. I can see the authors compare against a CTDS approach, but does this baseline represents well the previously proposed approach for CTDS? Why not including other previously proposed CTDS methods as baselines?

Minor comments/typos:
- line 56: "The guider policy allows agent to act sequentially conditioned on (...)" - agent -> the agents?
- line 155: "We compare three MARL frameworks:" is duplicated.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Multi-Agent Guided Policy Optimization (MAGPO), which follows the Centralized Training with Decentralized Execution (CTDE) paradigm. MAGPO alternates between improving a centralized guider policy and distilling it into a decentralized learner policy. After each distillation step, the guider is reset to match the learner so that the guider remains factorizable and the imitation gap does not grow. Empirically, MAGPO consistently outperforms strong CTDE baselines and matches or surpasses fully centralized training approaches.

### Strengths
Proposes a simple and conceptually elegant algorithmic framework for multi-agent RL.

Demonstrates strong empirical performance across benchmark tasks.

### Weaknesses
It is not clearly explained how the guider (teacher) is reset to the learner (student) in practice. While in theory a product of decentralized policies defines an autoregressive joint policy, the mechanism of implementing this reset may be nontrivial depending on the network architecture.

### Questions
How is the guider actually reset to the learner in implementation? Is this performed by zeroing out the action-conditioning parameters? Please clarify the exact procedure.

### Soundness
3

### Presentation
3

### Contribution
3
