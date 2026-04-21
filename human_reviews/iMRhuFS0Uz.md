# Mildly Constrained Evaluation Policy for Offline Reinforcement Learning

- Avg Score: 5.25
- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 6, 5

## Abstract
Offline reinforcement learning (RL) methodologies enforce constraints on the policy to adhere closely to the behavior policy, thereby stabilizing value learning and mitigating the selection of out-of-distribution (OOD) actions during test time. Conventional approaches apply identical constraints for both value learning and test time inference. However, our findings indicate that the constraints suitable for value estimation may in fact be excessively restrictive for action selection during test time.
To address this issue, we propose a \textit{Mildly Constrained Evaluation Policy (MCEP)} for test time inference with a more constrained \textit{target policy} for value estimation. Since the \textit{target policy} has been adopted in various prior approaches, MCEP can be seamlessly integrated with them as a plug-in. We instantiate MCEP based on TD3-BC [Fujimoto and Gu, 2021], AWAC [Nair et al., 2020] and DQL [Wang et al., 2023] algorithms. The empirical results on D4RL MuJoCo locomotion and high-dimensional humanoid tasks show that the MCEP brought significant performance improvement on classic offline RL methods and can further improve SOTA methods. The codes are open-sourced at link.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper focuses on the trade-off between value estimation stability and performance improvement caused by in offline RL. The authors propose a new algorithm by introducing a new mildly constrained policy to obtain both stable value estimation and good evaluation performance. The authors test their method on D4RL mujoco tasks to verify its effectiveness.

### Strengths
- The paper is clearly written and easy to follow.
- The trade-off between stability of value learning and policy improvement is important and not well-studied by previous work.
- The experiment part adequately explains how the policy constraint influences the evaluation and policy evaluations, which validates the motivation of this paper.

### Weaknesses
- The authors claim that a mild-constrained evaluation policy improves the final performances, but its effectiveness is questionable.
    - The improvement may be attributed to the policy constraint strengths of original method are not well selected. E.g., in fig.4, the performance of TD3BC on hopper-m can achieve >80 with $\alpha=10$. If we use this value as the baseline, then the improvement of the proposed method is actually limited. This also happens on other two settings plotted in fig.4. Meanwhile, if the original policy constraint strengths are suitable, a milder constraint in MCEP may actually degrade the performances (e.g., TD3+BC on medium-expert tasks).
    - For DQL and DQL-MCEP, there are no remarkable differences on most tasks. So why MCEP is effective on some baselines but helps little on others?
    - Based on the above analysis, we can find whether the additional evaluation policy improves the performances heavily depends on the strength of policy constraint in original baseline. MCEP can achieve better with better hyper-parameter, which is actually infeasible in offline RL setting, however.
    - Although the authors give an ablation study in sec. 5.4, the improvement of MCEP is not very significant and the results seem to be inconsistent with previous figures and tables (See questions).
- The authors only test their methods on mujoco tasks. To comprehensive verify the advantages of proposed method, more experiment results (e.g., on maze/kitchen/adroit) are needed. 

Minor issues:
- There are two `3)` in the first paragraph of sec.5.
- In the first paragraph in page 9, $\\tilde{\\alpha}, \\tilde{\\lambda}$ instead of $\\tilde{alpha}, \\tilde{lambda}$.

### Questions
- Which level are you using in fig.7? If it is "-medium", why are the performances of TD3BC with $\alpha=2.5$ and TD3BC-MCEP very different from the values reported in table 1? The hyper-parameters should be the same.

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies the policy constraint methods in offline reinforcement learning. It takes an interesting idea: decoupling the constraint strength for stable value estimation and for policy learning. Specifically, they find that while we need a restrictive policy constraint to mitigate extrapolation error in value estimation, a milder constraint is allowed for policy learning. Thus, apart from the target policy used in actor-critic learning of standard offline RL, a mildly constrained evaluation policy (MCEP) is proposed to be separately learned with a more relaxed policy constraint. This paper instantiates MCEP with existing offline RL method TD3+BC, AWAC, and DQL and demonstrates improved performance.

### Strengths
1. An ingenious idea of decoupling constraint strengths for stable value estimation and for policy learning. Two kinds of distributional shifts, namely OOD actions during Bellman bootstrapping and deployment, to my knowledge, are rarely distinguished in offline RL literature. 
2. The design of MCEP is simple and general for policy constraint methods.
3. Challenging humanoid tasks are introduced in the experiments, and MCEP demonstrates good performance.
4. The visualization of the toy example in Figure 2 illustrates the motivation well.

### Weaknesses
1. The most important finding of this paper, i.e. the difference between policy constraint strengths for stable value learning and for a performant evaluation policy, is validated empirically but lacks a theoretical analysis.
2. The improvement of MCEP upon DQL is limited, which doubts the benefits of MCEP for modern offline RL methods with better designs, such as ReBRAC.
3. The proposed MCEP is only applicable to policy constraint methods and does not outperform other kinds of sota offline RL methods, such as MCQ and EDAC.

### Questions
1. What do you mean by 'While the target policy may recover its performance by iterative policy improvement and policy evaluation, we observe that the evaluation policy may fail to do so.'
2. How does MCEP 'overcome this drawback' (state-agnostic constraint) discussed in Singh et al.?
3. As the authors claim in the introduction that the toy maze experiments 'validate the finding of (Czarnecki et al., 2019)', I recommend also mentioning Czarnecki et al. in Section 5.1.

Typos: 

- Section 4.3 DQL WITH MCEP should be a paragraph in parallel with TD3BC with MCEP and AWAC with MCEP. 
- $C\left(\pi_{\beta, \pi^E}\right)$ should be $C\left(\pi_{\beta}, \pi^E\right)$ in the Equation (10)
- There seem to be some explanation sentences missing after 'We next introduce the policy improvement step for the evaluation policy' and Equation (10).
- The caption of Figure 6 is incorrectly the same as that of Figure 7.

### Soundness
4 excellent

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In policy constraint offline reinforcement learning (RL) algorithms, it is a common practice that the constraints for both value learning and test time inference are the same. This paper argues that such a paradigm may hinder the performance of the agent during test time inference. To address this issue, they propose the Mildly Constrained Evaluation Policy (MCEP) for test time inference. The idea is quite simple and the implementation is also easy. MCEP has the same objective function as the policy trained during the offline phase, but it does not participate in the policy evaluation phase. The authors show that by doing so, the performance of the offline RL agents can be improved.

### Strengths
# Strengths

This paper is generally well-written, and the logical flow is clear. I would say this paper is also well-motivated and proposes an interesting test-time inference algorithm. The resulting method is very simple, and the authors provide some figures and a toy example to illustrate to the readers the key idea behind their method, which I personally like very much. The authors combine their method with three off-the-shelf offline RL algorithms, and conduct some experiments on the D4RL locomotion datasets. The authors also conduct experiments on the Humanoid datasets, where the authors collect the corresponding static datasets by themselves. One can observe performance improvement by building MCEP upon numerous base algorithms. To summarize, the strengths and the advantages of this manuscript are

- this paper is well-written with a clear logic flow

- the core idea and the resulting method of this paper is quite simple and easy to implement

- the improvements from the proposed method are significant on many base algorithms

- the authors provide source codes, and I believe that the results presented in this paper are reproducible

### Weaknesses
# Weaknesses

I think the submission has the following potential flaws

- (major) limited evaluation. Though the authors combine their proposed MCEP method with three offline RL algorithms, they only evaluate them on locomotion tasks, which are actually simple and easy to get a high return. So, my question is, can the proposed method benefit other domains like antmaze, kitchen, and adroit? These domains are known to be more challenging than the MuJoCo tasks. I strongly believe that the empirical evaluations on these domains are critical to show the effectiveness and advantages of the proposed methods. If the proposed methods fail in these domains, I also expect possible explanations from the authors. This paper feels quite incomplete without the experiments on these domains.

- (major) It turns out that the hyperparameter selection counts in MCEP. Based on the empirical results in Section 5.4, TD3BC-MCEP and AWAC-MCEP are slightly sensitive to the hyperparameters. This may cause issues when using the MCEP in practice. Can the authors further explain this phenomenon and are there any ways that we can get rid of it?

- (minor) inconsistent abbreviation for some of the algorithms, e.g., the authors write TD3-BC in the first few paragraphs while using TD3BC later. This is not a big issue and can be easily fixed, please check your submission for potential similar issues.

I will be happy to update my score if the concerns are addressed during the rebuttal or in the revised manuscript.

### Questions
It seems your method is not restricted to the policy constraints offline RL methods, can your method be applied to value-based offline RL algorithms like CQL? I would expect explanations from the authors if CQL-MCEP fails and underperforms vanilla CQL.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper introduces a new approach to address the problems in offline policy learning (e.g. extrapolation). The idea is to train an extra policy (called evaluation policy) based on the $Q$ function learned from the critic of a standard constrained actor-critic offline method. The idea is that using different constraint weights for the critic and evaluation policy should addresses the trade-off between evaluation performance and stable value estimate.

### Strengths
The paper is easy to follow and well written. As far as I know the idea is novel.

Experiments seem to be well conducted and results are fairly explained.

### Weaknesses
While the idea of the approach is interesting, I think the paper needs a bit more work. My main concern is that the contribution is limited and the results are not super clear and convincing to me.

- For example, given that $\pi_e$ is not involved in the optimization of $Q$, why are you training $\pi_e$ at each step and not only at the end? Training $\pi_e$ at the end will allow to do an analysis of the impact of the constraint. 
- Have you tried to train a greedy policy starting from the recovered Q, similarly to what done in the grid experiment?
- Why haven't you tested other environments in D4RL, eg Antmaze-v0?
- Figures are not readable when printed out. The font is too small.


Typos:

acheive -> achieve

priority. i.e. 

wrong latex commands in page 9

### Questions
See Weaknesses part.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
