# A Unified Framework for Reinforcement Learning under Policy and Dynamic Shifts

- Avg Score: 5.75
- Decision: Reject
- Scores: 8, 6, 6, 3

## Abstract
Training reinforcement learning policies using environment interaction data collected from varying policies or dynamics presents a fundamental challenge. Existing works often overlook the distribution discrepancies induced by policy or dynamics shifts, or rely on specialized algorithms with task priors, thus often resulting in suboptimal policy performances and high variances. In this paper, we identify a unified strategy for online RL policy learning under diverse settings of policy and dynamics shifts: transition occupancy matching. In light of this, we introduce a surrogate policy learning objective by considering the transition occupancy discrepancies and then cast it into a tractable \textit{min-max} optimization problem through dual reformulation. Our method, dubbed Occupancy-Matching Policy Optimization (OMPO), features a specialized actor-critic structure and a distribution discriminator. We conduct extensive experiments based on the OpenAI Gym, Meta-World, and Panda Robots environments, encompassing policy shifts under stationary and non-stationary dynamics, as well as domain adaption. The results demonstrate that OMPO outperforms the specialized baselines from different categories in all settings. We also find that OMPO exhibits particularly strong performance when combined with domain randomization, highlighting its potential in RL-based robotics applications.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work proposes a unified framework to address distributional shifts induced by policy changes and/or dynamic variations. This is done by firstly leveraging the concept of Transition Occupancy Distribution (TOD), which augments the state-action occupancy distribution with the next states (which “accounts” for the dynamics of the MDP). Then, the work develops a surrogate policy learning objective based on the TOD, and further reformulates such objective in a minimax optimization problem, ensuring that all terms are tractable. Finally, the work proposes a practical learning algorithm, OMPO, which implements this learning objective as an actor-critic paradigm. The work presents experiments in several continuous control environments, showing superior performance in all settings that accounts for policy or dynamic distributional shifts.

### Strengths
- The work approaches the challenge of handling distributional shifts in RL, which is very relevant and spawn different active areas of research. Therefore, it is well motivated.

- The proposed framework is very sound and elegant. It unifies and generalizes several challenges from Reinforcement Learning, often approached by different RL subareas (off-policy/offline RL, meta-RL, multi-task RL, sim-to-real transfer), into a single and generic learning objective.
    - It also empirically shows that OMPO can provide superior performance to several algorithms that are specialized in a single type of distributional shift, which provides strong evidence of the effectiveness from the theoretical framework and practical algorithm.

- The experimental setup is very complete. It brings all combinations of shifts (stationary environment, domain adaptation and non-stationary environments), which progressively cover all scenarios of shifts. For each case, it compares against solid and recent baselines. It also provides visualizations of the transition occupancy distributions (which explicit the problems addressed in the paper), as well as ablations for the main hyperparameters in OMPO.

- The Appendices are also rich and improve the clarity of the paper. For instance, they detail the experimental setup, hyperparameters, provide pseudocode, contrast with prior DICE methods, etc. Hence, the work looks very reproducible and mature.

### Weaknesses
While I do not have major concerns, I believe the paper could be improved in some directions, as detailed below:

- I believe the proof of Proposition 3 could be clearer and more didactic with a better explanation on the assumptions and on why some steps are taken. Perhaps starting with an initial “rationale” behind the proof (describing the strategy to be followed) would be helpful. 
    - In fact, this could be also extended to the Section 4.2, which would help clarify some of the steps taken to arrive in the tractable learning objective (see my question below).

- First, I believe the work provided sufficient empirical evidence to support the proposed framework. Nevertheless, one question remains: does OMPO scale for harder problems? For instance, does it work in meta-RL benchmarks such as Meta-World ML10, ML45? If not, why?

- Following the previous point, the paper does not well describe the limitations of the proposed technique. It only states the challenges of tuning the buffer size but does not bring more practical information. For instance, I am curious to know about the stability of the method and how sensible it is for other learning parameters (such as those on Table 2). Additionally, it would be interesting to describe the computational resources needed to run the method for the presented benchmarks.


**Minor Concerns:**

In Section 3, while describing the MDP, I believe the paper refers to the initial state distribution by two different symbols ($\mu_0$ and $\rho_0$). Based on the rest of the paper, I believe that the $\rho_0$ should be replaced with $\mu_0$. 


**Summary of the Review:**

The work provides a strong theoretical contribution by providing a framework that unifies addressing policy and dynamics shifts. The empirical part also provides a good support to the presented algorithm. The raised concerns are minor, and I recommend acceptance. Nonetheless, I am also stating medium confidence, as I do not have profound familiarity with DICE-related literature.

### Questions
- In the beginning of Section 4.2, the work motivates the dual reformulation due to the presence of the distribution induced by the current policy in the historical dynamics (which is unknown). Then, the paper includes the Bellman flow constraint, arriving at equations 7-8, which still depends on this unknown distribution. Could you please better justify this step? From my understanding, this seems needed to arrive at the tractable objective in Equation 12, but it is unclear if there is another justification.

- Figure 3: Would it not be possible to combine Domain Randomization with DARC? In some environments, DARC is a stronger baseline than SAC, and their combination could match or outperform OMPO-DR.



======================================== **POST-REBUTTAL** =========================================


After carefully checking other reviews and authors’ responses for all reviews, I understand that the paper improved during the rebuttal in many directions and, personally, addressed almost all of my concerns. The paper improved in clarity (more didactic and transparent in the proofs, besides new discussions) and richness in the experimental methodology (new ablations, longer runs).

I still believe that it is not clear whether OMPO would scale for more complex distributional shifts and harder environments. Nevertheless,  considering the proposed scope of this paper, I do not believe this is a major concern, as the current experiments are satisfactory to validate the proposed method. I am more confident that this paper is ready for acceptance. Therefore, I am raising my confidence (3 -> 4).

### Soundness
4 excellent

### Presentation
3 good

### Contribution
4 excellent

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper introduces a unified framework to tackle diverse settings of policy and dynamic shifts by performing transition occupancy matching, leading to a surrogate policy learning objective that can be cast into a tractable min-max optimization problem through employing dual formulation. Moreover, this paper conducts extensive experiments to demonstrate the efficacy of the proposed method under different policy and dynamic shifts.

### Strengths
1. The proposed unified framework performs consistently well in diverse settings with different policy and dynamic shifts. Notably, the authors claim they use the same set of hyperparameters for all experiments in the paper, making the results rather impressive.
2. The paper tackles the issue of policy and dynamic shifts, an important problem in deploying RL policy in real-world applications. 
3. The paper is clearly written and easy to follow.

### Weaknesses
1. The experimental evaluations under the Stationary environments setting can be improved. For example, 1M environment steps are not usually enough when evaluating on the `Humanoid` task. I would suggest the authors provide the results of their OMPO when training for more than 2.5M environment steps and at least compare with SAC on `Walker2d`, `Ant`, and `Humanoid`.

2. The authors should provide more intuition to explain why their OMPO outperforms the baseline methods under the Stationary environments setting. Is it a consequence of incorporating $R(s, a, s')$ into the training loss? Do the authors also employ the double-Q technique for OMPO or only use a single Q?

3. The pseudo-codes provided in Algorithm 1 should provide more training details. For example, calculating Eqs (23) and Eqs (25) requires sampling $s_0$ from the distribution $\mu_0$. How does the proposed method perform this operation exactly? I suggest the authors provide more training details, at least in the appendix.

### Questions
1. The proposed OMPO enjoys a low variance across different random seeds in terms of performance given stationary environments, as shown in Figure 2. Can the author provide some insights into this phenomenon? 

2. Eqs (23) and Eqs (25) minimize the $Q(s, a)$ specifically for $s\sim\mu_0$. Since $Q$ is parameterized by a neural network, the  $Q(s, a), s\sim\mu_0$ can be minimized spuriously low. How do the authors combat this potential training stability?

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper presents a unified framework, called Occupancy-Matching Policy Optimization (OMPO), for reinforcement learning under policy and dynamics shifts. The authors identify the challenges posed by these shifts and propose a surrogate policy learning objective that captures the transition occupancy discrepancies. They then formulate the objective as a tractable min-max optimization problem through dual reformulation. The proposed method is evaluated on benchmark environments and compared with several baselines, demonstrating its superior performance in all settings.

### Strengths
- The paper addresses an important problem in reinforcement learning, namely handling policy and dynamics shifts, which are common in real-world scenarios.
- The proposed OMPO framework provides a unified strategy for online RL policy learning under diverse settings of policy and dynamics shifts. The derivation of the algorithm is clear and comprehensive.
- The experimental results demonstrate that OMPO outperforms specialized baselines in various settings, showcasing its effectiveness in handling policy and dynamics shifts.

### Weaknesses
1. The proposed implementation is complicated. It introduces modules such as estimating the density ratio $\rho_T^\pi\left(s, a, s^{\prime}\right) / \rho_{\widehat{T}}^{\widehat{\pi}}\left(s, a, s^{\prime}\right)$ and performing min-max optimization. This can make the training unstable.
2. There is a lack of discussions on some related papers. See Q2.
3. The experiments are not thorough enough. See Q3 and Q4.

### Questions
1. In the related work, why do algorithms that modify the reward function require policy exploration in the source domain can provide broad data coverage? Is it due to the likelihood ratio that serves as the reward modification term? But OMPO also uses the ratio term and requires that the denominator is larger than zero.
2. Papers [1,2] also deals with the issue of dynamics shift and should be included as related works. What is the advantage of OMPO compared with these two algorithms?
3. Regarding the experiments, the change in environment parameters is limited. For example, the gravity in the target dynamics is only twice larger than that in the source dynamics. Is it possible to evaluate the algorithms with a more severe shift in dynamics?
4. How are the experiment settings related to policy shifts? It seems that all changes are made in environment parameters and related to dynamic shifts.

[1] Xue Z, Cai Q, Liu S, et al. State Regularized Policy Optimization on Data with Dynamics Shift. arXiv preprint arXiv:2306.03552, 2023.

[2] Cang C, Rajeswaran A, Abbeel P, et al. Behavioral priors and dynamics models: Improving performance and domain transfer in offline rl. arXiv preprint arXiv:2106.09119, 2021.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this paper, a policy learning algorithm is proposed, that is intended to learn from off policy observations, which also were collected in environments that are different from the target environment, and in particular in non-stationary environments.  

It is argued that differences between environments can be expressed via  the stationary (s,a,s') distribution, which extends the standard state-action occupancy distribution (s,a) by adding the next-sate s'.  A "surrrogate" objective is proposed, that involves averaging over the (s,a,s') distribution given in the data,
which is intended to be analogous to the various DICE type off-policy methods (AlgaeDICE and others) where averaging is over the empirical occupancy (s,a). 

Experiments are performed comparing the derived algorithm  to a number of current model free methods.

### Strengths
The problems of off-policy learning, domain adaptation and non-stationary environments are impotrtant. 
The experiments presented in the paper suggest  performance improvements over the copmeting algorithms.

### Weaknesses
This paper lacks any theoretical justification or proper motivation of the proposed objective. 

It is completely not clear why optimising the proposed objective (5) should yield good performance 
on a new unseen environment, or even on the same env. off policy.  In fact, there are several logical errors in the arguments. 

In more detail:

* The authors propose to construct the occupancy measure $\rho_{T}^{\pi}(s,a,s')$ and assume that observations data has such distribution. However, stantionary measures are generally ill-defined for  
non-stationary envirnoments and it is not clear what this means.  Thus when the authors write $\rho_{\hat{T}}^{\pi}(s,a,s')$, it appears that they must assume the data generating env. $\hat{T}$ is a regular env, contradicting the premise of the paper. 

* Even if $\rho_{\hat{T}}^{\pi}(s,a,s')$ is just computed from a finite data, it is not clear why information in it should be relevant to performance in a new environment. This might be true under some strong assumptions that are implicit, but such assumptions must be discussed. It trivially not true in the general case. 


* Further, even if we only have one environment, it is not clear why the objective (5) should be related to 
the standard objective $\mathcal{J}(\pi)$.  The inequlities in (2) are not tight, except in very degenerate cases. I.e. that gap between the right handside and left handside can be huge even for the optimal policy $\pi^*$. 


* It is generally not clear how specifically the introduction of $s'$ helps performance on the target environment. 




The encouraging experimental results do seem to indicate that there is something intersting about the proposed algorithm. However, a finished paper must provide an understanding of why the improvement happens,  or at least provide minimal theoretical grounding of the methods, both of which are absent from the current paper.

### Questions
Please see above.

### Soundness
1 poor

### Presentation
2 fair

### Contribution
2 fair
