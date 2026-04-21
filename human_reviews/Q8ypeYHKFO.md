# SafeDiffuser: Safe Planning with Diffusion Probabilistic Models

- Avg Score: 3.33
- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 3, 1

## Abstract
Diffusion model-based approaches have shown promise in data-driven planning. Although these planners are typically used in decision-critical applications, there are yet no known safety guarantees established for them.  In this paper, we address this limitation by introducing SafeDiffuser, a method to equip probabilistic diffusion models with safety guarantees via control barrier functions. The key idea of our approach is to embed finite-time diffusion invariance, i.e., a form of specification mainly consisting of safety constraints, into the denoising diffusion procedure. This way we enable data generation under safety constraints. We show that SafeDiffusers maintain the generative performance of diffusion models while also providing robustness in safe data generation. We finally test our method on a series of planning tasks, including maze path generation, legged robot locomotion, and 3D space manipulation, and demonstrate the advantages of robustness over vanilla diffusion models.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
Diffusion models have been a successful generative modelling mechanism in image synthesis. Many works have made creative extensions to use diffusion models for planning and trajectory synthesis. Most of the previous work doesn’t explicitly guide the diffusion process to respect hard safety constraints. Inspired by CBF’s and classical theorems known about barrier functions, this paper proposes to project the diffusion dynamics to improve safety constraints over the course of the diffusion process. Three formulations: robust safe diffuser, time-varying safe diffuser and relaxed safe diffuser are defined with the latter two fixing a critical issue with the first approach. The foundational idea of the contribution has mathematical backing and it is possible to prove from Nagumo's theorem that every point of the trajectory will eventually satisfy constraints. Experiments are performed on maze solving, robot locomotion and manipulation to demonstrate and compare with previous baselines.

### Strengths
Creative, impactful and sound contribution. As a result, I am leaning towards an acceptance score.
Additionally, I found the extensions of the Robust Safe-Diffuser to time-varying and relaxed Safe-Diffuser to be quite interesting. It is believable that the most basic formulation would get stuck in local traps as it is never allowed to violate safety. Additional deeper insights on this phenomenon are appreciated.

### Weaknesses
1) For the experimental comparisons, a detailed description of all the baselines would be useful.

2) I have some questions about the sufficiency of the baselines and comparisons. See below.

3) Would be nice to have a result on one more setup but it is not a major drawback.

### Questions
1) https://arxiv.org/pdf/2211.15657.pdf For example here, a one hot encoding is used to guide the diffusion process to satisfy constraints. Is it possible to compare to this? This is different from classifier-based guidance as we are conditioning on a random variable encoding constraint satisfaction.

2) https://arxiv.org/pdf/2205.09991.pdf Is it valid to compare against using diffusion guided by reward or goal-based RL and just use an usual off-the shelf CBF at runtime?

3) With standard RL, there can be one performance reward and another safety reward similar to what is usually done in safe constrained RL. Can something like that be amenable to the diffusion process?

4) Would it be necessary to have a large number of diffusion steps in some cases? 

5) Is there a possibility of trajectories not being consistent with the dynamics of the environment? i.e. the generated trajectories can have several violations of the underlying dynamics of the robot. If yes, are there some metrics quantifying the dynamics validity of trajectories.

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper presents a safe planning algorithm using a diffuser planner. The authors incorporate the safety constraints into planning, which is a preferable solution to the problem. The planning algorithm can potentially suffer from the "local traps" - the phenomenon inherited from the planner, however, authors propose a solution to this problem by relaxing the safety constraints. The algorithm is demonstrated on the maze and robot locomotion tasks.

### Strengths
1. The authors present theoretical results regarding their method
1. The authors develop the theory of safe diffusion planners
1. They demonstrate the behavior of the safe algorithm on the maze and the robot locomotion tasks.

### Weaknesses
1. Some of the theoretical developments and derivations are confusing. For example, the control barrier functions are developed in continuous time, while planning is done in discrete time. 
1. There are some issues with notation. For example, it appears that $u$ denotes a control signal for the continuous time systems as well as the diffusion process. 
1. It is not clear how the dynamics / one-step transitions are used in the planner. Perhaps I missed it, but I couldn't understand it at all. 

Overall the methodology is not clear to me, but perhaps authors can clear this up

### Questions
1. Why continuous dynamics and CBF for continuous if in practice it is discrete? This introduces extra estimation errors. 
1. The description of the local trap problem is slightly confusing for me. Could the authors elaborate on what is happening in more detail? Is the problem inherent to the diffusion planner?

### Soundness
1 poor

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
1: strong reject

### Rating Number
1

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper recognizes the importance of establishing safety guarantee in the application of diffusion process planning, and introduces three safe diffusers from the perspective of control theory and control barrier functions. Particularly, this paper attempts to address the local traps during diffusion procedure.

### Strengths
1. It's an interesting perspective to work around with finite-time diffusion invariance to guarantee safety constraint in the diffusion models.
2. Propose three safe diffusers and work through the theorems related to each of them respectively and demonstrate they achieve decent results in the experiment section.

### Weaknesses
1. Experimental design is not clear. The details of how time-varying weight $w_k$ or time varying function $\gamma_k$ are not revealed. I believe they have to be carefully chosen to guarantee the effect of the safe diffusers and ablation studies are needed here. More ablation studies are needed when determine number of diffusion time steps to reduce the computation time while still maintaining the model performance. 
2. (a) Some typos in the text: e.g., 'Cassifier-guidance' in the description of Table 1. (b) The ``ELBO'' in the tables should be negative. (c) Some typos in the theorems proof: e.g., proof of theorem 3.2, at the very bottom of page 12, "Replace $b(x_k^j) - \gamma(N, \epsilon)$ by $V(x_k^j)$'', $b$ and $\gamma$ should be switched over. (d) In theorem 3.2, All equations and inequalities should have labels. Some equations and statements should have period at the end but not comma everywhere.

### Questions
1. The figure 3 compares the samples between classifier-based guidance and time-varying-diffuser. It is hard to believe that the classifier-based guidance is much worse than the time-varying diffuser at diffusion time steps 4 and 3. Do you have some idea why it stays so noisy at the very end of denoting steps but become noiseless at time step 0 (ignore the constraint violation)? Also, why classifier-base guidance failed to identify the constraint and steer the trajectory away from the obstacles?
2. I have a hard time to understand what definition 2.3 Control Barrier Function really means and tries to delivery. Can you give an intuitive explanation for it? Or some geometric meaning?
3. In the proof of theorem 3.2, I don't really understand why $\frac{d(b(x_k^j) - \gamma(N, \epsilon))}{dt} + \epsilon(b(x_k^j) - \gamma(N, \epsilon))\geq 0$ (the third inequality) ``is equivalent to the last equation (inequality)''. What is exactly diffusion time $T$ here? I know $\gamma$ is a robust term, but what's the justification of its appearance?
4. In the experiments, what are their corresponding robust function $\gamma(\cdot)$, extended class function $\alpha(\cdot)$? Can you revel more details about them? What does $r$ represent in equation 14?

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair
