# On Stationary Point Convergence of PPO-Clip

- Decision: Accept (poster)
- Scores: 3, 6, 8

## Abstract
Proximal policy optimization (PPO) has gained popularity in reinforcement learning (RL). Its PPO-Clip variant is one the most frequently implemented algorithms and is one of the first-to-try algorithms in RL tasks. This variant uses a clipped surrogate objective function not typically found in other algorithms. Many works have demonstrated the practical performance of PPO-Clip, but the theoretical understanding of it is limited to specific settings. In this work, we provide a comprehensive analysis that shows the stationary point convergence of PPO-Clip and the convergence rate thereof. Our analysis is new and overcomes many challenges, including the non-smooth nature of the clip operator, the potentially unbounded score function, and the involvement of the ratio of two stochastic policies. Our results and techniques might share new insights into PPO-Clip.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper provides a theoretical analysis of the convergence of a variant of the popular PPO algorithm where the objective function is clipped.

### Strengths
The paper provides formal guarantees for the convergence of the clipping variant of PPO that has been used widely by reinforcement learning practitioners. This probably adds some reassurance that that this is a sound RL method to use.

### Weaknesses
The paper appears to be a good contribution to the filed of RL, but has little, if anything, to do with representation learning. ICLR'24 might not be the best venue for this paper.

Some minor typos:
Last paragraph on page 1: "this analysis rely" -> "this analysis relies"
Same place: "no longer involve" -> "no longer involves"

### Questions
Can you think of a possible impact your result can have on the field of representation learning?

### Soundness
3 good

### Presentation
3 good

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
This work takes a closer look at the theories behind the clipped surrogate objective of PPO (Proximal Policy Optimization). The authors provide a comprehensive analysis that proves the stationary point convergence of PPO-Clip and demonstrates the convergence rate.

### Strengths
The work is novel in that it investigates the theoretical convergence of PPO, whereas the field has overwhelmingly focused on the empirical performance and application of PPO (e.g., Dota 2 with PPO and ChatGPT, RLFH with PPO). This is partially because the clip operation is inherently non-smooth, thus posing challenges to empirical analysis. The authors' analysis seems sound and comprehensive; they also clearly listed out the necessary list of assumptions. The authors' Theorem 3.2. indicates "PPO-Clip has the same convergence property as the best current results available for policy gradient", which seems significant.

### Weaknesses
Please take this with a grain of salt, as I have primarily been using PPO under empirical settings. I struggle to understand how this work connects with the wider research community. What is the implication of this work? This work demonstrates PPO has stationary point convergence — can this property be used in some ways?

### Questions
> the unbounded score function makes the ratio of two policies arbitrarily large, even in the late stages of the optimization process.
Do the authors mean the ratio **could** become arbitrarily large?

### Soundness
4 excellent

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper presents theoretical analysis of convergence of PPO algorithm using clipping and version of two-time scale method, i.e., updating policy parameter with particular period. Analysis of PPO-clip is difficult due to non-smoothness of clipping operator and involves ratio of policies. Theorem 3.3 provides best iterate and averaged iterate convergence in terms of $||\nabla V(\theta_{n,1})||\to 0$.

### Strengths
- The paper tackles important question regrading theoretical analysis of practical implementation of PPO algorithm. The aim of the paper is clear and simple. 

- The analysis seems to be novel. The authors use two-time scale method to overcome the difficulty to deal with ratio of the policy and constructing particular set of events enables to derive recursive inequalities to bound the norm of gradient of the clipped loss function.

### Weaknesses
- Even though the aim of this paper is to tackle theoretical analysis of practical implementation of PPO, still there is some gap. Using a inner and outer loop style update seems to be far from practical implementation.
-  The estimate for advantage function, $\phi_n$ can be estimated with parameterized value network (e.g. neural network). However, I believe extending the proof to Actor-Critic setting is non-trivial. Hence, I believe the proof is setting restricted to Monte-Carlo setting. 
- Assumption 3.4. restricts the generality of $T$ and the learning rate. Providing simple examples on the learning rate, and $T$ would be helpful to understand the conditions about Assumption 3.4. For example, can we use $\frac{1}{k}$ or $\frac{1}{\sqrt{k}}$ as the step-size?

### Questions
- Above the paragraph of Step 1, what does it mean to have similar recursive inequality like policy gradient? Please provide more details.

- In Step 1, how does the estimated clipped PPO loss have gradient? Should it be sub-gradient due to the non-smoothness of the clipping operator? Please provide more clarifications.

- What is the intuitive meaning of $X_{n,k}$ and $Y_{n,k}$? The motivation of decomposition of error term  into $X_{n,k}$ and $Y_{n,k}$ is not really clear

- The introduction of $C_{n,k}$ in Step 2 is quite abrupt. Please provide more details.

- In Step 2, what is the meaning of bound of $\mathbb{E}[||X_k|| \mathcal{F} ]$? Depending on $\frac{1}{\sqrt{\pi_{\theta_{n-1,1}}(s,a)}}$ seems to be problematic. How is this term compensated? In deriving (14) where did $\frac{1}{\sqrt{\pi_{\theta_{n-1,1}(a\mid s)}}}$ go?

- I think there is typo in (13). $\nabla V(\theta_{n-1})$ should be $\nabla V(\theta_{n-1,1})$.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
