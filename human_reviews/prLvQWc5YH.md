# Tackling Underestimation Bias in Successor Features by Distributional Reinforcement Learning

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 5

## Abstract
The framework of successor features (SFs) and generalized policy improvement (GPI) yields the potential to achieve zero-shot transfer in reinforcement learning (RL) among different tasks. However, GPI always suffers from inaccurate value function approximation in practice, resulting in a ``zero-shot'' somewhat fantastical. This paper focuses on comprehending the underlying causes of inaccurate SFs and presents a methodology for improving their accuracy. Our contributions encompass four key aspects: (i) we theoretically study the underestimation phenomenon in SF\&GPI; (ii) we introduce distributional RL into SF\&GPI, and demonstrate its effectiveness in relieving such underestimation; (iii) we show that distributional SFs (DSFs) is provided with a lower generalization bound than original SFs; (iv) we put forward that the performance of SFs-based algorithms can be enhanced by incorporating DSFs. Furthermore, we verify the quality of employing DSFs on the platform of multi-objective RL (MORL). Simulation study demonstrates the superiority of our concept in addressing underestimation challenges.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In this paper, the authors identify an issue with underestimation bias when employing successor features with generalized policy improvement (GPI). They apply distributional RL to successor features to obtain a distributional GPI that they claim alleviates this issue.

### Strengths
* Connecting successor features and multi-objective RL. These two concepts are intimately related, yet I’m unaware of any published work making a direct connection. 
* Identifying the issue of underestimation with SFs. The intuition behind this makes sense but I found the explanation hard to follow.

### Weaknesses
Overall, I found the paper hard to follow as the math is a little too loose to make sense of what the authors are claiming. I'd suggest the authors go over the text with a careful eye and fix some of the notational issues. I’ll detail the major weaknesses and ask more direct questions about notation and other issues in the questions section below.

* I believe they are learning the wrong object with their proposed distributional SF algorithm. They should be learning the features' joint distribution, but they treat each feature as independent. This is implicitly done in how the author's sample $\tau$; to learn the proper distribution, you'd need the quantile of a multivariate random variable.
* The paper is poorly positioned in the literature regarding multi-dimensional reward functions in distributional RL. My above point on learning the wrong object has been solved by [1] and [2], where they learn the correct joint distribution. Furthermore, I expected a more in-depth discussion about [3] as they also learn “distributional SFs” (although they also aren’t learning the joint distribution).
* Assumption 1 seems dubious; this should be impossible with stochastic transitions; a single application of the Bellman operator will construct a mixture distribution, so, at the very least, you’d expect the target to be a mixture of Gaussians. 
* No justification is given for the additive noise model (Equation 6).
* It’s hard to judge the effectiveness of the approach as the empirical results don’t differentiate much between distributional GPI and regular GPI. I would have had to see a better-executed empirical study to be convinced.

---

[1] Pushi Zhang, Xiaoyu Chen, Li Zhao, Wei Xiong, Tao Qin, Tie-Yan Liu. Distributional Reinforcement Learning for Multi-Dimensional Reward Functions. NeurIPS 2021.

[2]  Dror Freirich, Tzahi Shimkin, Ron Meir, Aviv Tamar. Distributional Multivariate Policy Evaluation and Exploration with the Bellman GAN. ICML 2019.

[3] Michael Gimelfarb, Andre Barreto, Scott Sander, and Chi-Guhn Lee. Risk-Aware Transfer in Reinforcement Learning using Successor Features. NeurIPS 2021.

### Questions
- In Section 3.1, the prediction $\Psi(s', a', \theta_i)$ should be $\Psi(s, a, \theta_i)$? Only the TD target should have the next state-action term. This error is propagated from this point forward, e.g., the gradient term is wrong, and eq (5) contains the same error.
- In Section 3.1, why is there an expectation over s’ in the loss? Aren’t we trying to write down the stochastic approximation algorithm for learning SFs via TD? Citing equation 1 makes it seem like that’s what we’re trying to do.
- Theorem 1, why is $s’$, $a’$ defined as input to $\Delta$ but then $s'$ appears in the expectation? Also, why do we have an expectation over $s'$ again?
- Theorem 3 compares the expected return of the optimal policy with a risk measure of the estimated policy. Why? If you're being risk-sensitive, the goal is not to learn the mean-optimal policy.
- Algorithm 1, where is $\tau_e$ being used? Shouldn't it be used in Line 6 when computing the greedy action?
- Algorithm 1, quantiles are treated implicitly in some cases; this makes it hard to decipher what's going on.

### Soundness
1 poor

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In this work, the authors aim to address underestimation when using Successor Features and Generalized Policy Iteration (GPI). A common technique often used to prevent overestimation in the Q-values is by using the min operation with double Q-functions, which may in turn result in underestimation. Motivated by this insight,  the authors rely on theoretical analyses to show a similar trait is observed when updating the parameters of the successor features. This is induced by a mismatch of the parameters of the successor features between one that depends on a changing policy distribution and the other being the optimal policy. The authors proposed replacing successor features with its distributional form in order to limit or reduce the underestimation bias.

### Strengths
I have to be honest. It is difficult for me to identify what to write with regards to the strength of the paper. I can see that a lot of work has been done. However, the presentation and writing is not doing justice to the authors’ effort.

### Weaknesses
1. First of all, the writing and the overall presentation is not very clear and does not flow well at times. Despite reading the paper a couple of times, it still seems confusing and overly complicated. Lastly, some of the sentences in the paper do not even make sense and makes me wonder if they were generated by LLMs. Here are some examples: 
  a. “Explosively, we take an impressive TRL method - successor features (SFs) (Barreto et al., 2017; 2018; Carvalho et al., 2023) as an example, to study the underlying overestimation/underestimation bias.” 
  b.“They enrich our concepts mutually.”
  c.“Extensive quantitative evaluations support our analysis.”
2. It seems that the research question about addressing underestimation was motivated by RL/ But at the moment, I fail to understand the need of using risk-sensitive frameworks and multi-objective RL. What is the main motivation for considering these frameworks and theories? Furthermore, the lack of clarity from the section on bridging successor features and multi-objective RL does not help the cause.
3. Eq 4. Is y the target that you are regressing towards? It is confusing if that is not the case. 
4. It is very hard to read the paper when there are a large portion of different concepts and their corresponding theorems and equations. I would recommend moving most of these items into the appendix and use the main portion of the paper to explain what these different concepts are and how they are related to the research question that you are attempting to address. You can also move the pseudocode for Algorithm 1 into the appendix. This will also allow you to make more space for the section for conclusion and discussion. 
5. The overall paper structure should be re-visited. The fact that a whole chunk of related work is in your appendix is a missed opportunity for the readers that they can follow along. 
6. Although the author did provide the theoretical proofs showing the existence of the underestimation bias in the SF & GPI framework, this point will make a stronger case with empirical evidence as well.

### Questions
1. What is the purpose of analyzing using the risk-sensitive framework? 
2. What is the purpose of considering multi-objective RL which only further complicates the study?

### Soundness
2 fair

### Presentation
1 poor

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper theoretically studies the underestimation phenomenon in successor features and generalized policy improvement. The paper introduces distributional RL into the SF/GPI framework so as to mitigate underestimation and theoretically analyzes its generalization bounds. The experiments are run on multi-objective RL environments (testing transferabiity with GPI) in Mujoco. They compare the performance of their distributional variants to the standard SF/GPI variants.

### Strengths
The paper does indeed seem to show theoretically that the underestimation phenomena occurs in SF & GPI.

The paper does indeed seem to show that distributional SFs has a lower generalization bound than the original SFs.

### Weaknesses
My primary concerns have to do with clarity, as well as well as the experimental results.

There are many typos or awkward wording, including some that impact the reader's understanding.

Some examples include:
- "The results indicate that both the two DSFs-based algorithms (RDSFOLS and DGPI-WCPI)". The latter isn't even in Figure 2? I am assuming the latter refers to "WCDPI+DGPI", but this should be clarified.
Awkward wording:
- "resulting in a “zero-shot” somewhat fantastical"
- "For a new task w_{n+1}, it is practicable to evaluate all policies". Perhaps you mean practical?
- "standpoint to expose the mystery of underestimation". The word 'mystery' seems akward here.
- "Due to the disorder of exploration": I don't know what "disorder of explanation refers to"
- "is lack of stability"
- "DSFs exhibits" -> "exhibit"
- "We remark that δ_φ > 0 makes no focus". I didn't understand what was meant by "focus".
- "if the set of DSFs enough close"

There are many more beyond what was mentioned, and this does indeed negatively impact the readability of the paper.

The results in Figure 2 do not appear to be very compelling. It seems the proposed method does not significantly outperform the baselines.

### Questions
- Can you describe again the y-axis in Figure 2?
- Did you look at Q-value predictions of the agents to show/demonstrate lower underestimation?

### Soundness
3 good

### Presentation
1 poor

### Contribution
2 fair
