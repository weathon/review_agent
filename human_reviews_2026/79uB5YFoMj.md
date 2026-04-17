# Practical Adversarial Attacks on Stochastic Bandits via Fake Data Injection

- Decision: Reject
- Scores: 2, 6, 2, 2

## Abstract
Adversarial attacks on stochastic bandits have traditionally relied on some unrealistic assumptions, such as per-round reward manipulation and unbounded perturbations, limiting their relevance to real-world systems. We propose a more practical threat model, Fake Data Injection, which reflects realistic adversarial constraints: the attacker can inject only a limited number of bounded fake feedback samples into the learner’s history, simulating legitimate interactions. We design efficient attack strategies under this model, explicitly addressing both magnitude constraints (on reward values) and temporal constraints (on when and how often data can be injected). Our theoretical analysis shows that these attacks can mislead both Upper Confidence Bound (UCB) and Thompson Sampling algorithms into selecting a target arm in nearly all rounds while incurring only sublinear attack cost. Experiments on synthetic and real-world datasets validate the effectiveness of our strategies, revealing significant vulnerabilities in widely used stochastic bandit algorithms under practical adversarial scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a new attack model in MAB called Fake Data Injection (FDI). Unlike the standard threat model, in FDI, the attacker can not arbitrarily manipulate the feedback, and cannot attack in every round, which is closer to the real world scenario. Attack strategies are given for MAB and TS algorithm, and there are theoretical bounds of the trade-off between attack cost and attack performance.

While the motivations make sense to me, I don't feel that all of them are precisely reflected through the proposed attack model, and I'm not able to fully differentiate it from the existing attack model. Another limitation is that the attack is specific and customized to MAB and TS algorithms. For details please refer to ``weakness''.

### Strengths
1. The attack model takes more constraints and is closer to real-world application.

### Weaknesses
1. My main concern is the applicability of this model, because it needs to keep track of number of pulls. I don't think this can be known to the attacker in the real world?
2. Attack is limited to and customized for MAB and TS. In practice, it's infeasible to know exactly the recommendation algorithm. Can we extend this to, at least a class of algorithms?
3. Around Line 179, seems to me that the success of attack is still defined in terms of the amout of total corruption. In other words, whether $N^F$ is large or not seems not to matter, which seems inconsistent with real-world motivation. In this case, how is this different from the standard successful attack definition in the literature?
4. Is it possible to examine the performance of some existing corruption-robust algorithms? If not, what is the challenge? (This is just for curiosity, and I do NOT ask for this during the rebuttal)

Would like to increase the rating if these are addressed by the author(s)

### Questions
1. In Sec. 4.2.2, how would $f$ and $R_i$ affect the tradeoff in Thm. 4.4?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper studied fake data injection attacks on UCB and Thompson sampling algorithms. Different from prior works that allow attacker to perturb the reward in every step, this paper considered a more practical attack scenario where the attacker can only inject fake data in certain steps to guarantee sparseness and boundedness. The paper starts a motivating attack that injects a single but very negative reward to reduce empirical mean of non-target arms, and then extended this attack by distributing this big attack to multiple rounds and periodic stages. The paper provided theoretical analysis to show that even under sparse and bounded attack scenario, the attacker is still able to force the bandit to always pull some target arm while incurring only sublinear attack cost.

### Strengths
The paper extended traditional bandit attacks to a more realistic setting, where the attacker can only inject fake data in limited number of steps. This kind of attack is more stealthy and less likely to be detected.

The paper derived solid theoretical results to show the effectiveness and efficiency of the proposed attack algorithms.

The paper performed empirical study of the attacks on real-world data to show that the performance of the attack.

Overall, the paper is sound, solid, and provided a comprehensive investigation of the problem.

### Weaknesses
Fake data injection, although more stealthy, may make the data suspicious due to the injection. Traditional attacks does not alter the actions of the bandit learner, but only changes the reward. However, fake data injection causes a change in the action selected by the bandit algorithm. For example, if a batch of fake data all have the same action and are injected, then the learner may suddenly receive a set of data with exactly the same action, but the learner only chooses to take that action once. This data definitely result in an alarm and the learner may realize that it is being attacked. From this perspective, the fake data injection might as well be more suspicous.

### Questions
What if the bandit algorithm finds fake data injection suspicious? How to address it and how to justify fake data injection attack is more realistic and stealthy?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This work studies adversarial attacks on stochastic multi armed bandits (MABs).
In prior works, the adversary was only allowed to perturb the rewards obtained by the algorithm for the arm is actually picked during the online process.
In this work, the authors introduce a new notion for adversarial attacks on MABs called Fake Data Injection where the adversary is allowed to inject any arm and reward pair to MAB algorithm and algorithm considers it as if it chose the arm and updates it's beliefs. 
The authors claim that this attack model is more realistic as it can allow the adversary to attack only in limited rounds and also attack any arbitrary arm instead of just the arm chosen by the model. This could simulate situations such as fake reviews or fake engagement for any specific arm/content without the algorithm even recommending that arm in the currrent round. The authors also consider that the adversary cannot manipulate in every round but only a small fractions of rounds.

The authors consider attacks on UCB and Thompson sampling. 

First, the authors consider an unbounded reward model and show that if the adversary can add unbounded corruptions, then it can make the algorithm pick the target arm in all but $o(T)$ rounds using only $\tilde{O}({K \sigma \sqrt{\log T}})$ noise.

Then the authors consider the case of bounded corruptions, and show that under certain assumptions on the bounds of the corruption, the algorithm from unbounded setting can be updated to make multiple bounded attacks every time it needs to attack to get similar bounds on the corruption to ensure the model selects the target arm in all but $o(T)$ rounds.

The authors then consider a followup where the attacker can only inject atmost $f$ attacks in one go show how the attack can be adopted for the setting.

The authors also provide experiments where they simulate the setup on MovieLens25 dataset and show that their attack methods do work for UCB and with limited attack cost, the algorithm can indeed be tricked into selecting the target arm for most rounds.

### Strengths
- The paper introduces and interesting model for data corruption attacks where the adversary can attack any arm
- The consider the setting where the attacker has constraints on the bound of the rewards
- They have experiments using MovieLens 25M showing the effectiveness of the algorithm

### Weaknesses
- My main concern for this paper is that I believe the main results are not true and there is a bug in the analysis.

In the proof for Theorem 4.1. Let's consider any specific arm $i$, since each arm is corrupted once, the cost of corrupting $i$ according the definition of the corruption is $| r_i - \mu_i|$ where $r_i$ is the corrupted reward.

To ensure that after the corruption, eq (1) holds, we need
\begin{align*}
    & \hat{\mu}(t+1) \leq \hat{l}_k(t) \\
    \implies & \frac{\hat{\mu}_i(t) N_i(T) + r_i}{N_i(T) + 1} \leq  \hat{l}_k(t) \\
     \implies & r_i \leq \hat{l}_k(t) (N_i(T) + 1) - \hat{\mu}_i(t) N_i(T)
\end{align*}

Now consider corruption $| r_i - \mu_i|$. Since we are attacking the arm, it's obvious that we are trying to reduce it's mean, so $r_i - \mu_i$ is infact negative and $|r_i - \mu_i| = \mu_i - r_i$ not $r_i - \mu_i$

So we have
\begin{equation}
     r_i \leq \hat{l}_k(t) (N_i(T) + 1) - \hat{\mu}_i(t) N_i(T)
\end{equation}
and 
\begin{equation}
    C =  \mu_i - r_i
\end{equation}
Combining the two we get 
\begin{equation}
    C \geq \mu_i + \hat{\mu}_i(t) N_i(T) - \hat{l}_k(t) (N_i(T) + 1)
\end{equation}
This is completely opposite of what is claimed in the proof for Theorem 4.1 in Appendix Section A.2.2

- I don't agree with the claim that this model has more restrictions as it allows corrupting only a fraction of rounds not all. In the algorithm design, all the algorithms proposed by the authors are also just arbitrarily corrupting whenever they want. In fact I believe this problem is simpler as the learner does not have to analyze whether the arm would be picked by the algorithm or not and can inject noise whenever and wherever. 

- I don't agree with the claim that earlier works only considered the unbounded reward setting. Xu et all 2021 consider rewards bounded between 0 and 1. Moreover in the bounded case, the authors assume an upper bound on the lower limit of the boundary which is not well motivated.

### Questions
Can the proofs be fixed?
Can you formally compare the difficulty of this attack model with the existing model studied where we are only allowed to corrupt the arms picked by the algorithm?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studied the adversarial attacks on stochastic bandits. It designed and analyzed attack strategies which aim to prevent superior performance of classical TS and UCB algorithms and presented some numerical results.

### Strengths
The paper begined with an introduction, discussed some related works, proposed attack strategies which are accopanied by theoretical results and experimental results.

### Weaknesses
My main concern is about the motivation and hence contribution of this work. The author(s) **grounded the work on the statement that the existing works are with unrealistic assumptions on adversarial attacks**. The details are listed in the second paragraph in Section 2.2 and my corresponding concerns are as below:
1. 'Attacker can perturb the environment-generated reward in every round.' --- It is incorrect. Some existing works, such as \url{https://link.springer.com/article/10.1007/s10994-018-5758-5}, assume that the attacker can only attack a fraction of time steps, instead of all time steps.
2. 'The attacker modifies the reward of only the selected arm' --- It is incorrect for some existing works. Some papers such as \url{https://proceedings.mlr.press/v99/gupta19a/gupta19a.pdf} assume that the agent pull and arm only after attacks, in which case it is impossible for the attacker to only attack the selected arm.
3. 'The pre-attack reward and the attack values are unbounded' --- It is incorrect for some existing works. Bounded-reward case is usually a much easier one and there has been many works on that. Again, check the results in \url{https://proceedings.mlr.press/v99/gupta19a/gupta19a.pdf}

Based on my concerns above, 
1. I am a bit confused why this fake-data-injection setting is interesting to explore and what is its difference from the existing setting. It seems to me that the amount of injection is similar to the amount of corruption.
2. Moreover, the numerical/analytical performance of proposed strategies should be compared to existing ones, which I failed to find in the manuscript.
Overall, I suggest the author(s) to discuss more related works and convince the readers the necessity of this formulation and contribution of this work.

### Questions
In addition to my concerns raised in the **Weaknesses* section, I suggest the author(s) to be careful about citation formatting. For instance, in line 30, there should be '(Li et al., 2010)' instead of 'Li et al. (2010)'.

### Soundness
1

### Presentation
3

### Contribution
1
