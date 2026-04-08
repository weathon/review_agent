## Human Reviewer 1

### Summary
This paper proposes an approach based on the concept of R-learner to estimate the difference of Q-function and subsequently policy optimization in reinforcement learning. The authors provide theoretical guarantees and conduct experiments on both synthetic data and CartPole to support their approach.

### Strengths
- The use of R-learner in causal inference to improve RL convergence rate appears novel. 
- The author provides concrete theory on convergence guarantees. The technical analyses appear solid (Besides some typos, the proof seems sound). 
- Experiments indicate that the proposed methods outperform baselines in terms of the mean square error in estimating differences of Q.

### Weaknesses
1. The main theoretical implications of this paper need to be explained more clearly. The authors mention they proved theoretical guarantees of *improved* convergence rate. It is unclear what works/results this paper is comparing with. Given the large RL literature, the author might need to discuss their improvements over existing approaches/results in much more clarity to position their contributions.

2. Assumption 1 (sequential unconfoundedness) needs justification. While it may be natural for other settings, whether this assumption is proper in RL is unclear. The assumption suggests actions only affect rewards but not the Markovian dynamic, which seems restrictive and also makes the problem potentially much easier.

3. My understanding is that the ultimate goal of the authors' approach is to obtain better RL policies, i.e., estimating difference of Q is only a means towards this ultimate goal. However, the experiments only consider the mean square error in estimating difference of Q. So there's a mismatch between the goal and the experimental results. Moreover, this limits the scope of baseline methods as most methods aim to obtain RL policies other than merely estimating difference of Q.

The above are my main concerns. In addition, the following might help improve the paper:
1. Fix typos, e.g., Eq. (3) $\epsilon(A_t)$ should be $\epsilon_t$, Eq. (4) $A$ should be $A_t$. Terminologies are not consistent, e.g., OrthDiff-Q v.s. $\tau$-TL v.s. DiffQ.
2. Experiments can be described more precisely. For example, in the Misaligned exo-endo experiment, $\beta_{dense}$ is not defined, and it is unclear how the data is generated.

### Questions
1. Does Assumption 1 hold in the numerical examples?
2. Connection with literature: E.g., how is Assumption 1 related to "Exogenous State MDP" in Dietterich et al., 2018? Regarding reward/value estimation, how is the proposed method compared to, e.g., Pan & Scholkopf, 2024, and others?
3. How should we understand or estimate $\alpha$ and $\delta_0$ in Assumption 5? This looks like an assumption for the sake of proof, and it's unclear how they can be understood for a given problem.
4. The cross-fitting part seems computationally intensive. What is the runtime of the experiment?

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
6

### Confidence
2

---

## Human Reviewer 2

### Summary
The paper considers estimation of advantage function in offline RL setting. The authors use ideas from R learners, which was designed for heterogenous effect estimation, and propose an estimator for difference-in-Q function. By introducing three nuisances (Q, m, pi^b), the authors obtain an identifying conditional moments and relevant loss function, of which the proposed estimator is the empirical minimizer. The paper derives the estimation error, policy evaluation and policy optimization.

### Strengths
1. Incorporating ideas from R learner to estimating difference of Q function is interesting. The paper details the required error rate conditions for nuisances estimations for the proposed method to work, which is also technically challenging. 

2. The results presented are extensive, including policy evaluation and optimization.

### Weaknesses
Some suggestions on presentation 
1. Theorem 2 and 3: if the requirement on nuisances rates are the same, then it may be better to just state once. 

2. Theorem 3 is long. It may be better to move some of them out as assumptions and explain. 

3. More generally, the whole paper proceeds by introducing new definition and assumptions, and then stating algorithms / theorems. It may be better to add more explanations about assumptions and motivations.


Typos.
Line 133: u should be superscript, and why do we need X here?

### Questions
1. In theorem 3, is there a reason to expect the $\rho$ to increase?

2. What is the theoretical advantage of R-learning based method over traditional methods in terms of estimating (difference of ) Q function using offline data? Traditionally methods are based on Inverse Propensity Scoring, or its doubly robust variant. These methods can also be used to estimate difference in Q.

### Soundness
3

### Presentation
1

### Contribution
3

### Rating
6

### Confidence
2

---

## Human Reviewer 3

### Summary
The paper brings the tools from orthogonal ML into offline
reinforcement learning, with an orthogonalized estimator for the
"difference in Q function" in the offline policy
evaluation/optimization settings. The high level point is that the
difference of Q functions \tau(s_t) = Q^\pi(s_t,1) - Q^\pi(s_t,0) satisfies
the moment equation:

r_t + \gamma Q^\pi(s_{t+1},a_{t+1}) - E[r_t+\gamma
Q^\pi(s',\pi_b(s')|s_t] = (a_t - \pi_b(1|s_t))\tau(s_t) + \eps

This motivates a strategy where all unknown quantities above except
for \tau are estimated in the first stage (as nuisance parameters) and
then plug-in version of these are used to estimate \tau in the second
stage. This can be combined with dynamic programming for both policy
evaluation and policy optimization, resulting in the two main
algorithms in the paper.

As far as theoretical results, I think they can be summarized (and
simplified a bit) as follows: suppose all the nuisance functions can
be estimated at a n^{1/4} rate, then we can estimate \tau at a n^{1/2}
rate. For policy optimization, this becomes non-trivial because the
nuisance function at time $t$ is a function of the primary estimand
$\tau_{t+1}$, so one must show that this more complex nuisance
function can still be estimated at a fast rate.

Comments:

1. Overall, the paper is quite hard to read, even if you have a
background in both RL and orthogonal ML (as I do). I think the
notation could be improved significantly, and the theorem statements
are very difficult to parse/understand. Some simplifications would
help significantly I think, without compromising the main ideas of the
paper, for example:

- Work in the well-specified setting, so there is no need for \tau^n

- Don't worry about localized rademacher complexity

- Just do Massart margin condition

Basically, I get the impulse to consider the most general settings,
but I think this significantly compromises the readability of the
paper and isn't often very relevant toward the main points. One idea
is to write the main body of the paper in the simplest possible
setting that conveys the main technical and conceptual innovations and
then put the most general results in the appendix.

2. To this end, I actually couldn't figure out what is the technical
innovation here. I saw the paragraph saying that the policy
optimization part, specifically dealing with the dependence between
\hat{\tau}_{t+1} and the nuisance function for time t is the main
challenge. My first impression is that the margin assumption allows
one to turn a ell_2 guarantee into an ell_{\infty} guarantee, which
one can use to perhaps break this dependence? But honestly I didn't
really get it and the theorem statement is way too complicated for me
to try to infer what is happening.

3. I also think the paper could benefit from connecting to the prior
literature on offline RL theory. In particular how do requirements
like "bellman completeness" appear in the analysis. I don't think it
is possible to estimate Q functions, essentially at all, without such
assumptions in the offline RL setting, see e.g., [1]. So where does
such an assumption come into the analysis?

Overall, my main concern is that the paper is way too hard to read,
even for relative experts in the area. I think it will not be
appreciated by most of the ICLR audience. And I might be wrong about
this (due to difficulty in understanding the paper), but I currently
feel that in terms of technical novelty, the paper does not quite meet
the bar set by the conference.

References

[1] Dylan J. Foster, Akshay Krishnamurthy, David Simchi-Levi, Yunzong
Xu. Offline Reinforcement Learning: Fundamental Barriers for Value
Function Approximation. COLT 2022.

### Strengths
see above

### Weaknesses
see above

### Questions
see above

### Soundness
4

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
4

---

## Human Reviewer 4

### Summary
The paper introduces a novel approach with statistical theory for offline reinforcement learning (RL) through the development of a dynamic generalization of the R-learner. This method is designed for estimating and optimizing the difference between Q-functions under two different actions for discrete-valued actions. The key contributions and results of this work are as follows:
The authors extend the R-learner framework, commonly used in causal inference, to the offline RL setting. This generalization allows for the estimation of Q-function differences, which can be utilized to optimize decision-making for multiple actions.

One of the contributions of this work is the use of orthogonal estimation techniques, which improves convergence rates and as a result mitigates the effects of unstable estimations in behavior policies, which is a common issue in offline RL.

The proposed method focuses on estimating a more structured Q-function contrast, which adapts to underlying sparse or smoother structures in the data. This is crucial because causal contrasts, such as the difference in outcomes between actions, often exhibit more structure (e.g., sparsity) than the Q-functions themselves. The method, therefore, leverages this property to improve policy optimization.

Theoretical guarantees for the proposed method are given, ensuring consistency in policy optimization. The framework is capable of estimating causal contrasts in a sequential decision-making setting, with applications for learning dynamic treatment rules based on observational data. Synthetic experiments are presented to demonstrate the effectiveness of their proposed approach.

### Strengths
Originality: This paper has originality. First, it extends the R-learner from causal inference to the dynamic setting of offline RL. I agree that focusing on Q-function contrasts rather than estimating full Q-functions is a novel approach, offering advantages in terms of adaptability to structured or smoother forms like sparsity. Additionally, the use of orthogonal estimation to stabilize policy optimization and improve convergence rates is an original contribution to offline RL, leveraging causal inference techniques in a new domain. The method's ability to adapt to sparse structures in the Q-function contrast further enhances its uniqueness, aligning it with modern trends in machine learning that exploit underlying structure for efficiency. 

Quality: This paper has good quality. First, the application of orthogonal estimation techniques in this context seems a non-trivial approach to me, and it addresses critical challenges in offline RL such as slow convergence.

Clarity: This paper states its contributions clearly. In addition, as a theoretical work, the paper clearly introduced its mathematical setup: the assumptions and theorems are clearly specified. An illustration of the sequential decision making structure is given by figure 1. 

Significance: This paper seeks to address a major problem in offline RL. Specifically, the instability of policy optimization due to slow convergence of nuisance functions (Q-functions and behavior policies). The incorporation of orthogonal estimation improves the stability and reliability of learning. The focus on estimating Q-function contrasts rather than the entire Q-function offers a more adaptable and potentially more efficient way to optimize decisions, especially in environments where the contrast between actions is more structured or smoother. This could lead to better policy performance in practice, addressing the need for more efficient learning in complex decision-making problems.

### Weaknesses
As a theoretical work, the paper does not clearly highlight the main technical challenges in its analysis. While the incorporation of orthogonal estimation to improve the robustness of offline RL is a valid contribution, it would be stronger if the paper explained why this is technically challenging. Alternatively, discussing the difficulties in the theoretical analysis of the proposed algorithm would enhance its depth. 
Indeed, the paper claimed to have followed the analysis of Foster and Syrgkanis for orthogonal statistical learning. This is not an issue but the paper should also elaborate what makes the analysis in this paper different from the previous theoretical analysis, if applicable. For example, does the paper have a distinct error decomposition? Does the convergence analysis for Theorem 2 become different from previous work because some error term takes a different form? etc.

I think the following might be some helpful suggestion the authors could consider (if applicable).

(1) Include a paragraph after Theorem 2 or 3 explaining the key technical difficulties they encountered in proving those results.

(2) Explicitly compare their error decomposition or convergence analysis to previous work, pointing out any additional terms which that bring extra difficulty to the proof. 

(3) Add a short paragraph at the beginning of the theoretical analysis session to highlight the key technical challenge and novelty on a high level.

In summary, the paper could benefit from more explicit discussion of the theoretical challenges or innovations in its analysis.

Minor issue:
There seems to be some typos in the sentence from line 419-422.

### Questions
Please discuss the unique theoretical challenge or innovations, as pointed out in **Weaknesses**.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4