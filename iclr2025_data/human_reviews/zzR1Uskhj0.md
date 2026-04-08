## Human Reviewer 1

### Summary
The paper proposes an algorithm that achieves high probability regret bound (which is stronger than the expected regret bound) for the cross-learning contextual bandits under unknown context distribution by developing refined martingale inequalities.

### Strengths
1. The paper clearly presents the challenging point with detailed technical expressions and the novelty of the analysis.

### Weaknesses
1. The explanation is only focused on technical side without the explanation of the algorithm. I suggest authors to spend more time including more explanations and organizing the paper.

### Questions
1. What is the intuition behind the algorithm?
2. How the indicator function $F_e$ resolves the unbounded martingale inequalties?

### Soundness
2

### Presentation
1

### Contribution
2

### Rating
5

### Confidence
3

---

## Human Reviewer 2

### Summary
This paper addresses the challenge of achieving high-probability regret bounds in the adversarial contextual bandit framework, where the learner encounters varying contexts and must minimize cumulative loss over time. The focus is on "cross-learning" contextual bandits, where learners can observe losses for all possible contexts, not just the current one. 

Results leverage weak dependencies between epochs and refine existing martingale inequalities, by exploiting interdependencies in observations. This analysis ultimately shows that the algorithm is effective in adversarial settings, even with unknown context distributions.

### Strengths
The paper proposes a new look at an existing problem and provides a completely novel analysis in their work. The ideas and techniques proposed are completely new and can be of independent interest. This is particularly true for the martingale concentration result.

### Weaknesses
See questions.

### Questions
While the result is good, I am uncertain about its significance because neither the problem nor the algorithm proposed are new. There are no extensions of this result, no experiments. I am not sure if this is standalone result is significant enough to be published at a premier conference.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
5

### Confidence
3

---

## Human Reviewer 3

### Summary
The paper studies cross learning in contextual adversarial linear bandits where the learner observes the losses of all contexts in each round. Recent work in Schneider et al. proposed an algorithm with a regret upper bound only in expectation. The paper studies the same algorithm and proves that the regret upper bound holds with high probability.

### Strengths
- The paper proves a high probability lower bound which is stronger than the in expectation bound in the literature.
- The analysis uses a nice observation that the different epochs in the algorithm are only weakly dependent which enables to prove a small bound for the cumulative bias across all epochs
- While standard martingale inequalities cannot directly upper bound the cumulative bias, a novel technique is proposed to address this

### Weaknesses
Can the reduction in [1] be used to map the multi-context to a single context problem? The technique is proposed for non-adversarial losses, however, the action set map from distributional to fixed should not be affected by that.
I understand that the paper only focuses on analyzing an existing algorithm. However, a comparison with such technique in the related work is needed to justify the use of such algorithm or suggest alternative techniques to address the problem.

[1] "Contexts can be cheap: Solving stochastic contextual bandits with linear bandit algorithms." The Thirty Sixth Annual Conference on Learning Theory. PMLR, 2023.

### Questions
please see weaknesses

### Soundness
4

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
3

---

## Human Reviewer 4

### Summary
The paper studied adversarial context bandits in a special setting where the losses of arm $a_i$ could be observed under all contexts when the algorithm plays arm $a_i$. The goal, like in classical adversarial bandit problems, is to minimize the regret compared to the loss of the best arm in hindsight. The paper focuses on the setting where the loss sequence is adversarial and the context is stochastic with an unknown distribution. A recent work of SZ [NeurIPS’23] designed an algorithm with *expected* regret of $\tilde{O}(\sqrt{KT})$ in this setting, where $K$ is the number of arms. This paper conducted a renewed analysis of the algorithm in SZ  [NeurIPS’23], and the main result is that the algorithm could actually achieve $\tilde{O}(\sqrt{KT})$ with high probability.

The main technique of the paper is heavily influenced by the previous work of SZ [NeurIPS’23]. In a nutshell, the low-regret guarantee of the algorithm crucially relies on the concentration of unbiased estimation of $E_{c}[\ell_{t,c}(a)]$. Here, we cannot exactly compute the quantity since the distribution of the context is unknown. The key idea of SZ [NeurIPS’23] is to commit two steps for each EXP3 step and use one of them to estimate the distribution of the context. On top of that, this paper further utilized the weak dependency between epochs, and derived a martingale argument to get high-probability regret.

### Strengths
In general, my opinion of this paper is positive. The paper appears to require a great deal of background to be able to parse. Despite this, I believe the paper did reasonably well in terms of explaining the existing work and its techniques. Getting high probability bounds in adversarial bandits usually requires some neat observations and technical steps. Although I’m not able to follow all the steps in the short time frame, I do think the paper contains some nice technical observations and ideas.

### Weaknesses
Although I'm mostly supportive, I think I couldn’t strongly champion the paper due to the following reasons:
- The scope of contribution: although the paper does contain some neat technical observations, the contribution appears to be somehow incremental. After all, this is a new analysis of an existing algorithm, and the new analysis is not something that improves the previous bound (but instead is to get a high-probability bound). Again, I do acknowledge that such contributions are non-trivial. However, I do not think it’s enough for me to champion the paper.
- If the paper is going to be mainly accepted due to the techniques: then, I do not think the paper contains a substantial amount of new ideas. I appreciate the technical observations, and I agree that the steps are non-trivial. However, if the conceptual message is not as strong, and the merits of the paper mainly lie in the techniques, then the bar would inevitably be higher. 
- For a conference like ICLR, the lack of experiments could be an issue. I am *not* letting this affect my score since I often advocate learning theory papers. However, I do want to raise this point since it is common for ML conferences to ask for experiments.

### Questions
- Should the definition of regret on page 3 be reversed? As in, you are subtracting the loss of the best arm with the loss of a policy, which should be a negative value (if with positive regret). This would change the decomposition of the regret on page 5 as well, but it seems nothing would affect the correctness.
- I think the full algorithm description of the algorithm in SZ [NeurIPS’23] (or some simpler version of the description) could be shown much earlier in the paper. This would be helpful for readers who are not familiar with the previous algorithm.
- Also, stating the main theorem in a preliminary section looks very non-standard to me. I’m not letting this affect my score, but please consider re-organizing this.
- The meaning of 'with high probability' was never explained in the paper -- as in, it could mean with probability $1-1/K$ or with probability $0.99$. I think your bound gives the former, and this should be stated explicitly.

### Soundness
4

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 5

### Summary
The submission studies contextual bandits with cross learning. Previously, the existing regret bound held in expectation. The submission refines the regret analysis so that the regret bound holds with high probability. The main contribution is to show how the weak dependency structure can be exploited to solve a concentration difficulty in the previous analysis.

### Strengths
- The submission points out the difficulty that prevents the previous work from achieving a bound with high probability (lines 167–176).
- Identify the weak dependency between epochs (line 386).
- Devise a new technique to solve the unbounded issue induced by the weak dependency (the treatment for the Bias5e them in lines 395–411).

### Weaknesses
- (1) Section 3.2 contains several subtopics, such as the regret decomposition, the discussion of each decomposed term, and the analysis strategy for the challenging term. A better editorial layout would improve the readability.
- (2) The sentence “Notably, …” in line 111 is confusing. It seems unrealistic to be able to observe the loss for every context $c$. It also does not match the algorithm’s (Algorithm 1) behavior.

### Questions
- (1) Is the concept class $C$ a finite set? If so, what is the reason for assuming a finite concept class $C$? Practically speaking, the contextual information would be like a vector in a compact set, as it is very unlikely to see two identical users.
- (2) Where is the variable in Theorem 1 that characterizes the property of $C$? How does this variable appear in the bound proved in this submission?
- (3) Could you please provide more evidence or further discussion of the applicability of the technique developed here so that we can better appreciate its potential?

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