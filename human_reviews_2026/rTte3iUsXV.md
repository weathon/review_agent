# Displacement-Resistant Extensions of DPO with Nonconvex $f$-Divergences

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 8

## Abstract
DPO and related algorithms align language models by directly optimizing the RLHF objective: find a policy that maximizes the Bradley-Terry reward while staying close to a reference policy through a KL divergence penalty. Previous work showed that this approach could be further generalized: the original problem remains tractable even if the KL divergence is replaced by a family of $f$-divergence with a convex generating function $f$. Our first contribution is to show that convexity of $f$ is not essential. Instead, we identify a more general condition, referred to as DPO-inducing, that precisely characterizes when the RLHF problem remains tractable. Our next contribution is to establish a second condition on $f$ that is necessary to prevent probability displacement, a known empirical phenomenon in which the probabilities of the winner and the loser responses approach zero. We refer to any $f$ that satisfies this condition as displacement-resistant. We finally focus on a specific DPO-inducing and displacement-resistant $f$, leading to our novel SquaredPO loss. Compared to DPO, this new loss offers stronger theoretical guarantees while performing competitively in practice.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper studies a generalization of $f$-DPO to non-convex $f$ generator functions, and identifies a simple condition to enforce on the generator function $f$ to be more resistant to the phenomenon of "likelihood displacement" affecting DPO, by which the probabilities of chosen responses in the preference dataset tend to 0 as the training progresses. The identified condition is that the minimum of $f(t)$ should occur at $t \geq 1$. They call such functions "displacement-resistant".  The paper proposes a specific generator function that they term SquaredPO and compares it against DPO on TL;DR.

### Strengths
- S1) The paper discusses novel ideas on the realm of using $f$ divergences as penalty terms in KL-regularized RL, specifically in the context of $f$-DPO.
- S2) It comes with an extensive theoretical analysis of the discussed algorithms

### Weaknesses
- W1) The paper, while clear and well-written, was also a bit terse to read for me. This is highly subjective, and I don't weigh this weakness strongly in my evaluation.
- W2) The usage of a non-convex $f$ felt quite arbitrary, and there is no comparison with a convex displacement-resistant function. In a sense, the generalization to non-convex $f$ might be interesting, but it's unclear from the content of this paper what can we gain from it.
- W3) More in general, the comparison of the proposed technique is very limited as it only considers one baseline (DPO) and one dataset (TL;DR) , as noted by the authors in the limitations section.  While the authors claim that they leave further comparisons for future work, I don't think this is an aspect that can be relegated to future work.

### Questions
- Q1) Your "displacement-resistant" condition is a necessary, but is it sufficient to avoid likelihood displacement? If not, isn't the name "displacement-resistant" a bit misleading?
- Q2) Your displacement-resistant is quite simple. Wouldn't it predict that a simple change of variables, say $t' = t - (1- e^{-1})$ for the reverse KL, work just as well?
- Q3) I think you might want to clarify early on that the usage of $f$-divergences here concerns the generalization of the regularization penalty term only, to distinguish it from other usages in LLM post-training where they have been used as the loss to minimize in a distribution matching objective (https://arxiv.org/abs/2302.08215). Also, not sure if this is relevant, but I also note that there was another contemporary paper that studied the $f$-divergence generalization of the penalty term in the context of RLVR: https://arxiv.org/abs/2509.07430.

### Soundness
4

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
3

### Summary
Likelihood displacement has been know as one of the problems of DPO. This paper tries to solve this likelihood displacement issue by changing the KL divergenge regularization term in RLHF objective to nonconvex f-divergences. It theoretically shows that a convexity is not a necessary condition. From this theoretical finding, this paper suggests square of log function. It is both displacement resistant and DPO inducing. Empirical results on TL;DR and MT-Bench show that SQUAREDPO alleviates displacement while maintaining alignment performance comparable to standard DPO.

### Strengths
- Rigorous theoretical analysis.
- Easy to adapt loss (only regularization term is changed yet the validity is proved)

### Weaknesses
- Experiment is narrow.(model is only llama, the only baseline is naiveDPO)
- In experiment, the performance gain is marginal or even similar to naive DPO. Specifically, if the same performance for squareDPO be achieved with epoch 4 as DPO with epoch 1(Table1), why we should use SquareDPO?
- No performance analysis without LORA

### Questions
- How much displacement is mitigated? 
How it improved the model performance empirically?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes exploring DPO variants that are based on a generalization of f-divergences that does not require convexity of f. They also analyze how these variants such as SquaredPO prevent probability displacement. They derive a set of functions that are DPO-inducing and a set of functions that are displacement resistant. They show that SquaredPO in particular is more robust to over-optimization and mitigates displacement while maintaining comparable performance to DPO.

### Strengths
Analysis - The paper provides a thorough analysis of the properties of different objectives and characterizes a wide range of possible objective functions. They also provide direct empirical comparisons of changes in likelihood as well as win rate and benchmark performance. 

Clarity - The paper provides a clear presentation of ideas with detailed explanations. The theoretical definitions and interpretations walk through the key ideas and the experimental setup is well described and provides support for the claims.

### Weaknesses
Contributions - While the analysis is thorough and claims made are well supported, there is a lack of comparison to other methods that aim to achieve the same goal and it is unclear whether the convexity constraint is an issue. Figure 1 shows that there are multiple convex functions which are DPO-inducing and displacement-resistant, many of which have already been explored and have successfully mitigated over-optimization of displacement. As a result, without further comparison to these existing methods or a strong justification as to why the convexity constraint is limiting, it is unclear whether the paper provides a method or insights that goes beyond existing methods. The key ideas involved are also primarily extensions of existing ideas in f-DPO and the cited paper for Lemma 2.

### Questions
- Could you provide justification as to why the convexity of f may be a concern?
- Could you provide comparisons to existing methods such as f-DPO or Chi-PO?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This work dives deeper into DPO with f-divergences ($f$-DPO): (1) Proves more relaxed DPO-inducing condition, i.e., a sufficient condition that yields the $f$-DPO loss; (2) Proves a displacement-resistant condition, i.e., a necessary condition for $f$-DPO to avoid probability displacement issue; (3) proposes SQUAREDPO, a special case of $f$-DPO that satisfies the DPO-inducing and displacement-resistant conditions, and is empirically demonstrated to have better performance and mitigated displacement issue compared with vanilla DPO.

### Strengths
The perspective of looking at $f$-DPO is novel, with non-trivial theoretical results, which also yields a new DPO variant that is theoretically and empirically better than vanilla DPO. Many benchmarks are used in the experiments. There seem to be sufficient details to reproduce the experiments.

### Weaknesses
Figure 4 does not show significant performance difference between DPO and the proposed SQUAREDPO. 

The paper could be better if 

(1) Sufficient conditions for displacement-resistancy could be provided. 

(2) $f$-DPO with more choices of $f$ could be empirically compared, such as $\chi^2$.

### Questions
(1) What is $\mathbb{R} _ {++}$ in Corollary 1? 

(2) Both $\ln$ and $\log$ are used in the paper. Do they mean the same? 

(3) Is there any sufficient condition to prevent probability displacement? 

(4) What metrics are used in Figure 4?

### Soundness
3

### Presentation
3

### Contribution
3
