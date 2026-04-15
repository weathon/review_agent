# Last-Iterate Convergence Properties of Regret-Matching Algorithms in Games

- Decision: Reject
- Scores: 8, 8, 5, 3, 6

## Abstract
Algorithms based on regret matching, specifically regret matching$^+$ (RM$^+$), and its variants are the most popular approaches for solving large-scale two-player zero-sum games in practice. Unlike algorithms such as optimistic gradient descent ascent, which have strong last-iterate and ergodic convergence properties for zero-sum games, virtually nothing is known about the last-iterate properties of regret-matching algorithms. Since last-iterate convergence is an attractive property both for numerical optimization reasons and because no-regret learning is viewed as a plausible method of real-world learning in games. In this paper, we study the last-iterate convergence properties of various popular variants of RM$^+$. First, we show numerically that several practical variants such as simultaneous RM$^+$, alternating RM$^+$, and simultaneous predictive RM$^+$, all lack last-iterate convergence guarantees even on a simple $3\times 3$ game. Then, we go on to show that recent variants of these algorithms based on a *smoothing* technique do enjoy last-iterate convergence: we prove that *extragradient RM$^{+}$* and *smooth PRM$^+$*  enjoy asymptotic last-iterate convergence (without a rate) and $1/\sqrt{t}$ best-iterate convergence. Finally, we introduce restarted variants of these algorithms, and show that in both cases they enjoy linear-rate last-iterate convergence.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper investigates the last-iterate convergence property of the algorithm family of Regret Matching, which is popular in practice but lacks corresponding theoretical guarantees for its good empirical performance. This paper fills the gap in this part. Specifically, the authors first show that RM+ and some of its variants, such as alternating RM+ and PRM+, may not converge by a toy example. Then, they prove that, under a very strong assumption of strict Nash equilibrium (NE), an assumption that usually does not hold in practice, RM+ enjoys last-iterate asymptotic convergence. Consequently, focusing on ExRM+, another algorithm in the RM+ family, the authors first prove that it enjoys last-iterate convergence when there is a unique limit point. Later, the authors rule out the case of infinite limit points and thus prove the last-iterate convergence of RM+. Besides, the authors prove an $O(1/\sqrt{T})$ best-iterate convergence rate for ExRM+. From the best-iterate analysis, they find a simple strategy for obtaining a linear last-iterate convergence rate is to restart the algorithm whenever the best-iterate comes. However, the distance to NE is not observable. To this end, they find a proxy variable (upper bound) for the distance to NE, restart the algorithm by checking the condition on the proxy variable, and eventually achieve a linear last-iterate convergence rate. Moreover, they also prove a similar linear rate for another RM+ variant called SPRM+, following the same flow as before. Finally, empirical studies validate the effectiveness of the proposed methods.

### Strengths
This is a solid paper from my point of view. RM+ algorithms do not draw enough attention in the literature because of the lack of theoretical guarantees. This paper fills a gap in this point. The presentation is quite clear and intuitive. It is worth noting that the authors also give some illustrating examples to help readers understand the paper better, which is very good. The preliminaries are sufficient for readers with little background knowledge in this field or about RM+ algorithms. The solution is described step by step, which is quite clear and intuitive. Although the final solution (restarting mechanism) is simple, the obtained results are important since they show that RM+ algorithms are not only useful in practice but also theoretically guaranteed.

### Weaknesses
I do not see major weaknesses in this paper.

### Questions
I do not have any questions since the presentation is clear and intuitive.

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
4 excellent

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors study the last-iterate and best-iterate convergence of RM+ and variants in in 2-player zero-sum games. 
They show:
(1)	Empirically, RM+ and some of its simplest variants fail to converge on a specific 3X3 example recently used by Farina et al to show instability of RM+
(2)	Analytically, two slightly more elaborate variants of RM+, namely ExRM+ and SPRM+ do converge
(3)	Furthermore, restarting ExRM+ and SPRM+ at the right times improves the theoretical convergence rate to linear.

### Strengths
I think that the results are interesting. Last-iterate convergence to NE is an interesting question, and understanding which natural algorithms converge and which don’t is a natural question.

### Weaknesses
On the downside, to be honest I couldn’t find anything particularly innovative about the paper. Perhaps the careful analysis of ExRM+ in Section 4 would be interesting even to experts.

### Questions
1.	I’m confused about the notation \delta_{\ge}: aren’t all the strategies in \delta already probability distribution that whose sum is exactly 1?

2.	For the numerical run on the 3X3 algorithm, it would be very insightful to plot the actual trajectory of the algorithms (it should be relatively easy to plot in 2D since the strategies are probability distributions over 3 actions)

a.	Related, I’m curious if best-iterate convergence of RM+ etc is any better?

3.	The bulk of the work in Sections 4 and 5 is, AFAICT, to deal with solutions of the VI that are not NE. I’m a bit confused about it – when you introduced the VI notation I expected that NE would be the only solutions. Maybe you could give a simple example of a non-NE solution? That would really help build intuition.

4.	I probably missed something, but I didn’t understand why you need to talk about best iterate vs last iterate in Section 4.2. Does the guarantee of Lemma 1 not imply something about the convergence of last z^t?

5.	Appendix C, Claim 2, Case 2 typo: “equilibrium” should be “equilibria” (This is not a question :))

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The research paper studies  the convergence properties of various variants of the Regret Matching algorithm within the context of two-player zero-sum games. The authors present a mix of favorable and unfavorable outcomes in this regard.

To begin with, the authors offer empirical evidence demonstrating that RM+, alternating RM+, and the more recently proposed Predictive RM+ (PRM+) do not admit last-iterate convergence even in the case of simple 3x3 zero-sum games. Subsequently, the authors introduce and evaluate two recently proposed variants of the RM algorithm.

Specifically they consider the Extragradient RM+ (ExRM+) and establish that it displays asymptotic last-iterate convergence and $O(1/\sqrt{T})$ best-iterate convergence. Furthermore, the authors demonstrate that a version of ExRM+ incorporating restarts achieves linear last-iterate convergence.
In section 5, the authors extend the above last-iterate results to Smooth Predicative RM+ (that is another variant of RM+). They respectively provide insights into the asymptotic last-iterate and $O(1/\sqrt{T})$ best-iterate convergence of SPRM+. Additionally, they extend their linear-convergence findings to the variant of SPRM+ that employs restarts.

### Strengths
Regret matching algorithms comprise an intriguing class of algorithms due to their practical success in large extensive form games. Despite the latter success, it is worth noting that the theoretical aspects of RM algorithms have not been sufficiently studied, as also acknowledged by the authors. From this perspective, I find the paper to be sufficiently motivated and to offer interesting insights into the behavior of these algorithms.

I am particularly intrigued by the observation that RM+, alt-RM+, and PRM+ do not exhibit last-iterate convergence, while the alternating version of PRM+ appears to outperform other RM variants in the experiments provided. This finding adds an interesting dimension to the discussion. Additionally, the paper is well-written and employs techniques that appear to distinguish from the previous techniques establishing last-iterate convergence for ExtraGradient and OMD (I have not yet delved into the appendix in full detail though).

### Weaknesses
Despite the fact that the paper provides some interesting insights on the last-iterate convergence properties of RM algorithms, I find that the paper lacks a key take-way message that creates me some doubts on the significance of the results.

As previously mentioned, the empirical success of RM+ algorithms lacks a solid theoretical foundation. In this context, investigating the last-iterate convergence properties of RM or RM+ appears to be a reasonable step in the right direction. However, I do have reservations about the significance of establishing last-iterate convergence results for artificial RM variants, especially given the existing results for Extragradient and OMD.

Additionally, I suggest that the paper could enhance its value by presenting time-average results for the different RM variants. It would be particularly intriguing to see the time-average convergence rates of the alternating PMR+ algorithm, which, as demonstrated in the provided experiments, seems to significantly outperform other RM methods in terms of last-iterate convergence. Furthermore, an experimental comparison of the last-iterate properties of Extragradient and OMD would be a valuable addition.

In summary, I consider this paper to be on the cusp of meeting the threshold for acceptance. It does contain some interesting findings, but I remain somewhat uncertain about their overall importance. I am open to reconsidering my assessment and potentially raising my score if the authors address the aforementioned comments during the rebuttal phase.

### Questions
Is there a limit for the constant $c$ in Proposition 2? Could it potentially be exceedingly small, perhaps even exponentially so?

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
The paper considers the classical min-max matrix game and discusses the convergence properties of several algorithms.  In particular, the paper shows the last-iterate property holds for some algorithms, where the property means that the last update of the solution is the output of the algorithm. In general, many iterative algorithms based on online linear optimization (such as Hedge) have a solution based on the average of iterative updates. The experimental results compare several iterative algorithms.

### Strengths
The theoretical analyses are solid. In particular, the last-iterate property might be interesting for iterative algorithms, especially if the design is based on the online convex (linear) optimization. However, I feel that the asymptotic statement of the last-iterate property is not strong enough.

### Weaknesses
The min-max game itself is known to be an LP and thus solved in polynomial time. Iterative algorithms are one of the approaches to solving the LPs. The paper should consider comparing the state-of-the-art LP solvers with iterative algorithms. Although iterative algorithms are easy to implement, practical LP solvers have been improved for a long time and could be faster than naive iterative algorithms.

I do not understand why the last-iterate property is important. The convergence results can still be obtained by averaging the outputs of iterative OLO-based algorithms (such as Hedge). Maybe a more important issue is the speed itself, not whether the property holds or not.

I am afraid that the experimental data is rather too small for LP instances. For such small instances, I wonder that the sota LP solver such as Gurobi solve them much faster. 

In summary, the paper focuses only on iterative algorithms, but as a solver of a certain LP, there are more alternatives to compare.

### Questions
Are the analyses for the last iterate property useful for constructing a new online-to-batch conversion technique (e.g., averaging all outputs of OCO algorithms)?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 5

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors study the last iterate convergence of regret matching algorithms in normal-form games. They show that a large class of well-known regret matching algorithms are not guaranteed to converge in last iterates and provide  EXRM+  and SPRM+ as alternatives that converge at a rate of 1/sqrt(T), and whose convergence rate can further improved to be linear

### Strengths
The non-convergence examples in last iterates are interesting and the linear convergence rates under restarting seem interesting.

### Weaknesses
The convergence results for EXRM+ and SPRM+ seem to be direct from known results (?) (see [1] for EXRM, and [2] for SPRM+). The authors claim that their setting does not satisfy the monotonocity assumption, but zero-sum bimatrix games do satisfy the monotonicity assumption (correct me if I am wrong?)

### Questions
Can you clarify the comments at the bottom of page 5? It is not clear to me why your setting is not a monotone game setting?

Can you explain why the restarted variants of your algorithms converge faster?

[1] Gorbunov, Eduard, Nicolas Loizou, and Gauthier Gidel. "Extragradient method: O (1/k) last-iterate convergence for monotone variational inequalities and connections with cocoercivity." International Conference on Artificial Intelligence and Statistics. PMLR, 2022.

[2] Cai, Yang, Argyris Oikonomou, and Weiqiang Zheng. "Tight last-iterate convergence of the extragradient and the optimistic gradient descent-ascent algorithm for constrained monotone variational inequalities." arXiv preprint arXiv:2204.09228 (2022).

### Soundness
2 fair

### Presentation
4 excellent

### Contribution
2 fair
