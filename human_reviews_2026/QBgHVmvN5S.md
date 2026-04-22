# Bi-Criteria Metric Distortion

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 8

## Abstract
Selecting representatives based on voters' preferences is a fundamental problem in social choice theory. While cardinal utility functions offer a detailed representation of preferences, voters often cannot precisely quantify their affinity towards a given candidate. As a result, modern voting systems rely on ordinal rankings to simplistically represent preference profiles. In quantifying the suboptimality of solutions due to the loss of information when using ordinal preferences, the metric distortion framework models voters and candidates as points in a metric space, with distortion bounding the efficiency loss. Prior works within this framework use the distance between a voter and a candidate in the underlying metric as the cost of selecting the candidate for the given voter, with a goal of minimizing the sum (utilitarian) or maximum (egalitarian) of costs across voters. For deterministic election mechanisms selecting a single winning candidate, the best possible distortion is known to be 3 for any metric, as established by Gkatzelis, Halpern, and Shah (FOCS'20). In contrast, for randomized mechanisms, distortions cannot be lower than $2.112$, as shown by Charikar and Ramakrishnan (SODA'22), and there exists a mechanism with a distortion guarantee of $2.753$, according to Charikar, Ramakrishnan, Wang, and Wu (SODA'24 Best Paper Award). Our work asks: can one obtain a better approximation compared to an optimal candidate by selecting a committee of $k$ candidates ($k \ge 1$), where the cost of a voter is defined to be its distance to the closest candidate in the committee? We affirmatively answer this question by introducing the concept of bi-criteria approximation within the metric distortion framework. In the line metric, it is possible to achieve optimal cost with only $O(1)$ candidates. In contrast, we also prove that in both the two-dimensional and tree metrics -- which naturally generalize the line metric -- achieving optimal cost is impossible unless all candidates are selected. These results apply to both utilitarian and egalitarian objectives. Our results establish a stark separation between the line metric and the 2D or tree metric in the context of the metric distortion problem.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates the following problem in social choice theory under the metric distortion framework: if, instead of selecting a single winner, a deterministic mechanism is allowed to select multiple candidates (where the cost to a voter is their distance to the nearest selected candidate), then can this lead to smaller distortion relative to the optimal single candidate?

The authors obtained the following results:

1. Line metric (1D Euclidean): At most 4 candidates suffice to achieve optimal cost in both the utilitarian  and egalitarian cost setting.
2. 2-D Euclidean and tree metric: Even when selecting $m−1$ of $m$ candidates, the distortion cannot be smaller than $1+2/(m−1)$ for utilitarian and $3$ for egalitarian. As a corollary, it is impossible to achieve the optimal cost unless all candidates are selected.

### Strengths
This is a theoretical paper addressing a natural and interesting question in metric distortion. The results are logically sound and well contextualized within the existing literature. The writing is clear and well-organized.

### Weaknesses
1. The contribution is limited to deterministic algorithms and highly restricted metric classes (line, 2-D Euclidean, and tree metrics). It would be interesting to see whether similar results extend to randomized mechanisms or more general metrics.

2. The paper focuses on existence results without explicitly discussing the computational complexity or implementability of the proposed mechanisms.  Clarifying whether these mechanisms can be computed efficiently would strengthen the practical relevance of the work.

### Questions
See "weakness"

### Soundness
3

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
2

### Summary
For reasons of convenience and practicality, when soliciting preferences we often see ordinal preferences. These preferences mask the true intensity of certain opinions and the gap between them - a small gap between A and B and a large gap between B and C are represented simply as A,B,C. And so the optimal solution can be ambiguous.

The authors propose an approach of committees' to address this well-established problem, in a bi-criteria sense: select a small k-committee but compare its cost against the best single candidate. They establish bounds on the maximum level of distortion this approach can have. While fairly simple intuitively, the authors prove their methods bounds and claims rigorously.

### Strengths
* I think the content of the paper is good, and I feel that between the appendices and the main body a strong paper can be found (more in weaknesses/suggestions)

* The content is well motivated, I believe it is a novel contribution, and well-justified.

### Weaknesses
* I thought the visuals in the paper were quite weak and to the detriment of the clarity of the paper. There are only 2 figures. The first is quite large but has conveys relatively little information. The next (2/3) one doesn't convey much beyond the writing. I felt that understandability could be easily enhanced with more visuals that convey more meaning (there are several in the appendix that seem to be much better/efficient at conveying information)

* Perhaps the above could be overcome, but the writing in section 3/4 was not, in my opinion, up to par. The lemmas are given in succession, the connections are weakly explained, and it's not presented in an easy way to follow. I recognize that there are proofs in the appendix and the space provided is limited, but the main text is not up to par. Between lemma 9-11 and then theorem 12 for instance, we cover 2 different cases in the span of a few sentences, with choppy flow.

### Questions
I think this paper is solid content-wise when the supplementary material is considered. I think the writing and presentation are not. I know much of the content is in the appendices, but as is, the writing in Sections 3 and 4 is not yet there in my opinion. That is my primary critique, and revision there could easily raise my score.

### Soundness
4

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper investigates whether selecting multiple representatives (a committee) instead of a single candidate can reduce the efficiency loss that arises when elections rely only on ordinal preferences rather than full cardinal information. This problem is studied within the metric distortion framework, where voters and candidates are points in a metric space and the cost of selecting a candidate for a voter is their distance. 

The authors introduce a bi-criteria approximation perspective, asking whether selecting a small committee of size k≥1 can yield a total or maximum voter cost close to that of an optimal single candidate. They show that in the line metric, it is indeed possible to achieve the optimal cost with only a constant number of candidates. 

However, this improvement is shown to be unique to one-dimensional settings. In two-dimensional Euclidean and tree metrics, achieving optimal cost is impossible unless all candidates are selected.

### Strengths
The paper introduces a bi-criteria approximation framework to analyze how selecting multiple candidates can reduce efficiency loss in metric distortion, extending classical single-winner results to multi-candidate settings for both utilitarian and egalitarian objectives. It establishes precise trade-offs between committee size and approximation quality, proving that on the line metric two candidates can achieve optimal cost, while in higher-dimensional and tree metrics such improvement is impossible without selecting all candidates.

### Weaknesses
The main weakness of the paper is in the presentation, given that the introduction is too long and most proofs of correctness are relegated to the appendix, which is pretty long. On the other hand, the results are presented well and given that the paper is technically demanding, this choice is justified.

### Questions
No questions.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper studies the question of whether it is possible to achieve the cost of an optimal candidate in the metric distortion framework when selecting only a fixed number of candidates instead of a single one. For the line metric, it provides a positive answer both for the utilitarian objective (with just 2 candidates) and the egalitarian objective (with just 4 candidates). It also provides matching lower bounds. In contrast, even in the 2D Euclidean metric, guaranteeing optimality requires selecting all candidates.

### Strengths
The paper studies a very natural question: how many additional candidates are needed to guarantee optimal distortion. This to some extend related with k-committee selection, but the main question posed here is different. The paper obtains strong and non-trivial results both both the utilitarian and egalitarian objectives, and both the 1D case and more general metrics. The results on the line are particularly surprising: a constant number of candidates (and in fact a small constant at that) always suffices to guarantee optimality. I find this result very interesting and informative in practice. In cases where one can potentially select more than one candidate, the results of this paper explain how one can navigate the tradeoff between picking a small number of candidates and guaranteeing small distortion. The paper also provides an interesting impossibility result, showing that a similar guarantee cannot be attained in general metrics. In fact, selecting all but one candidates will always fail to guarantee optimality. This provides an interesting separation between 1D and more general metrics, something which hasn't been investigated a lot in this line of work, with some exceptions. Overall, the paper makes a concrete contribution to a natural problem. It provides a comprehensive understanding under different objectives and metric spaces, together with matching lower bounds. 

Furthermore, the writing of the paper is very good. It accurately places its contributions in the context of the existing work; all relevant references have been discussed. The technical component is also non-trivial and explained well in the main body.

### Weaknesses
The main weakness is that, a priori, selecting more than one winning candidate could trivialize the problem when comparing against the optimal omniscient solution that picks a single candidate. Still, the lower bounds of the paper show that this is far from trivial in general, and accomplishing it in 1D turns out to be highly non-trivial.

### Questions
- Is there some general property of the metric space that could be used to parameterize the number of candidates needed to reach optimality? I am trying to understand whether it is really only the 1D metric that has this nice property. Could you prove that, in some sense, any metric that is not isomorphic to the 1D will always fail to reach optimality with all but one candidates?

### Soundness
4

### Presentation
3

### Contribution
3
