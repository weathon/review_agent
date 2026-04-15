# Cost Adaptive Recourse Recommendation by Adaptive Preference Elicitation

- Decision: Reject
- Scores: 6, 5, 8, 5

## Abstract
Algorithmic recourse recommends a cost-efficient action to a subject to reverse an unfavorable machine learning classification decision. Most existing methods in the literature generate recourse under the assumption of complete knowledge about the cost function. In real-world practice, subjects could have distinct preferences, leading to incomplete information about the underlying cost function of the subject. This paper proposes a two-step approach that integrates preference learning to the recourse generation problem. In the first step, we design a question-answering framework to refine the confidence set of the Mahalanobis matrix cost of the subject sequentially. Then we generate recourse by utilizing two methods: gradient-based and graph-based cost-adaptive recourse that ensures validity while considering the whole confidence set of the cost matrix. The numerical evaluation demonstrates the benefits of our approach over state-of-the-art baselines in delivering cost-efficient recourse recommendations.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The work addresses the problem of personalized algorithmic recourse
where the cost function of the user has to be learned by interacting
with the user. The problem is formalized as learning a Mahalanobis
distance in state space, and combining it with two existing approaches
for differentiable and non-differentiable classifiers respectively.

### Strengths
In the cases in which the use is assumed to provide consistent
feedback, the approach is technically sound, and it is nicely adapted
to deal with both differentiable and non-differentible classifiers.

The manuscript is well written and clear, and the algorithmic solutions are well motivated.

The authors made a substantial effort in comparing with existing
alternatives, including a recent approach (De Toni et al, 2022) for
which no implementation is currently available.

### Weaknesses
User inconsistencies are dealt in a less principled way in my
opinion. The authors do provide in the supplement a (computationally
challenging) solution to account for inconsistencies. However this
assumes to have a given max percentage of inconsistencies. Approaches
to account for user inconsistencies typically assume a user response
model, where the probability of a wrong preference feedback grows with
the similarity between instances. This is one of the advantages of a
Bayesian approach to preference elicitation.
 
The approach provides cost-learning functionalities on top of two
existing approaches, a differentiable and a non-differentiable one. It
thus inherits the limitations of these approaches, e.g. for the
non-differentiable approach, the fact that recourse can only be
achieved by moving to an example in the training set that achieves
recourse. This is suboptimal, as a new user could achieve recourse in
a way that is different from training users and has lower cost. These
limitations should also be mentioned.

### Questions
Is it possible to incorporate a user response model modelling uncertainty in feedback?

Did you study how your approach behaves when faced with inconsistent users? This is major advantage of Bayesian approaches to preference elicitation..

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper presents ReAP, a method to generate personalised recourse suggestions via preference elicitation. The authors provide a technique to learn a personalized cost function, for each user, by asking preference questions. The authors also show extended versions of a gradient-based and graph-based approach for recourse that exploit their learned cost function. Lastly, they validate ReAP with some experiments.

### Strengths
The topic of the paper is timely, and I think it is a direction which needs to be explored by the recourse community in order to make this field progress (and more realistic). The paper is also clear in its exposition, the idea is well-motivated and the formalization of the problem is straightforward.

### Weaknesses
The main concern is that I do not see the advantages of this approach compared to the closets-related method PEAR [1]. While it is true that they seem to rely on some sort of causal graph, which might be hard to get, their technique can be in principle applied to different cost functions (even the one presented here) and they provide a Bayesian model to incorporate uncertainty over the user answers. Therefore, it is not clear to me the benefits of ReAP over PEAR.

In the introduction, the author says “ [...] our framework can perform well even when the dimension of the feature space grows large [...]”. First, this assertion is true only under a synthetic setting. Real-world users will find it difficult to compare profiles with too many features. For this reason, I would have expected to see experiments on “noisy” users, which is a common scenario in the preference elicitation literature [2]. Equation 2 seems to accommodate it, but I think it would have made the evaluation less synthetic to see some results concerning the role of $\epsilon$.

Lastly, ReAP disentangle the elicitation part from the recourse generation (Figure 1). However, if the number of features is truly high, we might just need to estimate only the cost of the features we need for recourse, rather than the full matrix $A$. Moreover, it might be desirable that we just ask the user the right questions to estimate the perfect recourse for him/her, rather than the ones minimizing the smallest projection distance. 

[1] De Toni, Giovanni, et al. “Personalized Algorithmic Recourse with Preference Elicitation" arXiv preprint arXiv:2205.13743 (2022). (it appears a new version of the paper was published the last May).
[2] Viappiani & Boutilier. "Optimal bayesian recommendation sets and myopically optimal choice query sets." NeurIPS (2010).

### Questions
* Why did the author not include PEAR as a baseline in the main paper (while it is present in the Appendix)? I think it is the most suitable competitor since its method can be applied in principle to all cost functions, it employs a more targeted approach to the recourse pair selection and it deals with the user’s uncertainty in a most principled way.

* How do you explain the fact that Watcher and ReAP achieve almost the same cost/validity in all four datasets (Table 1)? 

* In Figure 4, would it be possible to show the results also for a random strategy to select the recourse pairs to show the user? It is a common check to understand if the experimental setting is not too simple (considering also the concerns of Table 1).

* What happens if the users are “noisy” in their answers? For example, they gave a wrong answer to the preference questions. Equation 2 seems to accommodate this issue, but I do not see any experiments showing the effect of $\epsilon$ in the cost estimation quality and/or final recourse.

* What is the novelty of the approaches presented in Sections 4.1 and 4.2? Especially section 4.1 seems a trivial extension of Watcher et al.

### Soundness
3 good

### Presentation
3 good

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
This paper proposes a method for learning a human subject’s recourse preferences, so as to deliver better options in the problem of algorithmic recourse. Then, methods are presenting for providing recourses given a learned cost function. The paper is technically innovative, and the approach it presents is very reasonable. The method assumes a Mahalanobis cost function for the user, and it gradually restricts the space of cost functions that is consistent with queried user preferences. Query selection is done to find aggressive cuts of the candidate set space in order to efficiently reduce the candidate set size. Existing gradient-based and blackbox (graph-based) recourse selection algorithms are adapted to using the learned cost functions. Experiments are conducted to compare the proposed method with two common baselines, Wachter et al 2018 and DiCE. It’s shown that the method does reduce ground-truth user cost over time (using simulated, identifiable cost functions) and the method does at least as well as baselines.

### Strengths
- Very Important: The proposed method tackles an important problem in a reasonable way. It’s clear that algorithmic recourse mechanisms should be able to leverage subject feedback/preferences to improve recourse quality. The proposed method makes reasonable choices at every turn as it tackles this problem. The cost function family is well motivated, and it is nice that this method reduces to simpler Euclidean-distance-based approaches at T=0 queries. The cost inference pipeline is naturally integrated with different optimization procedures for recourse selection (although this could take some work, as evidenced by the integration with FACE). The approximation to the O(n^2) search in cost inference is not only reasonable but actually a nice feature of the method, since it encodes a bias for selecting pairs where there could be a lot of information gain in the query.
- Important: The development of sequential recourses when combining the method with FACE is technically impressive and likely to be useful for downstream applications with real subjects.
- Important: The evaluation shows that the proposed method does learn user preferences over time and thereby lowers ground-truth user cost over additional queries with simulated ground truth preferences. On a number of datasets, the method works at least as well as past baselines.
- Important: The paper provides a lot of context to its approach through extensive connections to related work.

### Weaknesses
- Important: So, why wasn’t a comparison conducted against Rawal and Lakkararaju (2022)? I agree that the Bradley-Terry cost model is not expressive enough to capture many realistic cost functions, but then again this criticism should apply to the Mahalanobis family as well. I do not understand why it is claimed without further substantiation that the proposed method would perform better in high dimensions than the Bradley-Terry model, and so I worry that this is a key missing baseline in the current work. It could be that their method works as well as as the ReAP, while being conceptually simpler than ReAP.
- Important: Another question about the method is its limited improvements over even simple baselines on common datasets. There is often no clear improvement over a method from 2018. I wonder if this is an artifact of the simple ground-truth cost function distribution — could this distribution produce more heterogenous cost functions that better separate methods (particularly by being poorly approximated by a simple Euclidean distance)? Regardless, this is a mark against the method. While there are more noticeable improvements against FACE on the sequential recourses (Table 2), it’s not clear whether these are statistically significant.

### Questions
- What is the size of the solution set returned by DiCE? Is the comparison with ReAP fair in terms of computational cost and subject query cost?
- Comment: You might also reference https://arxiv.org/pdf/2111.01235.pdf, which investigates using a distribution over plausible cost functions to find a set of recourses that could better satisfy a user.
- Why optimize over the worst case cost function? How might results vary based on the ground truth cost function distribution and the choice of optimizing over the worst case cost function vs some centroid cost function or expected cost function.

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes an approach to recourse generation when the user’s cost is unknown. To learn the user’s cost, pairwise comparisons between points are used and a space of consistent cost matrices is maintained. A min-max objective is used for choosing a recourse either via a gradient-based or a graph-based approach.

### Strengths
* Adding preference elicitation for learning user cost makes a nice extension to previous work
* The paper is clearly written
* The experimental results look encouraging

### Weaknesses
* It is hard to understand the effect of some approximations (for example, decomposed maximization over edges). An empirical evaluation can help shed some light on that.
* Some details are missing from the experiments (dimension d etc)

### Questions
* Queries are chosen from the positive set D_1 only. Does it make sense to consider points from D_0 or new points that have no label (since this is only for the purpose of preference learning), especially in the case of single recourse?
* The worst-case objective might be too conservative, resulting in longer paths. It would be interesting to compare it at least empirically to other choices (A* for example, or a random A \in U_P).
* Sequential recourse: the problem in Eq (7) doesn’t seem like the real problem that needs to be solved. The set U_P updates after every step, and this changes the argmax A and therefore w_ij. Am I missing anything?
* Computation of \bar{w}_ij: maximizing independently over each edge ij is even more conservative. Is it possible to estimate the difference in computed weights between independent and joint optimization (for some small problems)?
* Section 5.1: what is the dimension d in the experiments? Can you comment on the cost of solving (5) using MOSEK? How does it scale up with d?
* Section 5.3: how is lambda chosen for the gradient-based objective?
* In Table 2, what is the average path length?
* Section 6: “We … extend the heuristics to choose the questions from pairwise comparison to multiple-option questions.” I didn’t see where this was done. Can you point to the right section?

Other questions / comments:
* “addictive” => “additive”
* Typo: Algorithm 1, should be “\lambda > 0” in “parameters”?
* In section 3.2, consider adding an algorithm block instead of enumerating the steps.
* Also, in section 4.2 it would be good to have an algorithm block for the graph-based approach.
* Mean rank is referred to in “Recourse generation” (section 5.1), but defined only later, in 5.2.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
