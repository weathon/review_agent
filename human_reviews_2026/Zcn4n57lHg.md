# On Smoothness Bounds for Non-Clairvoyant Scheduling with Predictions

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 6, 8, 2

## Abstract
Algorithms with predictions leverage predictions for unknown inputs in online decision-making. These algorithms are analyzed by consistency, i.e., competitive ratio under perfect predictions, and robustness, i.e., competitive ratio under worst-case predictions. Smooth degrading performance with an increased prediction error is also desirable. This paper refines the notion of smoothness, a function of prediction error, defined as the competitive ratio over the problem instances where predictions are guaranteed to provide additional information. 

With our refined smoothness metric, we establish smoothness bounds for a few scheduling problems, including online total completion time minimization and makespan minimization. For a single machine to minimize the total completion time, we show a lower bound of $\eta$ and a $\eta^2$-smooth algorithm, where $\eta$ is the prediction error ($\eta \geq 1$); the bound holds for small errors. For parallel identical machines to minimize the makespan, we show a lower bound of $2 - O(\eta^{-2})$ and present an $O(\eta^2)$-smooth algorithm for small errors. Both bounds are tighter than the existing ones. For uniformly-related machines to minimize the makespan, we show a tight lower bound of $\lceil \log \eta \rceil$, matched by an $O(\log \eta)$-smooth algorithm.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The present paper studies non-clairvoyant scheduling scheduling with predictions. In this online scheduling problem, we are given job that need to be scheduled on a single or multiple machines over time. A non-clairvoyant algorithm only learns about a job's processing time once it is completed. In particular, the paper studies this problem for minimizing the average completion time on a single machine, and minimizing the makespan on parallel identical and uniformly related machines. 

In the classic online setting, these problems are well understood. The paper considers them in a learning-augmented setting, where initially a prediction on each jobs processing time is given. The quality of the prediction is measures by the maximum relative deviation factor $\eta$ of the predictions to the actual processing times. The authors propose a new idea to classify, for a fixed error value, all instances into those where the prediction error is correct, helpful, and not helpfu. Their key idea is that if their exists a prediction that achieves for the instance the fixed prediction error, then it is still useless if all predicted job lengths are the same. By that, they remove instances from the set of instance on which they want to achieve smooth guarantees, and thereby can show better bounds than previous work.

They then apply this new definition to the three aforementioned problem variants, and prove error-dependent performance guarantees depending on $\eta$.

### Strengths
- The paper proposes a new way to establish smoothness bounds for learning-augmented algorithms, and showcases it in non-clairvoyant scheduling, a well-studied and established class of problems in the area.
- The bounds are highly non-trivial and optimized. They improve over previous work.
- They prove lower bounds that show that previous results are best-possible
- The paper considers three different problems, which have all been studied before, and thereby nicely continues the story of learning-augmented scheduling.

I think that the paper is a nice contribution to the field of learning-augmented algorithms. It has a new conceptual idea and executes it across different settings, which showcases it applicability. It also gives a good overview over literature. I think it is a good fit for ICLR.

### Weaknesses
- One can argue that the results are somewhat incremental, but I think this is rather a minor weakness.
- I would prefer a slightly less technical main part. For a revised version, I would recommend to simplify or upper bound complicated expressions like in Theorem 4.1.

### Questions
As far as I know, the type of multiplicative prediction error has been introduced in a paper by Azar, Leonardi, and Touitou (STOC 2021). Perhaps you update these references.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies the non-clairvoyant scheduling problem in an online setting under the algorithms with predictions framework.  In addition to the standard notions of consistency and robustness, the authors revisit the notion of smoothness, which captures the performance of a learning-augmented algorithm when predictions are in the middle ground — not perfectly accurate but also not completely adversarial (i.e., there is a bounded prediction error).  This problem has been studied before, but the authors propose a different (superior) metric to measure prediction error, and upper / lower bounds on algorithm performance in three scenarios using this error metric.

### Strengths
The proposed method to measure prediction error (multiplicatively rather than as the $L_1$ difference) is natural and makes sense for the problem.

Smoothness bounds are known to be somewhat difficult to obtain in the literature on learning-augmented algorithms (in the sense that many papers omit it completely and focus on the sometimes easier to analyze consistency and robustness metrics).  This paper provides a useful definition of smoothness (as a partitioning of the universe of inputs) that could help to model some notion of smoothness in other problems.

The paper studies several settings: single machine scheduling, parallel machines scheduling, and uniform machines scheduling, and provides or completes a set of upper and lower bounds for each setting.

### Weaknesses
Figure 2 is quite hard to read in its current form due to small text.

The upper bounds and lower bounds are not all tight in the sense that the proposed algorithms do not attain the theoretically best possible smoothness — this is a relatively mild weakness since attaining optimality in this sense seems quite challenging in the literature on related problems.

Typically at an ICLR-type of conference I would expect to see some small numerical experiments in addition to the theory.  This might also allow for some comparison between the proposed algorithms and the algorithms that rely on a different prediction error metric in the literature.

It may be worth clarifying somewhere early on that $\eta \geq 1$ in your multiplicative model, because from the abstract a reader may get the impression that your upper bound is somehow below your lower bound (i.e., if $\eta$ was allowed to be $< 1$, $\eta^2 < \eta$).

### Questions
The refined notion of smoothness requires this definition of instances on which \eta is small enough to provide information (reasonable predictions, Definition 3.1).  Can you speak on whether this notion of smoothness can extend to other online problems?  I know there are some problems such as one-way trading that exhibit a brittleness [1] — even a prediction with a very small prediction error can be completely uninformative in some cases.  Would this pose a challenge for the proposed partitioning?

Am I understanding correctly that the constant vector $\mathbf{p}$ mentioned around line 193 is a vector where any arbitrary element is the same value (e.g., p[0] = p[1] = p[2] = ….)?

Can you speak to the utility of smoothness bounds in practice?  They initially strike me as a primarily theoretical exercise since a prediction model’s error is not generally known apriori.  This question doesn’t detract from the paper’s contribution, but I wonder if there is some secondary benefit from considering smoothness (e.g., even if the bounds themselves are not informative in practice, perhaps the algorithms designed with smoothness in mind are better at dealing with prediction errors in practice?)

[1] On Tradeoffs in Learning-Augmented Algorithms, Ziyad Benomar, Vianney Perchet, arXiv:2501.12770

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper considers 3 scheduling problems. Jobs are given, but their length is revealed only at their completion. This is called the non-clairvoyant model. The 3 scheduling problems are

1. Single machine. The goal is to minimize the sum of completion times.
2. Identical parallel machines. The goal is to minimize the maximum completion time.
3. Related parallel machines. Same objective. But machines have different speeds. Execution time is job length divided by machine speed.

This model is augmented with predictions on the job length. The error between the prediction and the actual job length is measured by $\eta$, the maximum over all jobs j, of the maximum over the ratios $p_j/p^\star_j$ and $p^\star_j/p_j$, where $p_j$ is the predicted and $p^\star_j$ the actual length. Note that if the error is 1, the optimal solution could be computed, at least in exponential time for the parallel machine cases.

The performance of a schedule is measured by the competitive ratio, which compares its objective value with the optimal solution, one could compute if the actual processing times were known. 

This paper is part of an active research area on learning augmented algorithms. There an important question is how to design algorithms with a competitive ratio degrading smoothly with the error.

The paper distinguishes 3 kind of instances, where an instance is the pair $\langle p, p^*\rangle$.

- $I_c$ is the set of instance with no error, $\eta(I)=1$.
- $I_r$ is the set of instance with error, for which a constant prediction $p'$ exists, that is $p'_j$ is independent of $j$, and with the same error, that is $\eta(p,p^\star)=\eta(p',p^\star)$.
- $I_s$ are all other instances.

The competitive ratio within each of these instance classes is called 

- consistency
- robustness
- smoothness 

The smoothness ratio is a function of eta, and the paper provides bounds for it. For the single machine problem: when eta is between 1 and 1.835, the smoothness is at least eta. When it is between 1.835 and $1+\sqrt3$, it is at least some specific function $\lambda(\eta)$. And above $1+\sqrt3$ it is at least 2. This lower bound is continuous. It is also shown that in all cases the smoothness of the _Shortest Processing Job First_ algorithm is at most $\eta^2$. Its analysis relies on previous work. There is a long discussion on how the worst case instances look like.

There are also lower and upper bounds given for the identical and the uniformly identical parallel machine scheduling problem. It improves over a work from 2023. For the identical parallel machine problem, there is an algorithm called SIMPLE from the 80's, which is 1+epsilon consistent and eta^2 robust, but not smooth. A modification is proposed where once a machine finishes all jobs assigned to it, steels pending jobs from other machines.

For the uniform parallel machine problem, an upper bound was know in 2024, and the paper provides an essentially matching lower bound.

### Strengths
Overall I think that this is a great paper. It is novel to distinguish instances where the prediction can actually tell something about the instance, in addition to a distinction by the prediction error. Then the results are good, sometimes tight. The results are not technically involved, but it does not always need to be the case in a good paper. The studied problems are among the most important online problems.  It leaves open problems, and hence will drive the community in an interesting direction.

Prediction error was mostly studied with respect to absolute difference. Considering the ratios for the error is in my opinion the better approach.

### Weaknesses
I had to read the definition of R(I) several times before understanding its deep meaning.

### Questions
Page 4 line 198, I found definition of $I_c$ confusing and propose $I_c=\\{I|\eta(T)=1\\}$.

I found the definition of $I_c, I_r, I_s$ not well explained. I would say that R(I) is the set of instances, for which there is a constant vector $q$ such that $\eta(p,q)=\eta(p,p^\star)$. In other words for a fixed prediction error, the set of possible actual job length vectors $p^\star$ include a constant one. Which means that the prediction error does not allow to partially order jobs according to their length, or even partition them according to the approximate length. These informations would be necessary to design a good schedule. Also I need some discussion on how this smoothness is related to the usual studied notion of the competitive ratio degrading smoothly with the prediction error.

Theorem 4.1 is hard to parse, because $\eta^*$ is defined at the end of the sentence.

I cannot parse Figure 3 on a black and white printout.

Page 8 line 386. For large error eta, the lower bound is 2, but the upper bound is 2-1/m. Hence something is not quite right.

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper considers non-clairvoyant scheduling with job size predictions for two different objective functions, total completion time minimization on a single machine and makespan minimization on identical and uniformly-related machines. In non-clairvoyant scheduling, the actual job sizes are unknown and only revealed once a job is completed. In the setting with predictions, we have a-priori access to predictions on the job sizes. Such learning-augmented algorithm are studied with respect to their consistency (the competitive ratio for correct predictions), their robustness (the competitive ratio for arbitrarily bad predictions) and their smoothness (an guarantee depending on a prediction error).

In this paper, the authors propose an alternative definition of smoothness, which only requires error-dependent guarantees for predictions that "reveal additional information". The author study this notion for the error $\eta$, which is the maximum ratio over all jobs between a predicted job size and an actual job size. For minimising the total completion time on a single machine, the authors prove an upper bound of $\eta^2$ and a lower bound of $\eta$. For minimising the makespan on identical machines, the authors show a lower bound of $2-\mathcal{O}(\eta^{-2})$ and an upper bound of $min(\eta^2,2)$. For minimising the makespan on uniformly related machines, they show a lower bound of $\lceil\log \eta \rceil$

### Strengths
* The paper gives more fine-grained error-dependent lower bounds for multiple non-clairvoyant scheduling problems. These lower bounds are non-trivial. In my opinion, error-dependent lower bounds are not sufficiently studied in the area of learning-augmented algorithms. Hence, I think that the given lower bounds are a significant contribution.

### Weaknesses
* I don't completely get the idea behind the partition of instances in the proposed smoothness definition. Smoothness bounds according to the new definition exclude predictions that "reveal no additional information". However, if the predictions reveal no additional information, then lower bounds for the original problem without predictions should always apply. Hence, even using the original definitions, such predictions should always lead to the algorithms guarantee ending up in the robustness case, the same as in the new definition. 
* The paper is missing a discussion of related work on flow time scheduling with predictions, e.g., "Distortion-Oblivious Algorithms for Scheduling on Multiple Machines" and "Flow time scheduling with uncertain processing time" by Azar, Leonardi and Touitou, and "Distortion-Oblivious Algorithms for Scheduling on Multiple Machines" by Azar, Peretz and Touitou. These works consider a more general problem, which contains total completion time minimization. Their distortion error is related to the multiplicative error that is considered in this paper. In particular, the upper bounds presented in these works should imply (weaker) upper bounds for the error measure used in this paper.
* The algorithmic results of Theorem 4.2 and Theorem 5.4 are not actually using the proposed new definition of smoothness. Instead, Theorem 4.2 is just an error-dependent bound and Theorem 5.4 is the minimum between an error-dependent bound and a robustness bound. Hence, both theorems just follow the original notions of consistency, robustness and smoothness. The proposed alternative notion of smoothness seems to only be used to give more fine-grained lower bounds. While these lower bounds are nice results, I think that algorithmic results are necessary to make a good case for establishing the proposed alternative notion of smoothness.
* The algorithmic results given in this paper are not very technically involved. I think the paper would benefit from more clearly stating that the main results are more fine-grained error-dependent lower bounds.

### Questions
* Could you elaborate on the advantages of your smoothness definition, taking the first weakness above into account?

### Soundness
3

### Presentation
2

### Contribution
2
