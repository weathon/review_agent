# Singleton-Optimized Conformal Prediction

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Conformal prediction can be used to construct prediction sets that cover the true outcome with a desired probability, but can sometimes lead to large prediction sets that are costly in practice. The most useful outcome is a singleton prediction---an unambiguous decision---yet existing efficiency-oriented methods primarily optimize average set size. Motivated by this, we propose a new non-conformity score that is motivated by minimizing the probability of producing non-singleton sets while maintaining coverage. Starting from a non-convex constrained optimization problem as a motivation, we provide a convex-geometric reformulation and associated algorithm for computing the non-conformity score and associated split conformal prediction sets in $O(K)$ time for $K$-class problems. Using this score in split conformal prediction, we introduce Singleton-Optimized Conformal Prediction (SOCOP). We evaluate our method in experiments on image classification and LLM multiple-choice answering, comparing with standard non-conformity scores such as the (negative) label probability estimates and their cumulative distribution function; both of which are motivated by aiming to optimize average length. The results show that SOCOP increases singleton frequency (sometimes by over 20\%) compared to the above scores, with minimal impact on average set size.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
**Summary**

This paper addresses a key limitation of conformal prediction: while it creates reliable prediction sets (e.g., a list of possible answers that is 95% likely to contain the true one), these sets often contain multiple items, making them ambiguous for automated decision-making. The authors argue that the most useful outcome is a set with just one answer (a "singleton"), as it provides a clear, actionable result. 
They introduce a method called SOCOP (Singleton-Optimized Conformal Prediction) which is specifically designed to maximize the frequency of these singleton predictions. 
Instead of only trying to make the prediction sets as small as possible on average, SOCOP try to produce unambiguous, single-answer outputs, creating a more practical trade-off for real-world applications where decision clarity is more important than a slightly smaller average set size.


**Contribution**

The main contribution is a novel and highly efficient method to achieve this goal. 
The authors derive a new "nonconformity score" from an optimization objective that directly penalizes non-singleton sets. 
Their key innovation is a geometric insight that transforms the complex task of calculating this score into a simple computational geometry problem: finding the lower convex hull of a set of 2D points. 
This allows the score to be computed in fast time, making the method practical even for problems with thousands of classes. 
Experiments on image classification and large language models show that SOCOP significantly increases the rate of singleton predictions (often by over 20%) with only a minimal cost to the average set size, proving its effectiveness and practical value.

### Strengths
*   The paper's main strength lies in its novel objective function and the corresponding insight into evaluating conformal prediction. By shifting the focus from minimizing average set size to minimizing the probability of non-singleton sets (`P(size > 1)`), the work introduces a practical and well-motivated criterion for efficiency that directly relates to the goal of automated decision-making. This reframing provides a valuable perspective on what constitutes a "useful" prediction set in many real-world applications.

*   This conceptual contribution is supported by a thorough and well-explained computational algorithm. The paper provides a deep discussion on how to make the proposed nonconformity score practical, culminating in an efficient O(K) method based on a geometric reformulation involving the lower convex hull. This ensures the method is not just a theoretical construct but a scalable tool that can be readily applied.

*   The work can be seen as a natural and timely extension of recent research on improving the efficiency of conformal sets, such as the work on length optimization by [1]. While that paper focused on minimizing the expected set size, this paper explores a related but distinct objective—maximizing singleton frequency. This progression shows a thoughtful engagement with the state-of-the-art and pushes the conversation forward on defining and optimizing different notions of efficiency in conformal prediction.


[Reference] 

[1] Kiyani, Shayan, George J. Pappas, and Hamed Hassani. 2024. “Length Optimization in Conformal Prediction.” In Neural Information Processing Systems. Vol. 37. Curran Associates, Inc.

### Weaknesses
*   The practical significance of the proposed objective could be further elaborated. While the goal of maximizing singletons is intuitive, the paper could benefit from a more direct discussion on the distinction between the method's singleton predictions and simply using the standard top-1 prediction. A clearer explanation of the trade-offs would help readers understand when the statistical guarantees of a conformal singleton justify the method's complexity, especially since it can still produce non-singleton sets.

*   The method introduces a hyperparameter, $\lambda$, that requires tuning, which can be a practical challenge. The paper could provide a more in-depth discussion on the sensitivity of the results to the choice of $\lambda$. It might also be beneficial to explore the potential impact of tuning bias on the final performance, a topic investigated in recent work such as [1], and consider if more principled selection methods could be developed.

*   The clarity of the writing in the methodology section could be improved. While the experimental results are presented clearly, the section explaining the method itself consists of large blocks of text that can make it difficult to follow the main ideas. The presentation would be enhanced by breaking down the key concepts more, perhaps by using more structured text or adding a high-level diagram to help illustrate the geometric approach.

[Reference] 

[2] Zeng, Hao, Kangdao Liu, Bingyi Jing, and Hongxin Wei. 2025. “Parametric Scaling Law of Tuning Bias in Conformal Prediction.” In Forty-Second International Conference on Machine Learning.

### Questions
1.  Could you please make the practical difference between a singleton set from your method and a standard top-1 prediction? For instance, in what specific scenarios would a decision-maker prefer a SOCOP singleton over a top-1 prediction, especially considering SOCOP still produces multi-sets in other cases? A more direct comparison of their respective risk profiles would be helpful.

2.  The performance of SOCOP depends on the hyperparameter $\lambda$, which is selected on a tuning set. How sensitive is the method's performance to the choice of $\lambda$ and the size of the tuning set? Could you comment on the potential for tuning bias, as discussed in works like [2], and perhaps provide practical guidance for users on how to select $\lambda$ robustly?

3.  Could you share your thoughts on potential future directions for this work? For example, could this singleton-optimization framework be extended to other, more complex conformal prediction methods beyond the split-conformal setting, such as those that aim for conditional coverage?

4. For theorem part, is there any insight for the P(size <1)'s increasing under proposed method?

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
4

### Summary
The paper targets increasing the probability that conformal prediction sets are singletons (sets of size one) in multi-class classification. It proposes a new nonconformity score derived from a Lagrangian relaxation of a joint objective balancing singleton frequency and expected set size, and introduces an O(K) per-instance algorithm via a lower convex hull construction. Empirically, the method increases singleton frequency with small impact on average set size while maintaining marginal coverage.

### Strengths
Clarity and organization: The paper is well written and easy to follow. The problem setup, the Lagrangian relaxation, and the geometric algorithm are clearly presented.

Novel scoring and efficient computation: The nonconformity score is new and is explicitly motivated by the singleton objective rather than average size alone. The geometric O(K) algorithm based on the lower convex hull is elegant and practically important.

### Weaknesses
1. The statement that “often the most desirable outcome is an unambiguous prediction, a singleton set containing only one label” needs stronger justification. Conformal prediction naturally returns sets of varying sizes; prioritizing singletons is application-dependent rather than universally optimal. Please provide concrete justification and cite papers explicitly values singleton sets.

2. Compared with APS [1], the method provides marginal but not conditional validity when the true $p(y\mid x)$ is given.

3. While the score is motivated by minimizing $P(|C(X)|>1) + \lambda E[|C(X)|]$, similar composite losses can potentially be handled within existing efficiency-oriented frameworks (e.g., [2]). Please clarify why a new score is necessary rather than adapting losses within those frameworks, and whether SOCOP can be integrated into them while preserving nestedness, monotonicity, and computational benefits.

[1] Romano, Yaniv, Matteo Sesia, and Emmanuel Candès. Classification with valid and adaptive coverage. NeurIPS 33 (2020): 3581–3591.

[2] Liang, Ruiting, Wanrong Zhu, and Rina Foygel Barber. Conformal prediction after efficiency-oriented model selection. arXiv:2408.07066 (2024).

### Questions
The definition of a singleton set should be more precise. The definition used in the formulas is $|C(X)| \le 1$, which differs from $|C(X)| = 1$. Empty prediction sets can occur in conformal prediction. For example, if $\hat p$ can almost perfectly predict the true class, then approximately an $\alpha$ fraction of least ambiguous sets will be empty. If we use $|C(X)| = 1$ as the definition of a singleton set and change the first term of the loss in line 135 to $I(|C(X)| \ne 1)$, this may harm the monotonicity of the score function.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a new conformal score function designed to produce prediction sets that minimize a weighted sum of the probability of singleton sets and the expected set size, subject to lower bound on marginal coverage. This generalizes the LAC score function proposed in Sadinle et al., 2019, which is a special case of this paper’s score function with the weight on expected set size set to infinity. Computing this score function is nontrivial, and the paper derives a way to compute it efficiently in O(K) time, where K is the number of classes. The method is validated on image and Q&A datasets and we see that the proposed method does a good job balancing average size and P(size > 1).

### Strengths
* The paper is very well written. Some parts I found particularly useful: the discussion of connections to Sadinle et al., 2019 and the comparison in Figs 2-3; the motivation for taking a linear combination of the singleton and size objectives starting on line 311
* This method can be easily extended to P(size > $k_0$) for any integer $k_0 \geq 1$. This seems quite practically useful. 
* The experiments are extensive and shed insight into the operating characteristics of the proposed method.

### Weaknesses
There are some places the writing could be improved. See Questions

### Questions
* Can you provide more intuition in the paragraph “Computing the nonconformity score”? My attempt is “The conformal score is the smallest “price” $\eta$ in terms of the set size metric (singleton + size objective) per unit of “coverage” such that if we “buy” all classes with price less than $\eta$, we will have purchased the true class. I'm not sure if this is precisely correct, but something like this would be useful. 
* Why do you use Plug-In sets (which do not have a marginal coverage guarantee) instead of APS (which is essentially the same procedure but with the threshold tuned to achieve a marginal coverage guarantee)?
* “Our method SOCOP outperforms Plug-In and RAPS in both Average Size and P(size > 1). We find this result remarkable […]” — I don’t think the second sentence is warranted. Plug-In/APS and RAPS target X-conditional coverage. How does SOCOP do on this metric? 
* On line 382, you say that ImageNet-V2 is more challenging test dataset. Can you briefly describe why in the text?
* A class-conditional extension of this method that is mentioned in the Discussion section seems quite important to me, as there are many classification tasks where class-conditional coverage is important. However, I don’t immediately see a way that the current methodology could be extended. Do you have thoughts on how? 


Small corrections/suggestions:
* Line 162: “result;” -> “result,”
* Line 200: “that, for k >= 3, that” -> “that, for k >= 3,”
* Line 231: “the the jumps”
* Line 267: “is set” -> “is the set”
* Line 337: “Results are reported” -> “Results for this method are reported” (initially I thought this sentence was referring to results for all methods, not just CPL)
* Line 365: “Imagenet” -> “ImageNet”
* Line 399: Fig 2 caption “for of”

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a new nonconformity score with the goal of reducing the probability of producing non-singleton sets. The score is motivated by an optimization problem that optimizes a combination of the singleton objective and the expected length of the prediction sets subject to coverage constraint. The authors provide an efficient algorithm through a geometric reformulation to compute the nonconformity score. The paper includes empirical evaluation on image classification and LLM multiple-choice question answering that shows the proposed method reduces average set size and increases the frequency of singleton predictions.

### Strengths
The motivation behind the problem of singleton sets is important in practical settings. The paper presents a theoretically sound and efficient algorithm for the problem. Empirical evaluation shows that the method reduces the non-singleton probability while maintaining the average set size. Additionally, the extensions discussed in Section 2.2 are appreciated.

### Weaknesses
1. The paper is not written very clearly and it is hard to follow. The notation is extensive and dense in Section 2, with several new symbols introduced with little intuition. Additionally, the definition of nested conformal prediction was deferred for later, while the concepts are used in the paper; e.g., l215 ‘the nested sets property implies that…’ – this hampers the readability of the paper. It is suggested to work on this in the revised version.
2. While the paper guarantees marginal validity, there is no discussion on the implications of this objective on conditional coverage. Such analysis (theoretical and empirical) is warranted to understand the underlying tradeoffs.
3. Apart from Coverage and Avg Size, the empirical evaluation should report metrics that measure adaptivity of the sets e.g., worst-slice coverage, SSCV. The current evaluation is limited in its scope of studying the Avg Size and P(Size > 1) that are directly optimized by the method. To understand the usefulness of the sets, the analysis will benefit from additional metrics.

### Questions
Comments/typos:

- P4 l201: extra‘that’
- P5 l231: extra ‘the’

### Soundness
3

### Presentation
2

### Contribution
3
