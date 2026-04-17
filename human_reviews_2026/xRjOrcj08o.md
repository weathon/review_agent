# Conformalized Decision Risk Assessment

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
High-stakes decisions in healthcare, energy, and public policy have long depended on human expertise and heuristics, but are now increasingly supported by predictive and optimization-based tools. A prevailing paradigm in operations research is predict-then-optimize, where predictive models estimate uncertain inputs and optimization models recommend decisions. However, such approaches often sideline human judgment, creating a disconnect between algorithmic outputs and expert intuition that undermines trust and adoption in practice.
To bridge this gap, we propose CREDO, a framework that, for any candidate decision proposed by human experts, provides a distribution-free upper bound on the probability of suboptimality---informed by both the optimization structure and the data distribution. By combining inverse optimization geometry with conformal generative prediction, CREDO delivers statistically rigorous yet practically interpretable risk certificates. This framework allows human decision-makers to audit and validate their decisions under uncertainty, strengthening the alignment between algorithmic tools and human intuition.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a new framework that can evaluate the probability of suboptimality for any decision with strong statistical guarantees. The authors reformulate the probability of suboptimality as the probability of the outcome variable belonging to an inverse feasible region. This probability can then be estimated using conformal prediction sets of varying levels of marginal coverage produced from $K$ samples from a generative model. The method is validated on two synthetic datasets and a real-world infrastructure planning problem.

### Strengths
The paper is well-written and well-organized. The theoretical contributions are meaningful in validating the proposed method. For example, Proposition 3 justified the use of generative models very well. This framework is a novel approach to handling risk in decision-making by quantifying it directly rather than being robust to it during optimization.

### Weaknesses
The authors compare their method with other robust decision-making methods using empirical confidence ranking. However, I am not convinced that this is the best metric to evaluate decision quality. How is the predicted decision’s rank in terms of its frequency in the ground truth optimal decision set informative of decision quality? This metric is instance-dependent, and so choosing an action $z$ that is optimal (i.e., $z \in \pi(Y; \theta)$), but rare among the optimal set of actions in the test set, would be discouraged by this metric. That doesn’t seem like a fair assessment of decision quality. An experiment evaluating decision quality seems crucial in building a case for viewing robust decision-making through a different lens. If the authors can clearly and strongly justify this metric or reproduce this experiment with a metric more indicative of decision quality, I will reconsider my score. 

Additionally, I believe that Kiyani et al. (2025) seems quite relevant to this line of work; however, it wasn’t included in the experiments. I believe adding this baseline can strengthen the paper.

_References_
* Kiyani et al. (2025), Decision theoretic foundations for conformal prediction: Optimal uncertainty quantification for risk-averse agents. https://arxiv.org/pdf/2502.02561.

### Questions
* In Figure 5 Column 2, why isn’t the Point Model just a flat line? It shouldn’t be changing with $K$ increasing.
* Why isn’t NS included in Columns 2 and 3 of Figure 5?
* Column 3 of Figure 5 appears to be inaccurately interpreted (Lines 432-437). While high accuracy, in the typical sense, indicates better performance, “accuracy”, as defined by the authors, seems to be like a loss (absolute difference between true and estimated risk). So, shouldn’t the method with lower “accuracy” be better?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a framework that provides distribution-free upper bounds on the probability that a given decision is suboptimal. The authors use inverse optimization space of the outcome and then construct a conformal set that is contained in the inverse space to produce the upper bound on the probability of suboptimal decision. The authors give a computational efficient algorithm when the objective function is a linear combination of decision and outcome. The authors validate the framework by experiments on synthetic and power-grid planning datasets.

### Strengths
- The paper's motivation is practical. The interpretable risk assessment in high-stakes domains is widely applicable and important.
- The theorems on conservativeness and the Monte Carlo interpretation demonstrate sound reasoning and attention to statistical guarantees.
- The idea of using inverse optimal space of outcome to find an upper bound on distribution-free probability is refreshing and can lead to potentially stronger results.

### Weaknesses
- The computational efficiency of the framework is not discussed. Finding the inverse space of the outcome where a given decision is optimal can be NP-hard for any objective function. It is also NP-hard to check whether the conformal set is included by the inverse space. The radius assumption here is still not enough since the inverse space can be non-convex.
- The upper bound that is found by the framework could be arbitrarily bad. That says there is no result on the lower bound of the probability of decision being suboptimal. I'm having this worry especially because the conformal set algorithm constructs the candidate space as a naive radius space. It is very easy to construct a case where the radius is arbitrarily small such that $\alpha$ is arbitrarily large.
- The linear form of objective function is not a very general form. A lot of utility functions in decision-making such as brier score cannot be converted to linear form.
- I found the paper is a little hard to follow. Some properties are not discussed. See more details in questions.

### Questions
- What are the intuitions on the generative model $\hat{f}$? How does it impact the quality of $\alpha$? Is there any guideline for choosing $\hat{f}$?
- Why the radius version of conformal set is taken instead of the more general one?
- What does this repeat K times do? What is the random variable here?

### Soundness
3

### Presentation
2

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
CREDO provides a distribution-free upper bound on the probability that a candidate decision is suboptimal, using inverse optimization geometry and conformal prediction with generative models. It enables practitioners to audit both algorithmic and expert-proposed decisions, offering statistically valid risk certificates. Theoretical guarantees and empirical results on synthetic and real-world tasks demonstrate that CREDO delivers conservative, interpretable, and actionable risk estimates, improving trust and decision quality compared to standard PTO and robust optimization approaches.

### Strengths
1. The paper addresses an under explored research area about how to provide rigorous, interpretable risk certificates for candidate decisions in high stakes, uncertain environments.
2. The "decide-then-assess" paradigm is a reasonable variation from the standard "predict-then-optimize" pipeline, and is well-motivated by practical needs for human-AI collaboration.
3. The use of inverse optimization geometry to characterize the optimality region for a decision is well done.
4. The integration of conformal prediction with generative models for risk estimation and the corresponding theoretical guarantees are clearly stated and proved.
5. The closed form solution for linear programs makes the method practical for large scale problems.
The experiments are well designed, covering both synthetic and real world settings (e.g., power grid planning).

### Weaknesses
1. While the method is general, the closed form efficiency is only for linear programs. For nonlinear or combinatorial problems, the computational cost of characterizing the inverse feasible region may be significant.
2. The approach assumes access to a well calibrated conditional model for the uncertain parameters. While the paper uses generative models to estimate the conditional distribution, in practice, any model that can accurately capture and sample from P(Y∣X) would suffice, including parametric or non-parametric approaches. I think it is limiting to claim the importance of generative models in this use case.
3. The method is conservative by design, but this can lead to loose risk estimates in some settings. The paper discusses this tradeoff, but more empirical analysis of the "tightness" of the certificates would strengthen the work.
4. The paper is motivated by human AI collaboration, but there is little discussion or experimentation on how practitioners actually use or interpret the risk certificates. A user study or qualitative feedback would be valuable.
5. The paper positions itself relative to robust optimization, DRO, and conformal prediction, but could more deeply discuss how CREDO compares to recent advances in human-in-the-loop optimization.

### Questions
1. For general nonlinear or combinatorial optimization problems, how is the inverse feasible region (\pi^{-1}(z;\theta)) practically characterized? Are there efficient relaxations that maintain the validity of the risk certificate, or does the method require exact computation?
2. The framework uses conformal prediction with generative models to construct inner approximations of the inverse feasible region. How sensitive is the risk estimate to the choice of conformal set (e.g., L2 balls vs. other shapes)?
3. What are the theoretical or empirical sample complexity requirements for the calibration set to ensure valid and non-trivial risk certificates, especially as the dimension of Y increases? How does the method perform with limited calibration data?
4. Can the CREDO framework be extended to settings where decisions are made sequentially or in multiple stages, with uncertainty revealed over time? What are the main challenges or limitations in such extensions?
5. Beyond generative models, have the authors empirically compared CREDO with approaches using quantile regression, Bayesian models, or ensemble methods for conditional uncertainty estimation?
6. Can the authors provide a more detailed analysis of the computational complexity of CREDO for both the linear and general cases, including the cost of generating samples, constructing conformal sets, and evaluating the inverse feasible region?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a method, to provide a statistically valid estimates of the probability of sub-optimality for any candidate decision proposed by a human expert. They do this through rewriting this problem in terms of an inverse optimization problem of the outcome and then construct a conformal set that is contained in the inverse space. They also provide a computationally efficient implfememntation of this method.

### Strengths
- The problem of designing ML-powered, distribution-free valid risk certificates is a timely and impactful problem. The authors make a meaningful step by connecting this problem to running CP on an inverse optimization problem.
- Their work has a very good balance of theory and experiment. The theoretical guarantees are of the interest of practice, and they showcase that it actually works in real world datasets.

### Weaknesses
My first concern is regarding "selection bias". Selection bias is well-known phenomena is statistics, which points toward the scenario where a decision maker wants to use an estimation to inform their decisions. The bias arises, when the estimation is performed without the knowledge of the down stream decision problem, and this can potentially disrupt the statistical guarantees of the original estimation. In the context of the problem that is studied in this paper, it shows itself as follows: say a decision maker want to pick a decision such that the risk certificates is larger than \tau. Now what they could do is to run your algorithm, get the certificate, and then filter out the decisions that the corresponding certificate is less than \tau, and pick one of the remaining. Now the issue is, from the viewpoint of the decision maker, what actually matters is that the risk certificate be statistically valid conditioned on the certificates larger than \tau. If this doesn't hold, then the algorithm suffers from "selection bias". The question is, whether this algorithm suffers from such a situation (my guess is yes), and if yes, to what extend (this could be certified in experiments), and if the bias is significant, how to fix it (this could at least be discussed\reported in the future works and limitations).

The second concern is regarding a proper discussion regarding the two recent works of [1] and [2]. Although the scope of those papers, particularly [1], is to "find" a low-risk action, rather than "certify" the low risk ones, however, one can use their method to derive risk certificates too. The idea is, you can run your favorite CP method, to get a set of labels C(x). Then if you run the max-min rule defined in these papers, you get both a certificate (as the max-min value), and an action (the argmax-min). Alternatively, you can ignore the outer max, and for each action candidate of your interest, you can just solve the inner min of their method, and that would give you a risk certificate that holds with high probability. There needs to be a discussion to distinguish this simple method with yours, in terms of scope, practicality, and use cases.

I like this paper, and if the authors provide a satisfying answer to these concerns I would be happy to vote for acceptance.


[1]: Decision Theoretic Foundations for Conformal Prediction: Optimal Uncertainty Quantification for Risk-Averse Agents, Kiyani et. al.

[2]: Certified Decisions, Andrews et. al.

### Questions
.

### Soundness
3

### Presentation
3

### Contribution
2
