# Federated Learning of Quantile Inference under Local Differential Privacy

- Avg Score: 4.40
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 4, 4

## Abstract
In this paper, we investigate federated learning for quantile inference under local differential privacy (LDP). We propose an estimator based on local stochastic gradient descent (SGD), whose local gradients are perturbed via a randomized mechanism with global parameters, making the procedure tolerant of communication and storage constraints without compromising statistical efficiency. Although the quantile loss and its corresponding gradient do not satisfy standard smoothness conditions typically assumed in existing literature, we establish asymptotic normality for our estimator as well as a functional central limit theorem. The proposed method accommodates data heterogeneity and allows each server to operate with an individual privacy budget. Furthermore, we construct confidence intervals for the target value through a self‐normalization approach, thereby circumventing the need to estimate additional nuisance parameters. Extensive numerical experiments and real data application validate the theoretical guarantees of the proposed methodology.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors consider the problem of estimating a quantile in a federated learning setting with a local differential privacy constraint.  They present a new algorithm for computing this central quantile and extensive asymptotic analysis that can be used for statistical inference.  They highlight how estimating the quantile in this LDP setting is equivalent to estimating a quantile in a non private setting, but with the distribution shifted in a known way (i.e. solving one gives a computable solution to the other).  To get around estimating nuisance parameters, the authors also present a self-normalized approach based on a functional central limit theorem.

### Strengths
Overall the paper tackles an interesting problem and presents extensive mathematical results.

### Weaknesses
The organization and writing of the paper is interesting, but feels terse or missing in details at times.  See questions.

### Questions
Why in Theorem 3.2 do you use $\ell^\infty$ instead of $D[0,1]$ (the Skorohod space)?  My understanding is that a partial sum process like this is not tight in $\ell^\infty$ so you have to work in $D[0,1]$.  Or if you want to work in $\ell^\infty$ you can linearly interpolate the partial sum process so that you have continuous paths (and then you are essentially working in C[0,1]).  But as stated, I'm not sure Thm 3.2 is actually correct since a lack of tightness implies you don't have a functional CLT (in the given space).  But let me know if I'm missing something as it's been a while since I worked on functional CLTs.

I would argue that (2.1) is not the goal, but that the goal is outlined in lines 147-149.  It just so happens that the minimizer of (2.1) gives you the desired parameter,  i.e. the goal is to estimate the quantile of some target population.

The discussion around Thm 2.1 is quite weak and should be expanded.  

The discussion/motivation for Thm 3.2 could be strengthened before being presented.  Just saying "it is necessary to strengthen" it, isn't really motivating the result.  

Line 48: "Guarantees convergence" is a bit vague.  Statistically or computationally?  

Line 61: Nasr should be in parentheses.  In general, the paper's handling of references is really all over the place.  References out of the sentence should be in parentheses, those that are part of the sentence shouldn't be, and you shouldn't have double parentheses like …(e.g.(Zhao…  I suggest the authors go through and clean these up.

Ln 124: When talking about estimation accuracy, I think the privacy parameters should be in the rate as well.  The choice of $\epsilon$ or $\delta$ play a large practical role.

Ln 130: The author should note the $\theta$ here is the one computed from the data, as opposed to the population level quantity.

Ln 151-161: The setup for the estimator is not very clear.  I suppose the authors are saying that at each step each client either updates locally or uses the current estimates from all of the other clients. But it is presented in a confusing fashion.  E.g. "$x_k^t$ is an independent realization of P_k", yes, but it is also just one of the datapoints held by a client.  "$E_m = n$" is divide an conquer?  So all clients have the same sample sizes?  I suppose $\mathcal I$ is ordered.  I would just add more details for the reader.

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
The paper proposes a federated algorithm for quantile estimation and inference under
Local Differential Privacy (LDP) using local stochastic gradient descent (Local-SGD). The
method perturbs local gradients via a randomised-response mechanism that ensures client-
level privacy while maintaining statistical efficiency. Key theoretical contributions include:
(i) an equivalence theorem reducing the LDP problem to a non-private one with adjusted
quantile levels, (ii) asymptotic normality and functional central limit theorem (FCLT) for the
non-smooth quantile loss, and (iii) a self-normalised confidence interval method that avoids
estimating nuisance parameters. Experiments and a real-data application validate the
theoretical results.

### Strengths
• Novel formulation combining federated learning, quantile inference, and local differential
privacy — a previously unexplored combination.
• Theoretical depth: establishes asymptotic normality and FCLT for non-smooth quantile
loss without the average-smoothness assumption.
• Elegant reduction (Theorem 2.1) that transforms the privatised problem into an
equivalent non-private version, enabling classical analysis.
• Self-normalised inference avoids density estimation, conserving privacy budget and
improving practical applicability.
• Comprehensive experiments demonstrating good accuracy and confidence interval
coverage across various privacy levels and heterogeneity settings.
• Clear discussion of limitations and directions for future research.

### Weaknesses
• The theoretical results depend on several strong regularity assumptions, particularly
bounded densities and step-size control, which may limit practical generality.
• The self-normalized approach yields heavier-tailed distributions, producing conservative
confidence intervals and lower test power.
• Dependence on a central aggregation server restricts use in fully decentralized federated
systems.
• Empirical validation is limited to tabular, one-dimensional quantiles — no high-
dimensional or complex FL tasks are explored.

• Certain proof steps (e.g., adaptation of Li et al., 2022 assumptions) rely on claims without
complete derivation; these could be more explicit for reproducibility.

### Questions
• Provide a proof sketch for Theorem 2.1 explicitly showing how the randomized response
induces the r_k scaling and τ̃_k shift.
• Expand on the removal of Assumption 1 from Li et al. (2022) — clarify exactly which steps
are substituted by the new bound E|q̄−Q*|² ≲ γ_m.
• Include empirical analysis of conservative confidence intervals — e.g., actual vs nominal
coverage rates.
• Test robustness under heavy-tailed client data (e.g., Cauchy distributions) to validate
bounded-density assumptions.
• Discuss communication-computation trade-offs more quantitatively under different local
update frequencies E_m.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper addresses a significant and challenging problem at the intersection of federated learning, statistical inference, and privacy preservation: performing quantile inference under local differential privacy (LDP). The authors propose a novel federated learning algorithm based on local Stochastic Gradient Descent (SGD), where client gradients are perturbed using an LDP mechanism. The key innovation is a theoretical analysis that establishes the asymptotic normality and a functional central limit theorem (FCLT) for their estimator, even though the quantile loss function is non-smooth. Leveraging this FCLT, they develop a self-normalized inference procedure that constructs valid confidence intervals without needing to estimate the asymptotic variance directly, thus avoiding additional privacy costs. Extensive simulations and a real-data application on government salary data demonstrate the method's effectiveness under various heterogeneity settings.

### Strengths
1.  Important and Timely Problem: The combination of federated quantile estimation with the strong privacy guarantee of LDP is highly relevant. Quantile inference is crucial for understanding data heterogeneity (e.g., in healthcare or economics), and LDP is the gold standard for privacy in distributed settings where a central server is not trusted. Tackling inference under these constraints is a substantial contribution.
2.  Significant Theoretical Contribution: The paper's primary strength is its theoretical analysis. Proving a functional central limit theorem for a local SGD-based estimator with a non-smooth loss function is a non-trivial and valuable advancement. This provides a solid foundation for inference in a setting where standard smoothness assumptions fail.
3.  Practical and Elegant Inference Solution: The use of self-normalization to construct confidence intervals is a clever and highly practical solution. It bypasses the notoriously difficult problem of estimating the asymptotic variance under LDP, which would typically require additional privacy budget or complex techniques. The resulting method is computationally efficient and integrates seamlessly with the federated learning workflow.
4.  Comprehensive Experimental Evaluation: The paper validates its theoretical claims with extensive simulations covering a wide range of heterogeneity scenarios (in quantile levels, privacy budgets, and data distributions). The inclusion of a real-world dataset (government salaries) strengthens the practical relevance of the findings. Comparisons against appropriate baselines like DP-SGD and a divide-and-conquer method are well-executed.

### Weaknesses
1.  Reliance on Strong Assumptions: The theoretical analysis relies on assumptions such as the boundedness of the parameter space and the uniform boundedness of the client density functions. While common in theoretical proofs, the practical implications of these assumptions could be discussed further (e.g., how to ensure parameter space boundedness in practice).
2.  Inherent Conservatism of Self-Normalization: The authors correctly note that self-normalization can lead to heavier-tailed limiting distributions than the standard normal. This often results in conservative confidence intervals (wider than necessary), which can reduce statistical power in hypothesis testing. This is a well-known trade-off of the method but is an important limitation to highlight.
3.  Centralized Server Dependency: The proposed framework requires a central server for aggregation and synchronization. The algorithm's applicability to fully decentralized or peer-to-peer federated learning environments, an active area of research, is not explored.
4.  Limited Analysis of Communication/Computation Trade-offs: While different communication strategies (C1, C5, Log) are compared, a more detailed analysis of the total communication cost and wall-clock time (computation + communication) compared to baseline methods would be beneficial for assessing real-world efficiency.

### Questions
1.  Have you quantified the degree of conservatism introduced by the self-normalization approach in your experiments? Are there potential adjustments or alternative techniques to mitigate this while remaining within the LDP constraint?
2.  Can your theoretical framework be extended to more complex models, such as high-dimensional or conditional quantile regression? What would be the main challenges?
3.  How would the algorithm perform under more realistic, asynchronous client participation patterns (e.g., clients joining at random times)? Does the theory accommodate such scenarios?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper propose to use federated learning (FL) for quantile inference under local differential privacy (LDP). The core idea of this paper is to design an LDP mechanism that can transform the LDP federated quantile estimation into the non-DP case. In addition, via constructing valid confidence intervals under LDP, the authors develop a self-normalized inference procedure without requiring direct estimation of the asymptotic
variance. The extensive experiments evaluate the effectiveness of the proposed method. Overall, this paper considers a new problem in this area. However, I have some concerns as follows: 1) I cannot see any baselines in the experiments. Although the authors claim that this is the first inference framework for federated quantile estimation, related works are existing and also can be adopted as baselines; 2) The theoretical analysis is not complete, especially for DP. The authors claim that they transform the LDP federated quantile estimation into the non-DP case. Can this transformation maintain DP guarantee? 3) Existing results in simulation cannot evaluate the proposed methods well. Some more experimental should be added.

### Strengths
This paper propose to use federated learning (FL) for quantile inference under local differential privacy (LDP). The core idea of this paper is to design an LDP mechanism that can transform the LDP federated quantile estimation into the non-DP case. In addition, via constructing valid confidence intervals under LDP, the authors develop a self-normalized inference procedure without requiring direct estimation of the asymptotic
variance. The extensive experiments evaluate the effectiveness of the proposed method. Overall, this paper considers a new problem in this area and well written.

### Weaknesses
However, I have some concerns as follows: 1) I cannot see any baselines in the experiments. Although the authors claim that this is the first inference framework for federated quantile estimation, related works are existing and also can be adopted as baselines; 2) The theoretical analysis is not complete, especially for DP. The authors claim that they transform the LDP federated quantile estimation into the non-DP case. Can this transformation maintain DP guarantee? 3) Existing results in simulation cannot evaluate the proposed methods well. Some more experimental should be added.

### Questions
1) I cannot see any baselines in the experiments. Although the authors claim that this is the first inference framework for federated quantile estimation, related works are existing and also can be adopted as baselines; 
2) The theoretical analysis is not complete, especially for DP. The authors claim that they transform the LDP federated quantile estimation into the non-DP case. Can this transformation maintain DP guarantee? 
3) Existing results in simulation cannot evaluate the proposed methods well. Some more experimental should be added.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper considers the statistical inference of federated learning for quantile loss via local stochastic gradient decent under local differential privacy. It proposes a local-SGD algorithm with randomized response on the binary quantile “gradient” supports heterogeneous clients, and outputs a Polyak–Ruppert averaged estimator. Then, its theoretical properties were explored and experimentally verified.

### Strengths
S1. This work designs an LDP mechanism that can transform the LDP federated quantile estimation into the non-DP case, and then derive the asymptotic normality and functional central limit theorem of the proposed estimator under non-DP cases.

S2. It is the first weak-convergence result for local SGD without the usual average-smoothness assumption in existing literature

### Weaknesses
My comments focus on presentation and experimental design issues.

W1. The introduction section reads like an extended related work. Missing the importance and/or motivation of the question. I suggest giving 1–2 concrete use cases, or moving a simplified version of Figure 1 to the intro to anchor the problem.

W2. The Methodology section lacks early notation and problem formulation. Even Q is not defined.

W3. In the Experiment section, placing synthetic results in the appendix is a bad idea. There is no direct comparison against prior work (e.g., Liu et al, 2023b). Table 1 lacks guidance on “what good looks like”, nominal 95% coverage with the shortest average CI length? Why not visually highlight when empirical quantiles fall inside/outside CIs?

W4. For the truthful response rate, I am unsure who sets $r_k$, whether it is policy-driven, user-chosen, or purely synthetic. The phrase "range from 0.6 to 0.9" in real data is missing a clear definition. I suggest adding the sensitivity of the experiment regarding the truthful response rate.

W5. In Asymptotic Analysis, are the theorems and algorithms (partially) verified in the experimental section? If not, is verification possible?

### Questions
See W1-W5.

### Soundness
2

### Presentation
1

### Contribution
2
