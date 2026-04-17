# DeNOTS: Stable Deep Neural ODEs for Time Series

- Decision: Accept (Poster)
- Scores: 6, 4, 2, 6

## Abstract
Neural Controlled Differential Equations (Neural CDEs) provide a principled framework for modelling irregular time series in continuous time. 
Their number of function evaluations (NFEs) acts as a natural analogue of depth in discrete neural networks and is typically controlled indirectly via solver tolerances. 
However, tightening tolerances increases numerical precision without necessarily improving expressiveness.
We propose a simple alternative: scaling the integration time horizon to increase NFEs and thereby "deepen" the model. 
Since enlarging the interval can cause uncontrolled growth in standard vector fields, we introduce a Negative Feedback (NF) mechanism that ensures provable stability without limiting flexibility. 
We further establish general risk bounds for Neural CDEs and quantify discretization error using Gaussian process theory, improving robustness to integration and interpolation error.
On four public benchmarks, our method, **DeNOTS**, outperforms existing approaches—including Neural RDEs and state space models—by up to $20$%. 
DeNOTS combines expressiveness, stability, and robustness for reliable continuous-time modelling.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces two refinements of Neural CDEs. The first idea is to increase the time horizon of the differential equation in order to increase expressivity. However, this introduces either instability (if the CDE is diverging) or forgetfulness (if the CDE is contractive). The second idea is to add a negative feedback term proportional to the current state, to avoid instability, and to modulate the magnitude of the feedback term through a gate, in order to reduce forgetfulness. This is equivalent to a standard GRU with a minus sign in the gate. Numerical experiments show that the proposed method outperforms both alternative Neural CDE-like models and SSMs. Some theoretical properties are derived, showing improved robustness of the proposed method.

### Strengths
Improving the expressivity and stability of neural ODEs/CDEs is an important line of work. The idea proposed by the authors is conceptually simple, yet seems novel and sound. The theoretical and numerical evidence is thorough and convincing. I did not check the proofs.

### Weaknesses
The paper discusses many different theoretical results and experiments, which is interesting, but at the same time makes it difficult to follow in some places. For instance:
- the last column of Table 3 is not very clear (I understand that it means that the assumption holds if (and only if?) the condition written is satisfied, but this is not explained). A related comment is that it is stated on line 232 that they analyze empirically whether Assumption 4.2 holds, but on line 243 that is always stands. I did not understand the argument on line 243 and how it relates to the empirical analysis.
- in Theorem 4.5, my understanding is that authors compute how local errors propagate in the ODE, and show a bound on the final state depending on the local errors. I do not understand where the MAP comes into play. It looks to me like $\hat x$ could be any estimator satisfying (8)?
- Section 4.2.1: this section looks interesting but is very difficult to follow. For instance, who is $S$ in Assumption 4.6? What does it mean to assume that the sequence is infinite in our setting? Where does the spectral density of Lemma 4.7 come from?
- Table 5: there is no confidence interval, so it is difficult to assess the robustness of the comparison between Sync-NF and Anti-NF. This makes the hypothesis of lines 405-406 quite brittle.

### Questions
See weaknesses.

### Soundness
3

### Presentation
2

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
This paper focus on solving downstream tasks for time series with global, sequence-wise targets (binary/multiclass/regression).
The core idea is to deliberately ​scale the integration time horizon​ instead of lowering solver tolerances to increase the model's "depth" (Number of Function Evaluations - NFE), which leads to superior expressiveness. 
They also introduced a negative feedback mechanism (derived from control theory) to enhance stability.

### Strengths
This article employs a variety of theoretical tools and proposes a straightforward and easy-to-use methodological innovation (time scaling and negative feedback).

### Weaknesses
Although this article is based on theoretical analysis, I am somewhat skeptical about whether they can correctly validate its arguments. In my view, it is not entirely logically rigorous. The specific details are listed in the Questions part.

### Questions
Theorem 3.1 concludes that scaling T makes the model 'exponentially' more expressive, so it's worth  scaling T. 
However, we can also easily reach the following conclusion: scaling T would also 'exponentially' impairs the model's performance. For example, as a well known result [1], for an ODE system, the error of the Euler numerical solution (which is required both in training and inference stages) is $|\varepsilon_n| \leq e^{T L} |\varepsilon_0| + \frac{R}{hL} (e^{TL} - 1)$, which is also exponential of T. In fact, the difficulty of training Neural ODEs or RNNs lies in the need for backpropagation-through-time to track the gradients of weights (e.g., in [2], "Flow-based models were previously limited by inefficient simulation-based training objectives that require an expensive integration of the ODE at training time."). 
What I'm concerned about is whether extending the time T will touch upon the core pain points of such models.
So at present, I cannot see that scaling time is an essential improvement. It can certainly enhance the model's fitting ability, but it will also increase the difficulty and error in training and inference for the model (also exponentially).

As for the issue of increasing NFE (network depth) that you mentioned, with the same error tolerance (under the adaptive step-size framework), the size of network weights, and the degree of function stiffness can also affect NFE. If you require a longer integration time to achieve a smoother vector field function, NFE may not necessarily increase, so there may not be a necessary connection between the two. And larger l_2 but small integration times, under the same error tolerance,  I cannot believe there would be any fundamental difference in the results.

[1] Hairer, Ernst, Gerhard Wanner, and Syvert P. Nørsett. Solving ordinary differential equations I: Nonstiff problems. Berlin, Heidelberg: Springer Berlin Heidelberg, 1993.

[2] Tong, Alexander, Nikolay Malkin, Kilian Fatras, Lazar Atanackovic, Yanlei Zhang, Guillaume Huguet, Guy Wolf, and Yoshua Bengio. "Simulation-free schr\" odinger bridges via score and flow matching." arXiv preprint arXiv:2307.03672 (2023).

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces DeNOTS, a new variant of Neural CDEs for time series. The authors proposes two main modificaitons: time scaling increaisng time interal imprvoe the expressive, while causing exploring vector fields, hence, propose use negative feedback to stablize long horizon integration. They claim that DeNOTS improves expressivlity and stabling as increasing time horizon.

### Strengths
try to study an important problem with theoreical motivaiton, to improve expreisivty and stalbity.

### Weaknesses
1. I am not convinced by the theoretical results and analysis. Theorem 3.1 shows that a larger integration horizon $T$ leads to a larger $L_F$, but this merely indicates greater output variance, not increased expressivity. To rigorously support the claim that longer integration time enhances expressivity, the authors should establish a functional inclusion relation between model families; e.g., for two horizons $T<T'$, show that the corresponding function classes $H_T \leq H_{T'}$. Without such reasoning, the argument that "increasing $T$ improve expressivity" is not theoretically substantiated.
2. Negative feedback may suppress dynamics rather than enhance expressivity. With the proposed negative-feedback mechanism, increasing $T$ does not always enrich the representation. Consider the scalar linearized dynamics $h' = af(x,h)-b h$ with $f(x,h)\approx \lambda h$. This yields the closed form solution: $h_t = h_0 e^{(a\lambda -b)t}$. With $a+b=1$, if $b$ is small and close to zero, $h_t \propto e^{\lambda t}$ diverges or vanishes exponentially depending on the sign of $\lambda$; if $b$ is large and close to one, $h_t\propto e^{-t}$ decays exponentially to zero; and if $b$ is balanced such that $(a\lambda -b)t\approx 0$, then $h_t$ remain nearly close to its initial $h_0$ with minimal dynamics. In all these regimes, longer $T$ does not meaningfully improve expressivity, contradicting the paper's central claim.
3. The reported gains (1–3 % in R^2 or AUROC) are within the expected variance and are not statistically significant. Given the small benchmark size and lack of confidence intervals, the empirical evidence does not convincingly demonstrate a substantive improvement over existing Neural ODE or CDE baselines.

### Questions
See the weakness.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes DENOTS (Deep Neural ODEs with Negative Feedback and Time Scaling), a novel neural ODE method for time series analysis. The authors introduce two main innovations: (1) time scaling to increase the number of function evaluations (NFE) without increasing weight norms, and (2) an anti-phase negative feedback (Anti-NF) mechanism to stabilize the dynamics. The paper claims that DENOTS achieves state-of-the-art performance on four public datasets. However, the experimental design is insufficient to fully validate the contributions, and the related work discussion is not comprehensive.

### Strengths
1.Novel Time Scaling Approach: The idea of scaling the integration time range to increase NFE without increasing weight norms is conceptually interesting and has potential to improve model expressivity.
2.Practical Performance: The experimental results demonstrate that DENOTS outperforms several baseline methods on multiple datasets, showing practical utility.
3.Stability Considerations: The inclusion of negative feedback mechanisms to address stability issues in neural ODEs is a thoughtful contribution.
4.Experimental Design: The paper includes experiments on multiple datasets and ablation studies to validate different components of the method.

### Weaknesses
1.Restrictive Assumptions: The theoretical analysis relies on restrictive assumptions (e.g., Assumptions 4.1 and 4.2) that may not hold in practice. The authors do not sufficiently discuss the rationality and limitations of these assumptions.
2.D/M Ratio Problem: The paper provides insufficient details on how to select D and M values, particularly regarding the determination of the D/M ratio. More specific parameter selection guidelines and analysis of how the D/M ratio affects model performance are needed.
3.Unclear Distinction from Existing Methods: The paper does not adequately explain the essential differences between the DENOTS method and existing GRU-ODE or other neural ODE variants. A clearer comparative analysis is needed.
4.Insufficient Ablation Study Analysis: It does not deeply analyze the interaction between time scaling and anti-phase negative feedback. The authors should further explore why the combination of these two techniques produces better performance.
5.Missing Computational Efficiency Analysis: The paper does not systematically compare computational efficiency with other methods, nor does it sufficiently discuss the computational cost incurred by increased NFE. A more comprehensive performance evaluation is needed.
6.Inadequate Related Work Discussion: The paper only briefly mentions that "prior work mostly sidesteps this topic" regarding time scaling, without citing specific related work. The discussion of negative feedback mechanism related work, particularly comparisons with methods like GRU-ODE-Bayes, is insufficient.

### Questions
1. How do you justify the restrictive assumptions (e.g., Assumptions 4.1 and 4.2) used in your theoretical analysis? Could you discuss their validity in practical scenarios?
2. Could you provide more details on how to select parameters D and M, particularly regarding the determination of the D/M ratio? How does this ratio affect model performance?
3. Could you provide a more detailed explanation of the anti-phase negative feedback mechanism? Specifically, why does passing -h instead of h to a standard PyTorch GRU solve the "forgetfulness" problem?
4. How does your method fundamentally differ from existing GRU-ODE or other neural ODE variants? Could you provide a detailed comparative analysis?
5. Could you provide a more in-depth analysis of the ablation study.Specifically, how do time scaling and anti-phase negative feedback interact to produce superior performance?
6. Could you provide a systematic comparison of computational efficiency with other methods? How do you balance the increased NFE with computational cost?
7. Could you provide a more comprehensive review of literature related to time scaling in neural ODEs? Are there any prior works that have explored similar ideas?
8. How does your negative feedback mechanism compare with existing methods like GRU-ODE-Bayes? Could you provide a detailed comparison?

### Soundness
2

### Presentation
3

### Contribution
2
