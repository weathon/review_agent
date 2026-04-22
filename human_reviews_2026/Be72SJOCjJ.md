# On the Sharp Input-Output Analysis of Nonlinear Systems under Adversarial Attacks

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
This paper is concerned with learning the input-output mapping of general nonlinear dynamical systems. While the existing literature focuses on Gaussian inputs and benign disturbances, we significantly broaden the scope of admissible control inputs and allow correlated, nonzero-mean, adversarial disturbances. With our reformulation as a linear combination of basis functions, we prove that the $\ell_2$-norm estimator overcomes the challenges posed by an adversary with access to the full information history, provided that the attack times are sparse, *i.e.*, the probability that the system is under adversarial attack at a given time is smaller than a certain threshold. We provide an estimation error bound that decays with the input memory length and prove its optimality by constructing a problem instance that suffers from the same bound under adversarial attacks. Our work provides a sharp input-output analysis for a generic nonlinear and partially observed system under significantly generalized assumptions compared to existing works.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper studies identification of nonlinear dynamical systems with partial observability and potentially adversarial disturbances. It models the system via a finite-memory nonlinear operator G* and analyzes an l2-norm estimator under bounded contraction. The authors prove matching upper and lower bounds on estimation error as functions of the contraction factor and memory length, and show empirical scaling trends on synthetic data. While the analysis is mathematically sound, the motivation and presentation are weak. The adversarial framing is poorly defined, as disturbances are modeled as small-probability corruptions, which is closer to sparse-noise robustness than genuine adversarial robustness.

### Strengths
Provides matching upper and lower bounds for a finite-memory setting. Handles non-Gaussian and correlated disturbances beyond the standard i.i.d. noise model. The mathematical exposition of the main theorems is careful and logically consistent.

### Weaknesses
The paper’s motivation and exposition are weak. The “adversarial attack” model corresponds only to rare sample corruptions and is not meaningfully adversarial, and it is not explained what real scenarios are captured by this model. While the authors argue that the small-probability corruption assumption is needed due to existing lower bounds, there are alternative approaches to handling this issue. In particular, rather than focusing on probabilistic sparsity assumptions, one could aim to characterize the fundamental learnability of nonlinear dynamical systems directly. Some examples in that direction (not cited in the paper) are: Safely Learning Dynamical Systems (Ahmadi, Chaudhry, Sindhwani & Tu, 2024) and Efficient PAC Learnability of Dynamical Systems Over Multilayer Networks (Qiu, Adiga, Marathe, Ravi, Rosenkrantz, Stearns & Vullikanti, 2024). A very recent example to an attempt in that direction is Universal Learning of Nonlinear Dynamics (Dogariu, Brahmbhatt & Hazan, 2025), and the present work could benefit from discussing the conceptual relationship. All of that to say, it is very unclear why is it relevant to study small probability disturbances in the context of controlling unknown nonlinear dynamical systems.

More broadly, the paper lacks engagement with the substantial machine-learning view of control literature, which is particularly expected at an ML venue. For example, Learning Nonlinear Dynamical Systems from a Single Trajectory (Foster, Rakhlin & Sarkar, 2020) already provides Lyapunov-based conditioning and optimal sample-complexity guarantees for learning a two-layer ReLU dynamical model. A review of more works on learning linear dynamical systems is missing, for example No-Regret Prediction in Marginally Stable Systems (Ghai, Lee, Singh, Zhang & Zhang, 2020) and A New Approach to Learning Linear Dynamical Systems (Bakshi, Liu, Moitra & Yau, 2023). In addition, Improper Learning for Non-Stochastic Control (Simchowitz, Singh & Hazan, 2020) tackles control of unknown linear dynamical systems under adversarial disturbances. The paper does not cite or connect to these lines of work, which undermines its relevance to the ML and online control communities.

Finally, while the theoretical contribution is sound, it is fairly limited in relevance, and the experiments are preliminary: small synthetic systems only, no real data, no ablations, and no comparison to prior algorithms.

### Questions
What realistic scenario motivates the “adversarial disturbance” model beyond occasional corruptions?

Why is it relevant to extend the input structure (non-gaussian)? Isn't it chosen by the learner? On that note, how is this method more general than considering the system without inputs?

### Soundness
4

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In this paper, researchers study the problem of learning input-output mapping for general nonlinear dynamical systems under disturbances from adversarial attacks. The paper’s work has focused on nonlinear systems with partially observed outputs, non-Gaussian control inputs, and correlated, nonzero-mean, possibly adversarial disturbances. This setting significantly expands the scope of the existing literature and significantly broadens the range of allowed control inputs. This paper reformulates a nonlinear dynamical system as a linear combination of basis functions and proves that l2-norm estimator overcomes the challenge as long as the probability of adversarial attacks on the system at a given moment is less than a certain threshold. This paper presents an estimation error bound that decays with the length of the input memory, and proves the optimality of the estimation error bound by constructing a problem instance.

### Strengths
S1: The paper relax the range of control inputs for the task of identifying nonlinear dynamic systems, and require only partial observed outputs, allowing adversarial disturbances with nonzero-mean. 

S2: The related work of this paper is fully investigated and the research gap in related fields is accurately grasped. Meanwhile, researchers propose the appropriate problem formulation and theoretical analysis. 

S3: The paper has a clear framework and the writing is fluent. The problem positioning and the theoretical derivation steps are clear. 

S4: The setting of input and disturbance in this paper is closer to the real scene, which is more practical than previous studies.

### Weaknesses
W1: The outline of your analysis is clear, but some parts lack specific details. Moving some of the reasoning from the appendix to the main body might make this section more readable.

W2: In addition to proving that the l2 norm estimation is optimal, the experiment should also compare the effect of partial observation output and full observation output on the error.

W3: Some symbols lack interpretation, leading to a reduction in the readability of parts, such as E for Equation (10) and ψ2 for Assumption 2.5.

### Questions
Q1: Your point is that your theory applies to general nonlinear dynamical systems. Does this mean that general nonlinear dynamical systems have Lipschitz continuity?

Q2: Are the basis functions used for nonlinear dynamic systems all nonlinear?

Q3: Do sub-Gaussian variables have any special impact on evaluating nonlinear dynamic systems compared to general variables?

### Soundness
3

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
2

### Summary
The paper considers the problem of learning the mapping from the control input to the partial observation of a nonlinear dynamical system under more general assumptions on the control input and the allowed adversarial attacks. The error bound is also presented for the proposed approximation model.

### Strengths
Overall, the paper considers an interesting problem and proposes an effective numerical method.

### Weaknesses
The presentation of the paper need to be revised based on the comments in the Questions part.

### Questions
1. Eq. (1) The problem is not properly stated. What is the meaning of "partially observed"? What is w in the original problem, which would not be the adversarial disturbances. What are f and g?. Are both of them totally unknown? Is x_0 given? 

2. G^\ast and \Phi a in Eq. (2) are very unclear and misleading. \Phi is just a set of M real-valued functions defined on (U)^{\tau+1}, and each component of the first term on the RHS of (2) is the linear combination of them. Also, I think the dimension of G^\ast should be r by M. In addition, it is unclear how G^\ast is defined in terms of (2) and (2) in the current form seems to be independent of w. 

3. lines 81-86 on page 2. The explanation here is also misleading. To make the approximation error smaller, one would usually need to increase the number of the basis functions. 

4. Lines 218-220. I cannot understand the explanation in the bracket. 

5. It would be helpful to provide some examples of the basis functions after assumption 2.4. How to choose the basis functions in practice? 

6. Again, it is unclear how G^\ast is defined in terms of (7). 

7. Theorem 3.1. The paper is aimed at approximating the input-output map. Then the main result should be stated in terms of the output error instead of the error of G and G^\ast.

### Soundness
2

### Presentation
2

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
This paper proposes to approximate the (finite impute-response functions or) input-output characteristics of a (partially observable) nonlinear system by a linear function in a given basis. The proposed method works in presence of sporadically occurring adversarial disturbances and non-Gaussian control inputs.

### Strengths
Indeed, quite a bit of work in this area in either on linear systems, or for IID perturbations. So this addresses a novel facet of the sysid problem.

The final bound is quite interpretable.

### Weaknesses
I am fine with most assumptions in the paper, to the extent such assumptions (like spectral radius like condition) are also required in the linear case. However, Assumption 2.8 sticks out like a sore thumb. In general, it is a reasonable expectation that works dealing with adversarial disturbances generalize the stochastic case; for example, this is the case for Simchowitz et al who can handle an oblivious adversary (I can give more example here if needed). But, Assumption 2.8 implies that most of the time the entire truncated history of perturbations is exactly zero. Thus, not only does it not generalize the stochastic case, but also, but it also places limits on identifiably of G even in the infinite sample regime. Now, irrespective of how apt this assumption is, it severely limits the applicability of this work. At the very least, this should be noted early, e.g., in the abstract. (I also wonder if a condition like that in Theorem 3.3 is necessary in this regime, or if something qualitatively weaker that captures the fact that the first few most recent w's are more likely to be uncorrupted qualifies.)

Lastly, in statistical learning, when approximating a function by a class that does not contain it, the best approximation also depends on the distribution of inputs (i.e., holds in a distributional L1 sense). I suspect this also holds here (I didn't see a concrete definition of G*). This would also limit using G* for control purposes (or under different inputs downstream).

### Questions
Can the authors please point out any aspect of the work (as summarized above) I might have misunderstood?

### Soundness
3

### Presentation
3

### Contribution
2
