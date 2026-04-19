# Causal-StoNet: Causal Inference for High-Dimensional Complex Data

- Decision: Accept (poster)
- Scores: 8, 6, 6

## Abstract
With the advancement of data science, the collection of increasingly complex datasets has become commonplace. In such datasets, the data dimension can be extremely high, and the underlying data generation process can be unknown and highly nonlinear. As a result, the task of making causal inference with high-dimensional complex data has become a fundamental problem in many disciplines, such as medicine, econometrics, and social science. However, the existing methods for causal inference are frequently developed under the assumption that the data dimension is low or that the underlying data generation process is linear or approximately linear. To address these challenges, this paper proposes a novel stochastic deep learning approach for conducting causal inference with high-dimensional complex data. The proposed approach is based on some deep learning techniques, including sparse deep learning theory and stochastic neural networks, that have been developed in recent literature. By using these techniques, the proposed approach can address both the high dimensionality and unknown data generation process in a coherent way. Furthermore, the proposed approach can also be used when missing values are present in the datasets.  Extensive numerical studies indicate that the proposed approach outperforms existing ones.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors propose an algorithm, CausalStoNet, which aims to give accurate causal inferences for systems of up to 100 variables (after variable reduction) under fairly relaxed nonlinearity conditions. The method is based on deep learning ideas.

### Strengths
This was a good paper, I thought. The applications were fruitful, and the explanation of the theory was clear. It was well-situated in the literature, and useful and correct extensions of the method were proposed.

### Weaknesses
Several methods are compared, but I don't see a discussion of looking for the best available methods. Are these methods in Table 2 the best available methods? See also my questions below.

### Questions
1.	I guess, in my mind, it does very little good to say a method can deal with high dimensions without saying how high. There are two examples, one for 43 variables and another for 100 (genome reduced to this). Some may not consider this to be high-dimensional, so it would be better to say up front what dimension one hopes to achieve with the method (after variable reduction is done).
2.	Also, it’s important to say what density such models can attain. It’s possible with some very high-dimensional data that the models may, in fact, be very dense, a situation that can be addressed currently in the linear Gaussian or non-Gaussian case. However, this hasn’t been addressed to my knowledge for models with more general connection functions, a possible advantage of the method in this paper. 
3.	There are recent papers that address a dense searches for the linear, Gaussian case and linear, non-Gaussian cases. The secret of some of these papers is to relax the Faithfulness condition, something which does allow some nonlinear functions to be addressed in a linear framework. The secret of others (like DirectLiNGAM) is to move to the linear, non-Gaussian regime. It would be really wonderful if an approach like the one in this paper could be shown to improve these algorithms (which look to be state-of-the-art) for some choices of nonlinear functions.
4.	Along these lines, I think it’s important when saying you’re outperforming methods to include in this the relaxation of assumptions that you’ve done and to compare only to other methods that relax assumptions in similar ways or that use stronger assumptions, explicitly noting this and show that with the stronger assumptions worse results are achieved. (The latter is not always the case.)
5.	The restriction to binary data for some of the theory is somewhat severe since very few real datasets consist entirely of binary variables, though perhaps I misunderstand.
6.	I’m not sure how “MNR” abbreviates “Missing At Random”; I think it must be “MAR.”


NOTE: I read the response; I thought it was convincing, so I'm raising my rating. Thanks.

### Soundness
4 excellent

### Presentation
3 good

### Contribution
4 excellent

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a deep-learning-based causal inference method, that assumes the nuisance parameters are nonlinear high-dimensional models, fit by a stochastic neural network with sparsity-pursuing properties. This paper also gives theoretical guarantees, building upon a series of papers by Liang and colleagues on Bayesian neural nets.

### Strengths
The paper tries to address an important problem in causal inference, by taking the nuisance parameters to be high-dimensional sparse nonlinear models. State-of-the-art deep learning methods are used, e.g. stochastic/Bayesian sparsity-pursuing neural nets, to address this challenge.

The theoretical results look sound, building upon a series of earlier works on statistical guarantees of Bayesian neural nets by Liang and colleagues.

### Weaknesses
1. Even though I selected ``good'' in Presentation, I believe the exposition can nonetheless be significantly improved. For example, some of the key assumptions and elements are delayed to Appendix so the flow seems a bit broken, in particular in the first several sections. I strongly recommend that the authors consider revising their manuscript to make the flow smoother.

2. In the simulation, this paper considered scenarios with p=100/200 and n=10000, which do not seem to be a very high-dimensional regime. What would be the performance if we further increase the covariate dimension p or decrease the sample size n?

3. Missing references: doubly-robustness should be traced back to Robins et al. 1994 JASA. Also, Farrell et al. was cited twice in the paper (one arxiv version, one joe published version). A recent paper led by Xiaohong Chen and colleagues (Chen, Liu, Ma, Zhang, to appear in JoE) also addresses a similar problem, though their nuisance models are slightly different (Barron space). This paper should also be cited.

4. Theory-practice gap: As we know, almost all theoretical works on deep learning do not reflect practice. For instance, sparse neural nets are generally difficult to fit, as persuasively argued in Farrell, Liang, and Misra ECTA 2021. The authors are recommended to comment on this, for readers to better understand what are the key elements that allow Causal-Stonet to learn the sparse neural nets.

5. In causal inference, often times $x$ has clear scientific meaning. For sparse models, I would imagine that the input layer sparsity seems to be more important. If putting sparsity in output layer, this is essentially saying that there is some sparse nonlinear representation of the input. By assuming sparse neural nets, however, this is saying that the nonlinear representation itself is also in some sense sparse. Is it really aligned with our view of the real world or is it more like a contrived modeling assumption? I hope that the authors could further comment on the modeling philosophy adopted in this paper.

### Questions
1. Do the theoretical results rely on equation (6)? If so, then the theoretical results seem to go against the conventional wisdom in the deep learning literature, that is, the neural parameters themselves are not scientifically meaningful so it is not that important to learn the neural net parameters.

2. In theory, the propensity model and outcome model are both sparse models with sparsity levels lower than n^{3/16}. In linear models, the sparsity allowed is n^{1 / 2} up to log factors. Could the authors provide a heuristic explanation on the rate n^{3 / 16}?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper introduces an approach to causal inference tailored for high-dimensional complex data. This approach draws upon recent developments in deep learning techniques, including sparse deep learning and stochastic neural networks. By leveraging these techniques, the proposed approach effectively tackles both the high dimensionality and the complexity of the underlying data generation process. Moreover, it can handle scenarios with missing values in the datasets.

### Strengths
- The proposed method Causal-StoNet is proven to have the universal approximation ability, making it a versatile tool for modeling outcome and propensity score functions.

- The paper provides a strong theoretical foundation for its approach, including proofs and mathematical support, enhancing its reliability.

### Weaknesses
1/ It seems to me that the proposed method is a direct application of stochastic neural networks. Please clearly highlight technical innovation of the proposed method.

2/ Can the authors explain which component of their method make it works well for high-dimensional confounder X? Why the existing methods such as: CEVAE, CFR Net are not possible to deal with high dimensions of X?

3/ It seems to me that many concepts and notations in the paper are unexplained. For example, why do we choose $\sigma_{0,n}^n$ to be a very small number while $\sigma_{1,n}^2$ is relatively large? Is there any rationale for this? What is $\pi(\theta)$ in Eq.~9? Is it the prior?

4/ Since $Y_{mis}^i$ is unobserved, how do you minimise Eq.~9?

### Questions
Please see section weaknesses

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair
