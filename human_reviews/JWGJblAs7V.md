# End-to-End Learning under Endogenous Uncertainty

- Decision: Reject
- Scores: 3, 5, 5, 3

## Abstract
How can we effectively learn to make decisions when there are no ground-truth counterfactual observations? We propose an end-to-end learning approach to the contextual stochastic optimization problem under decision-dependent uncertainty. We propose both exact methods and efficient sampling-based methods to implement our approach. We also introduce a new class of two-stage stochastic optimization problems to the end-to-end learning framework. Here, the first stage is an information-gathering problem to decide which random variable to ``poll'' and gain information about before making a second-stage decision based off of it. We provide theoretical analysis showing  (1) that optimally minimizing our proposed objective produces optimal decisions and (2) generalization bounds between in-sample and out-of-sample cost.
We  computationally test the proposed approach on multi-item assortment problems where demand is affected by cross-item complementary and supplementary effects. Overall, our method outperforms other benchmarks by more than 15\% and performs best in high noise, across any cost configuration, and when given sufficient data.
We also introduce an experiment for the information-gathering problem on a real-world electricity generation problem. We show our method proposes decisions with more than 7\% lower cost than other decision-making methods.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The paper proposes a new approach to solve contextual stochastic optimization problems under decision-dependent uncertainty. Their approach focuses on estimating the reward function via a plug-in approach, i.e., replacing the unknown stochastic random variable with a prediction conditional on the context and decision. Since learning the reward function is a non-convex problem, they provide a MIP formulation to solve it exactly for certain classes of plug-ins. The paper also discusses a new class of two-stage stochastic optimization problems related to information-gathering. Finally, they provide some numerics comparing their method to other approaches.

### Strengths
The paper's main strength is that they propose an interesting approach to estimating the reward function for offline policy evaluation that seems loosely justified theoretically by Proposition 3.1. They then propose a MIP formulation that provides a tractable way to solve their non-convex learning problem to optimality. It's overall interesting because it directly leverages structure of the stochastic optimization problem compared to more generic offline policy evaluation approaches. This reduces the complexity of learning the reward function which maybe beneficial in small data settings.

### Weaknesses
I think the paper can be improved in many different ways. Below outlines some issues that should be addressed:

1. First, **I would not classify their approach as "end-to-end"**, especially in the context of the works [1][2][3]. These works look to directly optimize the decision-loss whereas this paper is primarily focused on learning the reward function. To clarify, an end-to-end approach would look to solve $\hat{f} \in \arg\max_{f \in \mathcal{F}} \mathbb{E}_{\mathbf{z}\sim Z(\hat{\mathbf{w}}_f (\mathbf{x}),\mathbf{x})}\left[ g(\hat{\mathbf{w}}_f(\mathbf{x}), \mathbf{z}) \right]$ where $\hat{\mathbf{w}}_f (\mathbf{x})$ is equation (1) in the paper but parameterized by 
$f$. In contrast, **this paper lies more in the offline policy learning for contextual bandits literature** as it proposes the following estimate for the reward function $\hat{r}(w,x) = g(w, \hat{f}\_{\text{end-to-end}} (w,x))$. The policy $\hat{\mathbf{w}} (\mathbf{x})$ is just the policy that optimizes the reward function estimator. 

2. The paper claims "We provide theoretical analysis showing that (1) optimally minimizing our proposed objective produces optimal decisions". However, this does not seem to appear anywhere in the paper. The only theory is in section 3, which provides some limited intuition why using plug-in estimator is reasonable (proposition 3.1) and some generic generalization bounds (theorem 3.2). In general Theorem 3.2 seems somewhat problematic. **First**, the authors do not translate theorem into any sort of regret bound to show you obtain optimal decisions. **Second**, the theorem seems potentially incorrect/imprecise since you do not take an expectation of $\mathbf{x}$ so $l(f)$ is still random and not even well defined. My guess is you just want an expectation over all the random variables. **Third**, the assumption that $g(\mathbf{w},\mathbf{z})$ (typo in the theorem) is bounded seems hard to justify if $g$ is Lipschitz and $\mathbf{z}$ is unbounded given the examples discussed in the paper. 

3. The information gathering application feels somewhat out of place in this paper. I understand the authors are trying to motivate endogeneity for contextual stochastic optimization, but the application is confusing. Specifically, I do not understand why if you already have data about $Z_w(\mathbf{x})$ why you would need to make a specific decision $w$ to utilize the data in the optimization problem. For the opposite case, if you don't have any data related to $Z_w(\mathbf{x})$ for a specific decision $w$, then how are you supposed to model it in your two-stage contextual stochastic optimization problem. Everything is offline, so I don't understand the information gathering aspect of this example. 

4. **The numerics seem extremely limited**. Since the paper doesn't provide any theoretical guarantees in regards to learning "optimal decisions", it would be helpful to have more extensive experimental results and more details about the implementation.  For the assortment optimization experiment, the following is unclear: 
- How many trials did you do? It seems like you only did a single sample run, i.e., you only saw the performance of the different methods for a single training set. Training sets are also noisy so it would be nice to see the performance over many training sets and have confidence bands on the average performance. 
- It seems like your setting is almost well specified, as in the data generation procedure is contained in the class of linear models you are learning $f$ from. What happens if $z_k$ is not generated from a linear model, but something more non-convex or even a non-continuous function?
- More details and methods would help highlight the benefits of the paper's proposed approach. Why do you not consider an approach where you learn the map $x_n, w_n \rightarrow g(w_n, z_n)$ via a neural network? Is this what reward learning is? Additionally, how do you tune your KNN method? 
- What are the run times of your method? Since it's a MIP, it would be helpful to understand the trade-off of the increased computation time vs. improvement in decision quality. Additionally, the paper seems to suggest ways to speed up the computation, so it would be nice to see how these approaches improve the computation time. 


___
[1] Adam N Elmachtoub and Paul Grigas. Smart “predict, then optimize”. Management Science, 68(1): 9–26, 2022.
[2] Akshay Agrawal, Brandon Amos, Shane Barratt, Stephen Boyd, Steven Diamond, and J Zico Kolter. Differentiable convex optimization layers. Advances in neural information processing systems, 32, 2019.
[3] Brandon Amos and J Zico Kolter. Optnet: Differentiable optimization as a layer in neural networks. In International conference on machine learning, pp. 136–145. PMLR, 2017.

### Questions
Below are questions that perhaps don't fall directly in the categories of the weakness section: 

1. Is $\mathbf{x}$ fixed or random? It doesn't seem well defined in the paper which can interfere with the theory in the paper. 
2. Equation (7) is confusing. Is $\bar{\mathbf{z}}$ random on the left hand side like it is on the right hand side? If so, then I do not see how the $\approx$ makes sense since if $\bar{\mathbf{z}}]$ varies a lot then you shouldn't ever be able to estimate the constant on the right hand side.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper provides a formulation in "predict then optimize" situations where the optimization decisions affect the random realizations. The paper solves a version of the above problem for some structured cases via MIP optimization.

### Strengths
The prediction and then optimization problem is relevant to operations research and other areas. The mixed integer programming formulation of the above problem is novel. The paper also has reasonable empirical work to support their theory.
The introduction and motivation are well written and play a good role in building anticipation for the further sections.

### Weaknesses
1)The introduction is well written, but it motivates a very general setting of the problem, while the paper seems to address only a very structured version of the general problem (with examples in lines at the beginning of section 2). 

2) The setting as motivated is similar to "Performative Prediction" https://proceedings.mlr.press/v119/perdomo20a.html. The authors should consider contrasting their work with this and other works in performative machine learning.

3) Realizability assumptions are not explicitly made (e.g. f* should be part of the function class for (3) as a necessary condition). Other assumptions may be required based on the exact nature of the "sampling method"

4)Why is the non-convexity of z is an issue (Line 183) in solving (3). It is not clear the analogy made, r_n is not defined.

5) The integer program appears all of a sudden; it is not well motivated, and all the variables are not fully defined in the main text. Why is this the chosen model for the given applications ? 

6)The sampling method is fully described, e.g., does perturbing the mean squared predictor help? Section 2.1 seems to be overall hurriedly written or hurriedly shortened just to fit the page limit

7) I do not fully follow the message that section 3 is trying to convey. While for the applicability of proposition 3.1, one needs a function of having "infinite expressivity" (suitability defined), how does the Rademacher complexity of such a function class behave? So, in some sense, the bounds of Theorem 3.2 only make sense when $N = \infty$ (if we can suitably show the R_N(F) falls down suitably fast)

8)In the information gathering section, the setting is not well described, as it seems that the realization of z is fixed (but unknown and something that can be queried). Can you clarify

### Questions
Please refer above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The authors study an end-to-end approach that tackles endogenous problems, specifically contextual stochastic optimization problems under decision-dependent uncertainty. Their approach has a first information gathering stage and a second decision stage. They provide generalization bounds between in-sample and out-of-sample cost, and they empirically demonstrate the advantages of their approach.

### Strengths
For their proposed method, the authors provide both theoretical analysis and empirical results on synthetic and real-world data.

The paper is well-organized and easy to follow.

### Weaknesses
The authors only compared against one end-to-end baseline, while there has been many advances in this area [1]. [1] is a python library that  includes implementation of several end-to-end learning algorithms and testing environments. [2] benchmarks a few end-to-end learning methods. [3] and [4] are some examples of existing end-to-end methods that I am sure the authors are familiar with.  

Although not all existing end-to-end methods may apply to the experimental settings the authors considered, I encourage the authors to compare against at least one or two more end-to-end algorithms and especially the recent state-of-the-art, to showcase the benefits of proposed method in handling endogeneity. 

Apart from baselines, the experimental settings are limited. For example, table 1 only shows results in one single setting. This is synthetic data experiment, and thus it should be straight forward to obtain results on a wide range of settings (e.g. different number of products, costs, and noise levels). Having results on at least several different settings would greatly improve the work. 

The evaluation metrics currently do not include standard errors or confidence intervals. Thus, it is not clear whether there is statistically significant difference between the proposed method and the baselines. Providing either paired t-test results or standard errors would help showcase the advantages of the proposed method. 

[1] Tang, B. and Khalil, E.B., 2024. Pyepo: A pytorch-based end-to-end predict-then-optimize library for linear and integer programming. Mathematical Programming Computation, 16(3), pp.297-335.
[2] Geng, H., Ruan, H., Wang, R., Li, Y., Wang, Y., Chen, L. and Yan, J., 2023. Rethinking and Benchmarking Predict-then-Optimize Paradigm for Combinatorial Optimization Problems. arXiv preprint arXiv:2311.07633.
[3] Amos, B. and Kolter, J.Z., 2017, July. Optnet: Differentiable optimization as a layer in neural networks. In International conference on machine learning (pp. 136-145). PMLR.
[4] Kallus, N. and Mao, X., 2023. Stochastic optimization forests. Management Science, 69(4), pp.1975-1994.

### Questions
What are the technical difficulties and novelties in the proof of theoretical results? Specifically, what are the challenges in applying standard proof techniques using Rademacher complexity to the problem setting considered in this paper? It would be nice to highlight in section 3.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The manuscript proposes a joint prediction-and-optimization framework for the contextual stochastic optimization problem under endogenous uncertainty, and for a class of two-stage information-gathering problem. The framework has been tested using two specific examples and the results show the newly proposed framework outperforms baselines.

### Strengths
The manuscript provides a novel method to estimate the reward function when decision-dependent uncertainty is present. It also provides theoretical proof to the achievability of the formulations, even though loosely. Empirical experiments are also conducted to further prove the performance of the algorithm.

### Weaknesses
1. More experiments are needed to demonstrate the performance of the method.
2. The experimental results are not persuasive enough by simply comparing with the shown baseline models. It would be necessary to include other advanced methods.

### Questions
See above.

### Soundness
3

### Presentation
2

### Contribution
3
