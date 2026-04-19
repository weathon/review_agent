# Revisiting a Design Choice in Gradient Temporal Difference Learning

- Decision: Accept (Poster)
- Scores: 6, 8, 6

## Abstract
Off-policy learning enables a reinforcement learning (RL) agent to reason counterfactually about policies that are not executed and is one of the most important ideas in RL. It, however, can lead to instability when combined with function approximation and bootstrapping, two arguably indispensable ingredients for large-scale reinforcement learning. This is the notorious deadly triad. The seminal work Sutton et al. (2008) pioneers Gradient Temporal Difference learning (GTD) as the first solution to the deadly triad, which has enjoyed massive success thereafter. During the derivation of GTD, some intermediate algorithm, called $A^\top$TD, was invented but soon deemed inferior. In this paper, we revisit this $A^\top$TD and prove that a variant of $A^\top$TD, called $A_t^\top$TD, is also an effective solution to the deadly triad. Furthermore, this $A_t^\top$TD only needs one set of parameters and one learning rate. By contrast, GTD has two sets of parameters and two learning rates, making it hard to tune in practice.  We provide asymptotic analysis for $A^\top_t$TD and finite sample analysis for a variant of $A^\top_t$TD that additionally involves a projection operator. The convergence rate of this variant is on par with the canonical on-policy temporal difference learning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposed a new solution to solve double sampling issue in off-policy reinforcement learning. Specifically, this paper provided another method to estimate $A^T A$ and $A^T b$ by introducing a function $f(t)$. The authors also provided finite-time analysis for their method as well as some numerical results.

### Strengths
The paper expand the idea from $A^TTD$ and proposed a new method to solve double sampling issue in off-policy learning. Compared with $A^TTD$, this methods required less memory. The authors also provided the convergence analysis of their method.

### Weaknesses
This paper is really interesting to me. However, I have several questions. 

1. The advantage of selecting $f(t)$ as an increasing function over a constant one is not immediately clear. The authors state in lines 186–187 that classical convergence analysis can be applied to establish the convergence rate. Thus, it seems that a constant $f(t)$ could also ensure convergence. Additionally, the experimental results suggest that setting $f(t)=2$ is sufficient to resolve Baird’s counterexample, which further supports the idea of choosing $f(t)$ as a constant.

2. Relying on samples from several steps prior may introduce additional errors during policy improvement. Although this paper focuses exclusively on policy evaluation, it is important to mark that policy evaluation serves the purpose of policy improvement. In cases where the policy is continuously updated, the samples used to estimate $A^T$ may become inaccurate, introducing further errors.

3. A minor issue appears in lines 206–207, where I believe the correct notation should be $\bar{h}(\omega_{t_m})$ instead of $h(\omega_{t_m})$

### Questions
Please see my comments in Weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a new variant of the gradient temporal difference (GTD) learning algorithm for online off-policy policy evaluation with linear function approximation. The idea is to use two samples distanced away from each other to address the double sampling issue encountered during the derivation of GTD. The paper shows that when the distance between the two samples $f(t)$ used to estimate the gradient increases with a proper rate (e.g., $f(t)=\ln^2(t+1)$), the new algorithm converges asymptotically, while its variant with a projection operator has a convergent rate comparable to on-policy TD. The consequence of this new GTD variant is that 1) it reduces the need for an additional set of weights and step size, and 2) it requires an additional memory of size $O(\ln^2(t))$. Preliminary experiment results on Baird’s counterexample show the effectiveness of the proposed algorithm.

### Strengths
The strengths of the paper include its originality, quality, and clarity:
1. The idea in this paper is novel to the best of my knowledge. It’s neat to use two samples distanced away from each other to estimate the terms involving two $A$ matrices, which are independent if their gap is large. The sublinear memory requirement also renders this idea a practical approach.
2. The quality of the paper is also a strength. The asymptotic convergence of the proposed algorithm is novel and may bring value to the community.
3. The paper is very well written and easy to follow.

### Weaknesses
The paper has weaknesses in its significance and relevant work discussion:
1. The paper may be limited in its significance. 
  - On the theory side, the finite time analysis is based on a variant of the proposed algorithm with a projection step, which is absent in the actual algorithm. Thus, the comparison between its convergence rate in this case with that of on-policy TD may not be very valuable. Note that finite sample analysis of the actual algorithm is possible, as also pointed out in the paper. Obtaining such a result can strengthen the paper.
  - On the empirical side, the experiments presented in this paper only focus on Baird’s counterexample and are rather limited. Having more experiments, even in simple environments like FourRoom (Ghiassian & Sutton, 2021; Ghiassian et al., 2024), would help strengthen the claim that the proposed algorithm is effective. In addition to testing the proposed algorithm in environments like FourRoom, a comparison with other GTD algorithms can also make the paper stronger. Other researchers may find it useful to know how the proposed algorithm compares to others in terms of sample efficiency, stability, and hyperparameter sensitivity.
2. The paper also lacks a thorough related work discussion on off-policy policy evaluation (OPPE) with linear function approximation. There have been many follow-up works on the GTD algorithm (Mahadevan et al., 2014; Ghiassian et al., 2020; Yao, 2023). While some of them are cited in the paper, the relationship between these later ideas building on GTD and the proposed method is not thoroughly discussed, which could be useful and inspire future research. Note that Yao (2023) also introduces a GTD variant with one step-size, so it may be necessary to clarify its distinction with your approach. In addition, the paper may also benefit from discussing another line of work that addresses the deadly triad, the ETD family (Sutton et al., 2016; Hallak et al., 2016, 2017; He et al., 2023). Specifically, how does the proposed approach compare to these methods in terms of the optimality of the fixed point and the convergence property? Having a more thorough discussion of these relevant works would strengthen the paper’s positioning.

Ghiassian, S., Patterson, A., Garg, S., Gupta, D., White, A., & White, M. (2020). Gradient temporal-difference learning with regularized corrections. ICML.

Ghiassian, S., & Sutton, R. S. (2021). An empirical comparison of off-policy prediction learning algorithms in the four rooms environment. arXiv preprint arXiv:2109.05110.

Ghiassian, S., Rafiee, B., & Sutton, R. S. (2024). Off-Policy Prediction Learning: An Empirical Study of Online Algorithms. IEEE Transactions on Neural Networks and Learning Systems.

Hallak, A., Tamar, A., Munos, R., & Mannor, S. (2016). Generalized emphatic temporal difference learning: Bias-variance analysis. AAAI.

Hallak, A., & Mannor, S. (2017). Consistent on-line off-policy evaluation. ICML.

He, J., Che, F., Wan, Y., & Mahmood, A. R. (2023). Loosely consistent emphatic temporal-difference learning. UAI.

Mahadevan, S., Liu, B., Thomas, P., Dabney, W., Giguere, S., Jacek, N., ... & Liu, J. (2014). Proximal reinforcement learning: A new theory of sequential decision making in primal-dual spaces. arXiv preprint arXiv:1405.6757.

Sutton, R. S., Mahmood, A. R., & White, M. (2016). An emphatic approach to the problem of off-policy temporal-difference learning. JMLR.

Yao, H. (2023). A new Gradient TD Algorithm with only One Step-size: Convergence Rate Analysis using $ L $-$\lambda $ Smoothness. arXiv preprint arXiv:2307.15892.

### Questions
Here are a few questions that might affect the evaluation:
1. Do the choices of $f(t)$ depend on the induced Markov chain’s mixing time? If yes, where is it in the result? If not, why?
2. What is the convergence rate for GTD? Is it the same as the canonical on-policy TD as well?
3. In Line 182, when $f(t)$ was a constant function, what would happen by following the classical convergence results mentioned in Line 186? What’s the consequence of using such a $f(t)$?
4. In Line 227, the paper claims the technique of using a variable interval $T_m$ has not been used in any stochastic approximation and RL literature. Has it been used or studied in other fields? If yes, then it is worth mentioning the relevant literature.
5. In Line 482, the paper claims that the finite sample analysis **confirms** that the proposed algorithm converges reasonably fast, but this analysis is based on a variant of the proposed algorithm with a projection step. Does the efficiency of the variant with the projection step guarantee or generally suggest the efficiency of the base algorithm?
6. Is the proposed algorithm more or less sensitive to the learning rate compared to GTD? Since GTD has two hyperparameters, comparison methods like those in Ghiassian & Sutton, 2021 might be useful here.

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
3

### Summary
The paper aims to develop a convergent algorithm for off-policy temporal difference learning under linear function approximation. The proposed algorithm directly minimizes the L2-norm of expected TD updates (NEU) $||Aw+b||^2$, improving the memory requirement of estimating matrix $A$ from the previous ATD algorithm. Meanwhile, the proposed algorithm reduces the number of learning rates from two to one compared to GTD, another NEU minimization algorithm. It maintains the convergent property with a convergent rate $\tilde{O}(1/t)$. Moreover, the algorithm is tested on Baird’s counterexample and is shown to avoid divergence in the deadly triad.

### Strengths
The problem to tackle is well stated, which is to stabilize off-policy learning and improve the previous algorithm ATD and GTD: the proposed algorithm saves memory compared to the ATD algorithm and increases the convergence rate compared to GTD. Also, the paper is clearly written and easy to follow, with rigorously stated assumptions and lemmas.

### Weaknesses
The approach needs to be more motivated. GTD is known to suffer from a low convergent rate compared to TD. More experiments to compare the convergence speed and some intuition on why the proposed algorithm can fasten the learning would be great.

Also, the algorithm needs to fit better into the literature. ETD, introduced by Mahmood and colleagues (2015), is another stable off-policy algorithm. Also, a target network is suggested to help convergence (Zhang et al., 2021; Fellows et al., 2023; Che et al., 2024). Che et al. (2024) compare their TD algorithm with GTD on Baird’s counterexample, showing much faster convergence.

Reference

Mahmood, A. R., Yu, H., White, M., & Sutton, R. S. (2015). Emphatic temporal-difference learning. arXiv preprint arXiv:1507.01569.

Zhang, S., Yao, H., & Whiteson, S. (2021, July). Breaking the deadly triad with a target network. In International Conference on Machine Learning (pp. 12621-12631). PMLR.

Fellows, M., Smith, M. J., & Whiteson, S. (2023, July). Why target networks stabilise temporal difference methods. In International Conference on Machine Learning (pp. 9886-9909). PMLR.

Che, F., Xiao, C., Mei, J., Dai, B., Gummadi, R., Ramirez, O. A., ... & Schuurmans, D. (2024). Target Networks and Over-parameterization Stabilize Off-policy Bootstrapping with Function Approximation. arXiv preprint arXiv:2405.21043.

### Questions
Is it necessary to take an increasing function f(t)? Besides removing the dependence between data at step t and t+f(t) to establish the convergence proof, is there any other reason to use an increasing function?

### Soundness
3

### Presentation
3

### Contribution
3
