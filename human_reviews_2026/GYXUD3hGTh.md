# Improved high-dimensional estimation with Langevin dynamics and stochastic weight averaging

- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
Significant recent work has studied the ability of gradient descent to recover a hidden planted direction $\theta^\star \in S^{d-1}$ in different high-dimensional settings, including tensor PCA and single-index models. The key quantity that governs the ability of gradient descent to traverse these landscapes is the information exponent $k^\star$ (Ben Arous et al., (2021)), which corresponds to the order of the saddle at initialization in the population landscape. Ben Arous et al., (2021) showed that $n \gtrsim d^{\max(1, k^\star-1)}$ samples were necessary and sufficient for online SGD to recover $\theta^\star$, and Ben Arous et al., (2020) proved a similar lower bound for Langevin dynamics. More recently, Damian et al., (2023) showed it was possible to circumvent these lower bounds by running gradient descent on a smoothed landscape, and that this algorithm succeeds with $n \gtrsim d^{\max(1, k^\star/2)}$ samples, which is optimal in the worst case. This raises the question of whether it is possible to achieve the same rate without explicit smoothing. In this paper, we show that Langevin dynamics can succeed with $n \gtrsim d^{ k^\star/2 }$ samples if one considers the average iterate, rather than the last iterate. The key idea is that the combination of noise-injection and iterate averaging is able to emulate the effect of landscape smoothing. We apply this result to both the tensor PCA and single-index model settings. Finally, we conjecture that minibatch SGD can also achieve the same rate without adding any additional noise.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates the ability of Langevin dynamics on the sphere to recover a latent signal $\theta^\star$ in the spiked PCA and single-index models. The key idea is to consider the time average of the iterates instead of the final one. The author shows that this averaging procedure allows recovery of $\theta^\star$ with a sample complexity matching that of smoothed online SGD, suggesting that iterate averaging has an implicit smoothing effect. The paper also provides a heuristic connection between mini-batch SGD and Langevin dynamics.

### Strengths
The idea that averaging iterates can replace explicit loss smoothing to recover a planted signal with optimal sample complexity in the online regime is novel and conceptually appealing. The paper provides a clear theoretical treatment of this idea in the context of Langevin dynamics on the sphere, supported by sound derivations and proofs.

### Weaknesses
(1) The analysis is limited to the continuous-time setting, and it remains unclear how a discretized version of the proposed dynamics would perform in practice or whether it would preserve the same sample complexity guarantees.

(2) While the idea of using iterate averaging could potentially lead to simplified algorithms, the specific procedure analyzed in this work is not practical. It requires oracle knowledge of the information exponent $k^\star$, and depending on its parity, a different estimator must be used. 

(3) The use of iterate averaging is quite common in convex optimization, yet this connection to existing literature is not discussed.
 
(4) The heuristic connection between mini-batch SGD and Langevin dynamics presented in Section~5.2 is somewhat confusing, as it requires $\varepsilon = 1/\eta$ while both quantities are assumed to be small. 

(5) The algorithm implemented in the experiments differs from the one analyzed theoretically, and no empirical comparison is provided with smoothing SGD or with the partial-trace method.

### Questions
(1) The role of the empirical risk minimization (ERM) formulation (as opposed to an online or streaming setting) is mentioned as a distinction from prior work. However, the algorithm used in experiments appears very close to the standard spherical SGD analyzed in previous works. Is this distinction mostly of conceptual interest? Furthermore, other works such as [1] already provided a similar ``ERM'' formulation, so the novelty of this contribution is not entirely clear to me.

(2) The theoretical analysis is conducted in continuous time, but in practice, the Langevin dynamics must be discretized. The choice of step size can crucially affect both stability and sample complexity, as discussed for instance by Arous et al.(2021). Could the authors comment on how the results would translate to the discrete-time setting? In particular, what scaling of the discretization step would be required for the discrete algorithm to inherit the same statistical guarantees as the continuous process?


[1] Learning Multi-Index Models with Neural Networks via Mean-Field Langevin Dynamics, Mousavi-Hosseini et al. (2025)

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper shows how combining Langevin dynamics with stochastic weight averaging can enhance signal recovery in high-dimensional estimation tasks by navigating complex loss landscapes more effectively.

### Strengths
The paper conveys an interesting check about the possibility to dynamically smooth the landscape in a number of hard inference problems.
It is robust in terms of derivations and conclusions.

### Weaknesses
Not sure that it is relevant to Machine Learning applications where landscape is expected to be way smoother already thanks to overparametrization and large datasets. 
The paper is not very innovative both concerning ideas and execution. I do not need to specify further the very rich literature on the same subject already cited in the paper and in other parts of this review. Also little parts of the proof are contained in other papers (as explicitly mentioned in the text). 
I suggest the authors to stress the added value of their contribution more explicitly, also with reference to possible ML applications especially as they intendo to discuss SGD implications in terms of stochastic averaging.

### Questions
Concerning the general idea of dynamical smoothing can you comment on the relation between your proposal and https://www.pnas.org/doi/10.1073/pnas.1608103113 and other papers on the same lines such as https://arxiv.org/abs/1611.01838?
The finite learning rate could represent an effective correlation between the regions over which information is dynamically averaged.
Note that an optimal protocol for the coupling is considered in the contexts of dynamical explorations inspired by robust ensemble.

Possibly related to this discussion: in figure 1 and figure 2 is there a non monotonicity appearing in the behaviour when learning rate increases? Can you discuss why that is? Is there a best learning rate for the smoothing process? 
What are the drawbacks deteriorating the quality of the average information for large and small learning rates?

Also a significant chunk of literature about tensor PCA and its sample complexity for the success of Langevin dynamics was overlooked: https://arxiv.org/abs/1812.09066, https://arxiv.org/abs/1902.00139, https://arxiv.org/abs/1907.08226, please consider adding it.

### Soundness
4

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
3

### Summary
The paper proposes to solve the single index models by a Langevin algorithm and demonstrates that this achieves the state of the art theoretical scaling with respect to the information exponent of the link function, on par with smoothing the loss landscape.

### Strengths
The paper addresses the problem of learning the implanted direction with a Langevin algorithm, even though this was previously considered inefficient and shows that it is not. The paper nicely explains that it does not solve the problem of escaping from the "equator" for individual samples, but only once the averaging over the samples is performed. It also demonstrates this with numerical simulations.

### Weaknesses
It is not clear what are the implications of this work and what should a practitioner gather from these findings.
As far as I understand, Langevin dynamics only boosts the performance when the information exponent k is larger than 2 (otherwise max(1, k-1) = 1 and so is max(1, k/2)). It is not clear to me how relevant is the k > 2 case, especially if the single index model is viewed as a toy model of the neural network. In Example 1, k = 1 and k = 2 cases are often encountered, while cases with k > 2 seem made up.

### Questions
1. Compared to the landscape smoothing, what are the pros and cons of using the Langevin algorithm? Some clarification of how the landscape is smoothened would be useful. Can smoothing be applied in practice, if so, could it be shown in Figure 1?

2. What is the practical relevance of the findings in the paper, which practical problems are modeled by the problems considered here (single index models with k > 2 and tensor PCA)? In other words, in which problems should the practitioner consider applying the Langevin algorithm?

3. What is the intuition as to why the Hermite polynomials play an important role here and what is an intuitive meaning of the information exponent?

### Soundness
3

### Presentation
2

### Contribution
2
