# A Minimum Variance Path Principle for Accurate and Stable Score-Based Density Ratio Estimation

- Decision: Accept (Poster)
- Scores: 6, 2, 8

## Abstract
Score-based methods are powerful  across machine learning, but they face a paradox: theoretically path-independent, yet practically path-dependent.
	We resolve this by proving that practical training objectives differ from the ideal, ground-truth objective by a crucial, overlooked term: the path variance of the score function.
	We propose the MVP (**M**imum **V**ariance **P**ath) Principle to minimize this path variance. 
	Our key contribution is deriving a closed-form expression for the variance, making optimization tractable. 
	By parameterizing the path with a flexible Kumaraswamy Mixture Model, our method learns data-adaptive, low-variance paths without heuristic manual selection. 
	This principled optimization of the complete objective yields more accurate and stable estimators, establishing new state-of-the-art results on challenging benchmarks and providing a general framework for optimizing score-based interpolation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Score-based approaches for Density Ratio Estimation (DRE) have gained interests in recent literature. While previous works typically employ a fixed probability path beforehand, the authors observe that the choice of the path can have implications on the downstream performances. In order to resolve the issue, the authors propose to explicitly learn the path by minimizing the resulting variance. Specifically, a Kumaraswamy Mixture Model (KMM) is employed to model the schedule. Empirical experiments are provided, which demonstrate that the proposed method helps to improve the performances of the DRE method.

### Strengths
1. The idea of learning the interpolation path in general is well-motivated.
2. The proposed solution relying on closed-form path variance is interesting.

### Weaknesses
1. As the authors themselves noted in the Appendix, the bound in Lemma 4.1 is with a specific choice of n and m, without strong justification of these choices.
2. Relatively minor, the experiments are of smaller scale problems.

### Questions
1. In the main paper the STSM objective is highlighted, e.g. the equality in Equation 9 relies on STSM objective. However, in Appendix B.2.1. it was noted "We Use CTSM". Which objective is employed in the experiments?
2. The empirical performances of the methods naturally depend on many factors, e.g. the weighting function \lambda(t) as employed in the objective. Have the authors made attempts to isolate the other factors to investigate whether the empirical improvements are indeed due to the choice of path?
3. In terms of broader literature, the variance of the time score appears in information geometry context as the time score is the Fisher score with respect to t. Furthermore, learning the paths have been investigated in diffusion literature, e.g. [1].

[1] Neural Diffusion Models, Bartoh et al. ICML 2024

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The goal is to estimate the ratio of two densities $p_0$ and $p_1$ using their samples. The authors do so using the framework of time score matching, where a design choice if the path of distributions $(p_t)_{t \in [0, 1]}$ that is chosen by the user to link $p_0$ to $p_1$. 

The authors upper-bound the estimation error of the density-ratio in terms of the path choice, and minimize the upper-bound. They obtain closed-form results (optimal time-reparameterization aka "schedule" of a given path) and empirical results (numerically optimizing over a family of paths).

### Strengths
The authors identify an interesting question that has not been solved in the literature of density ratio estimation using the time score identity. Their analysis is rather novel, and is reminiscent of similar results in the flow matching literature (where the estimation error is upper bounded by the training loss).

### Weaknesses
## **Minor**

- **Terminology.** 

Line 159: the authors say the time score loss in (Choi et al., 2022) is known as the *sliced* time score matching objective. I have read (Choi et al., 2022) and have not come across the terminology of "sliced time score matching" in the main text. As far I know, (Choi et al., 2022)  just call their cost function "time score matching". Is "sliced" a terminology that you are adding? If so, it would be fair to explicitly say so. 

---

## **Math Questions**

- **Reweighting.**  
 In the time-score matching loss (Eq 6), do you use a reweighting function as (Choi et al., 2022) do?

- **On Section 4.1.**  
  Is $\theta^\*$ the minimizer of the *empirical* loss, the *population* loss, or neither? Is it simply a fixed parameter value of the time-score model? If it corresponds to the *true data parameter*, then shouldn’t $L_{\text{TSM}}(\theta^\*) = 0$, assuming the neural network can perfectly approximate the true time score?

- **On Section 4.2.**  
  You parameterize general paths between $p_0$ and $p_1$, but it is unclear how you enforce the boundary conditions. Specifically, how do you ensure that at $t = 1$, we have $p_t = p_1$?

- **Potential inconsistency in Section 4.1.**  
  My understanding is that:
  $$
  \text{estimation error} \le e^L \times L_{\text{TSM}}(\theta^\*) 
  \le e^L \times (L_{\text{STSM}}(\theta^\*) + \text{slack}),
  $$
  where $L$ is a Lipschitz constant of the path, and the slack term corresponds to the kinetic energy of the path.

  *Issue #1*. The paper states that “to minimize the final estimation error, one must minimize the ideal score-matching loss.”  
  I believe this is a **sufficient** but not **necessary** condition, since the ideal score-matching loss only provides an upper bound.  
  If that upper bound is loose, it may be possible to reduce the estimation error without reducing the ideal score-matching loss.

  *Issue #2*. You aim to minimize the estimation error with respect to the choice of path $p_t$ and therefore propose minimizing its upper bound over paths. However, your analysis focuses only on minimizing the *slack term* (the kinetic energy) while seemingly ignoring the term $L_{\text{STSM}}(\theta^\*)$, which also depends on the choice of path. It is possible, for instance, that a path minimizing the kinetic term actually increases $L_{\text{STSM}}(\theta^\*)$, in which case the upper bound (and the estimation error) may not improve.  

---

## **Missing Reference**

The authors claim that the literature lacks a theoretical analysis of how the choice of path $p_t$ impacts the estimation error of the density ratio.  
While this remains an open and interesting problem, it would be relevant to cite prior work that provides partial theoretical results in this direction.  

In particular, **Theorem 4 of [1]** derives an upper bound on the estimation error of the density ratio as a function of the path’s Lipschitz constant—closely related to your Lemma 4.1, with a similar proof structure (see Appendix D.3 of that work).  
This reference would be highly appropriate to discuss, as it represents the closest known theoretical result to your contribution.  

Another reference that is less directly related but might be interesting to the authors is [2]. It also investigates the optimal path for reducing the estimation error of a density ratio using the time score identity, in the specific case when all the densities $p_t$ are parameterized only by their normalizing constant $Z_t$. 

[1] Yu et al. *Density Ratio Estimation with Conditional Probability Paths.* ICML 2025.  

[2] Chehab et al. Provable benefits of annealing for estimating normalizing constants... NeurIPS 2023. 


---

## **Evaluation**

- **On Figure 2.**  
  Figure 2 does not seem to show a substantial performance difference between optimal and suboptimal paths $p_t$.  
  Could the authors comment on this observation?

### Questions
This is an interesting paper and I am glad the authors are tackling how the estimation error of the density-ratio depends on the choice of path: this is an open question in the community. 

Could the authors address my questions detailed in the weaknesses section? In there, my main concerns are my "Issue #2" and "Evaluation" sections. I would be happy to raise my score if these are addressed.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper studies density ratio estimation problems that integrate time-scores along some path. Even though any smooth path is in principle okay, the practical performance of such methods depends on the actual path when approximations of the time score are used. The authors quantify the approximation error trough variance of the time-score along the path and show that choosing a minimum-variance path is optimal, both theoretically and empirically.

### Strengths
This is in overall a strong contribution for the DRE literature. The idea of analysing the variance of the time-score is intuitive but to my best knowledge novel, and the authors deliver a well-executed study that makes the connection between the variance and the estimation quality clear. Theorem 4.2 is a worthy objective in itself, and the closed-form expressions provided in Proposition 4.3 are interesting. The practical method is well explained and the empirical results provided in broad set of experiments are strong, and already the explanation for (one) reason for the path-dependency of practical performance would be valuable as such even if not demonstrating improved accuracy in practice.

### Weaknesses
The empirical gain is clear, but not so major that this would be a transformative contribution that would qualitatively change the state-of-the-art.

### Questions
No specific questions.

### Soundness
4

### Presentation
4

### Contribution
3
