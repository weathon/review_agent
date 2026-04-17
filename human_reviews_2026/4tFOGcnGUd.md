# Flatness-Aware Stochastic Gradient Langevin Dynamics

- Decision: Reject
- Scores: 0, 6, 2, 6

## Abstract
Generalization in deep learning is closely tied to the pursuit of flat minima in the loss landscape, yet classical Stochastic Gradient Langevin Dynamics (SGLD) offers no mechanism to bias its dynamics toward such low-curvature solutions. This work introduces Flatness-Aware Stochastic Gradient Langevin Dynamics (fSGLD), designed to efficiently and provably seek flat minima in high-dimensional nonconvex optimization problems. At each iteration, fSGLD uses the stochastic gradient evaluated at parameters perturbed by isotropic Gaussian noise, commonly referred to as Random Weight Perturbation (RWP), thereby optimizing a randomized-smoothing objective that implicitly captures curvature information. Leveraging these properties, we prove that the invariant measure of fSGLD stays close to a stationary measure concentrated on the global minimizers of a loss function regularized by the Hessian trace whenever the inverse temperature and the scale of random weight perturbation are properly coupled. This result provides a rigorous theoretical explanation for the benefits of random weight perturbation. In particular, we establish non-asymptotic convergence guarantees in Wasserstein distance with the best known rate and derive an excess-risk bound for the Hessian-trace regularized objective. Extensive experiments on noisy-label and large-scale vision tasks, in both training-from-scratch and fine-tuning settings, demonstrate that fSGLD achieves superior or comparable generalization and robustness to baseline algorithms while maintaining the computational cost of SGD, about half that of SAM. Hessian-spectrum analysis further confirms that fSGLD converges to significantly flatter minima.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The paper advances the literature by establishing a formal coupling between the Langevin temperature and random perturbation scale that provably links stochastic weight perturbations to convergence toward globally flat minima.

### Strengths
1. **Clear theoretical contribution.**  
The paper rigorously establishes that a specific coupling between the inverse temperature $\beta$ and the perturbation scale $\sigma$ guarantees that the invariant measure of fSGLD concentrates on flat minima.

2. **Strong theoretical guarantees.**  
The paper provides non-asymptotic convergence bounds in Wasserstein distance and an excess-risk bound, matching the best known rates for SGLD under comparable assumptions.

3. **Empirical validation consistent with the theory.**  
The experiments demonstrate that the theoretically prescribed $\beta$–$\sigma$ coupling consistently improves generalization and stability across some noisy-label and fine-tuning tasks, and in general, that your method is better than the others in most cases. Importantly, you tuned the hyperparameters fairly.

### Weaknesses
1. **Questionable premise on flatness and generalization.**  
   Contrary to the claim made in the Abstract, I do not find the assertion that generalization in deep learning is tied to the pursuit of flatter minima convincing. For instance, Adam is known to converge to sharper minima than SGD, yet AdamW remains the default optimizer in large-scale LLM training. In such settings, convergence is not even expected, let alone a meaningful notion of selecting flatter or sharper minima. Moreover, [1] empirically challenges the link between generalization and flatness across a broad range of architectures and training setups, undermining the central motivation of the paper.

2. **Misleading explanation for the lack of adoption of SGLD.**  
   The Introduction attributes the limited use of SGLD in practice to its lack of an intrinsic bias toward flat minima. I find this reasoning unconvincing. The primary reason SGLD (and SGD-like samplers) are not used in modern deep learning is that they fail to effectively optimize highly nonconvex, large-scale architectures such as transformers and LLMs. The discussion on flatness is likely secondary or even irrelevant in this context.

3. **Missing discussion of implicit regularization in SAM-like algorithms.**  
   Several recent works have shown that SAM and its variants implicitly regularize toward flatter regions (See Thm 3.2 and Thm 3.5, and their discussions in [2]), not necessarily by minimizing the Hessian trace directly, but through implicit regularization of the objective function and by how gradient noise interacts with local curvature: sharp minima tend to amplify noise, effectively promoting escape toward flatter areas. The paper should discuss this implicit regularization mechanism and compare it conceptually to that of fSGLD. Furthermore, it would be valuable to explore whether a similar stochastic-curvature coupling could be incorporated into fSGLD to improve its dynamics.

4. **Limited novelty of the core method.**  
   Random Weight Perturbation (RWP) is a well-established technique, and SGLD is a classical stochastic optimization method. The proposed algorithm essentially combines two well-known components without introducing a genuinely new conceptual element. While the theoretical analysis is rigorous, the algorithmic contribution feels incremental given the existing literature.

5. **Insufficient experimental relevance.**  
   The experimental setup does not reflect the scale or complexity of contemporary deep learning practice. Evaluations on noisy-label datasets and ViT fine-tuning are limited in scope. To substantiate claims of improved generalization and practical relevance, the method should be tested on modern architectures such as small LLMs (e.g., NanoGPT).

**IMPORTANTLY:**  
Are you certain that the SDE presented in your Equation 24 — which is meant to model the dynamics of the discrete-time optimizer defined in Equation 7 — is derived correctly? Can you rigorously justify this correspondence in the sense of *weak approximations*, as proposed in [3]?  

1. The **expected value of the increments** appears to be **correct**, as it corresponds to the expectation of  $ \theta_{k+1} - \theta_k$ thereby yielding the appropriate drift term.  

2. The **covariance of the increments**, however, appears to be **incorrect**. While the contribution from the stochastic term $\xi$ is included, the covariance arising from the injected perturbations $\epsilon$ is **not accounted for**, which is problematic. Preliminary calculations suggest that this additional covariance term scales with $\sigma$ and may also depend on the Hessian of $U$, resembling the stochastic–curvature interaction observed in the analysis of SAM in [2].  

This is a **major concern** and must be **thoroughly addressed and clarified**.  
Please see my derivation below — I would be glad to be proven wrong.

---

**In summary:**

1. The stated motivation, linking generalization to flatness, is weak and empirically disputable.  
2. The comparison with SAM and related implicit-regularization methods is underdeveloped.  
3. The proposed method is conceptually incremental, combining two classical ideas without introducing a fundamentally new mechanism.  
4. The experimental validation is not sufficient to support claims of superiority over AdamW, especially given recent optimizers like SOAP, Muon, or Scion that demonstrate stronger empirical evidence on large-scale models.

**Most importantly, there seems to be a mistake in the derivation of the SDE of the method. I provide my derivation below.**

---

**[1]** *A Modern Look at the Relationship between Sharpness and Generalization.*  
Maksym Andriushchenko, Francesco Croce, Maximilian Müller, Matthias Hein, Nicolas Flammarion.

**[2]** *An SDE for Modeling SAM: Theory and Insights.*  
Enea Monzio Compagnoni, Luca Biggio, Antonio Orvieto, Frank Norbert Proske, Hans Kersting, Aurélien Lucchi.  

**[3]** *Stochastic Modified Equations and Adaptive Stochastic Gradient Algorithms.*  
Qi Li, Cheng Tai, Weinan E.

### Questions
Here is my derivation of the SDE for Equation 7:

### Expected value of the increments

We consider the fSGLD iteration

\begin{equation}
\theta_{k+1}
= \theta_k - \lambda  \nabla_\theta U(\theta_k + \epsilon_{k+1}, X_{k+1}) + \sqrt{2\lambda \beta^{-1}} \xi_{k+1},
\end{equation}

where
$ \epsilon_{k+1} \sim \mathcal{N}(0,\sigma^2 I_d) $,  $\xi_{k+1} \sim \mathcal{N}(0,I_d)$, and all variables are independent.

The increment is

\begin{equation}
\Delta \theta_k = \theta_{k+1} - \theta_k = - \lambda  \nabla_\theta U(\theta_k + \epsilon_{k+1}, X_{k+1}) + \sqrt{2\lambda \beta^{-1}}\ xi_{k+1}.
\end{equation}

Conditioning on $\theta_k$:

\begin{equation}
\mathbb{E}[\Delta \theta_k] = - \lambda G(\theta_k),
\end{equation}

where

\begin{equation}
G(\theta) := \mathbb{E}\left[ \nabla_\theta U(\theta + \epsilon, X) \right].
\end{equation}

---

### Covariance of the increments

Define the zero-mean stochastic component

\begin{equation}
\delta_{k+1} := \nabla_\theta U(\theta_k + \epsilon_{k+1}, X_{k+1}) - G(\theta_k), \qquad \mathbb{E}[\delta_{k+1}] = 0.
\end{equation}

Then

\begin{equation}
\Delta \theta_k = -\lambda G(\theta_k) - \lambda \delta_{k+1} + \sqrt{2\lambda \beta^{-1}} \xi_{k+1}.
\end{equation}

Since $\delta_{k+1}$ and $\xi_{k+1}$ are independent and centered,

\begin{equation}
\operatorname{Cov}(\Delta \theta_k) = \lambda^2  \mathbb{E}[\delta_{k+1}\delta_{k+1}^\top \mid \theta_k] + 2\lambda \beta^{-1} I_d.
\end{equation}

Define

\begin{equation}
\Sigma(\theta) := \operatorname{Cov}\left[  \nabla_\theta U(\theta + \epsilon, X)\right],
\end{equation}

so that

\begin{equation}
\operatorname{Cov}(\Delta \theta_k \mid \theta_k) = \lambda^2 \Sigma(\theta_k) + 2\lambda \beta^{-1} I_d.
\end{equation}

---

### Limiting SDE

The continuous-time limit satisfies

$$
d\theta_t = - \lambda \mathbb{E}\left[\nabla_\theta U(\theta_t + \epsilon, X)\right] dt + \sqrt{2\beta^{-1} I_d + \operatorname{Cov}\left[    \nabla_\theta U(\theta_t + \epsilon, X)\right]} dB_t,
$$

where $B_t$ is a standard $d$-dimensional Brownian motion.

Probably, if we expand the gradient of $U$ with Taylor, we get that the additional covariance term I derived scales like $\sigma^2 H H^{\top}$ where H is the Hessian of U. But I did not work the calculations out, sorry.

### Soundness
1

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
It is well known that the generalization property is closely associated with flat minima phenomenon, that says that flat local minima lead to better generalization performance. This work proposes and studies Flatness-Aware Stochastic Gradient Langevin Dynamics (fSGLD) that optimizes a randomized-smooth objective that implicitly captures curvature information by adding Random Weight Perturbation (RWP). 

More specifically, consider $\min_{\theta\in\mathbb{R}^{d}}u(\theta)$, with $u(\theta)=\mathbb{E}[U(\theta,X)]$. One approach is to replace $u(\theta)$ by $v(\theta):=u(\theta)+\frac{\sigma^{2}}{2}\text{tr}(H(\theta))$, where $H(\theta)$ is the Hessian of $u$ at $\theta$. However, $\text{tr}(H(\theta))$ can be expensive to compute. If one takes $g_{\epsilon}(\theta)=\mathbb{E}[u(\theta+\epsilon)]$, where $\epsilon\sim\mathcal{N}(0,\sigma^{2}I_{d})$, then $g_{\epsilon}(\theta)\approx u(\theta)+\frac{\sigma^{2}}{2}\text{tr}(H(\theta))$, which motivates the author(s) of the paper to propose fSGLD:
\begin{equation*}
\theta_{k+1}=\theta_{k}-\lambda\nabla_{\theta}U(\theta_{k}+\epsilon_{k+1},X_{k+1})+\sqrt{2\lambda\beta^{-1}}\xi_{k+1}.
\end{equation*}
Intuitively, $\theta_{k}$ will converge to a stationary distribution that is close to $e^{-\beta v(\theta)}$ when $\sigma$ and $\lambda$ are small, and the Gibbs distribution will concentrate around the global minimizer of $v(\theta)$ when $\beta$ is large. The paper establishes non-asymptotic convergence guarantees in Wasserstein distance with the best known rate and derive an excess-risk bound for the Hessian-trace regularized objective. Extensive numerical experiments are also provided.

### Strengths
(1) fSGLD is a novel algorithm that is naturally motivated by flat minima phenomenon and Random Weight Perturbation from the literature.

(2) There is solid theoretical analysis that establishes non-asymptotic Wasserstein convergence guarantees for the proposed algorithm. With the presence of Random Weight Perturbation, the analysis is quite sophisticated.

(3) Numerical experiments are extensive and illustrative.

### Weaknesses
(1) One major weakness I see is that there does not seem to be any discussion comparing the iteration complexity or non-asymptotic convergence bounds for fSGLD vs SGLD theoretically. If you can have some theoretical result, or even just some double-well example for which you can show theoretically that fSGLD can outperform SGLD, that will strengthen the paper. Based on your current result, if you view SGLD as a special case of fSGLD as $\sigma\rightarrow 0$, you can perhaps directly compare fSGLD vs SGLD. 

(2) Proposition 3.1. seems to be an asymptotic result. In the proof, it seems most steps (if not are) are non-asymptotic. Also, since Theorem 3.2 and Corollary 3.3. are non-asymptotic, I guess Proposition 3.1. can also be made non-asymptotic. It would be really nice if you can make Proposition 3.1. non-asymptotic. If the bound is too complicated, you can highlight the big O dependence in the statement of Proposition 3.1. and refer to the proof section for the explicit expressions.

### Questions
(1) In Assumption 2 and throughout the rest of the paper, $x^{'}$ should be $x'$ and $\theta^{'}$ should be $\theta'$.

(2) You stated 1-Wasserstein convergence result in Theorem 3.2 and then the 2-Wasserstein convergence result as a corollary in Corollary 3.3. I thought 2-Wasserstein upper bounds the 1-Wasserstein distance. Intuitively, I expect the bound in Corollary 3.3. trivially provides a 1-Wasserstein bound for Theorem 3.2. If you can obtain tighter 2-Wasserstein bound by establishing 1-Wasserstein bound first, it would be helpful if you can add some discussions and explanations.

(3) Flat minima phenomenon says that SGD favors flat local minima that lead to better generalization. In designing fSGLD, it seems that you are making the global minimum flatter. Does fSGLD also make local minima flatter or it will only lead to a flatter global minimum and if that is the case, will that help you with generalization performance?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a new method, flatness-aware stochastic Langevin dynamics (fSGLD), as an efficient method for finding flat minimizer in nonconvex problems. Specifically, fSGLD is a variant of the typical SGLD (SGD with an additive Gaussian noise), with the modification that the gradient is evaluated at the current weight perturbed by a Gaussian noise. Then this paper proves that, in a certain asymptotic sense, fSGLD converges to the stationary distribution of SGLD that applied to the original objective with sharpness penalty. This paper further provides empirical verifications on the effectiveness of fSGLD in CIFAR-type problems, matching or outperforming benchmark methods such as SGD, AdamW, and SAM.

### Strengths
See below.

### Weaknesses
See below.

### Questions
The idea of fSGLD comes from random weight perturbation. The main contribution of this work lies in that it provides theoretical guarantees on how fSGLD indeed produces an effect of sharpness penalty, measured by trace of the Hessian. Regarding this, I have the following main questions:

1. The Assumptions 2-3 are quite strong,in particular, the Assumption 3. It would be very helpful to provide concrete (nonconvex) examples that satisfy those conditions. I don’t see a meaningful example for Assumption 3 except for linear models. If I understand correctly, networks with more than two layers would typically violate Assumption 2. 
2. Proposition 3.1 seems rather weak as $\alpha$ and $\beta$ are tied together. In particular, the equivalence only holds as $\beta \to \infty$, in which case $\alpha\to 0$, meaning that the sharpness penalty is asymptotic zero. In this sense, Proposition 3.1 does not reveal a direct connection between fSGLD and flat minimizer.  
3. I find the condition $\alpha = o(\beta^{-1/4})$ not intuitive. Why is the exponent $1/4$ instead of $1/6$ or so? I tried to dig into the proof, and find that equation (16) only keeps order-4 components in the Taylor expansion – why do the higher order terms disappear here? One selling point of this paper (Line 71) is the promise of dealing with higher order terms. However, I don’t quite understand why order-6 and higher terms disappear – is it part of the assumption? 
4. Theorem 3.2 and Corollary 3.3 have the similar issue, where the convergence only holds in the strong asymptotic sense that requires $\alpha\to 0$, thus their implication to finding a flat minimizer is limited. 
5. The proof of Theorem 3.2 and Corollary 3.3 seems mainly using techniques from Zhang et al 2023. It would be helpful to clarify the novelty here. 

Besides, I have a couple of minor suggestions. 

6. Line 126. Computing the trace of Hessian does not need to be expensive in high dimensions. For example, one can avoid dimension-dependent cost by doing the trick of commuting gradient and linear operators and using monte carlo. 
7. In Theorem 3.2 and Corollary 3.3, certain quantities are called “constants”, e.g., D with underline. However, these quantities go to zero in order for the bounds to be useful, which means they are not constants. Additionally, it would be better to make $\dot{c}, D_1, D_2, D_3$ etc explicit in the main paper – at least clarify which given constants they depend on. 

8. Line206, "is is i.i.d." -> "is i.i.d."

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose an algorithm for flatness-aware optimisation in non-convex settings. Their method combines stochastic gradient Langevin dynamics (SGLD) and random weight perturbations. They propose using SGLD applied to a Laplacian-regularised modification of the objective, estimating the Laplacian through random weight perturbations. They compare this method against standard optimisation techniques and SAM, the most popular flatness-aware optimiser, demonstrating improved results.

### Strengths
* Addresses a classical yet current problem in machine learning: tractable flatness-aware optimisation.
* Provides a rigorous mathematical proof of convergence.
* Using its theory, the authors identify a good rate of coupling between SGLD temperature and the scale of Laplacian regularisation. They then show that this coupling works well in their experiments.
* Not only obtains convergence bounds but also provides excess risk bounds.
* The experiments consider a realistic setting.

### Weaknesses
* The dependence on dimension in the bound appears poor (exponential in $d$), requiring the step size to decay exponentially with $d$. While common in the SGLD literature (and thus not a fundamental flaw in novelty), the authors should mention this limitation in the main body, as dimension-dependence is critical for the large-scale settings targeted in the experiments.
* A large portion of the proof appears to be borrowed from previous papers on SGLD, making it difficult to isolate the novel theoretical contributions. I would suggest including a discussion on what is new in the proof technique or, if the technique is standard, a statement clarifying that it is adapted from prior work.
* In general, there is little discussion surrounding the theoretical results, making them somewhat opaque to the theoretical audience this paper will likely appeal to. For example, it would be helpful to decompose the error terms in the bound (e.g., from convergence, discretisation, and objective estimation). In particular, what is the error contributed by the RWP estimator, and how does it scale with dimension?
* There is a mismatch between the theory (online setting) and the experiments (multi-pass setting). While this may not invalidate the results, it is a detail that the authors should disclose and briefly discuss.
* The authors do not appear to use early stopping, which is common for the baselines considered. It would be interesting to know if early stopping interacts with the curvature of the final minimisers. A plot of the train curve for these methods might help with aleviating this concern.
* It is difficult to find details in the paper due to imprecise referencing to the appendix. For example, the constants $D_1, \dots, D_7$ in the main theoretical statements are critical for understanding dimension and temperature dependence. The authors should reference the specific equations (e.g., (65) and (67)) where these are defined, rather than referencing the entire appendix section. Similarly, I struggled to find the "compelxity analysis" referenced in the main body.

### Questions
* This work targets the online setting in the theory and the multi-pass setting in the experiments. How does the theory compare in the multi-pass setting, and how do the experiments compare in the online setting?
* Is the computational cost truly comparable to SGD? Does the RWP and Langevvin noise in fSGLD not necessitate a smaller step-size than standard SGD, thus requiring more steps to converge?
* How does early stopping affect the experiments and the curvature of the resulting solutions?
* How does the bound compare to standard SGLD bounds? Which error terms are brought about by the RWP estimate?

### Soundness
3

### Presentation
3

### Contribution
2
