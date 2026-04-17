# Black-box unadjusted Hamiltonian Monte Carlo

- Decision: Reject
- Scores: 4, 6, 2, 8

## Abstract
Hamiltonian Monte Carlo and underdamped Langevin Monte Carlo are state-of-the-art methods for taking samples from high-dimensional distributions with a differentiable density function. To generate samples, they numerically integrate Hamiltonian or Langevin dynamics. This numerical integration introduces an asymptotic bias in Monte Carlo estimators of expectation values, which can be eliminated by adjusting the dynamics with a Metropolis-Hastings (MH) proposal step. Alternatively, one can trade bias for variance by avoiding MH, and select an integration step size that ensures sufficiently small asymptotic bias, relative to the variance inherent in a finite set of samples. Such unadjusted methods often significantly outperform their adjusted counterparts in high-dimensional problems where sampling would otherwise be prohibitively expensive, yet are rarely used in statistical applications due to the absence of an automated way of choosing a step size. We propose just such an automatic tuning scheme that takes a user-provided asymptotic bias tolerance and selects a step size that ensures it. The key to the method is a relationship we establish between the energy change during integration and the asymptotic bias. We show that this procedure rigorously bounds the asymptotic bias for Gaussian target distributions. We then numerically show that the procedure works beyond Gaussians. To demonstrate the practicality of the proposed scheme, we provide a comprehensive comparison of adjusted and unadjusted samplers on Bayesian inference problems and on a statistical physics model in more than one million parameters. With our tuning scheme, the unadjusted methods achieve close to optimal performance, significantly and consistently outperforming their adjusted counterparts.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
Although adjusted sampling methods (which use Metropolis-Hastings algorithm for correction) do not have asymptotic bias due to finite step size, they are vulnerable to the curse of dimensionality. As such, the paper's goal is to find a good balance between efficiency and accuracy for unadjusted sampling methods, by adaptively changing step size based on the running estimate of EEVPD.

### Strengths
* The paper is nicely written and easy to follow
* The paper is supported by theoretical analysis and numerical experiments
* The proposed method is intuitive and seems effective

### Weaknesses
* The automatic tuning is too simple. This is not necessarily a weakness, but I want the authors to provide more context -- why haven't people come up with something similar before? If so, what is the conceptual challenge here? If not, how is the proposed method compared to previous methods that adaptively adjust step sizes?
* Similar to the first point, I'm not totally sure about the novelty and usefulness of the proposed method. My general impression is that HMC/LMC is a well-developed field. I'm willing to increase the score if the authors can justify more the novelty and contribution to the field.

### Questions
* Can you provide a concrete example to show how $\epsilon$ is adaptively changed when navigating through different parts of the landscape (for flatter regions, a larger $\epsilon$ is desired; for sharper regions, a smaller $\epsilon$ is desired)? Can the algorithm rediscover this trend?
* Why in Figure 1, the adjusted has a bigger variance than the unadjusted, despite the fact that they are both $1/t$?

### Soundness
2

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
3

### Summary
This paper studies how to adjust the integration stepsize automatically to control the bias in HMC and Langevin Monte Carlo while without using the Metropolis-Hastings (MH) correction. The key idea behind this paper is that the energy error during numerical integration directly controls the asymptotic bias. As a result, one can adaptively selects the step size to guarantees a bias tolerance by monitoring the energy error variance per dimension (EEVPD). 

Specific contributions include:
1. Analytical results for Gaussian distribution, showing the relationship betwen EEVPD and integration stepsize (Lemma 4.1) as well as the associated covariance matrix bias.
2. Numerical impelmentation of the automatic tuning scheme;
3. Results on multiple Bayesian inference benchmarks (e.g., Gaussian, Brownian, Item Response, Stochastic Volatility) and a statistical physics model with >1 million parameters.

### Strengths
1. The proposed method has strong theoretical fundation for Gaussian distribution. It also provides empirical results to show the effectiveness on non-Gaussian distributions.
2. The numerical results show that the proposed approach has achieved near-optimal RMSE with orders-of-magnitude fewer gradient evaluations than adjusted (MH) methods. It also outperform NUTS and manually tuned adjusted LMC/HMC.
3. Open-source codes are provided.

### Weaknesses
-The main weakness of this paper is the lack of theoretical justification for non-Gaussian cases. The reviewer understand that this paper still provides enough novel contribution even without theoretical justification about non-Gaussian cases. However, it will be great if the authors can provides enough explaination/clarification regarding when this framework will work or not work for non-Gaussian cases.

### Questions
--In order to better understand the strenth/weakness of this method in non-Gaussian distributions, can you provide the numerical results on some multi-modal distributions that are much different from Gaussian dsitributions (e.g., the Gaussian mixture distribution where each Gaussian distribution is far away from each other but has similar variance)?

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
The paper aims to develop a practical method for selecting the discretization step-size in Hamiltonian Monte Carlo (HMC) and Underdamped Langevin Monte Carlo (ULMC) algorithms. The theoretical analysis is restricted to basic computations in a Gaussian setting. Based on these computations, an algorithm is proposed—though its derivation lacks precision—and its performance is evaluated through numerical simulations on selected examples.

### Strengths
The paper addresses an important, timely problem relevant to the ICLR audience: selecting the step size in MCMC algorithms. This challenge has broad applications across machine learning and computational statistics.

I ike the idea of monitoring the expected change of the Hamiltonian.

### Weaknesses
### 1. Theoretical Contributions
The theoretical contributions of the paper are simple and of limited interest:
The justification for minimizing EEVPD is unconvincing:
  - There is no clear evidence that EEVPD can be easily approximated in practice.
  - In non-Gaussian settings, there is no reason to assume that EEVPD controls the Wasserstein error.
  - Even if a function $\phi$ exists such that an EEVPD smaller than $\varepsilon$ implies a Wasserstein error smaller than $\phi(\varepsilon)$, the form of $\phi$ would likely depend on the target distribution, limiting the generality of the result.

---

### 2. Methodological Contributions
The main contributions of the paper are methodological, yet they lack clarity and practical guidance:
- A pseudo-code for the proposed method is missing. Section 6 provides only a high-level description, which is insufficient for understanding how the method should be implemented.

---

### 3. Unjustified Claims
There are several strong and unjustified claims; some of them are listed below:
- **Line 67:** *"In statistics, MH-adjusted versions of these dynamics are used almost exclusively."*
  This is an overgeneralization and lacks nuance.
- **Line 77:** *"Thus, unadjusted methods provably outperform adjusted methods."*
  
- **Line 79:** *"Without a principled way to do so, general practitioners cannot use unadjusted algorithms in a black-box fashion, which gravely limits their applicability."*
  This claim is inconsistent with practice: practitioners routinely use stochastic gradient descent (SGD) and its variants in deep learning, despite the lack of a universally accepted principled method for selecting the learning rate.

---

### 4. Major Weakness: Practical Estimability
A critical weakness of the paper is its reliance on the following statement from **line 253**:
*"Crucially, this is a quantity that can easily be estimated in practice."*
The quantity referred to involves an expectation with respect to the stationary distribution. For arbitrary target distributions, it is unclear when stationarity is achieved or how closely the process approaches stationarity after a finite number of steps. As a result, the claim that this quantity can be easily estimated lacks justification.

---

### 5. The fact that the initialization error is disregarded, as written on line 80, is an important limitation.

### Questions
1. In multiple instances, such as on line 175, general statements are supported by one or two references. For example:
   *"Unadjusted methods have also been analyzed theoretically, establishing a bound on the asymptotic bias and the mixing time."*
   It is unclear how the cited works were selected. Are they the earliest papers on the topic, the most recent, or those providing the sharpest results?

2. Lines 88–90 state:
   *"Noting that chains are finite in practice, one could negotiate the tradeoff differently and reduce the variance at the cost of introducing some bias. This strategy is the focus of the present work."*
   Initially, I interpreted this as proposing a method to select the step-size by balancing bias and variance. However, I now suspect this is not the intended meaning. Is my understanding correct?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes a black‑box tuning scheme for unadjusted, gradient‑based samplers (HMC, underdamped LMC...). The key idea is to control asymptotic discretization bias by monitoring the Energy‑Error Variance Per Dimension (EEVPD), i.e. the variance of the one‑step Hamiltonian error normalized by dimension. This is cheap to estimate (online) quantity from the integrator’s energy changes. For Gaussian targets, the authors prove sharp, dimension‑wise bounds connecting EEVPD to (i) a covariance‑error divergence and (ii) Wasserstein‑2 distance, and they use these bounds to set a target EEVPD and adapt the step size automatically.

### Strengths
- The proposed criterion EEVPD is intuitive and simple to compute online. Furthermore it is backed with some theory in a Gaussian setting and for the more general case the authors show experimentally that a function of EEVPD upperbounds $b_cov$, suggesting some generality to the result proved in theorem 4.2
- The paper contains broad benchmarks with high‑dimensional case and the method is winning against NUTS/adjusted baselines at fixed error. 
- Overall I found the paper to be well-written, with clear theory and experiments. The paper is definitely interesting for anyone in the ICLR  community interested in Bayesian inference and MCMC algorithms.

### Weaknesses
The only weakness of this paper is the scope of the theoretical results. It would have certainly been better to have more general results (at least under log-concavity).

### Questions
N/A

### Soundness
4

### Presentation
4

### Contribution
3
