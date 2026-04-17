# DR-SAC: Distributionally Robust Soft Actor-Critic for Reinforcement Learning under Uncertainty

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Deep reinforcement learning (RL) has achieved remarkable success, yet its deployment in real-world scenarios is often limited by vulnerability to environmental uncertainties. Distributionally robust RL (DR-RL) algorithms have been proposed to resolve this challenge, but existing approaches are largely restricted to value-based methods in tabular settings. In this work, we introduce Distributionally Robust Soft Actor-Critic (DR-SAC), the first actor–critic based DR-RL algorithm for offline learning in continuous action spaces. DR-SAC maximizes the entropy-regularized rewards against the worst possible transition models within an KL-divergence constrained uncertainty set. We derive the distributionally robust version of the soft policy iteration with a convergence guarantee and incorporate a generative modeling approach to estimate the unknown nominal transition models. Experiment results on five continuous RL tasks demonstrate our algorithm achieves up to $9.8\times$ higher average reward than the SAC baseline under common perturbations. Additionally, DR-SAC significantly improves computing efficiency and applicability to large-scale problems compared with existing DR-RL algorithms.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Distributionally Robust Soft Actor-Critic (DR-SAC), a model-free reinforcement learning algorithm that integrates distributionally robust optimization into the Soft Actor-Critic framework. The key idea is to handle environment uncertainty via a KL-divergence–bounded uncertainty set over transition distributions. The authors derive a distributionally robust soft Bellman operator using KL duality, which introduces a scalar dual variable $\beta$ for each $(s,a)$ pair. To avoid solving numerous local optimizations, they replace these with a single functional optimization $g(s,a)$ over the entire dataset, justified by interchange theorems from variational analysis. To handle continuous state–action spaces and offline settings, the paper incorporates a VAE-based generative model to approximate the nominal transition distribution $p^0(s'|s,a)$, thereby mitigating the double-sampling issue that arises from nonlinear expectations in the KL dual form.

### Strengths
- **Methodological innovation**: Introducing KL-based DR soft Bellman operator and replacing numerous scalar $\beta$-optimizations with a single functional optimization are novel and computationally effective.
- **Efficiency-performance trade-off**: Demonstrates superior robustness compared to SAC/RFQI while reducing training time.

### Weaknesses
1. **Theory–implementation gap**: The convergence and monotonicity proofs apply only to tabular finite-action cases, whereas the implemented DR-SAC relies on continuous Gaussian policies and neural approximators.
2. **VAE dependency**: While the need for a generative model (to avoid double-sampling) is well justified, empirical validation of the VAE’s adequacy is limited (mostly latent-dimension sweeps) and lacks comparisons with alternative generative models such as flows or diffusion models.
3. **Limited robustness evaluation**: Perturbations focus on Gaussian noise and parameter scaling. Tests under more challenging conditions (e.g., heavy-tailed, regime-switching, or reset/restart scenarios) are absent. The adequacy of the KL-ball assumption versus, e.g., Wasserstein metrics remains unexamined.

### Questions
1. With errors from function approximation, VAE modeling, and functional optimization present, can the authors provide partial error–convergence guarantees (e.g., one-step contraction with bias)?
2. What is the quantitative rationale for choosing a VAE over flow-, diffusion-, or score-based transition models?
3. Do the authors believe the KL-ball assumption adequately captures realistic structural or regime shifts? Would a Wasserstein-ball or f-divergence family extension offer advantages?
4. (minor) Please fix typographical errors ("In iour algorithm") and ensure consistent terminology (e.g., "KL ball").

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
This work proposes a DR-SAC algorithm under a KL-divergence-based uncertainty set. It employs a Variational Autoencoder (VAE) to estimate the empirical distribution from the offline dataset. The proposed algorithm is model-free and can be applied to environments with continuous action spaces.

### Strengths
1. This algorithm can be applied in continuous action/state space setting. 

2. Using VAE to avoid double sampling in offline DR-RL problem. 

3. This work provide convergence guarantee.

### Weaknesses
1. The presentation of this work is not very clear or professional. The gradients used in the algorithm section are not defined where they first appear—I had to consult the appendix to understand their meaning. Moreover, the assumptions on which the convergence analysis relies are not stated explicitly, making it difficult to follow the theoretical results.

2. The proof section lacks sufficient rigor. The main theorem is presented without clearly stating the required assumptions, and the accompanying proof is incomplete and lacks detail. A more thorough and transparent exposition of the assumptions and proof steps is necessary to make the theoretical claims convincing.

### Questions
1. Could the authors provide a more detailed discussion on why the VAE can effectively avoid the double-sampling process, and whether the estimated distribution satisfies the requirement of being an unbiased estimator? In previous offline DR-RL works, the use of double-sampling and treating the empirical distribution derived from offline data has been theoretically justified — how does this approach maintain similar rigor?

2. Are there any additional requirements or assumptions regarding the offline dataset (e.g., data coverage, distributional support, or sample quality) for the proposed method to perform effectively?

### Soundness
2

### Presentation
1

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
This paper tackles the problem of developing a distributionally robust version of the Soft Actor-Critic (SAC) for offline reinforcement learning (RL) from a dataset with a continuous action space. Specifically, the authors develop a maximum entropy framework for policy iteration to accomplish the RL objective across an uncertainty set of transitions defined by KL-divergence, showing that their algorithm converges. They introduce a technique that reformulates the harder state-action optimizations into a single optimization problem over a function space. Through the use of a Variational Autoencoder (VAE), they are able to estimate the nominal transitions and thus generate samples without requiring access to a simulator.

### Strengths
1. The paper is well written and easy to follow. Namely, the paper systematically addresses the challenges discussed in section 3.1 in the subsequent sections.
2. The use of interchanging minimization and integration in a decomposable space under the stated assumption is clever.
3. The introduction of a VAE to estimate the nominal transition kernel for a given state-action pair, $p_{s,a}^0$ alleviates the double-sampling issue encountered in the calculation of the empirical risk by producing synthetic samples. This combined with the above point and the use of a neural network allows for the extension to a continuous action space.
4. Extensive experimentation indicates the validity of the author's theoretical claims.

### Weaknesses
1. There appears to be a fundamental disconnect between the convergence guarantees in Theorem 3.6 (which rely on the assumption that $|\mathcal{A|<\infty}$ and hence tabular) and the practical extension to continuous actions in the algorithm's construction. Namely, the convergence guarantees are not formally proven under continuous actions, but rather implied through the use of a neural network and empirical validation.
2. The use of a VAE to approximate the nominal transition kernel introduces additional risk due to the optimization occurring on this estimation and not the true environment. Should a given dataset not have sufficient coverage, the VAE would learn a model that is not necessarily robust to the true environmental shifts, but rather be robust to some other environmental shifts. The authors should consider including further discussion on this topic in the ablation studies in section 4.3 or additional theory-based discussion on this in the appendix. Can you provide a reference that provides some additional clarity on the above point?

### Questions
1. On line 122, adding "discounted" in front of "Markov Decision Process" adds clarity to exactly what problem setting you are considering.
2. Extra closing parenthesis on line 125.
3. Inconsistent notation of $\Delta(\mathcal{S}), \Delta(S),$ and $\Delta(|\mathcal{S}|)$ on line 126 and equation 5.
4. Addition of $\forall s\in\mathcal{S}$ to make equations 6-8 more precise.
5. Use of $|A|<\infty$ in Assumption 3.1 versus $|\mathcal{A}|<\infty$ elsewhere.
6. Please clarify what values $\tau$ takes, and it's impact in the algorithm.
7. It would help to make your work more appealing to a wider audience unfamiliar with deep learning by precisely clarifying that $\hat{\nabla}$ denotes the empirical stochastic gradient estimate in Algorithm 1.
8. Is the standard deviation plotted in figure 1.e?
9. Typo of "grind" instead of "grid" on lines 1190 and 1240.
10. Typo of "larges" on line 1379.
11. Addition of grid lines in figures 7 and 9 for consistency.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduce a new  Distributionally Robust SAC algorithm. Theoretical analysis and numerical examples are given to show the advantages of the proposed methods.

### Strengths
The paper introduce a new  Distributionally Robust SAC algorithm. Theoretical analysis and numerical examples are given to show the advantages of the proposed methods.

### Weaknesses
The benchmarks used are low dimension examples. It would be good to add some higher dimension examples with continuous action spaces such as Ant in Mujoco environments.

### Questions
Is there any limitation for the general distributional robust RL framework? 
Is it possible to learn the uncertainty of the environment and onine adapt?

### Soundness
3

### Presentation
3

### Contribution
3
