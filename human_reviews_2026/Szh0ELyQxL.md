# Information Shapes Koopman Representation

- Avg Score: 5.50
- Decision: Accept (Oral)
- Scores: 6, 6, 6, 4

## Abstract
The Koopman operator provides a powerful framework for modeling dynamical systems and has attracted growing interest from the machine learning community. However, its infinite-dimensional nature makes identifying suitable finite-dimensional subspaces challenging, especially for deep architectures. We argue that these difficulties come from suboptimal representation learning, where latent variables fail to balance expressivity and simplicity. This tension is closely related to the information bottleneck (IB) dilemma: constructing compressed representations that are both compact and predictive. Rethinking Koopman learning through this lens, we demonstrate that latent mutual information promotes simplicity, yet an overemphasis on simplicity may cause latent space to collapse onto a few dominant modes. In contrast, expressiveness is sustained by the von Neumann entropy, which prevents such collapse and encourages mode diversity. This insight leads us to propose an information-theoretic Lagrangian formulation that explicitly balances this tradeoff. Furthermore, we propose a new algorithm based on the Lagrangian formulation that encourages both simplicity and expressiveness, leading to a stable and interpretable Koopman representation. Beyond quantitative evaluations, we further visualize the learned manifolds under our representations, observing empirical results consistent with our theoretical predictions. Finally, we validate our approach across a diverse range of dynamical systems, demonstrating improved performance over existing Koopman learning methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents an information-theoretic framework for learning Koopman representations of nonlinear dynamical systems.In this paper, authors proposed a unified framework that balances simplicity and expressiveness in learning finite-dimensional Koopman representation.

### Strengths
(1) The proposed information-theoretic Lagrangian provides a clear mathematical formulation for balancing simplicity and expressiveness of predictive sufficiency;
(2) The information decomposition in Proposition 3 connecting mutual information components to Koopman spectral properties is insightful. This bridges dynamical systems theory with information theory smoothly;
(3) Experiments across three diverse domains demonstrates broad practical application.

### Weaknesses
(1) While the paper provides asymptotic and information-theoretic insights, it lacks finite-sample guarantees or non-asymptotic convergence results;
(2) The practical loss in Eq. 9 is a little bit heavy, with many tunable hyperparameters $\alpha, \beta, \gamma$. It is unclear how sensitive performance is to these values;
(3) Missing comparisons or discusstion of the following recent deep Koopman methods:
  (a) https://www.nature.com/articles/s41467-018-07210-0;
  (b) https://openreview.net/pdf?id=Svk7jjhlSu;
  (c) https://pubs.aip.org/aip/cha/article/27/10/103111/151485;
  (d) https://www.sciencedirect.com/science/article/abs/pii/S0021999124004431;
  (e) https://link.springer.com/article/10.1007/s00332-019-09567-y.

### Questions
(1) How sensitive are the results to the choice of von Neumann entropy regularization coefficient $\gamma$? What happens if you set $\gamma = 0$ (no entropy regularization) on all tasks quantitatively?
(2) The current formulation involves covariance and entropy computations, then how does the computational cost scale for high-dimensional latent spaces?
(3) How does your approach perform on stochastic dynamical systems, where the Koopman operator is no longer deterministic setting?
(4) The InfoNCE computation treats temporal neighbors as positives. But if the system has $T$-periodic dynamics, shouldn't $z_{T}$ also be a positive? How do you handle periodicity?

### Soundness
2

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
The paper proposes a new representation learning scheme based on the Information Bottleneck framework for learning Koopman operators. It addresses a common flaw in current methods, mode collapse, by introducing a novel information-theoretic Lagrangian that balances latent mutual information (for simplicity) against von Neumann entropy (to ensure expressiveness). The paper experimentally evaluates the proposed algorithm across a range of dynamical systems, demonstrating improved performance against baseline methods.

### Strengths
The paper addresses an important challenge in data-driven dynamical systems: learning stable, interpretable, and expressive finite-dimensional representations of the Koopman operator. The authors see the problem through a general, principled, and information-theoretic approach to manage the simplicity-expressiveness trade-off. The latent mutual information ($I(z_{t-n}; z_t)$) promotes temporal coherence but can lead to mode collapse, while von Neumann entropy ($S(\mathcal{C})$) promotes expressiveness and spectral diversity, actively preventing such collapse. This leads to a novel information-theoretic Lagrangian objective for Koopman learning. The paper is well-written and logically organized, making it easy to follow and comprehend. Its clarity is further enhanced by the inclusion of informative diagrams.

### Weaknesses
1. The paper's core premise is that standard representation learning (like VAEs) is insufficient for Koopman models because it prioritizes reconstruction over the more restrictive constraint of a linear predictive structure (structural consistency). While this claim is valid, the paper frames this as a relatively novel insight, overlooking a significant body of existing literature that has already identified and addressed this exact problem. For instance, works on VAMPNets (arXiv:1710.06012)  and, more recently, DPNets (	arXiv:2307.09912) are explicitly designed to learn representations that linearize the dynamics, moving far beyond simple reconstruction. By not citing or benchmarking against these more relevant baselines (as well as other related works, such as arxiv:2309.07200), the paper overstates the novelty of its problem formulation, making it difficult to assess how this new solution compares to other state-of-the-art methods targeting the same fundamental challenge.

2. There is a gap between the "pure" information-theoretic Lagrangian (Eq. 8) and the tractable loss function used for optimization (Eq. 9). The practical implementation relies on several approximations:
    
- The latent mutual information $I(z_{t-n}; z_t)$ is approximated using the InfoNCE contrastive loss, which is a lower bound, not the true MI.
- The von Neumann entropy $S(\frac{\mathcal{C}}{tr(\mathcal{C})})$ is calculated using a stochastic approximation of the covariance matrix $\mathcal{C}$ derived from a single minibatch (Algorithm 1, line 7). This estimate is likely to be very noisy and may introduce high variance into the gradients, potentially affecting training stability and the quality of the final manifold.

3. As the authors state in their appendix (Appendix D), the framework does not yet provide formal guarantees on sample complexity or the non-asymptotic convergence of the representation. It is unclear how many data samples are needed to reliably estimate the information-theoretic quantities (especially the minibatch-based entropy) and converge to a meaningful Koopman subspace.


_Minor_:

- In Eq. 14, line 1003, I believe that it should be $I(x_{t−1}; x_t) − I(z_{t−1}; z_t)$.
- Figure 7 is a bit misleading, in the decoding phase, as it is mentioned in the caption of the figure, “ The latent variables are subsequently decoded back to approximate the original states,” but in the diagram, it goes back to the original state. Therefore, one could wonder why $I(z_{t-1}; x_t | z_t)$ is not zero.

### Questions
1. How sensitive the algorithm is with respect to hyperparameter optimization? I haven't found any discussion on hyperparameter selection; is there any intuition or heuristic for choosing (k, alpha, beta, gamma)? Especially since the ablation study in Figure 5 clearly demonstrates that the balance between these terms is critical to success.
2. In the abstract, the Authors noted that KO's infinite-dimensional nature makes finding a good finite-dimensional subspace challenging, especially for deeper architectures. Could you elaborate on why deeper architectures tend to suffer more?
3. At the beginning of section 3.2, the authors said, “The latent mutual information quantifies the magnitude of error, but not the nature of the information lost in the Koopman representation.”, which sounds peculiar to me. Could you elaborate on what you mean by “nature of the information lost”?
4. I don’t understand the meaning of the red and blue arrows in Proposition 3. Does that mean we want to maximize the blue and minimize the red?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors explore a new, information-theoretic approach to Koopman operator learning, which is used to model dynamical systems. They argue that existing methods struggle because latent variables often fail to balance expressivity (capturing rich dynamics) and simplicity (remaining compact and interpretable). Drawing parallels to the information bottleneck principle, they introduce a Lagrangian formulation that explicitly trades off these two goals. In this setup, latent mutual information promotes simplicity, while von Neumann entropy encourages diversity and avoids mode collapse. They propose a corresponding algorithm that stabilizes learning, improves interpretability, and yields better empirical performance on various dynamical system benchmarks.

### Strengths
S1. The paper presents a fresh conceptual perspective by linking Koopman learning with the information bottleneck framework, which deepens theoretical understanding and connects two previously distinct research areas.

S2. The proposed Lagrangian and algorithmic formulation is both principled and practical, demonstrating consistent empirical improvements and more interpretable latent structures.

S3. The paper provides a well-structured and thorough analysis of the information components underlying the Koopman operator, offering a clear theoretical justification for the proposed Lagrangian formulation.

S4. The visualizations of the learned manifolds are insightful and effectively illustrate how the proposed method improves representation quality and interpretability.

S5. The quantitative results show that the proposed approach consistently outperforms existing methods, supporting its practical advantages and robustness.

### Weaknesses
W1. The framework, while elegant, is conceptually dense and abstract, making it difficult to assess how easily it can be implemented or scaled to complex, high-dimensional real-world systems.

W2. The empirical validation, though broad, seems to focus mainly on illustrative examples—more rigorous or comparative testing on large-scale benchmarks would strengthen the claims of generality and robustness.

W3. While the information-theoretic framing is creative, the use of mutual information and entropy as tradeoff terms resembles existing regularization and variational methods, making the contribution partly incremental rather than entirely groundbreaking.

### Questions
Q0. Does the proposed approach still model a true Koopman operator, or has it effectively become an information-theoretic Lagrangian framework—based on mutual information and entropy—that only mimics Koopman behavior?

Q1. How sensitive is the proposed Lagrangian formulation to the relative weighting between mutual information and von Neumann entropy? In other words, does performance degrade noticeably if this balance is not carefully tuned?

Q2. Can the authors demonstrate the scalability of their method by applying it to higher-dimensional or real-world dynamical systems to validate its robustness beyond synthetic benchmarks?

Q3. Are there other approximate Koopman operator methods that could be compared against the proposed approach? It would be helpful to see how this method performs relative to alternatives that may more closely capture the underlying manifold, even if the proposed one achieves better overall results.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The key idea of this paper is that learning a good Koopman latent space can be seen as an information bottleneck problem: the goal is a latent representation that is both simple (compact, linear) and expressive (able to predict the future).
To formalize that, the authors derive an information-theoretic Lagrangian with three main terms:
- Latent mutual information -- to keep temporal coherence and predictive power.

- Von Neumann entropy -- to prevent mode collapse and keep latent diversity.

- Conditional mutual information -- to suppress irrelevant or fast-dissipating information.

They build a new Koopman learning algorithm that implements this Lagrangian, combining VAEs with information-theoretic regularization. Experiments on chaotic systems (Lorenz, vortex flows), visual control tasks, and graph-structured dynamics show better long-term prediction and more stable latent manifolds than existing Koopman Autoencoders or PFNNs.
Conceptually, they link Koopman spectral modes to information flow -- coherent modes near |λ|=1 correspond to persistent information, while dissipative modes correspond to information loss.

### Strengths
- The connection between spectral theory and the information components of the Koopman representation is very interesting. This can bring good information-theoretic insight into dynamical representation learning.

### Weaknesses
1. The paper often claims that it learns a Koopman operator or a Koopman representation, but the actual results only show a **latent linear predictor** -- not a true Koopman operator. The authors present a latent-linear model and refers to the learned matrix $K_{\psi}$ as a “Koopman operator”. However, unlike DMD or EDMD, which explicitly enforce the Koopman relation φ(T(x)) = K φ(x) over sampled trajectories, this work only trains $K_{\psi}$ to minimize predictive error ($z_{t+1} ≈ K_{\psi}  z_t$) or a KL divergence with a linear-Gaussian prior. This ensures linear predictability in the learned coordinates, but not Koopman invariance of the feature map φ(x). The paper provides no invariance or reconstruction tests. Consequently, $K_{\psi}$ may simply act as a best-fit linear regressor in latent space rather than an approximation of the true Koopman operator. More clear, the paper trains a neural encoder $φ_θ(x)$ that maps x_t → z_t,
and a linear map $K_{\psi}$ that predicts: $z_{t+1} ≈ K_{\psi}  z_t$.  This looks like DMD in the latent space. However, the key difference is how $φ_θ$ is chosen and what it represents. In DMD/EDMD, φ is fixed (a known basis) and directly tied to the system's state, so the learned K approximates the true Koopman operator on that basis. In this paper, $φ_θ$ is learned jointly with $K_{\psi}$ to minimize reconstruction and prediction losses. There is no constraint ensuring $φ_θ$ produces Koopman-invariant coordinates. As a result, $K_{\psi}$ is just the best linear predictor in whatever feature space $φ_θ$ learns, not necessarily the Koopman projection of the true dynamics.

2. Some propositions depend on unproven assumptions as the proofs and closed-form identities (in the Appendix) repeatedly assume linear–Gaussian dynamics, full-rank covariances, and ergodicity to move from general nonlinear dynamics to tractable formulas. These assumptions are used without careful justification (or statements of when they hold). for instance,  Appendix F (derivations for conditional mutual information): the Gaussian closed-form identity and linear–Gaussian relations are introduced and used to produce equations (32)–(35). It writes z_{t} | z_{t−n} ∼ Normal(Kⁿ z_{t−n}, M_n) and uses linear Gaussian formulas to get closed-form mutual information (e.g., eq. (33)–(35)). Those closed forms are valid if the latent process is linear Gaussian. But the encoder is a neural network mapping high dimensional x to z; there is no general reason the induced z dynamics are exactly linear Gaussian. The paper uses the Gaussian identities to make spectral claims about Koopman eigenvalues and MI, and then presents those claims as general. Moreover
- Ergodicity: Remark F.4 after Proposition 2 uses an ergodic limit for long-time averages.
- Full-rank covariance / density matrix manipulations: von Neumann entropy derivations and water-filling solution in Appendix F.5 (equations around 44–49).

3. Calculating mutual information and von Neumann entropy for neural networks is challenging, but the paper does not adequately explain the methods used for their estimation in high-dimensional spaces.

4. The “von Neumann entropy” regularization uses S(C/tr(C)), but C may not be always positive semidefinite.

5. The approximation I(z_t;P_t) ≈ InfoNCE loss is questionable for temporal sequences; no justification for mutual information estimation validity.
6. The KL-based error bound (Proposition 2) assumes independence that may not hold for nonlinear latent dynamics.

### Questions
1. Can the authors formally prove that maximizing I(z_{t−n}; z_t) indeed selects eigenfunctions corresponding to Koopman modes with |λ|≈1? Or is this just a heuristic interpretation?

2. How is the von Neumann entropy computed robustly from noisy minibatch covariances? What happens if the covariance matrix is rank-deficient?

### Soundness
2

### Presentation
2

### Contribution
3
