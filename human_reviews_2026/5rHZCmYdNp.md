# SESaMo: Symmetry-Enforcing Stochastic Modulation for Normalizing Flows

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 2

## Abstract
Deep generative models have recently garnered significant attention across various fields, from physics to chemistry, where sampling from unnormalized Boltzmann-like distributions represents a fundamental challenge. In particular, autoregressive models and normalizing flows have become prominent due to their appealing ability to yield closed-form probability densities. Moreover, it is well-established that incorporating prior knowledge—such as symmetries—into deep neural networks can substantially improve training performances. In this context, recent advances have focused on developing symmetry-equivariant generative models, achieving remarkable results.
Building upon these foundations, this paper introduces Symmetry-Enforcing Stochastic Modulation (SESaMo). Similar to equivariant normalizing flows, SESaMo enables the incorporation of inductive biases (e.g., symmetries) into normalizing flows through a novel technique called \textit{stochastic modulation}. This approach enhances the flexibility of the generative model by enforcing exact symmetries while, for the first time, enabling the model to learn broken symmetries during training.
Our numerical experiments benchmark SESaMo in different scenarios, including an 8-Gaussian mixture model and physically relevant field theories, such as the $\phi^4$ theory and the Hubbard model.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces a new way to incorporate symmetries into discrete normalizing flows, especially suited for symmetry breaking. In this scheme the NF always maps to the same symmetry sector (canonical mode or cell). Samples are then possibly transformed to one of the other symmetry sectors (picked stochastically according to learned probability). To allow gradient flow to the parametrization of this “stochastic modulation”, the authors introduce the “self-reparametrized KL divergence” (bijectivity penalty; REINFORCE estimator to differentiate through the random variable). Since the learned probabilities can differ between the modes, the method allows for symmetry breaking, i.e. different mass per mode. The authors test on a Gaussian mixture with exact and broken Z8 symmetry, 2D complex $\phi^4$ scalar field with U(1) symmetry, and Hubbard model with Z4 symmetry.

### Strengths
- The idea is technically sound and novel
- The paper shows clear benefits in terms of effective sample size (ESS) and lower KL on the tasks considered, especially for symmetry breaking

### Weaknesses
- The proposed stochastic modulation seems to only work for discrete symmetries
- The symmetry sectors (and the corresponding transformations from the canonical cell) must be known a priori

### Questions
- Appendix A and B (especially figure 5) are quite crucial to understanding the method. In my opinion they should be in the main text, possibly at the cost of moving parts of the background into the appendix
- How does accuracy/training complexity scale with dimension and number of symmetry sectors?
- Can stochastic modulation be used with continuous NF (like flow matching)?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces SESaMo (Symmetry-Enforcing Stochastic Modulation), a framework that integrates group symmetry constraints into normalizing flows via a stochastic modulation mechanism. Specifically, the method augments a base flow g_\theta with a family of symmetry transformations S_u, where u is sampled from a learnable distribution p_{S,b}(u). This allows the model to both enforce exact symmetries and capture symmetry breaking. The approach is evaluated on several discrete-symmetry benchmarks, showing improved sampling performance and better mode coverage compared to RealNVP, canonicalization-based flows, and VMoNF.

### Strengths
The paper proposes a creative mechanism to integrate group symmetries into flow-based generative models using stochastic modulation. This differs from previous deterministic equivariant designs (e.g., Equivariant Flows) by introducing a learnable mixture structure that explicitly models symmetry breaking. Experiments demonstrate that SESaMo achieves competitive or superior ESS on discrete-symmetry datasets while maintaining interpretability of the learned symmetry-breaking parameter b.

### Weaknesses
1. The bijectivity penalty used to enforce separability is heuristic and not theoretically justified. The experimental comparisons only include RealNVP, VMoNF, and canonicalization methods. These are several years old and do not represent the current state of equivariant or symmetry-aware generative modeling. Modern approaches, including equivariant flow mathcing and equivariant diffusion models, should at least be discussed or justified as not directly applicable. Without this clarification, it is unclear whether SESaMo’s advantage is due to the stochastic modulation itself or simply the choice of older baselines.

2. The method assumes that each symmetry sector maps to a distinct, non-overlapping region of configuration space, which ensures bijectivity but restricts applicability to simple discrete symmetries.

3. The bijectivity penalty used to enforce separability is heuristic and not theoretically or experimentally justified.

4. The experiments focus on low-dimensional or lattice-based toy systems. There is no demonstration on realistic and high-dimensional systems, where the claimed symmetry advantages would be most meaningful.

### Questions
1. Could the authors clarify whether recent equivariant generative models cannot be directly applied to the symmetry-enforcing scenarios considered here? If so, please explain the technical limitations preventing their use. Otherwise, their inclusion as baselines would significantly strengthen the experimental evidence.

2. Could the authors discuss the potential of extending SESaMo to more complex or continuous symmetry groups?

3. The bijectivity penalty appears heuristic. How sensitive is the training process to the choice of its parameters (A and B)? Have the authors tried alternative formulations?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces SESaMo (Symmetry-Enforcing Stochastic Modulation), a simple and general framework for incorporating both exact and broken symmetries into normalizing flows (NFs). The method augments a standard flow with a symmetry transformation whose selection probabilities are learnable, allowing the model to capture multiple symmetric modes and symmetry-breaking effects without having to enforce symmetries as hard constraints within the architecture. Experiments on Gaussian mixtures and lattice field theory benchmarks ($\phi^4$ and Hubbard models) demonstrate improved sample efficiency and effective sample size compared to normalizing flow baselines, including with canonicalization.

### Strengths
* The proposed method is a simple and flexible way to impose both exact and broken symmetries on NFs without requiring explicitly equivariant architectures or canonicalization.
* The work is technically sound and supported with solid empirical validation.
* Furthermore, the proposed method integrates naturally with standard NFs (e.g. RealNVP) and requires relatively minimal overhead (a REINFORCE estimator to enable gradients through a stochastic variable).

### Weaknesses
* While the proposed method is elegant in principle, the exposition is overly abstract which makes the main idea underlying the method difficult to understand. For example, the authors provide several intuitive examples in increasing generality, but relegate them to the appendix (E, F). The paper could benefit from a clear algorithm box/pseudocode implementation describing the key components (flow, modulation map, and probability weights) and, in increasing generality, how each can be set or augmented with learned components. I believe this would make the work much more accessible.
* The paper presents the method and empirical results, but could benefit from a formal analysis of conditions for convergence, expressivity, or relations to existing equivariant NFs. For example, as far as I understand, the REINFORCE estimator and stochastic variable introduced for the modulation map is likely to introduce additional variance, but it is unclear to what extent this affects the training stability or the consistency of the results (see questions.) Additional experiments that quantify the variability of the results with SESaMo may further strengthen the paper.

### Questions
* If I understand correctly, the stochastic variable + REINFORCE estimator used in the modulation map is likely to increase variance. Furthermore, since the approach still minimizes reverse KL, the approach remains mode-seeking and it seems possible (with poor initialization or high-variance training) that a mode may be entirely missed. Did the authors observe this potential issue?
* Did the authors evaluate the method on datasets beyond toy or physical models where symmetry-enforcement is important, e.g. rotation/reflections with natural images?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents a sampling method that handles symmetries in unnormalized distributions in a principled way. It maps the random tensor of interest $x$ to a reduced space $\tilde x$. It learns a Normalizing Flow model to sample $\tilde x$ and then map it, along with a latent variable $u$, back to $x$.

Experiments are conducted on several distributions to validate the effectiveness of the proposed method.

### Strengths
- The paper introduces a systematic way to incorporate symmetries in NF.
- Experimental testbed is good. The complex $\phi^4$ and Hubbard models are known to be challenging distributions.
- The idea of using a penalty term to enforce bijectivity is interesting.
- The paper is well written and is mostly comprehensible.

### Weaknesses
I think several essential questions/issues are overlooked both in theory and experiments.
- Experimental results report only ESS and RKL (reverse KL divergence). However, the performance on two metrics can be good (i.e., high ESS and low RKL) even in the case of mode collapse (see [AdvNF 2025]), i.e., when the model learns and explores only a part of the target distribution. A standard metric robust to this problem is the negative log-likelihood (NLL).
- The baselines are weak. The state of the art baselines such as [FAB 2023] and [IDEM 2024] should be used.
- Are all $S_{T,u}$ independently trained for each $u$? If no, this could be a problem because the symmetry breaking bias may cause a probability mass drift even within a mode (of symmetry). If yes, the number of parameters involved will shoot up, and this should be discussed in the paper (comparing model sizes and training times of different methods/models). Or is the model conditioned on $u$ (i.e., $u$ is an input)?

In the core, the proposed approach factorizes $q(x)$ as $\sum_u q(x|u)p(u)$, where $u$ is a random variable corresponding to a symmetry mode. Theoretically, the contribution seems modest. More rigorous experiments could make the paper more appealing.

The presentation could be simplified to highlight the core contribution. There are too many variables and equations, and many are superfluous.


[AdvNF 2024] AdvNF: Reducing mode collapse in conditional normalising flows using adversarial learning, SciPost Phy. 2024
[FAB 2023] FLOW ANNEALED IMPORTANCE SAMPLING BOOTSTRAP, ICLR 2023
[IDEM 2024] Iterated Denoising Energy Matching for Sampling from Boltzmann Densities, NeurIPS 2024

### Questions
- Please answer my concerns raised in the Weaknesses section.
- It seems the paper considers only discrete symmetries as $u$ is discrete, and not continuous ones. But the experiments are performed for U(1) symmetry too. What is $u$ in that case?
- Although bijectivity has been enforced by a penalty term, its effectiveness has not been assessed. Experiments could be added to address this.
- Why do we need to learn $C_{T,z}$? Can't it be a heuristic map, or can't we sample $\tilde z$ directly? I would appreciate it if another example(s) could help me understand better.

- Minor typos:
  - Line 228: It think it should be $g_\theta(z)$ instead of just $g_\theta$.
  - Line 283: a -> an

### Soundness
2

### Presentation
3

### Contribution
2
