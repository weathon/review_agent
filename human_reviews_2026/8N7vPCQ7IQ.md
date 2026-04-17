# Convergence Theory of Decentralized Diffusion Models via Pseudo-Non-Markov Analysis

- Decision: Reject
- Scores: 6, 8, 2, 6

## Abstract
Diffusion probabilistic models (DPMs) have demonstrated remarkable success in generative tasks, supported by a solid foundation of convergence analysis. 
  Recently, decentralized DPMs have been proposed to enhance data security and enable cross-institutional collaboration.
  However, their unique decentralized structure renders existing analysis techniques inapplicable, leaving their theoretical convergence properties an open question.
  In this paper, we introduce a novel pseudo-non-Markovian method to analyze the convergence of both standard and decentralized DPMs within the context of the denoising diffusion probabilistic model (DDPM) sampler.
  Our key technical insight is to reframe the analysis of the backward transition. While the transition from $x_t$ to $x_s$ ($s<t$) is Markovian, we analyze its conditional form given the initial data $x_0$. This conditional transition becomes non-Markovian but gains a tractable analytical expression, allowing for a direct analysis of the discretization error on the Cartesian product space of $x_t\times x_s\times x_0$.
 We show that this method is readily extensible to the decentralized setting. To the best of our knowledge, our convergence theory represents the first of its kind applicable to the decentralized scenario.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper develops the first convergence analysis for decentralized denoising diffusion probabilistic models (DDPMs).
The key idea is a pseudo-non-Markovian framework: instead of analyzing the standard Markovian backward diffusion $x_t \to x_{t-1}$, the authors condition on the initial data $x_0$, making the chain non-Markov but analytically tractable.
This yields explicit discretization-error bounds on the product space $\mathcal{X}^{N}$ of all nodes, which naturally extend to a decentralized setup with consensus averaging.
The paper provides theorems establishing convergence under Lipschitz score and bounded communication noise assumptions, plus illustrative toy simulations showing scaling with network size and diffusion step.

### Strengths
- Novel theoretical framework: conditionalizing on $x_0$ to obtain a pseudo-non-Markov formulation is elegant and non-obvious.
- First decentralized convergence result: extends DDPM theory to multi-node, data-partitioned settings.
- Mathematical clarity: proofs outline how discretization and communication errors decompose additively.
- Potential generality: approach likely transferable to federated score matching or diffusion-based privacy mechanisms.
- Sound motivation: addresses both privacy (no raw-data sharing) and analytical tractability gaps.

### Weaknesses
- Assumption strength: bounded spectral gap and synchronous communication may be unrealistic; need discussion of asynchronous or lossy channels.
- Empirical validation minimal: toy 2-D examples only; no larger-scale decentralized experiments verifying theoretical rates.
- Tightness of bounds unclear: constants in main theorem not compared to centralized DDPM rates—difficult to assess practical relevance.
- Notation density: Section 3 introduces multiple stochastic kernels without clear hierarchy; risk of confusing readers unfamiliar with diffusion theory.
- Limited discussion of score-estimation error: assumes near-oracle scores; no analysis of training noise or stochastic gradients.

### Questions
- Error metric: Are convergence bounds in total variation, $W_2$, or KL? If $W_2$, what is the dependency on dimensionality d?
- Score approximation: How would estimation error $\|\hat s_\theta - s^\*\|$ propagate in the pseudo-non-Markov framework?
- Communication topology: Is the rate affected by the graph spectral gap $\lambda_2(L)$? Could results hold under time-varying or directed graphs?
- Decentralized bias: Does the product-space formulation assume perfect consensus each step, or is there a residual bias term $\mathcal{O}(\eta/\lambda_2)$?
- Relation to existing DDPM analyses: How does your bound compare to Nichol & Dhariwal (2021) or De Bortoli et al. (2022) in the centralized case?
- Extension to DDIM / score-based SDEs: Can the pseudo-non-Markov conditioning handle deterministic samplers or continuous-time diffusion limits?
- Empirical confirmation: Do small-scale experiments confirm scaling predicted by your rate (e.g., $\mathcal{O}(1/T)$ vs. $\mathcal{O}(1/\sqrt{N})$)?
- Practical implication: How large can communication noise be before it dominates discretization error?
- Assumption necessity: Is global Lipschitzness required, or could a local dissipativity condition suffice?
- Broader scope: Could this framework analyze federated generative training (e.g., decentralized score matching) rather than inference?
- Minor Typo: At line 2028: "As per Theorem 5.3 in ?"

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
4

### Summary
The paper addresses a timely and important gap: the lack of convergence guarantees for decentralized diffusion models (DPMs). The proposed pseudo-non-Markov analysis is conceptually clean, avoids heavy assumptions (e.g., log-Sobolev, Lipschitz gradients), and yields convergence rates comparable to state-of-the-art for standard DPMs. Crucially, it is the first work to provide a theoretical convergence guarantee for decentralized DPMs, which are increasingly used in privacy-sensitive and federated settings. Experiments on synthetic and real-world data (CIFAR-10, CelebA-HQ) validate the practical efficacy of the proposed dynamic sampling scheme. While not a breakthrough-level contribution, it is a solid, well-executed theoretical advance with clear practical implications

### Strengths
1. First convergence theory for decentralized DPMs, a practically important but theoretically ungrounded class of models.
2. A new method proposed (pseudo-non-Markov analysis) that simplifies convergence proofs and weakens assumptions compared to SDE-based methods.
3. Identification and mitigation of classifier error accumulation in decentralized sampling—a practical insight with theoretical backing.
4. Empirical validation showing dynamic domain blending improves generation quality over baselines.

### Weaknesses
1. The classifier approximation error accumulates linearly with the number of timesteps, which can dominate the total error for fine discretizations. While the authors propose a high-order training fix, no experiments validate its effectiveness.
2. Limited algorithmic novelty: The sampling and training algorithms (Algorithms 1–4) are straightforward extensions of existing decentralized/MoG-DPM approaches. The core contribution is theoretical, not methodological.
3. Training the classifier need access the whole set of data (noised in most cases, but this term breaks the confidentiality concept)

### Questions
Question on the classifier lower bound and practical training):  
Assumption 5.1 requires that both the true domain weights and their approximations are uniformly lower-bounded by a constant C_a > 0. However, in practice—especially with well-separated domains—the true posterior can become arbitrarily close to 0 or 1. How sensitive is the convergence bound in Theorem 5.2 to violations of this assumption? Have you observed training instability or degradation in sample quality when the classifier becomes overconfident?

Question on the relationship between discretization and classifier error:
Theorem 5.2 shows that the classifier error grows linearly with T, whereas discretization error decreases. Does this imply an optimal number of timesteps that balances these two competing terms?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors develop a new convergence analysis for diffusion models which they term the pseudo-non-Markovian method. Rather than directly consider the distribution $p(x_s|x_t)$ at each step, they condition on the initial point $x_0=y$, consider $p(x_s|x_t,y)$, and integrate over $y$. They use this to give a new analysis for DDPM which recovers the known bound in TV distance to a early-stopped distribution (up to poly-logarithmic factors) with discretization error $hd$ (where $h$ is step size). They extend the analysis to decentralized diffusion models, giving the first analysis of these models.

Decentralized diffusion models work by training a diffusion model for each class $l$ and a classifier at each noise level; a backward step consists of sampling from the mixture distribution of $p^l(x_s|x_t)$ with the classifier probabilities.

The authors expose a weakness of the standard decentralized diffusion models, which is the accumulation of error of the classification error as the number of steps, rather than the amount of time, and use this to suggest learning the derivative of $a$ as well.

### Strengths
The authors give a new and simple framework for analyzing diffusion models which does not require SDE theory, which recovers the known bound in TV distance (up to poly-logarithmic factors) to a early-stopped distribution. They give the first error analysis for decentralized diffusion models.

The observation of error accumulation from classification error is insightful, and the suggested algorithmic modification is promising.

### Weaknesses
It's not clear to me that the decentralized diffusion models necessitate a new framework for analysis. One could try to apply existing analysis for the error accrued during 1 step for the backwards diffusion for each class, and then use this to derive an error for the mixture distribution. If so, this would weakens the paper's contributions, as the paper currently suggests that the pseudo-non-Markov analysis is essential to analyzing decentralized diffusion models. 

Assumption 5.1 requires a uniform lower bound on the classification probabilities $a$; however, when t is close to 0, it is reasonable for one class to have probability close to 1 and the others to have probability close to 0.

The main theorems only give the TV distance to the slightly noised distribution $p_{t_{\min}}$ with constant step size, though it is known that a variable step size schedule gives better bounds in general cases (e.g. without smoothness assumptions on $p_0$, decreasing step sizes as $t\to 0$).

Some of the proofs are given purely as a sequence of equalities/inequalities, and can benefit from more exposition to guide the reader.

The forward error inequality (32) is incorrect. Convergence in KL divergence cannot give a bound in terms of the initial TV error. It is possible to use the 2nd moment assumption to first obtain some regularization, but that is a separate argument.

I like the idea with higher-order training for decentralized diffusion models, although this currently seems like an afterthought to the paper. Exploring this more centrally would improve the contribution of the paper.

### Questions
1. Is the pseudo-non-Markov analysis really necessary? Would the above sketched analysis work? If not, why not?
2. Where does the uniform lower bound on the classification probabilities $a$ appear in the proof? Is this necessary?

Minor:
* p. 3: "sota" -> "SOTA"
* Assumption 3.1 states "first moment" but (12) shows a second moment. The "moment" in Remark 3.2 is unspecified.
* p. 6 "the pseudo-non-Markov" -> pseudo-non-Markov analysis"
* p. 8: Missing period in 4th sentence
* p. 8, Theorem 5.2: Extraneous (x_s)
* p. 9: "way" in Sectino 4.1 -> method
* p. 9 "Considering the... we can whole-cluster..." - Rewrite this sentence.
* p. 9, Theorem 5.2: $T$ is not defined.
* p. 9: "formation" -> "formulation"
* p. 9: missing period in next-to-last sentence.

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
2

### Summary
This paper presents a pseudo-non-Markovian analysis framework for studying the convergence of both standard and decentralized diffusion probabilistic models (DPMs). To analyze the discretization error, it proposes to decompose the joint Cartesian space $(x_t,x_s, y)$ into different parts and analyze each part separately. This technique can lead to a linear $d$ dependence in the final results. In combination with the analysis of approximation error, it can be generalized to the setting for decentralized diffusion models.

### Strengths
(1) The paper is well written, which presents the proof pipeline very clearly.

(2) The theoretical results are solid, showing superiority over previous works.

(3) The generalization to decentralized diffusion models is very natural.

### Weaknesses
(1) The introduction of decentralized DPM is not very clear. It only shows the definition comes from Dong et al.2024, without a brief introduction on why it is defined.

(2) The description of the cluster partition is not very clear. What is the partition data distribution? For example, if each data is drawn I.I.d., the data distribution should be exactly the same as the total distribution (with correct normalization).

(3) Line 385: What is the $L(x_0)$ here?

### Questions
I do not check every detail of the proof, and I do not hold questions regarding other parts of the paper.

### Soundness
3

### Presentation
3

### Contribution
3
