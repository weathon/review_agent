# Policy transfer ensures fast learning for continuous-time LQR with entropy regularization

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Reinforcement Learning (RL) enables agents to learn optimal decision-making strategies through interaction with an environment, yet training from scratch on complex tasks can be highly inefficient. Transfer learning (TL), widely successful in large language models (LLMs), offers a promising direction for enhancing RL efficiency by leveraging pre-trained models. 

This paper investigates policy transfer, a TL approach that initializes learning in a target RL task using a policy from a related source task, in the context of continuous-time linear quadratic regulators (LQRs) with entropy regularization. We provide the first theoretical proof of policy transfer for continuous-time RL, proving that a policy optimal for one LQR serves as a near-optimal initialization for closely related LQRs, while preserving the original algorithm’s convergence rate. Furthermore, we introduce a novel policy learning algorithm for continuous-time LQRs that achieves global linear and local super-linear convergence. Our results demonstrate both theoretical guarantees and algorithmic benefits of transfer learning in continuous-time RL, addressing a gap in existing literature and extending prior work from discrete to continuous time settings. 

As a byproduct of our analysis, we derive the stability of a class of continuous-time score-based diffusion models via their connection with LQRs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents the first theoretical framework for policy transfer in continuous-time RL, focusing on entropy-regularized LQRs. The authors rigorously prove that an optimal policy for one LQR remains near-optimal for a closely related LQR, thereby ensuring efficient transfer and fast convergence.

### Strengths
This is the first formal proof of policy transferability in continuous-time RL, a problem significantly harder than its discrete counterpart. The results establish a solid foundation for understanding transfer efficiency under system perturbations. The proofs are rigorous. By generalizing the discrete-time IPO results to continuous-time, the paper fills a conceptual and technical gap between theory and control-oriented RL applications.

### Weaknesses
The paper is purely theoretical, lacking numerical simulations or empirical studies that could illustrate the benefits or limitations of policy transfer in practice.

Moreover, the notion of “closeness” between two LQRs is not concretely quantified. All results are stated in the general form of “as long as they are sufficiently close,” which limits the practical applicability of the theory for algorithm design.

### Questions
1. Ambiguity regarding the notion of "closeness":
    - In Theorem 1, it is unclear how large one should expect $\epsilon$ to be, especially under specific choices of the metric $d$?
    - In Theorem 5, similar clarification is needed on the expected magnitudes of $\epsilon$ and $C_2$?
2. It would be helpful if the authors could provide a simple numerical experiment to illustrate or verify the theoretical results.
3. Assumptions 4-5 are presented without any accompanying discussion. It would be valuable if the authors could elaborate on how restrictive these assumptions are in practice.
4. Throughout the proofs, Assumptions 1–5 are invoked but not explicitly referenced when applied. Clearer indication of where each assumption is used would improve the readability and rigor of the arguments.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
Summary: The paper studies policy transfer in reinforcement learning (RL) for continuous-time linear quadratic regulators (LQRs) with entropy regularization. The paper proves that the optimal policy of a source LQR task provides a near-optimal initialization for a target LQR with similar system parameters. The main contributions include: (i) a continuity result for the Riccati operator guaranteeing policy transferability between closely related LQRs; (ii) an Iterative Policy Optimization (IPO) algorithm with provable global linear and local super-linear convergence; and (iii) an application of the analysis to the stability of score-based diffusion models via the connection between LQRs and controlled SDEs.

### Strengths
1) The paper is well-written and well-organized. It provides an approach to connect entropy-regularized continuous-time LQRs with transfer learning. The proofs are sound and build upon solid stochastic control foundations (e.g., Riccati continuity, DPP, HJB).

2) The connection between LQRs and score-based diffusion models is interesting and well-motivated, illustrating broader implications of LQR analysis beyond control theory.

### Weaknesses
1) The paper frames policy transfer as a transfer learning problem but overlooks the extensive literature on meta-learning for control and meta-LQR. Examples: 
  
[1] Toso et al. "Meta-learning linear quadratic regulators: a policy gradient MAML approach for model-free LQR." 6th Annual Learning for Dynamics \& Control Conference. PMLR, 2024.

 [2] Aravind et al. "A Moreau envelope approach for LQR meta-policy estimation." 2024 IEEE 63rd Conference on Decision and Control (CDC). IEEE, 2024.

 [3] Muthirayan et al. "Meta-learning online control for linear dynamical systems." IEEE Transactions on Automatic Control (2025).

 [4] Richards et al. "Control-oriented meta-learning." The International Journal of Robotics Research (2023)

 [5] Zhang et al. "Multi-task imitation learning for linear dynamical systems." Learning for Dynamics and Control Conference. PMLR, 2023.

2) The aforementioned work and follow-up analyses on meta-adaptation and task similarity are highly relevant, as they provide theoretical and algorithmic frameworks for policy reuse across task distribution, precisely the problem studied here. Without comparing with respect to this line of work, the contribution risks appearing as an isolated instance of LQR transfer rather than a step toward data-efficient multi-task or meta-RL.

3) The IPO algorithm seems structurally identical to the discrete-time IPO in Guo et al. (2023), with the main novelty being a continuous-time adaptation. No significant methodological innovation or empirical verification seems to be introduced.

4) Although the work is primarily theoretical, even simple numerical examples (e.g., two LQRs with slightly perturbed dynamics) would enhance clarity regarding the policy transfer effect and convergence rates.

5) The exposition of transfer learning motivations is extensive and heavily references LLM/vision TL literature, but the discussion of control-specific challenges (e.g., transfer between dynamical systems, partial observability) is brief and lacks comparison to established control-theoretic notions of similarity or adaptation.

### Questions
1) How does the proposed framework compare to recent meta-LQR formulations, where task similarity is quantified via parameter-space or trajectory-space distances?

2) Could the authors formalize the notion of  "closely related" LQRs using information-theoretic or bisimulation-based measures, as often done in meta-control literature?

3) How might the entropy regularization interact with multi-task adaptation or robustness to heterogeneity across tasks?

Comments: 

1) The exposition is clear and mathematically precise, though the paper could better contextualize its novelty relative to multi-task and meta-RL control frameworks.

2) The link to diffusion models is interesting but somewhat tangential; emphasizing implications for meta-control or representation transfer might make the contribution more cohesive.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates policy transfer for continuous-time linear quadratic regulators (LQRs) with entropy regularization, extending prior analyses from discrete-time to continuous-time settings. The authors introduced an Iterative Policy Optimization (IPO) algorithm that achieves global linear and local super-linear convergence. As a byproduct of their analysis, they establish the stability of a class of continuous-time score-based diffusion models through their connection with LQRs.

### Strengths
1. The theoretical results are supported by clean mathematical derivations, leveraging properties of the Riccati equation and Gaussian policy structures.

2. The results provide a formal justification for policy reuse and initialization transfer between related LQR systems. 

3. The connection between LQRs and score-based diffusion models is interesting, yielding new stability results with potential implications for generative modeling.

### Weaknesses
1. The theoretical development appears to be a relatively direct extension of the prior work [1], without substantial methodological novelty. If the authors believe that extending the results to continuous settings is non-trivial, the paper should clearly articulate the specific challenges unique to the continuous-time setting and how their approach overcomes them.

2. The proposed IPO algorithm also appears to be a direct extension of the discrete-time version in [1]. The authors could strengthen the paper by clarifying what is fundamentally new in the continuous-time formulation, whether in algorithmic design, analysis, or convergence guarantees.


[1] Guo, Xin, Xinyu Li, and Renyuan Xu. "Fast policy learning for linear quadratic control with entropy regularization." arXiv preprint arXiv:2311.14168 (2023).

### Questions
1. It would strengthen the paper to discuss theoretical works on policy transfer in MDP settings, such as [2].

2. While the theoretical contributions are clear, the paper would benefit from empirical validation to illustrate the practical relevance of the results. Even simple numerical experiments on continuous-time LQRs could help demonstrate the effectiveness and convergence behavior of the proposed IPO algorithm and the benefits of policy transfer in practice.


[2] Fu, Haotian, et al. "Performance bounds for model and policy transfer in hidden-parameter mdps." The Eleventh International Conference on Learning Representations. 2023.

### Soundness
3

### Presentation
3

### Contribution
1
