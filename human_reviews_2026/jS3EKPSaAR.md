# PHYSICS-INFORMED RADIAL PHASE RETRIEVAL NEURAL NETWORK WITH HYBRID DEEP PRIORS AND DUAL PDE

- Decision: Reject
- Scores: 6, 2, 2, 4, 0

## Abstract
Phase retrieval from intensity-only measurements is severely ill-posed due to global-gauge and rotational symmetries. We consider outer-ring generalization: training with supervision from only a few inner rings and testing the model’s ability to reconstruct a broader set of unseen outer rings. We introduce a physics-informed hybrid network that combines (i) radial priors encoded by a smooth exponentiated spline and a \emph{monotone} outer-radius booster, (ii) two differentiable PDE branches---a Strang-split Kerr--NLSE pathway for high-frequency synthesis and a TIE-based low-pass pathway for coarse structure---and (iii) a strict radial projection enforcing output symmetry, together with a radius-dependent $\alpha$-fusion. Across the tested configurations, when trained only on a few rings (1-3), our model reconstructs more rings(4-9) than conventional methods, and achieves better stability in peak
positions and amplitude calibration under out-of-distribution settings. This provides some inspiration for enhancing the generalization of physics-informed neural networks when applied to optical inverse problems. Ablations isolate the contribution of the alpha fusion, PDE coupling, and monotone
boosting. We will release pseudo-code to facilitate reproducibility.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a physics-informed hybrid neural network for radial-phase retrieval from intensity-only measurements, addressing the challenging problem of outer-ring generalization. The authors combine three key components: (i) radial priors with smooth exponentiated splines and a monotone outer-radius booster, (ii) dual differentiable PDE branches (Kerr-NLSE for high-frequency synthesis and TIE for coarse structure), and (iii) strict radial projection with radius-dependent α-fusion. The method is trained on only 1-3 inner rings and tested on 4-9 outer rings. Experimental results demonstrate better reconstruction quality, better stability in peak positions and amplitude calibration, and improved generalization compared to baseline methods.

### Strengths
1) The paper combines multiple physics-informed components: (i) radial priors with monotone outer-radius boosting, (ii) dual PDE branches (Kerr-NLSE and TIE), and (iii) strict radial projection with radius-dependent α-fusion. This approach is original in addressing the outer-ring generalization challenge in phase retrieval.

2) The work demonstrates theoretical foundations, including Lipschitz stability analysis (Propositions 1-3), identifiability guarantees (Theorem 1), and local PL-type convergence (Theorem 2). The experimental design is thorough with comprehensive evaluations across multiple dimensions (E1-E7), including ablations, data efficiency studies, and robustness tests.

3) The paper makes contributions to physics-informed neural networks for inverse problems. The demonstrated ability to generalize from 1-3 rings to 4-9 rings addresses a fundamental challenge in computational imaging. The work has broad applicability beyond phase retrieval to other optical inverse problems, MRI, ultrasound, and PDE-based reconstruction tasks.

### Weaknesses
1) The paper relies on synthetic data with controlled Fraunhofer diffraction patterns. While robustness tests include noise and parameter mismatch, the absence of real experimental optical data raises concerns about domain transfer and practical applicability in actual settings.

2) The strict radial projection and entire framework assume effective radial symmetry. The paper acknowledges this limitation (Section 8) but does not provide an empirical evaluation of performance degradation under asymmetric conditions, such as astigmatism, ellipticity, or off-axis aberrations, which commonly occur in real optical systems.

3) The robustness experiments (E4) reveal that parameter mismatch in wavelength and focal length is the dominant failure mode, with precision dropping from 0.75 to 0.43 and recovered rings from 5.1 to 2.8. This sensitivity suggests the method may require precise system calibration, limiting practical deployment.

4) While the paper compares against U-Net and FNO, it does not evaluate against recent physics-informed phase retrieval methods or classical iterative approaches like Gerchberg-Saxton variants with regularization, making it difficult to assess the relative contribution of the specific architectural choices.

5) Although latency is reported (27.6ms vs 21.3ms for U-Net), the paper does not discuss training time, convergence speed, or the practical cost of the dual-PDE forward pass during inference at scale. The 30% increase in latency may be significant for real-time applications.

### Questions
1. Do the authors have plans to validate the method on real experimental optical data? What specific challenges do they anticipate in transitioning from synthetic to real data, and how might the method be adapted to handle domain shift?

2. *Can the authors provide quantitative results on how the method degrades under increasing levels of asymmetry (e.g., varying degrees of astigmatism)? At what point does the radial symmetry assumption break down?

3. Given the sensitivity to parameter mismatch, what calibration accuracy is needed for practical deployment? Could the method be extended to estimate or adapt to unknown system parameters jointly?

4. The local PL inequality (Theorem 2) requires specific assumptions. How often are these assumptions satisfied in practice, and what happens when they are violated?

5. Beyond the empirical generalization results, are there theoretical bounds on how many rings the method can extrapolate to, given training on N rings?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a physics-informed neural network for radial phase retrieval that combines optical priors with a manifold-based design. The model enforces strict radial symmetry by projecting inputs onto a quotient manifold, which removes global and angular phase ambiguities. It uses the nonlinear Schrödinger equation (NLSE) to model high-frequency wave propagation and the transport-of-intensity equation (TIE) to capture low-frequency behavior. These physics-based components are integrated within a learned neural framework. This resulted in more accurate and stable phase reconstructions than prior approaches.

### Strengths
* Physics-informed architecture: Integrates optical physics (NLSE and TIE) directly into the neural network design, grounding the model in physical principles rather than purely data-driven fitting.

* Compared to other approaches, such as U-Net and FNO, the proposed method achieves more accurate and stable phase reconstructions
* The method has a strong theoretical grounding, and this helps with the interpretability 

* The method performances has some generalizablity and appears to be robust to noise.

### Weaknesses
* The paper is difficult to follow. Main ideas and intuitions are not clearly stated. Several sections rely on dense mathematical notation (e.g., Hankel transforms, Lipschitz bounds, quotient manifolds) with limited intuitive explanation or guidance for the reader. The connection between the theoretical guarantees and the implemented network is unclear, and the paper quickly dives into architectural details (Figure 1) without first establishing sufficient conceptual context.

* The results are shown on synthetic images only. There is no validation on real experiments or physical hardware setups. Such experiments could help determine whether the strong symmetry assumption holds. The method relies on strict radial symmetry that may not be valid under optical aberrations or misalignment.

* How does the method compare against classical methods such as Gerchberg–Saxton or multi-plane TIE solvers? Only a few learned method results are shown. Are these the only methods available for comparison? I suggest the authors expand their comparison baseline to include well-established classical approaches.

* Despite the complexity of the proposed method, in terms of core reconstruction, the results are very close to those of a U-Net. I question if the added complexity is justified by the improvements.

* Some of the ablation results are unexpected, and the takeaways from each experiment are not clearly stated. For example, in Table 6, the performance on clean and noisy measurements is almost identical. The authors should discuss why this occurs and specify the noise level used in the experiments. In Table 5, I notice negative delta values, which indicate improvement when certain modules are removed. Does this mean that those components may not be beneficial? These points are not discussed or analyzed in the paper.

### Questions
1. Can the authors provide more intuition of the main ideas? The manuscript is difficult to follow, and it would help to see more discussion and diagrams of the manifold projection, the dual PDE branches, and the overall data flow.

2. How sensitive is the model to deviations from perfect radial symmetry? For example, would mild astigmatism in the input significantly degrade performance?

3. Are the generalization claims really valid? Simple extrapolation from 1-3 rings to 4-9 rings is not sufficient to claim generalizability.

### Soundness
3

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduce a physics-informed hybrid network that combines (i) radial priors encoded by a smooth exponentiated spline and a monotone outer-radius booster,(ii) two differentiable PDE branches—a Strang-split Kerr–NLSE pathway for high-frequency synthesis and a TIE-based low-pass pathway for coarse structure—and (iii) a strict radial projection enforcing output symmetry, together with a radius-dependent $\alpha$-fusion.

### Strengths
- The authors introduce a physics-informed method to solve phase retrieval problems and split into high-frequency and low-pass pathways.  
-

### Weaknesses
The writing is very poor and there are some abbreviations which may not give the full expressions at the first time.

### Questions
- This paper mainly discussion phase retrieval problems and solve by two pde equations. So why you can split the phase retrieval problems into these two and how?
- In Section 3, the authors introduce some theoretical definitions, assumptions and lemmas, what is the relationship between you introduce and phase retrieval?
- Line 100, NLSE, is it nonlinear schrodinger equations? Also Line 142, ``PL'' and Line 177, ``atan2''?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper deals with radial phase retrieval from intensity-only measurements, a highly ill-posed inverse problem, under the conditions of an outer ring generalization. A physics-informed hybrid network is proposed that (i) embeds radial priors via a smooth exponential spline and a monotonic outer radius booster, (ii) couples two differentiable PDE branches, and (iii) enforces a strict radial projection with a radius-dependent $\alpha$-fusion and output symmetry. Experimental results using synthetic data show that the method can recover more outer rings and reduce hallucinations compared to baselines.

### Strengths
- The designed network adapts the inductive bias with the ring-structured phase retrieval problem, rather than relying on generic CNN priors. This is a thoughtful specialization that many “physic-informed” works propose but do not actually implement.
-  The task “training on inner rings, testing on invisible outer rings” as an explicit setting for distribution shift is an original evaluation method for radial phase retrieval and could serve as a basis for subsequent benchmarks. Accurate recontruction of the outer ring is crucial for downstream optical tasks. The method that reduces hallucinations, improves ring fidelity, and remains stable at the same time is of great value to laboratories and systems that cannot afford dense multi-surface measurements. 
- The paper clearly separates the roles of (i) radial projection, (ii) outer radius boosting, (iii) NLSE vs. TIE paths, and (iv) $\alpha(\rho)$ fusion. This makes the method easier to understand and implement.

### Weaknesses
- The experiment is conducted using synthetic data with limited distribution shifts. For the proposed method, which is to be generalized to invisible outer rings and be physics-informed, the empirical scope of application is too narrow.
- For the comparison, the baselines are U-Net/FNO variants. Please compare the proposed method with physics-guided unrolled methods and plug-and-play / regularization by denoising (RED) algorithms [1, 2, 3]
- Stability, identifiability, and convergence sketches are based on assumptions such as Lipschitzness, radial Hankel linearization, and noise models, which may not apply under realistic optical conditions. The theory presented is promising, but has not yet been sufficiently investigated for practical application.
- The current "6. Experiments" and "7.Overall Experimental Analysis" sections are difficult to interpret. They offers only a limited interpretation of what the individual key figures reflect in terms of specific error modes, and contains figures/tables whose captions lack essential experimental details.

[1] Ulyanov, Dmitry, Andrea Vedaldi, and Victor Lempitsky. "Deep image prior." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.

[2] Metzler, Christopher, et al. "prDeep: Robust phase retrieval with a flexible deep network." International Conference on Machine Learning. PMLR, 2018.

[3] Mardani, Morteza, et al. "A Variational Perspective on Solving Inverse Problems with Diffusion Models." The Twelfth International Conference on Learning Representations.

### Questions
- Since the phase retrieval is highly ill-posed problem, how is phase ambiguity handled, and what happens if the actual field deviates slightly from perfect radial symmetry?
- What are the stability regions for NLSE (nonlinearity/dispersion) and TIE step size?
- Can the hybrid model signal outer rings with low reliability to avoid hallucinations?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 5

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
In this work, the authors address problem of phase retrieval from intensity-only measurements and explores outer-ring generalization, where a model trained on inner rings is evaluated on unseen outer rings. The authors propose a hybrid physics-informed network that integrates several components: (i) radial priors using spline-based and monotone boosting mechanisms, (ii) two PDE-based branches (Kerr-NLSE and TIE), and (iii) a radial projection with a radius-dependent α-fusion. The method is claimed to outperform baselines on synthetic optical datasets and to provide better generalization to unseen spatial frequencies.

### Strengths
1. Improving generalization of physics-informed neural networks in optical inverse problems is interesting and relevant.
2. The attempt to connect PDE-based modeling and learned priors is conceptually valuable.

### Weaknesses
The paper is not written well enough to be accepted and requires a lot more work to be in a readable format.

1. The paper assumes a high level of prior knowledge about the phase retrieval problem, but never explicitly defines it. For most readers at a general machine learning venue (such as ICLR), the problem setup (what is measured, what is reconstructed, and why it is ill-posed) needs to be clearly introduced. In its current form, the paper is not understandable. For example: (a) the writing is overly dense and often boils down to unexplained jargon (e.g., “monotone curvature reduces ring,” “Strang-split Kerr-NLSE pathway”) with undefined acronyms (the authors mention SLM, PINNs and much more but never define it!), which significantly hinders readability. (b)  Several sections read like lists of technical keywords rather than structured explanations, e.g. the “Design rules” paragraph is essentially a bullet list of concepts without narrative or connection.
2. Proposition 1 is trivial and adds little value. If it is meant to motivate the model’s architecture, the reasoning should be expanded or omitted.
3. The paper presents multiple architectural and mathematical components (radial priors, PDE branches, $\alpha$-fusion), but their interrelations and necessity are not clearly motivated. It is unclear why these components are combined or how they interact theoretically.
4. Despite being an imaging paper, no reconstructed images are shown, which severely limits interpretability. Visual examples are crucial to assess the claimed improvements in ring generalization, amplitude stability, or spatial structure.
5. Although the abstract mentions that pseudo-code will be released, the current version lacks sufficient experimental detail to reproduce the results. Precise dataset specifications, training protocols, and implementation details (e.g., optimizer, learning rate, loss definition) should be included. The dataset description is vague: “synthetic Fraunhofer (centered FFT, energy-normalized)” is not sufficient to reproduce the setup. Details such as the aperture geometry, sampling conditions, number of rings, and noise model must be specified.
6. Sections 6.4–6.7 are particularly difficult to parse; results are presented without context or interpretation.

### Questions
Given how difficult to read is the paper, here are some suggestions for improvement:

1. Begin with a clear and intuitive definition of the phase retrieval problem, including its mathematical formulation and physical motivation. Provide images if possible.
2. Greatly simplify and clarify the exposition. Replace jargon-heavy descriptions with explanations of what each component contributes conceptually.
3. Provide qualitative results (images) alongside quantitative metrics.
4. Clarify dataset generation and all experimental settings.
5. Remove trivial statements or overly technical statements (e.g., Proposition 1) or make them meaningful by linking them to design insights.
6. Revise the writing tone to be more informative and guide the reader through your reasoning. At the moment it seems “technical for the sake of sounding technical.”

### Soundness
1

### Presentation
1

### Contribution
1
