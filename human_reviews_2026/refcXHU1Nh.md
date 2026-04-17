# SafeFlowMatcher: Safe and Fast Planning using Flow Matching with Control Barrier Functions

- Decision: Accept (Poster)
- Scores: 6, 4, 8, 4, 6

## Abstract
Generative planners based on flow matching (FM) produce high-quality paths in a single or a few ODE steps, but their sampling dynamics offer no formal safety guarantees and can yield incomplete paths near constraints. We present SafeFlowMatcher, a planning framework that couples FM with control barrier functions (CBFs) to achieve both real-time efficiency and certified safety. SafeFlowMatcher uses a two-phase prediction-correction (PC) integrator: (i) a prediction phase integrates the learned FM once (or a few steps) to obtain a candidate path without intervention; (ii) a correction phase refines this path with a vanishing time‑scaled vector field and a CBF-based quadratic program that minimally perturbs the vector field. We prove a barrier certificate for the resulting flow system, establishing forward invariance of a robust safe set and finite-time convergence to the safe set. In addition, by enforcing safety only on the executed path—rather than all intermediate latent paths—SafeFlowMatcher avoids distributional drift and mitigates local trap problems. Moreover, SafeFlowMatcher attains faster, smoother, and safer paths than diffusion- and FM-based baselines on maze navigation, locomotion, and robot manipulation tasks. Extensive ablations corroborate the contributions of the PC integrator and the barrier certificate.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper “Practical Diffusion Planning via Temperature-Guided Reward Conditioning (TGDP)” introduces a diffusion-based planning method that removes the need for task-specific tuning in reward-conditioned decision-making. Building on Classifier-Free Guidance (CFG), the authors propose a temperature-guided framework where trajectory samples are reweighted during training according to a temperature parameter that biases learning toward higher-return trajectories while preserving data diversity. At inference, TGDP employs an adaptive guidance mechanism that dynamically adjusts the influence of reward conditioning based on the geometric alignment of denoising directions across temperature scales, thereby preventing under- or overguidance. Experiments on standard D4RL locomotion, Maze2D, and Kitchen benchmarks show that TGDP achieves performance on par with or exceeding existing diffusion planners, all while using a single default hyperparameter configuration and maintaining computational efficiency.

### Strengths
- The paper is well motivated and clearly written, providing a logical flow from problem statement to solution.

- The proposed TGDP method is technically sound and conceptually reasonable, effectively addressing CFG’s sensitivity to guide-scale tuning.

- The method design is elegant and practical, integrating temperature-guided weighting and adaptive guidance without significant computational or architectural overhead.

- Experiments are comprehensive and well executed, covering multiple standard benchmarks (D4RL, Maze2D, Kitchen) and demonstrating consistent performance gains.

- The analysis is insightful, linking theoretical intuition with empirical behavior and validating the method’s robustness and generality.

### Weaknesses
- The novelty is limited, as the contribution primarily targets a known issue (guide-scale sensitivity) within Classifier-Free Guidance rather than introducing a fundamentally new planning paradigm.

- The method introduces additional parameters, such as βₘₐₓ, which slightly increase algorithmic complexity and may require their own tuning for optimal stability.

- The experimental evaluation relies mainly on D4RL, an older and largely saturated benchmark suite; validation on more realistic or modern robotic planning tasks (e.g., navigation or manipulation with sensory inputs) would strengthen the paper’s practical relevance.

- While the adaptive guidance mechanism is elegant, it remains heuristic—its theoretical guarantees or convergence behavior under diverse diffusion architectures are not thoroughly analyzed.

- The temperature-based reweighting is conceptually intuitive but still handcrafted; there is no exploration of whether β or weighting could be learned adaptively or optimized end-to-end.

- The paper’s computational trade-offs (≈1.5× CFG cost) are reasonable but may still pose challenges for real-time or large-scale deployment scenarios.

### Questions
- How sensitive is TGDP to the choice of the **βₘₐₓ** parameter? Could the authors provide guidance or heuristics for selecting it, or explore whether it can be learned automatically?  
- Since TGDP mainly addresses **guide-scale sensitivity** in CFG, how do the authors envision extending this framework to **other conditioning paradigms** or more general decision-making problems?  
- The adaptive guidance mechanism is based on **collinearity heuristics** — can the authors provide deeper theoretical justification or empirical evidence (e.g., ablation or visualization) showing that this metric consistently distinguishes inter- and intra-modal cases?  
- Have the authors evaluated TGDP on **modern or realistic robotic benchmarks**, such as continuous manipulation or navigation tasks with raw sensory input? If not, what challenges do they foresee in transferring TGDP to such settings?  
- The method adds some computational overhead (~1.5× CFG). Could the authors analyze how this scales with **planning horizon** and **environment complexity**, and whether approximate or truncated versions can retain performance?  
- Can the **temperature-based reweighting** framework be extended to **learned or adaptive temperature schedules**, potentially improving flexibility across nonstationary datasets?  
- How does TGDP interact with **online fine-tuning or real-time replanning**, where data distributions evolve — does the adaptive mechanism remain stable in such dynamic conditions?

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
This paper introduces SafeFlowMatcher, a novel planning framework that integrates Flow Matching (FM) with Control Barrier Functions (CBFs) to achieve both real-time efficiency and certified safety in robotic path planning. The core of SafeFlowMatcher is a two-phase Prediction-Correction (PC) integrator. The prediction phase efficiently generates a candidate path using learned FM dynamics without safety interventions. The subsequent correction phase refines this path by (i) reducing integration errors through vanishing time-scaled flow dynamics (VTFD) and (ii) enforcing hard safety constraints using a CBF-based quadratic program (QP), minimally perturbing the vector field. This decoupling strategy prevents distributional drift and mitigates "local trap" problems often encountered by other certification-based generative planners. Extensive experiments on Maze2D navigation and high-dimensional locomotion tasks demonstrate that SafeFlowMatcher outperforms existing diffusion- and FM-based baselines in terms of safety (zero trap rate), planning performance (higher scores), path quality (smoother paths), and computational efficiency. Ablation studies confirm the necessity of both prediction and correction phases and the effectiveness of VTFD.

### Strengths
1. The paper proposes a unique combination of Flow Matching for efficient path generation and Control Barrier Functions for formal safety guarantees. This addresses a critical need in robotic planning for both speed and reliability. The two-phase PC integrator is a significant contribution. By separating the initial path generation (prediction) from safety enforcement (correction), SafeFlowMatcher avoids issues like distributional drift and local traps that plague other methods that attempt to enforce safety on intermediate latent states.
2.  The paper provides rigorous mathematical proofs for the forward invariance of a robust safe set and finite-time convergence to this set. This theoretical backing strengthens the credibility of the proposed safety guarantees.
3. By leveraging Flow Matching, SafeFlowMatcher demonstrates efficiency comparable to or better than diffusion-based methods, which often require many more sampling steps. The ability to achieve high performance with a small number of function evaluations (e.g., $T^p=1$) is a notable advantage.

### Weaknesses
1.  The paper highlights the reliance on well-defined CBFs. Could the authors elaborate on the practical challenges and potential solutions for designing or learning CBFs for more complex, dynamic, or unstructured real-world robotic environments beyond the relatively simple constraints presented? Are there any ongoing efforts to integrate CBF learning within SafeFlowMatcher?
2. Given that hyperparameter tuning is identified as future work, can the authors provide more insight into the sensitivity of SafeFlowMatcher's performance to the choice of $\epsilon$, $\rho$, $\delta$, $\alpha$, and $t_w$? Are there any heuristics or adaptive strategies that could be employed to simplify this tuning process for new tasks or environments?
3. While closed-form QP solutions are mentioned for simple cases, how does the computational overhead of the QP solver scale with a larger number of waypoints ($H$) or a more extensive set of complex safety constraints? Have the authors encountered scenarios where the QP solving time becomes a bottleneck, and what strategies could mitigate this?
4. The paper mentions developing a guidance-free version as future work. Could the authors expand on the specific challenges in achieving this (e.g., in terms of exploration, multimodal goal reaching) and their initial thoughts on how to approach such a development?
5. Lemma 1 assumes a symmetric, zero-mean distribution for the prediction error $\epsilon$. How robust is SafeFlowMatcher to deviations from this assumption, particularly if the prediction errors are biased or exhibit heavy tails? What are the practical implications of such deviations on safety guarantees and performance?
6. While Naive Safe FM serves as a baseline, could the authors provide a more in-depth discussion on why directly applying safety constraints from the beginning leads to a high trap rate? A clearer theoretical or empirical breakdown of the failure modes of Naive Safe FM would further strengthen the argument for the PC integrator.
7. For the vanishing time-scaled flow dynamics, the optimal $\alpha=2$ was found empirically. Is there a deeper theoretical justification for this specific value or for the general range of $\alpha$ that provides a good trade-off between accelerated convergence and stability?
8. The current experimental validation focuses on maze navigation and locomotion. How would SafeFlowMatcher perform on more complex robotic manipulation tasks (e.g., pick-and-place with moving obstacles) that typically involve higher-dimensional state and action spaces, more intricate safety constraints, and potentially longer planning horizons?

### Questions
See Weaknesses^

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces SafeFlowMatcher, a planning framework that integrates conditional flow matching with control barrier functions to generate trajectories that are efficient to sample and provably converge to a robust safe set. The method addresses a key limitation of flow-matching and diffusion planners, which produce high-quality paths but lack safety control and can become trapped when constraints are enforced at each latent step.
SafeFlowMatcher relies on a two-phase prediction-correction scheme. The prediction phase runs standard FM dynamics to propose an unconstrained path, while the correction phase refines it using a vanishing time-scaled FM field and a CBF-based quadratic program that projects waypoints back into the safe set, ensuring finite-time convergence under the derived barrier conditions.
On the theoretical side, the authors restate finite-time CBF results and provide a flow-invariance certificate guaranteeing safety for the corrected flow. Empirically, the framework is validated on Maze2D and MuJoCo locomotion. It consistently outperforms FM, DDIM, and SafeDiffuser-based baselines on the Maze2D environment, achieving higher scores while ablations confirm the role of each phase and the vanishing-scale mechanism.

### Strengths
- The paper identifies a genuine mismatch in existing “safe” generative planners: safety is typically enforced at intermediate latent steps of the denoising or flow-matching process, although only the final trajectory is executed. This “semantic misalignment” is addressed cleanly by decoupling generation and certification: safety is enforced only in the correction phase of the flow. This clarification is timely and relevant for the diffusion/FM-for-planning community.
- The prediction-correction (PC) scheme is well motivated. The prediction phase runs unconstrained FM dynamics to avoid distributional drift, while the correction phase refines the path using a vanishing time-scaled FM vector field and a one-constraint CBF-QP projecting waypoints into the safe set. The finite-time safety guarantee applies to this corrected flow. The ablation in section 4.2 (prediction-only / correction-only / full PC) clearly supports this design.
- The QP correction is computationally light: the authors define a convex problem with a single CBF constraint and closed-form projection, countering concerns about runtime overhead. The running time reported in the experiments shows that this approach is quite fast compared to other SOTA methods.
- The authors performed a serious experimental analysis and comparisons by comparing their approach to a wide variety of baselines (for which they have themselves reproduced the results). Experiments are coherent with the claims and SafeFlowMatcher always achieves a postivie barrier function with zero trap rate, outperforming all baselines while remaining efficient.

### Weaknesses
- On p.3, the authors introduce CBFs in their general control-affine form: $\dot x = f(x)+g(x)u$, which normally presupposes known dynamics; in experiments, they seem not to use a physical model, but instead plug in analytic obstacle functions. This distinction is not clearly stated.
- Definition 1 and Lemma 1 are written for general systems with Lipschitz $f,g$, but in practice, the “dynamics” is the FM vector field with vanishing scaling (cf. equation 10). The paper does state this in section 3.2, but the transition is quick. I think the authors should give a short explanation with intuitions; the CBF is not certified in the real environment, but on the generated flow. Furthermore, Definitions 1 and Lemma 1 should be more intuitively justified; currently, they are just "dropped" there without a clear explanation, which makes it difficult for the reader to understand their consequences and intuitive meanings.
- I noticed some clarity issues: on page 3, line 108, some distribution over paths $q$ appears, but has not been introduced anywhere. I also found that, in the main text, the environments and the assumptions made on those environments were not well-specified. For instance, what is the state space and what are the dynamics of the $2$D maze? Does the method assume access to the exact dynamics or just to a dataset containing random trajectories? If this is the case, from where do they come? A short description of the environments and assumptions would help.

### Questions
- Could you make explicit what you actually assume in the experiments? Am I correct that you do not know the environment transition dynamics and that the only environment-specific information you use is offline trajectories for FM training and analytic, differentiable obstacle/safety functions $b(x)$ with known centers and scales? Is there any case where you used the true simulator for $\nabla b$? 
- In the maze experiments, are you conditioning FM on start-goal or only on start and letting it flow to the high-density goal region? Could you clarify what is actually fed to the network at inference time?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes SafeFlowMatcher, a planning framework that aims to combine Flow Matching (FM) generative models with the certified safety of Control Barrier Functions (CBFs). The core contribution is a two-phase "Prediction-Correction" (PC) integrator. First, a "Prediction" phase uses the learned FM model to generate a candidate path in one or a few steps, without safety constraints. Second, a "Correction" phase refines this path using a proposed "vanishing time-scaled flow dynamics" (VTFD) and a CBF-based quadratic program (QP) to enforce safety and finite-time convergence to the safe set. The authors claim this two-phase approach decouples generation from certification, thereby avoiding the "local trap" problem. The method is evaluated on 2D maze navigation and locomotion tasks, showing improved safety and task performance and a zero "trap rate" compared to baselines like SafeDiffuser.

### Strengths
1. The paper identifies a clear and important problem in safe generative planning. Applying hard safety constraints (like CBFs) directly within the sampling loop of a generative model (like diffusion or flow matching) can distort the learned dynamics, leading to "distributional drift" and "local trap" failures. The goal of decoupling generation from certification is well-motivated. 

2. To further enhance sampling stability in the correction phase, this paper proposes a time-scaled flow dynamics method to contract the prediction errors and mitigate drifting.

3. The paper compares the proposed method with multiple baselines and conducts various ablation studies to demonstrate the efficiency and effectiveness of SafeFlowMatcher.

### Weaknesses
1. The theoretical guarantees for the correction phase (Lemma 2 and 3) rest on strong assumptions about the prediction error $\epsilon$ (e.g., symmetric, zero-mean, and a locally strongly convex negative log-density). There is no empirical validation to show that the actual integration error from a deep FM model satisfies these convenient properties.



2. In the experiments (Table 1), SafeFlowMatcher achieves zero trap rate. How does the proposed method guarantee the low trap rate? Can the method consistently achieve a zero trap rate in some higher-dimensional or more complex navigation tasks? Are there any theoretical guarantees?



3. The current evaluation only includes three simple environments: one maze and two locomotion environments. Can the author provide experimental results on robot manipulation tasks as well? For example, the environment used in SafeDiffuser might be a good starting point.



4. In Algorithm 1, SafeFlowMatcher first samples a trajectory and solves QP in a double for-loop. The authors report a very fast sampling time of ~4.7ms. To better understand the efficiency of the method, it would be more informative to report the computation time of both phases.



5. Does the QP always have a solution? What will Algorithm 1 do if the QP can not be solved? Will the safety guarantee still be satisfied in that case? 



6. Does the method always require a specific function $b$ as in equation (6)? How to obtain $b$ for environments with irregular geometry, agents with high-dimensional state space, or robots with image-based observation?



7. In line 214, "that places $\tau_0^p$ sufficiently close to ...", should the term here be $\tau_1^p$?

### Questions
See the weakness above.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents SafeFlowMatcher, a novel framework for safe robotic path planning that
effectively couples generative Flow Matching (FM) with Control Barrier Functions (CBFs). The
authors identify a key problem with existing generative planners: they either lack formal safety
guarantees or, if certification-based, suffer from local trap problems where interventions distort
the generative process and cause the plan to fail.

### Strengths
● Paper is very well written and easy to understand
● The key idea of a two-phase Prediction-Correction (PC) integrator is both simple and
highly effective. Decoupling the initial path generation from the subsequent safety
correction elegantly solves the local trap problem
● The paper provides a formal barrier certificate and a proof of finite-time convergence -
strong mathematical guarantees.
● The method is validated on maze navigation and high-dimensional locomotion tasks,
demonstrating superior performance across key metrics: safety, path quality, and
efficiency

### Weaknesses
● The paper's baselines are exclusively generative models (diffusion- and FM-based) .
While this is the direct field of contribution, classical sampling- or optimization-based
planners are still the gold standard in many robotics applications. A comparison against a
strong classical baseline (e.g., an optimization-based planner) would have provided a
better picture of the practical performance.

### Questions
● How does the framework handle dynamically changing environments? The CBF-QP is
solved at each control step, which seems well-suited for this, but the underlying FM
model is trained to generate paths based on a static map. How would the model perform
if an obstacle changed position mid-rollout?

### Soundness
3

### Presentation
3

### Contribution
3
